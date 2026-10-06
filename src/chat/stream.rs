//! SSE transform for streaming responses (port of open-sse/utils/stream.js
//! createSSEStream + streamHandler.js pipeWithDisconnect).

use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use super::usage::{estimate_usage, extract_usage, filter_usage_for_format, has_valid_usage, merge_usage};
use crate::exec::ByteStream;
use crate::jsv::{now_ms, now_s, truthy};
use crate::sse::{LineParser, Parsed, parse_sse_line};
use crate::translate::{self, CLAUDE, OLLAMA, OPENAI, OPENAI_RESPONSES, format_sse, has_valuable_content, translate_response};

const PENDING_COMPLETION_FLUSH_MS: u64 = 3000;
pub const DEFAULT_STALL_MS: u64 = 360_000;

/// What the stream saw, reported once when it ends (or is dropped).
pub struct StreamOutcome {
    pub content: String,
    pub thinking: String,
    pub usage: Option<Value>,
    pub ttft_ms: Option<i64>,
}

pub type OnComplete = Box<dyn FnOnce(StreamOutcome) + Send>;

pub enum Mode {
    Translate { target: String, source: String },
    Passthrough,
}

pub struct StreamOpts {
    pub mode: Mode,
    pub provider: String,
    pub model: String,
    /// Client request body (input-token estimation).
    pub body: Value,
    pub tool_name_map: Option<serde_json::Map<String, Value>>,
    pub renamed_tool_names: Option<serde_json::Map<String, Value>>,
    pub custom_tool_names: Vec<String>,
    pub session_id: Option<String>,
    pub stall_ms: u64,
    pub on_complete: Option<OnComplete>,
}

struct Transformer {
    mode_translate: bool,
    target: String,
    source: String,
    provider: String,
    body: Value,
    state: Value,
    usage: Option<Value>,
    total_len: usize,
    content: String,
    thinking: String,
    started: Instant,
    ttft: Option<i64>,
    cur_resp_event: Option<String>,
    resp_terminal_seen: bool,
    resp_done_sent: bool,
    stream_done_sent: bool,
    finalized: bool,
    client_done_sent: bool,
    on_complete: Option<OnComplete>,
    restore_map: Option<serde_json::Map<String, Value>>,
}

fn is_responses_terminal(event: Option<&str>, chunk: &Value) -> bool {
    let ty = event.map(str::to_owned).or_else(|| chunk["type"].as_str().map(str::to_owned));
    if matches!(ty.as_deref(), Some("response.completed" | "response.done" | "response.failed" | "error")) {
        return true;
    }
    matches!(chunk["response"]["status"].as_str(), Some("completed" | "failed"))
}

pub fn incomplete_responses_failure() -> String {
    format_sse(
        &json!({"event": "response.failed", "data": {"type": "response.failed", "response": {
            "id": format!("resp_{}", now_ms()), "status": "failed",
            "error": {"type": "stream_error", "code": "stream_disconnected", "message": "stream closed before response.completed"}}}}),
        OPENAI_RESPONSES,
    )
}

/// buildStreamErrorBytes(status, message, clientFormat)
pub fn stream_error_bytes(status: u16, message: &str, client_format: &str) -> String {
    let err = super::util::error_body(status, message)["error"].clone();
    if client_format == CLAUDE {
        format_sse(&json!({"type": "error", "error": err}), CLAUDE)
    } else {
        format_sse(&json!({"error": err}), client_format) + "data: [DONE]\n\n"
    }
}

fn fix_invalid_id(p: &mut Value) -> bool {
    let Some(id) = p["id"].as_str() else { return false };
    if id.is_empty() || !(id == "chat" || id == "completion" || id.chars().count() < 8) {
        return false;
    }
    let fb = [&p["extend_fields"]["requestId"], &p["extend_fields"]["traceId"]]
        .into_iter()
        .find(|v| truthy(v))
        .map(crate::jsv::js_string)
        .unwrap_or_else(|| radix36(now_ms() as u64));
    p["id"] = json!(format!("chatcmpl-{fb}"));
    true
}

fn radix36(mut n: u64) -> String {
    const D: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut s = vec![];
    while n > 0 {
        s.push(D[(n % 36) as usize]);
        n /= 36;
    }
    if s.is_empty() {
        s.push(b'0');
    }
    s.reverse();
    String::from_utf8(s).unwrap()
}

impl Transformer {
    fn new(o: &mut StreamOpts) -> Self {
        let (translate_mode, target, source) = match &o.mode {
            Mode::Translate { target, source } => (true, target.clone(), source.clone()),
            Mode::Passthrough => (false, String::new(), String::new()),
        };
        let mut state = if translate_mode { translate::resp::init_state(&source) } else { json!({}) };
        if translate_mode {
            state["provider"] = json!(o.provider);
            state["model"] = json!(o.model);
            state["toolNameMap"] = o.tool_name_map.clone().map(Value::Object).unwrap_or(Value::Null);
            if let Some(r) = &o.renamed_tool_names {
                state["renamedToolNames"] = Value::Object(r.clone());
            }
            state["customToolNames"] = json!(o.custom_tool_names);
            state["sessionId"] = o.session_id.clone().map(Value::String).unwrap_or(Value::Null);
            state["targetFormat"] = json!(target);
        }
        let mut restore_map = o.tool_name_map.clone();
        if let Some(r) = &o.renamed_tool_names {
            restore_map.get_or_insert_with(Default::default).extend(r.clone());
        }
        Transformer {
            mode_translate: translate_mode,
            target,
            source,
            provider: o.provider.clone(),
            body: std::mem::take(&mut o.body),
            state,
            usage: None,
            total_len: 0,
            content: String::new(),
            thinking: String::new(),
            started: Instant::now(),
            ttft: None,
            cur_resp_event: None,
            resp_terminal_seen: false,
            resp_done_sent: false,
            stream_done_sent: false,
            finalized: false,
            client_done_sent: false,
            on_complete: o.on_complete.take(),
            restore_map,
        }
    }

    fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        self.finalized = true;
        let mut fin = if self.mode_translate { Some(self.state["usage"].clone()).filter(|u| !u.is_null()) } else { self.usage.clone() };
        if !has_valid_usage(fin.as_ref()) && self.total_len > 0 {
            let fmt = if self.mode_translate { self.source.as_str() } else { OPENAI };
            let est = estimate_usage(&self.body, self.total_len, fmt);
            if self.mode_translate {
                self.state["usage"] = est.clone();
            } else {
                self.usage = Some(est.clone());
            }
            fin = Some(est);
        }
        if let Some(cb) = self.on_complete.take() {
            cb(StreamOutcome {
                content: std::mem::take(&mut self.content),
                thinking: std::mem::take(&mut self.thinking),
                usage: fin.filter(|u| has_valid_usage(Some(u))),
                ttft_ms: self.ttft,
            });
        }
    }

    fn emit(&mut self, out: &mut String, item: &Value, fmt: &str) {
        let s = format_sse(item, fmt);
        if s == "data: [DONE]\n\n" {
            if self.client_done_sent {
                return;
            }
            self.client_done_sent = true;
        }
        out.push_str(&s);
    }

    fn flush_pending_completion(&mut self, out: &mut String) {
        let items = translate_response(&self.target, &self.source, None, &mut self.state);
        for it in items.iter().filter(|v| !v.is_null()) {
            self.emit(out, it, &self.source.clone());
        }
        self.finalize();
    }

    fn completion_pending(&self) -> bool {
        self.mode_translate
            && self.target == OPENAI
            && self.source == OPENAI_RESPONSES
            && self.state["completionPending"] == json!(true)
            && self.state["completedSent"] != json!(true)
    }

    fn passthrough_line(&mut self, line: &str, out: &mut String) {
        let trimmed = line.trim();
        let mut output: Option<String> = None;
        let mut responses_terminal = false;
        if let Some(rest) = trimmed.strip_prefix("data:") {
            if rest.trim() != "[DONE]" {
                let Ok(mut p) = serde_json::from_str::<Value>(rest.trim()) else { return };
                let id_fixed = fix_invalid_id(&mut p);
                let mut injected = false;
                if p.get("choices").is_some() {
                    if !truthy(&p["object"]) {
                        p["object"] = json!("chat.completion.chunk");
                        injected = true;
                    }
                    if !truthy(&p["created"]) {
                        p["created"] = json!(now_s());
                        injected = true;
                    }
                }
                if p.get("prompt_filter_results").is_some() {
                    crate::jsv::del(&mut p, "prompt_filter_results");
                    injected = true;
                }
                if let Some(ch) = p["choices"].as_array_mut() {
                    for c in ch.iter_mut() {
                        if c.get("content_filter_results").is_some() {
                            crate::jsv::del(c, "content_filter_results");
                            injected = true;
                        }
                        if c["delta"]["tool_calls"].as_array().map(|a| a.is_empty()).unwrap_or(false) {
                            crate::jsv::del(&mut c["delta"], "tool_calls");
                            injected = true;
                        }
                    }
                }
                if !has_valuable_content(&p, OPENAI) {
                    return;
                }
                if let Some(map) = &self.restore_map {
                    if p["choices"][0]["delta"]["tool_calls"].is_array() {
                        crate::providers::opencode::restore_tool_names(&mut p, map);
                        injected = true;
                    }
                }
                let d = &p["choices"][0]["delta"];
                if let Some(c) = d["content"].as_str().filter(|s| !s.is_empty()) {
                    self.total_len += c.chars().count();
                    self.content.push_str(c);
                }
                if let Some(r) = d["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
                    self.total_len += r.chars().count();
                    self.thinking.push_str(r);
                }
                if let Some(ex) = extract_usage(&p) {
                    self.usage = merge_usage(self.usage.as_ref(), Some(&ex));
                }
                responses_terminal = is_responses_terminal(self.cur_resp_event.as_deref(), &p);
                let is_finish = truthy(&p["choices"][0]["finish_reason"]);
                if is_finish && !has_valid_usage(Some(&p["usage"])) {
                    let est = estimate_usage(&self.body, self.total_len, OPENAI);
                    p["usage"] = filter_usage_for_format(&est, OPENAI);
                    self.usage = Some(est);
                    output = Some(format!("data: {p}\n"));
                } else if is_finish && self.usage.is_some() {
                    p["usage"] = filter_usage_for_format(self.usage.as_ref().unwrap(), OPENAI);
                    output = Some(format!("data: {p}\n"));
                } else if id_fixed || injected {
                    output = Some(format!("data: {p}\n"));
                }
            }
        }
        if trimmed.starts_with("event:") {
            self.cur_resp_event = Some(trimmed[6..].trim().to_string());
        }
        let output = output.unwrap_or_else(|| {
            if line.starts_with("data:") && !line.starts_with("data: ") { format!("data: {}\n", &line[5..]) } else { format!("{line}\n") }
        });
        out.push_str(&output);
        if responses_terminal {
            self.finalize();
        }
    }

    fn translate_line(&mut self, line: &str, out: &mut String) {
        let trimmed = line.trim();
        if self.target == OPENAI_RESPONSES && trimmed.starts_with("event:") {
            self.cur_resp_event = Some(trimmed[6..].trim().to_string());
        }
        if trimmed.is_empty() {
            return;
        }
        let Some(parsed) = parse_sse_line(trimmed, Some(&self.target)) else { return };
        let is_resp_stream = self.target == OPENAI_RESPONSES;
        let keeps_resp = is_resp_stream && self.source == OPENAI_RESPONSES;
        let parsed_v = match parsed {
            Parsed::Done => json!({"done": true}),
            Parsed::Value(v) => v,
        };
        let resp_event = if is_resp_stream {
            self.cur_resp_event.clone().or_else(|| parsed_v["type"].as_str().map(str::to_owned))
        } else {
            None
        };
        if is_resp_stream && is_responses_terminal(resp_event.as_deref(), &parsed_v) {
            self.resp_terminal_seen = true;
        }
        if truthy(&parsed_v["done"]) && self.target != OLLAMA {
            if self.completion_pending() {
                self.flush_pending_completion(out);
            }
            if keeps_resp && !self.resp_terminal_seen {
                out.push_str(&incomplete_responses_failure());
                self.resp_terminal_seen = true;
            }
            if keeps_resp && !self.stream_done_sent {
                out.push_str("data: [DONE]\n\n");
            }
            self.stream_done_sent = true;
            if keeps_resp {
                self.resp_done_sent = true;
            }
            return;
        }
        self.accumulate(&parsed_v);
        if let Some(ex) = extract_usage(&parsed_v) {
            let prev = Some(self.state["usage"].clone()).filter(|u| u.is_object());
            self.state["usage"] = merge_usage(prev.as_ref(), Some(&ex)).unwrap_or(Value::Null);
        }
        if keeps_resp {
            if let Some(ev) = resp_event {
                let mut data = parsed_v;
                if let Some(map) = &self.restore_map {
                    crate::providers::opencode::restore_tool_names(&mut data, map);
                }
                out.push_str(&format_sse(&json!({"event": ev, "data": data}), &self.source));
                self.cur_resp_event = None;
                if self.resp_terminal_seen {
                    self.finalize();
                }
                return;
            }
        }
        self.cur_resp_event = None;
        let items = translate_response(&self.target, &self.source, Some(&parsed_v), &mut self.state);
        let source = self.source.clone();
        for mut item in items.into_iter().filter(|v| !v.is_null()) {
            if !has_valuable_content(&item, &source) {
                continue;
            }
            let is_finish = item["type"] == "message_delta" || truthy(&item["choices"][0]["finish_reason"]);
            if truthy(&self.state["finishReason"]) && is_finish {
                if !has_valid_usage(Some(&item["usage"])) && self.total_len > 0 {
                    let est = estimate_usage(&self.body, self.total_len, &source);
                    item["usage"] = filter_usage_for_format(&est, &source);
                    self.state["usage"] = est;
                } else if self.state["usage"].is_object() {
                    item["usage"] = filter_usage_for_format(&self.state["usage"], &source);
                }
            }
            self.emit(out, &item, &source);
        }
    }

    fn accumulate(&mut self, p: &Value) {
        let add = |s: &str, thinking: bool, this: &mut Self| {
            this.total_len += s.chars().count();
            if thinking { this.thinking.push_str(s) } else { this.content.push_str(s) }
        };
        if let Some(t) = p["delta"]["text"].as_str().filter(|s| !s.is_empty()) {
            add(t, false, self);
        }
        if let Some(t) = p["delta"]["thinking"].as_str().filter(|s| !s.is_empty()) {
            add(t, true, self);
        }
        if let Some(t) = p["choices"][0]["delta"]["content"].as_str().filter(|s| !s.is_empty()) {
            add(t, false, self);
        }
        if let Some(t) = p["choices"][0]["delta"]["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
            add(t, true, self);
        }
        let parts = if p["candidates"][0]["content"]["parts"].is_array() {
            p["candidates"][0]["content"]["parts"].as_array().cloned()
        } else {
            p["response"]["candidates"][0]["content"]["parts"].as_array().cloned()
        };
        for part in parts.into_iter().flatten() {
            if let Some(t) = part["text"].as_str().filter(|s| !s.is_empty()) {
                add(t, part["thought"] == json!(true), self);
            }
        }
        if let Some(t) = p["delta"].as_str().filter(|_| p["type"] == "response.output_text.delta") {
            add(t, false, self);
        }
    }

    fn line(&mut self, line: &str, out: &mut String) {
        if self.mode_translate { self.translate_line(line, out) } else { self.passthrough_line(line, out) }
    }

    fn flush(&mut self, rest: Option<String>, out: &mut String) {
        if !self.mode_translate {
            if let Some(b) = rest {
                let o = if b.starts_with("data:") && !b.starts_with("data: ") { format!("data: {}", &b[5..]) } else { b };
                out.push_str(&o);
                out.push('\n');
            }
            let gemini_family = matches!(self.provider.as_str(), "antigravity" | "gemini" | "vertex");
            if !self.stream_done_sent && !gemini_family {
                out.push_str("data: [DONE]\n\n");
            }
            self.finalize();
            return;
        }
        if let Some(b) = rest {
            if let Some(p) = parse_sse_line(b.trim(), Some(&self.target)) {
                let is_done = p.is_done() && self.target != OLLAMA;
                if !is_done {
                    let v = p.into_value();
                    if let Some(ex) = extract_usage(&v) {
                        let prev = Some(self.state["usage"].clone()).filter(|u| u.is_object());
                        self.state["usage"] = merge_usage(prev.as_ref(), Some(&ex)).unwrap_or(Value::Null);
                    }
                    self.accumulate(&v);
                    let items = translate_response(&self.target, &self.source, Some(&v), &mut self.state);
                    let source = self.source.clone();
                    for it in items.iter().filter(|v| !v.is_null()) {
                        self.emit(out, it, &source);
                    }
                }
            }
        }
        let items = translate_response(&self.target, &self.source, None, &mut self.state);
        let source = self.source.clone();
        for it in items.iter().filter(|v| !v.is_null()) {
            self.emit(out, it, &source);
        }
        let keeps_resp = self.target == OPENAI_RESPONSES && self.source == OPENAI_RESPONSES;
        if keeps_resp && !self.resp_terminal_seen {
            out.push_str(&incomplete_responses_failure());
            self.resp_terminal_seen = true;
        }
        if keeps_resp && !self.resp_done_sent && !self.stream_done_sent {
            out.push_str("data: [DONE]\n\n");
            self.resp_done_sent = true;
            self.stream_done_sent = true;
        }
        // OpenAI chat clients expect the sentinel even when the upstream spoke
        // another wire format (some hang without it).
        if self.source == OPENAI && !self.client_done_sent {
            out.push_str("data: [DONE]\n\n");
            self.client_done_sent = true;
        }
        self.finalize();
    }
}

impl Drop for Transformer {
    fn drop(&mut self) {
        // Client went away mid-stream: still record what we saw.
        self.finalize();
    }
}

/// Pipes an upstream SSE/NDJSON body through the format transform.
/// `abort_terminal` produces the bytes sent when the upstream dies or stalls
/// after the 200 header was already sent.
pub fn transform(mut upstream: ByteStream, mut opts: StreamOpts) -> ByteStream {
    let stall = Duration::from_millis(if opts.stall_ms == 0 { DEFAULT_STALL_MS } else { opts.stall_ms });
    let mut t = Transformer::new(&mut opts);
    let responses_passthrough = matches!(&opts.mode, Mode::Translate { target, source } if target == OPENAI_RESPONSES && source == OPENAI_RESPONSES);
    let client_format = match &opts.mode {
        Mode::Translate { source, .. } => source.clone(),
        Mode::Passthrough => OPENAI.to_string(),
    };
    let s = async_stream::stream! {
        let mut lines = LineParser::default();
        let mut pending_deadline: Option<tokio::time::Instant> = None;
        loop {
            let next = if let Some(dl) = pending_deadline {
                tokio::select! {
                    r = tokio::time::timeout(stall, upstream.next()) => Some(r),
                    _ = tokio::time::sleep_until(dl) => None,
                }
            } else {
                Some(tokio::time::timeout(stall, upstream.next()).await)
            };
            let mut out = String::new();
            let mut abort: Option<String> = None;
            match next {
                None => {
                    pending_deadline = None;
                    if t.completion_pending() {
                        t.flush_pending_completion(&mut out);
                    }
                }
                Some(Ok(Some(Ok(chunk)))) => {
                    if t.ttft.is_none() {
                        t.ttft = Some(t.started.elapsed().as_millis() as i64);
                    }
                    for l in lines.push(&chunk) {
                        t.line(&l, &mut out);
                    }
                    if t.completion_pending() && pending_deadline.is_none() {
                        pending_deadline = Some(tokio::time::Instant::now() + Duration::from_millis(PENDING_COMPLETION_FLUSH_MS));
                    }
                }
                Some(Ok(None)) => {
                    let rest = lines.finish();
                    t.flush(rest, &mut out);
                    if !out.is_empty() {
                        yield Ok(Bytes::from(out));
                    }
                    break;
                }
                Some(Ok(Some(Err(e)))) => {
                    abort = Some(format!("upstream connection lost: {e}"));
                }
                Some(Err(_)) => {
                    abort = Some("stream stall timeout".to_string());
                }
            }
            if let Some(msg) = abort.take() {
                tracing::warn!("stream aborted: {msg}");
                t.finalize();
                let term = if responses_passthrough {
                    format!("{}data: [DONE]\n\n", incomplete_responses_failure())
                } else {
                    stream_error_bytes(504, &msg, &client_format)
                };
                out.push_str(&term);
                yield Ok(Bytes::from(out));
                break;
            }
            if !out.is_empty() {
                yield Ok(Bytes::from(out));
            }
        }
    };
    s.boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: Vec<&'static str>, opts: StreamOpts) -> String {
        let up: ByteStream = futures::stream::iter(chunks.into_iter().map(|c| Ok(Bytes::from(c)))).boxed();
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let mut s = transform(up, opts);
            let mut out = String::new();
            while let Some(c) = s.next().await {
                out.push_str(&String::from_utf8_lossy(&c.unwrap()));
            }
            out
        })
    }

    fn opts(mode: Mode) -> StreamOpts {
        StreamOpts {
            mode,
            provider: "openai".into(),
            model: "m".into(),
            body: json!({"messages": [{"role": "user", "content": "hi"}]}),
            tool_name_map: None,
            renamed_tool_names: None,
            custom_tool_names: vec![],
            session_id: None,
            stall_ms: 0,
            on_complete: None,
        }
    }

    #[test]
    fn passthrough_adds_done_and_usage() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut o = opts(Mode::Passthrough);
        o.on_complete = Some(Box::new(move |r: StreamOutcome| tx.send((r.content, r.usage)).unwrap()));
        let out = run(vec!["data: {\"id\":\"chatcmpl-123456789\",\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n", "data: {\"id\":\"chatcmpl-123456789\",\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"], o);
        assert!(out.contains("\"usage\""), "{out}");
        assert!(out.trim_end().ends_with("data: [DONE]"));
        let (content, usage) = rx.recv().unwrap();
        assert_eq!(content, "Hello");
        assert!(usage.is_some());
    }

    #[test]
    fn claude_to_openai_translate() {
        let o = opts(Mode::Translate { target: CLAUDE.into(), source: OPENAI.into() });
        let out = run(vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"c\",\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ], o);
        assert!(out.contains("\"content\":\"Hi\""), "{out}");
        assert!(out.contains("\"finish_reason\":\"stop\""), "{out}");
    }

    #[test]
    fn openai_to_claude_translate() {
        let o = opts(Mode::Translate { target: OPENAI.into(), source: CLAUDE.into() });
        let out = run(vec![
            "data: {\"id\":\"chatcmpl-x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Yo\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ], o);
        assert!(out.contains("event: message_start"), "{out}");
        assert!(out.contains("\"text\":\"Yo\""), "{out}");
        assert!(out.contains("event: message_stop"), "{out}");
    }
}
