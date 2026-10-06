//! Streaming (SSE) translation, built as small state machines that consume
//! upstream events and emit events in the client's format.

use std::collections::HashMap;

use serde_json::{Value, json};

use super::response::{
    claude_stop_to_openai, gemini_finish_to_openai, new_id, openai_finish_to_claude,
};
use crate::catalog::Format;

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Incremental parser for `text/event-stream` bodies.
#[derive(Default)]
pub struct SseParser {
    buf: String,
    pending: Vec<u8>,
}

impl SseParser {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        // Keep incomplete UTF-8 sequences until the next chunk arrives.
        self.pending.extend_from_slice(chunk);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        let text: Vec<u8> = self.pending.drain(..valid).collect();
        self.buf
            .push_str(&String::from_utf8_lossy(&text).replace('\r', ""));

        let mut out = Vec::new();
        while let Some(pos) = self.buf.find("\n\n") {
            let block: String = self.buf.drain(..pos + 2).collect();
            if let Some(ev) = parse_block(&block) {
                out.push(ev);
            }
        }
        out
    }

    pub fn finish(&mut self) -> Vec<SseEvent> {
        let rest = std::mem::take(&mut self.buf);
        parse_block(&rest).into_iter().collect()
    }
}

fn parse_block(block: &str) -> Option<SseEvent> {
    let mut ev = SseEvent::default();
    let mut data = Vec::new();
    for line in block.lines() {
        if let Some(v) = line.strip_prefix("event:") {
            ev.event = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("data:") {
            data.push(v.strip_prefix(' ').unwrap_or(v).to_string());
        }
    }
    if data.is_empty() {
        return None;
    }
    ev.data = data.join("\n");
    Some(ev)
}

pub fn sse_data(v: &Value) -> String {
    format!("data: {v}\n\n")
}

pub fn sse_named(event: &str, v: &Value) -> String {
    format!("event: {event}\ndata: {v}\n\n")
}

// ---------------------------------------------------------------------------
// Usage observation (works on the upstream's native events)
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Copy, Debug)]
pub struct StreamUsage {
    pub prompt: i64,
    pub completion: i64,
}

impl StreamUsage {
    pub fn observe(&mut self, format: Format, data: &Value) {
        let n = |x: &Value| x.as_i64();
        match format {
            Format::OpenAI => {
                if let Some(u) = data.get("usage").filter(|u| u.is_object()) {
                    if let Some(p) = n(&u["prompt_tokens"]) {
                        self.prompt = p;
                    }
                    if let Some(c) = n(&u["completion_tokens"]) {
                        self.completion = c;
                    }
                }
            }
            Format::Claude => {
                let u = data
                    .get("message")
                    .map(|m| &m["usage"])
                    .or_else(|| data.get("usage"));
                if let Some(u) = u.filter(|u| u.is_object()) {
                    let input = n(&u["input_tokens"]).unwrap_or(0)
                        + n(&u["cache_read_input_tokens"]).unwrap_or(0)
                        + n(&u["cache_creation_input_tokens"]).unwrap_or(0);
                    if input > 0 {
                        self.prompt = input;
                    }
                    if let Some(c) = n(&u["output_tokens"]) {
                        self.completion = self.completion.max(c);
                    }
                }
            }
            Format::Gemini => {
                if let Some(u) = data.get("usageMetadata") {
                    if let Some(p) = n(&u["promptTokenCount"]) {
                        self.prompt = p;
                    }
                    let c = n(&u["candidatesTokenCount"]).unwrap_or(0)
                        + n(&u["thoughtsTokenCount"]).unwrap_or(0);
                    if c > 0 {
                        self.completion = c;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Upstream -> OpenAI chunks
// ---------------------------------------------------------------------------

struct ChunkBuilder {
    id: String,
    model: String,
    created: i64,
}

impl ChunkBuilder {
    fn new(model: &str) -> Self {
        Self {
            id: new_id("chatcmpl-"),
            model: model.to_string(),
            created: crate::db::now(),
        }
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })
    }

    fn usage_chunk(&self, u: StreamUsage) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": {"prompt_tokens": u.prompt, "completion_tokens": u.completion, "total_tokens": u.prompt + u.completion},
        })
    }
}

pub struct ClaudeToOpenAI {
    b: ChunkBuilder,
    tool_index: HashMap<i64, usize>,
    next_tool: usize,
    usage: StreamUsage,
    finished: bool,
}

impl ClaudeToOpenAI {
    pub fn new(model: &str) -> Self {
        Self {
            b: ChunkBuilder::new(model),
            tool_index: HashMap::new(),
            next_tool: 0,
            usage: StreamUsage::default(),
            finished: false,
        }
    }

    pub fn on_event(&mut self, data: &Value) -> Vec<Value> {
        self.usage.observe(Format::Claude, data);
        match data["type"].as_str().unwrap_or("") {
            "message_start" => {
                if let Some(id) = data["message"]["id"].as_str() {
                    self.b.id = id.to_string();
                }
                vec![
                    self.b
                        .chunk(json!({"role": "assistant", "content": ""}), None),
                ]
            }
            "content_block_start" => {
                let block = &data["content_block"];
                if block["type"] == "tool_use" {
                    let idx = self.next_tool;
                    self.next_tool += 1;
                    self.tool_index
                        .insert(data["index"].as_i64().unwrap_or(0), idx);
                    vec![self.b.chunk(
                        json!({"tool_calls": [{
                            "index": idx, "id": block["id"], "type": "function",
                            "function": {"name": block["name"], "arguments": ""}
                        }]}),
                        None,
                    )]
                } else {
                    vec![]
                }
            }
            "content_block_delta" => {
                let d = &data["delta"];
                match d["type"].as_str().unwrap_or("") {
                    "text_delta" => vec![self.b.chunk(json!({"content": d["text"]}), None)],
                    "thinking_delta" => vec![
                        self.b
                            .chunk(json!({"reasoning_content": d["thinking"]}), None),
                    ],
                    "input_json_delta" => {
                        let idx = self
                            .tool_index
                            .get(&data["index"].as_i64().unwrap_or(0))
                            .copied()
                            .unwrap_or(0);
                        vec![self.b.chunk(
                            json!({"tool_calls": [{"index": idx, "function": {"arguments": d["partial_json"]}}]}),
                            None,
                        )]
                    }
                    _ => vec![],
                }
            }
            "message_delta" => {
                let reason = data["delta"]["stop_reason"].as_str().unwrap_or("end_turn");
                self.finished = true;
                vec![
                    self.b.chunk(json!({}), Some(claude_stop_to_openai(reason))),
                    self.b.usage_chunk(self.usage),
                ]
            }
            _ => vec![],
        }
    }

    pub fn finish(&mut self) -> Vec<Value> {
        if self.finished {
            vec![]
        } else {
            self.finished = true;
            vec![
                self.b.chunk(json!({}), Some("stop")),
                self.b.usage_chunk(self.usage),
            ]
        }
    }
}

pub struct GeminiToOpenAI {
    b: ChunkBuilder,
    started: bool,
    next_tool: usize,
    finish_reason: Option<String>,
    usage: StreamUsage,
}

impl GeminiToOpenAI {
    pub fn new(model: &str) -> Self {
        Self {
            b: ChunkBuilder::new(model),
            started: false,
            next_tool: 0,
            finish_reason: None,
            usage: StreamUsage::default(),
        }
    }

    pub fn on_event(&mut self, data: &Value) -> Vec<Value> {
        self.usage.observe(Format::Gemini, data);
        let mut out = Vec::new();
        if !self.started {
            self.started = true;
            out.push(
                self.b
                    .chunk(json!({"role": "assistant", "content": ""}), None),
            );
        }
        let cand = &data["candidates"][0];
        for part in cand["content"]["parts"].as_array().into_iter().flatten() {
            if let Some(fc) = part.get("functionCall") {
                let idx = self.next_tool;
                self.next_tool += 1;
                out.push(self.b.chunk(
                    json!({"tool_calls": [{
                        "index": idx, "id": new_id("call_"), "type": "function",
                        "function": {"name": fc["name"], "arguments": fc.get("args").cloned().unwrap_or(json!({})).to_string()}
                    }]}),
                    None,
                ));
            } else if let Some(t) = part["text"].as_str() {
                let key = if part["thought"].as_bool() == Some(true) {
                    "reasoning_content"
                } else {
                    "content"
                };
                out.push(self.b.chunk(json!({ key: t }), None));
            }
        }
        if let Some(r) = cand["finishReason"].as_str() {
            self.finish_reason = Some(r.to_string());
        }
        out
    }

    pub fn finish(&mut self) -> Vec<Value> {
        let reason = if self.next_tool > 0 {
            "tool_calls"
        } else {
            gemini_finish_to_openai(self.finish_reason.as_deref().unwrap_or("STOP"))
        };
        vec![
            self.b.chunk(json!({}), Some(reason)),
            self.b.usage_chunk(self.usage),
        ]
    }
}

// ---------------------------------------------------------------------------
// OpenAI chunks -> Claude events
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
enum OpenBlock {
    None,
    Text,
    Thinking,
    Tool(usize),
}

pub struct OpenAIToClaude {
    model: String,
    started: bool,
    block_index: i64,
    open: OpenBlock,
    tool_blocks: HashMap<usize, i64>,
    stop_reason: Option<String>,
    usage: StreamUsage,
    done: bool,
}

impl OpenAIToClaude {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            started: false,
            block_index: -1,
            open: OpenBlock::None,
            tool_blocks: HashMap::new(),
            stop_reason: None,
            usage: StreamUsage::default(),
            done: false,
        }
    }

    fn start(&mut self, out: &mut Vec<(String, Value)>) {
        if self.started {
            return;
        }
        self.started = true;
        out.push((
            "message_start".into(),
            json!({"type": "message_start", "message": {
                "id": new_id("msg_"), "type": "message", "role": "assistant", "model": self.model,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": self.usage.prompt, "output_tokens": 0}
            }}),
        ));
    }

    fn close_block(&mut self, out: &mut Vec<(String, Value)>) {
        if self.open != OpenBlock::None {
            out.push((
                "content_block_stop".into(),
                json!({"type": "content_block_stop", "index": self.block_index}),
            ));
            self.open = OpenBlock::None;
        }
    }

    fn open_block(&mut self, kind: OpenBlock, block: Value, out: &mut Vec<(String, Value)>) {
        self.close_block(out);
        self.block_index += 1;
        self.open = kind;
        out.push((
            "content_block_start".into(),
            json!({"type": "content_block_start", "index": self.block_index, "content_block": block}),
        ));
    }

    pub fn on_chunk(&mut self, chunk: &Value) -> Vec<(String, Value)> {
        self.usage.observe(Format::OpenAI, chunk);
        let mut out = Vec::new();
        self.start(&mut out);
        let Some(choice) = chunk["choices"].get(0) else {
            return out;
        };
        let delta = &choice["delta"];

        let reasoning = delta["reasoning_content"]
            .as_str()
            .or_else(|| delta["reasoning"].as_str());
        if let Some(r) = reasoning.filter(|r| !r.is_empty()) {
            if self.open != OpenBlock::Thinking {
                self.open_block(
                    OpenBlock::Thinking,
                    json!({"type": "thinking", "thinking": "", "signature": ""}),
                    &mut out,
                );
            }
            out.push((
                "content_block_delta".into(),
                json!({"type": "content_block_delta", "index": self.block_index, "delta": {"type": "thinking_delta", "thinking": r}}),
            ));
        }

        if let Some(t) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            if self.open != OpenBlock::Text {
                self.open_block(
                    OpenBlock::Text,
                    json!({"type": "text", "text": ""}),
                    &mut out,
                );
            }
            out.push((
                "content_block_delta".into(),
                json!({"type": "content_block_delta", "index": self.block_index, "delta": {"type": "text_delta", "text": t}}),
            ));
        }

        for tc in delta["tool_calls"].as_array().into_iter().flatten() {
            let idx = tc["index"].as_u64().unwrap_or(0) as usize;
            if !self.tool_blocks.contains_key(&idx) {
                let block = json!({
                    "type": "tool_use",
                    "id": tc["id"].as_str().map(str::to_owned).unwrap_or_else(|| new_id("toolu_")),
                    "name": tc["function"]["name"].as_str().unwrap_or(""),
                    "input": {},
                });
                self.open_block(OpenBlock::Tool(idx), block, &mut out);
                self.tool_blocks.insert(idx, self.block_index);
            }
            if let Some(args) = tc["function"]["arguments"]
                .as_str()
                .filter(|a| !a.is_empty())
            {
                let index = self.tool_blocks[&idx];
                out.push((
                    "content_block_delta".into(),
                    json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": args}}),
                ));
            }
        }

        if let Some(f) = choice["finish_reason"].as_str() {
            self.stop_reason = Some(openai_finish_to_claude(f).to_string());
        }
        out
    }

    pub fn finish(&mut self) -> Vec<(String, Value)> {
        if self.done {
            return vec![];
        }
        self.done = true;
        let mut out = Vec::new();
        self.start(&mut out);
        self.close_block(&mut out);
        out.push((
            "message_delta".into(),
            json!({"type": "message_delta",
                   "delta": {"stop_reason": self.stop_reason.clone().unwrap_or_else(|| "end_turn".into()), "stop_sequence": null},
                   "usage": {"input_tokens": self.usage.prompt, "output_tokens": self.usage.completion}}),
        ));
        out.push(("message_stop".into(), json!({"type": "message_stop"})));
        out
    }
}

// ---------------------------------------------------------------------------
// Full pipeline: upstream SSE -> client SSE
// ---------------------------------------------------------------------------

enum ToHub {
    Passthrough,
    Claude(ClaudeToOpenAI),
    Gemini(GeminiToOpenAI),
}

pub struct StreamTranslator {
    inbound: Format,
    target: Format,
    parser: SseParser,
    to_hub: ToHub,
    to_claude: Option<OpenAIToClaude>,
    pub usage: StreamUsage,
    saw_done: bool,
}

impl StreamTranslator {
    pub fn new(inbound: Format, target: Format, model: &str) -> Self {
        let to_hub = if inbound == target {
            ToHub::Passthrough
        } else {
            match target {
                Format::OpenAI => ToHub::Passthrough,
                Format::Claude => ToHub::Claude(ClaudeToOpenAI::new(model)),
                Format::Gemini => ToHub::Gemini(GeminiToOpenAI::new(model)),
            }
        };
        let to_claude = (inbound == Format::Claude && target != Format::Claude)
            .then(|| OpenAIToClaude::new(model));
        Self {
            inbound,
            target,
            parser: SseParser::default(),
            to_hub,
            to_claude,
            usage: StreamUsage::default(),
            saw_done: false,
        }
    }

    fn passthrough(&self) -> bool {
        self.inbound == self.target
    }

    /// Feeds raw upstream bytes; returns bytes to send to the client.
    pub fn push(&mut self, chunk: &[u8]) -> String {
        let events = self.parser.push(chunk);
        if self.passthrough() {
            for ev in &events {
                if let Ok(v) = serde_json::from_str::<Value>(&ev.data) {
                    self.usage.observe(self.target, &v);
                }
            }
            return String::from_utf8_lossy(chunk).into_owned();
        }
        let mut out = String::new();
        for ev in events {
            self.handle(ev, &mut out);
        }
        out
    }

    pub fn finish(&mut self) -> String {
        let events = self.parser.finish();
        if self.passthrough() {
            for ev in &events {
                if let Ok(v) = serde_json::from_str::<Value>(&ev.data) {
                    self.usage.observe(self.target, &v);
                }
            }
            return String::new();
        }
        let mut out = String::new();
        for ev in events {
            self.handle(ev, &mut out);
        }
        let tail = match &mut self.to_hub {
            ToHub::Passthrough => vec![],
            ToHub::Claude(c) => c.finish(),
            ToHub::Gemini(g) => g.finish(),
        };
        self.emit_hub(tail, &mut out);
        match &mut self.to_claude {
            Some(c) => {
                for (name, v) in c.finish() {
                    out.push_str(&sse_named(&name, &v));
                }
            }
            None => {
                if !self.saw_done {
                    self.saw_done = true;
                    out.push_str("data: [DONE]\n\n");
                }
            }
        }
        out
    }

    fn handle(&mut self, ev: SseEvent, out: &mut String) {
        if ev.data.trim() == "[DONE]" {
            return; // emitted in finish()
        }
        let Ok(v) = serde_json::from_str::<Value>(&ev.data) else {
            return;
        };
        self.usage.observe(self.target, &v);
        let chunks = match &mut self.to_hub {
            ToHub::Passthrough => vec![v],
            ToHub::Claude(c) => c.on_event(&v),
            ToHub::Gemini(g) => g.on_event(&v),
        };
        self.emit_hub(chunks, out);
    }

    fn emit_hub(&mut self, chunks: Vec<Value>, out: &mut String) {
        for chunk in chunks {
            match &mut self.to_claude {
                Some(c) => {
                    for (name, v) in c.on_chunk(&chunk) {
                        out.push_str(&sse_named(&name, &v));
                    }
                }
                None => out.push_str(&sse_data(&chunk)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events(s: &str) -> Vec<(Option<String>, Value)> {
        let mut p = SseParser::default();
        let mut evs = p.push(s.as_bytes());
        evs.extend(p.finish());
        evs.into_iter()
            .filter(|e| e.data != "[DONE]")
            .map(|e| (e.event, serde_json::from_str(&e.data).unwrap()))
            .collect()
    }

    #[test]
    fn parser_handles_split_chunks() {
        let mut p = SseParser::default();
        assert!(p.push(b"data: {\"a\"").is_empty());
        let evs = p.push(b":1}\r\n\r\nevent: x\ndata: 2\n\n");
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "{\"a\":1}");
        assert_eq!(evs[1].event.as_deref(), Some("x"));
    }

    #[test]
    fn claude_stream_to_openai() {
        let upstream = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t\",\"name\":\"f\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        let mut t = StreamTranslator::new(Format::OpenAI, Format::Claude, "m");
        let mut out = t.push(upstream.as_bytes());
        out.push_str(&t.finish());
        assert!(out.ends_with("data: [DONE]\n\n"));
        let evs = events(&out);
        assert_eq!(evs[1].1["choices"][0]["delta"]["content"], "Hi");
        assert_eq!(
            evs[2].1["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "f"
        );
        assert_eq!(evs[4].1["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(evs[5].1["usage"]["completion_tokens"], 9);
        assert_eq!(t.usage.prompt, 7);
    }

    #[test]
    fn openai_stream_to_claude() {
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":1}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut t = StreamTranslator::new(Format::Claude, Format::OpenAI, "m");
        let mut out = t.push(upstream.as_bytes());
        out.push_str(&t.finish());
        let evs = events(&out);
        let names: Vec<_> = evs.iter().map(|e| e.0.clone().unwrap()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        let delta = &evs[9].1;
        assert_eq!(delta["delta"]["stop_reason"], "tool_use");
        assert_eq!(delta["usage"]["output_tokens"], 3);
    }

    #[test]
    fn gemini_stream_to_claude() {
        let upstream = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"A\"}]}}]}\r\n\r\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"B\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":2}}\r\n\r\n",
        );
        let mut t = StreamTranslator::new(Format::Claude, Format::Gemini, "g");
        let mut out = t.push(upstream.as_bytes());
        out.push_str(&t.finish());
        let evs = events(&out);
        let text: String = evs
            .iter()
            .filter_map(|e| e.1["delta"]["text"].as_str())
            .collect();
        assert_eq!(text, "AB");
        assert_eq!(evs.last().unwrap().0.as_deref(), Some("message_stop"));
        assert_eq!(t.usage.completion, 2);
    }
}
