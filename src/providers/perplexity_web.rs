//! Perplexity web (session cookie) executor — port of executors/perplexity-web.js.

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use regex::Regex;
use serde_json::{Value, json};

use crate::exec::{ByteStream, ExecArgs, ExecResult, Executor, Headers, Upstream, post_json};
use crate::jsv::{now_ms, truthy};

const API_VERSION: &str = "2.18";
const UA: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36";
const SESSION_MAX_AGE_MS: i64 = 3_600_000;
const SESSION_MAX_ENTRIES: usize = 200;

fn model_map(m: &str) -> Option<(&'static str, &'static str)> {
    Some(match m {
        "pplx-auto" => ("concise", "pplx_pro"),
        "pplx-sonar" => ("copilot", "experimental"),
        "pplx-gpt" => ("copilot", "gpt54"),
        "pplx-gemini" => ("copilot", "gemini31pro_high"),
        "pplx-sonnet" => ("copilot", "claude46sonnet"),
        "pplx-opus" => ("copilot", "claude46opus"),
        "pplx-nemotron" => ("copilot", "nv_nemotron_3_super"),
        _ => return None,
    })
}

fn thinking_map(m: &str) -> Option<&'static str> {
    Some(match m {
        "pplx-gpt" => "gpt54_thinking",
        "pplx-sonnet" => "claude46sonnetthinking",
        "pplx-opus" => "claude46opusthinking",
        _ => return None,
    })
}

static SESSIONS: LazyLock<Mutex<HashMap<String, (String, i64)>>> = LazyLock::new(Default::default);

#[derive(Clone, Debug)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

pub fn session_key(history: &[Turn]) -> String {
    let parts = history.iter().map(|h| format!("{}:{}", h.role, h.content)).collect::<Vec<_>>().join("\n");
    let mut hash: u32 = 0x811c9dc5;
    for u in parts.encode_utf16() {
        hash ^= u as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    format!("{hash:08x}")
}

fn session_lookup(history: &[Turn]) -> Option<String> {
    if history.is_empty() {
        return None;
    }
    let key = session_key(history);
    let mut s = SESSIONS.lock().unwrap();
    let (uuid, ts) = s.get(&key)?.clone();
    if now_ms() - ts > SESSION_MAX_AGE_MS {
        s.remove(&key);
        return None;
    }
    Some(uuid)
}

fn session_store(history: &[Turn], current: &str, response: &str, backend: Option<&str>) {
    let Some(b) = backend else { return };
    let mut full = history.to_vec();
    full.push(Turn { role: "user".into(), content: current.into() });
    full.push(Turn { role: "assistant".into(), content: response.into() });
    let mut s = SESSIONS.lock().unwrap();
    s.insert(session_key(&full), (b.to_string(), now_ms()));
    if s.len() > SESSION_MAX_ENTRIES {
        if let Some(k) = s.iter().min_by_key(|(_, v)| v.1).map(|(k, _)| k.clone()) {
            s.remove(&k);
        }
    }
}

pub fn clean_response(text: &str, strip: bool) -> String {
    static RES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
        [r"<[?]xml[^?]*[?]>", r"\[\d+\]", r"(?s)<grok:[^>]*>.*?</grok:[^>]*>", r"<grok:[^>]*/>", r"(?i)</?response\b[^>]*>"].iter().map(|r| Regex::new(r).unwrap()).collect()
    });
    static SP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" {2,}").unwrap());
    static NL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());
    let mut t = text.to_string();
    for r in RES.iter() {
        t = r.replace_all(&t, "").into_owned();
    }
    if strip {
        t = SP.replace_all(&t, " ").into_owned();
        t = NL.replace_all(&t, "\n\n").into_owned();
        t = t.trim().to_string();
    }
    t
}

pub struct Parsed {
    pub system: String,
    pub history: Vec<Turn>,
    pub current: String,
}

pub fn parse_messages(messages: &[Value]) -> Parsed {
    let mut system = String::new();
    let mut history = vec![];
    for m in messages {
        let mut role = m["role"].as_str().filter(|s| !s.is_empty()).unwrap_or("user").to_string();
        if role == "developer" {
            role = "system".into();
        }
        let content = match &m["content"] {
            Value::String(s) => s.clone(),
            Value::Array(a) => a.iter().filter(|c| c["type"] == "text").map(|c| c["text"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>().join(" "),
            _ => String::new(),
        };
        if content.trim().is_empty() {
            continue;
        }
        if role == "system" {
            system.push_str(&content);
            system.push('\n');
        } else if role == "user" || role == "assistant" {
            history.push(Turn { role, content });
        }
    }
    let mut current = String::new();
    if history.last().map(|h| h.role == "user").unwrap_or(false) {
        current = history.pop().unwrap().content;
    }
    Parsed { system, history, current }
}

fn tools_hint(tools: &Value) -> String {
    let Some(a) = tools.as_array().filter(|a| !a.is_empty()) else { return String::new() };
    let lines: Vec<String> = a
        .iter()
        .map(|t| {
            let f = if t["function"].is_object() { &t["function"] } else { t };
            let name = f["name"].as_str().filter(|s| !s.is_empty()).unwrap_or("unnamed");
            let desc: String = f["description"].as_str().unwrap_or("").split('\n').next().unwrap_or("").chars().take(200).collect();
            format!("- {name}: {desc}")
        })
        .collect();
    format!("Available tools (reference only, cannot invoke):\n{}", lines.join("\n"))
}

pub fn build_query(p: &Parsed, follow_up: Option<&str>, tools: &Value) -> String {
    if follow_up.is_some() {
        return p.current.clone();
    }
    let mut instr = vec![];
    if !p.system.trim().is_empty() {
        instr.push(p.system.trim().to_string());
    }
    let th = tools_hint(tools);
    if !th.is_empty() {
        instr.push(th);
    }
    instr.push("You have built-in web search. Answer questions directly using search results.".into());
    let mut obj = json!({"instructions": instr});
    if !p.history.is_empty() {
        obj["history"] = Value::Array(p.history.iter().map(|h| json!({"role": h.role, "content": h.content})).collect());
    }
    if !p.current.is_empty() {
        obj["query"] = json!(p.current);
    } else if p.history.is_empty() {
        obj["query"] = json!("");
    }
    let s = obj.to_string();
    let units: Vec<u16> = s.encode_utf16().collect();
    if units.len() > 96_000 { String::from_utf16_lossy(&units[units.len() - 96_000..]) } else { s }
}

fn build_body(query: &str, mode: &str, pref: &str, follow_up: Option<&str>) -> Value {
    let tz = std::env::var("TZ").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "UTC".into());
    json!({
        "query_str": query,
        "params": {
            "query_str": query,
            "search_focus": "internet",
            "mode": mode,
            "model_preference": pref,
            "sources": ["web"],
            "attachments": [],
            "frontend_uuid": uuid::Uuid::new_v4().to_string(),
            "frontend_context_uuid": uuid::Uuid::new_v4().to_string(),
            "version": API_VERSION,
            "language": "en-US",
            "timezone": tz,
            "search_recency_filter": null,
            "is_incognito": true,
            "use_schematized_api": true,
            "last_backend_uuid": follow_up,
        }
    })
}

enum Ev {
    Error(String),
    Thinking(String),
    Delta(String, String),
    Done(String),
    Backend(String),
}

fn pplx_sse_events(mut body: ByteStream) -> impl futures::Stream<Item = Value> + Send {
    async_stream::stream! {
        let mut lines = crate::sse::LineParser::default();
        let mut data: Vec<String> = vec![];
        let flush = |data: &mut Vec<String>| -> Option<Option<Value>> {
            if data.is_empty() { return Some(None); }
            let payload = data.join("\n");
            data.clear();
            let t = payload.trim();
            if t.is_empty() || t == "[DONE]" { return None; }
            Some(serde_json::from_str(t).ok())
        };
        loop {
            let c = body.next().await;
            let ls = match c {
                Some(Ok(c)) => lines.push(&c),
                Some(Err(_)) | None => {
                    if let Some(l) = lines.finish() {
                        if let Some(d) = l.strip_prefix("data:") { data.push(d.trim_start().to_string()); }
                    }
                    if let Some(Some(v)) = flush(&mut data) { yield v; }
                    return;
                }
            };
            for line in ls {
                if line.is_empty() {
                    match flush(&mut data) { None => return, Some(Some(v)) => yield v, Some(None) => {} }
                    continue;
                }
                if let Some(d) = line.strip_prefix("data:") { data.push(d.trim_start().to_string()); }
                if line == "event: end_of_stream" { return; }
            }
        }
    }
}

fn extract(body: ByteStream) -> impl futures::Stream<Item = Ev> + Send {
    async_stream::stream! {
        let events = pplx_sse_events(body);
        futures::pin_mut!(events);
        let mut full = String::new();
        let mut seen_len = 0usize;
        let mut seen: HashSet<String> = HashSet::new();
        while let Some(ev) = events.next().await {
            if truthy(&ev["error_code"]) || truthy(&ev["error_message"]) {
                let m = ev["error_message"].as_str().filter(|s| !s.is_empty()).map(str::to_owned).unwrap_or_else(|| format!("Perplexity error: {}", crate::jsv::js_string(&ev["error_code"])));
                yield Ev::Error(m);
                return;
            }
            if let Some(b) = ev["backend_uuid"].as_str().filter(|s| !s.is_empty()) { yield Ev::Backend(b.to_string()); }
            let blocks = ev["blocks"].as_array().cloned().unwrap_or_default();
            for block in &blocks {
                let usage = block["intended_usage"].as_str().unwrap_or("");
                if usage == "pro_search_steps" {
                    for step in block["plan_block"]["steps"].as_array().into_iter().flatten() {
                        if step["step_type"] == "SEARCH_WEB" {
                            for q in step["search_web_content"]["queries"].as_array().into_iter().flatten() {
                                let qr = q["query"].as_str().unwrap_or("");
                                if !qr.is_empty() && seen.insert(qr.to_string()) { yield Ev::Thinking(format!("Searching: {qr}")); }
                            }
                        } else if step["step_type"] == "READ_RESULTS" {
                            for u in step["read_results_content"]["urls"].as_array().into_iter().flatten().take(3) {
                                if let Some(u) = u.as_str().filter(|s| !s.is_empty()) {
                                    if seen.insert(u.to_string()) { yield Ev::Thinking(format!("Reading: {u}")); }
                                }
                            }
                        }
                    }
                }
                if usage == "plan" {
                    for g in block["plan_block"]["goals"].as_array().into_iter().flatten() {
                        let d = g["description"].as_str().unwrap_or("");
                        if !d.is_empty() && seen.insert(d.to_string()) { yield Ev::Thinking(d.to_string()); }
                    }
                }
                if !usage.contains("markdown") { continue; }
                let mb = &block["markdown_block"];
                let chunks: Vec<String> = mb["chunks"].as_array().into_iter().flatten().map(crate::jsv::js_string).collect();
                if chunks.is_empty() { continue; }
                if mb["progress"] == "DONE" {
                    full = chunks.join("");
                } else {
                    let cumulative = format!("{full}{}", chunks.join(""));
                    let cl = cumulative.chars().count();
                    if cl > seen_len {
                        let delta: String = cumulative.chars().skip(seen_len).collect();
                        full = cumulative;
                        seen_len = cl;
                        yield Ev::Delta(delta, full.clone());
                    }
                }
            }
            if blocks.is_empty() {
                if let Some(t) = ev["text"].as_str().map(str::trim).filter(|s| !s.is_empty()) {
                    let tl = t.chars().count();
                    if tl > seen_len {
                        let delta: String = t.chars().skip(seen_len).collect();
                        full = t.to_string();
                        seen_len = tl;
                        yield Ev::Delta(delta, full.clone());
                    }
                }
            }
            if truthy(&ev["final"]) || ev["status"] == "COMPLETED" { break; }
        }
        yield Ev::Done(full);
    }
}

fn err_json(status: u16, msg: &str, code: Option<&str>) -> Upstream {
    let mut e = json!({"message": msg, "type": if status == 400 { "invalid_request" } else { "upstream_error" }});
    if let Some(c) = code {
        e["code"] = json!(c);
    }
    Upstream::json(status, &json!({"error": e}))
}

pub struct PerplexityWeb;

#[async_trait]
impl Executor for PerplexityWeb {
    fn provider(&self) -> &str {
        "perplexity-web"
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let url = self.config()["baseUrl"].as_str().unwrap_or("https://www.perplexity.ai/rest/sse/perplexity_ask").to_string();
        let res = |up: Upstream, h: Vec<(String, String)>, body: Value| Ok(ExecResult { response: up, url: url.clone(), headers: h, body, response_format: Some("openai".into()) });
        let Some(msgs) = args.body["messages"].as_array().filter(|a| !a.is_empty()) else {
            return res(err_json(400, "Missing or empty messages array", None), vec![], args.body.clone());
        };
        let b = &args.body;
        let thinking = b["thinking"] == json!(true) || (!b["reasoning_effort"].is_null() && b["reasoning_effort"] != "none");
        let model = args.model;
        let (mode, pref) = match (thinking, thinking_map(model), model_map(model)) {
            (true, Some(t), _) => ("copilot".to_string(), t.to_string()),
            (_, _, Some((m, p))) => (m.to_string(), p.to_string()),
            _ => ("copilot".to_string(), model.to_string()),
        };
        let parsed = parse_messages(msgs);
        let follow = session_lookup(&parsed.history);
        let query = build_query(&parsed, follow.as_deref(), &b["tools"]);
        if query.trim().is_empty() {
            return res(err_json(400, "Empty query after processing", None), vec![], args.body.clone());
        }
        let body = build_body(&query, &mode, &pref, follow.as_deref());
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.set("Accept", "text/event-stream");
        h.set("Origin", "https://www.perplexity.ai");
        h.set("Referer", "https://www.perplexity.ai/");
        h.set("User-Agent", UA);
        h.set("X-App-ApiClient", "default");
        h.set("X-App-ApiVersion", API_VERSION);
        if let Some(t) = crate::exec::cred_str(args.creds, "accessToken") {
            h.set("Authorization", format!("Bearer {t}"));
        } else if let Some(k) = crate::exec::cred_str(args.creds, "apiKey") {
            h.set("Cookie", format!("__Secure-next-auth.session-token={k}"));
        }
        let up = match post_json(args.creds, &url, &h, &body, 60_000).await {
            Ok(u) => u,
            Err(e) => return res(err_json(502, &format!("Perplexity connection failed: {e}"), None), h.0, body),
        };
        if !up.ok() {
            let st = up.status;
            let msg = match st {
                401 | 403 => "Perplexity auth failed — session cookie may be expired. Re-paste your __Secure-next-auth.session-token.".to_string(),
                429 => "Perplexity rate limited. Wait a moment and retry.".to_string(),
                _ => format!("Perplexity returned HTTP {st}"),
            };
            return res(err_json(st, &msg, Some(&format!("HTTP_{st}"))), h.0, body);
        }
        let cid = format!("chatcmpl-pplx-{}", &uuid::Uuid::new_v4().to_string()[..12]);
        let created = now_ms() / 1000;
        let model = model.to_string();
        let events = extract(up.body);
        let history = parsed.history.clone();
        let current = parsed.current.clone();
        if args.stream {
            let s = async_stream::stream! {
                let chunk = |delta: Value, fin: Value| json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "system_fingerprint": null, "choices": [{"index": 0, "delta": delta, "finish_reason": fin, "logprobs": null}]});
                yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", chunk(json!({"role": "assistant"}), Value::Null))));
                let mut full = String::new();
                let mut backend: Option<String> = None;
                futures::pin_mut!(events);
                while let Some(ev) = events.next().await {
                    match ev {
                        Ev::Backend(b) => backend = Some(b),
                        Ev::Error(e) => { yield Ok(Bytes::from(format!("data: {}\n\n", chunk(json!({"content": format!("[Error: {e}]")}), Value::Null)))); break; }
                        Ev::Thinking(t) => yield Ok(Bytes::from(format!("data: {}\n\n", chunk(json!({"reasoning_content": format!("{t}\n")}), Value::Null)))),
                        Ev::Done(a) => { if !a.is_empty() { full = a; } break; }
                        Ev::Delta(d, a) => {
                            let dt = clean_response(&d, false);
                            if !dt.is_empty() { yield Ok(Bytes::from(format!("data: {}\n\n", chunk(json!({"content": dt}), Value::Null)))); }
                            if !a.is_empty() { full = a; }
                        }
                    }
                }
                yield Ok(Bytes::from(format!("data: {}\n\n", chunk(json!({}), json!("stop")))));
                yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                session_store(&history, &current, &clean_response(&full, true), backend.as_deref());
            };
            return res(Upstream::synthetic(200, "text/event-stream", s.boxed()), h.0, body);
        }
        let mut full = String::new();
        let mut backend: Option<String> = None;
        let mut thinking_parts = vec![];
        futures::pin_mut!(events);
        while let Some(ev) = events.next().await {
            match ev {
                Ev::Backend(b) => backend = Some(b),
                Ev::Error(e) => return res(Upstream::json(502, &json!({"error": {"message": e, "type": "upstream_error", "code": "PPLX_ERROR"}})), h.0, body),
                Ev::Thinking(t) => thinking_parts.push(t),
                Ev::Done(a) => {
                    if !a.is_empty() {
                        full = a;
                    }
                    break;
                }
                Ev::Delta(_, a) => {
                    if !a.is_empty() {
                        full = a;
                    }
                }
            }
        }
        let full = clean_response(&full, true);
        session_store(&parsed.history, &parsed.current, &full, backend.as_deref());
        let mut msg = json!({"role": "assistant", "content": full});
        if !thinking_parts.is_empty() {
            msg["reasoning_content"] = json!(thinking_parts.join("\n"));
        }
        let pt = (parsed.current.encode_utf16().count() as f64 / 4.0).ceil() as i64;
        let ct = (full.encode_utf16().count() as f64 / 4.0).ceil() as i64;
        let out = json!({
            "id": cid, "object": "chat.completion", "created": created, "model": args.model, "system_fingerprint": null,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop", "logprobs": null}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        });
        res(Upstream::json(200, &out), h.0, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_and_clean() {
        let m = json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]);
        let p = parse_messages(m.as_array().unwrap());
        assert_eq!(p.current, "q2");
        assert_eq!(p.history.len(), 2);
        let q: Value = serde_json::from_str(&build_query(&p, None, &Value::Null)).unwrap();
        assert_eq!(q["instructions"][0], "be brief");
        assert_eq!(q["query"], "q2");
        assert_eq!(clean_response("Hello [1] world  <grok:x>y</grok:x>\n\n\n\nend", true), "Hello world \n\nend");
        assert_eq!(session_key(&[]), "811c9dc5");
    }
}
