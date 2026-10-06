//! Cursor IDE executor — ConnectRPC protobuf over HTTP/2.
//! Port of executors/cursor.js, utils/cursorProtobuf.js, utils/cursorChecksum.js
//! and the openai→cursor request translator.
//!
//! Two upstream paths:
//! * AgentService.Run (agent.api5.cursor.sh, bidirectional h2 stream) for
//!   text-only conversations, with MCP tool declarations;
//! * ChatService.StreamUnifiedChatWithTools (api2.cursor.sh) otherwise.

use std::collections::HashMap;
use std::io::Read;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use serde_json::{Map, Value, json};

use crate::exec::{ExecArgs, ExecResult, Executor, Upstream, client_for};
use crate::jsv::{now_ms, truthy};
use crate::translate::ReqCtx;

// ===========================================================================
// protobuf primitives
// ===========================================================================

const VARINT: u8 = 0;
const FIXED64: u8 = 1;
const LEN: u8 = 2;

pub fn varint(mut v: u64) -> Vec<u8> {
    let mut out = vec![];
    while v >= 0x80 {
        out.push((v as u8 & 0x7f) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
    out
}

pub fn field_varint(n: u32, v: u64) -> Vec<u8> {
    let mut o = varint(((n << 3) | VARINT as u32) as u64);
    o.extend(varint(v));
    o
}

pub fn field_bytes(n: u32, v: &[u8]) -> Vec<u8> {
    let mut o = varint(((n << 3) | LEN as u32) as u64);
    o.extend(varint(v.len() as u64));
    o.extend_from_slice(v);
    o
}

pub fn field_str(n: u32, v: &str) -> Vec<u8> {
    field_bytes(n, v.as_bytes())
}

fn field_double(n: u32, v: f64) -> Vec<u8> {
    let mut o = varint(((n << 3) | FIXED64 as u32) as u64);
    o.extend_from_slice(&v.to_le_bytes());
    o
}

fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.concat()
}

#[derive(Clone, Debug)]
pub enum Pb {
    Int(u64),
    Bytes(Vec<u8>),
}

impl Pb {
    pub fn bytes(&self) -> &[u8] {
        match self {
            Pb::Bytes(b) => b,
            Pb::Int(_) => &[],
        }
    }
    pub fn int(&self) -> u64 {
        match self {
            Pb::Int(i) => *i,
            Pb::Bytes(_) => 0,
        }
    }
}

/// Decoded message: field number → values in order (plus field order).
#[derive(Default, Debug)]
pub struct Msg {
    pub fields: HashMap<u32, Vec<Pb>>,
    pub order: Vec<u32>,
}

impl Msg {
    pub fn has(&self, f: u32) -> bool {
        self.fields.contains_key(&f)
    }
    pub fn first(&self, f: u32) -> Option<&Pb> {
        self.fields.get(&f).and_then(|v| v.first())
    }
    pub fn str(&self, f: u32) -> String {
        self.first(f).map(|p| String::from_utf8_lossy(p.bytes()).into_owned()).unwrap_or_default()
    }
    pub fn all(&self, f: u32) -> &[Pb] {
        self.fields.get(&f).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

fn decode_varint(b: &[u8], mut pos: usize) -> (u64, usize) {
    let mut r: u64 = 0;
    let mut shift = 0;
    while pos < b.len() {
        let x = b[pos];
        if shift < 64 {
            r |= ((x & 0x7f) as u64) << shift;
        }
        pos += 1;
        if x & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (r, pos)
}

pub fn decode(b: &[u8]) -> Msg {
    let mut m = Msg::default();
    let mut pos = 0;
    while pos < b.len() {
        let (tag, p1) = decode_varint(b, pos);
        let field = (tag >> 3) as u32;
        let wt = (tag & 7) as u8;
        let (val, np) = match wt {
            0 => {
                let (v, p) = decode_varint(b, p1);
                (Pb::Int(v), p)
            }
            2 => {
                let (l, p2) = decode_varint(b, p1);
                let end = (p2 + l as usize).min(b.len());
                (Pb::Bytes(b[p2..end].to_vec()), end)
            }
            1 => {
                let end = (p1 + 8).min(b.len());
                (Pb::Bytes(b[p1..end].to_vec()), end)
            }
            5 => {
                let end = (p1 + 4).min(b.len());
                (Pb::Bytes(b[p1..end].to_vec()), end)
            }
            _ => break,
        };
        if np <= pos {
            break;
        }
        if !m.fields.contains_key(&field) {
            m.order.push(field);
        }
        m.fields.entry(field).or_default().push(val);
        pos = np;
    }
    m
}

pub fn wrap_frame(payload: &[u8]) -> Vec<u8> {
    let mut f = vec![0u8];
    f.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    f.extend_from_slice(payload);
    f
}

fn gunzip(b: &[u8]) -> Option<Vec<u8>> {
    let mut out = vec![];
    flate2::read::GzDecoder::new(b).read_to_end(&mut out).ok().map(|_| out)
}

fn decompress(payload: &[u8], flags: u8) -> Vec<u8> {
    if payload.len() > 10 && payload[0] == b'{' && payload[1] == b'"' && payload.starts_with(b"{\"error\"") {
        return payload.to_vec();
    }
    if (1..=3).contains(&flags) {
        if let Some(o) = gunzip(payload) {
            return o;
        }
        let mut out = vec![];
        if flate2::read::ZlibDecoder::new(payload).read_to_end(&mut out).is_ok() {
            return out;
        }
        let mut out = vec![];
        if flate2::read::DeflateDecoder::new(payload).read_to_end(&mut out).is_ok() {
            return out;
        }
    }
    payload.to_vec()
}

// ===========================================================================
// ChatService request encoding
// ===========================================================================

fn format_tool_name(name: &str) -> String {
    let base = if name.is_empty() { "tool" } else { name };
    if let Some(rest) = base.strip_prefix("mcp__") {
        if let Some(i) = rest.find("__") {
            let server = if rest[..i].is_empty() { "custom" } else { &rest[..i] };
            let tool = if rest[i + 2..].is_empty() { "tool" } else { &rest[i + 2..] };
            return format!("mcp_{server}_{tool}");
        }
        return format!("mcp_custom_{}", if rest.is_empty() { "tool" } else { rest });
    }
    if base.starts_with("mcp_") {
        return base.to_string();
    }
    format!("mcp_custom_{base}")
}

fn parse_tool_name(f: &str) -> (String, String) {
    let Some(tail) = f.strip_prefix("mcp_") else { return ("custom".into(), if f.is_empty() { "tool".into() } else { f.into() }) };
    match tail.find('_') {
        None => ("custom".into(), if tail.is_empty() { "tool".into() } else { tail.into() }),
        Some(i) => (if tail[..i].is_empty() { "custom".into() } else { tail[..i].into() }, if tail[i + 1..].is_empty() { "tool".into() } else { tail[i + 1..].into() }),
    }
}

fn parse_tool_id(id: &str) -> (String, Option<String>) {
    match id.find("\nmc_") {
        Some(i) => (id[..i].into(), Some(id[i + 4..].into())),
        None => (id.into(), None),
    }
}

fn encode_mcp_result(sel: &str, res: &str) -> Vec<u8> {
    cat(&[field_str(1, sel), field_str(2, res)])
}

fn encode_tool_result(tr: &Value) -> Vec<u8> {
    let orig = tr["tool_name"].as_str().or_else(|| tr["name"].as_str()).unwrap_or("");
    let name = format_tool_name(orig);
    let raw_args = tr["raw_args"].as_str().filter(|s| !s.is_empty()).unwrap_or("{}");
    let result = tr["result_content"].as_str().or_else(|| tr["result"].as_str()).unwrap_or("");
    let (call_id, model_call) = parse_tool_id(tr["tool_call_id"].as_str().unwrap_or(""));
    let idx = tr["tool_index"].as_u64().or_else(|| tr["index"].as_u64()).filter(|i| *i > 0).unwrap_or(1);
    let (server, sel) = parse_tool_name(&name);
    let mut cv2r = vec![field_varint(1, 19), field_bytes(28, &encode_mcp_result(&sel, result)), field_str(35, &call_id)];
    if let Some(m) = &model_call {
        cv2r.push(field_str(48, m));
    }
    cv2r.push(field_varint(49, idx));
    let mcp_tool = cat(&[field_str(1, &sel), field_str(3, raw_args), field_str(4, &server)]);
    let mut cv2c = vec![field_varint(1, 19), field_bytes(27, &field_bytes(1, &mcp_tool)), field_str(3, &call_id), field_str(9, &name), field_str(10, raw_args), field_varint(48, idx)];
    if let Some(m) = &model_call {
        cv2c.push(field_str(49, m));
    }
    let mut parts = vec![field_str(1, &call_id), field_str(2, &name), field_varint(3, idx)];
    if let Some(m) = &model_call {
        parts.push(field_str(12, m));
    }
    parts.push(field_str(5, raw_args));
    parts.push(field_bytes(8, &cat(&cv2r)));
    parts.push(field_bytes(11, &cat(&cv2c)));
    cat(&parts)
}

fn content_text(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter(|p| p["type"] == "text").map(|p| p["text"].as_str().unwrap_or("")).collect::<Vec<_>>().join(""),
        Value::Null => String::new(),
        o => crate::jsv::js_string(o),
    }
}

fn encode_message(content: &str, role: u64, id: &str, is_last: bool, has_tools: bool, results: &[Value]) -> Vec<u8> {
    let mut p = vec![field_str(1, content), field_varint(2, role), field_str(13, id)];
    for r in results {
        p.push(field_bytes(18, &encode_tool_result(r)));
    }
    p.push(field_varint(29, has_tools as u64));
    p.push(field_varint(47, if has_tools { 2 } else { 1 }));
    if is_last && has_tools {
        p.push(field_bytes(51, &varint(1)));
    }
    cat(&p)
}

fn encode_mcp_tool(t: &Value) -> Vec<u8> {
    let name = t["function"]["name"].as_str().or_else(|| t["name"].as_str()).unwrap_or("");
    let desc = t["function"]["description"].as_str().or_else(|| t["description"].as_str()).unwrap_or("");
    let schema = if truthy(&t["function"]["parameters"]) { &t["function"]["parameters"] } else if truthy(&t["input_schema"]) { &t["input_schema"] } else { &Value::Null };
    let mut p = vec![];
    if !name.is_empty() {
        p.push(field_str(1, name));
    }
    if !desc.is_empty() {
        p.push(field_str(2, desc));
    }
    if schema.as_object().map(|o| !o.is_empty()).unwrap_or(false) {
        p.push(field_str(3, &schema.to_string()));
    }
    p.push(field_str(4, "custom"));
    cat(&p)
}

fn encode_metadata() -> Vec<u8> {
    cat(&[
        field_str(1, crate::consts::node_platform()),
        field_str(2, crate::consts::node_arch()),
        field_str(3, "v20.0.0"),
        field_str(4, &std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| "/".into())),
        field_str(5, &crate::jsv::iso_from_ms(now_ms())),
    ])
}

/// encodeRequest(messages, model, tools, reasoningEffort, forceAgentMode)
pub fn encode_request(messages: &[Value], model: &str, tools: &[Value], effort: Option<&str>, force_agent: bool) -> Vec<u8> {
    let has_tools = !tools.is_empty();
    let agentic = has_tools || force_agent;
    let mut norm: Vec<Value> = vec![];
    for (i, m) in messages.iter().enumerate() {
        let calls = m["tool_calls"].as_array().map(|a| !a.is_empty()).unwrap_or(false);
        let results = m["tool_results"].as_array().filter(|a| !a.is_empty());
        if m["role"] == "assistant" && calls && results.is_some() {
            let res = results.unwrap().clone();
            let mut a = m.clone();
            a["tool_results"] = json!([]);
            norm.push(a);
            let next = messages.get(i + 1);
            let ids = |r: &[Value]| r.iter().filter_map(|t| t["tool_call_id"].as_str().map(str::to_owned)).collect::<std::collections::HashSet<_>>();
            let cur = ids(&res);
            let nxt = next.and_then(|n| n["tool_results"].as_array()).map(|a| ids(a)).unwrap_or_default();
            let next_has = next.map(|n| n["role"] == "assistant" && n["tool_results"].as_array().map(|a| !a.is_empty()).unwrap_or(false)).unwrap_or(false);
            let same = !cur.is_empty() && cur == nxt;
            if !(next_has && same) {
                norm.push(json!({"role": "assistant", "content": "", "tool_results": res}));
            }
            continue;
        }
        norm.push(m.clone());
    }
    let mut msgs = vec![];
    let mut ids = vec![];
    for (i, m) in norm.iter().enumerate() {
        let role = if m["role"] == "user" { 1 } else { 2 };
        let id = uuid::Uuid::new_v4().to_string();
        let results = m["tool_results"].as_array().cloned().unwrap_or_default();
        msgs.push(field_bytes(1, &encode_message(&content_text(&m["content"]), role, &id, i == norm.len() - 1, has_tools, &results)));
        ids.push((id, role));
    }
    let thinking = match effort {
        Some("medium") => 1,
        Some("high") => 2,
        _ => 0,
    };
    let setting = cat(&[field_str(1, "cursor\\aisettings"), field_bytes(3, &[]), field_bytes(6, &cat(&[field_bytes(1, &[]), field_bytes(2, &[])])), field_varint(8, 1), field_varint(9, 1)]);
    let mut p = msgs;
    p.push(field_varint(2, 1));
    p.push(field_bytes(3, &[]));
    p.push(field_varint(4, 1));
    p.push(field_bytes(5, &cat(&[field_str(1, model), field_bytes(4, &[])])));
    p.push(field_str(8, ""));
    p.push(field_varint(13, 1));
    p.push(field_bytes(15, &setting));
    p.push(field_varint(19, 1));
    p.push(field_str(23, &uuid::Uuid::new_v4().to_string()));
    p.push(field_bytes(26, &encode_metadata()));
    p.push(field_varint(27, agentic as u64));
    if agentic {
        p.push(field_bytes(29, &varint(1)));
    }
    for (id, role) in &ids {
        p.push(field_bytes(30, &cat(&[field_str(1, id), field_varint(3, *role)])));
    }
    for t in tools {
        p.push(field_bytes(34, &encode_mcp_tool(t)));
    }
    p.push(field_varint(35, 0));
    p.push(field_varint(38, 0));
    p.push(field_varint(46, if agentic { 2 } else { 1 }));
    p.push(field_str(47, ""));
    p.push(field_varint(48, if agentic { 0 } else { 1 }));
    p.push(field_varint(49, thinking));
    p.push(field_varint(51, 0));
    p.push(field_varint(53, 1));
    p.push(field_str(54, if agentic { "Agent" } else { "Ask" }));
    cat(&p)
}

pub fn generate_cursor_body(messages: &[Value], model: &str, tools: &[Value], effort: Option<&str>, force_agent: bool) -> Vec<u8> {
    wrap_frame(&field_bytes(1, &encode_request(messages, model, tools, effort, force_agent)))
}

// ===========================================================================
// ChatService response decoding
// ===========================================================================

#[derive(Default, Debug)]
pub struct Extracted {
    pub text: Option<String>,
    pub thinking: Option<String>,
    pub tool_call: Option<(String, String, String, bool)>,
}

fn extract_tool_call(b: &[u8]) -> Option<(String, String, String, bool)> {
    let tc = decode(b);
    let id = if tc.has(3) { tc.str(3).split('\n').next().unwrap_or("").to_string() } else { String::new() };
    let mut name = tc.str(9);
    let is_last = tc.first(11).map(|p| p.int() != 0).unwrap_or(false);
    let mut args = String::new();
    if let Some(mp) = tc.first(27) {
        let mp = decode(mp.bytes());
        if let Some(t) = mp.first(1) {
            let t = decode(t.bytes());
            if t.has(1) {
                name = t.str(1);
            }
            if t.has(3) {
                args = t.str(3);
            }
        }
    }
    if args.is_empty() && tc.has(10) {
        args = tc.str(10);
    }
    (!id.is_empty() && !name.is_empty()).then(|| (id, name, if args.is_empty() { "{}".into() } else { args }, is_last))
}

pub fn extract_text_from_response(payload: &[u8]) -> Extracted {
    let f = decode(payload);
    if let Some(tc) = f.first(1) {
        if let Some(t) = extract_tool_call(tc.bytes()) {
            return Extracted { tool_call: Some(t), ..Default::default() };
        }
    }
    if let Some(r) = f.first(2) {
        let n = decode(r.bytes());
        let text = n.has(1).then(|| n.str(1));
        let thinking = n.first(25).map(|t| decode(t.bytes())).filter(|t| t.has(1)).map(|t| t.str(1));
        if text.as_deref().map(|s| !s.is_empty()).unwrap_or(false) || thinking.as_deref().map(|s| !s.is_empty()).unwrap_or(false) {
            return Extracted { text, thinking, tool_call: None };
        }
    }
    Extracted::default()
}

fn is_composer(model: &str) -> bool {
    let id = model.rsplit('/').next().unwrap_or(model).to_lowercase();
    id == "composer" || id.starts_with("composer-")
}

fn visible_from_thinking(t: &str) -> String {
    match t.rfind("</think>") {
        Some(i) => t[i + 8..].trim_start().to_string(),
        None => String::new(),
    }
}

fn error_response(j: &Value) -> Upstream {
    let d = &j["error"]["details"][0]["debug"];
    let msg = [&d["details"]["title"], &d["details"]["detail"], &j["error"]["message"]].into_iter().find(|v| truthy(v)).map(crate::jsv::js_string).unwrap_or_else(|| "API Error".into());
    let rl = j["error"]["code"] == "resource_exhausted";
    Upstream::json(if rl { 429 } else { 400 }, &json!({"error": {"message": msg, "type": if rl { "rate_limit_error" } else { "api_error" }, "code": if truthy(&d["error"]) { d["error"].clone() } else { json!("unknown") }}}))
}

/// usageTracking.estimateUsage (OpenAI shape, no buffer).
pub fn estimate_usage(body: &Value, content_len: usize) -> Value {
    let input = if body.is_object() { (body.to_string().encode_utf16().count() as f64 / 4.0).ceil() as i64 } else { 0 };
    let output = if content_len == 0 { 0 } else { (content_len / 4).max(1) as i64 };
    json!({"prompt_tokens": input, "completion_tokens": output, "total_tokens": input + output, "estimated": true})
}

enum Frames {
    Error(Upstream),
    Ok(Vec<Vec<u8>>),
}

fn split_frames(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut out = vec![];
    let mut off = 0;
    while off + 5 <= buf.len() {
        let flags = buf[off];
        let len = u32::from_be_bytes([buf[off + 1], buf[off + 2], buf[off + 3], buf[off + 4]]) as usize;
        if off + 5 + len > buf.len() {
            break;
        }
        out.push(decompress(&buf[off + 5..off + 5 + len], flags));
        off += 5 + len;
    }
    out
}

struct ToolAcc {
    id: String,
    name: String,
    args: String,
    index: usize,
}

fn chunk(id: &str, created: i64, model: &str, delta: Value, fin: Value) -> String {
    format!("data: {}\n\n", json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}))
}

fn check_error_frame(p: &[u8], has_content: bool) -> Option<Option<Frames>> {
    if p.first() == Some(&b'{') {
        let text = String::from_utf8_lossy(p);
        if text.contains("\"error\"") {
            if has_content {
                return Some(None);
            }
            if let Ok(j) = serde_json::from_str::<Value>(&text) {
                return Some(Some(Frames::Error(error_response(&j))));
            }
        }
    }
    None
}

pub fn protobuf_to_sse(buf: &[u8], model: &str, body: &Value) -> Upstream {
    let id = format!("chatcmpl-cursor-{}", now_ms());
    let created = now_ms() / 1000;
    let mut chunks: Vec<String> = vec![];
    let mut total = String::new();
    let mut thinking = String::new();
    let mut emitted_comp = 0usize;
    let mut tools: Vec<ToolAcc> = vec![];
    for p in split_frames(buf) {
        let has_content = !chunks.is_empty() || !total.is_empty() || !tools.is_empty();
        match check_error_frame(&p, has_content) {
            Some(None) => break,
            Some(Some(Frames::Error(u))) => return u,
            _ => {}
        }
        let r = extract_text_from_response(&p);
        if let Some((tid, name, args, _last)) = r.tool_call {
            if chunks.is_empty() {
                chunks.push(chunk(&id, created, model, json!({"role": "assistant", "content": ""}), Value::Null));
            }
            if let Some(t) = tools.iter_mut().find(|t| t.id == tid) {
                t.args.push_str(&args);
                if !args.is_empty() {
                    chunks.push(chunk(&id, created, model, json!({"tool_calls": [{"index": t.index, "id": tid, "type": "function", "function": {"name": name, "arguments": args}}]}), Value::Null));
                }
            } else {
                let idx = tools.len();
                chunks.push(chunk(&id, created, model, json!({"tool_calls": [{"index": idx, "id": tid, "type": "function", "function": {"name": name, "arguments": args}}]}), Value::Null));
                tools.push(ToolAcc { id: tid, name, args, index: idx });
            }
        }
        if let Some(t) = r.text.filter(|s| !s.is_empty()) {
            total.push_str(&t);
            let d = if chunks.is_empty() && tools.is_empty() { json!({"role": "assistant", "content": t}) } else { json!({"content": t}) };
            chunks.push(chunk(&id, created, model, d, Value::Null));
        }
        if is_composer(model) {
            if let Some(th) = r.thinking.filter(|s| !s.is_empty()) {
                thinking.push_str(&th);
                let vis = visible_from_thinking(&thinking);
                if vis.len() > emitted_comp {
                    let delta = vis[emitted_comp..].to_string();
                    emitted_comp = vis.len();
                    total.push_str(&delta);
                    let d = if chunks.is_empty() && tools.is_empty() { json!({"role": "assistant", "content": delta}) } else { json!({"content": delta}) };
                    chunks.push(chunk(&id, created, model, d, Value::Null));
                }
            }
        }
    }
    let _ = tools.iter().map(|t| &t.name).count();
    if chunks.is_empty() && tools.is_empty() {
        chunks.push(chunk(&id, created, model, json!({"role": "assistant", "content": ""}), Value::Null));
    }
    let usage = estimate_usage(body, total.encode_utf16().count());
    chunks.push(format!(
        "data: {}\n\n",
        json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": if tools.is_empty() { "stop" } else { "tool_calls" }}], "usage": usage})
    ));
    chunks.push("data: [DONE]\n\n".into());
    let all = Bytes::from(chunks.concat());
    Upstream::synthetic(200, "text/event-stream", futures::stream::once(async move { Ok(all) }).boxed())
}

pub fn protobuf_to_json(buf: &[u8], model: &str, body: &Value) -> Upstream {
    let id = format!("chatcmpl-cursor-{}", now_ms());
    let created = now_ms() / 1000;
    let mut total = String::new();
    let mut thinking = String::new();
    let mut map: Vec<(String, String, String, bool)> = vec![];
    let mut done: Vec<Value> = vec![];
    let mut finalized = std::collections::HashSet::new();
    for p in split_frames(buf) {
        let has_content = !total.is_empty() || !map.is_empty();
        match check_error_frame(&p, has_content) {
            Some(None) => break,
            Some(Some(Frames::Error(u))) => return u,
            _ => {}
        }
        let r = extract_text_from_response(&p);
        if let Some((tid, name, args, last)) = r.tool_call {
            if let Some(e) = map.iter_mut().find(|e| e.0 == tid) {
                e.2.push_str(&args);
                e.3 = last;
            } else {
                map.push((tid.clone(), name, args, last));
            }
            if last {
                let e = map.iter().find(|e| e.0 == tid).unwrap();
                finalized.insert(tid.clone());
                done.push(json!({"id": e.0, "type": "function", "function": {"name": e.1, "arguments": e.2}}));
            }
        }
        if let Some(t) = r.text {
            total.push_str(&t);
        }
        if let Some(t) = r.thinking {
            thinking.push_str(&t);
        }
    }
    let vis = if is_composer(model) { visible_from_thinking(&thinking) } else { String::new() };
    let content = if total.is_empty() { vis } else { total };
    for e in &map {
        if !finalized.contains(&e.0) {
            done.push(json!({"id": e.0, "type": "function", "function": {"name": e.1, "arguments": e.2}}));
        }
    }
    let mut msg = json!({"role": "assistant", "content": if content.is_empty() { Value::Null } else { json!(content) }});
    if !done.is_empty() {
        msg["tool_calls"] = Value::Array(done.clone());
    }
    let out = json!({"id": id, "object": "chat.completion", "created": created, "model": model, "choices": [{"index": 0, "message": msg, "finish_reason": if done.is_empty() { "stop" } else { "tool_calls" }}], "usage": estimate_usage(body, content.encode_utf16().count())});
    Upstream::json(200, &out)
}

// ===========================================================================
// headers (cursorChecksum.js)
// ===========================================================================

pub fn cursor_checksum(machine_id: &str) -> String {
    let ts = (now_ms() / 1_000_000) as u64;
    let mut b = [(ts >> 40) as u8, (ts >> 32) as u8, (ts >> 24) as u8, (ts >> 16) as u8, (ts >> 8) as u8, ts as u8];
    let mut t: u8 = 165;
    for (i, x) in b.iter_mut().enumerate() {
        *x = ((*x ^ t) as u16 + (i % 256) as u16) as u8;
        t = *x;
    }
    use base64::Engine;
    let enc = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b);
    format!("{enc}{machine_id}")
}

pub fn cursor_headers(access_token: &str, machine_id: &str, ghost: bool) -> crate::exec::Headers {
    let clean = access_token.split_once("::").map(|(_, b)| b).unwrap_or(access_token);
    let mid = if machine_id.is_empty() { crate::jsv::sha256_hex(&format!("{clean}machineId")) } else { machine_id.to_string() };
    let session = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, clean.as_bytes()).to_string();
    let os = match std::env::consts::OS {
        "windows" => "windows",
        "macos" => "macos",
        _ => "linux",
    };
    let arch = if std::env::consts::ARCH == "aarch64" { "aarch64" } else { "x64" };
    let mut h = crate::exec::Headers::default();
    for (k, v) in [
        ("authorization", format!("Bearer {clean}")),
        ("connect-accept-encoding", "gzip".into()),
        ("connect-protocol-version", "1".into()),
        ("content-type", "application/connect+proto".into()),
        ("user-agent", "connect-es/1.6.1".into()),
        ("x-amzn-trace-id", format!("Root={}", uuid::Uuid::new_v4())),
        ("x-client-key", crate::jsv::sha256_hex(clean)),
        ("x-cursor-checksum", cursor_checksum(&mid)),
        ("x-cursor-client-version", "3.12.17".into()),
        ("x-cursor-client-commit", "0fb762053c34788bb7760d5673f8a6d4c8589d50".into()),
        ("x-cursor-client-type", "ide".into()),
        ("x-cursor-client-os", os.into()),
        ("x-cursor-client-arch", arch.into()),
        ("x-cursor-client-device-type", "desktop".into()),
        ("x-cursor-config-version", uuid::Uuid::new_v4().to_string()),
        ("x-cursor-timezone", std::env::var("TZ").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "UTC".into())),
        ("x-ghost-mode", if ghost { "true" } else { "false" }.into()),
        ("x-request-id", uuid::Uuid::new_v4().to_string()),
        ("x-session-id", session),
    ] {
        h.set(k, v);
    }
    h
}

// ===========================================================================
// request translator (openai-to-cursor.js) + response passthrough
// ===========================================================================

fn sanitize_tool_text(t: &str) -> String {
    t.chars().filter(|c| !matches!(*c as u32, 0..=8 | 0x0b | 0x0c | 0x0e..=0x1f | 0x7f)).collect()
}

fn esc_xml(t: &str) -> String {
    t.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn tool_result_block(name: &str, id: &str, text: &str) -> String {
    format!(
        "<tool_result>\n<tool_name>{}</tool_name>\n<tool_call_id>{}</tool_call_id>\n<result>{}</result>\n</tool_result>",
        esc_xml(if name.is_empty() { "tool" } else { name }),
        esc_xml(id),
        esc_xml(&sanitize_tool_text(text))
    )
}

fn extract_content(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter(|p| p.is_object() && p["type"] == "text" && p["text"].is_string()).map(|p| p["text"].as_str().unwrap()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

pub fn openai_to_cursor(_model: &str, body: &Value, _stream: bool, _rc: &ReqCtx) -> Value {
    let msgs = body["messages"].as_array().cloned().unwrap_or_default();
    let mut meta: HashMap<String, String> = HashMap::new();
    let mut remember = |id: &str, name: &str| {
        if id.is_empty() {
            return;
        }
        let n = if name.is_empty() { "tool" } else { name };
        meta.insert(id.to_string(), n.to_string());
        let norm = id.split('\n').next().unwrap_or("");
        if !norm.is_empty() && norm != id {
            meta.insert(norm.to_string(), n.to_string());
        }
    };
    for m in &msgs {
        if m["role"] == "assistant" {
            for tc in m["tool_calls"].as_array().into_iter().flatten() {
                remember(tc["id"].as_str().unwrap_or(""), tc["function"]["name"].as_str().unwrap_or("tool"));
            }
            for p in m["content"].as_array().into_iter().flatten() {
                if p["type"] == "tool_use" {
                    remember(p["id"].as_str().unwrap_or(""), p["name"].as_str().unwrap_or("tool"));
                }
            }
        }
    }
    let mut out = vec![];
    for m in &msgs {
        let role = m["role"].as_str().unwrap_or("");
        match role {
            "system" => out.push(json!({"role": "user", "content": format!("[System Instructions]\n{}", extract_content(&m["content"]))})),
            "tool" => {
                let id = m["tool_call_id"].as_str().unwrap_or("");
                let name = m["name"].as_str().map(str::to_owned).or_else(|| meta.get(id).cloned()).unwrap_or_else(|| "tool".into());
                out.push(json!({"role": "user", "content": tool_result_block(&name, id, &extract_content(&m["content"]))}));
            }
            "user" if m["content"].is_array() => {
                let mut parts = vec![];
                for b in m["content"].as_array().unwrap() {
                    if !b.is_object() {
                        continue;
                    }
                    if b["type"] == "text" {
                        if let Some(t) = b["text"].as_str() {
                            parts.push(t.to_string());
                        }
                    } else if b["type"] == "tool_result" {
                        let id = b["tool_use_id"].as_str().unwrap_or("");
                        let name = meta.get(id).or_else(|| meta.get(id.split('\n').next().unwrap_or(""))).cloned().unwrap_or_else(|| "tool".into());
                        parts.push(tool_result_block(&name, id, &extract_content(&b["content"])));
                    }
                }
                let j = parts.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n");
                if !j.is_empty() {
                    out.push(json!({"role": "user", "content": j}));
                }
            }
            "user" | "assistant" => {
                let content = extract_content(&m["content"]);
                if role == "assistant" && m["tool_calls"].as_array().map(|a| !a.is_empty()).unwrap_or(false) {
                    let tcs: Vec<Value> = m["tool_calls"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|tc| {
                            let mut t = tc.clone();
                            crate::jsv::del(&mut t, "index");
                            t
                        })
                        .collect();
                    out.push(json!({"role": "assistant", "content": content, "tool_calls": tcs}));
                } else if role == "assistant" && m["content"].is_array() {
                    let tcs: Vec<Value> = m["content"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|b| b["type"] == "tool_use")
                        .map(|b| json!({"id": b["id"].as_str().unwrap_or(""), "type": "function", "function": {"name": b["name"].as_str().unwrap_or("tool"), "arguments": if truthy(&b["input"]) { b["input"].to_string() } else { "{}".into() }}}))
                        .filter(|t| t["id"] != "")
                        .collect();
                    if !tcs.is_empty() {
                        out.push(json!({"role": "assistant", "content": content, "tool_calls": tcs}));
                    } else if !content.is_empty() {
                        out.push(json!({"role": "assistant", "content": content}));
                    }
                } else if !content.is_empty() {
                    out.push(json!({"role": role, "content": content}));
                }
            }
            _ => {}
        }
    }
    let mut rest: Map<String, Value> = body.as_object().cloned().unwrap_or_default();
    for k in ["user", "metadata", "tool_choice", "stream_options", "system"] {
        rest.shift_remove(k);
    }
    rest.insert("messages".into(), Value::Array(out));
    rest.insert("max_tokens".into(), json!(crate::consts::DEFAULT_MIN_TOKENS));
    Value::Object(rest)
}

pub fn cursor_to_openai(chunk: &Value, _state: &mut Value) -> Vec<Value> {
    vec![chunk.clone()]
}

// ===========================================================================
// AgentService (bidirectional)
// ===========================================================================

fn text_from(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter(|p| p["type"] == "text" && p["text"].is_string()).map(|p| p["text"].as_str().unwrap()).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

pub fn is_agent_capable(body: &Value) -> bool {
    let Some(m) = body["messages"].as_array().filter(|a| !a.is_empty()) else { return false };
    m.iter().all(|msg| match &msg["content"] {
        Value::Array(a) => a.iter().all(|p| p.is_null() || p["type"] == "text" || p.is_string()),
        Value::Null | Value::String(_) => true,
        _ => false,
    })
}

fn encode_history(m: &Value) -> Option<Vec<u8>> {
    let content = text_from(&m["content"]);
    let mut extras = vec![];
    if m["role"] == "assistant" {
        for tc in m["tool_calls"].as_array().into_iter().flatten() {
            extras.push(format!(
                "[tool_call id={} name={} args={}]",
                tc["id"].as_str().unwrap_or(""),
                tc["function"]["name"].as_str().unwrap_or("tool"),
                tc["function"]["arguments"].as_str().unwrap_or("{}")
            ));
        }
    }
    if m["role"] == "tool" {
        extras.push(format!("[tool_result id={}]", m["tool_call_id"].as_str().unwrap_or("")));
    }
    let body = std::iter::once(content).chain(extras).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n");
    if body.is_empty() {
        return None;
    }
    let text = field_str(1, &body);
    let inner = field_bytes(1, &field_bytes(1, &text));
    Some(if m["role"] == "assistant" { field_bytes(2, &inner) } else { field_bytes(1, &inner) })
}

/// google.protobuf.Value encoding.
pub fn encode_value(v: &Value) -> Vec<u8> {
    match v {
        Value::Null => field_varint(1, 0),
        Value::Bool(b) => field_varint(4, *b as u64),
        Value::Number(n) => field_double(2, n.as_f64().unwrap_or(0.0)),
        Value::String(s) => field_str(3, s),
        Value::Array(a) => field_bytes(6, &cat(&a.iter().map(|x| field_bytes(1, &encode_value(x))).collect::<Vec<_>>())),
        Value::Object(o) => field_bytes(5, &cat(&o.iter().map(|(k, x)| field_bytes(1, &cat(&[field_str(1, k), field_bytes(2, &encode_value(x))]))).collect::<Vec<_>>())),
    }
}

pub fn decode_value(b: &[u8]) -> Value {
    let f = decode(b);
    if f.has(1) {
        return Value::Null;
    }
    if let Some(x) = f.first(4) {
        return json!(x.int() != 0);
    }
    if let Some(x) = f.first(2) {
        let bytes = x.bytes();
        if bytes.len() == 8 {
            let d = f64::from_le_bytes(bytes.try_into().unwrap());
            return crate::jsv::jnum(d);
        }
        return json!(0);
    }
    if let Some(x) = f.first(3) {
        return json!(String::from_utf8_lossy(x.bytes()));
    }
    if let Some(x) = f.first(5) {
        let mut o = Map::new();
        for e in decode(x.bytes()).all(1) {
            let pair = decode(e.bytes());
            let k = pair.str(1);
            if !k.is_empty() {
                o.insert(k, pair.first(2).map(|p| decode_value(p.bytes())).unwrap_or(Value::Null));
            }
        }
        return Value::Object(o);
    }
    if let Some(x) = f.first(6) {
        return Value::Array(decode(x.bytes()).all(1).iter().map(|p| decode_value(p.bytes())).collect());
    }
    Value::Null
}

fn encode_mcp_tools(tools: &[Value]) -> Vec<u8> {
    cat(&tools
        .iter()
        .map(|t| {
            let f = if t["function"].is_object() { &t["function"] } else { t };
            let name = f["name"].as_str().or_else(|| t["name"].as_str()).unwrap_or("");
            let desc = f["description"].as_str().or_else(|| t["description"].as_str()).unwrap_or("");
            let schema = [&f["parameters"], &t["parameters"], &t["inputSchema"], &t["input_schema"]].into_iter().find(|v| truthy(v)).cloned().unwrap_or(json!({}));
            field_bytes(1, &cat(&[field_str(1, name), field_str(2, desc), field_bytes(3, &encode_value(&schema)), field_str(4, "9router"), field_str(5, name)]))
        })
        .collect::<Vec<_>>())
}

pub fn decode_mcp_args(b: &[u8]) -> (String, String, String, Value) {
    let m = decode(b);
    let mut args = Map::new();
    for e in m.all(2) {
        let pair = decode(e.bytes());
        let k = pair.str(1);
        if !k.is_empty() {
            args.insert(k, pair.first(2).map(|p| decode_value(p.bytes())).unwrap_or(Value::Null));
        }
    }
    (m.str(1), m.str(3), m.str(5), Value::Object(args))
}

pub fn build_agent_run_frame(messages: &[Value], model: &str, tools: &[Value]) -> Vec<u8> {
    let system = messages.iter().filter(|m| m["role"] == "system").map(|m| text_from(&m["content"])).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
    let chat: Vec<&Value> = messages.iter().filter(|m| m["role"] != "system").collect();
    let cur_idx = chat.iter().rposition(|m| m["role"] == "user");
    let current = cur_idx.map(|i| chat[i]).or(chat.last().copied());
    let hist_end = cur_idx.unwrap_or(chat.len().saturating_sub(1));
    let history: Vec<Vec<u8>> = chat[..hist_end].iter().filter_map(|m| encode_history(m)).collect();
    let raw = current.map(|c| text_from(&c["content"])).filter(|s| !s.is_empty()).unwrap_or_else(|| "Continue.".into());
    let user_text = if system.is_empty() { raw } else { format!("{system}\n\n{raw}") };
    let user_msg = cat(&[field_str(1, &user_text), field_str(2, &uuid::Uuid::new_v4().to_string()), field_bytes(3, &[]), field_varint(4, 1)]);
    let mut ua = field_bytes(1, &user_msg);
    if !history.is_empty() {
        ua.extend(field_bytes(7, &cat(&history.iter().map(|h| field_bytes(1, h)).collect::<Vec<_>>())));
    }
    let conv = field_bytes(1, &ua);
    let requested = cat(&[field_str(1, model), field_varint(7, 1)]);
    let details = cat(&[field_str(1, model), field_str(3, model), field_str(4, model)]);
    let mcp = encode_mcp_tools(tools);
    let mut run = vec![field_bytes(1, &[]), field_bytes(2, &conv), field_bytes(3, &details)];
    if !mcp.is_empty() {
        run.push(field_bytes(4, &mcp));
    }
    run.push(field_bytes(9, &requested));
    wrap_frame(&field_bytes(1, &cat(&run)))
}

fn exec_ids(m: &Msg) -> (u64, String) {
    (m.first(1).map(|p| p.int()).unwrap_or(0), m.str(15))
}

fn wrap_exec_client(id: u64, exec_id: &str, field: u32, payload: &[u8]) -> Vec<u8> {
    let mut p = vec![];
    if id != 0 {
        p.push(field_varint(1, id));
    }
    p.push(field_str(15, exec_id));
    p.push(field_bytes(field, payload));
    wrap_frame(&field_bytes(2, &cat(&p)))
}

fn request_context_response(m: &Msg) -> Vec<u8> {
    let (id, eid) = exec_ids(m);
    wrap_exec_client(id, &eid, 10, &field_bytes(1, &field_bytes(1, &[])))
}

fn reject_exec(m: &Msg) -> Option<Vec<u8>> {
    let (id, eid) = exec_ids(m);
    let variant = m.order.iter().copied().find(|f| *f != 1 && *f != 15)?;
    if ![2, 3, 4, 5, 7, 8, 9, 16, 20, 23].contains(&variant) {
        return None;
    }
    if variant == 9 {
        return Some(wrap_exec_client(id, &eid, 9, &[]));
    }
    let rejected = field_bytes(2, &field_str(2, "Tool not available in this environment. Use the MCP tools provided instead."));
    Some(wrap_exec_client(id, &eid, variant, &rejected))
}

fn kv_client(kv_id: u64, field: u32, payload: &[u8], meta: Option<&[u8]>) -> Vec<u8> {
    let mut p = vec![];
    if kv_id != 0 {
        p.push(field_varint(1, kv_id));
    }
    p.push(field_bytes(field, payload));
    if let Some(m) = meta.filter(|m| !m.is_empty()) {
        p.push(field_bytes(4, m));
    }
    wrap_frame(&field_bytes(3, &cat(&p)))
}

enum AgentEv {
    Text(String),
    Tool(String, String, String),
    Error(String),
    Done(Option<&'static str>),
}

/// Drives the AgentService duplex stream, emitting events.
fn agent_events(mut body: crate::exec::ByteStream, mut tx: futures::channel::mpsc::Sender<Result<Bytes, std::io::Error>>, model: String) -> impl futures::Stream<Item = AgentEv> + Send {
    async_stream::stream! {
        let composer = is_composer(&model);
        let mut pending: Vec<u8> = vec![];
        let mut finished = false;
        let mut thinking = String::new();
        let mut emitted_vis = 0usize;
        let mut emitted_text = false;
        while !finished {
            let Some(chunk) = body.next().await else { break };
            let Ok(chunk) = chunk else { break };
            pending.extend_from_slice(&chunk);
            let mut frames = vec![];
            while pending.len() >= 5 {
                let flags = pending[0];
                let len = u32::from_be_bytes([pending[1], pending[2], pending[3], pending[4]]) as usize;
                if pending.len() < 5 + len { break; }
                let mut payload: Vec<u8> = pending[5..5 + len].to_vec();
                pending.drain(..5 + len);
                if flags & 1 != 0 { payload = gunzip(&payload).unwrap_or(payload); }
                if flags & 2 == 0 { frames.push(payload); }
            }
            for payload in frames {
                if finished { break; }
                let sm = decode(&payload);
                if let Some(u) = sm.first(1) {
                    let up = decode(u.bytes());
                    if let Some(t) = up.first(1) {
                        let d = decode(t.bytes()).str(1);
                        if !d.is_empty() { emitted_text = true; yield AgentEv::Text(d); }
                    }
                    if let Some(t) = up.first(4) {
                        let d = decode(t.bytes()).str(1);
                        if !d.is_empty() {
                            thinking.push_str(&d);
                            if composer {
                                let vis = visible_from_thinking(&thinking);
                                if vis.len() > emitted_vis {
                                    let delta = vis[emitted_vis..].to_string();
                                    emitted_vis = vis.len();
                                    emitted_text = true;
                                    yield AgentEv::Text(delta);
                                }
                            }
                        }
                    }
                    if up.has(14) {
                        if !emitted_text && !thinking.is_empty() {
                            let fb = if composer { visible_from_thinking(&thinking) } else { thinking.trim().to_string() };
                            if !fb.is_empty() { emitted_text = true; yield AgentEv::Text(fb); }
                        }
                        finished = true;
                        yield AgentEv::Done(None);
                    }
                }
                if let Some(k) = sm.first(4) {
                    let kv = decode(k.bytes());
                    let kv_id = kv.first(1).map(|p| p.int()).unwrap_or(0);
                    let meta = kv.first(4).map(|p| p.bytes().to_vec());
                    if kv.has(2) {
                        let _ = tx.send(Ok(Bytes::from(kv_client(kv_id, 2, &field_bytes(1, &[]), meta.as_deref())))).await;
                    } else if kv.has(3) {
                        let _ = tx.send(Ok(Bytes::from(kv_client(kv_id, 3, &[], meta.as_deref())))).await;
                    }
                }
                if let Some(e) = sm.first(2) {
                    let er = decode(e.bytes());
                    if er.has(10) {
                        let _ = tx.send(Ok(Bytes::from(request_context_response(&er)))).await;
                    } else if let Some(a) = er.first(11) {
                        let (name, call_id, tool_name, args) = decode_mcp_args(a.bytes());
                        let n = if tool_name.is_empty() { name } else { tool_name };
                        finished = true;
                        if !n.is_empty() {
                            let id = if call_id.is_empty() { format!("call_{}", uuid::Uuid::new_v4()) } else { call_id };
                            yield AgentEv::Tool(id, n, args.to_string());
                            yield AgentEv::Done(Some("tool_calls"));
                        } else {
                            yield AgentEv::Error("Cursor AgentService requested an unsupported IDE tool".into());
                        }
                    } else if let Some(r) = reject_exec(&er) {
                        let _ = tx.send(Ok(Bytes::from(r))).await;
                    } else {
                        finished = true;
                        yield AgentEv::Error("Cursor AgentService requested an unsupported IDE tool".into());
                    }
                }
            }
        }
        tx.close_channel();
        if !finished {
            if !emitted_text && !thinking.is_empty() {
                let fb = if composer { visible_from_thinking(&thinking) } else { thinking.trim().to_string() };
                if !fb.is_empty() { yield AgentEv::Text(fb); }
            }
            yield AgentEv::Done(None);
        }
    }
}

pub struct Cursor;

impl Cursor {
    fn headers(&self, creds: &Value) -> Result<crate::exec::Headers, String> {
        let psd = &creds["providerSpecificData"];
        let mid = psd["machineId"].as_str().filter(|s| !s.is_empty()).ok_or("Machine ID is required for Cursor API")?;
        Ok(cursor_headers(creds["accessToken"].as_str().unwrap_or(""), mid, psd["ghostMode"] != json!(false)))
    }

    async fn execute_agent(&self, args: &ExecArgs<'_>) -> Result<ExecResult, String> {
        let endpoint = crate::registry::REG.oauth("cursor")["agentEndpoint"].as_str().filter(|s| !s.is_empty()).ok_or("Cursor AgentService endpoint is not configured")?.to_string();
        let url = format!("{endpoint}/agent.v1.AgentService/Run");
        let h = self.headers(args.creds)?;
        let tools = args.body["tools"].as_array().cloned().unwrap_or_default();
        let first = build_agent_run_frame(args.body["messages"].as_array().map(|v| v.as_slice()).unwrap_or(&[]), args.model, &tools);
        let (mut tx, rx) = futures::channel::mpsc::channel::<Result<Bytes, std::io::Error>>(16);
        tx.send(Ok(Bytes::from(first))).await.map_err(|e| e.to_string())?;
        let mut rb = client_for(args.creds).post(&url).version(reqwest::Version::HTTP_2);
        for (k, v) in &h.0 {
            rb = rb.header(k, v);
        }
        let resp = rb.body(reqwest::Body::wrap_stream(rx)).send().await.map_err(|e| format!("Cursor AgentService request failed: {e}"))?;
        let status = resp.status().as_u16();
        let hv = h.0.clone();
        if status != 200 {
            let text = resp.text().await.unwrap_or_default();
            return Ok(ExecResult {
                response: Upstream::json(if status == 0 { 500 } else { status }, &json!({"error": {"message": format!("Cursor AgentService {status}: {}", if text.is_empty() { "request failed" } else { &text }), "type": "api_error"}})),
                url,
                headers: hv,
                body: args.body.clone(),
                response_format: Some("openai".into()),
            });
        }
        let up = Upstream::from_reqwest(resp);
        let rid = format!("chatcmpl-msg_{}", now_ms());
        let created = now_ms() / 1000;
        let model = args.model.to_string();
        let events = agent_events(up.body, tx, model.clone());
        if !args.stream {
            let mut content = String::new();
            let mut tool_calls = vec![];
            let mut finish = "stop";
            let mut err = None;
            futures::pin_mut!(events);
            while let Some(e) = events.next().await {
                match e {
                    AgentEv::Text(t) => content.push_str(&t),
                    AgentEv::Tool(id, n, a) => {
                        tool_calls.push(json!({"id": id, "type": "function", "function": {"name": n, "arguments": a}}));
                        finish = "tool_calls";
                    }
                    AgentEv::Error(m) => err = Some(m),
                    AgentEv::Done(Some(f)) => finish = f,
                    AgentEv::Done(None) => {}
                }
            }
            if let Some(m) = err {
                return Ok(ExecResult { response: Upstream::json(400, &json!({"error": {"message": m, "type": "api_error"}})), url, headers: hv, body: args.body.clone(), response_format: Some("openai".into()) });
            }
            let mut msg = json!({"role": "assistant", "content": if content.is_empty() { Value::Null } else { json!(content) }});
            if !tool_calls.is_empty() {
                msg["tool_calls"] = Value::Array(tool_calls);
            }
            let out = json!({"id": rid, "object": "chat.completion", "created": created, "model": model, "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": estimate_usage(&args.body, content.encode_utf16().count())});
            return Ok(ExecResult { response: Upstream::json(200, &out), url, headers: hv, body: args.body.clone(), response_format: Some("openai".into()) });
        }
        let s = async_stream::stream! {
            futures::pin_mut!(events);
            while let Some(e) = events.next().await {
                match e {
                    AgentEv::Text(t) => yield Ok::<Bytes, String>(Bytes::from(chunk(&rid, created, &model, json!({"content": t}), Value::Null))),
                    AgentEv::Tool(id, n, a) => yield Ok(Bytes::from(chunk(&rid, created, &model, json!({"tool_calls": [{"index": 0, "id": id, "type": "function", "function": {"name": n, "arguments": a}}]}), Value::Null))),
                    AgentEv::Error(m) => {
                        yield Ok(Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n", json!({"error": {"message": m, "type": "api_error"}}))));
                        break;
                    }
                    AgentEv::Done(f) => {
                        yield Ok(Bytes::from(chunk(&rid, created, &model, json!({}), json!(f.unwrap_or("stop")))));
                        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                        break;
                    }
                }
            }
        };
        Ok(ExecResult { response: Upstream::synthetic(200, "text/event-stream", s.boxed()), url, headers: hv, body: args.body.clone(), response_format: Some("openai".into()) })
    }
}

#[async_trait]
impl Executor for Cursor {
    fn provider(&self) -> &str {
        "cursor"
    }
    async fn refresh_credentials(&self, _c: &Value) -> Option<Value> {
        None
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        if is_agent_capable(&args.body) {
            return match self.execute_agent(&args).await {
                Ok(r) => Ok(r),
                Err(e) => Ok(ExecResult {
                    response: Upstream::json(500, &json!({"error": {"message": e, "type": "connection_error", "code": ""}})),
                    url: format!("{}/agent.v1.AgentService/Run", crate::registry::REG.oauth("cursor")["agentEndpoint"].as_str().unwrap_or("")),
                    headers: vec![],
                    body: args.body.clone(),
                    response_format: None,
                }),
            };
        }
        let cfg = self.config();
        let url = format!("{}{}", cfg["baseUrl"].as_str().unwrap_or(""), cfg["chatPath"].as_str().unwrap_or(""));
        let h = self.headers(args.creds)?;
        let msgs = args.body["messages"].as_array().cloned().unwrap_or_default();
        let tools = args.body["tools"].as_array().cloned().unwrap_or_default();
        let ua = args.creds["rawHeaders"]["user-agent"].as_str().unwrap_or("");
        let force = ua.contains("claude-cli") || ua.contains("claude-code") || ua.contains("Claude Code");
        let pb = generate_cursor_body(&msgs, args.model, &tools, args.body["reasoning_effort"].as_str(), force);
        let r = crate::exec::post_raw(args.creds, &url, &h, pb, 60_000).await;
        let up = match r {
            Ok(u) => u,
            Err(e) => return Ok(ExecResult { response: Upstream::json(500, &json!({"error": {"message": e, "type": "connection_error", "code": ""}})), url, headers: h.0, body: args.body.clone(), response_format: None }),
        };
        let status = up.status;
        let bytes = up.bytes().await.unwrap_or_default();
        if status != 200 {
            let text = String::from_utf8_lossy(&bytes).into_owned();
            return Ok(ExecResult {
                response: Upstream::json(status, &json!({"error": {"message": format!("[{status}]: {}", if text.is_empty() { "Unknown error" } else { &text }), "type": "invalid_request_error", "code": ""}})),
                url,
                headers: h.0,
                body: args.body.clone(),
                response_format: None,
            });
        }
        let resp = if args.stream { protobuf_to_sse(&bytes, args.model, &args.body) } else { protobuf_to_json(&bytes, args.model, &args.body) };
        Ok(ExecResult { response: resp, url, headers: h.0, body: args.body.clone(), response_format: Some("openai".into()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_and_fields() {
        assert_eq!(varint(300), vec![0xac, 0x02]);
        let m = decode(&cat(&[field_str(1, "hi"), field_varint(2, 5), field_str(1, "x")]));
        assert_eq!(m.str(1), "hi");
        assert_eq!(m.all(1).len(), 2);
        assert_eq!(m.first(2).unwrap().int(), 5);
        let v = json!({"a": [1.5, "s", true, null], "b": {"c": 2}});
        assert_eq!(decode_value(&encode_value(&v)), v);
    }

    #[test]
    fn response_frames() {
        let text_frame = wrap_frame(&field_bytes(2, &field_str(1, "Hello")));
        let tc = cat(&[field_str(3, "call_1\nmc_x"), field_str(9, "mcp_custom_read"), field_str(10, "{\"p\":1}"), field_varint(11, 1)]);
        let tool_frame = wrap_frame(&field_bytes(1, &tc));
        let buf = cat(&[text_frame, tool_frame]);
        let j = protobuf_to_json(&buf, "gpt", &json!({}));
        let s = futures::executor::block_on(j.text());
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "Hello");
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["id"], "call_1");
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        let sse = futures::executor::block_on(protobuf_to_sse(&buf, "gpt", &json!({})).text());
        assert!(sse.contains("\"content\":\"Hello\"") && sse.ends_with("data: [DONE]\n\n"));
        let err = wrap_frame(br#"{"error":{"code":"resource_exhausted","message":"slow down"}}"#);
        assert_eq!(protobuf_to_json(&err, "gpt", &json!({})).status, 429);
    }

    #[test]
    fn names_and_translation() {
        assert_eq!(format_tool_name("mcp__srv__do"), "mcp_srv_do");
        assert_eq!(format_tool_name("Read"), "mcp_custom_Read");
        assert_eq!(parse_tool_name("mcp_custom_Read"), ("custom".into(), "Read".into()));
        let b = json!({"messages": [{"role": "system", "content": "s"}, {"role": "assistant", "content": null, "tool_calls": [{"id": "c", "index": 0, "type": "function", "function": {"name": "f", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "c", "content": "<x>"}], "tool_choice": "auto"});
        let t = openai_to_cursor("m", &b, true, &ReqCtx::default());
        assert_eq!(t["messages"][0]["content"], "[System Instructions]\ns");
        assert!(t["messages"][1]["tool_calls"][0].get("index").is_none());
        assert!(t["messages"][2]["content"].as_str().unwrap().contains("<tool_name>f</tool_name>"));
        assert!(t["messages"][2]["content"].as_str().unwrap().contains("&lt;x&gt;"));
        assert!(t.get("tool_choice").is_none());
        assert!(cursor_checksum("mid").ends_with("mid"));
        assert!(is_agent_capable(&json!({"messages": [{"role": "user", "content": "hi"}]})));
        assert!(!is_agent_capable(&json!({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})));
        assert!(!build_agent_run_frame(&[json!({"role": "user", "content": "hi"})], "auto", &[]).is_empty());
    }
}
