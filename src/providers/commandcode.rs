//! CommandCode (api.commandcode.ai/alpha/generate) — executor plus the
//! openai→commandcode request and commandcode→openai response translators.

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::exec::{ByteStream, ExecArgs, ExecResult, Executor, Headers, ParsedError, Upstream, base_execute, cred_str};
use crate::jsv::{js_string, now_ms, truthy};
use crate::translate::ReqCtx;
use crate::translate::concerns::{build_chunk, encode_data_uri, fallback_tool_call_id, parse_data_uri, reasoning_delta, to_openai_finish, to_openai_usage};

// ---------------------------------------------------------------------------
// Request translator
// ---------------------------------------------------------------------------

fn flatten_text(c: &Value) -> String {
    match c {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .filter_map(|p| p.as_str().map(str::to_owned).or_else(|| p["text"].as_str().map(str::to_owned)))
            .collect::<Vec<_>>()
            .join("\n"),
        o => js_string(o),
    }
}

fn image_block(mime: &str, image: String) -> Value {
    json!({"type": "image", "image": image, "mimeType": mime, "mediaType": mime})
}

fn to_native_image(part: &Value) -> Option<Value> {
    if part["type"] == "image_url" {
        let url = if part["image_url"].is_string() { part["image_url"].clone() } else { part["image_url"]["url"].clone() };
        let (mime, b64) = parse_data_uri(&url)?;
        return Some(image_block(&mime, encode_data_uri(&mime, &b64)));
    }
    if part["type"] == "image" {
        if let Some(img) = part["image"].as_str().filter(|s| s.starts_with("data:")) {
            let parsed = parse_data_uri(&part["image"]);
            let mime = part["mimeType"].as_str().map(str::to_owned).or(parsed.map(|p| p.0)).unwrap_or_else(|| "image/png".into());
            return Some(image_block(&mime, img.to_string()));
        }
        let src = &part["source"];
        if src["type"] == "base64" {
            if let Some(d) = src["data"].as_str() {
                let mime = src["media_type"].as_str().filter(|s| !s.is_empty()).unwrap_or("image/png");
                return Some(image_block(mime, encode_data_uri(mime, d)));
            }
        }
    }
    None
}

fn to_content_blocks(c: &Value) -> Vec<Value> {
    let text = |t: &str| json!({"type": "text", "text": t});
    match c {
        Value::Null => vec![text("")],
        Value::String(s) => vec![text(s)],
        Value::Array(a) => {
            let mut out = vec![];
            for p in a {
                if let Some(s) = p.as_str() {
                    out.push(text(s));
                } else if p.is_object() {
                    if p["type"] == "text" && p["text"].is_string() {
                        out.push(text(p["text"].as_str().unwrap()));
                    } else if let Some(img) = to_native_image(p) {
                        out.push(img);
                    } else if let Some(t) = p["text"].as_str() {
                        out.push(text(t));
                    }
                }
            }
            if out.is_empty() { vec![text("")] } else { out }
        }
        o => vec![text(&js_string(o))],
    }
}

fn safe_parse(s: &Value) -> Value {
    match s {
        Value::Null => json!({}),
        Value::String(x) => serde_json::from_str(x).unwrap_or_else(|_| json!({})),
        o => o.clone(),
    }
}

fn convert_messages(msgs: &[Value]) -> (Vec<Value>, String) {
    let mut out = vec![];
    let mut sys = vec![];
    for m in msgs {
        if m.is_null() {
            continue;
        }
        match m["role"].as_str().unwrap_or("") {
            "system" => {
                let t = flatten_text(&m["content"]);
                if !t.is_empty() {
                    sys.push(t);
                }
            }
            "tool" => {
                let value = m["content"].as_str().map(str::to_owned).unwrap_or_else(|| flatten_text(&m["content"]));
                out.push(json!({"role": "tool", "content": [{
                    "type": "tool-result",
                    "toolCallId": m["tool_call_id"].as_str().unwrap_or(""),
                    "toolName": m["name"].as_str().unwrap_or(""),
                    "output": {"type": "text", "value": value},
                }]}));
            }
            "assistant" => {
                let mut blocks = vec![];
                let rc = [&m["reasoning_content"], &m["thought"], &m["reasoning"]].into_iter().find(|v| truthy(v)).cloned();
                let has_tc = m["tool_calls"].as_array().map(|a| !a.is_empty()).unwrap_or(false);
                if rc.is_some() || has_tc {
                    blocks.push(json!({"type": "reasoning", "text": rc.unwrap_or(json!(" "))}));
                }
                let t = flatten_text(&m["content"]);
                if !t.is_empty() {
                    blocks.push(json!({"type": "text", "text": t}));
                }
                for tc in m["tool_calls"].as_array().into_iter().flatten() {
                    let f = &tc["function"];
                    blocks.push(json!({
                        "type": "tool-call",
                        "toolCallId": tc["id"].as_str().unwrap_or(""),
                        "toolName": f["name"].as_str().unwrap_or(""),
                        "input": safe_parse(&f["arguments"]),
                    }));
                }
                if blocks.is_empty() {
                    blocks.push(json!({"type": "text", "text": ""}));
                }
                out.push(json!({"role": "assistant", "content": blocks}));
            }
            _ => out.push(json!({"role": "user", "content": to_content_blocks(&m["content"])})),
        }
    }
    (out, sys.join("\n\n"))
}

fn convert_tools(tools: &Value) -> Option<Vec<Value>> {
    let a = tools.as_array().filter(|a| !a.is_empty())?;
    let mut r = vec![];
    for t in a {
        if t["type"] == "function" && t["function"].is_object() {
            let f = &t["function"];
            let mut o = json!({"name": f["name"], "description": f["description"], "input_schema": if truthy(&f["parameters"]) { f["parameters"].clone() } else { json!({"type": "object"}) }});
            if f.get("description").is_none() {
                crate::jsv::del(&mut o, "description");
            }
            r.push(o);
        } else if truthy(&t["name"]) && (truthy(&t["input_schema"]) || truthy(&t["parameters"])) {
            let mut o = json!({"name": t["name"], "description": t["description"], "input_schema": if truthy(&t["input_schema"]) { t["input_schema"].clone() } else { t["parameters"].clone() }});
            if t.get("description").is_none() {
                crate::jsv::del(&mut o, "description");
            }
            r.push(o);
        }
    }
    (!r.is_empty()).then_some(r)
}

pub fn openai_to_commandcode(model: &str, body: &Value, stream: bool, _rc: &ReqCtx) -> Value {
    let (messages, system) = convert_messages(body["messages"].as_array().map(|v| v.as_slice()).unwrap_or(&[]));
    let nn = |v: &Value| (!v.is_null()).then(|| v.clone());
    let mut params = json!({
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": nn(&body["max_tokens"]).or_else(|| nn(&body["max_output_tokens"])).unwrap_or(json!(crate::consts::DEFAULT_MAX_TOKENS)),
        "temperature": nn(&body["temperature"]).unwrap_or(json!(0.3)),
    });
    if !system.is_empty() {
        params["system"] = json!(system);
    }
    if let Some(t) = convert_tools(&body["tools"]) {
        params["tools"] = Value::Array(t);
    }
    if !body["top_p"].is_null() {
        params["top_p"] = body["top_p"].clone();
    }
    let today = &crate::jsv::iso_from_ms(now_ms())[..10];
    json!({
        "threadId": uuid::Uuid::new_v4().to_string(),
        "memory": "",
        "config": {
            "workingDir": std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default(),
            "date": today,
            "environment": crate::consts::node_platform(),
            "structure": [],
            "isGitRepo": false,
            "currentBranch": "",
            "mainBranch": "",
            "gitStatus": "",
            "recentCommits": [],
        },
        "params": params,
    })
}

// ---------------------------------------------------------------------------
// Response translator
// ---------------------------------------------------------------------------

fn ensure_state(state: &mut Value, model: &Value) {
    if !truthy(&state["responseId"]) {
        state["responseId"] = json!(format!("chatcmpl-{}", now_ms()));
        state["created"] = json!(now_ms() / 1000);
        if !truthy(&state["model"]) {
            state["model"] = if truthy(model) { model.clone() } else { json!("commandcode") };
        }
        state["chunkIndex"] = json!(0);
        state["toolIndex"] = json!(0);
        state["toolIndexById"] = json!({});
        state["finishReason"] = Value::Null;
        state["usage"] = Value::Null;
    }
}

fn mk(state: &Value, delta: Value, finish: Value) -> Value {
    build_chunk(state["responseId"].as_str().unwrap_or(""), state["created"].as_i64().unwrap_or(0), &state["model"], delta, finish)
}

fn bump(state: &mut Value, k: &str) -> i64 {
    let n = state[k].as_i64().unwrap_or(0);
    state[k] = json!(n + 1);
    n
}

/// One NDJSON event → OpenAI chunks. Err on an upstream `error` event.
pub fn commandcode_event(chunk: &Value, state: &mut Value) -> Result<Vec<Value>, String> {
    if chunk["object"] == "chat.completion.chunk" {
        return Ok(vec![chunk.clone()]);
    }
    let event = if let Some(s) = chunk.as_str() {
        let line = s.trim();
        let j = line.strip_prefix("data:").map(str::trim).unwrap_or(line);
        if j.is_empty() || j == "[DONE]" {
            return Ok(vec![]);
        }
        match serde_json::from_str::<Value>(j) {
            Ok(v) => v,
            Err(_) => return Ok(vec![]),
        }
    } else {
        chunk.clone()
    };
    if !event.is_object() || !truthy(&event["type"]) {
        return Ok(vec![]);
    }
    ensure_state(state, &event["model"]);
    let mut out = vec![];
    let first = |s: &Value| s["chunkIndex"].as_i64().unwrap_or(0) == 0;
    match event["type"].as_str().unwrap_or("") {
        "text-delta" => {
            let text = if truthy(&event["text"]) { event["text"].clone() } else if truthy(&event["delta"]) { event["delta"].clone() } else { json!("") };
            if truthy(&text) {
                let delta = if first(state) { json!({"role": "assistant", "content": text}) } else { json!({"content": text}) };
                bump(state, "chunkIndex");
                out.push(mk(state, delta, Value::Null));
            }
        }
        "reasoning-delta" => {
            if truthy(&event["text"]) {
                let delta = reasoning_delta(&event["text"], first(state));
                bump(state, "chunkIndex");
                out.push(mk(state, delta, Value::Null));
            }
        }
        "tool-input-start" => {
            let id = [&event["id"], &event["toolCallId"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or_else(|| fallback_tool_call_id(state["toolIndex"].as_i64()));
            let idx = match state["toolIndexById"][&id].as_i64() {
                Some(i) => i,
                None => {
                    let i = bump(state, "toolIndex");
                    state["toolIndexById"][&id] = json!(i);
                    i
                }
            };
            let mut delta = if first(state) { json!({"role": "assistant"}) } else { json!({}) };
            delta["tool_calls"] = json!([{"index": idx, "id": id, "type": "function", "function": {"name": event["toolName"].as_str().unwrap_or(""), "arguments": ""}}]);
            bump(state, "chunkIndex");
            out.push(mk(state, delta, Value::Null));
        }
        "tool-input-delta" => {
            let id = if truthy(&event["id"]) { js_string(&event["id"]) } else { js_string(&event["toolCallId"]) };
            if let Some(idx) = state["toolIndexById"][&id].as_i64() {
                let a = if truthy(&event["delta"]) { event["delta"].clone() } else if truthy(&event["inputTextDelta"]) { event["inputTextDelta"].clone() } else { json!("") };
                out.push(mk(state, json!({"tool_calls": [{"index": idx, "function": {"arguments": a}}]}), Value::Null));
            }
        }
        "tool-call" => {
            let id = js_string(&event["toolCallId"]);
            if state["toolIndexById"].get(&id).is_none() {
                let idx = bump(state, "toolIndex");
                state["toolIndexById"][&id] = json!(idx);
                let args = event["input"].as_str().map(str::to_owned).unwrap_or_else(|| if event["input"].is_null() { "{}".into() } else { event["input"].to_string() });
                let mut delta = if first(state) { json!({"role": "assistant"}) } else { json!({}) };
                delta["tool_calls"] = json!([{"index": idx, "id": event["toolCallId"], "type": "function", "function": {"name": event["toolName"].as_str().unwrap_or(""), "arguments": args}}]);
                bump(state, "chunkIndex");
                out.push(mk(state, delta, Value::Null));
            }
        }
        "finish-step" => {
            state["finishReason"] = to_openai_finish(&event["finishReason"], "commandcode");
            if truthy(&event["usage"]) {
                state["usage"] = event["usage"].clone();
            }
        }
        "finish" => {
            let fr = if truthy(&state["finishReason"]) {
                state["finishReason"].clone()
            } else {
                {
                    let stop = json!("stop");
                    to_openai_finish(if truthy(&event["finishReason"]) { &event["finishReason"] } else { &stop }, "commandcode")
                }
            };
            let mut c = mk(state, json!({}), fr);
            let total = if truthy(&event["totalUsage"]) { event["totalUsage"].clone() } else { state["usage"].clone() };
            if let Some(u) = to_openai_usage(&total, "commandcode") {
                c["usage"] = u;
            }
            out.push(c);
        }
        "error" => {
            let ev = if !event["error"].is_null() { &event["error"] } else if !event["message"].is_null() { &event["message"] } else { &json!("unknown") };
            let s = ev.as_str().map(str::to_owned).unwrap_or_else(|| ev.to_string());
            return Err(format!("[CommandCode error: {s}]"));
        }
        _ => {}
    }
    Ok(out)
}

pub fn commandcode_to_openai(chunk: Option<&Value>, state: &mut Value) -> Vec<Value> {
    chunk.map(|c| commandcode_event(c, state).unwrap_or_default()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

pub fn parse_commandcode_error(event: &Value) -> (u16, String, String) {
    if !event.is_object() {
        return (503, "CommandCode upstream error".into(), "server_error".into());
    }
    let ev = if !event["error"].is_null() { event["error"].clone() } else if !event["message"].is_null() { event["message"].clone() } else { json!("unknown") };
    let mut status: Option<i64> = None;
    let mut ty = "server_error".to_string();
    let as_int = |v: &Value| -> Option<i64> {
        match v {
            Value::Number(n) => n.as_i64(),
            Value::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    };
    let message = match &ev {
        Value::Object(_) => {
            if truthy(&ev["statusCode"]) {
                status = as_int(&ev["statusCode"]);
            } else if truthy(&ev["status"]) {
                status = as_int(&ev["status"]);
            }
            if let Some(t) = ev["type"].as_str().filter(|s| !s.is_empty()) {
                ty = t.into();
            }
            if truthy(&ev["message"]) { js_string(&ev["message"]) } else if truthy(&ev["error"]) { js_string(&ev["error"]) } else { ev.to_string() }
        }
        Value::String(s) => s.clone(),
        o => o.to_string(),
    };
    if truthy(&event["statusCode"]) {
        if let Some(s) = as_int(&event["statusCode"]) {
            status = Some(s);
        }
    }
    let status = match status {
        Some(s) if (400..=599).contains(&s) => s as u16,
        _ => {
            let l = message.to_lowercase();
            let (s, t) = if l.contains("rate limit") || l.contains("too many requests") {
                (429, "rate_limit_error")
            } else if l.contains("unauthorized") || l.contains("invalid api key") || l.contains("authentication") {
                (401, "authentication_error")
            } else if l.contains("payment required") || l.contains("billing") {
                (402, "billing_error")
            } else if l.contains("quota") || l.contains("forbidden") || l.contains("permission") {
                (403, "permission_error")
            } else if l.contains("not found") {
                (404, "invalid_request_error")
            } else if l.contains("unavailable") || l.contains("overloaded") || l.contains("server error") {
                (503, "server_error")
            } else {
                (503, ty.as_str())
            };
            ty = t.to_string();
            s
        }
    };
    (status, message, ty)
}

fn event_of(line: &str) -> Option<Value> {
    let t = line.trim();
    let j = t.strip_prefix("data:").map(str::trim).unwrap_or(t);
    serde_json::from_str(j).ok()
}

/// Peeks the NDJSON stream until the first meaningful event; surfaces an
/// upstream `error` event as an HTTP error, otherwise re-encodes as OpenAI SSE.
pub async fn inspect_and_wrap(up: Upstream, model: &str) -> Upstream {
    let status = up.status;
    let mut body = up.body;
    let mut raw: Vec<Bytes> = vec![];
    let mut buf = String::new();
    let mut detected: Option<Value> = None;
    'outer: loop {
        match body.next().await {
            None => {
                let t = buf.trim();
                if !t.is_empty() {
                    if let Some(e) = event_of(t).filter(|e| e["type"] == "error") {
                        detected = Some(e);
                    }
                }
                break;
            }
            Some(Err(_)) => break,
            Some(Ok(c)) => {
                buf.push_str(&String::from_utf8_lossy(&c));
                raw.push(c);
                while let Some(pos) = buf.find('\n') {
                    let line: String = buf.drain(..=pos).collect();
                    let t = line.trim();
                    if t.is_empty() {
                        continue;
                    }
                    let j = t.strip_prefix("data:").map(str::trim).unwrap_or(t);
                    if j.is_empty() || j == "[DONE]" {
                        break 'outer;
                    }
                    let Ok(ev) = serde_json::from_str::<Value>(j) else { continue };
                    if ev["type"] == "error" {
                        detected = Some(ev);
                        break 'outer;
                    }
                    if ["text-delta", "reasoning-delta", "tool-input-start", "tool-call", "finish", "finish-step"].iter().any(|t| ev["type"] == *t) {
                        break 'outer;
                    }
                }
            }
        }
    }
    if let Some(e) = detected {
        let (st, msg, ty) = parse_commandcode_error(&e);
        return Upstream::json(st, &json!({"error": {"message": format!("[CommandCode error: {msg}]"), "type": ty, "code": st}}));
    }
    let replay = futures::stream::iter(raw.into_iter().map(Ok)).chain(body).boxed();
    Upstream::synthetic(status, "text/event-stream", wrap_ndjson(replay, model.to_string()))
}

fn wrap_ndjson(mut body: ByteStream, model: String) -> ByteStream {
    let s = async_stream::stream! {
        let mut state = json!({"model": model});
        let mut lines = crate::sse::LineParser::default();
        while let Some(c) = body.next().await {
            let c = match c { Ok(c) => c, Err(e) => { yield Err(e); return; } };
            for line in lines.push(&c) {
                if line.trim().is_empty() { continue; }
                match commandcode_event(&json!(line), &mut state) {
                    Ok(out) => for o in out { yield Ok(Bytes::from(format!("data: {o}\n\n"))); },
                    Err(e) => { yield Err(e); return; }
                }
            }
        }
        if let Some(line) = lines.finish() {
            match commandcode_event(&json!(line), &mut state) {
                Ok(out) => for o in out { yield Ok(Bytes::from(format!("data: {o}\n\n"))); },
                Err(e) => { yield Err(e); return; }
            }
        }
        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
    };
    s.boxed()
}

pub struct CommandCode;

#[async_trait]
impl Executor for CommandCode {
    fn provider(&self) -> &str {
        "commandcode"
    }
    fn transform_request(&self, _m: &str, mut body: Value, _s: bool, _c: &Value) -> Value {
        body["stream"] = json!(true);
        body
    }
    fn build_headers(&self, creds: &Value, stream: bool, _u: &str, _m: &str, _b: &Value) -> Headers {
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.extend_obj(&self.config()["headers"]);
        h.set("x-session-id", uuid::Uuid::new_v4().to_string());
        if let Some(t) = cred_str(creds, "apiKey").or_else(|| cred_str(creds, "accessToken")) {
            h.set("Authorization", format!("Bearer {t}"));
        }
        if stream {
            h.set("Accept", "text/event-stream");
        }
        h
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let max = 2;
        let mut attempt = 0;
        loop {
            let a = ExecArgs { model: args.model, body: args.body.clone(), stream: args.stream, creds: &mut *args.creds, session_id: args.session_id.clone(), client_tool: args.client_tool.clone(), override_headers: args.override_headers.clone() };
            let mut res = base_execute(self, a).await?;
            if !res.response.ok() {
                return Ok(res);
            }
            let up = std::mem::replace(&mut res.response, Upstream::json(200, &json!({})));
            let wrapped = inspect_and_wrap(up, args.model).await;
            if !wrapped.ok() && attempt < max && [502, 503, 504].contains(&wrapped.status) {
                attempt += 1;
                tokio::time::sleep(std::time::Duration::from_millis(1000 * attempt as u64)).await;
                continue;
            }
            res.response = wrapped;
            res.response_format = Some("openai".into());
            return Ok(res);
        }
    }
    fn parse_error(&self, status: u16, body: &str) -> ParsedError {
        let parsed: Value = serde_json::from_str(if body.is_empty() { "{}" } else { body }).unwrap_or(Value::Null);
        let err = if truthy(&parsed["error"]) { &parsed["error"] } else { &parsed };
        let msg = [&err["message"], &parsed["message"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or_else(|| body.to_string());
        let code = [&err["code"], &err["statusCode"]].into_iter().find(|v| truthy(v)).and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())));
        ParsedError { status: code.map(|c| c as u16).filter(|c| *c > 0).unwrap_or(status), message: if msg.is_empty() { format!("CommandCode upstream error: {status}") } else { msg }, resets_at_ms: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_shape() {
        let body = json!({"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        ], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]});
        let r = openai_to_commandcode("m", &body, true, &ReqCtx::default());
        assert_eq!(r["params"]["system"], "sys");
        assert_eq!(r["params"]["messages"][1]["content"][0]["type"], "reasoning");
        assert_eq!(r["params"]["messages"][1]["content"][1]["input"]["a"], 1);
        assert_eq!(r["params"]["messages"][2]["content"][0]["output"]["value"], "ok");
        assert_eq!(r["params"]["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(r["params"]["max_tokens"], 64000);
    }

    #[test]
    fn response_events() {
        let mut st = json!({"model": "m"});
        let a = commandcode_to_openai(Some(&json!(r#"{"type":"text-delta","text":"Hi"}"#)), &mut st);
        assert_eq!(a[0]["choices"][0]["delta"]["role"], "assistant");
        let b = commandcode_to_openai(Some(&json!({"type": "tool-call", "toolCallId": "t", "toolName": "f", "input": {"x": 1}})), &mut st);
        assert_eq!(b[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], "{\"x\":1}");
        commandcode_to_openai(Some(&json!({"type": "finish-step", "finishReason": "tool-calls", "usage": {"inputTokens": 3, "outputTokens": 4}})), &mut st);
        let f = commandcode_to_openai(Some(&json!({"type": "finish"})), &mut st);
        assert_eq!(f[0]["choices"][0]["finish_reason"], "tool_calls");
        assert!(commandcode_event(&json!({"type": "error", "error": "boom"}), &mut st).is_err());
        assert_eq!(parse_commandcode_error(&json!({"type": "error", "error": "Rate limit hit"})).0, 429);
    }
}
