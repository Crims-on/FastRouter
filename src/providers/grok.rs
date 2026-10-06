//! Grok executors: grok-cli (Responses API on cli-chat-proxy.grok.com) and
//! grok-web (grok.com web chat with an SSO cookie). Ports of grok-cli.js / grok-web.js.

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use async_trait::async_trait;
use base64::Engine;
use bytes::Bytes;
use futures::StreamExt;
use regex::Regex;
use serde_json::{Value, json};

use crate::exec::{ExecArgs, ExecResult, Executor, Headers, ParsedError, Upstream, post_json};
use crate::jsv::{now_ms, truthy};

// ===========================================================================
// grok-cli
// ===========================================================================

static SERVER_ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(rs|fc|resp|msg)_").unwrap());
static NATIVE_ITEM_ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^(?:rs|msg|fc)_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$").unwrap());
const HOSTED: [&str; 8] = ["web_search", "x_search", "web_search_preview", "file_search", "image_generation", "code_interpreter", "mcp", "local_shell"];
const ALLOW: [&str; 16] = [
    "model", "input", "instructions", "tools", "tool_choice", "stream", "store", "reasoning", "include", "temperature", "top_p", "max_output_tokens", "parallel_tool_calls", "text", "metadata", "prompt_cache_key",
];
const EFFORTS: [&str; 4] = ["low", "medium", "high", "xhigh"];
const TURN_STORE_MAX: usize = 5000;
const SESSION_TTL_MS: i64 = 2 * 60 * 60 * 1000;

static TURNS: LazyLock<Mutex<indexmap_lite::Lru>> = LazyLock::new(Default::default);

mod indexmap_lite {
    use std::collections::{HashMap, VecDeque};
    /// Insertion-ordered map with oldest-first eviction (JS Map semantics).
    #[derive(Default)]
    pub struct Lru {
        pub map: HashMap<String, (i64, i64)>,
        pub order: VecDeque<String>,
    }
    impl Lru {
        pub fn take(&mut self, k: &str) -> Option<(i64, i64)> {
            let v = self.map.remove(k)?;
            self.order.retain(|x| x != k);
            Some(v)
        }
        pub fn insert(&mut self, k: String, v: (i64, i64), max: usize) {
            while self.map.len() >= max {
                match self.order.pop_front() {
                    Some(old) => {
                        self.map.remove(&old);
                    }
                    None => break,
                }
            }
            self.order.push_back(k.clone());
            self.map.insert(k, v);
        }
    }
}

pub fn count_user_turns(input: &Value) -> i64 {
    let Some(a) = input.as_array() else { return 1 };
    let n = a
        .iter()
        .filter(|i| i.is_object() && i["role"] == "user" && (i["type"].as_str().map(|t| t.is_empty() || t == "message").unwrap_or(true)))
        .count() as i64;
    n.max(1)
}

/// resolveGrokCliTurnIdx with a fresh request key (each execute is one request).
pub fn resolve_turn_idx(session_id: &str, input: &Value) -> i64 {
    let from_input = count_user_turns(input);
    if session_id.is_empty() {
        return from_input;
    }
    let now = now_ms();
    let mut st = TURNS.lock().unwrap();
    let prev = st.take(session_id).filter(|(_, last)| now - last <= SESSION_TTL_MS).map(|(t, _)| t).unwrap_or(0);
    let turn = if prev > 0 { from_input.max(prev + 1) } else { from_input };
    st.insert(session_id.to_string(), (turn, now), TURN_STORE_MAX);
    turn
}

pub fn normalize_effort(v: &Value) -> String {
    let e = v.as_str().map(|s| s.trim().to_lowercase()).unwrap_or_default();
    if e == "max" {
        return "xhigh".into();
    }
    if EFFORTS.contains(&e.as_str()) { e } else { "high".into() }
}

pub fn supports_reasoning_effort(model: &str) -> bool {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^grok-4\.5(?:$|-)").unwrap());
    RE.is_match(model)
}

fn resolve_session(creds: &Value, body: &Value) -> String {
    let explicit = json!({
        "prompt_cache_key": body["prompt_cache_key"],
        "session_id": body["session_id"],
        "conversation_id": body["conversation_id"],
        "metadata": body["metadata"],
    });
    let conn = creds["connectionId"].as_str().filter(|s| !s.is_empty()).or_else(|| creds["id"].as_str());
    crate::session::resolve_session_identity(&creds["rawHeaders"], &explicit, conn, creds["providerSpecificData"]["workspaceId"].as_str(), "grok-cli").0
}

fn stringify_output(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        o => o.to_string(),
    }
}

fn normalize_input_item(item: &Value) -> Option<Value> {
    if !item.is_object() {
        return Some(item.clone());
    }
    let mut clean = item.clone();
    crate::jsv::del(&mut clean, "internal_chat_message_metadata_passthrough");
    let native = |v: &Value| v.as_str().map(|s| NATIVE_ITEM_ID.is_match(s)).unwrap_or(false);
    let call_id = || if truthy(&item["call_id"]) { item["call_id"].clone() } else { item["id"].clone() };
    let name = item["name"].as_str().map(|s| s.trim().to_string()).unwrap_or_default();
    match item["type"].as_str().unwrap_or("") {
        "reasoning" => {
            if !native(&item["id"]) || !item["encrypted_content"].is_string() {
                return None;
            }
            Some(clean)
        }
        "custom_tool_call" => {
            let cid = call_id();
            if !truthy(&cid) || name.is_empty() {
                return None;
            }
            let inp = if !item["input"].is_null() { &item["input"] } else { &item["arguments"] };
            Some(json!({"type": "function_call", "call_id": cid, "name": name, "arguments": json!({"input": stringify_output(inp)}).to_string()}))
        }
        "custom_tool_call_output" | "function_call_output" => {
            let cid = call_id();
            if !truthy(&cid) {
                return None;
            }
            Some(json!({"type": "function_call_output", "call_id": cid, "output": stringify_output(&item["output"])}))
        }
        "function_call" => {
            let cid = call_id();
            if !truthy(&cid) || name.is_empty() {
                return None;
            }
            let mut o = json!({"type": "function_call"});
            if native(&item["id"]) {
                o["id"] = item["id"].clone();
            }
            o["call_id"] = cid;
            o["name"] = json!(name);
            o["arguments"] = json!(item["arguments"].as_str().map(str::to_owned).unwrap_or_else(|| if item["arguments"].is_null() { "{}".into() } else { item["arguments"].to_string() }));
            if item["status"].is_string() {
                o["status"] = item["status"].clone();
            }
            Some(o)
        }
        _ => Some(clean),
    }
}

pub fn normalize_grok_input(body: &mut Value) {
    let Some(a) = body["input"].as_array() else { return };
    let normalized: Vec<Value> = a.iter().filter_map(normalize_input_item).filter(|v| !v.is_null() && v != &json!(false)).collect();
    let call_ids: HashSet<String> = normalized.iter().filter(|i| i["type"] == "function_call" && truthy(&i["call_id"])).map(|i| crate::jsv::js_string(&i["call_id"])).collect();
    body["input"] = Value::Array(normalized.into_iter().filter(|i| i["type"] != "function_call_output" || call_ids.contains(&crate::jsv::js_string(&i["call_id"]))).collect());
}

fn strip_stored_refs(body: &mut Value) {
    let Some(a) = body["input"].as_array_mut() else { return };
    a.retain(|i| !(i.as_str().map(|s| SERVER_ID.is_match(s)).unwrap_or(false) || i["type"] == "item_reference"));
    for i in a.iter_mut() {
        if let Some(id) = i["id"].as_str() {
            if SERVER_ID.is_match(id) && !NATIVE_ITEM_ID.is_match(id) {
                crate::jsv::del(i, "id");
            }
        }
    }
}

fn normalize_tools(body: &mut Value) {
    let tools = body["tools"].as_array().cloned().unwrap_or_default();
    if tools.is_empty() {
        crate::jsv::del(body, "tools");
        crate::jsv::del(body, "tool_choice");
        return;
    }
    let mut valid = HashSet::new();
    let mut hosted = HashSet::new();
    let mut out = vec![];
    for tool in tools {
        if !tool.is_object() {
            continue;
        }
        let ty = tool["type"].as_str().unwrap_or("").to_string();
        if ty != "function" {
            if HOSTED.contains(&ty.as_str()) {
                hosted.insert(ty.clone());
                out.push(tool);
                continue;
            }
            let nested = ty.is_empty() && truthy(&tool["function"]);
            let bare = ty.is_empty() || tool["name"].is_string();
            if !nested && !bare {
                continue;
            }
        }
        let f = if tool["function"].is_object() { tool["function"].clone() } else { Value::Null };
        let raw = tool["name"].as_str().or_else(|| f["name"].as_str()).unwrap_or("");
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        let desc = tool["description"].as_str().or_else(|| f["description"].as_str()).unwrap_or("");
        let params = if ty == "custom" {
            json!({"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]})
        } else if tool["parameters"].is_object() {
            tool["parameters"].clone()
        } else if f["parameters"].is_object() {
            f["parameters"].clone()
        } else {
            json!({"type": "object", "properties": {}})
        };
        let n: String = name.chars().take(128).collect();
        let mut t = json!({"type": "function", "name": n});
        if !desc.is_empty() {
            t["description"] = json!(desc);
        }
        t["parameters"] = params;
        valid.insert(n);
        out.push(t);
    }
    if out.is_empty() {
        crate::jsv::del(body, "tools");
        crate::jsv::del(body, "tool_choice");
        return;
    }
    body["tools"] = Value::Array(out);
    let tc = body["tool_choice"].clone();
    if tc.is_object() && truthy(&tc) {
        let ct = tc["type"].as_str().unwrap_or("");
        if ct == "function" || ct == "custom" {
            let raw = if !tc["name"].is_null() { &tc["name"] } else { &tc["function"]["name"] };
            let name: String = raw.as_str().map(|s| s.trim().chars().take(128).collect()).unwrap_or_default();
            if name.is_empty() || !valid.contains(&name) {
                crate::jsv::del(body, "tool_choice");
            } else {
                body["tool_choice"] = json!({"type": "function", "name": name});
            }
        } else if !hosted.contains(ct) {
            crate::jsv::del(body, "tool_choice");
        }
    }
}

fn effort_from_model(m: &str) -> Option<&'static str> {
    EFFORTS.iter().find(|l| m.ends_with(&format!("-{l}"))).copied()
}

pub struct GrokCliPrepared {
    pub body: Value,
    pub session_id: String,
    pub req_id: String,
    pub turn: i64,
    pub model: String,
}

/// transformRequest: returns the body plus the per-request header identity.
pub fn grok_cli_transform(model: &str, mut body: Value, creds: &Value) -> GrokCliPrepared {
    let session_id = resolve_session(creds, &body);
    if let Some(n) = crate::translate::req::normalize_responses_input(&body["input"]) {
        body["input"] = Value::Array(n);
    }
    let empty = !truthy(&body["input"]) || body["input"].as_array().map(|a| a.is_empty()).unwrap_or(false);
    if empty {
        if let Some(msgs) = body["messages"].as_array().filter(|a| !a.is_empty()).cloned() {
            body["input"] = Value::Array(
                msgs.iter()
                    .map(|m| json!({"type": "message", "role": if truthy(&m["role"]) { m["role"].clone() } else { json!("user") }, "content": m["content"].as_str().map(str::to_owned).unwrap_or_else(|| if m["content"].is_null() { "\"\"".into() } else { m["content"].to_string() })}))
                    .collect(),
            );
            crate::jsv::del(&mut body, "messages");
        } else {
            body["input"] = json!([{"type": "message", "role": "user", "content": "..."}]);
        }
    }
    normalize_grok_input(&mut body);
    strip_stored_refs(&mut body);
    normalize_tools(&mut body);
    let turn = resolve_turn_idx(&session_id, &body["input"]);
    body["stream"] = json!(true);
    body["store"] = json!(false);

    let orig = body["model"].as_str().filter(|s| !s.is_empty()).unwrap_or(model).to_string();
    let effort = effort_from_model(&orig);
    let mut resolved = orig.clone();
    if let Some(e) = effort {
        resolved = resolved[..resolved.len() - e.len() - 1].to_string();
    }
    resolved = crate::registry::get_model_upstream_id("gcli", &resolved);
    if resolved == orig {
        resolved = crate::registry::get_model_upstream_id("grok-cli", &resolved);
    }
    body["model"] = json!(resolved);
    let supports = supports_reasoning_effort(&resolved);
    let eff_hint = if truthy(&body["reasoning_effort"]) { body["reasoning_effort"].clone() } else { effort.map(|e| json!(e)).unwrap_or(Value::Null) };
    if !body["reasoning"].is_object() {
        body["reasoning"] = json!({"summary": "concise"});
        if supports {
            body["reasoning"]["effort"] = json!(normalize_effort(&eff_hint));
        }
    } else {
        if supports {
            let e = if truthy(&body["reasoning"]["effort"]) { body["reasoning"]["effort"].clone() } else { eff_hint };
            body["reasoning"]["effort"] = json!(normalize_effort(&e));
        } else {
            crate::jsv::del(&mut body["reasoning"], "effort");
        }
        if !truthy(&body["reasoning"]["summary"]) {
            body["reasoning"]["summary"] = json!("concise");
        }
    }
    crate::jsv::del(&mut body, "reasoning_effort");
    if body["reasoning"]["effort"] != "none" {
        let mut inc = body["include"].as_array().cloned().unwrap_or_default();
        if !inc.iter().any(|v| v == "reasoning.encrypted_content") {
            inc.push(json!("reasoning.encrypted_content"));
        }
        body["include"] = Value::Array(inc);
    }
    if let Some(o) = body.as_object_mut() {
        o.retain(|k, _| ALLOW.contains(&k.as_str()));
    }
    GrokCliPrepared { body, session_id, req_id: uuid::Uuid::new_v4().to_string(), turn, model: resolved }
}

fn agent_id(creds: &Value) -> String {
    let psd = &creds["providerSpecificData"];
    if let Some(d) = psd["deviceId"].as_str().filter(|s| !s.is_empty()) {
        return d.to_string();
    }
    if let Some(a) = psd["agentId"].as_str().filter(|s| !s.is_empty()) {
        return a.to_string();
    }
    let mid = crate::exec::consistent_machine_id("grok-cli-agent");
    // JS: [m.slice(0,8), m.slice(8,12), "5"+m.slice(13,16), "a"+m.slice(17,20), m.slice(0,12)] on a 16-char id
    format!("{}-{}-5{}-a-{}", &mid[0..8], &mid[8..12], &mid[13..16], &mid[0..12])
}

pub struct GrokCli;

impl GrokCli {
    fn headers(&self, creds: &Value, stream: bool, p: &GrokCliPrepared) -> Headers {
        let cfg = self.config();
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.extend_obj(&cfg["headers"]);
        if let Some(t) = crate::exec::cred_str(creds, "accessToken").or_else(|| crate::exec::cred_str(creds, "apiKey")) {
            h.set("Authorization", format!("Bearer {t}"));
        }
        if stream {
            h.set("Accept", "text/event-stream");
        }
        let ident = cfg["clientIdentifier"].as_str().map(str::to_owned).or_else(|| h.get("x-grok-client-identifier").map(str::to_owned)).unwrap_or_else(|| crate::consts::s("GROK_CLI_CLIENT_IDENTIFIER").to_string());
        let ver = cfg["clientVersion"].as_str().map(str::to_owned).or_else(|| h.get("x-grok-client-version").map(str::to_owned)).unwrap_or_else(|| crate::consts::s("GROK_CLI_VERSION").to_string());
        h.set("x-grok-client-identifier", ident);
        h.set("x-grok-client-version", ver);
        let sid = if p.session_id.is_empty() { creds["connectionId"].as_str().map(str::to_owned).unwrap_or_else(|| uuid::Uuid::new_v4().to_string()) } else { p.session_id.clone() };
        h.set("x-grok-session-id", sid.clone());
        h.set("x-grok-conv-id", sid);
        h.set("x-grok-req-id", p.req_id.clone());
        h.set("x-grok-turn-idx", p.turn.to_string());
        h.set("x-grok-agent-id", agent_id(creds));
        h.set("x-grok-model-override", p.model.clone());
        let psd = &creds["providerSpecificData"];
        let pick = |a: &Value, b: &Value| a.as_str().filter(|s| !s.is_empty()).or_else(|| b.as_str().filter(|s| !s.is_empty())).map(str::to_owned);
        if let Some(e) = pick(&psd["email"], &creds["email"]) {
            h.set("x-email", e);
        }
        if let Some(u) = pick(&psd["userId"], &creds["userId"]).or_else(|| creds["providerUserId"].as_str().map(str::to_owned)) {
            h.set("x-userid", u);
        }
        h
    }
}

#[async_trait]
impl Executor for GrokCli {
    fn provider(&self) -> &str {
        "grok-cli"
    }
    fn build_url(&self, _m: &str, _s: bool, _i: usize, _c: &Value) -> Result<String, String> {
        Ok(self.config()["baseUrl"].as_str().unwrap_or("").to_string())
    }
    fn transform_request(&self, model: &str, body: Value, _s: bool, creds: &Value) -> Value {
        grok_cli_transform(model, body, creds).body
    }
    fn parse_error(&self, status: u16, body: &str) -> ParsedError {
        if status == 402 && !body.is_empty() {
            if let Ok(j) = serde_json::from_str::<Value>(body) {
                let m = if truthy(&j["error"]) { &j["error"] } else { &j["message"] };
                return ParsedError { status: 402, message: m.as_str().map(str::to_owned).unwrap_or_else(|| body.to_string()), resets_at_ms: None };
            }
        }
        ParsedError { status, message: if body.is_empty() { format!("HTTP {status}") } else { body.to_string() }, resets_at_ms: None }
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let url = self.build_url(args.model, args.stream, 0, args.creds)?;
        let p = grok_cli_transform(args.model, args.body.clone(), args.creds);
        let mut h = self.headers(args.creds, args.stream, &p);
        if let Some(o) = &args.override_headers {
            h.extend_obj(o);
        }
        let timeout = self.config()["timeoutMs"].as_u64().unwrap_or(60_000);
        let up = post_json(args.creds, &url, &h, &p.body, timeout).await?;
        Ok(ExecResult { response: up, url, headers: h.0, body: p.body, response_format: None })
    }
}

// ===========================================================================
// grok-web
// ===========================================================================

const GROK_UA: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36";

fn model_map(m: &str) -> Option<(&'static str, &'static str, bool)> {
    Some(match m {
        "grok-3" => ("grok-3", "MODEL_MODE_GROK_3", false),
        "grok-3-mini" => ("grok-3", "MODEL_MODE_GROK_3_MINI_THINKING", true),
        "grok-3-thinking" => ("grok-3", "MODEL_MODE_GROK_3_THINKING", true),
        "grok-4" => ("grok-4", "MODEL_MODE_GROK_4", false),
        "grok-4-mini" => ("grok-4-mini", "MODEL_MODE_GROK_4_MINI_THINKING", true),
        "grok-4-thinking" => ("grok-4", "MODEL_MODE_GROK_4_THINKING", true),
        "grok-4-heavy" => ("grok-4", "MODEL_MODE_HEAVY", true),
        "grok-4.1-mini" => ("grok-4-1-thinking-1129", "MODEL_MODE_GROK_4_1_MINI_THINKING", true),
        "grok-4.1-fast" => ("grok-4-1-thinking-1129", "MODEL_MODE_FAST", false),
        "grok-4.1-expert" => ("grok-4-1-thinking-1129", "MODEL_MODE_EXPERT", true),
        "grok-4.1-thinking" => ("grok-4-1-thinking-1129", "MODEL_MODE_GROK_4_1_THINKING", true),
        "grok-4.2" | "grok-4.20" | "grok-4.20-beta" => ("grok-420", "MODEL_MODE_GROK_420", false),
        _ => return None,
    })
}

fn rand_string(len: usize, alnum: bool) -> String {
    let chars: &[u8] = if alnum { b"abcdefghijklmnopqrstuvwxyz0123456789" } else { b"abcdefghijklmnopqrstuvwxyz" };
    let bytes = crate::jsv::rand_hex(len);
    bytes.as_bytes().chunks(2).map(|c| chars[u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap() as usize % chars.len()] as char).collect()
}

fn statsig_id() -> String {
    let coin = uuid::Uuid::new_v4().as_bytes()[0] < 128;
    let msg = if coin {
        format!("e:TypeError: Cannot read properties of null (reading 'children[\"{}\"]')", rand_string(5, true))
    } else {
        format!("e:TypeError: Cannot read properties of undefined (reading '{}')", rand_string(10, false))
    };
    base64::engine::general_purpose::STANDARD.encode(msg)
}

/// parseOpenAIMessages: flattens history into one prompt string.
pub fn flatten_messages(messages: &[Value]) -> String {
    let mut ex: Vec<(String, String)> = vec![];
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
        ex.push((role, content));
    }
    let last_user = ex.iter().rposition(|(r, _)| r == "user");
    ex.iter().enumerate().map(|(i, (r, t))| if Some(i) == last_user { t.clone() } else { format!("{r}: {t}") }).collect::<Vec<_>>().join("\n\n")
}

enum GrokEv {
    Error(String),
    Full(String),
    Delta(String),
    Fingerprint(String),
}

/// Parses grok.com NDJSON into content events.
fn grok_events(mut body: crate::exec::ByteStream) -> impl futures::Stream<Item = GrokEv> + Send {
    async_stream::stream! {
        let mut lines = crate::sse::LineParser::default();
        let mut fp = String::new();
        let mut pending: Vec<String> = vec![];
        loop {
            let chunk = body.next().await;
            let end = chunk.is_none();
            match chunk {
                Some(Ok(c)) => pending.extend(lines.push(&c)),
                Some(Err(e)) => { yield GrokEv::Error(e); return; }
                None => if let Some(l) = lines.finish() { pending.push(l) },
            }
            for line in pending.drain(..) {
                let Ok(ev) = serde_json::from_str::<Value>(line.trim()) else { continue };
                if truthy(&ev["error"]) {
                    let m = ev["error"]["message"].as_str().map(str::to_owned).unwrap_or_else(|| format!("Grok error: {}", crate::jsv::js_string(&ev["error"]["code"])));
                    yield GrokEv::Error(m);
                    return;
                }
                let r = &ev["result"]["response"];
                if !truthy(r) { continue; }
                if fp.is_empty() {
                    if let Some(h) = r["llmInfo"]["modelHash"].as_str() { fp = h.to_string(); yield GrokEv::Fingerprint(fp.clone()); }
                }
                if truthy(&r["modelResponse"]) {
                    let mr = &r["modelResponse"];
                    if let Some(m) = mr["message"].as_str().filter(|s| !s.is_empty()) { yield GrokEv::Full(m.to_string()); }
                    if let Some(h) = mr["metadata"]["llm_info"]["modelHash"].as_str() { fp = h.to_string(); yield GrokEv::Fingerprint(fp.clone()); }
                    continue;
                }
                if !r["token"].is_null() {
                    let t = crate::jsv::js_string(&r["token"]);
                    if !t.is_empty() { yield GrokEv::Delta(t); }
                }
            }
            if end { break; }
        }
    }
}

fn err_json(status: u16, msg: &str, code: Option<&str>) -> Upstream {
    let mut e = json!({"message": msg, "type": if status == 400 { "invalid_request" } else { "upstream_error" }});
    if let Some(c) = code {
        e["code"] = json!(c);
    }
    Upstream::json(status, &json!({"error": e}))
}

pub struct GrokWeb;

#[async_trait]
impl Executor for GrokWeb {
    fn provider(&self) -> &str {
        "grok-web"
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let url = self.config()["baseUrl"].as_str().unwrap_or("https://grok.com/rest/app-chat/conversations/new").to_string();
        let res = |up: Upstream, h: Vec<(String, String)>, body: Value| Ok(ExecResult { response: up, url: url.clone(), headers: h, body, response_format: Some("openai".into()) });
        let Some(msgs) = args.body["messages"].as_array().filter(|a| !a.is_empty()) else {
            return res(err_json(400, "Missing or empty messages array", None), vec![], args.body.clone());
        };
        let (gmodel, mode, _thinking) = model_map(args.model).unwrap_or(("grok-4-1-thinking-1129", "MODEL_MODE_FAST", false));
        let message = flatten_messages(msgs);
        if message.trim().is_empty() {
            return res(err_json(400, "Empty query after processing", None), vec![], args.body.clone());
        }
        let payload = json!({
            "temporary": true, "modelName": gmodel, "modelMode": mode, "message": message,
            "fileAttachments": [], "imageAttachments": [],
            "disableSearch": false, "enableImageGeneration": false, "returnImageBytes": false,
            "returnRawGrokInXaiRequest": false, "enableImageStreaming": false, "imageGenerationCount": 0,
            "forceConcise": false, "toolOverrides": {}, "enableSideBySide": true, "sendFinalMetadata": true,
            "isReasoning": false, "disableTextFollowUps": false, "disableMemory": true,
            "forceSideBySide": false, "isAsyncChat": false, "disableSelfHarmShortCircuit": false,
            "deviceEnvInfo": {"darkModeEnabled": false, "devicePixelRatio": 2, "screenWidth": 2056, "screenHeight": 1329, "viewportWidth": 2056, "viewportHeight": 1083},
        });
        let mut h = Headers::default();
        for (k, v) in [
            ("Accept", "*/*"),
            ("Accept-Encoding", "gzip, deflate, br, zstd"),
            ("Accept-Language", "en-US,en;q=0.9"),
            ("Baggage", "sentry-environment=production,sentry-release=d6add6fb0460641fd482d767a335ef72b9b6abb8,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c"),
            ("Cache-Control", "no-cache"),
            ("Content-Type", "application/json"),
            ("Origin", "https://grok.com"),
            ("Pragma", "no-cache"),
            ("Referer", "https://grok.com/"),
            ("Sec-Ch-Ua", "\"Google Chrome\";v=\"136\", \"Chromium\";v=\"136\", \"Not(A:Brand\";v=\"24\""),
            ("Sec-Ch-Ua-Mobile", "?0"),
            ("Sec-Ch-Ua-Platform", "\"macOS\""),
            ("Sec-Fetch-Dest", "empty"),
            ("Sec-Fetch-Mode", "cors"),
            ("Sec-Fetch-Site", "same-origin"),
            ("User-Agent", GROK_UA),
        ] {
            h.set(k, v);
        }
        h.set("x-statsig-id", statsig_id());
        h.set("x-xai-request-id", uuid::Uuid::new_v4().to_string());
        h.set("traceparent", format!("00-{}-{}-00", crate::jsv::rand_hex(16), crate::jsv::rand_hex(8)));
        if let Some(k) = crate::exec::cred_str(args.creds, "apiKey") {
            h.set("Cookie", format!("sso={}", k.strip_prefix("sso=").unwrap_or(k)));
        }
        let up = match post_json(args.creds, &url, &h, &payload, 60_000).await {
            Ok(u) => u,
            Err(e) => return res(err_json(502, &format!("Grok connection failed: {e}"), None), h.0, payload),
        };
        if !up.ok() {
            let st = up.status;
            let msg = match st {
                401 | 403 => "Grok auth failed — SSO cookie may be expired. Re-paste your sso cookie value from grok.com.".to_string(),
                429 => "Grok rate limited. Wait a moment and retry, or rotate cookies.".to_string(),
                _ => format!("Grok returned HTTP {st}"),
            };
            return res(err_json(st, &msg, Some(&format!("HTTP_{st}"))), h.0, payload);
        }
        let cid = format!("chatcmpl-grok-{}", &uuid::Uuid::new_v4().to_string()[..12]);
        let created = now_ms() / 1000;
        let model = args.model.to_string();
        let events = grok_events(up.body);
        if args.stream {
            let s = async_stream::stream! {
                let chunk = |delta: Value, fp: &str, fin: Value| json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "system_fingerprint": if fp.is_empty() { Value::Null } else { json!(fp) }, "choices": [{"index": 0, "delta": delta, "finish_reason": fin, "logprobs": null}]});
                yield Ok(Bytes::from(format!("data: {}\n\n", chunk(json!({"role": "assistant"}), "", Value::Null))));
                let mut fp = String::new();
                futures::pin_mut!(events);
                while let Some(ev) = events.next().await {
                    match ev {
                        GrokEv::Fingerprint(f) => fp = f,
                        GrokEv::Error(e) => {
                            yield Ok(Bytes::from(format!("data: {}\n\n", chunk(json!({"content": format!("[Error: {e}]")}), &fp, Value::Null))));
                            break;
                        }
                        GrokEv::Delta(d) => yield Ok(Bytes::from(format!("data: {}\n\n", chunk(json!({"content": d}), &fp, Value::Null)))),
                        GrokEv::Full(_) => {}
                    }
                }
                yield Ok(Bytes::from(format!("data: {}\n\n", chunk(json!({}), &fp, json!("stop")))));
                yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
            };
            return res(Upstream::synthetic(200, "text/event-stream", s.boxed()), h.0, payload);
        }
        let mut full = String::new();
        let mut fp = String::new();
        futures::pin_mut!(events);
        while let Some(ev) = events.next().await {
            match ev {
                GrokEv::Fingerprint(f) => fp = f,
                GrokEv::Error(e) => return res(Upstream::json(502, &json!({"error": {"message": e, "type": "upstream_error", "code": "GROK_ERROR"}})), h.0, payload),
                GrokEv::Full(m) => full = m,
                GrokEv::Delta(d) => full.push_str(&d),
            }
        }
        let toks = (full.chars().count() as f64 / 4.0).ceil() as i64;
        let out = json!({
            "id": cid, "object": "chat.completion", "created": created, "model": args.model,
            "system_fingerprint": if fp.is_empty() { Value::Null } else { json!(fp) },
            "choices": [{"index": 0, "message": {"role": "assistant", "content": full}, "finish_reason": "stop", "logprobs": null}],
            "usage": {"prompt_tokens": toks, "completion_tokens": toks, "total_tokens": toks * 2},
        });
        res(Upstream::json(200, &out), h.0, payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_transform() {
        let body = json!({"model": "grok-4.5-low", "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "function_call_output", "call_id": "orphan", "output": "x"},
            {"type": "custom_tool_call", "call_id": "c1", "name": "apply", "input": "patch"},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": {"ok": true}},
            {"type": "item_reference", "id": "msg_1"}
        ], "tools": [{"type": "custom", "name": "apply"}, {"type": "web_search"}, {"type": "function", "function": {"name": "f"}}], "max_tokens": 5, "foo": 1});
        let p = grok_cli_transform("grok-4.5-low", body, &json!({"connectionId": "conn1"}));
        let b = p.body;
        assert_eq!(b["input"].as_array().unwrap().len(), 3);
        assert_eq!(b["input"][1]["arguments"], "{\"input\":\"patch\"}");
        assert_eq!(b["input"][2]["output"], "{\"ok\":true}");
        assert_eq!(b["tools"][0]["parameters"]["required"][0], "input");
        assert_eq!(b["tools"][1]["type"], "web_search");
        assert_eq!(b["tools"][2]["name"], "f");
        assert_eq!(b["reasoning"]["effort"], "low");
        assert_eq!(b["model"], "grok-4.5");
        assert!(b.get("max_tokens").is_none() && b.get("foo").is_none());
        assert_eq!(b["include"][0], "reasoning.encrypted_content");
        assert_eq!(p.turn, 1);
    }

    #[test]
    fn web_flatten() {
        let m = json!([{"role": "system", "content": "s"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": [{"type": "text", "text": "c"}]}]);
        assert_eq!(flatten_messages(m.as_array().unwrap()), "system: s\n\nuser: a\n\nassistant: b\n\nc");
    }
}

pub fn pager_user_agent() -> &'static str {
    crate::consts::s("GROK_CLI_PAGER_USER_AGENT")
}

pub fn cli_version() -> &'static str {
    crate::consts::s("GROK_CLI_VERSION")
}
