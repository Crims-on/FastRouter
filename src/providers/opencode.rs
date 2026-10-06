//! OpenCode executors: opencode (free Zen tier, `Bearer public`), opencode-go
//! and opencode-zen (API key). Port of executors/opencode*.js and
//! utils/opencodeFingerprint.js.

use std::sync::{LazyLock, Mutex};

use async_trait::async_trait;
use regex::Regex;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::exec::{ExecArgs, ExecResult, Executor, Headers, base_execute, default_build_headers, default_build_url, default_transform_request};
use crate::jsv::{del, js_string, now_ms, truthy};
use crate::translate::req::{clamp_responses_call_id, coerce_responses_arguments, coerce_responses_output, normalize_responses_input};

const OPENCODE_UA: &str = "opencode/1.18.31";
const MAX_SESSION_LENGTH: usize = 256;
const MAX_TOOL_NAME_LEN: usize = 128;
const BASE62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const SESSION_FIELD: &str = "_opencodeSession";
const REQ_FIELD: &str = "_opencodeRequest";
pub const RENAMED_FIELD: &str = "__renamedToolNames";
const FINGERPRINT: [&str; 4] = ["bash", "glob", "grep", "read"];

static SESSION_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$").unwrap());
static REQUEST_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^msg_[0-9a-f]{12}[0-9A-Za-z]{14}$").unwrap());

fn has_valid_opencode_version(ua: &str) -> bool {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)opencode/(\d+)\.(\d+)(?:\.(\d+))?").unwrap());
    let Some(c) = RE.captures(ua) else { return false };
    let major: u64 = c[1].parse().unwrap_or(0);
    let minor: u64 = c[2].parse().unwrap_or(0);
    major > 1 || (major == 1 && minor >= 17)
}

fn unstable_random() -> String {
    let b = crate::jsv::rand_hex(14);
    b.as_bytes().chunks(2).map(|c| BASE62[u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap() as usize % 62] as char).collect()
}

fn time_hex(v: i128) -> String {
    (0..6).map(|i| format!("{:02x}", (v >> (40 - 8 * i)) & 0xff)).collect()
}

static SEQ: LazyLock<Mutex<(i64, i64)>> = LazyLock::new(Default::default);

pub fn generate_session_id(ts: i64) -> String {
    let counter = {
        let mut s = SEQ.lock().unwrap();
        if s.0 != ts {
            *s = (ts, 0);
        }
        s.1 += 1;
        s.1
    };
    let current = ts as i128 * 0x1000 + counter as i128;
    format!("ses_{}{}", time_hex(!current), unstable_random())
}

pub fn generate_request_id(ts: i64) -> String {
    format!("msg_{}{}", time_hex(ts as i128 * 0x1000 + 1), unstable_random())
}

fn digest_id(prefix: &str, seed: &str) -> String {
    let d = Sha256::digest(seed.as_bytes());
    let rand: String = d[6..20].iter().map(|b| BASE62[*b as usize % 62] as char).collect();
    format!("{prefix}_{}{rand}", hex::encode(&d[..6]))
}

pub fn translate_session_id(session_id: &str, client_tool: &str) -> String {
    if SESSION_RE.is_match(session_id.trim()) {
        return session_id.trim().to_string();
    }
    digest_id("ses", &format!("opencode\0{}\0{session_id}", if client_tool.is_empty() { "generic" } else { client_tool }))
}

fn normalize_session(v: &Value) -> Option<String> {
    let s = v.as_str()?.trim();
    (!s.is_empty() && s.len() <= MAX_SESSION_LENGTH).then(|| s.to_string())
}

fn header_ci<'a>(headers: &'a Value, name: &str) -> Option<&'a Value> {
    headers.as_object()?.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v)
}

static STABLE: LazyLock<Mutex<Vec<(String, String, i64)>>> = LazyLock::new(Default::default);

fn identity_key(creds: &Value) -> String {
    let conn = [&creds["connectionId"], &creds["id"]].into_iter().find(|v| truthy(v)).map(js_string);
    if let Some(c) = conn {
        return format!("opencode:conn:{}", c.chars().take(128).collect::<String>());
    }
    let raw = &creds["rawHeaders"];
    let auth = ["authorization", "Authorization", "x-api-key", "X-Api-Key"].iter().find_map(|k| raw[*k].as_str().filter(|s| !s.is_empty()));
    if let Some(a) = auth {
        return format!("opencode:auth:{}", &crate::jsv::sha256_hex(a)[..32]);
    }
    "opencode:default".into()
}

pub fn stable_session_id(creds: &Value) -> String {
    let key = identity_key(creds);
    let now = now_ms();
    let mut st = STABLE.lock().unwrap();
    st.retain(|e| now - e.2 <= 2 * 3_600_000);
    if let Some(pos) = st.iter().position(|e| e.0 == key) {
        let mut e = st.remove(pos);
        e.2 = now;
        let id = e.1.clone();
        st.push(e);
        return id;
    }
    let id = generate_session_id(now);
    if st.len() >= 1000 {
        st.remove(0);
    }
    st.push((key, id.clone(), now));
    id
}

fn last_user_text(body: &Value) -> String {
    let take600 = |s: &str| {
        let u: Vec<u16> = s.encode_utf16().collect();
        String::from_utf16_lossy(&u[u.len().saturating_sub(600)..])
    };
    let arr = body["messages"].as_array().or_else(|| body["input"].as_array());
    let Some(arr) = arr else {
        return body["input"].as_str().map(take600).unwrap_or_default();
    };
    for m in arr.iter().rev() {
        if m.is_null() || (truthy(&m["role"]) && m["role"] != "user") {
            continue;
        }
        if let Some(s) = m["content"].as_str().map(str::trim).filter(|s| !s.is_empty()) {
            return take600(s);
        }
        if let Some(a) = m["content"].as_array() {
            let t = a.iter().map(|p| p.as_str().map(str::to_owned).unwrap_or_else(|| p["text"].as_str().or_else(|| p["input_text"].as_str()).unwrap_or("").to_string())).collect::<Vec<_>>().join(" ");
            let t = t.trim();
            if !t.is_empty() {
                return take600(t);
            }
        }
    }
    String::new()
}

pub fn derive_request_id(session_id: &str, body: &Value) -> String {
    let t = last_user_text(body);
    if t.is_empty() {
        return generate_request_id(now_ms());
    }
    let id = digest_id("msg", &format!("opencode-req\0{session_id}\0{t}"));
    if REQUEST_RE.is_match(&id) { id } else { generate_request_id(now_ms()) }
}

fn body_has_session_hints(body: &Value) -> bool {
    let s = |v: &Value| v.as_str().map(|x| !x.trim().is_empty()).unwrap_or(false);
    if s(&body["session_id"]) || s(&body["conversation_id"]) || s(&body["prompt_cache_key"]) || s(&body["metadata"]["user_id"]) {
        return true;
    }
    if !body["request"]["sessionId"].is_null() && js_string(&body["request"]["sessionId"]) != "" {
        return true;
    }
    let arr = body["messages"].as_array().or_else(|| body["input"].as_array());
    let mut text = String::new();
    for m in arr.into_iter().flatten() {
        if m["role"] == "assistant" {
            match &m["content"] {
                Value::String(c) => text.push_str(c),
                Value::Array(a) => {
                    for p in a {
                        text.push_str(p["text"].as_str().or_else(|| p["output"].as_str()).unwrap_or(""));
                    }
                }
                _ => {}
            }
            if text.len() >= 50 {
                return true;
            }
        }
    }
    false
}

fn base_model_id(m: &str) -> String {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\([^()]+\)\s*$").unwrap());
    RE.replace(m, "").trim().to_string()
}

fn is_responses_model(m: &str) -> bool {
    let b = base_model_id(m);
    b == "muse-spark-1.2-contributor-free" || b == "muse-spark-1.3-contributor-free" || crate::registry::is_muse_spark_model(&b)
}

fn resolve_session(body: &Value, creds: &Value, provider_session: Option<&str>, client_tool: &str) -> String {
    let headers = &creds["rawHeaders"];
    let incoming = header_ci(headers, "x-opencode-session").and_then(normalize_session);
    if let Some(n) = &incoming {
        if SESSION_RE.is_match(n) {
            return n.clone();
        }
    }
    let hinted = incoming.or_else(|| provider_session.and_then(|p| normalize_session(&json!(p))));
    if let Some(h) = hinted {
        return translate_session_id(&h, client_tool);
    }
    if truthy(&creds["connectionId"]) || body_has_session_hints(body) {
        let s = crate::session::resolve_session_id(headers, body, creds["connectionId"].as_str(), "opencode");
        if !s.is_empty() {
            return translate_session_id(&s, client_tool);
        }
    }
    stable_session_id(creds)
}

fn resolve_request_id(body: &Value, creds: &Value, session: &str) -> String {
    if let Some(v) = header_ci(&creds["rawHeaders"], "x-opencode-request") {
        if let Some(n) = normalize_session(v).filter(|n| REQUEST_RE.is_match(n)) {
            return n;
        }
    }
    derive_request_id(session, body)
}

// ---------------------------------------------------------------------------
// Responses normalization shared by the three executors
// ---------------------------------------------------------------------------

pub fn normalize_responses_tools(body: &mut Value) {
    let Some(tools) = body["tools"].as_array().cloned() else { return };
    let mut valid = std::collections::HashSet::new();
    let mut out = vec![];
    for t in tools {
        if !t.is_object() {
            continue;
        }
        let f = if t["function"].is_object() { t["function"].clone() } else { Value::Null };
        let name = t["name"].as_str().or_else(|| f["name"].as_str()).unwrap_or("").trim().to_string();
        if name.is_empty() {
            continue;
        }
        let desc = t["description"].as_str().or_else(|| f["description"].as_str()).unwrap_or("").to_string();
        let mut params = if t["parameters"].is_object() {
            t["parameters"].clone()
        } else if f["parameters"].is_object() {
            f["parameters"].clone()
        } else {
            json!({"type": "object", "properties": {}})
        };
        if params["type"] == "object" && !truthy(&params["properties"]) {
            params["properties"] = json!({});
        }
        let n: String = name.chars().take(MAX_TOOL_NAME_LEN).collect();
        let mut nt = json!({"type": "function", "name": n});
        if !desc.is_empty() {
            nt["description"] = json!(desc);
        }
        nt["parameters"] = params;
        valid.insert(n);
        out.push(nt);
    }
    body["tools"] = Value::Array(out);
    if body["tool_choice"].is_object() && body["tool_choice"]["type"] == "function" {
        let n = body["tool_choice"]["name"].as_str().map(str::trim).unwrap_or("").to_string();
        if n.is_empty() || !valid.contains(&n) {
            del(body, "tool_choice");
        }
    }
}

pub fn sanitize_responses_items(body: &mut Value) {
    let Some(items) = body["input"].as_array().cloned() else { return };
    let mut out = vec![];
    for mut it in items {
        if !it.is_object() {
            out.push(it);
            continue;
        }
        if it["type"] == "reasoning" {
            continue;
        }
        del(&mut it, "encrypted_content");
        del(&mut it, "reasoning_encrypted_content");
        if it["type"] == "function_call" {
            let Some(n) = it["name"].as_str().map(str::trim).filter(|s| !s.is_empty()) else { continue };
            it["name"] = json!(n.chars().take(MAX_TOOL_NAME_LEN).collect::<String>());
            it["call_id"] = json!(clamp_responses_call_id(&it["call_id"]));
            it["arguments"] = json!(coerce_responses_arguments(&it["arguments"]));
        } else if it["type"] == "function_call_output" {
            it["call_id"] = json!(clamp_responses_call_id(&it["call_id"]));
            it["output"] = json!(coerce_responses_output(&it["output"]));
        }
        out.push(it);
    }
    body["input"] = Value::Array(out);
}

fn ensure_input(body: &mut Value) {
    if let Some(n) = normalize_responses_input(&body["input"]) {
        body["input"] = Value::Array(n);
    }
    if body["input"].as_array().map(|a| a.is_empty()).unwrap_or(true) {
        body["input"] = json!([{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}]);
    }
}

fn max_output_tokens(body: &mut Value) {
    if body.get("max_output_tokens").is_none() {
        if body.get("max_completion_tokens").is_some() {
            body["max_output_tokens"] = body["max_completion_tokens"].clone();
        } else if body.get("max_tokens").is_some() {
            body["max_output_tokens"] = body["max_tokens"].clone();
        }
    }
    del(body, "max_tokens");
    del(body, "max_completion_tokens");
}

fn simple_reasoning(out: &mut Value) {
    if out.get("reasoning_effort").is_some() && out.get("reasoning").is_none() {
        out["reasoning"] = json!({"effort": out["reasoning_effort"], "summary": "auto"});
    }
    if out["reasoning"].is_object() && !truthy(&out["reasoning"]["summary"]) {
        out["reasoning"]["summary"] = json!("auto");
    }
    del(out, "reasoning_effort");
}

// ---------------------------------------------------------------------------
// fingerprint tools
// ---------------------------------------------------------------------------

fn tool_name_of(t: &Value) -> String {
    if !t.is_object() {
        return String::new();
    }
    if let Some(n) = t["name"].as_str().map(str::trim).filter(|s| !s.is_empty()) {
        return n.to_string();
    }
    t["function"]["name"].as_str().map(|s| s.trim().to_string()).unwrap_or_default()
}

fn fp_key(name: &str) -> Option<&'static str> {
    let l = name.trim().to_lowercase();
    FINGERPRINT.iter().find(|f| **f == l).copied()
}

/// applyFingerprintTools → map sent-name → original name.
pub fn apply_fingerprint_tools(body: &mut Value, flat: bool) -> Map<String, Value> {
    let mut map = Map::new();
    if !body.is_object() {
        return map;
    }
    let had = body["tools"].as_array().map(|a| !a.is_empty()).unwrap_or(false);
    let mut tools: Vec<Value> = vec![];
    let mut seen = std::collections::HashSet::new();
    for t in body["tools"].as_array().cloned().unwrap_or_default() {
        if !t.is_object() {
            tools.push(t);
            continue;
        }
        let cur = tool_name_of(&t);
        let Some(key) = fp_key(&cur) else {
            tools.push(t);
            continue;
        };
        if !seen.insert(key) {
            continue;
        }
        if cur != key {
            map.insert(key.into(), json!(cur));
            let mut nt = t.clone();
            if t["function"].is_object() {
                nt["function"]["name"] = json!(key);
            } else {
                nt["name"] = json!(key);
            }
            tools.push(nt);
        } else {
            tools.push(t);
        }
    }
    for name in FINGERPRINT {
        if tools.iter().any(|t| fp_key(&tool_name_of(t)) == Some(name)) {
            continue;
        }
        let desc = "This tool is currently unavailable and must not be used.";
        tools.push(if flat {
            json!({"type": "function", "name": name, "description": desc, "parameters": {"type": "object", "properties": {}}})
        } else {
            json!({"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": {}}}})
        });
    }
    body["tools"] = Value::Array(tools);
    if !map.is_empty() && body["tool_choice"].is_object() {
        let tc = body["tool_choice"].clone();
        if let Some(n) = tc["name"].as_str() {
            if let Some(k) = fp_key(n).filter(|k| map.contains_key(*k)) {
                body["tool_choice"]["name"] = json!(k);
            }
        } else if let Some(n) = tc["function"]["name"].as_str() {
            if let Some(k) = fp_key(n).filter(|k| map.contains_key(*k)) {
                body["tool_choice"]["function"]["name"] = json!(k);
            }
        }
    }
    if !truthy(&body["tool_choice"]) {
        if flat {
            body["tool_choice"] = json!("auto");
        } else if !had {
            body["tool_choice"] = json!("none");
        }
    }
    map
}

/// restoreToolNames (in place).
pub fn restore_tool_names(p: &mut Value, map: &Map<String, Value>) {
    if map.is_empty() || p.is_null() {
        return;
    }
    if let Some(a) = p.as_array_mut() {
        for i in a.iter_mut() {
            restore_tool_names(i, map);
        }
        return;
    }
    if !p.is_object() {
        return;
    }
    let fix = |v: &mut Value| {
        if let Some(n) = v.as_str() {
            if let Some(o) = map.get(n) {
                *v = o.clone();
            }
        }
    };
    if p["type"] == "content_block_start" && p["content_block"]["type"] == "tool_use" {
        fix(&mut p["content_block"]["name"]);
    }
    if let Some(c) = p.get_mut("content").and_then(|c| c.as_array_mut()) {
        for b in c.iter_mut() {
            if b["type"] == "tool_use" {
                fix(&mut b["name"]);
            }
        }
    }
    if let Some(ch) = p.get_mut("choices").and_then(|c| c.as_array_mut()) {
        for c in ch.iter_mut() {
            for holder in ["delta", "message"] {
                if let Some(tcs) = c.get_mut(holder).and_then(|h| h.get_mut("tool_calls")).and_then(|t| t.as_array_mut()) {
                    for tc in tcs.iter_mut() {
                        if tc["function"].is_object() {
                            fix(&mut tc["function"]["name"]);
                        }
                    }
                }
            }
        }
    }
    if let Some(o) = p.get_mut("output").and_then(|o| o.as_array_mut()) {
        for it in o.iter_mut() {
            if it["type"] == "function_call" {
                fix(&mut it["name"]);
            }
        }
    }
    if p["item"]["type"] == "function_call" {
        fix(&mut p["item"]["name"]);
    }
}

fn normalize_reasoning(model: &str, body: &mut Value) {
    let cur = if body["reasoning"].is_object() { body["reasoning"].clone() } else { Value::Null };
    let req = body["reasoning_effort"].as_str().map(str::to_owned).or_else(|| cur["effort"].as_str().map(str::to_owned));
    let Some(req) = req else { return };
    let clean = base_model_id(if model.is_empty() { body["model"].as_str().unwrap_or("") } else { model });
    let levels = crate::caps::thinking_levels(Some("opencode"), &clean).unwrap_or_default();
    let mut effort = req.trim().to_lowercase();
    if (effort == "max" || effort == "ultra") && !levels.is_empty() && !levels.contains(&effort) {
        if effort == "ultra" && levels.iter().any(|l| l == "max") {
            effort = "max".into();
        } else if levels.iter().any(|l| l == "xhigh") {
            effort = "xhigh".into();
        }
    }
    let mut r = if cur.is_object() { cur } else { json!({}) };
    r["effort"] = json!(effort);
    if !truthy(&r["summary"]) {
        r["summary"] = json!("auto");
    }
    body["reasoning"] = r;
    del(body, "reasoning_effort");
}

// ---------------------------------------------------------------------------
// opencode (free)
// ---------------------------------------------------------------------------

pub struct OpenCode;

impl OpenCode {
    fn prepare(&self, args: &mut ExecArgs<'_>) {
        let session = resolve_session(&args.body, args.creds, args.session_id.as_deref(), args.client_tool.as_deref().unwrap_or(""));
        let req = resolve_request_id(&args.body, args.creds, &session);
        args.creds[SESSION_FIELD] = json!(session);
        args.creds[REQ_FIELD] = json!(req);
    }
}

#[async_trait]
impl Executor for OpenCode {
    fn provider(&self) -> &str {
        "opencode"
    }
    fn build_url(&self, model: &str, _s: bool, _i: usize, _c: &Value) -> Result<String, String> {
        let base = self.config()["baseUrl"].as_str().unwrap_or("https://opencode.ai").to_string();
        Ok(if is_responses_model(model) {
            format!("{base}/zen/v1/responses")
        } else if base_model_id(model) == "union-alpha" {
            format!("{base}/zen/v1/messages")
        } else {
            format!("{base}/zen/v1/chat/completions")
        })
    }
    fn build_headers(&self, creds: &Value, stream: bool, url: &str, _m: &str, _b: &Value) -> Headers {
        let raw = &creds["rawHeaders"];
        let get = |k: &str| header_ci(raw, k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let ua = get("user-agent");
        let session = creds[SESSION_FIELD].as_str().map(str::to_owned).unwrap_or_else(|| resolve_session(&Value::Null, creds, None, ""));
        let down_req = normalize_session(&json!(get("x-opencode-request"))).filter(|n| REQUEST_RE.is_match(n));
        let req = creds[REQ_FIELD].as_str().map(str::to_owned).or(down_req).unwrap_or_else(|| generate_request_id(now_ms()));
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.set("Authorization", "Bearer public");
        h.set("User-Agent", if has_valid_opencode_version(&ua) { ua } else { OPENCODE_UA.into() });
        let client = get("x-opencode-client");
        h.set("x-opencode-client", if client.is_empty() { "desktop".into() } else { client });
        h.set("x-opencode-session", session);
        h.set("x-opencode-request", req);
        let proj = get("x-opencode-project");
        h.set("x-opencode-project", if proj.is_empty() { "global".into() } else { proj });
        h.set("Accept", if stream { "text/event-stream" } else { "*/*" });
        if url.ends_with("/messages") {
            h.set("anthropic-version", crate::consts::ANTHROPIC_API_VERSION);
        }
        h
    }
    fn transform_request(&self, model: &str, mut body: Value, _s: bool, _c: &Value) -> Value {
        if body.is_object() && !model.is_empty() && !truthy(&body["model"]) {
            body["model"] = json!(model);
        }
        if body.is_object() {
            body["stream"] = json!(true);
        }
        let m = if model.is_empty() { body["model"].as_str().unwrap_or("").to_string() } else { model.to_string() };
        if is_responses_model(&m) && body.is_object() {
            let forced = self.config()["quirks"]["forceAutoToolChoiceModels"].as_array().map(|a| a.iter().any(|x| x.as_str() == Some(&base_model_id(model)))).unwrap_or(false);
            if body.get("tool_choice").is_some() && body["tool_choice"] != "auto" && forced {
                body["tool_choice"] = json!("auto");
            }
            ensure_input(&mut body);
            max_output_tokens(&mut body);
            normalize_reasoning(model, &mut body);
            body["stream"] = json!(true);
            body["store"] = json!(false);
            normalize_responses_tools(&mut body);
            sanitize_responses_items(&mut body);
            apply_fingerprint_tools(&mut body, true);
        } else {
            apply_fingerprint_tools(&mut body, false);
        }
        crate::exec::inject_reasoning_content("opencode", model, body)
    }
    async fn execute(&self, mut args: ExecArgs<'_>) -> Result<ExecResult, String> {
        self.prepare(&mut args);
        // Same rename the transform applies, recorded on the per-request
        // credentials so the response side can restore caller spellings.
        let mut probe = args.body.clone();
        let flat = is_responses_model(args.model);
        if flat {
            normalize_responses_tools(&mut probe);
        }
        let map = apply_fingerprint_tools(&mut probe, flat);
        if !map.is_empty() {
            args.creds[RENAMED_FIELD] = Value::Object(map);
        }
        base_execute(self, args).await
    }
}

// ---------------------------------------------------------------------------
// opencode-go / opencode-zen (API-key)
// ---------------------------------------------------------------------------

fn native_session(headers: &Value, strict: bool) -> Option<String> {
    let v = header_ci(headers, "x-opencode-session").and_then(normalize_session)?;
    if strict && !SESSION_RE.is_match(&v) {
        return None;
    }
    Some(v)
}

fn go_translated(session: &str, client_tool: &str) -> String {
    let d = crate::jsv::sha256_hex(&format!("opencode-go\0{}\0{session}", if client_tool.is_empty() { "generic" } else { client_tool }));
    format!("ses_{}", &d[..32])
}

pub struct OpenCodeGo;

fn go_is_responses(model: &str) -> bool {
    crate::registry::get_model_target_format("opencode-go", model).as_deref() == Some("openai-responses")
}

#[async_trait]
impl Executor for OpenCodeGo {
    fn provider(&self) -> &str {
        "opencode-go"
    }
    fn build_url(&self, model: &str, stream: bool, idx: usize, creds: &Value) -> Result<String, String> {
        if go_is_responses(model) {
            return Ok("https://opencode.ai/zen/go/v1/responses".into());
        }
        default_build_url("opencode-go", &self.config(), model, stream, idx, creds)
    }
    fn build_headers(&self, creds: &Value, stream: bool, url: &str, model: &str, body: &Value) -> Headers {
        let mut h = default_build_headers("opencode-go", &self.config(), creds, stream, url, model, body);
        let s = creds["_opencodeGoSession"].as_str().map(str::to_owned).unwrap_or_else(|| {
            native_session(&creds["rawHeaders"], false).unwrap_or_else(|| go_translated(&crate::session::resolve_session_id(&creds["rawHeaders"], &Value::Null, creds["connectionId"].as_str(), "opencode-go"), ""))
        });
        h.set("x-opencode-session", s);
        h
    }
    fn transform_request(&self, model: &str, body: Value, _s: bool, _c: &Value) -> Value {
        let m = if model.is_empty() { body["model"].as_str().unwrap_or("").to_string() } else { model.to_string() };
        let mut out = default_transform_request("opencode-go", model, body);
        if !go_is_responses(&m) {
            return out;
        }
        ensure_input(&mut out);
        max_output_tokens(&mut out);
        simple_reasoning(&mut out);
        out["stream"] = json!(true);
        out["store"] = json!(false);
        normalize_responses_tools(&mut out);
        sanitize_responses_items(&mut out);
        out
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let s = native_session(&args.creds["rawHeaders"], false).unwrap_or_else(|| {
            let resolved = args.session_id.as_deref().and_then(|p| normalize_session(&json!(p))).unwrap_or_else(|| crate::session::resolve_session_id(&args.creds["rawHeaders"], &args.body, args.creds["connectionId"].as_str(), "opencode-go"));
            go_translated(&resolved, args.client_tool.as_deref().unwrap_or(""))
        });
        args.creds["_opencodeGoSession"] = json!(s);
        base_execute(self, args).await
    }
}

pub struct OpenCodeZen;

fn zen_tools(body: &mut Value, flat: bool) {
    let mut present: std::collections::HashSet<String> = body["tools"].as_array().into_iter().flatten().map(tool_name_of).filter(|n| !n.is_empty()).collect();
    if !body["tools"].is_array() {
        body["tools"] = json!([]);
    }
    for name in FINGERPRINT {
        if present.contains(name) {
            continue;
        }
        let desc = format!("OpenCode built-in {name} tool");
        body["tools"].as_array_mut().unwrap().push(if flat {
            json!({"type": "function", "name": name, "description": desc, "parameters": {"type": "object", "properties": {}}})
        } else {
            json!({"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": {}}}})
        });
        present.insert(name.to_string());
    }
}

#[async_trait]
impl Executor for OpenCodeZen {
    fn provider(&self) -> &str {
        "opencode-zen"
    }
    fn build_url(&self, model: &str, stream: bool, idx: usize, creds: &Value) -> Result<String, String> {
        if crate::registry::is_muse_spark_model(&base_model_id(model)) {
            return Ok("https://opencode.ai/zen/v1/responses".into());
        }
        default_build_url("opencode-zen", &self.config(), model, stream, idx, creds)
    }
    fn build_headers(&self, creds: &Value, stream: bool, url: &str, model: &str, body: &Value) -> Headers {
        let mut h = default_build_headers("opencode-zen", &self.config(), creds, stream, url, model, body);
        let raw = &creds["rawHeaders"];
        let get = |k: &str| header_ci(raw, k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let ua = get("user-agent");
        h.set("User-Agent", if has_valid_opencode_version(&ua) { ua } else { OPENCODE_UA.into() });
        let client = get("x-opencode-client");
        h.set("x-opencode-client", if client.is_empty() { "desktop".into() } else { client });
        let s = creds["_opencodeZenSession"].as_str().map(str::to_owned).unwrap_or_else(|| {
            native_session(raw, true).unwrap_or_else(|| translate_session_id(&crate::session::resolve_session_id(raw, &Value::Null, creds["connectionId"].as_str(), "opencode-zen"), ""))
        });
        h.set("x-opencode-session", s);
        h
    }
    fn transform_request(&self, model: &str, body: Value, _s: bool, _c: &Value) -> Value {
        let m = if model.is_empty() { body["model"].as_str().unwrap_or("").to_string() } else { model.to_string() };
        let mut out = default_transform_request("opencode-zen", model, body);
        if out.is_object() {
            out["stream"] = json!(true);
        }
        if !crate::registry::is_muse_spark_model(&base_model_id(&m)) {
            zen_tools(&mut out, false);
            return out;
        }
        ensure_input(&mut out);
        max_output_tokens(&mut out);
        simple_reasoning(&mut out);
        out["stream"] = json!(true);
        out["store"] = json!(false);
        zen_tools(&mut out, true);
        normalize_responses_tools(&mut out);
        sanitize_responses_items(&mut out);
        out
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let s = native_session(&args.creds["rawHeaders"], true).unwrap_or_else(|| {
            let resolved = args.session_id.as_deref().and_then(|p| normalize_session(&json!(p))).unwrap_or_else(|| crate::session::resolve_session_id(&args.creds["rawHeaders"], &args.body, args.creds["connectionId"].as_str(), "opencode-zen"));
            translate_session_id(&resolved, args.client_tool.as_deref().unwrap_or(""))
        });
        args.creds["_opencodeZenSession"] = json!(s);
        base_execute(self, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        let s = generate_session_id(1_700_000_000_000);
        assert!(SESSION_RE.is_match(&s), "{s}");
        let r = generate_request_id(1_700_000_000_000);
        assert!(REQUEST_RE.is_match(&r), "{r}");
        assert!(SESSION_RE.is_match(&translate_session_id("abc", "")));
        let d = derive_request_id("ses_x", &json!({"messages": [{"role": "user", "content": "hello"}]}));
        assert_eq!(d, derive_request_id("ses_x", &json!({"messages": [{"role": "user", "content": "hello"}]})));
    }

    #[test]
    fn fingerprint() {
        let mut b = json!({"tools": [{"type": "function", "function": {"name": "Bash"}}, {"type": "function", "function": {"name": "bash"}}, {"type": "function", "function": {"name": "Edit"}}]});
        let m = apply_fingerprint_tools(&mut b, false);
        assert_eq!(m["bash"], "Bash");
        let names: Vec<String> = b["tools"].as_array().unwrap().iter().map(tool_name_of).collect();
        assert_eq!(names, vec!["bash", "Edit", "glob", "grep", "read"]);
        let mut chunk = json!({"choices": [{"delta": {"tool_calls": [{"function": {"name": "bash"}}]}}]});
        restore_tool_names(&mut chunk, &m);
        assert_eq!(chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "Bash");
        let mut empty = json!({"messages": []});
        apply_fingerprint_tools(&mut empty, false);
        assert_eq!(empty["tool_choice"], "none");
    }
}
