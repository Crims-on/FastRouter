//! Session-id helpers (port of 9router utils/sessionManager.js) and the
//! Gemini thought-signature cache (services/thoughtSignatureStore.js, memory only).

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use regex::Regex;
use serde_json::Value;

use crate::jsv::{now_ms, sha256_hex};

const SESSION_TTL_MS: i64 = 2 * 60 * 60 * 1000;

struct Entry {
    session_id: String,
    last_used: i64,
}

static RUNTIME: LazyLock<Mutex<HashMap<String, Entry>>> = LazyLock::new(Default::default);
static ASSISTANT: LazyLock<Mutex<HashMap<String, Entry>>> = LazyLock::new(Default::default);
static CONTINUATION: LazyLock<Mutex<HashMap<String, Entry>>> = LazyLock::new(Default::default);

pub fn generate_binary_style_id() -> String {
    format!("{}{}", uuid::Uuid::new_v4(), now_ms())
}

fn sweep(map: &mut HashMap<String, Entry>, cap: usize) {
    let now = now_ms();
    map.retain(|_, e| now - e.last_used <= SESSION_TTL_MS);
    if map.len() >= cap {
        if let Some(k) = map.iter().min_by_key(|(_, e)| e.last_used).map(|(k, _)| k.clone()) {
            map.remove(&k);
        }
    }
}

pub fn derive_session_id(connection_id: Option<&str>) -> String {
    let Some(cid) = connection_id.filter(|c| !c.is_empty()) else {
        return generate_binary_style_id();
    };
    let mut store = RUNTIME.lock().unwrap();
    if let Some(e) = store.get_mut(cid) {
        e.last_used = now_ms();
        return e.session_id.clone();
    }
    sweep(&mut store, 1000);
    let sid = generate_binary_style_id();
    store.insert(cid.to_string(), Entry { session_id: sid.clone(), last_used: now_ms() });
    sid
}

fn normalize(v: &Value) -> Option<String> {
    let s = v.as_str()?.trim();
    if s.is_empty() || s.len() > 256 {
        return None;
    }
    Some(s.to_string())
}

fn header(headers: &Value, key: &str) -> Option<String> {
    normalize(&headers[key]).or_else(|| normalize(&headers[key.to_lowercase().as_str()]))
}

fn claude_code_session(user_id: &Value) -> Option<String> {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_session_([a-f0-9-]+)$").unwrap());
    let u = user_id.as_str().filter(|s| !s.is_empty())?;
    if let Some(c) = RE.captures(u) {
        return Some(c[1].to_string());
    }
    if u.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<Value>(u) {
            return normalize(&v["session_id"]);
        }
    }
    None
}

fn antigravity_session(body: &Value) -> Option<String> {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^[a-z]+/([0-9a-f-]{36})/").unwrap());
    let sid = &body["request"]["sessionId"];
    if !sid.is_null() && *sid != Value::String(String::new()) {
        let s = match sid {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        return normalize(&Value::String(s));
    }
    let rid = body["requestId"].as_str()?;
    RE.captures(rid).and_then(|c| normalize(&Value::String(c[1].to_string())))
}

fn client_session_id(headers: &Value, body: &Value, scope: &str) -> Option<String> {
    if let Some(c) = claude_code_session(&body["metadata"]["user_id"]).or_else(|| header(headers, "x-claude-code-session-id")) {
        return Some(format!("claude:{c}"));
    }
    if let Some(a) = antigravity_session(body) {
        return Some(format!("antigravity:{a}"));
    }
    for k in ["x-session-id", "session-id", "session_id", "x-amp-thread-id"] {
        if let Some(v) = header(headers, k) {
            return Some(v);
        }
    }
    if scope != "kiro" {
        if let Some(v) = header(headers, "x-client-request-id") {
            return Some(v);
        }
    }
    normalize(&body["prompt_cache_key"])
        .or_else(|| normalize(&body["session_id"]))
        .or_else(|| normalize(&body["conversation_id"]))
        .or_else(|| if scope == "kiro" { None } else { normalize(&body["metadata"]["user_id"]) })
}

fn accumulate_assistant_text(body: &Value) -> String {
    let items = body["messages"].as_array().or_else(|| body["input"].as_array());
    let mut text = String::new();
    for item in items.into_iter().flatten() {
        if item["role"] != "assistant" {
            continue;
        }
        match &item["content"] {
            Value::String(s) => text.push_str(s),
            Value::Array(parts) => {
                for c in parts {
                    text.push_str(c["text"].as_str().or_else(|| c["output"].as_str()).unwrap_or(""));
                }
            }
            _ => {}
        }
        if text.chars().count() >= 50 {
            break;
        }
    }
    text
}

fn assistant_text_session_id(scope: &str, body: &Value) -> Option<String> {
    let text = accumulate_assistant_text(body);
    if text.chars().count() < 50 {
        return None;
    }
    let head: String = text.chars().take(50).collect();
    let hash = sha256_hex(&format!("{scope}:{head}"))[..16].to_string();
    let mut store = ASSISTANT.lock().unwrap();
    if let Some(e) = store.get_mut(&hash) {
        e.last_used = now_ms();
        return Some(e.session_id.clone());
    }
    sweep(&mut store, 5000);
    let sid = generate_binary_style_id();
    store.insert(hash, Entry { session_id: sid.clone(), last_used: now_ms() });
    Some(sid)
}

/// resolveSessionIdentity → (sessionId, ephemeral)
pub fn resolve_session_identity(
    headers: &Value,
    body: &Value,
    connection_id: Option<&str>,
    workspace_id: Option<&str>,
    scope: &str,
) -> (String, bool) {
    if let Some(c) = client_session_id(headers, body, scope) {
        return (c, false);
    }
    if scope != "kiro" {
        if let Some(a) = assistant_text_session_id(&format!("{scope}:{}", connection_id.unwrap_or("")), body) {
            return (a, false);
        }
    }
    if let Some(ws) = workspace_id.and_then(|w| normalize(&Value::String(w.to_string()))) {
        return (ws, false);
    }
    if scope == "kiro" {
        return (generate_binary_style_id(), true);
    }
    (derive_session_id(connection_id), false)
}

pub fn resolve_session_id(headers: &Value, body: &Value, connection_id: Option<&str>, scope: &str) -> String {
    resolve_session_identity(headers, body, connection_id, None, scope).0
}

pub fn resolve_continuation_id(session_id: &str, connection_id: &str, scope: &str, ephemeral: bool) -> String {
    if ephemeral {
        return uuid::Uuid::new_v4().to_string();
    }
    let key = format!("{scope}:{connection_id}:{session_id}");
    let mut store = CONTINUATION.lock().unwrap();
    if let Some(e) = store.get_mut(&key) {
        e.last_used = now_ms();
        return e.session_id.clone();
    }
    sweep(&mut store, 5000);
    let id = uuid::Uuid::new_v4().to_string();
    store.insert(key, Entry { session_id: id.clone(), last_used: now_ms() });
    id
}

/// Antigravity numeric session format "-<int63>".
pub fn to_numeric_session_id(session_id: Option<&str>) -> Option<String> {
    let v = session_id.map(str::trim).filter(|s| !s.is_empty() && s.len() <= 256)?;
    static NUM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^-?\d+$").unwrap());
    if NUM.is_match(v) {
        return Some(v.to_string());
    }
    use sha2::{Digest, Sha256};
    let h = Sha256::digest(v.as_bytes());
    let mut b = [0u8; 8];
    b.copy_from_slice(&h[..8]);
    let n = u64::from_be_bytes(b) & 0x7fff_ffff_ffff_ffff;
    Some(format!("-{n}"))
}

// ---------------------------------------------------------------------------
// Gemini thought signatures (RAM cache)
// ---------------------------------------------------------------------------

struct Sig {
    signature: String,
    family: Option<String>,
    expires_at: i64,
}

static SIGS: LazyLock<Mutex<HashMap<String, Sig>>> = LazyLock::new(Default::default);

pub fn signature_family(model: Option<&str>) -> Option<String> {
    let m = model?.to_lowercase();
    if m.is_empty() {
        return None;
    }
    if m.contains("claude") {
        return Some("claude".into());
    }
    if m.contains("gemini") {
        return Some("gemini".into());
    }
    Some(m)
}

pub fn store_thought_signature(tool_call_id: &str, signature: &str, session_id: Option<&str>, model: Option<&str>) {
    if tool_call_id.is_empty() || signature.is_empty() {
        return;
    }
    let family = signature_family(model);
    let mut map = SIGS.lock().unwrap();
    let now = now_ms();
    map.retain(|_, s| s.expires_at > now);
    while map.len() > 2000 {
        let Some(k) = map.keys().next().cloned() else { break };
        map.remove(&k);
    }
    let mut keys = vec![];
    if let Some(s) = session_id.filter(|s| !s.is_empty()) {
        keys.push(format!("{s}:{tool_call_id}"));
    }
    keys.push(tool_call_id.to_string());
    for k in keys {
        map.insert(k, Sig { signature: signature.to_string(), family: family.clone(), expires_at: now + 3_600_000 });
    }
}

pub fn get_thought_signature(tool_call_id: &str, session_id: Option<&str>, model: Option<&str>) -> Option<String> {
    if tool_call_id.is_empty() {
        return None;
    }
    let family = signature_family(model);
    let map = SIGS.lock().unwrap();
    let ok = |s: &Sig| s.expires_at > now_ms() && (s.family.is_none() || family.is_none() || s.family == family);
    if let Some(sid) = session_id.filter(|s| !s.is_empty()) {
        if let Some(s) = map.get(&format!("{sid}:{tool_call_id}")).filter(|s| ok(s)) {
            return Some(s.signature.clone());
        }
    }
    map.get(tool_call_id).filter(|s| ok(s)).map(|s| s.signature.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn numeric_session() {
        assert_eq!(to_numeric_session_id(Some("-123")).unwrap(), "-123");
        assert!(to_numeric_session_id(Some("abc")).unwrap().starts_with('-'));
    }

    #[test]
    fn claude_session_from_user_id() {
        let body = json!({"metadata": {"user_id": "user_x_account__session_ab-12"}});
        assert_eq!(resolve_session_id(&json!({}), &body, None, ""), "claude:ab-12");
    }
}
