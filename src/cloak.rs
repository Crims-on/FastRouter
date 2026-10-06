//! Claude Code identity helpers (port of 9router utils/claudeCloaking.js and
//! utils/claudeSignature.js). Anthropic only serves subscription OAuth tokens
//! for Claude Code-shaped requests, so these keep that shape.

use base64::Engine;
use serde_json::{Value, json};

use crate::consts::{CLAUDE_TOOL_SUFFIX, cc_default_tools, claude_cli_version};
use crate::jsv::{rand_hex, sha256_hex};

fn generate_billing_header(payload: &Value) -> String {
    let content = serde_json::to_string(payload).unwrap_or_default();
    let cch = &sha256_hex(&content)[..5];
    let build = &rand_hex(2)[..3];
    format!(
        "x-anthropic-billing-header: cc_version={}.{build}; cc_entrypoint=sdk-cli; cch={cch};",
        claude_cli_version()
    )
}

fn derive_uuid(seed: &str) -> String {
    let h = sha256_hex(seed);
    let nib = u8::from_str_radix(&h[16..17], 16).unwrap_or(0);
    format!(
        "{}-{}-4{}-{:x}{}-{}",
        &h[0..8],
        &h[8..12],
        &h[13..16],
        (nib & 0x3) | 0x8,
        &h[17..20],
        &h[20..32]
    )
}

fn fake_user_id(session_id: Option<&str>, api_key: &str) -> String {
    let device_id = if api_key.is_empty() { rand_hex(32) } else { sha256_hex(&format!("device:{api_key}")) };
    let account = if api_key.is_empty() { uuid::Uuid::new_v4().to_string() } else { derive_uuid(&format!("account:{api_key}")) };
    let clean = session_id
        .map(|s| {
            let s = s.trim();
            let lower = s.to_lowercase();
            if lower.starts_with("claude:") { s[7..].trim().to_string() } else { s.to_string() }
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    format!(r#"{{"device_id":"{device_id}","account_uuid":"{account}","session_id":"{clean}"}}"#)
}

pub fn extract_claude_session_id_from_user_id(user_id: &Value) -> Option<String> {
    let u = user_id.as_str().filter(|s| !s.is_empty())?;
    let strip = |s: &str| {
        let s = s.trim();
        if s.to_lowercase().starts_with("claude:") { s[7..].trim().to_string() } else { s.to_string() }
    };
    if u.starts_with('{') {
        let v: Value = serde_json::from_str(u).ok()?;
        let sid = v["session_id"].as_str().filter(|s| !s.is_empty())?;
        let c = strip(sid);
        return (!c.is_empty()).then_some(c);
    }
    let c = strip(u);
    (!c.is_empty()).then_some(c)
}

/// applyCloaking: billing header + fake metadata.user_id for OAuth tokens.
pub fn apply_cloaking(mut body: Value, api_key: &str, session_id: Option<&str>) -> Value {
    if !api_key.contains("sk-ant-oat") {
        return body;
    }
    let billing = json!({"type": "text", "text": generate_billing_header(&body)});
    match body["system"].take() {
        Value::Array(mut a) => {
            let already = a.first().and_then(|b| b["text"].as_str()).map(|t| t.starts_with("x-anthropic-billing-header:")).unwrap_or(false);
            if !already {
                a.insert(0, billing);
            }
            body["system"] = Value::Array(a);
        }
        Value::String(s) => body["system"] = json!([billing, {"type": "text", "text": s}]),
        _ => body["system"] = json!([billing]),
    }
    let existing = &body["metadata"]["user_id"];
    if !crate::jsv::truthy(existing) {
        body["metadata"]["user_id"] = json!(fake_user_id(session_id, api_key));
    }
    body
}

fn decoy_tools() -> Vec<Value> {
    cc_default_tools()
        .into_iter()
        .map(|n| json!({"name": n, "description": "This tool is currently unavailable.", "input_schema": {"type": "object", "properties": {}}}))
        .collect()
}

/// cloakClaudeTools → (body, suffixed→original map)
pub fn cloak_claude_tools(body: Value) -> (Value, Option<serde_json::Map<String, Value>>) {
    let Some(tools) = body["tools"].as_array().filter(|t| !t.is_empty()).cloned() else {
        return (body, None);
    };
    let mut map = serde_json::Map::new();
    let mut client_names = vec![];
    let mut decls = vec![];
    for tool in tools {
        if crate::jsv::truthy(&tool["type"]) {
            decls.push(tool);
            continue;
        }
        let name = tool["name"].as_str().unwrap_or("").to_string();
        let suffixed = format!("{name}{CLAUDE_TOOL_SUFFIX}");
        map.insert(suffixed.clone(), json!(name));
        client_names.push(name);
        let mut t = tool.clone();
        t["name"] = json!(suffixed);
        decls.push(t);
    }
    decls.extend(decoy_tools());
    let mut out = body.clone();
    out["tools"] = Value::Array(decls);
    if let Some(msgs) = out["messages"].as_array_mut() {
        for msg in msgs {
            if let Some(content) = msg["content"].as_array_mut() {
                for block in content {
                    if block["type"] == "tool_use" {
                        let n = block["name"].as_str().unwrap_or("").to_string();
                        block["name"] = json!(format!("{n}{CLAUDE_TOOL_SUFFIX}"));
                    }
                }
            }
        }
    }
    if body["tool_choice"]["type"] == "tool" {
        if let Some(n) = body["tool_choice"]["name"].as_str() {
            if client_names.iter().any(|c| c == n) {
                out["tool_choice"]["name"] = json!(format!("{n}{CLAUDE_TOOL_SUFFIX}"));
            }
        }
    }
    (out, (!map.is_empty()).then_some(map))
}

fn strip_cloak_suffix(name: &str) -> Option<String> {
    if !name.ends_with(CLAUDE_TOOL_SUFFIX) || cc_default_tools().contains(&name) {
        return None;
    }
    let orig = &name[..name.len() - CLAUDE_TOOL_SUFFIX.len()];
    (!orig.is_empty()).then(|| orig.to_string())
}

pub fn decloak_tool_names(mut body: Value, map: Option<&serde_json::Map<String, Value>>) -> Value {
    if let Some(content) = body["content"].as_array_mut() {
        for block in content {
            if block["type"] != "tool_use" {
                continue;
            }
            let name = block["name"].as_str().unwrap_or("").to_string();
            if let Some(orig) = map.and_then(|m| m.get(&name)) {
                block["name"] = orig.clone();
            } else if let Some(f) = strip_cloak_suffix(&name) {
                block["name"] = json!(f);
            }
        }
    }
    body
}

pub fn decloak_stream_chunk(mut chunk: Value, map: Option<&serde_json::Map<String, Value>>) -> Value {
    if chunk["type"] != "content_block_start" || chunk["content_block"]["type"] != "tool_use" {
        return chunk;
    }
    let Some(name) = chunk["content_block"]["name"].as_str().map(str::to_owned) else { return chunk };
    let orig = map.and_then(|m| m.get(&name)).and_then(|v| v.as_str().map(str::to_owned)).or_else(|| strip_cloak_suffix(&name));
    if let Some(o) = orig {
        chunk["content_block"]["name"] = json!(o);
    }
    chunk
}

/// Claude thinking signature validation.
pub fn is_valid_claude_signature(raw: &Value) -> bool {
    let Some(raw) = raw.as_str() else { return false };
    let sig = raw.trim();
    let sig = match sig.find('#') {
        Some(i) => sig[i + 1..].trim(),
        None => sig,
    };
    if sig.is_empty() || sig.len() > 32 * 1024 * 1024 {
        return false;
    }
    let dec = |s: &str| {
        base64::engine::general_purpose::STANDARD
            .decode(s.trim_end_matches('=').to_string() + &"=".repeat((4 - s.trim_end_matches('=').len() % 4) % 4))
            .ok()
    };
    match sig.as_bytes()[0] {
        b'E' => dec(sig).map(|d| d.first() == Some(&0x12)).unwrap_or(false),
        b'R' => {
            let Some(outer) = dec(sig) else { return false };
            if outer.first() != Some(&0x45) {
                return false;
            }
            let inner_s = String::from_utf8_lossy(&outer).to_string();
            dec(&inner_s).map(|d| d.first() == Some(&0x12)).unwrap_or(false)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloaking_injects_billing_and_user_id() {
        let b = apply_cloaking(json!({"system": "hi", "messages": []}), "sk-ant-oat01-x", Some("claude:abc"));
        assert!(b["system"][0]["text"].as_str().unwrap().starts_with("x-anthropic-billing-header:"));
        assert!(b["metadata"]["user_id"].as_str().unwrap().contains(r#""session_id":"abc""#));
        let plain = apply_cloaking(json!({"system": "hi"}), "sk-ant-api", None);
        assert_eq!(plain["system"], "hi");
    }

    #[test]
    fn default_signature_is_valid() {
        assert!(is_valid_claude_signature(&json!(crate::consts::sig_claude())));
        assert!(!is_valid_claude_signature(&json!("abc")));
    }
}
