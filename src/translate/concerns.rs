//! Shared translator helpers (port of 9router translator/concerns/* and
//! formats/maxTokens.js, formats/openai.js).

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};

use crate::consts::{DEFAULT_MAX_TOKENS, DEFAULT_MIN_TOKENS};
use crate::jsv::{del, js_string, now_ms, omit, truthy};

// ---------------------------------------------------------------------------
// max tokens
// ---------------------------------------------------------------------------

pub fn adjust_max_tokens(body: &Value, ceiling: Option<i64>) -> i64 {
    let ceiling = ceiling.unwrap_or(DEFAULT_MAX_TOKENS);
    let mut max = body["max_tokens"].as_i64().filter(|n| *n != 0).unwrap_or(DEFAULT_MAX_TOKENS);
    if body["tools"].as_array().map(|t| !t.is_empty()).unwrap_or(false) && max < DEFAULT_MIN_TOKENS {
        max = DEFAULT_MIN_TOKENS;
    }
    if let Some(budget) = body["thinking"]["budget_tokens"].as_i64().filter(|b| *b != 0) {
        if max <= budget {
            max = budget + 1024;
        }
    }
    max.min(ceiling)
}

// ---------------------------------------------------------------------------
// data URIs / content helpers
// ---------------------------------------------------------------------------

pub fn parse_data_uri(url: &Value) -> Option<(String, String)> {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)^data:([^;]+);base64,(.+)$").unwrap());
    let c = RE.captures(url.as_str()?)?;
    Some((c[1].to_string(), c[2].to_string()))
}

pub fn encode_data_uri(mime: &str, b64: &str) -> String {
    format!("data:{mime};base64,{b64}")
}

/// collapseTextParts: a lone text part becomes a plain string.
pub fn collapse_text_parts(parts: Vec<Value>) -> Value {
    if parts.len() == 1 && parts[0]["type"] == "text" {
        return parts[0]["text"].clone();
    }
    Value::Array(parts)
}

/// extractTextContent (formats/gemini.js)
pub fn extract_text_content(content: &Value, sep: &str) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .filter(|c| c["type"] == "text")
            .map(|c| match &c["text"] {
                Value::String(s) => s.clone(),
                Value::Null => String::new(),
                other => js_string(other),
            })
            .collect::<Vec<_>>()
            .join(sep),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// tool calls (concerns/toolCall.js)
// ---------------------------------------------------------------------------

static TOOL_ID_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-zA-Z0-9_-]+$").unwrap());
static TOOL_ID_BAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-zA-Z0-9_-]").unwrap());

pub fn fallback_tool_call_id(index: Option<i64>) -> String {
    match index {
        None => format!("call_{}", now_ms()),
        Some(i) => format!("call_{i}_{}", now_ms()),
    }
}

pub fn generate_tool_call_id(msg_index: usize, tc_index: usize, tool_name: &str) -> String {
    let name = if tool_name.is_empty() { String::new() } else { format!("_{}", TOOL_ID_BAD.replace_all(tool_name, "")) };
    format!("call_msg{msg_index}_tc{tc_index}{name}")
}

fn sanitize_tool_id(id: &Value) -> Option<String> {
    let s = id.as_str().filter(|s| !s.is_empty())?;
    let out = TOOL_ID_BAD.replace_all(s, "").to_string();
    (!out.is_empty()).then_some(out)
}

fn valid_id(v: &Value) -> bool {
    v.as_str().map(|s| TOOL_ID_RE.is_match(s)).unwrap_or(false)
}

pub fn ensure_tool_call_ids(body: &mut Value) {
    let Some(messages) = body["messages"].as_array_mut() else { return };
    for (i, msg) in messages.iter_mut().enumerate() {
        if msg["role"] == "assistant" {
            if let Some(tcs) = msg["tool_calls"].as_array_mut() {
                for (j, tc) in tcs.iter_mut().enumerate() {
                    if !truthy(&tc["id"]) || !valid_id(&tc["id"]) {
                        let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                        tc["id"] = json!(sanitize_tool_id(&tc["id"]).unwrap_or_else(|| generate_tool_call_id(i, j, &name)));
                    }
                    if !truthy(&tc["type"]) {
                        tc["type"] = json!("function");
                    }
                    let args = &tc["function"]["arguments"];
                    if truthy(args) && !args.is_string() {
                        let s = args.to_string();
                        tc["function"]["arguments"] = json!(s);
                    }
                }
            }
        }
        if msg["role"] == "tool" && truthy(&msg["tool_call_id"]) && !valid_id(&msg["tool_call_id"]) {
            msg["tool_call_id"] = json!(sanitize_tool_id(&msg["tool_call_id"]).unwrap_or_else(|| generate_tool_call_id(i, 0, "")));
        }
        if let Some(content) = msg["content"].as_array_mut() {
            for (k, block) in content.iter_mut().enumerate() {
                if block["type"] == "tool_use" && truthy(&block["id"]) && !valid_id(&block["id"]) {
                    let name = block["name"].as_str().unwrap_or("").to_string();
                    block["id"] = json!(sanitize_tool_id(&block["id"]).unwrap_or_else(|| generate_tool_call_id(i, k, &name)));
                }
                if block["type"] == "tool_result" && truthy(&block["tool_use_id"]) && !valid_id(&block["tool_use_id"]) {
                    block["tool_use_id"] = json!(sanitize_tool_id(&block["tool_use_id"]).unwrap_or_else(|| generate_tool_call_id(i, k, "")));
                }
            }
        }
    }
}

pub fn get_tool_call_ids(msg: &Value) -> Vec<Value> {
    if msg["role"] != "assistant" {
        return vec![];
    }
    let mut ids = vec![];
    for tc in msg["tool_calls"].as_array().into_iter().flatten() {
        if truthy(&tc["id"]) {
            ids.push(tc["id"].clone());
        }
    }
    for b in msg["content"].as_array().into_iter().flatten() {
        if b["type"] == "tool_use" && truthy(&b["id"]) {
            ids.push(b["id"].clone());
        }
    }
    ids
}

pub fn has_tool_results(msg: &Value, ids: &[Value]) -> bool {
    if ids.is_empty() {
        return false;
    }
    if msg["role"] == "tool" && truthy(&msg["tool_call_id"]) {
        return ids.contains(&msg["tool_call_id"]);
    }
    if msg["role"] == "user" {
        for b in msg["content"].as_array().into_iter().flatten() {
            if b["type"] == "tool_result" && ids.contains(&b["tool_use_id"]) {
                return true;
            }
        }
    }
    false
}

pub fn fix_missing_tool_responses(body: &mut Value) {
    let Some(messages) = body["messages"].as_array() else { return };
    let mut out = Vec::with_capacity(messages.len());
    for (i, msg) in messages.iter().enumerate() {
        out.push(msg.clone());
        let ids = get_tool_call_ids(msg);
        if ids.is_empty() {
            continue;
        }
        if let Some(next) = messages.get(i + 1) {
            if !has_tool_results(next, &ids) {
                for id in ids {
                    out.push(json!({"role": "tool", "tool_call_id": id, "content": ""}));
                }
            }
        }
    }
    body["messages"] = Value::Array(out);
}

pub fn default_claude_tool_type(tools: &mut Value) {
    if let Some(a) = tools.as_array_mut() {
        for t in a {
            if !truthy(&t["type"]) {
                t["type"] = json!("custom");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// finish reasons / usage / chunks
// ---------------------------------------------------------------------------

pub fn to_openai_finish(reason: &Value, format: &str) -> Value {
    let r = reason.as_str().unwrap_or("");
    let out = match format {
        "claude" => match r {
            "max_tokens" => "length",
            "tool_use" => "tool_calls",
            "refusal" => "content_filter",
            _ => "stop",
        },
        "commandcode" => match r {
            "stop" | "error" => "stop",
            "length" => "length",
            "tool-calls" | "tool_use" => "tool_calls",
            "content-filter" => "content_filter",
            "" => "stop",
            other => return json!(other),
        },
        "gemini" => match js_string(reason).to_uppercase().as_str() {
            "MAX_TOKENS" => "length",
            "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" => "content_filter",
            _ => "stop",
        },
        "kiro" | "ollama" => match r {
            "tool_calls" | "tool_use" => "tool_calls",
            "length" | "max_tokens" => "length",
            _ => "stop",
        },
        _ => {
            if truthy(reason) {
                return reason.clone();
            }
            "stop"
        }
    };
    json!(out)
}

pub fn from_openai_finish(reason: &Value, format: &str) -> Value {
    match format {
        "claude" => json!(match reason.as_str().unwrap_or("") {
            "length" => "max_tokens",
            "tool_calls" => "tool_use",
            "content_filter" => "refusal",
            _ => "end_turn",
        }),
        _ => reason.clone(),
    }
}

pub fn build_usage(prompt: i64, completion: i64, total: i64, cached: i64, cache_creation: i64, reasoning: i64) -> Value {
    let mut u = json!({"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total});
    if cached > 0 || cache_creation > 0 {
        let mut d = json!({});
        if cached > 0 {
            d["cached_tokens"] = json!(cached);
        }
        if cache_creation > 0 {
            d["cache_creation_tokens"] = json!(cache_creation);
        }
        u["prompt_tokens_details"] = d;
    }
    if reasoning > 0 {
        u["completion_tokens_details"] = json!({"reasoning_tokens": reasoning});
    }
    u
}

fn n(v: &Value) -> i64 {
    if v.is_number() { v.as_f64().unwrap_or(0.0) as i64 } else { 0 }
}

/// toOpenAIUsage(raw, kind)
pub fn to_openai_usage(raw: &Value, kind: &str) -> Option<Value> {
    if !raw.is_object() {
        return None;
    }
    Some(match kind {
        "claude" => {
            let (i, o) = (n(&raw["input_tokens"]), n(&raw["output_tokens"]));
            let (cr, cc) = (n(&raw["cache_read_input_tokens"]), n(&raw["cache_creation_input_tokens"]));
            let p = i + cr + cc;
            build_usage(p, o, p + o, cr, cc, 0)
        }
        "gemini" => {
            let cached = n(&raw["cachedContentTokenCount"]);
            let prompt = n(&raw["promptTokenCount"]);
            let thoughts = n(&raw["thoughtsTokenCount"]);
            let total = n(&raw["totalTokenCount"]);
            let mut cand = n(&raw["candidatesTokenCount"]);
            if cand == 0 && total > 0 {
                cand = (total - prompt - thoughts).max(0);
            }
            build_usage(prompt, cand + thoughts, total, cached, 0, thoughts)
        }
        "kiro" => {
            let (i, o) = (n(&raw["inputTokens"]), n(&raw["outputTokens"]));
            let cached = [n(&raw["cache_read_input_tokens"]), n(&raw["cachedTokens"]), n(&raw["cached_tokens"])]
                .into_iter()
                .find(|x| *x != 0)
                .unwrap_or(0);
            build_usage(i, o, i + o, cached.max(0), n(&raw["cache_creation_input_tokens"]).max(0), 0)
        }
        "ollama" => {
            let (i, o) = (n(&raw["prompt_eval_count"]), n(&raw["eval_count"]));
            build_usage(i, o, i + o, 0, 0, 0)
        }
        "commandcode" => {
            let (i, o) = (n(&raw["inputTokens"]), n(&raw["outputTokens"]));
            let total = if raw["totalTokens"].is_number() { n(&raw["totalTokens"]) } else { i + o };
            build_usage(i, o, total, 0, 0, 0)
        }
        _ => return None,
    })
}

pub fn build_chunk(id: &str, created: i64, model: &Value, delta: Value, finish: Value) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
}

pub fn reasoning_delta(text: &Value, with_role: bool) -> Value {
    if with_role {
        json!({"role": "assistant", "reasoning_content": text})
    } else {
        json!({"reasoning_content": text})
    }
}

/// extractReasoningText(delta)
pub fn extract_reasoning_text(delta: &Value) -> String {
    if !delta.is_object() {
        return String::new();
    }
    if let Some(s) = delta["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
        return s.into();
    }
    if let Some(s) = delta["reasoning"].as_str().filter(|s| !s.is_empty()) {
        return s.into();
    }
    if let Some(d) = delta["reasoning_details"].as_array() {
        return d
            .iter()
            .map(|x| match x {
                Value::String(s) => s.clone(),
                _ => x["text"].as_str().filter(|s| !s.is_empty()).or_else(|| x["content"].as_str()).unwrap_or("").to_string(),
            })
            .collect();
    }
    String::new()
}

// ---------------------------------------------------------------------------
// formats/openai.js — filterToOpenAIFormat
// ---------------------------------------------------------------------------

const VALID_OPENAI_CONTENT_TYPES: &[&str] = &["text", "image_url", "image", "input_audio", "audio_url", "file"];

pub fn filter_to_openai_format(body: &mut Value, preserve_cache: bool) {
    let Some(messages) = body["messages"].as_array().cloned() else { return };
    let strip_block = |b: &Value| {
        let rest = omit(b, &["signature", "cache_control"]);
        if preserve_cache && truthy(&b["cache_control"]) {
            let mut r = rest;
            r["cache_control"] = b["cache_control"].clone();
            r
        } else {
            rest
        }
    };
    let mapped: Vec<Value> = messages
        .into_iter()
        .map(|mut msg| {
            if msg["role"] == "developer" {
                msg["role"] = json!("system");
            }
            if msg["role"] == "tool" || (msg["role"] == "assistant" && truthy(&msg["tool_calls"])) {
                return msg;
            }
            if msg["content"].is_string() {
                return msg;
            }
            if let Some(content) = msg["content"].as_array() {
                let mut filtered = vec![];
                for b in content {
                    let ty = b["type"].as_str().unwrap_or("");
                    if ty == "thinking" || ty == "redacted_thinking" {
                        continue;
                    }
                    if VALID_OPENAI_CONTENT_TYPES.contains(&ty) || ty == "tool_result" {
                        filtered.push(strip_block(b));
                    }
                }
                if filtered.is_empty() {
                    filtered.push(json!({"type": "text", "text": ""}));
                }
                msg["content"] = Value::Array(filtered);
            }
            msg
        })
        .filter(|msg| {
            if msg["role"] == "tool" || (msg["role"] == "assistant" && truthy(&msg["tool_calls"])) {
                return true;
            }
            match &msg["content"] {
                Value::String(s) => !s.trim().is_empty(),
                Value::Array(a) => a.iter().any(|b| {
                    (b["type"] == "text" && b["text"].as_str().map(|t| !t.trim().is_empty()).unwrap_or(false)) || b["type"] != "text"
                }),
                _ => true,
            }
        })
        .collect();
    body["messages"] = Value::Array(mapped);

    if body["tools"].as_array().map(|t| t.is_empty()).unwrap_or(false) {
        del(body, "tools");
    }
    if let Some(tools) = body["tools"].as_array().cloned().filter(|t| !t.is_empty()) {
        let mut out = vec![];
        for tool in tools {
            if tool["type"] == "function" && truthy(&tool["function"]) {
                out.push(tool);
            } else if truthy(&tool["name"]) && (truthy(&tool["input_schema"]) || truthy(&tool["description"])) {
                out.push(json!({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": desc_string(&tool["description"]),
                        "parameters": if truthy(&tool["input_schema"]) { tool["input_schema"].clone() } else { json!({"type": "object", "properties": {}}) },
                    }
                }));
            } else if let Some(fds) = tool["functionDeclarations"].as_array() {
                for fd in fds {
                    out.push(json!({
                        "type": "function",
                        "function": {
                            "name": fd["name"],
                            "description": desc_string(&fd["description"]),
                            "parameters": if truthy(&fd["parameters"]) { fd["parameters"].clone() } else { json!({"type": "object", "properties": {}}) },
                        }
                    }));
                }
            } else {
                out.push(tool);
            }
        }
        body["tools"] = Value::Array(out);
    }

    let choice = body["tool_choice"].clone();
    if choice.is_object() {
        match choice["type"].as_str() {
            Some("auto") => body["tool_choice"] = json!("auto"),
            Some("any") => body["tool_choice"] = json!("required"),
            Some("tool") if truthy(&choice["name"]) => {
                body["tool_choice"] = json!({"type": "function", "function": {"name": choice["name"]}})
            }
            _ => {}
        }
    }
}

/// `String(x || "")`
pub fn desc_string(v: &Value) -> String {
    if truthy(v) { js_string(v) } else { String::new() }
}

// ---------------------------------------------------------------------------
// paramSupport.js — strip unsupported params
// ---------------------------------------------------------------------------

pub fn strip_unsupported_params(provider: &str, model: &str, body: &mut Value) {
    if model.is_empty() || !body.is_object() {
        return;
    }
    let lower = model.to_lowercase();
    if lower.contains("claude") {
        del(body, "temperature");
    }
    match provider {
        "github" => {
            if lower.contains("gpt-5.4") {
                del(body, "temperature");
            }
            static OPUS46: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)claude.*(opus|sonnet).*4\.6").unwrap());
            if lower.contains("claude") && !OPUS46.is_match(model) {
                del(body, "thinking");
                del(body, "reasoning_effort");
            }
        }
        "cloudflare-ai" => {
            for msg in body["messages"].as_array_mut().into_iter().flatten() {
                if let Some(parts) = msg["content"].as_array() {
                    let s: String = parts
                        .iter()
                        .map(|b| if b["type"] == "text" { b["text"].as_str().unwrap_or("").to_string() } else { String::new() })
                        .collect();
                    msg["content"] = json!(s);
                }
            }
        }
        "volcengine-ark" => {
            let mut caps: Vec<i64> = vec![];
            if lower.contains("glm-5") || lower.contains("kimi") {
                if let Some(m) = crate::caps::caps_for(Some(provider), model).max_output().filter(|m| *m > 0) {
                    caps.push(m);
                }
            }
            if lower.contains("kimi") {
                caps.push(32768);
            }
            if let Some(ceil) = caps.into_iter().min() {
                for k in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
                    if body[k].as_f64().map(|v| v > ceil as f64).unwrap_or(false) {
                        body[k] = json!(ceil);
                    }
                }
            }
        }
        "groq" | "mistral" | "cerebras" => {
            for msg in body["messages"].as_array_mut().into_iter().flatten() {
                if msg["role"] == "assistant" {
                    for k in ["reasoning_content", "reasoning", "reasoning_details"] {
                        del(msg, k);
                    }
                }
            }
        }
        _ => {}
    }
}

/// normalizeThinkingConfig: drop `thinking` when the last message is not from the user.
pub fn normalize_thinking_config(body: &mut Value) {
    let msgs = body["messages"].as_array().or_else(|| body["contents"].as_array());
    let last_is_user = match msgs {
        None => true,
        Some(m) if m.is_empty() => true,
        Some(m) => m[m.len() - 1]["role"] == "user",
    };
    if !last_is_user {
        del(body, "thinking");
    }
}

/// stripContentTypes (explicit strip[] opt-in per model)
pub fn strip_content_types(body: &mut Value, strip: &[String]) {
    if strip.is_empty() {
        return;
    }
    let image = strip.iter().any(|s| s == "image");
    let audio = strip.iter().any(|s| s == "audio");
    for msg in body["messages"].as_array_mut().into_iter().flatten() {
        let Some(c) = msg["content"].as_array() else { continue };
        let kept: Vec<Value> = c
            .iter()
            .filter(|p| {
                let t = p["type"].as_str().unwrap_or("");
                !((image && (t == "image_url" || t == "image")) || (audio && (t == "audio_url" || t == "input_audio")))
            })
            .cloned()
            .collect();
        msg["content"] = if kept.is_empty() { json!("") } else { Value::Array(kept) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_tool_responses_inserted() {
        let mut b = json!({"messages": [
            {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "user", "content": "hi"}
        ]});
        fix_missing_tool_responses(&mut b);
        assert_eq!(b["messages"][1]["role"], "tool");
    }

    #[test]
    fn ids_sanitized() {
        let mut b = json!({"messages": [{"role": "assistant", "tool_calls": [{"id": "a.b", "function": {"name": "f", "arguments": {"x": 1}}}]}]});
        ensure_tool_call_ids(&mut b);
        assert_eq!(b["messages"][0]["tool_calls"][0]["id"], "ab");
        assert_eq!(b["messages"][0]["tool_calls"][0]["function"]["arguments"], "{\"x\":1}");
    }
}
