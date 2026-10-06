//! chatCore helpers: format detection, client detection, tool dedup,
//! modality stripping, remote image prefetch, bypass responses and errors.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};

use crate::jsv::{now_ms, truthy};
use crate::registry::REG;
use crate::translate::{self, CLAUDE, OPENAI, OPENAI_RESPONSES};

/// services/provider.js detectFormat(body)
pub fn detect_format(body: &Value) -> &'static str {
    if (body["input"].is_array() || body["input"].is_string()) && truthy(&body["input"]) && !truthy(&body["messages"]) {
        return OPENAI_RESPONSES;
    }
    if truthy(&body["request"]["contents"]) && body["userAgent"] == "antigravity" {
        return translate::ANTIGRAVITY;
    }
    if body["contents"].is_array() {
        return translate::GEMINI;
    }
    let o = body.as_object();
    let has = |k: &str| o.map(|o| o.contains_key(k) && !o[k].is_null()).unwrap_or(false);
    if truthy(&body["stream_options"]) || truthy(&body["response_format"]) || has("logprobs") || has("top_logprobs") || has("n") || has("presence_penalty") || has("frequency_penalty") || truthy(&body["logit_bias"]) || truthy(&body["user"]) {
        return OPENAI;
    }
    if let Some(msgs) = body["messages"].as_array() {
        let first = &msgs.first().cloned().unwrap_or(Value::Null);
        if let Some(c) = first["content"].as_array() {
            if c.first().map(|x| x["type"] == "text").unwrap_or(false) && !body["model"].as_str().map(|m| m.contains('/')).unwrap_or(false) {
                if truthy(&body["system"]) || truthy(&body["anthropic_version"]) {
                    return CLAUDE;
                }
                if c.iter().any(|x| x["type"] == "image" && x["source"]["type"] == "base64") {
                    return CLAUDE;
                }
                if c.iter().any(|x| x["type"] == "image_url" && truthy(&x["image_url"]["url"])) {
                    return OPENAI;
                }
                if c.iter().any(|x| x["type"] == "tool_use" || x["type"] == "tool_result") {
                    return CLAUDE;
                }
            }
        }
        if body.get("system").is_some() || truthy(&body["anthropic_version"]) {
            return CLAUDE;
        }
    }
    OPENAI
}

/// translator/formats.js detectFormatByEndpoint
pub fn detect_format_by_endpoint(path: &str, body: &Value) -> Option<&'static str> {
    if path.contains("/v1/responses") || path.ends_with("/responses") {
        return Some(OPENAI_RESPONSES);
    }
    if path.contains("/v1/messages") {
        return Some(CLAUDE);
    }
    if path.contains("/chat/completions") && body["input"].is_array() {
        return Some(OPENAI);
    }
    None
}

/// getTargetFormat(provider, credentials)
pub fn get_target_format(provider: &str, creds: &Value) -> String {
    if crate::exec::is_openai_compatible(provider) {
        return if crate::exec::openai_compatible_api_type(provider, creds) == "responses" { OPENAI_RESPONSES.into() } else { OPENAI.into() };
    }
    if crate::exec::is_anthropic_compatible(provider) {
        return CLAUDE.into();
    }
    let t = REG.transport(provider);
    let t = if t.is_null() { REG.transport("openai") } else { t };
    t["format"].as_str().unwrap_or(OPENAI).to_string()
}

/// resolveTransport(provider, sourceFormat)
pub fn resolve_transport(provider: &str, source: &str) -> Option<Value> {
    REG.transport(provider)["transports"].as_array()?.iter().find(|t| t["format"] == source).cloned()
}

/// utils/clientDetector.js detectClientTool
pub fn detect_client_tool(headers: &Value, body: &Value) -> Option<&'static str> {
    let h = |k: &str| headers[k].as_str().unwrap_or("").to_lowercase();
    let ua = h("user-agent");
    if body["userAgent"] == "antigravity" {
        return Some("antigravity");
    }
    if ua.contains("githubcopilotchat") || h("openai-intent") == "conversation-panel" || h("x-initiator") == "user" {
        return Some("github-copilot");
    }
    if ua.contains("claude-cli") || ua.contains("claude-code") || h("x-app") == "cli" {
        return Some("claude");
    }
    if ua.contains("gemini-cli") {
        return Some("gemini-cli");
    }
    if ua.contains("codex-tui") || ua.contains("codex-cli") || ua.contains("codex_cli_rs") || ua.contains("codex desktop") || h("originator").starts_with("codex_") {
        return Some("codex");
    }
    if ua.contains("deepseek-tui") {
        return Some("deepseek-tui");
    }
    None
}

pub fn is_native_passthrough(tool: Option<&str>, provider: &str) -> bool {
    let p = if provider.starts_with("anthropic-compatible") { "anthropic" } else { provider };
    match tool {
        Some("claude") => p == "claude" || p == "anthropic",
        Some("gemini-cli") => p == "gemini-cli",
        Some("antigravity") => p == "antigravity",
        Some("codex") => p == "codex",
        _ => false,
    }
}

/// utils/toolDeduper.js dedupeTools
pub fn dedupe_tools(tools: &mut Value, client_tool: Option<&str>, model: &str) -> Vec<String> {
    let Some(arr) = tools.as_array() else { return vec![] };
    if arr.is_empty() {
        return vec![];
    }
    let name_of = |t: &Value| t["name"].as_str().or_else(|| t["function"]["name"].as_str()).unwrap_or("").to_string();
    let names: Vec<String> = arr.iter().map(name_of).collect();
    let mut strip: Vec<String> = vec![];
    if client_tool == Some("claude") {
        static BROWSER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^mcp__browsermcp__").unwrap());
        static CHROME: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^mcp__Claude_in_Chrome__").unwrap());
        let web = ["WebSearch", "WebFetch", "mcp__workspace__web_fetch"];
        let rules: [(&dyn Fn(&str) -> bool, &dyn Fn(&str) -> bool); 3] = [
            (&|n| n == "mcp__exa__web_search_exa" || n == "mcp__exa__web_fetch_exa", &|n| web.contains(&n)),
            (&|n| n == "mcp__tavily__tavily_search" || n == "mcp__tavily__tavily_extract", &|n| web.contains(&n)),
            (&|n| BROWSER.is_match(n), &|n| CHROME.is_match(n)),
        ];
        for (trig, st) in rules {
            if names.iter().any(|n| trig(n)) {
                for n in &names {
                    if st(n) && !strip.contains(n) {
                        strip.push(n.clone());
                    }
                }
            }
        }
    }
    let mut drop = std::collections::HashSet::new();
    if crate::registry::is_deepseek_model(model) {
        let mut seen = std::collections::HashSet::new();
        for (i, n) in names.iter().enumerate() {
            if n.is_empty() {
                continue;
            }
            if !seen.insert(n.clone()) {
                drop.insert(i);
            }
        }
    }
    if strip.is_empty() && drop.is_empty() {
        return vec![];
    }
    let out: Vec<Value> = arr.iter().enumerate().filter(|(i, t)| !drop.contains(i) && !strip.contains(&name_of(t))).map(|(_, t)| t.clone()).collect();
    let mut stripped: Vec<String> = drop.iter().map(|i| names[*i].clone()).collect();
    stripped.extend(strip);
    *tools = Value::Array(out);
    stripped
}

// ---------------------------------------------------------------------------
// modality stripping (translator/concerns/modality.js)
// ---------------------------------------------------------------------------

fn placeholder(cap: &str, last: bool) -> &'static str {
    match (cap, last) {
        ("vision", true) => "[image omitted: model has no vision support]",
        ("audioInput", true) => "[audio omitted: model has no audio support]",
        ("pdf", true) => "[file omitted: model has no document support]",
        ("vision", false) => "[Previous image omitted from context.]",
        ("audioInput", false) => "[Previous audio omitted from context.]",
        _ => "[Previous file omitted from context.]",
    }
}

fn cap_false(caps: &crate::caps::Caps, c: &str) -> bool {
    caps.get(c) == &json!(false)
}

fn filter_blocks(blocks: &[Value], cap_of: &dyn Fn(&Value) -> Option<&'static str>, caps: &crate::caps::Caps, last: bool, mk: &dyn Fn(&str) -> Value) -> Vec<Value> {
    let mut removed: Vec<&'static str> = vec![];
    let mut out = vec![];
    for b in blocks {
        if let Some(c) = cap_of(b) {
            if cap_false(caps, c) {
                if !removed.contains(&c) {
                    removed.push(c);
                }
                continue;
            }
        }
        out.push(b.clone());
    }
    for c in removed {
        out.push(mk(placeholder(c, last)));
    }
    out
}

fn cap_for_mime(m: &str) -> Option<&'static str> {
    if m.starts_with("image/") {
        Some("vision")
    } else if m.starts_with("audio/") {
        Some("audioInput")
    } else if m == "application/pdf" {
        Some("pdf")
    } else {
        None
    }
}

pub fn strip_unsupported_modalities(body: &mut Value, source: &str, caps: &crate::caps::Caps) -> bool {
    if !cap_false(caps, "vision") && !cap_false(caps, "audioInput") && !cap_false(caps, "pdf") {
        return false;
    }
    let text = |t: &str| json!({"type": "text", "text": t});
    match source {
        CLAUDE => {
            if let Some(msgs) = body["messages"].as_array_mut() {
                let last = msgs.len().saturating_sub(1);
                for (i, m) in msgs.iter_mut().enumerate() {
                    if let Some(c) = m["content"].as_array().cloned() {
                        m["content"] = Value::Array(filter_blocks(&c, &|b| match b["type"].as_str() { Some("image") => Some("vision"), Some("document") => Some("pdf"), _ => None }, caps, i == last, &text));
                    }
                }
            }
        }
        OPENAI_RESPONSES => {
            if let Some(items) = body["input"].as_array_mut() {
                let last = items.len().saturating_sub(1);
                for (i, it) in items.iter_mut().enumerate() {
                    if let Some(c) = it["content"].as_array().cloned() {
                        it["content"] = Value::Array(filter_blocks(&c, &|b| match b["type"].as_str() { Some("input_image") => Some("vision"), Some("input_file") => Some("pdf"), _ => None }, caps, i == last, &|t| json!({"type": "input_text", "text": t})));
                    }
                }
            }
        }
        translate::GEMINI | translate::GEMINI_CLI | translate::VERTEX | translate::ANTIGRAVITY => {
            let contents = if source == translate::ANTIGRAVITY { &mut body["request"]["contents"] } else { &mut body["contents"] };
            if let Some(cs) = contents.as_array_mut() {
                let last = cs.len().saturating_sub(1);
                for (i, c) in cs.iter_mut().enumerate() {
                    if let Some(p) = c["parts"].as_array().cloned() {
                        c["parts"] = Value::Array(filter_blocks(
                            &p,
                            &|b| b["inlineData"]["mimeType"].as_str().or_else(|| b["fileData"]["mimeType"].as_str()).and_then(cap_for_mime),
                            caps,
                            i == last,
                            &|t| json!({"text": t}),
                        ));
                    }
                }
            }
        }
        _ => {
            if let Some(msgs) = body["messages"].as_array_mut() {
                let last = msgs.len().saturating_sub(1);
                for (i, m) in msgs.iter_mut().enumerate() {
                    if cap_false(caps, "vision") {
                        crate::jsv::del(m, "images");
                        for k in ["experimental_attachments", "attachments"] {
                            if let Some(a) = m[k].as_array().cloned() {
                                m[k] = Value::Array(a.into_iter().filter(|x| !(x["contentType"].as_str().map(|c| c.starts_with("image/")).unwrap_or(false) || x["url"].as_str().map(|u| u.starts_with("data:image/")).unwrap_or(false))).collect());
                            }
                        }
                    }
                    if let Some(c) = m["content"].as_array().cloned() {
                        m["content"] = Value::Array(filter_blocks(
                            &c,
                            &|b| match b["type"].as_str() {
                                Some("image_url") | Some("image") => Some("vision"),
                                Some("input_audio") | Some("audio_url") => Some("audioInput"),
                                Some("file") => Some("pdf"),
                                _ => None,
                            },
                            caps,
                            i == last,
                            &text,
                        ));
                    }
                }
            }
        }
    }
    true
}

/// translator/concerns/prefetch.js — inline remote images for targets that need base64.
pub async fn prefetch_remote_images(body: &mut Value, source: &str, target: &str) -> usize {
    use translate::*;
    if ![GEMINI, GEMINI_CLI, VERTEX, ANTIGRAVITY, OLLAMA, KIRO, COMMANDCODE].contains(&target) {
        return 0;
    }
    let remote = |u: &str| u.starts_with("http://") || u.starts_with("https://");
    let mut n = 0;
    match source {
        CLAUDE => {
            for m in body["messages"].as_array_mut().into_iter().flatten() {
                for b in m["content"].as_array_mut().into_iter().flatten() {
                    if b["type"] == "image" && b["source"]["type"] == "url" {
                        let u = b["source"]["url"].as_str().unwrap_or("").to_string();
                        if remote(&u) {
                            if let Some(d) = crate::media::fetch_image_as_data_url(&u).await {
                                if let Some((mime, data)) = crate::translate::concerns::parse_data_uri(&json!(d)) {
                                    b["source"] = json!({"type": "base64", "media_type": mime, "data": data});
                                    n += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        GEMINI | GEMINI_CLI | VERTEX | ANTIGRAVITY => {
            let cs = if source == ANTIGRAVITY { &mut body["request"]["contents"] } else { &mut body["contents"] };
            for c in cs.as_array_mut().into_iter().flatten() {
                for p in c["parts"].as_array_mut().into_iter().flatten() {
                    let u = p["fileData"]["fileUri"].as_str().unwrap_or("").to_string();
                    if remote(&u) {
                        if let Some(d) = crate::media::fetch_image_as_data_url(&u).await {
                            if let Some((mime, data)) = crate::translate::concerns::parse_data_uri(&json!(d)) {
                                crate::jsv::del(p, "fileData");
                                p["inlineData"] = json!({"mimeType": mime, "data": data});
                                n += 1;
                            }
                        }
                    }
                }
            }
        }
        _ => {
            for m in body["messages"].as_array_mut().into_iter().flatten() {
                for b in m["content"].as_array_mut().into_iter().flatten() {
                    if b["type"] != "image_url" {
                        continue;
                    }
                    let is_str = b["image_url"].is_string();
                    let u = if is_str { b["image_url"].as_str() } else { b["image_url"]["url"].as_str() }.unwrap_or("").to_string();
                    if remote(&u) {
                        if let Some(d) = crate::media::fetch_image_as_data_url(&u).await {
                            if is_str {
                                b["image_url"] = json!(d);
                            } else {
                                b["image_url"]["url"] = json!(d);
                            }
                            n += 1;
                        }
                    }
                }
            }
        }
    }
    n
}

// ---------------------------------------------------------------------------
// errors (utils/error.js + config/errorConfig.js)
// ---------------------------------------------------------------------------

pub fn error_type(status: u16) -> (&'static str, &'static str) {
    match status {
        400 => ("invalid_request_error", "bad_request"),
        401 => ("authentication_error", "invalid_api_key"),
        402 => ("billing_error", "payment_required"),
        403 => ("permission_error", "insufficient_quota"),
        404 => ("invalid_request_error", "model_not_found"),
        406 => ("invalid_request_error", "model_not_supported"),
        429 => ("rate_limit_error", "rate_limit_exceeded"),
        500 => ("server_error", "internal_server_error"),
        502 => ("server_error", "bad_gateway"),
        503 => ("server_error", "service_unavailable"),
        504 => ("server_error", "gateway_timeout"),
        s if s >= 500 => ("server_error", "internal_server_error"),
        _ => ("invalid_request_error", ""),
    }
}

pub fn default_error_message(status: u16) -> &'static str {
    match status {
        400 => "Bad request",
        401 => "Invalid API key provided",
        402 => "Payment required",
        403 => "You exceeded your current quota",
        404 => "Model not found",
        406 => "Model not supported",
        429 => "Rate limit exceeded",
        500 => "Internal server error",
        502 => "Bad gateway - upstream provider error",
        503 => "Service temporarily unavailable",
        504 => "Gateway timeout",
        _ => "An error occurred",
    }
}

pub fn error_body(status: u16, message: &str) -> Value {
    let (t, c) = error_type(status);
    json!({"error": {"message": if message.is_empty() { default_error_message(status) } else { message }, "type": t, "code": c}})
}

/// parseUpstreamError(response, executor)
pub fn parse_upstream_error(status: u16, body: &str, ex: &dyn crate::exec::Executor) -> (u16, String, Option<i64>) {
    let p = ex.parse_error(status, body);
    let msg = if p.message.is_empty() { default_error_message(status).to_string() } else { p.message };
    // Default executor parse returns the raw body: mirror the JS JSON extraction.
    let msg = match serde_json::from_str::<Value>(&msg) {
        Ok(j) if j.is_object() => {
            let m = if truthy(&j["error"]["message"]) { j["error"]["message"].clone() } else if truthy(&j["message"]) { j["message"].clone() } else if truthy(&j["error"]) { j["error"].clone() } else { json!(msg) };
            m.as_str().map(str::to_owned).unwrap_or_else(|| m.to_string())
        }
        _ => msg,
    };
    (p.status, msg, p.resets_at_ms)
}

pub fn format_provider_error(message: &str, status: u16) -> String {
    format!("[{status}]: {message}")
}

/// formatRetryAfter
pub fn format_retry_after(until_ms: i64) -> String {
    let diff = until_ms - now_ms();
    if diff <= 0 {
        return "reset after 0s".into();
    }
    let total = (diff + 999) / 1000;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    let mut parts = vec![];
    if h > 0 {
        parts.push(format!("{h}h"));
    }
    if m > 0 {
        parts.push(format!("{m}m"));
    }
    if s > 0 || parts.is_empty() {
        parts.push(format!("{s}s"));
    }
    format!("reset after {}", parts.join(" "))
}

// ---------------------------------------------------------------------------
// misc
// ---------------------------------------------------------------------------

/// stripContinuityFields
pub fn strip_continuity_fields(body: &mut Value) {
    for m in body["messages"].as_array_mut().into_iter().flatten() {
        crate::jsv::del(m, "encrypted_content");
        crate::jsv::del(m, "reasoning_encrypted_content");
    }
}

/// shouldDefaultClaudeToolType + defaultClaudeToolType
pub fn maybe_default_claude_tool_type(provider: &str, final_format: &str, body: &mut Value) {
    if final_format == CLAUDE && body["tools"].is_array() && REG.transport(provider)["quirks"]["requireClaudeToolType"] == json!(true) {
        crate::translate::concerns::default_claude_tool_type(&mut body["tools"]);
    }
}

/// stripModelContextMarker: "model[1m]" → ("model", Some("1m"))
pub fn strip_model_context_marker(model: &str) -> (String, Option<String>) {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(.*)\[(\w+)\]$").unwrap());
    match RE.captures(model) {
        Some(c) => (c[1].to_string(), Some(c[2].to_string())),
        None => (model.to_string(), None),
    }
}

/// Project ID for gemini-cli / antigravity accounts (services/projectId.js).
pub async fn fetch_project_id(access_token: &str, provider: &str) -> Option<String> {
    let c = &crate::consts::C;
    let ep = if c["CLOUD_CODE_API"][provider].is_object() { &c["CLOUD_CODE_API"][provider] } else { &c["CLOUD_CODE_API"]["gemini-cli"] };
    let headers = if provider == "antigravity" { &c["ANTIGRAVITY_LOAD_CODE_ASSIST_HEADERS"] } else { &c["LOAD_CODE_ASSIST_HEADERS"] };
    let mut h = crate::exec::Headers::default();
    h.extend_obj(headers);
    h.set("Authorization", format!("Bearer {access_token}"));
    let meta = c["LOAD_CODE_ASSIST_METADATA"].clone();
    let up = crate::exec::post_json(&Value::Null, ep["loadCodeAssist"].as_str()?, &h, &json!({"metadata": meta}), 30_000).await.ok()?;
    if !up.ok() {
        return None;
    }
    let d: Value = serde_json::from_slice(&up.bytes().await.ok()?).ok()?;
    let pick = |p: &Value| p.as_str().map(|s| s.trim().to_string()).or_else(|| p["id"].as_str().map(|s| s.trim().to_string())).filter(|s| !s.is_empty());
    if let Some(p) = pick(&d["cloudaicompanionProject"]) {
        return Some(p);
    }
    let tier = d["allowedTiers"].as_array().and_then(|a| a.iter().find(|t| t["isDefault"] == json!(true)).and_then(|t| t["id"].as_str().map(|s| s.trim().to_string()))).filter(|s| !s.is_empty()).unwrap_or_else(|| "legacy-tier".into());
    for _ in 0..2 {
        let up = crate::exec::post_json(&Value::Null, ep["onboardUser"].as_str()?, &h, &json!({"tierId": tier, "metadata": meta}), 30_000).await.ok()?;
        if !up.ok() {
            return None;
        }
        let d: Value = serde_json::from_slice(&up.bytes().await.ok()?).ok()?;
        if d["done"] == json!(true) {
            return pick(&d["response"]["cloudaicompanionProject"]);
        }
        tokio::time::sleep(std::time::Duration::from_secs(12)).await;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(detect_format(&json!({"input": "hi"})), OPENAI_RESPONSES);
        assert_eq!(detect_format(&json!({"contents": []})), "gemini");
        assert_eq!(detect_format(&json!({"messages": [{"role": "user", "content": "x"}], "system": "s"})), CLAUDE);
        assert_eq!(detect_format(&json!({"messages": [{"role": "user", "content": "x"}], "n": 1, "system": "s"})), OPENAI);
        assert_eq!(detect_format(&json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "tool_result"}]}]})), CLAUDE);
        assert_eq!(detect_format_by_endpoint("/v1/messages", &json!({})), Some(CLAUDE));
        assert_eq!(strip_model_context_marker("cc/claude-opus[1m]"), ("cc/claude-opus".into(), Some("1m".into())));
        assert_eq!(detect_client_tool(&json!({"user-agent": "claude-cli/1.0"}), &json!({})), Some("claude"));
        assert!(is_native_passthrough(Some("claude"), "anthropic-compatible-x"));
    }

    #[test]
    fn dedupe() {
        let mut t = json!([{"name": "a"}, {"name": "a"}, {"function": {"name": "b"}}]);
        let s = dedupe_tools(&mut t, None, "deepseek-chat");
        assert_eq!(s, vec!["a"]);
        assert_eq!(t.as_array().unwrap().len(), 2);
        let mut t2 = json!([{"name": "WebSearch"}, {"name": "mcp__exa__web_search_exa"}]);
        assert_eq!(dedupe_tools(&mut t2, Some("claude"), "x"), vec!["WebSearch"]);
    }
}
