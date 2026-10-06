//! OpenAI Codex (ChatGPT backend, Responses API) executor (port of executors/codex.js).

use std::sync::LazyLock;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use regex::Regex;
use serde_json::{Value, json};

use crate::exec::{ExecArgs, ExecResult, Executor, Headers, ParsedError, Upstream, base_execute};
use crate::jsv::{del, js_string, now_ms, truthy};
use crate::registry::{REG, get_model_upstream_id};

pub struct Codex;

const ALLOWLIST: &[&str] = &[
    "model", "input", "instructions", "tools", "tool_choice", "stream", "store", "reasoning", "service_tier", "include",
    "prompt_cache_key", "client_metadata", "text", "parallel_tool_calls",
];
const HOSTED: &[&str] = &[
    "image_generation", "web_search", "web_search_preview", "file_search", "computer", "computer_use_preview", "code_interpreter", "mcp",
    "local_shell", "tool_search",
];

fn is_lite(model: &str) -> bool {
    let base = crate::registry::strip_paren_suffix(model);
    REG.provider_models("cx").iter().any(|e| e["id"] == base.as_str() && e["responsesLite"] == json!(true))
}

/// Removes `pattern` constraints containing Unicode property escapes (Codex rejects them).
pub fn strip_unsupported_patterns(v: &Value) -> Value {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(^|[^\\])(\\\\)*\\[pP]\{").unwrap());
    match v {
        Value::Array(a) => Value::Array(a.iter().map(strip_unsupported_patterns).collect()),
        Value::Object(o) => {
            let mut out = serde_json::Map::new();
            for (k, val) in o {
                if k == "pattern" && val.as_str().map(|p| RE.is_match(p)).unwrap_or(false) {
                    continue;
                }
                if k == "properties" && val.is_object() {
                    out.insert(k.clone(), Value::Object(val.as_object().unwrap().iter().map(|(n, s)| (n.clone(), strip_unsupported_patterns(s))).collect()));
                    continue;
                }
                out.insert(k.clone(), strip_unsupported_patterns(val));
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

fn normalize_tools(body: &mut Value) {
    let Some(tools) = body["tools"].as_array().cloned() else { return };
    let mut valid: Vec<String> = vec![];
    let mut out = vec![];
    for tool in tools {
        if !tool.is_object() {
            continue;
        }
        let ty = tool["type"].as_str().unwrap_or("").to_string();
        if ty == "namespace" {
            let mut t = tool.clone();
            if let Some(sub) = t["tools"].as_array_mut() {
                for st in sub.iter_mut() {
                    if let Some(n) = st["name"].as_str() {
                        let n: String = n.trim().chars().take(128).collect();
                        if !n.is_empty() {
                            valid.push(n);
                        }
                    }
                    if st["parameters"].is_object() {
                        st["parameters"] = strip_unsupported_patterns(&st["parameters"]);
                    }
                }
            }
            out.push(t);
            continue;
        }
        if ty != "function" {
            if ty == "custom" {
                out.push(tool);
                continue;
            }
            if ty.is_empty() || truthy(&tool["function"]) || tool["name"].is_string() {
                continue;
            }
            if HOSTED.contains(&ty.as_str()) {
                out.push(tool);
            }
            continue;
        }
        let f = if tool["function"].is_object() { tool["function"].clone() } else { Value::Null };
        let name = tool["name"].as_str().or_else(|| f["name"].as_str()).unwrap_or("").trim().to_string();
        if name.is_empty() {
            continue;
        }
        let desc = tool["description"].as_str().or_else(|| f["description"].as_str()).unwrap_or("").to_string();
        let params = if tool["parameters"].is_object() {
            tool["parameters"].clone()
        } else if f["parameters"].is_object() {
            f["parameters"].clone()
        } else {
            json!({"type": "object", "properties": {}})
        };
        let mut t = json!({"type": "function", "name": name.chars().take(128).collect::<String>()});
        if !desc.is_empty() {
            t["description"] = json!(desc);
        }
        t["parameters"] = strip_unsupported_patterns(&params);
        valid.push(name);
        out.push(t);
    }
    body["tools"] = Value::Array(out);
    if body["tool_choice"].is_object() && body["tool_choice"]["type"] == "function" {
        let n = body["tool_choice"]["name"].as_str().unwrap_or("").trim().to_string();
        if n.is_empty() || !valid.contains(&n) {
            del(body, "tool_choice");
        }
    }
}

fn normalize_effort(model: &str, value: &str) -> String {
    let levels = crate::caps::thinking_levels(Some("codex"), model);
    let has = |l: &str| levels.as_ref().map(|v| v.iter().any(|x| x == l)).unwrap_or(false);
    if has(value) {
        return value.into();
    }
    if is_lite(model) && (value == "none" || value == "minimal") {
        return "low".into();
    }
    if value == "ultra" && has("max") {
        return "max".into();
    }
    if value == "max" || value == "ultra" {
        return "xhigh".into();
    }
    value.into()
}

fn session_id(body: &Value, creds: &Value) -> String {
    crate::session::resolve_session_identity(
        &creds["rawHeaders"],
        body,
        creds["connectionId"].as_str(),
        creds["providerSpecificData"]["workspaceId"].as_str(),
        "codex",
    )
    .0
}

pub fn transform_codex_body(model: &str, mut body: Value, creds: &Value) -> (Value, String) {
    let compact = truthy(&body["_compact"]);
    del(&mut body, "_compact");
    let sid = session_id(&body, creds);
    if let Some(n) = crate::translate::req::normalize_responses_input(&body["input"]) {
        body["input"] = Value::Array(n);
    }
    let mm = if truthy(&body["model"]) { js_string(&body["model"]) } else { model.to_string() };
    let upstream = get_model_upstream_id("cx", &mm);
    let auto_ws = body["_autoCodexWebSearch"] == json!(true);
    del(&mut body, "_autoCodexWebSearch");
    let has_ws = |b: &Value| b["tools"].as_array().map(|t| t.iter().any(|x| x["type"] == "web_search")).unwrap_or(false);
    if auto_ws && !has_ws(&body) {
        let mut t = body["tools"].as_array().cloned().unwrap_or_default();
        t.push(json!({"type": "web_search"}));
        body["tools"] = Value::Array(t);
    }
    let mut converted_lite_prefix = false;
    let input_has_ws = body["input"].as_array().map(|a| a.iter().any(|i| i["type"] == "additional_tools" && i["tools"].as_array().map(|t| t.iter().any(|x| x["type"] == "web_search")).unwrap_or(false))).unwrap_or(false);
    if is_lite(&upstream) && body["input"].is_array() && (has_ws(&body) || input_has_ws) {
        let mut tools = body["tools"].as_array().cloned().unwrap_or_default();
        let key = |t: &Value| format!("{}:{}", js_string(&t["type"]), t["name"].as_str().or_else(|| t["function"]["name"].as_str()).unwrap_or(""));
        let mut seen: Vec<String> = tools.iter().map(key).collect();
        for item in body["input"].as_array().unwrap().clone() {
            if item["type"] != "additional_tools" {
                continue;
            }
            for t in item["tools"].as_array().into_iter().flatten() {
                let k = key(t);
                if !seen.contains(&k) {
                    tools.push(t.clone());
                    seen.push(k);
                }
            }
        }
        body["tools"] = Value::Array(tools);
        converted_lite_prefix = body["input"].as_array().unwrap().iter().any(|i| i["type"] == "additional_tools");
        let kept: Vec<Value> = body["input"].as_array().unwrap().iter().filter(|i| i["type"] != "additional_tools").cloned().collect();
        body["input"] = Value::Array(kept);
    }
    let lite = is_lite(&upstream) && !has_ws(&body);
    if !truthy(&body["input"]) || body["input"].as_array().map(|a| a.is_empty()).unwrap_or(false) {
        body["input"] = json!([{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}]);
    }
    if let Some(items) = body["input"].as_array_mut() {
        for it in items.iter_mut() {
            if it["role"] == "system" && (!truthy(&it["type"]) || it["type"] == "message") {
                it["role"] = json!("developer");
            }
        }
    }
    static SERVER_ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(rs|fc|resp|msg)_").unwrap());
    if let Some(items) = body["input"].as_array().cloned() {
        let kept: Vec<Value> = items
            .into_iter()
            .filter(|it| !(it.as_str().map(|s| SERVER_ID.is_match(s)).unwrap_or(false)) && it["type"] != "item_reference")
            .map(|mut it| {
                if let Some(id) = it["id"].as_str().map(str::to_owned) {
                    if SERVER_ID.is_match(&id) && !(lite && it["role"] == "developer" && id.starts_with("msg_")) {
                        del(&mut it, "id");
                    }
                }
                it
            })
            .collect();
        body["input"] = Value::Array(kept);
    }
    normalize_tools(&mut body);
    body["stream"] = json!(true);
    let instructions_empty = body["instructions"].as_str().map(|s| s.trim().is_empty()).unwrap_or(true);
    if !lite && !converted_lite_prefix && instructions_empty {
        body["instructions"] = json!(crate::consts::s("CODEX_DEFAULT_INSTRUCTIONS"));
    }
    body["store"] = json!(false);
    if !truthy(&body["prompt_cache_key"]) && !sid.is_empty() {
        body["prompt_cache_key"] = json!(sid);
    }
    body["model"] = json!(upstream);
    if lite {
        let mut input = body["input"].as_array().cloned().unwrap_or_default();
        if !input.iter().any(|i| i["type"] == "additional_tools") {
            let instr = body["instructions"].as_str().filter(|s| !s.trim().is_empty()).map(str::to_owned).unwrap_or_else(|| crate::consts::s("CODEX_DEFAULT_INSTRUCTIONS").to_string());
            let mut prefix = vec![json!({"type": "additional_tools", "role": "developer", "tools": body["tools"].as_array().cloned().unwrap_or_default()})];
            if !instr.is_empty() {
                prefix.push(json!({"type": "message", "role": "developer", "content": [{"type": "input_text", "text": instr}]}));
            }
            prefix.extend(input);
            input = prefix;
        }
        body["input"] = Value::Array(input);
        body["instructions"] = json!("");
        body["tools"] = Value::Null;
        if !truthy(&body["tool_choice"]) {
            body["tool_choice"] = json!("auto");
        }
        body["parallel_tool_calls"] = json!(false);
    } else {
        del(&mut body, "parallel_tool_calls");
    }
    let mut model_effort: Option<String> = None;
    let m = js_string(&body["model"]);
    for level in ["none", "minimal", "low", "medium", "high", "xhigh"] {
        if m.ends_with(&format!("-{level}")) {
            model_effort = Some(level.into());
            body["model"] = json!(m.replacen(&format!("-{level}"), "", 1));
            break;
        }
    }
    let bm = js_string(&body["model"]);
    if !truthy(&body["reasoning"]) {
        let src = body["reasoning_effort"].as_str().map(str::to_owned).or(model_effort).unwrap_or_else(|| if lite { "medium".into() } else { "low".into() });
        let e = normalize_effort(&bm, &src);
        body["reasoning"] = if lite { json!({"effort": e}) } else { json!({"effort": e, "summary": "auto"}) };
    } else {
        let e = normalize_effort(&bm, body["reasoning"]["effort"].as_str().unwrap_or(""));
        body["reasoning"]["effort"] = json!(e);
        if !lite && !truthy(&body["reasoning"]["summary"]) {
            body["reasoning"]["summary"] = json!("auto");
        }
    }
    if lite {
        body["reasoning"]["context"] = json!("all_turns");
    }
    del(&mut body, "reasoning_effort");
    if truthy(&body["reasoning"]["effort"]) && body["reasoning"]["effort"] != "none" {
        body["include"] = json!(["reasoning.encrypted_content"]);
    }
    if body["service_tier"] == "fast" {
        body["service_tier"] = json!("priority");
    }
    if truthy(&body["service_tier"]) && body["service_tier"] != "priority" {
        del(&mut body, "service_tier");
    }
    let keys: Vec<String> = body.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
    for k in keys {
        if !ALLOWLIST.contains(&k.as_str()) {
            del(&mut body, &k);
        }
    }
    if compact {
        body["_compact"] = json!(true);
    }
    (body, sid)
}

fn find_nested_message(v: &Value, depth: usize) -> Option<String> {
    if depth > 6 {
        return None;
    }
    match v {
        Value::Array(a) => a.iter().find_map(|x| find_nested_message(x, depth + 1)),
        Value::Object(o) => {
            for k in [&v["message"], &v["error"]["message"], &v["response"]["error"]["message"]] {
                if let Some(s) = k.as_str().filter(|s| !s.trim().is_empty()) {
                    return Some(s.to_string());
                }
            }
            o.values().find_map(|x| find_nested_message(x, depth + 1))
        }
        _ => None,
    }
}

const CAPACITY_MSG: &str = "Selected model is at capacity. Please try a different model.";

#[async_trait]
impl Executor for Codex {
    fn provider(&self) -> &str {
        "codex"
    }
    fn build_url(&self, _m: &str, _s: bool, _i: usize, _c: &Value) -> Result<String, String> {
        Ok(self.config()["baseUrl"].as_str().unwrap_or("").to_string())
    }
    fn build_headers(&self, creds: &Value, stream: bool, _u: &str, model: &str, body: &Value) -> Headers {
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.extend_obj(&self.config()["headers"]);
        if let Some(t) = creds["accessToken"].as_str().filter(|s| !s.is_empty()).or_else(|| creds["apiKey"].as_str()) {
            h.set("Authorization", format!("Bearer {t}"));
        }
        if stream {
            h.set("Accept", "text/event-stream");
        }
        let has_ws = body["tools"].as_array().map(|t| t.iter().any(|x| x["type"] == "web_search")).unwrap_or(false);
        if !model.is_empty() && is_lite(&get_model_upstream_id("cx", model)) && !has_ws {
            h.set("x-openai-internal-codex-responses-lite", "true");
        }
        let sid = creds["__codexSession"].as_str().filter(|s| !s.is_empty()).or_else(|| creds["connectionId"].as_str()).unwrap_or("default");
        h.set("session_id", sid);
        if h.get("originator").is_none() {
            h.set("originator", "codex_cli_rs");
        }
        let psd = &creds["providerSpecificData"];
        if let Some(a) = psd["workspaceId"].as_str().filter(|s| !s.is_empty()).or_else(|| psd["chatgptAccountId"].as_str().filter(|s| !s.is_empty())).or_else(|| psd["accountId"].as_str().filter(|s| !s.is_empty())) {
            h.set("ChatGPT-Account-ID", a);
        }
        h
    }
    fn transform_request(&self, _model: &str, body: Value, _s: bool, _c: &Value) -> Value {
        body
    }
    fn parse_error(&self, status: u16, body: &str) -> ParsedError {
        if status == 429 {
            if let Ok(j) = serde_json::from_str::<Value>(body) {
                let e = &j["error"];
                if e["type"] == "usage_limit_reached" {
                    let now = now_ms();
                    let mut reset = e["resets_at"].as_i64().filter(|r| *r > 0).map(|r| r * 1000).filter(|m| *m > now);
                    if reset.is_none() {
                        reset = e["resets_in_seconds"].as_i64().filter(|r| *r > 0).map(|r| now + r * 1000);
                    }
                    if reset.is_some() {
                        return ParsedError { status: 429, message: e["message"].as_str().unwrap_or(body).to_string(), resets_at_ms: reset };
                    }
                }
            }
        }
        ParsedError { status, message: if body.is_empty() { format!("HTTP {status}") } else { body.into() }, resets_at_ms: None }
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let (mut body, sid) = transform_codex_body(args.model, args.body.clone(), args.creds);
        let compact = truthy(&body["_compact"]);
        del(&mut body, "_compact");
        prefetch_images(&mut body).await;
        args.creds["__codexSession"] = json!(sid);
        let (attempts, delay) = (3, 2000u64);
        let mut attempt = 0;
        loop {
            let mut res = base_execute(self, ExecArgs { model: args.model, body: body.clone(), stream: args.stream, creds: args.creds, session_id: args.session_id.clone(), client_tool: args.client_tool.clone(), override_headers: args.override_headers.clone() }).await?;
            if compact {
                // Compaction endpoint (/responses/compact) — rebuild the request.
            }
            if !res.response.ok() {
                return Ok(res);
            }
            // Peek the first part of the SSE body for in-band transient errors.
            let mut prefix: Vec<Bytes> = vec![];
            let mut text = String::new();
            let mut matched: Option<(&str, bool)> = None;
            let mut stream = std::mem::replace(&mut res.response.body, futures::stream::empty().boxed());
            while text.len() < 256 * 1024 {
                let Some(c) = stream.next().await else { break };
                let Ok(c) = c else { break };
                text.push_str(&String::from_utf8_lossy(&c));
                prefix.push(c);
                let lower = text.to_lowercase();
                if let Some(p) = ["selected model is at capacity", "model_at_capacity"].into_iter().find(|p| lower.contains(p)) {
                    matched = Some((p, true));
                    break;
                }
                if let Some(p) = ["server_is_overloaded", "service_unavailable_error"].into_iter().find(|p| lower.contains(p)) {
                    matched = Some((p, false));
                    break;
                }
                if ["event: response.output_text.delta", "event: response.function_call_arguments.delta", "\"type\":\"response.output_text.delta\"", "\"type\":\"response.function_call_arguments.delta\""].iter().any(|p| lower.contains(p)) {
                    break;
                }
            }
            match matched {
                None => {
                    let pre = futures::stream::iter(prefix.into_iter().map(Ok));
                    res.response.body = pre.chain(stream).boxed();
                    return Ok(res);
                }
                Some((p, account_fallback)) => {
                    let msg = extract_sse_error(&text).unwrap_or_else(|| if account_fallback { CAPACITY_MSG.into() } else { p.to_string() });
                    if account_fallback || attempt >= attempts {
                        res.response = Upstream::json(503, &json!({"error": {"message": msg, "type": "server_error", "code": "service_unavailable"}}));
                        return Ok(res);
                    }
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
            }
        }
    }
}

fn extract_sse_error(text: &str) -> Option<String> {
    let re = Regex::new(r"(?i)Selected model is at capacity\. Please try a different model\.").unwrap();
    if let Some(m) = re.find(text) {
        return Some(m.as_str().to_string());
    }
    for line in text.lines() {
        let Some(d) = line.strip_prefix("data:") else { continue };
        let d = d.trim();
        if d.is_empty() || d == "[DONE]" {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(d) {
            if let Some(m) = find_nested_message(&v, 0) {
                return Some(m);
            }
        }
    }
    None
}

/// Inline remote images as data URIs (Codex cannot fetch URLs).
async fn prefetch_images(body: &mut Value) {
    let Some(items) = body["input"].as_array_mut() else { return };
    for item in items.iter_mut() {
        let Some(content) = item["content"].as_array_mut() else { continue };
        for c in content.iter_mut() {
            if c["type"] != "image_url" {
                continue;
            }
            let url = if c["image_url"].is_string() { js_string(&c["image_url"]) } else { c["image_url"]["url"].as_str().unwrap_or("").to_string() };
            let detail = c["image_url"]["detail"].as_str().unwrap_or("auto").to_string();
            if url.is_empty() {
                continue;
            }
            let final_url = if url.starts_with("data:") { url } else { crate::media::fetch_image_as_data_url(&url).await.unwrap_or(url) };
            *c = json!({"type": "input_image", "image_url": final_url, "detail": detail});
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_body_shaping() {
        let body = json!({"model": "gpt-5.6-terra", "input": "hello", "temperature": 0.3, "max_output_tokens": 10,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string", "pattern": "^\\p{L}+$"}}}}}]});
        let (b, _) = transform_codex_body("gpt-5.6-terra", body, &json!({}));
        assert!(b.get("temperature").is_none());
        assert!(b.get("max_output_tokens").is_none());
        assert_eq!(b["store"], false);
        assert_eq!(b["input"][0]["content"][0]["text"], "hello");
        assert_eq!(b["tools"][0]["name"], "f");
        assert!(b["tools"][0]["parameters"]["properties"]["a"].get("pattern").is_none());
        assert!(b["instructions"].as_str().unwrap().starts_with("You are Codex"));
    }
}
