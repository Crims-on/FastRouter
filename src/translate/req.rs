//! Request translators (port of translator/request/*.js except kiro/cursor/commandcode).

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use regex::Regex;
use serde_json::{Map, Value, json};

use super::concerns::{adjust_max_tokens, collapse_text_parts, desc_string, encode_data_uri, extract_text_content, parse_data_uri};
use super::gemini_fmt::{
    clean_json_schema, convert_openai_content_to_parts, default_safety_settings, generate_project_id, generate_request_id,
    generate_session_id, normalize_gemini_contents,
};
use super::thinking::budget_to_effort;
use crate::caps::caps_for;
use crate::consts::{CLAUDE_SYSTEM_PROMPT, sig_ag, sig_gemini_cli, sig_vertex};
use crate::jsv::{del, js_string, now_ms, safe_parse, truthy};
use crate::session::{derive_session_id, get_thought_signature, to_numeric_session_id};

/// Context a translator may need (mirrors the JS `credentials` argument).
#[derive(Default, Clone)]
pub struct Ctx {
    pub client_session_id: Option<String>,
    pub project_id: Option<String>,
    pub email: Option<String>,
    pub connection_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Claude → OpenAI
// ---------------------------------------------------------------------------

fn strip_billing_header(text: &str) -> String {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^x-anthropic-billing-header:[^\n]*(?:\r?\n)?").unwrap());
    RE.replace(text, "").to_string()
}

pub fn claude_to_openai(model: &str, body: &Value, stream: bool) -> Value {
    let mut result = json!({"model": model, "messages": [], "stream": stream});
    if truthy(&body["max_tokens"]) {
        result["max_tokens"] = json!(adjust_max_tokens(body, None));
    }
    if !body["temperature"].is_null() {
        result["temperature"] = body["temperature"].clone();
    }
    if truthy(&body["system"]) {
        let content = match &body["system"] {
            Value::Array(a) => a
                .iter()
                .map(|s| strip_billing_header(s["text"].as_str().unwrap_or("")))
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
            other => strip_billing_header(other.as_str().unwrap_or("")),
        };
        if !content.is_empty() {
            result["messages"].as_array_mut().unwrap().push(json!({"role": "system", "content": content}));
        }
    }
    for msg in body["messages"].as_array().into_iter().flatten() {
        let mut msg = msg.clone();
        for m in convert_claude_message(&mut msg) {
            result["messages"].as_array_mut().unwrap().push(m);
        }
    }
    fix_missing_tool_responses_openai(result["messages"].as_array_mut().unwrap());
    if let Some(tools) = body["tools"].as_array() {
        result["tools"] = Value::Array(
            tools
                .iter()
                .map(|t| {
                    json!({"type": "function", "function": {
                        "name": t["name"],
                        "description": desc_string(&t["description"]),
                        "parameters": if truthy(&t["input_schema"]) { t["input_schema"].clone() } else { json!({"type": "object", "properties": {}}) },
                    }})
                })
                .collect(),
        );
    }
    if truthy(&body["tool_choice"]) {
        let c = &body["tool_choice"];
        result["tool_choice"] = if c.is_string() {
            c.clone()
        } else {
            match c["type"].as_str() {
                Some("any") => json!("required"),
                Some("tool") => json!({"type": "function", "function": {"name": c["name"]}}),
                _ => json!("auto"),
            }
        };
    }
    if !body["reasoning_effort"].is_null() {
        result["reasoning_effort"] = body["reasoning_effort"].clone();
    } else if !body["reasoning"]["effort"].is_null() {
        result["reasoning_effort"] = body["reasoning"]["effort"].clone();
    }
    if !body["reasoning"].is_null() {
        result["reasoning"] = body["reasoning"].clone();
    }
    result
}

fn fix_missing_tool_responses_openai(messages: &mut Vec<Value>) {
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        let tcs = msg["tool_calls"].as_array().cloned().unwrap_or_default();
        if msg["role"] == "assistant" && !tcs.is_empty() {
            let ids: Vec<Value> = tcs.iter().map(|t| t["id"].clone()).collect();
            let mut responded = vec![];
            let mut insert = i + 1;
            for (j, next) in messages.iter().enumerate().skip(i + 1) {
                if next["role"] == "tool" && truthy(&next["tool_call_id"]) {
                    responded.push(next["tool_call_id"].clone());
                    insert = j + 1;
                } else {
                    break;
                }
            }
            let missing: Vec<Value> = ids
                .into_iter()
                .filter(|id| !responded.contains(id))
                .map(|id| json!({"role": "tool", "tool_call_id": id, "content": "[No response received]"}))
                .collect();
            if !missing.is_empty() {
                let n = missing.len();
                for (k, m) in missing.into_iter().enumerate() {
                    messages.insert(insert + k, m);
                }
                i = insert + n - 1;
            }
        }
        i += 1;
    }
}

fn system_reminder_text(content: &Value) -> String {
    let parts: Vec<String> = match content {
        Value::Array(a) => a.iter().filter(|c| c["type"] == "text").map(|c| c["text"].as_str().unwrap_or("").to_string()).collect(),
        Value::String(s) => vec![s.clone()],
        _ => vec![String::new()],
    };
    let text = parts.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n");
    if text.trim().is_empty() {
        return String::new();
    }
    format!("<instructions>\n{text}\n</instructions>")
}

fn convert_claude_message(msg: &mut Value) -> Vec<Value> {
    if msg["content"].is_object() {
        let c = msg["content"].take();
        msg["content"] = json!([c]);
    }
    if msg["role"] == "system" {
        let text = system_reminder_text(&msg["content"]);
        return if text.is_empty() { vec![] } else { vec![json!({"role": "user", "content": text})] };
    }
    let role = if msg["role"] == "user" || msg["role"] == "tool" { "user" } else { "assistant" };
    if let Some(s) = msg["content"].as_str() {
        return vec![json!({"role": role, "content": s})];
    }
    let Some(blocks) = msg["content"].as_array() else { return vec![] };
    let mut parts = vec![];
    let mut tool_calls = vec![];
    let mut tool_results = vec![];
    for block in blocks {
        match block["type"].as_str().unwrap_or("") {
            "text" => parts.push(json!({"type": "text", "text": block["text"]})),
            "image" => {
                if block["source"]["type"] == "base64" {
                    parts.push(json!({"type": "image_url", "image_url": {"url": encode_data_uri(
                        &js_string(&block["source"]["media_type"]), &js_string(&block["source"]["data"]))}}));
                }
            }
            "tool_use" => tool_calls.push(json!({
                "id": block["id"],
                "type": "function",
                "function": {"name": block["name"], "arguments": if truthy(&block["input"]) { block["input"].to_string() } else { "{}".into() }},
            })),
            "tool_result" => {
                let mut result_content = String::new();
                let mut images = vec![];
                match &block["content"] {
                    Value::String(s) => result_content = s.clone(),
                    Value::Array(a) => {
                        for c in a {
                            if c["type"] == "image" && c["source"]["type"] == "base64" {
                                images.push(json!({"type": "image_url", "image_url": {"url": encode_data_uri(
                                    &js_string(&c["source"]["media_type"]), &js_string(&c["source"]["data"]))}}));
                            }
                        }
                        let text_only: Vec<String> = a.iter().filter(|c| c["type"] == "text").map(|c| js_string(&c["text"])).collect();
                        result_content = text_only.join("\n");
                        if result_content.is_empty() && images.is_empty() {
                            result_content = block["content"].to_string();
                        }
                    }
                    other if truthy(other) => result_content = other.to_string(),
                    _ => {}
                }
                tool_results.push(json!({"role": "tool", "tool_call_id": block["tool_use_id"], "content": result_content}));
                if !images.is_empty() {
                    parts.push(json!({"type": "text", "text": format!("[Image from tool result {}]", js_string(&block["tool_use_id"]))}));
                    parts.extend(images);
                }
            }
            _ => {}
        }
    }
    if !tool_results.is_empty() {
        if !parts.is_empty() {
            tool_results.push(json!({"role": "user", "content": collapse_text_parts(parts)}));
        }
        return tool_results;
    }
    if !tool_calls.is_empty() {
        let mut r = json!({"role": "assistant"});
        if !parts.is_empty() {
            r["content"] = collapse_text_parts(parts);
        }
        r["tool_calls"] = Value::Array(tool_calls);
        return vec![r];
    }
    if !parts.is_empty() {
        return vec![json!({"role": role, "content": collapse_text_parts(parts)})];
    }
    if blocks.is_empty() {
        return vec![json!({"role": role, "content": ""})];
    }
    vec![]
}

// ---------------------------------------------------------------------------
// OpenAI → Claude
// ---------------------------------------------------------------------------

fn claude_blocks_from_message(msg: &Value) -> Vec<Value> {
    let mut blocks = vec![];
    let role = msg["role"].as_str().unwrap_or("");
    if role == "tool" {
        blocks.push(json!({"type": "tool_result", "tool_use_id": msg["tool_call_id"], "content": msg["content"]}));
    } else if role == "user" {
        match &msg["content"] {
            Value::String(s) => {
                if !s.is_empty() {
                    blocks.push(json!({"type": "text", "text": s}));
                }
            }
            Value::Array(parts) => {
                for part in parts {
                    let ty = part["type"].as_str().unwrap_or("");
                    if ty == "text" && truthy(&part["text"]) {
                        blocks.push(json!({"type": "text", "text": part["text"]}));
                    } else if ty == "tool_result" {
                        let mut b = json!({"type": "tool_result", "tool_use_id": part["tool_use_id"], "content": part["content"]});
                        if truthy(&part["is_error"]) {
                            b["is_error"] = part["is_error"].clone();
                        }
                        blocks.push(b);
                    } else if ty == "image_url" {
                        let url = &part["image_url"]["url"];
                        if let Some((mime, data)) = parse_data_uri(url) {
                            blocks.push(json!({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}));
                        } else if let Some(u) = url.as_str().filter(|u| u.starts_with("http://") || u.starts_with("https://")) {
                            blocks.push(json!({"type": "image", "source": {"type": "url", "url": u}}));
                        }
                    } else if ty == "image" && truthy(&part["source"]) {
                        blocks.push(json!({"type": "image", "source": part["source"]}));
                    } else if ty == "file" && truthy(&part["file"]) {
                        if let Some((mime, data)) = parse_data_uri(&part["file"]["file_data"]) {
                            if mime == "application/pdf" {
                                blocks.push(json!({"type": "document", "source": {"type": "base64", "media_type": mime, "data": data}}));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    } else if role == "assistant" {
        match &msg["content"] {
            Value::Array(parts) => {
                for part in parts {
                    let ty = part["type"].as_str().unwrap_or("");
                    if ty == "text" && truthy(&part["text"]) {
                        blocks.push(json!({"type": "text", "text": part["text"]}));
                    } else if ty == "tool_use" {
                        blocks.push(json!({"type": "tool_use", "id": part["id"], "name": part["name"], "input": part["input"]}));
                    } else if ty == "thinking" {
                        blocks.push(crate::jsv::omit(part, &["cache_control"]));
                    }
                }
            }
            c if truthy(c) => {
                let text = c.as_str().map(str::to_owned).unwrap_or_else(|| extract_text_content(c, "\n"));
                if !text.is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
            }
            _ => {}
        }
        for tc in msg["tool_calls"].as_array().into_iter().flatten() {
            if tc["type"] == "function" {
                let args = &tc["function"]["arguments"];
                blocks.push(json!({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": safe_parse(args, args.clone())}));
            }
        }
    }
    blocks
}

fn convert_openai_tool_choice(choice: &Value) -> Value {
    match choice {
        Value::Null => json!({"type": "auto"}),
        Value::String(s) => {
            if s == "required" {
                json!({"type": "any"})
            } else {
                json!({"type": "auto"})
            }
        }
        Value::Object(_) => {
            if truthy(&choice["function"]["name"]) {
                return json!({"type": "tool", "name": choice["function"]["name"]});
            }
            if matches!(choice["type"].as_str(), Some("auto" | "any" | "tool" | "none")) {
                return choice.clone();
            }
            json!({"type": "auto"})
        }
        _ => json!({"type": "auto"}),
    }
}

pub fn openai_to_claude(model: &str, body: &Value, stream: bool) -> Value {
    let ceiling = caps_for(None, model).max_output().filter(|m| *m != 0);
    let mut result = json!({"model": model, "max_tokens": adjust_max_tokens(body, ceiling), "stream": stream});
    if !body["temperature"].is_null() {
        result["temperature"] = body["temperature"].clone();
    }
    let mut messages: Vec<Value> = vec![];
    let mut system_parts: Vec<String> = vec![];
    if let Some(msgs) = body["messages"].as_array() {
        for m in msgs {
            if m["role"] == "system" {
                system_parts.push(m["content"].as_str().map(str::to_owned).unwrap_or_else(|| extract_text_content(&m["content"], "\n")));
            }
        }
        let mut current_role: Option<&str> = None;
        let mut current: Vec<Value> = vec![];
        let flush = |messages: &mut Vec<Value>, role: Option<&str>, current: &mut Vec<Value>| {
            if let Some(r) = role {
                if !current.is_empty() {
                    messages.push(json!({"role": r, "content": std::mem::take(current)}));
                }
            }
        };
        for msg in msgs.iter().filter(|m| m["role"] != "system") {
            let new_role = if msg["role"] == "user" || msg["role"] == "tool" { "user" } else { "assistant" };
            let blocks = claude_blocks_from_message(msg);
            let has_tool_use = blocks.iter().any(|b| b["type"] == "tool_use");
            let has_tool_result = blocks.iter().any(|b| b["type"] == "tool_result");
            if has_tool_result {
                let (tr, other): (Vec<Value>, Vec<Value>) = blocks.into_iter().partition(|b| b["type"] == "tool_result");
                flush(&mut messages, current_role, &mut current);
                if !tr.is_empty() {
                    messages.push(json!({"role": "user", "content": tr}));
                }
                if !other.is_empty() {
                    current_role = Some(new_role);
                    current.extend(other);
                }
                continue;
            }
            if current_role != Some(new_role) {
                flush(&mut messages, current_role, &mut current);
                current_role = Some(new_role);
            }
            current.extend(blocks);
            if has_tool_use {
                flush(&mut messages, current_role, &mut current);
            }
        }
        flush(&mut messages, current_role, &mut current);
        for m in messages.iter_mut().rev() {
            if m["role"] == "assistant" && m["content"].as_array().map(|a| !a.is_empty()).unwrap_or(false) {
                if let Some(c) = m["content"].as_array_mut() {
                    for b in c.iter_mut().rev() {
                        if matches!(b["type"].as_str(), Some("text" | "tool_use" | "tool_result" | "image")) {
                            b["cache_control"] = json!({"type": "ephemeral"});
                            break;
                        }
                    }
                }
                break;
            }
        }
    }
    result["messages"] = Value::Array(messages);
    let rf = &body["response_format"];
    if truthy(rf) {
        if rf["type"] == "json_schema" && truthy(&rf["json_schema"]["schema"]) {
            let schema = serde_json::to_string_pretty(&rf["json_schema"]["schema"]).unwrap_or_default();
            system_parts.push(format!(
                "You must respond with valid JSON that strictly follows this JSON schema:\n```json\n{schema}\n```\nRespond ONLY with the JSON object, no other text."
            ));
        } else if rf["type"] == "json_object" {
            system_parts.push("You must respond with valid JSON. Respond ONLY with a JSON object, no other text.".into());
        }
    }
    let cc = json!({"type": "text", "text": CLAUDE_SYSTEM_PROMPT});
    result["system"] = if system_parts.is_empty() {
        json!([cc])
    } else {
        json!([cc, {"type": "text", "text": system_parts.join("\n"), "cache_control": {"type": "ephemeral", "ttl": "1h"}}])
    };
    if let Some(tools) = body["tools"].as_array() {
        let mut out = vec![];
        for tool in tools {
            let ty = &tool["type"];
            if truthy(ty) && ty != "function" {
                out.push(tool.clone());
                continue;
            }
            let data = if tool["function"].is_null() { tool } else { &tool["function"] };
            out.push(json!({
                "name": data["name"],
                "description": if truthy(&data["description"]) { data["description"].clone() } else { json!("") },
                "input_schema": if truthy(&data["parameters"]) { data["parameters"].clone() } else if truthy(&data["input_schema"]) { data["input_schema"].clone() } else { json!({"type": "object", "properties": {}, "required": []}) },
            }));
        }
        if let Some(last) = out.last_mut() {
            last["cache_control"] = json!({"type": "ephemeral", "ttl": "1h"});
        }
        result["tools"] = Value::Array(out);
    }
    if truthy(&body["tool_choice"]) {
        result["tool_choice"] = convert_openai_tool_choice(&body["tool_choice"]);
    }
    result
}

pub fn openai_to_claude_for_antigravity(model: &str, body: &Value, stream: bool) -> Value {
    let mut r = openai_to_claude(model, body, stream);
    if let Some(sys) = r["system"].as_array() {
        let kept: Vec<Value> = sys.iter().filter(|b| !b["text"].as_str().map(|t| t.contains("You are Claude Code")).unwrap_or(false)).cloned().collect();
        if kept.is_empty() {
            del(&mut r, "system");
        } else {
            r["system"] = Value::Array(kept);
        }
    }
    r
}

// ---------------------------------------------------------------------------
// OpenAI → Gemini / Gemini CLI / Antigravity / Vertex
// ---------------------------------------------------------------------------

pub fn sanitize_gemini_function_name(name: &Value) -> String {
    static BAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-zA-Z0-9_.:\-]").unwrap());
    let Some(n) = name.as_str().filter(|s| !s.is_empty()) else { return "_unknown".into() };
    let mut s = BAD.replace_all(n, "_").to_string();
    let first_ok = s.chars().next().map(|c| c.is_ascii_alphabetic() || c == '_').unwrap_or(false);
    if !first_ok {
        s = format!("_{s}");
    }
    s.chars().take(64).collect()
}

fn try_parse_json(v: &Value) -> Value {
    safe_parse(v, Value::Null)
}

fn openai_to_gemini_base(model: &str, body: &Value, signature: &str, session_id: Option<&str>) -> Value {
    let mut result = json!({"model": model, "contents": [], "generationConfig": {}, "safetySettings": default_safety_settings()});
    for (src, dst) in [("temperature", "temperature"), ("top_p", "topP"), ("top_k", "topK"), ("max_tokens", "maxOutputTokens")] {
        if !body[src].is_null() {
            result["generationConfig"][dst] = body[src].clone();
        }
    }
    let msgs = body["messages"].as_array().cloned().unwrap_or_default();
    let mut id2name: HashMap<String, String> = HashMap::new();
    let mut responses: HashMap<String, Value> = HashMap::new();
    for m in &msgs {
        if m["role"] == "assistant" {
            for tc in m["tool_calls"].as_array().into_iter().flatten() {
                if tc["type"] == "function" && truthy(&tc["id"]) && truthy(&tc["function"]["name"]) {
                    id2name.insert(js_string(&tc["id"]), js_string(&tc["function"]["name"]));
                }
            }
        }
        if m["role"] == "tool" && truthy(&m["tool_call_id"]) {
            responses.insert(js_string(&m["tool_call_id"]), m["content"].clone());
        }
    }
    let mut contents: Vec<Value> = vec![];
    let n = msgs.len();
    for (i, msg) in msgs.iter().enumerate() {
        let role = msg["role"].as_str().unwrap_or("");
        let content = &msg["content"];
        if role == "system" && n > 1 {
            let text = content.as_str().map(str::to_owned).unwrap_or_else(|| extract_text_content(content, ""));
            result["systemInstruction"] = json!({"role": "user", "parts": [{"text": text}]});
        } else if role == "user" || (role == "system" && n == 1) {
            let parts = convert_openai_content_to_parts(content);
            if !parts.is_empty() {
                contents.push(json!({"role": "user", "parts": parts}));
            }
        } else if role == "assistant" {
            let mut parts = vec![];
            if truthy(&msg["reasoning_content"]) {
                parts.push(json!({"thought": true, "text": msg["reasoning_content"]}));
                parts.push(json!({"thoughtSignature": signature, "text": ""}));
            }
            if truthy(content) {
                let text = content.as_str().map(str::to_owned).unwrap_or_else(|| extract_text_content(content, ""));
                if !text.is_empty() {
                    parts.push(json!({"text": text}));
                }
            }
            if let Some(tcs) = msg["tool_calls"].as_array() {
                let mut ids = vec![];
                let mut first_seen = false;
                for tc in tcs {
                    if tc["type"] != "function" {
                        continue;
                    }
                    let args_src = if truthy(&tc["function"]["arguments"]) { tc["function"]["arguments"].clone() } else { json!("{}") };
                    let args = try_parse_json(&args_src);
                    let id = js_string(&tc["id"]);
                    let cached = if truthy(&tc["id"]) { get_thought_signature(&id, session_id, Some(model)) } else { None };
                    let call_sig = cached.or_else(|| (!first_seen).then(|| signature.to_string()));
                    first_seen = true;
                    let mut part = json!({"functionCall": {"id": tc["id"], "name": sanitize_gemini_function_name(&tc["function"]["name"]), "args": args}});
                    if let Some(s) = call_sig {
                        part["thoughtSignature"] = json!(s);
                    }
                    parts.push(part);
                    ids.push(id);
                }
                if !parts.is_empty() {
                    contents.push(json!({"role": "model", "parts": parts}));
                }
                let intermediate = i < n - 1;
                let has_actual = ids.iter().any(|f| responses.contains_key(f));
                if has_actual || intermediate {
                    let mut tool_parts = vec![];
                    for fid in &ids {
                        let resp = responses.get(fid).cloned().unwrap_or(json!(""));
                        let name = id2name.get(fid).cloned().unwrap_or_else(|| {
                            let p: Vec<&str> = fid.split('-').collect();
                            if p.len() > 2 { p[..p.len() - 2].join("-") } else { fid.clone() }
                        });
                        let mut parsed = try_parse_json(&resp);
                        if parsed.is_null() {
                            parsed = json!({"result": resp});
                        } else if !parsed.is_object() && !parsed.is_array() {
                            parsed = json!({"result": parsed});
                        }
                        tool_parts.push(json!({"functionResponse": {"id": fid, "name": sanitize_gemini_function_name(&json!(name)), "response": {"result": parsed}}}));
                    }
                    if !tool_parts.is_empty() {
                        contents.push(json!({"role": "user", "parts": tool_parts}));
                    }
                }
            } else if !parts.is_empty() {
                contents.push(json!({"role": "model", "parts": parts}));
            }
        }
    }
    if let Some(tools) = body["tools"].as_array().filter(|t| !t.is_empty()) {
        let mut decls = vec![];
        for t in tools {
            if truthy(&t["name"]) && truthy(&t["input_schema"]) {
                decls.push(json!({"name": sanitize_gemini_function_name(&t["name"]), "description": if truthy(&t["description"]) { t["description"].clone() } else { json!("") }, "parameters": clean_json_schema(t["input_schema"].clone())}));
            } else if t["type"] == "function" && truthy(&t["function"]) {
                let f = &t["function"];
                let params = if truthy(&f["parameters"]) { f["parameters"].clone() } else { json!({"type": "object", "properties": {}}) };
                decls.push(json!({"name": sanitize_gemini_function_name(&f["name"]), "description": if truthy(&f["description"]) { f["description"].clone() } else { json!("") }, "parameters": clean_json_schema(params)}));
            }
        }
        if !decls.is_empty() {
            result["tools"] = json!([{"functionDeclarations": decls}]);
        }
    }
    result["contents"] = Value::Array(normalize_gemini_contents(&contents));
    result
}

pub fn openai_to_gemini(model: &str, body: &Value, ctx: &Ctx) -> Value {
    openai_to_gemini_base(model, body, sig_ag(), ctx.client_session_id.as_deref())
}

pub fn openai_to_gemini_cli(model: &str, body: &Value, ctx: &Ctx) -> Value {
    let mut g = openai_to_gemini_base(model, body, sig_gemini_cli(), ctx.client_session_id.as_deref());
    if let Some(fds) = g["tools"][0]["functionDeclarations"].as_array_mut() {
        for f in fds {
            if truthy(&f["parameters"]) {
                f["parameters"] = clean_json_schema(f["parameters"].take());
            }
        }
    }
    g
}

fn session_for_envelope(ctx: &Ctx, antigravity: bool) -> String {
    to_numeric_session_id(ctx.client_session_id.as_deref()).unwrap_or_else(|| {
        if antigravity {
            derive_session_id(ctx.email.as_deref().or(ctx.connection_id.as_deref()))
        } else {
            generate_session_id()
        }
    })
}

pub fn wrap_cloud_code_envelope(model: &str, g: &Value, ctx: &Ctx, antigravity: bool) -> Value {
    let project = ctx.project_id.clone().unwrap_or_else(generate_project_id);
    let mut request = Map::new();
    request.insert("sessionId".into(), json!(session_for_envelope(ctx, antigravity)));
    request.insert("contents".into(), g["contents"].clone());
    for k in ["systemInstruction", "generationConfig", "tools"] {
        if !g[k].is_null() {
            request.insert(k.into(), g[k].clone());
        }
    }
    let mut env = json!({
        "project": project,
        "model": model,
        "userAgent": if antigravity { "antigravity" } else { "gemini-cli" },
        "requestId": if antigravity { format!("agent-{}", uuid::Uuid::new_v4()) } else { generate_request_id() },
        "request": Value::Object(request),
    });
    if !antigravity && !g["safetySettings"].is_null() {
        env["request"]["safetySettings"] = g["safetySettings"].clone();
    }
    if g["tools"].as_array().map(|t| !t.is_empty()).unwrap_or(false) {
        env["request"]["toolConfig"] = json!({"functionCallingConfig": {"mode": "VALIDATED"}});
    }
    env
}

fn wrap_cloud_code_envelope_for_claude(model: &str, claude: &Value, ctx: &Ctx) -> Value {
    let project = ctx.project_id.clone().unwrap_or_else(generate_project_id);
    let temp = if truthy(&claude["temperature"]) { claude["temperature"].clone() } else { json!(1) };
    let max = if truthy(&claude["max_tokens"]) { claude["max_tokens"].clone() } else { json!(4096) };
    let mut env = json!({
        "project": project,
        "model": model,
        "userAgent": "antigravity",
        "requestId": format!("agent-{}", uuid::Uuid::new_v4()),
        "request": {
            "sessionId": session_for_envelope(ctx, true),
            "contents": [],
            "generationConfig": {"temperature": temp, "maxOutputTokens": max},
        }
    });
    let mut id2name: HashMap<String, String> = HashMap::new();
    for m in claude["messages"].as_array().into_iter().flatten() {
        for b in m["content"].as_array().into_iter().flatten() {
            if b["type"] == "tool_use" && truthy(&b["id"]) && truthy(&b["name"]) {
                id2name.insert(js_string(&b["id"]), js_string(&b["name"]));
            }
        }
    }
    let mut contents = vec![];
    for msg in claude["messages"].as_array().into_iter().flatten() {
        let mut parts = vec![];
        if let Some(blocks) = msg["content"].as_array() {
            let mut first_seen = false;
            for b in blocks {
                match b["type"].as_str().unwrap_or("") {
                    "text" => parts.push(json!({"text": b["text"]})),
                    "tool_use" => {
                        let cached = if truthy(&b["id"]) { get_thought_signature(&js_string(&b["id"]), ctx.client_session_id.as_deref(), Some(model)) } else { None };
                        let sig = cached.or_else(|| (!first_seen).then(|| sig_ag().to_string()));
                        first_seen = true;
                        let mut part = json!({"functionCall": {"id": b["id"], "name": sanitize_gemini_function_name(&b["name"]), "args": if truthy(&b["input"]) { b["input"].clone() } else { json!({}) }}});
                        if let Some(s) = sig {
                            part["thoughtSignature"] = json!(s);
                        }
                        parts.push(part);
                    }
                    "tool_result" => {
                        let content = match &b["content"] {
                            Value::Array(a) => json!(a.iter().map(|c| if c["type"] == "text" { js_string(&c["text"]) } else { c.to_string() }).collect::<Vec<_>>().join("\n")),
                            other => other.clone(),
                        };
                        let name = id2name.get(&js_string(&b["tool_use_id"])).map(|n| sanitize_gemini_function_name(&json!(n))).unwrap_or_else(|| "tool".into());
                        let parsed = try_parse_json(&content);
                        parts.push(json!({"functionResponse": {"id": b["tool_use_id"], "name": name, "response": {"result": if truthy(&parsed) { parsed } else { content }}}}));
                    }
                    _ => {}
                }
            }
        } else if let Some(s) = msg["content"].as_str() {
            parts.push(json!({"text": s}));
        }
        if !parts.is_empty() {
            contents.push(json!({"role": if msg["role"] == "assistant" { "model" } else { "user" }, "parts": parts}));
        }
    }
    if let Some(tools) = claude["tools"].as_array() {
        let decls: Vec<Value> = tools
            .iter()
            .filter(|t| truthy(&t["name"]) && truthy(&t["input_schema"]))
            .map(|t| json!({"name": sanitize_gemini_function_name(&t["name"]), "description": if truthy(&t["description"]) { t["description"].clone() } else { json!("") }, "parameters": clean_json_schema(t["input_schema"].clone())}))
            .collect();
        if !decls.is_empty() {
            env["request"]["tools"] = json!([{"functionDeclarations": decls}]);
            env["request"]["toolConfig"] = json!({"functionCallingConfig": {"mode": "VALIDATED"}});
        }
    }
    let mut sys_parts = vec![];
    match &claude["system"] {
        Value::Array(a) => {
            for b in a {
                if truthy(&b["text"]) {
                    sys_parts.push(json!({"text": b["text"]}));
                }
            }
        }
        Value::String(s) => sys_parts.push(json!({"text": s})),
        _ => {}
    }
    if !sys_parts.is_empty() {
        env["request"]["systemInstruction"] = json!({"role": "user", "parts": sys_parts});
    }
    env["request"]["contents"] = Value::Array(normalize_gemini_contents(&contents));
    env
}

pub fn openai_to_antigravity(model: &str, body: &Value, stream: bool, ctx: &Ctx) -> Value {
    if model.to_lowercase().contains("claude") {
        let c = openai_to_claude_for_antigravity(model, body, stream);
        return wrap_cloud_code_envelope_for_claude(model, &c, ctx);
    }
    let g = openai_to_gemini_cli(model, body, &Ctx::default());
    wrap_cloud_code_envelope(model, &g, ctx, true)
}

pub fn openai_to_vertex(model: &str, body: &Value, ctx: &Ctx) -> Value {
    let mut g = openai_to_gemini(model, body, ctx);
    if let Some(contents) = g["contents"].as_array_mut() {
        for turn in contents {
            for part in turn["parts"].as_array_mut().into_iter().flatten() {
                if crate::jsv::has(part, "thoughtSignature") {
                    part["thoughtSignature"] = json!(sig_vertex());
                }
                if part["functionCall"].is_object() {
                    del(&mut part["functionCall"], "id");
                }
                if part["functionResponse"].is_object() {
                    del(&mut part["functionResponse"], "id");
                }
            }
        }
    }
    g
}

// ---------------------------------------------------------------------------
// Gemini / Antigravity → OpenAI
// ---------------------------------------------------------------------------

pub fn gemini_to_openai(model: &str, body: &Value, stream: bool) -> Value {
    let mut result = json!({"model": model, "messages": [], "stream": stream});
    let cfg = &body["generationConfig"];
    if truthy(cfg) {
        if truthy(&cfg["maxOutputTokens"]) {
            result["max_tokens"] = json!(adjust_max_tokens(&json!({"max_tokens": cfg["maxOutputTokens"], "tools": body["tools"]}), None));
        }
        if !cfg["temperature"].is_null() {
            result["temperature"] = cfg["temperature"].clone();
        }
        if !cfg["topP"].is_null() {
            result["top_p"] = cfg["topP"].clone();
        }
    }
    if truthy(&body["systemInstruction"]) {
        let text = extract_gemini_text(&body["systemInstruction"]);
        if !text.is_empty() {
            result["messages"].as_array_mut().unwrap().push(json!({"role": "system", "content": text}));
        }
    }
    for c in body["contents"].as_array().into_iter().flatten() {
        if let Some(m) = convert_gemini_content(c) {
            result["messages"].as_array_mut().unwrap().push(m);
        }
    }
    if let Some(tools) = body["tools"].as_array() {
        let mut out = vec![];
        for t in tools {
            for f in t["functionDeclarations"].as_array().into_iter().flatten() {
                out.push(json!({"type": "function", "function": {"name": f["name"], "description": if truthy(&f["description"]) { f["description"].clone() } else { json!("") }, "parameters": if truthy(&f["parameters"]) { f["parameters"].clone() } else { json!({"type": "object", "properties": {}}) }}}));
            }
        }
        result["tools"] = Value::Array(out);
    }
    result
}

fn extract_gemini_text(c: &Value) -> String {
    if let Some(s) = c.as_str() {
        return s.into();
    }
    c["parts"].as_array().map(|p| p.iter().map(|x| x["text"].as_str().unwrap_or("").to_string()).collect()).unwrap_or_default()
}

fn convert_gemini_content(content: &Value) -> Option<Value> {
    let role = if content["role"] == "user" { "user" } else { "assistant" };
    let parts_in = content["parts"].as_array()?;
    let mut parts = vec![];
    let mut tool_calls = vec![];
    for part in parts_in {
        if !part["text"].is_null() || crate::jsv::has(part, "text") {
            parts.push(json!({"type": "text", "text": part["text"]}));
        }
        if truthy(&part["inlineData"]) {
            parts.push(json!({"type": "image_url", "image_url": {"url": encode_data_uri(&js_string(&part["inlineData"]["mimeType"]), &js_string(&part["inlineData"]["data"]))}}));
        }
        if truthy(&part["functionCall"]) {
            let fc = &part["functionCall"];
            let id = if truthy(&fc["id"]) { fc["id"].clone() } else { json!(format!("call_{}", js_string(&fc["name"]))) };
            tool_calls.push(json!({"id": id, "type": "function", "function": {"name": fc["name"], "arguments": if truthy(&fc["args"]) { fc["args"].to_string() } else { "{}".into() }}}));
        }
        if truthy(&part["functionResponse"]) {
            let fr = &part["functionResponse"];
            let id = if truthy(&fr["id"]) { fr["id"].clone() } else { json!(format!("call_{}", js_string(&fr["name"]))) };
            let payload = if truthy(&fr["response"]["result"]) {
                fr["response"]["result"].clone()
            } else if truthy(&fr["response"]) {
                fr["response"].clone()
            } else {
                json!({})
            };
            return Some(json!({"role": "tool", "tool_call_id": id, "content": payload.to_string()}));
        }
    }
    if !tool_calls.is_empty() {
        let mut r = json!({"role": "assistant"});
        if !parts.is_empty() {
            r["content"] = if parts.len() == 1 { parts[0]["text"].clone() } else { Value::Array(parts) };
        }
        r["tool_calls"] = Value::Array(tool_calls);
        return Some(r);
    }
    if !parts.is_empty() {
        return Some(json!({"role": role, "content": collapse_text_parts(parts)}));
    }
    None
}

fn normalize_schema_types(schema: &Value) -> Value {
    if !schema.is_object() && !schema.is_array() {
        return schema.clone();
    }
    let mut r = schema.clone();
    if let Some(t) = r["type"].as_str().map(|s| s.to_lowercase()) {
        if r.is_object() {
            r["type"] = json!(t);
        }
    }
    del(&mut r, "enumDescriptions");
    if let Some(p) = r["properties"].as_object().cloned() {
        r["properties"] = Value::Object(p.into_iter().map(|(k, v)| (k, normalize_schema_types(&v))).collect());
    }
    if truthy(&r["items"]) {
        r["items"] = normalize_schema_types(&r["items"]);
    }
    r
}

pub fn antigravity_to_openai(model: &str, body: &Value, stream: bool) -> Value {
    let req = if truthy(&body["request"]) { &body["request"] } else { body };
    let mut result = json!({"model": model, "messages": [], "stream": stream});
    let cfg = &req["generationConfig"];
    if truthy(cfg) {
        if truthy(&cfg["maxOutputTokens"]) {
            result["max_tokens"] = json!(adjust_max_tokens(&json!({"max_tokens": cfg["maxOutputTokens"], "tools": req["tools"]}), None));
        }
        for (s, d) in [("temperature", "temperature"), ("topP", "top_p"), ("topK", "top_k")] {
            if !cfg[s].is_null() {
                result[d] = cfg[s].clone();
            }
        }
        if truthy(&cfg["thinkingConfig"]) {
            if let Some(e) = budget_to_effort(cfg["thinkingConfig"]["thinkingBudget"].as_f64().unwrap_or(0.0)) {
                result["reasoning_effort"] = json!(e);
            }
        }
    }
    if truthy(&req["systemInstruction"]) {
        let t = extract_gemini_text(&req["systemInstruction"]);
        if !t.is_empty() {
            result["messages"].as_array_mut().unwrap().push(json!({"role": "system", "content": t}));
        }
    }
    for c in req["contents"].as_array().into_iter().flatten() {
        result["messages"].as_array_mut().unwrap().extend(convert_antigravity_content(c));
    }
    if let Some(tools) = req["tools"].as_array() {
        let mut out = vec![];
        for t in tools {
            for f in t["functionDeclarations"].as_array().into_iter().flatten() {
                let p = normalize_schema_types(&f["parameters"]);
                out.push(json!({"type": "function", "function": {"name": f["name"], "description": if truthy(&f["description"]) { f["description"].clone() } else { json!("") }, "parameters": if truthy(&p) { p } else { json!({"type": "object", "properties": {}}) }}}));
            }
        }
        result["tools"] = Value::Array(out);
    }
    result
}

fn convert_antigravity_content(content: &Value) -> Vec<Value> {
    let role = match content["role"].as_str() {
        Some("model") => json!("assistant"),
        Some("user") => json!("user"),
        _ => content["role"].clone(),
    };
    let Some(parts) = content["parts"].as_array() else { return vec![] };
    let mut text_parts = vec![];
    let mut tool_calls = vec![];
    let mut tool_results = vec![];
    let mut reasoning = String::new();
    for part in parts {
        if part["thought"] == json!(true) && truthy(&part["text"]) {
            reasoning.push_str(part["text"].as_str().unwrap_or(""));
            continue;
        }
        if truthy(&part["thoughtSignature"]) && crate::jsv::has(part, "text") {
            if truthy(&part["text"]) {
                text_parts.push(json!({"type": "text", "text": part["text"]}));
            }
            continue;
        }
        if crate::jsv::has(part, "text") && part["text"] != "" {
            text_parts.push(json!({"type": "text", "text": part["text"]}));
        }
        if truthy(&part["inlineData"]) {
            text_parts.push(json!({"type": "image_url", "image_url": {"url": encode_data_uri(&js_string(&part["inlineData"]["mimeType"]), &js_string(&part["inlineData"]["data"]))}}));
        }
        if truthy(&part["functionCall"]) {
            let fc = &part["functionCall"];
            tool_calls.push(json!({"id": if truthy(&fc["id"]) { fc["id"].clone() } else { json!(format!("call_{}", js_string(&fc["name"]))) }, "type": "function", "function": {"name": fc["name"], "arguments": if truthy(&fc["args"]) { fc["args"].to_string() } else { "{}".into() }}}));
        }
        if truthy(&part["functionResponse"]) {
            let fr = &part["functionResponse"];
            let payload = if truthy(&fr["response"]["result"]) { fr["response"]["result"].clone() } else if truthy(&fr["response"]) { fr["response"].clone() } else { json!({}) };
            tool_results.push(json!({"role": "tool", "tool_call_id": if truthy(&fr["id"]) { fr["id"].clone() } else { json!(format!("call_{}", js_string(&fr["name"]))) }, "content": payload.to_string()}));
        }
    }
    let build_assistant = |text_parts: Vec<Value>, reasoning: &str, tool_calls: Vec<Value>| {
        let mut m = json!({"role": "assistant"});
        if !text_parts.is_empty() {
            m["content"] = collapse_text_parts(text_parts);
        }
        if !reasoning.is_empty() {
            m["reasoning_content"] = json!(reasoning);
        }
        if !tool_calls.is_empty() {
            m["tool_calls"] = Value::Array(tool_calls);
        }
        m
    };
    if !tool_results.is_empty() {
        if !tool_calls.is_empty() || !text_parts.is_empty() || !reasoning.is_empty() {
            tool_results.push(build_assistant(text_parts, &reasoning, tool_calls));
        }
        return tool_results;
    }
    if !tool_calls.is_empty() {
        return vec![build_assistant(text_parts, &reasoning, tool_calls)];
    }
    if !text_parts.is_empty() || !reasoning.is_empty() {
        let mut m = json!({"role": role});
        if !text_parts.is_empty() {
            m["content"] = collapse_text_parts(text_parts);
        }
        if !reasoning.is_empty() {
            m["reasoning_content"] = json!(reasoning);
        }
        return vec![m];
    }
    vec![]
}

// ---------------------------------------------------------------------------
// OpenAI Responses API ⇄ Chat Completions
// ---------------------------------------------------------------------------

pub fn normalize_responses_input(input: &Value) -> Option<Vec<Value>> {
    match input {
        Value::String(s) => {
            let text = if s.trim().is_empty() { "...".to_string() } else { s.clone() };
            Some(vec![json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})])
        }
        Value::Array(a) => {
            if a.is_empty() {
                return Some(vec![json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]})]);
            }
            Some(a.clone())
        }
        _ => None,
    }
}

static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

pub fn clamp_responses_call_id(id: &Value) -> String {
    match id.as_str().filter(|s| !s.is_empty()) {
        None => format!("call_{}_{}", now_ms(), CALL_SEQ.fetch_add(1, Ordering::Relaxed) + 1),
        Some(s) => s.chars().take(64).collect(),
    }
}

pub fn coerce_responses_arguments(v: &Value) -> String {
    match v {
        Value::Null => "{}".into(),
        Value::String(s) if s.is_empty() => "{}".into(),
        Value::String(s) => {
            if serde_json::from_str::<Value>(s).is_ok() {
                s.clone()
            } else {
                "{}".into()
            }
        }
        other => other.to_string(),
    }
}

pub fn coerce_responses_output(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Array(a) => a
            .iter()
            .map(|c| if !c["text"].is_null() { js_string(&c["text"]) } else { c.to_string() })
            .collect(),
        other => other.to_string(),
    }
}

fn normalize_tool_parameters(p: &Value) -> Value {
    if !truthy(p) {
        return json!({"type": "object", "properties": {}});
    }
    if p["type"] == "object" && !truthy(&p["properties"]) {
        let mut q = p.clone();
        q["properties"] = json!({});
        return q;
    }
    p.clone()
}

/// openaiResponsesToOpenAIRequest → (body, customToolNames)
pub fn responses_to_openai(body: &Value) -> (Value, Vec<String>) {
    if !truthy(&body["input"]) {
        return (body.clone(), vec![]);
    }
    let mut result = body.clone();
    let mut messages: Vec<Value> = vec![];
    if truthy(&body["instructions"]) {
        messages.push(json!({"role": "system", "content": body["instructions"]}));
    }
    let Some(items) = normalize_responses_input(&body["input"]) else { return (body.clone(), vec![]) };
    let mut current: Option<Value> = None;
    let mut pending_reasoning = String::new();
    let mut pending_encrypted = String::new();
    let mut additional_tools = vec![];
    let mut custom: Vec<String> = vec![];
    let extract_reasoning = |item: &Value| -> String {
        if let Some(s) = item["summary"].as_array() {
            let t = s.iter().map(|x| x["text"].as_str().unwrap_or("").to_string()).filter(|x| !x.is_empty()).collect::<Vec<_>>().join("\n");
            if !t.is_empty() {
                return t;
            }
        }
        if let Some(c) = item["content"].as_array() {
            let t = c.iter().map(|x| x["text"].as_str().unwrap_or("").to_string()).filter(|x| !x.is_empty()).collect::<Vec<_>>().join("\n");
            if !t.is_empty() {
                return t;
            }
        }
        String::new()
    };
    for item in &items {
        let item_type = if truthy(&item["type"]) {
            js_string(&item["type"])
        } else if truthy(&item["role"]) {
            "message".into()
        } else {
            String::new()
        };
        match item_type.as_str() {
            "message" => {
                if let Some(c) = current.take() {
                    messages.push(c);
                }
                let content = match &item["content"] {
                    Value::Array(a) => Value::Array(
                        a.iter()
                            .map(|c| match c["type"].as_str() {
                                Some("input_text") | Some("output_text") => json!({"type": "text", "text": c["text"]}),
                                Some("input_image") => {
                                    let url = if truthy(&c["image_url"]) { c["image_url"].clone() } else if truthy(&c["file_id"]) { c["file_id"].clone() } else { json!("") };
                                    json!({"type": "image_url", "image_url": {"url": url, "detail": if truthy(&c["detail"]) { c["detail"].clone() } else { json!("auto") }}})
                                }
                                _ => c.clone(),
                            })
                            .collect(),
                    ),
                    other => other.clone(),
                };
                let mut msg = json!({"role": item["role"], "content": content});
                if item["role"] == "assistant" {
                    if !pending_reasoning.is_empty() {
                        msg["reasoning_content"] = json!(pending_reasoning);
                    }
                    if !pending_encrypted.is_empty() {
                        msg["encrypted_content"] = json!(pending_encrypted);
                    }
                }
                pending_reasoning.clear();
                pending_encrypted.clear();
                messages.push(msg);
            }
            "function_call" | "custom_tool_call" => {
                if current.is_none() {
                    let mut m = json!({"role": "assistant", "content": null, "tool_calls": []});
                    if !pending_reasoning.is_empty() {
                        m["reasoning_content"] = json!(pending_reasoning);
                    }
                    if !pending_encrypted.is_empty() {
                        m["encrypted_content"] = json!(pending_encrypted);
                    }
                    pending_reasoning.clear();
                    pending_encrypted.clear();
                    current = Some(m);
                }
                let Some(name) = item["name"].as_str().filter(|n| !n.trim().is_empty()) else { continue };
                if item_type == "custom_tool_call" && !custom.iter().any(|c| c == name) {
                    custom.push(name.to_string());
                }
                let input = if item_type == "custom_tool_call" {
                    let s = item["input"].as_str().map(str::to_owned).unwrap_or_else(|| if item["input"].is_null() { "\"\"".into() } else { item["input"].to_string() });
                    json!({"input": s})
                } else {
                    item["arguments"].clone()
                };
                let args = match &input {
                    Value::String(s) => s.clone(),
                    Value::Null => "{}".into(),
                    other => other.to_string(),
                };
                current.as_mut().unwrap()["tool_calls"].as_array_mut().unwrap().push(json!({"id": item["call_id"], "type": "function", "function": {"name": name, "arguments": args}}));
            }
            "function_call_output" | "custom_tool_call_output" => {
                if let Some(c) = current.take() {
                    messages.push(c);
                }
                let content = item["output"].as_str().map(str::to_owned).unwrap_or_else(|| item["output"].to_string());
                messages.push(json!({"role": "tool", "tool_call_id": item["call_id"], "content": content}));
            }
            "additional_tools" => {
                if let Some(t) = item["tools"].as_array() {
                    additional_tools.extend(t.iter().cloned());
                }
            }
            "reasoning" => {
                let t = extract_reasoning(item);
                if !t.is_empty() {
                    pending_reasoning = if pending_reasoning.is_empty() { t } else { format!("{pending_reasoning}\n{t}") };
                }
                if let Some(e) = item["encrypted_content"].as_str().filter(|s| !s.is_empty()) {
                    pending_encrypted = e.to_string();
                }
            }
            _ => {}
        }
    }
    if let Some(c) = current.take() {
        messages.push(c);
    }
    result["messages"] = Value::Array(messages);
    let mut all_tools: Vec<Value> = body["tools"].as_array().cloned().unwrap_or_default();
    all_tools.extend(additional_tools);
    if !all_tools.is_empty() {
        let tools: Vec<Value> = all_tools
            .into_iter()
            .filter_map(|tool| {
                if truthy(&tool["function"]) {
                    return Some(tool);
                }
                let name = tool["name"].as_str().filter(|n| !n.trim().is_empty())?.to_string();
                if tool["type"] == "custom" {
                    if !custom.contains(&name) {
                        custom.push(name.clone());
                    }
                    let hint = [&tool["format"]["syntax"], &tool["format"]["definition"]]
                        .iter()
                        .filter(|v| truthy(v))
                        .map(|v| js_string(v))
                        .collect::<Vec<_>>()
                        .join("\n");
                    let desc = [desc_string(&tool["description"]), hint].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
                    return Some(json!({"type": "function", "function": {"name": name, "description": desc, "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string", "description": "Raw freeform input for this custom tool"}},
                        "required": ["input"],
                        "additionalProperties": false
                    }}}));
                }
                let mut f = json!({"name": name, "description": desc_string(&tool["description"]), "parameters": normalize_tool_parameters(&tool["parameters"])});
                if !tool["strict"].is_null() {
                    f["strict"] = tool["strict"].clone();
                }
                Some(json!({"type": "function", "function": f}))
            })
            .collect();
        result["tools"] = Value::Array(tools);
    }
    if !result["max_output_tokens"].is_null() || crate::jsv::has(&result, "max_output_tokens") {
        if result["max_tokens"].is_null() {
            result["max_tokens"] = result["max_output_tokens"].clone();
        }
        del(&mut result, "max_output_tokens");
    }
    for k in ["input", "instructions", "include", "prompt_cache_key", "store"] {
        del(&mut result, k);
    }
    if let Some(e) = result["reasoning"]["effort"].as_str().map(str::to_owned) {
        result["reasoning_effort"] = json!(e);
    }
    del(&mut result, "reasoning");
    del(&mut result, "client_metadata");
    (result, custom)
}

fn extract_instructions_text(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .map(|x| x["text"].as_str().or_else(|| x["content"].as_str()).unwrap_or("").to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn build_reasoning_input_item(msg: &Value) -> Option<Value> {
    let encrypted = msg["encrypted_content"]
        .as_str()
        .filter(|s| !s.is_empty())
        .or_else(|| msg["reasoning_encrypted_content"].as_str().filter(|s| !s.is_empty()))
        .or_else(|| msg["reasoning"]["encrypted_content"].as_str().filter(|s| !s.is_empty()))
        .unwrap_or("");
    let summary = if let Some(s) = msg["reasoning_content"].as_str().filter(|s| !s.trim().is_empty()) {
        s.to_string()
    } else if let Some(s) = msg["reasoning"].as_str().filter(|s| !s.trim().is_empty()) {
        s.to_string()
    } else if let Some(d) = msg["reasoning_details"].as_array() {
        d.iter()
            .map(|x| x["text"].as_str().or_else(|| x["content"].as_str()).unwrap_or("").to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        String::new()
    };
    if encrypted.is_empty() && summary.is_empty() {
        return None;
    }
    let mut item = json!({"type": "reasoning"});
    if !summary.is_empty() {
        item["summary"] = json!([{"type": "summary_text", "text": summary}]);
    }
    if !encrypted.is_empty() {
        item["encrypted_content"] = json!(encrypted);
    }
    Some(item)
}

pub fn openai_to_responses(model: &str, body: &Value) -> Value {
    if truthy(&body["input"]) {
        let mut out = body.clone();
        out["model"] = json!(model);
        out["stream"] = json!(true);
        if out["max_output_tokens"].is_null() {
            if !out["max_completion_tokens"].is_null() {
                out["max_output_tokens"] = out["max_completion_tokens"].clone();
            } else if !out["max_tokens"].is_null() {
                out["max_output_tokens"] = out["max_tokens"].clone();
            }
        }
        del(&mut out, "max_tokens");
        del(&mut out, "max_completion_tokens");
        return out;
    }
    let mut result = json!({"model": model, "input": [], "stream": true, "store": false});
    let mut has_system = false;
    let mut input = vec![];
    for msg in body["messages"].as_array().into_iter().flatten() {
        let role = msg["role"].as_str().unwrap_or("");
        if role == "system" || role == "developer" {
            if !has_system {
                result["instructions"] = json!(extract_instructions_text(&msg["content"]));
                has_system = true;
            }
            continue;
        }
        if role == "user" || role == "assistant" {
            if role == "assistant" {
                if let Some(r) = build_reasoning_input_item(msg) {
                    input.push(r);
                }
            }
            let ct = if role == "user" { "input_text" } else { "output_text" };
            let content: Vec<Value> = match &msg["content"] {
                Value::String(s) => vec![json!({"type": ct, "text": s})],
                Value::Array(a) => a
                    .iter()
                    .map(|c| {
                        let ty = c["type"].as_str().unwrap_or("");
                        if ty == "text" {
                            return json!({"type": ct, "text": c["text"]});
                        }
                        if ty == "image_url" {
                            let url = if c["image_url"].is_string() { c["image_url"].clone() } else { c["image_url"]["url"].clone() };
                            let detail = if truthy(&c["image_url"]["detail"]) { c["image_url"]["detail"].clone() } else { json!("auto") };
                            return json!({"type": "input_image", "image_url": url, "detail": detail});
                        }
                        if ty == "input_image" {
                            return c.clone();
                        }
                        let text = if truthy(&c["text"]) {
                            c["text"].clone()
                        } else if truthy(&c["content"]) {
                            c["content"].clone()
                        } else {
                            json!(c.to_string())
                        };
                        json!({"type": ct, "text": text.as_str().map(str::to_owned).unwrap_or_else(|| text.to_string())})
                    })
                    .collect(),
                _ => vec![],
            };
            if !content.is_empty() {
                input.push(json!({"type": "message", "role": role, "content": content}));
            }
        }
        if role == "assistant" {
            for tc in msg["tool_calls"].as_array().into_iter().flatten() {
                let name = tc["function"]["name"].as_str().map(|s| s.trim()).unwrap_or("");
                if name.is_empty() {
                    continue;
                }
                input.push(json!({
                    "type": "function_call",
                    "call_id": clamp_responses_call_id(&tc["id"]),
                    "name": name.chars().take(128).collect::<String>(),
                    "arguments": coerce_responses_arguments(&tc["function"]["arguments"]),
                }));
            }
        }
        if role == "tool" {
            input.push(json!({"type": "function_call_output", "call_id": clamp_responses_call_id(&msg["tool_call_id"]), "output": coerce_responses_output(&msg["content"])}));
        }
    }
    result["input"] = Value::Array(input);
    if !has_system {
        result["instructions"] = json!("");
    }
    if let Some(tools) = body["tools"].as_array() {
        let out: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                if t["type"] == "function" {
                    let name = t["function"]["name"].as_str().map(|s| s.trim()).unwrap_or("");
                    if name.is_empty() {
                        return None;
                    }
                    let mut o = json!({"type": "function", "name": name.chars().take(128).collect::<String>(), "description": desc_string(&t["function"]["description"]), "parameters": normalize_tool_parameters(&t["function"]["parameters"])});
                    if !t["function"]["strict"].is_null() {
                        o["strict"] = t["function"]["strict"].clone();
                    }
                    return Some(o);
                }
                Some(t.clone())
            })
            .collect();
        result["tools"] = Value::Array(out);
    }
    if !body["temperature"].is_null() {
        result["temperature"] = body["temperature"].clone();
    }
    if !body["max_output_tokens"].is_null() {
        result["max_output_tokens"] = body["max_output_tokens"].clone();
    } else if !body["max_completion_tokens"].is_null() {
        result["max_output_tokens"] = body["max_completion_tokens"].clone();
    } else if !body["max_tokens"].is_null() {
        result["max_output_tokens"] = body["max_tokens"].clone();
    }
    for k in ["top_p", "reasoning", "service_tier", "prompt_cache_key"] {
        if !body[k].is_null() {
            result[k] = body[k].clone();
        }
    }
    if !body["reasoning_effort"].is_null() {
        result["reasoning"] = json!({"effort": body["reasoning_effort"], "summary": "auto"});
    }
    result
}

// ---------------------------------------------------------------------------
// OpenAI → Ollama
// ---------------------------------------------------------------------------

fn ollama_text(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter(|b| b["type"] == "text" && truthy(&b["text"])).map(|b| js_string(&b["text"])).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

pub fn openai_to_ollama(model: &str, body: &Value, stream: bool) -> Value {
    let mut result = json!({"model": model, "messages": body["messages"].clone(), "stream": stream});
    if let Some(msgs) = body["messages"].as_array() {
        let mut map: HashMap<String, String> = HashMap::new();
        for m in msgs {
            if m["role"] == "assistant" {
                for tc in m["tool_calls"].as_array().into_iter().flatten() {
                    if truthy(&tc["id"]) && truthy(&tc["function"]["name"]) {
                        map.insert(js_string(&tc["id"]), js_string(&tc["function"]["name"]));
                    }
                }
            }
        }
        let mut out = vec![];
        for m in msgs {
            if m["role"] == "tool" {
                let r = ollama_text(&m["content"]);
                if r.is_empty() {
                    continue;
                }
                let name = map.get(&js_string(&m["tool_call_id"])).cloned().or_else(|| m["name"].as_str().map(str::to_owned)).unwrap_or_else(|| "unknown_tool".into());
                out.push(json!({"role": "tool", "tool_name": name, "content": r}));
                continue;
            }
            if m["role"] == "assistant" && truthy(&m["tool_calls"]) {
                let calls: Vec<Value> = m["tool_calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|tc| {
                        let a = &tc["function"]["arguments"];
                        let args = if a.is_string() {
                            safe_parse(&json!(if truthy(a) { a.as_str().unwrap() } else { "{}" }), json!({}))
                        } else if truthy(a) {
                            a.clone()
                        } else {
                            json!({})
                        };
                        json!({"type": "function", "function": {"index": if truthy(&tc["index"]) { tc["index"].clone() } else { json!(0) }, "name": tc["function"]["name"].as_str().unwrap_or(""), "arguments": args}})
                    })
                    .collect();
                out.push(json!({"role": "assistant", "content": ollama_text(&m["content"]), "tool_calls": calls}));
                continue;
            }
            let content = ollama_text(&m["content"]);
            let mut images = vec![];
            for b in m["content"].as_array().into_iter().flatten() {
                if b["type"] != "image_url" {
                    continue;
                }
                let url = if b["image_url"].is_string() { b["image_url"].clone() } else { b["image_url"]["url"].clone() };
                if let Some((_, data)) = parse_data_uri(&url) {
                    images.push(json!(data));
                }
            }
            if content.is_empty() && m["role"] != "assistant" {
                continue;
            }
            let mut o = json!({"role": m["role"], "content": content});
            if !images.is_empty() {
                o["images"] = Value::Array(images);
            }
            out.push(o);
        }
        result["messages"] = Value::Array(out);
    }
    for (s, d) in [("temperature", "temperature"), ("max_tokens", "num_predict"), ("top_p", "top_p")] {
        if !body[s].is_null() {
            result["options"][d] = body[s].clone();
        }
    }
    if body["tools"].is_array() {
        result["tools"] = body["tools"].clone();
    }
    if truthy(&body["tool_choice"]) {
        result["tool_choice"] = body["tool_choice"].clone();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_openai_claude() {
        let claude = json!({
            "model": "x", "max_tokens": 100,
            "system": [{"type": "text", "text": "be nice"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "calling"}, {"type": "tool_use", "id": "t1", "name": "ls", "input": {"path": "/"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a b"}, {"type": "text", "text": "thanks"}]}
            ],
            "tools": [{"name": "ls", "description": "list", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"}
        });
        let oa = claude_to_openai("m", &claude, false);
        assert_eq!(oa["messages"][0]["role"], "system");
        assert_eq!(oa["messages"][2]["tool_calls"][0]["function"]["name"], "ls");
        assert_eq!(oa["messages"][3]["role"], "tool");
        assert_eq!(oa["messages"][4]["content"], "thanks");
        assert_eq!(oa["tool_choice"], "required");
        let back = openai_to_claude("claude-sonnet-4-5", &oa, false);
        assert_eq!(back["system"][1]["text"], "be nice");
        assert_eq!(back["messages"][1]["content"][1]["type"], "tool_use");
        assert_eq!(back["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(back["tool_choice"]["type"], "any");
    }

    #[test]
    fn responses_roundtrip() {
        let r = json!({"model": "m", "instructions": "sys", "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"}
        ], "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}], "max_output_tokens": 50});
        let (oa, _) = responses_to_openai(&r);
        assert_eq!(oa["messages"][0]["role"], "system");
        assert_eq!(oa["messages"][2]["tool_calls"][0]["id"], "c1");
        assert_eq!(oa["max_tokens"], 50);
        let back = openai_to_responses("m", &oa);
        assert_eq!(back["instructions"], "sys");
        assert_eq!(back["input"][1]["type"], "function_call");
        assert_eq!(back["input"][2]["output"], "ok");
    }
}
