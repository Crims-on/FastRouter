//! Request translation. The OpenAI chat-completions shape is the hub format:
//! inbound requests are converted to it, then converted to the upstream format.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::catalog::Format;

/// Flattens OpenAI/Claude style content (string or array of parts) into plain text.
pub fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p {
                Value::String(s) => Some(s.clone()),
                _ => match p.get("type").and_then(Value::as_str) {
                    Some("text") | Some("input_text") | Some("output_text") | None => {
                        p.get("text").and_then(Value::as_str).map(str::to_owned)
                    }
                    Some("tool_result") => {
                        Some(content_text(p.get("content").unwrap_or(&Value::Null)))
                    }
                    _ => None,
                },
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Splits a `data:<mime>;base64,<data>` URL.
fn parse_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let mime = meta.strip_suffix(";base64")?;
    Some((mime.to_string(), data.to_string()))
}

fn copy_fields(from: &Value, to: &mut Map<String, Value>, fields: &[(&str, &str)]) {
    for (src, dst) in fields {
        if let Some(v) = from.get(*src)
            && !v.is_null()
        {
            to.insert((*dst).to_string(), v.clone());
        }
    }
}

fn stop_list(v: Option<&Value>) -> Option<Value> {
    match v? {
        Value::String(s) => Some(json!([s])),
        Value::Array(a) if !a.is_empty() => Some(Value::Array(a.clone())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Claude -> OpenAI
// ---------------------------------------------------------------------------

pub fn claude_to_openai(req: &Value) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    if let Some(system) = req.get("system") {
        let text = content_text(system);
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }

    for msg in req
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = msg.get("content").unwrap_or(&Value::Null);
        let Some(blocks) = content.as_array() else {
            messages.push(json!({"role": role, "content": content_text(content)}));
            continue;
        };

        if role == "assistant" {
            let mut text = String::new();
            let mut tool_calls = Vec::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
                    Some("tool_use") => tool_calls.push(json!({
                        "id": b.get("id").cloned().unwrap_or(Value::Null),
                        "type": "function",
                        "function": {
                            "name": b.get("name").cloned().unwrap_or(Value::Null),
                            "arguments": b.get("input").map(|i| i.to_string()).unwrap_or_else(|| "{}".into()),
                        }
                    })),
                    _ => {}
                }
            }
            let mut m = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { Value::String(text) }});
            if !tool_calls.is_empty() {
                m["tool_calls"] = Value::Array(tool_calls);
            }
            messages.push(m);
            continue;
        }

        // user: tool_results become separate `tool` messages, emitted first
        let mut parts = Vec::new();
        for b in blocks {
            match b.get("type").and_then(Value::as_str) {
                Some("tool_result") => {
                    let mut text = content_text(b.get("content").unwrap_or(&Value::Null));
                    if b.get("is_error").and_then(Value::as_bool) == Some(true)
                        && !text.starts_with("Error")
                    {
                        text = format!("Error: {text}");
                    }
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(Value::Null),
                        "content": text,
                    }));
                }
                Some("text") => parts.push(
                    json!({"type": "text", "text": b.get("text").cloned().unwrap_or_default()}),
                ),
                Some("image") => {
                    let src = b.get("source").cloned().unwrap_or_default();
                    let url = match src.get("type").and_then(Value::as_str) {
                        Some("base64") => format!(
                            "data:{};base64,{}",
                            src.get("media_type")
                                .and_then(Value::as_str)
                                .unwrap_or("image/png"),
                            src.get("data").and_then(Value::as_str).unwrap_or("")
                        ),
                        _ => src
                            .get("url")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    };
                    parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                }
                _ => {}
            }
        }
        if !parts.is_empty() {
            let all_text = parts.iter().all(|p| p["type"] == "text");
            let content = if all_text {
                Value::String(
                    parts
                        .iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            } else {
                Value::Array(parts)
            };
            messages.push(json!({"role": "user", "content": content}));
        }
    }

    let mut out = Map::new();
    out.insert(
        "model".into(),
        req.get("model").cloned().unwrap_or_default(),
    );
    out.insert("messages".into(), Value::Array(messages));
    copy_fields(
        req,
        &mut out,
        &[
            ("max_tokens", "max_tokens"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("stream", "stream"),
        ],
    );
    if let Some(stop) = stop_list(req.get("stop_sequences")) {
        out.insert("stop".into(), stop);
    }

    let tools: Vec<Value> = req
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|t| t.get("input_schema").is_some())
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.get("name").cloned().unwrap_or_default(),
                    "description": t.get("description").cloned().unwrap_or(Value::String(String::new())),
                    "parameters": t.get("input_schema").cloned().unwrap_or(json!({"type": "object"})),
                }
            })
        })
        .collect();
    if !tools.is_empty() {
        out.insert("tools".into(), Value::Array(tools));
        if let Some(tc) = req.get("tool_choice") {
            let mapped = match tc.get("type").and_then(Value::as_str) {
                Some("any") => json!("required"),
                Some("none") => json!("none"),
                Some("tool") => {
                    json!({"type": "function", "function": {"name": tc.get("name").cloned().unwrap_or_default()}})
                }
                _ => json!("auto"),
            };
            out.insert("tool_choice".into(), mapped);
        }
    }
    Value::Object(out)
}

// ---------------------------------------------------------------------------
// OpenAI -> Claude
// ---------------------------------------------------------------------------

fn openai_parts_to_claude(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) if s.is_empty() => vec![],
        Value::String(s) => vec![json!({"type": "text", "text": s})],
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") | Some("input_text") => {
                    let t = p.get("text").and_then(Value::as_str).unwrap_or("");
                    (!t.is_empty()).then(|| json!({"type": "text", "text": t}))
                }
                Some("image_url") => {
                    let url = p
                        .get("image_url")
                        .and_then(|i| i.get("url").or(Some(i)))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    Some(match parse_data_url(url) {
                        Some((mime, data)) => {
                            json!({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
                        }
                        None => json!({"type": "image", "source": {"type": "url", "url": url}}),
                    })
                }
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

fn push_merged(msgs: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = msgs.last_mut()
        && last["role"] == role
        && let Some(arr) = last["content"].as_array_mut()
    {
        arr.extend(blocks);
        return;
    }
    msgs.push(json!({"role": role, "content": blocks}));
}

pub fn openai_to_claude(req: &Value, model: &str) -> Value {
    let mut system = Vec::new();
    let mut msgs: Vec<Value> = Vec::new();

    for m in req
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let content = m.get("content").unwrap_or(&Value::Null);
        match m.get("role").and_then(Value::as_str).unwrap_or("user") {
            "system" | "developer" => {
                let t = content_text(content);
                if !t.is_empty() {
                    system.push(t);
                }
            }
            "assistant" => {
                let mut blocks = openai_parts_to_claude(content);
                for tc in m
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let args = tc["function"]["arguments"].as_str().unwrap_or("{}");
                    let input: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.get("id").cloned().unwrap_or_default(),
                        "name": tc["function"]["name"].clone(),
                        "input": input,
                    }));
                }
                push_merged(&mut msgs, "assistant", blocks);
            }
            "tool" | "function" => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id").cloned().unwrap_or_default(),
                    "content": content_text(content),
                });
                push_merged(&mut msgs, "user", vec![block]);
            }
            _ => push_merged(&mut msgs, "user", openai_parts_to_claude(content)),
        }
    }

    // Claude requires the conversation to start with a user turn.
    if msgs.first().map(|m| m["role"] != "user").unwrap_or(true) {
        msgs.insert(
            0,
            json!({"role": "user", "content": [{"type": "text", "text": "."}]}),
        );
    }

    let mut out = Map::new();
    out.insert("model".into(), json!(model));
    out.insert("messages".into(), Value::Array(msgs));
    if !system.is_empty() {
        out.insert("system".into(), json!(system.join("\n\n")));
    }
    let mut max_tokens = req
        .get("max_completion_tokens")
        .or_else(|| req.get("max_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(8192);
    copy_fields(
        req,
        &mut out,
        &[
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("stream", "stream"),
        ],
    );
    if let Some(stop) = stop_list(req.get("stop")) {
        out.insert("stop_sequences".into(), stop);
    }

    if let Some(effort) = req.get("reasoning_effort").and_then(Value::as_str) {
        let budget = match effort {
            "minimal" | "low" => 1024,
            "high" => 24576,
            _ => 8192,
        };
        if max_tokens <= budget {
            max_tokens = budget + 8192;
        }
        out.insert(
            "thinking".into(),
            json!({"type": "enabled", "budget_tokens": budget}),
        );
        out.remove("temperature");
        out.remove("top_p");
    }
    out.insert("max_tokens".into(), json!(max_tokens));

    let tools: Vec<Value> = req
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|t| {
            let f = t.get("function")?;
            Some(json!({
                "name": f.get("name")?,
                "description": f.get("description").cloned().unwrap_or(json!("")),
                "input_schema": f.get("parameters").cloned().unwrap_or(json!({"type": "object", "properties": {}})),
            }))
        })
        .collect();
    if !tools.is_empty() {
        out.insert("tools".into(), Value::Array(tools));
        if let Some(tc) = req.get("tool_choice") {
            let mapped = match tc {
                Value::String(s) if s == "required" => Some(json!({"type": "any"})),
                Value::String(s) if s == "none" => Some(json!({"type": "none"})),
                Value::String(_) => Some(json!({"type": "auto"})),
                Value::Object(_) => tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .map(|n| json!({"type": "tool", "name": n})),
                _ => None,
            };
            if let Some(m) = mapped {
                out.insert("tool_choice".into(), m);
            }
        }
    }
    Value::Object(out)
}

// ---------------------------------------------------------------------------
// OpenAI -> Gemini
// ---------------------------------------------------------------------------

/// Strips JSON-schema keywords the Gemini function-declaration schema rejects.
pub fn clean_gemini_schema(schema: &Value) -> Value {
    const DROP: &[&str] = &[
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "$comment",
        "definitions",
        "additionalProperties",
        "examples",
        "default",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "patternProperties",
        "propertyNames",
        "if",
        "then",
        "else",
        "not",
        "unevaluatedProperties",
        "dependentRequired",
        "dependentSchemas",
        "contentEncoding",
        "contentMediaType",
        "title",
        "strict",
    ];
    match schema {
        Value::Object(obj) => {
            let mut out = Map::new();
            for (k, v) in obj {
                if DROP.contains(&k.as_str()) {
                    continue;
                }
                match k.as_str() {
                    "const" => {
                        out.insert("enum".into(), json!([v]));
                    }
                    "format" => {
                        if matches!(v.as_str(), Some("enum") | Some("date-time")) {
                            out.insert(k.clone(), v.clone());
                        }
                    }
                    "type" => {
                        if let Value::Array(types) = v {
                            let first = types.iter().find(|t| t.as_str() != Some("null")).cloned();
                            out.insert("type".into(), first.unwrap_or(json!("string")));
                            if types.iter().any(|t| t.as_str() == Some("null")) {
                                out.insert("nullable".into(), json!(true));
                            }
                        } else {
                            out.insert(k.clone(), v.clone());
                        }
                    }
                    "properties" => {
                        let props = v
                            .as_object()
                            .map(|p| {
                                p.iter()
                                    .map(|(n, s)| (n.clone(), clean_gemini_schema(s)))
                                    .collect::<Map<_, _>>()
                            })
                            .unwrap_or_default();
                        out.insert(k.clone(), Value::Object(props));
                    }
                    _ => {
                        out.insert(k.clone(), clean_gemini_schema(v));
                    }
                }
            }
            // `required` may only reference declared properties.
            if let (Some(Value::Array(req)), Some(Value::Object(props))) =
                (out.get("required"), out.get("properties"))
            {
                let filtered: Vec<Value> = req
                    .iter()
                    .filter(|r| r.as_str().map(|s| props.contains_key(s)).unwrap_or(false))
                    .cloned()
                    .collect();
                out.insert("required".into(), Value::Array(filtered));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(clean_gemini_schema).collect()),
        other => other.clone(),
    }
}

fn openai_parts_to_gemini(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) if s.is_empty() => vec![],
        Value::String(s) => vec![json!({"text": s})],
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") | Some("input_text") => {
                    Some(json!({"text": p.get("text").cloned().unwrap_or_default()}))
                }
                Some("image_url") => {
                    let url = p
                        .get("image_url")
                        .and_then(|i| i.get("url"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    parse_data_url(url)
                        .map(|(mime, data)| json!({"inlineData": {"mimeType": mime, "data": data}}))
                }
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

pub fn openai_to_gemini(req: &Value) -> Value {
    let mut system = Vec::new();
    let mut contents: Vec<Value> = Vec::new();
    let mut tool_names: HashMap<String, String> = HashMap::new();

    let push = |contents: &mut Vec<Value>, role: &str, parts: Vec<Value>| {
        if parts.is_empty() {
            return;
        }
        if let Some(last) = contents.last_mut()
            && last["role"] == role
            && let Some(arr) = last["parts"].as_array_mut()
        {
            arr.extend(parts);
            return;
        }
        contents.push(json!({"role": role, "parts": parts}));
    };

    for m in req
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let content = m.get("content").unwrap_or(&Value::Null);
        match m.get("role").and_then(Value::as_str).unwrap_or("user") {
            "system" | "developer" => system.push(content_text(content)),
            "assistant" => {
                let mut parts = openai_parts_to_gemini(content);
                for tc in m
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                    if let Some(id) = tc.get("id").and_then(Value::as_str) {
                        tool_names.insert(id.to_string(), name.clone());
                    }
                    let args: Value =
                        serde_json::from_str(tc["function"]["arguments"].as_str().unwrap_or("{}"))
                            .unwrap_or(json!({}));
                    parts.push(json!({"functionCall": {"name": name, "args": args}}));
                }
                push(&mut contents, "model", parts);
            }
            "tool" | "function" => {
                let id = m.get("tool_call_id").and_then(Value::as_str).unwrap_or("");
                let name = tool_names
                    .get(id)
                    .cloned()
                    .or_else(|| m.get("name").and_then(Value::as_str).map(str::to_owned))
                    .unwrap_or_else(|| "tool".into());
                let text = content_text(content);
                let response = serde_json::from_str::<Value>(&text)
                    .ok()
                    .filter(Value::is_object)
                    .unwrap_or_else(|| json!({"content": text}));
                push(
                    &mut contents,
                    "user",
                    vec![json!({"functionResponse": {"name": name, "response": response}})],
                );
            }
            _ => push(&mut contents, "user", openai_parts_to_gemini(content)),
        }
    }

    let mut out = Map::new();
    out.insert("contents".into(), Value::Array(contents));
    let system: Vec<String> = system.into_iter().filter(|s| !s.is_empty()).collect();
    if !system.is_empty() {
        out.insert(
            "systemInstruction".into(),
            json!({"parts": [{"text": system.join("\n\n")}]}),
        );
    }

    let mut generation = Map::new();
    copy_fields(
        req,
        &mut generation,
        &[("temperature", "temperature"), ("top_p", "topP")],
    );
    if let Some(mt) = req
        .get("max_completion_tokens")
        .or_else(|| req.get("max_tokens"))
    {
        generation.insert("maxOutputTokens".into(), mt.clone());
    }
    if let Some(stop) = stop_list(req.get("stop")) {
        generation.insert("stopSequences".into(), stop);
    }
    if let Some(effort) = req.get("reasoning_effort").and_then(Value::as_str) {
        let budget = match effort {
            "minimal" | "low" => 1024,
            "high" => 24576,
            _ => 8192,
        };
        generation.insert(
            "thinkingConfig".into(),
            json!({"thinkingBudget": budget, "includeThoughts": true}),
        );
    }
    if !generation.is_empty() {
        out.insert("generationConfig".into(), Value::Object(generation));
    }

    let decls: Vec<Value> = req
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|t| {
            let f = t.get("function")?;
            let mut d = json!({"name": f.get("name")?, "description": f.get("description").cloned().unwrap_or(json!(""))});
            if let Some(p) = f.get("parameters") {
                let cleaned = clean_gemini_schema(p);
                let empty = cleaned.get("properties").and_then(Value::as_object).map(|p| p.is_empty()).unwrap_or(true);
                if !empty {
                    d["parameters"] = cleaned;
                }
            }
            Some(d)
        })
        .collect();
    if !decls.is_empty() {
        out.insert("tools".into(), json!([{"functionDeclarations": decls}]));
        if let Some(tc) = req.get("tool_choice") {
            let cfg = match tc {
                Value::String(s) if s == "none" => json!({"mode": "NONE"}),
                Value::String(s) if s == "required" => json!({"mode": "ANY"}),
                Value::Object(_) => {
                    json!({"mode": "ANY", "allowedFunctionNames": [tc["function"]["name"].clone()]})
                }
                _ => json!({"mode": "AUTO"}),
            };
            out.insert("toolConfig".into(), json!({"functionCallingConfig": cfg}));
        }
    }
    Value::Object(out)
}

/// Converts an inbound request into the body the upstream expects.
pub fn to_upstream(inbound: Format, target: Format, req: &Value, model: &str) -> Value {
    if inbound == target {
        let mut body = req.clone();
        body["model"] = json!(model);
        if target == Format::OpenAI && body.get("stream").and_then(Value::as_bool) == Some(true) {
            body["stream_options"] = json!({"include_usage": true});
        }
        return body;
    }
    let hub = match inbound {
        Format::OpenAI => req.clone(),
        Format::Claude => claude_to_openai(req),
        Format::Gemini => req.clone(),
    };
    match target {
        Format::OpenAI => {
            let mut body = hub;
            body["model"] = json!(model);
            if body.get("stream").and_then(Value::as_bool) == Some(true) {
                body["stream_options"] = json!({"include_usage": true});
            }
            body
        }
        Format::Claude => openai_to_claude(&hub, model),
        Format::Gemini => openai_to_gemini(&hub),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_roundtrip_tools() {
        let claude = json!({
            "model": "x",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "be nice"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "calling"},
                    {"type": "tool_use", "id": "t1", "name": "ls", "input": {"path": "/"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "a b"},
                    {"type": "text", "text": "thanks"}
                ]}
            ],
            "tools": [{"name": "ls", "description": "list", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"}
        });
        let oa = claude_to_openai(&claude);
        assert_eq!(oa["messages"][0]["role"], "system");
        assert_eq!(oa["messages"][2]["tool_calls"][0]["function"]["name"], "ls");
        assert_eq!(oa["messages"][3]["role"], "tool");
        assert_eq!(oa["messages"][4]["content"], "thanks");
        assert_eq!(oa["tool_choice"], "required");

        let back = openai_to_claude(&oa, "m");
        assert_eq!(back["system"], "be nice");
        assert_eq!(back["messages"][1]["content"][1]["type"], "tool_use");
        assert_eq!(back["messages"][1]["content"][1]["input"]["path"], "/");
        assert_eq!(back["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(back["messages"][2]["content"][1]["text"], "thanks");
        assert_eq!(back["tool_choice"]["type"], "any");
    }

    #[test]
    fn openai_to_gemini_tools() {
        let oa = json!({
            "model": "g",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}
            ],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {
                "type": "object", "additionalProperties": false,
                "properties": {"a": {"type": ["integer", "null"]}}, "required": ["a", "zzz"]
            }}}]
        });
        let g = openai_to_gemini(&oa);
        assert_eq!(g["systemInstruction"]["parts"][0]["text"], "sys");
        assert_eq!(g["contents"][1]["parts"][0]["functionCall"]["args"]["a"], 1);
        assert_eq!(
            g["contents"][2]["parts"][0]["functionResponse"]["name"],
            "f"
        );
        let params = &g["tools"][0]["functionDeclarations"][0]["parameters"];
        assert!(params.get("additionalProperties").is_none());
        assert_eq!(params["properties"]["a"]["nullable"], true);
        assert_eq!(params["required"], json!(["a"]));
    }
}
