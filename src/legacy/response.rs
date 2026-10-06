//! Non-streaming response translation.

use serde_json::{Value, json};

use crate::catalog::Format;

pub fn new_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

/// (prompt_tokens, completion_tokens) reported by a response body in the given format.
pub fn usage_of(format: Format, v: &Value) -> (i64, i64) {
    let n = |x: &Value| x.as_i64().unwrap_or(0);
    match format {
        Format::OpenAI => {
            let u = &v["usage"];
            (n(&u["prompt_tokens"]), n(&u["completion_tokens"]))
        }
        Format::Claude => {
            let u = &v["usage"];
            (
                n(&u["input_tokens"])
                    + n(&u["cache_read_input_tokens"])
                    + n(&u["cache_creation_input_tokens"]),
                n(&u["output_tokens"]),
            )
        }
        Format::Gemini => {
            let u = &v["usageMetadata"];
            (
                n(&u["promptTokenCount"]),
                n(&u["candidatesTokenCount"]) + n(&u["thoughtsTokenCount"]),
            )
        }
    }
}

pub fn claude_stop_to_openai(reason: &str) -> &'static str {
    match reason {
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        "refusal" => "content_filter",
        _ => "stop",
    }
}

pub fn openai_finish_to_claude(reason: &str) -> &'static str {
    match reason {
        "length" => "max_tokens",
        "tool_calls" | "function_call" => "tool_use",
        "content_filter" => "refusal",
        _ => "end_turn",
    }
}

pub fn gemini_finish_to_openai(reason: &str) -> &'static str {
    match reason {
        "MAX_TOKENS" => "length",
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => "content_filter",
        _ => "stop",
    }
}

fn openai_usage(prompt: i64, completion: i64) -> Value {
    json!({"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion})
}

pub fn claude_to_openai(v: &Value, model: &str) -> Value {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for b in v["content"].as_array().into_iter().flatten() {
        match b["type"].as_str() {
            Some("text") => text.push_str(b["text"].as_str().unwrap_or("")),
            Some("thinking") => reasoning.push_str(b["thinking"].as_str().unwrap_or("")),
            Some("tool_use") => tool_calls.push(json!({
                "id": b["id"],
                "type": "function",
                "function": {"name": b["name"], "arguments": b["input"].to_string()},
            })),
            _ => {}
        }
    }
    let mut message = json!({"role": "assistant", "content": text});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    let (p, c) = usage_of(Format::Claude, v);
    json!({
        "id": v["id"].as_str().map(str::to_owned).unwrap_or_else(|| new_id("chatcmpl-")),
        "object": "chat.completion",
        "created": crate::db::now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": claude_stop_to_openai(v["stop_reason"].as_str().unwrap_or("end_turn")),
        }],
        "usage": openai_usage(p, c),
    })
}

pub fn gemini_to_openai(v: &Value, model: &str) -> Value {
    let cand = &v["candidates"][0];
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for part in cand["content"]["parts"].as_array().into_iter().flatten() {
        if let Some(fc) = part.get("functionCall") {
            tool_calls.push(json!({
                "id": new_id("call_"),
                "type": "function",
                "function": {"name": fc["name"], "arguments": fc.get("args").cloned().unwrap_or(json!({})).to_string()},
            }));
        } else if let Some(t) = part["text"].as_str() {
            if part["thought"].as_bool() == Some(true) {
                reasoning.push_str(t);
            } else {
                text.push_str(t);
            }
        }
    }
    let mut message = json!({"role": "assistant", "content": text});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    let finish = if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
        "tool_calls"
    } else {
        gemini_finish_to_openai(cand["finishReason"].as_str().unwrap_or("STOP"))
    };
    let (p, c) = usage_of(Format::Gemini, v);
    json!({
        "id": v["responseId"].as_str().map(str::to_owned).unwrap_or_else(|| new_id("chatcmpl-")),
        "object": "chat.completion",
        "created": crate::db::now(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": openai_usage(p, c),
    })
}

pub fn openai_to_claude(v: &Value, model: &str) -> Value {
    let choice = &v["choices"][0];
    let msg = &choice["message"];
    let mut content = Vec::new();
    if let Some(r) = msg["reasoning_content"]
        .as_str()
        .or_else(|| msg["reasoning"].as_str())
        && !r.is_empty()
    {
        content.push(json!({"type": "thinking", "thinking": r, "signature": ""}));
    }
    let text = crate::translate::request::content_text(&msg["content"]);
    if !text.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }
    for tc in msg["tool_calls"].as_array().into_iter().flatten() {
        let input: Value =
            serde_json::from_str(tc["function"]["arguments"].as_str().unwrap_or("{}"))
                .unwrap_or(json!({}));
        content.push(json!({
            "type": "tool_use",
            "id": tc["id"].as_str().map(str::to_owned).unwrap_or_else(|| new_id("toolu_")),
            "name": tc["function"]["name"],
            "input": input,
        }));
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }
    let (p, c) = usage_of(Format::OpenAI, v);
    json!({
        "id": new_id("msg_"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": openai_finish_to_claude(choice["finish_reason"].as_str().unwrap_or("stop")),
        "stop_sequence": null,
        "usage": {"input_tokens": p, "output_tokens": c},
    })
}

/// Converts an upstream response body into the inbound client's format.
pub fn to_inbound(inbound: Format, target: Format, v: &Value, model: &str) -> Value {
    if inbound == target {
        return v.clone();
    }
    let hub = match target {
        Format::OpenAI => v.clone(),
        Format::Claude => claude_to_openai(v, model),
        Format::Gemini => gemini_to_openai(v, model),
    };
    match inbound {
        Format::Claude => openai_to_claude(&hub, model),
        _ => hub,
    }
}

/// Error body in the client's expected shape.
pub fn error_body(inbound: Format, status: u16, message: &str) -> Value {
    match inbound {
        Format::Claude => json!({
            "type": "error",
            "error": {"type": claude_error_type(status), "message": message},
        }),
        _ => json!({
            "error": {"message": message, "type": openai_error_type(status), "code": status},
        }),
    }
}

fn claude_error_type(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        529 => "overloaded_error",
        _ => "api_error",
    }
}

fn openai_error_type(status: u16) -> &'static str {
    match status {
        400 | 404 => "invalid_request_error",
        401 | 403 => "authentication_error",
        429 => "rate_limit_exceeded",
        _ => "api_error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_response_to_openai_and_back() {
        let c = json!({
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t", "name": "f", "input": {"x": 1}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let o = claude_to_openai(&c, "m");
        assert_eq!(o["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(o["usage"]["total_tokens"], 15);
        let back = openai_to_claude(&o, "m");
        assert_eq!(back["stop_reason"], "tool_use");
        assert_eq!(back["content"][1]["input"]["x"], 1);
    }

    #[test]
    fn gemini_response() {
        let g = json!({
            "candidates": [{"content": {"parts": [{"text": "think", "thought": true}, {"text": "ans"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "thoughtsTokenCount": 2}
        });
        let o = gemini_to_openai(&g, "g");
        assert_eq!(o["choices"][0]["message"]["content"], "ans");
        assert_eq!(o["choices"][0]["message"]["reasoning_content"], "think");
        assert_eq!(o["usage"]["completion_tokens"], 6);
    }
}
