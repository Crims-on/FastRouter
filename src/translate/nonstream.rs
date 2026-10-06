//! Non-streaming response conversion (port of handlers/chatCore/nonStreamingHandler.js
//! and sseToJsonHandler.js), generalised to a full target → openai → source chain.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{CLAUDE, GEMINI, GEMINI_CLI, ANTIGRAVITY, VERTEX, OLLAMA, OPENAI, OPENAI_RESPONSES};
use crate::jsv::{js_string, now_ms, now_s, truthy};

/// Provider JSON body (in `target` format) → OpenAI chat.completion.
pub fn to_openai(target: &str, body: &Value) -> Value {
    match target {
        OPENAI => body.clone(),
        GEMINI | GEMINI_CLI | ANTIGRAVITY | VERTEX => gemini_to_openai(body),
        CLAUDE => claude_to_openai(body),
        OLLAMA => super::resp::ollama_body_to_openai(body),
        OPENAI_RESPONSES => responses_to_openai(body),
        _ => body.clone(),
    }
}

fn gemini_to_openai(body: &Value) -> Value {
    let response = if truthy(&body["response"]) { &body["response"] } else { body };
    if response["candidates"][0].is_null() {
        return body.clone();
    }
    let candidate = &response["candidates"][0];
    let usage = if truthy(&response["usageMetadata"]) { &response["usageMetadata"] } else { &body["usageMetadata"] };
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = vec![];
    for part in candidate["content"]["parts"].as_array().into_iter().flatten() {
        if part["thought"] == json!(true) && truthy(&part["text"]) {
            reasoning.push_str(part["text"].as_str().unwrap_or(""));
        } else if crate::jsv::has(part, "text") {
            text.push_str(part["text"].as_str().unwrap_or(""));
        }
        if truthy(&part["functionCall"]) {
            let fc = &part["functionCall"];
            tool_calls.push(json!({
                "id": if truthy(&fc["id"]) { js_string(&fc["id"]) } else { format!("call_{}_{}_{}", js_string(&fc["name"]), now_ms(), tool_calls.len()) },
                "type": "function",
                "function": {"name": fc["name"], "arguments": if truthy(&fc["args"]) { fc["args"].to_string() } else { "{}".into() }},
            }));
        }
        let inline = if truthy(&part["inlineData"]) { &part["inlineData"] } else { &part["inline_data"] };
        if truthy(&inline["data"]) {
            let mime = inline["mimeType"].as_str().or_else(|| inline["mime_type"].as_str()).unwrap_or("image/png");
            text.push_str(&format!("\n![image](data:{mime};base64,{})\n", js_string(&inline["data"])));
        }
    }
    let mut message = json!({"role": "assistant"});
    if !text.is_empty() {
        message["content"] = json!(text);
    }
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls.clone());
    }
    if !truthy(&message["content"]) && !truthy(&message["tool_calls"]) {
        message["content"] = json!("");
    }
    let mut fr = candidate["finishReason"].as_str().unwrap_or("stop").to_lowercase();
    if fr == "max_tokens" {
        fr = "length".into();
    }
    if fr == "stop" && !tool_calls.is_empty() {
        fr = "tool_calls".into();
    }
    let mut result = json!({
        "id": format!("chatcmpl-{}", if truthy(&response["responseId"]) { js_string(&response["responseId"]) } else { now_ms().to_string() }),
        "object": "chat.completion",
        "created": now_s(),
        "model": if truthy(&response["modelVersion"]) { response["modelVersion"].clone() } else { json!("gemini") },
        "choices": [{"index": 0, "message": message, "finish_reason": fr}],
    });
    if truthy(usage) {
        let n = |k: &str| usage[k].as_i64().unwrap_or(0);
        result["usage"] = json!({
            "prompt_tokens": n("promptTokenCount") + n("thoughtsTokenCount"),
            "completion_tokens": n("candidatesTokenCount"),
            "total_tokens": n("totalTokenCount"),
        });
        if n("thoughtsTokenCount") > 0 {
            result["usage"]["completion_tokens_details"] = json!({"reasoning_tokens": n("thoughtsTokenCount")});
        }
    }
    result
}

fn claude_to_openai(body: &Value) -> Value {
    if truthy(&body["choices"]) || (truthy(&body["content"]) && !body["content"].is_array()) {
        return body.clone();
    }
    let mut text = String::new();
    let mut thinking = String::new();
    let mut tool_calls = vec![];
    for b in body["content"].as_array().into_iter().flatten() {
        match b["type"].as_str().unwrap_or("") {
            "text" => text.push_str(b["text"].as_str().unwrap_or("")),
            "thinking" => thinking.push_str(b["thinking"].as_str().unwrap_or("")),
            "tool_use" => tool_calls.push(json!({"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": if truthy(&b["input"]) { b["input"].to_string() } else { "{}".into() }}})),
            _ => {}
        }
    }
    let mut message = json!({"role": "assistant"});
    if !text.is_empty() {
        message["content"] = json!(text);
    }
    if !thinking.is_empty() {
        message["reasoning_content"] = json!(thinking);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    if !truthy(&message["content"]) && !truthy(&message["tool_calls"]) {
        message["content"] = json!("");
    }
    let fr = match body["stop_reason"].as_str().unwrap_or("stop") {
        "end_turn" | "stop_sequence" => "stop",
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        "refusal" => "content_filter",
        other => other,
    }
    .to_string();
    let mut result = json!({
        "id": format!("chatcmpl-{}", if truthy(&body["id"]) { js_string(&body["id"]) } else { now_ms().to_string() }),
        "object": "chat.completion",
        "created": now_s(),
        "model": if truthy(&body["model"]) { body["model"].clone() } else { json!("claude") },
        "choices": [{"index": 0, "message": message, "finish_reason": fr}],
    });
    if truthy(&body["usage"]) {
        let u = &body["usage"];
        let i = u["input_tokens"].as_i64().unwrap_or(0) + u["cache_read_input_tokens"].as_i64().unwrap_or(0) + u["cache_creation_input_tokens"].as_i64().unwrap_or(0);
        let o = u["output_tokens"].as_i64().unwrap_or(0);
        result["usage"] = json!({"prompt_tokens": i, "completion_tokens": o, "total_tokens": i + o});
        if let Some(c) = u["cache_read_input_tokens"].as_i64().filter(|c| *c > 0) {
            result["usage"]["prompt_tokens_details"] = json!({"cached_tokens": c});
        }
    }
    result
}

/// Responses API JSON (non-streaming) → chat.completion.
fn responses_to_openai(body: &Value) -> Value {
    if body["object"] != "response" && !body["output"].is_array() {
        return body.clone();
    }
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = vec![];
    for item in body["output"].as_array().into_iter().flatten() {
        match item["type"].as_str().unwrap_or("") {
            "message" => {
                for c in item["content"].as_array().into_iter().flatten() {
                    if let Some(t) = c["text"].as_str() {
                        text.push_str(t);
                    }
                }
            }
            "reasoning" => {
                for s in item["summary"].as_array().into_iter().flatten() {
                    reasoning.push_str(s["text"].as_str().unwrap_or(""));
                }
            }
            "function_call" | "custom_tool_call" => {
                let args = if item["type"] == "custom_tool_call" {
                    json!({"input": item["input"]}).to_string()
                } else {
                    item["arguments"].as_str().map(str::to_owned).unwrap_or_else(|| item["arguments"].to_string())
                };
                tool_calls.push(json!({"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": args}}));
            }
            _ => {}
        }
    }
    let mut message = json!({"role": "assistant", "content": text});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    let fr = if !tool_calls.is_empty() { "tool_calls" } else { "stop" };
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    let u = &body["usage"];
    let i = u["input_tokens"].as_i64().unwrap_or(0);
    let o = u["output_tokens"].as_i64().unwrap_or(0);
    let mut usage = json!({"prompt_tokens": i, "completion_tokens": o, "total_tokens": i + o});
    if let Some(c) = u["input_tokens_details"]["cached_tokens"].as_i64().filter(|c| *c > 0) {
        usage["prompt_tokens_details"] = json!({"cached_tokens": c});
    }
    json!({
        "id": body["id"].as_str().map(|s| s.replacen("resp_", "chatcmpl-", 1)).unwrap_or_else(|| format!("chatcmpl-{}", now_ms())),
        "object": "chat.completion",
        "created": body["created_at"].as_i64().unwrap_or_else(now_s),
        "model": if truthy(&body["model"]) { body["model"].clone() } else { json!("unknown") },
        "choices": [{"index": 0, "message": message, "finish_reason": fr}],
        "usage": usage,
    })
}

fn parse_tool_args(v: &Value) -> Value {
    match v {
        Value::Null => json!({}),
        Value::Object(_) => v.clone(),
        Value::String(s) if s.is_empty() => json!({}),
        Value::String(s) => serde_json::from_str(s).unwrap_or(json!({})),
        _ => json!({}),
    }
}

fn custom_input(args: &Value) -> String {
    let s = args.as_str().map(str::to_owned).unwrap_or_else(|| if truthy(args) { args.to_string() } else { "{}".into() });
    if let Ok(v) = serde_json::from_str::<Value>(&s) {
        if let Some(i) = v["input"].as_str() {
            return i.to_string();
        }
    }
    s
}

/// OpenAI chat.completion → client format JSON.
pub fn from_openai(source: &str, body: &Value, custom_tool_names: &[String]) -> Value {
    let Some(choice) = body["choices"].get(0) else { return body.clone() };
    let message = &choice["message"];
    let usage = &body["usage"];
    let n = |a: &str, b: &str| usage[a].as_i64().filter(|x| *x != 0).or_else(|| usage[b].as_i64()).unwrap_or(0);
    match source {
        CLAUDE => {
            let mut content = vec![];
            let reasoning = message["reasoning_content"].as_str().filter(|s| !s.is_empty()).or_else(|| message["provider_specific_fields"]["reasoning_content"].as_str()).unwrap_or("");
            if !reasoning.is_empty() {
                content.push(json!({"type": "thinking", "thinking": reasoning}));
            }
            if let Some(t) = message["content"].as_str().filter(|s| !s.is_empty()) {
                content.push(json!({"type": "text", "text": t}));
            }
            for tc in message["tool_calls"].as_array().into_iter().flatten() {
                let f = &tc["function"];
                content.push(json!({
                    "type": "tool_use",
                    "id": if truthy(&tc["id"]) { js_string(&tc["id"]) } else { format!("toolu_{}_{}", now_ms(), content.len()) },
                    "name": if truthy(&f["name"]) { f["name"].clone() } else if truthy(&tc["name"]) { tc["name"].clone() } else { json!("") },
                    "input": parse_tool_args(if truthy(&f["arguments"]) { &f["arguments"] } else { &tc["arguments"] }),
                }));
            }
            if content.is_empty() {
                content.push(json!({"type": "text", "text": ""}));
            }
            let mut out_usage = json!({"input_tokens": n("prompt_tokens", "input_tokens"), "output_tokens": n("completion_tokens", "output_tokens")});
            if let Some(c) = usage["prompt_tokens_details"]["cached_tokens"].as_i64().filter(|c| *c > 0) {
                out_usage["input_tokens"] = json!((n("prompt_tokens", "input_tokens") - c).max(0));
                out_usage["cache_read_input_tokens"] = json!(c);
            }
            json!({
                "id": js_string(&if truthy(&body["id"]) { body["id"].clone() } else { json!(format!("msg_{}", now_ms())) }).trim_start_matches("chatcmpl-").to_string(),
                "type": "message",
                "role": "assistant",
                "model": if truthy(&body["model"]) { body["model"].clone() } else { json!("unknown") },
                "content": content,
                "stop_reason": super::concerns::from_openai_finish(&choice["finish_reason"], "claude"),
                "stop_sequence": null,
                "usage": out_usage,
            })
        }
        OPENAI_RESPONSES => {
            let mut output = vec![];
            let reasoning = message["reasoning_content"].as_str().or_else(|| message["reasoning"].as_str()).unwrap_or("");
            if !reasoning.is_empty() {
                output.push(json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]}));
            }
            if let Some(t) = message["content"].as_str().filter(|s| !s.is_empty()) {
                output.push(json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": t, "annotations": []}]}));
            }
            for tc in message["tool_calls"].as_array().into_iter().flatten() {
                let f = &tc["function"];
                let custom = f["name"].as_str().map(|n| custom_tool_names.iter().any(|c| c == n)).unwrap_or(false);
                let id = crate::jsv::sn(&tc["id"]);
                let mut item = json!({"type": if custom { "custom_tool_call" } else { "function_call" }, "id": format!("{}_{id}", if custom { "ctc" } else { "fc" }), "call_id": id, "name": f["name"].as_str().unwrap_or("")});
                if custom {
                    item["input"] = json!(custom_input(&f["arguments"]));
                } else {
                    item["arguments"] = json!(f["arguments"].as_str().map(str::to_owned).unwrap_or_else(|| if truthy(&f["arguments"]) { f["arguments"].to_string() } else { "{}".into() }));
                }
                output.push(item);
            }
            let fr = choice["finish_reason"].as_str().unwrap_or("completed");
            let status = if fr == "tool_calls" || fr == "stop" { "completed" } else { fr };
            let p = n("prompt_tokens", "input_tokens");
            let c = n("completion_tokens", "output_tokens");
            json!({
                "id": format!("resp_{}", crate::jsv::sn(&body["id"])).replacen("resp_chatcmpl-", "resp_", 1),
                "object": "response",
                "created_at": body["created"].as_i64().unwrap_or_else(now_s),
                "model": if truthy(&body["model"]) { body["model"].clone() } else { json!("unknown") },
                "status": status,
                "background": false,
                "error": null,
                "output": output,
                "usage": {"input_tokens": p, "output_tokens": c, "total_tokens": usage["total_tokens"].as_i64().filter(|t| *t != 0).unwrap_or(p + c)},
            })
        }
        GEMINI | GEMINI_CLI | ANTIGRAVITY => {
            let mut parts = vec![];
            if let Some(r) = message["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
                parts.push(json!({"thought": true, "text": r}));
            }
            if let Some(t) = message["content"].as_str() {
                parts.push(json!({"text": t}));
            }
            for tc in message["tool_calls"].as_array().into_iter().flatten() {
                parts.push(json!({"functionCall": {"name": tc["function"]["name"], "args": parse_tool_args(&tc["function"]["arguments"])}}));
            }
            if parts.is_empty() {
                parts.push(json!({"text": ""}));
            }
            let fr = match choice["finish_reason"].as_str().unwrap_or("stop") {
                "length" => "MAX_TOKENS",
                "content_filter" => "SAFETY",
                _ => "STOP",
            };
            let p = n("prompt_tokens", "input_tokens");
            let c = n("completion_tokens", "output_tokens");
            let response = json!({
                "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": fr, "index": 0}],
                "usageMetadata": {"promptTokenCount": p, "candidatesTokenCount": c, "totalTokenCount": p + c},
                "modelVersion": body["model"],
                "responseId": body["id"],
            });
            if source == ANTIGRAVITY || source == GEMINI_CLI { json!({"response": response}) } else { response }
        }
        _ => body.clone(),
    }
}

/// Accumulates OpenAI chat chunks into one chat.completion (parseSSEToOpenAIResponse).
#[derive(Default)]
pub struct ChunkAggregator {
    first: Option<Value>,
    content: String,
    reasoning: String,
    tools: BTreeMap<i64, Value>,
    finish: Option<String>,
    usage: Option<Value>,
    pub error: Option<Value>,
    pub images: Vec<Value>,
}

impl ChunkAggregator {
    pub fn push(&mut self, chunk: &Value) {
        if truthy(&chunk["error"]) {
            self.error = Some(chunk["error"].clone());
            return;
        }
        if self.first.is_none() {
            self.first = Some(chunk.clone());
        }
        let choice = &chunk["choices"][0];
        let d = &choice["delta"];
        if let Some(c) = d["content"].as_str() {
            self.content.push_str(c);
        }
        if let Some(r) = d["reasoning_content"].as_str() {
            self.reasoning.push_str(r);
        }
        if let Some(imgs) = d["images"].as_array() {
            self.images.extend(imgs.iter().cloned());
        }
        if let Some(f) = choice["finish_reason"].as_str() {
            self.finish = Some(f.to_string());
        }
        if chunk["usage"].is_object() {
            self.usage = Some(chunk["usage"].clone());
        }
        for tc in d["tool_calls"].as_array().into_iter().flatten() {
            let idx = tc["index"].as_i64().unwrap_or(0);
            let e = self.tools.entry(idx).or_insert_with(|| json!({"id": tc["id"].as_str().unwrap_or(""), "type": "function", "function": {"name": "", "arguments": ""}}));
            if truthy(&tc["id"]) {
                e["id"] = tc["id"].clone();
            }
            if let Some(n) = tc["function"]["name"].as_str() {
                e["function"]["name"] = json!(e["function"]["name"].as_str().unwrap_or("").to_string() + n);
            }
            if let Some(a) = tc["function"]["arguments"].as_str() {
                e["function"]["arguments"] = json!(e["function"]["arguments"].as_str().unwrap_or("").to_string() + a);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.first.is_none()
    }

    pub fn finish(self, fallback_model: &str) -> Value {
        let first = self.first.unwrap_or(json!({}));
        let mut message = json!({"role": "assistant", "content": if self.content.is_empty() && !self.tools.is_empty() { Value::Null } else { json!(self.content) }});
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = json!(self.reasoning);
        }
        if !self.images.is_empty() {
            message["images"] = Value::Array(self.images);
        }
        let mut finish = self.finish.unwrap_or_else(|| "stop".into());
        if !self.tools.is_empty() {
            message["tool_calls"] = Value::Array(self.tools.into_values().collect());
            if finish == "stop" {
                finish = "tool_calls".into();
            }
        }
        let mut r = json!({
            "id": if truthy(&first["id"]) { first["id"].clone() } else { json!(format!("chatcmpl-{}", now_ms())) },
            "object": "chat.completion",
            "created": if truthy(&first["created"]) { first["created"].clone() } else { json!(now_s()) },
            "model": if truthy(&first["model"]) { first["model"].clone() } else { json!(fallback_model) },
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        });
        if let Some(u) = self.usage {
            r["usage"] = u;
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_json_to_openai_and_back() {
        let c = json!({"id": "msg_1", "type": "message", "content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t", "name": "f", "input": {"x": 1}}], "stop_reason": "tool_use", "usage": {"input_tokens": 10, "output_tokens": 5}});
        let o = to_openai(CLAUDE, &c);
        assert_eq!(o["choices"][0]["finish_reason"], "tool_calls");
        let back = from_openai(CLAUDE, &o, &[]);
        assert_eq!(back["stop_reason"], "tool_use");
        assert_eq!(back["content"][1]["input"]["x"], 1);
        let r = from_openai(OPENAI_RESPONSES, &o, &[]);
        assert_eq!(r["output"][1]["type"], "function_call");
    }

    #[test]
    fn aggregate() {
        let mut a = ChunkAggregator::default();
        a.push(&json!({"id": "x", "choices": [{"delta": {"content": "he"}}]}));
        a.push(&json!({"id": "x", "choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1}}));
        let r = a.finish("m");
        assert_eq!(r["choices"][0]["message"]["content"], "hello");
        assert_eq!(r["usage"]["prompt_tokens"], 1);
    }
}
