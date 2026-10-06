//! Streaming response translators (port of translator/response/*.js except
//! kiro/cursor/commandcode). All translators share one mutable JSON `state`
//! object, exactly like 9router's pipeline (`initState` + chained hops).

use serde_json::{Value, json};

use super::concerns::{
    build_chunk, build_usage, encode_data_uri, extract_reasoning_text, fallback_tool_call_id, from_openai_finish, reasoning_delta,
    to_openai_finish, to_openai_usage,
};
use crate::jsv::{js_string, map_size, now_ms, now_s, truthy};
use crate::session::store_thought_signature;

/// initState(sourceFormat)
pub fn init_state(source: &str) -> Value {
    let mut s = json!({
        "messageId": null,
        "model": null,
        "textBlockStarted": false,
        "thinkingBlockStarted": false,
        "inThinkingBlock": false,
        "currentBlockIndex": null,
        "toolCalls": {},
        "finishReason": null,
        "finishReasonSent": false,
        "usage": null,
        "contentBlockIndex": -1
    });
    if source == "openai-responses" {
        let extra = json!({
            "seq": 0,
            "responseId": format!("resp_{}", now_ms()),
            "created": now_s(),
            "started": false,
            "msgTextBuf": {},
            "msgItemAdded": {},
            "msgContentAdded": {},
            "msgItemDone": {},
            "reasoningId": "",
            "reasoningIndex": -1,
            "reasoningBuf": "",
            "reasoningPartAdded": false,
            "reasoningDone": false,
            "inThinking": false,
            "funcArgsBuf": {},
            "funcNames": {},
            "funcCallIds": {},
            "funcItemAdded": {},
            "funcArgsDone": {},
            "funcItemDone": {},
            "customToolNames": [],
            "responsesUsage": null,
            "completionPending": false,
            "completedSent": false
        });
        for (k, v) in extra.as_object().unwrap() {
            s[k] = v.clone();
        }
    }
    s
}

fn tool_name_lookup(state: &Value, name: &Value) -> Value {
    let mapped = &state["toolNameMap"][name.as_str().unwrap_or("")];
    if truthy(mapped) { mapped.clone() } else { name.clone() }
}

fn key(v: &Value) -> String {
    match v {
        Value::Null => "0".into(),
        other => js_string(other),
    }
}

// ---------------------------------------------------------------------------
// Claude → OpenAI
// ---------------------------------------------------------------------------

fn claude_chunk(state: &Value, delta: Value, finish: Value) -> Value {
    build_chunk(&format!("chatcmpl-{}", js_string(&state["messageId"])), now_s(), &state["model"], delta, finish)
}

pub fn claude_to_openai(chunk: &Value, state: &mut Value) -> Vec<Value> {
    let mut results = vec![];
    match chunk["type"].as_str().unwrap_or("") {
        "message_start" => {
            state["messageId"] = if truthy(&chunk["message"]["id"]) { chunk["message"]["id"].clone() } else { json!(format!("msg_{}", now_ms())) };
            state["model"] = chunk["message"]["model"].clone();
            state["toolCallIndex"] = json!(0);
            let u = &chunk["message"]["usage"];
            if u.is_object() {
                let i = u["input_tokens"].as_i64().unwrap_or(0);
                let cr = u["cache_read_input_tokens"].as_i64().unwrap_or(0);
                let cc = u["cache_creation_input_tokens"].as_i64().unwrap_or(0);
                let p = i + cr + cc;
                let mut su = json!({"prompt_tokens": p, "completion_tokens": 0, "total_tokens": p, "input_tokens": i, "output_tokens": 0});
                if cr > 0 {
                    su["cache_read_input_tokens"] = json!(cr);
                }
                if cc > 0 {
                    su["cache_creation_input_tokens"] = json!(cc);
                }
                state["usage"] = su;
            }
            results.push(claude_chunk(state, json!({"role": "assistant"}), Value::Null));
        }
        "content_block_start" => {
            let block = &chunk["content_block"];
            if block["type"] == "server_tool_use" {
                state["serverToolBlockIndex"] = chunk["index"].clone();
            } else if block["type"] == "text" {
                state["textBlockStarted"] = json!(true);
            } else if block["type"] == "tool_use" {
                let idx = state["toolCallIndex"].as_i64().unwrap_or(0);
                state["toolCallIndex"] = json!(idx + 1);
                let tc = json!({"index": idx, "id": block["id"], "type": "function", "function": {"name": tool_name_lookup(state, &block["name"]), "arguments": ""}});
                state["toolCalls"][key(&chunk["index"])] = tc.clone();
                results.push(claude_chunk(state, json!({"tool_calls": [tc]}), Value::Null));
            }
        }
        "content_block_delta" => {
            if !chunk["index"].is_null() && chunk["index"] == state["serverToolBlockIndex"] {
                return results;
            }
            let d = &chunk["delta"];
            if d["type"] == "text_delta" && truthy(&d["text"]) {
                results.push(claude_chunk(state, json!({"content": d["text"]}), Value::Null));
            } else if d["type"] == "thinking_delta" && truthy(&d["thinking"]) {
                results.push(claude_chunk(state, reasoning_delta(&d["thinking"], false), Value::Null));
            } else if d["type"] == "input_json_delta" && truthy(&d["partial_json"]) {
                let k = key(&chunk["index"]);
                if state["toolCalls"][&k].is_object() {
                    let prev = js_string(&state["toolCalls"][&k]["function"]["arguments"]);
                    state["toolCalls"][&k]["function"]["arguments"] = json!(prev + d["partial_json"].as_str().unwrap_or(""));
                    let tc = &state["toolCalls"][&k];
                    let delta = json!({"tool_calls": [{"index": tc["index"], "id": tc["id"], "function": {"arguments": d["partial_json"]}}]});
                    results.push(claude_chunk(state, delta, Value::Null));
                }
            }
        }
        "content_block_stop" => {
            if !chunk["index"].is_null() && chunk["index"] == state["serverToolBlockIndex"] {
                state["serverToolBlockIndex"] = json!(-1);
                return results;
            }
            state["textBlockStarted"] = json!(false);
            state["thinkingBlockStarted"] = json!(false);
        }
        "message_delta" => {
            let u = &chunk["usage"];
            if u.is_object() {
                let prev = state["usage"].clone();
                let pick = |k: &str| u[k].as_i64().or_else(|| prev[k].as_i64()).unwrap_or(0);
                let i = pick("input_tokens");
                let o = u["output_tokens"].as_i64().unwrap_or(0);
                let cr = pick("cache_read_input_tokens");
                let cc = pick("cache_creation_input_tokens");
                let p = i + cr + cc;
                let mut su = json!({"prompt_tokens": p, "completion_tokens": o, "total_tokens": p + o, "input_tokens": i, "output_tokens": o});
                if cr > 0 {
                    su["cache_read_input_tokens"] = json!(cr);
                }
                if cc > 0 {
                    su["cache_creation_input_tokens"] = json!(cc);
                }
                state["usage"] = su;
            }
            if truthy(&chunk["delta"]["stop_reason"]) {
                state["finishReason"] = to_openai_finish(&chunk["delta"]["stop_reason"], "claude");
                if chunk["delta"]["stop_reason"] == "refusal" && truthy(&chunk["delta"]["stop_details"]["explanation"]) {
                    results.push(claude_chunk(state, json!({"content": chunk["delta"]["stop_details"]["explanation"]}), Value::Null));
                }
                let mut fc = claude_chunk(state, json!({}), state["finishReason"].clone());
                if truthy(&state["usage"]) {
                    let su = &state["usage"];
                    fc["usage"] = to_openai_usage(
                        &json!({"input_tokens": su["input_tokens"].as_i64().unwrap_or(0), "output_tokens": su["output_tokens"].as_i64().unwrap_or(0),
                                "cache_read_input_tokens": su["cache_read_input_tokens"], "cache_creation_input_tokens": su["cache_creation_input_tokens"]}),
                        "claude",
                    )
                    .unwrap_or(Value::Null);
                }
                results.push(fc);
                state["finishReasonSent"] = json!(true);
            }
        }
        "message_stop" => {
            if !truthy(&state["finishReasonSent"]) {
                let fr = if truthy(&state["finishReason"]) {
                    state["finishReason"].clone()
                } else if map_size(&state["toolCalls"]) > 0 {
                    json!("tool_calls")
                } else {
                    json!("stop")
                };
                let mut c = claude_chunk(state, json!({}), fr);
                if state["usage"].is_object() {
                    let i = state["usage"]["input_tokens"].as_i64().unwrap_or(0);
                    let o = state["usage"]["output_tokens"].as_i64().unwrap_or(0);
                    c["usage"] = json!({"prompt_tokens": i, "completion_tokens": o, "total_tokens": i + o});
                }
                results.push(c);
                state["finishReasonSent"] = json!(true);
            }
        }
        _ => {}
    }
    results
}

// ---------------------------------------------------------------------------
// OpenAI → Claude
// ---------------------------------------------------------------------------

fn sanitize_tool_args(name: &str, args: &str) -> String {
    let Ok(mut v) = serde_json::from_str::<Value>(args) else { return args.to_string() };
    let n = name.strip_prefix("proxy_").unwrap_or(name);
    if n == "Read" && v.is_object() {
        let int_re = |s: &str, neg: bool| {
            let t = if neg { s.strip_prefix('-').unwrap_or(s) } else { s };
            !t.is_empty() && t.chars().all(|c| c.is_ascii_digit())
        };
        if let Some(s) = v["limit"].as_str().map(str::to_owned) {
            if int_re(&s, false) {
                v["limit"] = json!(s.parse::<i64>().unwrap_or(0));
            }
        }
        if let Some(s) = v["offset"].as_str().map(str::to_owned) {
            if int_re(&s, true) {
                v["offset"] = json!(s.parse::<i64>().unwrap_or(0));
            }
        }
        if let Some(l) = v["limit"].as_f64() {
            if l > 2000.0 {
                v["limit"] = json!(2000);
            }
            if l < 1.0 {
                crate::jsv::del(&mut v, "limit");
            }
        }
        if v["offset"].as_f64().map(|o| o < 0.0).unwrap_or(false) {
            v["offset"] = json!(0);
        }
        if crate::jsv::has(&v, "pages") {
            let fp = v["file_path"].as_str().map(|s| s.to_lowercase().ends_with(".pdf")).unwrap_or(false);
            let ok = v["pages"].as_str().map(|p| {
                let parts: Vec<&str> = p.split('-').collect();
                (parts.len() == 1 || parts.len() == 2) && parts.iter().all(|x| !x.is_empty() && x.chars().all(|c| c.is_ascii_digit()))
            });
            if !(fp && ok == Some(true)) {
                crate::jsv::del(&mut v, "pages");
            }
        }
    }
    v.to_string()
}

fn stop_thinking(state: &mut Value, results: &mut Vec<Value>) {
    if !truthy(&state["thinkingBlockStarted"]) {
        return;
    }
    results.push(json!({"type": "content_block_stop", "index": state["thinkingBlockIndex"]}));
    state["thinkingBlockStarted"] = json!(false);
}

fn stop_text(state: &mut Value, results: &mut Vec<Value>) {
    if !truthy(&state["textBlockStarted"]) || truthy(&state["textBlockClosed"]) {
        return;
    }
    state["textBlockClosed"] = json!(true);
    results.push(json!({"type": "content_block_stop", "index": state["textBlockIndex"]}));
    state["textBlockStarted"] = json!(false);
}

fn next_block(state: &mut Value) -> i64 {
    let n = state["nextBlockIndex"].as_i64().unwrap_or(0);
    state["nextBlockIndex"] = json!(n + 1);
    n
}

pub fn openai_to_claude(chunk: &Value, state: &mut Value) -> Vec<Value> {
    let mut results = vec![];
    if chunk["choices"][0].is_null() {
        return results;
    }
    let choice = &chunk["choices"][0];
    let delta = &choice["delta"];
    let u = &chunk["usage"];
    if u.is_object() {
        let p = u["prompt_tokens"].as_i64().unwrap_or(0);
        let o = u["completion_tokens"].as_i64().unwrap_or(0);
        let cr = u["prompt_tokens_details"]["cached_tokens"].as_i64().unwrap_or(0);
        let cc = u["prompt_tokens_details"]["cache_creation_tokens"].as_i64().unwrap_or(0);
        let mut su = json!({"input_tokens": p - cr - cc, "output_tokens": o});
        if cr > 0 {
            su["cache_read_input_tokens"] = json!(cr);
        }
        if cc > 0 {
            su["cache_creation_input_tokens"] = json!(cc);
        }
        state["usage"] = su;
    }
    if !truthy(&state["messageStartSent"]) {
        state["messageStartSent"] = json!(true);
        let mut mid = chunk["id"].as_str().map(|s| s.replacen("chatcmpl-", "", 1)).unwrap_or_default();
        if mid.is_empty() {
            mid = format!("msg_{}", now_ms());
        }
        if mid == "chat" || mid.chars().count() < 8 {
            mid = chunk["extend_fields"]["requestId"]
                .as_str()
                .or_else(|| chunk["extend_fields"]["traceId"].as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("msg_{}", now_ms()));
        }
        state["messageId"] = json!(mid);
        state["model"] = if truthy(&chunk["model"]) { chunk["model"].clone() } else { json!("unknown") };
        state["nextBlockIndex"] = json!(0);
        results.push(json!({"type": "message_start", "message": {
            "id": state["messageId"], "type": "message", "role": "assistant", "model": state["model"],
            "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 0, "output_tokens": 0}
        }}));
    }
    let reasoning = extract_reasoning_text(delta);
    if !reasoning.is_empty() {
        stop_text(state, &mut results);
        if !truthy(&state["thinkingBlockStarted"]) {
            let i = next_block(state);
            state["thinkingBlockIndex"] = json!(i);
            state["thinkingBlockStarted"] = json!(true);
            results.push(json!({"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": ""}}));
        }
        results.push(json!({"type": "content_block_delta", "index": state["thinkingBlockIndex"], "delta": {"type": "thinking_delta", "thinking": reasoning}}));
    }
    if truthy(&delta["content"]) {
        stop_thinking(state, &mut results);
        if !truthy(&state["textBlockStarted"]) {
            let i = next_block(state);
            state["textBlockIndex"] = json!(i);
            state["textBlockStarted"] = json!(true);
            state["textBlockClosed"] = json!(false);
            results.push(json!({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}));
        }
        results.push(json!({"type": "content_block_delta", "index": state["textBlockIndex"], "delta": {"type": "text_delta", "text": delta["content"]}}));
    }
    if let Some(tcs) = delta["tool_calls"].as_array() {
        for tc in tcs {
            let idx = if tc["index"].is_null() { "0".to_string() } else { js_string(&tc["index"]) };
            if truthy(&tc["id"]) && !state["toolCalls"].as_object().map(|o| o.contains_key(&idx)).unwrap_or(false) {
                stop_thinking(state, &mut results);
                stop_text(state, &mut results);
                let bi = next_block(state);
                let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                state["toolCalls"][&idx] = json!({"id": tc["id"], "name": name, "blockIndex": bi});
                let shown = name.strip_prefix("proxy_").unwrap_or(&name).to_string();
                results.push(json!({"type": "content_block_start", "index": bi, "content_block": {"type": "tool_use", "id": tc["id"], "name": shown, "input": {}}}));
            }
            if truthy(&tc["function"]["arguments"]) && state["toolCalls"][&idx].is_object() {
                let prev = state["toolArgBuffers"][&idx].as_str().unwrap_or("").to_string();
                state["toolArgBuffers"][&idx] = json!(prev + tc["function"]["arguments"].as_str().unwrap_or(""));
            }
        }
    }
    if truthy(&choice["finish_reason"]) {
        stop_thinking(state, &mut results);
        stop_text(state, &mut results);
        let tool_calls = state["toolCalls"].as_object().cloned().unwrap_or_default();
        for (idx, info) in tool_calls {
            if let Some(buf) = state["toolArgBuffers"][&idx].as_str().filter(|b| !b.is_empty()) {
                let sanitized = sanitize_tool_args(info["name"].as_str().unwrap_or(""), buf);
                results.push(json!({"type": "content_block_delta", "index": info["blockIndex"], "delta": {"type": "input_json_delta", "partial_json": sanitized}}));
            }
            results.push(json!({"type": "content_block_stop", "index": info["blockIndex"]}));
        }
        state["finishReason"] = choice["finish_reason"].clone();
        let usage = if truthy(&state["usage"]) { state["usage"].clone() } else { json!({"input_tokens": 0, "output_tokens": 0}) };
        results.push(json!({"type": "message_delta", "delta": {"stop_reason": from_openai_finish(&choice["finish_reason"], "claude")}, "usage": usage}));
        results.push(json!({"type": "message_stop"}));
    }
    results
}

// ---------------------------------------------------------------------------
// Gemini (incl. CLI / Antigravity / Vertex envelopes) → OpenAI
// ---------------------------------------------------------------------------

fn gemini_meta(state: &Value) -> (String, i64, Value) {
    (format!("chatcmpl-{}", js_string(&state["messageId"])), now_s(), state["model"].clone())
}

fn emit_function_call(fc: &Value, state: &mut Value, signature: Option<&str>) -> Value {
    let name = tool_name_lookup(state, &fc["name"]);
    let args = if truthy(&fc["args"]) { fc["args"].clone() } else { json!({}) };
    let idx = state["functionIndex"].as_i64().unwrap_or(0);
    state["functionIndex"] = json!(idx + 1);
    let call_id = if truthy(&fc["id"]) { js_string(&fc["id"]) } else { format!("{}-{}-{idx}", js_string(&name), now_ms()) };
    if let Some(sig) = signature {
        store_thought_signature(&call_id, sig, state["sessionId"].as_str(), state["model"].as_str());
    }
    state["geminiToolCallCount"] = json!(state["geminiToolCallCount"].as_i64().unwrap_or(0) + 1);
    let (id, created, model) = gemini_meta(state);
    build_chunk(&id, created, &model, json!({"tool_calls": [{"id": call_id, "index": idx, "type": "function", "function": {"name": name, "arguments": args.to_string()}}]}), Value::Null)
}

pub fn gemini_to_openai(chunk: &Value, state: &mut Value) -> Vec<Value> {
    let response = if truthy(&chunk["response"]) { &chunk["response"] } else { chunk };
    if response["candidates"][0].is_null() {
        return vec![];
    }
    let mut results = vec![];
    let candidate = &response["candidates"][0];
    if !truthy(&state["messageId"]) {
        state["messageId"] = if truthy(&response["responseId"]) { response["responseId"].clone() } else { json!(format!("msg_{}", now_ms())) };
        state["model"] = if truthy(&response["modelVersion"]) {
            response["modelVersion"].clone()
        } else if truthy(&state["model"]) {
            state["model"].clone()
        } else {
            json!("gemini")
        };
        state["functionIndex"] = json!(0);
        state["geminiToolCallCount"] = json!(0);
        let (id, created, model) = gemini_meta(state);
        results.push(build_chunk(&id, created, &model, json!({"role": "assistant"}), Value::Null));
    }
    for part in candidate["content"]["parts"].as_array().into_iter().flatten() {
        let sig = if truthy(&part["thoughtSignature"]) { &part["thoughtSignature"] } else { &part["thought_signature"] };
        if let Some(s) = sig.as_str() {
            state["pendingThoughtSignature"] = json!(s);
        }
        let is_thought = part["thought"] == json!(true);
        if truthy(sig) {
            let has_text = crate::jsv::has(part, "text") && part["text"] != "";
            let has_fc = truthy(&part["functionCall"]);
            if !has_text && !has_fc {
                continue;
            }
            if has_text {
                let (id, created, model) = gemini_meta(state);
                let d = if is_thought { reasoning_delta(&part["text"], false) } else { json!({"content": part["text"]}) };
                results.push(build_chunk(&id, created, &model, d, Value::Null));
            }
            if has_fc {
                let s = sig.as_str().map(str::to_owned);
                results.push(emit_function_call(&part["functionCall"], state, s.as_deref()));
                state["pendingThoughtSignature"] = Value::Null;
            }
            continue;
        }
        if crate::jsv::has(part, "text") && part["text"] != "" && !part["text"].is_null() {
            let (id, created, model) = gemini_meta(state);
            let d = if is_thought { reasoning_delta(&part["text"], false) } else { json!({"content": part["text"]}) };
            results.push(build_chunk(&id, created, &model, d, Value::Null));
        }
        if truthy(&part["functionCall"]) {
            let s = state["pendingThoughtSignature"].as_str().map(str::to_owned);
            results.push(emit_function_call(&part["functionCall"], state, s.as_deref()));
            state["pendingThoughtSignature"] = Value::Null;
        }
        let inline = if truthy(&part["inlineData"]) { &part["inlineData"] } else { &part["inline_data"] };
        if truthy(&inline["data"]) {
            let mime = inline["mimeType"].as_str().or_else(|| inline["mime_type"].as_str()).unwrap_or("image/png");
            let (id, created, model) = gemini_meta(state);
            results.push(build_chunk(&id, created, &model, json!({"images": [{"type": "image_url", "image_url": {"url": encode_data_uri(mime, &js_string(&inline["data"]))}}]}), Value::Null));
        }
    }
    let meta = if truthy(&response["usageMetadata"]) { &response["usageMetadata"] } else { &chunk["usageMetadata"] };
    if let Some(u) = to_openai_usage(meta, "gemini") {
        state["usage"] = u;
    }
    if truthy(&candidate["finishReason"]) {
        let mut fr = to_openai_finish(&candidate["finishReason"], "gemini");
        if fr == "stop" && state["geminiToolCallCount"].as_i64().unwrap_or(0) > 0 {
            fr = json!("tool_calls");
        }
        let (id, created, model) = gemini_meta(state);
        let mut fc = build_chunk(&id, created, &model, json!({}), fr.clone());
        if truthy(&state["usage"]) {
            fc["usage"] = state["usage"].clone();
        }
        results.push(fc);
        state["finishReason"] = fr;
    }
    results
}

// ---------------------------------------------------------------------------
// OpenAI → Antigravity / Gemini (client side)
// ---------------------------------------------------------------------------

pub fn openai_to_antigravity(chunk: &Value, state: &mut Value) -> Vec<Value> {
    let choice = &chunk["choices"][0];
    if choice.is_null() {
        if truthy(&chunk["usage"]) {
            state["_usage"] = chunk["usage"].clone();
        }
        return vec![];
    }
    let delta = &choice["delta"];
    let finish = &choice["finish_reason"];
    if !state["_toolCallAccum"].is_object() {
        state["_toolCallAccum"] = json!({});
    }
    if !truthy(&state["_responseId"]) {
        state["_responseId"] = if truthy(&chunk["id"]) { chunk["id"].clone() } else { json!(format!("resp_{}", now_ms())) };
    }
    if !truthy(&state["_modelVersion"]) {
        state["_modelVersion"] = if truthy(&chunk["model"]) { chunk["model"].clone() } else { json!("") };
    }
    let mut parts = vec![];
    if truthy(&delta["reasoning_content"]) {
        parts.push(json!({"thought": true, "text": delta["reasoning_content"]}));
    }
    if truthy(&delta["content"]) {
        parts.push(json!({"text": delta["content"]}));
    }
    if let Some(tcs) = delta["tool_calls"].as_array() {
        for tc in tcs {
            let idx = if tc["index"].is_null() { "0".to_string() } else { js_string(&tc["index"]) };
            if !state["_toolCallAccum"][&idx].is_object() {
                state["_toolCallAccum"][&idx] = json!({"id": "", "name": "", "arguments": ""});
            }
            let acc = &mut state["_toolCallAccum"][&idx];
            if truthy(&tc["id"]) {
                acc["id"] = tc["id"].clone();
            }
            if let Some(n) = tc["function"]["name"].as_str().filter(|s| !s.is_empty()) {
                acc["name"] = json!(js_string(&acc["name"]) + n);
            }
            if let Some(a) = tc["function"]["arguments"].as_str().filter(|s| !s.is_empty()) {
                acc["arguments"] = json!(js_string(&acc["arguments"]) + a);
            }
        }
        if parts.is_empty() && !truthy(finish) {
            return vec![];
        }
    }
    if truthy(finish) {
        let accum = state["_toolCallAccum"].as_object().cloned().unwrap_or_default();
        for (_, acc) in accum {
            let args: Value = serde_json::from_str(acc["arguments"].as_str().unwrap_or("")).unwrap_or(json!({}));
            let name = tool_name_lookup(state, &acc["name"]);
            parts.push(json!({"functionCall": {"name": name, "args": args}}));
        }
    }
    if parts.is_empty() && !truthy(finish) {
        return vec![];
    }
    if parts.is_empty() {
        parts.push(json!({"text": ""}));
    }
    let mut candidate = json!({"content": {"role": "model", "parts": parts}});
    if truthy(finish) {
        candidate["finishReason"] = json!(match finish.as_str().unwrap_or("") {
            "length" => "MAX_TOKENS",
            "content_filter" => "SAFETY",
            _ => "STOP",
        });
    }
    let mut response = json!({"candidates": [candidate], "modelVersion": state["_modelVersion"], "responseId": state["_responseId"]});
    let usage = if truthy(&chunk["usage"]) { chunk["usage"].clone() } else { state["_usage"].clone() };
    if truthy(&usage) {
        let mut um = json!({
            "promptTokenCount": usage["prompt_tokens"].as_i64().unwrap_or(0),
            "candidatesTokenCount": usage["completion_tokens"].as_i64().unwrap_or(0),
            "totalTokenCount": usage["total_tokens"].as_i64().unwrap_or(0),
        });
        if truthy(&usage["completion_tokens_details"]["reasoning_tokens"]) {
            um["thoughtsTokenCount"] = usage["completion_tokens_details"]["reasoning_tokens"].clone();
        }
        if truthy(&usage["prompt_tokens_details"]["cached_tokens"]) {
            um["cachedContentTokenCount"] = usage["prompt_tokens_details"]["cached_tokens"].clone();
        }
        response["usageMetadata"] = um;
    }
    vec![json!({"response": response})]
}

// ---------------------------------------------------------------------------
// Ollama → OpenAI
// ---------------------------------------------------------------------------

fn convert_ollama_tool_calls(tcs: &[Value]) -> Vec<Value> {
    tcs.iter()
        .enumerate()
        .map(|(i, tc)| {
            let a = &tc["function"]["arguments"];
            json!({
                "index": if tc["function"]["index"].is_null() { json!(i) } else { tc["function"]["index"].clone() },
                "id": if truthy(&tc["id"]) { tc["id"].clone() } else { json!(fallback_tool_call_id(Some(i as i64))) },
                "type": "function",
                "function": {"name": tc["function"]["name"].as_str().unwrap_or(""), "arguments": if a.is_string() { a.clone() } else { json!(if truthy(a) { a.to_string() } else { "{}".into() }) }},
            })
        })
        .collect()
}

pub fn ollama_to_openai(chunk: &Value, state: &mut Value) -> Vec<Value> {
    if !chunk.is_object() {
        return vec![];
    }
    if !truthy(&state["ollama"]) {
        state["ollama"] = json!({"id": format!("chatcmpl-{}", now_ms()), "created": now_s(), "model": if truthy(&chunk["model"]) { chunk["model"].clone() } else { state["model"].clone() }});
    }
    let o = state["ollama"].clone();
    let (id, created, model) = (o["id"].as_str().unwrap_or("").to_string(), o["created"].as_i64().unwrap_or(0), o["model"].clone());
    if truthy(&chunk["done"]) {
        let mut fr = to_openai_finish(&chunk["done_reason"], "ollama");
        if chunk["done_reason"] == "tool_calls" || truthy(&state["hadToolCalls"]) {
            fr = json!("tool_calls");
        }
        let mut c = build_chunk(&id, created, &model, json!({}), fr);
        c["usage"] = to_openai_usage(chunk, "ollama").unwrap_or(Value::Null);
        return vec![c];
    }
    let msg = &chunk["message"];
    if !truthy(msg) {
        return vec![];
    }
    let content = msg["content"].as_str().unwrap_or("");
    let thinking = msg["thinking"].as_str().unwrap_or("");
    let tcs = msg["tool_calls"].as_array();
    if content.is_empty() && thinking.is_empty() && tcs.is_none() {
        return vec![];
    }
    let mut delta = json!({});
    if !content.is_empty() {
        delta["content"] = json!(content);
    }
    if !thinking.is_empty() {
        delta["reasoning_content"] = json!(thinking);
    }
    if let Some(tcs) = tcs {
        state["hadToolCalls"] = json!(true);
        delta["tool_calls"] = Value::Array(convert_ollama_tool_calls(tcs));
    }
    vec![build_chunk(&id, created, &model, delta, Value::Null)]
}

pub fn ollama_body_to_openai(body: &Value) -> Value {
    let msg = &body["message"];
    let tcs = msg["tool_calls"].as_array().cloned().unwrap_or_default();
    let mut message = json!({"role": "assistant"});
    if truthy(&msg["content"]) {
        message["content"] = msg["content"].clone();
    }
    if truthy(&msg["thinking"]) {
        message["reasoning_content"] = msg["thinking"].clone();
    }
    if !tcs.is_empty() {
        message["tool_calls"] = Value::Array(convert_ollama_tool_calls(&tcs));
    }
    if !truthy(&message["content"]) && !truthy(&message["tool_calls"]) {
        message["content"] = json!("");
    }
    let mut fr = to_openai_finish(&body["done_reason"], "ollama");
    if !tcs.is_empty() {
        fr = json!("tool_calls");
    }
    json!({
        "id": format!("chatcmpl-{}", now_ms()),
        "object": "chat.completion",
        "created": now_s(),
        "model": if truthy(&body["model"]) { body["model"].clone() } else { json!("ollama") },
        "choices": [{"index": 0, "message": message, "finish_reason": fr}],
        "usage": to_openai_usage(body, "ollama"),
    })
}

// ---------------------------------------------------------------------------
// OpenAI chat chunks → Responses API events
// ---------------------------------------------------------------------------

fn to_responses_usage(u: &Value) -> Option<Value> {
    if !u.is_object() {
        return None;
    }
    let is_int = |v: &Value| v.as_i64().is_some() || v.as_u64().is_some();
    let inp = [&u["input_tokens"], &u["prompt_tokens"]].into_iter().find(|v| is_int(v))?.as_i64()?;
    let out = [&u["output_tokens"], &u["completion_tokens"]].into_iter().find(|v| is_int(v))?.as_i64()?;
    if inp + out <= 0 {
        return None;
    }
    let mut r = json!({"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out});
    if let Some(c) = [&u["input_tokens_details"]["cached_tokens"], &u["prompt_tokens_details"]["cached_tokens"]].into_iter().find(|v| is_int(v)) {
        r["input_tokens_details"] = json!({"cached_tokens": c});
    }
    if let Some(c) = [&u["output_tokens_details"]["reasoning_tokens"], &u["completion_tokens_details"]["reasoning_tokens"]].into_iter().find(|v| is_int(v)) {
        r["output_tokens_details"] = json!({"reasoning_tokens": c});
    }
    Some(r)
}

struct Emitter<'a> {
    state: &'a mut Value,
    events: Vec<Value>,
}

impl Emitter<'_> {
    fn emit(&mut self, event: &str, mut data: Value) {
        let seq = self.state["seq"].as_i64().unwrap_or(0) + 1;
        self.state["seq"] = json!(seq);
        data["sequence_number"] = json!(seq);
        self.events.push(json!({"event": event, "data": data}));
    }
}

fn record_completed(state: &mut Value, idx: i64, item: Value) {
    if !state["completedOutputItems"].is_object() {
        state["completedOutputItems"] = json!({});
    }
    state["completedOutputItems"][idx.to_string()] = item;
}

fn collect_completed(state: &Value) -> Vec<Value> {
    let mut items: Vec<(i64, Value)> = state["completedOutputItems"]
        .as_object()
        .map(|o| o.iter().map(|(k, v)| (k.parse().unwrap_or(0), v.clone())).collect())
        .unwrap_or_default();
    items.sort_by_key(|(k, _)| *k);
    items.into_iter().map(|(_, v)| v).collect()
}

fn start_reasoning(e: &mut Emitter, idx: i64) {
    if !truthy(&e.state["reasoningId"]) {
        let rid = format!("rs_{}_{idx}", js_string(&e.state["responseId"]));
        e.state["reasoningId"] = json!(rid);
        e.state["reasoningIndex"] = json!(idx);
        e.emit("response.output_item.added", json!({"type": "response.output_item.added", "output_index": idx, "item": {"id": rid, "type": "reasoning", "summary": []}}));
        e.emit("response.reasoning_summary_part.added", json!({"type": "response.reasoning_summary_part.added", "item_id": rid, "output_index": idx, "summary_index": 0, "part": {"type": "summary_text", "text": ""}}));
        e.state["reasoningPartAdded"] = json!(true);
    }
}

fn emit_reasoning_delta(e: &mut Emitter, text: &str) {
    if text.is_empty() {
        return;
    }
    let buf = e.state["reasoningBuf"].as_str().unwrap_or("").to_string() + text;
    e.state["reasoningBuf"] = json!(buf);
    let (rid, ri) = (e.state["reasoningId"].clone(), e.state["reasoningIndex"].clone());
    e.emit("response.reasoning_summary_text.delta", json!({"type": "response.reasoning_summary_text.delta", "item_id": rid, "output_index": ri, "summary_index": 0, "delta": text}));
}

fn close_reasoning(e: &mut Emitter) {
    if truthy(&e.state["reasoningId"]) && !truthy(&e.state["reasoningDone"]) {
        e.state["reasoningDone"] = json!(true);
        let (rid, ri, buf) = (e.state["reasoningId"].clone(), e.state["reasoningIndex"].clone(), e.state["reasoningBuf"].clone());
        e.emit("response.reasoning_summary_text.done", json!({"type": "response.reasoning_summary_text.done", "item_id": rid, "output_index": ri, "summary_index": 0, "text": buf}));
        e.emit("response.reasoning_summary_part.done", json!({"type": "response.reasoning_summary_part.done", "item_id": rid, "output_index": ri, "summary_index": 0, "part": {"type": "summary_text", "text": buf}}));
        let item = json!({"id": rid, "type": "reasoning", "summary": [{"type": "summary_text", "text": buf}]});
        e.emit("response.output_item.done", json!({"type": "response.output_item.done", "output_index": ri, "item": item}));
        let idx = ri.as_i64().unwrap_or(0);
        record_completed(e.state, idx, item);
    }
}

fn emit_text(e: &mut Emitter, idx: i64, content: &str) {
    let k = idx.to_string();
    let msg_id = format!("msg_{}_{idx}", js_string(&e.state["responseId"]));
    if !truthy(&e.state["msgItemAdded"][&k]) {
        e.state["msgItemAdded"][&k] = json!(true);
        e.emit("response.output_item.added", json!({"type": "response.output_item.added", "output_index": idx, "item": {"id": msg_id, "type": "message", "content": [], "role": "assistant"}}));
    }
    if !truthy(&e.state["msgContentAdded"][&k]) {
        e.state["msgContentAdded"][&k] = json!(true);
        e.emit("response.content_part.added", json!({"type": "response.content_part.added", "item_id": msg_id, "output_index": idx, "content_index": 0, "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}}));
    }
    e.emit("response.output_text.delta", json!({"type": "response.output_text.delta", "item_id": msg_id, "output_index": idx, "content_index": 0, "delta": content, "logprobs": []}));
    let prev = e.state["msgTextBuf"][&k].as_str().unwrap_or("").to_string();
    e.state["msgTextBuf"][&k] = json!(prev + content);
}

fn close_message(e: &mut Emitter, k: &str) {
    if truthy(&e.state["msgItemAdded"][k]) && !truthy(&e.state["msgItemDone"][k]) {
        e.state["msgItemDone"][k] = json!(true);
        let full = e.state["msgTextBuf"][k].as_str().unwrap_or("").to_string();
        let idx: i64 = k.parse().unwrap_or(0);
        let msg_id = format!("msg_{}_{k}", js_string(&e.state["responseId"]));
        e.emit("response.output_text.done", json!({"type": "response.output_text.done", "item_id": msg_id, "output_index": idx, "content_index": 0, "text": full, "logprobs": []}));
        e.emit("response.content_part.done", json!({"type": "response.content_part.done", "item_id": msg_id, "output_index": idx, "content_index": 0, "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": full}}));
        let item = json!({"id": msg_id, "type": "message", "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": full}], "role": "assistant"});
        e.emit("response.output_item.done", json!({"type": "response.output_item.done", "output_index": idx, "item": item}));
        record_completed(e.state, idx, item);
    }
}

fn is_custom_tool(state: &Value, name: &Value) -> bool {
    truthy(name) && state["customToolNames"].as_array().map(|a| a.contains(name)).unwrap_or(false)
}

fn extract_custom_input(args: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(args) {
        if let Some(s) = v["input"].as_str() {
            if v.is_object() {
                return s.to_string();
            }
        }
    }
    args.to_string()
}

fn emit_tool_call(e: &mut Emitter, tc: &Value) {
    let k = if tc["index"].is_null() { "0".to_string() } else { js_string(&tc["index"]) };
    let idx: i64 = k.parse().unwrap_or(0);
    if truthy(&tc["function"]["name"]) {
        e.state["funcNames"][&k] = tc["function"]["name"].clone();
    }
    if truthy(&tc["id"]) {
        e.state["funcCallIds"][&k] = tc["id"].clone();
    }
    let call_id = e.state["funcCallIds"][&k].clone();
    if !truthy(&e.state["funcItemAdded"][&k]) && truthy(&call_id) && truthy(&e.state["funcNames"][&k]) {
        e.state["funcItemAdded"][&k] = json!(true);
        let name = e.state["funcNames"][&k].clone();
        let custom = is_custom_tool(e.state, &name);
        let mut item = json!({"id": format!("{}_{}", if custom { "ctc" } else { "fc" }, js_string(&call_id)), "type": if custom { "custom_tool_call" } else { "function_call" }});
        if custom {
            item["input"] = json!("");
        } else {
            item["arguments"] = json!("");
        }
        item["call_id"] = call_id.clone();
        item["name"] = name;
        e.emit("response.output_item.added", json!({"type": "response.output_item.added", "output_index": idx, "item": item}));
    }
    if !truthy(&e.state["funcArgsBuf"][&k]) {
        e.state["funcArgsBuf"][&k] = json!("");
    }
    if let Some(args) = tc["function"]["arguments"].as_str().filter(|a| !a.is_empty()) {
        let ref_id = if truthy(&e.state["funcCallIds"][&k]) { e.state["funcCallIds"][&k].clone() } else { tc["id"].clone() };
        let name = e.state["funcNames"][&k].clone();
        if truthy(&e.state["funcItemAdded"][&k]) && truthy(&ref_id) && !is_custom_tool(e.state, &name) {
            e.emit("response.function_call_arguments.delta", json!({"type": "response.function_call_arguments.delta", "item_id": format!("fc_{}", js_string(&ref_id)), "output_index": idx, "delta": args}));
        }
        let prev = e.state["funcArgsBuf"][&k].as_str().unwrap_or("").to_string();
        e.state["funcArgsBuf"][&k] = json!(prev + args);
    }
}

fn close_tool_call(e: &mut Emitter, k: &str) {
    let call_id = e.state["funcCallIds"][k].clone();
    if truthy(&call_id) && !truthy(&e.state["funcItemDone"][k]) {
        let args = e.state["funcArgsBuf"][k].as_str().filter(|s| !s.is_empty()).unwrap_or("{}").to_string();
        let name = e.state["funcNames"][k].clone();
        let custom = is_custom_tool(e.state, &name);
        let idx: i64 = k.parse().unwrap_or(0);
        let cid = js_string(&call_id);
        if custom {
            let input = extract_custom_input(&args);
            e.emit("response.custom_tool_call_input.delta", json!({"type": "response.custom_tool_call_input.delta", "item_id": format!("ctc_{cid}"), "output_index": idx, "delta": input}));
            e.emit("response.custom_tool_call_input.done", json!({"type": "response.custom_tool_call_input.done", "item_id": format!("ctc_{cid}"), "output_index": idx, "input": input}));
        } else {
            e.emit("response.function_call_arguments.done", json!({"type": "response.function_call_arguments.done", "item_id": format!("fc_{cid}"), "output_index": idx, "arguments": args}));
        }
        let mut item = json!({"id": format!("{}_{cid}", if custom { "ctc" } else { "fc" }), "type": if custom { "custom_tool_call" } else { "function_call" }});
        if custom {
            item["input"] = json!(extract_custom_input(&args));
        } else {
            item["arguments"] = json!(args);
        }
        item["call_id"] = call_id;
        item["name"] = if truthy(&name) { name } else { json!("") };
        e.emit("response.output_item.done", json!({"type": "response.output_item.done", "output_index": idx, "item": item}));
        record_completed(e.state, idx, item);
        e.state["funcItemDone"][k] = json!(true);
        e.state["funcArgsDone"][k] = json!(true);
    }
}

fn send_completed(e: &mut Emitter) {
    if !truthy(&e.state["completedSent"]) {
        e.state["completedSent"] = json!(true);
        let mut resp = json!({
            "id": e.state["responseId"], "object": "response", "created_at": e.state["created"], "status": "completed",
            "background": false, "error": null, "output": collect_completed(e.state)
        });
        if truthy(&e.state["responsesUsage"]) {
            resp["usage"] = e.state["responsesUsage"].clone();
        }
        e.emit("response.completed", json!({"type": "response.completed", "response": resp}));
    }
}

fn keys_of(v: &Value) -> Vec<String> {
    v.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default()
}

fn flush_responses_events(state: &mut Value) -> Vec<Value> {
    if truthy(&state["completedSent"]) {
        return vec![];
    }
    let mut e = Emitter { state, events: vec![] };
    for k in keys_of(&e.state["msgItemAdded"]) {
        close_message(&mut e, &k);
    }
    close_reasoning(&mut e);
    for k in keys_of(&e.state["funcCallIds"]) {
        close_tool_call(&mut e, &k);
    }
    send_completed(&mut e);
    e.events
}

pub fn openai_to_responses(chunk: Option<&Value>, state: &mut Value) -> Vec<Value> {
    let Some(chunk) = chunk else { return flush_responses_events(state) };
    if let Some(u) = to_responses_usage(&chunk["usage"]) {
        state["responsesUsage"] = u;
    }
    if chunk["choices"].as_array().map(|c| c.is_empty()).unwrap_or(true) {
        return if truthy(&state["completionPending"]) && truthy(&state["responsesUsage"]) { flush_responses_events(state) } else { vec![] };
    }
    let choice = chunk["choices"][0].clone();
    let idx = choice["index"].as_i64().unwrap_or(0);
    let delta = choice["delta"].clone();
    let mut e = Emitter { state, events: vec![] };
    if !truthy(&e.state["started"]) {
        e.state["started"] = json!(true);
        if truthy(&chunk["id"]) {
            e.state["responseId"] = json!(format!("resp_{}", js_string(&chunk["id"])));
        }
        let (rid, created) = (e.state["responseId"].clone(), e.state["created"].clone());
        e.emit("response.created", json!({"type": "response.created", "response": {"id": rid, "object": "response", "created_at": created, "status": "in_progress", "background": false, "error": null, "output": []}}));
        e.emit("response.in_progress", json!({"type": "response.in_progress", "response": {"id": rid, "object": "response", "created_at": created, "status": "in_progress"}}));
    }
    let reasoning = extract_reasoning_text(&delta);
    if !reasoning.is_empty() {
        start_reasoning(&mut e, idx);
        emit_reasoning_delta(&mut e, &reasoning);
    }
    if let Some(c) = delta["content"].as_str().filter(|c| !c.is_empty()) {
        let mut content = c.to_string();
        if content.contains("<think>") {
            e.state["inThinking"] = json!(true);
            content = content.replacen("<think>", "", 1);
            start_reasoning(&mut e, idx);
        }
        if content.contains("</think>") {
            let mut parts = content.splitn(2, "</think>");
            let think = parts.next().unwrap_or("").to_string();
            let text = parts.next().unwrap_or("").to_string();
            if !think.is_empty() {
                emit_reasoning_delta(&mut e, &think);
            }
            close_reasoning(&mut e);
            e.state["inThinking"] = json!(false);
            content = text;
        }
        if truthy(&e.state["inThinking"]) && !content.is_empty() {
            emit_reasoning_delta(&mut e, &content);
            return e.events;
        }
        if !content.is_empty() {
            close_reasoning(&mut e);
            emit_text(&mut e, idx, &content);
        }
    }
    if let Some(tcs) = delta["tool_calls"].as_array().filter(|t| !t.is_empty()) {
        close_reasoning(&mut e);
        close_message(&mut e, &idx.to_string());
        for tc in tcs {
            emit_tool_call(&mut e, tc);
        }
    }
    if truthy(&choice["finish_reason"]) {
        for k in keys_of(&e.state["msgItemAdded"]) {
            close_message(&mut e, &k);
        }
        close_reasoning(&mut e);
        for k in keys_of(&e.state["funcCallIds"]) {
            close_tool_call(&mut e, &k);
        }
        let flush_reaches_us = e.state["targetFormat"] == "openai";
        if truthy(&e.state["responsesUsage"]) || !flush_reaches_us {
            send_completed(&mut e);
        } else {
            e.state["completionPending"] = json!(true);
        }
    }
    e.events
}

// ---------------------------------------------------------------------------
// Responses API events → OpenAI chat chunks
// ---------------------------------------------------------------------------

fn resp_chunk(state: &Value, delta: Value, finish: Value) -> Value {
    let model = if truthy(&state["model"]) { state["model"].clone() } else { json!("unknown") };
    build_chunk(state["chatId"].as_str().unwrap_or(""), state["created"].as_i64().unwrap_or_else(now_s), &model, delta, finish)
}

fn compute_finish(state: &Value) -> Value {
    if state["toolCallIndex"].as_i64().unwrap_or(0) > 0 || truthy(&state["currentToolCallId"]) {
        json!("tool_calls")
    } else {
        json!("stop")
    }
}

pub fn responses_to_openai(chunk: Option<&Value>, state: &mut Value) -> Vec<Value> {
    let Some(chunk) = chunk else {
        if truthy(&state["finishReasonSent"]) || !truthy(&state["started"]) {
            return vec![];
        }
        let fr = compute_finish(state);
        state["finishReasonSent"] = json!(true);
        state["finishReason"] = fr.clone();
        if !truthy(&state["chatId"]) {
            state["chatId"] = json!(format!("chatcmpl-{}", now_ms()));
        }
        let mut c = resp_chunk(state, json!({}), fr);
        if state["usage"].is_object() {
            c["usage"] = state["usage"].clone();
        }
        return vec![c];
    };
    let event = if truthy(&chunk["type"]) { js_string(&chunk["type"]) } else { js_string(&chunk["event"]) };
    let data = if truthy(&chunk["data"]) { &chunk["data"] } else { chunk };
    if !truthy(&state["started"]) {
        state["started"] = json!(true);
        state["chatId"] = json!(format!("chatcmpl-{}", now_ms()));
        state["created"] = json!(now_s());
        state["toolCallIndex"] = json!(0);
        state["currentToolCallId"] = Value::Null;
        if !state["respToolChatIndex"].is_object() {
            state["respToolChatIndex"] = json!({});
        }
        if !state["respToolArgsEmitted"].is_object() {
            state["respToolArgsEmitted"] = json!({});
        }
    }
    let item_type = data["item"]["type"].as_str().unwrap_or("");
    let is_fc = item_type == "function_call" || item_type == "custom_tool_call";
    match event.as_str() {
        "response.output_text.delta" => {
            let d = data["delta"].as_str().unwrap_or("");
            if d.is_empty() {
                return vec![];
            }
            vec![resp_chunk(state, json!({"content": d}), Value::Null)]
        }
        "response.output_item.added" if is_fc => {
            let item = &data["item"];
            let call_id = if truthy(&item["call_id"]) { item["call_id"].clone() } else { json!(fallback_tool_call_id(None)) };
            state["currentToolCallId"] = call_id.clone();
            let k = if truthy(&item["id"]) {
                js_string(&item["id"])
            } else if truthy(&data["item_id"]) {
                js_string(&data["item_id"])
            } else {
                js_string(&call_id)
            };
            let idx = if let Some(i) = state["respToolChatIndex"][&k].as_i64() {
                i
            } else {
                let i = state["toolCallIndex"].as_i64().unwrap_or(0);
                state["toolCallIndex"] = json!(i + 1);
                state["respToolChatIndex"][&k] = json!(i);
                i
            };
            vec![resp_chunk(state, json!({"tool_calls": [{"index": idx, "id": call_id, "type": "function", "function": {"name": item["name"].as_str().unwrap_or(""), "arguments": ""}}]}), Value::Null)]
        }
        "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
            let d = data["delta"].as_str().unwrap_or("");
            if d.is_empty() {
                return vec![];
            }
            let known = data["item_id"].as_str().and_then(|k| state["respToolChatIndex"][k].as_i64());
            let idx = known.unwrap_or_else(|| (state["toolCallIndex"].as_i64().filter(|n| *n != 0).unwrap_or(1) - 1).max(0));
            state["respToolArgsEmitted"][idx.to_string()] = json!(true);
            vec![resp_chunk(state, json!({"tool_calls": [{"index": idx, "function": {"arguments": d}}]}), Value::Null)]
        }
        "response.output_item.done" if is_fc => {
            let k = if truthy(&data["item"]["id"]) { js_string(&data["item"]["id"]) } else { js_string(&data["item_id"]) };
            let idx = state["respToolChatIndex"][&k]
                .as_i64()
                .filter(|i| *i != 0 || !k.is_empty())
                .unwrap_or_else(|| (state["toolCallIndex"].as_i64().filter(|n| *n != 0).unwrap_or(1) - 1).max(0));
            if let Some(args) = data["item"]["arguments"].as_str().filter(|a| !a.is_empty()) {
                if !truthy(&state["respToolArgsEmitted"][idx.to_string()]) {
                    state["respToolArgsEmitted"][idx.to_string()] = json!(true);
                    return vec![resp_chunk(state, json!({"tool_calls": [{"index": idx, "function": {"arguments": args}}]}), Value::Null)];
                }
            }
            vec![]
        }
        "response.completed" | "response.done" => {
            let u = &data["response"]["usage"];
            if u.is_object() {
                let i = u["input_tokens"].as_i64().filter(|n| *n != 0).or_else(|| u["prompt_tokens"].as_i64()).unwrap_or(0);
                let o = u["output_tokens"].as_i64().filter(|n| *n != 0).or_else(|| u["completion_tokens"].as_i64()).unwrap_or(0);
                let cr = u["input_tokens_details"]["cached_tokens"].as_i64().filter(|n| *n != 0).or_else(|| u["cache_read_input_tokens"].as_i64()).unwrap_or(0);
                state["usage"] = build_usage(i, o, i + o, cr, 0, 0);
            }
            if !truthy(&state["finishReasonSent"]) {
                let fr = compute_finish(state);
                state["finishReasonSent"] = json!(true);
                state["finishReason"] = fr.clone();
                let mut c = resp_chunk(state, json!({}), fr);
                if state["usage"].is_object() {
                    c["usage"] = state["usage"].clone();
                }
                return vec![c];
            }
            vec![]
        }
        "error" | "response.failed" => {
            if truthy(&state["finishReasonSent"]) {
                return vec![];
            }
            let err = if truthy(&data["error"]) { data["error"].clone() } else { data["response"]["error"].clone() };
            if truthy(&err) {
                state["error"] = err.clone();
                state["finishReasonSent"] = json!(true);
                let msg = if truthy(&err["message"]) { js_string(&err["message"]) } else { err.to_string() };
                return vec![resp_chunk(state, json!({"content": format!("[Error] {msg}")}), json!("stop"))];
            }
            vec![]
        }
        "response.reasoning_summary_text.delta" => {
            let d = data["delta"].as_str().unwrap_or("");
            if d.is_empty() {
                return vec![];
            }
            vec![resp_chunk(state, reasoning_delta(&json!(d), false), Value::Null)]
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_chunks_to_responses_events() {
        let mut st = init_state("openai-responses");
        st["targetFormat"] = json!("openai");
        let mut evs = vec![];
        evs.extend(openai_to_responses(Some(&json!({"id": "x", "choices": [{"delta": {"content": "hi"}}]})), &mut st));
        evs.extend(openai_to_responses(Some(&json!({"id": "x", "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{}"}}]}}]})), &mut st));
        evs.extend(openai_to_responses(Some(&json!({"id": "x", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]})), &mut st));
        evs.extend(openai_to_responses(Some(&json!({"id": "x", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})), &mut st));
        let names: Vec<&str> = evs.iter().map(|e| e["event"].as_str().unwrap()).collect();
        assert_eq!(names[0], "response.created");
        assert_eq!(*names.last().unwrap(), "response.completed");
        let done = evs.last().unwrap();
        assert_eq!(done["data"]["response"]["usage"]["input_tokens"], 3);
        // matches 9router: the completed snapshot carries only the function_call item
        assert_eq!(done["data"]["response"]["output"].as_array().unwrap().len(), 1);
        assert_eq!(names.len(), 13);
    }

    #[test]
    fn responses_events_to_openai() {
        let mut st = init_state("openai");
        let a = responses_to_openai(Some(&json!({"type": "response.output_item.added", "item": {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "f"}})), &mut st);
        assert_eq!(a[0]["choices"][0]["delta"]["tool_calls"][0]["id"], "c1");
        let b = responses_to_openai(Some(&json!({"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "{}"})), &mut st);
        assert_eq!(b[0]["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        let c = responses_to_openai(Some(&json!({"type": "response.completed", "response": {"usage": {"input_tokens": 5, "output_tokens": 1}}})), &mut st);
        assert_eq!(c[0]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(c[0]["usage"]["prompt_tokens"], 5);
    }
}
