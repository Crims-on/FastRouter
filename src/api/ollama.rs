//! Ollama-compatible `/api/chat` and `/api/tags` (port of
//! src/app/api/v1/api/chat/route.js + utils/ollamaTransform.js).

use std::collections::BTreeMap;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use super::{authorize, client_req, parse_body};
use crate::AppState;
use crate::chat::core::json_response;

fn ndjson(v: &Value) -> String {
    format!("{v}\n")
}

fn end_line(model: &str) -> String {
    ndjson(&json!({"model": model, "message": {"role": "assistant", "content": ""}, "done": true}))
}

/// Ollama messages carry `images: [base64]`; turn them into OpenAI image parts.
fn ollama_to_openai_messages(body: &mut Value) {
    for m in body["messages"].as_array_mut().into_iter().flatten() {
        let Some(imgs) = m["images"].as_array().cloned().filter(|a| !a.is_empty()) else { continue };
        let text = m["content"].as_str().unwrap_or("").to_string();
        let mut parts = vec![json!({"type": "text", "text": text})];
        for i in imgs {
            if let Some(s) = i.as_str() {
                let url = if s.starts_with("data:") || s.starts_with("http") { s.to_string() } else { format!("data:image/png;base64,{s}") };
                parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
            }
        }
        m["content"] = Value::Array(parts);
        crate::jsv::del(m, "images");
    }
    if let Some(o) = body["options"].as_object().cloned() {
        if let Some(t) = o.get("temperature") {
            body["temperature"] = t.clone();
        }
        if let Some(t) = o.get("top_p") {
            body["top_p"] = t.clone();
        }
        if let Some(n) = o.get("num_predict") {
            body["max_tokens"] = n.clone();
        }
    }
    crate::jsv::del(body, "options");
}

fn to_ollama(resp: Response, model: String) -> Response {
    let ok = resp.status().is_success();
    let ct = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    if !ok {
        return resp;
    }
    let mut body = resp.into_body().into_data_stream();
    let is_sse = ct.contains("event-stream");
    let s = async_stream::stream! {
        if !is_sse {
            // Non-streaming OpenAI completion → single Ollama response.
            let mut buf = Vec::new();
            while let Some(Ok(c)) = body.next().await {
                buf.extend_from_slice(&c);
            }
            let v: Value = serde_json::from_slice(&buf).unwrap_or(Value::Null);
            let msg = &v["choices"][0]["message"];
            let mut m = json!({"role": "assistant", "content": msg["content"].as_str().unwrap_or("")});
            if let Some(tcs) = msg["tool_calls"].as_array().filter(|a| !a.is_empty()) {
                m["tool_calls"] = Value::Array(tcs.iter().map(|tc| json!({"function": {"name": tc["function"]["name"], "arguments": serde_json::from_str::<Value>(tc["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}))}})).collect());
            }
            let u = &v["usage"];
            let out = json!({"model": model, "created_at": crate::jsv::iso_from_ms(crate::jsv::now_ms()), "message": m, "done": true, "done_reason": v["choices"][0]["finish_reason"].as_str().unwrap_or("stop"), "prompt_eval_count": u["prompt_tokens"].as_i64().unwrap_or(0), "eval_count": u["completion_tokens"].as_i64().unwrap_or(0)});
            yield Ok::<Bytes, String>(Bytes::from(ndjson(&out)));
            return;
        }
        let mut lines = crate::sse::LineParser::default();
        let mut pending: BTreeMap<i64, (String, String)> = BTreeMap::new();
        let mut done = false;
        while let Some(chunk) = body.next().await {
            let Ok(chunk) = chunk else { break };
            let mut out = String::new();
            for l in lines.push(&chunk) {
                let Some(data) = l.strip_prefix("data:").map(str::trim) else { continue };
                if data == "[DONE]" {
                    if !done {
                        out.push_str(&end_line(&model));
                        done = true;
                    }
                    continue;
                }
                let Ok(p) = serde_json::from_str::<Value>(data) else { continue };
                let d = &p["choices"][0]["delta"];
                for tc in d["tool_calls"].as_array().into_iter().flatten() {
                    let e = pending.entry(tc["index"].as_i64().unwrap_or(0)).or_default();
                    if let Some(n) = tc["function"]["name"].as_str() { e.0.push_str(n); }
                    if let Some(a) = tc["function"]["arguments"].as_str() { e.1.push_str(a); }
                }
                if let Some(c) = d["content"].as_str().filter(|s| !s.is_empty()) {
                    out.push_str(&ndjson(&json!({"model": model, "message": {"role": "assistant", "content": c}, "done": false})));
                }
                let fin = p["choices"][0]["finish_reason"].as_str().unwrap_or("");
                if fin == "tool_calls" || fin == "stop" {
                    if !pending.is_empty() {
                        let calls: Vec<Value> = std::mem::take(&mut pending).into_values().map(|(n, a)| json!({"function": {"name": n, "arguments": serde_json::from_str::<Value>(if a.is_empty() { "{}" } else { &a }).unwrap_or(json!({}))}})).collect();
                        out.push_str(&ndjson(&json!({"model": model, "message": {"role": "assistant", "content": "", "tool_calls": calls}, "done": true})));
                        done = true;
                    } else if fin == "stop" && !done {
                        out.push_str(&end_line(&model));
                        done = true;
                    }
                }
            }
            if !out.is_empty() {
                yield Ok(Bytes::from(out));
            }
        }
        if !done {
            yield Ok(Bytes::from(end_line(&model)));
        }
    };
    let mut r = Response::new(axum::body::Body::from_stream(s.map(|r| r.map_err(std::io::Error::other))));
    r.headers_mut().insert("content-type", "application/x-ndjson".parse().unwrap());
    r.headers_mut().insert("access-control-allow-origin", "*".parse().unwrap());
    r
}

pub async fn chat(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let key = match authorize(&st, &headers, None) {
        Ok(k) => k,
        Err(r) => return r,
    };
    let mut body = match parse_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let model = body["model"].as_str().unwrap_or("llama3.2").to_string();
    ollama_to_openai_messages(&mut body);
    let resp = crate::chat::handle_chat(st.db.clone(), body, client_req(&headers, "/api/chat", key)).await;
    to_ollama(resp, model)
}

pub async fn tags(State(st): State<AppState>) -> Response {
    let models = super::models::build_models_list(&st.db, &[super::models::LLM], true).await;
    let mut list: Vec<Value> = models
        .iter()
        .filter_map(|m| m["id"].as_str())
        .map(|id| json!({"name": id, "model": id, "modified_at": "2025-12-26T00:00:00Z", "size": 0, "digest": "", "details": {"format": "api", "family": id.split('/').next().unwrap_or(""), "parameter_size": "", "quantization_level": ""}}))
        .collect();
    if list.is_empty() {
        list = vec![
            json!({"name": "llama3.2", "modified_at": "2025-12-26T00:00:00Z", "size": 2000000000u64, "digest": "abc123def456", "details": {"format": "gguf", "family": "llama", "parameter_size": "3B", "quantization_level": "Q4_K_M"}}),
            json!({"name": "qwen2.5", "modified_at": "2025-12-26T00:00:00Z", "size": 4000000000u64, "digest": "def456abc123", "details": {"format": "gguf", "family": "qwen", "parameter_size": "7B", "quantization_level": "Q4_K_M"}}),
        ];
    }
    json_response(200, &json!({"models": list}), &[])
}
