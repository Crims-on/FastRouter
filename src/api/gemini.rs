//! `/v1beta/models/{model}:generateContent|streamGenerateContent` (port of
//! src/app/api/v1beta/models/[...path]/route.js). The Gemini body goes through
//! the regular translator (tools and multimodal parts included) instead of
//! the lossy text-only conversion upstream does.

use std::collections::HashSet;

use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use super::{KeyQuery, authorize, client_req, parse_body};
use crate::AppState;
use crate::chat::accounts::{self, Selection};
use crate::chat::core::{json_response, sse_response};
use crate::jsv::truthy;
use crate::registry::REG;

fn finish_map(f: &str) -> &'static str {
    match f {
        "length" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        _ => "STOP",
    }
}

/// One OpenAI chunk → Gemini streaming chunk.
fn openai_chunk_to_gemini(p: &Value, model: &str, tools: &mut std::collections::BTreeMap<i64, (String, String)>) -> Option<Value> {
    let choice = p["choices"].get(0)?;
    let d = &choice["delta"];
    let mut parts = vec![];
    if let Some(r) = d["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
        parts.push(json!({"text": r, "thought": true}));
    }
    if let Some(c) = d["content"].as_str().filter(|s| !s.is_empty()) {
        parts.push(json!({"text": c}));
    }
    for tc in d["tool_calls"].as_array().into_iter().flatten() {
        let e = tools.entry(tc["index"].as_i64().unwrap_or(0)).or_default();
        if let Some(n) = tc["function"]["name"].as_str() {
            e.0.push_str(n);
        }
        if let Some(a) = tc["function"]["arguments"].as_str() {
            e.1.push_str(a);
        }
    }
    let fin = choice["finish_reason"].as_str().filter(|s| !s.is_empty());
    if fin.is_some() {
        for (_, (name, args)) in std::mem::take(tools) {
            let a: Value = serde_json::from_str(&args).unwrap_or(json!({}));
            parts.push(json!({"functionCall": {"name": name, "args": a}}));
        }
    }
    if parts.is_empty() && fin.is_none() {
        return None;
    }
    let mut cand = json!({"content": {"role": "model", "parts": if parts.is_empty() { json!([{"text": ""}]) } else { json!(parts) }}, "index": 0});
    if let Some(f) = fin {
        cand["finishReason"] = json!(finish_map(f));
    }
    let mut out = json!({"candidates": [cand]});
    if fin.is_some() && p["usage"].is_object() {
        let u = &p["usage"];
        out["usageMetadata"] = json!({
            "promptTokenCount": u["prompt_tokens"].as_i64().unwrap_or(0),
            "candidatesTokenCount": u["completion_tokens"].as_i64().unwrap_or(0),
            "totalTokenCount": u["total_tokens"].as_i64().unwrap_or(0),
        });
        if let Some(r) = u["completion_tokens_details"]["reasoning_tokens"].as_i64().filter(|n| *n > 0) {
            out["usageMetadata"]["thoughtsTokenCount"] = json!(r);
        }
        out["modelVersion"] = if truthy(&p["model"]) { p["model"].clone() } else { json!(model) };
    }
    Some(out)
}

fn to_gemini_sse(resp: Response, model: String) -> Response {
    if !resp.status().is_success() {
        return resp;
    }
    let is_sse = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).map(|c| c.contains("event-stream")).unwrap_or(false);
    if !is_sse {
        return resp;
    }
    let mut body = resp.into_body().into_data_stream();
    let s = async_stream::stream! {
        let mut lines = crate::sse::LineParser::default();
        let mut tools = std::collections::BTreeMap::new();
        while let Some(chunk) = body.next().await {
            let Ok(chunk) = chunk else { break };
            let mut out = String::new();
            for l in lines.push(&chunk) {
                let Some(data) = l.strip_prefix("data:").map(str::trim) else { continue };
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let Ok(p) = serde_json::from_str::<Value>(data) else { continue };
                let g = if p["candidates"].is_array() {
                    Some(p)
                } else if p["response"]["candidates"].is_array() {
                    Some(p["response"].clone())
                } else if p["error"].is_object() {
                    Some(json!({"error": p["error"]}))
                } else {
                    openai_chunk_to_gemini(&p, &model, &mut tools)
                };
                if let Some(g) = g {
                    out.push_str(&format!("data: {g}\r\n\r\n"));
                }
            }
            if !out.is_empty() {
                yield Ok::<Bytes, String>(Bytes::from(out));
            }
        }
    };
    let mut r = sse_response(s.boxed(), &[]);
    r.headers_mut().remove("connection");
    r
}

fn tts_model_ids() -> HashSet<String> {
    let mut s = HashSet::new();
    for m in REG.provider_models("gemini") {
        if m["kind"] == "tts" || m["type"] == "tts" {
            if let Some(i) = m["id"].as_str() {
                s.insert(i.to_string());
            }
        }
    }
    for m in REG.provider_models("gemini-tts-models") {
        if let Some(i) = m["id"].as_str() {
            s.insert(i.to_string());
        }
    }
    s
}

fn normalize_native(m: &str) -> String {
    let m = m.strip_prefix("models/").unwrap_or(m);
    m.strip_prefix("gemini/").unwrap_or(m).to_string()
}

fn is_native_tts(model: &str, body: &Value) -> bool {
    if model.contains('/') && !model.starts_with("gemini/") && !model.starts_with("models/") {
        return false;
    }
    let audio = body["generationConfig"]["responseModalities"].as_array().map(|a| a.iter().any(|m| m.as_str().map(|s| s.eq_ignore_ascii_case("audio")).unwrap_or(false))).unwrap_or(false);
    audio || tts_model_ids().contains(&normalize_native(model))
}

async fn forward_native(st: &AppState, body: &Value, model: &str, action: &str, query: &str) -> Response {
    let id = normalize_native(model);
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || "_.:-".contains(c)) {
        return json_response(400, &json!({"error": {"message": "Invalid model"}}), &[]);
    }
    let mut exclude = HashSet::new();
    let mut last: Option<(u16, String)> = None;
    loop {
        let creds = match accounts::get_provider_credentials(&st.db, "gemini", &exclude, Some(&id), None).await {
            Selection::Creds(c) => c,
            _ => {
                let (s, m) = last.unwrap_or((503, "No active credentials for provider: gemini".into()));
                return json_response(s, &json!({"error": {"message": m}}), &[]);
            }
        };
        let cid = creds["connectionId"].as_str().unwrap_or("").to_string();
        let qs: Vec<&str> = query.split('&').filter(|p| !p.is_empty() && !p.starts_with("key=")).collect();
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{id}{action}{}", if qs.is_empty() { String::new() } else { format!("?{}", qs.join("&")) });
        let mut rb = crate::exec::client_for(&creds).post(&url).header("content-type", "application/json").timeout(std::time::Duration::from_secs(300)).body(body.to_string());
        if let Some(k) = creds["apiKey"].as_str().filter(|s| !s.is_empty()) {
            rb = rb.header("x-goog-api-key", k);
        } else if let Some(t) = creds["accessToken"].as_str().filter(|s| !s.is_empty()) {
            rb = rb.bearer_auth(t);
        } else {
            return json_response(404, &json!({"error": {"message": "No Gemini API key configured"}}), &[]);
        }
        match rb.send().await {
            Ok(r) if r.status().is_success() => {
                accounts::clear_account_error(&st.db, &cid, &creds["_connection"], Some(&id));
                let ct = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("application/json").to_string();
                let up = crate::exec::Upstream::from_reqwest(r);
                let mut resp = Response::new(axum::body::Body::from_stream(up.body.map(|r| r.map_err(std::io::Error::other))));
                resp.headers_mut().insert("content-type", ct.parse().unwrap());
                resp.headers_mut().insert("access-control-allow-origin", "*".parse().unwrap());
                return resp;
            }
            Ok(r) => {
                let s = r.status().as_u16();
                let t = r.text().await.unwrap_or_default();
                if accounts::mark_account_unavailable(&st.db, &cid, s, &t, "gemini", Some(&id), None).should_fallback {
                    exclude.insert(cid);
                    last = Some((s, t));
                    continue;
                }
                let mut resp = Response::new(axum::body::Body::from(t));
                *resp.status_mut() = axum::http::StatusCode::from_u16(s).unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
                resp.headers_mut().insert("access-control-allow-origin", "*".parse().unwrap());
                return resp;
            }
            Err(e) => {
                let s = if e.is_timeout() { 504 } else { 502 };
                let t = e.to_string();
                if accounts::mark_account_unavailable(&st.db, &cid, s, &t, "gemini", Some(&id), None).should_fallback {
                    exclude.insert(cid);
                    last = Some((s, t));
                    continue;
                }
                return json_response(s, &json!({"error": {"message": t}}), &[]);
            }
        }
    }
}

pub async fn generate(State(st): State<AppState>, Path(path): Path<String>, Query(q): Query<KeyQuery>, uri: axum::http::Uri, headers: HeaderMap, body: Bytes) -> Response {
    let key = match authorize(&st, &headers, q.key.as_deref()) {
        Ok(k) => k,
        Err(r) => return r,
    };
    let mut body = match parse_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let stream = path.contains(":streamGenerateContent");
    let action = if stream { ":streamGenerateContent" } else { ":generateContent" };
    let model = path.replace(":streamGenerateContent", "").replace(":generateContent", "");
    let model = model.strip_prefix("models/").unwrap_or(&model).to_string();
    if is_native_tts(&model, &body) {
        return forward_native(&st, &body, &model, action, uri.query().unwrap_or("")).await;
    }
    body["model"] = json!(model);
    body["stream"] = json!(stream);
    let resp = crate::chat::handle_chat(st.db.clone(), body, client_req(&headers, "/v1beta/models", key)).await;
    if stream { to_gemini_sse(resp, model) } else { resp }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_conversion() {
        let mut t = Default::default();
        let g = openai_chunk_to_gemini(&json!({"choices": [{"delta": {"content": "hi"}}]}), "m", &mut t).unwrap();
        assert_eq!(g["candidates"][0]["content"]["parts"][0]["text"], "hi");
        assert!(openai_chunk_to_gemini(&json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "{\"a\":"}}]}}]}), "m", &mut t).is_none());
        let g = openai_chunk_to_gemini(&json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"}]}), "m", &mut t).unwrap();
        assert_eq!(g["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["a"], 1);
        assert!(is_native_tts("gemini-2.5-flash", &json!({"generationConfig": {"responseModalities": ["AUDIO"]}})));
    }
}
