//! Public OpenAI / Anthropic / Gemini / Ollama compatible endpoints.

pub mod gemini;
pub mod models;
pub mod ollama;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use bytes::Bytes;
use serde_json::{Value, json};

use crate::AppState;
use crate::chat::core::{error_response, json_response};
use crate::chat::{self, ClientReq};

pub fn routes() -> Router<AppState> {
    let chat_paths = [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/v1/chat/completions",
        "/v1/messages",
        "/v1/v1/messages",
        "/messages",
        "/v1/responses",
        "/v1/v1/responses",
        "/responses",
        "/codex/responses",
        "/codex/{*rest}",
    ];
    let mut r = Router::new();
    for p in chat_paths {
        r = r.route(p, post(chat_endpoint).options(preflight));
    }
    r.route("/v1/responses/compact", post(responses_compact).options(preflight))
        .route("/v1/messages/count_tokens", post(count_tokens).options(preflight))
        .route("/v1/models", get(models::list).options(preflight))
        .route("/v1", get(models::list).options(preflight))
        .route("/models", get(models::list).options(preflight))
        .route("/v1/models/{*path}", get(models::by_kind).options(preflight))
        .route("/v1beta/models", get(models::gemini_list).options(preflight))
        .route("/v1beta/models/{*path}", post(gemini::generate).options(preflight))
        .route("/api/chat", post(ollama::chat).options(preflight))
        .route("/v1/api/chat", post(ollama::chat).options(preflight))
        .route("/api/tags", get(ollama::tags).options(preflight))
        .route("/v1/embeddings", post(crate::mediaapi::embeddings::handle).options(preflight))
        .route("/v1/images/generations", post(crate::mediaapi::images::handle).options(preflight))
        .route("/v1/audio/speech", post(crate::mediaapi::tts::handle).options(preflight))
        .route("/v1/audio/voices", get(crate::mediaapi::tts::voices).options(preflight))
        .route("/v1/audio/transcriptions", post(crate::mediaapi::stt::handle).options(preflight))
        .route("/v1/search", post(crate::mediaapi::search::handle).options(preflight))
        .route("/v1/web/fetch", post(crate::mediaapi::fetch::handle).options(preflight))
        .route("/v1/videos/{action}", post(crate::mediaapi::video::create).get(crate::mediaapi::video::get).options(preflight))
        .route("/v1/systemone", post(crate::mediaapi::systemone::handle).options(preflight))
        .route("/systemone", post(crate::mediaapi::systemone::handle).options(preflight))
        .route("/health", get(|| async { "ok" }))
        .route("/v1/{*rest}", any(not_found))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
}

async fn preflight() -> Response {
    (
        [
            ("access-control-allow-origin", "*"),
            ("access-control-allow-methods", "GET, POST, OPTIONS"),
            ("access-control-allow-headers", "*"),
        ],
        "",
    )
        .into_response()
}

async fn not_found(method: Method, uri: Uri) -> Response {
    error_response(404, &format!("Unknown endpoint: {method} {}", uri.path()), &[])
}

/// Enforces the "require API key" setting. Ok(key name) when a valid key was sent.
pub fn authorize(st: &AppState, headers: &HeaderMap, query_key: Option<&str>) -> Result<Option<String>, Response> {
    let key = crate::auth::client_key(headers).or_else(|| query_key.map(str::to_owned));
    let name = key.as_deref().and_then(|k| st.db.check_api_key(k));
    let required = crate::auth::api_key_required(st) || crate::chat::accounts::settings(&st.db)["requireApiKey"] == json!(true);
    if required {
        match (&key, &name) {
            (None, _) => return Err(error_response(401, "Missing API key", &[])),
            (Some(_), None) => return Err(error_response(401, "Invalid API key", &[])),
            _ => {}
        }
    }
    Ok(name)
}

pub fn parse_body(bytes: &Bytes) -> Result<Value, Response> {
    match serde_json::from_slice::<Value>(bytes) {
        Ok(v) if v.is_object() => Ok(v),
        _ => Err(error_response(400, "Invalid JSON body", &[])),
    }
}

pub fn client_req(headers: &HeaderMap, endpoint: &str, key_name: Option<String>) -> ClientReq {
    ClientReq { headers: chat::headers_json(headers), endpoint: endpoint.to_string(), api_key: key_name }
}

#[derive(serde::Deserialize, Default)]
pub struct KeyQuery {
    pub key: Option<String>,
}

async fn chat_endpoint(State(st): State<AppState>, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
    let key = match authorize(&st, &headers, None) {
        Ok(k) => k,
        Err(r) => return r,
    };
    let body = match parse_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let mut path = uri.path().replace("/v1/v1/", "/v1/");
    if path.starts_with("/codex") || path == "/responses" {
        path = "/v1/responses".into();
    }
    chat::handle_chat(st.db.clone(), body, client_req(&headers, &path, key)).await
}

async fn responses_compact(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let key = match authorize(&st, &headers, None) {
        Ok(k) => k,
        Err(r) => return r,
    };
    let mut body = match parse_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    body["_compact"] = json!(true);
    chat::handle_chat(st.db.clone(), body, client_req(&headers, "/v1/responses/compact", key)).await
}

fn count_chars(v: &Value) -> usize {
    match v {
        Value::Null => 0,
        Value::String(s) => s.encode_utf16().count(),
        Value::Number(n) => n.to_string().len(),
        Value::Bool(b) => b.to_string().len(),
        Value::Array(a) => a.iter().map(count_chars).sum(),
        Value::Object(o) => o.iter().map(|(k, v)| k.encode_utf16().count() + count_chars(v)).sum(),
    }
}

fn count_block(b: &Value) -> usize {
    match b {
        Value::String(s) => s.encode_utf16().count(),
        Value::Object(_) => match b["type"].as_str() {
            Some("text") => count_chars(&b["text"]),
            Some("tool_use") => count_chars(&b["name"]) + count_chars(&b["input"]),
            Some("tool_result") => count_chars(&b["content"]),
            Some("thinking") => count_chars(&b["thinking"]),
            _ => count_chars(b),
        },
        Value::Null => 0,
        other => count_chars(other),
    }
}

/// estimateAnthropicInputTokens
pub fn estimate_anthropic_input_tokens(body: &Value) -> usize {
    let mut total = count_chars(&body["system"]) + count_chars(&body["tools"]);
    for m in body["messages"].as_array().into_iter().flatten() {
        total += match &m["content"] {
            Value::String(s) => s.encode_utf16().count(),
            Value::Array(a) => a.iter().map(count_block).sum(),
            other => count_chars(other),
        };
    }
    total.div_ceil(4)
}

async fn count_tokens(body: Bytes) -> Response {
    match serde_json::from_slice::<Value>(&body) {
        Ok(b) => json_response(200, &json!({"input_tokens": estimate_anthropic_input_tokens(&b)}), &[]),
        Err(_) => json_response(400, &json!({"error": "Invalid JSON body"}), &[]),
    }
}

pub fn status_only(code: u16) -> Response {
    StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR).into_response()
}

pub type KeyQ = Query<KeyQuery>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimate() {
        let b = json!({"system": "abcd", "messages": [{"role": "user", "content": "12345678"}]});
        assert_eq!(estimate_anthropic_input_tokens(&b), 3);
    }
}
