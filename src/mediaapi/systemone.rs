//! POST /v1/systemone (port of systemone.js + systemoneCore.js).

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use bytes::Bytes;
use serde_json::{Value, json};

use super::*;
use crate::AppState;

async fn core(ctx: Ctx, body: Value) -> MediaResult {
    let cfg = media_cfg(&ctx.provider, "systemoneConfig");
    let url = ctx.creds["providerSpecificData"]["baseUrl"].as_str().filter(|s| !s.is_empty()).map(str::to_owned).or_else(|| cfg["baseUrl"].as_str().map(str::to_owned));
    let Some(url) = url else {
        return MediaResult::err(400, format!("Provider '{}' does not support System One.", ctx.provider));
    };
    let mut req = body.clone();
    req["model"] = json!(ctx.model);
    let token = key_of(&ctx.creds);
    let mut rb = client(&ctx.creds).post(url).header("content-type", "application/json").timeout(std::time::Duration::from_secs(60));
    if !token.is_empty() {
        rb = rb.bearer_auth(token);
    }
    for (k, v) in cfg["headers"].as_object().into_iter().flatten() {
        rb = rb.header(k.as_str(), crate::jsv::js_string(v));
    }
    rb = rb.header("x-opencode-session", crate::providers::opencode::generate_session_id(crate::jsv::now_ms()));
    let r = match rb.body(req.to_string()).send().await {
        Ok(r) => r,
        Err(e) => return provider_err(502, &e.to_string()),
    };
    if !r.status().is_success() {
        return upstream_error(r).await;
    }
    let Ok(v) = r.json::<Value>().await else { return MediaResult::err(502, format!("Invalid JSON response from {}", ctx.provider)) };
    let mut res = MediaResult::json(&v);
    if v["usage"].is_object() {
        res.usage = Some(json!({"prompt_tokens": v["usage"]["input_tokens"].as_i64().unwrap_or(0), "completion_tokens": v["usage"]["output_tokens"].as_i64().unwrap_or(0)}));
    }
    res
}

pub async fn handle(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let key = match crate::api::authorize(&st, &headers, None) {
        Ok(k) => k,
        Err(r) => return r,
    };
    let body = match crate::api::parse_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let Some(model_str) = body["model"].as_str().filter(|s| !s.is_empty()).map(str::to_owned) else {
        return error_response(400, "Missing model", &[]);
    };
    if body["state"].is_null() {
        return error_response(400, "Missing required field: state", &[]);
    }
    if !body["questions"].is_object() {
        return error_response(400, "Missing required field: questions", &[]);
    }
    let Some((provider, model)) = resolve_model(&st.db, &model_str) else {
        return error_response(400, "Invalid model format", &[]);
    };
    let started = std::time::Instant::now();
    let db = st.db.clone();
    with_accounts(st.db.clone(), &provider, &model, false, None, |ctx| {
        let (b, db, key, ms) = (body.clone(), db.clone(), key.clone(), model_str.clone());
        async move {
            let (p, m, c) = (ctx.provider.clone(), ctx.model.clone(), ctx.connection_id.clone());
            let r = core(ctx, b).await;
            if r.ok {
                if let Some(u) = &r.usage {
                    crate::chat::core::UsageSink { db, provider: p, model: m, requested_model: ms, connection: c, api_key: key, endpoint: "/v1/systemone".into(), stream: false, started }.record(Some(u), 200, None);
                }
            }
            r
        }
    })
    .await
}
