//! The `/v1` API: OpenAI- and Anthropic-compatible endpoints with fallback.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde_json::{Value, json};

use crate::AppState;
use crate::auth;
use crate::catalog::Format;
use crate::db::{Db, UsageRecord, now};
use crate::pricing;
use crate::router::{Target, available_models};
use crate::translate::{request, response, stream::StreamTranslator};

/// Records a usage row when dropped, so aborted streams are still logged.
struct UsageGuard {
    db: Arc<Db>,
    rec: UsageRecord,
    started: Instant,
}

impl Drop for UsageGuard {
    fn drop(&mut self) {
        self.rec.latency_ms = self.started.elapsed().as_millis() as i64;
        self.rec.cost = pricing::estimate(
            &self.rec.model,
            self.rec.prompt_tokens,
            self.rec.completion_tokens,
        );
        self.db.insert_usage(&self.rec);
    }
}

fn error_response(inbound: Format, status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(response::error_body(inbound, status.as_u16(), message)),
    )
        .into_response()
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle(state, Format::OpenAI, headers, body).await
}

pub async fn messages(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    handle(state, Format::Claude, headers, body).await
}

/// Cheap token estimate for Claude Code's `count_tokens` preflight.
pub async fn count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(r) = check_key(&state, Format::Claude, &headers) {
        return r;
    }
    let chars = String::from_utf8_lossy(&body).chars().count();
    Json(json!({"input_tokens": chars / 4})).into_response()
}

pub async fn models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = check_key(&state, Format::OpenAI, &headers) {
        return r;
    }
    let data: Vec<Value> = available_models(&state.db)
        .into_iter()
        .map(|(id, owner)| json!({"id": id, "object": "model", "created": 0, "owned_by": owner}))
        .collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

#[allow(clippy::result_large_err)]
fn check_key(
    state: &AppState,
    inbound: Format,
    headers: &HeaderMap,
) -> Result<Option<String>, Response> {
    let key = auth::client_key(headers);
    let name = key.as_deref().and_then(|k| state.db.check_api_key(k));
    if auth::api_key_required(state) && name.is_none() {
        return Err(error_response(
            inbound,
            StatusCode::UNAUTHORIZED,
            "Invalid or missing API key",
        ));
    }
    Ok(name)
}

async fn handle(state: AppState, inbound: Format, headers: HeaderMap, body: Bytes) -> Response {
    let key_name = match check_key(&state, inbound, &headers) {
        Ok(k) => k,
        Err(r) => return r,
    };
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                inbound,
                StatusCode::BAD_REQUEST,
                &format!("Invalid JSON body: {e}"),
            );
        }
    };
    let Some(requested) = req.get("model").and_then(Value::as_str).map(str::to_owned) else {
        return error_response(inbound, StatusCode::BAD_REQUEST, "Missing `model`");
    };
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let strategy = state
        .db
        .get_setting("strategy")
        .unwrap_or_else(|| "fallback".into());
    let targets = state.router.resolve(&state.db, &requested, &strategy);
    if targets.is_empty() {
        return error_response(
            inbound,
            StatusCode::NOT_FOUND,
            &format!(
                "No provider connection can serve model `{requested}`. Add one in the dashboard or use `prefix/model`."
            ),
        );
    }

    let anthropic_beta = headers.get("anthropic-beta").cloned();
    let mut last_err: (StatusCode, String) =
        (StatusCode::BAD_GATEWAY, "no upstream attempted".into());

    for target in targets {
        let started = Instant::now();
        let guard_rec = UsageRecord {
            ts: now(),
            api_key: key_name.clone(),
            requested_model: requested.clone(),
            provider: target.conn.provider.clone(),
            connection: target.conn.name.clone(),
            model: target.model.clone(),
            stream,
            ..Default::default()
        };

        let upstream = match send(
            &state,
            &target,
            inbound,
            &req,
            stream,
            anthropic_beta.as_ref(),
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(target = %target.qualified(), "upstream request failed: {e}");
                state
                    .router
                    .cooldown(&target.conn.id, Duration::from_secs(30));
                record_failure(&state.db, guard_rec, started, 502, &e.to_string());
                last_err = (
                    StatusCode::BAD_GATEWAY,
                    format!("{}: {e}", target.qualified()),
                );
                continue;
            }
        };

        let status = upstream.status();
        if !status.is_success() {
            let text = upstream.text().await.unwrap_or_default();
            tracing::warn!(target = %target.qualified(), %status, "upstream error: {}", truncate(&text, 300));
            apply_cooldown(&state, &target, status);
            record_failure(
                &state.db,
                guard_rec,
                started,
                status.as_u16() as i64,
                &truncate(&text, 500),
            );
            last_err = (
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                format!("{}: {}", target.qualified(), upstream_message(&text)),
            );
            continue;
        }

        let mut guard = UsageGuard {
            db: state.db.clone(),
            rec: guard_rec,
            started,
        };
        guard.rec.status = status.as_u16() as i64;
        let target_fmt = target.provider.format;

        if !stream {
            let bytes = match upstream.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    guard.rec.status = 502;
                    guard.rec.error = Some(e.to_string());
                    last_err = (StatusCode::BAD_GATEWAY, e.to_string());
                    continue;
                }
            };
            let Ok(v) = serde_json::from_slice::<Value>(&bytes) else {
                guard.rec.status = 502;
                guard.rec.error = Some("upstream returned non-JSON body".into());
                last_err = (
                    StatusCode::BAD_GATEWAY,
                    "upstream returned non-JSON body".into(),
                );
                continue;
            };
            let (p, c) = response::usage_of(target_fmt, &v);
            guard.rec.prompt_tokens = p;
            guard.rec.completion_tokens = c;
            let out = response::to_inbound(inbound, target_fmt, &v, &requested);
            return Json(out).into_response();
        }

        // Streaming: translate on the fly; the guard logs usage when the stream ends.
        let mut translator = StreamTranslator::new(inbound, target_fmt, &requested);
        let mut upstream_stream = upstream.bytes_stream();
        let body_stream = async_stream::stream! {
            let mut guard = guard;
            while let Some(chunk) = upstream_stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        let out = translator.push(&bytes);
                        guard.rec.prompt_tokens = translator.usage.prompt;
                        guard.rec.completion_tokens = translator.usage.completion;
                        if !out.is_empty() {
                            yield Ok::<Bytes, std::io::Error>(Bytes::from(out));
                        }
                    }
                    Err(e) => {
                        guard.rec.error = Some(format!("stream interrupted: {e}"));
                        break;
                    }
                }
            }
            let tail = translator.finish();
            guard.rec.prompt_tokens = translator.usage.prompt;
            guard.rec.completion_tokens = translator.usage.completion;
            if !tail.is_empty() {
                yield Ok(Bytes::from(tail));
            }
        };
        let mut resp = Response::new(Body::from_stream(body_stream));
        let h = resp.headers_mut();
        h.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        h.insert("x-accel-buffering", HeaderValue::from_static("no"));
        if let Ok(v) = HeaderValue::from_str(&target.qualified()) {
            h.insert("x-fastrouter-target", v);
        }
        return resp;
    }

    error_response(
        inbound,
        last_err.0,
        &format!("All providers failed. Last error — {}", last_err.1),
    )
}

fn record_failure(db: &Arc<Db>, mut rec: UsageRecord, started: Instant, status: i64, err: &str) {
    rec.status = status;
    rec.error = Some(err.to_string());
    rec.latency_ms = started.elapsed().as_millis() as i64;
    db.insert_usage(&rec);
}

fn apply_cooldown(state: &AppState, target: &Target, status: reqwest::StatusCode) {
    let (key, secs) = match status.as_u16() {
        401..=403 => (target.conn.id.clone(), 300),
        429 => (format!("{}|{}", target.conn.id, target.model), 60),
        500..=599 => (format!("{}|{}", target.conn.id, target.model), 15),
        _ => return,
    };
    state.router.cooldown(&key, Duration::from_secs(secs));
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

fn upstream_message(text: &str) -> String {
    serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|v| {
            v["error"]["message"]
                .as_str()
                .or_else(|| v["message"].as_str())
                .or_else(|| v["error"].as_str())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| truncate(text, 300))
}

/// Builds and sends the upstream HTTP request for a target.
pub async fn send(
    state: &AppState,
    target: &Target,
    inbound: Format,
    req: &Value,
    stream: bool,
    anthropic_beta: Option<&HeaderValue>,
) -> Result<reqwest::Response, reqwest::Error> {
    let base = target.base_url();
    let key = target.conn.api_key.trim();
    let mut body = request::to_upstream(inbound, target.provider.format, req, &target.model);

    let builder = match target.provider.format {
        Format::OpenAI => {
            let mut b = state.http.post(format!("{base}/chat/completions"));
            if !key.is_empty() {
                b = b.bearer_auth(key);
            }
            if target.provider.id == "openrouter" {
                b = b
                    .header("HTTP-Referer", "https://github.com/fastrouter")
                    .header("X-Title", "FastRouter");
            }
            b
        }
        Format::Claude => {
            let mut b = state
                .http
                .post(format!("{base}/messages"))
                .header("anthropic-version", "2023-06-01");
            if key.starts_with("sk-ant-oat") {
                // Claude Code subscription OAuth token.
                b = b
                    .bearer_auth(key)
                    .header("anthropic-beta", oauth_beta(anthropic_beta));
                prepend_claude_code_system(&mut body);
            } else {
                if !key.is_empty() {
                    b = b.header("x-api-key", key);
                }
                if let (Some(beta), Format::Claude) = (anthropic_beta, inbound) {
                    b = b.header("anthropic-beta", beta.clone());
                }
            }
            b
        }
        Format::Gemini => {
            let url = if stream {
                format!(
                    "{base}/models/{}:streamGenerateContent?alt=sse",
                    target.model
                )
            } else {
                format!("{base}/models/{}:generateContent", target.model)
            };
            let mut b = state.http.post(url);
            if !key.is_empty() {
                b = b.header("x-goog-api-key", key);
            }
            b
        }
    };
    if stream && target.provider.format != Format::Gemini {
        body["stream"] = json!(true);
    }
    builder.json(&body).send().await
}

fn oauth_beta(client_beta: Option<&HeaderValue>) -> String {
    let mut betas = vec!["oauth-2025-04-20".to_string()];
    if let Some(v) = client_beta.and_then(|v| v.to_str().ok()) {
        betas.extend(
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s != "oauth-2025-04-20"),
        );
    }
    betas.join(",")
}

/// Anthropic only accepts subscription OAuth tokens for Claude Code-shaped requests.
fn prepend_claude_code_system(body: &mut Value) {
    const IDENT: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
    let ident = json!({"type": "text", "text": IDENT});
    let system = match body.get("system").cloned() {
        Some(Value::String(s)) if s.starts_with(IDENT) => return,
        Some(Value::String(s)) => json!([ident, {"type": "text", "text": s}]),
        Some(Value::Array(mut a)) => {
            if a.first()
                .and_then(|b| b["text"].as_str())
                .map(|t| t.starts_with(IDENT))
                .unwrap_or(false)
            {
                return;
            }
            a.insert(0, ident);
            Value::Array(a)
        }
        _ => json!([ident]),
    };
    body["system"] = system;
}

/// Sends a tiny request through a connection; used by the dashboard "Test" button.
pub async fn test_target(state: &AppState, target: &Target) -> Result<String, String> {
    let req = json!({
        "model": target.model,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 16,
    });
    let started = Instant::now();
    let resp = send(state, target, Format::OpenAI, &req, false, None)
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!(
            "HTTP {}: {}",
            status.as_u16(),
            upstream_message(&text)
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|_| "non-JSON response".to_string())?;
    let out = response::to_inbound(Format::OpenAI, target.provider.format, &v, &target.model);
    let reply = request::content_text(&out["choices"][0]["message"]["content"]);
    state.router.clear_cooldown(&target.conn.id);
    Ok(format!(
        "{} replied “{}” in {} ms",
        target.model,
        truncate(reply.trim(), 60),
        started.elapsed().as_millis()
    ))
}

/// Fetches the model list from an upstream connection.
pub async fn fetch_models(state: &AppState, target: &Target) -> Result<Vec<String>, String> {
    let base = target.base_url();
    let key = target.conn.api_key.trim();
    let req = match target.provider.format {
        Format::OpenAI => {
            let b = state.http.get(format!("{base}/models"));
            if key.is_empty() {
                b
            } else {
                b.bearer_auth(key)
            }
        }
        Format::Claude => {
            let b = state
                .http
                .get(format!("{base}/models?limit=100"))
                .header("anthropic-version", "2023-06-01");
            if key.starts_with("sk-ant-oat") {
                b.bearer_auth(key)
                    .header("anthropic-beta", "oauth-2025-04-20")
            } else {
                b.header("x-api-key", key)
            }
        }
        Format::Gemini => state
            .http
            .get(format!("{base}/models?pageSize=1000"))
            .header("x-goog-api-key", key),
    };
    let resp = req.send().await.map_err(|e| e.to_string())?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!(
            "HTTP {}: {}",
            status.as_u16(),
            upstream_message(&text)
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|_| "non-JSON response".to_string())?;
    let mut models: Vec<String> = match target.provider.format {
        Format::Gemini => v["models"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|m| {
                m["supportedGenerationMethods"]
                    .as_array()
                    .map(|a| a.iter().any(|x| x == "generateContent"))
                    .unwrap_or(true)
            })
            .filter_map(|m| {
                m["name"]
                    .as_str()
                    .map(|n| n.trim_start_matches("models/").to_string())
            })
            .collect(),
        _ => v["data"]
            .as_array()
            .or_else(|| v["models"].as_array())
            .into_iter()
            .flatten()
            .filter_map(|m| {
                m["id"]
                    .as_str()
                    .or_else(|| m["name"].as_str())
                    .map(str::to_owned)
            })
            .collect(),
    };
    models.sort();
    models.dedup();
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_code_system_prepended_once() {
        let mut b = json!({"system": "hello"});
        prepend_claude_code_system(&mut b);
        prepend_claude_code_system(&mut b);
        assert_eq!(b["system"].as_array().unwrap().len(), 2);
        assert_eq!(b["system"][1]["text"], "hello");
    }
}
