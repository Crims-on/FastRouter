//! handleChatCore: translate → execute → (refresh) → respond, for one
//! provider/model/account (port of open-sse/handlers/chatCore.js and its
//! streaming / non-streaming / forced-SSE handlers).

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{HeaderValue, StatusCode};
use axum::response::Response;
use futures::StreamExt;
use serde_json::{Map, Value, json};

use super::stream::{self, Mode, StreamOpts, StreamOutcome};
use super::usage::{canonicalize_usage, extract_usage_from_body, filter_usage_for_format};
use super::util::*;
use crate::db::{Db, UsageRecord};
use crate::exec::{ExecArgs, Upstream};
use crate::jsv::{js_string, now_ms, now_s, truthy};
use crate::registry::{self, REG};
use crate::translate::{self, ANTIGRAVITY, CLAUDE, GEMINI, GEMINI_CLI, OPENAI, OPENAI_RESPONSES, ReqCtx};

pub struct CoreArgs {
    pub db: Arc<Db>,
    pub body: Value,
    pub provider: String,
    pub model: String,
    pub creds: Value,
    /// Lower-cased client headers as a JSON object.
    pub headers: Value,
    pub endpoint: String,
    pub source_override: Option<String>,
    pub connection_id: String,
    pub api_key: Option<String>,
    pub requested_model: String,
    pub provider_thinking: Value,
    pub provider_overrides: Value,
}

pub struct CoreResult {
    pub ok: bool,
    pub status: u16,
    pub error: Option<String>,
    pub resets_at_ms: Option<i64>,
    pub response: Response,
    /// Credentials after any refresh during the call.
    pub creds: Value,
}

const FORWARDED: &[&str] = &["retry-after", "x-should-retry"];

pub fn forwarded_headers(h: &reqwest::header::HeaderMap) -> Vec<(String, String)> {
    h.iter()
        .filter(|(k, _)| {
            let k = k.as_str();
            FORWARDED.contains(&k) || k.starts_with("anthropic-ratelimit-")
        })
        .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.as_str().to_string(), v.to_string())))
        .collect()
}

pub fn json_response(status: u16, body: &Value, extra: &[(String, String)]) -> Response {
    let mut r = Response::new(Body::from(body.to_string()));
    *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    let h = r.headers_mut();
    h.insert("content-type", HeaderValue::from_static("application/json"));
    h.insert("access-control-allow-origin", HeaderValue::from_static("*"));
    for (k, v) in extra {
        if let (Ok(k), Ok(v)) = (axum::http::HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(v)) {
            h.insert(k, v);
        }
    }
    r
}

pub fn error_response(status: u16, message: &str, extra: &[(String, String)]) -> Response {
    json_response(status, &error_body(status, message), extra)
}

pub fn sse_response(body: crate::exec::ByteStream, extra: &[(String, String)]) -> Response {
    let mut r = Response::new(Body::from_stream(body.map(|r| r.map_err(std::io::Error::other))));
    let h = r.headers_mut();
    h.insert("content-type", HeaderValue::from_static("text/event-stream"));
    h.insert("cache-control", HeaderValue::from_static("no-cache"));
    h.insert("connection", HeaderValue::from_static("keep-alive"));
    h.insert("access-control-allow-origin", HeaderValue::from_static("*"));
    h.insert("x-accel-buffering", HeaderValue::from_static("no"));
    for (k, v) in extra {
        if let (Ok(k), Ok(v)) = (axum::http::HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(v)) {
            h.insert(k, v);
        }
    }
    r
}

fn err_result(status: u16, msg: String, resets: Option<i64>, extra: &[(String, String)], creds: Value) -> CoreResult {
    CoreResult { ok: false, status, response: error_response(status, &msg, extra), error: Some(msg), resets_at_ms: resets, creds }
}

/// Records one request in the usage table.
pub struct UsageSink {
    pub db: Arc<Db>,
    pub provider: String,
    pub model: String,
    pub requested_model: String,
    pub connection: String,
    pub api_key: Option<String>,
    pub endpoint: String,
    pub stream: bool,
    pub started: Instant,
}

impl UsageSink {
    pub fn record(&self, usage: Option<&Value>, status: u16, error: Option<String>) {
        let c = usage.and_then(canonicalize_usage).unwrap_or_else(|| json!({}));
        let p = c["prompt_tokens"].as_i64().unwrap_or(0);
        let o = c["completion_tokens"].as_i64().unwrap_or(0);
        let cached = c["cached_tokens"].as_i64().unwrap_or(0);
        self.db.insert_usage(&UsageRecord {
            ts: crate::db::now(),
            api_key: self.api_key.clone(),
            requested_model: self.requested_model.clone(),
            provider: self.provider.clone(),
            connection: self.connection.clone(),
            model: self.model.clone(),
            prompt_tokens: p,
            completion_tokens: o,
            cached_tokens: cached,
            status: status as i64,
            latency_ms: self.started.elapsed().as_millis() as i64,
            cost: crate::caps::estimate_cost(&self.provider, &self.model, p, o, cached),
            stream: self.stream,
            endpoint: self.endpoint.clone(),
            error,
        });
    }
}

/// handleChatCore
pub async fn handle_chat_core(a: CoreArgs) -> CoreResult {
    let CoreArgs { db, mut body, provider, model, mut creds, headers, endpoint, source_override, connection_id, api_key, requested_model, provider_thinking, provider_overrides } = a;
    let started = Instant::now();
    let source = source_override.unwrap_or_else(|| detect_format(&body).to_string());
    let alias = REG.alias_of(&provider);
    let model_target = registry::get_model_target_format(&alias, &model);
    let supported = registry::get_model_supported_formats(&alias, &model);
    let runtime_transport = resolve_transport(&provider, &source);
    let use_transport = if supported.as_ref().map(|s| s.iter().any(|f| f == &source)).unwrap_or(true) {
        runtime_transport
    } else {
        // The model only speaks other formats: use the transport serving the
        // model's target format (e.g. Muse Spark → /v1/responses) rather than
        // the provider's default endpoint.
        model_target.as_deref().and_then(|t| resolve_transport(&provider, t))
    };
    let target = use_transport
        .as_ref()
        .and_then(|t| t["format"].as_str().map(str::to_owned))
        .or(model_target)
        .unwrap_or_else(|| get_target_format(&provider, &creds));
    if let Some(t) = &use_transport {
        creds["runtimeTransport"] = t.clone();
    }
    let strip = registry::get_model_strip(&alias, &model);
    let upstream_model = registry::get_model_upstream_id(&alias, &model);

    // Provider-level thinking override (only when the client didn't set one).
    if let Some(mode) = provider_thinking["mode"].as_str().filter(|m| *m != "auto") {
        if mode == "on" && !truthy(&body["thinking"]) {
            body["thinking"] = json!({"type": "enabled", "budget_tokens": 10000});
        } else if mode == "off" && !truthy(&body["thinking"]) {
            body["thinking"] = json!({"type": "disabled"});
        } else if !truthy(&body["reasoning_effort"]) {
            body["reasoning_effort"] = json!(mode);
        }
    }

    let client_streaming = body["stream"] == json!(true) || matches!(source.as_str(), ANTIGRAVITY | GEMINI | GEMINI_CLI);
    let transport = REG.transport(&provider);
    let provider_requires_stream = transport["forceStream"] == json!(true);
    let mut stream = provider_requires_stream || body["stream"] != json!(false);
    let model_type = registry::get_model_type(&alias, &model);
    let lower_model = model.to_lowercase();
    let image_gen = model_type.as_deref() == Some("imageGen") || lower_model.contains("image") || lower_model.contains("imagen");
    if image_gen && (provider == "antigravity" || provider == "gemini-cli") {
        stream = false;
    }
    let client_tool = detect_client_tool(&headers, &body);
    if client_tool == Some("deepseek-tui") && body["stream"] != json!(true) {
        stream = false;
    }
    let accept = headers["accept"].as_str().unwrap_or("");
    if accept.contains("application/json") && !accept.contains("text/event-stream") && body["stream"] != json!(true) && !provider_requires_stream {
        stream = false;
    }
    if matches!(source.as_str(), GEMINI | GEMINI_CLI | ANTIGRAVITY) {
        // `stream` only carried the URL action; it is not a Gemini body field.
        crate::jsv::del(&mut body, "stream");
    }
    let passthrough = is_native_passthrough(client_tool, &provider);
    creds["rawHeaders"] = headers.clone();
    tracing::debug!("FORMAT {source} → {target} | stream={stream} | {provider}/{model}");

    if !passthrough {
        let caps = crate::caps::caps_for(Some(&provider), &model);
        strip_unsupported_modalities(&mut body, &source, &caps);
        prefetch_remote_images(&mut body, &source, &target).await;
    }

    let mut tool_name_map: Option<Map<String, Value>> = None;
    let mut custom_tool_names: Vec<String> = vec![];
    let mut session_id: Option<String> = None;
    let mut tbody: Value;
    if passthrough {
        tbody = body.clone();
        tbody["model"] = json!(translate::thinking::strip_thinking_suffix(&upstream_model));
        if provider == "codex" {
            let mut sfx = json!({});
            translate::thinking::apply_thinking(&source, &upstream_model, &mut sfx, Some(&provider), None);
            if truthy(&sfx["reasoning_effort"]) {
                let mut r = if tbody["reasoning"].is_object() { tbody["reasoning"].clone() } else { json!({}) };
                r["effort"] = sfx["reasoning_effort"].clone();
                tbody["reasoning"] = r;
                crate::jsv::del(&mut tbody, "reasoning_effort");
            }
        }
        if client_tool == Some("claude") {
            let m = js_string(&tbody["model"]);
            translate::claude_fmt::normalize_claude_passthrough(&mut tbody, &m);
        }
    } else {
        let mut rc = ReqCtx {
            provider: provider.clone(),
            api_key: Some(crate::exec::api_key_or_token(&creds)).filter(|s| !s.is_empty()),
            connection_id: Some(connection_id.clone()).filter(|s| !s.is_empty()),
            headers: headers.clone(),
            strip,
            psd: creds["providerSpecificData"].clone(),
            ..Default::default()
        };
        rc.ctx.project_id = creds["projectId"].as_str().map(str::to_owned);
        rc.ctx.email = creds["email"].as_str().map(str::to_owned).or_else(|| creds["providerSpecificData"]["email"].as_str().map(str::to_owned));
        rc.ctx.connection_id = rc.connection_id.clone();
        let t = translate::translate_request(&source, &target, &upstream_model, &body, stream, &mut rc);
        tbody = t.body;
        tool_name_map = t.tool_name_map;
        custom_tool_names = t.custom_tool_names;
        session_id = Some(t.session_id);
        if tbody.is_object() {
            tbody["model"] = json!(translate::thinking::strip_thinking_suffix(&upstream_model));
        }
        strip_continuity_fields(&mut tbody);
    }
    let session_seed = session_id.unwrap_or_else(|| crate::session::resolve_session_id(&headers, &body, Some(&connection_id), &provider));
    creds["_clientSessionId"] = json!(session_seed);

    if tbody["tools"].is_array() {
        let mut tools = tbody["tools"].take();
        let stripped = dedupe_tools(&mut tools, client_tool, &model);
        if !stripped.is_empty() {
            tracing::debug!("TOOLDEDUP stripped {}", stripped.len());
        }
        tbody["tools"] = tools;
    }
    let final_format = if passthrough { source.clone() } else { target.clone() };
    if model_type.as_deref() == Some("tts") {
        if let Some(m) = tbody["messages"].as_array_mut() {
            m.retain(|x| x["role"] != "tool");
            crate::jsv::del(&mut tbody, "tools");
        }
    }
    maybe_default_claude_tool_type(&provider, &final_format, &mut tbody);
    if passthrough && client_tool == Some("claude") {
        translate::claude_fmt::anchor_claude_cache(&mut tbody);
    }

    let ex = crate::providers::get_executor(&provider);
    let override_headers = Some(provider_overrides["headers"].clone()).filter(|h| h.is_object());
    let sink = UsageSink {
        db: db.clone(),
        provider: provider.clone(),
        model: model.clone(),
        requested_model,
        connection: connection_id.clone(),
        api_key: api_key.clone(),
        endpoint,
        stream,
        started,
    };

    let exec_once = |creds: &mut Value| {
        let ex = ex.clone();
        let tbody = tbody.clone();
        let model = model.clone();
        let sid = session_seed.clone();
        let ct = client_tool.map(str::to_owned);
        let oh = override_headers.clone();
        let mut c = std::mem::take(creds);
        async move {
            let r = ex.execute(ExecArgs { model: &model, body: tbody, stream, creds: &mut c, session_id: Some(sid), client_tool: ct, override_headers: oh }).await;
            (r, c)
        }
    };

    let (res, c) = exec_once(&mut creds).await;
    creds = c;
    let mut result = match res {
        Ok(r) => r,
        Err(e) => {
            let msg = format_provider_error(&e, 502);
            tracing::error!("ERROR 502 · {provider}/{model} · {msg}");
            sink.record(None, 502, Some(msg.clone()));
            return err_result(502, msg, None, &[], creds);
        }
    };

    // 401/403 → refresh credentials and retry once.
    if !ex.no_auth() && matches!(result.response.status, 401 | 403) {
        let mut refreshed: Option<Value> = None;
        for attempt in 0..3u64 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(attempt * 1000)).await;
            }
            if let Some(raw) = ex.refresh_credentials(&creds).await {
                if crate::oauth::refresh::is_unrecoverable(&raw) {
                    break;
                }
                let patch = crate::oauth::refresh::merge_refreshed(&provider, &creds, &raw);
                if truthy(&raw["refreshToken"]) && raw["refreshToken"] != creds["refreshToken"] {
                    if truthy(&raw["accessToken"]) {
                        creds["accessToken"] = raw["accessToken"].clone();
                    }
                    creds["refreshToken"] = raw["refreshToken"].clone();
                }
                refreshed = Some(patch);
                break;
            }
        }
        if let Some(patch) = refreshed.filter(|p| truthy(&p["accessToken"]) || truthy(&p["copilotToken"])) {
            tracing::info!("TOKEN REFRESHED · {provider}/{model}");
            let existing = creds["providerSpecificData"].clone();
            crate::oauth::refresh::merge_into(&mut creds, &patch);
            let mut persist = patch.clone();
            persist["testStatus"] = json!("active");
            super::accounts::update_provider_credentials(&db, &connection_id, &persist, &existing);
            let (r2, c2) = exec_once(&mut creds).await;
            creds = c2;
            if let Ok(r2) = r2 {
                if r2.response.ok() {
                    result = r2;
                }
            }
        } else {
            tracing::warn!("{provider} token refresh failed");
        }
    }

    let renamed = creds.as_object_mut().and_then(|o| o.shift_remove("__renamedToolNames")).and_then(|v| v.as_object().cloned());
    let response_format = result.response_format.clone().unwrap_or_else(|| target.clone());
    let up = result.response;
    let fwd = forwarded_headers(&up.headers);

    if !up.ok() {
        let status = up.status;
        let text = up.text().await;
        let (code, message, resets) = parse_upstream_error(status, &text, ex.as_ref());
        let msg = format_provider_error(&message, code);
        tracing::warn!("ERROR {code} · {provider}/{model} · {}", msg.chars().take(300).collect::<String>());
        sink.record(None, code, Some(msg.clone()));
        return err_result(code, msg, resets, &fwd, creds);
    }

    if (!client_streaming && provider_requires_stream) || !stream {
        let resp = non_streaming(up, &provider, &model, &source, &response_format, tool_name_map.as_ref(), renamed.as_ref(), &custom_tool_names, &sink, &fwd).await;
        return match resp {
            Ok(r) => CoreResult { ok: true, status: 200, error: None, resets_at_ms: None, response: r, creds },
            Err((s, m)) => {
                sink.record(None, s, Some(m.clone()));
                err_result(s, m, None, &fwd, creds)
            }
        };
    }

    // Streaming
    let ct = up.content_type().to_lowercase();
    if !ct.is_empty() && !ct.contains("text/event-stream") && !ct.contains("application/json") && !ct.contains("ndjson") && !ct.contains("eventstream") && !ct.contains("octet-stream") {
        let status = up.status;
        let text = up.text().await;
        let title = regex::Regex::new(r"(?i)<title>([^<]+)</title>").unwrap().captures(&text).map(|c| c[1].to_string()).unwrap_or_default();
        let tag = regex::Regex::new(r"<[^>]*>").unwrap();
        let clean = |s: &str| tag.replace_all(s, "").replace(['\r', '\n'], " ").trim().chars().take(160).collect::<String>();
        let short = if !title.is_empty() { clean(&title) } else if text.len() < 200 { clean(&text) } else { format!("Upstream returned non-SSE response ({ct})") };
        let msg = format!("[{status}]: {short}");
        sink.record(None, status, Some(msg.clone()));
        return CoreResult { ok: false, status, error: Some(msg.clone()), resets_at_ms: None, response: json_response(status, &json!({"error": {"message": msg}}), &[]), creds };
    }
    // Upstream ignored `stream` and answered with plain JSON: convert, then replay as SSE.
    if ct.contains("application/json") && !ct.contains("stream") {
        let resp = non_streaming(up, &provider, &model, &source, &response_format, tool_name_map.as_ref(), renamed.as_ref(), &custom_tool_names, &sink, &fwd).await;
        return match resp {
            Ok(r) => {
                let bytes = axum::body::to_bytes(r.into_body(), usize::MAX).await.unwrap_or_default();
                let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
                CoreResult { ok: true, status: 200, error: None, resets_at_ms: None, response: sse_response(json_to_sse(&v, &source), &fwd), creds }
            }
            Err((s, m)) => {
                sink.record(None, s, Some(m.clone()));
                err_result(s, m, None, &fwd, creds)
            }
        };
    }

    let is_responses_provider = transport["format"] == OPENAI_RESPONSES;
    let ua = headers["user-agent"].as_str().unwrap_or("").to_lowercase();
    let droid = ua.contains("droid") || ua.contains("codex-cli");
    let mode = if is_responses_provider && response_format == OPENAI_RESPONSES && !droid {
        let to = match source.as_str() {
            OPENAI_RESPONSES => OPENAI_RESPONSES,
            CLAUDE => CLAUDE,
            ANTIGRAVITY | GEMINI | GEMINI_CLI => ANTIGRAVITY,
            _ => OPENAI,
        };
        Mode::Translate { target: OPENAI_RESPONSES.into(), source: to.into() }
    } else if response_format != source {
        Mode::Translate { target: response_format.clone(), source: source.clone() }
    } else {
        Mode::Passthrough
    };
    let stall = transport["stallTimeoutMs"].as_u64().unwrap_or(0);
    let on_complete: stream::OnComplete = Box::new(move |o: StreamOutcome| {
        sink.record(o.usage.as_ref(), 200, None);
    });
    let s = stream::transform(
        up.body,
        StreamOpts {
            mode,
            provider: provider.clone(),
            model: model.clone(),
            body,
            tool_name_map,
            renamed_tool_names: renamed,
            custom_tool_names,
            session_id: creds["_clientSessionId"].as_str().map(str::to_owned),
            stall_ms: stall,
            on_complete: Some(on_complete),
        },
    );
    CoreResult { ok: true, status: 200, error: None, resets_at_ms: None, response: sse_response(s, &fwd), creds }
}

/// Replays a complete client-format JSON body as an SSE stream.
fn json_to_sse(v: &Value, source: &str) -> crate::exec::ByteStream {
    let openai = translate::nonstream::to_openai(source, v);
    let mut out = String::new();
    if source == OPENAI_RESPONSES {
        // Minimal Responses event sequence around the final object.
        out.push_str(&translate::format_sse(&json!({"event": "response.created", "data": {"type": "response.created", "response": {"id": v["id"], "object": "response", "status": "in_progress", "output": []}}}), source));
        out.push_str(&translate::format_sse(&json!({"event": "response.completed", "data": {"type": "response.completed", "response": v}}), source));
        out.push_str("data: [DONE]\n\n");
    } else {
        let mut state = translate::resp::init_state(source);
        let mut chunks = vec![];
        let msg = &openai["choices"][0]["message"];
        let id = openai["id"].clone();
        let mk = |delta: Value, fin: Value| json!({"id": id, "object": "chat.completion.chunk", "created": now_s(), "model": openai["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]});
        chunks.push(mk(json!({"role": "assistant"}), Value::Null));
        if let Some(r) = msg["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
            chunks.push(mk(json!({"reasoning_content": r}), Value::Null));
        }
        if let Some(c) = msg["content"].as_str().filter(|s| !s.is_empty()) {
            chunks.push(mk(json!({"content": c}), Value::Null));
        }
        for (i, tc) in msg["tool_calls"].as_array().into_iter().flatten().enumerate() {
            chunks.push(mk(json!({"tool_calls": [{"index": i, "id": tc["id"], "type": "function", "function": tc["function"]}]}), Value::Null));
        }
        let mut last = mk(json!({}), openai["choices"][0]["finish_reason"].clone());
        if openai["usage"].is_object() {
            last["usage"] = openai["usage"].clone();
        }
        chunks.push(last);
        for c in chunks {
            for item in translate::translate_response(OPENAI, source, Some(&c), &mut state) {
                out.push_str(&translate::format_sse(&item, source));
            }
        }
        for item in translate::translate_response(OPENAI, source, None, &mut state) {
            out.push_str(&translate::format_sse(&item, source));
        }
        if source != CLAUDE {
            out.push_str("data: [DONE]\n\n");
        }
    }
    let b = bytes::Bytes::from(out);
    futures::stream::once(async move { Ok(b) }).boxed()
}

/// Collects an SSE upstream (in `target` format) into one OpenAI chat.completion.
pub fn sse_to_openai(raw: &str, target: &str, fallback_model: &str) -> Result<Value, Value> {
    // Responses upstream: the terminal event carries the whole response.
    if target == OPENAI_RESPONSES {
        let mut parser = crate::sse::EventParser::default();
        let mut evs = parser.push(raw.as_bytes());
        evs.extend(parser.finish());
        for e in evs.iter().rev() {
            if let Ok(v) = serde_json::from_str::<Value>(&e.data) {
                let ty = e.event.clone().or_else(|| v["type"].as_str().map(str::to_owned)).unwrap_or_default();
                if (ty == "response.completed" || ty == "response.done") && v["response"].is_object() {
                    return Ok(json!({"__responses": v["response"]}));
                }
                if ty == "response.failed" || ty == "error" {
                    return Err(if v["response"]["error"].is_object() { v["response"]["error"].clone() } else if v["error"].is_object() { v["error"].clone() } else { v });
                }
            }
        }
    }
    let mut agg = translate::nonstream::ChunkAggregator::default();
    let mut state = translate::resp::init_state(OPENAI);
    let mut lines = crate::sse::LineParser::default();
    let mut all = lines.push(raw.as_bytes());
    all.extend(lines.finish());
    for l in all {
        let Some(p) = crate::sse::parse_sse_line(&l, Some(target)) else { continue };
        if p.is_done() && target != translate::OLLAMA {
            continue;
        }
        let v = p.into_value();
        if truthy(&v["error"]) && target != CLAUDE {
            return Err(v["error"].clone());
        }
        if v["type"] == "error" {
            return Err(v["error"].clone());
        }
        let items = if target == OPENAI { vec![v] } else { translate::translate_response(target, OPENAI, Some(&v), &mut state) };
        for it in items {
            agg.push(&it);
        }
    }
    if target != OPENAI {
        for it in translate::translate_response(target, OPENAI, None, &mut state) {
            agg.push(&it);
        }
    }
    if let Some(e) = agg.error.take() {
        return Err(e);
    }
    if agg.is_empty() {
        return Err(json!({"message": "Invalid SSE response for non-streaming request"}));
    }
    Ok(agg.finish(fallback_model))
}

#[allow(clippy::too_many_arguments)]
async fn non_streaming(
    up: Upstream,
    provider: &str,
    model: &str,
    source: &str,
    target: &str,
    tool_name_map: Option<&Map<String, Value>>,
    renamed: Option<&Map<String, Value>>,
    custom: &[String],
    sink: &UsageSink,
    fwd: &[(String, String)],
) -> Result<Response, (u16, String)> {
    let ct = up.content_type().to_lowercase();
    let bytes = up.bytes().await.map_err(|e| (502, format!("upstream read failed: {e}")))?;
    let text = String::from_utf8_lossy(&bytes);
    // `openai` here means "already converted to an OpenAI chat.completion".
    let (mut body, body_fmt) = if ct.contains("text/event-stream") || (target != OPENAI && text.trim_start().starts_with("data:")) {
        match sse_to_openai(&text, target, model) {
            Ok(v) if v.get("__responses").is_some() => (v["__responses"].clone(), OPENAI_RESPONSES.to_string()),
            Ok(v) => (v, OPENAI.to_string()),
            Err(e) => {
                let m = if e["message"].is_string() { js_string(&e["message"]) } else { e.to_string() };
                return Err((502, m));
            }
        }
    } else {
        match serde_json::from_slice::<Value>(&bytes) {
            Ok(v) => (v, target.to_string()),
            Err(_) => {
                // NDJSON (ollama) bodies.
                if target == translate::OLLAMA {
                    match sse_to_openai(&text, target, model) {
                        Ok(v) => (v, OPENAI.to_string()),
                        Err(_) => return Err((502, format!("Invalid JSON response from {provider}"))),
                    }
                } else {
                    return Err((502, format!("Invalid JSON response from {provider}")));
                }
            }
        }
    };
    if REG.transport(provider)["quirks"]["clineEnvelope"] == json!(true) && body["success"] == json!(true) && body["data"].is_object() {
        body = body["data"].clone();
    }
    body = crate::cloak::decloak_tool_names(body, tool_name_map);
    let usage = extract_usage_from_body(&body);
    let mut out = if body_fmt == source {
        body
    } else {
        let o = if body_fmt == OPENAI { body } else { translate::nonstream::to_openai(&body_fmt, &body) };
        if source == OPENAI { o } else { translate::nonstream::from_openai(source, &o, custom) }
    };
    let is_claude_msg = source == CLAUDE && out["type"] == "message";
    let is_resp = source == OPENAI_RESPONSES && out["object"] == "response";
    if let Some(choice) = out["choices"].get_mut(0) {
        if choice["message"]["tool_calls"].as_array().map(|a| !a.is_empty()).unwrap_or(false) && choice["finish_reason"] != "tool_calls" {
            choice["finish_reason"] = json!("tool_calls");
        }
    }
    if !is_claude_msg && !is_resp && out.is_object() {
        if !truthy(&out["object"]) {
            out["object"] = json!("chat.completion");
        }
        if !truthy(&out["created"]) {
            out["created"] = json!(now_s());
        }
        crate::jsv::del(&mut out, "prompt_filter_results");
        for c in out["choices"].as_array_mut().into_iter().flatten() {
            crate::jsv::del(c, "content_filter_results");
            if truthy(&c["message"]["reasoning_content"]) && truthy(&c["message"]["content"]) {
                crate::jsv::del(&mut c["message"], "reasoning_content");
            }
        }
    }
    if out["usage"].is_object() {
        out["usage"] = filter_usage_for_format(&out["usage"], source);
    }
    if let Some(r) = renamed {
        crate::providers::opencode::restore_tool_names(&mut out, r);
    }
    sink.record(usage.as_ref(), 200, None);
    let _ = now_ms;
    Ok(json_response(200, &out, fwd))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_collect_claude() {
        let raw = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\",\"usage\":{\"input_tokens\":3}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n";
        let v = sse_to_openai(raw, CLAUDE, "x").unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "ok");
        let r = sse_to_openai("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"output\":[]}}\n\n", OPENAI_RESPONSES, "x").unwrap();
        assert_eq!(r["__responses"]["id"], "r");
    }
}
