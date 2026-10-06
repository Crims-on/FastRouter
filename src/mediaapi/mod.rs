//! Non-chat endpoints: embeddings, images, speech, transcription, search,
//! fetch and video (ports of 9router's media handlers). They share one
//! account-rotation loop and a refresh-on-401 request helper.

pub mod embeddings;
pub mod fetch;
pub mod images;
pub mod search;
pub mod stt;
pub mod systemone;
pub mod tts;
pub mod video;

use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;

use axum::response::Response;
use serde_json::{Value, json};

use crate::chat::accounts::{self, Selection};
use crate::chat::core::{error_response, json_response};
use crate::db::Db;
use crate::jsv::truthy;

pub struct MediaResult {
    pub ok: bool,
    pub status: u16,
    pub error: Option<String>,
    pub response: Response,
    pub usage: Option<Value>,
}

impl MediaResult {
    pub fn ok(response: Response) -> Self {
        MediaResult { ok: true, status: 200, error: None, response, usage: None }
    }
    pub fn err(status: u16, msg: impl Into<String>) -> Self {
        let m = msg.into();
        MediaResult { ok: false, status, response: error_response(status, &m, &[]), error: Some(m), usage: None }
    }
    pub fn json(v: &Value) -> Self {
        Self::ok(json_response(200, v, &[]))
    }
}

/// `[provider/model] message` style upstream error (formatProviderError).
pub fn provider_err(status: u16, message: &str) -> MediaResult {
    MediaResult::err(status, crate::chat::util::format_provider_error(message, status))
}

/// Resolves `provider/model` (aliases, custom nodes) like chat does.
pub fn resolve_model(db: &Db, model_str: &str) -> Option<(String, String)> {
    match crate::chat::get_model_info(db, model_str)? {
        crate::chat::ModelInfo::Provider { provider, model } => Some((provider, model)),
        crate::chat::ModelInfo::Combo(_) => None,
    }
}

pub struct Ctx {
    pub db: Arc<Db>,
    pub provider: String,
    pub model: String,
    pub creds: Value,
    pub connection_id: String,
}

/// Account loop shared by every media endpoint (handleEmbeddings & co).
pub async fn with_accounts<F, Fut>(db: Arc<Db>, provider: &str, model: &str, no_auth: bool, preferred: Option<&str>, mut run: F) -> Response
where
    F: FnMut(Ctx) -> Fut,
    Fut: Future<Output = MediaResult>,
{
    if no_auth {
        // Local/keyless services still honour a configured connection (e.g. a
        // custom base URL for SD WebUI, ComfyUI, Coqui or SearXNG).
        let conn = db.connections_for(&accounts::provider_id(provider), true).into_iter().next();
        let (creds, connection_id) = match conn {
            Some(c) => (accounts::credentials_from_connection(&c), c["id"].as_str().unwrap_or("").to_string()),
            None => (json!({"providerSpecificData": {}}), String::new()),
        };
        let r = run(Ctx { db: db.clone(), provider: provider.into(), model: model.into(), creds, connection_id }).await;
        return r.response;
    }
    let mut exclude: HashSet<String> = HashSet::new();
    let mut last: Option<(u16, String)> = None;
    loop {
        let sel = accounts::get_provider_credentials(&db, provider, &exclude, Some(model), None).await;
        let creds = match sel {
            Selection::Creds(c) => c,
            Selection::AllRateLimited { retry_after_ms, last_error, last_error_code } => {
                let (st, msg) = last.clone().unwrap_or((last_error_code.and_then(|v| v.as_u64()).map(|n| n as u16).unwrap_or(503), last_error.unwrap_or_else(|| "Unavailable".into())));
                let human = crate::chat::util::format_retry_after(retry_after_ms);
                let secs = ((retry_after_ms - crate::jsv::now_ms() + 999) / 1000).max(1);
                return json_response(st, &json!({"error": {"message": format!("[{provider}/{model}] {msg} ({human})")}}), &[("retry-after".into(), secs.to_string())]);
            }
            Selection::None => {
                return match last {
                    None => error_response(400, &format!("No credentials for provider: {provider}"), &[]),
                    Some((s, m)) => error_response(s, &m, &[]),
                };
            }
        };
        // Preferred connection pin (x-connection-id) when it is available.
        if let Some(p) = preferred {
            if creds["connectionId"] != p && !exclude.contains(p) {
                if let Some(conn) = db.get_connection(p).filter(|c| c["provider"] == provider && c["isActive"] != json!(false)) {
                    let c = accounts::credentials_from_connection(&conn);
                    let cid = p.to_string();
                    let c = accounts::check_and_refresh_token(&db, provider, &c).await;
                    let r = run(Ctx { db: db.clone(), provider: provider.into(), model: model.into(), creds: c.clone(), connection_id: cid.clone() }).await;
                    if r.ok {
                        accounts::clear_account_error(&db, &cid, &c["_connection"], Some(model));
                        return r.response;
                    }
                    exclude.insert(cid.clone());
                    if !accounts::mark_account_unavailable(&db, &cid, r.status, r.error.as_deref().unwrap_or(""), provider, Some(model), None).should_fallback {
                        return r.response;
                    }
                    last = Some((r.status, r.error.unwrap_or_default()));
                    continue;
                }
            }
        }
        let cid = creds["connectionId"].as_str().unwrap_or("").to_string();
        let creds = accounts::check_and_refresh_token(&db, provider, &creds).await;
        let snapshot = creds["_connection"].clone();
        let r = run(Ctx { db: db.clone(), provider: provider.into(), model: model.into(), creds, connection_id: cid.clone() }).await;
        if r.ok {
            accounts::clear_account_error(&db, &cid, &snapshot, Some(model));
            return r.response;
        }
        let fb = accounts::mark_account_unavailable(&db, &cid, r.status, r.error.as_deref().unwrap_or(""), provider, Some(model), None);
        if fb.should_fallback && cid != "noauth" {
            exclude.insert(cid);
            last = Some((r.status, r.error.unwrap_or_default()));
            continue;
        }
        return r.response;
    }
}

/// Sends a request built from the current credentials; on 401/403 refreshes
/// the token (persisting it) and retries once.
pub async fn send_with_refresh<B>(ctx: &mut Ctx, build: B) -> Result<reqwest::Response, String>
where
    B: Fn(&Value) -> Result<reqwest::RequestBuilder, String>,
{
    let r = build(&ctx.creds)?.send().await.map_err(|e| e.to_string())?;
    let st = r.status().as_u16();
    if st != 401 && st != 403 {
        return Ok(r);
    }
    let ex = crate::providers::get_executor(&ctx.provider);
    if ex.no_auth() {
        return Ok(r);
    }
    let Some(raw) = ex.refresh_credentials(&ctx.creds).await else { return Ok(r) };
    if crate::oauth::refresh::is_unrecoverable(&raw) {
        return Ok(r);
    }
    let patch = crate::oauth::refresh::merge_refreshed(&ctx.provider, &ctx.creds, &raw);
    if !(truthy(&patch["accessToken"]) || truthy(&patch["apiKey"])) {
        return Ok(r);
    }
    let existing = ctx.creds["providerSpecificData"].clone();
    crate::oauth::refresh::merge_into(&mut ctx.creds, &patch);
    let mut persist = patch.clone();
    persist["testStatus"] = json!("active");
    accounts::update_provider_credentials(&ctx.db, &ctx.connection_id, &persist, &existing);
    build(&ctx.creds)?.send().await.map_err(|e| e.to_string())
}

/// parseUpstreamError for a raw reqwest response.
pub async fn upstream_error(r: reqwest::Response) -> MediaResult {
    let st = r.status().as_u16();
    let text = r.text().await.unwrap_or_default();
    let ex = crate::exec::DefaultExecutor { id: String::new() };
    let (code, msg, _) = crate::chat::util::parse_upstream_error(st, &text, &ex);
    provider_err(code, &msg)
}

pub fn bearer(creds: &Value) -> String {
    let k = creds["apiKey"].as_str().filter(|s| !s.is_empty()).or_else(|| creds["accessToken"].as_str()).unwrap_or("");
    format!("Bearer {k}")
}

pub fn key_of(creds: &Value) -> String {
    creds["apiKey"].as_str().filter(|s| !s.is_empty()).or_else(|| creds["accessToken"].as_str()).unwrap_or("").to_string()
}

pub fn client(creds: &Value) -> reqwest::Client {
    crate::exec::client_for(creds)
}

pub fn now_s() -> i64 {
    crate::jsv::now_s()
}

/// Fetch URL → base64 (public hosts only).
pub async fn url_to_base64(url: &str) -> Result<String, String> {
    use base64::Engine;
    let u = reqwest::Url::parse(url).map_err(|e| e.to_string())?;
    if !crate::media::host_is_public(u.host_str().unwrap_or("")).await {
        return Err("refusing to fetch private address".into());
    }
    let r = crate::exec::http_client(None).get(url).send().await.map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        return Err(format!("Failed to fetch image: {}", r.status().as_u16()));
    }
    let b = r.bytes().await.map_err(|e| e.to_string())?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&b))
}

pub fn size_to_aspect(size: &Value) -> &'static str {
    match size.as_str().unwrap_or("") {
        "1024x1792" => "9:16",
        "1792x1024" => "16:9",
        "1024x1536" => "2:3",
        "1536x1024" => "3:2",
        _ => "1:1",
    }
}

pub fn media_cfg(provider: &str, key: &str) -> Value {
    crate::registry::REG.media(provider)[key].clone()
}
