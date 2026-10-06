//! /v1/videos/{generations,edits,extensions} and GET /v1/videos/{id}
//! (port of videoGeneration.js, videoCore.js and videoProviders/*).

use std::collections::HashSet;
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use base64::Engine;
use bytes::Bytes;
use serde_json::{Value, json};

use super::*;
use crate::AppState;
use crate::chat::accounts::{self, Selection};

const DEFAULT_PROVIDER: &str = "xai";
const B64URL: base64::engine::GeneralPurpose = base64::engine::general_purpose::URL_SAFE_NO_PAD;

fn video_cfg(p: &str) -> Option<Value> {
    Some(media_cfg(p, "videoConfig")).filter(|v| v.is_object())
}

pub fn sanitize(text: &str, creds: &Value) -> String {
    let re = regex::Regex::new(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]{8,}").unwrap();
    let mut out = re.replace_all(text, "Bearer [redacted]").to_string();
    for k in ["accessToken", "refreshToken", "apiKey"] {
        if let Some(s) = creds[k].as_str().filter(|s| s.len() >= 8) {
            out = out.replace(s, "[redacted]");
        }
    }
    out
}

struct Plan {
    method: &'static str,
    url: String,
    headers: Vec<(String, String)>,
    body: Option<Vec<u8>>,
}

fn op_re() -> regex::Regex {
    regex::Regex::new(r"^projects/[^/]+/locations/[^/]+/publishers/[^/]+/models/[^/]+/operations/[^/]+$").unwrap()
}

fn decode_job(id: &str) -> Option<String> {
    if id.is_empty() || id.len() > 1024 || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return None;
    }
    let d = String::from_utf8(B64URL.decode(id).ok()?).ok()?;
    (B64URL.encode(&d) == id && op_re().is_match(&d)).then_some(d)
}

fn vertex_body(b: &Value) -> Value {
    let mut inst = json!({"prompt": b["prompt"]});
    let image = if !b["image"].is_null() { &b["image"] } else { &b["image_url"] };
    if image.is_object() {
        inst["image"] = image.clone();
    } else if let Some(s) = image.as_str() {
        let re = regex::Regex::new(r"(?s)^data:([^;]+);base64,(.*)$").unwrap();
        inst["image"] = match re.captures(s) {
            Some(c) => json!({"bytesBase64Encoded": &c[2], "mimeType": &c[1]}),
            None => json!({"gcsUri": s}),
        };
    }
    if b["video"].is_object() {
        inst["video"] = b["video"].clone();
    }
    let mut p = serde_json::Map::new();
    let num = |v: &Value| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()));
    if let Some(n) = num(&b["n"]) {
        p.insert("sampleCount".into(), json!(n));
    }
    if let Some(n) = num(&b["duration"]) {
        p.insert("durationSeconds".into(), json!(n));
    }
    for (src, dst) in [("aspect_ratio", "aspectRatio"), ("resolution", "resolution"), ("negative_prompt", "negativePrompt"), ("storage_uri", "storageUri")] {
        if truthy(&b[src]) {
            p.insert(dst.into(), b[src].clone());
        }
    }
    if !b["seed"].is_null() {
        p.insert("seed".into(), b["seed"].clone());
    }
    if !b["generate_audio"].is_null() {
        p.insert("generateAudio".into(), json!(truthy(&b["generate_audio"])));
    }
    let mut out = json!({"instances": [inst]});
    if !p.is_empty() {
        out["parameters"] = Value::Object(p);
    }
    out
}

fn from_vertex_op(j: &Value) -> Value {
    let Some(name) = j["name"].as_str() else { return j.clone() };
    let id = B64URL.encode(name);
    if j["error"].is_object() {
        return json!({"id": id, "request_id": id, "status": "failed", "error": j["error"]});
    }
    if j["done"] != json!(true) {
        return json!({"id": id, "request_id": id, "status": "pending"});
    }
    let samples = [&j["response"]["videos"], &j["response"]["generateVideoResponse"]["generatedSamples"]].into_iter().find(|v| v.is_array()).cloned().unwrap_or(json!([]));
    let pick = |vs: [&Value; 3]| vs.into_iter().find(|v| truthy(v)).cloned().unwrap_or(Value::Null);
    let vids: Vec<Value> = samples
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let url = pick([&s["gcsUri"], &s["video"]["uri"], &s["uri"]]);
            let b64 = pick([&s["bytesBase64Encoded"], &s["video"]["bytesBase64Encoded"], &Value::Null]);
            let mime = pick([&s["mimeType"], &s["video"]["mimeType"], &json!("video/mp4")]);
            json!({"url": url, "b64_json": b64, "mime_type": mime})
        })
        .collect();
    json!({"id": id, "request_id": id, "status": "completed", "video": vids.first().cloned().unwrap_or(Value::Null), "videos": vids})
}

async fn plan(provider: &str, cfg: &Value, action: Option<&str>, request_id: Option<&str>, raw: &[u8], ct: Option<&str>, idem: Option<&str>, creds: &Value) -> Result<Plan, String> {
    let token = creds["accessToken"].as_str().filter(|s| !s.is_empty()).or_else(|| creds["apiKey"].as_str()).unwrap_or("").to_string();
    let base = cfg["baseUrl"].as_str().unwrap_or("").trim_end_matches('/').to_string();
    match provider {
        "openrouter" => {
            let mut h = vec![("accept".to_string(), "application/json".to_string())];
            for (k, v) in cfg["headers"].as_object().into_iter().flatten() {
                h.push((k.clone(), crate::jsv::js_string(v)));
            }
            if !token.is_empty() {
                h.push(("authorization".into(), format!("Bearer {token}")));
            }
            if let Some(id) = request_id {
                return Ok(Plan { method: "GET", url: format!("{base}/{}", crate::oauth::enc(id)), headers: h, body: None });
            }
            if action != Some("generations") {
                return Err(format!("OpenRouter video supports 'generations' only (got '{}')", action.unwrap_or("")));
            }
            if ct.map(|c| !c.contains("application/json")).unwrap_or(false) {
                return Err("OpenRouter video requires an application/json body".into());
            }
            h.push(("content-type".into(), "application/json".into()));
            Ok(Plan { method: "POST", url: base, headers: h, body: Some(raw.to_vec()) })
        }
        "vertex" | "vertex-partner" => {
            if ct.map(|c| !c.contains("application/json")).unwrap_or(false) {
                return Err("Vertex video requires an application/json body".into());
            }
            let sa = crate::oauth::refresh::parse_vertex_sa_json(&creds["apiKey"]);
            let project = sa.as_ref().and_then(|s| s["project_id"].as_str().map(str::to_owned)).or_else(|| creds["projectId"].as_str().map(str::to_owned)).or_else(|| creds["providerSpecificData"]["projectId"].as_str().map(str::to_owned)).ok_or("Vertex video requires a project_id — use Service Account JSON or set providerSpecificData.projectId")?;
            let location = creds["providerSpecificData"]["location"].as_str().unwrap_or("us-central1").to_string();
            let mut tok = creds["accessToken"].as_str().unwrap_or("").to_string();
            if let Some(sa) = &sa {
                tok = crate::oauth::refresh::refresh_vertex_token(sa).await.map(|t| t.0).ok_or("Vertex video: failed to mint access token from service account JSON")?;
            }
            if tok.is_empty() {
                return Err("Vertex video requires Service Account JSON or an OAuth access token (raw API keys are not supported)".into());
            }
            let base = if base.is_empty() { "https://aiplatform.googleapis.com".to_string() } else { base };
            let h = vec![("accept".to_string(), "application/json".to_string()), ("content-type".into(), "application/json".into()), ("authorization".into(), format!("Bearer {tok}"))];
            if let Some(id) = request_id {
                let op = decode_job(id).ok_or("Invalid Vertex video job id")?;
                let model_path = &op[..op.find("/operations/").unwrap()];
                return Ok(Plan { method: "POST", url: format!("{base}/v1/{model_path}:fetchPredictOperation"), headers: h, body: Some(json!({"operationName": op}).to_string().into_bytes()) });
            }
            if action != Some("generations") {
                return Err(format!("Vertex video supports 'generations' only (got '{}')", action.unwrap_or("")));
            }
            let b: Value = serde_json::from_slice(raw).map_err(|_| "Invalid JSON body")?;
            let m = b["model"].as_str().ok_or("Vertex video requires a model (e.g. vertex/veo-3.1-generate-preview)")?;
            if !m.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c)) {
                return Err("Invalid Vertex video model id".into());
            }
            if !truthy(&b["prompt"]) && !truthy(&b["image"]) && !truthy(&b["image_url"]) {
                return Err("Vertex video requires a prompt or an image".into());
            }
            Ok(Plan { method: "POST", url: format!("{base}/v1/projects/{project}/locations/{location}/publishers/google/models/{m}:predictLongRunning"), headers: h, body: Some(vertex_body(&b).to_string().into_bytes()) })
        }
        _ => {
            let mut h = vec![("accept".to_string(), "application/json".to_string())];
            if !token.is_empty() {
                h.push(("authorization".into(), format!("Bearer {token}")));
            }
            if let Some(id) = request_id {
                return Ok(Plan { method: "GET", url: format!("{base}/{}", crate::oauth::enc(id)), headers: h, body: None });
            }
            if let Some(c) = ct {
                h.push(("content-type".into(), c.into()));
            }
            if let Some(i) = idem {
                h.push(("idempotency-key".into(), i.into()));
            }
            Ok(Plan { method: "POST", url: format!("{base}/{}", action.unwrap_or("generations")), headers: h, body: Some(raw.to_vec()) })
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn proxy_core(ctx: &mut Ctx, action: Option<&str>, request_id: Option<&str>, raw: &[u8], ct: Option<&str>, idem: Option<&str>) -> MediaResult {
    let provider = ctx.provider.clone();
    let Some(cfg) = video_cfg(&provider) else {
        return MediaResult::err(400, format!("Provider '{provider}' does not support video generation"));
    };
    if request_id.is_none() && !matches!(action, Some("generations" | "edits" | "extensions")) {
        return MediaResult::err(400, format!("Unknown video action: {}", action.unwrap_or("")));
    }
    let to = Duration::from_millis(std::env::var("VIDEO_FETCH_TIMEOUT_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(120_000));
    let mut attempt = 0;
    loop {
        attempt += 1;
        let p = match plan(&provider, &cfg, action, request_id, raw, ct, idem, &ctx.creds).await {
            Ok(p) => p,
            Err(e) => return MediaResult::err(400, format!("[{provider}] {e}")),
        };
        let c = client(&ctx.creds);
        let mut rb = if p.method == "GET" { c.get(&p.url) } else { c.post(&p.url) };
        for (k, v) in &p.headers {
            rb = rb.header(k.as_str(), v.as_str());
        }
        if let Some(b) = p.body {
            rb = rb.body(b);
        }
        let r = match rb.timeout(to).send().await {
            Ok(r) => r,
            Err(e) if e.is_timeout() => return MediaResult::err(408, format!("[{provider}] video {} aborted: {e}", p.method)),
            Err(e) => return MediaResult::err(502, sanitize(&format!("[{provider}] video upstream fetch failed: {e}"), &ctx.creds)),
        };
        let st = r.status().as_u16();
        if matches!(st, 401 | 403) && attempt == 1 && truthy(&ctx.creds["refreshToken"]) {
            if let Some(raw_r) = crate::oauth::refresh::refresh_for_provider(&provider, &ctx.creds).await {
                let patch = crate::oauth::refresh::merge_refreshed(&provider, &ctx.creds, &raw_r);
                if truthy(&patch["accessToken"]) {
                    let existing = ctx.creds["providerSpecificData"].clone();
                    crate::oauth::refresh::merge_into(&mut ctx.creds, &patch);
                    let mut persist = patch;
                    persist["testStatus"] = json!("active");
                    accounts::update_provider_credentials(&ctx.db, &ctx.connection_id, &persist, &existing);
                    continue;
                }
            }
        }
        let ctype = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("application/json").to_string();
        let text = r.text().await.unwrap_or_default();
        if !(200..300).contains(&st) {
            let raw_msg = if text.is_empty() { format!("HTTP {st}") } else { text.clone() };
            let m = sanitize(&raw_msg, &ctx.creds);
            return MediaResult::err(st, format!("[{provider}] {}", m.chars().take(2000).collect::<String>()));
        }
        let (body, ctype) = if provider.starts_with("vertex") {
            match serde_json::from_str::<Value>(&text) {
                Ok(v) => (from_vertex_op(&v).to_string(), "application/json".to_string()),
                Err(_) => (text, ctype),
            }
        } else {
            (text, ctype)
        };
        let mut resp = Response::new(axum::body::Body::from(body));
        *resp.status_mut() = axum::http::StatusCode::from_u16(st).unwrap_or(axum::http::StatusCode::OK);
        resp.headers_mut().insert("content-type", ctype.parse().unwrap_or_else(|_| "application/json".parse().unwrap()));
        resp.headers_mut().insert("access-control-allow-origin", "*".parse().unwrap());
        if let Ok(v) = ctx.connection_id.parse() {
            resp.headers_mut().insert("x-9router-connection-id", v);
        }
        return MediaResult::ok(resp);
    }
}

pub async fn create(State(st): State<AppState>, Path(action): Path<String>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = crate::api::authorize(&st, &headers, None) {
        return r;
    }
    let ct = headers.get("content-type").and_then(|v| v.to_str().ok()).map(str::to_owned);
    let is_json = ct.as_deref().map(|c| c.contains("application/json")).unwrap_or(false);
    let mut parsed: Option<Value> = None;
    if is_json {
        match serde_json::from_slice::<Value>(&body) {
            Ok(v) => parsed = Some(v),
            Err(_) => return error_response(400, "Invalid JSON body", &[]),
        }
    }
    let (provider, model) = match parsed.as_ref().and_then(|p| p["model"].as_str()) {
        None => (DEFAULT_PROVIDER.to_string(), None),
        Some(m) => match resolve_model(&st.db, m) {
            None => return error_response(400, "Combos are not supported for video generation", &[]),
            Some((p, mm)) if video_cfg(&p).is_some() => (p, Some(mm)),
            Some((p, _)) => {
                if !m.contains('/') {
                    (DEFAULT_PROVIDER.to_string(), Some(m.to_string()))
                } else {
                    return error_response(400, &format!("Provider '{p}' does not support video generation"), &[]);
                }
            }
        },
    };
    let mut raw = body.to_vec();
    if let (Some(p), Some(m)) = (&parsed, &model) {
        if p["model"].as_str() != Some(m) {
            let mut p2 = p.clone();
            p2["model"] = json!(m);
            raw = p2.to_string().into_bytes();
        }
    }
    let preferred = headers.get("x-connection-id").and_then(|v| v.to_str().ok()).map(str::to_owned);
    let idem = headers.get("idempotency-key").and_then(|v| v.to_str().ok()).map(str::to_owned);
    let m = model.clone().unwrap_or_default();
    let mut exclude: HashSet<String> = HashSet::new();
    let mut last: Option<(u16, String)> = None;
    loop {
        let creds = match accounts::get_provider_credentials(&st.db, &provider, &exclude, model.as_deref(), None).await {
            Selection::Creds(c) => c,
            Selection::AllRateLimited { last_error, .. } => {
                let (s, e) = last.clone().unwrap_or((503, last_error.unwrap_or_else(|| "Unavailable".into())));
                return error_response(s, &format!("[{provider}/{}] {e}", if m.is_empty() { "video" } else { &m }), &[]);
            }
            Selection::None => {
                return match last {
                    None => error_response(400, &format!("No credentials for provider: {provider}"), &[]),
                    Some((s, e)) => error_response(s, &e, &[]),
                };
            }
        };
        let mut creds = creds;
        if let Some(p) = &preferred {
            if let Some(conn) = st.db.get_connection(p).filter(|c| c["provider"] == provider.as_str() && !exclude.contains(p)) {
                creds = accounts::credentials_from_connection(&conn);
            }
        }
        let cid = creds["connectionId"].as_str().unwrap_or("").to_string();
        let creds = accounts::check_and_refresh_token(&st.db, &provider, &creds).await;
        let snapshot = creds["_connection"].clone();
        let mut ctx = Ctx { db: st.db.clone(), provider: provider.clone(), model: m.clone(), creds, connection_id: cid.clone() };
        let r = proxy_core(&mut ctx, Some(&action), None, &raw, ct.as_deref(), idem.as_deref()).await;
        if r.ok {
            accounts::clear_account_error(&st.db, &cid, &snapshot, model.as_deref());
            return r.response;
        }
        let err = sanitize(r.error.as_deref().unwrap_or(""), &ctx.creds);
        let fb = accounts::mark_account_unavailable(&st.db, &cid, r.status, &err, &provider, model.as_deref(), None);
        if fb.should_fallback && matches!(r.status, 401 | 403 | 429) {
            exclude.insert(cid);
            last = Some((r.status, r.error.unwrap_or_default()));
            continue;
        }
        return r.response;
    }
}

#[derive(serde::Deserialize, Default)]
pub struct GetQuery {
    pub provider: Option<String>,
}

pub async fn get(State(st): State<AppState>, Path(id): Path<String>, Query(q): Query<GetQuery>, headers: HeaderMap) -> Response {
    if let Err(r) = crate::api::authorize(&st, &headers, None) {
        return r;
    }
    if id.is_empty() {
        return error_response(400, "Missing video request id", &[]);
    }
    let preferred = headers.get("x-connection-id").and_then(|v| v.to_str().ok()).map(str::to_owned);
    let conn = preferred.as_deref().and_then(|p| st.db.get_connection(p));
    let provider = conn
        .as_ref()
        .and_then(|c| c["provider"].as_str().map(str::to_owned))
        .filter(|p| video_cfg(p).is_some())
        .or_else(|| q.provider.clone().filter(|p| video_cfg(p).is_some()))
        .unwrap_or_else(|| DEFAULT_PROVIDER.to_string());
    let creds = match conn.filter(|c| c["provider"] == provider.as_str()) {
        Some(c) => accounts::credentials_from_connection(&c),
        None => match accounts::get_provider_credentials(&st.db, &provider, &HashSet::new(), None, None).await {
            Selection::Creds(c) => c,
            _ => return error_response(400, &format!("No credentials for provider: {provider}"), &[]),
        },
    };
    let cid = creds["connectionId"].as_str().unwrap_or("").to_string();
    let creds = accounts::check_and_refresh_token(&st.db, &provider, &creds).await;
    let snapshot = creds["_connection"].clone();
    let mut ctx = Ctx { db: st.db.clone(), provider: provider.clone(), model: String::new(), creds, connection_id: cid.clone() };
    let r = proxy_core(&mut ctx, None, Some(&id), &[], None, None).await;
    if r.ok {
        accounts::clear_account_error(&st.db, &cid, &snapshot, None);
    } else {
        accounts::mark_account_unavailable(&st.db, &cid, r.status, &sanitize(r.error.as_deref().unwrap_or(""), &ctx.creds), &provider, None, None);
    }
    r.response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_jobs() {
        let name = "projects/p/locations/us-central1/publishers/google/models/veo/operations/123";
        let id = B64URL.encode(name);
        assert_eq!(decode_job(&id).as_deref(), Some(name));
        assert!(decode_job("bad!").is_none());
        let done = from_vertex_op(&json!({"name": name, "done": true, "response": {"videos": [{"gcsUri": "gs://x"}]}}));
        assert_eq!(done["video"]["url"], "gs://x");
        assert_eq!(sanitize("Bearer abcdefghijk", &json!({})), "Bearer [redacted]");
    }
}
