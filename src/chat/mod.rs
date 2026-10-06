//! Chat request pipeline (port of src/sse/handlers/chat.js + services/model.js,
//! services/combo.js and utils/bypassHandler.js).

pub mod accounts;
pub mod core;
pub mod stream;
pub mod usage;
pub mod util;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};

use axum::response::Response;
use futures::future::BoxFuture;
use serde_json::{Value, json};

use self::accounts::Selection;
use self::core::{CoreArgs, error_response, json_response, sse_response};
use crate::db::Db;
use crate::jsv::{now_ms, now_s, truthy};
use crate::registry::REG;
use crate::translate::{self, CLAUDE, OPENAI};

/// Client request context shared by every attempt.
#[derive(Clone)]
pub struct ClientReq {
    pub headers: Value,
    pub endpoint: String,
    pub api_key: Option<String>,
}

pub fn headers_json(h: &axum::http::HeaderMap) -> Value {
    let mut o = serde_json::Map::new();
    for (k, v) in h {
        if let Ok(s) = v.to_str() {
            o.insert(k.as_str().to_lowercase(), json!(s));
        }
    }
    Value::Object(o)
}

// ---------------------------------------------------------------------------
// Model resolution
// ---------------------------------------------------------------------------

pub enum ModelInfo {
    Provider { provider: String, model: String },
    Combo(String),
}

static RESERVED: LazyLock<HashSet<String>> = LazyLock::new(|| {
    let mut s: HashSet<String> = ["xmtp", "xiaomi-tokenplan"].iter().map(|x| x.to_string()).collect();
    for e in &REG.entries {
        if let Some(i) = e["id"].as_str() {
            s.insert(i.into());
        }
        if let Some(a) = e["alias"].as_str() {
            s.insert(a.into());
        }
        for a in e["aliases"].as_array().into_iter().flatten() {
            if let Some(a) = a.as_str() {
                s.insert(a.into());
            }
        }
    }
    s
});

fn resolve_provider_alias(a: &str) -> String {
    accounts::provider_id(a)
}

fn resolve_alias_target(v: &Value) -> Option<(String, String)> {
    if let Some(s) = v.as_str() {
        let (p, m) = s.split_once('/')?;
        return Some((resolve_provider_alias(p), m.to_string()));
    }
    if truthy(&v["provider"]) && truthy(&v["model"]) {
        return Some((resolve_provider_alias(v["provider"].as_str()?), v["model"].as_str()?.to_string()));
    }
    None
}

/// getModelInfo(modelStr)
pub fn get_model_info(db: &Db, model_str: &str) -> Option<ModelInfo> {
    if let Some((prefix, model)) = model_str.split_once('/') {
        if !RESERVED.contains(prefix) {
            for ty in ["openai-compatible", "anthropic-compatible", "custom-embedding"] {
                if let Some(n) = db.list_nodes(Some(ty)).into_iter().find(|n| n["prefix"] == prefix) {
                    return Some(ModelInfo::Provider { provider: n["id"].as_str()?.to_string(), model: model.to_string() });
                }
            }
            // A node id used directly as the prefix.
            if db.get_node(prefix).is_some() {
                return Some(ModelInfo::Provider { provider: prefix.to_string(), model: model.to_string() });
            }
        }
        return Some(ModelInfo::Provider { provider: resolve_provider_alias(prefix), model: model.to_string() });
    }
    if db.get_combo(model_str).is_some() {
        return Some(ModelInfo::Combo(model_str.to_string()));
    }
    let aliases = db.model_aliases();
    if let Some((p, m)) = aliases.get(model_str).and_then(resolve_alias_target) {
        return Some(ModelInfo::Provider { provider: p, model: m });
    }
    if model_str == "grok-build" {
        return Some(ModelInfo::Provider { provider: "grok-cli".into(), model: "grok-build".into() });
    }
    Some(ModelInfo::Provider { provider: crate::registry::infer_provider_from_model(model_str).to_string(), model: model_str.to_string() })
}

pub fn combo_models(db: &Db, model_str: &str) -> Option<Vec<String>> {
    if model_str.contains('/') {
        return None;
    }
    db.get_combo(model_str).map(|c| c.models).filter(|m| !m.is_empty())
}

// ---------------------------------------------------------------------------
// Bypass (Claude CLI warmup / title requests)
// ---------------------------------------------------------------------------

fn text_of(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter(|b| b["type"] == "text").filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join(" "),
        _ => String::new(),
    }
}

/// handleBypassRequest
pub fn bypass(body: &Value, model: &str, user_agent: &str, cc_filter_naming: bool) -> Option<Response> {
    if !user_agent.contains("claude-cli") {
        return None;
    }
    let msgs = body["messages"].as_array().filter(|m| !m.is_empty())?;
    let last = msgs.last()?;
    let mut hit = last["role"] == "assistant" && last["content"][0]["text"] == "{";
    if !hit && text_of(&msgs[0]["content"]) == "Warmup" {
        hit = true;
    }
    if !hit && msgs.len() == 1 && msgs[0]["role"] == "user" && text_of(&msgs[0]["content"]) == "count" {
        hit = true;
    }
    if !hit {
        let ut = msgs.iter().filter(|m| m["role"] == "user").map(|m| text_of(&m["content"])).collect::<Vec<_>>().join(" ");
        if ut.contains("Please write a 5-10 word title for the following conversation:") {
            hit = true;
        }
    }
    let mut naming = false;
    if !hit && cc_filter_naming {
        let sys_msg = msgs.iter().find(|m| m["role"] == "system").map(|m| text_of(&m["content"])).unwrap_or_default();
        let sys_body = match &body["system"] {
            Value::Array(a) => a.iter().filter(|s| s["type"] == "text").filter_map(|s| s["text"].as_str()).collect::<Vec<_>>().join(" "),
            Value::String(s) => s.clone(),
            _ => String::new(),
        };
        let st = if sys_msg.is_empty() { sys_body } else { sys_msg };
        if st.contains("isNewTopic") {
            hit = true;
            naming = true;
        }
    }
    if !hit {
        return None;
    }
    let text = if naming {
        let ut = msgs.iter().find(|m| m["role"] == "user").map(|m| text_of(&m["content"])).unwrap_or_default();
        let title = ut.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
        json!({"isNewTopic": true, "title": title}).to_string()
    } else {
        "CLI Command Execution: Clear Terminal".to_string()
    };
    let source = util::detect_format(body);
    let openai = json!({
        "id": format!("chatcmpl-{}", now_ms()), "object": "chat.completion", "created": now_s(), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    });
    if body["stream"] == json!(false) {
        let out = if source == OPENAI { openai } else { translate::nonstream::from_openai(source, &openai, &[]) };
        return Some(json_response(200, &out, &[]));
    }
    let mut state = translate::resp::init_state(source);
    let mut out = String::new();
    let base = |delta: Value, fin: Value| json!({"id": openai["id"], "object": "chat.completion.chunk", "created": openai["created"], "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]});
    let mut fin = base(json!({}), json!("stop"));
    fin["usage"] = openai["usage"].clone();
    for c in [base(json!({"role": "assistant", "content": text}), Value::Null), fin] {
        for it in translate::translate_response(OPENAI, source, Some(&c), &mut state) {
            out.push_str(&translate::format_sse(&it, source));
        }
    }
    for it in translate::translate_response(OPENAI, source, None, &mut state) {
        out.push_str(&translate::format_sse(&it, source));
    }
    if source != CLAUDE {
        out.push_str("data: [DONE]\n\n");
    }
    let b = bytes::Bytes::from(out);
    Some(sse_response(Box::pin(futures::stream::once(async move { Ok(b) })), &[]))
}

// ---------------------------------------------------------------------------
// Combos
// ---------------------------------------------------------------------------

static ROTATION: LazyLock<Mutex<HashMap<String, (usize, i64)>>> = LazyLock::new(Default::default);

/// getRotatedModels(models, comboName, strategy, stickyLimit)
pub fn rotated_models(models: &[String], combo: &str, strategy: &str, sticky: i64) -> Vec<String> {
    if models.len() <= 1 || strategy != "round-robin" {
        return models.to_vec();
    }
    let sticky = if sticky > 0 { sticky } else { 1 };
    let mut map = ROTATION.lock().unwrap();
    let (idx, count) = map.get(combo).copied().unwrap_or((0, 0));
    let cur = idx % models.len();
    let mut out = models.to_vec();
    out.rotate_left(cur);
    let next = count + 1;
    if next >= sticky {
        map.insert(combo.to_string(), ((cur + 1) % models.len(), 0));
    } else {
        map.insert(combo.to_string(), (cur, next));
    }
    out
}

pub fn reset_rotation(combo: Option<&str>) {
    let mut m = ROTATION.lock().unwrap();
    match combo {
        Some(c) => {
            m.remove(c);
        }
        None => m.clear(),
    }
}

fn trailing_user<'a>(arr: &'a Value) -> &'a [Value] {
    let Some(a) = arr.as_array() else { return &[] };
    let mut i = a.len();
    while i > 0 && !matches!(a[i - 1]["role"].as_str(), Some("assistant" | "model")) {
        i -= 1;
    }
    &a[i..]
}

/// detectRequiredCapabilities(body) → {"vision","pdf","audioInput","videoInput"}
pub fn required_capabilities(body: &Value) -> HashSet<&'static str> {
    let mut req = HashSet::new();
    fn by_mime(m: &str, req: &mut HashSet<&'static str>) {
        if m.starts_with("image/") {
            req.insert("vision");
        } else if m == "application/pdf" {
            req.insert("pdf");
        } else if m.starts_with("audio/") {
            req.insert("audioInput");
        } else if m.starts_with("video/") {
            req.insert("videoInput");
        }
    }
    fn data_mime(s: &str) -> Option<String> {
        let r = s.strip_prefix("data:")?;
        let end = r.find([';', ',']).unwrap_or(r.len());
        Some(r[..end].to_string()).filter(|m| !m.is_empty())
    }
    fn scan_block(b: &Value, req: &mut HashSet<&'static str>) {
        let t = b["type"].as_str().unwrap_or("");
        if matches!(t, "image_url" | "image" | "input_image") {
            req.insert("vision");
        }
        if matches!(t, "input_audio" | "audio_url" | "audio") {
            req.insert("audioInput");
        }
        if matches!(t, "input_video" | "video_url" | "video") {
            req.insert("videoInput");
        }
        if matches!(t, "file" | "document" | "input_file") {
            let m = if let Some(f) = b["input_audio"]["format"].as_str() {
                Some(format!("audio/{f}"))
            } else if let Some(d) = b["file"]["file_data"].as_str() {
                data_mime(d)
            } else if let Some(m) = b["source"]["media_type"].as_str() {
                Some(m.to_string())
            } else if let Some(d) = b["source"]["data"].as_str() {
                data_mime(d)
            } else {
                None
            };
            match m {
                Some(m) => by_mime(&m, req),
                None => {
                    req.insert("pdf");
                }
            }
        }
        if let Some(m) = b["inlineData"]["mimeType"].as_str().or_else(|| b["fileData"]["mimeType"].as_str()) {
            by_mime(m, req);
        }
    }
    for m in trailing_user(&body["messages"]) {
        if m["images"].as_array().map(|a| !a.is_empty()).unwrap_or(false) {
            req.insert("vision");
        }
        let att = if m["experimental_attachments"].is_array() { &m["experimental_attachments"] } else { &m["attachments"] };
        for a in att.as_array().into_iter().flatten() {
            let mime = a["contentType"].as_str().map(str::to_owned).or_else(|| a["mediaType"].as_str().map(str::to_owned)).or_else(|| a["url"].as_str().and_then(data_mime));
            match mime {
                Some(m) => by_mime(&m, &mut req),
                None if truthy(&a["url"]) || truthy(&a["data"]) => {
                    req.insert("vision");
                }
                None => {}
            }
        }
        if truthy(&m["image_url"]) || truthy(&m["image"]) {
            req.insert("vision");
        }
        if truthy(&m["audio_url"]) || truthy(&m["audio"]) {
            req.insert("audioInput");
        }
        for b in m["content"].as_array().into_iter().flatten() {
            scan_block(b, &mut req);
        }
        if let Some(s) = m["content"].as_str() {
            if s.contains("data:image/") {
                req.insert("vision");
            } else if s.contains("data:audio/") {
                req.insert("audioInput");
            } else if s.contains("data:application/pdf") {
                req.insert("pdf");
            }
        }
    }
    for it in trailing_user(&body["input"]) {
        for b in it["content"].as_array().into_iter().flatten() {
            scan_block(b, &mut req);
        }
    }
    let contents = if body["contents"].is_array() { &body["contents"] } else { &body["request"]["contents"] };
    for c in trailing_user(contents) {
        for b in c["parts"].as_array().into_iter().flatten() {
            scan_block(b, &mut req);
        }
    }
    req
}

/// reorderByCapabilities: models that satisfy the required capabilities first.
pub fn reorder_by_capabilities(models: Vec<String>, req: &HashSet<&'static str>) -> Vec<String> {
    if req.is_empty() || models.len() <= 1 {
        return models;
    }
    let tier = |m: &String| {
        let (p, model) = m.split_once('/').unwrap_or(("", m.as_str()));
        let pid = accounts::provider_id(p);
        let caps = crate::caps::caps_for(if p.is_empty() { None } else { Some(&pid) }, model);
        if req.iter().all(|c| caps.get(c) == &json!(true)) { 0 } else { 2 }
    };
    let mut v: Vec<(usize, i32, String)> = models.into_iter().enumerate().map(|(i, m)| (i, tier(&m), m)).collect();
    v.sort_by_key(|(i, t, _)| (*t, *i));
    v.into_iter().map(|(_, _, m)| m).collect()
}

async fn body_error_text(resp: Response) -> (Response, String, Option<String>) {
    let (parts, body) = resp.into_parts();
    let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
    let mut text = String::new();
    let mut retry_after = None;
    if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
        let e = if truthy(&v["error"]["message"]) { &v["error"]["message"] } else if truthy(&v["error"]) { &v["error"] } else { &v["message"] };
        text = e.as_str().map(str::to_owned).unwrap_or_else(|| if e.is_null() { String::new() } else { e.to_string() });
        retry_after = v["retryAfter"].as_str().map(str::to_owned);
    }
    (Response::from_parts(parts, axum::body::Body::from(bytes)), text, retry_after)
}

/// handleComboChat: try each model in order until one succeeds.
pub async fn handle_combo(db: Arc<Db>, body: Value, models: Vec<String>, combo: &str, req: ClientReq) -> Response {
    let st = accounts::settings(&db);
    let strategy = st["comboStrategies"][combo]["fallbackStrategy"].as_str().or_else(|| st["comboStrategy"].as_str()).unwrap_or("fallback").to_string();
    let sticky = st["comboStickyRoundRobinLimit"].as_i64().unwrap_or(1);
    let mut list = rotated_models(&models, combo, &strategy, sticky);
    let required = required_capabilities(&body);
    if !required.is_empty() {
        list = reorder_by_capabilities(list, &required);
    }
    let mut last_err: Option<String> = None;
    let mut last_status: Option<u16> = None;
    for (i, m) in list.iter().enumerate() {
        tracing::info!("COMBO {combo}: trying {}/{}: {m}", i + 1, list.len());
        let resp = single_model(db.clone(), body.clone(), m.clone(), req.clone(), None, 1).await;
        let status = resp.status().as_u16();
        if (200..300).contains(&status) {
            return resp;
        }
        let (resp, text, _) = body_error_text(resp).await;
        let fb = accounts::check_fallback_error(status, &text, 0, None);
        if !fb.should_fallback {
            return resp;
        }
        if fb.cooldown_ms > 0 && fb.cooldown_ms <= 5000 && matches!(status, 502 | 503 | 504) {
            tokio::time::sleep(std::time::Duration::from_millis(fb.cooldown_ms as u64)).await;
        }
        last_err = Some(if text.is_empty() { status.to_string() } else { text });
        last_status.get_or_insert(status);
    }
    let msg = last_err.unwrap_or_else(|| "All combo models unavailable".into());
    let status = if msg.to_lowercase().contains("no credentials") { 503 } else { last_status.unwrap_or(503) };
    json_response(status, &json!({"error": {"message": msg}}), &[])
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// handleChat: body already parsed; API-key auth already enforced by the route.
pub async fn handle_chat(db: Arc<Db>, mut body: Value, req: ClientReq) -> Response {
    let raw_model = body["model"].as_str().unwrap_or("").to_string();
    let (model_str, marker) = util::strip_model_context_marker(&raw_model);
    if marker.is_some() {
        body["model"] = json!(model_str);
    }
    if model_str.is_empty() {
        return error_response(400, "Missing model", &[]);
    }
    let st = accounts::settings(&db);
    let ua = req.headers["user-agent"].as_str().unwrap_or("").to_string();
    if let Some(r) = bypass(&body, &model_str, &ua, st["ccFilterNaming"] == json!(true)) {
        return r;
    }
    if let Some(models) = combo_models(&db, &model_str) {
        return handle_combo(db, body, models, &model_str, req).await;
    }
    let requested = marker.map(|m| format!("{}[{m}]", model_str.split_once('/').map(|x| x.1).unwrap_or(&model_str)));
    single_model(db, body, model_str, req, requested, 0).await
}

/// handleSingleModelChat: account loop with fallback.
pub fn single_model(db: Arc<Db>, body: Value, model_str: String, req: ClientReq, requested_model: Option<String>, depth: u8) -> BoxFuture<'static, Response> {
    Box::pin(async move {
        let Some(info) = get_model_info(&db, &model_str) else {
            return error_response(400, "Invalid model format", &[]);
        };
        let (provider, model) = match info {
            ModelInfo::Combo(name) => {
                if depth > 2 {
                    return error_response(400, "Combo nesting too deep", &[]);
                }
                let models = combo_models(&db, &name).unwrap_or_default();
                return handle_combo(db, body, models, &name, req).await;
            }
            ModelInfo::Provider { provider, model } => (provider, model),
        };
        let mut exclude: HashSet<String> = HashSet::new();
        let mut last_error: Option<String> = None;
        let mut last_status: Option<u16> = None;
        let mut last_headers: Vec<(String, String)> = vec![];
        loop {
            let sel = accounts::get_provider_credentials(&db, &provider, &exclude, Some(&model), requested_model.as_deref().or(Some(&model))).await;
            let creds = match sel {
                Selection::Creds(c) => c,
                Selection::AllRateLimited { retry_after_ms, last_error: le, .. } => {
                    let msg = last_error.clone().or(le).unwrap_or_else(|| "Unavailable".into());
                    let human = util::format_retry_after(retry_after_ms);
                    let secs = ((retry_after_ms - now_ms() + 999) / 1000).max(1);
                    let mut h = last_headers.clone();
                    h.retain(|(k, _)| k != "retry-after");
                    h.push(("retry-after".into(), secs.to_string()));
                    return json_response(503, &json!({"error": {"message": format!("[{provider}/{model}] {msg} ({human})")}}), &h);
                }
                Selection::None => {
                    if exclude.is_empty() {
                        return error_response(404, &format!("No active credentials for provider: {provider}"), &[]);
                    }
                    return error_response(last_status.unwrap_or(503), last_error.as_deref().unwrap_or("All accounts unavailable"), &last_headers);
                }
            };
            let connection_id = creds["connectionId"].as_str().unwrap_or("").to_string();
            let mut creds = accounts::check_and_refresh_token(&db, &provider, &creds).await;
            fill_node_defaults(&db, &provider, &mut creds);
            if (provider == "antigravity" || provider == "gemini-cli") && !truthy(&creds["projectId"]) {
                if let Some(pid) = util::fetch_project_id(creds["accessToken"].as_str().unwrap_or(""), &provider).await {
                    creds["projectId"] = json!(pid);
                    let _ = db.update_connection(&connection_id, &json!({"projectId": pid}));
                }
            }
            let st = accounts::settings(&db);
            let conn_snapshot = creds["_connection"].clone();
            let mut b = body.clone();
            b["model"] = json!(format!("{provider}/{model}"));
            let source_override = util::detect_format_by_endpoint(&req.endpoint, &body).map(str::to_owned);
            let res = core::handle_chat_core(CoreArgs {
                db: db.clone(),
                body: b,
                provider: provider.clone(),
                model: model.clone(),
                creds,
                headers: req.headers.clone(),
                endpoint: req.endpoint.clone(),
                source_override,
                connection_id: connection_id.clone(),
                api_key: req.api_key.clone(),
                requested_model: model_str.clone(),
                provider_thinking: st["providerThinking"][&provider].clone(),
                provider_overrides: st["providerOverrides"][&provider].clone(),
            })
            .await;
            if res.ok {
                accounts::clear_account_error(&db, &connection_id, &conn_snapshot, Some(&model));
                return res.response;
            }
            let err = res.error.clone().unwrap_or_default();
            let fb = accounts::mark_account_unavailable(&db, &connection_id, res.status, &err, &provider, Some(&model), res.resets_at_ms);
            if fb.should_fallback {
                let name = res.creds["connectionName"].as_str().unwrap_or(&connection_id).to_string();
                tracing::warn!("FALLBACK ⇄ ACC:{name} UNAVAILABLE ({}) → NEXT ACCOUNT", res.status);
                exclude.insert(connection_id);
                last_error = Some(err);
                last_status = Some(res.status);
                last_headers = res.response.headers().iter().filter(|(k, _)| k.as_str() == "retry-after" || k.as_str() == "x-should-retry" || k.as_str().starts_with("anthropic-ratelimit-")).filter_map(|(k, v)| v.to_str().ok().map(|v| (k.to_string(), v.to_string()))).collect();
                if connection_id_is_virtual(&exclude) {
                    return res.response;
                }
                continue;
            }
            return res.response;
        }
    })
}

fn connection_id_is_virtual(exclude: &HashSet<String>) -> bool {
    exclude.contains("noauth")
}

/// Custom nodes: make sure the connection carries the node's base URL / API type.
fn fill_node_defaults(db: &Db, provider: &str, creds: &mut Value) {
    if !(crate::exec::is_openai_compatible(provider) || crate::exec::is_anthropic_compatible(provider)) {
        return;
    }
    let Some(node) = db.get_node(provider) else { return };
    if !creds["providerSpecificData"].is_object() {
        creds["providerSpecificData"] = json!({});
    }
    for k in ["baseUrl", "apiType", "prefix"] {
        if !truthy(&creds["providerSpecificData"][k]) && truthy(&node[k]) {
            creds["providerSpecificData"][k] = node[k].clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation() {
        let m: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        assert_eq!(rotated_models(&m, "t", "round-robin", 1), m);
        assert_eq!(rotated_models(&m, "t", "round-robin", 1)[0], "b");
        assert_eq!(rotated_models(&m, "t", "fallback", 1)[0], "a");
    }

    #[test]
    fn model_info() {
        let db = Db::open_in_memory().unwrap();
        db.upsert_node("openai-compatible-x", "openai-compatible", &json!({"prefix": "my", "baseUrl": "http://h/v1"})).unwrap();
        db.upsert_combo("best", &["openai/gpt-4o".to_string()]).unwrap();
        db.set_model_alias("fast", "groq/llama").unwrap();
        let p = |s: &str| match get_model_info(&db, s) {
            Some(ModelInfo::Provider { provider, model }) => format!("{provider}|{model}"),
            Some(ModelInfo::Combo(c)) => format!("combo:{c}"),
            None => "none".into(),
        };
        assert_eq!(p("my/llama3"), "openai-compatible-x|llama3");
        assert_eq!(p("cc/claude-x"), format!("{}|claude-x", REG.resolve_alias("cc")));
        assert_eq!(p("best"), "combo:best");
        assert_eq!(p("fast"), "groq|llama");
        assert_eq!(p("claude-sonnet-4"), "anthropic|claude-sonnet-4");
        assert_eq!(p("gpt-5.1"), "codex|gpt-5.1");
    }

    #[test]
    fn caps_required() {
        let b = json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "hi"}]});
        assert!(required_capabilities(&b).is_empty());
        let b2 = json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]});
        assert!(required_capabilities(&b2).contains("vision"));
    }
}
