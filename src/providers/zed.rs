//! Zed hosted LLM aggregator (cloud.zed.dev) — port of executors/zed.js and
//! the token/model helpers of shared/zedAuth.js.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::exec::{ExecArgs, ExecResult, Executor, Headers, ParsedError, Upstream, client_for};
use crate::jsv::{js_string, now_ms, truthy};
use crate::translate;

pub const ZED_CLOUD_BASE_URL: &str = "https://cloud.zed.dev";
pub const ZED_WEB_BASE_URL: &str = "https://zed.dev";
const LLM_TOKEN_TTL_MS: i64 = 50 * 60 * 1000;
const MODEL_CACHE_TTL_MS: i64 = 60 * 60 * 1000;

const H_EXPIRED: &str = "x-zed-expired-token";
const H_OUTDATED: &str = "x-zed-outdated-token";
const H_STATUS: &str = "x-zed-client-supports-status-messages";
const H_STREAM_ENDED: &str = "x-zed-client-supports-stream-ended-request-completion-status";
const H_XAI: &str = "x-zed-client-supports-x-ai";
const H_SYSTEM: &str = "x-zed-system-id";

static TOKENS: LazyLock<Mutex<HashMap<String, (String, i64)>>> = LazyLock::new(Default::default);
static MODELS: LazyLock<Mutex<HashMap<String, (Value, i64)>>> = LazyLock::new(Default::default);

pub fn user_auth_header(creds: &Value) -> Result<String, String> {
    let psd = &creds["providerSpecificData"];
    let uid = [&psd["userId"], &creds["userId"]].into_iter().find(|v| truthy(v)).map(js_string);
    let tok = crate::exec::cred_str(creds, "accessToken").or_else(|| crate::exec::cred_str(creds, "apiKey"));
    match (uid, tok) {
        (Some(u), Some(t)) => Ok(format!("{u} {t}")),
        _ => Err("Zed credential is missing userId or accessToken".into()),
    }
}

fn system_id(creds: &Value) -> String {
    [&creds["providerSpecificData"]["systemId"], &creds["systemId"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or_default()
}

fn norm_id(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(a) => a.first().and_then(|x| x.as_str()).unwrap_or("").to_string(),
        Value::Object(_) => v["id"].as_str().map(str::to_owned).unwrap_or_else(|| v.to_string()),
        o => js_string(o),
    }
}

async fn fetch_json(creds: &Value, method: reqwest::Method, url: &str, h: &Headers, body: Option<Value>) -> Result<Value, (u16, String)> {
    let mut rb = client_for(creds).request(method, url);
    for (k, v) in &h.0 {
        rb = rb.header(k, v);
    }
    if let Some(b) = body {
        rb = rb.body(b.to_string());
    }
    let r = rb.send().await.map_err(|e| (0, e.to_string()))?;
    let st = r.status().as_u16();
    let text = r.text().await.unwrap_or_default();
    let data: Value = if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or_else(|_| json!({"raw": text})) };
    if !(200..300).contains(&st) {
        let m = [&data["message"], &data["error"]["message"], &data["error"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or(if text.is_empty() { format!("HTTP {st}") } else { text });
        return Err((st, m));
    }
    Ok(data)
}

pub async fn fetch_user(creds: &Value) -> Result<Value, String> {
    let mut h = Headers::default();
    h.set("Accept", "application/json");
    h.set("Authorization", user_auth_header(creds)?);
    let sid = system_id(creds);
    if !sid.is_empty() {
        h.set(H_SYSTEM, sid);
    }
    fetch_json(creds, reqwest::Method::GET, &format!("{ZED_CLOUD_BASE_URL}/client/users/me"), &h, None).await.map_err(|e| e.1)
}

pub fn resolve_org_id(creds: &Value, user: Option<&Value>) -> String {
    let psd = &creds["providerSpecificData"];
    let explicit = norm_id(if truthy(&psd["organizationId"]) { &psd["organizationId"] } else { &psd["defaultOrganizationId"] });
    if !explicit.is_empty() {
        return explicit;
    }
    let Some(u) = user else { return String::new() };
    let from_user = norm_id(if truthy(&u["default_organization_id"]) { &u["default_organization_id"] } else { &u["defaultOrganizationId"] });
    if !from_user.is_empty() {
        return from_user;
    }
    let orgs = u["organizations"].as_array().cloned().unwrap_or_default();
    let org = orgs.iter().find(|o| truthy(&o["is_personal"])).or(orgs.first());
    org.map(|o| norm_id(&o["id"])).unwrap_or_default()
}

fn token_tail(creds: &Value) -> String {
    let t = crate::exec::api_key_or_token(creds);
    let t = if let Some(a) = crate::exec::cred_str(creds, "accessToken") { a.to_string() } else { t };
    let n = t.chars().count();
    t.chars().skip(n.saturating_sub(16)).collect()
}

pub async fn fetch_llm_token(creds: &Value, force: bool) -> Result<String, String> {
    let mut org = resolve_org_id(creds, None);
    if org.is_empty() {
        let user = fetch_user(creds).await?;
        org = resolve_org_id(creds, Some(&user));
    }
    if org.is_empty() {
        return Err("No Zed organization selected".into());
    }
    let psd = &creds["providerSpecificData"];
    let uid = [&psd["userId"], &creds["userId"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or_else(|| "unknown".into());
    let key = format!("{uid}:{org}:{}", token_tail(creds));
    if !force {
        if let Some((t, exp)) = TOKENS.lock().unwrap().get(&key).cloned() {
            if exp > now_ms() {
                return Ok(t);
            }
        }
    }
    let mut h = Headers::default();
    h.set("Content-Type", "application/json");
    h.set("Accept", "application/json");
    h.set("Authorization", user_auth_header(creds)?);
    let sid = system_id(creds);
    if !sid.is_empty() {
        h.set(H_SYSTEM, sid);
    }
    let data = fetch_json(creds, reqwest::Method::POST, &format!("{ZED_CLOUD_BASE_URL}/client/llm_tokens"), &h, Some(json!({"organization_id": org}))).await.map_err(|e| e.1)?;
    let token = data["token"].as_str().map(str::to_owned).or_else(|| data["token"][0].as_str().map(str::to_owned)).or_else(|| data["token"]["value"].as_str().map(str::to_owned)).ok_or("Zed did not return an LLM token")?;
    TOKENS.lock().unwrap().insert(key, (token.clone(), now_ms() + LLM_TOKEN_TTL_MS));
    Ok(token)
}

/// zedLlmFetch: LLM bearer request, re-minting the token once on 401/expired.
pub async fn llm_fetch(creds: &Value, method: reqwest::Method, path: &str, headers: &Headers, body: Option<&Value>) -> Result<reqwest::Response, String> {
    let url = format!("{ZED_CLOUD_BASE_URL}{path}");
    let mut force = false;
    loop {
        let token = fetch_llm_token(creds, force).await?;
        let mut rb = client_for(creds).request(method.clone(), &url);
        for (k, v) in &headers.0 {
            rb = rb.header(k, v);
        }
        rb = rb.header("Authorization", format!("Bearer {token}"));
        if let Some(b) = body {
            rb = rb.body(b.to_string());
        }
        let r = rb.send().await.map_err(|e| e.to_string())?;
        let refresh = r.status().as_u16() == 401 || r.headers().contains_key(H_EXPIRED) || r.headers().contains_key(H_OUTDATED);
        if refresh && !force {
            force = true;
            continue;
        }
        return Ok(r);
    }
}

pub fn map_model(m: &Value) -> Option<Value> {
    let id = norm_id(&m["id"]);
    if id.is_empty() {
        return None;
    }
    let pick = |a: &str, b: &str| if !m[a].is_null() { m[a].clone() } else { m[b].clone() };
    let name = [&m["display_name"], &m["displayName"]].into_iter().find(|v| truthy(v)).cloned().unwrap_or(json!(id));
    Some(json!({
        "id": id,
        "name": name,
        "provider": m["provider"],
        "isLatest": truthy(&m["is_latest"]),
        "contextLength": pick("max_token_count", "maxTokenCount"),
        "maxOutputTokens": pick("max_output_tokens", "maxOutputTokens"),
        "supportsTools": truthy(&m["supports_tools"]),
        "supportsImages": truthy(&m["supports_images"]),
        "supportsThinking": truthy(&m["supports_thinking"]),
        "isDisabled": truthy(&m["is_disabled"]),
    }))
}

/// resolveZedModels → { models, rawById } (cached for an hour).
pub async fn resolve_models(creds: &Value, force: bool) -> Result<Value, String> {
    if !truthy(&creds["accessToken"]) {
        return Ok(Value::Null);
    }
    let psd = &creds["providerSpecificData"];
    let org = [&psd["organizationId"], &psd["defaultOrganizationId"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or_else(|| "default".into());
    let key = format!("{}:{org}:{}", psd["userId"].as_str().unwrap_or("unknown"), token_tail(creds));
    if !force {
        if let Some((v, exp)) = MODELS.lock().unwrap().get(&key).cloned() {
            if exp > now_ms() {
                return Ok(v);
            }
        }
    }
    let mut h = Headers::default();
    h.set("Accept", "application/json");
    h.set(H_XAI, "true");
    let r = llm_fetch(creds, reqwest::Method::GET, "/models", &h, None).await?;
    let st = r.status().as_u16();
    let text = r.text().await.unwrap_or_default();
    if !(200..300).contains(&st) {
        return Err(format!("Zed models failed: {st} {text}"));
    }
    let data: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let raw = data["models"].as_array().cloned().unwrap_or_default();
    let models: Vec<Value> = raw.iter().filter_map(map_model).filter(|m| !truthy(&m["isDisabled"])).collect();
    let mut by_id = serde_json::Map::new();
    for r in &raw {
        let id = norm_id(&r["id"]);
        if !id.is_empty() {
            by_id.insert(id, r.clone());
        }
    }
    let entry = json!({"models": models, "rawById": by_id, "defaultModel": norm_id(&data["default_model"])});
    MODELS.lock().unwrap().insert(key, (entry.clone(), now_ms() + MODEL_CACHE_TTL_MS));
    Ok(entry)
}

pub fn normalize_provider(value: &Value, model: &str) -> &'static str {
    match value.as_str().unwrap_or("").to_lowercase().as_str() {
        "anthropic" => return "anthropic",
        "openai" | "open_ai" => return "open_ai",
        "google" | "gemini" => return "google",
        "xai" | "x_ai" | "x-ai" => return "x_ai",
        _ => {}
    }
    let m = model.to_lowercase();
    if m.contains("claude") {
        "anthropic"
    } else if m.contains("gemini") {
        "google"
    } else if m.contains("grok") || m.contains("xai") {
        "x_ai"
    } else {
        "open_ai"
    }
}

fn provider_format(p: &str) -> &'static str {
    match p {
        "anthropic" => translate::CLAUDE,
        "google" => translate::GEMINI,
        "open_ai" => translate::OPENAI_RESPONSES,
        _ => translate::OPENAI,
    }
}

pub fn build_provider_request(provider: &str, model: &str, body: &Value, stream: bool) -> Value {
    match provider {
        "anthropic" => translate::req::openai_to_claude(model, body, true),
        "google" => {
            let mut g = translate::req::openai_to_gemini(model, body, &Default::default());
            crate::jsv::del(&mut g, "safetySettings");
            g
        }
        "open_ai" => translate::req::openai_to_responses(model, body),
        _ => {
            let mut b = if body.is_object() { body.clone() } else { json!({}) };
            b["model"] = json!(model);
            b["stream"] = json!(stream);
            b
        }
    }
}

enum Line {
    Done,
    Event(Value),
    Status(Value),
}

fn unwrap_line(line: &str) -> Option<Line> {
    let mut t = line.trim_end_matches('\r').trim();
    if t.is_empty() {
        return None;
    }
    if let Some(r) = t.strip_prefix("data:") {
        t = r.trim_start();
    }
    if t == "[DONE]" {
        return Some(Line::Done);
    }
    let v: Value = serde_json::from_str(t).ok()?;
    if v.get("event").is_some() {
        return Some(Line::Event(v["event"].clone()));
    }
    if v.get("status").is_some() {
        return Some(Line::Status(v["status"].clone()));
    }
    Some(Line::Event(v))
}

fn normalize_status(s: &Value) -> Option<Value> {
    match s {
        Value::Null => None,
        Value::String(x) => Some(json!({"type": x})),
        Value::Object(o) => {
            if let Some((k, v)) = o.iter().next() {
                if v.is_object() {
                    let mut out = json!({"type": k});
                    for (a, b) in v.as_object().unwrap() {
                        out[a] = b.clone();
                    }
                    return Some(out);
                }
            }
            Some(s.clone())
        }
        _ => None,
    }
}

fn error_chunk(model: &str, msg: &str) -> Value {
    json!({"id": format!("chatcmpl-zed-error-{}", now_ms()), "object": "chat.completion.chunk", "created": now_ms() / 1000, "model": model,
        "choices": [{"index": 0, "delta": {"content": format!("[Zed error] {msg}")}, "finish_reason": "stop"}]})
}

fn wrap_stream(up: Upstream, provider: &'static str, model: String) -> Upstream {
    let fmt = provider_format(provider);
    let mut body = up.body;
    let s = async_stream::stream! {
        let mut state = translate::resp::init_state(fmt);
        if fmt == translate::OPENAI { state["model"] = json!(model); }
        let mut lines = crate::sse::LineParser::default();
        let mut done = false;
        let emit = |items: Vec<Value>| items.into_iter().filter(|i| !i.is_null()).map(|i| Ok::<Bytes, String>(Bytes::from(format!("data: {i}\n\n")))).collect::<Vec<_>>();
        let mut pending: Vec<String> = vec![];
        let mut ended = false;
        while !done {
            match body.next().await {
                Some(Ok(c)) => pending.extend(lines.push(&c)),
                Some(Err(e)) => { yield Err(e); return; }
                None => { if let Some(l) = lines.finish() { pending.push(l); } ended = true; }
            }
            for line in pending.drain(..) {
                if done { break; }
                match unwrap_line(&line) {
                    None => {}
                    Some(Line::Done) => {
                        for b in emit(translate::translate_response(fmt, translate::OPENAI, None, &mut state)) { yield b; }
                        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                        done = true;
                    }
                    Some(Line::Status(st)) => {
                        let st = normalize_status(&st);
                        let ty = st.as_ref().map(|s| js_string(&s["type"])).unwrap_or_default();
                        let failed = st.as_ref().map(|s| truthy(&s["failed"])).unwrap_or(false);
                        if ty == "failed" || failed {
                            let s = st.unwrap();
                            let f = if failed { s["failed"].clone() } else { s };
                            let m = [&f["message"], &f["error"], &f["code"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or_else(|| "request failed".into());
                            yield Ok(Bytes::from(format!("data: {}\n\n", error_chunk(&model, &m))));
                            for b in emit(translate::translate_response(fmt, translate::OPENAI, None, &mut state)) { yield b; }
                            yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                            done = true;
                        } else if ty == "stream_ended" {
                            for b in emit(translate::translate_response(fmt, translate::OPENAI, None, &mut state)) { yield b; }
                            yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                            done = true;
                        }
                    }
                    Some(Line::Event(ev)) => {
                        for b in emit(translate::translate_response(fmt, translate::OPENAI, Some(&ev), &mut state)) { yield b; }
                    }
                }
            }
            if ended && !done {
                for b in emit(translate::translate_response(fmt, translate::OPENAI, None, &mut state)) { yield b; }
                yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                done = true;
            }
        }
    };
    Upstream::synthetic(up.status, "text/event-stream", s.boxed())
}

pub struct Zed;

#[async_trait]
impl Executor for Zed {
    fn provider(&self) -> &str {
        "zed"
    }
    async fn refresh_credentials(&self, _c: &Value) -> Option<Value> {
        None
    }
    fn parse_error(&self, status: u16, body: &str) -> ParsedError {
        let p: Value = serde_json::from_str(if body.is_empty() { "{}" } else { body }).unwrap_or(Value::Null);
        let code = [&p["code"], &p["error"]["code"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or_default();
        let raw = [&p["message"], &p["error"]["message"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or_else(|| body.to_string());
        let message = if code == "trial_blocked" {
            format!("Zed trial access is blocked upstream. The account can list hosted models, but Zed is refusing completions until trial/billing access is enabled or unblocked. Zed says: {raw}")
        } else if !code.is_empty() {
            format!("Zed {code}: {raw}")
        } else if raw.is_empty() {
            format!("Zed upstream error: {status}")
        } else {
            raw
        };
        ParsedError { status, message, resets_at_ms: None }
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let model = args.model.to_string();
        let raw_provider = match resolve_models(args.creds, false).await {
            Ok(cat) => {
                let mut raw = cat["rawById"][&model].clone();
                if raw.is_null() {
                    if let Ok(c2) = resolve_models(args.creds, true).await {
                        raw = c2["rawById"][&model].clone();
                    }
                }
                raw["provider"].clone()
            }
            Err(e) => {
                tracing::warn!("zed model catalog unavailable, inferring provider for {model}: {e}");
                Value::Null
            }
        };
        let provider = normalize_provider(&raw_provider, &model);
        let preq = build_provider_request(provider, &model, &args.body, args.stream);
        let thread = if truthy(&args.body["thread_id"]) { args.body["thread_id"].clone() } else { args.creds["_clientSessionId"].clone() };
        let mut payload = json!({"thread_id": thread, "prompt_id": args.body["prompt_id"], "provider": provider, "model": model, "provider_request": preq});
        if payload["thread_id"].is_null() {
            crate::jsv::del(&mut payload, "thread_id");
        }
        if payload["prompt_id"].is_null() {
            crate::jsv::del(&mut payload, "prompt_id");
        }
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.set("Accept", "application/x-ndjson, text/event-stream, */*");
        h.set("User-Agent", "9router/zed");
        h.set("x-zed-version", "0.200.0");
        h.set(H_STATUS, "true");
        h.set(H_STREAM_ENDED, "true");
        let url = format!("{ZED_CLOUD_BASE_URL}/completions");
        let resp = llm_fetch(args.creds, reqwest::Method::POST, "/completions", &h, Some(&payload)).await?;
        let up = Upstream::from_reqwest(resp);
        let ok = up.ok();
        let up = if ok { wrap_stream(up, provider, model) } else { up };
        Ok(ExecResult {
            response: up,
            url,
            headers: vec![("Content-Type".into(), "application/json".into()), ("Authorization".into(), "Bearer <zed-llm-token>".into())],
            body: payload,
            response_format: ok.then(|| "openai".into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn providers_and_lines() {
        assert_eq!(normalize_provider(&json!("Google"), "x"), "google");
        assert_eq!(normalize_provider(&Value::Null, "claude-4"), "anthropic");
        assert_eq!(normalize_provider(&Value::Null, "grok-4"), "x_ai");
        assert_eq!(normalize_provider(&Value::Null, "gpt-5"), "open_ai");
        assert!(matches!(unwrap_line("data: [DONE]"), Some(Line::Done)));
        assert!(matches!(unwrap_line("{\"status\":\"stream_ended\"}"), Some(Line::Status(_))));
        assert_eq!(normalize_status(&json!({"failed": {"message": "x"}})).unwrap()["type"], "failed");
        assert_eq!(user_auth_header(&json!({"accessToken": "t", "providerSpecificData": {"userId": 5}})).unwrap(), "5 t");
    }
}

/// Model ids from the live Zed catalog (for /v1/models).
pub async fn list_model_ids(creds: &Value) -> Option<Vec<String>> {
    let r = resolve_models(creds, false).await.ok()?;
    Some(r["models"].as_array()?.iter().filter_map(|m| m["id"].as_str().map(str::to_owned)).collect())
}
