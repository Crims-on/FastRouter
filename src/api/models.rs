//! `/v1/models`, `/v1/models/{kind}` and `/v1beta/models` (port of
//! src/app/api/v1/models/route.js buildModelsList).

use std::collections::{HashMap, HashSet};

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::Response;
use serde_json::{Value, json};

use crate::AppState;
use crate::chat::accounts;
use crate::chat::core::json_response;
use crate::db::Db;
use crate::exec::{is_anthropic_compatible, is_openai_compatible};
use crate::jsv::truthy;
use crate::registry::REG;

pub const LLM: &str = "llm";
const INTERNAL_HEADER: &str = "x-9r-internal-models-fetch";

fn model_kind(m: &Value) -> &'static str {
    let k = m["kind"].as_str().or_else(|| m["type"].as_str()).unwrap_or("");
    match k {
        "image" => "image",
        "tts" => "tts",
        "embedding" => "embedding",
        "stt" => "stt",
        "imageToText" => "imageToText",
        "video" => "video",
        _ => LLM,
    }
}

fn infer_kind(id: &str) -> &'static str {
    let l = id.to_lowercase();
    if l.contains("embed") {
        "embedding"
    } else if ["tts", "speech", "audio", "voice"].iter().any(|k| l.contains(k)) {
        "tts"
    } else if ["image", "imagen", "dall-e", "dalle", "flux", "sdxl", "sd-", "stable-diffusion"].iter().any(|k| l.contains(k)) {
        "image"
    } else {
        LLM
    }
}

pub fn provider_kinds(id: &str) -> Vec<String> {
    if id.starts_with("custom-embedding-") {
        return vec!["embedding".into()];
    }
    let m = REG.media(id);
    match m["serviceKinds"].as_array() {
        Some(a) if !a.is_empty() => a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect(),
        _ => vec![LLM.into()],
    }
}

fn provider_matches(id: &str, kinds: &[&str]) -> bool {
    if is_openai_compatible(id) || is_anthropic_compatible(id) {
        return kinds.contains(&LLM);
    }
    let pk = provider_kinds(id);
    kinds.iter().any(|k| pk.iter().any(|p| p == k))
}

fn caps_json(provider: &str, model: &str) -> Value {
    crate::caps::caps_for(Some(provider), model).0
}

async fn fetch_compatible_ids(conn: &Value) -> Vec<String> {
    let key = conn["apiKey"].as_str().unwrap_or("");
    let base = conn["providerSpecificData"]["baseUrl"].as_str().unwrap_or("").trim().trim_end_matches('/').to_string();
    if key.is_empty() || base.is_empty() {
        return vec![];
    }
    let provider = conn["provider"].as_str().unwrap_or("");
    let mut url = format!("{base}/models");
    let client = crate::exec::client_for(conn);
    let mut rb;
    if is_openai_compatible(provider) {
        rb = client.get(&url).bearer_auth(key);
    } else if is_anthropic_compatible(provider) {
        if url.ends_with("/messages/models") {
            url.truncate(url.len() - 9);
        }
        rb = client.get(&url).header("x-api-key", key).header("anthropic-version", "2023-06-01").bearer_auth(key);
    } else {
        return vec![];
    }
    rb = rb.header(INTERNAL_HEADER, "1").timeout(std::time::Duration::from_secs(5));
    let Ok(r) = rb.send().await else { return vec![] };
    if !r.status().is_success() {
        return vec![];
    }
    let Ok(d) = r.json::<Value>().await else { return vec![] };
    let arr = if d.is_array() { d } else if d["data"].is_array() { d["data"].clone() } else if d["models"].is_array() { d["models"].clone() } else { d["results"].clone() };
    let mut seen = HashSet::new();
    arr.as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| m["id"].as_str().or_else(|| m["name"].as_str()).or_else(|| m["model"].as_str()).or_else(|| m.as_str()).map(str::to_owned))
        .filter(|s| !s.trim().is_empty() && seen.insert(s.clone()))
        .collect()
}

/// Live catalogs for providers whose model list is per-account.
async fn live_models(provider: &str, conn: &Value) -> Option<Vec<String>> {
    let creds = accounts::credentials_from_connection(conn);
    match provider {
        "zed" => crate::providers::zed::list_model_ids(&creds).await,
        "qoder" | "qoder-cn" => crate::providers::qoder::list_model_ids(provider, &creds).await,
        _ => None,
    }
}

fn strip_prefixes(id: &str, prefixes: &[&str]) -> String {
    for p in prefixes {
        if let Some(rest) = id.strip_prefix(&format!("{p}/")) {
            return rest.to_string();
        }
    }
    id.to_string()
}

/// buildModelsList(kindFilter)
pub async fn build_models_list(db: &Db, kinds: &[&str], skip_dynamic: bool) -> Vec<Value> {
    let conns: Vec<Value> = db.list_connections().into_iter().filter(|c| c["isActive"] != json!(false)).collect();
    let st = accounts::settings(db);
    let custom_models: Vec<Value> = st["customModels"].as_array().cloned().unwrap_or_default();
    let aliases = db.model_aliases();
    let disabled = |alias: &str, id: &str| st["disabledModels"][alias].as_array().map(|a| a.iter().any(|x| x == id)).unwrap_or(false);
    let mut models: Vec<Value> = vec![];

    for combo in db.list_combos() {
        let kind = st["comboKinds"][&combo.name].as_str().unwrap_or(LLM).to_string();
        if !kinds.contains(&kind.as_str()) {
            continue;
        }
        let mut e = json!({"id": combo.name, "object": "model", "owned_by": "combo"});
        if kind == "webSearch" || kind == "webFetch" {
            e["kind"] = json!(kind);
        } else {
            let mut ctx = i64::MAX;
            let mut out = i64::MAX;
            for seat in &combo.models {
                let (p, m) = seat.split_once('/').unwrap_or(("", seat));
                let pid = accounts::provider_id(p);
                let c = crate::caps::caps_for(if p.is_empty() { None } else { Some(&pid) }, m);
                if let Some(v) = c.get("contextWindow").as_i64() {
                    ctx = ctx.min(v);
                }
                if let Some(v) = c.max_output() {
                    out = out.min(v);
                }
            }
            if ctx != i64::MAX {
                e["context_length"] = json!(ctx);
            }
            if out != i64::MAX {
                e["max_completion_tokens"] = json!(out);
            }
        }
        models.push(e);
    }

    let mut first_conn: Vec<(String, Value)> = vec![];
    let mut seen_p = HashSet::new();
    for c in &conns {
        let p = c["provider"].as_str().unwrap_or("").to_string();
        if seen_p.insert(p.clone()) {
            first_conn.push((p, c.clone()));
        }
    }
    // Free no-auth providers are usable without a connection.
    for e in &REG.entries {
        let id = e["id"].as_str().unwrap_or("");
        if e["hidden"] != json!(true) && accounts::is_free_no_auth(id) && seen_p.insert(id.to_string()) {
            first_conn.push((id.to_string(), json!({"provider": id, "providerSpecificData": {}})));
        }
    }

    for (pid, conn) in first_conn {
        if !provider_matches(&pid, kinds) {
            continue;
        }
        let static_alias = REG.alias_of(&pid);
        let node = db.get_node(&pid);
        let prefix = conn["providerSpecificData"]["prefix"].as_str().filter(|s| !s.trim().is_empty()).map(str::to_owned).or_else(|| node.as_ref().and_then(|n| n["prefix"].as_str().map(str::to_owned)));
        let out_alias = prefix.unwrap_or_else(|| static_alias.clone()).trim().to_string();
        let pmodels = REG.provider_models(&static_alias);
        let static_kind: HashMap<String, &'static str> = pmodels.iter().filter_map(|m| m["id"].as_str().map(|i| (i.to_string(), model_kind(m)))).collect();
        let enabled: Vec<String> = conn["providerSpecificData"]["enabledModels"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).filter(|s| !s.trim().is_empty()).collect()).unwrap_or_default();
        let explicit = !enabled.is_empty();
        let compat = is_openai_compatible(&pid) || is_anthropic_compatible(&pid);
        let mut raw: Vec<String> = if explicit { enabled } else { pmodels.iter().filter_map(|m| m["id"].as_str().map(str::to_owned)).collect() };
        if compat && raw.is_empty() && !skip_dynamic {
            let mut c2 = conn.clone();
            if !truthy(&c2["providerSpecificData"]["baseUrl"]) {
                if let Some(n) = &node {
                    c2["providerSpecificData"]["baseUrl"] = n["baseUrl"].clone();
                }
            }
            raw = fetch_compatible_ids(&c2).await;
        }
        if !explicit && !skip_dynamic {
            if let Some(live) = live_models(&pid, &conn).await.filter(|v| !v.is_empty()) {
                raw = live;
            }
        }
        let prefixes = [out_alias.as_str(), static_alias.as_str(), pid.as_str()];
        let ids: Vec<String> = raw.iter().map(|m| strip_prefixes(m, &prefixes)).filter(|m| !m.trim().is_empty()).collect();
        let mut custom_kind: HashMap<String, String> = HashMap::new();
        let custom_ids: Vec<String> = custom_models
            .iter()
            .filter(|m| {
                let a = m["providerAlias"].as_str().unwrap_or("");
                truthy(&m["id"]) && (a == static_alias || a == out_alias || a == pid)
            })
            .filter_map(|m| {
                let id = m["id"].as_str()?.trim().to_string();
                let k = m["type"].as_str().or_else(|| m["kind"].as_str()).unwrap_or(LLM).to_string();
                if !kinds.contains(&k.as_str()) && !(k == "imageToText" && kinds.contains(&LLM)) {
                    return None;
                }
                custom_kind.insert(id.clone(), k);
                Some(id)
            })
            .collect();
        let alias_ids: Vec<String> = aliases
            .values()
            .filter_map(|v| v.as_str())
            .filter(|f| prefixes.iter().any(|p| f.starts_with(&format!("{p}/"))))
            .map(|f| strip_prefixes(f, &prefixes))
            .collect();
        let mut seen = HashSet::new();
        for id in ids.into_iter().chain(custom_ids).chain(alias_ids) {
            if !seen.insert(id.clone()) {
                continue;
            }
            let kind = custom_kind.get(&id).map(|s| s.as_str()).or_else(|| static_kind.get(&id).copied()).unwrap_or_else(|| infer_kind(&id));
            let as_llm = kind == "imageToText" && kinds.contains(&LLM);
            if !kinds.contains(&kind) && !as_llm {
                continue;
            }
            if disabled(&out_alias, &id) || disabled(&static_alias, &id) {
                continue;
            }
            let mut m = json!({"id": format!("{out_alias}/{id}"), "object": "model", "owned_by": out_alias});
            if kind == LLM || as_llm {
                let caps = caps_json(&pid, &id);
                if let Some(v) = caps["contextWindow"].as_i64() {
                    m["context_length"] = json!(v);
                }
                if let Some(v) = caps["maxOutput"].as_i64() {
                    m["max_completion_tokens"] = json!(v);
                }
                m["capabilities"] = caps;
            }
            models.push(m);
        }
        let media = REG.media(&pid);
        if kinds.contains(&"webSearch") && media["searchConfig"].is_object() {
            models.push(json!({"id": format!("{out_alias}/search"), "object": "model", "kind": "webSearch", "owned_by": out_alias}));
        }
        if kinds.contains(&"webFetch") && media["fetchConfig"].is_object() {
            models.push(json!({"id": format!("{out_alias}/fetch"), "object": "model", "kind": "webFetch", "owned_by": out_alias}));
        }
    }
    let mut seen = HashSet::new();
    models.retain(|m| m["id"].as_str().map(|i| seen.insert(i.to_string())).unwrap_or(false));
    models
}

fn cors_json(status: u16, v: &Value) -> Response {
    json_response(status, v, &[])
}

pub async fn list(State(st): State<AppState>, headers: HeaderMap) -> Response {
    let skip = headers.get(INTERNAL_HEADER).and_then(|v| v.to_str().ok()) == Some("1");
    let data = build_models_list(&st.db, &[LLM], skip).await;
    cors_json(200, &json!({"object": "list", "data": data}))
}

pub async fn by_kind(State(st): State<AppState>, Path(path): Path<String>) -> Response {
    let kinds: Option<&[&str]> = match path.as_str() {
        "image" => Some(&["image"]),
        "tts" => Some(&["tts"]),
        "stt" => Some(&["stt"]),
        "embedding" => Some(&["embedding"]),
        "image-to-text" => Some(&["imageToText"]),
        "video" => Some(&["video"]),
        "web" => Some(&["webSearch", "webFetch"]),
        _ => None,
    };
    if let Some(k) = kinds {
        let data = build_models_list(&st.db, k, false).await;
        return cors_json(200, &json!({"object": "list", "data": data}));
    }
    let models = build_models_list(&st.db, &[LLM], false).await;
    match models.into_iter().find(|m| m["id"] == path) {
        Some(m) => cors_json(200, &m),
        None => cors_json(404, &json!({"error": {"message": format!("The model '{path}' does not exist or you do not have access to it."), "type": "invalid_request_error", "code": "model_not_found"}})),
    }
}

/// GET /v1beta/models — Gemini-style listing.
pub async fn gemini_list() -> Response {
    let mut out = vec![];
    let mut seen = HashSet::new();
    let mut add = |name: String, display: &str, desc: String, methods: Value| {
        if seen.insert(name.clone()) {
            out.push(json!({"name": name, "displayName": display, "description": desc, "supportedGenerationMethods": methods, "inputTokenLimit": 128000, "outputTokenLimit": 8192}));
        }
    };
    for (p, list) in REG.models.iter() {
        for m in list.as_array().into_iter().flatten() {
            let id = m["id"].as_str().unwrap_or("");
            let name = m["name"].as_str().unwrap_or(id);
            add(format!("models/{p}/{id}"), name, format!("{p} model: {name}"), json!(["generateContent"]));
            if p == "gemini" {
                add(format!("models/{id}"), name, format!("Gemini model: {name}"), json!(["generateContent", "streamGenerateContent"]));
            }
        }
    }
    cors_json(200, &json!({"models": out}))
}
