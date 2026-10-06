//! Account selection, per-model locks and backoff (port of src/sse/services/auth.js,
//! open-sse/services/accountFallback.js and src/sse/services/tokenRefresh.js).

use std::collections::HashSet;
use std::sync::LazyLock;

use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::db::Db;
use crate::jsv::{iso_from_ms, js_string, now_ms, parse_iso_ms, truthy};
use crate::registry::REG;

pub const BACKOFF_BASE_MS: i64 = 2000;
pub const BACKOFF_MAX_MS: i64 = 5 * 60 * 1000;
pub const BACKOFF_MAX_LEVEL: i64 = 15;
pub const TRANSIENT_COOLDOWN_MS: i64 = 30 * 1000;
pub const MAX_RATE_LIMIT_COOLDOWN_MS: i64 = 30 * 60 * 1000;
const COOLDOWN_LONG: i64 = 2 * 60 * 1000;
const COOLDOWN_SHORT: i64 = 5 * 1000;
pub const MODEL_LOCK_PREFIX: &str = "modelLock_";
pub const MODEL_LOCK_ALL: &str = "modelLock___all";
const TOKEN_EXPIRY_BUFFER_MS: i64 = 5 * 60 * 1000;

static SELECTION: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

enum Rule {
    Text(Option<&'static str>, &'static str, Option<i64>),
    Status(u16, Option<i64>),
}

/// ERROR_RULES (None cooldown = exponential backoff).
const RULES: &[Rule] = &[
    Rule::Text(Some("codex"), "model is not supported when using codex with a chatgpt account", Some(MAX_RATE_LIMIT_COOLDOWN_MS)),
    Rule::Text(None, "no credentials", Some(COOLDOWN_LONG)),
    Rule::Text(None, "request not allowed", Some(COOLDOWN_SHORT)),
    Rule::Text(None, "improperly formed request", Some(COOLDOWN_LONG)),
    Rule::Text(None, "rate limit", None),
    Rule::Text(None, "too many requests", None),
    Rule::Text(None, "quota exceeded", None),
    Rule::Text(None, "capacity", None),
    Rule::Text(None, "overloaded", None),
    Rule::Status(401, Some(COOLDOWN_LONG)),
    Rule::Status(402, Some(COOLDOWN_LONG)),
    Rule::Status(403, Some(COOLDOWN_LONG)),
    Rule::Status(404, Some(COOLDOWN_LONG)),
    Rule::Status(429, None),
];

pub fn quota_cooldown(level: i64) -> i64 {
    let l = (level - 1).max(0).min(40);
    (BACKOFF_BASE_MS.saturating_mul(1i64 << l.min(30))).min(BACKOFF_MAX_MS)
}

pub struct Fallback {
    pub should_fallback: bool,
    pub cooldown_ms: i64,
    pub new_backoff_level: Option<i64>,
}

/// checkFallbackError(status, errorText, backoffLevel, provider)
pub fn check_fallback_error(status: u16, error_text: &str, backoff_level: i64, provider: Option<&str>) -> Fallback {
    let lower = error_text.to_lowercase();
    let backoff = |lvl: i64| {
        let n = (lvl + 1).min(BACKOFF_MAX_LEVEL);
        Fallback { should_fallback: true, cooldown_ms: quota_cooldown(n), new_backoff_level: Some(n) }
    };
    for r in RULES {
        match r {
            Rule::Text(p, text, cd) => {
                if p.is_some() && *p != provider {
                    continue;
                }
                if !lower.is_empty() && lower.contains(text) {
                    return match cd {
                        Some(c) => Fallback { should_fallback: true, cooldown_ms: *c, new_backoff_level: None },
                        None => backoff(backoff_level),
                    };
                }
            }
            Rule::Status(s, cd) => {
                if *s == status {
                    return match cd {
                        Some(c) => Fallback { should_fallback: true, cooldown_ms: *c, new_backoff_level: None },
                        None => backoff(backoff_level),
                    };
                }
            }
        }
    }
    if (400..500).contains(&status) && !matches!(status, 401 | 402 | 403 | 429) {
        return Fallback { should_fallback: false, cooldown_ms: 0, new_backoff_level: None };
    }
    Fallback { should_fallback: true, cooldown_ms: TRANSIENT_COOLDOWN_MS, new_backoff_level: None }
}

pub fn model_lock_key(model: Option<&str>) -> String {
    match model.filter(|m| !m.is_empty()) {
        Some(m) => format!("{MODEL_LOCK_PREFIX}{m}"),
        None => MODEL_LOCK_ALL.to_string(),
    }
}

fn time_of(v: &Value) -> Option<i64> {
    match v {
        Value::String(s) if !s.is_empty() => parse_iso_ms(s),
        Value::Number(n) => n.as_i64(),
        _ => None,
    }
}

pub fn is_model_lock_active(conn: &Value, model: Option<&str>) -> bool {
    let key = model_lock_key(model);
    let v = if truthy(&conn[&key]) { &conn[&key] } else { &conn[MODEL_LOCK_ALL] };
    time_of(v).map(|t| t > now_ms()).unwrap_or(false)
}

pub fn earliest_model_lock_until(conn: &Value) -> Option<i64> {
    let now = now_ms();
    conn.as_object()?
        .iter()
        .filter(|(k, _)| k.starts_with(MODEL_LOCK_PREFIX))
        .filter_map(|(_, v)| time_of(v))
        .filter(|t| *t > now)
        .min()
}

/// Settings document (global strategies, per-provider overrides, toggles).
pub fn settings(db: &Db) -> Value {
    let s = db.setting_json("settings");
    if s.is_object() { s } else { json!({}) }
}

pub fn provider_id(provider: &str) -> String {
    match provider {
        "xmtp" => "xiaomi-tokenplan".into(),
        p => REG.resolve_alias(p),
    }
}

pub fn is_free_no_auth(provider: &str) -> bool {
    REG.entry(provider).map(|e| e["category"] == "free").unwrap_or(false) && REG.transport(provider)["noAuth"] == json!(true)
}

pub enum Selection {
    Creds(Value),
    AllRateLimited { retry_after_ms: i64, last_error: Option<String>, last_error_code: Option<Value> },
    None,
}

/// getProviderCredentials(provider, exclude, model, {requestedModel})
pub async fn get_provider_credentials(db: &Db, provider: &str, exclude: &HashSet<String>, model: Option<&str>, requested_model: Option<&str>) -> Selection {
    let _guard = SELECTION.lock().await;
    let pid = provider_id(provider);
    if is_free_no_auth(&pid) {
        return Selection::Creds(json!({
            "id": "noauth", "connectionId": "noauth", "connectionName": "Public", "isActive": true,
            "accessToken": "public", "providerSpecificData": {},
        }));
    }
    let conns = db.connections_for(&pid, true);
    if conns.is_empty() {
        tracing::warn!("No credentials for {provider}");
        return Selection::None;
    }
    let requested = requested_model.or(model);
    let available: Vec<&Value> = conns
        .iter()
        .filter(|c| {
            let id = c["id"].as_str().unwrap_or("");
            if exclude.contains(id) || is_model_lock_active(c, model) {
                return false;
            }
            if pid == "codex" {
                if let (Some(en), Some(rm)) = (c["providerSpecificData"]["enabledModels"].as_array(), requested) {
                    if !en.is_empty() && !en.iter().any(|m| m == rm) {
                        return false;
                    }
                }
            }
            true
        })
        .collect();
    if available.is_empty() {
        let locked: Vec<&Value> = conns.iter().filter(|c| is_model_lock_active(c, model)).collect();
        if let Some(earliest) = locked.iter().filter_map(|c| earliest_model_lock_until(c)).min() {
            let first = locked[0];
            return Selection::AllRateLimited {
                retry_after_ms: earliest,
                last_error: first["lastError"].as_str().map(str::to_owned),
                last_error_code: Some(first["errorCode"].clone()).filter(|v| !v.is_null()),
            };
        }
        return Selection::None;
    }
    let st = settings(db);
    let po = &st["providerStrategies"][&pid];
    let strategy = po["fallbackStrategy"].as_str().or_else(|| st["fallbackStrategy"].as_str()).unwrap_or("fill-first");
    let conn: Value = if strategy == "round-robin" {
        let sticky = po["stickyRoundRobinLimit"].as_i64().or_else(|| st["stickyRoundRobinLimit"].as_i64()).filter(|n| *n > 0).unwrap_or(3);
        let last = |c: &Value| time_of(&c["lastUsedAt"]);
        let prio = |c: &Value| c["priority"].as_i64().filter(|p| *p != 0).unwrap_or(999);
        let mut by_recency = available.clone();
        by_recency.sort_by(|a, b| match (last(a), last(b)) {
            (None, None) => prio(a).cmp(&prio(b)),
            (None, _) => std::cmp::Ordering::Greater,
            (_, None) => std::cmp::Ordering::Less,
            (Some(x), Some(y)) => y.cmp(&x),
        });
        let cur = by_recency[0];
        let count = cur["consecutiveUseCount"].as_i64().unwrap_or(0);
        if last(cur).is_some() && count < sticky {
            let _ = db.update_connection(cur["id"].as_str().unwrap_or(""), &json!({"lastUsedAt": iso_from_ms(now_ms()), "consecutiveUseCount": count + 1}));
            cur.clone()
        } else {
            let mut oldest = available.clone();
            oldest.sort_by(|a, b| match (last(a), last(b)) {
                (None, None) => prio(a).cmp(&prio(b)),
                (None, _) => std::cmp::Ordering::Less,
                (_, None) => std::cmp::Ordering::Greater,
                (Some(x), Some(y)) => x.cmp(&y),
            });
            let c = oldest[0];
            let _ = db.update_connection(c["id"].as_str().unwrap_or(""), &json!({"lastUsedAt": iso_from_ms(now_ms()), "consecutiveUseCount": 1}));
            c.clone()
        }
    } else {
        available[0].clone()
    };
    Selection::Creds(credentials_from_connection(&conn))
}

/// The credentials object executors see, built from a stored connection.
pub fn credentials_from_connection(conn: &Value) -> Value {
    let mut c = json!({});
    for k in ["authType", "apiKey", "accessToken", "refreshToken", "idToken", "expiresAt", "expiresIn", "lastRefreshAt", "projectId", "testStatus", "lastError", "email"] {
        if !conn[k].is_null() {
            c[k] = conn[k].clone();
        }
    }
    let name = ["displayName", "name", "email", "id"].iter().map(|k| &conn[*k]).find(|v| truthy(v)).cloned().unwrap_or(Value::Null);
    c["connectionName"] = name;
    c["connectionId"] = conn["id"].clone();
    c["providerSpecificData"] = if conn["providerSpecificData"].is_object() { conn["providerSpecificData"].clone() } else { json!({}) };
    if truthy(&conn["providerSpecificData"]["copilotToken"]) {
        c["copilotToken"] = conn["providerSpecificData"]["copilotToken"].clone();
    }
    c["_connection"] = conn.clone();
    c
}

fn github_monthly_reset_ms(status: u16, text: &str, provider: &str) -> Option<i64> {
    if provider_id(provider) != "github" || status != 402 || !text.to_lowercase().contains("you've reached your additional usage limit for your plan") {
        return None;
    }
    let iso = iso_from_ms(now_ms());
    let y: i64 = iso[0..4].parse().ok()?;
    let m: i64 = iso[5..7].parse().ok()?;
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    Some(crate::jsv::days_from_civil(ny, nm, 1) * 86_400_000)
}

/// markAccountUnavailable(connectionId, status, errorText, provider, model, resetsAtMs)
pub fn mark_account_unavailable(db: &Db, connection_id: &str, status: u16, error_text: &str, provider: &str, model: Option<&str>, resets_at_ms: Option<i64>) -> Fallback {
    if connection_id.is_empty() || connection_id == "noauth" {
        return Fallback { should_fallback: false, cooldown_ms: 0, new_backoff_level: None };
    }
    let conn = db.get_connection(connection_id);
    let level = conn.as_ref().and_then(|c| c["backoffLevel"].as_i64()).unwrap_or(0);
    let gh = github_monthly_reset_ms(status, error_text, provider);
    let now = now_ms();
    let fb = if let Some(g) = gh {
        Fallback { should_fallback: true, cooldown_ms: g - now, new_backoff_level: Some(0) }
    } else if let Some(r) = resets_at_ms.filter(|r| *r > now) {
        let cd = if provider_id(provider) == "antigravity" { r - now } else { (r - now).min(MAX_RATE_LIMIT_COOLDOWN_MS) };
        Fallback { should_fallback: true, cooldown_ms: cd, new_backoff_level: Some(0) }
    } else {
        check_fallback_error(status, error_text, level, Some(&provider_id(provider)))
    };
    if !fb.should_fallback {
        return fb;
    }
    let reason: String = if error_text.is_empty() { "Provider error".into() } else { error_text.chars().take(200).collect() };
    let key = model_lock_key(if gh.is_some() { None } else { model });
    let mut patch = json!({
        "testStatus": "unavailable", "lastError": reason, "errorCode": status,
        "lastErrorAt": iso_from_ms(now), "backoffLevel": fb.new_backoff_level.unwrap_or(level),
    });
    patch[&key] = json!(iso_from_ms(now + fb.cooldown_ms));
    let _ = db.update_connection(connection_id, &patch);
    let name = conn.as_ref().map(|c| js_string(&c["name"])).unwrap_or_default();
    tracing::warn!("{} locked {key} for {}s [{status}]", if name.is_empty() || name == "null" { connection_id.chars().take(8).collect() } else { name }, fb.cooldown_ms / 1000);
    fb
}

/// clearAccountError(connectionId, currentConnection, model)
pub fn clear_account_error(db: &Db, connection_id: &str, conn: &Value, model: Option<&str>) {
    if connection_id.is_empty() || connection_id == "noauth" {
        return;
    }
    let now = now_ms();
    let Some(obj) = conn.as_object() else { return };
    let locks: Vec<&String> = obj.keys().filter(|k| k.starts_with(MODEL_LOCK_PREFIX)).collect();
    if !truthy(&conn["testStatus"]) && !truthy(&conn["lastError"]) && locks.is_empty() {
        return;
    }
    let model_key = model.map(|m| format!("{MODEL_LOCK_PREFIX}{m}"));
    let to_clear: Vec<&String> = locks
        .iter()
        .copied()
        .filter(|k| {
            if model.is_some() && (Some(k.as_str()) == model_key.as_deref() || k.as_str() == MODEL_LOCK_ALL) {
                return true;
            }
            time_of(&conn[k.as_str()]).map(|t| t <= now).unwrap_or(false)
        })
        .collect();
    if to_clear.is_empty() && conn["testStatus"] != "unavailable" && !truthy(&conn["lastError"]) {
        return;
    }
    let remaining = locks.iter().filter(|k| !to_clear.contains(k)).filter(|k| time_of(&conn[k.as_str()]).map(|t| t > now).unwrap_or(false)).count();
    let mut patch = json!({});
    for k in &to_clear {
        patch[k.as_str()] = Value::Null;
    }
    if remaining == 0 {
        for (k, v) in [("testStatus", json!("active")), ("lastError", Value::Null), ("errorCode", Value::Null), ("lastErrorAt", Value::Null), ("backoffLevel", json!(0))] {
            patch[k] = v;
        }
    }
    let _ = db.update_connection(connection_id, &patch);
}

fn expires_at_from(v: &Value) -> Option<String> {
    time_of(v).map(iso_from_ms)
}

/// updateProviderCredentials(connectionId, newCredentials)
pub fn update_provider_credentials(db: &Db, connection_id: &str, nc: &Value, existing_psd: &Value) -> bool {
    if connection_id.is_empty() || connection_id == "noauth" {
        return false;
    }
    let mut u = json!({});
    for k in ["accessToken", "refreshToken", "idToken", "lastRefreshAt", "expiresAt", "projectId"] {
        if truthy(&nc[k]) {
            u[k] = nc[k].clone();
        }
    }
    if let Some(n) = nc["expiresIn"].as_f64().filter(|n| *n > 0.0) {
        u["expiresAt"] = json!(iso_from_ms(now_ms() + (n * 1000.0) as i64));
        u["expiresIn"] = nc["expiresIn"].clone();
    } else if truthy(&nc["expiresAt"]) {
        if let Some(e) = expires_at_from(&nc["expiresAt"]) {
            let ms = parse_iso_ms(&e).unwrap_or(0);
            u["expiresIn"] = json!(((ms - now_ms()) / 1000).max(1));
            u["expiresAt"] = json!(e);
        }
    }
    let mut psd: Option<Value> = None;
    if nc["providerSpecificData"].is_object() {
        let mut p = if existing_psd.is_object() { existing_psd.clone() } else { json!({}) };
        for (k, v) in nc["providerSpecificData"].as_object().unwrap() {
            p[k] = v.clone();
        }
        psd = Some(p);
    }
    if truthy(&nc["copilotToken"]) || truthy(&nc["copilotTokenExpiresAt"]) {
        let mut p = psd.clone().unwrap_or_else(|| if existing_psd.is_object() { existing_psd.clone() } else { json!({}) });
        if truthy(&nc["copilotToken"]) {
            p["copilotToken"] = nc["copilotToken"].clone();
        }
        if truthy(&nc["copilotTokenExpiresAt"]) {
            p["copilotTokenExpiresAt"] = nc["copilotTokenExpiresAt"].clone();
        }
        psd = Some(p);
    }
    if let Some(p) = psd {
        u["providerSpecificData"] = p;
    }
    if nc["testStatus"].is_string() {
        u["testStatus"] = nc["testStatus"].clone();
    }
    // Stored providerSpecificData is replaced (not shallow-merged) here, so send
    // the merged object; merge_doc merges objects shallowly anyway.
    matches!(db.update_connection(connection_id, &u), Ok(Some(_)))
}

/// checkAndRefreshToken(provider, credentials): proactive refresh before dispatch.
pub async fn check_and_refresh_token(db: &Db, provider: &str, credentials: &Value) -> Value {
    let mut creds = credentials.clone();
    let cid = creds["connectionId"].as_str().unwrap_or("").to_string();
    if !cid.is_empty() && cid != "noauth" {
        if let Some(latest) = db.get_connection(&cid) {
            let l = time_of(&latest["lastRefreshAt"]);
            let c = time_of(&creds["lastRefreshAt"]);
            let newer = l.is_some() && (c.is_none() || l > c);
            if newer && truthy(&latest["refreshToken"]) && latest["refreshToken"] != creds["refreshToken"] {
                creds["refreshToken"] = latest["refreshToken"].clone();
                if truthy(&latest["accessToken"]) {
                    creds["accessToken"] = latest["accessToken"].clone();
                }
                if truthy(&latest["expiresAt"]) {
                    creds["expiresAt"] = latest["expiresAt"].clone();
                }
                creds["lastRefreshAt"] = latest["lastRefreshAt"].clone();
            }
        }
    }
    if crate::oauth::refresh::should_refresh(provider, &creds) {
        tracing::info!("refreshing {provider} credentials proactively");
        if let Some(raw) = crate::oauth::refresh::refresh_for_provider(provider, &creds).await {
            if !crate::oauth::refresh::is_unrecoverable(&raw) {
                let nc = crate::oauth::refresh::merge_refreshed(provider, &creds, &raw);
                if truthy(&nc["accessToken"]) || truthy(&nc["apiKey"]) || truthy(&nc["copilotToken"]) {
                    update_provider_credentials(db, &cid, &nc, &creds["providerSpecificData"]);
                    crate::oauth::refresh::merge_into(&mut creds, &nc);
                }
            }
        }
    }
    if provider == "github" {
        let tok = creds["providerSpecificData"]["copilotToken"].clone();
        let exp = creds["providerSpecificData"]["copilotTokenExpiresAt"].as_f64().map(|s| (s * 1000.0) as i64).unwrap_or(0);
        if !truthy(&tok) || exp - now_ms() < TOKEN_EXPIRY_BUFFER_MS {
            if let Some(r) = crate::oauth::refresh::copilot_token(creds["accessToken"].as_str().unwrap_or("")).await {
                let mut psd = if creds["providerSpecificData"].is_object() { creds["providerSpecificData"].clone() } else { json!({}) };
                psd["copilotToken"] = r["token"].clone();
                psd["copilotTokenExpiresAt"] = r["expiresAt"].clone();
                let _ = db.update_connection(&cid, &json!({"providerSpecificData": psd.clone()}));
                creds["providerSpecificData"] = psd;
                creds["copilotToken"] = r["token"].clone();
            }
        }
    }
    creds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_rules() {
        let f = check_fallback_error(429, "", 0, None);
        assert!(f.should_fallback);
        assert_eq!(f.cooldown_ms, 2000);
        assert_eq!(f.new_backoff_level, Some(1));
        assert_eq!(check_fallback_error(429, "", 3, None).cooldown_ms, 16000);
        assert!(!check_fallback_error(400, "bad", 0, None).should_fallback);
        assert!(check_fallback_error(400, "Rate limit hit", 0, None).should_fallback);
        assert_eq!(check_fallback_error(500, "", 0, None).cooldown_ms, TRANSIENT_COOLDOWN_MS);
        assert_eq!(check_fallback_error(401, "", 0, None).cooldown_ms, COOLDOWN_LONG);
    }

    #[tokio::test]
    async fn selection_and_locks() {
        let db = Db::open_in_memory().unwrap();
        let a = db.insert_connection(&json!({"provider": "openai", "apiKey": "a", "name": "A", "priority": 1})).unwrap();
        let b = db.insert_connection(&json!({"provider": "openai", "apiKey": "b", "name": "B", "priority": 2})).unwrap();
        let ex = HashSet::new();
        let Selection::Creds(c) = get_provider_credentials(&db, "openai", &ex, Some("gpt-4o"), None).await else { panic!() };
        assert_eq!(c["apiKey"], "a");
        mark_account_unavailable(&db, &a, 429, "rate limit", "openai", Some("gpt-4o"), None);
        let Selection::Creds(c) = get_provider_credentials(&db, "openai", &ex, Some("gpt-4o"), None).await else { panic!() };
        assert_eq!(c["apiKey"], "b");
        // other model is unaffected
        let Selection::Creds(c) = get_provider_credentials(&db, "openai", &ex, Some("gpt-4.1"), None).await else { panic!() };
        assert_eq!(c["apiKey"], "a");
        mark_account_unavailable(&db, &b, 429, "rate limit", "openai", Some("gpt-4o"), None);
        assert!(matches!(get_provider_credentials(&db, "openai", &ex, Some("gpt-4o"), None).await, Selection::AllRateLimited { .. }));
        let conn = db.get_connection(&a).unwrap();
        clear_account_error(&db, &a, &conn, Some("gpt-4o"));
        let conn = db.get_connection(&a).unwrap();
        assert!(conn.get("modelLock_gpt-4o").is_none());
        assert_eq!(conn["testStatus"], "active");
    }
}
