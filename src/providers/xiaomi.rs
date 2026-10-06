//! Xiaomi MiMo executor (port of executors/xiaomi-mimo.js + shared/mimoAccount.js).
//! v2.6 models route through the account service (weekly quota) when a
//! desktop passToken is available; everything else uses the sk- cloud API.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use async_trait::async_trait;
use base64::Engine;
use regex::Regex;
use serde_json::{Value, json};
use sha1::Digest;

use crate::exec::{ExecArgs, ExecResult, Executor, Headers, base_execute, client_for, default_build_headers, default_build_url, default_transform_request};
use crate::jsv::{js_string, now_ms, truthy};

const ACCOUNT_MODELS: [&str; 3] = ["mimo-v2.6-pro", "mimo-v2.6-flash", "mimo-v2.6-pro-ultraspeed"];
const COOKIE_KEY: &str = "__mimoAccountCookie";
pub const API_UA: &str = "miNative PC/Normal Windows_NT/10.0.19045 SDKV/1.0.0 DEVT/PC DEVS/Windows APP/miaccount_desktop APPV/0.1.0";
const SSO_UA: &str = "MiClaw/1.0";
const ACCOUNT_HOST: &str = "account.xiaomi.com";
const COOKIE_TTL_MS: i64 = 30 * 60 * 1000;

static CACHE: LazyLock<Mutex<HashMap<String, (String, i64)>>> = LazyLock::new(Default::default);

fn bare(model: &str) -> &str {
    model.split_once('/').map(|(_, b)| b).unwrap_or(model)
}

pub fn is_account_route(model: &str, creds: &Value) -> bool {
    ACCOUNT_MODELS.contains(&bare(model)) && (truthy(&creds[COOKIE_KEY]) || truthy(&creds["providerSpecificData"]["mimoPassToken"]))
}

pub fn server_base(psd: &Value) -> &'static str {
    match psd["region"].as_str().unwrap_or("").to_lowercase().as_str() {
        "cn" => "https://mimo-server-cn.xiaomimimo.com",
        "ams" => "https://mimo-server-ams.xiaomimimo.com",
        "ru" => "https://mimo-server-ru.xiaomimimo.com",
        "in" => "https://mimo-server-in.xiaomimimo.com",
        _ => "https://mimo-server-sgp.xiaomimimo.com",
    }
}

fn sid_for(region: &str) -> &'static str {
    match region.to_lowercase().as_str() {
        "cn" => "mimopc",
        "ams" => "mimoams",
        "ru" => "mimoru",
        "in" => "mimoin",
        _ => "mimosgp",
    }
}

fn desktop_cookie_path() -> std::path::PathBuf {
    let home = dirs::home_dir().unwrap_or_default();
    match std::env::consts::OS {
        "windows" => home.join("AppData/Roaming/Xiaomi MiMo/Partitions/xiaomi-account/Network/Cookies"),
        "macos" => home.join("Library/Application Support/Xiaomi MiMo/Partitions/xiaomi-account/Network/Cookies"),
        _ => home.join(".config/Xiaomi MiMo/Partitions/xiaomi-account/Network/Cookies"),
    }
}

/// Reads passToken/userId/cUserId from MiMo Desktop's Chromium cookie DB.
pub fn read_desktop_cookies() -> Option<HashMap<String, String>> {
    let src = desktop_cookie_path();
    if !src.exists() {
        return None;
    }
    let tmp = std::env::temp_dir().join(format!("9r-mimo-cookies-{}-{}.db", std::process::id(), crate::jsv::rand_hex(4)));
    std::fs::copy(&src, &tmp).ok()?;
    let res = (|| {
        let db = rusqlite::Connection::open_with_flags(&tmp, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).ok()?;
        let mut st = db.prepare("SELECT name, value FROM cookies WHERE host_key = ?").ok()?;
        let rows = st.query_map([format!(".{ACCOUNT_HOST}")], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))).ok()?;
        let jar: HashMap<String, String> = rows.flatten().collect();
        jar.contains_key("passToken").then_some(jar)
    })();
    let _ = std::fs::remove_file(&tmp);
    res
}

fn cookie_header(jar: &[(String, String)]) -> String {
    jar.iter().filter(|(_, v)| !v.is_empty()).map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("; ")
}

fn jar_set(jar: &mut Vec<(String, String)>, k: &str, v: &str) {
    if let Some(e) = jar.iter_mut().find(|(a, _)| a == k) {
        e.1 = v.to_string();
    } else {
        jar.push((k.to_string(), v.to_string()));
    }
}

fn absorb(jar: &mut Vec<(String, String)>, h: &reqwest::header::HeaderMap) {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^([^=]+)=([^;]*)").unwrap());
    for c in h.get_all(reqwest::header::SET_COOKIE) {
        if let Some(m) = c.to_str().ok().and_then(|s| RE.captures(s.trim())) {
            if !m[2].is_empty() {
                jar_set(jar, &m[1], &m[2]);
            }
        }
    }
}

fn client_sign(nonce: &str, ssecurity: &str) -> String {
    let input = if ssecurity.trim().is_empty() { format!("nonce={nonce}") } else { format!("nonce={nonce}&{ssecurity}") };
    let d = sha1::Sha1::digest(input.as_bytes());
    let b = base64::engine::general_purpose::STANDARD.encode(d);
    url_encode_component(&b)
}

pub fn url_encode_component(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.!~*'()".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn proxy_client(psd: &Value, region: &str, redirect: bool) -> reqwest::Client {
    // CN is always direct.
    let proxy = if region == "cn" { None } else { psd["connectionProxyEnabled"].as_bool().filter(|b| *b).and(psd["connectionProxyUrl"].as_str()) };
    let mut b = crate::exec::client_builder().connect_timeout(std::time::Duration::from_secs(20));
    if !redirect {
        b = b.redirect(reqwest::redirect::Policy::none());
    }
    if let Some(p) = proxy.filter(|p| !p.is_empty()) {
        if let Ok(px) = reqwest::Proxy::all(p) {
            b = b.proxy(px);
        }
    }
    b.build().unwrap_or_default()
}

async fn acquire(pass: &[(String, String)], psd: &Value) -> Option<String> {
    let region = psd["region"].as_str().unwrap_or("").to_lowercase();
    let sid = sid_for(&region);
    let mut jar: Vec<(String, String)> = pass.to_vec();
    let p1jar: Vec<(String, String)> = ["userId", "passToken", "cUserId"].iter().filter_map(|k| pass.iter().find(|(a, _)| a == k).cloned()).collect();
    let url = format!("https://{ACCOUNT_HOST}/pass/serviceLogin?_locale=zh_CN&_snsNone=true&sid={}&_json=true", url_encode_component(sid));
    let r = proxy_client(psd, &region, true).get(&url).header("Cookie", cookie_header(&p1jar)).header("User-Agent", SSO_UA).header("Accept", "application/json").send().await.ok()?;
    let headers = r.headers().clone();
    let raw = r.text().await.ok()?;
    let clean = raw.strip_prefix("&&&START&&&").unwrap_or(&raw);
    static NONCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""nonce"\s*:\s*(\d+)"#).unwrap());
    let raw_nonce = NONCE.captures(clean).map(|c| c[1].to_string());
    let j: Value = serde_json::from_str(clean).ok()?;
    let nonce = raw_nonce.or_else(|| (!j["nonce"].is_null()).then(|| js_string(&j["nonce"])))?;
    if j["code"] != json!(0) || !truthy(&j["location"]) || !truthy(&j["ssecurity"]) {
        tracing::info!("[mimoAccount] desktopPhase fail: phase1 sid={sid} code={}", j["code"]);
        return None;
    }
    absorb(&mut jar, &headers);
    let loc = js_string(&j["location"]);
    let sep = if loc.contains('?') { "&" } else { "?" };
    let mut current = format!("{loc}{sep}clientSign={}", client_sign(&nonce, &js_string(&j["ssecurity"])));
    let c2 = proxy_client(psd, &region, false);
    for _ in 0..8 {
        let Ok(res) = c2.get(&current).header("User-Agent", SSO_UA).send().await else { break };
        absorb(&mut jar, res.headers());
        let st = res.status().as_u16();
        let next = res.headers().get("location").and_then(|v| v.to_str().ok()).map(str::to_owned);
        match next {
            Some(l) if (300..400).contains(&st) => {
                current = reqwest::Url::parse(&current).and_then(|b| b.join(&l)).map(|u| u.to_string()).unwrap_or(l);
            }
            _ => break,
        }
    }
    let sid_key = format!("{sid}_serviceToken");
    if !jar.iter().any(|(k, v)| k == "serviceToken" && !v.is_empty()) {
        if let Some(v) = jar.iter().find(|(k, _)| *k == sid_key).map(|(_, v)| v.clone()) {
            jar_set(&mut jar, "serviceToken", &v);
        }
    }
    if !jar.iter().any(|(k, v)| k == "serviceToken" && !v.is_empty()) {
        tracing::info!("[mimoAccount] phase2 no serviceToken sid={sid}");
        return None;
    }
    static KEEP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_(ph|slh)$").unwrap());
    let out: Vec<(String, String)> = jar.into_iter().filter(|(k, v)| !v.is_empty() && (k == "serviceToken" || k == "userId" || KEEP.is_match(k))).collect();
    Some(cookie_header(&out))
}

/// getMimoAccountCookie: cached service cookie for the connection's passToken.
pub async fn account_cookie(psd: &Value) -> Option<String> {
    let base = server_base(psd);
    let pass: Vec<(String, String)> = if let Some(p) = psd["mimoPassToken"].as_str().filter(|s| !s.is_empty()) {
        let mut v = vec![("passToken".to_string(), p.to_string())];
        if let Some(u) = psd["mimoUserId"].as_str() {
            v.insert(0, ("userId".into(), u.into()));
        }
        if let Some(c) = psd["mimoCUserId"].as_str() {
            v.push(("cUserId".into(), c.into()));
        }
        v
    } else {
        let jar = read_desktop_cookies()?;
        let mut v: Vec<(String, String)> = jar.into_iter().collect();
        v.sort_by_key(|(k, _)| match k.as_str() {
            "userId" => 0,
            "passToken" => 1,
            "cUserId" => 2,
            _ => 3,
        });
        v
    };
    let pt = pass.iter().find(|(k, _)| k == "passToken").map(|(_, v)| v.clone())?;
    let key = crate::jsv::sha256_hex(&format!("{base}|{pt}"));
    if let Some((c, at)) = CACHE.lock().unwrap().get(&key).cloned() {
        if now_ms() - at < COOKIE_TTL_MS {
            return Some(c);
        }
    }
    let c = acquire(&pass, psd).await?;
    CACHE.lock().unwrap().insert(key, (c.clone(), now_ms()));
    Some(c)
}

pub fn invalidate_cookie_cache() {
    CACHE.lock().unwrap().clear();
}

/// getMimoAccountUsage → weekly quota.
pub async fn account_usage(creds: &Value) -> Value {
    let psd = &creds["providerSpecificData"];
    let Some(cookie) = account_cookie(psd).await else {
        return json!({"error": if truthy(&psd["mimoPassToken"]) || read_desktop_cookies().is_some() { "session-failed" } else { "no-session" }});
    };
    let r = client_for(creds).get(format!("{}/api/user/usage", server_base(psd))).header("User-Agent", API_UA).header("Cookie", cookie).header("Accept", "application/json").timeout(std::time::Duration::from_secs(10)).send().await;
    match r {
        Ok(r) if r.status().is_success() => {
            let d: Value = r.json().await.unwrap_or(Value::Null);
            if d["code"] != json!(0) || !truthy(&d["data"]) {
                return json!({"error": "bad-response"});
            }
            json!({"percent": d["data"]["percent"], "resetDate": d["data"]["resetDate"], "resetAt": d["data"]["resetAt"]})
        }
        Ok(r) => json!({"error": format!("http-{}", r.status().as_u16())}),
        Err(e) => json!({"error": e.to_string()}),
    }
}

pub struct XiaomiMimo;

#[async_trait]
impl Executor for XiaomiMimo {
    fn provider(&self) -> &str {
        "xiaomi-mimo"
    }
    fn build_url(&self, model: &str, stream: bool, idx: usize, creds: &Value) -> Result<String, String> {
        if is_account_route(model, creds) {
            return Ok(format!("{}/api/route/chat/completions", server_base(&creds["providerSpecificData"])));
        }
        default_build_url("xiaomi-mimo", &self.config(), model, stream, idx, creds)
    }
    fn build_headers(&self, creds: &Value, stream: bool, url: &str, model: &str, body: &Value) -> Headers {
        if is_account_route(model, creds) {
            if let Some(c) = creds[COOKIE_KEY].as_str() {
                let mut h = Headers::default();
                h.set("Content-Type", "application/json");
                h.set("Accept", if stream { "text/event-stream" } else { "application/json" });
                h.set("User-Agent", API_UA);
                h.set("Cookie", c);
                return h;
            }
        }
        default_build_headers("xiaomi-mimo", &self.config(), creds, stream, url, model, body)
    }
    fn transform_request(&self, model: &str, body: Value, _s: bool, creds: &Value) -> Value {
        let orig = body.clone();
        let mut out = default_transform_request("xiaomi-mimo", model, body);
        if is_account_route(model, creds) {
            let raw = [&out["reasoning_effort"], &orig["reasoning_effort"], &orig["output_config"]["effort"]].into_iter().find(|v| truthy(v)).cloned();
            if let Some(e) = raw {
                crate::jsv::del(&mut out, "reasoning_effort");
                let l = js_string(&e).to_lowercase();
                let norm = if l == "xhigh" { "high".to_string() } else { l };
                if !out["output_config"].is_object() {
                    out["output_config"] = json!({});
                }
                out["output_config"]["effort"] = json!(norm);
            }
            if out["temperature"].is_null() {
                out["temperature"] = json!(1.0);
            }
            if out["top_p"].is_null() {
                out["top_p"] = json!(0.95);
            }
        }
        out
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        if !is_account_route(args.model, args.creds) {
            return base_execute(self, args).await;
        }
        let psd = args.creds["providerSpecificData"].clone();
        let Some(cookie) = account_cookie(&psd).await else { return base_execute(self, args).await };
        args.creds[COOKIE_KEY] = json!(cookie);
        let again = ExecArgs { model: args.model, body: args.body.clone(), stream: args.stream, creds: &mut *args.creds, session_id: args.session_id.clone(), client_tool: args.client_tool.clone(), override_headers: args.override_headers.clone() };
        let res = base_execute(self, again).await?;
        if res.response.status == 401 {
            invalidate_cookie_cache();
            if let Some(fresh) = account_cookie(&psd).await {
                args.creds[COOKIE_KEY] = json!(fresh);
                return base_execute(self, args).await;
            }
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing() {
        assert!(!is_account_route("mimo-v2.6-pro", &json!({})));
        assert!(is_account_route("xm/mimo-v2.6-pro", &json!({"providerSpecificData": {"mimoPassToken": "p"}})));
        assert_eq!(server_base(&json!({"region": "CN"})), "https://mimo-server-cn.xiaomimimo.com");
        assert_eq!(url_encode_component("a+b/="), "a%2Bb%2F%3D");
        let t = XiaomiMimo.transform_request("mimo-v2.6-pro", json!({"messages": [], "reasoning_effort": "xhigh"}), true, &json!({"__mimoAccountCookie": "c"}));
        assert_eq!(t["output_config"]["effort"], "high");
        assert_eq!(t["top_p"], 0.95);
    }
}
