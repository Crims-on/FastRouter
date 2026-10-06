//! Interactive login flows (port of src/lib/oauth/providers/* and the
//! /api/oauth/* routes): authorization code (+PKCE), device code, browser
//! token, and the import paths (Cursor, Zed, Codex token, Kiro, iFlow
//! cookie, GitLab PAT, Xiaomi MiMo key).

use std::time::Duration;

use base64::Engine;
use rsa::rand_core::{OsRng, RngCore};
use serde_json::{Value, json};
use sha2::Digest;

use super::{decode_jwt_payload, email_from_jwt, enc, form};
use crate::jsv::{iso_from_ms, js_string, now_ms, truthy};
use crate::registry::REG;

const B64URL: base64::engine::GeneralPurpose = base64::engine::general_purpose::URL_SAFE_NO_PAD;
const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Flow {
    /// Browser redirect to our /callback with `code`.
    AuthCode { pkce: bool },
    /// User code + polling.
    Device,
    /// Browser redirect with a ready-to-use token (Kimchi).
    BrowserToken,
    /// Paste credentials from the IDE.
    Import,
}

pub fn flow(provider: &str) -> Option<Flow> {
    Some(match provider {
        "claude" | "codex" | "xai" | "gitlab" => Flow::AuthCode { pkce: true },
        "gemini-cli" | "antigravity" | "iflow" | "cline" | "clinepass" | "zed" | "xiaomi-mimo" => Flow::AuthCode { pkce: false },
        "github" | "kiro" | "kimi" | "kimi-coding" | "kilocode" | "codebuddy-cn" | "codebuddy-intl" | "qoder" | "qoder-cn" | "grok-cli" | "muse" | "glm" => Flow::Device,
        "kimchi" => Flow::BrowserToken,
        "cursor" => Flow::Import,
        _ => return None,
    })
}

/// Providers whose login must land on a fixed loopback port (their OAuth
/// client only whitelists that redirect URI).
pub fn fixed_redirect(provider: &str) -> Option<(u16, &'static str)> {
    match provider {
        "codex" => Some((1455, "/auth/callback")),
        "xai" => Some((56121, "/callback")),
        "zed" => Some((58443, "/")),
        _ => None,
    }
}

fn cfg(provider: &str) -> Value {
    let p = if provider == "kimi-coding" { "kimi" } else { provider };
    let mut c = REG.oauth(p).clone();
    if !c.is_object() {
        c = json!({});
    }
    match p {
        "gemini-cli" => {
            c["clientId"] = json!(crate::consts::C["GOOGLE_OAUTH_CLIENT"]["clientId"]);
            c["clientSecret"] = json!(crate::consts::C["GOOGLE_OAUTH_CLIENT"]["clientSecret"]);
        }
        "antigravity" => {
            c["clientId"] = json!(crate::consts::C["ANTIGRAVITY_OAUTH_CLIENT"]["clientId"]);
            c["clientSecret"] = json!(crate::consts::C["ANTIGRAVITY_OAUTH_CLIENT"]["clientSecret"]);
        }
        "xai" => {
            c["clientId"] = REG.transport("xai")["clientId"].clone();
            c["authorizeUrl"] = json!("https://auth.x.ai/oauth2/authorize");
            c["tokenUrl"] = json!("https://auth.x.ai/oauth2/token");
            c["scope"] = json!("openid profile email offline_access grok-cli:access api:access");
        }
        "kimi" => {
            if let Ok(v) = std::env::var("KIMI_CODING_OAUTH_CLIENT_ID").or_else(|_| std::env::var("KIMI_OAUTH_CLIENT_ID")) {
                c["clientId"] = json!(v);
            } else if !truthy(&c["clientId"]) {
                c["clientId"] = REG.transport("kimi")["clientId"].clone();
            }
        }
        _ => {}
    }
    c
}

pub fn random_b64url(n: usize) -> String {
    let mut b = vec![0u8; n];
    OsRng.fill_bytes(&mut b);
    B64URL.encode(b)
}

pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
    pub state: String,
}

pub fn pkce(bytes: usize) -> Pkce {
    let verifier = random_b64url(bytes);
    let challenge = B64URL.encode(sha2::Sha256::digest(verifier.as_bytes()));
    Pkce { verifier, challenge, state: random_b64url(32) }
}

fn http() -> reqwest::Client {
    crate::exec::http_client(None)
}

async fn json_or_text(r: reqwest::Response) -> (u16, Value, String) {
    let st = r.status().as_u16();
    let t = r.text().await.unwrap_or_default();
    (st, serde_json::from_str(&t).unwrap_or(Value::Null), t)
}

fn ok(st: u16) -> bool {
    (200..300).contains(&st)
}

// ---------------------------------------------------------------------------
// Authorization code
// ---------------------------------------------------------------------------

/// Everything needed to finish a browser login later.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct AuthStart {
    pub auth_url: String,
    pub state: String,
    pub code_verifier: String,
    pub redirect_uri: String,
    pub meta: Value,
}

fn qs(pairs: &[(&str, &str)]) -> String {
    pairs.iter().map(|(k, v)| format!("{k}={}", enc(v))).collect::<Vec<_>>().join("&")
}

/// generateAuthData(provider, redirectUri, meta)
pub async fn start_auth(provider: &str, redirect_uri: &str, meta: Value) -> Result<AuthStart, String> {
    let c = cfg(provider);
    let p = pkce(if provider == "xai" { 96 } else { 32 });
    let mut meta = if meta.is_object() { meta } else { json!({}) };
    let (url, state, verifier) = match provider {
        "claude" => (
            format!("{}?{}", c["authorizeUrl"].as_str().unwrap_or(""), qs(&[("code", "true"), ("client_id", c["clientId"].as_str().unwrap_or("")), ("response_type", "code"), ("redirect_uri", redirect_uri), ("scope", &scopes(&c)), ("code_challenge", &p.challenge), ("code_challenge_method", "S256"), ("state", &p.state)])),
            p.state.clone(),
            p.verifier.clone(),
        ),
        "codex" => {
            let mut pairs = vec![("response_type", "code".to_string()), ("client_id", js_string(&c["clientId"])), ("redirect_uri", redirect_uri.to_string()), ("scope", js_string(&c["scope"])), ("code_challenge", p.challenge.clone()), ("code_challenge_method", "S256".into())];
            for (k, v) in c["extraParams"].as_object().into_iter().flatten() {
                pairs.push((k.as_str(), js_string(v)));
            }
            pairs.push(("state", p.state.clone()));
            let q: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}={}", urlencode(v))).collect();
            (format!("{}?{}", c["authorizeUrl"].as_str().unwrap_or(""), q.join("&")), p.state.clone(), p.verifier.clone())
        }
        "xai" => {
            let (au, tu) = xai_discovery().await;
            meta["tokenUrl"] = json!(tu);
            let nonce = crate::jsv::rand_hex(16);
            let q = [("response_type", "code"), ("client_id", c["clientId"].as_str().unwrap_or("")), ("redirect_uri", redirect_uri), ("scope", c["scope"].as_str().unwrap_or("")), ("code_challenge", &p.challenge), ("code_challenge_method", "S256"), ("state", &p.state), ("nonce", &nonce), ("plan", "generic"), ("referrer", "cli-proxy-api")]
                .iter()
                .map(|(k, v)| format!("{k}={}", urlencode(v)))
                .collect::<Vec<_>>()
                .join("&");
            (format!("{au}?{q}"), p.state.clone(), p.verifier.clone())
        }
        "gemini-cli" | "antigravity" => (
            format!("{}?{}", c["authorizeUrl"].as_str().unwrap_or(""), qs(&[("client_id", c["clientId"].as_str().unwrap_or("")), ("response_type", "code"), ("redirect_uri", redirect_uri), ("scope", &scopes(&c)), ("state", &p.state), ("access_type", "offline"), ("prompt", "consent")])),
            p.state.clone(),
            p.verifier.clone(),
        ),
        "iflow" => (
            format!("{}?{}", c["authorizeUrl"].as_str().unwrap_or(""), qs(&[("loginMethod", c["extraParams"]["loginMethod"].as_str().unwrap_or("phone")), ("type", c["extraParams"]["type"].as_str().unwrap_or("phone")), ("redirect", redirect_uri), ("state", &p.state), ("client_id", c["clientId"].as_str().unwrap_or(""))])),
            p.state.clone(),
            p.verifier.clone(),
        ),
        "cline" | "clinepass" => (format!("{}?{}", c["authorizeUrl"].as_str().unwrap_or(""), qs(&[("client_type", "extension"), ("callback_url", redirect_uri), ("redirect_uri", redirect_uri)])), p.state.clone(), p.verifier.clone()),
        "gitlab" => {
            let base = meta["baseUrl"].as_str().filter(|s| !s.is_empty()).unwrap_or(c["defaultBaseUrl"].as_str().unwrap_or("https://gitlab.com")).trim_end_matches('/').to_string();
            let cid = meta["clientId"].as_str().unwrap_or("").to_string();
            if cid.is_empty() {
                return Err("GitLab OAuth needs an application Client ID (or use a Personal Access Token)".into());
            }
            (
                format!("{base}{}?{}", c["authorizeUrlPath"].as_str().unwrap_or("/oauth/authorize"), qs(&[("client_id", &cid), ("redirect_uri", redirect_uri), ("response_type", "code"), ("state", &p.state), ("scope", c["scope"].as_str().unwrap_or("api read_user")), ("code_challenge", &p.challenge), ("code_challenge_method", "S256")])),
                p.state.clone(),
                p.verifier.clone(),
            )
        }
        "kimchi" => {
            let base = c["webAppUrl"].as_str().unwrap_or("https://app.kimchi.dev").trim_end_matches('/').to_string();
            (format!("{base}/cli-auth?{}", qs(&[("callback", redirect_uri), ("state", &p.state)])), p.state.clone(), p.verifier.clone())
        }
        "zed" => {
            let port = reqwest::Url::parse(redirect_uri).ok().and_then(|u| u.port()).unwrap_or(58443);
            let z = zed_native_auth(port, meta["systemId"].as_str())?;
            meta["systemId"] = json!(z.system_id);
            (z.auth_url, p.state.clone(), z.private_key_pem)
        }
        "xiaomi-mimo" => {
            let (pk, sk) = x25519_keypair();
            let key_name = format!("fastrouter-xmd-{}", &crate::jsv::sha256_hex(&format!("{}-{}", std::env::consts::OS, crate::exec::hostname()))[..8]);
            let url = format!("https://platform.xiaomimimo.com/authorize?{}", qs(&[("pk", &pk), ("redirect_uri", redirect_uri), ("kn", "mimocode"), ("key_name", &key_name)]));
            (url, p.state.clone(), sk)
        }
        other => return Err(format!("Provider {other} does not use a browser login")),
    };
    Ok(AuthStart { auth_url: url, state, code_verifier: verifier, redirect_uri: redirect_uri.to_string(), meta })
}

fn urlencode(s: &str) -> String {
    // encodeURIComponent
    let mut o = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'(' | b')' => o.push(b as char),
            _ => o.push_str(&format!("%{b:02X}")),
        }
    }
    o
}

fn scopes(c: &Value) -> String {
    match &c["scopes"] {
        Value::Array(a) => a.iter().filter_map(|s| s.as_str()).collect::<Vec<_>>().join(" "),
        Value::String(s) => s.clone(),
        _ => js_string(&c["scope"]),
    }
}

async fn xai_discovery() -> (String, String) {
    static CACHE: std::sync::OnceLock<(String, String)> = std::sync::OnceLock::new();
    if let Some(c) = CACHE.get() {
        return c.clone();
    }
    let mut out = ("https://auth.x.ai/oauth2/authorize".to_string(), "https://auth.x.ai/oauth2/token".to_string());
    if let Ok(r) = http().get("https://auth.x.ai/.well-known/openid-configuration").header("accept", "application/json").timeout(Duration::from_secs(10)).send().await {
        if let Ok(d) = r.json::<Value>().await {
            let valid = |v: &Value| v.as_str().and_then(|s| reqwest::Url::parse(s).ok()).filter(|u| u.scheme() == "https" && u.host_str().map(|h| h == "x.ai" || h.ends_with(".x.ai")).unwrap_or(false)).map(|u| u.to_string());
            if let (Some(a), Some(t)) = (valid(&d["authorization_endpoint"]), valid(&d["token_endpoint"])) {
                out = (a, t);
            }
        }
    }
    let _ = CACHE.set(out.clone());
    out
}

/// Extracts `code` (and `state`) from a pasted callback URL, query or bare code.
pub fn parse_callback_input(input: &str) -> (String, Option<String>) {
    let t = input.trim();
    let query = if let Some(i) = t.find('?') { &t[i + 1..] } else if t.contains("code=") { t } else { "" };
    if !query.is_empty() {
        let pairs: Vec<(String, String)> = url_pairs(query);
        let get = |k: &str| pairs.iter().find(|(a, _)| a == k).map(|(_, b)| b.clone());
        if let Some(c) = get("code") {
            return (c, get("state"));
        }
    }
    (t.to_string(), None)
}

pub fn url_pairs(q: &str) -> Vec<(String, String)> {
    q.trim_start_matches('?')
        .split('&')
        .filter(|s| !s.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (pct_decode(k), pct_decode(v))
        })
        .collect()
}

pub fn pct_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'+' {
            out.push(b' ');
            i += 1;
        } else if b[i] == b'%' && i + 2 < b.len() + 0 + 1 && i + 2 <= b.len() - 1 {
            match std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                Some(v) => {
                    out.push(v);
                    i += 3;
                }
                None => {
                    out.push(b'%');
                    i += 1;
                }
            }
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// exchangeTokens(provider, code, redirectUri, verifier, state, meta) → mapped tokens.
pub async fn exchange(provider: &str, code: &str, a: &AuthStart) -> Result<Value, String> {
    let c = cfg(provider);
    let client = http();
    let redirect = a.redirect_uri.as_str();
    let verifier = a.code_verifier.as_str();
    match provider {
        "claude" => {
            let (code, st) = match code.split_once('#') {
                Some((c, s)) => (c.to_string(), s.to_string()),
                None => (code.to_string(), String::new()),
            };
            let r = client
                .post(c["tokenUrl"].as_str().unwrap_or(""))
                .header("accept", "application/json")
                .json(&json!({"code": code, "state": if st.is_empty() { a.state.clone() } else { st }, "grant_type": "authorization_code", "client_id": c["clientId"], "redirect_uri": redirect, "code_verifier": verifier}))
                .send()
                .await
                .map_err(|e| e.to_string())?;
            let (s, v, t) = json_or_text(r).await;
            if !ok(s) {
                return Err(format!("Token exchange failed: {t}"));
            }
            Ok(json!({"accessToken": v["access_token"], "refreshToken": v["refresh_token"], "expiresIn": v["expires_in"], "scope": v["scope"]}))
        }
        "codex" => {
            let v = post_form(&client, c["tokenUrl"].as_str().unwrap_or(""), &[("grant_type", "authorization_code"), ("client_id", c["clientId"].as_str().unwrap_or("")), ("code", code), ("redirect_uri", redirect), ("code_verifier", verifier)], "Token exchange failed").await?;
            Ok(codex_tokens(&v))
        }
        "xai" => {
            let tu = a.meta["tokenUrl"].as_str().map(str::to_owned).unwrap_or_else(|| "https://auth.x.ai/oauth2/token".into());
            let v = post_form(&client, &tu, &[("grant_type", "authorization_code"), ("client_id", c["clientId"].as_str().unwrap_or("")), ("code", code), ("redirect_uri", redirect), ("code_verifier", verifier)], "xAI token exchange failed").await?;
            let mut m = json!({"accessToken": v["access_token"], "refreshToken": v["refresh_token"], "expiresIn": v["expires_in"], "scope": v["scope"]});
            if let Some(id) = v["id_token"].as_str() {
                if let Some(e) = email_from_jwt(id) {
                    m["email"] = json!(e);
                }
                m["providerSpecificData"] = json!({"idToken": id});
            }
            Ok(m)
        }
        "gemini-cli" | "antigravity" => {
            let v = post_form(&client, c["tokenUrl"].as_str().unwrap_or(""), &[("grant_type", "authorization_code"), ("client_id", c["clientId"].as_str().unwrap_or("")), ("client_secret", c["clientSecret"].as_str().unwrap_or("")), ("code", code), ("redirect_uri", redirect)], "Token exchange failed").await?;
            let at = v["access_token"].as_str().unwrap_or("").to_string();
            let mut ui = client.get(format!("{}?alt=json", c["userInfoUrl"].as_str().unwrap_or(""))).bearer_auth(&at);
            if provider == "antigravity" {
                ui = ui.header("x-request-source", "local");
            }
            let user: Value = match ui.send().await {
                Ok(r) if r.status().is_success() => r.json().await.unwrap_or(json!({})),
                _ => json!({}),
            };
            let project = google_project(provider, &c, &at).await;
            Ok(json!({"accessToken": at, "refreshToken": v["refresh_token"], "expiresIn": v["expires_in"], "scope": v["scope"], "email": user["email"], "projectId": project}))
        }
        "iflow" => {
            let basic = B64.encode(format!("{}:{}", js_string(&c["clientId"]), js_string(&c["clientSecret"])));
            let r = client
                .post(c["tokenUrl"].as_str().unwrap_or(""))
                .header("content-type", "application/x-www-form-urlencoded")
                .header("accept", "application/json")
                .header("authorization", format!("Basic {basic}"))
                .body(form(&[("grant_type", "authorization_code"), ("code", code), ("redirect_uri", redirect), ("client_id", c["clientId"].as_str().unwrap_or("")), ("client_secret", c["clientSecret"].as_str().unwrap_or(""))]))
                .send()
                .await
                .map_err(|e| e.to_string())?;
            let (s, v, t) = json_or_text(r).await;
            if !ok(s) {
                return Err(format!("Token exchange failed: {t}"));
            }
            let at = v["access_token"].as_str().unwrap_or("");
            let r = client.get(format!("{}?accessToken={}", c["userInfoUrl"].as_str().unwrap_or(""), urlencode(at))).header("accept", "application/json").send().await.map_err(|e| e.to_string())?;
            let (s, u, t) = json_or_text(r).await;
            if !ok(s) {
                return Err(format!("Failed to fetch user info: {t}"));
            }
            if u["success"] != json!(true) {
                return Err(format!("User info request failed: {}", u["message"].as_str().unwrap_or("Unknown error")));
            }
            let d = &u["data"];
            let key = d["apiKey"].as_str().map(str::trim).unwrap_or("");
            if key.is_empty() {
                return Err("Empty API key returned from iFlow".into());
            }
            let email = d["email"].as_str().filter(|s| !s.trim().is_empty()).or_else(|| d["phone"].as_str().filter(|s| !s.trim().is_empty())).ok_or("Missing account email/phone in user info")?;
            Ok(json!({"accessToken": at, "refreshToken": v["refresh_token"], "expiresIn": v["expires_in"], "apiKey": key, "email": email.trim(), "displayName": if truthy(&d["nickname"]) { d["nickname"].clone() } else { d["name"].clone() }}))
        }
        "cline" | "clinepass" => {
            let mut t = None;
            let mut b = code.to_string();
            let pad = (4 - b.len() % 4) % 4;
            b.push_str(&"=".repeat(pad));
            if let Ok(bytes) = B64.decode(b.replace('-', "+").replace('_', "/")) {
                let s = String::from_utf8_lossy(&bytes).to_string();
                if let Some(i) = s.rfind('}') {
                    if let Ok(d) = serde_json::from_str::<Value>(&s[..=i]) {
                        t = Some(json!({"access_token": d["accessToken"], "refresh_token": d["refreshToken"], "email": d["email"], "firstName": d["firstName"], "lastName": d["lastName"], "expires_at": d["expiresAt"]}));
                    }
                }
            }
            let t = match t {
                Some(t) => t,
                None => {
                    let url = c["tokenExchangeUrl"].as_str().or_else(|| c["tokenUrl"].as_str()).unwrap_or("");
                    let r = client.post(url).header("accept", "application/json").json(&json!({"grant_type": "authorization_code", "code": code, "client_type": "extension", "redirect_uri": redirect})).send().await.map_err(|e| e.to_string())?;
                    let (s, d, txt) = json_or_text(r).await;
                    if !ok(s) {
                        return Err(format!("Cline token exchange failed: {txt}"));
                    }
                    let pick = |k: &str| if truthy(&d["data"][k]) { d["data"][k].clone() } else { d[k].clone() };
                    json!({"access_token": pick("accessToken"), "refresh_token": pick("refreshToken"), "email": d["data"]["userInfo"]["email"].as_str().unwrap_or(""), "expires_at": pick("expiresAt")})
                }
            };
            let exp = t["expires_at"].as_str().and_then(crate::jsv::parse_iso_ms).map(|ms| (ms - now_ms()) / 1000).unwrap_or(3600);
            Ok(json!({"accessToken": t["access_token"], "refreshToken": t["refresh_token"], "expiresIn": exp, "email": t["email"], "providerSpecificData": {"firstName": t["firstName"], "lastName": t["lastName"]}}))
        }
        "gitlab" => {
            let base = a.meta["baseUrl"].as_str().filter(|s| !s.is_empty()).unwrap_or(c["defaultBaseUrl"].as_str().unwrap_or("https://gitlab.com")).trim_end_matches('/').to_string();
            let cid = a.meta["clientId"].as_str().unwrap_or("");
            let mut pairs = vec![("client_id", cid), ("grant_type", "authorization_code"), ("code", code), ("redirect_uri", redirect), ("code_verifier", verifier)];
            let secret = a.meta["clientSecret"].as_str().unwrap_or("");
            if !secret.is_empty() {
                pairs.push(("client_secret", secret));
            }
            let v = post_form(&client, &format!("{base}{}", c["tokenUrlPath"].as_str().unwrap_or("/oauth/token")), &pairs, "GitLab token exchange failed").await?;
            let user: Value = match client.get(format!("{base}{}", c["userInfoUrlPath"].as_str().unwrap_or("/api/v4/user"))).bearer_auth(v["access_token"].as_str().unwrap_or("")).send().await {
                Ok(r) if r.status().is_success() => r.json().await.unwrap_or(json!({})),
                _ => json!({}),
            };
            let email = user["email"].as_str().filter(|s| !s.is_empty()).or_else(|| user["public_email"].as_str()).unwrap_or("");
            Ok(json!({"accessToken": v["access_token"], "refreshToken": v["refresh_token"], "expiresIn": v["expires_in"], "scope": v["scope"], "email": email, "providerSpecificData": {"username": user["username"].as_str().unwrap_or(""), "email": email, "name": user["name"].as_str().unwrap_or(""), "baseUrl": base, "clientId": cid, "authKind": "oauth"}}))
        }
        "kimchi" => {
            let tok = code.trim();
            if tok.is_empty() {
                return Err("Missing Kimchi token".into());
            }
            let r = client.get(c["validationUrl"].as_str().unwrap_or("https://api.cast.ai/v1/llm/openai/supported-providers")).bearer_auth(tok).header("accept", "application/json").send().await.map_err(|e| e.to_string())?;
            if !r.status().is_success() {
                return Err(format!("Kimchi token validation failed: {}", r.status().as_u16()));
            }
            let user: Value = match c["userInfoUrl"].as_str() {
                Some(u) => match client.get(u).bearer_auth(tok).header("accept", "application/json").send().await {
                    Ok(r) if r.status().is_success() => r.json().await.unwrap_or(json!({})),
                    _ => json!({}),
                },
                None => json!({}),
            };
            let uid = if user["id"].is_null() { String::new() } else { js_string(&user["id"]) };
            let username = user["username"].as_str().unwrap_or("");
            let email = user["email"].as_str().map(str::to_owned).or_else(|| (!uid.is_empty()).then(|| format!("kimchi-user-{uid}")));
            Ok(json!({"accessToken": tok, "refreshToken": null, "email": email, "displayName": user["name"].as_str().or(Some(username).filter(|s| !s.is_empty())), "providerSpecificData": {"authMethod": "browser_token", "userId": uid, "username": username}}))
        }
        "zed" => {
            let (uid, enc_tok) = parse_zed_callback(code)?;
            let at = zed_decrypt(&enc_tok, verifier)?;
            let system_id = a.meta["systemId"].as_str().map(str::to_owned).unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            zed_tokens(&at, &uid, &system_id, "oauth").await
        }
        "xiaomi-mimo" => {
            let u = if code.contains("u=") { url_pairs(code.split_once('?').map(|x| x.1).unwrap_or(code)).into_iter().find(|(k, _)| k == "u").map(|(_, v)| v).unwrap_or_default() } else { code.to_string() };
            let r = xiaomi_decrypt(verifier, &u)?;
            let uid = r["uid"].as_str().unwrap_or("").to_string();
            Ok(json!({
                "accessToken": r["sk"], "refreshToken": null, "expiresIn": 365 * 24 * 3600,
                "email": if uid.is_empty() { Value::Null } else { json!(format!("{uid}@xiaomi")) },
                "displayName": if uid.is_empty() { json!("Xiaomi MiMo") } else { json!(format!("Xiaomi {uid}")) },
                "providerSpecificData": {"uid": if uid.is_empty() { Value::Null } else { json!(uid) }, "baseUrl": r["url"], "authMethod": "oauth"},
            }))
        }
        other => Err(format!("Provider {other} does not support code exchange")),
    }
}

async fn post_form(client: &reqwest::Client, url: &str, pairs: &[(&str, &str)], label: &str) -> Result<Value, String> {
    let r = client.post(url).header("content-type", "application/x-www-form-urlencoded").header("accept", "application/json").body(form(pairs)).send().await.map_err(|e| e.to_string())?;
    let (s, v, t) = json_or_text(r).await;
    if !ok(s) {
        return Err(format!("{label}: {t}"));
    }
    Ok(v)
}

pub fn codex_info(id_token: &str) -> Value {
    let p = decode_jwt_payload(id_token).unwrap_or(json!({}));
    let auth = &p["https://api.openai.com/auth"];
    json!({
        "email": p["email"],
        "chatgptAccountId": if truthy(&auth["chatgpt_account_id"]) { auth["chatgpt_account_id"].clone() } else { p["account_id"].clone() },
        "chatgptPlanType": if truthy(&auth["chatgpt_plan_type"]) { auth["chatgpt_plan_type"].clone() } else { p["plan_type"].clone() },
    })
}

fn codex_tokens(v: &Value) -> Value {
    let info = codex_info(v["id_token"].as_str().unwrap_or(""));
    let mut m = json!({"accessToken": v["access_token"], "refreshToken": v["refresh_token"], "idToken": v["id_token"], "expiresIn": v["expires_in"], "lastRefreshAt": iso_from_ms(now_ms())});
    let email = info["email"].as_str().map(str::to_owned).or_else(|| email_from_jwt(v["access_token"].as_str().unwrap_or("")));
    if let Some(e) = email {
        m["email"] = json!(e);
    }
    if truthy(&info["chatgptAccountId"]) || truthy(&info["chatgptPlanType"]) {
        m["providerSpecificData"] = json!({"chatgptAccountId": info["chatgptAccountId"], "chatgptPlanType": info["chatgptPlanType"]});
    }
    m
}

fn platform_enum() -> i64 {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => 2,
        ("macos", _) => 1,
        ("linux", "aarch64") => 4,
        ("linux", _) => 3,
        ("windows", _) => 5,
        _ => 0,
    }
}

async fn google_project(provider: &str, c: &Value, at: &str) -> String {
    let client = http();
    let meta = json!({"ideType": 9, "platform": platform_enum(), "pluginType": 2});
    let pick = |d: &Value| d["cloudaicompanionProject"]["id"].as_str().or_else(|| d["cloudaicompanionProject"].as_str()).unwrap_or("").to_string();
    if provider == "gemini-cli" {
        let r = client.post("https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist").bearer_auth(at).json(&json!({"metadata": meta, "mode": 1})).send().await;
        return match r {
            Ok(r) if r.status().is_success() => pick(&r.json().await.unwrap_or(json!({}))),
            _ => String::new(),
        };
    }
    let ua = c["loadCodeAssistUserAgent"].as_str().unwrap_or("antigravity").to_string();
    let r = client.post(c["loadCodeAssistEndpoint"].as_str().unwrap_or("https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist")).bearer_auth(at).header("user-agent", &ua).header("x-request-source", "local").json(&json!({"metadata": meta})).send().await;
    let d: Value = match r {
        Ok(r) if r.status().is_success() => r.json().await.unwrap_or(json!({})),
        _ => json!({}),
    };
    let project = pick(&d);
    let tier = d["allowedTiers"].as_array().and_then(|a| a.iter().find(|t| t["isDefault"] == json!(true)).and_then(|t| t["id"].as_str().map(|s| s.trim().to_string()))).unwrap_or_else(|| "legacy-tier".into());
    if !project.is_empty() {
        // Fire-and-forget onboarding, like 9router.
        let (at, ep) = (at.to_string(), c["onboardUserEndpoint"].as_str().unwrap_or("https://cloudcode-pa.googleapis.com/v1internal:onboardUser").to_string());
        tokio::spawn(async move {
            for _ in 0..10 {
                match http().post(&ep).bearer_auth(&at).header("user-agent", &ua).header("x-request-source", "local").json(&json!({"tierId": tier, "metadata": meta})).send().await {
                    Ok(r) if r.status().is_success() => {
                        if r.json::<Value>().await.map(|v| v["done"] == json!(true)).unwrap_or(false) {
                            break;
                        }
                    }
                    Ok(_) => {}
                    Err(_) => break,
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
    }
    project
}

// ---------------------------------------------------------------------------
// Zed (RSA native-app sign-in)
// ---------------------------------------------------------------------------

pub struct ZedAuth {
    pub auth_url: String,
    pub private_key_pem: String,
    pub system_id: String,
}

pub fn zed_native_auth(port: u16, system_id: Option<&str>) -> Result<ZedAuth, String> {
    use rsa::pkcs1::{EncodeRsaPrivateKey, EncodeRsaPublicKey};
    let key = rsa::RsaPrivateKey::new(&mut OsRng, 2048).map_err(|e| e.to_string())?;
    let pub_der = key.to_public_key().to_pkcs1_der().map_err(|e| e.to_string())?;
    let pem = key.to_pkcs1_pem(rsa::pkcs1::LineEnding::LF).map_err(|e| e.to_string())?.to_string();
    let sid = system_id.map(str::to_owned).unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    // b64urlPadded
    let pk = base64::engine::general_purpose::URL_SAFE.encode(pub_der.as_bytes());
    let url = format!("https://zed.dev/native_app_signin?native_app_port={port}&native_app_public_key={}&system_id={}", urlencode(&pk), urlencode(&sid));
    Ok(ZedAuth { auth_url: url, private_key_pem: pem, system_id: sid })
}

pub fn parse_zed_callback(input: &str) -> Result<(String, String), String> {
    let raw = input.trim();
    if raw.is_empty() {
        return Err("Missing Zed callback URL".into());
    }
    let data: Value = match serde_json::from_str::<Value>(raw) {
        Ok(v) if v.is_object() => v,
        _ => {
            let q = raw.split_once('?').map(|x| x.1).unwrap_or(raw);
            let mut o = serde_json::Map::new();
            for (k, v) in url_pairs(q) {
                o.insert(k, json!(v));
            }
            Value::Object(o)
        }
    };
    let uid = [&data["user_id"], &data["userId"]].into_iter().find(|v| truthy(v)).map(js_string);
    let tok = [&data["access_token"], &data["accessToken"], &data["token"]].into_iter().find(|v| truthy(v)).map(js_string);
    match (uid, tok) {
        (Some(u), Some(t)) => Ok((u, t)),
        _ => Err("Zed callback must include user_id and access_token".into()),
    }
}

pub fn zed_decrypt(enc_b64url: &str, pem: &str) -> Result<String, String> {
    use rsa::pkcs1::DecodeRsaPrivateKey;
    let key = rsa::RsaPrivateKey::from_pkcs1_pem(pem).map_err(|e| format!("Failed to decrypt Zed access token: {e}"))?;
    let data = B64URL.decode(enc_b64url.trim_end_matches('=')).or_else(|_| B64.decode(enc_b64url)).map_err(|e| format!("Failed to decrypt Zed access token: {e}"))?;
    if let Ok(p) = key.decrypt(rsa::Oaep::new::<sha2_010::Sha256>(), &data) {
        return String::from_utf8(p).map_err(|e| e.to_string());
    }
    match key.decrypt(rsa::Pkcs1v15Encrypt, &data) {
        Ok(p) => String::from_utf8(p).map_err(|_| "Failed to decrypt Zed access token: invalid UTF-8".into()),
        Err(e) => Err(format!("Failed to decrypt Zed access token: {e}")),
    }
}

async fn zed_tokens(at: &str, uid: &str, system_id: &str, method: &str) -> Result<Value, String> {
    let creds = json!({"accessToken": at, "providerSpecificData": {"userId": uid, "systemId": system_id}});
    let user = crate::providers::zed::fetch_user(&creds).await.ok();
    if method == "imported" && user.is_none() {
        return Err("Zed token validation failed".into());
    }
    let org = crate::providers::zed::resolve_org_id(&creds, user.as_ref());
    let u = user.unwrap_or(json!({}));
    let name = [&u["name"], &u["display_name"], &u["username"]].into_iter().find(|v| truthy(v)).cloned().unwrap_or(json!(format!("Zed {uid}")));
    Ok(json!({"accessToken": at, "refreshToken": null, "expiresIn": null, "email": u["email"], "displayName": name, "providerSpecificData": {"authMethod": method, "userId": uid, "systemId": system_id, "organizationId": org}}))
}

// ---------------------------------------------------------------------------
// Xiaomi MiMo desktop OAuth (X25519 + AES-256-GCM)
// ---------------------------------------------------------------------------

/// (base64 SPKI public key, base64 raw private key)
pub fn x25519_keypair() -> (String, String) {
    let sk = x25519_dalek::StaticSecret::random_from_rng(OsRng);
    let pk = x25519_dalek::PublicKey::from(&sk);
    let mut spki = hex::decode("302a300506032b656e032100").unwrap();
    spki.extend_from_slice(pk.as_bytes());
    (B64.encode(spki), B64.encode(sk.to_bytes()))
}

pub fn xiaomi_decrypt(sk_b64: &str, enc_b64: &str) -> Result<Value, String> {
    use aes_gcm::aead::{Aead, KeyInit};
    let raw = B64.decode(enc_b64.trim().replace(' ', "+")).map_err(|e| e.to_string())?;
    if raw.len() < 12 + 32 + 16 + 1 {
        return Err(format!("Encrypted payload too short: {} bytes", raw.len()));
    }
    let sk_bytes: [u8; 32] = B64.decode(sk_b64).map_err(|e| e.to_string())?.try_into().map_err(|_| "bad key".to_string())?;
    let sk = x25519_dalek::StaticSecret::from(sk_bytes);
    let eph: [u8; 32] = raw[12..44].try_into().unwrap();
    let shared = sk.diffie_hellman(&x25519_dalek::PublicKey::from(eph));
    let key = sha2::Sha256::digest(shared.as_bytes());
    let cipher = aes_gcm::Aes256Gcm::new_from_slice(&key).map_err(|e| e.to_string())?;
    let plain = cipher.decrypt(aes_gcm::Nonce::from_slice(&raw[..12]), &raw[44..]).map_err(|_| "Xiaomi callback decryption failed".to_string())?;
    let v: Value = serde_json::from_slice(&plain).map_err(|e| e.to_string())?;
    if !v.is_object() {
        return Err("Decrypted payload is not a valid object".into());
    }
    Ok(json!({"uid": v["uid"], "sk": v["sk"], "url": if truthy(&v["url"]) { v["url"].clone() } else { json!("https://api.xiaomimimo.com/v1") }}))
}

// ---------------------------------------------------------------------------
// Device code
// ---------------------------------------------------------------------------

/// Device-code start: what to show the user plus state for polling.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DeviceStart {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: String,
    pub interval: u64,
    pub expires_in: u64,
    pub code_verifier: String,
    pub extra: Value,
}

pub enum Poll {
    Pending,
    Done(Value),
    Error(String),
}

fn form_req(url: &str, pairs: &[(&str, &str)]) -> reqwest::RequestBuilder {
    http().post(url).header("content-type", "application/x-www-form-urlencoded").header("accept", "application/json").body(form(pairs))
}

/// requestDeviceCode(provider, options)
pub async fn device_start(provider: &str, options: &Value) -> Result<DeviceStart, String> {
    let c = cfg(provider);
    let s = |v: &Value| v.as_str().unwrap_or("").to_string();
    let std_resp = |d: &Value| DeviceStart {
        device_code: s(&d["device_code"]),
        user_code: s(&d["user_code"]),
        verification_uri: s(&d["verification_uri"]),
        verification_uri_complete: if truthy(&d["verification_uri_complete"]) { s(&d["verification_uri_complete"]) } else { s(&d["verification_uri"]) },
        interval: d["interval"].as_u64().unwrap_or(5),
        expires_in: d["expires_in"].as_u64().unwrap_or(900),
        ..Default::default()
    };
    match provider {
        "github" => {
            let r = form_req(c["deviceCodeUrl"].as_str().unwrap_or(""), &[("client_id", c["clientId"].as_str().unwrap_or("")), ("scope", c["scopes"].as_str().unwrap_or("read:user"))]).send().await.map_err(|e| e.to_string())?;
            let (st, d, t) = json_or_text(r).await;
            if !ok(st) {
                return Err(format!("Device code request failed: {t}"));
            }
            Ok(std_resp(&d))
        }
        "grok-cli" | "muse" => {
            let ua = crate::providers::grok::pager_user_agent();
            let mut pairs = vec![("client_id", c["clientId"].as_str().unwrap_or(""))];
            if provider == "grok-cli" {
                pairs.push(("scope", c["scope"].as_str().unwrap_or("")));
                if let Some(r) = c["referrer"].as_str() {
                    pairs.push(("referrer", r));
                }
            }
            let mut rb = form_req(c["deviceCodeUrl"].as_str().unwrap_or(""), &pairs);
            rb = if provider == "grok-cli" { rb.header("user-agent", ua) } else { rb.header("x-api-version", "1.0.0") };
            let r = rb.send().await.map_err(|e| e.to_string())?;
            let (st, d, t) = json_or_text(r).await;
            if !ok(st) {
                return Err(format!("{} device code request failed: {t}", if provider == "muse" { "Muse Code" } else { "Grok CLI" }));
            }
            Ok(std_resp(&d))
        }
        "kimi" | "kimi-coding" => {
            let dev = uuid::Uuid::new_v4().to_string();
            let mut rb = form_req(c["deviceCodeUrl"].as_str().unwrap_or(""), &[("client_id", c["clientId"].as_str().unwrap_or(""))]);
            for (k, v) in crate::exec::kimi_headers(Some(&dev)) {
                rb = rb.header(k, v);
            }
            let (st, d, t) = json_or_text(rb.send().await.map_err(|e| e.to_string())?).await;
            if !ok(st) {
                return Err(format!("Device code request failed: {t}"));
            }
            let au = c["authorizeDeviceUrl"].as_str().unwrap_or("https://www.kimi.com/code/authorize_device");
            let mut out = std_resp(&d);
            if out.verification_uri.is_empty() {
                out.verification_uri = au.into();
            }
            if !truthy(&d["verification_uri_complete"]) {
                out.verification_uri_complete = format!("{au}?user_code={}", out.user_code);
            }
            out.extra = json!({"_kimiDeviceId": dev});
            Ok(out)
        }
        "kilocode" => {
            let r = http().post(c["initiateUrl"].as_str().unwrap_or("")).header("content-type", "application/json").send().await.map_err(|e| e.to_string())?;
            let st = r.status().as_u16();
            if st == 429 {
                return Err("Too many pending authorization requests. Please try again later.".into());
            }
            let (_, d, t) = json_or_text(r).await;
            if !ok(st) {
                return Err(format!("Device auth initiation failed: {t}"));
            }
            Ok(DeviceStart { device_code: s(&d["code"]), user_code: s(&d["code"]), verification_uri: s(&d["verificationUrl"]), verification_uri_complete: s(&d["verificationUrl"]), interval: 3, expires_in: d["expiresIn"].as_u64().unwrap_or(300), ..Default::default() })
        }
        "codebuddy-cn" | "codebuddy-intl" => {
            let domain = if provider == "codebuddy-cn" { "copilot.tencent.com" } else { "www.codebuddy.ai" };
            let r = http()
                .post(format!("{}?platform={}", c["stateUrl"].as_str().unwrap_or(""), c["platform"].as_str().unwrap_or("CLI")))
                .header("content-type", "application/json")
                .header("accept", "application/json")
                .header("user-agent", c["userAgent"].as_str().unwrap_or(""))
                .header("x-requested-with", "XMLHttpRequest")
                .header("x-domain", domain)
                .header("x-no-authorization", "true")
                .header("x-no-user-id", "true")
                .header("x-product", "SaaS")
                .body("{}")
                .send()
                .await
                .map_err(|e| e.to_string())?;
            let (st, d, t) = json_or_text(r).await;
            if !ok(st) {
                return Err(format!("CodeBuddy state request failed: {t}"));
            }
            if d["code"] != json!(0) || !truthy(&d["data"]["state"]) || !truthy(&d["data"]["authUrl"]) {
                return Err(format!("CodeBuddy state error: {}", d["msg"].as_str().unwrap_or("missing state/authUrl")));
            }
            Ok(DeviceStart { device_code: s(&d["data"]["state"]), user_code: String::new(), verification_uri: s(&d["data"]["authUrl"]), verification_uri_complete: s(&d["data"]["authUrl"]), interval: c["pollInterval"].as_u64().unwrap_or(5000) / 1000, expires_in: 600, ..Default::default() })
        }
        "qoder" | "qoder-cn" => {
            let p = pkce(32);
            let nonce = uuid::Uuid::new_v4().to_string();
            let machine = uuid::Uuid::new_v4().to_string();
            let login = c["loginUrl"].as_str().unwrap_or("https://qoder.com/device/selectAccounts");
            let url = format!("{login}?{}", qs(&[("challenge", &p.challenge), ("challenge_method", "S256"), ("machine_id", &machine), ("nonce", &nonce)]));
            Ok(DeviceStart { device_code: nonce.clone(), user_code: nonce[..8].to_uppercase(), verification_uri: login.into(), verification_uri_complete: url, interval: 2, expires_in: 300, code_verifier: p.verifier, extra: json!({"_qoderNonce": nonce, "_qoderMachineId": machine}) })
        }
        "kiro" => {
            let region = options["region"].as_str().map(str::trim).filter(|s| !s.is_empty()).unwrap_or("us-east-1").to_string();
            if !regex::Regex::new(r"^[a-z]{2}-[a-z]+-\d{1,2}$").unwrap().is_match(&region) {
                return Err("Invalid region".into());
            }
            let start_url = options["startUrl"].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).unwrap_or_else(|| s(&c["startUrl"]));
            let method = if options["authMethod"] == "idc" { "idc" } else { "builder-id" };
            let r = http().post(format!("https://oidc.{region}.amazonaws.com/client/register")).header("accept", "application/json").json(&json!({"clientName": c["clientName"], "clientType": c["clientType"], "scopes": c["scopes"], "grantTypes": c["grantTypes"], "issuerUrl": c["issuerUrl"]})).send().await.map_err(|e| e.to_string())?;
            let (st, ci, t) = json_or_text(r).await;
            if !ok(st) {
                return Err(format!("Client registration failed: {t}"));
            }
            let r = http().post(format!("https://oidc.{region}.amazonaws.com/device_authorization")).header("accept", "application/json").json(&json!({"clientId": ci["clientId"], "clientSecret": ci["clientSecret"], "startUrl": start_url})).send().await.map_err(|e| e.to_string())?;
            let (st, d, t) = json_or_text(r).await;
            if !ok(st) {
                return Err(format!("Device authorization failed: {t}"));
            }
            Ok(DeviceStart {
                device_code: s(&d["deviceCode"]),
                user_code: s(&d["userCode"]),
                verification_uri: s(&d["verificationUri"]),
                verification_uri_complete: s(&d["verificationUriComplete"]),
                interval: d["interval"].as_u64().unwrap_or(5),
                expires_in: d["expiresIn"].as_u64().unwrap_or(600),
                code_verifier: String::new(),
                extra: json!({"_clientId": ci["clientId"], "_clientSecret": ci["clientSecret"], "_region": region, "_authMethod": method, "_startUrl": start_url}),
            })
        }
        "glm" => {
            let poll = crate::jsv::rand_hex(32);
            let r = http().post(c["cliInitUrl"].as_str().unwrap_or("")).bearer_auth(&poll).json(&json!({"provider": c["providerId"].as_str().unwrap_or("zai")})).send().await.map_err(|e| e.to_string())?;
            let (st, p, t) = json_or_text(r).await;
            if !ok(st) {
                return Err(format!("ZCode OAuth init failed: {t}"));
            }
            if !glm_ok(&p["code"]) || !p["data"].is_object() {
                return Err(p["msg"].as_str().unwrap_or("ZCode OAuth init returned no data").into());
            }
            let d = &p["data"];
            if !truthy(&d["flow_id"]) || !truthy(&d["authorize_url"]) {
                return Err("ZCode OAuth init response missing flow_id/authorize_url".into());
            }
            let exp = d["expires_at"].as_f64().map(|raw| {
                let ms = if raw > 1e12 { raw } else { raw * 1000.0 };
                ((ms as i64 - now_ms()) / 1000).max(0) as u64
            });
            Ok(DeviceStart { device_code: s(&d["flow_id"]), user_code: String::new(), verification_uri: s(&d["authorize_url"]), verification_uri_complete: s(&d["authorize_url"]), interval: d["poll_interval_sec"].as_u64().unwrap_or(3), expires_in: exp.filter(|e| *e > 0).unwrap_or(300), extra: json!({"_zcodePollToken": poll}), ..Default::default() })
        }
        other => Err(format!("Provider {other} does not support device code flow")),
    }
}

fn glm_ok(c: &Value) -> bool {
    c.is_null() || *c == json!(0) || *c == json!(200) || *c == json!("0") || *c == json!("200")
}

fn pending_or(d: &Value, st: u16) -> Option<Poll> {
    match d["error"].as_str() {
        Some("authorization_pending" | "slow_down") => Some(Poll::Pending),
        Some(e) => Some(Poll::Error(d["error_description"].as_str().or_else(|| d["message"].as_str()).map(|m| format!("{e}: {m}")).unwrap_or_else(|| e.to_string()))),
        None if !ok(st) => Some(Poll::Error(format!("HTTP {st}"))),
        None => None,
    }
}

/// pollForToken(provider, deviceCode, verifier, extra) → mapped tokens.
pub async fn device_poll(provider: &str, d: &DeviceStart) -> Poll {
    match device_poll_inner(provider, d).await {
        Ok(p) => p,
        Err(e) => Poll::Error(e),
    }
}

async fn device_poll_inner(provider: &str, d: &DeviceStart) -> Result<Poll, String> {
    let c = cfg(provider);
    let dc = d.device_code.as_str();
    let grant = "urn:ietf:params:oauth:grant-type:device_code";
    match provider {
        "github" => {
            let r = form_req(c["tokenUrl"].as_str().unwrap_or(""), &[("client_id", c["clientId"].as_str().unwrap_or("")), ("device_code", dc), ("grant_type", grant)]).send().await.map_err(|e| e.to_string())?;
            let (st, v, _) = json_or_text(r).await;
            if let Some(p) = pending_or(&v, st) {
                return Ok(p);
            }
            let Some(at) = v["access_token"].as_str() else { return Ok(Poll::Error("no_access_token".into())) };
            let h = |rb: reqwest::RequestBuilder| rb.bearer_auth(at).header("accept", "application/json").header("x-github-api-version", c["apiVersion"].as_str().unwrap_or("2022-11-28")).header("user-agent", c["userAgent"].as_str().unwrap_or("GitHubCopilotChat/0.26.7"));
            let cp: Value = match h(http().get(c["copilotTokenUrl"].as_str().unwrap_or(""))).send().await {
                Ok(r) if r.status().is_success() => r.json().await.unwrap_or(json!({})),
                _ => json!({}),
            };
            let u: Value = match h(http().get(c["userInfoUrl"].as_str().unwrap_or("https://api.github.com/user"))).send().await {
                Ok(r) if r.status().is_success() => r.json().await.unwrap_or(json!({})),
                _ => json!({}),
            };
            Ok(Poll::Done(json!({
                "accessToken": at, "refreshToken": v["refresh_token"], "expiresIn": v["expires_in"],
                "name": if truthy(&u["login"]) { u["login"].clone() } else { u["name"].clone() },
                "displayName": if truthy(&u["name"]) { u["name"].clone() } else { u["login"].clone() },
                "email": u["email"],
                "providerSpecificData": {"copilotToken": cp["token"], "copilotTokenExpiresAt": cp["expires_at"], "githubUserId": u["id"], "githubLogin": u["login"], "githubName": u["name"], "githubEmail": u["email"]},
            })))
        }
        "grok-cli" => {
            let ua = crate::providers::grok::pager_user_agent();
            let r = form_req(c["tokenUrl"].as_str().unwrap_or(""), &[("grant_type", grant), ("device_code", dc), ("client_id", c["clientId"].as_str().unwrap_or(""))]).header("user-agent", ua).send().await.map_err(|e| e.to_string())?;
            let (st, v, _) = json_or_text(r).await;
            if let Some(p) = pending_or(&v, st) {
                return Ok(p);
            }
            let Some(at) = v["access_token"].as_str() else { return Ok(Poll::Error("no_access_token".into())) };
            let user: Value = match http().get("https://cli-chat-proxy.grok.com/v1/user").bearer_auth(at).header("accept", "application/json").header("user-agent", ua).header("x-xai-token-auth", "xai-grok-cli").header("x-grok-client-version", crate::providers::grok::cli_version()).send().await {
                Ok(r) if r.status().is_success() => r.json().await.unwrap_or(Value::Null),
                _ => Value::Null,
            };
            let email = v["id_token"].as_str().and_then(email_from_jwt).or_else(|| email_from_jwt(at)).or_else(|| user["email"].as_str().map(str::to_owned));
            let name = [user["firstName"].as_str(), user["lastName"].as_str()].into_iter().flatten().collect::<Vec<_>>().join(" ");
            let exp_at = v["expires_in"].as_i64().map(|e| iso_from_ms(now_ms() + e * 1000));
            Ok(Poll::Done(json!({
                "accessToken": at, "refreshToken": v["refresh_token"], "expiresIn": v["expires_in"], "expiresAt": exp_at, "scope": v["scope"],
                "email": email, "displayName": if name.trim().is_empty() { Value::Null } else { json!(name.trim()) },
                "providerSpecificData": {"authMethod": "device_code", "idToken": v["id_token"], "email": email, "userId": if truthy(&user["userId"]) { user["userId"].clone() } else { user["principalId"].clone() }, "hasGrokCodeAccess": user["hasGrokCodeAccess"], "subscriptionTier": user["subscriptionTier"]},
            })))
        }
        "muse" => {
            let r = form_req(c["tokenUrl"].as_str().unwrap_or(""), &[("grant_type", grant), ("device_code", dc), ("client_id", c["clientId"].as_str().unwrap_or(""))]).header("x-api-version", "1.0.0").send().await.map_err(|e| e.to_string())?;
            let (st, v, _) = json_or_text(r).await;
            if let Some(p) = pending_or(&v, st) {
                return Ok(p);
            }
            let Some(at) = v["access_token"].as_str() else { return Ok(Poll::Error("no_access_token".into())) };
            // Mint the Model API key (retry transient errors; device codes are one-shot).
            let mut last = (0u16, String::new());
            for attempt in 0..3 {
                let r = http().post("https://api.meta.ai/muse-code/key").bearer_auth(at).header("accept", "application/json").header("x-api-version", "1.0.0").json(&json!({"onboard": true})).send().await.map_err(|e| e.to_string())?;
                let st = r.status().as_u16();
                let t = r.text().await.unwrap_or_default();
                last = (st, t);
                if !(st == 429 || st >= 500) || attempt == 2 {
                    break;
                }
                tokio::time::sleep(Duration::from_secs(5 * (attempt + 1))).await;
            }
            let (st, t) = last;
            if !ok(st) {
                let mut msg = format!("{st} {}", t.chars().take(200).collect::<String>());
                if let Ok(e) = serde_json::from_str::<Value>(&t) {
                    if truthy(&e["title"]) || truthy(&e["detail"]) {
                        msg = [e["title"].as_str(), e["detail"].as_str()].into_iter().flatten().collect::<Vec<_>>().join(": ");
                        if let Some(u) = e["action_url"].as_str() {
                            msg.push_str(&format!(" — {u}"));
                        }
                    }
                }
                return Ok(Poll::Error(format!("Muse Code key mint failed: {msg}")));
            }
            let p: Value = serde_json::from_str(&t).map_err(|_| "Muse Code key mint returned invalid JSON".to_string())?;
            if p["is_subs_active"] == json!(false) {
                return Ok(Poll::Error("Muse Code subscription is inactive — activate it on muse.ai first".into()));
            }
            let Some(key) = p["api_key"].as_str() else {
                let au = p["action_url"].as_str().or_else(|| p["require_payment_action_url"].as_str());
                return Ok(Poll::Error(match au {
                    Some(u) => format!("Muse Code subscription required: {u}"),
                    None => "Muse Code key response is missing api_key".into(),
                }));
            };
            Ok(Poll::Done(json!({"accessToken": key, "refreshToken": null, "expiresIn": null, "email": p["user_email"].as_str().map(|s| s.trim().to_lowercase()), "providerSpecificData": {"authMethod": "device_code", "oauthAccessToken": at, "subscriptionTier": p["subs_tier_name"]}})))
        }
        "kimi" | "kimi-coding" => {
            let dev = d.extra["_kimiDeviceId"].as_str().map(str::to_owned);
            let mut rb = form_req(c["tokenUrl"].as_str().unwrap_or(""), &[("grant_type", grant), ("client_id", c["clientId"].as_str().unwrap_or("")), ("device_code", dc)]);
            for (k, v) in crate::exec::kimi_headers(dev.as_deref()) {
                rb = rb.header(k, v);
            }
            let (st, v, _) = json_or_text(rb.send().await.map_err(|e| e.to_string())?).await;
            if let Some(p) = pending_or(&v, st) {
                return Ok(p);
            }
            let Some(at) = v["access_token"].as_str() else { return Ok(Poll::Error("no_access_token".into())) };
            let mut psd = json!({"authMethod": "device_code"});
            if let Some(d) = dev {
                psd["deviceId"] = json!(d);
            }
            Ok(Poll::Done(json!({"accessToken": at, "refreshToken": v["refresh_token"], "expiresIn": v["expires_in"], "providerSpecificData": psd})))
        }
        "kilocode" => {
            let r = http().get(format!("{}/{dc}", c["pollUrlBase"].as_str().unwrap_or(""))).send().await.map_err(|e| e.to_string())?;
            match r.status().as_u16() {
                202 => return Ok(Poll::Pending),
                403 => return Ok(Poll::Error("Authorization denied by user".into())),
                410 => return Ok(Poll::Error("Authorization code expired".into())),
                s if !ok(s) => return Ok(Poll::Error(format!("Poll failed: {s}"))),
                _ => {}
            }
            let v: Value = r.json().await.map_err(|e| e.to_string())?;
            if v["status"] != "approved" || !truthy(&v["token"]) {
                return Ok(Poll::Pending);
            }
            let tok = v["token"].as_str().unwrap_or("");
            let org = match http().get(format!("{}/api/profile", c["apiBaseUrl"].as_str().unwrap_or("https://api.kilo.ai"))).bearer_auth(tok).send().await {
                Ok(r) if r.status().is_success() => r.json::<Value>().await.ok().and_then(|p| p["organizations"][0]["id"].as_str().map(str::to_owned)),
                _ => None,
            };
            let mut m = json!({"accessToken": tok, "refreshToken": null, "expiresIn": null, "email": v["userEmail"]});
            if let Some(o) = org {
                m["providerSpecificData"] = json!({"orgId": o});
            }
            Ok(Poll::Done(m))
        }
        "codebuddy-cn" | "codebuddy-intl" => {
            let domain = if provider == "codebuddy-cn" { "copilot.tencent.com" } else { "www.codebuddy.ai" };
            let r = http()
                .get(format!("{}?state={}", c["tokenUrl"].as_str().unwrap_or(""), urlencode(dc)))
                .header("accept", "application/json")
                .header("user-agent", c["userAgent"].as_str().unwrap_or(""))
                .header("x-requested-with", "XMLHttpRequest")
                .header("x-domain", domain)
                .header("x-no-authorization", "true")
                .header("x-no-user-id", "true")
                .header("x-no-enterprise-id", "true")
                .header("x-no-department-info", "true")
                .header("x-product", "SaaS")
                .send()
                .await
                .map_err(|e| e.to_string())?;
            if !r.status().is_success() {
                return Ok(Poll::Error("request_failed".into()));
            }
            let v: Value = r.json().await.map_err(|e| e.to_string())?;
            if v["code"] == json!(0) && truthy(&v["data"]["accessToken"]) {
                return Ok(Poll::Done(json!({"accessToken": v["data"]["accessToken"], "refreshToken": v["data"]["refreshToken"].as_str().unwrap_or(""), "expiresIn": v["data"]["expiresIn"].as_i64().unwrap_or(86400), "providerSpecificData": {}})));
            }
            if v["code"] == json!(11217) {
                return Ok(Poll::Pending);
            }
            Ok(Poll::Error(v["msg"].as_str().unwrap_or("unknown_error").into()))
        }
        "qoder" | "qoder-cn" => {
            let url = format!("{}?nonce={}&verifier={}&challenge_method=S256", c["deviceTokenUrl"].as_str().unwrap_or(""), urlencode(dc), urlencode(&d.code_verifier));
            let r = http().get(url).header("accept", "application/json").header("user-agent", "Go-http-client/2.0").timeout(Duration::from_secs(15)).send().await.map_err(|e| e.to_string())?;
            let st = r.status().as_u16();
            if st == 202 || st == 404 {
                return Ok(Poll::Pending);
            }
            let t = r.text().await.unwrap_or_default();
            if !ok(st) {
                let m = serde_json::from_str::<Value>(&t).ok().and_then(|b| b["message"].as_str().map(str::to_owned));
                return Ok(Poll::Error(format!("Qoder device token poll failed: {}", m.unwrap_or(format!("HTTP {st}")))));
            }
            let b: Value = serde_json::from_str(&t).map_err(|e| format!("Qoder device token poll: invalid JSON response ({e})"))?;
            let Some(tok) = b["token"].as_str() else { return Ok(Poll::Error("Qoder device token poll returned 200 but no token".into())) };
            let exp_ms = qoder_expiry(&b["expires_at"], &b["expires_in"]);
            let ui: Value = match http().get(c["userInfoUrl"].as_str().unwrap_or("")).bearer_auth(tok).header("accept", "application/json").header("user-agent", "Go-http-client/2.0").timeout(Duration::from_secs(15)).send().await {
                Ok(r) if r.status().is_success() => r.json().await.unwrap_or(json!({})),
                _ => json!({}),
            };
            let name = ui["name"].as_str().or_else(|| ui["username"].as_str()).unwrap_or("").trim().to_string();
            let uid = b["user_id"].as_str().map(str::to_owned).unwrap_or_else(|| if b["user_id"].is_null() { String::new() } else { js_string(&b["user_id"]) });
            let email = ui["email"].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).or_else(|| (!uid.is_empty()).then(|| format!("qoder-user-{uid}")));
            let expires_in = ((exp_ms - now_ms()) / 1000).max(24 * 3600);
            Ok(Poll::Done(json!({
                "accessToken": tok, "refreshToken": b["refresh_token"].as_str().unwrap_or(""), "expiresIn": expires_in, "email": email,
                "displayName": if name.is_empty() { Value::Null } else { json!(name) },
                "providerSpecificData": {"authMethod": "device", "userId": uid, "machineId": d.extra["_qoderMachineId"].as_str().unwrap_or(""), "organizationId": ui["organization_id"].as_str().unwrap_or("").trim()},
            })))
        }
        "kiro" => {
            let region = d.extra["_region"].as_str().unwrap_or("us-east-1");
            if !regex::Regex::new(r"^[a-z]{2}-[a-z]+-\d{1,2}$").unwrap().is_match(region) {
                return Ok(Poll::Error("Invalid region".into()));
            }
            let r = http().post(format!("https://oidc.{region}.amazonaws.com/token")).header("accept", "application/json").json(&json!({"clientId": d.extra["_clientId"], "clientSecret": d.extra["_clientSecret"], "deviceCode": dc, "grantType": grant})).send().await.map_err(|e| e.to_string())?;
            let (_, v, _) = json_or_text(r).await;
            let Some(at) = v["accessToken"].as_str() else {
                return Ok(match v["error"].as_str().unwrap_or("authorization_pending") {
                    "authorization_pending" | "slow_down" => Poll::Pending,
                    e => Poll::Error(v["error_description"].as_str().or_else(|| v["message"].as_str()).map(|m| format!("{e}: {m}")).unwrap_or_else(|| e.to_string())),
                });
            };
            let mut arn = v["profileArn"].clone();
            if !truthy(&arn) {
                arn = crate::oauth::refresh::fetch_kiro_profile_arn(at).await.map(Value::String).unwrap_or(Value::Null);
            }
            Ok(Poll::Done(json!({
                "accessToken": at, "refreshToken": v["refreshToken"], "expiresIn": v["expiresIn"], "email": email_from_jwt(at),
                "providerSpecificData": {"profileArn": arn, "clientId": d.extra["_clientId"], "clientSecret": d.extra["_clientSecret"], "region": region, "authMethod": d.extra["_authMethod"].as_str().unwrap_or("builder-id"), "startUrl": d.extra["_startUrl"]},
            })))
        }
        "glm" => {
            let Some(poll) = d.extra["_zcodePollToken"].as_str() else { return Ok(Poll::Error("Missing ZCode poll token — restart the login flow".into())) };
            let r = http().get(format!("{}/{}", c["cliPollUrl"].as_str().unwrap_or(""), urlencode(dc))).bearer_auth(poll).send().await.map_err(|e| e.to_string())?;
            if !r.status().is_success() {
                return Ok(Poll::Error(format!("ZCode poll failed (HTTP {})", r.status().as_u16())));
            }
            let p: Value = r.json().await.map_err(|e| e.to_string())?;
            if !glm_ok(&p["code"]) {
                return Ok(Poll::Error(p["msg"].as_str().unwrap_or("ZCode poll failed").into()));
            }
            let data = &p["data"];
            match data["status"].as_str() {
                Some("pending") => return Ok(Poll::Pending),
                Some("failed") => return Ok(Poll::Error("ZCode authorization failed or was cancelled".into())),
                Some("ready") => {}
                _ => return Ok(Poll::Pending),
            }
            let pid = c["providerId"].as_str().unwrap_or("zai");
            let pd = if data[pid].is_object() { &data[pid] } else { &data[data["providerId"].as_str().unwrap_or("")] };
            let zat = [&pd["access_token"], &pd["accessToken"], &data["accessToken"], &data["access_token"]].into_iter().find(|v| truthy(v)).and_then(|v| v.as_str()).ok_or("ZCode poll response missing access token")?;
            let (key, biz) = glm_plan_key(&c, zat).await?;
            let user = &data["user"];
            let rt = [&pd["refresh_token"], &pd["refreshToken"], &data["refresh_token"], &data["refreshToken"]].into_iter().find(|v| truthy(v)).cloned();
            let mut psd = json!({"authMethod": "cli_poll", "username": user["name"], "userId": user["user_id"], "zcodeJwtToken": data["token"], "zaiBusinessToken": biz});
            if let Some(rt) = rt {
                psd["zaiRefreshToken"] = rt;
            }
            let dn = if truthy(&user["name"]) { user["name"].clone() } else { user["email"].clone() };
            Ok(Poll::Done(json!({"accessToken": key, "refreshToken": null, "email": user["email"], "displayName": dn, "providerSpecificData": psd})))
        }
        other => Ok(Poll::Error(format!("Provider {other} does not support device code flow"))),
    }
}

fn qoder_expiry(at: &Value, inn: &Value) -> i64 {
    if let Some(n) = at.as_i64().filter(|n| *n > 0) {
        return n;
    }
    if let Some(s) = at.as_str().map(str::trim).filter(|s| !s.is_empty()) {
        if let Ok(n) = s.parse::<i64>() {
            if n > 0 {
                return n;
            }
        }
        if let Some(ms) = crate::jsv::parse_iso_ms(s) {
            return ms;
        }
    }
    if let Some(n) = inn.as_f64().filter(|n| *n >= 0.0) {
        return now_ms() + (n * 1000.0) as i64;
    }
    now_ms() + 30 * 24 * 3600 * 1000
}

async fn glm_json(rb: reqwest::RequestBuilder, label: &str) -> Result<Value, String> {
    let r = rb.send().await.map_err(|e| e.to_string())?;
    let st = r.status().as_u16();
    let t = r.text().await.unwrap_or_default();
    if !ok(st) {
        return Err(format!("Z.ai {label} request failed (HTTP {st}): {}", t.chars().take(200).collect::<String>()));
    }
    let p: Value = serde_json::from_str(&t).map_err(|_| format!("Z.ai {label} response is not valid JSON"))?;
    if !glm_ok(&p["code"]) || p["success"] == json!(false) {
        return Err(p["msg"].as_str().map(str::to_owned).unwrap_or(format!("Z.ai {label} returned business error {}", p["code"])));
    }
    Ok(if p.get("data").map(|d| !d.is_null()).unwrap_or(false) { p["data"].clone() } else { p })
}

async fn glm_plan_key(c: &Value, zat: &str) -> Result<(String, String), String> {
    let lp = glm_json(http().post(c["businessLoginUrl"].as_str().unwrap_or("")).json(&json!({"token": zat})), "Z.ai business login").await?;
    let biz = lp["access_token"].as_str().or_else(|| lp["accessToken"].as_str()).map(str::trim).filter(|s| !s.is_empty()).ok_or("Z.ai business login response is missing access_token")?.to_string();
    let base = c["apiBaseUrl"].as_str().unwrap_or("https://api.z.ai");
    let ci = glm_json(http().get(format!("{base}/api/biz/customer/getCustomerInfo")).bearer_auth(&biz), "customer info").await?;
    let is_default = |n: &Value| {
        let s = n.as_str().unwrap_or("").trim().to_lowercase();
        s.contains("默认机构") || s.contains("默认项目") || s == "default"
    };
    let orgs: Vec<(Value, Vec<Value>)> = ci["organizations"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|o| (o.clone(), o["projects"].as_array().into_iter().flatten().filter(|p| js_string(&p["projectType"]).trim() != "2").cloned().collect::<Vec<_>>()))
        .filter(|(o, ps)| truthy(&o["organizationId"]) && !ps.is_empty())
        .collect();
    let (org, projects) = orgs.iter().find(|(o, _)| is_default(&o["organizationName"])).or_else(|| orgs.first()).ok_or("Unable to resolve Z.ai organization and project for the coding plan")?;
    let project = projects.iter().find(|p| is_default(&p["projectName"])).unwrap_or(&projects[0]);
    let list = format!("{base}/api/biz/v1/organization/{}/projects/{}/api_keys", js_string(&org["organizationId"]), js_string(&project["projectId"]));
    let keys = glm_json(http().get(&list).bearer_auth(&biz), "api keys").await?;
    let name = c["planApiKeyName"].as_str().unwrap_or("zcode-api-key");
    let mut entry = keys.as_array().and_then(|a| a.iter().find(|k| k["name"] == name).cloned());
    if entry.is_none() {
        entry = Some(glm_json(http().post(&list).bearer_auth(&biz).json(&json!({"name": name})), "api key create").await?);
    }
    let ak = entry.and_then(|e| e["apiKey"].as_str().map(|s| s.trim().to_string())).filter(|s| !s.is_empty()).ok_or("Z.ai api_keys response is missing apiKey")?;
    let sec = glm_json(http().get(format!("{list}/copy/{}", urlencode(&ak))).bearer_auth(&biz), "api key copy").await?;
    let sk = sec["secretKey"].as_str().map(str::trim).filter(|s| !s.is_empty()).ok_or("Z.ai api key copy response is missing secretKey")?;
    Ok((format!("{ak}.{sk}"), biz))
}

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

/// Validates pasted credentials for import-style logins; returns the
/// connection document fields.
pub async fn import(provider: &str, method: &str, f: &Value) -> Result<Value, String> {
    let s = |k: &str| f[k].as_str().map(str::trim).unwrap_or("").to_string();
    match (provider, method) {
        ("cursor", _) => {
            let (at, mid) = (s("accessToken"), s("machineId"));
            if at.is_empty() {
                return Err("Access token is required".into());
            }
            if mid.is_empty() {
                return Err("Machine ID is required".into());
            }
            if at.len() < 50 {
                return Err("Invalid token format. Token appears too short.".into());
            }
            let hexish = mid.replace('-', "");
            if hexish.len() < 32 || !hexish.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err("Invalid machine ID format. Expected UUID format.".into());
            }
            let p = decode_jwt_payload(&at).unwrap_or(json!({}));
            Ok(json!({"authType": "oauth", "accessToken": at, "refreshToken": null, "expiresIn": 86400, "email": if truthy(&p["email"]) { p["email"].clone() } else { p["sub"].clone() }, "providerSpecificData": {"machineId": mid, "authMethod": "imported", "provider": "Imported", "userId": if truthy(&p["sub"]) { p["sub"].clone() } else { p["user_id"].clone() }}}))
        }
        ("zed", _) => {
            let (at, uid) = (s("accessToken"), s("userId"));
            if at.is_empty() {
                return Err("Access token is required".into());
            }
            if uid.is_empty() {
                return Err("User id is required".into());
            }
            let sid = Some(s("systemId")).filter(|x| !x.is_empty()).unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let mut t = zed_tokens(&at, &uid, &sid, "imported").await?;
            t["authType"] = json!("oauth");
            Ok(t)
        }
        ("codex", _) => {
            let tok = s("accessToken");
            if tok.is_empty() {
                return Err("Access token is required".into());
            }
            let p = decode_jwt_payload(&tok).unwrap_or(json!({}));
            let auth = &p["https://api.openai.com/auth"];
            let email = [&p["https://api.openai.com/profile"]["email"], &p["email"], &p["preferred_username"]].into_iter().find(|v| truthy(v)).cloned().unwrap_or(Value::Null);
            let mut psd = json!({"authMethod": "access_token"});
            for (a, b) in [("chatgpt_account_id", "chatgptAccountId"), ("chatgpt_plan_type", "chatgptPlanType")] {
                if truthy(&auth[a]) {
                    psd[b] = auth[a].clone();
                }
            }
            if truthy(&p["exp"]) {
                psd["jwtExp"] = p["exp"].clone();
            }
            let name = Some(s("name")).filter(|n| !n.is_empty()).map(Value::String).unwrap_or_else(|| if email.is_null() { json!("ChatGPT Access Token") } else { email.clone() });
            Ok(json!({"authType": "access_token", "accessToken": tok, "name": name, "email": email, "providerSpecificData": psd}))
        }
        ("kiro", "import") => {
            let rt = s("refreshToken");
            if rt.is_empty() {
                return Err("Refresh token is required".into());
            }
            let (cid, cs) = (s("clientId"), s("clientSecret"));
            let idc = !cid.is_empty() && !cs.is_empty();
            let region = Some(s("region")).filter(|r| !r.is_empty()).unwrap_or_else(|| "us-east-1".into());
            let psd = if idc { json!({"clientId": cid, "clientSecret": cs, "region": region, "authMethod": "idc"}) } else { json!({}) };
            let creds = json!({"refreshToken": rt, "providerSpecificData": psd});
            let t = crate::oauth::refresh::refresh_kiro(&rt, &psd, &creds).await.ok_or("Kiro token refresh failed — check the refresh token")?;
            let at = t["accessToken"].as_str().unwrap_or("");
            let mut out_psd = json!({"profileArn": if truthy(&f["profileArn"]) { f["profileArn"].clone() } else { t["providerSpecificData"]["profileArn"].clone() }, "authMethod": if idc { "idc" } else { "imported" }, "provider": if idc { "Enterprise" } else { "Imported" }});
            if idc {
                out_psd["clientId"] = json!(cid);
                out_psd["clientSecret"] = json!(cs);
                out_psd["region"] = json!(region);
            }
            Ok(json!({"authType": "oauth", "accessToken": at, "refreshToken": if truthy(&t["refreshToken"]) { t["refreshToken"].clone() } else { json!(rt) }, "expiresIn": t["expiresIn"].as_i64().unwrap_or(3600), "email": email_from_jwt(at), "providerSpecificData": out_psd}))
        }
        ("kiro", "api-key") => {
            let key = s("apiKey");
            if key.is_empty() {
                return Err("API key is required".into());
            }
            let region = Some(s("region")).filter(|r| !r.is_empty()).unwrap_or_else(|| "us-east-1".into());
            let ok_models = crate::providers::kiro::list_api_key_models(&key, &region).await;
            if let Err(e) = ok_models {
                return Err(format!("API key validation failed: {e}"));
            }
            Ok(json!({"authType": "api_key", "accessToken": key, "refreshToken": null, "expiresIn": 365 * 24 * 3600, "email": email_from_jwt(&key), "providerSpecificData": {"region": region, "authMethod": "api_key", "provider": "API Key"}}))
        }
        ("iflow", _) => {
            let cookie = s("cookie");
            if cookie.is_empty() {
                return Err("Cookie is required".into());
            }
            if !cookie.contains("BXAuth=") {
                return Err("Cookie must contain BXAuth field".into());
            }
            let cookie = if cookie.ends_with(';') { cookie } else { format!("{cookie};") };
            let ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36";
            let r = http().get("https://platform.iflow.cn/api/openapi/apikey").header("cookie", &cookie).header("accept", "application/json, text/plain, */*").header("user-agent", ua).send().await.map_err(|e| e.to_string())?;
            let (st, g, t) = json_or_text(r).await;
            if !ok(st) {
                return Err(format!("Failed to fetch API key info: {t}"));
            }
            if g["success"] != json!(true) {
                return Err(format!("API key fetch failed: {}", g["message"].as_str().unwrap_or("")));
            }
            let name = g["data"]["name"].as_str().ok_or("Missing name in API key info")?.to_string();
            let r = http().post("https://platform.iflow.cn/api/openapi/apikey").header("cookie", &cookie).header("user-agent", ua).header("origin", "https://platform.iflow.cn").header("referer", "https://platform.iflow.cn/").json(&json!({"name": name})).send().await.map_err(|e| e.to_string())?;
            let (st, p, t) = json_or_text(r).await;
            if !ok(st) {
                return Err(format!("Failed to refresh API key: {t}"));
            }
            if p["success"] != json!(true) {
                return Err(format!("API key refresh failed: {}", p["message"].as_str().unwrap_or("")));
            }
            let key = p["data"]["apiKey"].as_str().ok_or("Missing API key in response")?;
            let bx = regex::Regex::new(r"BXAuth=([^;]+)").unwrap().captures(&cookie).map(|c| format!("BXAuth={};", &c[1])).unwrap_or_default();
            let nm = p["data"]["name"].as_str().unwrap_or(&name).to_string();
            Ok(json!({"authType": "cookie", "name": nm, "email": nm, "apiKey": key, "providerSpecificData": {"cookie": bx, "expireTime": p["data"]["expireTime"]}}))
        }
        ("gitlab", _) => {
            let tok = s("token");
            if tok.is_empty() {
                return Err("Personal Access Token is required".into());
            }
            let base = Some(s("baseUrl")).filter(|b| !b.is_empty()).unwrap_or_else(|| "https://gitlab.com".into()).trim_end_matches('/').to_string();
            let r = http().get(format!("{base}/api/v4/user")).header("private-token", &tok).header("accept", "application/json").send().await.map_err(|e| e.to_string())?;
            let (st, u, t) = json_or_text(r).await;
            if !ok(st) {
                return Err(format!("GitLab token verification failed: {t}"));
            }
            let email = u["email"].as_str().filter(|s| !s.is_empty()).or_else(|| u["public_email"].as_str()).unwrap_or("").to_string();
            let dn = [&u["name"], &u["username"]].into_iter().find(|v| truthy(v)).cloned().unwrap_or(json!(email));
            Ok(json!({"authType": "oauth", "accessToken": tok, "refreshToken": null, "email": email, "displayName": dn, "providerSpecificData": {"username": u["username"].as_str().unwrap_or(""), "email": email, "name": u["name"].as_str().unwrap_or(""), "baseUrl": base, "authKind": "personal_access_token"}}))
        }
        ("xiaomi-mimo", _) => {
            let key = s("apiKey");
            let pass = s("mimoPassToken");
            if key.is_empty() && pass.is_empty() {
                return Err("API key is required".into());
            }
            if !key.is_empty() && !key.starts_with("sk-") {
                return Err("Invalid key format — expected sk- prefix".into());
            }
            let base = Some(s("baseUrl")).filter(|b| !b.is_empty()).unwrap_or_else(|| "https://api.xiaomimimo.com/v1".into()).trim_end_matches('/').to_string();
            let mut validated = false;
            if !key.is_empty() {
                if let Ok(r) = http().get(format!("{base}/models")).bearer_auth(&key).header("x-mimo-source", "mimocode-cli").timeout(Duration::from_secs(10)).send().await {
                    validated = r.status().is_success();
                }
            }
            let session = key.is_empty();
            let uid = s("uid");
            Ok(json!({
                "authType": if session { "oauth" } else { "api_key" }, "accessToken": if key.is_empty() { Value::Null } else { json!(key) },
                "expiresIn": 365 * 24 * 3600, "email": if uid.is_empty() { Value::Null } else { json!(format!("{uid}@xiaomi")) },
                "displayName": if uid.is_empty() { json!("Xiaomi MiMo") } else { json!(format!("Xiaomi {uid}{}", if session { " (Session)" } else { "" })) },
                "testStatus": if validated || session { "active" } else { "untested" },
                "providerSpecificData": {"uid": if uid.is_empty() { Value::Null } else { json!(uid) }, "baseUrl": base, "authMethod": if session { "session" } else { "api_key" }, "provider": if session { "Session Login" } else { "API Key" }, "region": Some(s("region")).filter(|r| !r.is_empty()).unwrap_or_else(|| "cn".into()), "mimoPassToken": if pass.is_empty() { Value::Null } else { json!(pass) }, "mimoUserId": f["mimoUserId"], "mimoCUserId": f["mimoCUserId"]},
            }))
        }
        _ => Err(format!("No import method '{method}' for {provider}")),
    }
}

/// Kiro social (Google/GitHub) login URL; the redirect is a `kiro://` URL the
/// user copies back.
pub fn kiro_social_start(idp: &str) -> AuthStart {
    let p = pkce(32);
    let idp_name = if idp == "google" { "Google" } else { "Github" };
    let redirect = "kiro://kiro.kiroAgent/authenticate-success";
    let url = format!("https://prod.us-east-1.auth.desktop.kiro.dev/login?idp={idp_name}&redirect_uri={}&code_challenge={}&code_challenge_method=S256&state={}&prompt=select_account", urlencode(redirect), p.challenge, p.state);
    AuthStart { auth_url: url, state: p.state, code_verifier: p.verifier, redirect_uri: redirect.into(), meta: json!({"idp": idp}) }
}

pub async fn kiro_social_exchange(code: &str, a: &AuthStart) -> Result<Value, String> {
    let r = http().post("https://prod.us-east-1.auth.desktop.kiro.dev/oauth/token").json(&json!({"code": code, "code_verifier": a.code_verifier, "redirect_uri": a.redirect_uri})).send().await.map_err(|e| e.to_string())?;
    let (st, v, t) = json_or_text(r).await;
    if !ok(st) {
        return Err(format!("Token exchange failed: {t}"));
    }
    let idp = a.meta["idp"].as_str().unwrap_or("google");
    let at = v["accessToken"].as_str().unwrap_or("");
    Ok(json!({"accessToken": at, "refreshToken": v["refreshToken"], "expiresIn": v["expiresIn"].as_i64().unwrap_or(3600), "email": email_from_jwt(at), "providerSpecificData": {"profileArn": v["profileArn"], "authMethod": idp, "provider": if idp == "google" { "Google" } else { "Github" }}}))
}

// ---------------------------------------------------------------------------
// Connection persistence (createProviderConnection)
// ---------------------------------------------------------------------------

/// Stores mapped tokens as a connection, updating an existing OAuth row for
/// the same account (createProviderConnection dedupe rules).
pub fn save_connection(db: &crate::db::Db, provider: &str, tokens: &Value) -> anyhow::Result<String> {
    let provider = if provider == "kimi-coding" { "kimi" } else { provider };
    let mut doc = tokens.clone();
    if !doc["authType"].is_string() {
        doc["authType"] = json!("oauth");
    }
    doc["provider"] = json!(provider);
    if let Some(e) = doc["expiresIn"].as_i64().filter(|e| *e > 0) {
        if !truthy(&doc["expiresAt"]) {
            doc["expiresAt"] = json!(iso_from_ms(now_ms() + e * 1000));
        }
    }
    if !doc["testStatus"].is_string() {
        doc["testStatus"] = json!("active");
    }
    if let Some(o) = doc.as_object_mut() {
        o.retain(|_, v| !v.is_null());
    }
    if doc["authType"] == "oauth" {
        if let Some(email) = doc["email"].as_str().filter(|s| !s.is_empty()).map(str::to_owned) {
            let in_ws = doc["providerSpecificData"]["chatgptAccountId"].as_str().map(str::to_owned);
            let in_user = doc["providerSpecificData"]["username"].as_str().map(str::to_owned);
            let existing = db.connections_for(provider, false).into_iter().find(|c| {
                if c["authType"] != "oauth" || c["email"] != email.as_str() {
                    return false;
                }
                let ex_ws = c["providerSpecificData"]["chatgptAccountId"].as_str().map(str::to_owned);
                if provider == "codex" {
                    return in_ws.is_some() && in_ws == ex_ws;
                }
                match (&in_ws, &ex_ws) {
                    (Some(a), Some(b)) => return a == b,
                    (Some(_), None) | (None, Some(_)) => return false,
                    _ => {}
                }
                let ex_user = c["providerSpecificData"]["username"].as_str().map(str::to_owned);
                match (&in_user, &ex_user) {
                    (Some(a), Some(b)) => a == b,
                    (None, None) => true,
                    _ => false,
                }
            });
            if let Some(ex) = existing {
                let id = ex["id"].as_str().unwrap_or("").to_string();
                db.update_connection(&id, &doc)?;
                return Ok(id);
            }
        }
    }
    if !doc["name"].is_string() {
        let n = [&doc["displayName"], &doc["email"]].into_iter().find(|v| truthy(v)).cloned().unwrap_or_else(|| json!(format!("Account {}", db.connections_for(provider, false).len() + 1)));
        doc["name"] = n;
    }
    db.insert_connection(&doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        let p = pkce(32);
        assert_eq!(B64URL.encode(sha2::Sha256::digest(p.verifier.as_bytes())), p.challenge);
        assert_eq!(parse_callback_input("http://localhost/callback?code=abc%2Fd&state=s1"), ("abc/d".into(), Some("s1".into())));
        assert_eq!(parse_callback_input("rawcode#st").0, "rawcode#st");
        assert_eq!(pct_decode("a%20b+c%"), "a b c%");
        assert_eq!(flow("github"), Some(Flow::Device));
        let (u, t) = parse_zed_callback("http://127.0.0.1:58443/?user_id=7&access_token=xyz").unwrap();
        assert_eq!((u.as_str(), t.as_str()), ("7", "xyz"));
    }

    #[test]
    fn zed_rsa_roundtrip() {
        use rsa::pkcs1::DecodeRsaPrivateKey;
        let z = zed_native_auth(58443, None).unwrap();
        assert!(z.auth_url.contains("native_app_public_key="));
        let key = rsa::RsaPrivateKey::from_pkcs1_pem(&z.private_key_pem).unwrap();
        let ct = key.to_public_key().encrypt(&mut OsRng, rsa::Oaep::new::<sha2_010::Sha256>(), b"secret-token").unwrap();
        assert_eq!(zed_decrypt(&B64URL.encode(ct), &z.private_key_pem).unwrap(), "secret-token");
    }

    #[test]
    fn xiaomi_roundtrip() {
        use aes_gcm::aead::{Aead, KeyInit};
        let (pk_spki, sk) = x25519_keypair();
        let spki = B64.decode(pk_spki).unwrap();
        let client_pub: [u8; 32] = spki[12..].try_into().unwrap();
        // Server side: ephemeral key, ECDH with the client's public key.
        let eph = x25519_dalek::StaticSecret::random_from_rng(OsRng);
        let eph_pub = x25519_dalek::PublicKey::from(&eph);
        let shared = eph.diffie_hellman(&x25519_dalek::PublicKey::from(client_pub));
        let key = sha2::Sha256::digest(shared.as_bytes());
        let nonce = [7u8; 12];
        let ct = aes_gcm::Aes256Gcm::new_from_slice(&key).unwrap().encrypt(aes_gcm::Nonce::from_slice(&nonce), br#"{"uid":"42","sk":"sk-abc"}"#.as_ref()).unwrap();
        let mut payload = nonce.to_vec();
        payload.extend_from_slice(eph_pub.as_bytes());
        payload.extend_from_slice(&ct);
        let r = xiaomi_decrypt(&sk, &B64.encode(payload)).unwrap();
        assert_eq!(r["sk"], "sk-abc");
        assert_eq!(r["uid"], "42");
    }

    #[test]
    fn dedupe_save() {
        let db = crate::db::Db::open_in_memory().unwrap();
        let a = save_connection(&db, "claude", &json!({"accessToken": "a", "email": "x@y", "expiresIn": 60})).unwrap();
        let b = save_connection(&db, "claude", &json!({"accessToken": "b", "email": "x@y"})).unwrap();
        assert_eq!(a, b);
        assert_eq!(db.get_connection(&a).unwrap()["accessToken"], "b");
        let c = save_connection(&db, "codex", &json!({"accessToken": "c", "email": "x@y", "providerSpecificData": {"chatgptAccountId": "w1"}})).unwrap();
        let d = save_connection(&db, "codex", &json!({"accessToken": "d", "email": "x@y", "providerSpecificData": {"chatgptAccountId": "w2"}})).unwrap();
        assert_ne!(c, d);
    }
}
