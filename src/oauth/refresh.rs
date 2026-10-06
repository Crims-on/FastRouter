//! Token refresh (port of services/tokenRefresh.js, tokenRefresh/providers.js,
//! oauthCredentialManager.js and the executors' refreshCredentials).

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use base64::Engine;
use serde_json::{Value, json};

use crate::exec::client_for;
use crate::jsv::{iso_from_ms, now_ms, parse_iso_ms, truthy};
use crate::oauth::form;
use crate::registry::REG;

const GOOGLE_TOKEN: &str = "https://oauth2.googleapis.com/token";

// ---------------------------------------------------------------------------
// dedup (tokenRefresh/dedup.js): one in-flight refresh per (provider, token),
// result reused for 10 s.
// ---------------------------------------------------------------------------

type Slot = std::sync::Arc<tokio::sync::Mutex<Option<(Option<Value>, i64)>>>;
static DEDUP: LazyLock<Mutex<HashMap<String, Slot>>> = LazyLock::new(Default::default);

async fn dedup<F, Fut>(key: &str, token: &str, f: F) -> Option<Value>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Option<Value>>,
{
    if token.is_empty() {
        return f().await;
    }
    let k = format!("{key}:{token}");
    let slot = DEDUP.lock().unwrap().entry(k.clone()).or_default().clone();
    let mut g = slot.lock().await;
    if let Some((r, at)) = g.as_ref() {
        if now_ms() - at < 10_000 {
            return r.clone();
        }
    }
    let r = f().await;
    *g = Some((r.clone(), now_ms()));
    drop(g);
    let mut map = DEDUP.lock().unwrap();
    map.retain(|_, s| s.try_lock().map(|x| x.as_ref().map(|(_, at)| now_ms() - at < 10_000).unwrap_or(true)).unwrap_or(true));
    r
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

async fn post(creds: &Value, url: &str, ct: &str, body: String, extra: &[(String, String)]) -> Result<(u16, String), String> {
    let mut rb = client_for(creds).post(url).header("Content-Type", ct).header("Accept", "application/json");
    for (k, v) in extra {
        rb = rb.header(k, v);
    }
    let r = rb.body(body).timeout(std::time::Duration::from_secs(30)).send().await.map_err(|e| e.to_string())?;
    let st = r.status().as_u16();
    Ok((st, r.text().await.unwrap_or_default()))
}

fn ok_json(r: Result<(u16, String), String>, what: &str) -> Option<Value> {
    match r {
        Ok((st, body)) if (200..300).contains(&st) => serde_json::from_str(&body).ok(),
        Ok((st, body)) => {
            tracing::warn!("token refresh failed for {what}: HTTP {st} {}", body.chars().take(300).collect::<String>());
            None
        }
        Err(e) => {
            tracing::warn!("token refresh error for {what}: {e}");
            None
        }
    }
}

fn std_tokens(t: &Value, old_refresh: &str) -> Value {
    json!({
        "accessToken": t["access_token"],
        "refreshToken": if truthy(&t["refresh_token"]) { t["refresh_token"].clone() } else { json!(old_refresh) },
        "expiresIn": t["expires_in"],
    })
}

// ---------------------------------------------------------------------------
// provider refreshers
// ---------------------------------------------------------------------------

pub async fn refresh_google_token(refresh_token: &str, client_id: &str, client_secret: &str) -> Option<Value> {
    if refresh_token.is_empty() {
        return None;
    }
    let (rt, cid, cs) = (refresh_token.to_string(), client_id.to_string(), client_secret.to_string());
    dedup(&format!("google:{cid}"), refresh_token, || async move {
        let body = form(&[("grant_type", "refresh_token"), ("refresh_token", &rt), ("client_id", &cid), ("client_secret", &cs)]);
        let t = ok_json(post(&Value::Null, GOOGLE_TOKEN, "application/x-www-form-urlencoded", body, &[]).await, "google")?;
        Some(std_tokens(&t, &rt))
    })
    .await
}

fn kimi_extra(creds: &Value) -> Vec<(String, String)> {
    crate::exec::kimi_headers(creds["providerSpecificData"]["deviceId"].as_str())
}

/// refreshAccessToken(provider, ...) — generic OAuth2 refresh with per-provider profiles.
pub async fn refresh_access_token(provider: &str, refresh_token: &str, creds: &Value) -> Option<Value> {
    let cfg = REG.transport(provider).clone();
    let oauth = REG.oauth(provider).clone();
    let url = match provider {
        "claude" => crate::consts::C["OAUTH_ENDPOINTS"]["anthropic"]["token"].as_str().map(str::to_owned),
        "iflow" => crate::consts::C["OAUTH_ENDPOINTS"]["iflow"]["token"].as_str().map(str::to_owned),
        "github" => crate::consts::C["OAUTH_ENDPOINTS"]["github"]["token"].as_str().map(str::to_owned),
        _ => None,
    }
    .or_else(|| cfg["refreshUrl"].as_str().map(str::to_owned))
    .or_else(|| oauth["tokenUrl"].as_str().map(str::to_owned));
    let Some(url) = url.filter(|_| cfg.is_object()) else {
        tracing::warn!("No refresh URL configured for provider: {provider}");
        return None;
    };
    if refresh_token.is_empty() {
        return None;
    }
    let client_id = cfg["clientId"].as_str().or_else(|| oauth["clientId"].as_str()).unwrap_or("").to_string();
    let client_secret = cfg["clientSecret"].as_str().or_else(|| oauth["clientSecret"].as_str()).unwrap_or("").to_string();
    let json_body = provider == "claude";
    let include_secret = match provider {
        "claude" => false,
        "github" => !client_secret.is_empty(),
        _ => true,
    };
    let mut extra: Vec<(String, String)> = vec![];
    if provider == "iflow" {
        extra.push(("Authorization".into(), format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{client_id}:{client_secret}")))));
    }
    if provider == "kimi" || provider == "kimi-coding" {
        extra.extend(kimi_extra(creds));
    }
    let rt = refresh_token.to_string();
    let p = provider.to_string();
    let creds = creds.clone();
    dedup(provider, refresh_token, || async move {
        let mut payload = vec![("grant_type", "refresh_token".to_string()), ("refresh_token", rt.clone()), ("client_id", client_id.clone())];
        if include_secret && !client_secret.is_empty() {
            payload.push(("client_secret", client_secret.clone()));
        }
        let (ct, body) = if json_body {
            let mut o = serde_json::Map::new();
            for (k, v) in &payload {
                o.insert(k.to_string(), json!(v));
            }
            ("application/json", Value::Object(o).to_string())
        } else {
            ("application/x-www-form-urlencoded", form(&payload.iter().map(|(k, v)| (*k, v.as_str())).collect::<Vec<_>>()))
        };
        let t = ok_json(post(&creds, &url, ct, body, &extra).await, &p)?;
        Some(std_tokens(&t, &rt))
    })
    .await
}

pub async fn refresh_cline(refresh_token: &str) -> Option<Value> {
    let url = REG.transport("cline")["refreshUrl"].as_str().or_else(|| REG.oauth("cline")["refreshUrl"].as_str())?.to_string();
    let rt = refresh_token.to_string();
    dedup("cline", refresh_token, || async move {
        let body = json!({"refreshToken": rt, "grantType": "refresh_token", "clientType": "extension"}).to_string();
        let b = ok_json(post(&Value::Null, &url, "application/json", body, &[]).await, "cline")?;
        let t = if truthy(&b["data"]) { b["data"].clone() } else { b };
        let at = t["accessToken"].as_str()?.to_string();
        let expires_in = match t["expiresAt"].as_str().and_then(parse_iso_ms) {
            Some(ms) => ((ms - now_ms()) / 1000).max(1),
            None => t["expiresIn"].as_i64().or_else(|| t["expires_in"].as_i64()).unwrap_or(3600),
        };
        Some(json!({"accessToken": at, "refreshToken": if truthy(&t["refreshToken"]) { t["refreshToken"].clone() } else { json!(rt) }, "expiresIn": expires_in}))
    })
    .await
}

fn classify_permanent(text: &str) -> (bool, String) {
    let p: Value = serde_json::from_str(text).unwrap_or(Value::Null);
    let code = [&p["error"]["code"], &p["error"], &p["error_code"]].into_iter().find(|v| v.is_string()).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let desc = p["error_description"].as_str().or_else(|| p["message"].as_str()).unwrap_or(text).to_string();
    let c = format!("{code} {desc}").to_lowercase();
    (["refresh_token_expired", "refresh_token_reused", "refresh_token_invalidated", "invalid_grant"].iter().any(|m| c.contains(m)), code)
}

pub async fn refresh_codex(refresh_token: &str) -> Option<Value> {
    let rt = refresh_token.to_string();
    dedup("codex", refresh_token, || async move {
        let url = crate::consts::C["OAUTH_ENDPOINTS"]["openai"]["token"].as_str().unwrap_or("https://auth.openai.com/oauth/token").to_string();
        let cid = REG.transport("codex")["clientId"].as_str().or_else(|| REG.oauth("codex")["clientId"].as_str()).unwrap_or("").to_string();
        let body = json!({"client_id": cid, "grant_type": "refresh_token", "refresh_token": rt}).to_string();
        match post(&Value::Null, &url, "application/json", body, &[]).await {
            Ok((st, text)) if (200..300).contains(&st) => {
                let t: Value = serde_json::from_str(&text).ok()?;
                Some(json!({"accessToken": t["access_token"], "refreshToken": if truthy(&t["refresh_token"]) { t["refresh_token"].clone() } else { json!(rt) }, "idToken": t["id_token"], "expiresIn": t["expires_in"]}))
            }
            Ok((st, text)) => {
                let (perm, code) = classify_permanent(&text);
                if perm {
                    tracing::error!("Codex refresh token already used or invalid. Re-auth required. ({st} {code})");
                    return Some(json!({"error": "unrecoverable_refresh_error", "code": code}));
                }
                tracing::warn!("Failed to refresh Codex token: {st} {text}");
                None
            }
            Err(e) => {
                tracing::warn!("Network error refreshing Codex token: {e}");
                None
            }
        }
    })
    .await
}

pub async fn fetch_kiro_profile_arn(access_token: &str) -> Option<String> {
    if access_token.is_empty() {
        return None;
    }
    let r = client_for(&Value::Null)
        .post("https://codewhisperer.us-east-1.amazonaws.com/ListAvailableProfiles")
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .header("Authorization", format!("Bearer {access_token}"))
        .body(json!({"maxResults": 10}).to_string())
        .send()
        .await
        .ok()?;
    if !r.status().is_success() {
        return None;
    }
    let d: Value = r.json().await.ok()?;
    d["profiles"].as_array()?.iter().find_map(|p| p["arn"].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned))
}

async fn kiro_arn_patch(psd: &Value, access: &str, refreshed: &Value) -> Option<Value> {
    if truthy(&psd["profileArn"]) {
        return None;
    }
    let arn = match refreshed.as_str().map(str::trim).filter(|s| !s.is_empty()) {
        Some(a) => Some(a.to_string()),
        None => fetch_kiro_profile_arn(access).await,
    }?;
    Some(json!({"profileArn": arn}))
}

pub async fn refresh_kiro(refresh_token: &str, psd: &Value, creds: &Value) -> Option<Value> {
    let rt = refresh_token.to_string();
    let psd = psd.clone();
    let creds = creds.clone();
    dedup("kiro", refresh_token, || async move {
        let am = psd["authMethod"].as_str().unwrap_or("");
        if am == "external_idp" {
            let cid = psd["clientId"].as_str().or_else(|| psd["client_id"].as_str()).unwrap_or("").trim().to_string();
            let ep = psd["tokenEndpoint"].as_str().or_else(|| psd["token_endpoint"].as_str()).unwrap_or("").trim().to_string();
            let scope = match if !psd["scope"].is_null() { &psd["scope"] } else { &psd["scopes"] } {
                Value::Array(a) => a.iter().filter_map(|x| x.as_str()).map(str::trim).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" "),
                Value::String(s) => s.trim().to_string(),
                _ => String::new(),
            };
            let host_ok = reqwest::Url::parse(&ep).ok().filter(|u| u.scheme() == "https").and_then(|u| u.host_str().map(str::to_lowercase)).map(|h| ["login.microsoftonline.com", "login.microsoft.com", "login.windows.net"].contains(&h.as_str())).unwrap_or(false);
            if cid.is_empty() || scope.is_empty() || !host_ok {
                tracing::warn!("Invalid Kiro external_idp refresh config");
                return None;
            }
            let body = form(&[("grant_type", "refresh_token"), ("client_id", &cid), ("refresh_token", &rt), ("scope", &scope)]);
            let t = ok_json(post(&creds, &ep, "application/x-www-form-urlencoded", body, &[]).await, "kiro external_idp")?;
            let mut np = psd.clone();
            np["authMethod"] = json!("external_idp");
            np["clientId"] = json!(cid);
            np["tokenEndpoint"] = json!(ep);
            np["scope"] = json!(scope);
            let mut out = std_tokens(&t, &rt);
            out["providerSpecificData"] = np;
            return Some(out);
        }
        let (cid, cs) = (psd["clientId"].as_str().unwrap_or(""), psd["clientSecret"].as_str().unwrap_or(""));
        let t = if !cid.is_empty() && !cs.is_empty() {
            let region = psd["region"].as_str().filter(|s| !s.is_empty());
            let ep = match (am == "idc", region) {
                (true, Some(r)) => format!("https://oidc.{r}.amazonaws.com/token"),
                _ => "https://oidc.us-east-1.amazonaws.com/token".into(),
            };
            ok_json(post(&creds, &ep, "application/json", json!({"clientId": cid, "clientSecret": cs, "refreshToken": rt, "grantType": "refresh_token"}).to_string(), &[]).await, "kiro aws")?
        } else {
            let url = REG.transport("kiro")["tokenUrl"].as_str().unwrap_or("https://prod.us-east-1.auth.desktop.kiro.dev/refreshToken").to_string();
            ok_json(post(&creds, &url, "application/json", json!({"refreshToken": rt}).to_string(), &[("User-Agent".into(), "kiro-cli/1.0.0".into())]).await, "kiro social")?
        };
        let access = t["accessToken"].as_str().unwrap_or("").to_string();
        let mut out = json!({"accessToken": access, "refreshToken": if truthy(&t["refreshToken"]) { t["refreshToken"].clone() } else { json!(rt) }, "expiresIn": t["expiresIn"]});
        if let Some(p) = kiro_arn_patch(&psd, &access, &t["profileArn"]).await {
            out["providerSpecificData"] = p;
        }
        Some(out)
    })
    .await
}

pub async fn refresh_xai(refresh_token: &str) -> Option<Value> {
    let rt = refresh_token.to_string();
    dedup("xai", refresh_token, || async move {
        let cid = REG.transport("xai")["clientId"].as_str().unwrap_or("").to_string();
        let mut token_url = "https://auth.x.ai/oauth2/token".to_string();
        if let Ok(r) = client_for(&Value::Null).get("https://auth.x.ai/.well-known/openid-configuration").header("Accept", "application/json").send().await {
            if let Ok(d) = r.json::<Value>().await {
                if let Some(u) = d["token_endpoint"].as_str().filter(|u| reqwest::Url::parse(u).ok().and_then(|x| x.host_str().map(|h| h == "x.ai" || h.ends_with(".x.ai"))).unwrap_or(false)) {
                    token_url = u.to_string();
                }
            }
        }
        match post(&Value::Null, &token_url, "application/x-www-form-urlencoded", form(&[("grant_type", "refresh_token"), ("client_id", &cid), ("refresh_token", &rt)]), &[]).await {
            Ok((st, text)) if (200..300).contains(&st) => {
                let t: Value = serde_json::from_str(&text).ok()?;
                Some(json!({"accessToken": t["access_token"], "refreshToken": if truthy(&t["refresh_token"]) { t["refresh_token"].clone() } else { json!(rt) }, "expiresIn": t["expires_in"], "idToken": t["id_token"]}))
            }
            Ok((_, text)) => {
                if text.contains("invalid_grant") || text.contains("invalid_request") {
                    return Some(json!({"error": "invalid_grant"}));
                }
                None
            }
            Err(_) => None,
        }
    })
    .await
}

async fn refresh_codebuddy(id: &str, domain: &str, refresh_token: &str) -> Option<Value> {
    let oauth = REG.oauth(id).clone();
    let url = oauth["refreshUrl"].as_str()?.to_string();
    let ua = oauth["userAgent"].as_str().unwrap_or("").to_string();
    let rt = refresh_token.to_string();
    let domain = domain.to_string();
    dedup(id, refresh_token, || async move {
        let extra = vec![
            ("User-Agent".to_string(), ua),
            ("X-Requested-With".into(), "XMLHttpRequest".into()),
            ("X-Domain".into(), domain),
            ("X-Refresh-Token".into(), rt.clone()),
            ("X-Auth-Refresh-Source".into(), "plugin".into()),
            ("X-Product".into(), "SaaS".into()),
        ];
        let d = ok_json(post(&Value::Null, &url, "application/json", "{}".into(), &extra).await, "codebuddy")?;
        if d["code"] != json!(0) || !truthy(&d["data"]["accessToken"]) {
            return None;
        }
        let x = &d["data"];
        Some(json!({"accessToken": x["accessToken"], "refreshToken": if truthy(&x["refreshToken"]) { x["refreshToken"].clone() } else { json!(rt) }, "expiresIn": x["expiresIn"]}))
    })
    .await
}

// ---------------------------------------------------------------------------
// Vertex service-account JWT
// ---------------------------------------------------------------------------

pub fn parse_vertex_sa_json(api_key: &Value) -> Option<Value> {
    let p: Value = serde_json::from_str(api_key.as_str()?).ok()?;
    (p["type"] == "service_account" && truthy(&p["client_email"]) && truthy(&p["private_key"]) && truthy(&p["project_id"])).then_some(p)
}

static VERTEX_CACHE: LazyLock<Mutex<HashMap<String, (String, i64)>>> = LazyLock::new(Default::default);

pub fn sign_rs256_jwt(private_key_pem: &str, claims: &Value) -> Result<String, String> {
    use rsa::pkcs1v15::SigningKey;
    use rsa::pkcs8::DecodePrivateKey;
    use rsa::signature::{SignatureEncoding, Signer};
    let key = rsa::RsaPrivateKey::from_pkcs8_pem(private_key_pem).or_else(|_| {
        use rsa::pkcs1::DecodeRsaPrivateKey;
        rsa::RsaPrivateKey::from_pkcs1_pem(private_key_pem)
    });
    let key = key.map_err(|e| e.to_string())?;
    let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let header = b64.encode(json!({"alg": "RS256"}).to_string());
    let payload = b64.encode(claims.to_string());
    let input = format!("{header}.{payload}");
    let sk = SigningKey::<sha2_010::Sha256>::new(key);
    let sig = sk.sign(input.as_bytes());
    Ok(format!("{input}.{}", b64.encode(sig.to_bytes())))
}

/// refreshVertexToken(saJson) → (accessToken, expiresAtMs)
pub async fn refresh_vertex_token(sa: &Value) -> Option<(String, i64)> {
    let email = sa["client_email"].as_str()?.to_string();
    if let Some((t, exp)) = VERTEX_CACHE.lock().unwrap().get(&email).cloned() {
        if exp - now_ms() > 300_000 {
            return Some((t, exp));
        }
    }
    let now = now_ms() / 1000;
    let pk = sa["private_key"].as_str()?.replace("\\n", "\n");
    let jwt = match sign_rs256_jwt(&pk, &json!({"scope": "https://www.googleapis.com/auth/cloud-platform", "iss": email, "aud": GOOGLE_TOKEN, "iat": now, "exp": now + 3600})) {
        Ok(j) => j,
        Err(e) => {
            tracing::error!("Vertex token error: {e}");
            return None;
        }
    };
    let body = form(&[("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"), ("assertion", &jwt)]);
    let t = ok_json(post(&Value::Null, GOOGLE_TOKEN, "application/x-www-form-urlencoded", body, &[]).await, "vertex")?;
    let tok = t["access_token"].as_str()?.to_string();
    let exp = now_ms() + t["expires_in"].as_i64().unwrap_or(3600) * 1000;
    VERTEX_CACHE.lock().unwrap().insert(email, (tok.clone(), exp));
    Some((tok, exp))
}

// ---------------------------------------------------------------------------
// GitHub Copilot
// ---------------------------------------------------------------------------

pub async fn copilot_token(github_token: &str) -> Option<Value> {
    let url = REG.oauth("github")["copilotTokenUrl"].as_str().unwrap_or("https://api.github.com/copilot_internal/v2/token").to_string();
    let (vscode, chat, ua, api) = crate::providers::github::copilot_constants();
    let gt = github_token.to_string();
    dedup("copilot", github_token, || async move {
        let r = client_for(&Value::Null)
            .get(&url)
            .header("Authorization", format!("token {gt}"))
            .header("User-Agent", ua)
            .header("Editor-Version", format!("vscode/{vscode}"))
            .header("Editor-Plugin-Version", format!("copilot-chat/{chat}"))
            .header("Accept", "application/json")
            .header("x-github-api-version", api)
            .send()
            .await
            .ok()?;
        if !r.status().is_success() {
            tracing::warn!("Failed to refresh Copilot token: {}", r.status());
            return None;
        }
        let d: Value = r.json().await.ok()?;
        Some(json!({"token": d["token"], "expiresAt": d["expires_at"]}))
    })
    .await
}

/// GithubExecutor.refreshCredentials
pub async fn refresh_github_copilot(creds: &Value) -> Option<Value> {
    let mut cp = copilot_token(creds["accessToken"].as_str().unwrap_or("")).await;
    if cp.is_none() {
        if let Some(rt) = creds["refreshToken"].as_str().filter(|s| !s.is_empty()) {
            if let Some(gh) = refresh_access_token("github", rt, creds).await {
                if let Some(at) = gh["accessToken"].as_str() {
                    cp = copilot_token(at).await;
                    let mut out = gh.clone();
                    if let Some(c) = cp {
                        out["copilotToken"] = c["token"].clone();
                        out["copilotTokenExpiresAt"] = c["expiresAt"].clone();
                    }
                    return Some(out);
                }
            }
        }
    }
    let c = cp?;
    Some(json!({"accessToken": creds["accessToken"], "refreshToken": creds["refreshToken"], "copilotToken": c["token"], "copilotTokenExpiresAt": c["expiresAt"]}))
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

fn transport_secret(id: &str) -> (String, String) {
    let t = REG.transport(id);
    (t["clientId"].as_str().unwrap_or("").to_string(), t["clientSecret"].as_str().unwrap_or("").to_string())
}

/// refreshTokenByProvider + executor overrides → raw refresh result.
pub async fn refresh_for_provider(provider: &str, creds: &Value) -> Option<Value> {
    let rt = creds["refreshToken"].as_str().unwrap_or("").to_string();
    match provider {
        "vertex" | "vertex-partner" => {
            let sa = parse_vertex_sa_json(&creds["apiKey"])?;
            let (t, exp) = refresh_vertex_token(&sa).await?;
            return Some(json!({"accessToken": t, "expiresAt": iso_from_ms(exp)}));
        }
        "github" => return refresh_github_copilot(creds).await,
        "zed" | "cursor" | "qoder" | "qoder-cn" | "kilocode" | "windsurf" => return None,
        _ => {}
    }
    if rt.is_empty() {
        return None;
    }
    let mut out = match provider {
        "gemini-cli" | "antigravity" | "gemini" => {
            let (cid, cs) = transport_secret(provider);
            refresh_google_token(&rt, &cid, &cs).await
        }
        "claude" => refresh_access_token("claude", &rt, creds).await,
        "codex" => refresh_codex(&rt).await,
        "iflow" => refresh_access_token("iflow", &rt, creds).await,
        "kiro" => refresh_kiro(&rt, &creds["providerSpecificData"], creds).await,
        "xai" | "grok-cli" | "gcli" => refresh_xai(&rt).await,
        "codebuddy-cn" => refresh_codebuddy("codebuddy-cn", "copilot.tencent.com", &rt).await,
        "codebuddy-intl" => refresh_codebuddy("codebuddy-intl", "www.codebuddy.ai", &rt).await,
        "cline" | "clinepass" => refresh_cline(&rt).await,
        "gitlab" => refresh_gitlab(&rt, creds).await,
        "kimi" | "kimi-coding" => refresh_access_token("kimi", &rt, creds).await,
        other => refresh_access_token(other, &rt, creds).await,
    }?;
    if matches!(provider, "gemini-cli" | "antigravity") && truthy(&creds["projectId"]) && !truthy(&out["projectId"]) {
        out["projectId"] = creds["projectId"].clone();
    }
    Some(out)
}

/// GitLab OAuth (self-managed instances too): the token endpoint, client and
/// PKCE verifier come from the connection created at sign-in.
async fn refresh_gitlab(rt: &str, creds: &Value) -> Option<Value> {
    let psd = &creds["providerSpecificData"];
    if psd["authKind"] != "oauth" {
        return None;
    }
    let base = psd["baseUrl"].as_str().filter(|s| !s.is_empty()).unwrap_or("https://gitlab.com").trim_end_matches('/').to_string();
    let mut pairs: Vec<(&str, String)> = vec![("grant_type", "refresh_token".into()), ("refresh_token", rt.into()), ("client_id", psd["clientId"].as_str().unwrap_or("").into())];
    for (k, f) in [("client_secret", "clientSecret"), ("redirect_uri", "redirectUri"), ("code_verifier", "codeVerifier")] {
        if let Some(v) = psd[f].as_str().filter(|s| !s.is_empty()) {
            pairs.push((k, v.into()));
        }
    }
    let body = pairs.iter().map(|(k, v)| format!("{k}={}", crate::oauth::enc(v))).collect::<Vec<_>>().join("&");
    let url = format!("{base}/oauth/token");
    let creds = creds.clone();
    dedup("gitlab", rt, || async move {
        let r = post(&creds, &url, "application/x-www-form-urlencoded", body, &[]).await;
        if let Ok((st, b)) = &r {
            if (400..500).contains(st) && b.contains("invalid_grant") {
                return Some(json!({"error": "invalid_grant"}));
            }
        }
        let v = ok_json(r, "gitlab")?;
        Some(json!({"accessToken": v["access_token"].as_str()?, "refreshToken": v["refresh_token"], "expiresIn": v["expires_in"]}))
    })
    .await
}

pub fn is_unrecoverable(r: &Value) -> bool {
    matches!(r["error"].as_str(), Some("unrecoverable_refresh_error") | Some("refresh_token_reused") | Some("invalid_request") | Some("invalid_grant"))
}

/// mergeRefreshedCredentials → patch to apply to the stored connection.
pub fn merge_refreshed(provider: &str, current: &Value, r: &Value) -> Value {
    if is_unrecoverable(r) {
        return r.clone();
    }
    let mut next = json!({});
    for k in ["accessToken", "apiKey", "token"] {
        if truthy(&r[k]) {
            next[k] = r[k].clone();
        }
    }
    let rt = if !r["refreshToken"].is_null() { &r["refreshToken"] } else { &current["refreshToken"] };
    if truthy(rt) {
        next["refreshToken"] = rt.clone();
    }
    let idt = if !r["idToken"].is_null() { &r["idToken"] } else { &current["idToken"] };
    if truthy(idt) {
        next["idToken"] = idt.clone();
    }
    if let Some(n) = r["expiresIn"].as_f64().filter(|n| *n > 0.0) {
        next["expiresIn"] = r["expiresIn"].clone();
        next["expiresAt"] = json!(iso_from_ms(now_ms() + (n * 1000.0) as i64));
    } else if truthy(&r["expiresAt"]) {
        next["expiresAt"] = r["expiresAt"].clone();
    }
    if truthy(&r["projectId"]) {
        next["projectId"] = r["projectId"].clone();
    }
    if r["providerSpecificData"].is_object() {
        let mut p = if current["providerSpecificData"].is_object() { current["providerSpecificData"].clone() } else { json!({}) };
        for (k, v) in r["providerSpecificData"].as_object().unwrap() {
            p[k] = v.clone();
        }
        next["providerSpecificData"] = p;
    }
    for k in ["copilotToken", "copilotTokenExpiresAt"] {
        if truthy(&r[k]) {
            next[k] = r[k].clone();
        }
    }
    let track = REG.oauth(provider)["trackRefreshAt"] == json!(true);
    if track || ["accessToken", "apiKey", "token", "refreshToken", "copilotToken"].iter().any(|k| truthy(&next[*k])) {
        next["lastRefreshAt"] = if truthy(&r["lastRefreshAt"]) { r["lastRefreshAt"].clone() } else { json!(iso_from_ms(now_ms())) };
    }
    next
}

/// Applies a refresh patch to in-flight credentials.
pub fn merge_into(creds: &mut Value, patch: &Value) {
    let Some(o) = patch.as_object() else { return };
    for (k, v) in o {
        if k == "providerSpecificData" && v.is_object() {
            if !creds["providerSpecificData"].is_object() {
                creds["providerSpecificData"] = json!({});
            }
            for (a, b) in v.as_object().unwrap() {
                creds["providerSpecificData"][a] = b.clone();
            }
        } else if !v.is_null() {
            creds[k] = v.clone();
        }
    }
    if let Some(exp) = patch["expiresIn"].as_f64() {
        if !truthy(&patch["expiresAt"]) && exp > 0.0 {
            creds["expiresAt"] = json!(iso_from_ms(now_ms() + (exp * 1000.0) as i64));
        }
    }
}

fn time_ms(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_f64().map(|f| if f < 1e12 { (f * 1000.0) as i64 } else { f as i64 }),
        Value::String(s) if !s.is_empty() => parse_iso_ms(s),
        _ => None,
    }
}

pub fn refresh_lead_ms(provider: &str) -> i64 {
    let c = &crate::consts::C["REFRESH_LEAD_MS"];
    c[provider].as_i64().or_else(|| if provider == "kimi-coding" { c["kimi"].as_i64() } else { None }).or_else(|| REG.oauth(provider)["refreshLeadMs"].as_i64()).unwrap_or(5 * 60 * 1000)
}

/// shouldRefreshCredentials(provider, credentials)
pub fn should_refresh(provider: &str, creds: &Value) -> bool {
    let exp = time_ms(if !creds["expiresAt"].is_null() { &creds["expiresAt"] } else { &creds["tokenExpiresAt"] });
    if let Some(e) = exp {
        if e - now_ms() < refresh_lead_ms(provider) {
            return true;
        }
    }
    if let Some(max_age) = REG.oauth(provider)["maxRefreshAgeMs"].as_i64() {
        if truthy(&creds["refreshToken"]) {
            let last = time_ms([&creds["lastRefreshAt"], &creds["lastRefresh"], &creds["providerSpecificData"]["lastRefreshAt"]].into_iter().find(|v| !v.is_null()).unwrap_or(&Value::Null));
            return last.map(|l| now_ms() - l >= max_age).unwrap_or(true);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging() {
        let cur = json!({"refreshToken": "old", "providerSpecificData": {"a": 1}});
        let p = merge_refreshed("codex", &cur, &json!({"accessToken": "new", "expiresIn": 3600, "providerSpecificData": {"b": 2}}));
        assert_eq!(p["refreshToken"], "old");
        assert_eq!(p["providerSpecificData"], json!({"a": 1, "b": 2}));
        assert!(p["expiresAt"].is_string() && p["lastRefreshAt"].is_string());
        let mut c = cur.clone();
        merge_into(&mut c, &p);
        assert_eq!(c["accessToken"], "new");
        assert!(should_refresh("codex", &json!({"expiresAt": iso_from_ms(now_ms() + 1000), "refreshToken": "x"})));
        assert!(!should_refresh("claude", &json!({"expiresAt": iso_from_ms(now_ms() + 10 * 3_600_000)})));
        assert!(is_unrecoverable(&json!({"error": "invalid_grant"})));
    }

    #[test]
    fn jwt_signing() {
        use rsa::pkcs8::EncodePrivateKey;
        let mut rng = rsa::rand_core::OsRng;
        let k = rsa::RsaPrivateKey::new(&mut rng, 1024).unwrap();
        let pem = k.to_pkcs8_pem(rsa::pkcs8::LineEnding::LF).unwrap();
        let jwt = sign_rs256_jwt(&pem, &json!({"iss": "a"})).unwrap();
        assert_eq!(jwt.split('.').count(), 3);
        assert_eq!(crate::oauth::decode_jwt_payload(&jwt).unwrap()["iss"], "a");
    }
}
