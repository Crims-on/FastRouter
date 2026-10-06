//! Qoder (qoder / qoder-cn) executor — COSY-signed agent_chat_generation.
//! Port of executors/qoder.js + shared/qoder/* + services/qoderModels.js.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use aes::cipher::{BlockEncryptMut, KeyIvInit, block_padding::Pkcs7};
use async_trait::async_trait;
use base64::Engine;
use bytes::Bytes;
use futures::StreamExt;
use md5::{Digest as _, Md5};
use sha2::Digest as _;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::exec::{ByteStream, ExecArgs, ExecResult, Executor, Headers, Upstream, client_for};
use crate::jsv::{js_string, now_ms, truthy};

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

pub const IDE_VERSION: &str = "1.0.0";
pub const CLIENT_TYPE: &str = "5";
const DATA_POLICY: &str = "disagree";
const LOGIN_VERSION: &str = "v2";
const MACHINE_OS: &str = "x86_64_windows";
const MACHINE_TYPE: &str = "5";
const CHAT_SIG_PATH: &str = "/api/v2/service/pro/sse/agent_chat_generation";
const IMAGE_UPLOAD_SIG_PATH: &str = "/api/v2/image/upload";
const MAX_PAYLOAD_BYTES: usize = 6 * 1024 * 1024;
const INLINE_FALLBACK_MAX_BYTES: usize = 512 * 1024;
const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const TIER_HEADROOM: f64 = 0.15;
const RSA_PUBLIC_KEY: &str = "-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDA8iMH5c02LilrsERw9t6Pv5Nc
4k6Pz1EaDicBMpdpxKduSZu5OANqUq8er4GM95omAGIOPOh+Nx0spthYA2BqGz+l
6HRkPJ7S236FZz73In/KVuLnwI8JJ2CbuJap8kvheCCZpmAWpb/cPx/3Vr/J6I17
XcW+ML9FoCI6AOvOzwIDAQAB
-----END PUBLIC KEY-----";

pub struct RegionBases {
    pub chat: &'static str,
    pub chat_alt: &'static str,
    pub open_api: &'static str,
    pub center: &'static str,
    pub login: &'static str,
    pub website: &'static str,
}

pub fn region_bases(region: &str) -> RegionBases {
    if region == "cn" {
        RegionBases {
            chat: "https://gateway.qoder.com.cn",
            chat_alt: "https://gateway.qoder.com.cn",
            open_api: "https://openapi.qoder.com.cn",
            center: "https://gateway.qoder.com.cn",
            login: "https://qoder.com.cn/device/selectAccounts",
            website: "https://qoder.com.cn",
        }
    } else {
        RegionBases {
            chat: "https://api3.qoder.sh",
            chat_alt: "https://api2.qoder.sh",
            open_api: "https://openapi.qoder.sh",
            center: "https://center.qoder.sh",
            login: "https://qoder.com/device/selectAccounts",
            website: "https://qoder.com",
        }
    }
}

pub fn region_of(provider: &str) -> &'static str {
    if provider == "qoder-cn" { "cn" } else { "intl" }
}

pub fn inference_base(creds: &Value, region: &str) -> &'static str {
    let b = region_bases(region);
    if region == "cn" {
        return b.chat;
    }
    let raw = crate::exec::cred_str(creds, "apiKey").or_else(|| crate::exec::cred_str(creds, "accessToken")).unwrap_or("");
    if !raw.starts_with("pt-") && (raw.starts_with("jt-") || creds["accessToken"].as_str().unwrap_or("").starts_with("jt-")) {
        return b.chat_alt;
    }
    b.chat
}

// ---------------------------------------------------------------------------
// body encoding + COSY signing
// ---------------------------------------------------------------------------

/// qoderEncodeBody: base64 → [tail][mid][head] → alphabet substitution.
pub fn encode_body(plain: &[u8]) -> Vec<u8> {
    const STD: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const CUSTOM: &[u8] = b"_doRTgHZBKcGVjlvpC,@aFSx#DPuNJme&i*MzLOEn)sUrthbf%Y^w.(kIQyXqWA!";
    let mut table = [0u8; 128];
    for i in 0..64 {
        table[STD[i] as usize] = CUSTOM[i];
    }
    table[b'=' as usize] = b'$';
    let std = B64.encode(plain).into_bytes();
    let n = std.len();
    let a = n / 3;
    let mut re = Vec::with_capacity(n);
    re.extend_from_slice(&std[n - a..]);
    re.extend_from_slice(&std[a..n - a]);
    re.extend_from_slice(&std[..a]);
    re.into_iter().map(|c| if c < 128 && table[c as usize] != 0 { table[c as usize] } else { c }).collect()
}

fn md5_hex(b: &[u8]) -> String {
    hex::encode(Md5::digest(b))
}

fn rsa_encrypt_b64(data: &[u8]) -> Result<String, String> {
    use rsa::pkcs8::DecodePublicKey;
    let key = rsa::RsaPublicKey::from_public_key_pem(RSA_PUBLIC_KEY).map_err(|e| e.to_string())?;
    let mut rng = rsa::rand_core::OsRng;
    let enc = key.encrypt(&mut rng, rsa::Pkcs1v15Encrypt, data).map_err(|e| e.to_string())?;
    Ok(B64.encode(enc))
}

fn aes_cbc_b64(plain: &str, key: &str) -> String {
    let k = key.as_bytes();
    let enc = cbc::Encryptor::<aes::Aes128>::new_from_slices(k, k).expect("16-byte key");
    B64.encode(enc.encrypt_padded_vec_mut::<Pkcs7>(plain.as_bytes()))
}

fn sig_path(url: &str) -> String {
    let p = reqwest::Url::parse(url).map(|u| u.path().to_string()).unwrap_or_default();
    p.strip_prefix("/algo").map(str::to_owned).unwrap_or(p)
}

pub struct CosyCreds {
    pub user_id: String,
    pub auth_token: String,
    pub name: String,
    pub email: String,
    pub machine_id: String,
}

pub fn cosy_creds(creds: &Value) -> CosyCreds {
    let psd = &creds["providerSpecificData"];
    CosyCreds {
        user_id: js_or_empty(&psd["userId"]),
        auth_token: js_or_empty(&creds["accessToken"]),
        name: js_or_empty(&creds["displayName"]),
        email: js_or_empty(&creds["email"]),
        machine_id: js_or_empty(&psd["machineId"]),
    }
}

fn js_or_empty(v: &Value) -> String {
    if v.is_null() { String::new() } else { js_string(v) }
}

/// buildCosyHeaders(body, url, creds)
pub fn cosy_headers(body: &[u8], url: &str, c: &CosyCreds) -> Result<Headers, String> {
    if c.user_id.is_empty() {
        return Err("cosy: user id is empty".into());
    }
    if c.auth_token.is_empty() {
        return Err("cosy: auth token is empty".into());
    }
    let aes_key: String = uuid::Uuid::new_v4().to_string().chars().take(16).collect();
    let info = aes_cbc_b64(&json!({"uid": c.user_id, "security_oauth_token": c.auth_token, "name": c.name, "aid": "", "email": c.email}).to_string(), &aes_key);
    let cosy_key = rsa_encrypt_b64(aes_key.as_bytes())?;
    let ts = (now_ms() / 1000).to_string();
    let payload = json!({"version": "v1", "requestId": uuid::Uuid::new_v4().to_string(), "info": info, "cosyVersion": IDE_VERSION, "ideVersion": ""}).to_string();
    let payload_b64 = B64.encode(payload);
    let sp = sig_path(url);
    let mut sig_input: Vec<u8> = format!("{payload_b64}\n{cosy_key}\n{ts}\n").into_bytes();
    sig_input.extend_from_slice(body);
    sig_input.extend_from_slice(format!("\n{sp}").as_bytes());
    let sig = md5_hex(&sig_input);
    let machine = if c.machine_id.is_empty() { uuid::Uuid::new_v4().to_string() } else { c.machine_id.clone() };
    let mut h = Headers::default();
    h.set("Authorization", format!("Bearer COSY.{payload_b64}.{sig}"));
    h.set("Cosy-Key", cosy_key);
    h.set("Cosy-User", c.user_id.clone());
    h.set("Cosy-Date", ts);
    h.set("Cosy-Version", IDE_VERSION);
    h.set("Cosy-Machineid", machine.clone());
    h.set("Cosy-Machinetoken", machine);
    h.set("Cosy-Machinetype", MACHINE_TYPE);
    h.set("Cosy-Machineos", MACHINE_OS);
    h.set("Cosy-Clienttype", CLIENT_TYPE);
    h.set("Cosy-Clientip", "127.0.0.1");
    h.set("Cosy-Bodyhash", md5_hex(body));
    h.set("Cosy-Bodylength", body.len().to_string());
    h.set("Cosy-Sigpath", sp);
    h.set("Cosy-Data-Policy", DATA_POLICY);
    h.set("Cosy-Organization-Id", "");
    h.set("Cosy-Organization-Tags", "");
    h.set("Login-Version", LOGIN_VERSION);
    h.set("X-Request-Id", uuid::Uuid::new_v4().to_string());
    Ok(h)
}

// ---------------------------------------------------------------------------
// PAT exchange + model catalog
// ---------------------------------------------------------------------------

pub fn is_pat(t: &str) -> bool {
    t.starts_with("pt-")
}

static PAT_CACHE: LazyLock<Mutex<HashMap<String, (String, String, i64)>>> = LazyLock::new(Default::default);
static CATALOG: LazyLock<Mutex<HashMap<String, (Value, i64)>>> = LazyLock::new(Default::default);

async fn exchange_job_token(creds: &Value, pat: &str, region: &str) -> Result<(String, i64), String> {
    let url = format!("{}/api/v1/jobToken/exchange", region_bases(region).open_api);
    let r = client_for(creds)
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .header("User-Agent", "qodercli/1.0.0")
        .header("Cosy-Version", IDE_VERSION)
        .header("Cosy-ClientType", CLIENT_TYPE)
        .body(json!({"personal_token": pat}).to_string())
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let st = r.status().as_u16();
    let text = r.text().await.unwrap_or_default();
    if !(200..300).contains(&st) {
        return Err(format!("qoder PAT exchange failed: {st} {}", text.chars().take(200).collect::<String>()));
    }
    let d: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let tok = d["token"].as_str().filter(|s| !s.is_empty()).ok_or("qoder PAT exchange returned no job token")?.to_string();
    let mut exp = now_ms() + 24 * 3_600_000;
    if let Some(s) = d["expires_at"].as_str() {
        if let Some(ms) = crate::jsv::parse_iso_ms(s) {
            exp = ms;
        }
    } else if let Some(n) = d["expires_in"].as_i64().filter(|n| *n > 0) {
        exp = now_ms() + n;
    }
    Ok((tok, exp))
}

async fn user_id_for_job_token(creds: &Value, tok: &str, region: &str) -> String {
    let url = format!("{}/api/v1/userinfo", region_bases(region).open_api);
    let Ok(r) = client_for(creds).get(&url).header("Authorization", format!("Bearer {tok}")).header("Accept", "application/json").header("User-Agent", "qodercli/1.0.0").send().await else { return String::new() };
    if !r.status().is_success() {
        return String::new();
    }
    let d: Value = r.json().await.unwrap_or(Value::Null);
    [&d["id"], &d["userId"], &d["user_id"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or_default()
}

/// resolveQoderCredentials: PAT → job token (+ userId); others unchanged.
pub async fn resolve_credentials(creds: &Value, region: &str) -> Result<Value, String> {
    let raw = crate::exec::cred_str(creds, "apiKey").or_else(|| crate::exec::cred_str(creds, "accessToken")).unwrap_or("").to_string();
    if !is_pat(&raw) {
        return Ok(creds.clone());
    }
    let key = format!("{region}:{raw}");
    let cached = PAT_CACHE.lock().unwrap().get(&key).cloned();
    let (tok, uid) = match cached.filter(|c| c.2 - now_ms() > 300_000) {
        Some(c) => (c.0, c.1),
        None => {
            let (tok, exp) = exchange_job_token(creds, &raw, region).await?;
            let uid = user_id_for_job_token(creds, &tok, region).await;
            PAT_CACHE.lock().unwrap().insert(key, (tok.clone(), uid.clone(), exp));
            (tok, uid)
        }
    };
    let mut out = creds.clone();
    out["accessToken"] = json!(tok);
    crate::jsv::del(&mut out, "apiKey");
    let mut psd = json!({"authMethod": "pat"});
    if let Some(o) = creds["providerSpecificData"].as_object() {
        for (k, v) in o {
            psd[k] = v.clone();
        }
    }
    psd["userId"] = json!(if uid.is_empty() { js_or_empty(&creds["providerSpecificData"]["userId"]) } else { uid });
    psd["machineId"] = json!(js_or_empty(&creds["providerSpecificData"]["machineId"]));
    out["providerSpecificData"] = psd;
    Ok(out)
}

fn catalog_key(creds: &Value, region: &str) -> String {
    let psd = &creds["providerSpecificData"];
    let seed = [&psd["userId"], &creds["refreshToken"], &creds["accessToken"]].into_iter().find(|v| truthy(v)).map(js_string).unwrap_or_else(|| "anonymous".into());
    crate::jsv::sha256_hex(&format!("qoder:{region}:{seed}"))
}

/// resolveQoderModels → {models:[...], rawConfigs:{key: cfg}} or Null.
pub async fn resolve_models(creds: &Value, region: &str, force: bool) -> Value {
    let Ok(resolved) = resolve_credentials(creds, region).await else { return Value::Null };
    let c = cosy_creds(&resolved);
    if c.auth_token.is_empty() || c.user_id.is_empty() {
        return Value::Null;
    }
    let key = catalog_key(&resolved, region);
    if !force {
        if let Some((v, exp)) = CATALOG.lock().unwrap().get(&key).cloned() {
            if exp > now_ms() {
                return v;
            }
        }
    }
    let url = format!("{}/algo/api/v2/model/list", inference_base(&resolved, region));
    let Ok(mut h) = cosy_headers(&[], &url, &c) else { return Value::Null };
    h.set("Accept", "application/json");
    h.set("Accept-Encoding", "identity");
    let mut rb = client_for(creds).get(&url).timeout(std::time::Duration::from_secs(15));
    for (k, v) in &h.0 {
        rb = rb.header(k, v);
    }
    let Ok(r) = rb.send().await else { return Value::Null };
    if !r.status().is_success() {
        return Value::Null;
    }
    let body: Value = r.json().await.unwrap_or(Value::Null);
    let Some(chat) = body["chat"].as_array() else { return Value::Null };
    let mut models = vec![];
    let mut raw = Map::new();
    for e in chat {
        let Some(k) = e["key"].as_str().filter(|s| !s.is_empty()) else { continue };
        raw.insert(k.to_string(), e.clone());
        if e["enable"] == json!(false) {
            continue;
        }
        models.push(json!({
            "id": k,
            "name": e["display_name"].as_str().filter(|s| !s.is_empty()).unwrap_or(k),
            "contextLength": e["max_input_tokens"].as_i64().filter(|n| *n > 0).unwrap_or(131_072),
            "isVL": truthy(&e["is_vl"]),
            "isReasoning": truthy(&e["is_reasoning"]),
            "maxOutputTokens": e["max_output_tokens"].as_i64().unwrap_or(0),
            "description": e["description"].as_str().unwrap_or(""),
        }));
    }
    let entry = json!({"models": models, "rawConfigs": raw});
    CATALOG.lock().unwrap().insert(key, (entry.clone(), now_ms() + 3_600_000));
    entry
}

/// routableQoderModels: visible models first, then hidden configs.
pub fn routable_models(catalog: &Value) -> Vec<Value> {
    let mut out = vec![];
    let mut seen = std::collections::HashSet::new();
    for m in catalog["models"].as_array().into_iter().flatten() {
        let id = js_string(&m["id"]);
        if seen.insert(id.clone()) {
            out.push(json!({"id": id, "name": m["name"], "hidden": false}));
        }
    }
    for (k, cfg) in catalog["rawConfigs"].as_object().into_iter().flatten() {
        if seen.insert(k.clone()) {
            out.push(json!({"id": k, "name": cfg["display_name"].as_str().unwrap_or(k), "hidden": true}));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// attachments
// ---------------------------------------------------------------------------

static DATA_URI_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"data:[^;]+;base64,[A-Za-z0-9+/=\s]+").unwrap());

fn mime_ext(m: &str) -> &'static str {
    let m = m.to_lowercase();
    if m.contains("png") {
        "png"
    } else if m.contains("jpeg") || m.contains("jpg") {
        "jpg"
    } else if m.contains("gif") {
        "gif"
    } else if m.contains("webp") {
        "webp"
    } else if m.contains("bmp") {
        "bmp"
    } else if m.contains("pdf") {
        "pdf"
    } else {
        "bin"
    }
}

fn decoded_bytes(b64: &str) -> usize {
    b64.chars().filter(|c| !c.is_whitespace()).count() * 3 / 4
}

fn stub(name: Option<&str>, mime: Option<&str>, bytes: usize, reason: &str) -> String {
    let label = name.filter(|s| !s.is_empty()).or(mime.filter(|s| !s.is_empty())).unwrap_or("attachment");
    let size = if bytes > 0 { format!(", {bytes} bytes") } else { String::new() };
    format!("[file omitted: {label}{size} — {reason}]")
}

fn upload_url_from(j: &Value) -> Option<String> {
    if !j.is_object() {
        return None;
    }
    let r = if j["result"].is_object() { &j["result"] } else { j };
    for a in [&r["imageUrls"], &r["image_urls"], &j["imageUrls"], &j["image_urls"]] {
        if let Some(s) = a[0].as_str().filter(|s| !s.is_empty()) {
            return Some(s.to_string());
        }
    }
    for k in ["imageUrl", "image_url", "url", "ossUrl", "oss_url", "originalUrl", "originUrl", "link", "image"] {
        let v = if !r[k].is_null() { &r[k] } else { &j[k] };
        if let Some(s) = v.as_str().filter(|s| !s.is_empty()) {
            return Some(s.to_string());
        }
    }
    j["body"].as_str().and_then(|b| serde_json::from_str::<Value>(b).ok()).and_then(|b| upload_url_from(&b))
}

async fn upload_image(creds: &Value, region: &str, buf: Vec<u8>, mime: &str) -> Result<String, String> {
    let url = format!("{}/algo{IMAGE_UPLOAD_SIG_PATH}?request_id={}", inference_base(creds, region), uuid::Uuid::new_v4());
    let boundary = format!("----9routerQoder{:x}{}", now_ms(), &crate::jsv::rand_hex(6));
    let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"image.{}\"\r\nContent-Type: {}\r\n\r\n", mime_ext(mime), if mime.is_empty() { "application/octet-stream" } else { mime }).into_bytes();
    body.extend_from_slice(&buf);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let mut h = cosy_headers(&body, &url, &cosy_creds(creds))?;
    h.set("Accept", "application/json");
    h.set("Content-Type", format!("multipart/form-data; boundary={boundary}"));
    h.set("AI-CLIENT-TIMESTAMP", (now_ms() / 1000).to_string());
    h.set("Accept-Encoding", "identity");
    let mut rb = client_for(creds).put(&url);
    for (k, v) in &h.0 {
        rb = rb.header(k, v);
    }
    let r = rb.body(body).send().await.map_err(|e| e.to_string())?;
    let st = r.status().as_u16();
    let text = r.text().await.unwrap_or_default();
    if !(200..300).contains(&st) {
        return Err(format!("HTTP {st}{}", if text.is_empty() { String::new() } else { format!(": {}", text.chars().take(180).collect::<String>()) }));
    }
    let j: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    upload_url_from(&j).ok_or_else(|| "upload response missing url".into())
}

enum Up {
    Url(String),
    Keep,
    Stub(usize),
}

struct AttachCtx<'a> {
    creds: &'a Value,
    region: &'a str,
    cache: HashMap<String, String>,
}

async fn upload_data(ctx: &mut AttachCtx<'_>, b64: &str, mime: &str) -> Up {
    let compact: String = b64.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.is_empty() {
        return Up::Stub(0);
    }
    let bytes = decoded_bytes(&compact);
    if bytes > MAX_IMAGE_BYTES {
        return Up::Stub(bytes);
    }
    let Ok(buf) = B64.decode(&compact) else { return Up::Stub(bytes) };
    let digest = hex::encode(sha2::Sha256::digest(&buf));
    if let Some(u) = ctx.cache.get(&digest) {
        return Up::Url(u.clone());
    }
    match upload_image(ctx.creds, ctx.region, buf, mime).await {
        Ok(u) => {
            ctx.cache.insert(digest, u.clone());
            Up::Url(u)
        }
        Err(e) => {
            tracing::warn!("qoder image upload failed ({e})");
            if bytes <= INLINE_FALLBACK_MAX_BYTES { Up::Keep } else { Up::Stub(bytes) }
        }
    }
}

fn img(url: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": url}})
}
fn txt(t: String) -> Value {
    json!({"type": "text", "text": t})
}

fn strip_large_data_uris(s: &str, reason: &str) -> String {
    DATA_URI_RE
        .replace_all(s, |c: &regex::Captures| {
            let m = &c[0];
            let parsed = crate::translate::concerns::parse_data_uri(&json!(m.trim()));
            let bytes = parsed.as_ref().map(|p| decoded_bytes(&p.1)).unwrap_or(m.len());
            if bytes <= INLINE_FALLBACK_MAX_BYTES {
                return m.to_string();
            }
            stub(None, parsed.as_ref().map(|p| p.0.as_str()), bytes, reason)
        })
        .into_owned()
}

async fn rewrite_block(b: &Value, ctx: &mut AttachCtx<'_>) -> Option<Value> {
    if !b.is_object() {
        return Some(b.clone());
    }
    let ty = b["type"].as_str().unwrap_or("");
    if ty == "image_url" {
        let raw = b["image_url"].as_str().or_else(|| b["image_url"]["url"].as_str()).unwrap_or("").to_string();
        if raw.is_empty() {
            return None;
        }
        if raw.starts_with("http://") || raw.starts_with("https://") {
            return Some(img(&raw));
        }
        let Some((mime, data)) = crate::translate::concerns::parse_data_uri(&json!(raw)) else {
            return Some(txt(stub(Some("attachment"), None, 0, "unreadable data URI")));
        };
        if !mime.to_lowercase().starts_with("image/") {
            return Some(txt(stub(Some("file"), Some(&mime), decoded_bytes(&data), "non-image bytes are not inlined into Qoder context")));
        }
        return Some(match upload_data(ctx, &data, &mime).await {
            Up::Url(u) => img(&u),
            Up::Keep => img(&raw),
            Up::Stub(n) => txt(stub(Some("image"), Some(&mime), n, "upload failed; not inlined")),
        });
    }
    if ty == "image" {
        let src = &b["source"];
        if src["type"] == "url" {
            if let Some(u) = src["url"].as_str() {
                return Some(img(u));
            }
        }
        if src["type"] == "base64" && truthy(&src["data"]) {
            let mime = src["media_type"].as_str().filter(|s| !s.is_empty()).unwrap_or("image/png").to_string();
            let data = js_string(&src["data"]);
            return Some(match upload_data(ctx, &data, &mime).await {
                Up::Url(u) => img(&u),
                Up::Keep => img(&format!("data:{mime};base64,{data}")),
                Up::Stub(n) => txt(stub(Some("image"), Some(&mime), n, "upload failed; not inlined")),
            });
        }
    }
    if ty == "file" && truthy(&b["file"]) {
        let f = &b["file"];
        let name = f["filename"].as_str().or_else(|| f["name"].as_str()).unwrap_or("file").to_string();
        let fd = f["file_data"].as_str().unwrap_or("");
        let parsed = if fd.is_empty() { None } else { crate::translate::concerns::parse_data_uri(&json!(fd)) };
        let b64 = parsed.as_ref().map(|p| p.1.clone()).or_else(|| (!fd.is_empty() && !fd.starts_with("data:")).then(|| fd.to_string()));
        let mime = parsed.as_ref().map(|p| p.0.clone()).or_else(|| f["format"].as_str().map(str::to_owned)).unwrap_or_else(|| "application/octet-stream".into());
        if let Some(d) = &b64 {
            if mime.to_lowercase().starts_with("image/") {
                if let Up::Url(u) = upload_data(ctx, d, &mime).await {
                    return Some(img(&u));
                }
            }
        }
        return Some(txt(stub(Some(&name), Some(&mime), decoded_bytes(b64.as_deref().unwrap_or("")), "Qoder reads documents via its file API, not inlined bytes")));
    }
    if ty == "document" && truthy(&b["source"]) {
        let src = &b["source"];
        let name = b["title"].as_str().unwrap_or("document").to_string();
        if src["type"] == "base64" && truthy(&src["data"]) {
            let mime = src["media_type"].as_str().filter(|s| !s.is_empty()).unwrap_or("application/pdf").to_string();
            let data = js_string(&src["data"]);
            if mime.to_lowercase().starts_with("image/") {
                if let Up::Url(u) = upload_data(ctx, &data, &mime).await {
                    return Some(img(&u));
                }
            }
            return Some(txt(stub(Some(&name), Some(&mime), decoded_bytes(&data), "Qoder reads documents via its file API, not inlined bytes")));
        }
    }
    if let Some(t) = b["text"].as_str() {
        if t.contains("data:") && t.len() > 8192 {
            let mut nb = b.clone();
            nb["text"] = json!(strip_large_data_uris(t, "inlined data URI stripped from Qoder context"));
            return Some(nb);
        }
    }
    Some(b.clone())
}

async fn rewrite_attachments(messages: &mut [Value], creds: &Value, region: &str) {
    let mut ctx = AttachCtx { creds, region, cache: HashMap::new() };
    for m in messages.iter_mut() {
        if !m.is_object() {
            continue;
        }
        if let Some(imgs) = m["images"].as_array().cloned() {
            let extras: Vec<Value> = imgs.iter().map(|u| img(&js_string(u))).collect();
            let mut c = if let Some(a) = m["content"].as_array() { a.clone() } else { vec![txt(m["content"].as_str().unwrap_or("").to_string())] };
            c.extend(extras);
            m["content"] = Value::Array(c);
            crate::jsv::del(m, "images");
        }
        let content = m["content"].clone();
        m["content"] = match &content {
            Value::String(s) if s.contains("data:") && s.len() > 8192 => json!(strip_large_data_uris(s, "inlined data URI stripped from Qoder context")),
            Value::Array(a) => {
                let mut out = vec![];
                for b in a {
                    if let Some(n) = rewrite_block(b, &mut ctx).await {
                        out.push(n);
                    }
                }
                if out.is_empty() { json!("") } else { Value::Array(out) }
            }
            other => other.clone(),
        };
    }
    if serde_json::to_vec(messages).map(|v| v.len()).unwrap_or(0) > MAX_PAYLOAD_BYTES {
        for m in messages.iter_mut() {
            if let Some(s) = m["content"].as_str().filter(|s| s.contains("data:")).map(str::to_owned) {
                m["content"] = json!(DATA_URI_RE.replace_all(&s, |c: &regex::Captures| stub(None, None, c[0].len(), "payload over Qoder size budget")).into_owned());
            } else if let Some(a) = m["content"].as_array_mut() {
                for b in a.iter_mut() {
                    if b["type"] == "image_url" {
                        let raw = b["image_url"].as_str().or_else(|| b["image_url"]["url"].as_str()).unwrap_or("");
                        if raw.starts_with("data:") {
                            *b = txt(stub(Some("image"), None, 0, "payload over Qoder size budget"));
                            continue;
                        }
                    }
                    if let Some(t) = b["text"].as_str().filter(|t| t.contains("data:")).map(str::to_owned) {
                        b["text"] = json!(DATA_URI_RE.replace_all(&t, |c: &regex::Captures| stub(None, None, c[0].len(), "payload over Qoder size budget")).into_owned());
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// message normalization + payload
// ---------------------------------------------------------------------------

fn extract_text(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Array(a) => a.iter().filter_map(|i| i["text"].as_str().map(str::to_owned)).collect::<Vec<_>>().join("\n"),
        o => js_string(o),
    }
}

fn normalize_content(c: &Value) -> Value {
    match c {
        Value::String(_) => return c.clone(),
        Value::Null => return json!(""),
        Value::Array(_) => {}
        o => return json!(js_string(o)),
    }
    let mut blocks: Vec<Value> = vec![];
    let mut texts: Vec<String> = vec![];
    let mut has_image = false;
    for it in c.as_array().unwrap() {
        if !it.is_object() {
            continue;
        }
        let ty = it["type"].as_str().unwrap_or("");
        let push_text = |t: String, blocks: &mut Vec<Value>, texts: &mut Vec<String>, has_image: bool| {
            if t.is_empty() {
                return;
            }
            if has_image || !blocks.is_empty() {
                blocks.push(txt(t));
            } else {
                texts.push(t);
            }
        };
        let iu = if ty == "image_url" { it["image_url"].as_str().filter(|s| !s.is_empty()).or_else(|| it["image_url"]["url"].as_str().filter(|s| !s.is_empty())) } else { None };
        if let Some(u) = iu {
            blocks.push(img(u));
            has_image = true;
        } else if ty == "image" && truthy(&it["source"]) {
            let s = &it["source"];
            let url = if s["type"] == "base64" && truthy(&s["data"]) {
                Some(format!("data:{};base64,{}", s["media_type"].as_str().filter(|x| !x.is_empty()).unwrap_or("image/png"), js_string(&s["data"])))
            } else {
                s["url"].as_str().filter(|x| !x.is_empty()).map(str::to_owned)
            };
            if let Some(u) = url {
                blocks.push(img(&u));
                has_image = true;
            }
        } else if ty == "file" {
            let n = it["file"]["filename"].as_str().or_else(|| it["file"]["name"].as_str()).unwrap_or("file");
            push_text(format!("[file omitted: {n} — Qoder reads documents via its file API, not inlined bytes]"), &mut blocks, &mut texts, has_image);
        } else if ty == "document" {
            let n = it["title"].as_str().unwrap_or("document");
            push_text(format!("[file omitted: {n} — Qoder reads documents via its file API, not inlined bytes]"), &mut blocks, &mut texts, has_image);
        } else if let Some(t) = it["text"].as_str() {
            push_text(t.to_string(), &mut blocks, &mut texts, has_image);
        }
    }
    if !has_image {
        return json!(texts.join("\n"));
    }
    if !texts.is_empty() {
        blocks.insert(0, txt(texts.join("\n")));
    }
    Value::Array(blocks)
}

pub fn normalize_messages(messages: &[Value]) -> (Vec<Value>, String) {
    let mut sys = vec![];
    let mut out = vec![];
    for m in messages {
        if !m.is_object() {
            continue;
        }
        if m["role"] == "system" {
            let t = extract_text(&m["content"]);
            if !t.is_empty() {
                sys.push(t);
            }
            continue;
        }
        let mut c = m.clone();
        c["content"] = normalize_content(&m["content"]);
        out.push(c);
    }
    (out, sys.join("\n\n"))
}

fn last_user_text(msgs: &[Value]) -> String {
    for m in msgs.iter().rev() {
        if m["role"] != "user" {
            continue;
        }
        if let Some(s) = m["content"].as_str() {
            return s.to_string();
        }
        if m["content"].is_array() {
            return extract_text(&m["content"]);
        }
    }
    String::new()
}

fn stable_hash(prefix: &str, parts: &[&str]) -> String {
    let mut s = prefix.to_string();
    for p in parts {
        s.push('\0');
        s.push_str(p);
    }
    crate::jsv::sha256_hex(&s)[..16].to_string()
}

fn stable_record_id(model: &str, msgs: &[Value], tools: &Value, max_tokens: i64) -> String {
    let mut s = format!("qoder-record\0{model}");
    for m in msgs {
        if !m.is_object() {
            continue;
        }
        if let Some(r) = m["role"].as_str().filter(|r| !r.is_empty()) {
            s.push('\0');
            s.push_str(r);
        }
        if let Some(c) = m["content"].as_str().filter(|c| !c.is_empty()) {
            s.push('\0');
            s.push_str(c);
        } else if m["content"].is_array() {
            s.push('\0');
            s.push_str(&m["content"].to_string());
        }
    }
    if truthy(tools) {
        s.push('\0');
        s.push_str(&tools.to_string());
    }
    s.push_str(&format!("\0mt={max_tokens}"));
    crate::jsv::sha256_hex(&s)[..16].to_string()
}

fn truncate(s: &str, n: usize) -> String {
    let u: Vec<u16> = s.encode_utf16().collect();
    if u.len() > n { format!("{}...", String::from_utf16_lossy(&u[..n])) } else { s.to_string() }
}

// context tiers ------------------------------------------------------------

pub fn parse_tier_count(v: &Value) -> i64 {
    match v {
        Value::Number(n) => n.as_f64().filter(|f| f.is_finite() && *f > 0.0).map(|f| f.floor() as i64).unwrap_or(0),
        Value::String(s) => {
            static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(\d+(?:\.\d+)?)\s*([KM])?$").unwrap());
            let up = s.trim().to_uppercase();
            let Some(c) = RE.captures(&up) else { return 0 };
            let n: f64 = c[1].parse().unwrap_or(0.0);
            let mult = match c.get(2).map(|m| m.as_str()) {
                Some("K") => 1000.0,
                Some("M") => 1_000_000.0,
                _ => 1.0,
            };
            let r = n * mult;
            if r > 0.0 { r.floor() as i64 } else { 0 }
        }
        _ => 0,
    }
}

#[derive(Clone, Debug)]
pub struct Tier {
    pub name: String,
    pub tokens: i64,
    pub is_default: bool,
}

pub fn context_tiers(cfg: &Value) -> Vec<Tier> {
    let list = if cfg["context_config"].is_array() { &cfg["context_config"] } else { &cfg["contextConfig"] };
    let mut by: Vec<Tier> = vec![];
    for e in list.as_array().into_iter().flatten() {
        if !e.is_object() {
            continue;
        }
        let raw = ["tokenCount", "token_count", "max_input_tokens", "maxInputTokens", "contextLength", "context_length"].iter().map(|k| &e[*k]).find(|v| !v.is_null()).cloned().unwrap_or(Value::Null);
        let n = parse_tier_count(&raw);
        if n == 0 {
            continue;
        }
        let is_def = e["isDefault"] == json!(true) || e["is_default"] == json!(true) || e["default"] == json!(true);
        let name = ["name", "label", "display_name", "displayName", "key", "id"]
            .iter()
            .map(|k| &e[*k])
            .find(|v| !v.is_null())
            .and_then(|v| v.as_str().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()))
            .unwrap_or_else(|| {
                if n >= 1_000_000 && n % 1_000_000 == 0 {
                    format!("{}M", n / 1_000_000)
                } else if n >= 1000 && n % 1000 == 0 {
                    format!("{}K", n / 1000)
                } else {
                    n.to_string()
                }
            });
        if let Some(t) = by.iter_mut().find(|t| t.tokens == n) {
            t.name = name;
            t.is_default = t.is_default || is_def;
        } else {
            by.push(Tier { name, tokens: n, is_default: is_def });
        }
    }
    by.sort_by_key(|t| t.tokens);
    by
}

pub fn estimate_tokens(system: &str, messages: &[Value], tools: &Value) -> i64 {
    let text = json!({"system": system, "messages": messages, "tools": if tools.is_null() { json!([]) } else { tools.clone() }}).to_string();
    let units: Vec<u16> = text.encode_utf16().collect();
    let cjk = units.iter().filter(|&&u| (0x1100..=0x11ff).contains(&u) || (0x2e80..=0x9fff).contains(&u) || (0xac00..=0xd7af).contains(&u) || (0xf900..=0xfaff).contains(&u) || (0xff00..=0xffef).contains(&u)).count();
    (cjk as f64 + (units.len() - cjk) as f64 / 4.0).ceil() as i64
}

pub fn resolve_tier(cfg: &Value, system: &str, messages: &[Value], tools: &Value, preference: Option<&str>) -> Option<(Tier, String)> {
    let tiers = context_tiers(cfg);
    if tiers.is_empty() {
        return None;
    }
    let mode = preference.map(str::trim).filter(|s| !s.is_empty()).unwrap_or("auto");
    let largest = tiers.last().unwrap().clone();
    let default = tiers.iter().find(|t| t.is_default).unwrap_or(&tiers[0]).clone();
    let est = estimate_tokens(system, messages, tools);
    let need = (est as f64 * (1.0 + TIER_HEADROOM)).ceil() as i64;
    match mode.to_lowercase().as_str() {
        "max" => return Some((largest, "forced:max".into())),
        "default" => return Some((default, "forced:default".into())),
        "auto" => {}
        _ => {
            let wanted: String = mode.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_uppercase();
            let as_count = parse_tier_count(&json!(wanted));
            if let Some(t) = tiers.iter().find(|t| t.name.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_uppercase() == wanted || (as_count > 0 && t.tokens == as_count)) {
                return Some((t.clone(), format!("forced:{}", t.name)));
            }
        }
    }
    let cur = parse_tier_count(if !cfg["max_input_tokens"].is_null() { &cfg["max_input_tokens"] } else { &cfg["maxInputTokens"] });
    let limit = if cur > 0 { cur } else { default.tokens };
    if need <= limit {
        return None;
    }
    let fits = tiers.iter().find(|t| t.tokens >= need && t.tokens > limit).cloned();
    let tier = fits.clone().unwrap_or(largest);
    if tier.tokens <= limit {
        return None;
    }
    Some((tier, if fits.is_some() { "auto:fits".into() } else { "auto:largest".into() }))
}

fn apply_tier(payload: &mut Value, tier: &Tier) {
    payload["parameters"]["context_length"] = json!(tier.tokens);
    payload["chat_context"]["extra"]["ideModelConfigOverride"]["max_input_tokens"] = json!(tier.tokens);
    if payload["model_config"].is_object() {
        payload["model_config"]["max_input_tokens"] = json!(tier.tokens);
    }
}

async fn build_payload(model: &str, body: &Value, creds: &Value, region: &str) -> Result<(String, Value), String> {
    let key = model.strip_prefix("qoder/").unwrap_or(model).to_string();
    let mut cfg = resolve_models(creds, region, false).await["rawConfigs"][&key].clone();
    if cfg.is_null() {
        cfg = resolve_models(creds, region, true).await["rawConfigs"][&key].clone();
        if cfg.is_null() {
            return Err(format!("qoder: model_config for \"{key}\" not yet known (run a model list fetch or check upstream connectivity)"));
        }
    }
    cfg["key"] = json!(key);
    let mut incoming: Vec<Value> = body["messages"].as_array().cloned().unwrap_or_default();
    rewrite_attachments(&mut incoming, creds, region).await;
    let (messages, system) = normalize_messages(&incoming);
    let tools = &body["tools"];
    let is_reasoning = truthy(&cfg["is_reasoning"]);
    let max_out = cfg["max_output_tokens"].as_f64().or_else(|| cfg["max_output_tokens"].as_str().and_then(|s| s.parse().ok())).unwrap_or(0.0) as i64;
    let mut max_tokens: i64 = if max_out > 0 { max_out } else { 32_768 };
    for k in ["max_tokens", "max_completion_tokens"] {
        if let Some(n) = body[k].as_f64() {
            if n > 0.0 && (n as i64) < max_tokens {
                max_tokens = n as i64;
            }
        }
    }
    let last = last_user_text(&messages);
    let uid = js_or_empty(&creds["providerSpecificData"]["userId"]);
    let session = stable_hash("qoder-session", &[&uid, &key]);
    let record = stable_record_id(&key, &messages, tools, max_tokens);
    let tier = resolve_tier(&cfg, &system, &messages, tools, std::env::var("QODER_CONTEXT_TIER").ok().as_deref());
    let mut payload = json!({
        "request_id": uuid::Uuid::new_v4().to_string(),
        "request_set_id": record,
        "chat_record_id": record,
        "session_id": session,
        "stream": true,
        "chat_task": "FREE_INPUT",
        "is_reply": true,
        "is_retry": false,
        "source": 1,
        "version": "3",
        "session_type": "qodercli",
        "agent_id": "agent_common",
        "task_id": "common",
        "code_language": "",
        "chat_prompt": "",
        "image_urls": null,
        "aliyun_user_type": "",
        "system": system,
        "messages": messages,
        "tools": if tools.is_array() { tools.clone() } else { json!([]) },
        "parameters": {"max_tokens": max_tokens},
        "chat_context": {
            "chatPrompt": "",
            "imageUrls": null,
            "extra": {"context": [], "modelConfig": {"key": key, "is_reasoning": is_reasoning}, "originalContent": last},
            "features": [],
            "text": last,
        },
        "model_config": cfg,
        "business": {"product": "cli", "version": "1.0.0", "type": "agent", "stage": "start", "id": uuid::Uuid::new_v4().to_string(), "name": truncate(&last, 30), "begin_at": now_ms()},
    });
    if let Some((t, reason)) = tier {
        tracing::info!("qoder context tier {} ({} tokens, {reason})", t.name, t.tokens);
        apply_tier(&mut payload, &t);
    }
    Ok((key, payload))
}

// ---------------------------------------------------------------------------
// SSE envelope unwrapping
// ---------------------------------------------------------------------------

pub fn is_billing_block(inner: &str) -> bool {
    if inner.is_empty() {
        return false;
    }
    if inner.to_lowercase().contains("pricingurl") {
        return true;
    }
    if let Ok(p) = serde_json::from_str::<Value>(inner) {
        let code = if p["code"].is_null() { String::new() } else { js_string(&p["code"]) };
        if ["110", "112", "10605"].contains(&code.as_str()) {
            return true;
        }
    }
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""code"\s*:\s*"(112|10605)""#).unwrap());
    RE.is_match(inner)
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Null => Some(0.0),
        _ => None,
    }
    .filter(|f| f.is_finite())
}

fn first_present<'a>(vals: &[&'a Value]) -> &'a Value {
    vals.iter().find(|v| !v.is_null()).copied().unwrap_or(&Value::Null)
}

pub fn canonical_usage(u: &Value) -> Option<Value> {
    if !u.is_object() {
        return None;
    }
    let prompt = num(first_present(&[&u["prompt_tokens"], &u["input_tokens"]])).filter(|_| !first_present(&[&u["prompt_tokens"], &u["input_tokens"]]).is_null());
    let completion = num(first_present(&[&u["completion_tokens"], &u["output_tokens"]])).filter(|_| !first_present(&[&u["completion_tokens"], &u["output_tokens"]]).is_null());
    if prompt.is_none() && completion.is_none() {
        return None;
    }
    let mut details = if u["prompt_tokens_details"].is_object() { u["prompt_tokens_details"].clone() } else { json!({}) };
    let cached_v = first_present(&[&details["cached_tokens"], &u["cached_tokens"], &u["prompt_cache_hit_tokens"], &u["cache_read_input_tokens"]]).clone();
    let cc_v = first_present(&[&details["cache_creation_tokens"], &u["cache_creation_input_tokens"]]).clone();
    let p = prompt.unwrap_or(0.0);
    let c = completion.unwrap_or(0.0);
    let total = if u["total_tokens"].is_null() { p + c } else { num(&u["total_tokens"]).unwrap_or(p + c) };
    let mut out = json!({"prompt_tokens": crate::jsv::jnum(p), "completion_tokens": crate::jsv::jnum(c), "total_tokens": crate::jsv::jnum(total)});
    if !cached_v.is_null() {
        if let Some(n) = num(&cached_v) {
            out["cached_tokens"] = crate::jsv::jnum(n);
            details["cached_tokens"] = crate::jsv::jnum(n);
        }
    }
    if !cc_v.is_null() {
        if let Some(n) = num(&cc_v) {
            details["cache_creation_tokens"] = crate::jsv::jnum(n);
        }
    }
    if details.as_object().map(|o| !o.is_empty()).unwrap_or(false) {
        out["prompt_tokens_details"] = details;
    }
    if u["completion_tokens_details"].is_object() {
        out["completion_tokens_details"] = u["completion_tokens_details"].clone();
    }
    let r = first_present(&[&u["reasoning_tokens"], &u["completion_tokens_details"]["reasoning_tokens"]]);
    if !r.is_null() {
        if let Some(n) = num(r) {
            out["reasoning_tokens"] = crate::jsv::jnum(n);
        }
    }
    Some(out)
}

/// createQoderSseCoalescer — holds finish + usage frames into one terminal chunk.
pub struct Coalescer {
    model: String,
    pending_finish: Option<Value>,
    pending_usage: Option<Value>,
    meta_id: Option<String>,
    meta_created: Option<Value>,
    meta_model: Option<String>,
    pub done: bool,
    finish_forwarded: bool,
}

fn sse_line(s: &str) -> Bytes {
    Bytes::from(format!("data: {}\n\n", s.replace("\r\n", "").replace('\n', "")))
}

impl Coalescer {
    pub fn new(model: &str) -> Self {
        Coalescer { model: model.into(), pending_finish: None, pending_usage: None, meta_id: None, meta_created: None, meta_model: None, done: false, finish_forwarded: false }
    }

    fn terminal(&mut self, out: &mut Vec<Bytes>) {
        if self.pending_finish.is_none() && self.pending_usage.is_none() {
            return;
        }
        let mut o = json!({
            "id": self.meta_id.clone().unwrap_or_else(|| format!("qoder-{}", now_ms())),
            "object": "chat.completion.chunk",
            "created": self.meta_created.clone().unwrap_or(json!(now_ms() / 1000)),
            "model": self.meta_model.clone().unwrap_or_else(|| self.model.clone()),
            "choices": [{"index": 0, "delta": {}, "finish_reason": self.pending_finish.clone().unwrap_or(json!("stop"))}],
        });
        if let Some(u) = self.pending_usage.take() {
            o["usage"] = u;
        }
        self.pending_finish = None;
        out.push(sse_line(&o.to_string()));
    }

    fn emit_done(&mut self, out: &mut Vec<Bytes>) {
        if !self.done {
            out.push(Bytes::from_static(b"data: [DONE]\n\n"));
            self.done = true;
        }
    }

    pub fn flush(&mut self, out: &mut Vec<Bytes>) {
        if self.done {
            return;
        }
        if self.pending_usage.is_some() || (self.pending_finish.is_some() && !self.finish_forwarded) {
            self.terminal(out);
        }
        self.emit_done(out);
    }

    pub fn handle(&mut self, inner: &str, out: &mut Vec<Bytes>) {
        if self.done || inner.is_empty() {
            return;
        }
        if inner == "[DONE]" {
            self.flush(out);
            return;
        }
        let Ok(p) = serde_json::from_str::<Value>(inner) else {
            out.push(sse_line(inner));
            return;
        };
        if !p.is_object() {
            out.push(sse_line(inner));
            return;
        }
        if let Some(id) = p["id"].as_str().filter(|s| !s.is_empty()) {
            self.meta_id = Some(id.into());
        }
        if p["created"].is_number() {
            self.meta_created = Some(p["created"].clone());
        }
        if let Some(m) = p["model"].as_str().filter(|s| !s.is_empty()) {
            self.meta_model = Some(m.into());
        }
        if let Some(u) = canonical_usage(&p["usage"]) {
            self.pending_usage = Some(u);
        }
        let c0 = &p["choices"][0];
        let finish = [&c0["finish_reason"], &c0["delta"]["finish_reason"], &p["finish_reason"]].into_iter().find(|v| truthy(v)).cloned();
        let d = &c0["delta"];
        let valuable = d.is_object()
            && (d["content"].as_str().map(|s| !s.is_empty()).unwrap_or(false)
                || d["reasoning_content"].as_str().map(|s| !s.is_empty()).unwrap_or(false)
                || d["tool_calls"].as_array().map(|a| !a.is_empty()).unwrap_or(false)
                || truthy(&d["role"]));
        if valuable {
            out.push(sse_line(inner));
            if let Some(f) = finish {
                self.finish_forwarded = true;
                self.pending_finish = if self.pending_usage.is_some() { Some(f) } else { None };
            }
            if self.pending_finish.is_some() && self.pending_usage.is_some() {
                self.terminal(out);
                self.emit_done(out);
            }
            return;
        }
        if let Some(f) = finish {
            self.pending_finish = Some(f);
        }
        if (self.pending_finish.is_some() || self.finish_forwarded) && self.pending_usage.is_some() {
            if self.pending_finish.is_none() {
                self.pending_finish = Some(json!("stop"));
            }
            self.terminal(out);
            self.emit_done(out);
        }
    }
}

fn envelope_inner(env: &Value) -> String {
    match &env["body"] {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        o => o.to_string(),
    }
}

/// wrapQoderSSE: peek for first-frame errors, then unwrap envelopes.
async fn wrap_sse(up: Upstream, model: String) -> Upstream {
    let status = up.status;
    let mut body = up.body;
    let mut lines = crate::sse::LineParser::default();
    let mut consumed: Vec<String> = vec![];
    let mut drained = false;
    // peek
    'peek: loop {
        let new_lines = match body.next().await {
            Some(Ok(c)) => lines.push(&c),
            _ => {
                drained = true;
                lines.finish().into_iter().collect()
            }
        };
        for l in new_lines {
            consumed.push(l.clone());
            let t = l.trim_end_matches('\r').trim();
            let Some(data) = t.strip_prefix("data:") else { continue };
            let data = data.trim_start();
            if data == "[DONE]" {
                break 'peek;
            }
            let Ok(env) = serde_json::from_str::<Value>(data) else { break 'peek };
            let sv = match &env["statusCodeValue"] {
                Value::Null => f64::NAN,
                v => num(v).unwrap_or(f64::NAN),
            };
            let sv = if sv.is_nan() { 200.0 } else { sv };
            let inner = envelope_inner(&env);
            if sv != 200.0 {
                let svi = sv as i64;
                let st = if is_billing_block(&inner) { 403 } else if (400..=599).contains(&svi) && sv.fract() == 0.0 { svi as u16 } else { 502 };
                let msg = if inner.is_empty() { format!("upstream status {}", crate::jsv::jnum(sv)) } else { inner };
                return Upstream::json(st, &json!({"error": {"message": msg, "code": crate::jsv::jnum(sv)}}));
            }
            break 'peek;
        }
        if drained {
            break;
        }
    }
    let s = async_stream::stream! {
        let mut co = Coalescer::new(&model);
        let mut pending = consumed;
        let mut drained = drained;
        loop {
            for line in pending.drain(..) {
                let t = line.trim_end_matches('\r').trim();
                if t.is_empty() || co.done { continue; }
                let Some(data) = t.strip_prefix("data:") else { continue };
                let data = data.trim_start();
                let mut out = vec![];
                if data == "[DONE]" {
                    co.flush(&mut out);
                } else if let Ok(env) = serde_json::from_str::<Value>(data) {
                    let sv = num(&env["statusCodeValue"]).filter(|n| *n != 0.0).unwrap_or(200.0);
                    let inner = envelope_inner(&env);
                    if sv != 200.0 {
                        tracing::error!("[QODER] error envelope status={sv} body={}", truncate(&inner, 300));
                        if is_billing_block(&inner) {
                            let msg = if inner.is_empty() { format!("qoder billing block ({sv})") } else { inner.clone() };
                            out.push(Bytes::from(format!("data: {}\n\n", json!({"error": {"message": msg, "code": "qoder_billing_block", "status": 403, "type": "quota_error"}}))));
                        } else {
                            let msg = if inner.is_empty() { format!("upstream status {sv}") } else { inner.clone() };
                            let c = json!({"id": format!("qoder-error-{}", now_ms()), "object": "chat.completion.chunk", "created": now_ms() / 1000, "model": model,
                                "choices": [{"index": 0, "delta": {"content": format!("\n[qoder error {}: {}]", crate::jsv::jnum(sv), truncate(&msg, 200))}, "finish_reason": "stop"}]});
                            out.push(Bytes::from(format!("data: {c}\n\n")));
                        }
                        out.push(Bytes::from_static(b"data: [DONE]\n\n"));
                        co.done = true;
                    } else if !inner.is_empty() {
                        co.handle(&inner, &mut out);
                    }
                }
                for b in out { yield Ok::<Bytes, String>(b); }
            }
            if co.done || drained { break; }
            match body.next().await {
                Some(Ok(c)) => pending = lines.push(&c),
                _ => { pending = lines.finish().into_iter().collect(); drained = true; }
            }
        }
        if !co.done {
            let mut out = vec![];
            co.flush(&mut out);
            for b in out { yield Ok(b); }
        }
    };
    Upstream::synthetic(status, "text/event-stream", Box::pin(s) as ByteStream)
}

// ---------------------------------------------------------------------------
// executor
// ---------------------------------------------------------------------------

pub struct Qoder {
    pub id: String,
}

fn err401(msg: String) -> Upstream {
    Upstream::json(401, &json!({"error": {"message": msg}}))
}

#[async_trait]
impl Executor for Qoder {
    fn provider(&self) -> &str {
        &self.id
    }
    async fn refresh_credentials(&self, _c: &Value) -> Option<Value> {
        None
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let region = region_of(&self.id);
        let url_for = |c: &Value| format!("{}/algo{CHAT_SIG_PATH}?FetchKeys=llm_model_result&AgentId=agent_common&Encode=1", inference_base(c, region));
        let fail = |up: Upstream, url: String, body: Value| Ok(ExecResult { response: up, url, headers: vec![], body, response_format: None });
        let raw = crate::exec::cred_str(args.creds, "apiKey").or_else(|| crate::exec::cred_str(args.creds, "accessToken")).unwrap_or("").to_string();
        let creds = if is_pat(&raw) {
            match resolve_credentials(args.creds, region).await {
                Ok(c) => c,
                Err(e) => return fail(err401(format!("qoder PAT exchange failed: {e}")), url_for(args.creds), args.body.clone()),
            }
        } else {
            args.creds.clone()
        };
        let url = url_for(&creds);
        if !truthy(&creds["providerSpecificData"]["userId"]) {
            return fail(err401("qoder credential is missing userId; reconnect the account".into()), url, args.body.clone());
        }
        if !truthy(&creds["accessToken"]) {
            return fail(err401("qoder credential is missing accessToken; reconnect the account".into()), url, args.body.clone());
        }
        let (key, payload) = match build_payload(args.model, &args.body, &creds, region).await {
            Ok(p) => p,
            Err(e) => return fail(Upstream::json(400, &json!({"error": {"message": e}})), url, args.body.clone()),
        };
        let encoded = encode_body(payload.to_string().as_bytes());
        let cosy = match cosy_headers(&encoded, &url, &cosy_creds(&creds)) {
            Ok(h) => h,
            Err(e) => return fail(err401(format!("qoder cosy signing failed: {e}")), url, args.body.clone()),
        };
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.set("Accept", "text/event-stream");
        h.set("Cache-Control", "no-cache");
        h.set("X-Model-Key", key.clone());
        h.set("X-Model-Source", payload["model_config"]["source"].as_str().filter(|s| !s.is_empty()).unwrap_or("system"));
        h.set("Accept-Encoding", "identity");
        for (k, v) in cosy.0 {
            h.set(&k, v);
        }
        let timeout = self.config()["timeoutMs"].as_u64().unwrap_or(60_000);
        let up = crate::exec::post_raw(&creds, &url, &h, encoded, timeout).await?;
        if !up.ok() {
            return Ok(ExecResult { response: up, url, headers: h.0, body: payload, response_format: None });
        }
        let wrapped = wrap_sse(up, format!("{}/{key}", self.id)).await;
        let ok = wrapped.ok();
        Ok(ExecResult { response: wrapped, url, headers: h.0, body: payload, response_format: ok.then(|| "openai".into()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_roundtrip_shape() {
        // "hello world" → b64 "aGVsbG8gd29ybGQ=" (16 chars, a=5)
        let e = String::from_utf8(encode_body(b"hello world")).unwrap();
        assert_eq!(e.len(), 16);
        assert!(e.ends_with("&TRHk") || e.contains('$'));
    }

    #[test]
    fn cosy_and_messages() {
        let c = CosyCreds { user_id: "u".into(), auth_token: "dt-x".into(), name: String::new(), email: String::new(), machine_id: "m".into() };
        let h = cosy_headers(b"abc", "https://api3.qoder.sh/algo/api/v2/model/list", &c).unwrap();
        assert_eq!(h.get("Cosy-Sigpath"), Some("/api/v2/model/list"));
        assert_eq!(h.get("Cosy-Bodyhash"), Some("900150983cd24fb0d6963f7d28e17f72"));
        assert!(h.get("Authorization").unwrap().starts_with("Bearer COSY."));
        let msgs = json!([{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]);
        let (m, sys) = normalize_messages(msgs.as_array().unwrap());
        assert_eq!(sys, "s");
        assert_eq!(m[0]["content"][0]["text"], "a");
        assert_eq!(m[0]["content"][1]["type"], "image_url");
    }

    #[test]
    fn tiers_and_billing() {
        let cfg = json!({"max_input_tokens": 180000, "context_config": [{"name": "200K", "tokenCount": "200K", "isDefault": true}, {"tokenCount": "1M"}]});
        assert_eq!(context_tiers(&cfg).len(), 2);
        assert!(resolve_tier(&cfg, "hi", &[], &Value::Null, None).is_none());
        assert_eq!(resolve_tier(&cfg, "hi", &[], &Value::Null, Some("max")).unwrap().0.tokens, 1_000_000);
        assert!(is_billing_block("{\"code\":112}"));
        assert!(!is_billing_block("{\"code\":1}"));
    }

    #[test]
    fn coalescer_merges_usage() {
        let mut co = Coalescer::new("m");
        let mut out = vec![];
        co.handle(r#"{"id":"a","choices":[{"delta":{"content":"hi"}}]}"#, &mut out);
        co.handle(r#"{"choices":[{"delta":{"finish_reason":"stop"}}]}"#, &mut out);
        co.handle(r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}"#, &mut out);
        let s: String = out.iter().map(|b| String::from_utf8_lossy(b).to_string()).collect();
        assert!(s.contains("\"usage\""));
        assert!(s.ends_with("data: [DONE]\n\n"));
        assert!(co.done);
    }
}

/// Routable model ids from the live Qoder catalog (for /v1/models).
pub async fn list_model_ids(provider: &str, creds: &Value) -> Option<Vec<String>> {
    let cat = resolve_models(creds, region_of(provider), false).await;
    if cat.is_null() {
        return None;
    }
    Some(routable_models(&cat).iter().filter_map(|m| m["id"].as_str().map(str::to_owned)).collect())
}
