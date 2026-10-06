//! Upstream executors (port of 9router open-sse/executors/base.js + default.js).
//! An executor turns a translated body into an HTTP call for one provider.
//! Credentials are a JSON object mirroring 9router's `credentials`
//! (`apiKey`, `accessToken`, `refreshToken`, `expiresAt`, `projectId`,
//! `providerSpecificData`, ...).

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{Value, json};

use crate::consts::ANTHROPIC_API_VERSION;
use crate::jsv::{js_string, now_ms, truthy};
use crate::registry::REG;

pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>>;

/// A provider response: status, headers and a byte stream body.
pub struct Upstream {
    pub status: u16,
    pub headers: reqwest::header::HeaderMap,
    pub body: ByteStream,
}

impl Upstream {
    pub fn from_reqwest(r: reqwest::Response) -> Self {
        let status = r.status().as_u16();
        let headers = r.headers().clone();
        let body = r.bytes_stream().map(|c| c.map_err(|e| e.to_string())).boxed();
        Upstream { status, headers, body }
    }

    pub fn synthetic(status: u16, content_type: &str, body: ByteStream) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Ok(v) = content_type.parse() {
            headers.insert(reqwest::header::CONTENT_TYPE, v);
        }
        Upstream { status, headers, body }
    }

    pub fn json(status: u16, v: &Value) -> Self {
        let b = Bytes::from(v.to_string());
        Self::synthetic(status, "application/json", futures::stream::once(async move { Ok(b) }).boxed())
    }

    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub fn content_type(&self) -> String {
        self.headers.get(reqwest::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
    }

    pub async fn bytes(self) -> Result<Bytes, String> {
        let mut out = Vec::new();
        let mut s = self.body;
        while let Some(c) = s.next().await {
            out.extend_from_slice(&c?);
        }
        Ok(Bytes::from(out))
    }

    pub async fn text(self) -> String {
        self.bytes().await.map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default()
    }
}

pub struct ExecArgs<'a> {
    pub model: &'a str,
    pub body: Value,
    pub stream: bool,
    pub creds: &'a mut Value,
    pub session_id: Option<String>,
    pub client_tool: Option<String>,
    pub override_headers: Option<Value>,
}

pub struct ExecResult {
    pub response: Upstream,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
    /// Format of the (possibly re-encoded) response body when it differs from the target format.
    pub response_format: Option<String>,
}

pub enum RetryDelay {
    Veto,
    Default,
    Ms(u64),
}

pub struct ParsedError {
    pub status: u16,
    pub message: String,
    pub resets_at_ms: Option<i64>,
}

// ---------------------------------------------------------------------------
// HTTP clients (optionally per proxy URL)
// ---------------------------------------------------------------------------

static CLIENTS: LazyLock<Mutex<HashMap<String, reqwest::Client>>> = LazyLock::new(Default::default);

pub fn http_client(proxy: Option<&str>) -> reqwest::Client {
    let key = proxy.unwrap_or("").to_string();
    let mut map = CLIENTS.lock().unwrap();
    if let Some(c) = map.get(&key) {
        return c.clone();
    }
    let mut b = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(90))
        .redirect(reqwest::redirect::Policy::limited(10));
    if let Some(p) = proxy.filter(|p| !p.is_empty()) {
        if let Ok(px) = reqwest::Proxy::all(p) {
            b = b.proxy(px);
        }
    }
    let c = b.build().expect("http client");
    map.insert(key, c.clone());
    c
}

/// Client for a connection, honouring `providerSpecificData.connectionProxyUrl`.
pub fn client_for(creds: &Value) -> reqwest::Client {
    let psd = &creds["providerSpecificData"];
    if psd["connectionProxyEnabled"] == json!(true) {
        if let Some(u) = psd["connectionProxyUrl"].as_str().filter(|u| !u.is_empty()) {
            return http_client(Some(u));
        }
    }
    http_client(None)
}

pub fn no_redirect_client() -> reqwest::Client {
    static C: LazyLock<reqwest::Client> = LazyLock::new(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap()
    });
    C.clone()
}

// ---------------------------------------------------------------------------
// Header helpers
// ---------------------------------------------------------------------------

/// Ordered, case-insensitive header list (JS objects keep insertion order).
#[derive(Default, Clone, Debug)]
pub struct Headers(pub Vec<(String, String)>);

impl Headers {
    pub fn set(&mut self, k: &str, v: impl Into<String>) {
        let v = v.into();
        if let Some(e) = self.0.iter_mut().find(|(a, _)| a == k) {
            e.1 = v;
        } else {
            self.0.push((k.to_string(), v));
        }
    }
    pub fn get(&self, k: &str) -> Option<&str> {
        self.0.iter().find(|(a, _)| a.eq_ignore_ascii_case(k)).map(|(_, v)| v.as_str())
    }
    pub fn has_exact(&self, k: &str) -> bool {
        self.0.iter().any(|(a, _)| a == k)
    }
    pub fn remove(&mut self, k: &str) {
        self.0.retain(|(a, _)| a != k);
    }
    pub fn extend_obj(&mut self, v: &Value) {
        if let Some(o) = v.as_object() {
            for (k, val) in o {
                self.set(k, js_string(val));
            }
        }
    }
}

pub fn set_auth(h: &mut Headers, spec: &Value, token: &str) {
    let header = spec["header"].as_str().unwrap_or("Authorization");
    let scheme = spec["scheme"].as_str().unwrap_or("bearer");
    let v = if scheme == "bearer" { format!("Bearer {token}") } else { token.to_string() };
    h.set(header, v);
}

pub fn cred_str<'a>(creds: &'a Value, k: &str) -> Option<&'a str> {
    creds[k].as_str().filter(|s| !s.is_empty())
}

pub fn api_key_or_token(creds: &Value) -> String {
    cred_str(creds, "apiKey").or_else(|| cred_str(creds, "accessToken")).unwrap_or("").to_string()
}

fn apply_auth(h: &mut Headers, desc: &Value, creds: &Value) {
    if desc["combined"] == json!(true) {
        let token = cred_str(creds, "apiKey").or_else(|| cred_str(creds, "accessToken")).unwrap_or("undefined");
        set_auth(h, desc, token);
        if desc["anthropicVersion"] == json!(true) && !h.has_exact("anthropic-version") {
            h.set("anthropic-version", ANTHROPIC_API_VERSION);
        }
        return;
    }
    if let Some(k) = cred_str(creds, "apiKey") {
        set_auth(h, &desc["apiKey"], k);
    } else if let Some(t) = cred_str(creds, "accessToken") {
        set_auth(h, &desc["oauth"], t);
    }
    if desc["anthropicVersion"] == json!(true) && !h.has_exact("anthropic-version") {
        h.set("anthropic-version", ANTHROPIC_API_VERSION);
    }
}

pub fn kimi_headers(device_id: Option<&str>) -> Vec<(String, String)> {
    let model = match std::env::consts::OS {
        "macos" => format!("macOS {}", crate::consts::node_arch()),
        "windows" => format!("Windows {}", crate::consts::node_arch()),
        "linux" => format!("Linux {}", crate::consts::node_arch()),
        o => format!("{o} {}", crate::consts::node_arch()),
    };
    let host = hostname();
    let id = device_id.map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).unwrap_or_else(|| format!("kimi-{}", now_ms()));
    vec![
        ("X-Msh-Platform".into(), "9router".into()),
        ("X-Msh-Version".into(), env!("CARGO_PKG_VERSION").into()),
        ("X-Msh-Device-Name".into(), host),
        ("X-Msh-Device-Model".into(), model),
        ("X-Msh-Device-Id".into(), id),
    ]
}

pub fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

pub fn cline_access_token(token: &str) -> String {
    let t = token.trim();
    if t.is_empty() {
        return String::new();
    }
    if t.to_lowercase().starts_with("workos:") {
        return t.to_string();
    }
    static JWT: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+").unwrap());
    if JWT.is_match(t) { format!("workos:{t}") } else { t.to_string() }
}

pub fn cline_headers(token: &str) -> Vec<(String, String)> {
    let v = env!("CARGO_PKG_VERSION");
    let mut h = vec![
        ("HTTP-Referer".to_string(), "https://cline.bot".to_string()),
        ("X-Title".into(), "Cline".into()),
        ("User-Agent".into(), format!("9Router/{v}")),
        ("X-PLATFORM".into(), crate::consts::node_platform().into()),
        ("X-PLATFORM-VERSION".into(), "v22.0.0".into()),
        ("X-CLIENT-TYPE".into(), "9router".into()),
        ("X-CLIENT-VERSION".into(), v.into()),
        ("X-CORE-VERSION".into(), v.into()),
        ("X-IS-MULTIROOT".into(), "false".into()),
    ];
    let at = cline_access_token(token);
    if !at.is_empty() {
        h.push(("Authorization".into(), format!("Bearer {at}")));
    }
    h
}

// ---------------------------------------------------------------------------
// Anthropic beta flags (providers/shared.js)
// ---------------------------------------------------------------------------

pub fn select_anthropic_beta(model: &str, body: Option<&Value>) -> String {
    let wants_summary = body.map(|b| b["thinking"]["display"] == "summarized").unwrap_or(false);
    let mut flags: Vec<&str> = vec![
        "claude-code-20250219",
        "oauth-2025-04-20",
        "interleaved-thinking-2025-05-14",
        "context-management-2025-06-27",
        "prompt-caching-scope-2026-01-05",
        "structured-outputs-2025-12-15",
        "fast-mode-2026-02-01",
        "redact-thinking-2026-02-12",
        "token-efficient-tools-2026-03-28",
    ];
    if wants_summary {
        flags.retain(|f| *f != "redact-thinking-2026-02-12");
    }
    if model.starts_with("claude-opus") || model.starts_with("claude-sonnet") {
        flags.push("advanced-tool-use-2025-11-20");
        flags.push("effort-2025-11-24");
    }
    flags.join(",")
}

pub fn merge_anthropic_beta(values: &[Option<&str>]) -> String {
    let mut out: Vec<String> = vec![];
    for v in values.iter().flatten() {
        for f in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if !out.iter().any(|x| x == f) {
                out.push(f.to_string());
            }
        }
    }
    out.join(",")
}

// ---------------------------------------------------------------------------
// Retry config
// ---------------------------------------------------------------------------

fn retry_entry(cfg: &Value, status: u16) -> (u32, u64) {
    let defaults = json!({"429": {"attempts": 0, "delayMs": 0}, "502": {"attempts": 3, "delayMs": 3000}, "503": {"attempts": 3, "delayMs": 2000}, "504": {"attempts": 2, "delayMs": 3000}});
    let key = status.to_string();
    let e = if !cfg["retry"][&key].is_null() { &cfg["retry"][&key] } else { &defaults[&key] };
    match e {
        Value::Null => (0, 2000),
        Value::Number(n) => (n.as_u64().unwrap_or(0) as u32, 2000),
        o => (o["attempts"].as_u64().unwrap_or(0) as u32, o["delayMs"].as_u64().unwrap_or(2000)),
    }
}

// ---------------------------------------------------------------------------
// Executor trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Executor: Send + Sync {
    fn provider(&self) -> &str;

    fn config(&self) -> Value {
        let p = self.provider();
        let t = REG.transport(p);
        if t.is_null() { REG.transport("openai").clone() } else { t.clone() }
    }

    fn no_auth(&self) -> bool {
        self.config()["noAuth"] == json!(true)
    }

    fn base_urls(&self) -> Vec<String> {
        let c = self.config();
        if let Some(a) = c["baseUrls"].as_array() {
            return a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect();
        }
        c["baseUrl"].as_str().map(|s| vec![s.to_string()]).unwrap_or_default()
    }

    fn build_url(&self, model: &str, stream: bool, idx: usize, creds: &Value) -> Result<String, String> {
        default_build_url(self.provider(), &self.config(), model, stream, idx, creds)
    }

    fn build_headers(&self, creds: &Value, stream: bool, url: &str, model: &str, body: &Value) -> Headers {
        default_build_headers(self.provider(), &self.config(), creds, stream, url, model, body)
    }

    fn transform_request(&self, model: &str, body: Value, _stream: bool, _creds: &Value) -> Value {
        default_transform_request(self.provider(), model, body)
    }

    fn parse_error(&self, status: u16, body: &str) -> ParsedError {
        ParsedError { status, message: if body.is_empty() { format!("HTTP {status}") } else { body.to_string() }, resets_at_ms: None }
    }

    fn should_retry(&self, status: u16, idx: usize) -> bool {
        status == 429 && idx + 1 < self.base_urls().len().max(1)
    }

    /// Whether `compute_retry_delay` should be consulted (buffers the error body).
    fn has_retry_hook(&self) -> bool {
        false
    }

    /// computeRetryDelay hook: derive the wait from the failed response.
    fn compute_retry_delay(&self, _status: u16, _headers: &reqwest::header::HeaderMap, _body: &str, _attempt: u32, _default_ms: u64) -> RetryDelay {
        RetryDelay::Default
    }

    async fn refresh_credentials(&self, creds: &Value) -> Option<Value> {
        crate::oauth::refresh::refresh_for_provider(self.provider(), creds).await
    }

    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        base_execute(self, args).await
    }
}

/// BaseExecutor.execute: URL fallback list + per-status retries.
pub async fn base_execute<E: Executor + ?Sized>(ex: &E, args: ExecArgs<'_>) -> Result<ExecResult, String> {
    let cfg = ex.config();
    let n = ex.base_urls().len().max(1);
    let mut attempts: HashMap<usize, u32> = HashMap::new();
    let mut last_err = String::new();
    let mut idx = 0usize;
    let timeout_ms = cfg["timeoutMs"].as_u64().unwrap_or(60_000);
    while idx < n {
        let url = ex.build_url(args.model, args.stream, idx, args.creds)?;
        let body = ex.transform_request(args.model, args.body.clone(), args.stream, args.creds);
        let mut headers = ex.build_headers(args.creds, args.stream, &url, args.model, &body);
        if let Some(o) = &args.override_headers {
            headers.extend_obj(o);
        }
        let client = client_for(args.creds);
        let mut rb = client.post(&url);
        for (k, v) in &headers.0 {
            rb = rb.header(k.as_str(), v.as_str());
        }
        let fut = rb.body(body.to_string()).send();
        match tokio::time::timeout(Duration::from_millis(timeout_ms), fut).await {
            Ok(Ok(resp)) => {
                let status = resp.status().as_u16();
                let (max, delay) = retry_entry(&cfg, status);
                let used = *attempts.entry(idx).or_insert(0);
                let mut up = Upstream::from_reqwest(resp);
                if max > 0 && used < max && !(200..300).contains(&status) {
                    let mut wait = Some(delay);
                    if ex.has_retry_hook() {
                        let hdrs = up.headers.clone();
                        let bytes = up.bytes().await.unwrap_or_default();
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        wait = match ex.compute_retry_delay(status, &hdrs, &text, used + 1, delay) {
                            RetryDelay::Veto => None,
                            RetryDelay::Default => Some(delay),
                            RetryDelay::Ms(ms) => Some(ms),
                        };
                        up = Upstream { status, headers: hdrs, body: futures::stream::once(async move { Ok(bytes) }).boxed() };
                    }
                    if let Some(w) = wait {
                        attempts.insert(idx, used + 1);
                        tracing::debug!("{} retry {}/{} after status {status}", ex.provider(), used + 1, max);
                        tokio::time::sleep(Duration::from_millis(w)).await;
                        continue;
                    }
                }
                if ex.should_retry(status, idx) {
                    idx += 1;
                    continue;
                }
                return Ok(ExecResult { response: up, url, headers: headers.0, body, response_format: None });
            }
            Ok(Err(e)) => {
                last_err = e.to_string();
                let (max, delay) = retry_entry(&cfg, 502);
                let used = attempts.entry(idx).or_insert(0);
                if *used < max {
                    *used += 1;
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    continue;
                }
                if idx + 1 < n {
                    idx += 1;
                    continue;
                }
                return Err(last_err);
            }
            Err(_) => {
                last_err = "fetch connect timeout".into();
                if idx + 1 < n {
                    idx += 1;
                    continue;
                }
                return Err(last_err);
            }
        }
    }
    Err(if last_err.is_empty() { format!("All {n} URLs failed") } else { last_err })
}

// ---------------------------------------------------------------------------
// Default behaviour (DefaultExecutor)
// ---------------------------------------------------------------------------

pub fn is_openai_compatible(p: &str) -> bool {
    p.starts_with("openai-compatible-")
}
pub fn is_anthropic_compatible(p: &str) -> bool {
    p.starts_with("anthropic-compatible-")
}

/// chat vs responses for openai-compatible custom nodes.
pub fn openai_compatible_api_type(provider: &str, creds: &Value) -> &'static str {
    match creds["providerSpecificData"]["apiType"].as_str() {
        Some("responses") => "responses",
        Some("chat") => "chat",
        _ => {
            if provider.contains("responses") { "responses" } else { "chat" }
        }
    }
}

pub fn default_build_url(provider: &str, cfg: &Value, model: &str, stream: bool, idx: usize, creds: &Value) -> Result<String, String> {
    let rt = &creds["runtimeTransport"];
    if let Some(b) = rt["baseUrl"].as_str().filter(|s| !s.is_empty()) {
        return Ok(format!("{b}{}", rt["urlSuffix"].as_str().unwrap_or("")));
    }
    if is_openai_compatible(provider) {
        let base = creds["providerSpecificData"]["baseUrl"].as_str().filter(|s| !s.is_empty()).unwrap_or(crate::consts::s("OPENAI_COMPAT_BASE"));
        let path = if openai_compatible_api_type(provider, creds) == "responses" { "/responses" } else { "/chat/completions" };
        return Ok(format!("{}{path}", base.trim_end_matches('/')));
    }
    if is_anthropic_compatible(provider) {
        let base = creds["providerSpecificData"]["baseUrl"].as_str().filter(|s| !s.is_empty()).unwrap_or(crate::consts::s("ANTHROPIC_COMPAT_BASE"));
        return Ok(format!("{}/messages", base.trim_end_matches('/')));
    }
    let base_urls: Vec<String> = cfg["baseUrls"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default();
    let base = base_urls.get(idx).cloned().or_else(|| base_urls.first().cloned()).or_else(|| cfg["baseUrl"].as_str().map(str::to_owned)).unwrap_or_default();
    if cfg["format"] == "gemini" {
        return Ok(format!("{base}/{model}:{}", if stream { "streamGenerateContent?alt=sse" } else { "generateContent" }));
    }
    if let Some(sfx) = cfg["urlSuffix"].as_str() {
        return Ok(format!("{base}{sfx}"));
    }
    if base.contains("{accountId}") {
        let acct = creds["providerSpecificData"]["accountId"].as_str().filter(|s| !s.is_empty()).ok_or_else(|| format!("{provider} requires accountId in providerSpecificData"))?;
        return Ok(base.replace("{accountId}", acct));
    }
    Ok(base)
}

pub fn default_build_headers(provider: &str, cfg: &Value, creds: &Value, stream: bool, _url: &str, model: &str, body: &Value) -> Headers {
    let rt = &creds["runtimeTransport"];
    let mut h = Headers::default();
    h.set("Content-Type", "application/json");
    h.extend_obj(if rt.is_object() { &rt["headers"] } else { &cfg["headers"] });
    let desc = if rt.is_object() && rt["auth"].is_object() {
        rt["auth"].clone()
    } else if cfg["auth"].is_object() && !is_openai_compatible(provider) && !is_anthropic_compatible(provider) {
        cfg["auth"].clone()
    } else if is_anthropic_compatible(provider) {
        json!({"apiKey": {"header": "x-api-key", "scheme": "raw"}, "oauth": {"header": "Authorization", "scheme": "bearer"}, "anthropicVersion": true})
    } else if cfg["format"] == "claude" && !is_openai_compatible(provider) {
        json!({"combined": true, "header": "x-api-key", "scheme": "raw", "anthropicVersion": true})
    } else {
        json!({"combined": true, "header": "Authorization", "scheme": "bearer"})
    };
    for hook in desc["hooks"].as_array().into_iter().flatten() {
        match hook.as_str().unwrap_or("") {
            "kimiHeaders" => {
                for (k, v) in kimi_headers(creds["providerSpecificData"]["deviceId"].as_str()) {
                    h.set(&k, v);
                }
            }
            "museHeaders" => {
                if cred_str(creds, "accessToken").is_some() && cred_str(creds, "apiKey").is_none() {
                    h.set("x-api-version", "1.0.0");
                }
            }
            "clineHeaders" => {
                for (k, v) in cline_headers(&api_key_or_token(creds)) {
                    h.set(&k, v);
                }
            }
            "kilocodeOrg" => {
                if let Some(o) = creds["providerSpecificData"]["orgId"].as_str().filter(|s| !s.is_empty()) {
                    h.set("X-Kilocode-OrganizationID", o);
                }
            }
            _ => {}
        }
    }
    apply_auth(&mut h, &desc, creds);

    let is_claude_model = model.starts_with("claude-");
    let client_beta = creds["rawHeaders"]["anthropic-beta"].as_str();
    if !model.is_empty() && (provider == "claude" || (is_anthropic_compatible(provider) && is_claude_model)) {
        h.set("Anthropic-Beta", merge_anthropic_beta(&[Some(&select_anthropic_beta(model, Some(body))), client_beta]));
    } else if provider == "anthropic" && client_beta.is_some() {
        let existing = h.get("Anthropic-Beta").map(str::to_owned);
        h.set("Anthropic-Beta", merge_anthropic_beta(&[existing.as_deref(), client_beta]));
    }
    if provider == "claude" && !h.has_exact("x-claude-code-session-id") {
        let token = api_key_or_token(creds);
        if token.contains("sk-ant-oat") {
            if let Some(sid) = crate::cloak::extract_claude_session_id_from_user_id(&body["metadata"]["user_id"]) {
                h.set("x-claude-code-session-id", sid);
            }
        }
    }
    if is_anthropic_compatible(provider) {
        let base = creds["providerSpecificData"]["baseUrl"].as_str().unwrap_or("");
        let official = base.is_empty() || base.contains("api.anthropic.com");
        if !official {
            if let Some(k) = cred_str(creds, "apiKey") {
                if h.get("Authorization").is_none() {
                    h.set("Authorization", format!("Bearer {k}"));
                }
            }
            for k in ["anthropic-dangerous-direct-browser-access", "Anthropic-Dangerous-Direct-Browser-Access", "x-app", "X-App"] {
                h.remove(k);
            }
            for k in ["anthropic-beta", "Anthropic-Beta"] {
                if let Some(v) = h.0.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone()) {
                    let f: Vec<&str> = v.split(',').map(str::trim).filter(|s| !s.is_empty() && *s != "claude-code-20250219").collect();
                    if f.is_empty() {
                        h.remove(k);
                    } else {
                        h.set(k, f.join(","));
                    }
                }
            }
        }
    }
    if stream {
        h.set("Accept", "text/event-stream");
    }
    h
}

/// DefaultExecutor.transformRequest: json_schema fallback, quirks, param stripping, reasoning injection.
pub fn default_transform_request(provider: &str, model: &str, body: Value) -> Value {
    let mut b = json_schema_fallback(provider, body);
    if b.is_object() {
        if REG.transport(provider)["quirks"]["dropClientMetadata"] == json!(true) {
            crate::jsv::del(&mut b, "client_metadata");
        }
        crate::translate::concerns::strip_unsupported_params(provider, model, &mut b);
    }
    inject_reasoning_content(provider, model, b)
}

fn json_schema_fallback(provider: &str, body: Value) -> Value {
    if !is_openai_compatible(provider) {
        return body;
    }
    let rf = &body["response_format"];
    if rf["type"] != "json_schema" || !truthy(&rf["json_schema"]["schema"]) {
        return body;
    }
    let schema = serde_json::to_string_pretty(&rf["json_schema"]["schema"]).unwrap_or_default();
    let prompt = format!("You must respond with valid JSON that strictly follows this JSON schema:\n```json\n{schema}\n```\nRespond ONLY with the JSON object, no other text.");
    let mut out = body.clone();
    let mut msgs = body["messages"].as_array().cloned().unwrap_or_default();
    if let Some(sys) = msgs.iter_mut().find(|m| m["role"] == "system") {
        if let Some(s) = sys["content"].as_str().map(str::to_owned) {
            sys["content"] = json!(format!("{s}\n\n{prompt}"));
        } else if let Some(a) = sys["content"].as_array_mut() {
            a.push(json!({"type": "text", "text": format!("\n\n{prompt}")}));
        }
    } else {
        msgs.insert(0, json!({"role": "system", "content": prompt}));
    }
    out["messages"] = Value::Array(msgs);
    out["response_format"] = json!({"type": "json_object"});
    out
}

/// utils/reasoningContentInjector.js
pub fn inject_reasoning_content(provider: &str, model: &str, body: Value) -> Value {
    let provider_rule = REG.transport(provider)["reasoningInject"]["scope"].as_str().map(str::to_owned);
    let model_rule = if model.to_lowercase().starts_with("kimi-") {
        Some("toolCalls".to_string())
    } else if model.to_lowercase().contains("deepseek") {
        Some("all".to_string())
    } else {
        None
    };
    let rule = provider_rule.or(model_rule);
    let mut b = body;
    if provider == "deepseek" && (model == "deepseek-v4-pro-max" || model == "deepseek-v4-pro-none") {
        let max = model.ends_with("-max");
        b["model"] = json!("deepseek-v4-pro");
        b["extra_body"]["thinking"]["type"] = json!(if max { "enabled" } else { "disabled" });
        if max {
            b["reasoning_effort"] = json!("max");
        } else {
            crate::jsv::del(&mut b, "reasoning_effort");
        }
    }
    let Some(scope) = rule else { return b };
    if let Some(msgs) = b["messages"].as_array_mut() {
        for m in msgs.iter_mut() {
            if m["role"] != "assistant" {
                continue;
            }
            if m["reasoning_content"].as_str().map(|s| !s.is_empty()).unwrap_or(false) {
                continue;
            }
            if scope == "toolCalls" && !m["tool_calls"].as_array().map(|a| !a.is_empty()).unwrap_or(false) {
                continue;
            }
            m["reasoning_content"] = json!(" ");
        }
    }
    b
}

/// Generic executor driven entirely by the registry transport.
pub struct DefaultExecutor {
    pub id: String,
}

#[async_trait]
impl Executor for DefaultExecutor {
    fn provider(&self) -> &str {
        &self.id
    }
}

pub type DynExecutor = Arc<dyn Executor>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_headers_and_urls() {
        let creds = json!({"apiKey": "k"});
        let cfg = REG.transport("groq").clone();
        assert_eq!(default_build_url("groq", &cfg, "m", true, 0, &creds).unwrap(), "https://api.groq.com/openai/v1/chat/completions");
        let h = default_build_headers("groq", &cfg, &creds, true, "", "m", &json!({}));
        assert_eq!(h.get("Authorization"), Some("Bearer k"));
        let gcfg = REG.transport("gemini").clone();
        assert!(default_build_url("gemini", &gcfg, "gemini-2.5-flash", true, 0, &creds).unwrap().ends_with("gemini-2.5-flash:streamGenerateContent?alt=sse"));
        let gh = default_build_headers("gemini", &gcfg, &creds, false, "", "x", &json!({}));
        assert_eq!(gh.get("x-goog-api-key"), Some("k"));
    }
}

/// One POST with headers + JSON body under the provider's connect timeout.
pub async fn post_json(creds: &Value, url: &str, headers: &Headers, body: &Value, timeout_ms: u64) -> Result<Upstream, String> {
    post_raw(creds, url, headers, body.to_string().into_bytes(), timeout_ms).await
}

pub async fn post_raw(creds: &Value, url: &str, headers: &Headers, body: Vec<u8>, timeout_ms: u64) -> Result<Upstream, String> {
    let mut rb = client_for(creds).post(url);
    for (k, v) in &headers.0 {
        rb = rb.header(k.as_str(), v.as_str());
    }
    match tokio::time::timeout(Duration::from_millis(timeout_ms.max(1)), rb.body(body).send()).await {
        Ok(Ok(r)) => Ok(Upstream::from_reqwest(r)),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("fetch connect timeout".into()),
    }
}

/// node-machine-id's machineIdSync(): sha256 of the OS machine id.
pub fn machine_id_raw() -> String {
    static ID: LazyLock<String> = LazyLock::new(|| {
        let raw = ["/var/lib/dbus/machine-id", "/etc/machine-id"]
            .iter()
            .find_map(|p| std::fs::read_to_string(p).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()))
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        crate::jsv::sha256_hex(&raw)
    });
    ID.clone()
}

/// getConsistentMachineId(salt)
pub fn consistent_machine_id(salt: &str) -> String {
    crate::jsv::sha256_hex(&format!("{}{salt}", machine_id_raw()))[..16].to_string()
}
