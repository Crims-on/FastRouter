//! Provider-specific executors (port of 9router open-sse/executors/*).

pub mod antigravity;
pub mod codex;
pub mod commandcode;
pub mod cursor;
pub mod github;
pub mod grok;
pub mod kiro;
pub mod opencode;
pub mod perplexity_web;
pub mod qoder;
pub mod vertex;
pub mod xiaomi;
pub mod zed;

use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use hmac::{Hmac, KeyInit, Mac};
use regex::Regex;
use serde_json::{Value, json};
use sha2::Sha256;

use crate::exec::{
    DefaultExecutor, DynExecutor, ExecArgs, ExecResult, Executor, Headers, ParsedError, Upstream, base_execute, client_for, cred_str,
    default_transform_request,
};
use crate::jsv::{del, now_ms, truthy};

/// getExecutor(provider)
pub fn get_executor(provider: &str) -> DynExecutor {
    match provider {
        "antigravity" => Arc::new(antigravity::Antigravity),
        "azure" => Arc::new(Azure),
        "gemini-cli" => Arc::new(GeminiCli),
        "github" => github::instance(),
        "iflow" => Arc::new(IFlow),
        "qoder" => Arc::new(qoder::Qoder { id: "qoder".into() }),
        "qoder-cn" => Arc::new(qoder::Qoder { id: "qoder-cn".into() }),
        "kiro" => Arc::new(kiro::Kiro),
        "kimchi" => Arc::new(Kimchi),
        "codex" => Arc::new(codex::Codex),
        "cursor" | "cu" => Arc::new(cursor::Cursor),
        "vertex" => Arc::new(vertex::Vertex { id: "vertex".into() }),
        "vertex-partner" => Arc::new(vertex::Vertex { id: "vertex-partner".into() }),
        "opencode" => Arc::new(opencode::OpenCode),
        "opencode-go" => Arc::new(opencode::OpenCodeGo),
        "opencode-zen" => Arc::new(opencode::OpenCodeZen),
        "grok-web" => Arc::new(grok::GrokWeb),
        "grok-cli" | "gcli" | "gb" => Arc::new(grok::GrokCli),
        "perplexity-web" => Arc::new(perplexity_web::PerplexityWeb),
        "ollama-local" => Arc::new(OllamaLocal),
        "commandcode" => Arc::new(commandcode::CommandCode),
        "xiaomi-tokenplan" => Arc::new(XiaomiTokenplan),
        "xiaomi-mimo" => Arc::new(xiaomi::XiaomiMimo),
        "mimo-free" | "mmf" => Arc::new(MimoFree),
        "codebuddy-cn" => Arc::new(CodeBuddy { id: "codebuddy-cn".into(), intl: false }),
        "codebuddy-intl" => Arc::new(CodeBuddy { id: "codebuddy-intl".into(), intl: true }),
        "zed" => Arc::new(zed::Zed),
        other => Arc::new(DefaultExecutor { id: other.to_string() }),
    }
}

// ---------------------------------------------------------------------------
// Azure OpenAI
// ---------------------------------------------------------------------------

pub struct Azure;

#[async_trait]
impl Executor for Azure {
    fn provider(&self) -> &str {
        "azure"
    }
    fn build_url(&self, model: &str, _stream: bool, _idx: usize, creds: &Value) -> Result<String, String> {
        let psd = &creds["providerSpecificData"];
        let endpoint = psd["azureEndpoint"].as_str().filter(|s| !s.is_empty()).map(str::to_owned).or_else(|| std::env::var("AZURE_ENDPOINT").ok()).unwrap_or_else(|| "https://api.openai.com".into());
        let version = psd["apiVersion"].as_str().filter(|s| !s.is_empty()).map(str::to_owned).or_else(|| std::env::var("AZURE_API_VERSION").ok()).unwrap_or_else(|| "2024-10-01-preview".into());
        let deployment = psd["deployment"].as_str().filter(|s| !s.is_empty()).unwrap_or(if model.is_empty() { "gpt-4" } else { model });
        Ok(format!("{}/openai/deployments/{deployment}/chat/completions?api-version={version}", endpoint.trim_end_matches('/')))
    }
    fn build_headers(&self, creds: &Value, stream: bool, _url: &str, _model: &str, _body: &Value) -> Headers {
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.extend_obj(&self.config()["headers"]);
        if let Some(k) = cred_str(creds, "apiKey").or_else(|| cred_str(creds, "accessToken")) {
            h.set("api-key", k);
        }
        if let Some(o) = creds["providerSpecificData"]["organization"].as_str().filter(|s| !s.is_empty()) {
            h.set("OpenAI-Organization", o);
        }
        if stream {
            h.set("Accept", "text/event-stream");
        }
        h
    }
    fn transform_request(&self, _m: &str, body: Value, _s: bool, _c: &Value) -> Value {
        body
    }
}

// ---------------------------------------------------------------------------
// Ollama local / Xiaomi token plan
// ---------------------------------------------------------------------------

pub struct OllamaLocal;

#[async_trait]
impl Executor for OllamaLocal {
    fn provider(&self) -> &str {
        "ollama-local"
    }
    fn build_url(&self, _m: &str, _s: bool, _i: usize, creds: &Value) -> Result<String, String> {
        let host = creds["providerSpecificData"]["baseUrl"].as_str().map(str::trim).filter(|s| !s.is_empty()).unwrap_or("http://localhost:11434");
        Ok(format!("{}/api/chat", host.trim_end_matches('/')))
    }
}

pub struct XiaomiTokenplan;

#[async_trait]
impl Executor for XiaomiTokenplan {
    fn provider(&self) -> &str {
        "xiaomi-tokenplan"
    }
    fn build_url(&self, _m: &str, _s: bool, _i: usize, creds: &Value) -> Result<String, String> {
        let cfg = self.config();
        let region = creds["providerSpecificData"]["region"].as_str().unwrap_or("");
        let base = cfg["regions"][region]
            .as_str()
            .or_else(|| cfg["regions"][cfg["defaultRegion"].as_str().unwrap_or("")].as_str())
            .unwrap_or("https://token-plan-sgp.xiaomimimo.com/v1")
            .to_string();
        if creds["runtimeTransport"]["format"] == "claude" {
            static V1: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/v1/?$").unwrap());
            return Ok(format!("{}/anthropic/v1/messages", V1.replace(&base, "")));
        }
        Ok(format!("{base}/chat/completions"))
    }
}

// ---------------------------------------------------------------------------
// CodeBuddy (Tencent) — cn + intl
// ---------------------------------------------------------------------------

pub struct CodeBuddy {
    pub id: String,
    pub intl: bool,
}

static AGENT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)you are claude code|claude.?code.+official.+cli|anthropic.+official.+cli|anxthxropic.+official.+cli|you are (?:cursor|windsurf|cline|aider|continue|copilot|cody)|you are an? (?:ai )?(?:coding |code )?agent|cc_entrypoint\s*=\s*(?:cli|vscode|jetbrains|gui)|claude.?code.+issues|give feedback.+claude.?code|you are .{0,30}(?:powerful )?ai agent|orchestration capabilities|OhMyOpenCode|<agent-identity>|<Role>|<Behavior_Instructions>").unwrap()
});

fn codebuddy_parse_error(status: u16, body: &str) -> ParsedError {
    if let Ok(d) = serde_json::from_str::<Value>(body) {
        let msg = d["msg"].as_str().or_else(|| d["message"].as_str()).or_else(|| d["error"]["message"].as_str()).unwrap_or("").to_string();
        static LIMIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)超出频率限制|frequency limit|限额").unwrap());
        if d["code"] == json!(6004) || LIMIT.is_match(&msg) {
            return ParsedError { status: 429, message: if msg.is_empty() { "CodeBuddy frequency limit (6004)".into() } else { msg }, resets_at_ms: None };
        }
    }
    ParsedError { status, message: if body.is_empty() { format!("HTTP {status}") } else { body.into() }, resets_at_ms: None }
}

#[async_trait]
impl Executor for CodeBuddy {
    fn provider(&self) -> &str {
        &self.id
    }
    fn transform_request(&self, model: &str, body: Value, _s: bool, _c: &Value) -> Value {
        let mut t = default_transform_request(&self.id, model, body);
        t["stream"] = json!(true);
        let eff = t["reasoning_effort"].as_str().map(str::to_owned);
        match eff.as_deref() {
            Some("none") | Some("off") => {
                del(&mut t, "reasoning_effort");
            }
            Some(_) => t["reasoning_summary"] = json!("auto"),
            None => {}
        }
        if self.intl {
            let src = t["messages"].as_array().cloned().unwrap_or_default();
            let mut out = vec![json!({"role": "system", "content": "You are CodeBuddy Code."})];
            for m in src {
                if !m.is_object() || m["role"] == "system" || m["role"] == "developer" {
                    continue;
                }
                if m["role"] == "user" && m["content"].is_string() {
                    let mut c = m.clone();
                    c["content"] = json!([{"type": "text", "text": m["content"]}]);
                    out.push(c);
                } else {
                    out.push(m);
                }
            }
            t["messages"] = Value::Array(out);
        } else if let Some(msgs) = t["messages"].as_array_mut() {
            const NEUTRAL: &str = "You are a helpful AI assistant that helps with software engineering tasks.";
            for m in msgs.iter_mut() {
                if m["role"] != "system" {
                    continue;
                }
                let text = match &m["content"] {
                    Value::String(s) => s.clone(),
                    Value::Array(a) => a.iter().map(|b| b["text"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>().join("\n"),
                    _ => String::new(),
                };
                if text.is_empty() {
                    continue;
                }
                if text.chars().count() > 2000 || AGENT_PATTERN.is_match(&text) {
                    m["content"] = if m["content"].is_string() { json!(NEUTRAL) } else { json!([{"type": "text", "text": NEUTRAL}]) };
                }
            }
        }
        t
    }
    fn parse_error(&self, status: u16, body: &str) -> ParsedError {
        codebuddy_parse_error(status, body)
    }
}

// ---------------------------------------------------------------------------
// Gemini CLI (Cloud Code Assist)
// ---------------------------------------------------------------------------

pub struct GeminiCli;

pub fn gemini_cli_user_agent(model: &str) -> String {
    let v = crate::consts::s("GEMINI_CLI_VERSION");
    let arch = match crate::consts::node_arch() {
        "ia32" => "x86",
        a => a,
    };
    format!("GeminiCLI/{v}/{} ({}; {arch}; terminal)", if model.is_empty() { "unknown" } else { model }, crate::consts::node_platform())
}

#[async_trait]
impl Executor for GeminiCli {
    fn provider(&self) -> &str {
        "gemini-cli"
    }
    fn build_url(&self, _m: &str, stream: bool, _i: usize, _c: &Value) -> Result<String, String> {
        Ok(format!("{}:{}", self.config()["baseUrl"].as_str().unwrap_or(""), if stream { "streamGenerateContent?alt=sse" } else { "generateContent" }))
    }
    fn build_headers(&self, creds: &Value, stream: bool, _u: &str, model: &str, _b: &Value) -> Headers {
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.set("Authorization", format!("Bearer {}", creds["accessToken"].as_str().unwrap_or("")));
        h.set("User-Agent", gemini_cli_user_agent(model));
        h.set("X-Goog-Api-Client", crate::consts::s("GEMINI_CLI_API_CLIENT"));
        h.set("Accept", if stream { "text/event-stream" } else { "application/json" });
        h
    }
    fn transform_request(&self, model: &str, body: Value, _s: bool, creds: &Value) -> Value {
        if truthy(&body["request"]) && truthy(&body["model"]) {
            return body;
        }
        let project = if truthy(&creds["projectId"]) { creds["projectId"].clone() } else { body["project"].clone() };
        json!({"project": project, "model": model, "request": body})
    }
}

// ---------------------------------------------------------------------------
// iFlow (HMAC-signed)
// ---------------------------------------------------------------------------

pub struct IFlow;

#[async_trait]
impl Executor for IFlow {
    fn provider(&self) -> &str {
        "iflow"
    }
    fn build_url(&self, _m: &str, _s: bool, _i: usize, _c: &Value) -> Result<String, String> {
        Ok(self.config()["baseUrl"].as_str().unwrap_or("").to_string())
    }
    fn build_headers(&self, creds: &Value, stream: bool, _u: &str, _m: &str, _b: &Value) -> Headers {
        let cfg = self.config();
        let session = format!("session-{}", uuid::Uuid::new_v4());
        let ts = now_ms();
        let ua = cfg["headers"]["User-Agent"].as_str().unwrap_or("iFlow-Cli").to_string();
        let key = crate::exec::api_key_or_token(creds);
        let sig = if key.is_empty() {
            String::new()
        } else {
            let mut mac = Hmac::<Sha256>::new_from_slice(key.as_bytes()).unwrap();
            mac.update(format!("{ua}:{session}:{ts}").as_bytes());
            hex::encode(mac.finalize().into_bytes())
        };
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.extend_obj(&cfg["headers"]);
        h.set("session-id", session);
        h.set("x-iflow-timestamp", ts.to_string());
        h.set("x-iflow-signature", sig);
        if let Some(k) = cred_str(creds, "apiKey") {
            h.set("Authorization", format!("Bearer {k}"));
        }
        if stream {
            h.set("Accept", "text/event-stream");
        }
        h
    }
    fn transform_request(&self, _m: &str, mut body: Value, stream: bool, _c: &Value) -> Value {
        if stream && body["messages"].is_array() && !truthy(&body["stream_options"]) {
            body["stream_options"] = json!({"include_usage": true});
        }
        body
    }
}

// ---------------------------------------------------------------------------
// Kimchi
// ---------------------------------------------------------------------------

pub struct Kimchi;

#[async_trait]
impl Executor for Kimchi {
    fn provider(&self) -> &str {
        "kimchi"
    }
    fn transform_request(&self, model: &str, body: Value, _s: bool, _c: &Value) -> Value {
        let mut t = default_transform_request("kimchi", model, body);
        if !t.is_object() {
            return t;
        }
        // merge top-level system
        let sys_text = match &t["system"] {
            Value::String(s) => s.clone(),
            Value::Array(a) => a.iter().map(|p| p.as_str().map(str::to_owned).unwrap_or_else(|| p["text"].as_str().unwrap_or("").to_string())).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n"),
            _ => String::new(),
        };
        let sys_text = sys_text.trim().to_string();
        if !sys_text.is_empty() {
            if let Some(msgs) = t["messages"].as_array_mut() {
                if let Some(existing) = msgs.iter_mut().find(|m| m["role"] == "system") {
                    if let Some(s) = existing["content"].as_str().map(str::to_owned) {
                        existing["content"] = json!(format!("{sys_text}\n\n{s}"));
                    } else if let Some(a) = existing["content"].as_array_mut() {
                        a.insert(0, json!({"type": "text", "text": sys_text}));
                    }
                } else {
                    msgs.insert(0, json!({"role": "system", "content": sys_text}));
                }
            }
        }
        for k in ["anthropic_version", "anthropic_beta", "client_metadata", "mcp_servers", "stop_sequences", "thinking", "top_k", "system"] {
            del(&mut t, k);
        }
        static ANTH: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)(^|[-_/])(?:claude|anthropic)(?:[-_/]|$)").unwrap());
        if ANTH.is_match(model) {
            for k in ["reasoning_effort", "reasoning", "thinking"] {
                del(&mut t, k);
            }
        }
        if let Some(msgs) = t["messages"].as_array_mut() {
            for m in msgs.iter_mut() {
                del(m, "cache_control");
                if let Some(c) = m["content"].as_array_mut() {
                    for p in c.iter_mut() {
                        del(p, "cache_control");
                        del(p, "signature");
                    }
                }
                if m["role"] == "assistant" && m["reasoning_content"].as_str().map(|s| s.chars().count() > 8).unwrap_or(false) {
                    del(m, "reasoning_content");
                }
            }
        }
        if let Some(tools) = t["tools"].as_array_mut() {
            for tool in tools.iter_mut() {
                del(tool, "cache_control");
            }
        }
        t
    }
}

// ---------------------------------------------------------------------------
// MiMo free (bootstrap JWT)
// ---------------------------------------------------------------------------

pub struct MimoFree;

const MIMO_BOOTSTRAP_URL: &str = "https://api.xiaomimimo.com/api/free-ai/bootstrap";
pub const MIMO_SYSTEM_MARKER: &str = "You are MiMoCode, an interactive CLI tool that helps users with software engineering tasks.";
const MIMO_UAS: [&str; 3] = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
];

static MIMO_JWT: LazyLock<std::sync::Mutex<Option<(String, i64)>>> = LazyLock::new(Default::default);
static MIMO_SESSION: LazyLock<String> = LazyLock::new(|| {
    const CH: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let r = uuid::Uuid::new_v4();
    let r2 = uuid::Uuid::new_v4();
    let bytes: Vec<u8> = r.as_bytes().iter().chain(r2.as_bytes()).copied().collect();
    let s: String = bytes.iter().take(24).map(|b| CH[*b as usize % CH.len()] as char).collect();
    format!("ses_{s}")
});

fn random_ua() -> &'static str {
    MIMO_UAS[uuid::Uuid::new_v4().as_bytes()[0] as usize % 3]
}

fn mimo_fingerprint() -> String {
    let user = std::env::var("USER").unwrap_or_else(|_| "unknown-user".into());
    crate::jsv::sha256_hex(&format!("{}|{}|{}|unknown-cpu|{user}", crate::exec::hostname(), crate::consts::node_platform(), crate::consts::node_arch()))
}

fn jwt_exp_ms(jwt: &str) -> i64 {
    use base64::Engine;
    let part = jwt.split('.').nth(1).unwrap_or("");
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(part.trim_end_matches('='))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok());
    decoded.and_then(|v| v["exp"].as_i64()).map(|e| e * 1000).unwrap_or_else(|| now_ms() + 3_000_000)
}

async fn mimo_bootstrap(creds: &Value, force: bool) -> Result<String, String> {
    if !force {
        if let Some((jwt, exp)) = MIMO_JWT.lock().unwrap().clone() {
            if now_ms() < exp - 300_000 {
                return Ok(jwt);
            }
        }
    }
    let resp = client_for(creds)
        .post(MIMO_BOOTSTRAP_URL)
        .header("Content-Type", "application/json")
        .header("User-Agent", random_ua())
        .body(json!({"client": mimo_fingerprint()}).to_string())
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("MiMo bootstrap failed: {}", resp.status().as_u16()));
    }
    let v: Value = resp.json().await.map_err(|e| e.to_string())?;
    let jwt = v["jwt"].as_str().ok_or("MiMo bootstrap returned no JWT")?.to_string();
    *MIMO_JWT.lock().unwrap() = Some((jwt.clone(), jwt_exp_ms(&jwt)));
    Ok(jwt)
}

#[async_trait]
impl Executor for MimoFree {
    fn provider(&self) -> &str {
        "mimo-free"
    }
    fn no_auth(&self) -> bool {
        true
    }
    fn transform_request(&self, _m: &str, body: Value, _s: bool, _c: &Value) -> Value {
        let Some(msgs) = body["messages"].as_array() else { return body };
        if msgs.iter().any(|m| m["role"] == "system" && m["content"].as_str().map(|c| c.contains(MIMO_SYSTEM_MARKER)).unwrap_or(false)) {
            return body;
        }
        let mut b = body.clone();
        let mut out = vec![json!({"role": "system", "content": MIMO_SYSTEM_MARKER})];
        out.extend(msgs.iter().cloned());
        b["messages"] = Value::Array(out);
        b
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let url = self.config()["baseUrl"].as_str().unwrap_or("").to_string();
        let body = self.transform_request(args.model, args.body.clone(), args.stream, args.creds);
        let mut jwt = mimo_bootstrap(args.creds, false).await?;
        for attempt in 0..2 {
            let mut h = Headers::default();
            h.set("Content-Type", "application/json");
            h.set("X-Mimo-Source", "mimocode-cli-free");
            h.set("User-Agent", random_ua());
            h.set("x-session-affinity", MIMO_SESSION.as_str());
            h.set("Accept", if args.stream { "text/event-stream" } else { "application/json" });
            h.set("Authorization", format!("Bearer {jwt}"));
            let mut rb = client_for(args.creds).post(&url);
            for (k, v) in &h.0 {
                rb = rb.header(k, v);
            }
            let resp = rb.body(body.to_string()).send().await.map_err(|e| e.to_string())?;
            let st = resp.status().as_u16();
            if (st == 401 || st == 403) && attempt == 0 {
                jwt = mimo_bootstrap(args.creds, true).await?;
                continue;
            }
            return Ok(ExecResult { response: Upstream::from_reqwest(resp), url, headers: h.0, body, response_format: None });
        }
        unreachable!()
    }
}

// ---------------------------------------------------------------------------
// helpers shared by special executors
// ---------------------------------------------------------------------------

/// Re-encodes a stream of OpenAI chunks as an SSE byte stream.
pub fn sse_from_chunks(chunks: impl futures::Stream<Item = Value> + Send + 'static, done: bool) -> crate::exec::ByteStream {
    let s = chunks.map(|c| Ok::<Bytes, String>(Bytes::from(format!("data: {c}\n\n"))));
    if done {
        s.chain(futures::stream::once(async { Ok(Bytes::from_static(b"data: [DONE]\n\n")) })).boxed()
    } else {
        s.boxed()
    }
}

/// Wrapper used by executors whose upstream speaks a different protocol:
/// converts results into an OpenAI-chunk SSE `Upstream`.
pub fn openai_sse_upstream(status: u16, stream: crate::exec::ByteStream) -> Upstream {
    Upstream::synthetic(status, "text/event-stream", stream)
}


#[allow(dead_code)]
pub async fn passthrough_execute<E: Executor + ?Sized>(ex: &E, args: ExecArgs<'_>) -> Result<ExecResult, String> {
    base_execute(ex, args).await
}
