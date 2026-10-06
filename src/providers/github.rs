//! GitHub Copilot executor (port of executors/github.js).

use std::collections::HashSet;
use std::sync::{Arc, LazyLock, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::consts::ANTHROPIC_API_VERSION;
use crate::exec::{ExecArgs, ExecResult, Executor, Headers, Upstream, base_execute, client_for};
use crate::jsv::{del, now_ms, truthy};
use crate::translate::{self, ReqCtx};

pub struct Github {
    known_codex: Mutex<HashSet<String>>,
}

static INSTANCE: LazyLock<Arc<Github>> = LazyLock::new(|| Arc::new(Github { known_codex: Mutex::new(HashSet::new()) }));

pub fn instance() -> crate::exec::DynExecutor {
    INSTANCE.clone()
}

pub fn copilot_constants() -> (String, String, String, String) {
    let c = &crate::consts::C["GITHUB_COPILOT"];
    (
        c["VSCODE_VERSION"].as_str().unwrap_or("1.110.0").into(),
        c["COPILOT_CHAT_VERSION"].as_str().unwrap_or("0.38.0").into(),
        c["USER_AGENT"].as_str().unwrap_or("GitHubCopilotChat/0.38.0").into(),
        c["API_VERSION"].as_str().unwrap_or("2025-04-01").into(),
    )
}

fn is_claude(model: &str) -> bool {
    model.to_lowercase().contains("claude")
}

fn supports_responses(model: &str) -> bool {
    let m = model.to_lowercase();
    !(m.contains("gemini") || m.contains("claude"))
}

fn sanitize_for_chat(body: &Value) -> Value {
    let mut b = body.clone();
    if let Some(msgs) = b["messages"].as_array_mut() {
        for m in msgs.iter_mut() {
            if !truthy(&m["content"]) || m["content"].is_string() {
                continue;
            }
            if let Some(parts) = m["content"].as_array() {
                let clean: Vec<Value> = parts
                    .iter()
                    .map(|p| {
                        if p["type"] == "text" || p["type"] == "image_url" {
                            return p.clone();
                        }
                        let t = if truthy(&p["text"]) { p["text"].clone() } else if truthy(&p["content"]) { p["content"].clone() } else { json!(p.to_string()) };
                        json!({"type": "text", "text": t.as_str().map(str::to_owned).unwrap_or_else(|| t.to_string())})
                    })
                    .filter(|p| p["text"] != "")
                    .collect();
                m["content"] = if clean.is_empty() { Value::Null } else { Value::Array(clean) };
            }
        }
    }
    b
}

/// Re-encodes an upstream SSE body through a translator into OpenAI chat SSE.
pub fn translate_sse_body(up: Upstream, target: &'static str, model: String, tool_name_map: Option<serde_json::Map<String, Value>>, stream_requested: bool) -> Upstream {
    let mut state = translate::resp::init_state(if target == translate::OPENAI_RESPONSES { "openai-responses" } else { target });
    state["model"] = json!(model);
    if let Some(m) = tool_name_map {
        state["toolNameMap"] = Value::Object(m);
    }
    let status = up.status;
    let headers = up.headers.clone();
    let mut body = up.body;
    let s = async_stream::stream! {
        let mut parser = crate::sse::LineParser::default();
        while let Some(chunk) = body.next().await {
            let Ok(chunk) = chunk else { break };
            for line in parser.push(&chunk) {
                let Some(parsed) = crate::sse::parse_sse_line(&line, None) else { continue };
                if parsed.is_done() {
                    if stream_requested { yield Ok(Bytes::from_static(b"data: [DONE]\n\n")); }
                    continue;
                }
                let v = parsed.into_value();
                let out = if target == translate::OPENAI_RESPONSES {
                    translate::resp::responses_to_openai(Some(&v), &mut state)
                } else {
                    translate::translate_response(target, translate::OPENAI, Some(&v), &mut state)
                };
                for c in out { yield Ok(Bytes::from(translate::format_sse(&c, "openai"))); }
            }
        }
        if let Some(line) = parser.finish() {
            if let Some(parsed) = crate::sse::parse_sse_line(&line, None) {
                if !parsed.is_done() {
                    let v = parsed.into_value();
                    let out = if target == translate::OPENAI_RESPONSES {
                        translate::resp::responses_to_openai(Some(&v), &mut state)
                    } else {
                        translate::translate_response(target, translate::OPENAI, Some(&v), &mut state)
                    };
                    for c in out { yield Ok(Bytes::from(translate::format_sse(&c, "openai"))); }
                }
            }
        }
    };
    Upstream { status, headers, body: s.boxed() }
}

impl Github {
    fn headers(&self, creds: &Value, stream: bool) -> Headers {
        let (vscode, chat, ua, api) = copilot_constants();
        let token = creds["copilotToken"].as_str().filter(|s| !s.is_empty()).or_else(|| creds["accessToken"].as_str()).unwrap_or("");
        let mut h = Headers::default();
        h.set("Authorization", format!("Bearer {token}"));
        h.set("Content-Type", "application/json");
        h.set("copilot-integration-id", "vscode-chat");
        h.set("editor-version", format!("vscode/{vscode}"));
        h.set("editor-plugin-version", format!("copilot-chat/{chat}"));
        h.set("user-agent", ua);
        h.set("openai-intent", "conversation-panel");
        h.set("x-github-api-version", api);
        h.set("x-request-id", uuid::Uuid::new_v4().to_string());
        h.set("x-vscode-user-agent-library-version", "electron-fetch");
        h.set("X-Initiator", "user");
        h.set("anthropic-version", ANTHROPIC_API_VERSION);
        h.set("Accept", if stream { "text/event-stream" } else { "application/json" });
        h
    }

    async fn post(&self, url: &str, h: &Headers, body: &Value, creds: &Value) -> Result<reqwest::Response, String> {
        let mut rb = client_for(creds).post(url);
        for (k, v) in &h.0 {
            rb = rb.header(k, v);
        }
        rb.body(body.to_string()).send().await.map_err(|e| e.to_string())
    }

    async fn with_responses(&self, args: &ExecArgs<'_>) -> Result<ExecResult, String> {
        let url = self.config()["responsesUrl"].as_str().unwrap_or("https://api.githubcopilot.com/responses").to_string();
        let h = self.headers(args.creds, args.stream);
        let body = translate::req::openai_to_responses(args.model, &args.body);
        let resp = self.post(&url, &h, &body, args.creds).await?;
        let up = Upstream::from_reqwest(resp);
        if !up.ok() {
            return Ok(ExecResult { response: up, url, headers: h.0, body, response_format: None });
        }
        let up = translate_sse_body(up, translate::OPENAI_RESPONSES, args.model.to_string(), None, args.stream);
        Ok(ExecResult { response: up, url, headers: h.0, body, response_format: Some("openai".into()) })
    }

    async fn with_messages(&self, args: &ExecArgs<'_>) -> Result<ExecResult, String> {
        let url = self.config()["messagesUrl"].as_str().unwrap_or("https://api.githubcopilot.com/v1/messages").to_string();
        let h = self.headers(args.creds, args.stream);
        let mut rc = ReqCtx { provider: "github".into(), headers: args.creds["rawHeaders"].clone(), ..Default::default() };
        let tr = translate::translate_request(translate::OPENAI, translate::CLAUDE, args.model, &args.body, true, &mut rc);
        let mut body = tr.body;
        body["stream"] = json!(true);
        let resp = self.post(&url, &h, &body, args.creds).await?;
        let up = Upstream::from_reqwest(resp);
        if !up.ok() {
            return Ok(ExecResult { response: up, url, headers: h.0, body, response_format: None });
        }
        let up = translate_sse_body(up, translate::CLAUDE, args.model.to_string(), tr.tool_name_map, args.stream);
        Ok(ExecResult { response: up, url, headers: h.0, body, response_format: Some("openai".into()) })
    }
}

#[async_trait]
impl Executor for Github {
    fn provider(&self) -> &str {
        "github"
    }
    fn build_url(&self, _m: &str, _s: bool, _i: usize, _c: &Value) -> Result<String, String> {
        Ok(self.config()["baseUrl"].as_str().unwrap_or("").to_string())
    }
    fn build_headers(&self, creds: &Value, stream: bool, _u: &str, _m: &str, _b: &Value) -> Headers {
        self.headers(creds, stream)
    }
    fn transform_request(&self, model: &str, body: Value, _s: bool, _c: &Value) -> Value {
        let mut t = body;
        let re = regex::Regex::new(r"(?i)gpt-5|o[134]-").unwrap();
        if re.is_match(model) && !t["max_tokens"].is_null() {
            t["max_completion_tokens"] = t["max_tokens"].clone();
            del(&mut t, "max_tokens");
        }
        if t["reasoning_effort"] == "none" {
            del(&mut t, "reasoning_effort");
        }
        crate::translate::concerns::strip_unsupported_params("github", model, &mut t);
        t
    }
    async fn refresh_credentials(&self, creds: &Value) -> Option<Value> {
        crate::oauth::refresh::refresh_github_copilot(creds).await
    }
    async fn execute(&self, mut args: ExecArgs<'_>) -> Result<ExecResult, String> {
        // Copilot needs a short-lived Copilot token minted from the GitHub token.
        let needs = !truthy(&args.creds["copilotToken"]) || {
            let exp = &args.creds["copilotTokenExpiresAt"];
            let ms = exp.as_i64().map(|e| if e < 1_000_000_000_000 { e * 1000 } else { e });
            ms.map(|m| m - now_ms() < 300_000).unwrap_or(false)
        };
        if needs && truthy(&args.creds["accessToken"]) {
            if let Some(r) = self.refresh_credentials(args.creds).await {
                crate::oauth::refresh::merge_into(args.creds, &r);
                args.creds["__refreshed"] = json!(true);
            }
        }
        let model = args.model.to_string();
        if is_claude(&model) {
            return self.with_messages(&args).await;
        }
        if self.known_codex.lock().unwrap().contains(&model) && supports_responses(&model) {
            return self.with_responses(&args).await;
        }
        let original = args.body.clone();
        args.body = sanitize_for_chat(&original);
        let mut res = base_execute(self, ExecArgs { model: args.model, body: args.body.clone(), stream: args.stream, creds: args.creds, session_id: args.session_id.clone(), client_tool: args.client_tool.clone(), override_headers: args.override_headers.clone() }).await?;
        if res.response.status == 400 && supports_responses(&model) {
            let text = std::mem::replace(&mut res.response, Upstream::json(400, &json!({}))).text().await;
            if text.contains("not accessible via the /chat/completions endpoint") || text.contains("The requested model is not supported") {
                self.known_codex.lock().unwrap().insert(model.clone());
                args.body = original;
                return self.with_responses(&args).await;
            }
            res.response = Upstream::synthetic(400, "application/json", futures::stream::once(async move { Ok(Bytes::from(text)) }).boxed());
        }
        Ok(res)
    }
}
