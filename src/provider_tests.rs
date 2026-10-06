//! Full-pipeline tests for every registry provider.
//!
//! Every outbound hostname is resolved to a local mock that speaks HTTPS with
//! a self-signed certificate (see `exec::TEST_UPSTREAM`). Each provider's real
//! executor, URL, headers, translation and response parsing run unchanged; the
//! mock answers in whatever wire format the request path implies.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Bytes;
use axum::http::{HeaderMap, Method};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::AppState;
use crate::db::Db;
use crate::registry::REG;

pub const REPLY: &str = "pong";

#[derive(Clone, Debug)]
pub struct Hit {
    pub host: String,
    pub path: String,
    pub headers: HeaderMap,
    pub body: Bytes,
}

pub static HITS: Mutex<Vec<Hit>> = Mutex::new(Vec::new());

fn sse(events: Vec<String>) -> Response {
    ([("content-type", "text/event-stream")], events.concat()).into_response()
}

fn data(v: Value) -> String {
    format!("data: {v}\n\n")
}

fn gemini_chunk(text: &str, last: bool) -> Value {
    let mut c = json!({"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "index": 0}], "modelVersion": "m"});
    if last {
        c["candidates"][0]["finishReason"] = json!("STOP");
        c["usageMetadata"] = json!({"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4});
    }
    c
}

/// Builds a reply in the wire format the request path implies.
pub fn respond(path: &str, query: &str, body: &Value) -> Response {
    let stream = body["stream"] == json!(true) || path.contains("streamGenerateContent") || query.contains("alt=sse");
    if path.contains(":loadCodeAssist") {
        return axum::Json(json!({"cloudaicompanionProject": "p", "currentTier": {"id": "free-tier"}, "allowedTiers": [{"id": "free-tier", "isDefault": true}]})).into_response();
    }
    if path.contains(":onboardUser") {
        return axum::Json(json!({"done": true, "response": {"cloudaicompanionProject": {"id": "p"}}})).into_response();
    }
    if path.to_ascii_lowercase().contains("generatecontent") {
        let wrap = path.contains("v1internal");
        let w = |v: Value| if wrap { json!({"response": v, "traceId": "t"}) } else { v };
        if stream {
            return sse(vec![data(w(gemini_chunk("po", false))), data(w(gemini_chunk("ng", true)))]);
        }
        return axum::Json(w(gemini_chunk(REPLY, true))).into_response();
    }
    if path.ends_with("/messages") {
        if stream {
            let ev = |name: &str, v: Value| format!("event: {name}\ndata: {v}\n\n");
            return sse(vec![
                ev("message_start", json!({"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "content": [], "usage": {"input_tokens": 3, "output_tokens": 0}}})),
                ev("content_block_start", json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})),
                ev("content_block_delta", json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": REPLY}})),
                ev("content_block_stop", json!({"type": "content_block_stop", "index": 0})),
                ev("message_delta", json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}})),
                ev("message_stop", json!({"type": "message_stop"})),
            ]);
        }
        return axum::Json(json!({"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "content": [{"type": "text", "text": REPLY}], "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 1}})).into_response();
    }
    if path.ends_with("/responses") {
        let item = json!({"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": REPLY, "annotations": []}]});
        let done = json!({"id": "resp_1", "object": "response", "status": "completed", "model": body["model"], "output": [item], "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}});
        if stream {
            let ev = |name: &str, v: Value| format!("event: {name}\ndata: {v}\n\n");
            return sse(vec![
                ev("response.created", json!({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "model": body["model"], "output": []}})),
                ev("response.output_item.added", json!({"type": "response.output_item.added", "output_index": 0, "item": {"id": "msg_1", "type": "message", "role": "assistant", "content": []}})),
                ev("response.output_text.delta", json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": REPLY})),
                ev("response.output_item.done", json!({"type": "response.output_item.done", "output_index": 0, "item": item})),
                ev("response.completed", json!({"type": "response.completed", "response": done})),
            ]);
        }
        return axum::Json(done).into_response();
    }
    if path.ends_with("/api/chat") {
        let lines = [json!({"model": body["model"], "message": {"role": "assistant", "content": REPLY}, "done": false}), json!({"model": body["model"], "message": {"role": "assistant", "content": ""}, "done": true, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 1})];
        if stream {
            return ([("content-type", "application/x-ndjson")], lines.iter().map(|l| format!("{l}\n")).collect::<String>()).into_response();
        }
        return axum::Json(json!({"model": body["model"], "message": {"role": "assistant", "content": REPLY}, "done": true, "prompt_eval_count": 3, "eval_count": 1})).into_response();
    }
    // OpenAI chat completions (default).
    if stream {
        return sse(vec![
            data(json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": body["model"], "choices": [{"index": 0, "delta": {"role": "assistant", "content": "po"}, "finish_reason": null}]})),
            data(json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": body["model"], "choices": [{"index": 0, "delta": {"content": "ng"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})),
            "data: [DONE]\n\n".into(),
        ]);
    }
    axum::Json(json!({"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})).into_response()
}

async fn handler(req: axum::extract::Request) -> Response {
    let (parts, body) = req.into_parts();
    let (method, uri, headers) = (parts.method, parts.uri, parts.headers);
    let host = uri.host().map(str::to_owned).or_else(|| headers.get("host").and_then(|h| h.to_str().ok()).map(str::to_owned)).unwrap_or_default();
    // Bidirectional streams: answer before the request body ends.
    if let Some(r) = crate::provider_mocks::early(uri.path()) {
        HITS.lock().unwrap().push(Hit { host, path: uri.path().to_string(), headers, body: Bytes::new() });
        return r;
    }
    let body = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
    let hit = Hit { host, path: uri.path().to_string(), headers, body };
    if std::env::var("MOCK_DEBUG").is_ok() {
        eprintln!("MOCK {method} {} {}?{} {}", hit.host, uri.path(), uri.query().unwrap_or(""), String::from_utf8_lossy(&hit.body).chars().take(300).collect::<String>());
    }
    HITS.lock().unwrap().push(hit.clone());
    if let Some(r) = crate::provider_mocks::special(&method, &hit) {
        return r;
    }
    if method == Method::GET {
        return axum::Json(json!({"object": "list", "data": [{"id": "m-1"}], "models": []})).into_response();
    }
    let v: Value = serde_json::from_slice(&hit.body).unwrap_or(Value::Null);
    respond(uri.path(), uri.query().unwrap_or(""), &v)
}

async fn serve_conn(stream: tokio::net::TcpStream, tls: tokio_rustls::TlsAcceptor, app: Router) {
    use hyper_util::rt::{TokioExecutor, TokioIo};
    let svc = hyper_util::service::TowerToHyperService::new(app);
    let mut first = [0u8; 1];
    if stream.peek(&mut first).await.is_err() {
        return;
    }
    let b = hyper_util::server::conn::auto::Builder::new(TokioExecutor::new());
    if first[0] == 0x16 {
        if let Ok(s) = tls.accept(stream).await {
            let _ = b.serve_connection_with_upgrades(TokioIo::new(s), svc).await;
        }
    } else {
        let _ = b.serve_connection_with_upgrades(TokioIo::new(stream), svc).await;
    }
}

/// Starts (once) the mock upstream and installs the resolver hook. It runs
/// on its own thread and runtime so it outlives any single test.
pub async fn mock_upstream() -> SocketAddr {
    static ADDR: std::sync::OnceLock<SocketAddr> = std::sync::OnceLock::new();
    *ADDR.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
            rt.block_on(async move {
                let ck = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
                let key = tokio_rustls::rustls::pki_types::PrivateKeyDer::Pkcs8(ck.signing_key.serialize_der().into());
                let mut cfg = tokio_rustls::rustls::ServerConfig::builder_with_provider(Arc::new(tokio_rustls::rustls::crypto::ring::default_provider()))
                    .with_safe_default_protocol_versions()
                    .unwrap()
                    .with_no_client_auth()
                    .with_single_cert(vec![ck.cert.der().clone()], key)
                    .unwrap();
                cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
                let tls = tokio_rustls::TlsAcceptor::from(Arc::new(cfg));
                let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                tx.send(l.local_addr().unwrap()).unwrap();
                let app = Router::new().fallback(handler);
                while let Ok((s, _)) = l.accept().await {
                    tokio::spawn(serve_conn(s, tls.clone(), app.clone()));
                }
            });
        });
        let addr = rx.recv().unwrap();
        let _ = crate::exec::TEST_UPSTREAM.set(addr);
        addr
    })
}

pub fn fake_connection(id: &str) -> Value {
    let far = "2099-01-01T00:00:00.000Z";
    let auth = REG.entry(id).and_then(|e| e["authType"].as_str()).unwrap_or("apikey");
    json!({
        "provider": id, "authType": if auth == "oauth" { "oauth" } else { "apikey" }, "name": "t", "isActive": true,
        "apiKey": "k-123", "accessToken": "t-123", "refreshToken": "r-123", "expiresAt": far, "projectId": "p",
        "copilotToken": "c-123", "copilotTokenExpiresAt": 4_070_908_800i64,
        "providerSpecificData": {"projectId": "p", "region": "us-east-1", "accountId": "acc", "resourceName": "res", "deployment": "dep", "machineId": "m", "userId": "u", "systemId": "s", "profileArn": "arn:aws:codewhisperer:us-east-1:1:profile/x", "cookie": "a=b", "copilotToken": "c-123", "copilotTokenExpiresAt": 4_070_908_800i64},
    })
}

pub async fn router_with(db: Arc<Db>) -> String {
    let state = AppState { db, config: Arc::new(crate::config::Config { host: "127.0.0.1".into(), port: 0, data_dir: std::env::temp_dir(), initial_password: "x".into(), require_api_key: Some(false) }) };
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, crate::app(state)).await.unwrap() });
    format!("http://{addr}")
}

fn llm_model(id: &str) -> String {
    REG.models_by_provider_id(id).iter().find(|m| m["kind"].is_null() && m["type"].is_null()).and_then(|m| m["id"].as_str()).unwrap_or("test-model").to_string()
}

fn content_of(stream: bool, text: &str) -> String {
    if !stream {
        let v: Value = serde_json::from_str(text).unwrap_or(Value::Null);
        return v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
    }
    let mut out = String::new();
    for line in text.lines() {
        if let Some(d) = line.strip_prefix("data: ") {
            if let Ok(v) = serde_json::from_str::<Value>(d) {
                out.push_str(v["choices"][0]["delta"]["content"].as_str().unwrap_or(""));
            }
        }
    }
    out
}

/// Chats through one provider (non-streaming and streaming); `Ok` when the
/// mock's reply comes back intact.
pub async fn chat_through(base: &str, provider: &str) -> Result<(), String> {
    let alias = REG.alias_of(provider);
    let prefix = if REG.resolve_alias(&alias) == provider { alias } else { provider.to_string() };
    let model = format!("{prefix}/{}", llm_model(provider));
    let c = reqwest::Client::builder().no_proxy().build().unwrap();
    for stream in [false, true] {
        let r = c.post(format!("{base}/v1/chat/completions")).json(&json!({"model": model, "stream": stream, "messages": [{"role": "user", "content": "Say pong"}]})).send().await.map_err(|e| e.to_string())?;
        let st = r.status().as_u16();
        let t = r.text().await.unwrap_or_default();
        let got = content_of(stream, &t);
        if stream && std::env::var("MOCK_DEBUG").is_ok() {
            eprintln!("OPENAI-STREAM {provider}:\n{t}");
        }
        if st != 200 || !got.contains(REPLY) {
            return Err(format!("stream={stream} status={st} body={}", t.chars().take(300).collect::<String>()));
        }
        if stream && t.matches("data: [DONE]").count() != 1 {
            return Err(format!("expected exactly one [DONE]: {}", t.chars().take(600).collect::<String>()));
        }
    }
    // Anthropic Messages client (streaming).
    let r = c.post(format!("{base}/v1/messages")).json(&json!({"model": model, "max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "Say pong"}]})).send().await.map_err(|e| e.to_string())?;
    let st = r.status().as_u16();
    let t = r.text().await.unwrap_or_default();
    let text: String = t.lines().filter_map(|l| l.strip_prefix("data: ")).filter_map(|d| serde_json::from_str::<Value>(d).ok()).filter_map(|v| v["delta"]["text"].as_str().map(str::to_owned)).collect();
    if st != 200 || !text.contains(REPLY) || !t.contains("message_stop") {
        return Err(format!("claude client status={st} body={}", t.chars().take(if std::env::var("MOCK_DEBUG").is_ok() { 5000 } else { 300 }).collect::<String>()));
    }
    // OpenAI Responses client (non-streaming).
    let r = c.post(format!("{base}/v1/responses")).json(&json!({"model": model, "stream": false, "input": "Say pong"})).send().await.map_err(|e| e.to_string())?;
    let st = r.status().as_u16();
    let t = r.text().await.unwrap_or_default();
    if st != 200 || !t.contains(REPLY) {
        return Err(format!("responses client status={st} body={}", t.chars().take(300).collect::<String>()));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_llm_provider_end_to_end() {
    mock_upstream().await;
    let mut failures = vec![];
    let mut passed = 0;
    for e in &REG.entries {
        let id = e["id"].as_str().unwrap();
        if let Ok(only) = std::env::var("ONLY_PROVIDER") {
            if only.split(',').all(|o| o != id) {
                continue;
            }
        }
        if !crate::api::models::provider_kinds(id).iter().any(|k| k == "llm") {
            continue;
        }
        let db = Arc::new(Db::open_in_memory().unwrap());
        if !crate::chat::accounts::is_free_no_auth(id) {
            let mut c = fake_connection(id);
            crate::provider_mocks::adjust_connection(id, &mut c);
            db.insert_connection(&c).unwrap();
        }
        let base = router_with(db).await;
        match tokio::time::timeout(std::time::Duration::from_secs(30), chat_through(&base, id)).await {
            Ok(Ok(())) => passed += 1,
            Ok(Err(e)) => failures.push(format!("{id}: {e}")),
            Err(_) => failures.push(format!("{id}: timeout")),
        }
    }
    eprintln!("{passed} providers passed");
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}


fn kind_model(id: &str, kind: &str) -> String {
    let want = match kind {
        "webSearch" | "webFetch" => return String::new(),
        k => k,
    };
    if let Some(m) = REG.models_by_provider_id(id).iter().find(|m| m["kind"] == want || m["type"] == want).and_then(|m| m["id"].as_str()) {
        return m.to_string();
    }
    let cfg_key = match kind {
        "tts" => "ttsConfig",
        "stt" => "sttConfig",
        "image" => "imageConfig",
        "embedding" => "embeddingConfig",
        "video" => "videoConfig",
        _ => "",
    };
    let cfg = crate::mediaapi::media_cfg(id, cfg_key);
    cfg["defaultModel"].as_str().or_else(|| cfg["models"][0]["id"].as_str()).unwrap_or("test-model").to_string()
}

/// Calls one media endpoint through FastRouter for `id`.
pub async fn media_through(base: &str, id: &str, kind: &str) -> Result<(), String> {
    let alias = REG.alias_of(id);
    let prefix = if REG.resolve_alias(&alias) == id { alias } else { id.to_string() };
    let model = format!("{prefix}/{}", kind_model(id, kind));
    let c = reqwest::Client::builder().no_proxy().build().unwrap();
    let post = |path: &str, body: Value| c.post(format!("{base}{path}")).json(&body);
    let r = match kind {
        "embedding" => post("/v1/embeddings", json!({"model": model, "input": "hello"})),
        "image" => post("/v1/images/generations", json!({"model": model, "prompt": "a cat", "response_format": "b64_json", "image": if id == "topaz" { json!("data:image/png;base64,iVBORw0KGgo=") } else { Value::Null }})),
        "tts" => post("/v1/audio/speech", json!({"model": model, "input": "hello", "voice": "alloy"})),
        "stt" => {
            let form = reqwest::multipart::Form::new().text("model", model.clone()).part("file", reqwest::multipart::Part::bytes(vec![1u8; 64]).file_name("a.wav").mime_str("audio/wav").unwrap());
            c.post(format!("{base}/v1/audio/transcriptions")).multipart(form)
        }
        "webSearch" => post("/v1/search", json!({"provider": prefix, "query": "rust axum"})),
        "webFetch" => post("/v1/web/fetch", json!({"provider": prefix, "url": "https://example.com/page"})),
        "video" => post("/v1/videos/generations", json!({"model": model, "prompt": "a cat"})),
        "systemone" => post("/v1/systemone", json!({"model": model, "state": {"goal": "test"}, "questions": {"q1": "hello"}})),
        other => return Err(format!("unknown kind {other}")),
    };
    let r = r.send().await.map_err(|e| e.to_string())?;
    let st = r.status().as_u16();
    let ct = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let bytes = r.bytes().await.unwrap_or_default();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let ok = st == 200
        && match kind {
            "embedding" => v["data"][0]["embedding"].is_array(),
            "image" => v["data"][0]["b64_json"].is_string() || v["data"][0]["url"].is_string(),
            "tts" => !ct.contains("json") && bytes.len() > 10 || v["audio"].is_string(),
            "stt" => v["text"].is_string(),
            "webSearch" => v["results"].as_array().is_some_and(|a| !a.is_empty()) || v["answer"]["text"].as_str().is_some_and(|s| !s.is_empty()),
            "webFetch" => v["content"]["text"].as_str().is_some_and(|s| !s.is_empty()),
            _ => true,
        };
    if ok { Ok(()) } else { Err(format!("status={st} ct={ct} body={}", text.chars().take(300).collect::<String>())) }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_media_provider_end_to_end() {
    mock_upstream().await;
    let mut failures = vec![];
    let mut passed = 0;
    for e in &REG.entries {
        let id = e["id"].as_str().unwrap();
        if let Ok(only) = std::env::var("ONLY_PROVIDER") {
            if only.split(',').all(|o| o != id) {
                continue;
            }
        }
        for kind in crate::api::models::provider_kinds(id) {
            if kind == "llm" || kind == "imageToText" {
                continue;
            }
            let has_engine = ["espeak-ng", "espeak", "say"].iter().any(|b| std::process::Command::new(b).arg("--version").output().is_ok());
            if id == "local-device" && !has_engine {
                eprintln!("skipping local-device: no espeak-ng/say on this machine");
                continue;
            }
            let db = Arc::new(Db::open_in_memory().unwrap());
            let mut c = fake_connection(id);
            crate::provider_mocks::adjust_connection(id, &mut c);
            db.insert_connection(&c).unwrap();
            let base = router_with(db).await;
            match tokio::time::timeout(std::time::Duration::from_secs(30), media_through(&base, id, &kind)).await {
                Ok(Ok(())) => passed += 1,
                Ok(Err(e)) => failures.push(format!("{id} [{kind}]: {e}")),
                Err(_) => failures.push(format!("{id} [{kind}]: timeout")),
            }
        }
    }
    eprintln!("{passed} provider/kind pairs passed");
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
