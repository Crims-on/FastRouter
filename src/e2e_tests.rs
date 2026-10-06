//! End-to-end tests: the real router in front of a mock upstream that speaks
//! OpenAI chat, Anthropic messages, OpenAI Responses and Gemini.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use serde_json::{Value, json};

use crate::AppState;
use crate::db::Db;

#[derive(Clone, Default)]
struct Mock {
    hits: Arc<AtomicUsize>,
    last: Arc<std::sync::Mutex<Value>>,
}

fn last_user_text(b: &Value) -> String {
    let m = b["messages"].as_array().and_then(|a| a.iter().rev().find(|m| m["role"] == "user")).cloned().unwrap_or(json!({}));
    match &m["content"] {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

fn sse(events: Vec<String>) -> Response {
    ([("content-type", "text/event-stream")], events.join("")).into_response()
}

fn key_of(h: &HeaderMap) -> String {
    h.get("authorization").and_then(|v| v.to_str().ok()).map(|s| s.trim_start_matches("Bearer ").to_string()).or_else(|| h.get("x-api-key").and_then(|v| v.to_str().ok()).map(str::to_owned)).unwrap_or_default()
}

async fn openai_chat(State(m): State<Mock>, h: HeaderMap, body: axum::Json<Value>) -> Response {
    m.hits.fetch_add(1, Ordering::SeqCst);
    *m.last.lock().unwrap() = body.0.clone();
    if key_of(&h) == "bad" {
        return (StatusCode::TOO_MANY_REQUESTS, axum::Json(json!({"error": {"message": "Rate limit reached"}}))).into_response();
    }
    let text = format!("echo: {}", last_user_text(&body));
    if body["tools"].is_array() && body["tool_choice"] == "required" {
        let tc = json!({"id": "call_1", "type": "function", "function": {"name": body["tools"][0]["function"]["name"], "arguments": "{\"q\":\"x\"}"}});
        if body["stream"] == json!(true) {
            return sse(vec![
                format!("data: {}\n\n", json!({"id": "chatcmpl-abcdefgh", "model": body["model"], "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": tc["function"]["name"], "arguments": ""}}]}}]})),
                format!("data: {}\n\n", json!({"id": "chatcmpl-abcdefgh", "model": body["model"], "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"q\":\"x\"}"}}]}}]})),
                format!("data: {}\n\n", json!({"id": "chatcmpl-abcdefgh", "model": body["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})),
                "data: [DONE]\n\n".into(),
            ]);
        }
        return axum::Json(json!({"id": "chatcmpl-abcdefgh", "object": "chat.completion", "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [tc]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})).into_response();
    }
    if body["stream"] == json!(true) {
        let mut ev = vec![];
        for (i, piece) in [&text[..text.len() / 2], &text[text.len() / 2..]].iter().enumerate() {
            let mut d = json!({"content": piece});
            if i == 0 {
                d["role"] = json!("assistant");
            }
            ev.push(format!("data: {}\n\n", json!({"id": "chatcmpl-abcdefgh", "object": "chat.completion.chunk", "created": 1, "model": body["model"], "choices": [{"index": 0, "delta": d, "finish_reason": null}]})));
        }
        ev.push(format!("data: {}\n\n", json!({"id": "chatcmpl-abcdefgh", "object": "chat.completion.chunk", "created": 1, "model": body["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}})));
        ev.push("data: [DONE]\n\n".into());
        return sse(ev);
    }
    axum::Json(json!({"id": "chatcmpl-abcdefgh", "object": "chat.completion", "created": 1, "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}})).into_response()
}

async fn anthropic_messages(State(m): State<Mock>, h: HeaderMap, body: axum::Json<Value>) -> Response {
    m.hits.fetch_add(1, Ordering::SeqCst);
    *m.last.lock().unwrap() = body.0.clone();
    assert!(!key_of(&h).is_empty());
    let text = format!("claude: {}", last_user_text(&body));
    if body["stream"] == json!(true) {
        let e = |t: &str, d: Value| format!("event: {t}\ndata: {d}\n\n");
        return sse(vec![
            e("message_start", json!({"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "content": [], "usage": {"input_tokens": 9, "output_tokens": 1}}})),
            e("content_block_start", json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})),
            e("content_block_delta", json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})),
            e("content_block_stop", json!({"type": "content_block_stop", "index": 0})),
            e("message_delta", json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 6}})),
            e("message_stop", json!({"type": "message_stop"})),
        ]);
    }
    axum::Json(json!({"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": {"input_tokens": 9, "output_tokens": 6}})).into_response()
}

async fn responses(State(m): State<Mock>, body: axum::Json<Value>) -> Response {
    m.hits.fetch_add(1, Ordering::SeqCst);
    *m.last.lock().unwrap() = body.0.clone();
    let resp = json!({"id": "resp_1", "object": "response", "status": "completed", "model": body["model"], "output": [{"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "resp ok"}]}], "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}});
    if body["stream"] == json!(true) {
        let e = |t: &str, d: Value| format!("event: {t}\ndata: {d}\n\n");
        return sse(vec![
            e("response.created", json!({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}})),
            e("response.output_item.added", json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}})),
            e("response.output_text.delta", json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "resp ok"})),
            e("response.output_item.done", json!({"type": "response.output_item.done", "output_index": 0, "item": resp["output"][0]})),
            e("response.completed", json!({"type": "response.completed", "response": resp})),
        ]);
    }
    axum::Json(resp).into_response()
}

async fn gemini(State(m): State<Mock>, Path(p): Path<String>, body: axum::Json<Value>) -> Response {
    m.hits.fetch_add(1, Ordering::SeqCst);
    *m.last.lock().unwrap() = body.0.clone();
    let chunk = json!({"candidates": [{"content": {"role": "model", "parts": [{"text": "gem ok"}]}, "finishReason": "STOP", "index": 0}], "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6}});
    if p.contains("streamGenerateContent") {
        return sse(vec![format!("data: {chunk}\n\n")]);
    }
    axum::Json(chunk).into_response()
}

async fn start(router: Router) -> String {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
    format!("http://{addr}")
}

struct Env {
    base: String,
    mock: Mock,
    db: Arc<Db>,
    up: String,
}

async fn env() -> Env {
    let mock = Mock::default();
    let up = start(
        Router::new()
            .route("/v1/chat/completions", post(openai_chat))
            .route("/v1/messages", post(anthropic_messages))
            .route("/v1/responses", post(responses))
            .route("/v1/models", axum::routing::get(|| async { axum::Json(json!({"object": "list", "data": [{"id": "m-1", "object": "model"}]})) }))
            .route("/v1beta/models/{p}", post(gemini))
            .route("/v1/embeddings", post(|b: axum::Json<Value>| async move { axum::Json(json!({"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}], "model": b["model"], "usage": {"prompt_tokens": 3, "total_tokens": 3}})) }))
            .route("/v1/images/generations", post(|b: axum::Json<Value>| async move { axum::Json(json!({"created": 1, "data": [{"b64_json": "aGVsbG8=", "revised_prompt": b["prompt"]}]})) }))
            .route("/v1/audio/speech", post(|b: axum::Json<Value>| async move { assert_eq!(b["voice"], "nova"); ([("content-type", "audio/mpeg")], vec![7u8; 200]) }))
            .route("/v1/audio/transcriptions", post(|mut mp: axum::extract::Multipart| async move {
                let mut model = String::new();
                let mut size = 0;
                while let Some(f) = mp.next_field().await.unwrap() {
                    if f.name() == Some("model") { model = f.text().await.unwrap(); } else { size = f.bytes().await.unwrap().len(); }
                }
                axum::Json(json!({"text": format!("{model}:{size}")}))
            }))
            .with_state(mock.clone()),
    )
    .await;
    let db = Arc::new(Db::open_in_memory().unwrap());
    db.upsert_node("openai-compatible-t1", "openai-compatible", &json!({"name": "Mock", "prefix": "mock", "baseUrl": format!("{up}/v1"), "apiType": "chat"})).unwrap();
    db.insert_connection(&json!({"provider": "openai-compatible-t1", "authType": "apikey", "apiKey": "good", "name": "m1", "providerSpecificData": {"baseUrl": format!("{up}/v1"), "prefix": "mock"}})).unwrap();
    db.upsert_node("anthropic-compatible-t2", "anthropic-compatible", &json!({"name": "MockA", "prefix": "ant", "baseUrl": format!("{up}/v1")})).unwrap();
    db.insert_connection(&json!({"provider": "anthropic-compatible-t2", "authType": "apikey", "apiKey": "k", "name": "a1", "providerSpecificData": {"baseUrl": format!("{up}/v1")}})).unwrap();
    db.upsert_node("openai-compatible-responses-t3", "openai-compatible", &json!({"name": "MockR", "prefix": "resp", "baseUrl": format!("{up}/v1"), "apiType": "responses"})).unwrap();
    db.insert_connection(&json!({"provider": "openai-compatible-responses-t3", "authType": "apikey", "apiKey": "k", "providerSpecificData": {"baseUrl": format!("{up}/v1"), "apiType": "responses"}})).unwrap();
    let state = AppState { db: db.clone(), config: Arc::new(crate::config::Config { host: "127.0.0.1".into(), port: 0, data_dir: std::env::temp_dir(), initial_password: "x".into(), require_api_key: Some(false) }) };
    let base = start(crate::app(state)).await;
    Env { base, mock, db, up }
}

async fn post_json(url: &str, body: Value) -> (u16, String) {
    let r = reqwest::Client::new().post(url).json(&body).send().await.unwrap();
    (r.status().as_u16(), r.text().await.unwrap())
}

#[tokio::test]
async fn openai_client_to_openai_compatible() {
    let e = env().await;
    let (s, t) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "mock/llama", "messages": [{"role": "user", "content": "hello"}], "stream": false})).await;
    assert_eq!(s, 200, "{t}");
    let v: Value = serde_json::from_str(&t).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "echo: hello");
    assert_eq!(e.mock.last.lock().unwrap()["model"], "llama");
    let (s, t) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "mock/llama", "messages": [{"role": "user", "content": "hello"}], "stream": true})).await;
    assert_eq!(s, 200);
    assert!(t.contains("\"content\":\"ech"), "{t}");
    assert!(t.trim_end().ends_with("data: [DONE]"), "{t}");
    // usage recorded
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(e.db.recent_usage(10, 0).len() >= 2);
}

#[tokio::test]
async fn claude_client_to_openai_compatible() {
    let e = env().await;
    let (s, t) = post_json(&format!("{}/v1/messages", e.base), json!({"model": "mock/llama", "max_tokens": 100, "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "stream": true})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("event: message_start"), "{t}");
    assert!(t.contains("echo: hi") || t.contains("\"text\":\"ech"), "{t}");
    assert!(t.contains("event: message_stop"), "{t}");
    let (s, t) = post_json(&format!("{}/v1/messages", e.base), json!({"model": "mock/llama", "max_tokens": 100, "messages": [{"role": "user", "content": "yo"}], "stream": false})).await;
    assert_eq!(s, 200, "{t}");
    let v: Value = serde_json::from_str(&t).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["content"][0]["text"], "echo: yo");
}

#[tokio::test]
async fn openai_client_to_anthropic_compatible() {
    let e = env().await;
    let (s, t) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "ant/claude-sonnet-4", "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "x"}], "stream": true})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("claude: x"), "{t}");
    assert!(t.contains("[DONE]"));
    let up = e.mock.last.lock().unwrap().clone();
    assert!(up["system"].is_array() || up["system"].is_string(), "{up}");
    let (s, t) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "ant/claude-sonnet-4", "messages": [{"role": "user", "content": "x"}]})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("claude: x"), "{t}");
}

#[tokio::test]
async fn responses_client_and_upstream() {
    let e = env().await;
    // Responses client → chat upstream
    let (s, t) = post_json(&format!("{}/v1/responses", e.base), json!({"model": "mock/llama", "input": "hey", "stream": true})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("response.completed"), "{t}");
    assert!(t.contains("echo: hey") || t.contains("ech"), "{t}");
    // Chat client → responses upstream (non-stream)
    let (s, t) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "resp/gpt-x", "messages": [{"role": "user", "content": "q"}], "stream": false})).await;
    assert_eq!(s, 200, "{t}");
    let v: Value = serde_json::from_str(&t).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "resp ok", "{t}");
    // Responses client → responses upstream (stream passthrough)
    let (s, t) = post_json(&format!("{}/v1/responses", e.base), json!({"model": "resp/gpt-x", "input": "q", "stream": true})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("response.completed"), "{t}");
}

#[tokio::test]
async fn account_fallback_and_locks() {
    let e = env().await;
    let bad = e.db.insert_connection(&json!({"provider": "openai-compatible-t1", "apiKey": "bad", "name": "bad", "priority": 0, "providerSpecificData": {"baseUrl": format!("{}/v1", e.up)}})).unwrap();
    let (s, t) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "mock/llama", "messages": [{"role": "user", "content": "a"}], "stream": false})).await;
    assert_eq!(s, 200, "{t}");
    let c = e.db.get_connection(&bad).unwrap();
    assert!(c["modelLock_llama"].is_string(), "{c}");
    assert_eq!(c["testStatus"], "unavailable");
    let hits = e.mock.hits.load(Ordering::SeqCst);
    // locked account is skipped next time
    let (s, _) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "mock/llama", "messages": [{"role": "user", "content": "b"}], "stream": false})).await;
    assert_eq!(s, 200);
    assert_eq!(e.mock.hits.load(Ordering::SeqCst), hits + 1);
}

#[tokio::test]
async fn combo_falls_through() {
    let e = env().await;
    e.db.upsert_combo("best", &["openai/gpt-4o".to_string(), "mock/llama".to_string()]).unwrap();
    let (s, t) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "best", "messages": [{"role": "user", "content": "c"}], "stream": false})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("echo: c"));
    let (s, t) = post_json(&format!("{}/v1/models", e.base), json!({})).await;
    let _ = (s, t);
    let r = reqwest::get(format!("{}/v1/models", e.base)).await.unwrap().text().await.unwrap();
    assert!(r.contains("\"best\""), "{r}");
}

#[tokio::test]
async fn gemini_and_ollama_clients() {
    let e = env().await;
    let (s, t) = post_json(&format!("{}/v1beta/models/mock/llama:generateContent", e.base), json!({"contents": [{"role": "user", "parts": [{"text": "gq"}]}]})).await;
    assert_eq!(s, 200, "{t}");
    let v: Value = serde_json::from_str(&t).unwrap();
    assert_eq!(v["candidates"][0]["content"]["parts"][0]["text"], "echo: gq", "{t}");
    let (s, t) = post_json(&format!("{}/v1beta/models/mock/llama:streamGenerateContent?alt=sse", e.base), json!({"contents": [{"role": "user", "parts": [{"text": "gs"}]}]})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("candidates") && !t.contains("[DONE]"), "{t}");
    let (s, t) = post_json(&format!("{}/api/chat", e.base), json!({"model": "mock/llama", "messages": [{"role": "user", "content": "ol"}]})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("\"done\":true"), "{t}");
    let (s, t) = post_json(&format!("{}/api/chat", e.base), json!({"model": "mock/llama", "stream": false, "messages": [{"role": "user", "content": "ol"}]})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("echo: ol"), "{t}");
}

#[tokio::test]
async fn tool_calls_roundtrip_claude_client() {
    let e = env().await;
    let body = json!({"model": "mock/llama", "max_tokens": 50, "stream": true, "tool_choice": {"type": "any"}, "tools": [{"name": "search", "description": "s", "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}], "messages": [{"role": "user", "content": "find"}]});
    let (s, t) = post_json(&format!("{}/v1/messages", e.base), body).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("\"type\":\"tool_use\""), "{t}");
    assert!(t.contains("\"stop_reason\":\"tool_use\""), "{t}");
}

#[tokio::test]
async fn errors_and_auth() {
    let e = env().await;
    let (s, t) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "groq/llama", "messages": [{"role": "user", "content": "a"}]})).await;
    assert_eq!(s, 404, "{t}");
    assert!(t.contains("No active credentials"));
    e.db.set_setting_json("settings", &json!({"requireApiKey": true})).unwrap();
    let (s, _) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "mock/llama", "messages": [{"role": "user", "content": "a"}]})).await;
    assert_eq!(s, 401);
}

#[tokio::test]
async fn media_endpoints_via_custom_node() {
    let e = env().await;
    let (s, t) = post_json(&format!("{}/v1/embeddings", e.base), json!({"model": "mock/emb", "input": "hi"})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("0.1"));
    let (s, t) = post_json(&format!("{}/v1/images/generations", e.base), json!({"model": "mock/img", "prompt": "cat"})).await;
    assert_eq!(s, 200, "{t}");
    assert!(t.contains("aGVsbG8="));
    let r = reqwest::Client::new().post(format!("{}/v1/images/generations?response_format=binary", e.base)).json(&json!({"model": "mock/img", "prompt": "cat"})).send().await.unwrap();
    assert_eq!(r.headers()["content-type"], "image/png");
    assert_eq!(r.bytes().await.unwrap().as_ref(), b"hello");
    let r = reqwest::Client::new().post(format!("{}/v1/audio/speech", e.base)).json(&json!({"model": "mock/tts-1/nova", "input": "hello"})).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.bytes().await.unwrap().len(), 200);
    let form = reqwest::multipart::Form::new().text("model", "mock/whisper-1").part("file", reqwest::multipart::Part::bytes(vec![1u8; 42]).file_name("a.wav"));
    let r = reqwest::Client::new().post(format!("{}/v1/audio/transcriptions", e.base)).multipart(form).send().await.unwrap();
    let t = r.text().await.unwrap();
    assert!(t.contains("whisper-1:42"), "{t}");
    // search: unknown provider and validation
    let (s, _) = post_json(&format!("{}/v1/search", e.base), json!({"provider": "nope", "query": "x"})).await;
    assert_eq!(s, 400);
    let (s, t) = post_json(&format!("{}/v1/web/fetch", e.base), json!({"provider": "firecrawl", "url": "http://127.0.0.1/x"})).await;
    assert_eq!(s, 400, "{t}");
    let r = reqwest::get(format!("{}/v1/models/embedding", e.base)).await.unwrap().text().await.unwrap();
    assert!(r.contains("\"object\":\"list\""), "{r}");
}

#[tokio::test]
async fn dashboard_flow() {
    let e = env().await;
    crate::auth::bootstrap(&e.db, "pw123456").unwrap();
    let c = reqwest::Client::builder().cookie_store(true).redirect(reqwest::redirect::Policy::none()).build().unwrap();
    // Unauthenticated → login.
    let r = c.get(format!("{}/dashboard", e.base)).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 303);
    let r = c.post(format!("{}/login", e.base)).form(&[("password", "pw123456")]).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 303);
    for p in ["/dashboard", "/dashboard/providers", "/dashboard/providers/claude", "/dashboard/providers/kiro", "/dashboard/combos", "/dashboard/models", "/dashboard/keys", "/dashboard/usage", "/dashboard/settings"] {
        let r = c.get(format!("{}{p}", e.base)).send().await.unwrap();
        assert_eq!(r.status().as_u16(), 200, "{p}");
    }
    // Create a custom node from the UI and chat through it.
    let r = c.post(format!("{}/dashboard/providers/new-node", e.base)).form(&[("ty", "openai-compatible"), ("prefix", "zz"), ("base_url", &format!("{}/v1", e.up)), ("api_key", "good")]).send().await.unwrap();
    let loc = r.headers()["location"].to_str().unwrap().to_string();
    assert!(loc.contains("ok="), "{loc}");
    let id = e.db.list_nodes(None).into_iter().find(|n| n["prefix"] == "zz").unwrap()["id"].as_str().unwrap().to_string();
    let conn = e.db.connections_for(&id, false)[0]["id"].as_str().unwrap().to_string();
    let r = c.post(format!("{}/dashboard/connections/{conn}/test", e.base)).form(&[("x", "1")]).send().await.unwrap();
    let loc = crate::oauth::flows::pct_decode(r.headers()["location"].to_str().unwrap());
    assert!(loc.contains("ok=") && loc.contains("m-1"), "{loc}");
    let (st, body) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": "zz/m-1", "messages": [{"role": "user", "content": "hey"}]})).await;
    assert_eq!(st, 200, "{body}");
    // Combo + alias via forms.
    c.post(format!("{}/dashboard/combos", e.base)).form(&[("name", "dash"), ("models", "zz/m-1")]).send().await.unwrap();
    c.post(format!("{}/dashboard/aliases", e.base)).form(&[("alias", "short"), ("target", "zz/m-1")]).send().await.unwrap();
    for m in ["dash", "short"] {
        let (st, body) = post_json(&format!("{}/v1/chat/completions", e.base), json!({"model": m, "messages": [{"role": "user", "content": "hey"}]})).await;
        assert_eq!(st, 200, "{m}: {body}");
    }
    // OAuth: a browser login starts a pending flow; a forged state is rejected.
    let r = c.post(format!("{}/dashboard/oauth/claude/start", e.base)).form(&[("x", "1")]).send().await.unwrap();
    assert!(r.headers()["location"].to_str().unwrap().starts_with("/dashboard/oauth/flow/"));
    let t = reqwest::get(format!("{}/callback?code=abc&state=forged", e.base)).await.unwrap().text().await.unwrap();
    assert!(t.contains("No pending login"));
    // Routing settings persist into the settings document.
    c.post(format!("{}/dashboard/settings/routing", e.base)).form(&[("strategy", "round-robin"), ("combo_strategy", "round-robin"), ("sticky", "4")]).send().await.unwrap();
    let st = crate::chat::accounts::settings(&e.db);
    assert_eq!(st["fallbackStrategy"], "round-robin");
    assert_eq!(st["stickyRoundRobinLimit"], 4);
}
