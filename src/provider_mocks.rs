//! Mock replies for providers whose upstream protocol is not one of the
//! standard JSON/SSE formats (see `provider_tests`).

use axum::body::Bytes;
use axum::http::Method;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::provider_tests::{Hit, REPLY};

fn sse(body: String) -> Response {
    ([("content-type", "text/event-stream")], body).into_response()
}

fn data(v: Value) -> String {
    format!("data: {v}\n\n")
}

/// Responses that must be sent before the request body finishes.
pub fn early(path: &str) -> Option<Response> {
    if path.ends_with("/agent.v1.AgentService/Run") {
        use crate::providers::cursor::{field_bytes, field_str, wrap_frame};
        let text = wrap_frame(&field_bytes(1, &field_bytes(1, &field_str(1, REPLY))));
        let done = wrap_frame(&field_bytes(1, &field_bytes(14, &[])));
        return Some(([("content-type", "application/connect+proto")], [text, done].concat()).into_response());
    }
    None
}

const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

fn b64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn audio() -> Vec<u8> {
    let mut v = b"ID3".to_vec();
    v.extend(std::iter::repeat_n(7u8, 4096));
    v
}

fn bin(ct: &'static str, b: Vec<u8>) -> Response {
    ([("content-type", ct)], b).into_response()
}

fn js(v: Value) -> Response {
    axum::Json(v).into_response()
}

fn result_item() -> Value {
    json!({"title": "Axum", "url": "https://docs.rs/axum", "link": "https://docs.rs/axum", "name": "Axum", "snippet": "web framework", "content": "web framework", "description": "web framework", "text": "web framework", "raw_content": "web framework"})
}

/// Media protocol replies (images, audio, embeddings, search, fetch).
fn media(method: &Method, h: &Hit, body: &Value) -> Option<Response> {
    let p = h.path.as_str();
    let host = h.host.as_str();
    let get = *method == Method::GET;
    // --- embeddings
    if p.ends_with("/embeddings") {
        return Some(js(json!({"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}], "model": body["model"], "usage": {"prompt_tokens": 2, "total_tokens": 2}})));
    }
    if p.ends_with(":embedContent") {
        return Some(js(json!({"embedding": {"values": [0.1, 0.2]}})));
    }
    if p.ends_with(":batchEmbedContents") {
        return Some(js(json!({"embeddings": [{"values": [0.1, 0.2]}]})));
    }
    // --- Gemini image / audio output
    if p.contains(":generateContent") || p.contains(":streamGenerateContent") {
        let gc = if body["generationConfig"].is_object() { &body["generationConfig"] } else { &body["request"]["generationConfig"] };
        let mods = gc["responseModalities"].to_string().to_uppercase();
        let inline = if mods.contains("AUDIO") {
            Some(json!({"mimeType": "audio/L16;codec=pcm;rate=24000", "data": b64(&[0u8; 4800])}))
        } else if mods.contains("IMAGE") || body["model"].as_str().unwrap_or("").contains("image") {
            Some(json!({"mimeType": "image/png", "data": PNG_B64}))
        } else {
            None
        };
        if let Some(d) = inline {
            let c = json!({"candidates": [{"content": {"role": "model", "parts": [{"inlineData": d}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4}});
            let c = if p.contains("v1internal") { json!({"response": c}) } else { c };
            if p.contains("streamGenerateContent") {
                return Some(sse(data(c)));
            }
            return Some(js(c));
        }
        return None;
    }
    // --- images
    if p.ends_with("/images/generations") {
        return Some(js(json!({"created": 1, "data": [{"b64_json": PNG_B64}]})));
    }
    if p.ends_with("/responses") && body["tools"].to_string().contains("image_generation") {
        let item = json!({"type": "image_generation_call", "id": "ig_1", "status": "completed", "result": PNG_B64});
        return Some(sse([
            format!("event: response.output_item.done\n{}", data(json!({"type": "response.output_item.done", "output_index": 0, "item": item}))),
            format!("event: response.completed\n{}", data(json!({"type": "response.completed", "response": {"id": "r1", "status": "completed", "output": [item]}}))),
        ].concat()));
    }
    if p.contains("/stable-image/generate") {
        return Some(js(json!({"image": PNG_B64, "finish_reason": "SUCCESS"})));
    }
    if host.contains("api.bfl.ai") {
        if get {
            return Some(js(json!({"status": "Ready", "result": {"sample": "https://cdn.example/x.png"}})));
        }
        return Some(js(json!({"id": "t1", "polling_url": "https://api.bfl.ai/v1/get_result?id=t1"})));
    }
    if host.contains("fal.run") {
        if p.ends_with("/status") {
            return Some(js(json!({"status": "COMPLETED"})));
        }
        if get {
            return Some(js(json!({"images": [{"url": "https://cdn.example/x.png"}]})));
        }
        return Some(js(json!({"request_id": "r1", "status_url": "https://queue.fal.run/req/r1/status", "response_url": "https://queue.fal.run/req/r1"})));
    }
    if host.contains("nanobananaapi.ai") {
        if get {
            return Some(js(json!({"code": 200, "data": {"successFlag": 1, "response": {"resultImageUrl": "https://cdn.example/x.png"}}})));
        }
        return Some(js(json!({"code": 200, "msg": "success", "data": {"taskId": "t1"}})));
    }
    if host.contains("runwayml.com") {
        if get {
            return Some(js(json!({"id": "t1", "status": "SUCCEEDED", "output": ["https://cdn.example/x.png"]})));
        }
        return Some(js(json!({"id": "t1"})));
    }
    if p.contains("/ai/run/") {
        return Some(js(json!({"success": true, "result": {"image": PNG_B64}})));
    }
    if host.contains("huggingface.co") && !p.contains("asr") && !body.is_null() && body["inputs"].is_string() {
        return Some(bin("image/png", vec![137, 80, 78, 71, 13, 10, 26, 10, 0, 0]));
    }
    if p.ends_with("/sdapi/v1/txt2img") {
        return Some(js(json!({"images": [PNG_B64]})));
    }
    if host.starts_with("comfy") {
        if p == "/prompt" {
            return Some(js(json!({"prompt_id": "p1", "number": 1})));
        }
        if p.starts_with("/history") {
            return Some(js(json!({"p1": {"status": {"completed": true, "status_str": "success"}, "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}}}})));
        }
        if p == "/view" {
            return Some(bin("image/png", vec![137, 80, 78, 71, 13, 10, 26, 10, 1, 2]));
        }
    }
    if host.contains("topazlabs.com") {
        return Some(bin("image/png", vec![137, 80, 78, 71, 13, 10, 26, 10, 3, 4]));
    }
    // --- text to speech
    if p.ends_with("/audio/speech") || p.contains("/text-to-speech") || p.ends_with("/tts/bytes") || p.ends_with("/v1/tts") || p.contains("/tts/stream") || p.ends_with("/api/tts") || p.ends_with("/v1/speech") || p.starts_with("/tfettts") {
        return Some(bin("audio/mpeg", audio()));
    }
    if p.ends_with("/tts/v1/voice") {
        return Some(js(json!({"audioContent": b64(&audio())})));
    }
    if p.ends_with("/t2a_v2") {
        let hex: String = audio().iter().map(|b| format!("{b:02x}")).collect();
        return Some(js(json!({"data": {"audio": hex, "status": 2}, "extra_info": {"audio_format": "mp3"}, "base_resp": {"status_code": 0, "status_msg": "success"}})));
    }
    if p.ends_with("/chat/completions") && body["modalities"].to_string().contains("audio") {
        return Some(sse([
            data(json!({"choices": [{"index": 0, "delta": {"audio": {"data": b64(&audio()[..2048]), "transcript": "hi"}}}]})),
            data(json!({"choices": [{"index": 0, "delta": {"audio": {"data": b64(&audio()[2048..])}}, "finish_reason": "stop"}]})),
            "data: [DONE]\n\n".into(),
        ].concat()));
    }
    if p.ends_with("/chat/completions") && body["audio"].is_object() && host.contains("xiaomimimo") {
        return Some(js(json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": "", "audio": {"data": b64(&audio()), "format": "wav"}}, "finish_reason": "stop"}]})));
    }
    if host == "www.bing.com" && p == "/translator" {
        return Some(([("content-type", "text/html"), ("set-cookie", "MUID=abc; path=/")], "<script>var params_AbusePreventionHelper = [1700000000000,\"tok-123\",3600000];</script>").into_response());
    }
    if host == "translate.google.com" && p == "/" {
        return Some(([("content-type", "text/html")], r#"<script>window.WIZ_global_data = {"FdrFJe":"-123","cfb2h":"boq_translate-webserver_20240101"};</script>"#).into_response());
    }
    if host == "translate.google.com" && p.contains("batchexecute") {
        let inner = json!([b64(&audio())]).to_string();
        let line = json!([["wrb.fr", "jQ1olc", inner, null, null, null, "generic"]]).to_string();
        return Some(([("content-type", "application/json")], format!(")]}}'\n\n123\n{line}\n")).into_response());
    }
    // --- speech to text
    if p.ends_with("/audio/transcriptions") || (host.contains("huggingface.co") && body.is_null()) {
        return Some(js(json!({"text": REPLY})));
    }
    if p.ends_with("/v1/listen") {
        return Some(js(json!({"results": {"channels": [{"alternatives": [{"transcript": REPLY}]}]}})));
    }
    if host.contains("assemblyai.com") {
        if p.ends_with("/upload") {
            return Some(js(json!({"upload_url": "https://cdn.assemblyai.com/upload/1"})));
        }
        if get {
            return Some(js(json!({"id": "t1", "status": "completed", "text": REPLY})));
        }
        return Some(js(json!({"id": "t1", "status": "queued"})));
    }
    // --- web fetch
    if host.contains("firecrawl.dev") {
        return Some(js(json!({"success": true, "data": {"markdown": "# Page\nhello", "metadata": {"title": "Page"}}})));
    }
    if host == "r.jina.ai" {
        return Some(([("content-type", "text/plain")], "Title: Page\n\nMarkdown Content:\nhello").into_response());
    }
    if p.ends_with("/extract") && host.contains("tavily") {
        return Some(js(json!({"results": [{"url": "https://example.com/page", "raw_content": "hello"}]})));
    }
    if p.ends_with("/contents") && host.contains("exa.ai") {
        return Some(js(json!({"results": [{"title": "Page", "text": "hello"}]})));
    }
    if p.ends_with("/api/web_fetch") {
        return Some(js(json!({"title": "Page", "content": "hello", "links": []})));
    }
    if host.contains("fetch.tinyfish.ai") {
        return Some(js(json!({"results": [{"title": "Page", "text": "hello", "links": []}]})));
    }
    // --- web search
    if host.contains("ydc-index.io") {
        return Some(js(json!({"results": {"web": [result_item()], "news": [result_item()]}})));
    }
    if host.contains("xquik.com") {
        return Some(js(json!({"tweets": [{"id": "1", "text": "axum is great", "author": {"username": "rust", "name": "Rust"}}], "has_next_page": false})));
    }
    if host.contains("api.z.ai") && p.contains("web_search_prime") {
        let payload = json!({"results": [{"title": "Axum", "link": "https://docs.rs/axum", "content": "web framework"}]}).to_string();
        return Some(js(json!({"jsonrpc": "2.0", "id": body["id"], "result": {"content": [{"type": "text", "text": payload}]}})));
    }
    let search_hosts = [("exa.ai", "/search"), ("linkup.so", "/search"), ("search.brave.com", "/search"), ("search.tinyfish.ai", ""), ("tavily.com", "/search"), ("serper.dev", "/"), ("ollama.com", "/api/web_search"), ("googleapis.com", "/customsearch"), ("searchapi.io", "/search"), ("searx", "/search")];
    if search_hosts.iter().any(|(h, path)| host.contains(h) && p.contains(path)) {
        let it = result_item();
        return Some(js(json!({"organic": [it], "web": {"results": [it]}, "results": [it], "items": [it], "organic_results": [it], "news": [it], "search_information": {"total_results": 1}})));
    }
    None
}

/// Per-protocol upstream replies.
pub fn special(method: &Method, h: &Hit) -> Option<Response> {
    let p = h.path.as_str();
    let body: Value = serde_json::from_slice(&h.body).unwrap_or(Value::Null);
    if let Some(r) = media(method, h, &body) {
        return Some(r);
    }
    // Kiro / CodeWhisperer: AWS binary event stream.
    if p.ends_with("/generateAssistantResponse") {
        use crate::providers::kiro::encode_event_frame;
        let ev = |t: &str, payload: Value| encode_event_frame(&[(":event-type", t), (":message-type", "event"), (":content-type", "application/json")], payload.to_string().as_bytes());
        let mut out = ev("assistantResponseEvent", json!({"content": REPLY}));
        out.extend(ev("messageStopEvent", json!({"stopReason": "end_turn"})));
        out.extend(ev("metadataEvent", json!({"tokenUsage": {"inputTokens": 3, "outputTokens": 1}})));
        return Some(([("content-type", "application/vnd.amazon.eventstream")], out).into_response());
    }
    // CommandCode: SSE of AI-SDK style events.
    if p.ends_with("/alpha/generate") {
        return Some(sse([
            data(json!({"type": "text-delta", "text": REPLY, "model": "m"})),
            data(json!({"type": "finish-step", "finishReason": "stop", "usage": {"inputTokens": 3, "outputTokens": 1}})),
            data(json!({"type": "finish", "finishReason": "stop", "totalUsage": {"inputTokens": 3, "outputTokens": 1}})),
        ]
        .concat()));
    }
    // grok.com web: NDJSON.
    if p.ends_with("/rest/app-chat/conversations/new") {
        let lines = [json!({"result": {"response": {"token": "po"}}}), json!({"result": {"response": {"token": "ng"}}}), json!({"result": {"response": {"modelResponse": {"message": REPLY}}}})];
        return Some(([("content-type", "application/json")], lines.iter().map(|l| format!("{l}\n")).collect::<String>()).into_response());
    }
    // perplexity.ai web: SSE with markdown blocks.
    if p.ends_with("/rest/sse/perplexity_ask") {
        let block = |chunks: &[&str], progress: &str| json!({"intended_usage": "ask_text_0_markdown", "markdown_block": {"chunks": chunks, "progress": progress}});
        return Some(sse([
            format!("event: message\n{}", data(json!({"backend_uuid": "b-1", "blocks": [block(&["po"], "IN_PROGRESS")]}))),
            format!("event: message\n{}", data(json!({"blocks": [block(&["ng"], "IN_PROGRESS")]}))),
            format!("event: message\n{}", data(json!({"blocks": [block(&[REPLY], "DONE")], "final": true, "status": "COMPLETED"}))),
            "event: end_of_stream\ndata: {}\n\n".into(),
        ]
        .concat()));
    }
    // Qoder: model catalogue + enveloped SSE.
    if p.ends_with("/algo/api/v2/model/list") {
        let ids = ["ultimate", "test-model", "auto", "performance", "efficient", "lite"];
        return Some(axum::Json(json!({"chat": ids.iter().map(|k| json!({"key": k, "display_name": k, "enable": true, "max_input_tokens": 100000})).collect::<Vec<_>>()})).into_response());
    }
    if p.ends_with("/agent_chat_generation") {
        let env = |inner: Value| data(json!({"statusCodeValue": 200, "body": inner.to_string()}));
        return Some(sse([
            env(json!({"id": "q1", "choices": [{"index": 0, "delta": {"role": "assistant", "content": REPLY}}]})),
            env(json!({"id": "q1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})),
            env(json!({"id": "q1", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})),
            "data: [DONE]\n\n".into(),
        ]
        .concat()));
    }
    // Zed cloud.
    if p == "/client/users/me" {
        return Some(axum::Json(json!({"user": {"id": 1}, "organizations": [{"id": "org-1", "is_personal": true}]})).into_response());
    }
    if p == "/client/llm_tokens" {
        return Some(axum::Json(json!({"token": "llm-token"})).into_response());
    }
    if p == "/models" && h.host.contains("zed.dev") {
        return Some(axum::Json(json!({"models": [{"id": "test-model", "provider": "open_ai", "display_name": "Test"}], "default_model": "test-model"})).into_response());
    }
    if p == "/completions" && h.host.contains("zed.dev") {
        let item = json!({"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": REPLY, "annotations": []}]});
        let lines = [
            json!({"event": {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}}}),
            json!({"event": {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": REPLY}}),
            json!({"event": {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": [item], "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}}}}),
        ];
        return Some(([("content-type", "application/x-ndjson")], lines.iter().map(|l| format!("{l}\n")).collect::<String>()).into_response());
    }
    // MiMo free: anonymous JWT bootstrap.
    if p.ends_with("/api/free-ai/bootstrap") {
        return Some(axum::Json(json!({"jwt": "eyJhbGciOiJub25lIn0.eyJleHAiOjQwNzA5MDg4MDB9.x"})).into_response());
    }
    let _ = (method, body, Bytes::new());
    None
}

/// Connection fields some providers need to reach the mock.
pub fn adjust_connection(id: &str, c: &mut Value) {
    match id {
        "ollama-local" => c["providerSpecificData"]["baseUrl"] = json!("http://ollama.local"),
        "zed" => c["providerSpecificData"]["organizationId"] = json!("org-1"),
        "aws-polly" => c["providerSpecificData"]["accessKeyId"] = json!("AKIATEST"),
        "google-pse" => c["providerSpecificData"]["cx"] = json!("cx-1"),
        "sdwebui" => c["providerSpecificData"]["baseUrl"] = json!("http://sdwebui.local/sdapi/v1/txt2img"),
        "comfyui" => c["providerSpecificData"]["baseUrl"] = json!("http://comfy.local"),
        "searxng" => c["providerSpecificData"]["baseUrl"] = json!("https://searx.example/search"),
        "coqui" => c["providerSpecificData"]["baseUrl"] = json!("http://coqui.local/api/tts"),
        "tortoise" => c["providerSpecificData"]["baseUrl"] = json!("http://tortoise.local/api/tts"),
        "selfhosted-tts" => c["providerSpecificData"]["baseUrl"] = json!("http://tts.local"),
        "selfhosted-stt" => c["providerSpecificData"]["baseUrl"] = json!("http://stt.local/v1/audio/transcriptions"),
        "selfhosted-embedding" => c["providerSpecificData"]["baseUrl"] = json!("http://embed.local/v1"),
        _ => {}
    }
}
