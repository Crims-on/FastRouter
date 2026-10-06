//! POST /v1/images/generations (port of imageGeneration.js, imageGenerationCore.js
//! and imageProviders/*).

use std::time::Duration;

use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use base64::Engine;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use super::*;
use crate::AppState;
use crate::chat::core::sse_response;
use crate::exec::{ExecArgs, Executor};

const POLL_INTERVAL: Duration = Duration::from_millis(1500);
const POLL_TIMEOUT: Duration = Duration::from_secs(120);
const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    OpenAi,
    Gemini,
    Codex,
    SdWebUi,
    ComfyUi,
    HuggingFace,
    NanoBanana,
    Antigravity,
    Fal,
    Stability,
    Bfl,
    Runway,
    Cloudflare,
}

fn kind(provider: &str) -> Option<Kind> {
    Some(match provider {
        "gemini" => Kind::Gemini,
        "codex" => Kind::Codex,
        "sdwebui" => Kind::SdWebUi,
        "comfyui" => Kind::ComfyUi,
        "huggingface" => Kind::HuggingFace,
        "nanobanana" => Kind::NanoBanana,
        "antigravity" => Kind::Antigravity,
        "fal-ai" => Kind::Fal,
        "stability-ai" => Kind::Stability,
        "black-forest-labs" => Kind::Bfl,
        "runwayml" => Kind::Runway,
        "cloudflare-ai" => Kind::Cloudflare,
        p if crate::exec::is_openai_compatible(p) => Kind::OpenAi,
        p if media_cfg(p, "imageConfig")["baseUrl"].is_string() => Kind::OpenAi,
        _ => return None,
    })
}

fn img_cfg(p: &str) -> Value {
    media_cfg(p, "imageConfig")
}

fn base_url(p: &str) -> String {
    img_cfg(p)["baseUrl"].as_str().unwrap_or("").to_string()
}

fn no_auth(k: Kind) -> bool {
    matches!(k, Kind::SdWebUi | Kind::ComfyUi)
}

enum Body {
    Json(Value),
    Form(Vec<(String, String)>),
}

fn openai_body(provider: &str, model: &str, b: &Value) -> Value {
    let mut full = json!({"model": model, "prompt": b["prompt"], "n": b["n"].as_i64().unwrap_or(1), "size": b["size"].as_str().unwrap_or("1024x1024")});
    for k in ["quality", "style", "response_format"] {
        if truthy(&b[k]) {
            full[k] = b[k].clone();
        }
    }
    if let Some(fields) = img_cfg(provider)["bodyFields"].as_array() {
        let mut r = json!({});
        for f in fields.iter().filter_map(|f| f.as_str()) {
            if !full[f].is_null() {
                r[f] = full[f].clone();
            }
        }
        return r;
    }
    full
}

async fn hf_source_image(b: &Value) -> Option<String> {
    let raw = b["image"].as_str().or_else(|| b["images"][0].as_str())?.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    if raw.starts_with("http://") || raw.starts_with("https://") {
        return url_to_base64(&raw).await.ok();
    }
    let re = regex::Regex::new(r"(?i)^data:image/[^;]+;base64,(.+)$").unwrap();
    Some(re.captures(&raw).map(|c| c[1].to_string()).unwrap_or(raw))
}

fn hf_entry(model: &str) -> Value {
    img_cfg("huggingface")["modelMap"].get(model).cloned().unwrap_or(Value::Null)
}

async fn cf_image_input(v: &Value) -> Option<(Vec<u8>, String)> {
    if let Some(a) = v.as_array() {
        let bytes: Vec<u8> = a.iter().filter_map(|x| x.as_u64().map(|n| n as u8)).collect();
        let b = B64.encode(&bytes);
        return Some((bytes, b));
    }
    let s = v.as_str()?.trim();
    if s.is_empty() {
        return None;
    }
    let b64 = if s.starts_with("http://") || s.starts_with("https://") {
        url_to_base64(s).await.ok()?
    } else {
        let re = regex::Regex::new(r"(?i)^data:image/[^;]+;base64,(.+)$").unwrap();
        re.captures(s).map(|c| c[1].to_string()).unwrap_or_else(|| s.to_string())
    };
    let bytes = B64.decode(&b64).unwrap_or_default();
    Some((bytes, b64))
}

fn dims(b: &Value) -> Vec<(String, i64)> {
    let mut out = vec![];
    if let Some((w, h)) = b["size"].as_str().and_then(|s| s.split_once('x')) {
        if let (Ok(w), Ok(h)) = (w.parse::<i64>(), h.parse::<i64>()) {
            out.push(("width".into(), w));
            out.push(("height".into(), h));
        }
    }
    for k in ["width", "height"] {
        if let Some(n) = b[k].as_i64().or_else(|| b[k].as_str().and_then(|s| s.parse().ok())) {
            out.retain(|(x, _)| x != k);
            out.push((k.into(), n));
        }
    }
    out
}

const CF_MULTIPART: &[&str] = &["@cf/black-forest-labs/flux-2-dev", "@cf/black-forest-labs/flux-2-klein-4b", "@cf/black-forest-labs/flux-2-klein-9b"];
const CF_OPTIONAL: &[&str] = &["negative_prompt", "guidance", "seed", "num_steps", "steps", "strength"];

async fn build_body(k: Kind, provider: &str, model: &str, b: &Value) -> Result<Body, String> {
    let prompt = b["prompt"].clone();
    Ok(Body::Json(match k {
        Kind::OpenAi => openai_body(provider, model, b),
        Kind::Gemini => json!({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}),
        Kind::SdWebUi => {
            let size = b["size"].as_str().unwrap_or("1024x1024");
            let (w, h) = size.split_once('x').map(|(w, h)| (w.parse().unwrap_or(512), h.parse().unwrap_or(512))).unwrap_or((512, 512));
            json!({"prompt": prompt, "width": w, "height": h, "steps": 20, "batch_size": b["n"].as_i64().unwrap_or(1)})
        }
        Kind::ComfyUi => json!({"prompt": prompt}),
        Kind::HuggingFace => {
            let e = hf_entry(model);
            let task = if e.is_string() { "text-to-image" } else { e["task"].as_str().unwrap_or("text-to-image") };
            if task == "image-to-image" {
                let img = hf_source_image(b).await.ok_or_else(|| format!("HuggingFace: model \"{model}\" requires a source image. Send it as \"image\" (or \"images\") in the request body."))?;
                json!({"inputs": img, "parameters": {"prompt": prompt}})
            } else {
                json!({"inputs": prompt})
            }
        }
        Kind::NanoBanana => {
            let edit = truthy(&b["image"]) || b["images"].as_array().map(|a| !a.is_empty()).unwrap_or(false);
            let mut r = json!({"prompt": prompt, "type": if edit { "IMAGETOIAMGE" } else { "TEXTTOIAMGE" }, "numImages": b["n"].as_i64().unwrap_or(1), "image_size": size_to_aspect(&b["size"]), "callBackUrl": "https://localhost/callback"});
            if edit {
                let mut urls: Vec<Value> = b["images"].as_array().cloned().unwrap_or_default().into_iter().filter(truthy).collect();
                if truthy(&b["image"]) {
                    urls.push(b["image"].clone());
                }
                r["imageUrls"] = json!(urls);
            }
            r
        }
        Kind::Fal => {
            let mut r = json!({"prompt": prompt, "num_images": b["n"].as_i64().unwrap_or(1)});
            if truthy(&b["size"]) {
                r["image_size"] = json!(size_to_aspect(&b["size"]));
            }
            if truthy(&b["image"]) {
                r["image_url"] = b["image"].clone();
            }
            r
        }
        Kind::Stability => {
            let mut r = json!({"prompt": prompt, "output_format": b["output_format"].as_str().unwrap_or("png").to_lowercase()});
            if truthy(&b["size"]) {
                r["aspect_ratio"] = json!(size_to_aspect(&b["size"]));
            }
            if truthy(&b["style"]) {
                r["style_preset"] = b["style"].clone();
            }
            if model.contains("sd3") {
                r["model"] = json!(model);
            }
            r
        }
        Kind::Bfl => {
            let mut r = json!({"prompt": prompt});
            if let Some((w, h)) = b["size"].as_str().and_then(|s| s.split_once('x')) {
                if let Ok(w) = w.parse::<i64>() {
                    r["width"] = json!(w);
                }
                if let Ok(h) = h.parse::<i64>() {
                    r["height"] = json!(h);
                }
            }
            if truthy(&b["image"]) {
                r["image_prompt"] = b["image"].clone();
            }
            r
        }
        Kind::Runway => {
            let ratio = size_to_aspect(&b["size"]);
            if !model.contains("image") {
                let mut r = json!({"promptText": prompt, "model": model, "ratio": ratio, "duration": 5});
                if truthy(&b["image"]) {
                    r["promptImage"] = b["image"].clone();
                }
                r
            } else {
                let mut r = json!({"promptText": prompt, "model": model, "ratio": ratio});
                if truthy(&b["image"]) {
                    r["referenceImages"] = json!([{"uri": b["image"]}]);
                }
                r
            }
        }
        Kind::Cloudflare => {
            if CF_MULTIPART.contains(&model) {
                let mut form = vec![("prompt".to_string(), crate::jsv::js_string(&prompt))];
                for (k, v) in dims(b) {
                    form.push((k, v.to_string()));
                }
                for k in CF_OPTIONAL {
                    if !b[*k].is_null() && b[*k] != "" {
                        form.push((k.to_string(), crate::jsv::js_string(&b[*k])));
                    }
                }
                return Ok(Body::Form(form));
            }
            let mut r = json!({"prompt": prompt});
            for (k, v) in dims(b) {
                r[k] = json!(v);
            }
            for k in CF_OPTIONAL {
                if !b[*k].is_null() && b[*k] != "" {
                    r[*k] = b[*k].clone();
                }
            }
            if let Some((bytes, b64)) = cf_image_input(&b["image"]).await {
                r["image_b64"] = json!(b64);
                r["image"] = json!(bytes);
            }
            let mask = [&b["mask_image"], &b["maskImage"], &b["mask"]].into_iter().find(|v| truthy(v)).cloned().unwrap_or(Value::Null);
            if let Some((bytes, b64)) = cf_image_input(&mask).await {
                r["mask_b64"] = json!(b64);
                r["mask"] = json!(bytes.clone());
                r["mask_image"] = json!(bytes);
            }
            r
        }
        Kind::Codex => codex_body(model, b),
        Kind::Antigravity => Value::Null,
    }))
}

fn build_url(k: Kind, provider: &str, model: &str, creds: &Value) -> Result<String, String> {
    let base = base_url(provider);
    Ok(match k {
        Kind::OpenAi => {
            if crate::exec::is_openai_compatible(provider) {
                let b = creds["providerSpecificData"]["baseUrl"].as_str().unwrap_or("https://api.openai.com/v1").trim_end_matches('/');
                format!("{b}/images/generations")
            } else {
                base
            }
        }
        Kind::Gemini => format!("{base}/{}:generateContent?key={}", model.strip_prefix("models/").unwrap_or(model), crate::oauth::enc(&key_of(creds))),
        Kind::SdWebUi | Kind::ComfyUi | Kind::NanoBanana => {
            let ov = creds["providerSpecificData"]["baseUrl"].as_str().filter(|s| !s.trim().is_empty());
            ov.map(str::to_owned).unwrap_or(base)
        }
        Kind::HuggingFace => {
            if let Some(o) = creds["providerSpecificData"]["baseUrl"].as_str().map(str::trim).filter(|s| !s.is_empty()) {
                if model.contains("..") || model.contains("//") || model.contains('?') || model.contains('#') {
                    return Err(format!("HuggingFace: invalid model ID \"{model}\""));
                }
                format!("{}/{model}", o.trim_end_matches('/'))
            } else {
                let e = hf_entry(model);
                let path = if e.is_string() { e.as_str().unwrap().to_string() } else { e["path"].as_str().map(str::to_owned).ok_or_else(|| format!("HuggingFace: no HuggingFace router mapping for model \"{model}\". Set a custom base URL on the connection."))? };
                format!("{base}/{path}")
            }
        }
        Kind::Fal | Kind::Bfl => format!("{base}/{model}"),
        Kind::Stability => format!("{base}/{}", if model.contains("ultra") { "ultra" } else if model.contains("sd3") { "sd3" } else { "core" }),
        Kind::Runway => format!("{base}/{}", if model.contains("image") { "text_to_image" } else { "image_to_video" }),
        Kind::Cloudflare => {
            let acct = creds["providerSpecificData"]["accountId"].as_str().filter(|s| !s.is_empty()).ok_or("cloudflare-ai requires accountId in providerSpecificData")?;
            format!("{base}/{acct}/ai/run/{model}")
        }
        Kind::Codex => crate::registry::REG.transport("codex")["baseUrl"].as_str().unwrap_or("https://chatgpt.com/backend-api/codex/responses").to_string(),
        Kind::Antigravity => String::new(),
    })
}

fn codex_account_id(creds: &Value) -> String {
    if let Some(a) = creds["providerSpecificData"]["chatgptAccountId"].as_str().filter(|s| !s.is_empty()) {
        return a.to_string();
    }
    crate::oauth::decode_jwt_payload(creds["idToken"].as_str().unwrap_or(""))
        .and_then(|p| p["https://api.openai.com/auth"]["chatgpt_account_id"].as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn headers_for(k: Kind, provider: &str, creds: &Value, multipart: bool) -> Vec<(String, String)> {
    let key = key_of(creds);
    let mut h: Vec<(String, String)> = vec![];
    if !multipart {
        h.push(("content-type".into(), "application/json".into()));
    }
    match k {
        Kind::OpenAi => {
            for (a, b) in img_cfg(provider)["headers"].as_object().into_iter().flatten() {
                h.push((a.clone(), crate::jsv::js_string(b)));
            }
            if !key.is_empty() {
                h.push(("authorization".into(), format!("Bearer {key}")));
            }
        }
        Kind::Gemini | Kind::SdWebUi | Kind::ComfyUi => {}
        Kind::HuggingFace | Kind::NanoBanana | Kind::Cloudflare => {
            if !key.is_empty() {
                h.push(("authorization".into(), format!("Bearer {key}")));
            }
        }
        Kind::Fal => h.push(("authorization".into(), format!("Key {key}"))),
        Kind::Stability => {
            h.push(("authorization".into(), format!("Bearer {key}")));
            h.push(("accept".into(), "application/json".into()));
        }
        Kind::Bfl => h.push(("x-key".into(), key)),
        Kind::Runway => {
            h.push(("authorization".into(), format!("Bearer {key}")));
            h.push(("x-runway-version".into(), "2024-11-06".into()));
        }
        Kind::Codex => {
            let v = crate::consts::s("CODEX_CLI_VERSION");
            h = vec![
                ("accept".into(), "text/event-stream, application/json".into()),
                ("authorization".into(), format!("Bearer {}", creds["accessToken"].as_str().unwrap_or(""))),
                ("chatgpt-account-id".into(), codex_account_id(creds)),
                ("content-type".into(), "application/json".into()),
                ("originator".into(), "codex_cli_rs".into()),
                ("session_id".into(), uuid::Uuid::new_v4().to_string()),
                ("user-agent".into(), format!("codex_cli_rs/{v}")),
                ("version".into(), v.to_string()),
                ("x-client-request-id".into(), uuid::Uuid::new_v4().to_string()),
            ];
        }
        Kind::Antigravity => {}
    }
    h
}

const CODEX_TOOL_IMAGE_MODELS: &[&str] = &["gpt-image-1.5", "gpt-image-2", "gpt-image-2.5", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"];

fn to_data_url(v: &Value) -> Option<String> {
    let s = v.as_str().filter(|s| !s.is_empty())?;
    if s.to_lowercase().starts_with("data:image/") || s.starts_with("http://") || s.starts_with("https://") {
        Some(s.to_string())
    } else {
        Some(format!("data:image/png;base64,{s}"))
    }
}

fn codex_body(model: &str, b: &Value) -> Value {
    let mut refs: Vec<String> = b["images"].as_array().into_iter().flatten().filter_map(to_data_url).collect();
    if let Some(u) = to_data_url(&b["image"]) {
        refs.push(u);
    }
    let detail = b["image_detail"].as_str().unwrap_or("high");
    let (resp_model, tool_model) = if CODEX_TOOL_IMAGE_MODELS.contains(&model) { ("gpt-5.5".to_string(), Some(model)) } else { (model.strip_suffix("-image").unwrap_or(model).to_string(), None) };
    let mut tool = json!({"type": "image_generation", "output_format": b["output_format"].as_str().unwrap_or("png").to_lowercase()});
    if let Some(tm) = tool_model {
        tool["action"] = json!(if refs.is_empty() { "generate" } else { "edit" });
        tool["model"] = json!(tm);
    }
    for k in ["size", "quality", "background"] {
        if b[k].as_str().map(|s| !s.is_empty()).unwrap_or(false) {
            tool[k] = b[k].clone();
        }
    }
    let mut content = vec![];
    for (i, u) in refs.iter().enumerate() {
        content.push(json!({"type": "input_text", "text": format!("<image name=image{}>", i + 1)}));
        content.push(json!({"type": "input_image", "image_url": u, "detail": detail}));
        content.push(json!({"type": "input_text", "text": "</image>"}));
    }
    content.push(json!({"type": "input_text", "text": b["prompt"]}));
    json!({
        "model": resp_model, "instructions": "",
        "input": [{"type": "message", "role": "user", "content": content}],
        "tools": [tool],
        "tool_choice": if tool_model.is_some() { json!({"type": "image_generation"}) } else { json!("auto") },
        "parallel_tool_calls": false, "prompt_cache_key": uuid::Uuid::new_v4().to_string(),
        "stream": true, "store": false,
        "reasoning": if tool_model.is_some() { json!({"effort": "medium", "summary": "auto"}) } else { Value::Null },
    })
}

/// Parses the Codex Responses SSE; calls `on_event` for progress/partials.
async fn codex_stream(resp: reqwest::Response, mut on_event: impl FnMut(&str, Value)) -> Option<String> {
    let mut parser = crate::sse::EventParser::default();
    let mut body = resp.bytes_stream();
    let mut image = None;
    let mut bytes_received = 0usize;
    let mut last = std::time::Instant::now() - Duration::from_secs(1);
    let mut handle = |e: crate::sse::SseEvent, image: &mut Option<String>, n: usize, on_event: &mut dyn FnMut(&str, Value)| {
        let Some(name) = e.event.clone() else { return };
        if last.elapsed() > Duration::from_millis(200) {
            last = std::time::Instant::now();
            on_event("progress", json!({"stage": name, "bytesReceived": n}));
        }
        let Ok(d) = serde_json::from_str::<Value>(&e.data) else { return };
        if name == "response.image_generation_call.partial_image" {
            if let Some(p) = d["partial_image_b64"].as_str() {
                on_event("partial_image", json!({"b64_json": p, "index": d["partial_image_index"]}));
            }
        }
        if name == "response.output_item.done" && d["item"]["type"] == "image_generation_call" {
            if let Some(r) = d["item"]["result"].as_str() {
                *image = Some(r.to_string());
            }
        }
    };
    while let Some(Ok(c)) = body.next().await {
        bytes_received += c.len();
        for e in parser.push(&c) {
            handle(e, &mut image, bytes_received, &mut on_event);
        }
    }
    for e in parser.finish() {
        handle(e, &mut image, bytes_received, &mut on_event);
    }
    image
}

async fn poll_json(client: &reqwest::Client, url: &str, headers: &[(String, String)], done: impl Fn(&Value) -> Result<bool, String>) -> Result<Value, String> {
    let deadline = std::time::Instant::now() + POLL_TIMEOUT;
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(POLL_INTERVAL).await;
        let mut rb = client.get(url);
        for (k, v) in headers {
            rb = rb.header(k.as_str(), v.as_str());
        }
        let r = rb.send().await.map_err(|e| e.to_string())?;
        if !r.status().is_success() {
            return Err(format!("status {}", r.status().as_u16()));
        }
        let s: Value = r.json().await.map_err(|e| e.to_string())?;
        if done(&s)? {
            return Ok(s);
        }
    }
    Err("polling timeout".into())
}

fn images_or_placeholder(images: Vec<Value>, prompt: &Value) -> Value {
    json!({"created": now_s(), "data": if images.is_empty() { json!([{"b64_json": "", "revised_prompt": prompt}]) } else { json!(images) }})
}

fn gemini_normalize(v: &Value, prompt: &Value) -> Value {
    let cands = if v["candidates"].is_array() { &v["candidates"] } else { &v["response"]["candidates"] };
    let imgs = cands[0]["content"]["parts"].as_array().into_iter().flatten().filter_map(|p| p["inlineData"]["data"].as_str().map(|d| json!({"b64_json": d}))).collect();
    images_or_placeholder(imgs, prompt)
}

fn cf_item(s: &str) -> Option<Value> {
    if s.is_empty() {
        return None;
    }
    let re = regex::Regex::new(r"(?i)^data:image/[^;]+;base64,").unwrap();
    if re.is_match(s) {
        return Some(json!({"b64_json": re.replace(s, "").to_string()}));
    }
    if s.starts_with("http://") || s.starts_with("https://") {
        return Some(json!({"url": s}));
    }
    Some(json!({"b64_json": s}))
}

fn cf_normalize(v: &Value) -> Value {
    if truthy(&v["created"]) && v["data"].is_array() {
        return v.clone();
    }
    let result = if v.get("result").map(|r| !r.is_null()).unwrap_or(false) { &v["result"] } else { v };
    if let Some(q) = result["responses"].as_array().and_then(|a| a.iter().find(|i| i["success"] != json!(false))) {
        return cf_normalize(&q["result"]);
    }
    let img = result.as_str().map(str::to_owned).or_else(|| result["image"].as_str().map(str::to_owned)).or_else(|| result["data"][0]["b64_json"].as_str().map(str::to_owned)).or_else(|| result["data"][0]["url"].as_str().map(str::to_owned)).unwrap_or_default();
    json!({"created": now_s(), "data": cf_item(&img).map(|i| vec![i]).unwrap_or_default()})
}

async fn binary_response(final_body: &Value, body: &Value) -> Option<Response> {
    let first = &final_body["data"][0];
    let mut b64 = first["b64_json"].as_str().filter(|s| !s.is_empty()).map(str::to_owned);
    if b64.is_none() {
        if let Some(u) = first["url"].as_str() {
            b64 = url_to_base64(u).await.ok();
        }
    }
    let bytes = B64.decode(b64?).ok()?;
    let fmt = body["output_format"].as_str().unwrap_or("png").to_lowercase();
    let mime = match fmt.as_str() {
        "jpeg" | "jpg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    };
    let ext = if fmt == "jpeg" { "jpg".to_string() } else { fmt };
    let mut r = Response::new(axum::body::Body::from(bytes));
    r.headers_mut().insert("content-type", mime.parse().unwrap());
    r.headers_mut().insert("content-disposition", format!("inline; filename=\"image.{ext}\"").parse().unwrap());
    r.headers_mut().insert("access-control-allow-origin", "*".parse().unwrap());
    Some(r)
}

async fn antigravity_image(ctx: &mut Ctx, body: &Value) -> Result<Value, String> {
    let ex = crate::providers::antigravity::Antigravity;
    let mut target = if crate::providers::antigravity::is_image_model(&ctx.model) || ctx.model.to_lowercase().contains("image") { ctx.model.clone() } else { "gemini-3.1-flash-image".into() };
    if let Some(s) = body["size"].as_str() {
        let suffix = size_to_aspect(&json!(s)).replace(':', "x");
        if !target.contains(&suffix) {
            target = format!("{target}-{suffix}");
        }
    }
    let mut parts = vec![json!({"text": body["prompt"]})];
    let input = if truthy(&body["image"]) { body["image"].clone() } else { body["images"][0].clone() };
    if let Some(s) = input.as_str() {
        let re = regex::Regex::new(r"^data:(image/[^;]+);base64,(.+)$").unwrap();
        if let Some(c) = re.captures(s) {
            parts.insert(0, json!({"inlineData": {"mimeType": &c[1], "data": &c[2]}}));
        } else if s.len() > 100 && !s.starts_with("http") {
            parts.insert(0, json!({"inlineData": {"mimeType": "image/png", "data": s}}));
        }
    }
    if !truthy(&ctx.creds["projectId"]) {
        if let Some(pid) = crate::chat::util::fetch_project_id(ctx.creds["accessToken"].as_str().unwrap_or(""), "antigravity").await {
            ctx.creds["projectId"] = json!(pid);
        }
    }
    let r = ex.execute(ExecArgs { model: &target, body: json!({"contents": [{"role": "user", "parts": parts}]}), stream: false, creds: &mut ctx.creds, session_id: None, client_tool: None, override_headers: None }).await?;
    let st = r.response.status;
    let text = r.response.text().await;
    if !(200..300).contains(&st) {
        return Err(if text.is_empty() { format!("HTTP {st}") } else { text });
    }
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

async fn core(mut ctx: Ctx, body: Value, stream_to_client: bool, binary: bool) -> MediaResult {
    let Some(k) = kind(&ctx.provider) else {
        return MediaResult::err(400, format!("Provider '{}' does not support image generation", ctx.provider));
    };
    let provider = ctx.provider.clone();
    let model = ctx.model.clone();
    let prompt = body["prompt"].clone();
    let final_body: Value = if k == Kind::Antigravity {
        match antigravity_image(&mut ctx, &body).await {
            Ok(v) => gemini_normalize(&v, &prompt),
            Err(e) => return provider_err(502, &e),
        }
    } else {
        let req = match build_body(k, &provider, &model, &body).await {
            Ok(b) => b,
            Err(e) => return MediaResult::err(400, e),
        };
        if let Err(e) = build_url(k, &provider, &model, &ctx.creds) {
            return MediaResult::err(400, e);
        }
        let multipart = matches!(req, Body::Form(_));
        let req_json = match &req {
            Body::Json(v) => Some(v.to_string()),
            Body::Form(_) => None,
        };
        let form_pairs = match &req {
            Body::Form(f) => f.clone(),
            _ => vec![],
        };
        let r = send_with_refresh(&mut ctx, |c| {
            let url = build_url(k, &provider, &model, c)?;
            let mut rb = client(c).post(url).timeout(Duration::from_secs(600));
            for (a, b) in headers_for(k, &provider, c, multipart) {
                rb = rb.header(a, b);
            }
            Ok(match &req_json {
                Some(s) => rb.body(s.clone()),
                None => {
                    let mut f = reqwest::multipart::Form::new();
                    for (a, b) in &form_pairs {
                        f = f.text(a.clone(), b.clone());
                    }
                    rb.multipart(f)
                }
            })
        })
        .await;
        let r = match r {
            Ok(r) => r,
            Err(e) => return provider_err(502, &e),
        };
        if !r.status().is_success() {
            return upstream_error(r).await;
        }
        let hdrs = headers_for(k, &provider, &ctx.creds, multipart);
        let http = client(&ctx.creds);
        let parsed: Result<Value, String> = match k {
            Kind::Codex => {
                if stream_to_client {
                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
                    tokio::spawn(async move {
                        let tx2 = tx.clone();
                        let img = codex_stream(r, |ev, d| {
                            let _ = tx2.send(Bytes::from(format!("event: {ev}\ndata: {d}\n\n")));
                        })
                        .await;
                        let (ev, d) = match img {
                            Some(b) => ("done", json!({"created": now_s(), "data": [{"b64_json": b}]})),
                            None => ("error", json!({"message": "Codex did not return an image. Account may not be entitled (Plus/Pro required)."})),
                        };
                        let _ = tx.send(Bytes::from(format!("event: {ev}\ndata: {d}\n\n")));
                    });
                    let s = tokio_stream_from(rx);
                    return MediaResult::ok(sse_response(s, &[]));
                }
                match codex_stream(r, |_, _| {}).await {
                    Some(b) => Ok(json!({"created": now_s(), "data": [{"b64_json": b}]})),
                    None => Err("Codex did not return an image. Account may not be entitled (Plus/Pro required).".into()),
                }
            }
            Kind::HuggingFace => r.bytes().await.map(|b| json!({"created": now_s(), "data": [{"b64_json": B64.encode(&b)}]})).map_err(|e| e.to_string()),
            Kind::Cloudflare => {
                let ct = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_lowercase();
                if ct.starts_with("image/") {
                    r.bytes().await.map(|b| json!({"created": now_s(), "data": [{"b64_json": B64.encode(&b)}]})).map_err(|e| e.to_string())
                } else {
                    r.json::<Value>().await.map(|v| cf_normalize(&v)).map_err(|e| e.to_string())
                }
            }
            Kind::NanoBanana => async {
                let sub: Value = r.json().await.map_err(|e| e.to_string())?;
                if sub["code"] != json!(200) {
                    return Err(sub["msg"].as_str().unwrap_or("NanoBanana submit failed").to_string());
                }
                let task = sub["data"]["taskId"].as_str().ok_or("NanoBanana: no taskId returned")?;
                let url = format!("{}?taskId={}", img_cfg("nanobanana")["pollUrl"].as_str().unwrap_or(""), crate::oauth::enc(task));
                let s = poll_json(&http, &url, &hdrs, |s| match s["data"]["successFlag"].as_i64() {
                    Some(1) => Ok(true),
                    Some(2) | Some(3) => Err(s["data"]["errorMessage"].as_str().unwrap_or("NanoBanana generation failed").to_string()),
                    _ => Ok(false),
                })
                .await?;
                Ok(s["data"].clone())
            }
            .await,
            Kind::Fal => async {
                let sub: Value = r.json().await.map_err(|e| e.to_string())?;
                let su = sub["status_url"].as_str().ok_or("Fal: no status_url")?.to_string();
                let ru = sub["response_url"].as_str().ok_or("Fal: no response_url")?.to_string();
                poll_json(&http, &su, &hdrs, |s| match s["status"].as_str() {
                    Some("COMPLETED") => Ok(true),
                    Some("FAILED") => Err(s["error"].as_str().unwrap_or("Fal generation failed").to_string()),
                    _ => Ok(false),
                })
                .await?;
                let mut rb = http.get(&ru);
                for (a, b) in &hdrs {
                    rb = rb.header(a.as_str(), b.as_str());
                }
                rb.send().await.map_err(|e| e.to_string())?.json::<Value>().await.map_err(|e| e.to_string())
            }
            .await,
            Kind::Bfl => async {
                let sub: Value = r.json().await.map_err(|e| e.to_string())?;
                let pu = sub["polling_url"].as_str().ok_or("BFL: no polling_url returned")?.to_string();
                let h = vec![("x-key".to_string(), key_of(&ctx.creds)), ("accept".to_string(), "application/json".to_string())];
                poll_json(&http, &pu, &h, |s| match s["status"].as_str() {
                    Some("Ready") => Ok(true),
                    Some("Error") | Some("Failed") => Err(s["error"].as_str().unwrap_or("BFL generation failed").to_string()),
                    _ => Ok(false),
                })
                .await
            }
            .await,
            Kind::Runway => async {
                let sub: Value = r.json().await.map_err(|e| e.to_string())?;
                let id = sub["id"].as_str().ok_or("Runway: no task id returned")?;
                let url = format!("{}/tasks/{id}", base_url("runwayml"));
                poll_json(&http, &url, &hdrs, |s| match s["status"].as_str() {
                    Some("SUCCEEDED") => Ok(true),
                    Some("FAILED") | Some("CANCELLED") => Err(s["failure"].as_str().unwrap_or("Runway task failed").to_string()),
                    _ => Ok(false),
                })
                .await
            }
            .await,
            _ => r.json::<Value>().await.map_err(|e| e.to_string()),
        };
        let parsed = match parsed {
            Ok(v) => v,
            Err(e) => return MediaResult::err(502, e),
        };
        let normalized = match k {
            Kind::Gemini => gemini_normalize(&parsed, &prompt),
            Kind::SdWebUi => json!({"created": now_s(), "data": parsed["images"].as_array().into_iter().flatten().map(|i| json!({"b64_json": i})).collect::<Vec<_>>()}),
            Kind::NanoBanana => {
                let url = parsed["response"]["resultImageUrl"].as_str().or_else(|| parsed["response"]["originImageUrl"].as_str());
                json!({"created": now_s(), "data": url.map(|u| vec![json!({"url": u, "revised_prompt": prompt})]).unwrap_or_default()})
            }
            Kind::Fal => {
                let imgs: Vec<Value> = if let Some(a) = parsed["images"].as_array() { a.clone() } else if truthy(&parsed["image"]) { vec![parsed["image"].clone()] } else { vec![] };
                json!({"created": now_s(), "data": imgs.iter().map(|i| json!({"url": if i["url"].is_string() { i["url"].clone() } else { i.clone() }})).collect::<Vec<_>>()})
            }
            Kind::Stability => json!({"created": now_s(), "data": parsed["image"].as_str().map(|i| vec![json!({"b64_json": i})]).unwrap_or_default()}),
            Kind::Bfl => json!({"created": now_s(), "data": parsed["result"]["sample"].as_str().map(|u| vec![json!({"url": u})]).unwrap_or_default()}),
            Kind::Runway => json!({"created": now_s(), "data": parsed["output"].as_array().into_iter().flatten().map(|u| json!({"url": u})).collect::<Vec<_>>()}),
            Kind::Cloudflare => cf_normalize(&parsed),
            _ => parsed.clone(),
        };
        if truthy(&normalized["created"]) && normalized["data"].is_array() { normalized } else { parsed }
    };
    if binary {
        if let Some(r) = binary_response(&final_body, &body).await {
            return MediaResult::ok(r);
        }
    }
    MediaResult::json(&final_body)
}

fn tokio_stream_from(mut rx: tokio::sync::mpsc::UnboundedReceiver<Bytes>) -> crate::exec::ByteStream {
    let s = async_stream::stream! {
        while let Some(b) = rx.recv().await {
            yield Ok::<Bytes, String>(b);
        }
    };
    s.boxed()
}

#[derive(serde::Deserialize, Default)]
pub struct ImgQuery {
    pub response_format: Option<String>,
}

pub async fn single(db: std::sync::Arc<crate::db::Db>, body: Value, model_str: String, stream_to_client: bool, binary: bool, preferred: Option<String>) -> Response {
    let Some((provider, model)) = resolve_model(&db, &model_str) else {
        return error_response(400, "Invalid model format", &[]);
    };
    let k = kind(&provider);
    let na = k.map(no_auth).unwrap_or(false);
    with_accounts(db, &provider, &model, na, preferred.as_deref(), |ctx| {
        let b = body.clone();
        async move { core(ctx, b, stream_to_client, binary).await }
    })
    .await
}

/// Media combos: try each model until one succeeds (handleComboChat semantics).
pub async fn combo<F, Fut>(db: &crate::db::Db, name: &str, models: Vec<String>, mut f: F) -> Response
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = Response>,
{
    let st = crate::chat::accounts::settings(db);
    let strategy = st["comboStrategies"][name]["fallbackStrategy"].as_str().or_else(|| st["comboStrategy"].as_str()).unwrap_or("fallback").to_string();
    let list = crate::chat::rotated_models(&models, name, &strategy, st["comboStickyRoundRobinLimit"].as_i64().unwrap_or(1));
    let mut last: Option<(u16, String)> = None;
    for m in list {
        let r = f(m).await;
        let s = r.status().as_u16();
        if (200..300).contains(&s) {
            return r;
        }
        let (parts, b) = r.into_parts();
        let bytes = axum::body::to_bytes(b, usize::MAX).await.unwrap_or_default();
        let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        let msg = v["error"]["message"].as_str().or_else(|| v["error"].as_str()).unwrap_or("").to_string();
        if !crate::chat::accounts::check_fallback_error(s, &msg, 0, None).should_fallback {
            return Response::from_parts(parts, axum::body::Body::from(bytes));
        }
        last.get_or_insert((s, msg));
    }
    let (s, m) = last.unwrap_or((503, "All combo models unavailable".into()));
    json_response(s, &json!({"error": {"message": m}}), &[])
}

pub async fn handle(State(st): State<AppState>, Query(q): Query<ImgQuery>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = crate::api::authorize(&st, &headers, None) {
        return r;
    }
    let body = match crate::api::parse_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let Some(model_str) = body["model"].as_str().filter(|s| !s.is_empty()).map(str::to_owned) else {
        return error_response(400, "Missing model", &[]);
    };
    if !truthy(&body["prompt"]) {
        return error_response(400, "Missing required field: prompt", &[]);
    }
    let wants_stream = headers.get("accept").and_then(|v| v.to_str().ok()).map(|a| a.contains("text/event-stream")).unwrap_or(false);
    let binary = q.response_format.as_deref() == Some("binary");
    let preferred = headers.get("x-connection-id").and_then(|v| v.to_str().ok()).map(str::to_owned);
    if let Some(models) = crate::chat::combo_models(&st.db, &model_str) {
        let db = st.db.clone();
        return combo(&st.db, &model_str, models, |m| single(db.clone(), body.clone(), m, wants_stream, binary, preferred.clone())).await;
    }
    single(st.db.clone(), body, model_str, wants_stream, binary, preferred).await
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters() {
        assert!(kind("openai") == Some(Kind::OpenAi));
        assert!(kind("venice") == Some(Kind::OpenAi));
        let b = openai_body("xai", "grok-2-image", &json!({"prompt": "p", "size": "1x1", "quality": "hd"}));
        assert!(b.get("size").is_none() && b["prompt"] == "p");
        let c = codex_body("gpt-image-2", &json!({"prompt": "cat", "image": "AAA"}));
        assert_eq!(c["model"], "gpt-5.5");
        assert_eq!(c["tools"][0]["action"], "edit");
        assert_eq!(cf_normalize(&json!({"result": {"image": "abc"}}))["data"][0]["b64_json"], "abc");
    }
}
