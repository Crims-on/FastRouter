//! POST /v1/audio/transcriptions (port of stt.js, sttCore.js, geminiLiveStt.js).

use std::time::Duration;

use axum::extract::{Multipart, State};
use axum::http::HeaderMap;
use axum::response::Response;
use base64::Engine;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};

use super::*;
use crate::AppState;
use crate::registry::REG;

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

#[derive(Clone, Default)]
pub struct Form {
    pub fields: std::collections::HashMap<String, String>,
    pub file: Option<(Vec<u8>, String, String)>, // bytes, filename, content-type
}

impl Form {
    fn get(&self, k: &str) -> Option<&str> {
        self.fields.get(k).map(|s| s.as_str()).filter(|s| !s.is_empty())
    }
}

fn stt_cfg(p: &str) -> Value {
    media_cfg(p, "sttConfig")
}

fn credentialed(p: &str) -> bool {
    let c = stt_cfg(p);
    c.is_null() || c["authType"] != "none"
}

fn audio_mime(f: &(Vec<u8>, String, String)) -> String {
    let t = f.2.to_lowercase();
    if t.starts_with("audio/") {
        return t;
    }
    let ext = f.1.rsplit('.').next().unwrap_or("").to_lowercase();
    match ext.as_str() {
        "mp3" => "audio/mpeg",
        "mp4" | "m4a" => "audio/mp4",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "webm" => "audio/webm",
        "aac" => "audio/aac",
        "opus" => "audio/opus",
        _ => "application/octet-stream",
    }
    .into()
}

fn auth(cfg: &Value, token: &str) -> Vec<(String, String)> {
    if token.is_empty() {
        return vec![];
    }
    let v = match cfg["authHeader"].as_str().unwrap_or("bearer") {
        "token" => ("authorization", format!("Token {token}")),
        "x-api-key" => ("x-api-key", token.to_string()),
        "key" => ("authorization", format!("Key {token}")),
        "authorization" => ("authorization", token.to_string()),
        _ => ("authorization", format!("Bearer {token}")),
    };
    vec![(v.0.to_string(), v.1)]
}

async fn up_err(r: reqwest::Response) -> MediaResult {
    let st = r.status().as_u16();
    let txt = r.text().await.unwrap_or_default();
    let msg = serde_json::from_str::<Value>(&txt)
        .ok()
        .and_then(|j| j["error"]["message"].as_str().map(str::to_owned).or_else(|| j["error"].as_str().map(str::to_owned)).or_else(|| j["message"].as_str().map(str::to_owned)))
        .unwrap_or(if txt.is_empty() { format!("Upstream error ({st})") } else { txt });
    MediaResult::err(st, msg)
}

fn model_transport(provider: &str, model: &str, db: &crate::db::Db) -> Option<String> {
    let st = crate::chat::accounts::settings(db);
    if let Some(t) = st["customModels"].as_array().and_then(|a| a.iter().find(|c| c["type"] == "stt" && c["providerAlias"] == provider && c["id"] == model)).and_then(|c| c["transport"].as_str()).filter(|s| !s.trim().is_empty()) {
        return Some(t.trim().into());
    }
    REG.provider_models(&REG.alias_of(provider)).iter().find(|m| m["id"] == model && m["kind"].as_str().unwrap_or("llm") == "stt").and_then(|m| m["transport"].as_str()).map(str::to_owned).filter(|s| !s.is_empty())
}

async fn gemini_live(cfg: &Value, file: &(Vec<u8>, String, String), model: &str, token: &str, form: &Form) -> Result<(String, Vec<String>), (u16, String)> {
    if file.0.is_empty() {
        return Err((400, "Empty audio file".into()));
    }
    let base = cfg["baseUrl"].as_str().unwrap_or("https://generativelanguage.googleapis.com/v1beta/models");
    let mut u = reqwest::Url::parse(base).map_err(|e| (502, e.to_string()))?;
    let _ = u.set_scheme("wss");
    if !u.path().starts_with("/ws/") {
        let p = format!("/ws/api{}", u.path());
        u.set_path(&p);
    }
    let url = format!("{}/{}:bidiGenerateContent?key={}", u.as_str().trim_end_matches('/'), crate::oauth::enc(model), crate::oauth::enc(token));
    let instruction = form.get("prompt").unwrap_or("Transcribe the spoken audio verbatim.").to_string();
    let lang = form.get("language").unwrap_or("");
    let num = |k: &str, d: u64| form.get(k).and_then(|s| s.parse::<u64>().ok()).filter(|n| *n > 0).map(|n| n.min(300_000)).unwrap_or(d);
    let setup_to = Duration::from_millis(num("setup_timeout_ms", 10_000));
    let turn_to = Duration::from_millis(num("turn_timeout_ms", 60_000));
    let system = form.get("system_instruction").map(str::to_owned).unwrap_or_else(|| if lang.is_empty() { instruction.clone() } else { format!("{instruction} Language: {lang}.") });
    let mime = audio_mime(file);
    let (mut ws, _) = tokio::time::timeout(setup_to, tokio_tungstenite::connect_async(&url)).await.map_err(|_| (504, "Gemini Live timed out waiting for setupComplete".to_string()))?.map_err(|_| (502, "Gemini Live websocket connection failed".to_string()))?;
    use tokio_tungstenite::tungstenite::Message;
    let setup = json!({"setup": {"model": format!("models/{model}"), "generationConfig": {"responseModalities": ["TEXT"], "inputAudioTranscription": {}}, "systemInstruction": {"parts": [{"text": system}]}}});
    ws.send(Message::text(setup.to_string())).await.map_err(|_| (502, "Gemini Live websocket connection failed".to_string()))?;
    let mut text = String::new();
    let mut chunks = vec![];
    let mut deadline = tokio::time::Instant::now() + setup_to;
    let mut streamed = false;
    loop {
        let msg = match tokio::time::timeout_at(deadline, ws.next()).await {
            Err(_) => return Err((504, if streamed { "Gemini Live transcription timed out" } else { "Gemini Live timed out waiting for setupComplete" }.into())),
            Ok(None) | Ok(Some(Err(_))) => {
                if !text.trim().is_empty() {
                    return Ok((text, chunks));
                }
                return Err((502, "Gemini Live socket closed before completion".into()));
            }
            Ok(Some(Ok(m))) => m,
        };
        let data = match msg {
            Message::Text(t) => t.to_string(),
            Message::Binary(b) => String::from_utf8_lossy(&b).into_owned(),
            Message::Close(_) => {
                if !text.trim().is_empty() {
                    return Ok((text, chunks));
                }
                return Err((502, "Gemini Live socket closed before completion".into()));
            }
            _ => continue,
        };
        let Ok(f) = serde_json::from_str::<Value>(&data) else { continue };
        if f["error"].is_object() {
            return Err((502, format!("Gemini Live error: {}", f["error"]["message"].as_str().unwrap_or("unknown"))));
        }
        let sc = &f["serverContent"];
        if let Some(d) = sc["inputTranscription"]["text"].as_str().filter(|d| !d.trim().is_empty()) {
            text.push_str(d);
            chunks.push(d.to_string());
        }
        if f.get("setupComplete").is_some() || sc.get("setupComplete").is_some() {
            deadline = tokio::time::Instant::now() + turn_to;
            streamed = true;
            for c in file.0.chunks(16_384) {
                let fr = json!({"realtimeInput": {"mediaChunks": [{"mimeType": mime, "data": B64.encode(c)}]}});
                ws.send(Message::text(fr.to_string())).await.map_err(|_| (502, "Gemini Live socket closed while streaming audio".to_string()))?;
            }
            let fin = json!({"clientContent": {"turns": [{"parts": [{"text": system}]}], "turnComplete": true}});
            let _ = ws.send(Message::text(fin.to_string())).await;
            continue;
        }
        if sc["turnComplete"] == json!(true) {
            let _ = ws.close(None).await;
            return Ok((text, chunks));
        }
    }
}

async fn core(ctx: Ctx, form: Form, transport: Option<String>) -> MediaResult {
    let Some(file) = form.file.clone() else { return MediaResult::err(400, "Missing required field: file") };
    let mut cfg = stt_cfg(&ctx.provider);
    if cfg.is_null() {
        if crate::exec::is_openai_compatible(&ctx.provider) {
            cfg = json!({"authType": "apikey", "authHeader": "bearer", "format": "openai", "baseUrl": format!("{}/audio/transcriptions", ctx.creds["providerSpecificData"]["baseUrl"].as_str().unwrap_or("https://api.openai.com/v1").trim_end_matches('/'))});
        } else {
            return MediaResult::err(400, format!("Provider '{}' does not support STT", ctx.provider));
        }
    } else if let Some(o) = ctx.creds["providerSpecificData"]["baseUrl"].as_str().filter(|s| !s.is_empty()) {
        cfg["baseUrl"] = json!(o.trim_end_matches('/'));
    }
    let token = if cfg["authType"] == "none" { String::new() } else { key_of(&ctx.creds) };
    if cfg["authType"] != "none" && token.is_empty() {
        return MediaResult::err(401, format!("No credentials for STT provider: {}", ctx.provider));
    }
    let base = cfg["baseUrl"].as_str().unwrap_or("").to_string();
    let model = ctx.model.clone();
    let c = client(&ctx.creds);
    let hs = auth(&cfg, &token);
    let with = |mut rb: reqwest::RequestBuilder| {
        for (k, v) in &hs {
            rb = rb.header(k.as_str(), v.as_str());
        }
        rb.timeout(Duration::from_secs(600))
    };
    let marker = transport.unwrap_or_else(|| cfg["format"].as_str().unwrap_or("openai").to_string());
    let res: Result<MediaResult, String> = async {
        match marker.as_str() {
            "gemini-live" => match gemini_live(&cfg, &file, &model, &token, &form).await {
                Ok((t, ch)) => Ok(if form.get("response_format").map(|f| f.trim().to_lowercase()) == Some("verbose_json".into()) {
                    MediaResult::json(&json!({"text": t, "segments": ch.iter().enumerate().map(|(i, s)| json!({"id": i, "text": s})).collect::<Vec<_>>()}))
                } else {
                    MediaResult::json(&json!({"text": t}))
                }),
                Err((s, m)) => Ok(MediaResult::err(s, m)),
            },
            "deepgram" => {
                let mut u = reqwest::Url::parse(&base).map_err(|e| e.to_string())?;
                {
                    let mut q = u.query_pairs_mut();
                    q.append_pair("model", &model).append_pair("smart_format", "true").append_pair("punctuate", "true");
                    match form.get("language").map(str::trim).filter(|s| !s.is_empty()) {
                        Some(l) => q.append_pair("language", l),
                        None => q.append_pair("detect_language", "true"),
                    };
                }
                let r = with(c.post(u)).header("content-type", audio_mime(&file)).body(file.0.clone()).send().await.map_err(|e| e.to_string())?;
                if !r.status().is_success() {
                    return Ok(up_err(r).await);
                }
                let d: Value = r.json().await.map_err(|e| e.to_string())?;
                Ok(MediaResult::json(&json!({"text": d["results"]["channels"][0]["alternatives"][0]["transcript"].as_str().unwrap_or("")})))
            }
            "assemblyai" => {
                let up = with(c.post("https://api.assemblyai.com/v2/upload")).header("content-type", "application/octet-stream").body(file.0.clone()).send().await.map_err(|e| e.to_string())?;
                if !up.status().is_success() {
                    return Ok(up_err(up).await);
                }
                let u: Value = up.json().await.map_err(|e| e.to_string())?;
                let sub = with(c.post(&base)).json(&json!({"audio_url": u["upload_url"], "speech_models": [model], "language_detection": true})).send().await.map_err(|e| e.to_string())?;
                if !sub.status().is_success() {
                    return Ok(up_err(sub).await);
                }
                let s: Value = sub.json().await.map_err(|e| e.to_string())?;
                let id = crate::jsv::js_string(&s["id"]);
                let start = std::time::Instant::now();
                while start.elapsed() < Duration::from_secs(120) {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    let Ok(p) = with(c.get(format!("{base}/{id}"))).send().await else { continue };
                    if !p.status().is_success() {
                        continue;
                    }
                    let r: Value = p.json().await.unwrap_or(Value::Null);
                    match r["status"].as_str() {
                        Some("completed") => return Ok(MediaResult::json(&json!({"text": r["text"].as_str().unwrap_or("")}))),
                        Some("error") => return Ok(MediaResult::err(500, r["error"].as_str().unwrap_or("AssemblyAI failed"))),
                        _ => {}
                    }
                }
                Ok(MediaResult::err(504, "AssemblyAI timeout after 120s"))
            }
            "nvidia-asr" => {
                let f = reqwest::multipart::Form::new().part("file", reqwest::multipart::Part::bytes(file.0.clone()).file_name(if file.1.is_empty() { "audio.wav".to_string() } else { file.1.clone() })).text("model", model.clone());
                let r = with(c.post(&base)).multipart(f).send().await.map_err(|e| e.to_string())?;
                if !r.status().is_success() {
                    return Ok(up_err(r).await);
                }
                let d: Value = r.json().await.map_err(|e| e.to_string())?;
                Ok(MediaResult::json(&json!({"text": d["text"].as_str().or_else(|| d["transcript"].as_str()).unwrap_or("")})))
            }
            "huggingface-asr" => {
                if model.contains("..") || model.contains("//") {
                    return Ok(MediaResult::err(400, "Invalid model ID"));
                }
                let r = with(c.post(format!("{}/{model}", base.trim_end_matches('/')))).header("content-type", audio_mime(&file)).body(file.0.clone()).send().await.map_err(|e| e.to_string())?;
                if !r.status().is_success() {
                    return Ok(up_err(r).await);
                }
                let d: Value = r.json().await.map_err(|e| e.to_string())?;
                Ok(MediaResult::json(&json!({"text": d["text"].as_str().unwrap_or("")})))
            }
            "gemini-stt" => {
                let mut prompt = form.get("prompt").map(|s| s.trim().to_string()).unwrap_or_else(|| "Generate a transcript of the speech. Return only the transcribed text, no commentary.".into());
                if let Some(l) = form.get("language").map(str::trim).filter(|s| !s.is_empty()) {
                    prompt.push_str(&format!(" Language: {l}."));
                }
                let r = c
                    .post(format!("{base}/{model}:generateContent?key={}", crate::oauth::enc(&token)))
                    .json(&json!({"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": audio_mime(&file), "data": B64.encode(&file.0)}}]}]}))
                    .timeout(Duration::from_secs(600))
                    .send()
                    .await
                    .map_err(|e| e.to_string())?;
                if !r.status().is_success() {
                    return Ok(up_err(r).await);
                }
                let d: Value = r.json().await.map_err(|e| e.to_string())?;
                let t: String = d["candidates"][0]["content"]["parts"].as_array().into_iter().flatten().filter_map(|p| p["text"].as_str()).collect();
                Ok(MediaResult::json(&json!({"text": t})))
            }
            _ => {
                let mut f = reqwest::multipart::Form::new().part("file", reqwest::multipart::Part::bytes(file.0.clone()).file_name(if file.1.is_empty() { "audio.wav".to_string() } else { file.1.clone() }).mime_str(&audio_mime(&file)).map_err(|e| e.to_string())?).text("model", model.clone());
                for k in ["language", "prompt", "response_format", "temperature"] {
                    if let Some(v) = form.get(k) {
                        f = f.text(k, v.to_string());
                    }
                }
                let r = with(c.post(&base)).multipart(f).send().await.map_err(|e| e.to_string())?;
                if !r.status().is_success() {
                    return Ok(up_err(r).await);
                }
                let ct = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("application/json").to_string();
                let txt = r.text().await.unwrap_or_default();
                let mut resp = Response::new(axum::body::Body::from(txt));
                resp.headers_mut().insert("content-type", ct.parse().unwrap());
                resp.headers_mut().insert("access-control-allow-origin", "*".parse().unwrap());
                Ok(MediaResult::ok(resp))
            }
        }
    }
    .await;
    res.unwrap_or_else(|e| MediaResult::err(502, e))
}

pub async fn read_form(mut mp: Multipart) -> Result<Form, String> {
    let mut f = Form::default();
    while let Some(field) = mp.next_field().await.map_err(|e| e.to_string())? {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" || field.file_name().is_some() {
            let fname = field.file_name().unwrap_or("audio.wav").to_string();
            let ct = field.content_type().unwrap_or("").to_string();
            let b = field.bytes().await.map_err(|e| e.to_string())?;
            if name == "file" || f.file.is_none() {
                f.file = Some((b.to_vec(), fname, ct));
            }
        } else {
            let v = field.text().await.map_err(|e| e.to_string())?;
            f.fields.insert(name, v);
        }
    }
    Ok(f)
}

pub async fn handle(State(st): State<AppState>, headers: HeaderMap, mp: Multipart) -> Response {
    if let Err(r) = crate::api::authorize(&st, &headers, None) {
        return r;
    }
    let form = match read_form(mp).await {
        Ok(f) => f,
        Err(_) => return error_response(400, "Invalid multipart form data", &[]),
    };
    let Some(model_str) = form.get("model").map(str::to_owned) else {
        return error_response(400, "Missing model", &[]);
    };
    if form.file.is_none() {
        return error_response(400, "Missing required field: file", &[]);
    }
    let Some((provider, model)) = resolve_model(&st.db, &model_str) else {
        return error_response(400, "Invalid model format", &[]);
    };
    let transport = model_transport(&provider, &model, &st.db);
    with_accounts(st.db.clone(), &provider, &model, !credentialed(&provider), None, |ctx| {
        let (form, transport) = (form.clone(), transport.clone());
        async move { core(ctx, form, transport).await }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        assert_eq!(audio_mime(&(vec![], "a.m4a".into(), "".into())), "audio/mp4");
        assert_eq!(auth(&json!({"authHeader": "token"}), "k")[0].1, "Token k");
        assert!(credentialed("openai"));
    }
}
