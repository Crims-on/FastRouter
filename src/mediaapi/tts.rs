//! POST /v1/audio/speech and GET /v1/audio/voices (port of tts.js, ttsCore.js
//! and ttsProviders/*). AWS Polly (SigV4) is implemented here as well.

use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use base64::Engine;
use bytes::Bytes;
use futures::StreamExt;
use hmac::{KeyInit, Mac};
use serde_json::{Value, json};
use sha2::Digest;

use super::*;
use crate::AppState;
use crate::registry::REG;

const UA: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36";
const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

pub struct Audio {
    pub base64: String,
    pub format: String,
}

fn tts_cfg(p: &str) -> Value {
    media_cfg(p, "ttsConfig")
}

/// Providers that need a stored connection (CREDENTIALED_PROVIDERS).
pub fn credentialed(provider: &str) -> bool {
    let cfg = tts_cfg(provider);
    if cfg.is_null() {
        // Custom nodes and unknown providers go through the account loop.
        return true;
    }
    cfg["authType"] != "none" && REG.transport(provider)["noAuth"] != json!(true)
}

fn http() -> reqwest::Client {
    crate::exec::http_client(None)
}

async fn upstream_msg(r: reqwest::Response) -> String {
    let st = r.status().as_u16();
    let text = r.text().await.unwrap_or_default();
    match serde_json::from_str::<Value>(&text) {
        Ok(p) => p["error"]["message"].as_str().or_else(|| p["message"].as_str()).or_else(|| p["detail"]["message"].as_str()).or_else(|| p["detail"].as_str()).map(str::to_owned).unwrap_or(if text.is_empty() { format!("Upstream error ({st})") } else { text }),
        Err(_) => if text.is_empty() { format!("Upstream error ({st})") } else { text },
    }
}

async fn to_audio(r: reqwest::Response, default: &str) -> Result<Audio, String> {
    let ct = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let b = r.bytes().await.map_err(|e| e.to_string())?;
    if b.len() < 100 {
        return Err("Upstream returned empty audio".into());
    }
    let format = if ct.contains("wav") { "wav" } else if ct.contains("mpeg") || ct.contains("mp3") { "mp3" } else if ct.contains("ogg") { "ogg" } else { default };
    Ok(Audio { base64: B64.encode(&b), format: format.into() })
}

/// parseModelVoice(model, defaultModel, defaultVoice, knownModels)
pub fn parse_model_voice(model: &str, default_model: &str, default_voice: &str, known: &[String]) -> (String, String) {
    if model.is_empty() {
        return (default_model.into(), default_voice.into());
    }
    let mut k: Vec<&String> = known.iter().filter(|s| !s.is_empty()).collect();
    k.sort_by_key(|s| std::cmp::Reverse(s.len()));
    for id in k {
        if model == id {
            return (id.clone(), default_voice.into());
        }
        if let Some(v) = model.strip_prefix(&format!("{id}/")) {
            return (id.clone(), v.into());
        }
    }
    if let Some(i) = model.rfind('/').filter(|i| *i > 0) {
        return (model[..i].into(), model[i + 1..].into());
    }
    (if default_model.is_empty() { model.into() } else { default_model.into() }, if default_voice.is_empty() { model.into() } else { default_voice.into() })
}

// ---------------------------------------------------------------------------
// Free web TTS (Google Translate, Bing/Edge)
// ---------------------------------------------------------------------------

static GOOGLE_TOKEN: LazyLock<Mutex<Option<(String, String, std::time::Instant)>>> = LazyLock::new(Default::default);
static GOOGLE_IDX: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

async fn google_tts(text: &str, model: &str) -> Result<Audio, String> {
    let lang = if model.is_empty() { "en" } else { model };
    let cached = GOOGLE_TOKEN.lock().unwrap().clone().filter(|t| t.2.elapsed() < Duration::from_secs(11 * 60));
    let (fsid, bl) = match cached {
        Some((a, b, _)) => (a, b),
        None => {
            let r = http().get("https://translate.google.com/").header("user-agent", UA).send().await.map_err(|e| e.to_string())?;
            if !r.status().is_success() {
                return Err(format!("Google translate fetch failed: {}", r.status().as_u16()));
            }
            let html = r.text().await.unwrap_or_default();
            let f = regex::Regex::new(r#""FdrFJe":"(.*?)""#).unwrap().captures(&html).map(|c| c[1].to_string());
            let b = regex::Regex::new(r#""cfb2h":"(.*?)""#).unwrap().captures(&html).map(|c| c[1].to_string());
            let (Some(f), Some(b)) = (f, b) else { return Err("Failed to parse Google token".into()) };
            *GOOGLE_TOKEN.lock().unwrap() = Some((f.clone(), b.clone(), std::time::Instant::now()));
            (f, b)
        }
    };
    let clean: String = text.chars().map(|c| if "@^*()\\/-_+=><\"'\u{201c}\u{201d}\u{3010}\u{3011}".contains(c) { ' ' } else { c }).collect::<String>().replace(", ", ". ");
    let rpc = "jQ1olc";
    let idx = GOOGLE_IDX.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    let reqid = idx * 100_000 + 1000 + (crate::jsv::now_ms() as u64 % 9000);
    let payload = json!([clean, lang, null, "undefined", [0]]).to_string();
    let freq = json!([[[rpc, payload, null, "generic"]]]).to_string();
    let url = format!(
        "https://translate.google.com/_/TranslateWebserverUi/data/batchexecute?rpcids={rpc}&f.sid={}&bl={}&hl={lang}&soc-app=1&soc-platform=1&soc-device=1&_reqid={reqid}&rt=c",
        crate::oauth::enc(&fsid),
        crate::oauth::enc(&bl)
    );
    let r = http().post(url).header("content-type", "application/x-www-form-urlencoded").header("referer", "https://translate.google.com/").body(format!("f.req={}", crate::oauth::enc(&freq))).send().await.map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        return Err(format!("Google TTS failed: {}", r.status().as_u16()));
    }
    let data = r.text().await.unwrap_or_default();
    let line = data.split('\n').nth(3).ok_or("Google TTS returned empty audio")?;
    let split: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
    let inner: Value = serde_json::from_str(split[0][2].as_str().unwrap_or("[]")).map_err(|e| e.to_string())?;
    let b64 = inner[0].as_str().unwrap_or("");
    if b64.len() < 100 {
        return Err("Google TTS returned empty audio".into());
    }
    Ok(Audio { base64: b64.into(), format: "mp3".into() })
}

static BING_TOKEN: LazyLock<Mutex<Option<(String, String, String, std::time::Instant)>>> = LazyLock::new(Default::default);

async fn bing_token(force: bool) -> Result<(String, String, String), String> {
    if !force {
        if let Some((k, t, c, at)) = BING_TOKEN.lock().unwrap().clone() {
            if at.elapsed() < Duration::from_secs(300) {
                return Ok((k, t, c));
            }
        }
    }
    let r = http().get("https://www.bing.com/translator").header("user-agent", UA).header("accept-language", "vi,en-US;q=0.9,en;q=0.8").send().await.map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        return Err(format!("Bing translator fetch failed: {}", r.status().as_u16()));
    }
    let cookie = r.headers().get_all("set-cookie").iter().filter_map(|v| v.to_str().ok()).map(|c| c.split(';').next().unwrap_or("").to_string()).collect::<Vec<_>>().join("; ");
    let html = r.text().await.unwrap_or_default();
    let c = regex::Regex::new(r"params_AbusePreventionHelper\s*=\s*\[([^,]+),([^,]+),").unwrap().captures(&html).ok_or("Failed to parse Bing token")?;
    let (k, t) = (c[1].to_string(), c[2].replace('"', ""));
    *BING_TOKEN.lock().unwrap() = Some((k.clone(), t.clone(), cookie.clone(), std::time::Instant::now()));
    Ok((k, t, cookie))
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

async fn edge_tts(text: &str, model: &str) -> Result<Audio, String> {
    let voice = if model.is_empty() { "vi-VN-HoaiMyNeural" } else { model };
    let parts: Vec<&str> = voice.split('-').collect();
    let lang = parts.iter().take(2).copied().collect::<Vec<_>>().join("-");
    let gender = if voice.to_lowercase().contains("male") { "Male" } else { "Female" };
    let ssml = format!("<speak version='1.0' xml:lang='{lang}'><voice xml:lang='{lang}' xml:gender='{gender}' name='{voice}'><prosody rate='0.00%'>{}</prosody></voice></speak>", xml_escape(text));
    let send = |tok: (String, String, String)| {
        let ssml = ssml.clone();
        async move {
            let body = format!("ssml={}&token={}&key={}", crate::oauth::enc(&ssml), crate::oauth::enc(&tok.1), crate::oauth::enc(&tok.0));
            let mut rb = http()
                .post("https://www.bing.com/tfettts?isVertical=1&&IG=1&IID=translator.5023&SFX=1")
                .header("content-type", "application/x-www-form-urlencoded")
                .header("accept", "*/*")
                .header("origin", "https://www.bing.com")
                .header("referer", "https://www.bing.com/translator")
                .header("user-agent", UA)
                .body(body);
            if !tok.2.is_empty() {
                rb = rb.header("cookie", tok.2);
            }
            rb.send().await.map_err(|e| e.to_string())
        }
    };
    let mut r = send(bing_token(false).await?).await?;
    if matches!(r.status().as_u16(), 429 | 403) {
        r = send(bing_token(true).await?).await?;
    }
    if !r.status().is_success() {
        let s = r.status().as_u16();
        let b = r.text().await.unwrap_or_default();
        return Err(format!("Bing TTS failed: {s}{}", if b.is_empty() { String::new() } else { format!(" - {b}") }));
    }
    let b = r.bytes().await.map_err(|e| e.to_string())?;
    if b.len() < 1024 {
        return Err("Bing TTS returned empty audio".into());
    }
    Ok(Audio { base64: B64.encode(&b), format: "mp3".into() })
}

async fn local_device(text: &str, model: &str) -> Result<Audio, String> {
    let dir = std::env::temp_dir().join(format!("tts-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mp3 = dir.join("out.mp3");
    let res = async {
        if cfg!(target_os = "macos") {
            let aiff = dir.join("out.aiff");
            let mut args: Vec<String> = vec![];
            if !model.is_empty() && model != "default" {
                args.extend(["-v".into(), model.into()]);
            }
            args.extend(["-o".into(), aiff.to_string_lossy().into(), text.into()]);
            let st = tokio::process::Command::new("say").args(&args).status().await.map_err(|e| e.to_string())?;
            if !st.success() {
                return Err("say failed".to_string());
            }
            let st = tokio::process::Command::new("ffmpeg").args(["-y", "-i", &aiff.to_string_lossy(), "-codec:a", "libmp3lame", "-qscale:a", "4", &mp3.to_string_lossy()]).status().await.map_err(|e| e.to_string())?;
            if !st.success() {
                return Err("ffmpeg failed".to_string());
            }
        } else {
            // Linux/Windows: espeak-ng (or espeak) → wav → mp3.
            let wav = dir.join("out.wav");
            let mut ok = false;
            // Try the requested voice first, then the engine's default voice.
            let voices: Vec<Option<&str>> = if model.is_empty() || model == "default" { vec![None] } else { vec![Some(model), None] };
            'outer: for bin in ["espeak-ng", "espeak"] {
                for v in &voices {
                    let mut c = tokio::process::Command::new(bin);
                    if let Some(v) = v {
                        c.args(["-v", v]);
                    }
                    if let Ok(st) = c.args(["-w", &wav.to_string_lossy(), text]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().await {
                        if st.success() {
                            ok = true;
                            break 'outer;
                        }
                    }
                }
            }
            if !ok {
                return Err("No local speech engine found (install espeak-ng)".to_string());
            }
            let st = tokio::process::Command::new("ffmpeg").args(["-y", "-i", &wav.to_string_lossy(), "-codec:a", "libmp3lame", "-qscale:a", "4", &mp3.to_string_lossy()]).status().await;
            if !matches!(st, Ok(s) if s.success()) {
                let b = std::fs::read(&wav).map_err(|e| e.to_string())?;
                return Ok(Audio { base64: B64.encode(b), format: "wav".into() });
            }
        }
        let b = std::fs::read(&mp3).map_err(|e| e.to_string())?;
        Ok(Audio { base64: B64.encode(b), format: "mp3".into() })
    }
    .await;
    let _ = std::fs::remove_dir_all(&dir);
    res
}

// ---------------------------------------------------------------------------
// Keyed adapters
// ---------------------------------------------------------------------------

async fn openai_tts(text: &str, model: &str, creds: &Value) -> Result<Audio, String> {
    let key = key_of(creds);
    if key.is_empty() {
        return Err("No OpenAI API key configured".into());
    }
    let mut tts_model = tts_cfg("openai")["defaultModel"].as_str().unwrap_or("gpt-4o-mini-tts").to_string();
    let mut voice = "alloy".to_string();
    if model.contains('/') {
        let p: Vec<&str> = model.split('/').collect();
        if p.len() == 2 {
            tts_model = p[0].into();
            voice = p[1].into();
        }
    } else if !model.is_empty() {
        voice = model.into();
    }
    let base = creds["baseUrl"].as_str().or_else(|| creds["providerSpecificData"]["baseUrl"].as_str()).unwrap_or("https://api.openai.com").trim_end_matches('/').to_string();
    let base = base.strip_suffix("/v1").unwrap_or(&base).to_string();
    let r = client(creds).post(format!("{base}/v1/audio/speech")).bearer_auth(&key).json(&json!({"model": tts_model, "voice": voice, "input": text})).send().await.map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        let s = r.status().as_u16();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        return Err(v["error"]["message"].as_str().map(str::to_owned).unwrap_or(format!("OpenAI TTS failed: {s}")));
    }
    let b = r.bytes().await.map_err(|e| e.to_string())?;
    Ok(Audio { base64: B64.encode(&b), format: "mp3".into() })
}

async fn elevenlabs(text: &str, model: &str, creds: &Value) -> Result<Audio, String> {
    let key = key_of(creds);
    if key.is_empty() {
        return Err("ElevenLabs API key required".into());
    }
    let (mid, vid) = match model.split_once('/') {
        Some((a, b)) => (a.to_string(), b.to_string()),
        None => ("eleven_flash_v2_5".to_string(), model.to_string()),
    };
    let r = client(creds).post(format!("https://api.elevenlabs.io/v1/text-to-speech/{vid}")).header("xi-api-key", key).json(&json!({"text": text, "model_id": mid, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}})).send().await.map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        let s = r.status().as_u16();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        return Err(v["detail"]["message"].as_str().map(str::to_owned).unwrap_or(format!("ElevenLabs TTS failed: {s}")));
    }
    let b = r.bytes().await.map_err(|e| e.to_string())?;
    if b.len() < 1024 {
        return Err("ElevenLabs TTS returned empty audio".into());
    }
    Ok(Audio { base64: B64.encode(&b), format: "mp3".into() })
}

async fn openrouter_tts(text: &str, model: &str, creds: &Value) -> Result<Audio, String> {
    let key = key_of(creds);
    if key.is_empty() {
        return Err("No OpenRouter API key configured".into());
    }
    let cfg = tts_cfg("openrouter");
    let mut tts_model = cfg["defaultModel"].as_str().unwrap_or("openai/gpt-4o-mini-tts").to_string();
    let mut voice = "alloy".to_string();
    if let Some(i) = model.rfind('/') {
        let (m, v) = (&model[..i], &model[i + 1..]);
        if m.contains('/') {
            tts_model = m.into();
            voice = v.into();
        } else {
            voice = model.into();
        }
    } else if !model.is_empty() {
        voice = model.into();
    }
    let mut rb = client(creds).post(cfg["baseUrl"].as_str().unwrap_or("https://openrouter.ai/api/v1/chat/completions")).bearer_auth(&key);
    for (k, v) in cfg["headers"].as_object().into_iter().flatten() {
        rb = rb.header(k.as_str(), crate::jsv::js_string(v));
    }
    let r = rb.json(&json!({"model": tts_model, "modalities": ["text", "audio"], "audio": {"voice": voice, "format": "wav"}, "stream": true, "messages": [{"role": "user", "content": text}]})).send().await.map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        let s = r.status().as_u16();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        return Err(v["error"]["message"].as_str().map(str::to_owned).unwrap_or(format!("OpenRouter TTS failed: {s}")));
    }
    // Each delta carries its own base64 segment (padded independently), so
    // decode per chunk and re-encode the joined bytes.
    let mut audio: Vec<u8> = vec![];
    let mut lp = crate::sse::LineParser::default();
    let mut s = r.bytes_stream();
    while let Some(Ok(c)) = s.next().await {
        for l in lp.push(&c) {
            if let Some(d) = l.strip_prefix("data:").map(str::trim).filter(|d| *d != "[DONE]") {
                if let Ok(v) = serde_json::from_str::<Value>(d) {
                    if let Some(a) = v["choices"][0]["delta"]["audio"]["data"].as_str() {
                        if let Ok(b) = B64.decode(a.trim()) {
                            audio.extend_from_slice(&b);
                        }
                    }
                }
            }
        }
    }
    if audio.is_empty() {
        return Err("OpenRouter TTS returned no audio data".into());
    }
    Ok(Audio { base64: B64.encode(&audio), format: "wav".into() })
}

fn gemini_tts_models() -> Vec<String> {
    let mut v: Vec<String> = vec![];
    for m in tts_cfg("gemini")["models"].as_array().into_iter().flatten().chain(REG.provider_models("gemini-tts-models").iter()).chain(REG.provider_models("gemini").iter().filter(|m| m["kind"] == "tts" || m["type"] == "tts")) {
        if let Some(i) = m["id"].as_str() {
            if !v.iter().any(|x| x == i) {
                v.push(i.into());
            }
        }
    }
    v
}

pub fn pcm_to_wav(pcm: &[u8]) -> Vec<u8> {
    let (rate, ch, bits) = (24000u32, 1u16, 16u16);
    let mut h = Vec::with_capacity(44 + pcm.len());
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&ch.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate * ch as u32 * bits as u32 / 8).to_le_bytes());
    h.extend_from_slice(&(ch * bits / 8).to_le_bytes());
    h.extend_from_slice(&bits.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    h.extend_from_slice(pcm);
    h
}

async fn gemini_tts(text: &str, model: &str, creds: &Value, language: &str) -> Result<Audio, String> {
    let key = key_of(creds);
    if key.is_empty() {
        return Err("No Gemini API key configured".into());
    }
    let known = gemini_tts_models();
    let def = known.first().cloned().unwrap_or_else(|| "gemini-3.1-flash-tts-preview".into());
    let (mid, vid) = if model.is_empty() {
        (def.clone(), "Kore".to_string())
    } else if let Some(id) = known.iter().find(|id| model == id.as_str() || model.starts_with(&format!("{id}/"))) {
        (id.clone(), model.strip_prefix(&format!("{id}/")).unwrap_or("Kore").to_string())
    } else {
        (def.clone(), model.to_string())
    };
    let prompt = if regex::Regex::new(r":\s").unwrap().is_match(text) { text.to_string() } else if !language.is_empty() { format!("Say in {language}: {text}") } else { format!("Say: {text}") };
    let base = tts_cfg("gemini")["baseUrl"].as_str().unwrap_or("https://generativelanguage.googleapis.com/v1beta/models").to_string();
    let r = client(creds)
        .post(format!("{base}/{mid}:generateContent?key={}", crate::oauth::enc(&key)))
        .json(&json!({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": vid}}}}}))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        let s = r.status().as_u16();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        return Err(v["error"]["message"].as_str().map(str::to_owned).unwrap_or(format!("Gemini TTS failed: {s}")));
    }
    let d: Value = r.json().await.map_err(|e| e.to_string())?;
    let b64 = d["candidates"][0]["content"]["parts"].as_array().and_then(|p| p.iter().find_map(|x| x["inlineData"]["data"].as_str()));
    let Some(b64) = b64 else {
        let reason = d["candidates"][0]["finishReason"].as_str().or_else(|| d["promptFeedback"]["blockReason"].as_str()).unwrap_or("unknown");
        return Err(format!("Gemini TTS returned no audio (finishReason: {reason}, voice: {vid}, model: {mid})"));
    };
    let pcm = B64.decode(b64).map_err(|e| e.to_string())?;
    Ok(Audio { base64: B64.encode(pcm_to_wav(&pcm)), format: "wav".into() })
}

async fn mimo_tts(text: &str, model: &str, creds: &Value, style: &str, language: &str) -> Result<Audio, String> {
    let key = key_of(creds);
    if key.is_empty() {
        return Err("xiaomi-mimo API key required".into());
    }
    let (mid, vid) = parse_model_voice(model, "mimo-v2.5-tts", "mimo_default", &["mimo-v2.5-tts".to_string()]);
    let mut ins = vec![];
    if !language.is_empty() {
        ins.push(format!("Speak in {language}."));
    }
    if !style.is_empty() {
        ins.push(style.to_string());
    }
    let mut msgs = vec![json!({"role": "assistant", "content": text})];
    if !ins.is_empty() {
        msgs.insert(0, json!({"role": "user", "content": ins.join(" ")}));
    }
    let r = client(creds).post("https://api.xiaomimimo.com/v1/chat/completions").bearer_auth(&key).json(&json!({"model": mid, "stream": false, "messages": msgs, "audio": {"format": "wav", "voice": if vid.is_empty() { "mimo_default".to_string() } else { vid }}})).send().await.map_err(|e| e.to_string())?;
    let ok = r.status().is_success();
    let st = r.status().as_u16();
    let raw = r.text().await.unwrap_or_default();
    let d: Value = serde_json::from_str(&raw).unwrap_or(json!({}));
    if !ok {
        return Err(d["error"]["message"].as_str().map(str::to_owned).unwrap_or(if raw.is_empty() { format!("MiMo TTS error ({st})") } else { raw }));
    }
    let a = d["choices"][0]["message"]["audio"]["data"].as_str().ok_or_else(|| d["error"]["message"].as_str().unwrap_or("MiMo TTS returned no audio").to_string())?;
    Ok(Audio { base64: a.into(), format: d["choices"][0]["message"]["audio"]["format"].as_str().unwrap_or("wav").into() })
}

async fn selfhosted_tts(text: &str, model: &str, creds: &Value, fmt: &str) -> Result<Audio, String> {
    let raw = creds["providerSpecificData"]["baseUrl"].as_str().or_else(|| creds["baseUrl"].as_str()).unwrap_or("http://localhost:8880").trim_end_matches('/').to_string();
    let base = raw.strip_suffix("/v1/audio/speech").unwrap_or(&raw);
    let base = base.strip_suffix("/v1").unwrap_or(base).to_string();
    let parts: Vec<&str> = model.split('/').filter(|s| !s.is_empty()).collect();
    let (m, v) = match parts.len() {
        0 => ("kokoro".to_string(), "af_heart".to_string()),
        1 => (parts[0].to_string(), "af_heart".to_string()),
        _ => (parts[0].to_string(), parts[1..].join("/")),
    };
    let fmt = if fmt == "json" { "mp3" } else { fmt };
    let mut rb = client(creds).post(format!("{base}/v1/audio/speech")).json(&json!({"model": m, "voice": v, "input": text, "response_format": fmt}));
    let k = key_of(creds);
    if !k.is_empty() {
        rb = rb.bearer_auth(k);
    }
    let r = rb.send().await.map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        let s = r.status().as_u16();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        return Err(v["error"]["message"].as_str().map(str::to_owned).unwrap_or(format!("Self-hosted TTS failed: {s}")));
    }
    let b = r.bytes().await.map_err(|e| e.to_string())?;
    Ok(Audio { base64: B64.encode(&b), format: fmt.into() })
}

// ---------------------------------------------------------------------------
// AWS Polly (SigV4)
// ---------------------------------------------------------------------------

fn hmac_sha256(key: &[u8], data: &str) -> Vec<u8> {
    let mut m = hmac::Hmac::<sha2::Sha256>::new_from_slice(key).expect("hmac");
    m.update(data.as_bytes());
    m.finalize().into_bytes().to_vec()
}

fn sha256_hex(b: &[u8]) -> String {
    hex::encode(sha2::Sha256::digest(b))
}

/// SigV4 headers for a JSON POST.
pub fn sigv4_headers(method: &str, host: &str, path: &str, region: &str, service: &str, access_key: &str, secret: &str, body: &[u8], amz_date: &str) -> Vec<(String, String)> {
    let date = &amz_date[..8];
    let payload_hash = sha256_hex(body);
    let canonical_headers = format!("content-type:application/json\nhost:{host}\nx-amz-date:{amz_date}\n");
    let signed = "content-type;host;x-amz-date";
    let canonical = format!("{method}\n{path}\n\n{canonical_headers}\n{signed}\n{payload_hash}");
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let sts = format!("AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}", sha256_hex(canonical.as_bytes()));
    let k = hmac_sha256(format!("AWS4{secret}").as_bytes(), date);
    let k = hmac_sha256(&k, region);
    let k = hmac_sha256(&k, service);
    let k = hmac_sha256(&k, "aws4_request");
    let sig = hex::encode(hmac_sha256(&k, &sts));
    vec![
        ("content-type".into(), "application/json".into()),
        ("x-amz-date".into(), amz_date.into()),
        ("authorization".into(), format!("AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed}, Signature={sig}")),
    ]
}

async fn polly(text: &str, model: &str, creds: &Value) -> Result<Audio, String> {
    let secret = key_of(creds);
    let psd = &creds["providerSpecificData"];
    let access = psd["accessKeyId"].as_str().unwrap_or("");
    if secret.is_empty() || access.is_empty() {
        return Err("AWS Polly needs the Secret Access Key as API key and providerSpecificData.accessKeyId".into());
    }
    let region = psd["region"].as_str().filter(|s| !s.is_empty()).unwrap_or("us-east-1");
    let engines = ["standard", "neural", "long-form", "generative"];
    let (engine, voice) = match model.split_once('/') {
        Some((e, v)) if engines.contains(&e) => (e.to_string(), v.to_string()),
        _ if engines.contains(&model) => (model.to_string(), "Joanna".to_string()),
        _ => ("neural".to_string(), if model.is_empty() { "Joanna".to_string() } else { model.to_string() }),
    };
    let host = format!("polly.{region}.amazonaws.com");
    let body = json!({"OutputFormat": "mp3", "Text": text, "VoiceId": voice, "Engine": engine}).to_string();
    let iso = crate::jsv::iso_from_ms(crate::jsv::now_ms());
    let amz = format!("{}{}{}T{}{}{}Z", &iso[0..4], &iso[5..7], &iso[8..10], &iso[11..13], &iso[14..16], &iso[17..19]);
    let hs = sigv4_headers("POST", &host, "/v1/speech", region, "polly", access, &secret, body.as_bytes(), &amz);
    let mut rb = client(creds).post(format!("https://{host}/v1/speech")).body(body);
    for (k, v) in hs {
        rb = rb.header(k, v);
    }
    let r = rb.send().await.map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        return Err(upstream_msg(r).await);
    }
    to_audio(r, "mp3").await
}

// ---------------------------------------------------------------------------
// Generic config-driven formats
// ---------------------------------------------------------------------------

async fn generic(provider: &str, text: &str, model: &str, creds: &Value) -> Result<Option<Audio>, String> {
    let cfg = tts_cfg(provider);
    let Some(format) = cfg["format"].as_str() else { return Ok(None) };
    let key = creds["apiKey"].as_str().unwrap_or("").to_string();
    if cfg["authType"] != "none" && key.is_empty() && !matches!(format, "aws-polly") {
        return Err(format!("{provider} API key required"));
    }
    let models: Vec<String> = REG.provider_models(&REG.alias_of(provider)).iter().filter(|m| m["kind"] == "tts" || m["type"] == "tts").filter_map(|m| m["id"].as_str().map(str::to_owned)).chain(cfg["models"].as_array().into_iter().flatten().filter_map(|m| m["id"].as_str().map(str::to_owned))).collect();
    let def = models.first().cloned().unwrap_or_default();
    let (mid, vid) = parse_model_voice(model, &def, "", &models);
    let base = creds["providerSpecificData"]["baseUrl"].as_str().filter(|s| !s.is_empty() && cfg["authType"] == "none").map(str::to_owned).unwrap_or_else(|| cfg["baseUrl"].as_str().unwrap_or("").to_string());
    let c = client(creds);
    let post = |url: String| c.post(url).header("content-type", "application/json");
    let r = match format {
        "hyperbolic" => {
            let r = post(base).bearer_auth(&key).json(&json!({"text": text})).send().await.map_err(|e| e.to_string())?;
            if !r.status().is_success() {
                return Err(upstream_msg(r).await);
            }
            let d: Value = r.json().await.map_err(|e| e.to_string())?;
            return Ok(Some(Audio { base64: d["audio"].as_str().unwrap_or("").into(), format: "mp3".into() }));
        }
        "deepgram" => {
            let mut u = reqwest::Url::parse(&base).map_err(|e| e.to_string())?;
            u.query_pairs_mut().append_pair("model", if mid.is_empty() { "aura-asteria-en" } else { &mid });
            (post(u.to_string()).header("authorization", format!("Token {key}")).json(&json!({"text": text})), "mp3")
        }
        "nvidia-tts" => (post(base).bearer_auth(&key).json(&json!({"input": {"text": text}, "voice": if vid.is_empty() { "default" } else { &vid }, "model": mid})), "wav"),
        "huggingface-tts" => {
            if mid.is_empty() || mid.contains("..") {
                return Err("Invalid HuggingFace model ID".into());
            }
            (post(format!("{base}/{mid}")).bearer_auth(&key).json(&json!({"inputs": text})), "wav")
        }
        "fish-audio" => {
            let mut b = json!({"text": text, "format": "mp3"});
            if !vid.is_empty() {
                b["reference_id"] = json!(vid);
            }
            (post(base).bearer_auth(&key).header("model", if mid.is_empty() { "s2.1-pro-free" } else { &mid }).json(&b), "mp3")
        }
        "inworld" => {
            let r = post(base).header("authorization", format!("Basic {key}")).json(&json!({"text": text, "voiceId": if vid.is_empty() { "Alex" } else { &vid }, "modelId": if mid.is_empty() { "inworld-tts-1.5-mini" } else { &mid }, "audioConfig": {"audioEncoding": "MP3"}})).send().await.map_err(|e| e.to_string())?;
            if !r.status().is_success() {
                return Err(upstream_msg(r).await);
            }
            let d: Value = r.json().await.map_err(|e| e.to_string())?;
            let a = d["audioContent"].as_str().ok_or("Inworld TTS returned no audio")?;
            return Ok(Some(Audio { base64: a.into(), format: "mp3".into() }));
        }
        "cartesia" => {
            let mut b = json!({"model_id": if mid.is_empty() { "sonic-2" } else { &mid }, "transcript": text, "output_format": {"container": "mp3", "bit_rate": 128000, "sample_rate": 44100}});
            if !vid.is_empty() {
                b["voice"] = json!({"mode": "id", "id": vid});
            }
            (post(base).header("x-api-key", &key).header("cartesia-version", "2024-06-10").json(&b), "mp3")
        }
        "playht" => {
            let (uid, k) = key.split_once(':').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((String::new(), key.clone()));
            (
                post(base).header("accept", "audio/mpeg").header("x-user-id", uid).bearer_auth(if k.is_empty() { &key } else { &k }).json(&json!({"text": text, "voice": if vid.is_empty() { "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json" } else { &vid }, "voice_engine": if mid.is_empty() { "PlayDialog" } else { &mid }, "output_format": "mp3", "speed": 1})),
                "mp3",
            )
        }
        "coqui" => {
            let mut b = json!({"text": text});
            if !vid.is_empty() {
                b["speaker_id"] = json!(vid);
            }
            (post(base).json(&b), "wav")
        }
        "tortoise" => (post(base).json(&json!({"text": text, "voice": if vid.is_empty() { "random" } else { &vid }})), "wav"),
        "openai" | "openai-speech" => {
            let mut rb = post(base).json(&json!({"model": mid, "input": text, "voice": if vid.is_empty() { "alloy" } else { &vid }, "response_format": "mp3", "speed": 1.0}));
            if !key.is_empty() {
                rb = rb.bearer_auth(&key);
            }
            (rb, "mp3")
        }
        "minimax-tts" => {
            let r = post(base).bearer_auth(&key).json(&json!({"model": if mid.is_empty() { "speech-2.8-hd" } else { &mid }, "text": text, "stream": false, "language_boost": "auto", "output_format": "hex", "voice_setting": {"voice_id": if vid.is_empty() { "English_expressive_narrator" } else { &vid }, "speed": 1, "vol": 1, "pitch": 0}, "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1}})).send().await.map_err(|e| e.to_string())?;
            let ok = r.status().is_success();
            let st = r.status().as_u16();
            let raw = r.text().await.unwrap_or_default();
            let d: Value = serde_json::from_str(&raw).unwrap_or(json!({}));
            let br = if d["base_resp"].is_object() { &d["base_resp"] } else { &d["baseResp"] };
            let code = br["status_code"].as_i64().or_else(|| br["statusCode"].as_i64()).unwrap_or(0);
            let msg = br["status_msg"].as_str().or_else(|| br["statusMsg"].as_str()).or_else(|| d["message"].as_str()).unwrap_or("").to_string();
            if !ok {
                return Err(if !msg.is_empty() { msg } else if !raw.is_empty() { raw } else { format!("MiniMax TTS error ({st})") });
            }
            if code != 0 {
                return Err(if msg.is_empty() { "MiniMax TTS upstream error".into() } else { msg });
            }
            let hexs = d["data"]["audio"].as_str().unwrap_or("").trim().to_string();
            let bytes = hex::decode(&hexs).map_err(|_| if hexs.is_empty() { "MiniMax TTS returned no audio".to_string() } else { "MiniMax TTS returned invalid audio".to_string() })?;
            let fmt = d["extra_info"]["audio_format"].as_str().or_else(|| d["extraInfo"]["audioFormat"].as_str()).unwrap_or("mp3");
            return Ok(Some(Audio { base64: B64.encode(bytes), format: fmt.into() }));
        }
        "aws-polly" => return polly(text, model, creds).await.map(Some),
        _ => return Ok(None),
    };
    let (rb, def_fmt) = r;
    let resp = rb.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(upstream_msg(resp).await);
    }
    to_audio(resp, def_fmt).await.map(Some)
}

/// handleTtsCore
pub async fn synthesize(provider: &str, model: &str, text: &str, creds: &Value, fmt: &str, language: &str, style: &str) -> Result<Audio, (u16, String)> {
    let t = text.trim();
    if t.is_empty() {
        return Err((400, "Missing required field: input".into()));
    }
    let r = match provider {
        "google-tts" => google_tts(t, model).await,
        "edge-tts" => edge_tts(t, model).await,
        "local-device" => local_device(t, model).await,
        "elevenlabs" => elevenlabs(t, model, creds).await,
        "openai" => openai_tts(t, model, creds).await,
        "openrouter" => openrouter_tts(t, model, creds).await,
        "gemini" => gemini_tts(t, model, creds, language).await,
        "xiaomi-mimo" => mimo_tts(t, model, creds, style, language).await,
        "selfhosted-tts" => selfhosted_tts(t, model, creds, fmt).await,
        "aws-polly" => polly(t, model, creds).await,
        p if crate::exec::is_openai_compatible(p) => openai_tts(t, model, creds).await,
        p => match generic(p, t, model, creds).await {
            Ok(Some(a)) => Ok(a),
            Ok(None) => return Err((400, format!("Provider '{p}' does not support TTS via this route."))),
            Err(e) => Err(e),
        },
    };
    r.map_err(|e| (502, e))
}

pub fn audio_response(a: Audio, response_format: &str) -> Response {
    if response_format == "json" {
        return json_response(200, &json!({"audio": a.base64, "format": a.format}), &[]);
    }
    let bytes = B64.decode(&a.base64).unwrap_or_default();
    let mut r = Response::new(axum::body::Body::from(bytes));
    r.headers_mut().insert("content-type", format!("audio/{}", a.format).parse().unwrap());
    r.headers_mut().insert("access-control-allow-origin", "*".parse().unwrap());
    r
}

#[derive(serde::Deserialize, Default)]
pub struct TtsQuery {
    pub response_format: Option<String>,
}

async fn single(db: std::sync::Arc<crate::db::Db>, body: Value, model_str: String, fmt: String) -> Response {
    let Some((provider, model)) = resolve_model(&db, &model_str) else {
        return error_response(400, "Invalid model format", &[]);
    };
    let text = crate::jsv::js_string(&body["input"]);
    let lang = body["language"].as_str().unwrap_or("").to_string();
    let style = body["style"].as_str().unwrap_or("").to_string();
    let cred = credentialed(&provider);
    with_accounts(db, &provider, &model, !cred, None, |ctx| {
        let (text, lang, style, fmt) = (text.clone(), lang.clone(), style.clone(), fmt.clone());
        async move {
            match synthesize(&ctx.provider, &ctx.model, &text, &ctx.creds, &fmt, &lang, &style).await {
                Ok(a) => MediaResult::ok(audio_response(a, &fmt)),
                Err((s, m)) => MediaResult::err(s, m),
            }
        }
    })
    .await
}

pub async fn handle(State(st): State<AppState>, Query(q): Query<TtsQuery>, headers: HeaderMap, body: Bytes) -> Response {
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
    if !truthy(&body["input"]) {
        return error_response(400, "Missing required field: input", &[]);
    }
    // OpenAI clients send `response_format` in the body; ?response_format wins.
    let fmt = q.response_format.or_else(|| body["response_format"].as_str().filter(|f| *f == "json").map(str::to_owned)).unwrap_or_else(|| "mp3".into());
    if let Some(models) = crate::chat::combo_models(&st.db, &model_str) {
        let db = st.db.clone();
        return super::images::combo(&st.db, &model_str, models, |m| single(db.clone(), body.clone(), m, fmt.clone())).await;
    }
    single(st.db.clone(), body, model_str, fmt).await
}

// ---------------------------------------------------------------------------
// Voices
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize, Default)]
pub struct VoicesQuery {
    pub provider: Option<String>,
    pub lang: Option<String>,
}

fn first_key(db: &crate::db::Db, provider: &str) -> Option<String> {
    db.connections_for(provider, true).first().and_then(|c| c["apiKey"].as_str().map(str::to_owned)).filter(|s| !s.is_empty())
}

pub async fn list_voices(db: &crate::db::Db, provider: &str) -> Result<Vec<Value>, String> {
    let mut out = vec![];
    match provider {
        "edge-tts" => {
            let r = http().get("https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list?trustedclienttoken=6A5AA1D4EAFF4E9FB37E23D68491D6F4").header("user-agent", UA).send().await.map_err(|e| e.to_string())?;
            let v: Value = r.json().await.map_err(|e| e.to_string())?;
            for x in v.as_array().into_iter().flatten() {
                let loc = x["Locale"].as_str().unwrap_or("");
                let name = x["FriendlyName"].as_str().or_else(|| x["ShortName"].as_str()).unwrap_or("").replace("Microsoft ", "").replace(" Online (Natural) - ", " (");
                out.push(json!({"id": x["ShortName"], "name": name, "lang": loc.split('-').next().unwrap_or(""), "gender": x["Gender"]}));
            }
        }
        "elevenlabs" => {
            let k = first_key(db, "elevenlabs").ok_or("No ElevenLabs connection found")?;
            let r = http().get("https://api.elevenlabs.io/v1/voices").header("xi-api-key", k).send().await.map_err(|e| e.to_string())?;
            let v: Value = r.json().await.map_err(|e| e.to_string())?;
            for x in v["voices"].as_array().into_iter().flatten() {
                out.push(json!({"id": x["voice_id"], "name": x["name"], "lang": x["labels"]["language"].as_str().unwrap_or("en").split('-').next(), "gender": x["labels"]["gender"].as_str().unwrap_or("")}));
            }
        }
        "deepgram" => {
            let k = first_key(db, "deepgram").ok_or("No Deepgram connection found")?;
            let r = http().get("https://api.deepgram.com/v1/models").header("authorization", format!("Token {k}")).send().await.map_err(|e| e.to_string())?;
            let v: Value = r.json().await.map_err(|e| e.to_string())?;
            for m in v["tts"].as_array().into_iter().flatten() {
                let id = m["canonical_name"].as_str().or_else(|| m["name"].as_str()).unwrap_or("");
                let lang = m["languages"][0].as_str().map(str::to_owned).unwrap_or_else(|| id.rsplit('-').next().unwrap_or("en").to_string());
                let g = m["metadata"]["tags"].as_array().and_then(|t| t.iter().find(|x| *x == "masculine" || *x == "feminine")).cloned().unwrap_or(json!(""));
                out.push(json!({"id": id, "name": m["name"].as_str().unwrap_or(id), "lang": lang, "gender": g}));
            }
        }
        "inworld" => {
            let k = first_key(db, "inworld").ok_or("No Inworld connection found")?;
            let r = http().get("https://api.inworld.ai/tts/v1/voices").header("authorization", format!("Basic {k}")).send().await.map_err(|e| e.to_string())?;
            let v: Value = r.json().await.map_err(|e| e.to_string())?;
            for x in v["voices"].as_array().into_iter().flatten() {
                out.push(json!({"id": x["voiceId"], "name": x["displayName"].as_str().or_else(|| x["voiceId"].as_str()), "lang": x["languages"][0].as_str().unwrap_or("en"), "gender": x["gender"].as_str().unwrap_or("")}));
            }
        }
        "gemini" => {
            for (id, g) in [("Zephyr", "Female"), ("Puck", "Male"), ("Charon", "Male"), ("Kore", "Female"), ("Fenrir", "Male"), ("Leda", "Female"), ("Orus", "Male"), ("Aoede", "Female"), ("Callirrhoe", "Female"), ("Autonoe", "Female"), ("Enceladus", "Male"), ("Iapetus", "Male"), ("Umbriel", "Male"), ("Algieba", "Male"), ("Despina", "Female"), ("Erinome", "Female"), ("Algenib", "Male"), ("Rasalgethi", "Male"), ("Laomedeia", "Female"), ("Achernar", "Female"), ("Alnilam", "Male"), ("Schedar", "Male"), ("Gacrux", "Female"), ("Pulcherrima", "Female"), ("Achird", "Male"), ("Zubenelgenubi", "Male"), ("Vindemiatrix", "Female"), ("Sadachbia", "Male"), ("Sadaltager", "Male"), ("Sulafat", "Female")] {
                out.push(json!({"id": id, "name": id, "lang": "en", "gender": g}));
            }
        }
        "local-device" => {
            if cfg!(target_os = "macos") {
                if let Ok(o) = tokio::process::Command::new("say").args(["-v", "?"]).output().await {
                    let re = regex::Regex::new(r"^([^\s].*?)\s{2,}([a-z]{2})_([A-Z]{2})").unwrap();
                    for l in String::from_utf8_lossy(&o.stdout).lines() {
                        if let Some(c) = re.captures(l) {
                            out.push(json!({"id": c[1].trim(), "name": c[1].trim(), "lang": &c[2], "gender": ""}));
                        }
                    }
                }
            } else if let Ok(o) = tokio::process::Command::new("espeak-ng").arg("--voices").output().await {
                for l in String::from_utf8_lossy(&o.stdout).lines().skip(1) {
                    let p: Vec<&str> = l.split_whitespace().collect();
                    if p.len() >= 4 {
                        out.push(json!({"id": p[1], "name": p[3], "lang": p[1].split('-').next().unwrap_or(""), "gender": if p[2].contains('M') { "Male" } else { "Female" }}));
                    }
                }
            }
        }
        _ => return Err("provider must be one of: elevenlabs, deepgram, inworld, edge-tts, local-device, gemini".into()),
    }
    Ok(out)
}

pub async fn voices(State(st): State<AppState>, Query(q): Query<VoicesQuery>) -> Response {
    let Some(p) = q.provider.filter(|p| !p.is_empty()) else {
        return json_response(400, &json!({"error": {"message": "provider must be one of: elevenlabs, deepgram, inworld, edge-tts, local-device, gemini", "type": "invalid_request_error"}}), &[]);
    };
    match list_voices(&st.db, &p).await {
        Ok(v) => {
            let alias = REG.entry(&p).and_then(|e| e["alias"].as_str().map(str::to_owned)).unwrap_or_else(|| p.clone());
            let data: Vec<Value> = v
                .into_iter()
                .filter(|x| q.lang.as_deref().map(|l| x["lang"] == l).unwrap_or(true))
                .map(|mut x| {
                    x["model"] = json!(format!("{alias}/{}", crate::jsv::js_string(&x["id"])));
                    x
                })
                .collect();
            json_response(200, &json!({"object": "list", "data": data}), &[])
        }
        Err(e) => json_response(if e.starts_with("provider must") { 400 } else { 502 }, &json!({"error": {"message": e, "type": "server_error"}}), &[]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        let known = vec!["tts-1".to_string(), "tts-1-hd".to_string()];
        assert_eq!(parse_model_voice("tts-1-hd/nova", "", "", &known), ("tts-1-hd".into(), "nova".into()));
        assert_eq!(parse_model_voice("a/b/c", "", "", &[]), ("a/b".into(), "c".into()));
        let w = pcm_to_wav(&[0u8; 10]);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(w.len(), 54);
        assert!(!credentialed("edge-tts"));
        assert!(credentialed("elevenlabs"));
        // AWS documented SigV4 example shape (deterministic signature).
        let h = sigv4_headers("POST", "polly.us-east-1.amazonaws.com", "/v1/speech", "us-east-1", "polly", "AKID", "SECRET", b"{}", "20250101T000000Z");
        assert!(h[2].1.starts_with("AWS4-HMAC-SHA256 Credential=AKID/20250101/us-east-1/polly/aws4_request"));
    }
}

/// Whether `synthesize` has an adapter for this provider.
pub(crate) fn supports(p: &str) -> bool {
    matches!(p, "google-tts" | "edge-tts" | "local-device" | "elevenlabs" | "openai" | "openrouter" | "gemini" | "xiaomi-mimo" | "selfhosted-tts" | "aws-polly") || crate::exec::is_openai_compatible(p) || tts_cfg(p)["format"].is_string()
}
