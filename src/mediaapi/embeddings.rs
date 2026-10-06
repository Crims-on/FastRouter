//! POST /v1/embeddings (port of embeddings.js + embeddingsCore.js + adapters).

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use bytes::Bytes;
use serde_json::{Value, json};

use super::*;
use crate::AppState;
use crate::exec::is_openai_compatible;

pub(crate) enum Adapter {
    OpenAi(String),
    Gemini,
    Node,
    SelfHosted,
}

const OPENAI_COMPAT: &[&str] = &["openai", "openrouter", "mistral", "voyage-ai", "fireworks", "together", "nebius", "github", "nvidia", "jina-ai", "vercel-ai-gateway"];

pub(crate) fn adapter(provider: &str) -> Option<Adapter> {
    match provider {
        "gemini" | "google_ai_studio" => Some(Adapter::Gemini),
        "selfhosted-embedding" => Some(Adapter::SelfHosted),
        p if OPENAI_COMPAT.contains(&p) => Some(Adapter::OpenAi(p.into())),
        p if is_openai_compatible(p) || p.starts_with("custom-embedding-") => Some(Adapter::Node),
        // Any other provider that declares an embeddings endpoint.
        p if media_cfg(p, "embeddingConfig")["baseUrl"].is_string() => Some(Adapter::OpenAi(p.into())),
        _ => None,
    }
}

fn openai_url(provider: &str) -> String {
    media_cfg(provider, "embeddingConfig")["baseUrl"].as_str().map(str::to_owned).unwrap_or_else(|| if provider == "jina-ai" { "https://api.jina.ai/v1/embeddings".into() } else { String::new() })
}

fn openai_body(model: &str, input: &Value, encoding: &Value, dims: &Value) -> Value {
    let mut b = json!({"model": model, "input": input});
    if truthy(encoding) {
        b["encoding_format"] = encoding.clone();
    }
    if let Some(d) = dims.as_f64().or_else(|| dims.as_str().and_then(|s| s.parse().ok())).filter(|d| *d > 0.0) {
        b["dimensions"] = json!(d as i64);
    }
    b
}

fn node_url(creds: &Value, required: bool) -> Result<String, String> {
    let raw = creds["providerSpecificData"]["baseUrl"].as_str().map(str::trim).filter(|s| !s.is_empty());
    let raw = match raw {
        Some(r) => r.to_string(),
        None if required => {
            return Err("Self-hosted Embedding needs an endpoint: set this connection's baseUrl to the OpenAI base URL of your server, e.g. http://host:8080/v1 (note the /v1 — \"/embeddings\" is appended to it). Refusing to fall back to api.openai.com, which would send your input and API key to OpenAI.".into());
        }
        None => "https://api.openai.com/v1".into(),
    };
    let b = raw.trim_end_matches('/');
    let b = b.strip_suffix("/embeddings").unwrap_or(b);
    Ok(format!("{b}/embeddings"))
}

fn gemini_model_path(m: &str) -> String {
    if m.starts_with("models/") { m.to_string() } else { format!("models/{m}") }
}

fn gemini_body(model: &str, input: &Value, dims: &Value) -> Value {
    let m = gemini_model_path(model);
    let od = dims.as_f64().filter(|d| *d > 0.0).map(|d| d as i64);
    let one = |t: &Value| {
        let mut r = json!({"model": m, "content": {"parts": [{"text": crate::jsv::js_string(t)}]}});
        if let Some(d) = od {
            r["outputDimensionality"] = json!(d);
        }
        r
    };
    match input.as_array() {
        Some(a) => json!({"requests": a.iter().map(one).collect::<Vec<_>>()}),
        None => one(input),
    }
}

fn gemini_normalize(b: &Value, model: &str) -> Value {
    if b["object"] == "list" && b["data"].is_array() {
        return b.clone();
    }
    let items: Vec<Value> = if let Some(e) = b["embeddings"].as_array() {
        e.iter().enumerate().map(|(i, x)| json!({"object": "embedding", "index": i, "embedding": x["values"].as_array().cloned().unwrap_or_default()})).collect()
    } else if b["embedding"]["values"].is_array() {
        vec![json!({"object": "embedding", "index": 0, "embedding": b["embedding"]["values"]})]
    } else {
        vec![]
    };
    json!({"object": "list", "data": items, "model": model, "usage": {"prompt_tokens": 0, "total_tokens": 0}})
}

async fn core(mut ctx: Ctx, body: Value) -> MediaResult {
    let input = body["input"].clone();
    if !input.is_string() && !input.is_array() {
        return MediaResult::err(400, "input must be a string or array of strings");
    }
    let Some(ad) = adapter(&ctx.provider) else {
        return MediaResult::err(400, format!("Provider '{}' does not support embeddings.", ctx.provider));
    };
    let model = ctx.model.clone();
    let encoding = if truthy(&body["encoding_format"]) { body["encoding_format"].clone() } else { json!("float") };
    let (url_fn, req_body): (Box<dyn Fn(&Value) -> Result<String, String> + Send + Sync>, Value) = match &ad {
        Adapter::OpenAi(p) => {
            let u = openai_url(p);
            (Box::new(move |_| Ok(u.clone())), openai_body(&model, &input, &encoding, &body["dimensions"]))
        }
        Adapter::Node => (Box::new(|c| node_url(c, false)), openai_body(&model, &input, &encoding, &body["dimensions"])),
        Adapter::SelfHosted => (Box::new(|c| node_url(c, true)), openai_body(&model, &input, &encoding, &body["dimensions"])),
        Adapter::Gemini => {
            let op = if input.is_array() { "batchEmbedContents" } else { "embedContent" };
            let mp = gemini_model_path(&model);
            (Box::new(move |c| Ok(format!("https://generativelanguage.googleapis.com/v1beta/{mp}:{op}?key={}", crate::oauth::enc(&key_of(c))))), gemini_body(&model, &input, &body["dimensions"]))
        }
    };
    if let Err(e) = url_fn(&ctx.creds) {
        return MediaResult::err(400, format!("[{}/{}] {e}", ctx.provider, ctx.model));
    }
    let extra = match &ad {
        Adapter::OpenAi(p) => media_cfg(p, "embeddingConfig")["headers"].clone(),
        _ => Value::Null,
    };
    let is_gemini = matches!(ad, Adapter::Gemini);
    let rb = req_body.to_string();
    let r = send_with_refresh(&mut ctx, |c| {
        let mut b = client(c).post(url_fn(c)?).header("content-type", "application/json").timeout(std::time::Duration::from_secs(120)).body(rb.clone());
        if !is_gemini {
            b = b.header("authorization", bearer(c));
        }
        for (k, v) in extra.as_object().into_iter().flatten() {
            b = b.header(k.as_str(), crate::jsv::js_string(v));
        }
        Ok(b)
    })
    .await;
    let r = match r {
        Ok(r) => r,
        Err(e) => return provider_err(502, &e),
    };
    if !r.status().is_success() {
        return upstream_error(r).await;
    }
    let Ok(v) = r.json::<Value>().await else {
        return MediaResult::err(502, format!("Invalid JSON response from {}", ctx.provider));
    };
    let out = if is_gemini { gemini_normalize(&v, &ctx.model) } else { v };
    let mut res = MediaResult::json(&out);
    res.usage = Some(out["usage"].clone());
    res
}

fn exact_usage(u: &Value) -> Option<(i64, i64)> {
    if !u.is_object() || u["estimated"] == json!(true) {
        return None;
    }
    let p = u["prompt_tokens"].as_i64().or_else(|| u["input_tokens"].as_i64())?;
    let c = u["completion_tokens"].as_i64().or_else(|| u["output_tokens"].as_i64()).unwrap_or(0);
    (p > 0 && c == 0 && u["total_tokens"].as_i64() == Some(p)).then_some((p, 0))
}

pub async fn handle(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let key = match crate::api::authorize(&st, &headers, None) {
        Ok(k) => k,
        Err(r) => return r,
    };
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
    let Some((provider, model)) = resolve_model(&st.db, &model_str) else {
        return error_response(400, "Invalid model format", &[]);
    };
    let started = std::time::Instant::now();
    let db = st.db.clone();
    let b = body.clone();
    let p2 = provider.clone();
    let m2 = model.clone();
    let model_str2 = model_str.clone();
    with_accounts(st.db.clone(), &provider, &model, false, None, move |ctx| {
        let b = b.clone();
        let db = db.clone();
        let (p2, m2, ms, key) = (p2.clone(), m2.clone(), model_str2.clone(), key.clone());
        async move {
            let conn = ctx.connection_id.clone();
            let r = core(ctx, b).await;
            if r.ok {
                if let Some((p, _)) = r.usage.as_ref().and_then(exact_usage) {
                    crate::chat::core::UsageSink { db, provider: p2, model: m2, requested_model: ms, connection: conn, api_key: key, endpoint: "/v1/embeddings".into(), stream: false, started }.record(Some(&json!({"prompt_tokens": p, "completion_tokens": 0})), 200, None);
                }
            }
            r
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bodies() {
        let g = gemini_body("text-embedding-004", &json!(["a", "b"]), &json!(256));
        assert_eq!(g["requests"][1]["content"]["parts"][0]["text"], "b");
        assert_eq!(g["requests"][0]["outputDimensionality"], 256);
        let n = gemini_normalize(&json!({"embeddings": [{"values": [0.1]}]}), "m");
        assert_eq!(n["data"][0]["embedding"][0], 0.1);
        assert!(node_url(&json!({}), true).is_err());
        assert_eq!(node_url(&json!({"providerSpecificData": {"baseUrl": "http://h/v1/embeddings/"}}), true).unwrap(), "http://h/v1/embeddings");
        assert!(matches!(adapter("venice"), Some(Adapter::OpenAi(_))));
    }
}
