//! POST /v1/web/fetch (port of fetch.js + handlers/fetch/index.js).

use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use bytes::Bytes;
use serde_json::{Value, json};

use super::*;
use crate::AppState;
use crate::chat::accounts;
use crate::registry::REG;

fn truncate(t: &str, max: Option<i64>) -> String {
    match max.filter(|m| *m > 0) {
        Some(m) => t.chars().take(m as usize).collect(),
        None => t.to_string(),
    }
}

fn jina_title(t: &str) -> Option<String> {
    regex::Regex::new(r"(?mi)^\s*Title:\s*(.+)$").unwrap().captures(t).map(|c| c[1].trim().to_string()).or_else(|| regex::Regex::new(r"(?m)^\s*#\s+(.+)$").unwrap().captures(t).map(|c| c[1].trim().to_string()))
}

#[allow(clippy::too_many_arguments)]
fn data(provider: &str, url: &str, title: Option<String>, fmt: &str, text: &str, links: Option<Value>, cost: &Value, resp_ms: i64, up_ms: i64) -> Value {
    let mut d = json!({
        "provider": provider, "url": url, "title": title,
        "content": {"format": fmt, "text": text, "length": text.encode_utf16().count()},
        "metadata": {"author": null, "published_at": null, "language": null},
        "usage": {"fetch_cost_usd": cost},
        "metrics": {"response_time_ms": resp_ms, "upstream_latency_ms": up_ms},
    });
    if let Some(l) = links.filter(|l| l.is_array()) {
        d["links"] = l;
    }
    d
}

pub async fn fetch_core(provider: &str, url: &str, fmt: &str, max: Option<i64>, cfg: &Value, creds: &Value) -> Result<Value, (u16, String)> {
    let started = Instant::now();
    let to = Duration::from_millis(cfg["timeoutMs"].as_u64().unwrap_or(15_000));
    let key = key_of(creds);
    let cost = cfg["costPerQuery"].clone();
    let c = client(creds);
    let send = |rb: reqwest::RequestBuilder| async move { rb.timeout(to).send().await.map_err(|e| (if e.is_timeout() { 504 } else { 502 }, e.to_string())) };
    let auth = |rb: reqwest::RequestBuilder, h: &str| if key.is_empty() { rb } else if h == "x-api-key" { rb.header("x-api-key", key.clone()) } else { rb.bearer_auth(key.clone()) };
    let read = |r: reqwest::Response| async move {
        let ct = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        let t = r.text().await.unwrap_or_default();
        let j = if ct.contains("application/json") { serde_json::from_str::<Value>(&t).ok() } else { None };
        (j, t)
    };
    let errmsg = |j: &Option<Value>, fallback: String| j.as_ref().map(|j| if j["error"].is_string() { j["error"].as_str().unwrap().to_string() } else if j["error"]["message"].is_string() { j["error"]["message"].as_str().unwrap().to_string() } else if j["message"].is_string() { j["message"].as_str().unwrap().to_string() } else { fallback.clone() }).unwrap_or(fallback);
    let t0 = Instant::now();
    match provider {
        "firecrawl" => {
            let r = send(auth(c.post("https://api.firecrawl.dev/v1/scrape"), "bearer").json(&json!({"url": url, "formats": [fmt]}))).await?;
            let up = t0.elapsed().as_millis() as i64;
            let st = r.status().as_u16();
            let (j, _) = read(r).await;
            if !(200..300).contains(&st) {
                return Err((st, errmsg(&j, format!("Firecrawl error: {st}"))));
            }
            let d = j.map(|j| j["data"].clone()).unwrap_or(json!({}));
            let text = d["markdown"].as_str().or_else(|| d["html"].as_str()).or_else(|| d["text"].as_str()).unwrap_or("");
            Ok(data("firecrawl", url, d["metadata"]["title"].as_str().map(str::to_owned), fmt, &truncate(text, max), None, &cost, started.elapsed().as_millis() as i64, up))
        }
        "jina-reader" => {
            let r = send(auth(c.post("https://r.jina.ai/"), "bearer").json(&json!({"url": url}))).await?;
            let up = t0.elapsed().as_millis() as i64;
            let st = r.status().as_u16();
            let body = r.text().await.unwrap_or_default();
            if !(200..300).contains(&st) {
                return Err((st, if body.is_empty() { format!("Jina error: {st}") } else { body.chars().take(500).collect() }));
            }
            Ok(data("jina-reader", url, jina_title(&body), fmt, &truncate(&body, max), None, &cost, started.elapsed().as_millis() as i64, up))
        }
        "tavily" => {
            let r = send(auth(c.post("https://api.tavily.com/extract"), "bearer").json(&json!({"urls": [url], "extract_depth": "basic"}))).await?;
            let up = t0.elapsed().as_millis() as i64;
            let st = r.status().as_u16();
            let (j, _) = read(r).await;
            if !(200..300).contains(&st) {
                return Err((st, errmsg(&j, format!("Tavily error: {st}"))));
            }
            let text = j.as_ref().and_then(|j| j["results"][0]["raw_content"].as_str()).unwrap_or("").to_string();
            Ok(data("tavily", url, None, fmt, &truncate(&text, max), None, &cost, started.elapsed().as_millis() as i64, up))
        }
        "exa" => {
            let r = send(auth(c.post("https://api.exa.ai/contents"), "x-api-key").json(&json!({"ids": [url], "text": true}))).await?;
            let up = t0.elapsed().as_millis() as i64;
            let st = r.status().as_u16();
            let (j, _) = read(r).await;
            if !(200..300).contains(&st) {
                return Err((st, errmsg(&j, format!("Exa error: {st}"))));
            }
            let f = j.map(|j| j["results"][0].clone()).unwrap_or(json!({}));
            Ok(data("exa", url, f["title"].as_str().map(str::to_owned), fmt, &truncate(f["text"].as_str().unwrap_or(""), max), None, &cost, started.elapsed().as_millis() as i64, up))
        }
        "ollama" => {
            let base = cfg["baseUrl"].as_str().unwrap_or("https://ollama.com/api/web_fetch");
            let r = send(auth(c.post(base), "bearer").json(&json!({"url": url}))).await?;
            let up = t0.elapsed().as_millis() as i64;
            let st = r.status().as_u16();
            let (j, t) = read(r).await;
            if !(200..300).contains(&st) {
                return Err((st, errmsg(&j, if t.is_empty() { format!("Ollama error: {st}") } else { t.chars().take(500).collect() })));
            }
            let j = j.filter(|j| j["content"].is_string()).ok_or((502, "Ollama returned an empty or invalid web fetch response".to_string()))?;
            Ok(data("ollama", url, j["title"].as_str().map(str::to_owned), fmt, &truncate(j["content"].as_str().unwrap(), max), Some(j["links"].clone()), &cost, started.elapsed().as_millis() as i64, up))
        }
        "tinyfish" => {
            if !["markdown", "html"].contains(&fmt) {
                return Err((400, format!("Unsupported TinyFish format: {fmt}")));
            }
            let r = send(c.post(cfg["baseUrl"].as_str().unwrap_or("")).header("x-api-key", key.clone()).json(&json!({"urls": [url], "format": fmt}))).await?;
            let up = t0.elapsed().as_millis() as i64;
            let st = r.status().as_u16();
            let (j, _) = read(r).await;
            if !(200..300).contains(&st) {
                return Err((st, errmsg(&j, format!("TinyFish error: {st}"))));
            }
            let j = j.unwrap_or(json!({}));
            if j["errors"][0].is_object() {
                let f = &j["errors"][0];
                return Err((f["status"].as_u64().unwrap_or(502) as u16, format!("TinyFish fetch failed: {}", f["error"].as_str().unwrap_or("unknown error"))));
            }
            let p = &j["results"][0];
            let Some(text) = p["text"].as_str() else { return Err((502, "TinyFish returned no extractable content".into())) };
            let mut d = data("tinyfish", url, p["title"].as_str().map(str::to_owned), fmt, &truncate(text, max), Some(p["links"].clone()), &cost, started.elapsed().as_millis() as i64, up);
            d["metadata"] = json!({"author": p["author"], "published_at": p["published_date"], "language": p["language"]});
            Ok(d)
        }
        _ => Err((400, format!("Unsupported provider: {provider}"))),
    }
}

async fn single(db: std::sync::Arc<crate::db::Db>, body: Value, input: String) -> Response {
    let pid = accounts::provider_id(&input);
    if REG.entry(&pid).is_none() && REG.media(&pid).is_null() {
        return error_response(400, &format!("Unknown provider: {input}"), &[]);
    }
    let cfg = media_cfg(&pid, "fetchConfig");
    if !cfg.is_object() {
        return error_response(400, &format!("Provider {pid} does not support web fetch"), &[]);
    }
    let url = body["url"].as_str().unwrap_or("").to_string();
    let fmt = body["format"].as_str().unwrap_or("markdown").to_string();
    let max = body["max_characters"].as_i64();
    let no_auth = REG.entry(&pid).map(|e| e["noAuth"] == json!(true)).unwrap_or(false);
    let lock = format!("webfetch:{pid}");
    with_accounts(db, &pid, &lock, no_auth, None, |ctx| {
        let (pid, url, fmt, cfg) = (pid.clone(), url.clone(), fmt.clone(), cfg.clone());
        async move {
            match fetch_core(&pid, &url, &fmt, max, &cfg, &ctx.creds).await {
                Ok(d) => MediaResult::json(&d),
                Err((s, e)) => MediaResult::err(s, e),
            }
        }
    })
    .await
}

pub async fn handle(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = crate::api::authorize(&st, &headers, None) {
        return r;
    }
    let body = match crate::api::parse_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let Some(input) = body["provider"].as_str().or_else(|| body["model"].as_str()).filter(|s| !s.is_empty()).map(str::to_owned) else {
        return error_response(400, "Missing required field: provider (or model)", &[]);
    };
    let Some(url) = body["url"].as_str().filter(|s| !s.is_empty()) else {
        return error_response(400, "Missing required field: url", &[]);
    };
    let Ok(u) = reqwest::Url::parse(url) else {
        return error_response(400, "Invalid URL format", &[]);
    };
    if !matches!(u.scheme(), "http" | "https") || !crate::media::host_is_public(u.host_str().unwrap_or("")).await {
        return error_response(400, "URL must resolve to a public address", &[]);
    }
    let input = input.strip_suffix("/fetch").unwrap_or(&input).to_string();
    if let Some(models) = crate::chat::combo_models(&st.db, &input) {
        let db = st.db.clone();
        return super::images::combo(&st.db, &input, models, |m| single(db.clone(), body.clone(), m.strip_suffix("/fetch").unwrap_or(&m).to_string())).await;
    }
    single(st.db.clone(), body, input).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        assert_eq!(jina_title("Title: Hello\nURL: x").as_deref(), Some("Hello"));
        assert_eq!(truncate("abcdef", Some(3)), "abc");
    }
}
