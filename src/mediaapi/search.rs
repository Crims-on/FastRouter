//! POST /v1/search (port of search.js, handlers/search/{index,callers,normalizers,chatSearch}.js).

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

const GLOBAL_TIMEOUT: Duration = Duration::from_secs(15);

fn entry(id: &str) -> Value {
    REG.entry(id).cloned().unwrap_or(Value::Null)
}

fn now_iso() -> String {
    crate::jsv::iso_from_ms(crate::jsv::now_ms())
}

pub struct Params {
    pub query: String,
    pub search_type: String,
    pub max_results: i64,
    pub token: String,
    pub country: String,
    pub language: String,
    pub time_range: String,
    pub offset: Option<i64>,
    pub domain_filter: Vec<String>,
    pub content_options: Value,
    pub provider_options: Value,
    pub psd: Value,
}

fn setting(p: &Params, k: &str) -> Option<String> {
    p.provider_options[k].as_str().or_else(|| p.psd[k].as_str()).map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned)
}

async fn base_url(cfg: &Value, p: &Params) -> Result<String, String> {
    if let Some(o) = setting(p, "baseUrl") {
        let u = reqwest::Url::parse(&o).map_err(|_| format!("Invalid baseUrl: {o}"))?;
        if !matches!(u.scheme(), "http" | "https") {
            return Err(format!("Invalid baseUrl protocol: {}:", u.scheme()));
        }
        if !crate::media::host_is_public(u.host_str().unwrap_or("")).await {
            return Err("baseUrl must be a public address".into());
        }
        return Ok(o.trim_end_matches('/').to_string());
    }
    Ok(cfg["baseUrl"].as_str().unwrap_or("").trim_end_matches('/').to_string())
}

fn split_domains(d: &[String]) -> (Vec<String>, Vec<String>) {
    let inc = d.iter().filter(|x| !x.starts_with('-')).cloned().collect();
    let exc = d.iter().filter_map(|x| x.strip_prefix('-').map(str::to_owned)).collect();
    (inc, exc)
}

fn page_no(offset: Option<i64>, max: i64) -> Option<i64> {
    offset.filter(|o| *o > 0 && max > 0).map(|o| o / max + 1)
}

fn qs(pairs: &[(&str, String)]) -> String {
    pairs.iter().map(|(k, v)| format!("{}={}", crate::oauth::enc(k), crate::oauth::enc(v))).collect::<Vec<_>>().join("&")
}

pub struct Req {
    pub url: String,
    pub method: &'static str,
    pub headers: Vec<(String, String)>,
    pub body: Option<Value>,
}

fn get(url: String, headers: Vec<(&str, String)>) -> Req {
    Req { url, method: "GET", headers: headers.into_iter().map(|(a, b)| (a.to_string(), b)).collect(), body: None }
}

fn post(url: String, headers: Vec<(&str, String)>, body: Value) -> Req {
    let mut h: Vec<(String, String)> = vec![("content-type".into(), "application/json".into())];
    h.extend(headers.into_iter().map(|(a, b)| (a.to_string(), b)));
    Req { url, method: "POST", headers: h, body: Some(body) }
}

/// buildSearchRequest
pub async fn build_request(id: &str, cfg: &Value, p: &Params) -> Result<Req, String> {
    let base = base_url(cfg, p).await?;
    let (inc, exc) = split_domains(&p.domain_filter);
    let bearer = || vec![("authorization", format!("Bearer {}", p.token))];
    Ok(match id {
        "serper" => {
            let mut b = json!({"q": p.query, "num": p.max_results});
            if !p.country.is_empty() {
                b["gl"] = json!(p.country.to_lowercase());
            }
            if !p.language.is_empty() {
                b["hl"] = json!(p.language);
            }
            post(format!("{base}{}", if p.search_type == "news" { "/news" } else { "/search" }), vec![("x-api-key", p.token.clone())], b)
        }
        "brave-search" => {
            let mut q = vec![("q", p.query.clone()), ("count", p.max_results.to_string())];
            if !p.country.is_empty() {
                q.push(("country", p.country.clone()));
            }
            if !p.language.is_empty() {
                q.push(("search_lang", p.language.clone()));
            }
            get(format!("{base}{}?{}", if p.search_type == "news" { "/news/search" } else { "/web/search" }, qs(&q)), vec![("accept", "application/json".into()), ("x-subscription-token", p.token.clone())])
        }
        "perplexity" => {
            let mut b = json!({"query": p.query, "max_results": p.max_results});
            if !p.country.is_empty() {
                b["country"] = json!(p.country);
            }
            if !p.language.is_empty() {
                b["search_language_filter"] = json!([p.language]);
            }
            if !p.domain_filter.is_empty() {
                b["search_domain_filter"] = json!(p.domain_filter);
            }
            post(base, bearer(), b)
        }
        "exa" => {
            let mut b = json!({"query": p.query, "numResults": p.max_results, "type": "auto", "text": true, "highlights": true});
            if !inc.is_empty() {
                b["includeDomains"] = json!(inc);
            }
            if !exc.is_empty() {
                b["excludeDomains"] = json!(exc);
            }
            if p.search_type == "news" {
                b["category"] = json!("news");
            }
            post(base, vec![("x-api-key", p.token.clone())], b)
        }
        "tavily" => {
            let mut b = json!({"query": p.query, "max_results": p.max_results, "topic": if p.search_type == "news" { "news" } else { "general" }});
            if !inc.is_empty() {
                b["include_domains"] = json!(inc);
            }
            if !exc.is_empty() {
                b["exclude_domains"] = json!(exc);
            }
            if !p.country.is_empty() {
                b["country"] = json!(p.country);
            }
            post(base, bearer(), b)
        }
        "google-pse" => {
            let cx = setting(p, "cx");
            if p.token.is_empty() || cx.is_none() {
                return Err("Google Programmable Search requires both apiKey and cx".into());
            }
            let mut q = vec![("key", p.token.clone()), ("cx", cx.unwrap()), ("q", p.query.clone()), ("num", p.max_results.min(10).to_string())];
            if !p.country.is_empty() {
                q.push(("gl", p.country.to_lowercase()));
            }
            if !p.language.is_empty() {
                q.push(("hl", p.language.clone()));
            }
            if let Some(d) = match p.time_range.as_str() {
                "day" => Some("d1"),
                "week" => Some("w1"),
                "month" => Some("m1"),
                "year" => Some("y1"),
                _ => None,
            } {
                q.push(("dateRestrict", d.into()));
            }
            if let Some(o) = p.offset.filter(|o| *o > 0) {
                q.push(("start", (o + 1).min(91).to_string()));
            }
            get(format!("{base}?{}", qs(&q)), vec![("accept", "application/json".into())])
        }
        "linkup" => {
            if p.token.is_empty() {
                return Err("Linkup Search requires an API key".into());
            }
            let depth = setting(p, "depth").filter(|d| ["fast", "standard", "deep"].contains(&d.as_str())).unwrap_or_else(|| "standard".into());
            let mut b = json!({"q": p.query, "depth": depth, "outputType": "searchResults", "maxResults": p.max_results});
            if !inc.is_empty() {
                b["includeDomains"] = json!(inc);
            }
            if !exc.is_empty() {
                b["excludeDomains"] = json!(exc);
            }
            let days = match p.time_range.as_str() {
                "day" => Some(1),
                "week" => Some(7),
                "month" => Some(30),
                "year" => Some(365),
                _ => None,
            };
            if let Some(d) = days {
                let now = crate::jsv::now_ms();
                b["fromDate"] = json!(crate::jsv::iso_from_ms(now - d * 86_400_000)[..10]);
                b["toDate"] = json!(crate::jsv::iso_from_ms(now)[..10]);
            }
            post(base, bearer(), b)
        }
        "searchapi" => {
            if p.token.is_empty() {
                return Err("SearchAPI requires an API key".into());
            }
            let mut q = vec![("engine", if p.search_type == "news" { "google_news" } else { "google" }.to_string()), ("q", p.query.clone()), ("api_key", p.token.clone())];
            if !p.country.is_empty() {
                q.push(("gl", p.country.to_lowercase()));
            }
            if !p.language.is_empty() {
                q.push(("hl", p.language.clone()));
            }
            if let Some(pg) = page_no(p.offset, p.max_results) {
                q.push(("page", pg.to_string()));
            }
            get(format!("{base}?{}", qs(&q)), vec![("accept", "application/json".into())])
        }
        "youcom" => {
            if p.token.is_empty() {
                return Err("You.com Search requires an API key".into());
            }
            let mut q = vec![("query", p.query.clone()), ("count", p.max_results.min(100).to_string())];
            if !p.time_range.is_empty() && p.time_range != "any" {
                q.push(("freshness", p.time_range.clone()));
            }
            if let Some(o) = p.offset.filter(|o| *o > 0 && p.max_results > 0) {
                q.push(("offset", (o / p.max_results).min(9).to_string()));
            }
            if !p.country.is_empty() {
                q.push(("country", p.country.clone()));
            }
            if !p.language.is_empty() {
                q.push(("language", p.language.clone()));
            }
            if !inc.is_empty() {
                q.push(("include_domains", inc.join(",")));
            }
            if !exc.is_empty() {
                q.push(("exclude_domains", exc.join(",")));
            }
            if p.content_options["full_page"] == json!(true) {
                q.push(("livecrawl", if p.search_type == "news" { "news" } else { "web" }.into()));
                q.push(("livecrawl_formats", if p.content_options["format"] == "markdown" { "markdown" } else { "html" }.into()));
            }
            get(format!("{base}?{}", qs(&q)), vec![("accept", "application/json".into()), ("x-api-key", p.token.clone())])
        }
        "searxng" => {
            let u = if base.ends_with("/search") { base.clone() } else { format!("{base}/search") };
            let mut q = vec![("q", p.query.clone()), ("format", "json".into()), ("categories", if p.search_type == "news" { "news" } else { "general" }.into())];
            if !p.language.is_empty() {
                q.push(("language", p.language.clone()));
            }
            if !p.time_range.is_empty() && p.time_range != "any" {
                q.push(("time_range", p.time_range.clone()));
            }
            if let Some(pg) = page_no(p.offset, p.max_results) {
                q.push(("pageno", pg.to_string()));
            }
            get(format!("{u}?{}", qs(&q)), vec![("accept", "application/json".into())])
        }
        "xquik" => {
            if p.token.is_empty() {
                return Err("Xquik requires an API key".into());
            }
            let qt = setting(p, "queryType");
            if let Some(t) = &qt {
                if t != "Latest" && t != "Top" {
                    return Err("Xquik queryType must be Latest or Top".into());
                }
            }
            let mut q = vec![("q", p.query.clone()), ("limit", p.max_results.to_string())];
            if let Some(c) = setting(p, "cursor") {
                q.push(("cursor", c));
            }
            if let Some(t) = qt {
                q.push(("queryType", t));
            }
            if !p.language.is_empty() {
                q.push(("language", p.language.clone()));
            }
            get(format!("{base}?{}", qs(&q)), vec![("accept", "application/json".into()), ("x-api-key", p.token.clone())])
        }
        "tinyfish" => {
            if !p.search_type.is_empty() && !["web", "news", "research_paper"].contains(&p.search_type.as_str()) {
                return Err("Unsupported TinyFish search type".into());
            }
            let mut q = vec![("query", p.query.clone())];
            if !p.search_type.is_empty() && p.search_type != "web" {
                q.push(("domain_type", p.search_type.clone()));
            }
            if !p.country.is_empty() {
                q.push(("location", p.country.clone()));
            }
            if !p.language.is_empty() {
                q.push(("language", p.language.clone()));
            }
            if !inc.is_empty() {
                q.push(("include_domains", inc.join(",")));
            }
            if !exc.is_empty() {
                q.push(("exclude_domains", exc.join(",")));
            }
            if let Some(o) = p.offset.filter(|o| *o > 0) {
                if o >= 110 {
                    return Err("TinyFish search offset exceeds available pages".into());
                }
                if o % 10 + p.max_results > 10 {
                    return Err("TinyFish search offset and max_results must fit within one page".into());
                }
                q.push(("page", (o / 10).to_string()));
            }
            get(format!("{}?{}", cfg["baseUrl"].as_str().unwrap_or(""), qs(&q)), vec![("accept", "application/json".into()), ("x-api-key", p.token.clone())])
        }
        "ollama-search" => {
            let mut b = json!({"query": p.query, "max_results": p.max_results});
            if !p.country.is_empty() {
                b["country"] = json!(p.country);
            }
            if !p.language.is_empty() {
                b["language"] = json!(p.language);
            }
            post(base, if p.token.is_empty() { vec![] } else { bearer() }, b)
        }
        "glm" => post(
            base,
            if p.token.is_empty() { vec![] } else { bearer() },
            json!({"jsonrpc": "2.0", "id": format!("9r-{}", crate::jsv::now_ms()), "method": "tools/call", "params": {"name": "web_search_prime", "arguments": {"search_query": p.query, "count": p.max_results}}}),
        ),
        _ => post(base, if p.token.is_empty() { vec![] } else { bearer() }, json!({"query": p.query, "max_results": p.max_results, "search_type": p.search_type})),
    })
}

struct Item {
    title: Value,
    url: Value,
    snippet: Value,
    score: Value,
    published_at: Value,
    favicon_url: Value,
    author: Value,
    source_type: Value,
    image_url: Value,
    full_text: Value,
    text_format: Value,
}

impl Default for Item {
    fn default() -> Self {
        Item { title: Value::Null, url: Value::Null, snippet: Value::Null, score: Value::Null, published_at: Value::Null, favicon_url: Value::Null, author: Value::Null, source_type: Value::Null, image_url: Value::Null, full_text: Value::Null, text_format: Value::Null }
    }
}

fn or_s(v: &Value) -> Value {
    if truthy(v) { v.clone() } else { json!("") }
}
fn or_n(v: &Value) -> Value {
    if truthy(v) { v.clone() } else { Value::Null }
}

fn make(pid: &str, it: Item, idx: usize, now: &str) -> Value {
    let url = it.url.as_str().unwrap_or("").to_string();
    let display = if url.is_empty() {
        Value::Null
    } else {
        let re = regex::Regex::new(r"^https?://(www\.)?").unwrap();
        json!(re.replace(&url, "").split('?').next().unwrap_or("").to_string())
    };
    let ft = it.full_text.as_str().filter(|s| !s.is_empty());
    json!({
        "title": or_s(&it.title), "url": url, "display_url": display, "snippet": or_s(&it.snippet), "position": idx + 1,
        "score": it.score.as_f64().map(|s| json!(s.clamp(0.0, 1.0))).unwrap_or(Value::Null),
        "published_at": or_n(&it.published_at), "favicon_url": or_n(&it.favicon_url),
        "content": ft.map(|t| json!({"format": if truthy(&it.text_format) { it.text_format.clone() } else { json!("text") }, "text": t, "length": t.encode_utf16().count()})).unwrap_or(Value::Null),
        "metadata": {"author": or_n(&it.author), "language": null, "source_type": or_n(&it.source_type), "image_url": or_n(&it.image_url)},
        "citation": {"provider": pid, "retrieved_at": now, "rank": idx + 1},
        "provider_raw": null,
    })
}

fn arr(v: &Value) -> Vec<Value> {
    v.as_array().cloned().unwrap_or_default()
}

fn first(vs: &[&Value]) -> Value {
    vs.iter().find(|v| truthy(v)).map(|v| (*v).clone()).unwrap_or(Value::Null)
}

/// normalizeSearchResponse → (results, totalResults, pagination)
pub fn normalize(pid: &str, d: &Value, search_type: &str) -> (Vec<Value>, Value, Option<Value>) {
    let now = now_iso();
    let map = |items: Vec<Value>, f: &dyn Fn(&Value) -> Item| items.iter().enumerate().map(|(i, x)| make(pid, f(x), i, &now)).collect::<Vec<_>>();
    match pid {
        "serper" => {
            let items = arr(if search_type == "news" { &d["news"] } else { &d["organic"] });
            let r = map(items, &|x| Item { title: x["title"].clone(), url: x["link"].clone(), snippet: first(&[&x["snippet"], &x["description"]]), published_at: x["date"].clone(), ..Default::default() });
            (r, if d["searchParameters"]["totalResults"].is_number() { d["searchParameters"]["totalResults"].clone() } else { Value::Null }, None)
        }
        "brave-search" => {
            let c = if search_type == "news" { if truthy(&d["news"]) { &d["news"] } else { d } } else { &d["web"] };
            let r = map(arr(&c["results"]), &|x| Item { title: x["title"].clone(), url: x["url"].clone(), snippet: x["description"].clone(), published_at: first(&[&x["page_age"], &x["age"]]), favicon_url: first(&[&x["meta_url"]["favicon"], &x["favicon"]]), ..Default::default() });
            (r, c["totalCount"].clone(), None)
        }
        "perplexity" => {
            let r = map(arr(&d["results"]), &|x| Item { title: x["title"].clone(), url: x["url"].clone(), snippet: x["snippet"].clone(), published_at: first(&[&x["date"], &x["last_updated"]]), ..Default::default() });
            let n = r.len();
            (r, json!(n), None)
        }
        "exa" => {
            let r = map(arr(&d["results"]), &|x| Item {
                title: x["title"].clone(),
                url: x["url"].clone(),
                snippet: if truthy(&x["highlights"][0]) { x["highlights"][0].clone() } else { json!(x["text"].as_str().map(|t| t.chars().take(300).collect::<String>()).unwrap_or_default()) },
                score: x["score"].clone(),
                published_at: x["publishedDate"].clone(),
                favicon_url: x["favicon"].clone(),
                author: x["author"].clone(),
                image_url: x["image"].clone(),
                full_text: x["text"].clone(),
                text_format: json!("text"),
                ..Default::default()
            });
            let n = r.len();
            (r, json!(n), None)
        }
        "tavily" => {
            let r = map(arr(&d["results"]), &|x| Item { title: x["title"].clone(), url: x["url"].clone(), snippet: x["content"].clone(), score: x["score"].clone(), published_at: x["published_date"].clone(), full_text: x["raw_content"].clone(), text_format: json!("text"), ..Default::default() });
            let n = r.len();
            (r, json!(n), None)
        }
        "tinyfish" => {
            let r = map(arr(&d["results"]), &|x| Item { title: x["title"].clone(), url: x["url"].clone(), snippet: x["snippet"].clone(), published_at: x["date"].clone(), source_type: x["publisher"].clone(), author: x["authors"].as_array().map(|a| json!(a.iter().filter_map(|s| s.as_str()).collect::<Vec<_>>().join(", "))).unwrap_or(Value::Null), ..Default::default() });
            (r, d["total_results"].clone(), None)
        }
        "google-pse" => {
            let r = map(arr(&d["items"]), &|x| Item { title: x["title"].clone(), url: x["link"].clone(), snippet: x["snippet"].clone(), image_url: first(&[&x["pagemap"]["cse_image"][0]["src"], &x["pagemap"]["cse_thumbnail"][0]["src"], &x["pagemap"]["metatags"][0]["og:image"]]), ..Default::default() });
            let raw = first(&[&d["searchInformation"]["totalResults"], &d["queries"]["request"][0]["totalResults"]]);
            let t = raw.as_f64().or_else(|| raw.as_str().and_then(|s| s.parse().ok()));
            (r, t.map(|t| json!(t as i64)).unwrap_or(Value::Null), None)
        }
        "linkup" => {
            let r = map(arr(&d["results"]), &|x| Item { title: first(&[&x["name"], &x["title"]]), url: x["url"].clone(), snippet: first(&[&x["content"], &x["snippet"]]), source_type: if truthy(&x["type"]) { x["type"].clone() } else { json!("web") }, image_url: first(&[&x["image_url"], &x["imageUrl"]]), full_text: x["content"].clone(), text_format: json!("text"), ..Default::default() });
            let n = r.len();
            (r, json!(n), None)
        }
        "searchapi" => {
            let items = if d["organic_results"].is_array() { arr(&d["organic_results"]) } else { arr(&d["top_stories"]) };
            let r = map(items, &|x| Item { title: x["title"].clone(), url: x["link"].clone(), snippet: first(&[&x["snippet"], &x["description"]]), published_at: first(&[&x["date"], &x["published_at"]]), favicon_url: x["favicon"].clone(), author: x["source"].clone(), image_url: x["thumbnail"].clone(), ..Default::default() });
            let raw = &d["search_information"]["total_results"];
            let t = raw.as_f64().or_else(|| raw.as_str().and_then(|s| s.parse().ok()));
            let n = r.len();
            (r, t.map(|t| json!(t as i64)).unwrap_or(json!(n)), None)
        }
        "youcom" => {
            let sec = if search_type == "news" { &d["results"]["news"] } else { &d["results"]["web"] };
            let st = search_type.to_string();
            let r = map(arr(sec), &|x| {
                let fs = x["snippets"].as_array().and_then(|a| a.iter().find(|v| v.is_string())).cloned();
                let (lt, lf) = if x["markdown"].is_string() { (x["markdown"].clone(), "markdown") } else if x["html"].is_string() { (x["html"].clone(), "html") } else { (Value::Null, "html") };
                Item { title: x["title"].clone(), url: x["url"].clone(), snippet: fs.unwrap_or_else(|| if x["description"].is_string() { x["description"].clone() } else { json!("") }), published_at: x["page_age"].clone(), favicon_url: x["favicon_url"].clone(), image_url: x["thumbnail_url"].clone(), source_type: json!(st), text_format: if lt.is_null() { Value::Null } else { json!(lf) }, full_text: lt, ..Default::default() }
            });
            let n = r.len();
            (r, json!(n), None)
        }
        "searxng" => {
            let r = map(arr(&d["results"]), &|x| Item {
                title: x["title"].clone(),
                url: x["url"].clone(),
                snippet: first(&[&x["content"], &x["snippet"]]),
                published_at: first(&[&x["publishedDate"], &x["published_date"]]),
                source_type: x["engines"].as_array().map(|a| json!(a.iter().filter_map(|s| s.as_str()).collect::<Vec<_>>().join(", "))).unwrap_or_else(|| first(&[&x["engine"], &x["category"]])),
                image_url: first(&[&x["thumbnail"], &x["img_src"]]),
                ..Default::default()
            });
            let n = r.len();
            (r, json!(n), None)
        }
        "xquik" => {
            let r = map(arr(&d["tweets"]), &|x| {
                let user = x["author"]["username"].as_str().unwrap_or("");
                let name = x["author"]["name"].as_str().unwrap_or("");
                let id = crate::jsv::js_string(&x["id"]);
                let id = if id == "null" { String::new() } else { id };
                let url = if !user.is_empty() && !id.is_empty() { format!("https://x.com/{}/status/{}", crate::oauth::enc(user), crate::oauth::enc(&id)) } else if !id.is_empty() { format!("https://x.com/i/web/status/{}", crate::oauth::enc(&id)) } else { String::new() };
                let author = if !user.is_empty() { json!(format!("@{user}")) } else if !name.is_empty() { json!(name) } else { Value::Null };
                let title = author.as_str().map(|a| format!("{a} on X")).unwrap_or_else(|| "X post".into());
                Item { title: json!(title), url: json!(url), snippet: or_s(&x["text"]), published_at: x["createdAt"].clone(), author, image_url: x["media"].as_array().and_then(|m| m.iter().find_map(|i| i["mediaUrl"].as_str().map(|s| json!(s)))).unwrap_or(Value::Null), source_type: json!("x_post"), full_text: x["text"].clone(), text_format: json!("text"), ..Default::default() }
            });
            (r, Value::Null, Some(json!({"has_more": d["has_next_page"] == json!(true), "next_cursor": d["next_cursor"].as_str().filter(|s| !s.is_empty())})))
        }
        "ollama-search" => {
            let items = if d["results"].is_array() { arr(&d["results"]) } else { arr(d) };
            let r = map(items, &|x| Item { title: x["title"].clone(), url: x["url"].clone(), snippet: first(&[&x["content"], &x["snippet"]]), full_text: x["content"].clone(), text_format: json!("text"), published_at: x["published_at"].clone(), source_type: x["source"].clone(), ..Default::default() });
            let n = r.len();
            (r, json!(n), None)
        }
        "glm" => {
            let payload = match d["result"]["content"][0]["text"].as_str() {
                Some(t) => serde_json::from_str(t).unwrap_or(json!({})),
                None => d.clone(),
            };
            let items = if payload["results"].is_array() { arr(&payload["results"]) } else if payload["news"].is_array() { arr(&payload["news"]) } else { arr(&payload) };
            let r = map(items, &|x| Item { title: x["title"].clone(), url: first(&[&x["link"], &x["url"]]), snippet: x["content"].clone(), published_at: first(&[&x["publish_date"], &x["published_at"]]), favicon_url: x["icon"].clone(), source_type: x["media"].clone(), ..Default::default() });
            let n = r.len();
            (r, json!(n), None)
        }
        _ => (vec![], Value::Null, None),
    }
}

// ---------------------------------------------------------------------------
// Chat-based search (searchViaChat)
// ---------------------------------------------------------------------------

fn norm_cit(c: &Value) -> Option<Value> {
    if let Some(s) = c.as_str() {
        return Some(json!({"url": s}));
    }
    if truthy(&c["url"]) {
        return Some(c.clone());
    }
    None
}

fn expand_segment(text: &str, seg: &Value) -> String {
    let (Some(s), Some(e)) = (seg["startIndex"].as_u64(), seg["endIndex"].as_u64()) else { return String::new() };
    let chars: Vec<char> = text.chars().collect();
    let start = (s as usize).saturating_sub(150);
    let end = ((e as usize) + 250).min(chars.len());
    if start >= end {
        return String::new();
    }
    let mut out: String = chars[start..end].iter().collect::<String>().trim().to_string();
    if start > 0 {
        out = format!("...{}", regex::Regex::new(r"^\S+").unwrap().replace(&out, ""));
    }
    if end < chars.len() {
        out = format!("{}...", regex::Regex::new(r"\S+$").unwrap().replace(&out, ""));
    }
    out.trim().to_string()
}

fn extract_chat_answer(provider: &str, d: &Value) -> (String, Vec<Value>, i64) {
    let mut cits = vec![];
    let tokens = d["usage"]["total_tokens"].as_i64().or_else(|| d["usageMetadata"]["totalTokenCount"].as_i64()).or_else(|| d["response"]["usageMetadata"]["totalTokenCount"].as_i64()).unwrap_or(0);
    let text: String;
    match provider {
        "gemini" => {
            let c = &d["candidates"][0];
            text = c["content"]["parts"].as_array().into_iter().flatten().filter_map(|p| p["text"].as_str()).collect();
            for ch in c["groundingMetadata"]["groundingChunks"].as_array().into_iter().flatten() {
                let w = &ch["web"];
                if let Some(u) = w["uri"].as_str().or_else(|| w["url"].as_str()) {
                    cits.push(json!({"url": u, "title": w["title"].as_str().unwrap_or("")}));
                }
            }
        }
        "antigravity" => {
            let r = if d["response"].is_object() { &d["response"] } else { d };
            let c = &r["candidates"][0];
            text = c["content"]["parts"].as_array().into_iter().flatten().filter_map(|p| p["text"].as_str()).collect();
            let g = &c["groundingMetadata"];
            let mut order: Vec<String> = vec![];
            let mut src: std::collections::HashMap<String, (String, Vec<String>, Vec<String>)> = Default::default();
            let by_idx: Vec<Option<String>> = g["groundingChunks"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|ch| {
                    let u = ch["web"]["uri"].as_str().or_else(|| ch["web"]["url"].as_str()).unwrap_or("");
                    if u.is_empty() {
                        return None;
                    }
                    if !src.contains_key(u) {
                        order.push(u.to_string());
                        src.insert(u.to_string(), (ch["web"]["title"].as_str().unwrap_or("").to_string(), vec![], vec![]));
                    }
                    Some(u.to_string())
                })
                .collect();
            for s in g["groundingSupports"].as_array().into_iter().flatten() {
                let grounded = s["segment"]["text"].as_str().unwrap_or("").to_string();
                let ex = expand_segment(&text, &s["segment"]);
                let ex = if ex.is_empty() { grounded.clone() } else { ex };
                for i in s["groundingChunkIndices"].as_array().into_iter().flatten().filter_map(|i| i.as_u64()) {
                    if let Some(Some(u)) = by_idx.get(i as usize) {
                        let e = src.get_mut(u).unwrap();
                        if !grounded.is_empty() && !e.1.contains(&grounded) {
                            e.1.push(grounded.clone());
                        }
                        if !ex.is_empty() && !e.2.contains(&ex) {
                            e.2.push(ex.clone());
                        }
                    }
                }
            }
            for u in order {
                let (t, sn, cx) = &src[&u];
                let snip = if sn.is_empty() { t.clone() } else { sn.join(" | ") };
                let content = if cx.is_empty() { snip.clone() } else { cx.join("\n\n") };
                cits.push(json!({"url": u, "title": t, "snippet": snip, "content": content}));
            }
        }
        "xai" | "perplexity-agent" => {
            let mut t = String::new();
            for item in d["output"].as_array().into_iter().flatten() {
                for p in item["content"].as_array().into_iter().flatten() {
                    if let Some(s) = p["text"].as_str() {
                        t.push_str(s);
                    }
                    for a in p["annotations"].as_array().into_iter().flatten() {
                        if let Some(c) = norm_cit(if truthy(&a["url"]) { a } else { &a["url_citation"] }) {
                            cits.push(c);
                        }
                    }
                }
                for r in item["results"].as_array().into_iter().flatten() {
                    if let Some(u) = r["url"].as_str().or_else(|| r["link"].as_str()) {
                        cits.push(json!({"url": u, "title": r["title"].as_str().unwrap_or(""), "snippet": r["snippet"].as_str().unwrap_or("")}));
                    }
                }
            }
            if cits.is_empty() {
                cits.extend(d["citations"].as_array().into_iter().flatten().filter_map(norm_cit));
            }
            text = t;
        }
        "kimi" | "minimax" => {
            let msg = &d["choices"][0]["message"];
            text = msg["content"].as_str().unwrap_or("").to_string();
            for it in d["web_search_results"].as_array().into_iter().flatten() {
                if let Some(u) = it["url"].as_str().or_else(|| it["link"].as_str()) {
                    cits.push(json!({"url": u, "title": it["title"].as_str().unwrap_or(""), "snippet": it["snippet"].as_str().or_else(|| it["summary"].as_str()).unwrap_or("")}));
                }
            }
            if cits.is_empty() {
                for call in msg["tool_calls"].as_array().into_iter().flatten() {
                    let a = &call["function"]["arguments"];
                    let parsed: Value = match a {
                        Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
                        o => o.clone(),
                    };
                    let items = [&parsed["search_results"], &parsed["results"], &parsed["references"]].into_iter().find(|v| v.is_array()).cloned().unwrap_or(json!([]));
                    for it in items.as_array().into_iter().flatten() {
                        if let Some(u) = it["url"].as_str().or_else(|| it["link"].as_str()) {
                            cits.push(json!({"url": u, "title": it["title"].as_str().unwrap_or(""), "snippet": it["snippet"].as_str().or_else(|| it["summary"].as_str()).unwrap_or("")}));
                        }
                    }
                }
            }
        }
        _ => {
            // openai-style (openai, perplexity, vercel-ai-gateway)
            let msg = &d["choices"][0]["message"];
            text = msg["content"].as_str().unwrap_or("").to_string();
            let ann: Vec<Value> = msg["annotations"].as_array().into_iter().flatten().filter_map(|a| a["url_citation"].as_object().map(|u| json!({"url": u.get("url"), "title": u.get("title").and_then(|t| t.as_str()).unwrap_or("")}))).collect();
            if !ann.is_empty() {
                cits = ann;
            } else {
                cits.extend(d["citations"].as_array().into_iter().flatten().filter_map(norm_cit));
            }
        }
    }
    (text, cits, tokens)
}

async fn chat_search(provider: &str, query: &str, max_results: Option<i64>, creds: &Value) -> Result<Value, (u16, String)> {
    let start = Instant::now();
    let cfg = media_cfg(provider, "searchViaChat");
    let token = key_of(creds);
    if token.is_empty() {
        return Err((401, "Missing credentials (apiKey or accessToken)".into()));
    }
    let model = cfg["defaultModel"].as_str().unwrap_or("").to_string();
    let endpoint = cfg["endpoint"].as_str().map(|e| e.replace("{model}", &model)).filter(|s| !s.is_empty()).unwrap_or_else(|| match provider {
        "vercel-ai-gateway" => "https://ai-gateway.vercel.sh/v1/chat/completions".into(),
        _ => "https://api.openai.com/v1/chat/completions".into(),
    });
    let mut headers: Vec<(String, String)> = vec![("content-type".into(), "application/json".into())];
    let body = match provider {
        "gemini" => {
            headers.push(("x-goog-api-key".into(), token.clone()));
            json!({"contents": [{"role": "user", "parts": [{"text": query}]}], "tools": [{"google_search": {}}]})
        }
        "antigravity" => {
            let Some(pid) = creds["projectId"].as_str().filter(|s| !s.is_empty()) else {
                return Err((401, "Antigravity account has no projectId — reconnect the account".into()));
            };
            headers.push(("authorization".into(), format!("Bearer {token}")));
            headers.push(("user-agent".into(), crate::consts::s("ANTIGRAVITY_IDE_USER_AGENT").into()));
            json!({"project": pid, "model": model, "userAgent": "antigravity", "requestType": "search", "request": {"contents": [{"role": "user", "parts": [{"text": query}]}], "tools": [{"googleSearch": {}}], "generationConfig": {"temperature": 1.0, "maxOutputTokens": 8192}}})
        }
        "xai" => {
            headers.push(("authorization".into(), format!("Bearer {token}")));
            json!({"model": model, "input": [{"role": "user", "content": query}], "tools": [{"type": "web_search"}]})
        }
        "perplexity-agent" => {
            headers.push(("authorization".into(), format!("Bearer {token}")));
            json!({"model": model, "input": query, "tools": [{"type": "web_search"}]})
        }
        "kimi" => {
            headers.push(("authorization".into(), format!("Bearer {token}")));
            json!({"model": model, "messages": [{"role": "user", "content": query}], "tools": [{"type": "builtin_function", "function": {"name": "$web_search"}}]})
        }
        "minimax" => {
            headers.push(("authorization".into(), format!("Bearer {token}")));
            json!({"model": model, "messages": [{"role": "user", "content": query}], "tools": [{"type": "web_search"}]})
        }
        "perplexity" => {
            headers.push(("authorization".into(), format!("Bearer {token}")));
            json!({"model": model, "messages": [{"role": "user", "content": query}]})
        }
        "openai" | "vercel-ai-gateway" => {
            headers.push(("authorization".into(), format!("Bearer {token}")));
            let mut b = json!({"model": model, "messages": [{"role": "user", "content": query}]});
            if !model.to_lowercase().contains("search") {
                b["tools"] = json!([{"type": "web_search"}]);
            }
            b
        }
        _ => return Err((400, format!("Unsupported chat-search provider: {provider}"))),
    };
    let mut rb = client(creds).post(&endpoint).timeout(Duration::from_secs(15)).body(body.to_string());
    for (k, v) in &headers {
        rb = rb.header(k.as_str(), v.as_str());
    }
    let up_start = Instant::now();
    let r = rb.send().await.map_err(|e| if e.is_timeout() { (504, "Upstream timeout".to_string()) } else { (502, format!("Network error: {e}")) })?;
    let latency = up_start.elapsed().as_millis() as i64;
    let st = r.status().as_u16();
    let d: Value = r.json().await.map_err(|_| (502, format!("Invalid upstream response (status {st})")))?;
    if !(200..300).contains(&st) {
        let m = d["error"]["message"].as_str().map(str::to_owned).or_else(|| d["error"].as_str().map(str::to_owned)).or_else(|| d["message"].as_str().map(str::to_owned)).unwrap_or(format!("Upstream HTTP {st}"));
        return Err((st, m));
    }
    let (text, cits, tokens) = extract_chat_answer(provider, &d);
    let limit = max_results.filter(|m| *m > 0).unwrap_or(10) as usize;
    let now = now_iso();
    let results: Vec<Value> = cits
        .into_iter()
        .take(limit)
        .enumerate()
        .map(|(i, c)| json!({"title": c["title"].as_str().unwrap_or(""), "url": c["url"], "snippet": c["snippet"].as_str().unwrap_or(""), "position": i + 1, "score": null, "published_at": null, "favicon_url": null, "content": if truthy(&c["content"]) { c["content"].clone() } else { Value::Null }, "metadata": {}, "citation": {"provider": provider, "retrieved_at": now, "rank": i + 1}, "provider_raw": null}))
        .collect();
    Ok(json!({
        "provider": provider, "query": query, "results": results,
        "answer": {"source": provider, "text": text, "model": model},
        "usage": {"queries_used": 1, "search_cost_usd": 0, "llm_tokens": tokens},
        "metrics": {"response_time_ms": start.elapsed().as_millis() as i64, "upstream_latency_ms": latency, "total_results_available": null},
        "errors": [],
    }))
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

async fn dedicated(pid: &str, cfg: &Value, body: &Value, query: &str, creds: &Value, started: Instant) -> Result<Value, (u16, String)> {
    let t0 = Instant::now();
    let token = key_of(creds);
    if cfg["authType"] != "none" && token.is_empty() {
        return Err((401, format!("No credentials for provider: {pid}")));
    }
    let max = body["max_results"].as_i64().filter(|n| *n > 0).or_else(|| cfg["defaultMaxResults"].as_i64()).unwrap_or(5).min(cfg["maxMaxResults"].as_i64().unwrap_or(100));
    let p = Params {
        query: query.into(),
        search_type: body["search_type"].as_str().map(str::to_owned).unwrap_or_else(|| cfg["searchTypes"][0].as_str().unwrap_or("web").to_string()),
        max_results: max,
        token,
        country: body["country"].as_str().unwrap_or("").into(),
        language: body["language"].as_str().unwrap_or("").into(),
        time_range: body["time_range"].as_str().unwrap_or("").into(),
        offset: body["offset"].as_i64(),
        domain_filter: body["domain_filter"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default(),
        content_options: body["content_options"].clone(),
        provider_options: body["provider_options"].clone(),
        psd: creds["providerSpecificData"].clone(),
    };
    let req = build_request(pid, cfg, &p).await.map_err(|e| (400, e))?;
    let remaining = GLOBAL_TIMEOUT.saturating_sub(started.elapsed()).max(Duration::from_secs(1));
    let to = Duration::from_millis(cfg["timeoutMs"].as_u64().unwrap_or(10_000)).min(remaining);
    let c = client(creds);
    let mut rb = if req.method == "GET" { c.get(&req.url) } else { c.post(&req.url) };
    for (k, v) in &req.headers {
        let clean: String = v.chars().filter(|ch| (*ch as u32) <= 0xff).collect();
        rb = rb.header(k.as_str(), clean.trim());
    }
    if let Some(b) = &req.body {
        rb = rb.body(b.to_string());
    }
    let r = rb.timeout(to).send().await.map_err(|e| if e.is_timeout() { (504, format!("{pid} timeout: {e}")) } else { (502, format!("{pid} error: {e}")) })?;
    if !r.status().is_success() {
        let s = r.status().as_u16();
        let t: String = r.text().await.unwrap_or_default().chars().take(200).collect();
        return Err((s, format!("{pid} returned {s}: {t}")));
    }
    let d: Value = r.json().await.map_err(|e| (502, format!("{pid} error: {e}")))?;
    let (results, total, pagination) = normalize(pid, &d, &p.search_type);
    let off = if pid == "tinyfish" { p.offset.filter(|o| *o > 0).map(|o| (o % 10) as usize).unwrap_or(0) } else { 0 };
    let results: Vec<Value> = results.into_iter().skip(off).take(p.max_results as usize).collect();
    let mut usage = json!({"queries_used": 1, "search_cost_usd": cfg["costPerQuery"]});
    if let Some(cr) = cfg["creditsPerResult"].as_f64() {
        usage["provider_credits_used"] = json!(results.len() as f64 * cr);
    }
    let ms = t0.elapsed().as_millis() as i64;
    let mut out = json!({"provider": pid, "query": p.query, "results": results, "answer": null, "usage": usage, "metrics": {"response_time_ms": ms, "upstream_latency_ms": ms, "total_results_available": total}, "errors": []});
    if let Some(pg) = pagination {
        out["pagination"] = pg;
    }
    Ok(out)
}

pub async fn search_core(pid: &str, body: &Value, creds: &Value) -> MediaResult {
    let started = Instant::now();
    let raw = body["query"].as_str().unwrap_or("");
    if raw.chars().any(|c| (c as u32) < 0x20 && !matches!(c, '\t' | '\n' | '\r') || c as u32 == 0x7f) {
        return search_err(400, "Query contains invalid control characters");
    }
    let clean = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if clean.is_empty() {
        return search_err(400, "Query is empty after normalization");
    }
    let cfg = media_cfg(pid, "searchConfig");
    let via_chat = media_cfg(pid, "searchViaChat").is_object();
    let r = if cfg.is_object() {
        dedicated(pid, &cfg, body, &clean, creds, started).await
    } else if via_chat {
        chat_search(pid, &clean, body["max_results"].as_i64(), creds).await
    } else {
        return search_err(400, &format!("Provider {pid} does not support web search"));
    };
    match r {
        Ok(d) => MediaResult::json(&d),
        Err((s, e)) => {
            if ![400, 401, 403, 404].contains(&s) && started.elapsed() < GLOBAL_TIMEOUT && via_chat && cfg.is_object() {
                if let Ok(d) = chat_search(pid, &clean, body["max_results"].as_i64(), creds).await {
                    return MediaResult::json(&d);
                }
            }
            search_err(s, &e)
        }
    }
}

fn search_err(status: u16, msg: &str) -> MediaResult {
    MediaResult { ok: false, status, error: Some(msg.into()), response: json_response(status, &json!({"error": {"message": msg, "code": status}}), &[]), usage: None }
}

async fn single(db: std::sync::Arc<crate::db::Db>, body: Value, input: String) -> Response {
    let pid = accounts::provider_id(&input);
    let e = entry(&pid);
    if e.is_null() && REG.media(&pid).is_null() {
        return error_response(400, &format!("Unknown provider: {input}"), &[]);
    }
    if !media_cfg(&pid, "searchConfig").is_object() && !media_cfg(&pid, "searchViaChat").is_object() {
        return error_response(400, &format!("Provider {pid} does not support web search"), &[]);
    }
    let mut core_body = json!({"query": body["query"].as_str().unwrap_or("").trim(), "provider": pid});
    for k in ["max_results", "search_type", "country", "language", "time_range", "offset", "domain_filter", "content_options", "provider_options"] {
        if !body[k].is_null() {
            core_body[k] = body[k].clone();
        }
    }
    if e["noAuth"] == json!(true) || media_cfg(&pid, "searchConfig")["authType"] == "none" {
        let creds = db.connections_for(&pid, true).first().map(accounts::credentials_from_connection).unwrap_or_else(|| json!({"providerSpecificData": {}}));
        return search_core(&pid, &core_body, &creds).await.response;
    }
    // Credential lookup may borrow another provider's keys (credentialFallback).
    let lock = format!("websearch:{pid}");
    let has_own = !db.connections_for(&pid, true).is_empty();
    let cred_provider = if has_own { pid.clone() } else { e["credentialFallback"].as_str().map(str::to_owned).filter(|f| !db.connections_for(f, true).is_empty()).unwrap_or(pid.clone()) };
    let pid2 = pid.clone();
    with_accounts(db, &cred_provider, &lock, false, None, |ctx| {
        let (b, p) = (core_body.clone(), pid2.clone());
        async move { search_core(&p, &b, &ctx.creds).await }
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
    if body["query"].as_str().map(|q| q.trim().is_empty()).unwrap_or(true) {
        return error_response(400, "Missing required field: query", &[]);
    }
    // `alias/search` model ids from /v1/models/web are accepted too.
    let input = input.strip_suffix("/search").unwrap_or(&input).to_string();
    if let Some(models) = crate::chat::combo_models(&st.db, &input) {
        let db = st.db.clone();
        return super::images::combo(&st.db, &input, models, |m| single(db.clone(), body.clone(), m.strip_suffix("/search").unwrap_or(&m).to_string())).await;
    }
    single(st.db.clone(), body, input).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn builders_and_normalizers() {
        let cfg = media_cfg("serper", "searchConfig");
        let p = Params { query: "rust".into(), search_type: "web".into(), max_results: 3, token: "k".into(), country: "US".into(), language: "".into(), time_range: "".into(), offset: None, domain_filter: vec![], content_options: Value::Null, provider_options: Value::Null, psd: Value::Null };
        let r = build_request("serper", &cfg, &p).await.unwrap();
        assert!(r.url.ends_with("/search"));
        assert_eq!(r.body.unwrap()["gl"], "us");
        let (res, _, _) = normalize("serper", &json!({"organic": [{"title": "T", "link": "https://www.a.com/x?y=1", "snippet": "s"}]}), "web");
        assert_eq!(res[0]["display_url"], "a.com/x");
        let (txt, cits, _) = extract_chat_answer("gemini", &json!({"candidates": [{"content": {"parts": [{"text": "ans"}]}, "groundingMetadata": {"groundingChunks": [{"web": {"uri": "u", "title": "t"}}]}}]}));
        assert_eq!(txt, "ans");
        assert_eq!(cits[0]["url"], "u");
    }
}
