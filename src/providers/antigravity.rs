//! Antigravity (Google Cloud Code, IDE flavour) executor — port of executors/antigravity.js.

use std::sync::LazyLock;

use async_trait::async_trait;
use regex::Regex;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::exec::{Executor, Headers, RetryDelay};
use crate::jsv::{now_ms, truthy};
use crate::session::{get_thought_signature, resolve_session_id, to_numeric_session_id};
use crate::translate::gemini_fmt::{clean_json_schema, normalize_gemini_contents};

pub struct Antigravity;

const MAX_RETRY_AFTER_MS: u64 = 10_000;
const TRANSIENT_RETRY_MAX_MS: u64 = 15_000;
const MAX_OUTPUT_TOKENS: i64 = 64_000;
const BLACKLIST: [&str; 7] = ["output_config", "thinking", "reasoning_effort", "reasoning", "enable_thinking", "thinking_budget", "thinkingConfig"];

static IDE_REQ_ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^agent/[^/]+/\d+/[^/]+/\d+$").unwrap());
static TRANSIENT: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?i)high\s+traffic",
        r"(?i)agent\s+(execution\s+)?terminated\s+due\s+to\s+error",
        r"(?i)capacity",
        r"(?i)temporarily\s+unavailable",
        r"(?i)timeout",
        r"(?i)stream\s+(ended|closed|terminated|interrupted)",
        r"(?i)empty\s+response",
    ]
    .iter()
    .map(|r| Regex::new(r).unwrap())
    .collect()
});

pub fn sanitize_function_name(name: &str) -> String {
    if name.is_empty() {
        return "_unknown".into();
    }
    static BAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-zA-Z0-9_.:\-]").unwrap());
    let mut s = BAD.replace_all(name, "_").into_owned();
    if !s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        s = format!("_{s}");
    }
    s.chars().take(64).collect()
}

pub fn is_image_model(model: &str) -> bool {
    model.to_lowercase().contains("image") || model.to_lowercase().contains("imagen")
}

fn parse_image_config(model: &str) -> Value {
    static RES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d+)x(\d+)$").unwrap());
    let mut ar = "1:1".to_string();
    if let Some(c) = RES.captures(model) {
        let w: u64 = c[1].parse().unwrap_or(1);
        let h: u64 = c[2].parse().unwrap_or(1);
        if w <= 16 && h <= 16 {
            ar = format!("{w}:{h}");
        } else {
            fn gcd(a: u64, b: u64) -> u64 {
                if b == 0 { a } else { gcd(b, a % b) }
            }
            let d = gcd(w, h).max(1);
            ar = format!("{}:{}", w / d, h / d);
        }
    }
    json!({"aspectRatio": ar})
}

fn uuid_from_seed(seed: &str) -> String {
    let h = Sha256::digest(if seed.is_empty() { "antigravity" } else { seed }.as_bytes());
    let mut b = [0u8; 16];
    b.copy_from_slice(&h[..16]);
    b[6] = (b[6] & 0x0f) | 0x50;
    b[8] = (b[8] & 0x3f) | 0x80;
    let x = hex::encode(b);
    format!("{}-{}-{}-{}-{}", &x[0..8], &x[8..12], &x[12..16], &x[16..20], &x[20..])
}

fn build_ide_request_id(body: &Value, request: &Value, creds: &Value, model: &str, request_type: &str) -> String {
    if let Some(r) = body["requestId"].as_str() {
        if IDE_REQ_ID.is_match(r) {
            return r.to_string();
        }
    }
    let pick = |v: &Value| v.as_str().filter(|s| !s.is_empty()).map(str::to_owned);
    let sid = pick(&request["sessionId"])
        .or_else(|| pick(&body["request"]["sessionId"]))
        .or_else(|| pick(&creds["_clientSessionId"]))
        .or_else(|| pick(&creds["connectionId"]))
        .or_else(|| pick(&creds["email"]))
        .unwrap_or_else(|| "anonymous".into());
    let conv = uuid_from_seed(&format!("antigravity:conversation:{sid}"));
    let traj = uuid_from_seed(&format!("antigravity:trajectory:{sid}:{model}:{request_type}"));
    let count = request["contents"].as_array().map(|a| a.len() as i64).unwrap_or(1);
    let step = (count * 2 - 1).max(1);
    format!("agent/{conv}/{}/{traj}/{step}", now_ms())
}

fn generate_project_id() -> String {
    let r = uuid::Uuid::new_v4();
    let b = r.as_bytes();
    let adj = ["useful", "bright", "swift", "calm", "bold"][b[0] as usize % 5];
    let noun = ["fuze", "wave", "spark", "flow", "core"][b[1] as usize % 5];
    format!("{adj}-{noun}-{}", &r.to_string()[..5])
}

/// Applies ANTIGRAVITY_PROMPT_REWRITES to a system text.
pub fn rewrite_prompt(text: &str) -> String {
    static RULES: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
        vec![
            (Regex::new(&regex::escape("You are a Claude agent, built on Anthropic's Claude Agent SDK.")).unwrap(), ""),
            (
                Regex::new(r"(?i)You are Hermes(?: Agent)?(?:,\s*(?:an intelligent AI assistant|an AI assistant|an AI agent))?(?:,?\s*(?:built|created)\s+by\s+Nous Research)?\.").unwrap(),
                "You are an AI assistant.",
            ),
            (Regex::new(r"(?im)^x-anthropic-billing-header:[^\n]*(?:\r?\n)*").unwrap(), ""),
        ]
    });
    let mut out = text.to_string();
    for (re, to) in RULES.iter() {
        out = re.replace_all(&out, *to).into_owned();
    }
    static OC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)opencode").unwrap());
    OC.replace_all(&out, |c: &regex::Captures| match &c[0] {
        "OpenCode" => "Antigravity".to_string(),
        "OPENCODE" => "ANTIGRAVITY".to_string(),
        _ => "antigravity".to_string(),
    })
    .into_owned()
}

fn strip_blacklisted(v: &mut Value) {
    if let Some(o) = v.as_object_mut() {
        for k in BLACKLIST {
            o.shift_remove(k);
        }
    }
}

fn parse_retry_headers(h: &reqwest::header::HeaderMap) -> Option<u64> {
    let get = |k: &str| h.get(k).and_then(|v| v.to_str().ok()).map(str::trim).map(str::to_owned);
    if let Some(ra) = get("retry-after") {
        if let Some(n) = crate::jsv::parse_int_prefix(&ra).filter(|n| *n > 0) {
            return Some(n as u64 * 1000);
        }
        if let Ok(d) = httpdate_parse(&ra) {
            let diff = d - now_ms();
            return (diff > 0).then_some(diff as u64);
        }
    }
    if let Some(n) = get("x-ratelimit-reset-after").and_then(|s| crate::jsv::parse_int_prefix(&s)).filter(|n| *n > 0) {
        return Some(n as u64 * 1000);
    }
    if let Some(ts) = get("x-ratelimit-reset").and_then(|s| crate::jsv::parse_int_prefix(&s)) {
        let diff = ts * 1000 - now_ms();
        return (diff > 0).then_some(diff as u64);
    }
    None
}

/// Minimal RFC 7231 date parser ("Sun, 06 Nov 1994 08:49:37 GMT") → epoch ms.
fn httpdate_parse(s: &str) -> Result<i64, ()> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 5 {
        return Err(());
    }
    let day: i64 = parts[1].parse().map_err(|_| ())?;
    let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let mon = months.iter().position(|m| *m == parts[2]).ok_or(())? as i64 + 1;
    let year: i64 = parts[3].parse().map_err(|_| ())?;
    let hms: Vec<i64> = parts[4].split(':').filter_map(|x| x.parse().ok()).collect();
    if hms.len() != 3 {
        return Err(());
    }
    Ok(crate::jsv::days_from_civil(year, mon, day) * 86_400_000 + (hms[0] * 3600 + hms[1] * 60 + hms[2]) * 1000)
}

pub fn parse_retry_from_message(msg: &str) -> Option<u64> {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)reset after (\d+h)?(\d+m)?(\d+s)?").unwrap());
    let c = RE.captures(msg)?;
    let n = |i: usize| c.get(i).and_then(|m| m.as_str()[..m.as_str().len() - 1].parse::<u64>().ok()).unwrap_or(0);
    let total = n(1) * 3_600_000 + n(2) * 60_000 + n(3) * 1000;
    (total > 0).then_some(total)
}

fn transient(status: u16, msg: &str) -> bool {
    status == 429 || [500, 502, 503, 504].contains(&status) || TRANSIENT.iter().any(|r| r.is_match(msg))
}

impl Antigravity {
    fn transform_image(&self, model: &str, body: &Value, creds: &Value, project: Value) -> Value {
        static SUFFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"-(\d+)x(\d+)$").unwrap());
        let clean = SUFFIX.replace(model, "").into_owned();
        let src = if body["request"]["contents"].is_array() { &body["request"]["contents"] } else { &body["contents"] };
        let mut contents = vec![];
        for c in src.as_array().into_iter().flatten() {
            let tp: Vec<Value> = c["parts"].as_array().into_iter().flatten().filter(|p| p.get("text").is_some()).map(|p| json!({"text": p["text"]})).collect();
            if !tp.is_empty() {
                contents.push(json!({"role": if truthy(&c["role"]) { c["role"].clone() } else { json!("user") }, "parts": tp}));
            }
        }
        let conn = creds["email"].as_str().filter(|s| !s.is_empty()).or_else(|| creds["connectionId"].as_str());
        let session_id = resolve_session_id(&creds["rawHeaders"], body, conn, "antigravity");
        let request = json!({
            "contents": contents,
            "generationConfig": {"temperature": 1.0, "topP": 0.95, "topK": 40, "maxOutputTokens": 8192, "imageConfig": parse_image_config(model)},
            "sessionId": session_id,
        });
        json!({
            "project": project,
            "model": clean,
            "userAgent": "antigravity",
            "requestType": "image_gen",
            "requestId": build_ide_request_id(body, &request, creds, &clean, "image_gen"),
            "request": request,
        })
    }
}

#[async_trait]
impl Executor for Antigravity {
    fn provider(&self) -> &str {
        "antigravity"
    }

    fn build_url(&self, model: &str, stream: bool, idx: usize, _c: &Value) -> Result<String, String> {
        let urls = self.base_urls();
        let base = urls.get(idx).or(urls.first()).cloned().unwrap_or_default();
        let action = if stream && !is_image_model(model) { "streamGenerateContent?alt=sse" } else { "generateContent" };
        Ok(format!("{base}/v1internal:{action}"))
    }

    fn build_headers(&self, creds: &Value, _stream: bool, _u: &str, _m: &str, _b: &Value) -> Headers {
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        h.set("Authorization", format!("Bearer {}", creds["accessToken"].as_str().unwrap_or("")));
        let ua = self.config()["headers"]["User-Agent"].as_str().map(str::to_owned).unwrap_or_else(|| crate::consts::C["ANTIGRAVITY_HEADERS"]["User-Agent"].as_str().unwrap_or("").to_string());
        h.set("User-Agent", ua);
        h
    }

    fn transform_request(&self, model: &str, mut body: Value, stream: bool, creds: &Value) -> Value {
        let project = if truthy(&creds["projectId"]) { creds["projectId"].clone() } else { json!(generate_project_id()) };
        if !stream {
            crate::jsv::del(&mut body, "stream_options");
        }
        if is_image_model(model) {
            return self.transform_image(model, &body, creds, project);
        }
        let raw_sid = body["request"]["sessionId"].as_str().filter(|s| !s.is_empty()).map(str::to_owned).unwrap_or_else(|| {
            let conn = creds["email"].as_str().filter(|s| !s.is_empty()).or_else(|| creds["connectionId"].as_str());
            resolve_session_id(&creds["rawHeaders"], &body, conn, "antigravity")
        });
        let session_id = to_numeric_session_id(Some(&raw_sid)).unwrap_or(raw_sid);
        let sig_model = body["model"].as_str().filter(|s| !s.is_empty()).unwrap_or(model).to_string();
        let default_sig = crate::consts::sig_ag();

        let raw_contents: Vec<Value> = body["request"]["contents"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|c| {
                let mut c = c.clone();
                let parts_in = c["parts"].as_array().cloned();
                if parts_in.as_ref().map(|p| p.iter().any(|x| truthy(&x["functionResponse"]))).unwrap_or(false) {
                    c["role"] = json!("user");
                }
                let parts: Vec<Value> = parts_in
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|p| {
                        if truthy(&p["thought"]) && !truthy(&p["functionCall"]) {
                            return false;
                        }
                        !(truthy(&p["thoughtSignature"]) && !truthy(&p["functionCall"]) && !truthy(&p["text"]))
                    })
                    .collect();
                let mut first_seen = false;
                let parts: Vec<Value> = parts
                    .into_iter()
                    .map(|mut p| {
                        if !truthy(&p["functionCall"]) {
                            return p;
                        }
                        let call_id = p["functionCall"]["id"].as_str().filter(|s| !s.is_empty()).map(str::to_owned);
                        let cached = call_id.and_then(|id| get_thought_signature(&id, Some(&session_id), Some(&sig_model)));
                        let sig = p["thoughtSignature"].as_str().filter(|s| !s.is_empty()).map(str::to_owned).or(cached.clone()).or_else(|| (!first_seen).then(|| default_sig.to_string()));
                        first_seen = true;
                        if let Some(s) = sig {
                            p["thoughtSignature"] = json!(s);
                        } else if truthy(&p["thoughtSignature"]) && cached.is_none() {
                            crate::jsv::del(&mut p, "thoughtSignature");
                        }
                        p
                    })
                    .collect();
                c["parts"] = Value::Array(parts);
                c
            })
            .collect();
        let contents = normalize_gemini_contents(&raw_contents);

        let mut tools: Option<Value> = body["request"]["tools"].as_array().filter(|a| !a.is_empty()).map(|groups| {
            let mut seen = std::collections::HashSet::new();
            let mut decls = vec![];
            for g in groups {
                for f in g["functionDeclarations"].as_array().into_iter().flatten() {
                    let name = sanitize_function_name(f["name"].as_str().unwrap_or(""));
                    if !seen.insert(name.clone()) {
                        continue;
                    }
                    let mut d = f.clone();
                    d["name"] = json!(name);
                    d["parameters"] = if truthy(&f["parameters"]) {
                        clean_json_schema(f["parameters"].clone())
                    } else {
                        json!({"type": "object", "properties": {"reason": {"type": "string", "description": "Brief explanation"}}, "required": ["reason"]})
                    };
                    decls.push(d);
                }
            }
            if decls.is_empty() { json!([]) } else { json!([{"functionDeclarations": decls}]) }
        });
        if tools.is_none() && body["request"]["tools"].is_array() {
            tools = Some(body["request"]["tools"].clone());
        }

        let mut req: Map<String, Value> = body["request"].as_object().cloned().unwrap_or_default();
        req.shift_remove("tools");
        req.shift_remove("toolConfig");
        let mut req = Value::Object(req);
        strip_blacklisted(&mut req);
        if let Some(parts) = req["systemInstruction"]["parts"].as_array_mut() {
            for p in parts.iter_mut() {
                if let Some(t) = p["text"].as_str() {
                    p["text"] = json!(rewrite_prompt(t));
                }
            }
        }
        let mut gcfg = if req["generationConfig"].is_object() { req["generationConfig"].clone() } else { json!({}) };
        if gcfg["maxOutputTokens"].as_f64().map(|n| n > MAX_OUTPUT_TOKENS as f64).unwrap_or(false) {
            gcfg["maxOutputTokens"] = json!(MAX_OUTPUT_TOKENS);
        }
        req["generationConfig"] = gcfg;
        req["contents"] = Value::Array(contents);
        let has_tools = tools.as_ref().and_then(|t| t.as_array()).map(|a| !a.is_empty()).unwrap_or(false);
        if let Some(t) = tools {
            req["tools"] = t;
        }
        req["sessionId"] = json!(session_id);
        crate::jsv::del(&mut req, "safetySettings");
        if has_tools {
            req["toolConfig"] = json!({"functionCallingConfig": {"mode": "VALIDATED"}});
        }

        strip_blacklisted(&mut body);
        crate::jsv::del(&mut body, "requestType");
        let model_out = if truthy(&body["model"]) { body["model"].clone() } else { json!(model) };
        let request_id = build_ide_request_id(&body, &req, creds, model, "agent");
        let mut out = if body.is_object() { body } else { json!({}) };
        out["project"] = project;
        out["model"] = model_out;
        out["userAgent"] = json!("antigravity");
        out["requestId"] = json!(request_id);
        out["request"] = req;
        out
    }

    fn has_retry_hook(&self) -> bool {
        true
    }

    fn compute_retry_delay(&self, status: u16, headers: &reqwest::header::HeaderMap, body: &str, attempt: u32, _d: u64) -> RetryDelay {
        let mut retry = parse_retry_headers(headers);
        let ej: Value = serde_json::from_str(body).unwrap_or(Value::Null);
        let msg = [&ej["error"]["message"], &ej["message"], &ej["error"], &json!(body)]
            .iter()
            .filter(|v| truthy(v))
            .map(|v| v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string()))
            .collect::<Vec<_>>()
            .join("\n");
        if retry.is_none() {
            retry = parse_retry_from_message(&msg);
        }
        if let Some(ms) = retry {
            return if ms <= MAX_RETRY_AFTER_MS { RetryDelay::Ms(ms) } else { RetryDelay::Veto };
        }
        if !transient(status, &msg) {
            return RetryDelay::Veto;
        }
        let cap = if status == 429 { MAX_RETRY_AFTER_MS } else { TRANSIENT_RETRY_MAX_MS };
        RetryDelay::Ms((1000u64 << attempt.min(20)).min(cap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        assert_eq!(sanitize_function_name("1bad name!"), "_1bad_name_");
        assert_eq!(parse_image_config("x-image-1024x768")["aspectRatio"], "4:3");
        assert_eq!(parse_image_config("x-image-16x9")["aspectRatio"], "16:9");
        assert_eq!(parse_retry_from_message("quota will reset after 1h2m3s."), Some(3_723_000));
        assert_eq!(rewrite_prompt("Use OpenCode and opencode"), "Use Antigravity and antigravity");
        assert_eq!(rewrite_prompt("x-anthropic-billing-header: abc\nHello"), "Hello");
        let u = uuid_from_seed("a");
        assert_eq!(u.len(), 36);
        assert_eq!(&u[14..15], "5");
    }

    #[test]
    fn transform_basic() {
        let body = json!({"model": "gemini-3-pro", "request": {
            "contents": [{"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}, "id": "c1"}}]}, {"role": "model", "parts": [{"functionResponse": {"name": "f", "response": {}}}]}],
            "tools": [{"functionDeclarations": [{"name": "f", "parameters": {"type": "object", "additionalProperties": false}}]}],
            "generationConfig": {"maxOutputTokens": 100000},
            "thinking": {}
        }, "reasoning_effort": "high"});
        let creds = json!({"accessToken": "t", "projectId": "p"});
        let out = Antigravity.transform_request("gemini-3-pro", body, true, &creds);
        assert_eq!(out["project"], "p");
        assert!(out.get("reasoning_effort").is_none());
        assert_eq!(out["request"]["generationConfig"]["maxOutputTokens"], 64000);
        assert_eq!(out["request"]["toolConfig"]["functionCallingConfig"]["mode"], "VALIDATED");
        assert!(out["request"].get("thinking").is_none());
        // first content is model → normalize prepends a user turn
        assert_eq!(out["request"]["contents"][0]["role"], "user");
        assert!(out["request"]["contents"][1]["parts"][0]["thoughtSignature"].is_string());
        assert!(out["requestId"].as_str().unwrap().starts_with("agent/"));
    }
}
