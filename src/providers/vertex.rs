//! Google Vertex AI executor (vertex = Gemini, vertex-partner = OpenAI-compatible
//! partner models). Port of executors/vertex.js.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};

use crate::exec::{ExecArgs, ExecResult, Executor, Headers, post_json};
use crate::jsv::truthy;
use crate::oauth::refresh::{parse_vertex_sa_json, refresh_google_token, refresh_vertex_token};

static PROJECT_CACHE: LazyLock<Mutex<HashMap<String, String>>> = LazyLock::new(Default::default);

/// gcloud ADC "authorized_user" JSON pasted as the API key.
pub fn parse_vertex_adc_json(api_key: &Value) -> Option<Value> {
    let p: Value = serde_json::from_str(api_key.as_str()?).ok()?;
    (p["type"] == "authorized_user" && truthy(&p["client_id"]) && truthy(&p["client_secret"]) && truthy(&p["refresh_token"])).then_some(p)
}

async fn resolve_project_id(creds: &Value, api_key: &str) -> Option<String> {
    if let Some(p) = PROJECT_CACHE.lock().unwrap().get(api_key) {
        return Some(p.clone());
    }
    let url = format!("https://aiplatform.googleapis.com/v1/publishers/google/models/__probe__:generateContent?key={api_key}");
    let r = crate::exec::client_for(creds).post(&url).header("Content-Type", "application/json").body("{}").send().await.ok()?;
    let j: Value = r.json().await.unwrap_or(Value::Null);
    let msg = j[0]["error"]["message"].as_str().or_else(|| j["error"]["message"].as_str()).unwrap_or("");
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"projects/([^/]+)/").unwrap());
    let pid = RE.captures(msg).map(|c| c[1].to_string())?;
    PROJECT_CACHE.lock().unwrap().insert(api_key.to_string(), pid.clone());
    Some(pid)
}

pub struct Vertex {
    pub id: String,
}

#[async_trait]
impl Executor for Vertex {
    fn provider(&self) -> &str {
        &self.id
    }

    fn build_url(&self, model: &str, stream: bool, _i: usize, creds: &Value) -> Result<String, String> {
        let sa = parse_vertex_sa_json(&creds["apiKey"]);
        let adc = parse_vertex_adc_json(&creds["apiKey"]);
        let oauth = sa.is_some() || adc.is_some() || truthy(&creds["accessToken"]);
        let raw_key = if oauth { None } else { creds["apiKey"].as_str().filter(|s| !s.is_empty()) };
        let project = sa
            .as_ref()
            .and_then(|s| s["project_id"].as_str().map(str::to_owned))
            .or_else(|| adc.as_ref().and_then(|a| a["quota_project_id"].as_str().map(str::to_owned)))
            .or_else(|| creds["providerSpecificData"]["projectId"].as_str().filter(|s| !s.is_empty()).map(str::to_owned));
        if self.id == "vertex-partner" {
            let p = project.ok_or("Vertex partner models require a project_id. Add it in providerSpecificData or use Service Account JSON.")?;
            let url = format!("https://aiplatform.googleapis.com/v1/projects/{p}/locations/global/endpoints/openapi/chat/completions");
            return Ok(match raw_key {
                Some(k) => format!("{url}?key={k}"),
                None => url,
            });
        }
        let action = if stream { "streamGenerateContent" } else { "generateContent" };
        if oauth {
            let p = project.ok_or("Vertex OAuth/ADC requires a project_id. Add quota_project_id to your ADC JSON or set providerSpecificData.projectId.")?;
            let loc = creds["providerSpecificData"]["location"].as_str().filter(|s| !s.is_empty()).unwrap_or("us-central1");
            let mut url = format!("https://aiplatform.googleapis.com/v1/projects/{p}/locations/{loc}/publishers/google/models/{model}:{action}");
            if stream {
                url.push_str("?alt=sse");
            }
            return Ok(url);
        }
        let mut url = format!("https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:{action}");
        if stream {
            url.push_str("?alt=sse");
        }
        if let Some(k) = raw_key {
            url.push_str(&format!("{}key={k}", if stream { "&" } else { "?" }));
        }
        Ok(url)
    }

    fn build_headers(&self, creds: &Value, stream: bool, _u: &str, _m: &str, _b: &Value) -> Headers {
        let mut h = Headers::default();
        h.set("Content-Type", "application/json");
        if let Some(t) = crate::exec::cred_str(creds, "accessToken") {
            h.set("Authorization", format!("Bearer {t}"));
        }
        if stream {
            h.set("Accept", "text/event-stream");
        }
        h
    }

    fn transform_request(&self, _m: &str, body: Value, _s: bool, _c: &Value) -> Value {
        body
    }

    async fn refresh_credentials(&self, creds: &Value) -> Option<Value> {
        let sa = parse_vertex_sa_json(&creds["apiKey"])?;
        let (tok, exp) = refresh_vertex_token(&sa).await?;
        Some(json!({"accessToken": tok, "expiresAt": crate::jsv::iso_from_ms(exp)}))
    }

    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let sa = parse_vertex_sa_json(&args.creds["apiKey"]);
        let adc = parse_vertex_adc_json(&args.creds["apiKey"]);
        if let Some(sa) = &sa {
            let (tok, _) = refresh_vertex_token(sa).await.ok_or("Vertex: failed to mint access token from Service Account JSON")?;
            args.creds["accessToken"] = json!(tok);
        }
        if let Some(a) = &adc {
            let r = refresh_google_token(a["refresh_token"].as_str().unwrap_or(""), a["client_id"].as_str().unwrap_or(""), a["client_secret"].as_str().unwrap_or("")).await;
            let tok = r.as_ref().and_then(|v| v["accessToken"].as_str()).ok_or("Vertex: failed to refresh access token from ADC JSON (authorized_user)")?;
            args.creds["accessToken"] = json!(tok);
        }
        if self.id == "vertex-partner" && sa.is_none() && adc.is_none() && !truthy(&args.creds["providerSpecificData"]["projectId"]) {
            let key = args.creds["apiKey"].as_str().unwrap_or("").to_string();
            let pid = resolve_project_id(args.creds, &key).await.ok_or("Vertex: could not resolve project_id from API key. Please add it manually in provider settings.")?;
            if !args.creds["providerSpecificData"].is_object() {
                args.creds["providerSpecificData"] = json!({});
            }
            args.creds["providerSpecificData"]["projectId"] = json!(pid);
        }
        let url = self.build_url(args.model, args.stream, 0, args.creds)?;
        let h = self.build_headers(args.creds, args.stream, &url, args.model, &args.body);
        let body = self.transform_request(args.model, args.body.clone(), args.stream, args.creds);
        let up = post_json(args.creds, &url, &h, &body, self.config()["timeoutMs"].as_u64().unwrap_or(60_000)).await?;
        Ok(ExecResult { response: up, url, headers: h.0, body, response_format: None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls() {
        let v = Vertex { id: "vertex".into() };
        let u = v.build_url("gemini-2.5-pro", true, 0, &json!({"apiKey": "AIza"})).unwrap();
        assert_eq!(u, "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse&key=AIza");
        let u = v.build_url("g", false, 0, &json!({"accessToken": "t", "providerSpecificData": {"projectId": "p", "location": "eu"}})).unwrap();
        assert_eq!(u, "https://aiplatform.googleapis.com/v1/projects/p/locations/eu/publishers/google/models/g:generateContent");
        let vp = Vertex { id: "vertex-partner".into() };
        assert!(vp.build_url("m", true, 0, &json!({"apiKey": "k"})).is_err());
    }
}
