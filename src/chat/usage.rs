//! Token usage helpers (port of open-sse/utils/usageTracking.js).

use serde_json::{Map, Value, json};

use crate::translate::{CLAUDE, GEMINI, GEMINI_CLI, ANTIGRAVITY, OPENAI_RESPONSES};

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
    .filter(|f| f.is_finite())
}

fn jnum(f: f64) -> Value {
    if f.fract() == 0.0 && f.abs() < 9e15 { json!(f as i64) } else { json!(f) }
}

/// normalizeUsage: keep finite numeric token fields + details objects.
pub fn normalize_usage(u: &Value) -> Option<Value> {
    if !u.is_object() {
        return None;
    }
    let mut out = Map::new();
    for k in ["prompt_tokens", "completion_tokens", "total_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "cached_tokens", "reasoning_tokens"] {
        if let Some(f) = u.get(k).and_then(num) {
            out.insert(k.into(), jnum(f));
        }
    }
    for k in ["prompt_tokens_details", "completion_tokens_details"] {
        if u[k].is_object() {
            out.insert(k.into(), u[k].clone());
        }
    }
    (!out.is_empty()).then_some(Value::Object(out))
}

fn or0(v: &Value) -> Value {
    if crate::jsv::truthy(v) { v.clone() } else { json!(0) }
}

/// extractUsage(chunk) — any wire format → normalized OpenAI-ish usage.
pub fn extract_usage(chunk: &Value) -> Option<Value> {
    if !chunk.is_object() {
        return None;
    }
    let ty = chunk["type"].as_str().unwrap_or("");
    if ty == "message_start" && chunk["message"]["usage"].is_object() {
        let u = &chunk["message"]["usage"];
        return normalize_usage(&json!({
            "prompt_tokens": or0(&u["input_tokens"]), "completion_tokens": or0(&u["output_tokens"]),
            "cache_read_input_tokens": u["cache_read_input_tokens"], "cache_creation_input_tokens": u["cache_creation_input_tokens"],
        }));
    }
    if ty == "message_delta" && chunk["usage"].is_object() {
        let u = &chunk["usage"];
        return normalize_usage(&json!({
            "prompt_tokens": or0(&u["input_tokens"]), "completion_tokens": or0(&u["output_tokens"]),
            "cache_read_input_tokens": u["cache_read_input_tokens"], "cache_creation_input_tokens": u["cache_creation_input_tokens"],
        }));
    }
    if (ty == "response.completed" || ty == "response.done") && chunk["response"]["usage"].is_object() {
        let u = &chunk["response"]["usage"];
        let cached = &u["input_tokens_details"]["cached_tokens"];
        let pick = |a: &str, b: &str| if crate::jsv::truthy(&u[a]) { u[a].clone() } else { or0(&u[b]) };
        let mut v = json!({
            "prompt_tokens": pick("input_tokens", "prompt_tokens"),
            "completion_tokens": pick("output_tokens", "completion_tokens"),
            "cached_tokens": cached,
            "reasoning_tokens": u["output_tokens_details"]["reasoning_tokens"],
        });
        if crate::jsv::truthy(cached) {
            v["prompt_tokens_details"] = json!({"cached_tokens": cached});
        }
        return normalize_usage(&v);
    }
    if chunk["usage"].is_object() && chunk["usage"].get("prompt_tokens").is_some() {
        let u = &chunk["usage"];
        let c = &u["prompt_tokens_details"]["cached_tokens"];
        return normalize_usage(&json!({
            "prompt_tokens": u["prompt_tokens"], "completion_tokens": or0(&u["completion_tokens"]),
            "cached_tokens": if crate::jsv::truthy(c) { c.clone() } else { u["prompt_cache_hit_tokens"].clone() },
            "reasoning_tokens": u["completion_tokens_details"]["reasoning_tokens"],
            "prompt_tokens_details": u["prompt_tokens_details"], "completion_tokens_details": u["completion_tokens_details"],
        }));
    }
    let meta = if chunk["usageMetadata"].is_object() { &chunk["usageMetadata"] } else { &chunk["response"]["usageMetadata"] };
    if meta.is_object() {
        return normalize_usage(&json!({
            "prompt_tokens": or0(&meta["promptTokenCount"]), "completion_tokens": or0(&meta["candidatesTokenCount"]),
            "total_tokens": meta["totalTokenCount"], "cached_tokens": meta["cachedContentTokenCount"], "reasoning_tokens": meta["thoughtsTokenCount"],
        }));
    }
    if chunk["done"] == json!(true) && chunk["prompt_eval_count"].is_number() {
        let p = chunk["prompt_eval_count"].as_i64().unwrap_or(0);
        let c = chunk["eval_count"].as_i64().unwrap_or(0);
        return normalize_usage(&json!({"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}));
    }
    None
}

/// Usage straight from a non-streaming response body (requestDetail.extractUsageFromResponse).
pub fn extract_usage_from_body(body: &Value) -> Option<Value> {
    if body["type"] == "message" && body["usage"].is_object() {
        let u = &body["usage"];
        return normalize_usage(&json!({
            "prompt_tokens": or0(&u["input_tokens"]), "completion_tokens": or0(&u["output_tokens"]),
            "cache_read_input_tokens": u["cache_read_input_tokens"], "cache_creation_input_tokens": u["cache_creation_input_tokens"],
        }));
    }
    if body["object"] == "response" && body["usage"].is_object() {
        return extract_usage(&json!({"type": "response.completed", "response": body}));
    }
    extract_usage(body)
}

/// mergeUsage: field-wise max for numbers, latest for nested objects.
pub fn merge_usage(prev: Option<&Value>, next: Option<&Value>) -> Option<Value> {
    let Some(prev) = prev.filter(|p| p.is_object()) else { return next.cloned() };
    let Some(next) = next else { return Some(prev.clone()) };
    let mut merged = prev.clone();
    for (k, v) in next.as_object().into_iter().flatten() {
        if let Some(f) = v.as_f64().filter(|f| f.is_finite()) {
            let cur = merged[k].as_f64().unwrap_or(0.0);
            merged[k] = jnum(cur.max(f));
        } else if v.is_object() || v.is_array() {
            merged[k] = v.clone();
        }
    }
    Some(merged)
}

pub fn has_valid_usage(u: Option<&Value>) -> bool {
    let Some(u) = u.filter(|u| u.is_object()) else { return false };
    ["prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens", "promptTokenCount", "candidatesTokenCount"]
        .iter()
        .any(|k| u[*k].as_f64().map(|f| f > 0.0).unwrap_or(false))
}

/// filterUsageForFormat: keep only the fields a client format understands.
pub fn filter_usage_for_format(u: &Value, format: &str) -> Value {
    if !u.is_object() {
        return u.clone();
    }
    let fields: &[&str] = match format {
        CLAUDE => &["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "estimated"],
        GEMINI | GEMINI_CLI | ANTIGRAVITY => &["promptTokenCount", "candidatesTokenCount", "totalTokenCount", "cachedContentTokenCount", "thoughtsTokenCount", "estimated"],
        OPENAI_RESPONSES => &["input_tokens", "output_tokens", "input_tokens_details", "output_tokens_details", "estimated"],
        _ => &["prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "reasoning_tokens", "prompt_tokens_details", "completion_tokens_details", "estimated"],
    };
    let mut out = Map::new();
    for f in fields {
        if let Some(v) = u.get(*f) {
            out.insert((*f).into(), v.clone());
        }
    }
    Value::Object(out)
}

pub fn estimate_input_tokens(body: &Value) -> i64 {
    if !body.is_object() {
        return 0;
    }
    let n = body.to_string().encode_utf16().count() as i64;
    (n + 3) / 4
}

pub fn estimate_output_tokens(len: usize) -> i64 {
    if len == 0 { 0 } else { ((len / 4) as i64).max(1) }
}

/// estimateUsage(body, contentLength, format) — fallback when the provider reports none.
pub fn estimate_usage(body: &Value, content_len: usize, format: &str) -> Value {
    let i = estimate_input_tokens(body);
    let o = estimate_output_tokens(content_len);
    if format == CLAUDE {
        json!({"input_tokens": i, "output_tokens": o, "estimated": true})
    } else {
        json!({"prompt_tokens": i, "completion_tokens": o, "total_tokens": i + o, "estimated": true})
    }
}

/// canonicalizeUsage: prompt includes cache read + creation, cached = cache read portion.
pub fn canonicalize_usage(u: &Value) -> Option<Value> {
    if !u.is_object() {
        return None;
    }
    let n = |v: &Value| num(v).unwrap_or(0.0) as i64;
    let first = |a: &str, b: &str| if u.get(a).map(|x| !x.is_null()).unwrap_or(false) { &u[a] } else { &u[b] };
    let completion = n(first("completion_tokens", "output_tokens"));
    let reasoning = n(&u["reasoning_tokens"]);
    let cache_creation = n(if u.get("cache_creation_input_tokens").map(|x| !x.is_null()).unwrap_or(false) {
        &u["cache_creation_input_tokens"]
    } else {
        &u["prompt_tokens_details"]["cache_creation_tokens"]
    });
    let mut prompt = n(first("prompt_tokens", "input_tokens"));
    let cached;
    let has = |k: &str| u.get(k).map(|x| !x.is_null()).unwrap_or(false);
    if !has("cached_tokens") && (has("cache_read_input_tokens") || has("cache_creation_input_tokens")) {
        cached = n(&u["cache_read_input_tokens"]);
        prompt += cached + cache_creation;
    } else {
        cached = n(if has("cached_tokens") { &u["cached_tokens"] } else { &u["prompt_tokens_details"]["cached_tokens"] });
    }
    let mut r = json!({
        "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
        "cached_tokens": cached, "cache_creation_input_tokens": cache_creation,
    });
    if reasoning > 0 {
        r["reasoning_tokens"] = json!(reasoning);
    }
    Some(r)
}

/// Converts an OpenAI-shaped usage object into the client's wire shape.
pub fn usage_for_client(u: &Value, format: &str) -> Value {
    let p = u["prompt_tokens"].as_i64();
    let c = u["completion_tokens"].as_i64();
    match format {
        CLAUDE if p.is_some() && u.get("input_tokens").is_none() => {
            let mut o = json!({"input_tokens": p.unwrap_or(0), "output_tokens": c.unwrap_or(0)});
            if u["estimated"] == json!(true) {
                o["estimated"] = json!(true);
            }
            o
        }
        _ => filter_usage_for_format(u, format),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::translate::OPENAI;

    #[test]
    fn extract_and_merge() {
        let a = extract_usage(&json!({"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 5}}})).unwrap();
        let b = extract_usage(&json!({"type": "message_delta", "usage": {"output_tokens": 42}})).unwrap();
        let m = merge_usage(Some(&a), Some(&b)).unwrap();
        assert_eq!(m["prompt_tokens"], 10);
        assert_eq!(m["completion_tokens"], 42);
        assert_eq!(m["cache_read_input_tokens"], 5);
        let c = canonicalize_usage(&m).unwrap();
        assert_eq!(c["prompt_tokens"], 15);
        assert_eq!(c["cached_tokens"], 5);
        let g = extract_usage(&json!({"response": {"usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4}}})).unwrap();
        assert_eq!(g["completion_tokens"], 4);
        assert!(has_valid_usage(Some(&g)));
        assert!(!has_valid_usage(Some(&json!({}))));
    }

    #[test]
    fn filter() {
        let u = json!({"prompt_tokens": 1, "completion_tokens": 2, "input_tokens": 1, "output_tokens": 2, "foo": 1});
        assert_eq!(filter_usage_for_format(&u, CLAUDE), json!({"input_tokens": 1, "output_tokens": 2}));
        assert_eq!(filter_usage_for_format(&u, OPENAI), json!({"prompt_tokens": 1, "completion_tokens": 2}));
    }
}
