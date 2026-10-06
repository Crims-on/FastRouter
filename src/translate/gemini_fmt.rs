//! Gemini helpers (port of translator/formats/gemini.js).

use serde_json::{Map, Value, json};

use crate::jsv::truthy;

const UNSUPPORTED: &[&str] = &[
    "minLength", "maxLength", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems", "format", "multipleOf",
    "uniqueItems", "contains", "unevaluatedProperties", "unevaluatedItems", "contentSchema", "prefixItems",
    "additionalItems", "default", "examples", "$schema", "$defs", "definitions", "const", "$ref", "$comment",
    "deprecated", "readOnly", "writeOnly", "additionalProperties", "propertyNames", "patternProperties",
    "enumDescriptions", "anyOf", "oneOf", "allOf", "not", "dependencies", "dependentSchemas", "dependentRequired",
    "title", "optional", "if", "then", "else", "contentMediaType", "contentEncoding", "cornerRadius", "fillColor",
    "fontFamily", "fontSize", "fontWeight", "gap", "padding", "strokeColor", "strokeThickness", "textColor",
    "errorMessage", "errorMessages", "x-errorMessage", "x-errorMessages", "markdownDescription",
    "x-intellij-html-description", "x-taplo-info", "x-taplo", "doNotSuggest", "suggestSortText", "minProperties",
    "maxProperties",
];

pub fn default_safety_settings() -> Value {
    json!([
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "OFF"}
    ])
}

fn split_data_url(url: &str) -> Option<(String, String)> {
    let comma = url.find(',')?;
    let mime_part = &url[5..comma];
    let mime = mime_part.split(';').next().unwrap_or("").to_string();
    Some((mime, url[comma + 1..].to_string()))
}

pub fn convert_openai_content_to_parts(content: &Value) -> Vec<Value> {
    let mut parts = vec![];
    match content {
        Value::String(s) => parts.push(json!({"text": s})),
        Value::Array(items) => {
            for item in items {
                let ty = item["type"].as_str().unwrap_or("");
                let url = item["image_url"]["url"].as_str().unwrap_or("");
                if ty == "text" {
                    parts.push(json!({"text": item["text"]}));
                } else if ty == "image_url" && url.starts_with("data:") {
                    if let Some((mime, data)) = split_data_url(url) {
                        parts.push(json!({"inlineData": {"mime_type": mime, "data": data}}));
                    }
                } else if ty == "image_url" && (url.starts_with("http://") || url.starts_with("https://")) {
                    parts.push(json!({"fileData": {"fileUri": url, "mimeType": "image/*"}}));
                } else if ty == "input_audio" && truthy(&item["input_audio"]["data"]) {
                    let format = item["input_audio"]["format"].as_str().filter(|s| !s.is_empty()).unwrap_or("wav");
                    let mime = if format == "mp3" { "audio/mpeg".to_string() } else { format!("audio/{format}") };
                    parts.push(json!({"inlineData": {"mime_type": mime, "data": item["input_audio"]["data"]}}));
                } else if ty == "audio_url" && item["audio_url"]["url"].as_str().map(|u| u.starts_with("data:")).unwrap_or(false) {
                    if let Some((mime, data)) = split_data_url(item["audio_url"]["url"].as_str().unwrap()) {
                        parts.push(json!({"inlineData": {"mime_type": mime, "data": data}}));
                    }
                } else if ty == "file" && item["file"]["file_data"].as_str().map(|u| u.starts_with("data:")).unwrap_or(false) {
                    if let Some((mime, data)) = split_data_url(item["file"]["file_data"].as_str().unwrap()) {
                        parts.push(json!({"inlineData": {"mime_type": mime, "data": data}}));
                    }
                }
            }
        }
        _ => {}
    }
    parts
}

pub fn generate_request_id() -> String {
    format!("agent-{}", uuid::Uuid::new_v4())
}

pub fn generate_session_id() -> String {
    format!("{}{}", uuid::Uuid::new_v4(), crate::jsv::now_ms())
}

pub fn generate_project_id() -> String {
    let adj = ["useful", "bright", "swift", "calm", "bold"];
    let noun = ["fuze", "wave", "spark", "flow", "core"];
    let r = uuid::Uuid::new_v4();
    let b = r.as_bytes();
    format!("{}-{}-{}", adj[b[0] as usize % 5], noun[b[1] as usize % 5], &uuid::Uuid::new_v4().to_string()[..5])
}

fn walk(v: &mut Value, f: &mut dyn FnMut(&mut Map<String, Value>)) {
    match v {
        Value::Object(o) => {
            f(o);
            for (_, child) in o.iter_mut() {
                walk(child, f);
            }
        }
        Value::Array(a) => {
            for child in a.iter_mut() {
                walk(child, f);
            }
        }
        _ => {}
    }
}

fn remove_unsupported(v: &mut Value) {
    match v {
        Value::Array(a) => a.iter_mut().for_each(remove_unsupported),
        Value::Object(o) => {
            let keys: Vec<String> = o.keys().cloned().collect();
            for k in keys {
                if UNSUPPORTED.contains(&k.as_str()) || k.starts_with("x-") {
                    o.shift_remove(&k);
                    continue;
                }
                if let Some(child) = o.get_mut(&k) {
                    if child.is_object() || child.is_array() {
                        remove_unsupported(child);
                    }
                }
            }
        }
        _ => {}
    }
}

fn select_best(items: &[Value]) -> usize {
    let mut best = 0;
    let mut best_score = -1;
    for (i, item) in items.iter().enumerate() {
        let ty = &item["type"];
        let score = if ty == "object" || truthy(&item["properties"]) {
            3
        } else if ty == "array" || truthy(&item["items"]) {
            2
        } else if truthy(ty) && ty != "null" {
            1
        } else {
            0
        };
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    best
}

fn flatten_any_one(o: &mut Map<String, Value>) {
    for key in ["anyOf", "oneOf"] {
        let Some(list) = o.get(key).and_then(|v| v.as_array()).cloned() else { continue };
        if list.is_empty() {
            continue;
        }
        let non_null: Vec<Value> = list.into_iter().filter(|s| !s.is_null() && s["type"] != "null").collect();
        if non_null.is_empty() {
            continue;
        }
        let sel = non_null[select_best(&non_null)].clone();
        o.shift_remove(key);
        if let Some(so) = sel.as_object() {
            for (k, v) in so {
                o.insert(k.clone(), v.clone());
            }
        }
    }
}

fn merge_all_of(o: &mut Map<String, Value>) {
    let Some(all) = o.get("allOf").and_then(|v| v.as_array()).cloned() else { return };
    let mut props: Option<Map<String, Value>> = None;
    let mut req: Option<Vec<Value>> = None;
    for item in &all {
        if let Some(p) = item["properties"].as_object() {
            let m = props.get_or_insert_with(Map::new);
            for (k, v) in p {
                m.insert(k.clone(), v.clone());
            }
        }
        if let Some(r) = item["required"].as_array() {
            let m = req.get_or_insert_with(Vec::new);
            for x in r {
                if !m.contains(x) {
                    m.push(x.clone());
                }
            }
        }
    }
    o.shift_remove("allOf");
    if let Some(p) = props {
        let mut base = o.get("properties").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        for (k, v) in p {
            base.insert(k, v);
        }
        o.insert("properties".into(), Value::Object(base));
    }
    if let Some(r) = req {
        let mut base = o.get("required").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        base.extend(r);
        o.insert("required".into(), Value::Array(base));
    }
}

/// cleanJSONSchemaForAntigravity(schema) — mutates and returns.
pub fn clean_json_schema(mut schema: Value) -> Value {
    if !schema.is_object() && !schema.is_array() {
        return schema;
    }
    // const → enum
    walk(&mut schema, &mut |o| {
        if o.contains_key("const") && !o.get("enum").map(truthy).unwrap_or(false) {
            let c = o.shift_remove("const").unwrap();
            o.insert("enum".into(), json!([c]));
        }
    });
    // enum values → strings + type
    walk(&mut schema, &mut |o| {
        if let Some(e) = o.get("enum").and_then(|v| v.as_array()).cloned() {
            o.insert("enum".into(), Value::Array(e.iter().map(|v| json!(crate::jsv::js_string(v))).collect()));
            if !o.get("type").map(truthy).unwrap_or(false) {
                o.insert("type".into(), json!("string"));
            }
        }
    });
    walk(&mut schema, &mut merge_all_of);
    // prefixItems → items
    walk(&mut schema, &mut |o| {
        if let Some(pi) = o.get("prefixItems").and_then(|v| v.as_array()).cloned() {
            if !pi.is_empty() {
                let variants: Vec<Value> = pi.into_iter().filter(|s| !s.is_null() && s["type"] != "null").collect();
                let has_items = o.get("items").map(truthy).unwrap_or(false);
                if !has_items && variants.len() == 1 {
                    o.insert("items".into(), variants[0].clone());
                } else if !has_items && variants.len() > 1 {
                    o.insert("items".into(), json!({"anyOf": variants}));
                }
                o.shift_remove("prefixItems");
            }
        }
    });
    walk(&mut schema, &mut flatten_any_one);
    walk(&mut schema, &mut |o| {
        if let Some(types) = o.get("type").and_then(|v| v.as_array()).cloned() {
            let non_null: Vec<Value> = types.into_iter().filter(|t| t != "null").collect();
            o.insert("type".into(), non_null.first().cloned().unwrap_or(json!("string")));
        }
    });
    walk(&mut schema, &mut |o| {
        if o.get("properties").map(truthy).unwrap_or(false) && !o.get("type").map(truthy).unwrap_or(false) {
            o.insert("type".into(), json!("object"));
        }
    });
    walk(&mut schema, &mut |o| {
        if o.get("type") == Some(&json!("array")) && !o.get("items").map(truthy).unwrap_or(false) {
            o.insert("items".into(), json!({"type": "string"}));
        }
    });
    remove_unsupported(&mut schema);
    walk(&mut schema, &mut |o| {
        let props = o.get("properties").and_then(|p| p.as_object()).cloned();
        if let (Some(req), Some(props)) = (o.get("required").and_then(|v| v.as_array()).cloned(), props) {
            let valid: Vec<Value> = req.into_iter().filter(|f| f.as_str().map(|s| props.contains_key(s)).unwrap_or(false)).collect();
            if valid.is_empty() {
                o.shift_remove("required");
            } else {
                o.insert("required".into(), Value::Array(valid));
            }
        }
    });
    fn placeholders(v: &mut Value) {
        let reason = json!({"reason": {"type": "string", "description": "Brief explanation of why you are calling this tool"}});
        match v {
            Value::Object(o) => {
                if o.is_empty() {
                    o.insert("type".into(), json!("object"));
                    o.insert("properties".into(), reason);
                    o.insert("required".into(), json!(["reason"]));
                    return;
                }
                if o.get("type") == Some(&json!("object")) {
                    let empty = o.get("properties").and_then(|p| p.as_object()).map(|p| p.is_empty()).unwrap_or(true);
                    if empty {
                        o.insert("properties".into(), reason);
                        o.insert("required".into(), json!(["reason"]));
                    }
                }
                for (_, child) in o.iter_mut() {
                    if child.is_object() || child.is_array() {
                        placeholders(child);
                    }
                }
            }
            Value::Array(a) => a.iter_mut().for_each(placeholders),
            _ => {}
        }
    }
    placeholders(&mut schema);
    schema
}

/// normalizeGeminiContents
pub fn normalize_gemini_contents(contents: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = vec![];
    for c in contents {
        if !truthy(&c["role"]) || !c["parts"].is_array() {
            continue;
        }
        let parts: Vec<Value> = c["parts"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p.as_object().map(|o| !o.is_empty()).unwrap_or(!p.is_null()))
            .cloned()
            .collect();
        if parts.is_empty() {
            continue;
        }
        if let Some(last) = out.last_mut() {
            if last["role"] == c["role"] {
                last["parts"].as_array_mut().unwrap().extend(parts);
                continue;
            }
        }
        let mut nc = c.clone();
        nc["parts"] = Value::Array(parts);
        out.push(nc);
    }
    if !out.is_empty() && out[0]["role"] != "user" {
        out.insert(0, json!({"role": "user", "parts": [{"text": "..."}]}));
    }
    if out.last().map(|l| l["role"] == "model").unwrap_or(false) {
        let fn_calls: Vec<Value> = out.last().unwrap()["parts"].as_array().unwrap().iter().filter(|p| truthy(&p["functionCall"])).cloned().collect();
        if !fn_calls.is_empty() {
            let responses: Vec<Value> = fn_calls
                .iter()
                .map(|p| {
                    let call = &p["functionCall"];
                    let mut fr = json!({"name": if truthy(&call["name"]) { call["name"].clone() } else { json!("tool") }, "response": {"result": "Continue."}});
                    if truthy(&call["id"]) {
                        fr["id"] = call["id"].clone();
                    }
                    json!({"functionResponse": fr})
                })
                .collect();
            out.push(json!({"role": "user", "parts": responses}));
        } else {
            out.push(json!({"role": "user", "parts": [{"text": "Continue."}]}));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_cleaning() {
        let s = json!({
            "type": "object", "additionalProperties": false, "$schema": "x",
            "properties": {
                "a": {"type": ["string", "null"], "format": "uri"},
                "b": {"anyOf": [{"type": "null"}, {"type": "object", "properties": {"c": {"const": 3}}}]},
                "d": {"type": "array"}
            },
            "required": ["a", "zz"]
        });
        let c = clean_json_schema(s);
        assert!(c.get("additionalProperties").is_none());
        assert_eq!(c["properties"]["a"]["type"], "string");
        assert!(c["properties"]["a"].get("format").is_none());
        assert_eq!(c["properties"]["b"]["type"], "object");
        assert_eq!(c["properties"]["b"]["properties"]["c"]["enum"], json!(["3"]));
        assert_eq!(c["properties"]["d"]["items"]["type"], "string");
        assert_eq!(c["required"], json!(["a"]));
    }
}
