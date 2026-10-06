//! Provider registry. The data (`data/*.json`) is generated from 9router's
//! provider registry (MIT, © decolua and contributors) so FastRouter knows
//! every provider, transport, model, OAuth and media config it supports.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

const REGISTRY_JSON: &str = include_str!("../data/registry.json");
const BUILT_JSON: &str = include_str!("../data/providers_built.json");

pub struct Registry {
    /// Raw registry entries, in 9router order.
    pub entries: Vec<Value>,
    /// PROVIDERS[id] — runtime transport config.
    pub providers: serde_json::Map<String, Value>,
    /// PROVIDER_MODELS[alias] — model lists keyed by alias (or id).
    pub models: serde_json::Map<String, Value>,
    pub oauth: serde_json::Map<String, Value>,
    pub media: serde_json::Map<String, Value>,
    alias_to_id: HashMap<String, String>,
    id_to_alias: HashMap<String, String>,
}

pub static REG: LazyLock<Registry> = LazyLock::new(|| {
    let entries: Vec<Value> = serde_json::from_str(&crate::secrets::substitute(REGISTRY_JSON)).expect("registry.json");
    let built: Value = serde_json::from_str(&crate::secrets::substitute(BUILT_JSON)).expect("providers_built.json");
    let take = |k: &str| built[k].as_object().cloned().unwrap_or_default();

    let mut alias_to_id: HashMap<String, String> = [
        ("el", "elevenlabs"),
        ("jina", "jina-ai"),
        ("jina-ai", "jina-ai"),
        ("polly", "aws-polly"),
        ("aws-polly", "aws-polly"),
    ]
    .into_iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect();
    let mut id_to_alias = HashMap::new();
    for e in &entries {
        let id = e["id"].as_str().unwrap_or_default().to_string();
        alias_to_id.insert(id.clone(), id.clone());
        if let Some(a) = e["alias"].as_str() {
            alias_to_id.insert(a.to_string(), id.clone());
            if a != id {
                id_to_alias.insert(id.clone(), a.to_string());
            }
        }
        for a in e["aliases"].as_array().into_iter().flatten() {
            if let Some(a) = a.as_str() {
                alias_to_id.insert(a.to_string(), id.clone());
            }
        }
    }
    Registry {
        entries,
        providers: take("PROVIDERS"),
        models: take("PROVIDER_MODELS"),
        oauth: take("PROVIDER_OAUTH"),
        media: take("PROVIDER_MEDIA"),
        alias_to_id,
        id_to_alias,
    }
});

static NULL: Value = Value::Null;

impl Registry {
    pub fn entry(&self, id: &str) -> Option<&Value> {
        self.entries.iter().find(|e| e["id"] == id)
    }

    /// PROVIDERS[id] (Null when absent).
    pub fn transport(&self, id: &str) -> &Value {
        self.providers.get(id).unwrap_or(&NULL)
    }

    pub fn oauth(&self, id: &str) -> &Value {
        self.oauth.get(id).unwrap_or(&NULL)
    }

    pub fn media(&self, id: &str) -> &Value {
        self.media.get(id).unwrap_or(&NULL)
    }

    pub fn resolve_alias(&self, alias_or_id: &str) -> String {
        self.alias_to_id.get(alias_or_id).cloned().unwrap_or_else(|| alias_or_id.to_string())
    }

    /// PROVIDER_ID_TO_ALIAS[id] || id
    pub fn alias_of(&self, id: &str) -> String {
        self.id_to_alias.get(id).cloned().unwrap_or_else(|| id.to_string())
    }

    pub fn provider_models(&self, alias_or_id: &str) -> &[Value] {
        self.models.get(alias_or_id).and_then(|m| m.as_array()).map(|a| a.as_slice()).unwrap_or(&[])
    }

    pub fn models_by_provider_id(&self, id: &str) -> &[Value] {
        self.provider_models(&self.alias_of(id))
    }

    pub fn has_models_key(&self, alias: &str) -> bool {
        self.models.contains_key(alias)
    }
}

static SUFFIX_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\([^()]+\)\s*$").unwrap());
static DIGIT_DASH_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d)-(\d)").unwrap());

/// Strips a trailing `(value)` thinking suffix.
pub fn strip_paren_suffix(id: &str) -> String {
    SUFFIX_RE.replace(id, "").trim().to_string()
}

pub fn normalize_model_id(id: &str) -> String {
    // Same non-overlapping semantics as JS replace(/(\d)-(\d)/g, "$1.$2").
    DIGIT_DASH_RE.replace_all(id, "$1.$2").to_string()
}

fn find_model<'a>(models: &'a [Value], model_id: &str, alias: &str) -> Option<&'a Value> {
    let base = strip_paren_suffix(model_id);
    if let Some(m) = models.iter().find(|m| m["id"] == model_id || m["id"] == base.as_str()) {
        return Some(m);
    }
    if alias != "kr" && alias != "kiro" {
        return None;
    }
    let normalized = normalize_model_id(&base);
    if normalized == base {
        return None;
    }
    models.iter().find(|m| m["id"] == normalized.as_str())
}

pub fn find_model_entry(alias: &str, model_id: &str) -> Option<&'static Value> {
    find_model(REG.provider_models(alias), model_id, alias)
}

fn is_opencode_alias(alias: &str) -> bool {
    alias.is_empty() || ["oc", "opencode", "ocg", "opencode-go", "ocz", "opencode-zen"].contains(&alias)
}

pub fn is_muse_spark_model(id: &str) -> bool {
    let clean = strip_paren_suffix(id);
    let base = clean.rsplit('/').next().unwrap_or(&clean);
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^muse[-_]?spark(?:$|[-_:.\s])").unwrap());
    RE.is_match(base)
}

pub fn is_deepseek_model(id: &str) -> bool {
    let clean = strip_paren_suffix(id);
    let base = clean.rsplit('/').next().unwrap_or(&clean);
    base.to_ascii_lowercase().starts_with("deepseek-")
}

/// (supportedFormats, targetFormat) for OpenCode models outside the curated registry.
fn opencode_family(model: &str) -> Option<(Vec<&'static str>, Option<&'static str>)> {
    let base = strip_paren_suffix(model);
    let lower = base.to_ascii_lowercase();
    if lower.starts_with("grok") || lower.starts_with("gpt") || is_muse_spark_model(&base) || Regex::new(r"(?i)^muse[-_]?spark").unwrap().is_match(&base) {
        return Some((vec!["openai-responses"], Some("openai-responses")));
    }
    if base.starts_with("deepseek-v4-pro") || base.starts_with("deepseek-v4-flash") {
        return Some((vec!["openai", "claude", "openai-responses"], None));
    }
    if base.starts_with("minimax") || base.starts_with("qwen") {
        return Some((vec!["openai", "claude"], None));
    }
    if lower.starts_with("claude-") {
        return Some((vec!["claude"], None));
    }
    None
}

pub fn get_model_target_format(alias: &str, model: &str) -> Option<String> {
    if is_opencode_alias(alias) && is_muse_spark_model(model) {
        return Some("openai-responses".into());
    }
    if !REG.has_models_key(alias) {
        return None;
    }
    if let Some(found) = find_model_entry(alias, model) {
        return found["targetFormat"].as_str().map(str::to_owned);
    }
    if is_opencode_alias(alias) {
        return opencode_family(model).and_then(|f| f.1.map(str::to_owned));
    }
    None
}

pub fn get_model_supported_formats(alias: &str, model: &str) -> Option<Vec<String>> {
    if !REG.has_models_key(alias) {
        return None;
    }
    if let Some(found) = find_model_entry(alias, model) {
        return found["supportedFormats"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect());
    }
    if is_opencode_alias(alias) {
        return Some(
            opencode_family(model)
                .map(|f| f.0.into_iter().map(str::to_owned).collect())
                .unwrap_or_else(|| vec!["openai".into()]),
        );
    }
    None
}

pub fn get_model_type(alias: &str, model: &str) -> Option<String> {
    let found = find_model_entry(alias, model)?;
    found["kind"].as_str().or_else(|| found["type"].as_str()).map(str::to_owned)
}

pub fn get_model_strip(alias: &str, model: &str) -> Vec<String> {
    find_model_entry(alias, model)
        .and_then(|m| m["strip"].as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect())
        .unwrap_or_default()
}

pub fn get_model_upstream_id(alias: &str, model: &str) -> String {
    let (base, suffix) = match SUFFIX_RE.find(model) {
        Some(m) => (model[..m.start()].trim().to_string(), m.as_str().to_string()),
        None => (model.to_string(), String::new()),
    };
    let found = find_model(REG.provider_models(alias), &base, alias);
    let resolved = found.and_then(|f| f["upstreamModelId"].as_str().or_else(|| f["id"].as_str()));
    if let Some(resolved) = resolved {
        let (rbase, preset) = match SUFFIX_RE.find(resolved) {
            Some(m) => (resolved[..m.start()].trim().to_string(), m.as_str().to_string()),
            None => (resolved.to_string(), String::new()),
        };
        return rbase + if suffix.is_empty() { &preset } else { &suffix };
    }
    if alias == "cx" && base.ends_with("-review") {
        return base[..base.len() - "-review".len()].to_string() + &suffix;
    }
    base + &suffix
}

/// Infers a provider from a bare model name (services/model.js).
pub fn infer_provider_from_model(model: &str) -> &'static str {
    let m = model.to_ascii_lowercase();
    if m == "codex-auto-review" {
        return "codex";
    }
    let re56 = Regex::new(r"^gpt-[56]\.").unwrap();
    if re56.is_match(&m) || m.starts_with("gpt-6-") || m.starts_with("gpt-daybreak-") || m.starts_with("gpt-reserve") {
        return "codex";
    }
    if m.starts_with("claude-") {
        return "anthropic";
    }
    if m.starts_with("gemini-") {
        return "gemini";
    }
    if m.starts_with("gpt-") {
        return "openai";
    }
    if Regex::new(r"^o[134]").unwrap().is_match(&m) {
        return "openai";
    }
    if m.starts_with("deepseek-") {
        return "openrouter";
    }
    "openai"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_loads() {
        assert_eq!(REG.entries.len(), 129);
        assert_eq!(REG.resolve_alias("cc"), "claude");
        assert_eq!(REG.alias_of("claude"), "cc");
        assert_eq!(REG.transport("groq")["baseUrl"], "https://api.groq.com/openai/v1/chat/completions");
        assert!(!REG.provider_models("cc").is_empty());
    }

    #[test]
    fn upstream_ids() {
        assert_eq!(get_model_upstream_id("zz", "foo(high)"), "foo(high)");
        assert_eq!(normalize_model_id("claude-sonnet-4-5"), "claude-sonnet-4.5");
    }
}
