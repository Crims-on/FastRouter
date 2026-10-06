//! Model capabilities, thinking levels, vision heuristics and pricing.
//! Tables come from 9router (`data/capabilities.json`, `data/pricing.json`).

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};

use crate::jsv::{glob_match, spread};

const CAPS_JSON: &str = include_str!("../data/capabilities.json");
const PRICING_JSON: &str = include_str!("../data/pricing.json");

static CAPS: LazyLock<Value> = LazyLock::new(|| {
    let mut v: Value = serde_json::from_str(CAPS_JSON).expect("capabilities.json");
    // Aliases defined in capabilities.js after the table literal.
    let provider = v["PROVIDER"].clone();
    for (alias, src) in [("qoder-cn", "qoder"), ("cx", "codex"), ("dv", "devin-cli"), ("devin", "devin-cli")] {
        if !provider[src].is_null() {
            v["PROVIDER"][alias] = provider[src].clone();
        }
    }
    v
});

static PRICING: LazyLock<Value> = LazyLock::new(|| serde_json::from_str(PRICING_JSON).expect("pricing.json"));

/// Resolved capability object (always merged over DEFAULT).
#[derive(Clone, Debug)]
pub struct Caps(pub Value);

impl Caps {
    pub fn get(&self, k: &str) -> &Value {
        &self.0[k]
    }
    pub fn flag(&self, k: &str) -> bool {
        self.0[k].as_bool().unwrap_or(false)
    }
    pub fn reasoning(&self) -> bool {
        self.flag("reasoning")
    }
    pub fn max_output(&self) -> Option<i64> {
        self.0["maxOutput"].as_i64()
    }
    pub fn thinking_format(&self) -> Option<&str> {
        self.0["thinkingFormat"].as_str()
    }
    pub fn can_disable(&self) -> bool {
        self.0["thinkingCanDisable"].as_bool() != Some(false)
    }
}

static NOT_VISION: LazyLock<Regex> = LazyLock::new(|| {
    let sep = "[-_/:.]";
    let parts = [
        format!("(^|{sep})(image|img)({sep}|$)"),
        "stable-image".into(),
        "gen[0-9]_image".into(),
        "nanobanana".into(),
        "imagine".into(),
        "t2v".into(),
        "i2v".into(),
        "flux".into(),
        "dall".into(),
        "sdxl".into(),
        "diffusion".into(),
        "embed".into(),
        "rerank".into(),
        "guard".into(),
        "moderation".into(),
        "tts".into(),
        "stt".into(),
        "whisper".into(),
        "voice".into(),
        "speech".into(),
        "audio".into(),
    ];
    Regex::new(&format!("(?i){}", parts.join("|"))).unwrap()
});

static VISION_NAME: LazyLock<Regex> = LazyLock::new(|| {
    let sep = "[-_/:.]";
    let parts = [
        format!("(^|{sep})(vision|vl|vlm|multimodal|omni|visual)({sep}|$)"),
        format!("[0-9]\\.[0-9]+v({sep}|$)"),
        format!("(^|{sep})glm-[0-9]+v({sep}|$)"),
        "(^|[-_/:.])(llava|pixtral|internvl|cogvlm|minicpm-v|moondream|idefics|fuyu)".into(),
    ];
    Regex::new(&format!("(?i){}", parts.join("|"))).unwrap()
});

pub fn looks_like_vision_model(model: &str) -> bool {
    if model.is_empty() {
        return false;
    }
    let id = model.to_lowercase();
    if NOT_VISION.is_match(&id) {
        return false;
    }
    VISION_NAME.is_match(&id)
}

fn refine(base: &Value, model: &str) -> Caps {
    let mut result = spread(&CAPS["DEFAULT"], base);
    if result["vision"] != json!(true) && looks_like_vision_model(model) {
        result["vision"] = json!(true);
    }
    Caps(result)
}

const COMMANDCODE_TEXT_ONLY: &[&str] = &[
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-flash-fast",
    "zai-org/glm-5.3",
    "zai-org/glm-5.2",
    "zai-org/glm-5.2-fast",
    "zai-org/glm-5.1",
    "zai-org/glm-5",
    "minimaxai/minimax-m2.7",
    "minimax/minimax-m2.7-free",
    "minimaxai/minimax-m2.5",
    "xiaomi/mimo-v2.5-pro",
    "qwen/qwen3.6-max-preview",
    "qwen/qwen3.7-max",
    "meituan/longcat-2.0:free",
    "stepfun/step-3.5-flash",
    "tencent/hy4-preview",
    "tencent/hy3",
    "tencent/hy3-paid",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "poolside/laguna-s-2.1-free",
    "inclusionai/ling-3.0-flash-free",
    "inclusionai/ling-3.0-flash-sante:free",
];

fn is_commandcode_text_only(model: &str) -> bool {
    let key = model.to_lowercase();
    COMMANDCODE_TEXT_ONLY.iter().any(|id| {
        let base = id.rsplit('/').next().unwrap_or(id);
        key == *id || key == base || key.ends_with(&format!("/{base}"))
    })
}

/// getCapabilitiesForModel(provider, model)
pub fn caps_for(provider: Option<&str>, model: &str) -> Caps {
    let default = &CAPS["DEFAULT"];
    if model.is_empty() {
        return Caps(default.clone());
    }
    let base_model = model.rsplit('/').next().unwrap_or(model);

    if matches!(provider, Some("commandcode") | Some("cmc")) {
        let pc = &CAPS["PROVIDER"]["commandcode"];
        if !pc[model].is_null() {
            return Caps(spread(default, &pc[model]));
        }
        if !pc[base_model].is_null() {
            return Caps(spread(default, &pc[base_model]));
        }
        return Caps(spread(
            default,
            &json!({
                "reasoning": true,
                "thinkingFormat": "commandcode",
                "thinkingEffortSupported": true,
                "vision": !is_commandcode_text_only(model),
                "contextWindow": 1000000,
                "maxOutput": 384000,
            }),
        ));
    }

    if let Some(p) = provider {
        let pc = &CAPS["PROVIDER"][p];
        if !pc[model].is_null() {
            return Caps(spread(default, &pc[model]));
        }
        if !pc[base_model].is_null() {
            return Caps(spread(default, &pc[base_model]));
        }
    }

    let mc = &CAPS["MODEL"];
    if !mc[base_model].is_null() {
        return refine(&mc[base_model], model);
    }
    if !mc[model].is_null() {
        return refine(&mc[model], model);
    }

    for entry in CAPS["PATTERN"].as_array().into_iter().flatten() {
        let pattern = entry["pattern"].as_str().unwrap_or("");
        if glob_match(pattern, base_model) || glob_match(pattern, model) {
            return refine(&entry["caps"], model);
        }
    }
    refine(&Value::Null, model)
}

// ---------------------------------------------------------------------------
// Kiro effort path (config/kiroConstants.js) — needed by thinking levels.
// ---------------------------------------------------------------------------

pub fn parse_claude_version(model: &str) -> Option<(i64, Option<i64>)> {
    static RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?:^|[/.])claude(?:[/.][a-z]+)*[/.](\d+)(?:[/.](\d+))?(?:[/.]|$)").unwrap());
    let normalized = model.to_lowercase().replace('-', ".");
    let c = RE.captures(&normalized)?;
    let major = c.get(1)?.as_str().parse().ok()?;
    let minor = c.get(2).and_then(|m| m.as_str().parse().ok());
    Some((major, minor))
}

pub fn resolve_kiro_effort_path(model: &str) -> Option<&'static str> {
    static GPT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:^|[/.])gpt[/.]5[/.]6(?:[/.]|$)").unwrap());
    let normalized = model.to_lowercase().replace('-', ".");
    if GPT.is_match(&normalized) {
        return Some("reasoning");
    }
    if !normalized.contains("claude") {
        return None;
    }
    let (major, minor) = parse_claude_version(model)?;
    let date_suffix_minor = minor.map(|m| m >= 1000).unwrap_or(false);
    if major < 4 || (major == 4 && (minor.is_none() || minor.unwrap() <= 5 || date_suffix_minor)) {
        None
    } else {
        Some("output_config")
    }
}

// ---------------------------------------------------------------------------
// Thinking levels (providers/thinkingLevels.js)
// ---------------------------------------------------------------------------

fn format_levels(fmt: &str) -> &'static [&'static str] {
    match fmt {
        "openai" => &["none", "minimal", "low", "medium", "high", "xhigh"],
        "claude-adaptive" | "claude-budget" => &["none", "low", "medium", "high", "xhigh", "max"],
        "gemini-level" => &["minimal", "low", "medium", "high"],
        "zai" | "minimax" => &["none", "thinking"],
        "kimi" => &["none", "low", "medium", "high", "max"],
        "deepseek" => &["none", "high", "max"],
        "commandcode" => &["none", "low", "medium", "high", "xhigh", "max"],
        _ => &["none", "low", "medium", "high"],
    }
}

const CODEX_56: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max"];
const CODEX_56_ULTRA: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"];

fn pattern_thinking(provider: Option<&str>, model: &str) -> Option<&'static [&'static str]> {
    let table: &[(Option<&str>, &str, &'static [&'static str])] = &[
        (None, "*claude*4.6*", &["none", "low", "medium", "high", "max"]),
        (None, "*claude*4-6*", &["none", "low", "medium", "high", "max"]),
        (Some("codex"), "*gpt-6*", CODEX_56),
        (Some("codex"), "*gpt-5.6-sol*", CODEX_56_ULTRA),
        (Some("codex"), "*gpt-5.6-terra*", CODEX_56_ULTRA),
        (Some("codex"), "*gpt-5.6-luna*", CODEX_56),
        (None, "*codex*", &["low", "medium", "high", "xhigh"]),
        (None, "*mimo*v2.6*", &["none", "low", "medium", "high", "xhigh"]),
        (None, "*mimo*v2.5-pro*", &["none", "low", "medium", "high", "xhigh"]),
        (None, "*deepseek-v4.*", &["none", "low", "medium", "high", "xhigh", "max"]),
        (Some("codebuddy-cn"), "glm-5.3*", &["low", "high", "max"]),
        (Some("codebuddy-cn"), "glm-5.2", &["high", "xhigh"]),
        (Some("codebuddy-cn"), "deepseek-v4*", &["low", "high", "xhigh"]),
        (Some("codebuddy-cn"), "hy3*", &["low", "high"]),
        (Some("codebuddy-cn"), "hy4*", &["high"]),
        (Some("codebuddy-intl"), "deepseek-v4*", &["low", "high", "xhigh"]),
    ];
    table
        .iter()
        .find(|(p, pat, _)| (p.is_none() || *p == provider) && glob_match(pat, model))
        .map(|(_, _, l)| *l)
}

/// getThinkingLevels(provider, model) — None when the model has no reasoning.
pub fn thinking_levels(provider: Option<&str>, model: &str) -> Option<Vec<String>> {
    if provider == Some("kiro") && resolve_kiro_effort_path(model).is_none() {
        return None;
    }
    let caps = caps_for(provider, model);
    if !caps.reasoning() {
        return None;
    }
    let base_id = crate::registry::strip_paren_suffix(model);
    let model_levels: Option<Vec<String>> = if provider == Some("codex") {
        crate::registry::REG
            .provider_models("cx")
            .iter()
            .find(|e| e["id"] == base_id.as_str())
            .and_then(|e| e["thinkingLevels"].as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect())
    } else {
        None
    };
    let mut levels: Vec<String> = model_levels.unwrap_or_else(|| {
        pattern_thinking(provider, model)
            .unwrap_or_else(|| format_levels(caps.thinking_format().unwrap_or("")))
            .iter()
            .map(|s| s.to_string())
            .collect()
    });
    if caps.0["thinkingCanDisable"] == json!(false) {
        levels.retain(|l| l != "none");
    }
    Some(levels)
}

// ---------------------------------------------------------------------------
// Pricing (USD per 1M tokens)
// ---------------------------------------------------------------------------

pub fn pricing_for(provider: Option<&str>, model: &str) -> Option<Value> {
    if model.is_empty() {
        return None;
    }
    if let Some(p) = provider {
        let v = &PRICING["PROVIDER"][p][model];
        if !v.is_null() {
            return Some(v.clone());
        }
    }
    if PRICING["FREE_NS"].as_array().into_iter().flatten().any(|ns| ns.as_str().map(|n| model.starts_with(n)).unwrap_or(false)) {
        return Some(PRICING["ZERO"].clone());
    }
    let base = model.rsplit('/').next().unwrap_or(model);
    for k in [base, model] {
        let v = &PRICING["MODEL"][k];
        if !v.is_null() {
            return Some(v.clone());
        }
    }
    for entry in PRICING["PATTERN"].as_array().into_iter().flatten() {
        let pat = entry["pattern"].as_str().unwrap_or("");
        if glob_match(pat, base) || glob_match(pat, model) {
            return Some(entry["pricing"].clone());
        }
    }
    None
}

/// Estimated cost in USD.
pub fn estimate_cost(provider: &str, model: &str, prompt: i64, completion: i64, cached: i64) -> f64 {
    let Some(p) = pricing_for(Some(provider), model).or_else(|| pricing_for(None, model)) else {
        return 0.0;
    };
    let rate = |k: &str| p[k].as_f64().unwrap_or(0.0);
    let non_cached = (prompt - cached).max(0) as f64;
    let cached_rate = if p["cached"].is_number() { rate("cached") } else { rate("input") };
    (non_cached * rate("input") + cached as f64 * cached_rate + completion as f64 * rate("output")) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_lookup() {
        let c = caps_for(Some("anthropic"), "claude-sonnet-4-5");
        assert!(c.reasoning());
        assert_eq!(c.thinking_format(), Some("claude-budget"));
        let g = caps_for(None, "gemini-2.5-flash");
        assert_eq!(g.thinking_format(), Some("gemini-budget"));
        assert!(!caps_for(None, "llama-3.3-70b").reasoning());
        assert!(looks_like_vision_model("qwen3-vl-plus"));
        assert!(!looks_like_vision_model("gpt-image-1"));
    }

    #[test]
    fn kiro_effort_path() {
        assert_eq!(resolve_kiro_effort_path("claude-sonnet-4.5"), None);
        assert_eq!(resolve_kiro_effort_path("claude-opus-4.7"), Some("output_config"));
        assert_eq!(resolve_kiro_effort_path("gpt-5.6-sol"), Some("reasoning"));
    }

    #[test]
    fn pricing() {
        assert!(pricing_for(None, "claude-sonnet-4-5").is_some());
    }
}
