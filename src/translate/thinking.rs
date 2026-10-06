//! Unified thinking normalization (port of translator/concerns/thinkingUnified.js
//! and concerns/thinking.js).

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};

use crate::caps::{Caps, caps_for, thinking_levels};
use crate::jsv::{del, truthy};
use crate::registry::REG;

#[derive(Clone, Debug, PartialEq)]
pub enum Mode {
    None,
    Auto,
    Level(String),
    Budget(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Intent {
    pub mode: Mode,
    pub display: Option<String>,
}

impl Intent {
    fn of(mode: Mode) -> Self {
        Intent { mode, display: None }
    }
}

pub fn level_to_budget(level: &str) -> Option<f64> {
    Some(match level.to_lowercase().as_str() {
        "none" => 0.0,
        "minimal" => 512.0,
        "low" => 1024.0,
        "medium" => 8192.0,
        "high" => 24576.0,
        "xhigh" => 32768.0,
        "max" => 128000.0,
        _ => return None,
    })
}

pub fn budget_to_level(b: f64) -> Option<&'static str> {
    if b.is_nan() || b <= 0.0 {
        return None;
    }
    Some(if b <= 768.0 {
        "minimal"
    } else if b <= 4096.0 {
        "low"
    } else if b <= 16384.0 {
        "medium"
    } else if b <= 28672.0 {
        "high"
    } else if b <= 80384.0 {
        "xhigh"
    } else {
        "max"
    })
}

pub fn budget_to_effort(b: f64) -> Option<&'static str> {
    if b <= 0.0 {
        return None;
    }
    Some(if b <= 2048.0 { "low" } else if b <= 16384.0 { "medium" } else { "high" })
}

fn effort_to_thinking_level(e: &str) -> String {
    let e = e.trim().to_lowercase();
    match e.as_str() {
        "none" | "off" => "minimal".into(),
        "xhigh" | "max" => "high".into(),
        _ => e,
    }
}

static SUFFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(.*)\(([^()]+)\)\s*$").unwrap());

pub fn strip_thinking_suffix(model: &str) -> String {
    match SUFFIX.captures(model) {
        Some(c) => c[1].trim().to_string(),
        None => model.to_string(),
    }
}

/// parseSuffix → (cleanModel, override)
pub fn parse_suffix(model: &str) -> (String, Option<Intent>) {
    let Some(c) = SUFFIX.captures(model) else { return (model.to_string(), None) };
    let clean = c[1].trim().to_string();
    let raw = c[2].trim().to_lowercase();
    let ov = match raw.as_str() {
        "none" | "off" => Some(Mode::None),
        "auto" => Some(Mode::Auto),
        "ultra" => Some(Mode::Level(raw.clone())),
        _ if raw.chars().all(|ch| ch.is_ascii_digit()) && !raw.is_empty() => Some(Mode::Budget(raw.parse().unwrap())),
        _ if level_to_budget(&raw).is_some() => Some(Mode::Level(raw.clone())),
        _ => None,
    };
    (clean, ov.map(Intent::of))
}

fn level_mode(e: &str) -> Mode {
    let e = e.to_lowercase();
    match e.as_str() {
        "none" | "off" => Mode::None,
        "auto" => Mode::Auto,
        _ => Mode::Level(e),
    }
}

fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Null => Some(0.0),
        _ => None,
    }
}

/// extractThinking(body)
pub fn extract_thinking(body: &Value) -> Option<Intent> {
    if !body.is_object() {
        return None;
    }
    if let Some(oc) = body["output_config"]["effort"].as_str().filter(|s| !s.is_empty()) {
        return Some(Intent::of(level_mode(oc)));
    }
    let effort = if !body["reasoning_effort"].is_null() {
        &body["reasoning_effort"]
    } else if body["reasoning"].is_object() {
        &body["reasoning"]["effort"]
    } else {
        &Value::Null
    };
    if let Some(e) = effort.as_str().filter(|s| !s.is_empty()) {
        return Some(Intent::of(level_mode(e)));
    }
    let t = &body["thinking"];
    if t.is_object() {
        if t["type"] == "disabled" {
            return Some(Intent::of(Mode::None));
        }
        if t["type"] == "adaptive" || t["type"] == "enabled" {
            if let Some(b) = number(&t["budget_tokens"]).filter(|b| b.is_finite() && *b > 0.0) {
                if !t["budget_tokens"].is_null() {
                    return Some(Intent::of(Mode::Budget(b)));
                }
            }
            return Some(Intent::of(Mode::Auto));
        }
    }
    let tc = if truthy(&body["thinkingConfig"]) {
        &body["thinkingConfig"]
    } else if truthy(&body["generationConfig"]["thinkingConfig"]) {
        &body["generationConfig"]["thinkingConfig"]
    } else {
        &body["request"]["generationConfig"]["thinkingConfig"]
    };
    if tc.is_object() {
        if let Some(l) = tc["thinkingLevel"].as_str() {
            return Some(Intent::of(Mode::Level(l.to_lowercase())));
        }
        if !tc["thinkingBudget"].is_null() {
            if let Some(tb) = number(&tc["thinkingBudget"]).filter(|x| x.is_finite()) {
                return Some(Intent::of(if tb == 0.0 {
                    Mode::None
                } else if tb < 0.0 {
                    Mode::Auto
                } else {
                    Mode::Budget(tb)
                }));
            }
        }
    }
    if body["enable_thinking"] == json!(false) {
        return Some(Intent::of(Mode::None));
    }
    if body["enable_thinking"] == json!(true) {
        if let Some(tb) = number(&body["thinking_budget"]).filter(|b| b.is_finite() && *b > 0.0) {
            if !body["thinking_budget"].is_null() {
                return Some(Intent::of(Mode::Budget(tb)));
            }
        }
        return Some(Intent::of(Mode::Auto));
    }
    None
}

/// captureThinking(body)
pub fn capture_thinking(body: &Value) -> Option<Intent> {
    let mut cfg = extract_thinking(body)?;
    if cfg.mode == Mode::None {
        return Some(cfg);
    }
    let display = if body["reasoning"].is_object() {
        body["reasoning"]["summary"].as_str().filter(|s| !s.is_empty() && *s != "none").map(|_| "summarized".to_string())
    } else if body["reasoning_effort"].is_string() {
        Some("summarized".to_string())
    } else {
        None
    };
    cfg.display = display;
    Some(cfg)
}

fn format_to_native(target: &str) -> &'static str {
    match target {
        "openai" | "openai-responses" | "openai-response" | "codex" => "openai",
        "claude" => "claude-budget",
        "gemini" | "gemini-cli" | "vertex" | "antigravity" => "gemini-budget",
        "kiro" => "kiro",
        "commandcode" => "commandcode",
        _ => "openai",
    }
}

fn resolve_format(target: &str, model: &str, provider: Option<&str>) -> String {
    if target == "commandcode" {
        return "commandcode".into();
    }
    if let Some(p) = provider {
        if let Some(f) = REG.transport(p)["thinkingFormat"].as_str() {
            return f.into();
        }
    }
    let caps = caps_for(provider, model);
    let openai_wire = target == "openai" || target == "openai-responses";
    if let Some(f) = caps.thinking_format() {
        let native_only = ["gemini-level", "gemini-budget", "claude-budget", "claude-adaptive", "kiro"].contains(&f);
        if !(openai_wire && native_only) {
            return f.into();
        }
    }
    format_to_native(target).into()
}

fn to_budget(cfg: &Mode, range: &Value) -> Option<f64> {
    let mut b = match cfg {
        Mode::Budget(b) => *b,
        Mode::Level(l) => level_to_budget(l)?,
        Mode::Auto => return Some(-1.0),
        Mode::None => return None,
    };
    if range.is_object() {
        if let Some(min) = range["min"].as_f64() {
            if b < min {
                b = min;
            }
        }
        if let Some(max) = range["max"].as_f64() {
            if b > max {
                b = max;
            }
        }
    }
    Some(b)
}

fn to_level(cfg: &Mode) -> Option<String> {
    match cfg {
        Mode::Level(l) => Some(l.clone()),
        Mode::Budget(b) => Some(budget_to_level(*b).unwrap_or("medium").to_string()),
        Mode::Auto => Some("auto".into()),
        Mode::None => None,
    }
}

fn normalize_openai_level(level: String, supported: &Option<Vec<String>>) -> String {
    if level != "max" && level != "ultra" {
        return level;
    }
    let has = |l: &str| supported.as_ref().map(|s| s.iter().any(|x| x == l)).unwrap_or(false);
    if has(&level) {
        return level;
    }
    if level == "ultra" && has("max") {
        return "max".into();
    }
    "xhigh".into()
}

fn gemini_gc(body: &mut Value) -> &mut Value {
    if body["request"].is_object() {
        if !body["request"]["generationConfig"].is_object() {
            body["request"]["generationConfig"] = json!({});
        }
        return &mut body["request"]["generationConfig"];
    }
    if !body["generationConfig"].is_object() {
        body["generationConfig"] = json!({});
    }
    &mut body["generationConfig"]
}

fn ensure_gemini_floor(body: &mut Value, floor: i64, caps: &Caps) {
    let cap = caps.max_output().unwrap_or(floor);
    let target = floor.min(cap);
    let gc = gemini_gc(body);
    let cur = gc["maxOutputTokens"].as_f64();
    if cur.map(|c| c < target as f64).unwrap_or(true) {
        gc["maxOutputTokens"] = json!(target);
    }
}

fn gemini_budget_floor(b: f64) -> i64 {
    if b == -1.0 || !b.is_finite() {
        32768
    } else if b <= 1024.0 {
        8192
    } else if b <= 8192.0 {
        16384
    } else if b <= 24576.0 {
        32768
    } else {
        65535
    }
}

fn gemini_level_floor(level: &str) -> i64 {
    match level {
        "minimal" => 4096,
        "low" => 8192,
        "medium" => 16384,
        _ => 65535,
    }
}

fn strip_all(body: &mut Value) {
    for k in ["thinking", "reasoning_effort", "reasoning", "thinkingConfig", "enable_thinking", "thinking_budget", "output_config"] {
        del(body, k);
    }
    if body["generationConfig"].is_object() {
        del(&mut body["generationConfig"], "thinkingConfig");
    }
    if body["request"]["generationConfig"].is_object() {
        del(&mut body["request"]["generationConfig"], "thinkingConfig");
    }
    if body["params"].is_object() {
        del(&mut body["params"], "reasoning_effort");
        del(&mut body["params"], "thinking");
    }
}

fn jb(b: f64) -> Value {
    crate::jsv::jnum(b)
}

fn apply_format(fmt: &str, body: &mut Value, cfg: &Mode, caps: &Caps, supported: &Option<Vec<String>>, display: Option<&str>) {
    let none = *cfg == Mode::None;
    let can_disable = caps.can_disable();
    let eff = if none && !can_disable { Mode::Level("minimal".into()) } else { cfg.clone() };
    let with_display = |mut v: Value| {
        if let Some(d) = display {
            v["display"] = json!(d);
        }
        v
    };
    let has = |l: &str| supported.as_ref().map(|s| s.iter().any(|x| x == l)).unwrap_or(false);
    match fmt {
        "openai" => {
            if none && can_disable {
                body["reasoning_effort"] = json!("none");
                return;
            }
            if let Some(l) = to_level(&eff) {
                body["reasoning_effort"] = json!(normalize_openai_level(l, supported));
            }
        }
        "claude-adaptive" => {
            if none && can_disable {
                body["thinking"] = json!({"type": "disabled"});
                return;
            }
            if can_disable {
                body["thinking"] = with_display(json!({"type": "adaptive"}));
            } else {
                del(body, "thinking");
            }
            let level = to_level(&eff);
            let effort = match level.as_deref() {
                Some("auto") => "high".to_string(),
                Some("xhigh") if !has("xhigh") => "high".to_string(),
                Some(l) => l.to_string(),
                None => return,
            };
            body["output_config"] = json!({"effort": effort});
        }
        "claude-budget" => {
            if none && can_disable {
                body["thinking"] = json!({"type": "disabled"});
                return;
            }
            let budget = to_budget(&eff, caps.get("thinkingRange"));
            body["thinking"] = if budget == Some(-1.0) {
                with_display(json!({"type": "enabled"}))
            } else {
                let b = budget.filter(|b| *b != 0.0).unwrap_or(8192.0);
                with_display(json!({"type": "enabled", "budget_tokens": jb(b)}))
            };
        }
        "gemini-level" => {
            let level = if none {
                "minimal".to_string()
            } else {
                let raw = if eff == Mode::Auto { "high".to_string() } else { to_level(&eff).unwrap_or_else(|| "high".into()) };
                effort_to_thinking_level(&raw)
            };
            let include = level != "minimal";
            gemini_gc(body)["thinkingConfig"] = json!({"thinkingLevel": level, "includeThoughts": include});
            ensure_gemini_floor(body, gemini_level_floor(&level), caps);
        }
        "gemini-budget" => {
            if none && can_disable {
                gemini_gc(body)["thinkingConfig"] = json!({"thinkingBudget": 0, "includeThoughts": false});
                return;
            }
            let budget = to_budget(&eff, caps.get("thinkingRange")).unwrap_or(-1.0);
            gemini_gc(body)["thinkingConfig"] = json!({"thinkingBudget": jb(budget), "includeThoughts": true});
            ensure_gemini_floor(body, gemini_budget_floor(budget), caps);
        }
        "zai" => {
            if none && can_disable {
                body["enable_thinking"] = json!(false);
                del(body, "thinking");
                return;
            }
            body["thinking"] = json!({"type": "enabled"});
            if caps.flag("thinkingEffortSupported") {
                let l = to_level(&eff);
                body["reasoning_effort"] = json!(match l.as_deref() {
                    Some("low") | Some("minimal") => "low",
                    Some("high") | Some("medium") => "high",
                    _ => "max",
                });
            }
        }
        "qwen" => {
            if none && can_disable {
                body["enable_thinking"] = json!(false);
                return;
            }
            body["enable_thinking"] = json!(true);
            if let Some(b) = to_budget(&eff, caps.get("thinkingRange")).filter(|b| b.is_finite() && *b > 0.0) {
                body["thinking_budget"] = jb(b);
            }
        }
        "deepseek" => {
            if none && can_disable {
                body["thinking"] = json!({"type": "disabled"});
                return;
            }
            body["thinking"] = json!({"type": "enabled"});
            let l = to_level(&eff);
            let want = if matches!(l.as_deref(), Some("xhigh") | Some("max")) { "max" } else { "high" };
            let v = if want == "max" && supported.is_some() && !has("max") { "high" } else { want };
            body["reasoning_effort"] = json!(v);
        }
        "kimi" => {
            if none && can_disable {
                body["thinking"] = json!({"type": "disabled"});
                return;
            }
            let e = match to_level(&eff).as_deref() {
                Some("auto") => Some("high"),
                Some("minimal") => Some("low"),
                Some("xhigh") => Some("max"),
                Some(l @ ("low" | "medium" | "high" | "max")) => Some(match l {
                    "low" => "low",
                    "medium" => "medium",
                    "high" => "high",
                    _ => "max",
                }),
                _ => None,
            };
            if let Some(e) = e {
                body["reasoning_effort"] = json!(e);
            }
        }
        "minimax" => {
            body["thinking"] = json!({"type": if none && can_disable { "disabled" } else { "adaptive" }});
        }
        "hunyuan" => {
            if none && can_disable {
                body["thinking"] = json!({"type": "disabled"});
                return;
            }
            let b = to_budget(&eff, caps.get("thinkingRange"));
            body["thinking"] = if b == Some(-1.0) {
                json!({"type": "enabled"})
            } else {
                json!({"type": "enabled", "budget_tokens": jb(b.filter(|x| *x != 0.0).unwrap_or(8192.0))})
            };
        }
        "step" => {
            if none && can_disable {
                return;
            }
            if let Some(l) = to_level(&eff) {
                body["reasoning_effort"] = json!(if l == "xhigh" || l == "max" { "high".to_string() } else { l });
            }
        }
        "tokenrouter" => {
            if none || eff == Mode::Auto {
                return;
            }
            if let Some(l) = to_level(&eff) {
                body["reasoning_effort"] = json!(l);
            }
        }
        "commandcode" => {
            if !body["params"].is_object() {
                body["params"] = json!({});
            }
            if none && can_disable {
                del(&mut body["params"], "reasoning_effort");
                return;
            }
            if let Some(l) = to_level(&eff) {
                body["params"]["reasoning_effort"] = json!(l);
            }
        }
        _ => {}
    }
}

/// applyThinking(targetFormat, model, body, provider, intent)
pub fn apply_thinking(target: &str, model: &str, body: &mut Value, provider: Option<&str>, intent: Option<&Intent>) {
    if !body.is_object() {
        return;
    }
    let (clean, ov) = parse_suffix(model);
    let cfg = ov.or_else(|| intent.cloned()).or_else(|| extract_thinking(body));
    let caps = caps_for(provider, &clean);
    if !caps.reasoning() {
        strip_all(body);
        return;
    }
    let Some(cfg) = cfg else { return };
    let fmt = resolve_format(target, &clean, provider);
    let supported = thinking_levels(provider, &clean);
    let display = body["thinking"]["display"].as_str().map(str::to_owned).or_else(|| intent.and_then(|i| i.display.clone()));
    strip_all(body);
    apply_format(&fmt, body, &cfg.mode, &caps, &supported, display.as_deref());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_to_claude_budget() {
        let mut b = json!({"reasoning_effort": "high"});
        apply_thinking("claude", "claude-sonnet-4-5", &mut b, Some("anthropic"), None);
        assert_eq!(b["thinking"]["budget_tokens"], 24576);
        assert!(b.get("reasoning_effort").is_none());
    }

    #[test]
    fn suffix_override() {
        let (m, ov) = parse_suffix("gemini-2.5-pro(8192)");
        assert_eq!(m, "gemini-2.5-pro");
        assert_eq!(ov.unwrap().mode, Mode::Budget(8192.0));
        let mut b = json!({});
        apply_thinking("gemini", "gemini-2.5-pro(low)", &mut b, Some("gemini"), None);
        assert_eq!(b["generationConfig"]["thinkingConfig"]["thinkingBudget"], 1024);
    }

    #[test]
    fn non_reasoning_model_strips() {
        let mut b = json!({"reasoning_effort": "high"});
        apply_thinking("openai", "llama-3.3-70b-versatile", &mut b, Some("groq"), None);
        assert!(b.get("reasoning_effort").is_none());
    }
}
