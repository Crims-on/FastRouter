//! Kiro (AWS CodeWhisperer / Amazon Q) — request translators, AWS EventStream
//! decoding, integrity gate and executor. Port of executors/kiro.js,
//! config/kiroConstants.js, translator/concerns/kiroConversation.js,
//! utils/kiroSessionReplay.js and the kiro request/response translators.

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::exec::{ByteStream, ExecArgs, ExecResult, Executor, Headers, Upstream, base_execute};
use crate::jsv::{js_string, now_ms, truthy};
use crate::translate::ReqCtx;
use crate::translate::concerns::{build_chunk, fallback_tool_call_id, parse_data_uri, reasoning_delta, to_openai_finish, to_openai_usage};
use crate::translate::thinking::{Intent, Mode, extract_thinking, level_to_budget, parse_suffix};

// ===========================================================================
// constants (config/kiroConstants.js)
// ===========================================================================

pub const AGENTIC_SUFFIX: &str = "-agentic";
pub const THINKING_SUFFIX: &str = "-thinking";
const TOOL_NAME_MAX: usize = 64;
const TOOL_DESC_MAX: usize = 10237;
const TOOL_ID_MAX: usize = 64;
pub const CODEWHISPERER_TARGET: &str = "AmazonCodeWhispererStreamingService.GenerateAssistantResponse";
const ENDPOINT_FALLBACK: [u16; 3] = [401, 403, 404];
pub const THINKING_BUDGET_DEFAULT: i64 = 16000;
const DEFAULT_ARN_BUILDER: &str = "arn:aws:codewhisperer:us-east-1:638616132270:profile/AAAACCCCXXXX";
const DEFAULT_ARN_SOCIAL: &str = "arn:aws:codewhisperer:us-east-1:699475941385:profile/EHGA3GRVQMUK";
const TOOL_RESULTS_PLACEHOLDER: &str = "Tool results provided.";
const EMPTY_USER_PLACEHOLDER: &str = "continue";

pub fn default_profile_arn(auth_method: &str) -> &'static str {
    if auth_method == "google" || auth_method == "github" { DEFAULT_ARN_SOCIAL } else { DEFAULT_ARN_BUILDER }
}

pub struct KiroModel {
    pub upstream: String,
    pub agentic: bool,
    pub thinking: bool,
}

pub fn resolve_kiro_model(model: &str) -> KiroModel {
    let mut up = model.to_string();
    let mut agentic = false;
    let mut thinking = false;
    if let Some(s) = up.strip_suffix(AGENTIC_SUFFIX) {
        agentic = true;
        up = s.to_string();
    }
    if let Some(s) = up.strip_suffix(THINKING_SUFFIX) {
        thinking = true;
        up = s.to_string();
    }
    KiroModel { upstream: up, agentic, thinking }
}

fn apply_thinking_override(body: &Value, ov: Option<&Intent>) -> Value {
    let Some(ov) = ov else { return body.clone() };
    let mut next = body.clone();
    match &ov.mode {
        Mode::Budget(b) => {
            for k in ["output_config", "reasoning_effort", "reasoning"] {
                crate::jsv::del(&mut next, k);
            }
            next["thinking"] = json!({"type": "enabled", "budget_tokens": crate::jsv::jnum(*b)});
        }
        m => {
            let e = match m {
                Mode::Level(l) => l.clone(),
                Mode::None => "none".into(),
                Mode::Auto => "auto".into(),
                Mode::Budget(_) => unreachable!(),
            };
            if !next["output_config"].is_object() {
                next["output_config"] = json!({});
            }
            next["output_config"]["effort"] = json!(e);
        }
    }
    next
}

fn header_ci<'a>(headers: &'a Value, name: &str) -> Option<&'a str> {
    headers.as_object()?.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).and_then(|(_, v)| v.as_str())
}

fn contains_tag(text: &str) -> bool {
    text.contains("<thinking_mode>") && (text.contains("<thinking_mode>enabled</thinking_mode>") || text.contains("<thinking_mode>interleaved</thinking_mode>"))
}

fn contains_thinking_mode_tag(body: &Value) -> bool {
    for m in body["messages"].as_array().into_iter().flatten() {
        if m["role"] != "system" && m["role"] != "user" {
            continue;
        }
        match &m["content"] {
            Value::String(s) if contains_tag(s) => return true,
            Value::Array(a) if a.iter().any(|p| p["text"].as_str().map(contains_tag).unwrap_or(false)) => return true,
            _ => {}
        }
    }
    body["system"].as_str().map(contains_tag).unwrap_or(false)
}

/// resolveKiroThinkingBudget → None when thinking is disabled.
pub fn resolve_thinking_budget(body: &Value, headers: &Value, model: &str) -> Option<i64> {
    if let Some(cfg) = extract_thinking(body) {
        return match cfg.mode {
            Mode::None => None,
            Mode::Level(l) if l == "disabled" => None,
            Mode::Budget(b) => Some(b as i64),
            Mode::Level(l) => Some(level_to_budget(&l).map(|b| b as i64).unwrap_or(THINKING_BUDGET_DEFAULT)),
            Mode::Auto => Some(THINKING_BUDGET_DEFAULT),
        };
    }
    if let Some(b) = header_ci(headers, "anthropic-beta") {
        if b.to_lowercase().contains("interleaved-thinking") {
            return Some(THINKING_BUDGET_DEFAULT);
        }
    }
    if contains_thinking_mode_tag(body) {
        return Some(THINKING_BUDGET_DEFAULT);
    }
    let m = model.to_lowercase();
    if !m.is_empty() && (m.contains("thinking") || m.contains("-reason")) {
        return Some(THINKING_BUDGET_DEFAULT);
    }
    None
}

fn effort_of(body: &Value) -> Option<String> {
    [&body["output_config"]["effort"], &body["reasoning_effort"], if body["reasoning"].is_object() { &body["reasoning"]["effort"] } else { &Value::Null }]
        .into_iter()
        .find(|v| !v.is_null())
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase())
}

fn lacks_xhigh(model: &str) -> bool {
    match crate::caps::parse_claude_version(model) {
        None => true,
        Some((maj, min)) => maj == 4 && min.map(|m| m <= 6).unwrap_or(false),
    }
}

pub fn extract_effort_level(body: &Value, model: &str) -> Option<String> {
    let e = effort_of(body)?;
    match e.as_str() {
        "none" | "off" | "disabled" => None,
        "xhigh" => Some(if lacks_xhigh(model) { "high".into() } else { "xhigh".into() }),
        "max" | "low" | "medium" | "high" => Some(e),
        _ => None,
    }
}

fn extract_gpt_effort(body: &Value) -> Option<String> {
    let e = effort_of(body)?;
    match e.as_str() {
        "max" => Some("xhigh".into()),
        "low" | "medium" | "high" | "xhigh" => Some(e),
        _ => None,
    }
}

pub fn additional_fields_for_model(body: &Value, model: &str) -> Option<Value> {
    let path = crate::caps::resolve_kiro_effort_path(model)?;
    if path == "reasoning" {
        return extract_gpt_effort(body).map(|e| json!({"reasoning": {"effort": e}}));
    }
    extract_effort_level(body, model).map(|e| json!({"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": e}}))
}

fn uses_native_gpt_effort(body: &Value, model: &str) -> bool {
    crate::caps::resolve_kiro_effort_path(model) == Some("reasoning") && extract_gpt_effort(body).is_some()
}

pub fn thinking_prefix(budget: i64) -> String {
    let b = if budget == 0 { THINKING_BUDGET_DEFAULT } else { budget }.clamp(1, 32000);
    format!("<thinking_mode>enabled</thinking_mode>\n<max_thinking_length>{b}</max_thinking_length>")
}

pub fn agentic_prompt() -> &'static str {
    crate::consts::s("KIRO_AGENTIC_SYSTEM_PROMPT")
}

// ===========================================================================
// conversation canonicalization (translator/concerns/kiroConversation.js)
// ===========================================================================

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        o => o.to_string(),
    }
}

fn append_text(target: &mut Value, extra: &str) {
    if extra.is_empty() {
        return;
    }
    let cur = text_of(&target["content"]);
    target["content"] = json!(if cur.is_empty() { extra.to_string() } else { format!("{cur}\n\n{extra}") });
}

fn trim_cp(v: &str, n: usize) -> String {
    v.chars().take(n).collect()
}

fn unique_name(raw: &str, idx: usize, used: &mut HashSet<String>) -> String {
    static BAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-zA-Z0-9_-]").unwrap());
    let cleaned = BAD.replace_all(raw.trim(), "_").trim_matches('_').to_string();
    let base = trim_cp(if cleaned.is_empty() { format!("tool_{}", idx + 1) } else { cleaned }.as_str(), TOOL_NAME_MAX);
    let mut cand = base.clone();
    let mut sfx = 2;
    while used.contains(&cand) {
        let tail = format!("_{sfx}");
        sfx += 1;
        cand = format!("{}{tail}", &base[..base.len().min(TOOL_NAME_MAX - tail.len())]);
    }
    used.insert(cand.clone());
    cand
}

fn clean_schema(v: &Value) -> Value {
    match v {
        Value::Array(a) => Value::Array(a.iter().map(clean_schema).collect()),
        Value::Object(o) => {
            let mut out = Map::new();
            for (k, c) in o {
                if k == "additionalProperties" {
                    continue;
                }
                if k == "required" && c.as_array().map(|a| a.is_empty()).unwrap_or(false) {
                    continue;
                }
                out.insert(k.clone(), clean_schema(c));
            }
            Value::Object(out)
        }
        o => o.clone(),
    }
}

fn normalize_root_schema(s: &Value) -> Value {
    let mut c = clean_schema(if s.is_object() { s } else { &Value::Null });
    if !c.is_object() {
        c = json!({});
    }
    c["type"] = json!("object");
    if !c["properties"].is_object() {
        c["properties"] = json!({});
    }
    if let Some(req) = c["required"].as_array().cloned() {
        let mut seen = HashSet::new();
        let r: Vec<Value> = req.into_iter().filter(|n| n.as_str().map(|s| c["properties"].get(s).is_some() && seen.insert(s.to_string())).unwrap_or(false)).collect();
        if r.is_empty() {
            crate::jsv::del(&mut c, "required");
        } else {
            c["required"] = Value::Array(r);
        }
    }
    c
}

/// normalizeKiroToolSpecs → (specs, nameMap original→sanitized)
pub fn normalize_tool_specs(tools: &Value) -> (Vec<Value>, Vec<(String, String)>) {
    let mut specs = vec![];
    let mut map: Vec<(String, String)> = vec![];
    let mut used = HashSet::new();
    for (i, t) in tools.as_array().into_iter().flatten().enumerate() {
        if !t.is_object() {
            continue;
        }
        let raw = if !t["function"]["name"].is_null() { &t["function"]["name"] } else { &t["name"] };
        let Some(raw) = raw.as_str().filter(|s| !s.trim().is_empty()) else { continue };
        if map.iter().any(|(o, _)| o == raw) {
            continue;
        }
        let name = unique_name(raw, i, &mut used);
        map.push((raw.to_string(), name.clone()));
        let rd = [&t["function"]["description"], &t["description"]].into_iter().find(|v| !v.is_null()).cloned().unwrap_or(json!(format!("Tool: {raw}")));
        let rd = if truthy(&rd) { js_string(&rd) } else { format!("Tool: {raw}") };
        let schema = [&t["function"]["parameters"], &t["parameters"], &t["input_schema"]].into_iter().find(|v| !v.is_null()).cloned().unwrap_or(json!({}));
        specs.push(json!({"toolSpecification": {"name": name, "description": trim_cp(&rd, TOOL_DESC_MAX), "inputSchema": {"json": normalize_root_schema(&schema)}}}));
    }
    (specs, map)
}

fn tool_call_text(name: &Value, input: &Value) -> String {
    let n = if truthy(name) { js_string(name) } else { "unknown".into() };
    let empty = json!({});
    format!("[Tool call: {n}({})]", text_of(if truthy(input) { input } else { &empty }))
}

fn tool_result_text(r: &Value) -> String {
    let content = if let Some(a) = r["content"].as_array() {
        a.iter().map(|p| text_of(if !p["text"].is_null() { &p["text"] } else { p })).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n")
    } else {
        text_of(&r["content"])
    };
    format!("[Tool result{}: {content}]", if r["status"] == "error" { " (error)" } else { "" })
}

fn merge_user(target: &mut Value, src: &Value) {
    append_text(target, &text_of(&src["content"]));
    if let Some(imgs) = src["images"].as_array().filter(|a| !a.is_empty()) {
        let mut t = target["images"].as_array().cloned().unwrap_or_default();
        t.extend(imgs.iter().cloned());
        target["images"] = Value::Array(t);
    }
    if let Some(res) = src["userInputMessageContext"]["toolResults"].as_array().filter(|a| !a.is_empty()) {
        if !target["userInputMessageContext"].is_object() {
            target["userInputMessageContext"] = json!({});
        }
        let mut t = target["userInputMessageContext"]["toolResults"].as_array().cloned().unwrap_or_default();
        t.extend(res.iter().cloned());
        target["userInputMessageContext"]["toolResults"] = Value::Array(t);
    }
}

fn merge_assistant(target: &mut Value, src: &Value) {
    append_text(target, &text_of(&src["content"]));
    if let Some(tu) = src["toolUses"].as_array().filter(|a| !a.is_empty()) {
        let mut t = target["toolUses"].as_array().cloned().unwrap_or_default();
        t.extend(tu.iter().cloned());
        target["toolUses"] = Value::Array(t);
    }
}

fn normalize_turns(history: &[Value], current: Option<&Value>, model_id: &str) -> Vec<Value> {
    let mut raw: Vec<Value> = history.to_vec();
    if let Some(c) = current {
        raw.push(c.clone());
    }
    let mut turns: Vec<Value> = vec![];
    for r in raw {
        let u = truthy(&r["userInputMessage"]);
        let a = truthy(&r["assistantResponseMessage"]);
        if u == a {
            continue;
        }
        let turn = if u { json!({"userInputMessage": r["userInputMessage"]}) } else { json!({"assistantResponseMessage": r["assistantResponseMessage"]}) };
        match turns.last_mut() {
            Some(prev) if u && truthy(&prev["userInputMessage"]) => merge_user(&mut prev["userInputMessage"], &turn["userInputMessage"]),
            Some(prev) if a && truthy(&prev["assistantResponseMessage"]) => merge_assistant(&mut prev["assistantResponseMessage"], &turn["assistantResponseMessage"]),
            _ => turns.push(turn),
        }
    }
    if turns.first().map(|t| truthy(&t["assistantResponseMessage"])).unwrap_or(false) {
        turns.insert(0, json!({"userInputMessage": {"content": "continue", "modelId": model_id}}));
    }
    if turns.is_empty() || truthy(&turns.last().unwrap()["assistantResponseMessage"]) {
        turns.push(json!({"userInputMessage": {"content": "continue", "modelId": model_id}}));
    }
    for t in turns.iter_mut() {
        if truthy(&t["userInputMessage"]) {
            let u = &mut t["userInputMessage"];
            let has_results = u["userInputMessageContext"]["toolResults"].as_array().map(|a| !a.is_empty()).unwrap_or(false);
            let c = text_of(&u["content"]).trim().to_string();
            u["content"] = json!(if c.is_empty() { if has_results { TOOL_RESULTS_PLACEHOLDER } else { EMPTY_USER_PLACEHOLDER }.to_string() } else { c });
            if !truthy(&u["modelId"]) {
                u["modelId"] = json!(model_id);
            }
            if u["userInputMessageContext"].get("tools").is_some() {
                crate::jsv::del(&mut u["userInputMessageContext"], "tools");
            }
        } else {
            let a = &mut t["assistantResponseMessage"];
            let c = text_of(&a["content"]).trim().to_string();
            a["content"] = json!(if c.is_empty() { "...".to_string() } else { c });
        }
    }
    turns
}

fn reserve_tool_id(v: &str, turn: usize, call: usize, name: &str, used: &mut HashSet<String>) -> String {
    static BAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-zA-Z0-9_-]").unwrap());
    let s = BAD.replace_all(v, "").into_owned();
    let generated = format!("call_msg{turn}_tc{call}_{}", if name.is_empty() { "tool" } else { name });
    let base = trim_cp(if s.is_empty() { &generated } else { &s }, TOOL_ID_MAX);
    let mut cand = base.clone();
    let mut sfx = 2;
    while used.contains(&cand) {
        let tail = format!("_{sfx}");
        sfx += 1;
        cand = format!("{}{tail}", base.chars().take(TOOL_ID_MAX - tail.len()).collect::<String>());
    }
    used.insert(cand.clone());
    cand
}

fn normalize_tool_input(input: &Value) -> Option<Value> {
    match input {
        Value::Object(_) => Some(input.clone()),
        Value::String(s) => serde_json::from_str::<Value>(s).ok().filter(|v| v.is_object()),
        Value::Null => Some(json!({})),
        _ => None,
    }
}

fn normalize_tool_result(r: &Value) -> Value {
    let content: Vec<Value> = if let Some(a) = r["content"].as_array() {
        a.iter().map(|p| json!({"text": text_of(if !p["text"].is_null() { &p["text"] } else { p })})).collect()
    } else {
        vec![json!({"text": text_of(&r["content"])})]
    };
    json!({
        "toolUseId": r["toolUseId"].as_str().unwrap_or(""),
        "status": if r["status"] == "error" { "error" } else { "success" },
        "content": if content.is_empty() { vec![json!({"text": ""})] } else { content },
    })
}

fn flatten_results(user: &mut Value, results: &[Value]) {
    for r in results {
        append_text(user, &tool_result_text(r));
    }
}

fn clean_user_context(user: &mut Value) {
    let Some(ctx) = user.get_mut("userInputMessageContext").and_then(|c| c.as_object_mut()) else { return };
    if ctx.get("toolResults").and_then(|v| v.as_array()).map(|a| a.is_empty()).unwrap_or(true) {
        ctx.shift_remove("toolResults");
    }
    if ctx.get("tools").and_then(|v| v.as_array()).map(|a| a.is_empty()).unwrap_or(true) {
        ctx.shift_remove("tools");
    }
    if ctx.is_empty() {
        crate::jsv::del(user, "userInputMessageContext");
    }
}

#[derive(Default)]
pub struct Repairs {
    pub missing_results: usize,
    pub orphan_results: usize,
    pub invalid_tool_uses: usize,
}

fn reconcile_pair(assistant: &mut Value, user: &mut Value, turn: usize, name_map: &[(String, String)], spec_names: &HashSet<String>, used: &mut HashSet<String>, rep: &mut Repairs) {
    let calls = assistant["toolUses"].as_array().cloned().unwrap_or_default();
    let results: Vec<Value> = user["userInputMessageContext"]["toolResults"].as_array().map(|a| a.iter().map(normalize_tool_result).collect()).unwrap_or_default();
    if calls.is_empty() {
        if !results.is_empty() {
            flatten_results(user, &results);
            rep.orphan_results += results.len();
        }
        if user["userInputMessageContext"].is_object() {
            crate::jsv::del(&mut user["userInputMessageContext"], "toolResults");
        }
        clean_user_context(user);
        return;
    }
    struct Rec {
        call: Value,
        idx: usize,
        key: String,
        name: Value,
        input: Option<Value>,
        result: Option<Value>,
    }
    let mut recs: Vec<Rec> = calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let key = c["toolUseId"].as_str().unwrap_or("").to_string();
            let name = c["name"].as_str().and_then(|n| name_map.iter().find(|(o, _)| o == n).map(|(_, s)| json!(s))).unwrap_or_else(|| c["name"].clone());
            Rec { call: c.clone(), idx: i, key, name, input: normalize_tool_input(&c["input"]), result: None }
        })
        .collect();
    let mut orphans = vec![];
    for r in results {
        let id = r["toolUseId"].as_str().unwrap_or("").to_string();
        if let Some(rec) = recs.iter_mut().find(|x| x.key == id && x.result.is_none()) {
            rec.result = Some(r);
        } else {
            orphans.push(r);
        }
    }
    let mut kept_calls = vec![];
    let mut kept_results = vec![];
    for rec in recs {
        let has_spec = rec.name.as_str().map(|n| spec_names.contains(n)).unwrap_or(false);
        let valid = rec.result.is_some() && has_spec && rec.input.is_some();
        if !valid {
            append_text(assistant, &tool_call_text(&rec.name, &rec.call["input"]));
            if rec.result.is_none() {
                rep.missing_results += 1;
            }
            if !(has_spec && rec.input.is_some()) {
                rep.invalid_tool_uses += 1;
            }
            if let Some(r) = rec.result {
                flatten_results(user, &[r]);
                rep.orphan_results += 1;
            }
            continue;
        }
        let name = rec.name.as_str().unwrap_or("").to_string();
        let id = reserve_tool_id(&rec.key, turn, rec.idx, &name, used);
        kept_calls.push(json!({"toolUseId": id, "name": name, "input": rec.input.unwrap()}));
        let mut r = rec.result.unwrap();
        r["toolUseId"] = json!(id);
        kept_results.push(r);
    }
    if !orphans.is_empty() {
        flatten_results(user, &orphans);
        rep.orphan_results += orphans.len();
    }
    if kept_calls.is_empty() {
        crate::jsv::del(assistant, "toolUses");
    } else {
        assistant["toolUses"] = Value::Array(kept_calls);
    }
    if !user["userInputMessageContext"].is_object() {
        user["userInputMessageContext"] = json!({});
    }
    if kept_results.is_empty() {
        crate::jsv::del(&mut user["userInputMessageContext"], "toolResults");
    } else {
        user["userInputMessageContext"]["toolResults"] = Value::Array(kept_results);
    }
    clean_user_context(user);
}

pub fn validate_conversation(history: &[Value], current: &Value, specs: &[Value]) -> Vec<String> {
    let mut errors = vec![];
    let mut turns: Vec<&Value> = history.iter().collect();
    turns.push(current);
    let spec_names: HashSet<&str> = specs.iter().filter_map(|s| s["toolSpecification"]["name"].as_str()).collect();
    let mut used = HashSet::new();
    for i in 0..turns.len() {
        let expected_user = i % 2 == 0;
        let is_user = truthy(&turns[i]["userInputMessage"]);
        if is_user != expected_user {
            errors.push(format!("role:{i}"));
        }
        if !is_user {
            let calls = turns[i]["assistantResponseMessage"]["toolUses"].as_array().cloned().unwrap_or_default();
            let results = turns.get(i + 1).and_then(|t| t["userInputMessage"]["userInputMessageContext"]["toolResults"].as_array().cloned()).unwrap_or_default();
            let rids: Vec<&Value> = results.iter().map(|r| &r["toolUseId"]).collect();
            if calls.len() != results.len() || calls.iter().any(|c| !rids.contains(&&c["toolUseId"])) {
                errors.push(format!("pair:{i}"));
            }
            for c in &calls {
                let id = js_string(&c["toolUseId"]);
                if !truthy(&c["toolUseId"]) || !used.insert(id) {
                    errors.push(format!("id:{i}"));
                }
                if !c["name"].as_str().map(|n| spec_names.contains(n)).unwrap_or(false) {
                    errors.push(format!("spec:{i}"));
                }
            }
        } else if i == 0 && turns[0]["userInputMessage"]["userInputMessageContext"]["toolResults"].as_array().map(|a| !a.is_empty()).unwrap_or(false) {
            errors.push("orphan:0".into());
        }
    }
    if !truthy(&current["userInputMessage"]["content"]) {
        errors.push("current".into());
    }
    errors
}

fn flatten_all(turns: &mut [Value], rep: &mut Repairs) {
    for t in turns.iter_mut() {
        if let Some(calls) = t["assistantResponseMessage"]["toolUses"].as_array().cloned().filter(|a| !a.is_empty()) {
            for c in &calls {
                append_text(&mut t["assistantResponseMessage"], &tool_call_text(&c["name"], &c["input"]));
            }
            rep.invalid_tool_uses += calls.len();
            crate::jsv::del(&mut t["assistantResponseMessage"], "toolUses");
        }
        if let Some(res) = t["userInputMessage"]["userInputMessageContext"]["toolResults"].as_array().cloned().filter(|a| !a.is_empty()) {
            flatten_results(&mut t["userInputMessage"], &res);
            rep.orphan_results += res.len();
            crate::jsv::del(&mut t["userInputMessage"]["userInputMessageContext"], "toolResults");
            clean_user_context(&mut t["userInputMessage"]);
        }
    }
}

pub struct Canonical {
    pub history: Vec<Value>,
    pub current: Value,
    pub valid: bool,
    pub errors: Vec<String>,
}

pub fn canonicalize(history: &[Value], current: Option<&Value>, model_id: &str, specs: &[Value], name_map: &[(String, String)]) -> Canonical {
    let mut turns = normalize_turns(history, current, model_id);
    let mut rep = Repairs::default();
    let spec_names: HashSet<String> = specs.iter().filter_map(|s| s["toolSpecification"]["name"].as_str().map(str::to_owned)).collect();
    let mut used = HashSet::new();
    let mut i = 0;
    while i < turns.len() {
        if i == 0 {
            if let Some(lead) = turns[0]["userInputMessage"]["userInputMessageContext"]["toolResults"].as_array().cloned().filter(|a| !a.is_empty()) {
                flatten_results(&mut turns[0]["userInputMessage"], &lead);
                rep.orphan_results += lead.len();
                crate::jsv::del(&mut turns[0]["userInputMessage"]["userInputMessageContext"], "toolResults");
                clean_user_context(&mut turns[0]["userInputMessage"]);
            }
        }
        if i + 2 < turns.len() + 0 && truthy(&turns[i + 1]["assistantResponseMessage"]) && truthy(&turns[i + 2]["userInputMessage"]) {
            let (left, right) = turns.split_at_mut(i + 2);
            reconcile_pair(&mut left[i + 1]["assistantResponseMessage"], &mut right[0]["userInputMessage"], i + 1, name_map, &spec_names, &mut used, &mut rep);
        }
        i += 2;
    }
    let last = turns.len() - 1;
    if !turns[last]["userInputMessage"]["userInputMessageContext"].is_object() {
        turns[last]["userInputMessage"]["userInputMessageContext"] = json!({});
    }
    if !specs.is_empty() {
        turns[last]["userInputMessage"]["userInputMessageContext"]["tools"] = Value::Array(specs.to_vec());
    }
    clean_user_context(&mut turns[last]["userInputMessage"]);
    let mut errors = validate_conversation(&turns[..last], &turns[last], specs);
    if !errors.is_empty() {
        flatten_all(&mut turns, &mut rep);
        errors = validate_conversation(&turns[..last], &turns[last], specs);
    }
    let current = turns.pop().unwrap();
    Canonical { history: turns, current, valid: errors.is_empty(), errors }
}

// ===========================================================================
// session replay (utils/kiroSessionReplay.js)
// ===========================================================================

struct SessionStart {
    start: Value,
    model_id: String,
    system_prompt: String,
    last_used: i64,
}

static SESSION_STARTS: LazyLock<Mutex<(HashMap<String, SessionStart>, Vec<String>)>> = LazyLock::new(Default::default);

fn ensure_model_id(m: &mut Value, model_id: &str) {
    if truthy(&m["userInputMessage"]) && !truthy(&m["userInputMessage"]["modelId"]) && !model_id.is_empty() {
        m["userInputMessage"]["modelId"] = json!(model_id);
    }
}

fn prefix_user(m: &Value, prefix: &str, model_id: &str) -> Value {
    let mut out = if m.is_null() { json!({"userInputMessage": {"content": ""}}) } else { m.clone() };
    if !truthy(&out["userInputMessage"]) {
        out["userInputMessage"] = json!({"content": ""});
    }
    ensure_model_id(&mut out, model_id);
    if !prefix.is_empty() {
        let c = out["userInputMessage"]["content"].as_str().unwrap_or("").to_string();
        out["userInputMessage"]["content"] = json!(if c.is_empty() { prefix.to_string() } else { format!("{prefix}\n\n{c}") });
    }
    out
}

pub struct Replay {
    pub history: Vec<Value>,
    pub current: Value,
}

pub fn apply_session_replay(conversation_id: &str, connection_id: &str, model_id: &str, system_prompt: &str, content_prefix: &str, current_prefix: &str, history: &[Value], current: &Value) -> Replay {
    let key = format!("{connection_id}:{conversation_id}");
    let mut base = history.to_vec();
    let base_current = if current.is_null() { json!({"userInputMessage": {"content": ""}}) } else { current.clone() };
    let first_user = base.iter().position(|m| truthy(&m["userInputMessage"]));
    let has_results = |m: &Value| m["userInputMessage"]["userInputMessageContext"]["toolResults"].as_array().map(|a| !a.is_empty()).unwrap_or(false);
    let can_replace = |base: &[Value], fu: Option<usize>| fu == Some(0) && !has_results(&base[0]);
    let mut store = SESSION_STARTS.lock().unwrap();
    let now = now_ms();
    store.0.retain(|_, v| now - v.last_used <= 2 * 3_600_000);
    if !conversation_id.is_empty() {
        if let Some(e) = store.0.get_mut(&key) {
            if e.model_id == model_id && e.system_prompt == system_prompt {
                e.last_used = now;
                let mut start = e.start.clone();
                ensure_model_id(&mut start, model_id);
                if can_replace(&base, first_user) {
                    base[0] = start;
                } else {
                    base.insert(0, start);
                    if base.len() == 1 {
                        base.push(json!({"assistantResponseMessage": {"content": "..."}}));
                    }
                }
                for m in base.iter_mut() {
                    ensure_model_id(m, model_id);
                }
                return Replay { history: base, current: prefix_user(&base_current, current_prefix, model_id) };
            }
        }
    }
    let start;
    let mut next_current = base_current.clone();
    ensure_model_id(&mut next_current, model_id);
    if can_replace(&base, first_user) {
        start = prefix_user(&base[0], content_prefix, model_id);
        base[0] = start.clone();
        next_current = prefix_user(&base_current, current_prefix, model_id);
    } else if first_user.is_some() {
        start = prefix_user(&json!({"userInputMessage": {"content": "", "modelId": model_id}}), content_prefix, model_id);
        base.insert(0, start.clone());
        next_current = prefix_user(&base_current, current_prefix, model_id);
    } else {
        start = prefix_user(&base_current, content_prefix, model_id);
        next_current = start.clone();
    }
    if !conversation_id.is_empty() {
        if store.0.len() >= 5000 {
            if let Some(old) = store.1.first().cloned() {
                store.0.remove(&old);
                store.1.remove(0);
            }
        }
        if !store.0.contains_key(&key) {
            store.1.push(key.clone());
        }
        store.0.insert(key, SessionStart { start, model_id: model_id.into(), system_prompt: system_prompt.into(), last_used: now });
    }
    for m in base.iter_mut() {
        ensure_model_id(m, model_id);
    }
    Replay { history: base, current: next_current }
}

// ===========================================================================
// request translators
// ===========================================================================

fn image_format(mime: &str) -> String {
    mime.split('/').nth(1).filter(|s| !s.is_empty()).unwrap_or(mime).to_string()
}

struct Builder {
    history: Vec<Value>,
    user: Vec<String>,
    assistant: Vec<String>,
    results: Vec<Value>,
    images: Vec<Value>,
    role: Option<String>,
    model: String,
    user_model_id: String,
}

impl Builder {
    fn flush(&mut self) {
        match self.role.as_deref() {
            Some("user") => {
                let c = self.user.join("\n\n").trim().to_string();
                let c = if c.is_empty() { if self.results.is_empty() { EMPTY_USER_PLACEHOLDER.into() } else { TOOL_RESULTS_PLACEHOLDER.into() } } else { c };
                let mut m = json!({"userInputMessage": {"content": c, "modelId": self.user_model_id}});
                if !self.images.is_empty() {
                    m["userInputMessage"]["images"] = Value::Array(std::mem::take(&mut self.images));
                }
                if !self.results.is_empty() {
                    m["userInputMessage"]["userInputMessageContext"] = json!({"toolResults": std::mem::take(&mut self.results)});
                }
                self.history.push(m);
                self.user.clear();
            }
            Some("assistant") => {
                let c = self.assistant.join("\n\n").trim().to_string();
                self.history.push(json!({"assistantResponseMessage": {"content": if c.is_empty() { "...".to_string() } else { c }}}));
                self.assistant.clear();
            }
            _ => {}
        }
    }

    fn attach_tool_uses(&mut self, uses: Vec<Value>) {
        self.flush();
        if let Some(last) = self.history.last_mut() {
            if truthy(&last["assistantResponseMessage"]) {
                last["assistantResponseMessage"]["toolUses"] = Value::Array(uses);
            }
        }
        self.role = None;
    }

    fn finish(mut self) -> (Vec<Value>, Value) {
        if self.role.is_some() {
            self.flush();
        }
        let mut current = Value::Null;
        if let Some(i) = self.history.iter().rposition(|m| truthy(&m["userInputMessage"])) {
            current = self.history.remove(i);
        }
        for m in self.history.iter_mut() {
            if m["userInputMessage"]["userInputMessageContext"].as_object().map(|o| o.is_empty()).unwrap_or(false) {
                crate::jsv::del(&mut m["userInputMessage"], "userInputMessageContext");
            }
            if truthy(&m["userInputMessage"]) && !truthy(&m["userInputMessage"]["modelId"]) {
                m["userInputMessage"]["modelId"] = json!(self.model);
            }
        }
        let mut merged: Vec<Value> = vec![];
        for cur in self.history {
            if truthy(&cur["userInputMessage"]) && merged.last().map(|p| truthy(&p["userInputMessage"])).unwrap_or(false) {
                let prev = merged.last_mut().unwrap();
                let pc = text_of(&prev["userInputMessage"]["content"]);
                prev["userInputMessage"]["content"] = json!(format!("{pc}\n\n{}", text_of(&cur["userInputMessage"]["content"])));
                let cctx = cur["userInputMessage"]["userInputMessageContext"].clone();
                if truthy(&cctx) {
                    if !truthy(&prev["userInputMessage"]["userInputMessageContext"]) {
                        prev["userInputMessage"]["userInputMessageContext"] = cctx;
                    } else {
                        for k in ["toolResults", "tools"] {
                            if let Some(a) = cctx[k].as_array().filter(|a| !a.is_empty()) {
                                let mut t = prev["userInputMessage"]["userInputMessageContext"][k].as_array().cloned().unwrap_or_default();
                                t.extend(a.iter().cloned());
                                prev["userInputMessage"]["userInputMessageContext"][k] = Value::Array(t);
                            }
                        }
                    }
                }
            } else {
                merged.push(cur);
            }
        }
        if current.is_null() {
            current = json!({"userInputMessage": {"content": "", "modelId": self.model}});
        }
        (merged, current)
    }
}

fn safe_parse_args(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        Value::Null => json!({}),
        o => o.clone(),
    }
}

fn convert_openai_messages(messages: &[Value], model: &str) -> (Vec<Value>, Value) {
    let mut b = Builder { history: vec![], user: vec![], assistant: vec![], results: vec![], images: vec![], role: None, model: model.into(), user_model_id: String::new() };
    for msg in messages {
        let orig_role = msg["role"].as_str().unwrap_or("").to_string();
        let was_system = orig_role == "system";
        let role = if orig_role == "system" || orig_role == "tool" { "user".to_string() } else { orig_role.clone() };
        if b.role.is_some() && b.role.as_deref() != Some(role.as_str()) {
            b.flush();
        }
        b.role = Some(role.clone());
        if role == "user" {
            let mut content = String::new();
            if let Some(s) = msg["content"].as_str() {
                content = s.to_string();
            } else if let Some(a) = msg["content"].as_array() {
                let mut parts = vec![];
                for c in a {
                    if c["type"] == "text" || truthy(&c["text"]) {
                        parts.push(c["text"].as_str().unwrap_or("").to_string());
                    } else if c["type"] == "image_url" {
                        let url = c["image_url"]["url"].as_str().unwrap_or("");
                        if let Some((mime, data)) = parse_data_uri(&json!(url)) {
                            b.images.push(json!({"format": image_format(&mime), "source": {"bytes": data}}));
                        } else if url.starts_with("http://") || url.starts_with("https://") {
                            parts.push(format!("[Image: {url}]"));
                        }
                    } else if c["type"] == "image" && c["source"]["type"] == "base64" && truthy(&c["source"]["data"]) {
                        let mt = c["source"]["media_type"].as_str().filter(|s| !s.is_empty()).unwrap_or("image/jpeg");
                        b.images.push(json!({"format": image_format(mt), "source": {"bytes": c["source"]["data"]}}));
                    }
                }
                content = parts.join("\n");
                for blk in a.iter().filter(|c| c["type"] == "tool_result") {
                    let text = if let Some(arr) = blk["content"].as_array() {
                        arr.iter().map(|c| c["text"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>().join("\n")
                    } else {
                        blk["content"].as_str().unwrap_or("").to_string()
                    };
                    b.results.push(json!({"toolUseId": blk["tool_use_id"], "status": if truthy(&blk["is_error"]) { "error" } else { "success" }, "content": [{"text": text}]}));
                }
            }
            if orig_role == "tool" {
                let tc = msg["content"].as_str().unwrap_or("").to_string();
                b.results.push(json!({"toolUseId": msg["tool_call_id"], "status": if truthy(&msg["is_error"]) || msg["status"] == "error" { "error" } else { "success" }, "content": [{"text": tc}]}));
            } else if !content.is_empty() {
                b.user.push(if was_system { format!("<instructions>\n{content}\n</instructions>") } else { content });
            }
        } else if role == "assistant" {
            let mut text = String::new();
            let mut uses: Vec<Value> = vec![];
            if let Some(a) = msg["content"].as_array() {
                text = a.iter().filter(|c| c["type"] == "text").map(|c| c["text"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>().join("\n").trim().to_string();
                uses = a.iter().filter(|c| c["type"] == "tool_use").cloned().collect();
            } else if let Some(s) = msg["content"].as_str() {
                text = s.trim().to_string();
            }
            if let Some(tc) = msg["tool_calls"].as_array().filter(|a| !a.is_empty()) {
                uses = tc.clone();
            }
            if !text.is_empty() {
                b.assistant.push(text);
            }
            if !uses.is_empty() {
                let mapped = uses
                    .iter()
                    .map(|tc| {
                        let id = if truthy(&tc["id"]) { tc["id"].clone() } else { json!(uuid::Uuid::new_v4().to_string()) };
                        if truthy(&tc["function"]) {
                            json!({"toolUseId": id, "name": tc["function"]["name"], "input": safe_parse_args(&tc["function"]["arguments"])})
                        } else {
                            json!({"toolUseId": id, "name": tc["name"], "input": if truthy(&tc["input"]) { tc["input"].clone() } else { json!({}) }})
                        }
                    })
                    .collect();
                b.attach_tool_uses(mapped);
            }
        }
    }
    b.finish()
}

fn convert_claude_messages(messages: &[Value], model: &str) -> (Vec<Value>, Value) {
    let mut b = Builder { history: vec![], user: vec![], assistant: vec![], results: vec![], images: vec![], role: None, model: model.into(), user_model_id: model.into() };
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("").to_string();
        if b.role.is_some() && b.role.as_deref() != Some(role.as_str()) {
            b.flush();
        }
        b.role = Some(role.clone());
        if role == "user" {
            if let Some(s) = msg["content"].as_str() {
                b.user.push(s.to_string());
            } else if let Some(a) = msg["content"].as_array() {
                for blk in a {
                    match blk["type"].as_str().unwrap_or("") {
                        "text" => b.user.push(blk["text"].as_str().unwrap_or("").to_string()),
                        "image" if blk["source"]["type"] == "base64" => {
                            let mt = blk["source"]["media_type"].as_str().filter(|s| !s.is_empty()).unwrap_or("image/jpeg");
                            b.images.push(json!({"format": image_format(mt), "source": {"bytes": blk["source"]["data"]}}));
                        }
                        "tool_result" => {
                            let rc = match &blk["content"] {
                                Value::String(s) => s.clone(),
                                Value::Array(arr) => {
                                    let mut has_img = false;
                                    for c in arr {
                                        if c["type"] == "image" && c["source"]["type"] == "base64" {
                                            has_img = true;
                                            let it = c["source"]["media_type"].as_str().filter(|s| !s.is_empty()).unwrap_or("image/jpeg");
                                            b.images.push(json!({"format": image_format(it), "source": {"bytes": c["source"]["data"]}}));
                                        }
                                    }
                                    let t = arr.iter().filter(|c| c["type"] == "text").map(|c| c["text"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>().join("\n");
                                    if !t.is_empty() {
                                        t
                                    } else if has_img {
                                        "(image attached)".into()
                                    } else {
                                        blk["content"].to_string()
                                    }
                                }
                                Value::Null => String::new(),
                                o if !truthy(o) => String::new(),
                                o => o.to_string(),
                            };
                            b.results.push(json!({"toolUseId": blk["tool_use_id"], "status": if truthy(&blk["is_error"]) { "error" } else { "success" }, "content": [{"text": rc}]}));
                        }
                        _ => {}
                    }
                }
            }
        } else if role == "assistant" {
            let mut text = String::new();
            let mut uses = vec![];
            if let Some(s) = msg["content"].as_str() {
                text = s.to_string();
            } else if let Some(a) = msg["content"].as_array() {
                for blk in a {
                    if blk["type"] == "text" {
                        text.push_str(blk["text"].as_str().unwrap_or(""));
                    } else if blk["type"] == "tool_use" {
                        uses.push(json!({"toolUseId": blk["id"], "name": blk["name"], "input": if truthy(&blk["input"]) { blk["input"].clone() } else { json!({}) }}));
                    }
                }
            }
            if !text.is_empty() {
                b.assistant.push(text);
            }
            if !uses.is_empty() {
                b.attach_tool_uses(uses);
            }
        }
    }
    b.finish()
}

fn claude_system_text(s: &Value) -> String {
    match s {
        Value::String(x) => x.clone(),
        Value::Array(a) => a.iter().map(|x| x.as_str().map(str::to_owned).unwrap_or_else(|| x["text"].as_str().unwrap_or("").to_string())).filter(|x| !x.is_empty()).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

fn build_payload(model: &str, body: &Value, rc: &ReqCtx, claude: bool) -> Value {
    let (clean, ov) = parse_suffix(model);
    let km = resolve_kiro_model(&clean);
    let up = km.upstream.clone();
    let tb = apply_thinking_override(body, ov.as_ref());
    let budget = resolve_thinking_budget(&tb, &rc.headers, &clean);
    let additional = additional_fields_for_model(&tb, &up);
    let native_gpt = uses_native_gpt_effort(&tb, &up);
    let (specs, name_map) = normalize_tool_specs(&body["tools"]);
    let msgs = body["messages"].as_array().map(|v| v.as_slice()).unwrap_or(&[]);
    let (history, current) = if claude { convert_claude_messages(msgs, &up) } else { convert_openai_messages(msgs, &up) };
    let auth = rc.psd["authMethod"].as_str().unwrap_or("");
    let bound = matches!(auth, "api_key" | "idc" | "external_idp");
    let arn = rc.psd["profileArn"].as_str().filter(|s| !s.is_empty()).map(str::to_owned).unwrap_or_else(|| if bound { String::new() } else { default_profile_arn(auth).to_string() });
    let ts = crate::jsv::iso_from_ms(now_ms());
    let mut parts = vec![];
    if let Some(b) = budget {
        if !native_gpt {
            parts.push(thinking_prefix(b));
        }
    }
    if km.agentic {
        parts.push(agentic_prompt().to_string());
    }
    if claude {
        let s = claude_system_text(&body["system"]);
        if !s.is_empty() {
            parts.push(s);
        }
    }
    let system_prompt = parts.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
    let time_ctx = format!("[Context: Current time is {ts}]");
    let content_prefix = [system_prompt.clone(), time_ctx.clone()].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
    let conn = rc.connection_id.clone().unwrap_or_default();
    let (conv_id, ephemeral) = crate::session::resolve_session_identity(&rc.headers, body, rc.connection_id.as_deref(), None, "kiro");
    let _cont = crate::session::resolve_continuation_id(&conv_id, &conn, "kiro", ephemeral);
    let replay = apply_session_replay(&conv_id, &conn, &up, &system_prompt, &content_prefix, &time_ctx, &history, &current);
    let canon = canonicalize(&replay.history, Some(&replay.current), &up, &specs, &name_map);
    if !canon.valid {
        tracing::error!("[Kiro] refusing invalid conversation ({} → kiro): {} | turns={}", if claude { "claude" } else { "openai" }, canon.errors.join(", "), canon.history.len() + 1);
        return Value::Null;
    }
    let rcur = &canon.current["userInputMessage"];
    let mut uim = json!({"content": rcur["content"].as_str().unwrap_or(""), "modelId": up, "origin": "AI_EDITOR"});
    if claude {
        if truthy(&rcur["userInputMessageContext"]) {
            uim["userInputMessageContext"] = rcur["userInputMessageContext"].clone();
        }
        if truthy(&rcur["images"]) {
            uim["images"] = rcur["images"].clone();
        }
    } else {
        if rcur["images"].as_array().map(|a| !a.is_empty()).unwrap_or(false) {
            uim["images"] = rcur["images"].clone();
        }
        if truthy(&rcur["userInputMessageContext"]) {
            uim["userInputMessageContext"] = rcur["userInputMessageContext"].clone();
        }
    }
    let mut payload = json!({"conversationState": {"chatTriggerType": "MANUAL", "conversationId": conv_id, "currentMessage": {"userInputMessage": uim}, "history": canon.history}});
    if !arn.is_empty() {
        payload["profileArn"] = json!(arn);
    }
    if let Some(a) = additional {
        payload["additionalModelRequestFields"] = a;
    }
    let max_tokens = if claude { body["max_tokens"].as_f64().filter(|n| *n != 0.0).map(crate::jsv::jnum).unwrap_or(json!(32000)) } else { json!(32000) };
    let mut ic = json!({"maxTokens": max_tokens});
    if body.get("temperature").is_some() {
        ic["temperature"] = body["temperature"].clone();
    }
    if body.get("top_p").is_some() {
        ic["topP"] = body["top_p"].clone();
    }
    payload["inferenceConfig"] = ic;
    let mut restored = Map::new();
    for (o, s) in &name_map {
        if o != s {
            restored.insert(s.clone(), json!(o));
        }
    }
    if !restored.is_empty() {
        payload["_toolNameMap"] = Value::Object(restored);
    }
    payload
}

pub fn openai_to_kiro(model: &str, body: &Value, _stream: bool, rc: &ReqCtx) -> Value {
    build_payload(model, body, rc, false)
}

pub fn claude_to_kiro(model: &str, body: &Value, _stream: bool, rc: &ReqCtx) -> Value {
    build_payload(model, body, rc, true)
}

// ===========================================================================
// response translators
// ===========================================================================

fn restore_name(state: &Value, name: &Value) -> Value {
    let raw = name.as_str().unwrap_or("");
    state["toolNameMap"].get(raw).cloned().unwrap_or_else(|| json!(raw))
}

pub fn kiro_to_openai(chunk: Option<&Value>, state: &mut Value) -> Vec<Value> {
    let Some(chunk) = chunk else { return vec![] };
    if chunk["object"] == "chat.completion.chunk" && chunk["choices"].is_array() {
        if state["toolNameMap"].as_object().map(|m| m.is_empty()).unwrap_or(true) {
            return vec![chunk.clone()];
        }
        let mut c = chunk.clone();
        for ch in c["choices"].as_array_mut().unwrap() {
            if let Some(tcs) = ch["delta"]["tool_calls"].as_array_mut() {
                for tc in tcs.iter_mut() {
                    if truthy(&tc["function"]["name"]) {
                        tc["function"]["name"] = restore_name(state, &tc["function"]["name"]);
                    }
                }
            }
        }
        return vec![c];
    }
    let mut data = chunk.clone();
    if let Some(s) = chunk.as_str() {
        let mut ev = String::new();
        let mut d = String::new();
        for line in s.split('\n') {
            if let Some(x) = line.strip_prefix("event:") {
                ev = x.trim().into();
            } else if let Some(x) = line.strip_prefix(":event-type:") {
                ev = x.trim().into();
            } else if let Some(x) = line.strip_prefix("data:") {
                d = x.trim().into();
            } else if line.starts_with(":content-type:") {
            } else if !line.trim().is_empty() && !line.starts_with(':') {
                d = line.trim().into();
            }
        }
        if d.is_empty() {
            return vec![];
        }
        data = serde_json::from_str::<Value>(&d).unwrap_or_else(|_| json!({"text": d}));
        if data.is_object() {
            data["_eventType"] = json!(ev);
        } else {
            data = json!({"text": d, "_eventType": ev});
        }
    }
    if !truthy(&state["responseId"]) {
        state["responseId"] = json!(format!("chatcmpl-{}", now_ms()));
        state["created"] = json!(now_ms() / 1000);
        state["chunkIndex"] = json!(0);
    }
    let et = if truthy(&data["_eventType"]) { js_string(&data["_eventType"]) } else { data["event"].as_str().unwrap_or("").to_string() };
    let model = if truthy(&state["model"]) { state["model"].clone() } else { json!("kiro") };
    let mk = |state: &Value, delta: Value, fin: Value| build_chunk(state["responseId"].as_str().unwrap_or(""), state["created"].as_i64().unwrap_or(0), &model, delta, fin);
    let first = state["chunkIndex"].as_i64().unwrap_or(0) == 0;
    let bump = |state: &mut Value| state["chunkIndex"] = json!(state["chunkIndex"].as_i64().unwrap_or(0) + 1);
    if et == "assistantResponseEvent" || truthy(&data["assistantResponseEvent"]) {
        let c = [&data["assistantResponseEvent"]["content"], &data["content"]].into_iter().find(|v| truthy(v)).cloned().unwrap_or(json!(""));
        if !truthy(&c) {
            return vec![];
        }
        let mut d = if first { json!({"role": "assistant"}) } else { json!({}) };
        d["content"] = c;
        let out = mk(state, d, Value::Null);
        bump(state);
        return vec![out];
    }
    if et == "reasoningContentEvent" || truthy(&data["reasoningContentEvent"]) {
        let r = if truthy(&data["reasoningContentEvent"]) { data["reasoningContentEvent"].clone() } else { data.clone() };
        let c = if r.is_string() { r } else { [&r["text"], &r["content"], &data["content"]].into_iter().find(|v| truthy(v)).cloned().unwrap_or(json!("")) };
        if !truthy(&c) {
            return vec![];
        }
        let out = mk(state, reasoning_delta(&c, first), Value::Null);
        bump(state);
        return vec![out];
    }
    if et == "toolUseEvent" || truthy(&data["toolUseEvent"]) {
        state["hadToolUse"] = json!(true);
        let tu = if truthy(&data["toolUseEvent"]) { data["toolUseEvent"].clone() } else { data.clone() };
        let id = if truthy(&tu["toolUseId"]) { tu["toolUseId"].clone() } else { json!(fallback_tool_call_id(None)) };
        let mut d = if first { json!({"role": "assistant"}) } else { json!({}) };
        let input = if truthy(&tu["input"]) { tu["input"].clone() } else { json!({}) };
        d["tool_calls"] = json!([{"index": 0, "id": id, "type": "function", "function": {"name": restore_name(state, &tu["name"]), "arguments": input.to_string()}}]);
        let out = mk(state, d, Value::Null);
        bump(state);
        return vec![out];
    }
    if et == "messageStopEvent" || et == "done" || truthy(&data["messageStopEvent"]) {
        let fr = to_openai_finish(&json!(if truthy(&state["hadToolUse"]) { "tool_use" } else { "stop" }), "kiro");
        state["finishReason"] = fr.clone();
        let mut out = mk(state, json!({}), fr);
        if state["usage"].is_object() {
            out["usage"] = state["usage"].clone();
        }
        return vec![out];
    }
    if et == "usageEvent" || truthy(&data["usageEvent"]) {
        if let Some(u) = to_openai_usage(if truthy(&data["usageEvent"]) { &data["usageEvent"] } else { &data }, "kiro") {
            state["usage"] = u;
        }
    }
    vec![]
}

fn finish_to_claude(r: &Value) -> &'static str {
    match r.as_str() {
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        _ => "end_turn",
    }
}

pub fn kiro_to_claude(chunk: Option<&Value>, state: &mut Value) -> Vec<Value> {
    let Some(chunk) = chunk else { return vec![] };
    let data = if let Some(s) = chunk.as_str() {
        let t = s.trim();
        if t.is_empty() || t == "[DONE]" {
            return vec![];
        }
        match serde_json::from_str::<Value>(t.strip_prefix("data:").map(str::trim).unwrap_or(t)) {
            Ok(v) => v,
            Err(_) => return vec![],
        }
    } else {
        chunk.clone()
    };
    if !truthy(&data["choices"][0]) {
        return vec![];
    }
    let mut res = vec![];
    let choice = &data["choices"][0];
    let delta = if choice["delta"].is_object() { choice["delta"].clone() } else { json!({}) };
    if data["usage"].is_object() {
        let u = &data["usage"];
        let mut su = json!({"input_tokens": u["prompt_tokens"].as_f64().map(crate::jsv::jnum).unwrap_or(json!(0)), "output_tokens": u["completion_tokens"].as_f64().map(crate::jsv::jnum).unwrap_or(json!(0))});
        let cr = if !u["cache_read_input_tokens"].is_null() { &u["cache_read_input_tokens"] } else { &u["prompt_tokens_details"]["cached_tokens"] };
        let cc = if !u["cache_creation_input_tokens"].is_null() { &u["cache_creation_input_tokens"] } else { &u["prompt_tokens_details"]["cache_creation_tokens"] };
        if cr.is_number() {
            su["cache_read_input_tokens"] = cr.clone();
        }
        if cc.is_number() {
            su["cache_creation_input_tokens"] = cc.clone();
        }
        state["usage"] = su;
    }
    let next_idx = |state: &mut Value| {
        let n = state["nextBlockIndex"].as_i64().unwrap_or(0);
        state["nextBlockIndex"] = json!(n + 1);
        n
    };
    let stop_thinking = |state: &mut Value, res: &mut Vec<Value>| {
        if truthy(&state["thinkingBlockStarted"]) {
            res.push(json!({"type": "content_block_stop", "index": state["thinkingBlockIndex"]}));
            state["thinkingBlockStarted"] = json!(false);
        }
    };
    let stop_text = |state: &mut Value, res: &mut Vec<Value>| {
        if truthy(&state["textBlockStarted"]) && !truthy(&state["textBlockClosed"]) {
            state["textBlockClosed"] = json!(true);
            res.push(json!({"type": "content_block_stop", "index": state["textBlockIndex"]}));
            state["textBlockStarted"] = json!(false);
        }
    };
    if !truthy(&state["messageStartSent"]) {
        state["messageStartSent"] = json!(true);
        let id = data["id"].as_str().map(|s| s.replacen("chatcmpl-", "", 1)).filter(|s| !s.is_empty()).unwrap_or_else(|| format!("msg_{}", now_ms()));
        state["messageId"] = json!(id);
        state["model"] = if truthy(&data["model"]) { data["model"].clone() } else { json!("kiro") };
        state["nextBlockIndex"] = json!(0);
        res.push(json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant", "model": state["model"], "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 0, "output_tokens": 0}}}));
    }
    let rcont = if truthy(&delta["reasoning_content"]) { delta["reasoning_content"].clone() } else { delta["reasoning"].clone() };
    if truthy(&rcont) {
        stop_text(state, &mut res);
        if !truthy(&state["thinkingBlockStarted"]) {
            let i = next_idx(state);
            state["thinkingBlockIndex"] = json!(i);
            state["thinkingBlockStarted"] = json!(true);
            res.push(json!({"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": ""}}));
        }
        res.push(json!({"type": "content_block_delta", "index": state["thinkingBlockIndex"], "delta": {"type": "thinking_delta", "thinking": rcont}}));
    }
    if truthy(&delta["content"]) {
        stop_thinking(state, &mut res);
        if !truthy(&state["textBlockStarted"]) {
            let i = next_idx(state);
            state["textBlockIndex"] = json!(i);
            state["textBlockStarted"] = json!(true);
            state["textBlockClosed"] = json!(false);
            res.push(json!({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}));
        }
        res.push(json!({"type": "content_block_delta", "index": state["textBlockIndex"], "delta": {"type": "text_delta", "text": delta["content"]}}));
    }
    if let Some(tcs) = delta["tool_calls"].as_array() {
        if !state["toolCalls"].is_object() {
            state["toolCalls"] = json!({});
        }
        if !state["toolArgBuffers"].is_object() {
            state["toolArgBuffers"] = json!({});
        }
        for tc in tcs {
            let idx = tc["index"].as_i64().unwrap_or(0).to_string();
            if truthy(&tc["id"]) {
                stop_thinking(state, &mut res);
                stop_text(state, &mut res);
                let bi = next_idx(state);
                let name = restore_name(state, &tc["function"]["name"]);
                state["toolCalls"][&idx] = json!({"id": tc["id"], "name": name, "blockIndex": bi});
                res.push(json!({"type": "content_block_start", "index": bi, "content_block": {"type": "tool_use", "id": tc["id"], "name": name, "input": {}}}));
            }
            if let Some(a) = tc["function"]["arguments"].as_str().filter(|s| !s.is_empty()) {
                if state["toolCalls"].get(&idx).is_some() {
                    let prev = state["toolArgBuffers"][&idx].as_str().unwrap_or("").to_string();
                    state["toolArgBuffers"][&idx] = json!(prev + a);
                }
            }
        }
    }
    if truthy(&choice["finish_reason"]) {
        stop_thinking(state, &mut res);
        stop_text(state, &mut res);
        if let Some(tcs) = state["toolCalls"].as_object().cloned() {
            for (idx, info) in tcs {
                if let Some(buf) = state["toolArgBuffers"][&idx].as_str().filter(|s| !s.is_empty()) {
                    res.push(json!({"type": "content_block_delta", "index": info["blockIndex"], "delta": {"type": "input_json_delta", "partial_json": buf}}));
                }
                res.push(json!({"type": "content_block_stop", "index": info["blockIndex"]}));
            }
        }
        state["finishReason"] = choice["finish_reason"].clone();
        let fu = if truthy(&state["usage"]) { state["usage"].clone() } else { json!({"input_tokens": 0, "output_tokens": 0}) };
        res.push(json!({"type": "message_delta", "delta": {"stop_reason": finish_to_claude(&choice["finish_reason"])}, "usage": fu}));
        res.push(json!({"type": "message_stop"}));
    }
    res
}

// ===========================================================================
// AWS EventStream
// ===========================================================================

const MAX_MESSAGE_BYTES: usize = 24 * 1024 * 1024;
const MAX_HEADERS_BYTES: usize = 128 * 1024;

pub fn crc32(data: &[u8]) -> u32 {
    static TABLE: LazyLock<[u32; 256]> = LazyLock::new(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut v = i as u32;
            for _ in 0..8 {
                v = (v >> 1) ^ if v & 1 != 0 { 0xedb88320 } else { 0 };
            }
            *e = v;
        }
        t
    });
    let mut crc = 0xffff_ffffu32;
    for b in data {
        crc = TABLE[((crc ^ *b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xffff_ffff
}

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

pub struct Frame {
    pub headers: HashMap<String, Value>,
    pub payload: Value,
}

pub fn parse_event_frame(data: &[u8]) -> Result<Frame, String> {
    if data.len() < 16 {
        return Err("AWS EventStream frame is shorter than 16 bytes".into());
    }
    let total = be32(data, 0) as usize;
    let hlen = be32(data, 4) as usize;
    if total != data.len() {
        return Err("AWS EventStream frame length does not match its prelude".into());
    }
    if total > MAX_MESSAGE_BYTES || hlen > MAX_HEADERS_BYTES || hlen > total - 16 {
        return Err("AWS EventStream frame bounds are invalid".into());
    }
    if be32(data, 8) != crc32(&data[..8]) {
        return Err("AWS EventStream prelude CRC mismatch".into());
    }
    if be32(data, total - 4) != crc32(&data[..total - 4]) {
        return Err("AWS EventStream message CRC mismatch".into());
    }
    let mut headers = HashMap::new();
    let mut off = 12;
    let end = off + hlen;
    let need = |off: usize, n: usize| if off + n > end { Err("AWS EventStream header exceeds its declared bounds".to_string()) } else { Ok(()) };
    while off < end {
        need(off, 1)?;
        let nl = data[off] as usize;
        off += 1;
        need(off, nl + 1)?;
        let name = String::from_utf8_lossy(&data[off..off + nl]).into_owned();
        off += nl;
        if headers.contains_key(&name) {
            return Err(format!("AWS EventStream contains duplicate header: {name}"));
        }
        let ty = data[off];
        off += 1;
        let v = match ty {
            0 | 1 => json!(ty == 0),
            2 => {
                need(off, 1)?;
                off += 1;
                json!(data[off - 1] as i8)
            }
            3 => {
                need(off, 2)?;
                off += 2;
                json!(i16::from_be_bytes([data[off - 2], data[off - 1]]))
            }
            4 => {
                need(off, 4)?;
                off += 4;
                json!(be32(data, off - 4) as i32)
            }
            5 | 8 => {
                need(off, 8)?;
                off += 8;
                Value::Null
            }
            6 | 7 => {
                need(off, 2)?;
                let l = u16::from_be_bytes([data[off], data[off + 1]]) as usize;
                off += 2;
                need(off, l)?;
                let b = &data[off..off + l];
                off += l;
                if ty == 7 { json!(String::from_utf8_lossy(b)) } else { Value::Null }
            }
            9 => {
                need(off, 16)?;
                off += 16;
                Value::Null
            }
            _ => return Err(format!("AWS EventStream header {name} has unknown type {ty}")),
        };
        headers.insert(name, v);
    }
    let payload = &data[end..total - 4];
    let text = String::from_utf8_lossy(payload);
    if payload.is_empty() || text.trim().is_empty() {
        return Ok(Frame { headers, payload: Value::Null });
    }
    serde_json::from_str(&text).map(|p| Frame { headers, payload: p }).map_err(|e| format!("AWS EventStream payload is not valid JSON ({e})"))
}

/// Encodes one EventStream frame (used by tests and the mock upstream).
pub fn encode_event_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    let mut h = vec![];
    for (k, v) in headers {
        h.push(k.len() as u8);
        h.extend_from_slice(k.as_bytes());
        h.push(7);
        h.extend_from_slice(&(v.len() as u16).to_be_bytes());
        h.extend_from_slice(v.as_bytes());
    }
    let total = 16 + h.len() + payload.len();
    let mut out = vec![];
    out.extend_from_slice(&(total as u32).to_be_bytes());
    out.extend_from_slice(&(h.len() as u32).to_be_bytes());
    let c = crc32(&out);
    out.extend_from_slice(&c.to_be_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(payload);
    let c2 = crc32(&out);
    out.extend_from_slice(&c2.to_be_bytes());
    out
}

fn normalize_stop_reason(v: &Value) -> Option<String> {
    static CAMEL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"([a-z])([A-Z])").unwrap());
    static SEP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\s-]+").unwrap());
    let s = if v.is_null() { String::new() } else { js_string(v) };
    let r = SEP.replace_all(&CAMEL.replace_all(s.trim(), "${1}_${2}").to_lowercase(), "_").into_owned();
    match r.as_str() {
        "endturn" | "end_turn" | "stop" | "stop_sequence" => Some("end_turn".into()),
        "tooluse" | "tool_use" | "tool_calls" => Some("tool_use".into()),
        "maxtokens" | "max_tokens" | "max_output_tokens" | "length" => Some("max_tokens".into()),
        "" => None,
        _ => Some(r),
    }
}

fn stop_disposition(reason: Option<&str>, has_tools: bool) -> &'static str {
    static REFUSAL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:content.*filter|guardrail|safety|policy|blocked)").unwrap());
    match reason {
        Some("malformed_model_output") | Some("invalid_model_output") => "retryable_protocol_failure",
        Some("cancelled") | Some("pause_turn") | Some("model_context_window_exceeded") => "terminal_incomplete",
        Some(r) if r == "refusal" || REFUSAL.is_match(r) => "terminal_refusal",
        Some("max_tokens") => {
            if has_tools {
                "terminal_incomplete"
            } else {
                "length"
            }
        }
        Some(r) if r != "end_turn" && r != "tool_use" => "unknown_failure",
        _ if has_tools || reason == Some("tool_use") => "tool_use",
        None | Some("end_turn") => "complete",
        _ => "unknown_failure",
    }
}

fn merge_stop_reason(cur: Option<String>, inc: Option<String>) -> Option<String> {
    let Some(i) = inc else { return cur };
    let Some(c) = cur else { return Some(i) };
    let sev = |r: &str| match stop_disposition(Some(r), false) {
        "terminal_refusal" => 6,
        "terminal_incomplete" => 5,
        "unknown_failure" => 4,
        "retryable_protocol_failure" => 3,
        "length" => 2,
        _ => 1,
    };
    if sev(&i) > sev(&c) { Some(i) } else { Some(c) }
}

fn encode_sse_error(code: &str, message: &str, details: Option<Value>) -> Bytes {
    let mut e = json!({"message": message, "type": "upstream_error", "code": code});
    if let Some(d) = details {
        e["details"] = d;
    }
    Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n", json!({"error": e})))
}

struct Tool {
    id: String,
    name: String,
    kind: Option<&'static str>,
    chunks: Vec<String>,
    object: Value,
    bytes: usize,
}

/// transformEventStreamToSSE as a synchronous push transformer.
pub struct EventStreamToSse {
    model: String,
    response_id: String,
    created: i64,
    context_window: i64,
    max_tool_bytes: usize,
    buffer: Vec<u8>,
    chunk_index: i64,
    tool_counter: i64,
    tools: Vec<Tool>,
    buffered_tool_bytes: usize,
    has_text: bool,
    has_reasoning: bool,
    has_code: bool,
    has_tool_calls: bool,
    saw_tool_use: bool,
    explicit_stop: bool,
    stop_reason: Option<String>,
    provenance: Option<String>,
    transport: String,
    total_content: usize,
    ctx_pct: f64,
    has_ctx: bool,
    has_metering: bool,
    usage: Option<Value>,
    in_thinking: bool,
    tool_validation_error: Option<String>,
    pub validated_frames: usize,
    pub finished: bool,
    event_counts: Map<String, Value>,
    pub diagnostics: Option<Value>,
}

impl EventStreamToSse {
    pub fn new(model: &str, max_tool_bytes: usize) -> Self {
        let up = resolve_kiro_model(model).upstream;
        let cw = crate::caps::caps_for(Some("kiro"), &up).get("contextWindow").as_i64().unwrap_or(200_000);
        EventStreamToSse {
            model: model.into(),
            response_id: format!("chatcmpl-{}", now_ms()),
            created: now_ms() / 1000,
            context_window: if cw > 0 { cw } else { 200_000 },
            max_tool_bytes,
            buffer: vec![],
            chunk_index: 0,
            tool_counter: 0,
            tools: vec![],
            buffered_tool_bytes: 0,
            has_text: false,
            has_reasoning: false,
            has_code: false,
            has_tool_calls: false,
            saw_tool_use: false,
            explicit_stop: false,
            stop_reason: None,
            provenance: None,
            transport: "consuming_response".into(),
            total_content: 0,
            ctx_pct: 0.0,
            has_ctx: false,
            has_metering: false,
            usage: None,
            in_thinking: false,
            tool_validation_error: None,
            validated_frames: 0,
            finished: false,
            event_counts: Map::new(),
            diagnostics: None,
        }
    }

    fn diag(&self, over: Value) -> Value {
        let mut d = json!({
            "terminal_provenance": self.provenance.clone().unwrap_or_else(|| "clean_eventstream_eof".into()),
            "transport_state": self.transport,
            "stop_reason": self.stop_reason,
            "stop_disposition": stop_disposition(self.stop_reason.as_deref(), self.has_tool_calls),
            "response_state": if self.has_tool_calls { "valid_tool" } else if self.has_text || self.has_reasoning || self.has_code { "text_reasoning" } else if self.explicit_stop { "explicit_stop" } else { "no_semantic_output" },
            "event_counts": self.event_counts,
            "incomplete_frame_bytes": self.buffer.len(),
        });
        for (k, v) in over.as_object().into_iter().flatten() {
            d[k] = v.clone();
        }
        d
    }

    fn sse(&self, delta: Value, fin: Value, usage: Option<&Value>) -> Bytes {
        let mut c = json!({"id": self.response_id, "object": "chat.completion.chunk", "created": self.created, "model": self.model, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]});
        if let Some(u) = usage {
            c["usage"] = u.clone();
        }
        Bytes::from(format!("data: {c}\n\n"))
    }

    fn emit(&mut self, out: &mut Vec<Bytes>, mut delta: Value) {
        if self.chunk_index == 0 {
            let mut d = json!({"role": "assistant"});
            for (k, v) in delta.as_object().unwrap() {
                d[k] = v.clone();
            }
            delta = d;
        }
        self.chunk_index += 1;
        out.push(self.sse(delta, Value::Null, None));
    }

    fn fail(&mut self, out: &mut Vec<Bytes>, prov: &str, code: &str, msg: &str, extra: Value) {
        self.finished = true;
        self.provenance = Some(prov.into());
        self.transport = extra["transport_state"].as_str().unwrap_or("corrupt_frame").into();
        let mut over = json!({"stop_disposition": extra["stop_disposition"].as_str().unwrap_or("terminal_incomplete")});
        for (k, v) in extra.as_object().into_iter().flatten() {
            over[k] = v.clone();
        }
        let d = self.diag(over);
        self.diagnostics = Some(d.clone());
        out.push(encode_sse_error(code, msg, Some(d)));
    }

    fn bound(&self) -> Result<(), (bool, String)> {
        if self.buffered_tool_bytes <= self.max_tool_bytes {
            Ok(())
        } else {
            Err((true, "Kiro buffered tool input exceeded the integrity memory bound".into()))
        }
    }

    fn append_tool_input(&mut self, ti: usize, input: &Value) -> Result<(), (bool, String)> {
        if input.is_null() {
            return Ok(());
        }
        let t = &mut self.tools[ti];
        match input {
            Value::String(s) => {
                if t.kind.is_some() && t.kind != Some("string") {
                    return Err((false, "Kiro tool input changed fragment type".into()));
                }
                t.kind = Some("string");
                t.chunks.push(s.clone());
                self.buffered_tool_bytes += s.len();
            }
            Value::Object(_) => {
                if t.kind.is_some() && t.kind != Some("object") {
                    return Err((false, "Kiro tool input changed fragment type".into()));
                }
                t.kind = Some("object");
                self.buffered_tool_bytes -= t.bytes.min(self.buffered_tool_bytes);
                t.object = input.clone();
                t.bytes = input.to_string().len();
                self.buffered_tool_bytes += t.bytes;
            }
            _ => return Err((false, "Kiro tool input must be a JSON object".into())),
        }
        self.bound()
    }

    fn emit_tools(&mut self, out: &mut Vec<Bytes>) -> Result<(), String> {
        let tools = std::mem::take(&mut self.tools);
        for t in tools {
            let parsed: Result<Value, String> = match t.kind {
                None => Err("Kiro tool call is missing input".into()),
                Some("object") => Ok(t.object.clone()),
                _ => serde_json::from_str::<Value>(&t.chunks.concat()).map_err(|e| e.to_string()).and_then(|v| if v.is_object() { Ok(v) } else { Err("not an object".into()) }).map_err(|e| format!("Kiro tool input must be valid object JSON ({e})")),
            };
            let parsed = parsed.and_then(|input| {
                if t.name == "tool_call" {
                    if !input["name"].as_str().map(|s| !s.trim().is_empty()).unwrap_or(false) {
                        return Err("Invalid Kiro tool_call payload: missing nested MCP tool name".to_string());
                    }
                    if input.get("arguments").is_none() {
                        return Err("Invalid Kiro tool_call payload: missing nested MCP tool arguments".to_string());
                    }
                }
                Ok(input)
            });
            let input = match parsed {
                Ok(i) => i,
                Err(e) => {
                    if self.tool_validation_error.is_none() {
                        self.tool_validation_error = Some(e.clone());
                    }
                    tracing::error!("[Kiro] dropping unusable tool call {} ({}): {e}", t.id, t.name);
                    continue;
                }
            };
            let idx = self.tool_counter;
            self.tool_counter += 1;
            self.emit(out, json!({"tool_calls": [{"index": idx, "id": t.id, "type": "function", "function": {"name": t.name, "arguments": ""}}]}));
            let ser = input.to_string();
            self.emit(out, json!({"tool_calls": [{"index": idx, "function": {"arguments": ser}}]}));
            self.total_content += t.name.chars().count() + ser.chars().count();
            self.has_tool_calls = true;
        }
        self.buffered_tool_bytes = 0;
        if self.stop_reason.as_deref() == Some("tool_use") && !self.has_tool_calls && !self.has_text && !self.has_reasoning && !self.has_code {
            return Err("Kiro tool_use stop reason did not include a complete tool call".into());
        }
        Ok(())
    }

    fn process_event(&mut self, ev: Frame, out: &mut Vec<Bytes>) -> Result<bool, (bool, String)> {
        let mt = ev.headers.get(":message-type").and_then(|v| v.as_str()).unwrap_or("");
        if mt == "error" || mt == "exception" {
            let m = ev.payload["message"].as_str().map(str::to_owned).unwrap_or_else(|| format!("Kiro upstream sent an EventStream {mt}"));
            self.fail(out, "upstream_eventstream_error", "kiro_upstream_eventstream_error", &m, json!({"transport_state": "upstream_error"}));
            return Ok(false);
        }
        let et = ev.headers.get(":event-type").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let known = ["assistantResponseEvent", "reasoningContentEvent", "codeEvent", "toolUseEvent", "messageStopEvent", "metadataEvent", "MetadataEvent", "contextUsageEvent", "meteringEvent", "metricsEvent"];
        let key = if known.contains(&et.as_str()) { et.clone() } else { "other".into() };
        let n = self.event_counts.get(&key).and_then(|v| v.as_i64()).unwrap_or(0) + 1;
        self.event_counts.insert(key, json!(n));
        let p = &ev.payload;
        match et.as_str() {
            "assistantResponseEvent" if p["content"].is_string() => {
                let mut content = p["content"].as_str().unwrap().to_string();
                if self.in_thinking {
                    match content.find("</thinking>") {
                        None => content.clear(),
                        Some(end) => {
                            self.in_thinking = false;
                            let rest = &content[end + 11..];
                            content = rest.strip_prefix('\n').unwrap_or(rest).to_string();
                        }
                    }
                } else if let Some(start) = content.find("<thinking>") {
                    match content[start + 10..].find("</thinking>").map(|e| e + start + 10) {
                        None => {
                            self.in_thinking = true;
                            content.truncate(start);
                        }
                        Some(end) => {
                            let rest = &content[end + 11..];
                            content = format!("{}{}", &content[..start], rest.strip_prefix('\n').unwrap_or(rest));
                        }
                    }
                }
                if !content.is_empty() || !self.has_reasoning {
                    self.has_text |= !content.is_empty();
                    self.total_content += content.encode_utf16().count();
                    self.emit(out, json!({"content": content}));
                }
            }
            "reasoningContentEvent" => {
                let v = if truthy(&p["reasoningContentEvent"]) { &p["reasoningContentEvent"] } else { p };
                let c = if let Some(s) = v.as_str() { s.to_string() } else { [&v["text"], &v["content"]].into_iter().find(|x| truthy(x)).map(js_string).unwrap_or_default() };
                if !c.is_empty() {
                    self.has_reasoning = true;
                    self.total_content += c.encode_utf16().count();
                    self.emit(out, json!({"reasoning_content": c}));
                }
            }
            "codeEvent" if p["content"].is_string() => {
                self.has_code = true;
                let c = p["content"].as_str().unwrap().to_string();
                self.total_content += c.encode_utf16().count();
                self.emit(out, json!({"content": c}));
            }
            "toolUseEvent" => {
                self.saw_tool_use = true;
                let values: Vec<Value> = if let Some(a) = p.as_array() { a.clone() } else { vec![p.clone()] };
                if values.first().map(|v| !truthy(v)).unwrap_or(true) {
                    return Err((false, "Kiro toolUseEvent is empty".into()));
                }
                for v in values {
                    let name = v["name"].as_str().map(|s| s.trim().to_string()).unwrap_or_default();
                    if name.is_empty() {
                        return Err((false, "Kiro toolUseEvent is missing a tool name".into()));
                    }
                    let id = match &v["toolUseId"] {
                        Value::Null => format!("call_{}_{}", self.created, self.tools.len() + 1),
                        Value::String(s) if !s.trim().is_empty() => s.clone(),
                        _ => return Err((false, "Kiro toolUseEvent has an invalid toolUseId".into())),
                    };
                    let ti = match self.tools.iter().position(|t| t.id == id) {
                        Some(i) => {
                            if self.tools[i].name != name {
                                return Err((false, "Kiro tool name changed between fragments".into()));
                            }
                            i
                        }
                        None => {
                            self.buffered_tool_bytes += id.len() + name.len() + 32;
                            self.tools.push(Tool { id, name, kind: None, chunks: vec![], object: Value::Null, bytes: 0 });
                            self.bound()?;
                            self.tools.len() - 1
                        }
                    };
                    self.append_tool_input(ti, &v["input"])?;
                }
            }
            "messageStopEvent" => {
                self.explicit_stop = true;
                let raw = if !p["stopReason"].is_null() { &p["stopReason"] } else { &p["stop_reason"] };
                let r = normalize_stop_reason(raw).unwrap_or_else(|| if self.saw_tool_use { "tool_use".into() } else { "end_turn".into() });
                let merged = merge_stop_reason(self.stop_reason.clone(), Some(r));
                if merged != self.stop_reason {
                    self.provenance = Some("message_stop_event".into());
                }
                self.stop_reason = merged;
            }
            "metadataEvent" | "MetadataEvent" => {
                let m = [&p["metadataEvent"], &p["metadata"]].into_iter().find(|v| truthy(v)).unwrap_or(p);
                let raw = if !m["stopReason"].is_null() { &m["stopReason"] } else { &m["stop_reason"] };
                if let Some(r) = normalize_stop_reason(raw) {
                    self.explicit_stop = true;
                    let merged = merge_stop_reason(self.stop_reason.clone(), Some(r));
                    if merged != self.stop_reason {
                        self.provenance = Some("metadata_stop_reason".into());
                    }
                    self.stop_reason = merged;
                }
            }
            "contextUsageEvent" => {
                if let Some(pct) = num_like(&p["contextUsagePercentage"]) {
                    self.ctx_pct = pct;
                    self.has_ctx = true;
                }
            }
            "meteringEvent" => {
                self.has_metering = true;
                let m = if truthy(&p["meteringEvent"]) { &p["meteringEvent"] } else { p };
                if let Some(c) = num_like(&m["usage"]) {
                    let mut u = self.usage.clone().unwrap_or(json!({}));
                    u["kiro_credits"] = crate::jsv::jnum(c);
                    u["kiro_credit_unit"] = json!(m["unit"].as_str().unwrap_or("credit"));
                    self.usage = Some(u);
                }
            }
            "metricsEvent" => {
                let m = if truthy(&p["metricsEvent"]) { &p["metricsEvent"] } else { p };
                let pr = num_like(&m["inputTokens"]).unwrap_or(0.0);
                let co = num_like(&m["outputTokens"]).unwrap_or(0.0);
                if pr != 0.0 || co != 0.0 {
                    let mut u = self.usage.clone().unwrap_or(json!({}));
                    u["prompt_tokens"] = crate::jsv::jnum(pr);
                    u["completion_tokens"] = crate::jsv::jnum(co);
                    u["total_tokens"] = crate::jsv::jnum(pr + co);
                    let cr = num_like(if truthy(&m["cacheReadInputTokens"]) { &m["cacheReadInputTokens"] } else { &m["cache_read_input_tokens"] }).unwrap_or(0.0);
                    let cc = num_like(if truthy(&m["cacheCreationInputTokens"]) { &m["cacheCreationInputTokens"] } else { &m["cache_creation_input_tokens"] }).unwrap_or(0.0);
                    if cr != 0.0 {
                        u["cache_read_input_tokens"] = crate::jsv::jnum(cr);
                    }
                    if cc != 0.0 {
                        u["cache_creation_input_tokens"] = crate::jsv::jnum(cc);
                    }
                    self.usage = Some(u);
                }
            }
            _ => {}
        }
        Ok(true)
    }

    /// processBytes: returns false when the stream must stop.
    pub fn push(&mut self, chunk: &[u8], out: &mut Vec<Bytes>) -> bool {
        if self.buffer.len() + chunk.len() > MAX_MESSAGE_BYTES {
            self.fail(out, "corrupt_eventstream_frame", "kiro_missing_terminal", "Kiro EventStream buffered bytes exceed the protocol bound", json!({}));
            return false;
        }
        self.buffer.extend_from_slice(chunk);
        while self.buffer.len() >= 12 {
            if be32(&self.buffer, 8) != crc32(&self.buffer[..8]) {
                self.fail(out, "corrupt_eventstream_frame", "kiro_missing_terminal", "Kiro EventStream prelude CRC mismatch", json!({}));
                return false;
            }
            let total = be32(&self.buffer, 0) as usize;
            let hl = be32(&self.buffer, 4) as usize;
            if total < 16 || total > MAX_MESSAGE_BYTES || hl > MAX_HEADERS_BYTES || hl > total - 16 {
                self.fail(out, "corrupt_eventstream_frame", "kiro_missing_terminal", "Kiro EventStream frame bounds are invalid", json!({}));
                return false;
            }
            if self.buffer.len() < total {
                break;
            }
            let frame: Vec<u8> = self.buffer.drain(..total).collect();
            let ev = match parse_event_frame(&frame) {
                Ok(e) => e,
                Err(e) => {
                    self.fail(out, "corrupt_eventstream_frame", "kiro_missing_terminal", &e, json!({}));
                    return false;
                }
            };
            self.transport = "valid_complete_frame".into();
            self.validated_frames += 1;
            match self.process_event(ev, out) {
                Ok(true) => {}
                Ok(false) => return false,
                Err((exceeded, msg)) => {
                    if !exceeded {
                        if self.tool_validation_error.is_none() {
                            self.tool_validation_error = Some(msg.clone());
                        }
                        tracing::error!("[Kiro] tool fragment rejected, keeping {} buffered tool(s): {msg}", self.tools.len());
                        continue;
                    }
                    let t = self.transport.clone();
                    self.fail(out, "integrity_buffer_exceeded", "kiro_integrity_buffer_exceeded", &msg, json!({"transport_state": t, "stop_disposition": "terminal_incomplete"}));
                    return false;
                }
            }
        }
        true
    }

    fn terminal_code(d: &str) -> &'static str {
        match d {
            "retryable_protocol_failure" => "kiro_retryable_protocol_failure",
            "terminal_refusal" => "kiro_terminal_refusal",
            "terminal_incomplete" => "kiro_terminal_incomplete",
            _ => "kiro_unknown_stop_reason",
        }
    }

    pub fn finish(&mut self, out: &mut Vec<Bytes>) {
        if self.finished {
            return;
        }
        if !self.buffer.is_empty() {
            self.fail(out, "incomplete_eventstream_frame", "kiro_missing_terminal", "Kiro EventStream ended with a truncated frame", json!({"transport_state": "incomplete_frame"}));
            return;
        }
        self.transport = "clean_eof".into();
        let truncation = |r: &Option<String>| matches!(r.as_deref(), Some("model_context_window_exceeded") | Some("max_tokens"));
        let declared = stop_disposition(self.stop_reason.as_deref(), self.saw_tool_use);
        let declared_trunc = declared == "terminal_incomplete" && truncation(&self.stop_reason) && self.chunk_index > 0;
        let bad = ["retryable_protocol_failure", "terminal_incomplete", "terminal_refusal", "unknown_failure"];
        if !declared_trunc && bad.contains(&declared) {
            let prov = self.provenance.clone().unwrap_or_else(|| "metadata_stop_reason".into());
            let msg = format!("Kiro ended with non-success stop reason: {}", self.stop_reason.clone().unwrap_or_default());
            let t = self.transport.clone();
            self.fail(out, &prov, Self::terminal_code(declared), &msg, json!({"transport_state": t, "stop_disposition": declared}));
            return;
        }
        if let Err(e) = self.emit_tools(out) {
            let t = self.transport.clone();
            self.fail(out, "invalid_tool_call", "invalid_kiro_tool_call", &e, json!({"transport_state": t, "stop_disposition": "retryable_protocol_failure"}));
            return;
        }
        if let Some(e) = self.tool_validation_error.clone() {
            if !self.has_tool_calls && !self.has_text && !self.has_reasoning && !self.has_code {
                let t = self.transport.clone();
                self.fail(out, "invalid_tool_call", "invalid_kiro_tool_call", &e, json!({"transport_state": t, "stop_disposition": "retryable_protocol_failure"}));
                return;
            }
        }
        let has_output = self.has_text || self.has_reasoning || self.has_code || self.has_tool_calls;
        if !has_output && !self.explicit_stop {
            let t = self.transport.clone();
            self.fail(out, "empty_response_eof", "kiro_missing_terminal", "Kiro EventStream ended without model output", json!({"transport_state": t}));
            return;
        }
        let disp = stop_disposition(self.stop_reason.as_deref(), self.has_tool_calls);
        let trunc = disp == "terminal_incomplete" && truncation(&self.stop_reason) && self.chunk_index > 0;
        if !trunc && bad.contains(&disp) {
            let prov = self.provenance.clone().unwrap_or_else(|| "metadata_stop_reason".into());
            let msg = format!("Kiro ended with non-success stop reason: {}", self.stop_reason.clone().unwrap_or_default());
            let t = self.transport.clone();
            self.fail(out, &prov, Self::terminal_code(disp), &msg, json!({"transport_state": t, "stop_disposition": disp}));
            return;
        }
        if self.has_metering && self.has_ctx && !self.usage.as_ref().map(|u| truthy(&u["total_tokens"])).unwrap_or(false) {
            let completion = if self.total_content > 0 { (self.total_content / 4).max(1) as i64 } else { 0 };
            let prompt = (self.ctx_pct * self.context_window as f64 / 100.0).floor() as i64;
            let mut u = self.usage.clone().unwrap_or(json!({}));
            u["prompt_tokens"] = json!(prompt);
            u["completion_tokens"] = json!(completion);
            u["total_tokens"] = json!(prompt + completion);
            self.usage = Some(u);
        }
        let fr = if trunc {
            "length"
        } else if self.has_tool_calls {
            "tool_calls"
        } else if disp == "length" {
            "length"
        } else {
            "stop"
        };
        out.push(self.sse(json!({}), json!(fr), self.usage.as_ref()));
        out.push(Bytes::from_static(b"data: [DONE]\n\n"));
        self.finished = true;
        let prov = self.provenance.clone().unwrap_or_else(|| "clean_eventstream_eof".into());
        let t = self.transport.clone();
        self.diagnostics = Some(self.diag(json!({"terminal_provenance": prov, "transport_state": t, "stop_disposition": if trunc { "length" } else { disp }})));
    }
}

fn num_like(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Null => None,
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
    .filter(|f| f.is_finite())
}

/// Streams an EventStream response as OpenAI SSE (no integrity buffering).
pub fn eventstream_to_sse(mut body: ByteStream, model: String) -> ByteStream {
    let s = async_stream::stream! {
        let mut t = EventStreamToSse::new(&model, KIRO_REPAIR_BUFFER_MAX_BYTES / 2);
        while !t.finished {
            match body.next().await {
                Some(Ok(c)) => {
                    let mut out = vec![];
                    let frames = t.validated_frames;
                    let chunks = t.chunk_index;
                    let ok = t.push(&c, &mut out);
                    if ok && t.validated_frames > frames && t.chunk_index == chunks { out.push(Bytes::from_static(b": kiro-upstream\n\n")); }
                    for b in out { yield Ok::<Bytes, String>(b); }
                    if !ok { break; }
                }
                Some(Err(e)) => {
                    let mut out = vec![];
                    t.fail(&mut out, "upstream_read_error", "kiro_missing_terminal", &e, json!({"transport_state": "upstream_error"}));
                    for b in out { yield Ok(b); }
                    break;
                }
                None => break,
            }
        }
        let mut out = vec![];
        t.finish(&mut out);
        for b in out { yield Ok(b); }
    };
    s.boxed()
}

// ===========================================================================
// integrity gate
// ===========================================================================

const KIRO_REPAIR_BUFFER_MAX_BYTES: usize = 8 * 1024 * 1024;
const KIRO_SHORT_FINAL_MAX_CHARS: usize = 800;

fn env_pos(name: &str, d: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| crate::jsv::parse_int_prefix(&v)).filter(|n| *n > 0).map(|n| n as u64).unwrap_or(d)
}

fn is_ellipsis_only(s: &str) -> bool {
    matches!(s.trim(), "..." | "…")
}

pub fn is_short_future_action(v: &str) -> bool {
    static SHORT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^(?:(?:(?:現在|接著|接下來|下一步)[，,:：\s]*(?:我(?:只)?(?:會|要|將|再)?\s*)?|我只再)(?:補|查|確認|驗證|追(?:查|蹤)?|繼續|檢查|測試)|我(?:會|要|將)(?:再|重新)?(?:補(?:齊|查)?|抓取|查(?:詢)?|確認|驗證|追(?:查|蹤)?|繼續|檢查|測試)|(?:(?:next|now|then)\b[\s,:-]*)?(?:i(?:'ll| will| am going to| need to)|let me)\s+(?:verify|check|confirm|validate|investigate|trace|continue|follow up|test)\b)").unwrap());
    static OBSERVED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)^目前證據顯示.{1,700}[。.!?；;]\s*最後補查\s+504\s+access\s+log[，,]\s*確認\s+host[／/]路徑與是否為集中流量[。.!]?$").unwrap());
    static EN_FUT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^(?:(?:next|now|then)\b[\s,:-]*)?(?:i(?:'ll| will| am going to| need to)|let me)\s+(?:verify|check|confirm|validate|investigate|trace|continue|follow up|test)\b").unwrap());
    static EN_RES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)(?:[:;\n]|[.!?]\s+\S|\b(?:status|checksum|response|deployment)\s+(?:is|are|was|were|matches?|equals?|returned)\b)").unwrap());
    static ZH_FUT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:(?:現在|接著|接下來|下一步)[，,:：\s]*(?:我(?:只)?(?:會|要|將|再)?\s*)?|我只再|我(?:會|要|將)(?:再|重新)?)(?:補|抓取|查|確認|驗證|追|繼續|檢查|測試)").unwrap());
    static ZH_RES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:[。！？]\s*\S|(?:版本|狀態|回應|結果|部署|校驗碼)(?:是|為|等於|顯示))").unwrap());
    static WAIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)(?:請(?:你|先)|你(?:先|需要|可以|提供|確認|批准|允許)|等待(?:你|使用者)|等你|核准|同意|授權|\b(?:after|when|once)\s+you\b|\byour\s+(?:approval|confirmation|permission|input)\b|\bwait(?:ing)?\s+for\s+you\b|\bplease\s+(?:approve|confirm|provide|send)\b)").unwrap());
    static DONE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)(?:已(?:經)?完成|完成(?:了|驗證|確認)|修復完成|確認無誤|驗證(?:完成|通過)|測試(?:均)?通過|結論|總結|\b(?:done|completed|fixed|verified|confirmed|passed|in conclusion|summary)\b|\b(?:is|are) complete\b)").unwrap());
    static EVID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)(?:顯示|發現|因此|成功|失敗|正常|無錯誤|沒有錯誤|\b(?:found|shows?|showed|because|therefore|succeeded|failed|healthy|green|no errors?)\b)").unwrap());
    let t = v.trim().replace('’', "'");
    if OBSERVED.is_match(&t) {
        return true;
    }
    if EN_FUT.is_match(&t) && EN_RES.is_match(&t) {
        return false;
    }
    if ZH_FUT.is_match(&t) && ZH_RES.is_match(&t) {
        return false;
    }
    let len = t.encode_utf16().count();
    len > 0 && len <= KIRO_SHORT_FINAL_MAX_CHARS && SHORT.is_match(&t) && !WAIT.is_match(&t) && !DONE.is_match(&t) && !EVID.is_match(&t)
}

#[derive(Default)]
struct Inspect {
    content: String,
    reasoning: String,
    has_tool_calls: bool,
    error: Option<Value>,
}

fn inspect_chunk(chunk: &[u8], st: &mut Inspect) {
    for line in String::from_utf8_lossy(chunk).split('\n') {
        let Some(d) = line.strip_prefix("data: ") else { continue };
        let d = d.trim();
        if d.is_empty() || d == "[DONE]" {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<Value>(d) else { continue };
        if truthy(&ev["error"]) {
            st.error = Some(ev["error"].clone());
        }
        for c in ev["choices"].as_array().into_iter().flatten() {
            if let Some(s) = c["delta"]["content"].as_str() {
                st.content.push_str(s);
            }
            if let Some(s) = c["delta"]["reasoning_content"].as_str() {
                st.reasoning.push_str(s);
            }
            if c["delta"]["tool_calls"].as_array().map(|a| !a.is_empty()).unwrap_or(false) {
                st.has_tool_calls = true;
            }
        }
    }
}

struct Attempt {
    kind: &'static str,
    message: Option<String>,
    diagnostics: Value,
    bytes: Vec<u8>,
}

struct GateOpts {
    max_bytes: usize,
    ttft: Duration,
    stall: Duration,
    repair: bool,
}

async fn read_attempt(up: Upstream, model: &str, o: &GateOpts, attempt: &str) -> Attempt {
    let mut t = EventStreamToSse::new(model, (o.max_bytes / 2).max(1));
    let mut body = up.body;
    let mut bytes: Vec<u8> = vec![];
    let mut out_state = Inspect::default();
    let mut saw = false;
    let mut err: Option<String> = None;
    'read: while !t.finished {
        let timeout = if saw { o.stall } else { o.ttft };
        let next = tokio::time::timeout(timeout, body.next()).await;
        let mut out = vec![];
        match next {
            Err(_) => {
                err = Some(format!("Kiro integrity validation {}", if saw { "stalled" } else { "timed out before first chunk" }));
                break 'read;
            }
            Ok(None) => break,
            Ok(Some(Err(e))) => {
                t.fail(&mut out, "upstream_read_error", "kiro_missing_terminal", &e, json!({"transport_state": "upstream_error"}));
            }
            Ok(Some(Ok(c))) => {
                saw = true;
                t.push(&c, &mut out);
            }
        }
        for b in out {
            bytes.extend_from_slice(&b);
            inspect_chunk(&b, &mut out_state);
        }
        if bytes.len() > o.max_bytes {
            return Attempt { kind: "terminal_stop", message: Some(format!("Kiro integrity buffer exceeded {} bytes", o.max_bytes)), diagnostics: json!({"terminal_provenance": "integrity_buffer_exceeded"}), bytes: vec![] };
        }
    }
    if let Some(e) = err {
        return Attempt {
            kind: "missing_terminal",
            message: Some(e),
            diagnostics: json!({"attempt": attempt, "terminal_provenance": "transport_read_error", "transport_state": "upstream_error", "stop_reason": null, "stop_disposition": "terminal_incomplete", "response_state": "no_semantic_output", "event_counts": {}, "incomplete_frame_bytes": 0}),
            bytes: vec![],
        };
    }
    let mut out = vec![];
    t.finish(&mut out);
    for b in out {
        bytes.extend_from_slice(&b);
        inspect_chunk(&b, &mut out_state);
    }
    let d = t.diagnostics.clone().unwrap_or(Value::Null);
    let pick = |k: &str, def: Value| if d[k].is_null() || (d[k].is_string() && d[k] == "") { def } else { d[k].clone() };
    let safe = json!({
        "attempt": attempt,
        "terminal_provenance": pick("terminal_provenance", json!("missing_terminal_diagnostics")),
        "transport_state": pick("transport_state", json!("unknown")),
        "stop_reason": d["stop_reason"],
        "stop_disposition": pick("stop_disposition", json!("terminal_incomplete")),
        "response_state": pick("response_state", json!("no_semantic_output")),
        "event_counts": pick("event_counts", json!({})),
        "incomplete_frame_bytes": pick("incomplete_frame_bytes", json!(0)),
    });
    let disp = safe["stop_disposition"].as_str().unwrap_or("").to_string();
    let prov = safe["terminal_provenance"].as_str().unwrap_or("").to_string();
    let msg = out_state.error.as_ref().and_then(|e| e["message"].as_str().map(str::to_owned));
    if disp == "retryable_protocol_failure" {
        return Attempt { kind: if prov == "invalid_tool_call" { "invalid_tool" } else { "retryable_stop" }, message: msg, diagnostics: safe, bytes: vec![] };
    }
    if ["terminal_incomplete", "terminal_refusal", "unknown_failure"].contains(&disp.as_str()) {
        let kind = if prov == "upstream_eventstream_error" {
            "upstream_error"
        } else if prov == "integrity_buffer_exceeded" || prov == "metadata_stop_reason" || prov == "message_stop_event" {
            "terminal_stop"
        } else {
            "missing_terminal"
        };
        return Attempt { kind, message: msg, diagnostics: safe, bytes: vec![] };
    }
    if out_state.error.is_some() {
        return Attempt { kind: "missing_terminal", message: msg, diagnostics: safe, bytes: vec![] };
    }
    if !out_state.has_tool_calls {
        if is_ellipsis_only(&out_state.content) || (out_state.content.trim().is_empty() && is_ellipsis_only(&out_state.reasoning)) {
            return Attempt { kind: "ellipsis", message: None, diagnostics: safe, bytes: vec![] };
        }
        if is_short_future_action(&out_state.content) {
            return Attempt { kind: "short_final", message: None, diagnostics: safe, bytes: vec![] };
        }
    }
    Attempt { kind: "complete", message: None, diagnostics: safe, bytes }
}

fn failure_sse(a: &Attempt) -> Bytes {
    let disp = a.diagnostics["stop_disposition"].as_str().unwrap_or("");
    let code = if a.diagnostics["terminal_provenance"] == "integrity_buffer_exceeded" {
        "kiro_integrity_buffer_exceeded"
    } else if a.kind == "upstream_error" {
        "kiro_upstream_eventstream_error"
    } else if disp == "terminal_refusal" {
        "kiro_terminal_refusal"
    } else if disp == "terminal_incomplete" {
        "kiro_terminal_incomplete"
    } else {
        "kiro_unknown_stop_reason"
    };
    encode_sse_error(code, a.message.as_deref().unwrap_or("Kiro stream ended with a terminal failure"), Some(a.diagnostics.clone()))
}

fn repair_body(body: &Value, kind: &str) -> Value {
    let mut b = body.clone();
    let instr = match kind {
        "tool" => "Retry the previous response because its Kiro tool_call wrapper was malformed. If you use the wrapper tool named tool_call, its input must contain a non-empty name and an arguments field.",
        "ellipsis" => "Retry the previous response because it ended with only an ellipsis. Return the complete final answer, not only ... or ….",
        "short_final" => "Retry the previous response because its final only announced a future action. Complete the check now and return the result or a concrete blocker.",
        _ => "Retry the previous incomplete Kiro response.",
    };
    let m = &mut b["conversationState"]["currentMessage"]["userInputMessage"];
    if m.is_object() {
        let c = m["content"].as_str().unwrap_or("").to_string();
        m["content"] = json!(if c.is_empty() { instr.to_string() } else { format!("{c}\n\n{instr}") });
    }
    b
}

// ===========================================================================
// executor
// ===========================================================================

pub struct Kiro;

impl Kiro {
    fn ordered_urls(&self, creds: &Value) -> Vec<String> {
        let base = self.base_urls();
        let region = creds["providerSpecificData"]["region"].as_str().map(str::trim).filter(|s| !s.is_empty()).unwrap_or("us-east-1").to_string();
        static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"([a-z]+)\.[a-z0-9-]+\.amazonaws\.com").unwrap());
        let regionalize = |u: &String| if region != "us-east-1" && u.contains("amazonaws.com") { RE.replace(u, format!("${{1}}.{region}.amazonaws.com")).into_owned() } else { u.clone() };
        let amazon: Vec<String> = base.iter().filter(|u| u.contains("amazonaws.com")).map(regionalize).collect();
        let others: Vec<String> = base.iter().filter(|u| !u.contains("amazonaws.com")).cloned().collect();
        let q: Vec<String> = amazon.iter().filter(|u| u.contains("://q.")).cloned().collect();
        let rest: Vec<String> = amazon.iter().filter(|u| !u.contains("://q.")).cloned().collect();
        if q.is_empty() { amazon.into_iter().chain(others).collect() } else { q.into_iter().chain(rest).chain(others).collect() }
    }

    async fn gate(&self, raw: Upstream, args_body: Value, model: String, creds: Value, session_id: Option<String>, client_tool: Option<String>, override_headers: Option<Value>) -> Bytes {
        let legacy = env_pos("KIRO_TOOL_CALL_REPAIR_TIMEOUT_MS", env_pos("STREAM_FIRST_CHUNK_TIMEOUT_MS", 200_000));
        let o = GateOpts {
            max_bytes: env_pos("KIRO_TOOL_CALL_REPAIR_BUFFER_MAX_BYTES", KIRO_REPAIR_BUFFER_MAX_BYTES as u64) as usize,
            ttft: Duration::from_millis(env_pos("KIRO_TOOL_CALL_REPAIR_TTFT_TIMEOUT_MS", legacy)),
            stall: Duration::from_millis(env_pos("KIRO_TOOL_CALL_REPAIR_STALL_TIMEOUT_MS", legacy)),
            repair: creds["providerSpecificData"]["kiroToolCallRepair"] != json!(false) && std::env::var("KIRO_TOOL_CALL_REPAIR").map(|v| v != "false").unwrap_or(true),
        };
        let first = read_attempt(raw, &model, &o, "initial").await;
        match first.kind {
            "complete" => return Bytes::from(first.bytes),
            "terminal_stop" | "upstream_error" => return failure_sse(&first),
            "invalid_tool" if !o.repair => return encode_sse_error("invalid_kiro_tool_call", first.message.as_deref().unwrap_or(""), Some(first.diagnostics)),
            _ => {}
        }
        let kind = match first.kind {
            "ellipsis" => Some("ellipsis"),
            "short_final" => Some("short_final"),
            "invalid_tool" => Some("tool"),
            _ => None,
        };
        let body = kind.map(|k| repair_body(&args_body, k)).unwrap_or(args_body);
        let mut creds2 = creds.clone();
        let retry = base_execute(self, ExecArgs { model: &model, body, stream: true, creds: &mut creds2, session_id, client_tool, override_headers }).await;
        let retry = match retry {
            Ok(r) => r,
            Err(e) => return encode_sse_error("kiro_integrity_retry_upstream_error", &e, Some(json!({"status": 502}))),
        };
        if !retry.response.ok() {
            let st = retry.response.status;
            let text = retry.response.text().await;
            let text: String = text.chars().take(4096.min(o.max_bytes)).collect();
            return encode_sse_error("kiro_integrity_retry_upstream_error", &if text.is_empty() { format!("Kiro integrity retry failed with HTTP {st}") } else { text }, Some(json!({"status": st})));
        }
        let second = read_attempt(retry.response, &model, &o, "retry").await;
        match second.kind {
            "complete" => return Bytes::from(second.bytes),
            "terminal_stop" | "upstream_error" => return failure_sse(&second),
            _ => {}
        }
        let code = match second.kind {
            "ellipsis" => "kiro_ellipsis_retry_failed",
            "short_final" => "kiro_short_final_retry_failed",
            "invalid_tool" => "kiro_tool_call_repair_retry_failed",
            _ => "kiro_missing_terminal_retry_failed",
        };
        encode_sse_error(code, &format!("Kiro integrity validation failed after one bounded retry: {}", second.message.clone().unwrap_or_else(|| second.kind.to_string())), Some(json!({"attempts": [first.diagnostics, second.diagnostics]})))
    }
}

#[async_trait]
impl Executor for Kiro {
    fn provider(&self) -> &str {
        "kiro"
    }
    fn build_url(&self, _m: &str, _s: bool, idx: usize, creds: &Value) -> Result<String, String> {
        let u = self.ordered_urls(creds);
        Ok(u.get(idx).or(u.first()).cloned().unwrap_or_else(|| self.config()["baseUrl"].as_str().unwrap_or("").to_string()))
    }
    fn build_headers(&self, creds: &Value, _stream: bool, url: &str, _m: &str, _b: &Value) -> Headers {
        let mut h = Headers::default();
        h.extend_obj(&self.config()["headers"]);
        h.set("Amz-Sdk-Request", "attempt=1; max=3");
        h.set("Amz-Sdk-Invocation-Id", uuid::Uuid::new_v4().to_string());
        if url.contains("://codewhisperer.") {
            h.set("X-Amz-Target", CODEWHISPERER_TARGET);
        } else {
            h.remove("X-Amz-Target");
        }
        let psd = &creds["providerSpecificData"];
        let am = psd["authMethod"].as_str().unwrap_or("");
        let api_key = crate::exec::cred_str(creds, "apiKey").or_else(|| if am == "api_key" { crate::exec::cred_str(creds, "accessToken") } else { None });
        if am == "api_key" && api_key.is_some() {
            h.set("Authorization", format!("Bearer {}", api_key.unwrap()));
            h.set("TokenType", "API_KEY");
        } else if let Some(t) = crate::exec::cred_str(creds, "accessToken") {
            h.set("Authorization", format!("Bearer {t}"));
            if am == "external_idp" {
                h.set("TokenType", "EXTERNAL_IDP");
            }
        }
        if let Some(t) = crate::exec::cred_str(creds, "accessToken") {
            h.set("x-amz-sso-bearer", t);
        }
        h.set("x-amzn-kiro-agent-mode", "spec");
        h.set("x-amzn-codewhisperer-machine-id", "kiro-desktop");
        if let Some(a) = psd["profileArn"].as_str().filter(|s| !s.is_empty()) {
            h.set("x-amzn-codewhisperer-profile-arn", a);
        }
        h
    }
    fn should_retry(&self, status: u16, idx: usize) -> bool {
        let n = self.base_urls().len().max(1);
        let has_fallback = idx + 1 < n;
        (status == 429 && has_fallback) || (has_fallback && ENDPOINT_FALLBACK.contains(&status))
    }
    fn transform_request(&self, _m: &str, body: Value, _s: bool, _c: &Value) -> Value {
        body
    }
    async fn execute(&self, args: ExecArgs<'_>) -> Result<ExecResult, String> {
        let body = args.body.clone();
        let model = args.model.to_string();
        let creds = args.creds.clone();
        let (sid, ct, oh) = (args.session_id.clone(), args.client_tool.clone(), args.override_headers.clone());
        let mut res = base_execute(self, args).await?;
        if !res.response.ok() {
            return Ok(res);
        }
        let raw = std::mem::replace(&mut res.response, Upstream::json(200, &json!({})));
        let status = raw.status;
        let this = Kiro;
        let s = async_stream::stream! {
            yield Ok::<Bytes, String>(Bytes::from_static(b": kiro-validation\n\n"));
            let fut = this.gate(raw, body, model, creds, sid, ct, oh);
            futures::pin_mut!(fut);
            let mut tick = tokio::time::interval(Duration::from_millis(10_000));
            tick.tick().await;
            loop {
                tokio::select! {
                    b = &mut fut => { yield Ok(b); break; }
                    _ = tick.tick() => { yield Ok(Bytes::from_static(b": kiro-validation\n\n")); }
                }
            }
        };
        res.response = Upstream::synthetic(status, "text/event-stream", s.boxed());
        res.response_format = Some("openai".into());
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(event: &str, payload: Value) -> Vec<u8> {
        encode_event_frame(&[(":message-type", "event"), (":event-type", event), (":content-type", "application/json")], payload.to_string().as_bytes())
    }

    fn run(frames: Vec<Vec<u8>>) -> (String, Option<Value>) {
        let mut t = EventStreamToSse::new("claude-sonnet-4.5", 1 << 20);
        let mut out = vec![];
        let all: Vec<u8> = frames.concat();
        // feed in odd-sized pieces to exercise buffering
        for c in all.chunks(7) {
            t.push(c, &mut out);
        }
        t.finish(&mut out);
        (out.iter().map(|b| String::from_utf8_lossy(b).to_string()).collect(), t.diagnostics)
    }

    #[test]
    fn eventstream_text_and_tools() {
        let (s, d) = run(vec![
            frame("assistantResponseEvent", json!({"content": "Hi <thinking>x</thinking>\nthere"})),
            frame("toolUseEvent", json!({"toolUseId": "t1", "name": "read", "input": "{\"pa"})),
            frame("toolUseEvent", json!({"toolUseId": "t1", "name": "read", "input": "th\":1}"})),
            frame("messageStopEvent", json!({"stopReason": "tool_use"})),
            frame("metricsEvent", json!({"inputTokens": 10, "outputTokens": 5})),
        ]);
        assert!(s.contains("\"content\":\"Hi there\""), "{s}");
        assert!(s.contains("{\\\"path\\\":1}"));
        assert!(s.contains("\"finish_reason\":\"tool_calls\""));
        assert!(s.contains("\"prompt_tokens\":10"));
        assert!(s.ends_with("data: [DONE]\n\n"));
        assert_eq!(d.unwrap()["stop_disposition"], "tool_use");
    }

    #[test]
    fn eventstream_errors() {
        let (s, _) = run(vec![encode_event_frame(&[(":message-type", "exception")], br#"{"message":"boom"}"#)]);
        assert!(s.contains("kiro_upstream_eventstream_error"));
        let mut bad = frame("assistantResponseEvent", json!({"content": "x"}));
        bad[9] ^= 1;
        let (s, _) = run(vec![bad]);
        assert!(s.contains("prelude CRC mismatch"));
        let (s, _) = run(vec![]);
        assert!(s.contains("without model output"));
    }

    #[test]
    fn request_translation() {
        let body = json!({"messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "mcp__a__b", "arguments": "{\"x\":1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        ], "tools": [{"type": "function", "function": {"name": "mcp__a__b", "parameters": {"type": "object", "properties": {"x": {"type": "number"}}, "additionalProperties": false}}}], "reasoning_effort": "high"});
        let rc = ReqCtx { connection_id: Some("conn-test-1".into()), ..Default::default() };
        let p = openai_to_kiro("claude-sonnet-4.5-thinking", &body, true, &rc);
        let cs = &p["conversationState"];
        assert_eq!(cs["currentMessage"]["userInputMessage"]["modelId"], "claude-sonnet-4.5");
        let hist = cs["history"].as_array().unwrap();
        assert_eq!(hist.len(), 2);
        assert!(hist[0]["userInputMessage"]["content"].as_str().unwrap().contains("<instructions>"));
        assert!(hist[0]["userInputMessage"]["content"].as_str().unwrap().contains("<thinking_mode>enabled"));
        assert_eq!(hist[1]["assistantResponseMessage"]["toolUses"][0]["name"], "mcp__a__b");
        assert_eq!(cs["currentMessage"]["userInputMessage"]["userInputMessageContext"]["toolResults"][0]["toolUseId"], "c1");
        assert_eq!(cs["currentMessage"]["userInputMessage"]["userInputMessageContext"]["tools"][0]["toolSpecification"]["name"], "mcp__a__b");
        assert!(p["profileArn"].as_str().unwrap().starts_with("arn:aws:codewhisperer"));
        assert_eq!(p["inferenceConfig"]["maxTokens"], 32000);
    }

    #[test]
    fn helpers() {
        assert!(is_short_future_action("Let me verify the deployment"));
        assert!(!is_short_future_action("Let me verify: status is green."));
        assert_eq!(normalize_stop_reason(&json!("EndTurn")).as_deref(), Some("end_turn"));
        assert_eq!(stop_disposition(Some("max_tokens"), false), "length");
        let mut st = json!({"toolNameMap": {"mcp_a_b": "mcp__a__b"}});
        let out = kiro_to_claude(Some(&json!({"id": "chatcmpl-1", "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "t", "function": {"name": "mcp_a_b", "arguments": "{}"}}]}}]})), &mut st);
        assert_eq!(out[1]["content_block"]["name"], "mcp__a__b");
    }
}

/// listAvailableApiKeyModels — validates a Kiro/Amazon Q API key.
pub async fn list_api_key_models(api_key: &str, region: &str) -> Result<Vec<Value>, String> {
    if !regex::Regex::new(r"^[a-z]{2}-[a-z]+-\d{1,2}$").unwrap().is_match(region) {
        return Err("Invalid region".into());
    }
    let r = crate::exec::http_client(None)
        .get(format!("https://q.{region}.amazonaws.com/ListAvailableModels?origin=AI_EDITOR"))
        .bearer_auth(api_key)
        .header("TokenType", "API_KEY")
        .header("Accept", "application/json")
        .header("User-Agent", "AWS-SDK-JS/3.0.0 kiro-ide/1.0.0")
        .header("X-Amz-User-Agent", "aws-sdk-js/3.0.0 kiro-ide/1.0.0")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        return Err(format!("Failed to list API-key models: {}", r.text().await.unwrap_or_default()));
    }
    let d: Value = r.json().await.map_err(|e| e.to_string())?;
    let m = d["models"].as_array().cloned().unwrap_or_default();
    if m.is_empty() {
        return Err("API key returned no available models".into());
    }
    Ok(m)
}
