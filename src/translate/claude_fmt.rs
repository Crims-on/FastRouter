//! Claude request shaping (port of translator/formats/claude.js).

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};

use crate::caps::caps_for;
use crate::cloak::{apply_cloaking, is_valid_claude_signature};
use crate::consts::{DEFAULT_MAX_TOKENS, sig_claude};
use crate::jsv::{del, omit, truthy};
use crate::registry::{REG, is_deepseek_model};

fn cache_5m() -> Value {
    json!({"type": "ephemeral"})
}
fn cache_1h() -> Value {
    json!({"type": "ephemeral", "ttl": "1h"})
}

pub fn last_cacheable_tool_index(tools: &Value) -> Option<usize> {
    let a = tools.as_array()?;
    (0..a.len()).rev().find(|i| a[*i]["defer_loading"] != json!(true))
}

fn is_contentful_block(b: &Value) -> bool {
    if b.is_null() {
        return false;
    }
    match b["type"].as_str().unwrap_or("") {
        "text" => b["text"].as_str().map(|t| !t.trim().is_empty()).unwrap_or(false),
        "tool_use" | "tool_result" | "image" | "document" | "container_upload" => true,
        _ => false,
    }
}

pub fn has_valid_content(msg: &Value) -> bool {
    match &msg["content"] {
        Value::String(s) => !s.trim().is_empty(),
        Value::Object(_) => is_contentful_block(&msg["content"]),
        Value::Array(a) => a.iter().any(is_contentful_block),
        _ => false,
    }
}

fn normalize_message_content(msg: &mut Value) {
    if msg["content"].is_object() {
        let mut c = msg["content"].take();
        del(&mut c, "cache_control");
        msg["content"] = json!([c]);
    }
}

fn count_cache_control_blocks(body: &Value) -> usize {
    let mut n = 0;
    for b in body["system"].as_array().into_iter().flatten() {
        if truthy(&b["cache_control"]) {
            n += 1;
        }
    }
    for t in body["tools"].as_array().into_iter().flatten() {
        if truthy(&t["cache_control"]) {
            n += 1;
        }
    }
    for m in body["messages"].as_array().into_iter().flatten() {
        match &m["content"] {
            Value::Array(a) => n += a.iter().filter(|b| truthy(&b["cache_control"])).count(),
            Value::Object(_) if truthy(&m["content"]["cache_control"]) => n += 1,
            _ => {}
        }
    }
    n
}

fn cap_cache_control_blocks(body: &mut Value) {
    // Collect (location) of marked blocks in document order; head anchors are the
    // last system block and the last cacheable tool.
    #[derive(Clone, Copy)]
    enum Loc {
        Sys(usize),
        Tool(usize),
        Msg(usize, usize),
    }
    let sys_len = body["system"].as_array().map(|a| a.len()).unwrap_or(0);
    let last_tool = last_cacheable_tool_index(&body["tools"]);
    let mut marked: Vec<(Loc, bool)> = vec![];
    for (i, b) in body["system"].as_array().into_iter().flatten().enumerate() {
        if truthy(&b["cache_control"]) {
            marked.push((Loc::Sys(i), sys_len > 0 && i == sys_len - 1));
        }
    }
    for (i, t) in body["tools"].as_array().into_iter().flatten().enumerate() {
        if truthy(&t["cache_control"]) {
            marked.push((Loc::Tool(i), Some(i) == last_tool));
        }
    }
    for (mi, m) in body["messages"].as_array().into_iter().flatten().enumerate() {
        for (bi, b) in m["content"].as_array().into_iter().flatten().enumerate() {
            if truthy(&b["cache_control"]) {
                marked.push((Loc::Msg(mi, bi), false));
            }
        }
    }
    let head = marked.iter().filter(|m| m.1).count();
    let rest: Vec<Loc> = marked.iter().filter(|m| !m.1).map(|m| m.0).collect();
    let keep = 4usize.saturating_sub(head);
    let drop_n = rest.len().saturating_sub(keep);
    for loc in rest.into_iter().take(drop_n) {
        let target = match loc {
            Loc::Sys(i) => &mut body["system"][i],
            Loc::Tool(i) => &mut body["tools"][i],
            Loc::Msg(m, b) => &mut body["messages"][m]["content"][b],
        };
        del(target, "cache_control");
    }
}

/// fixToolUseOrdering: drop text after tool_use; merge consecutive same-role messages.
pub fn fix_tool_use_ordering(messages: Vec<Value>) -> Vec<Value> {
    if messages.len() <= 1 {
        return messages;
    }
    let mut messages = messages;
    for msg in messages.iter_mut() {
        if msg["role"] != "assistant" {
            continue;
        }
        let Some(content) = msg["content"].as_array() else { continue };
        if !content.iter().any(|b| b["type"] == "tool_use") {
            continue;
        }
        let mut out = vec![];
        let mut found = false;
        for b in content {
            let ty = b["type"].as_str().unwrap_or("");
            if ty == "tool_use" {
                found = true;
                out.push(b.clone());
            } else if ty == "thinking" || ty == "redacted_thinking" || !found {
                out.push(b.clone());
            }
        }
        msg["content"] = Value::Array(out);
    }
    let as_blocks = |c: &Value| -> Vec<Value> {
        match c {
            Value::Array(a) => a.clone(),
            other => vec![json!({"type": "text", "text": other})],
        }
    };
    let mut merged: Vec<Value> = vec![];
    for msg in messages {
        if let Some(last) = merged.last_mut() {
            if last["role"] == msg["role"] {
                let lc = as_blocks(&last["content"]);
                let mc = as_blocks(&msg["content"]);
                let mut tr: Vec<Value> = lc.iter().filter(|b| b["type"] == "tool_result").cloned().collect();
                tr.extend(mc.iter().filter(|b| b["type"] == "tool_result").cloned());
                let mut other: Vec<Value> = lc.iter().filter(|b| b["type"] != "tool_result").cloned().collect();
                other.extend(mc.iter().filter(|b| b["type"] != "tool_result").cloned());
                tr.extend(other);
                last["content"] = Value::Array(tr);
                continue;
            }
        }
        let content = as_blocks(&msg["content"]);
        merged.push(json!({"role": msg["role"], "content": content}));
    }
    merged
}

fn handles_thinking_blocks(provider: &str) -> bool {
    provider == "claude" || provider.starts_with("anthropic-compatible") || provider == "deepseek"
}

fn thinking_placeholder(provider: &str, unsigned: bool) -> Value {
    let mut b = json!({"type": "thinking", "thinking": "."});
    if provider != "deepseek" && !unsigned {
        b["signature"] = json!(sig_claude());
    }
    b
}

static SRVTOOLU: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^srvtoolu_[a-zA-Z0-9_]+$").unwrap());

fn foreign_server_tool_use(b: &Value) -> bool {
    b["type"] == "server_tool_use" && !SRVTOOLU.is_match(&crate::jsv::sn(&b["id"]))
}

const TRAILING_USER_PLACEHOLDER: &str = "Continue.";

pub fn ensure_trailing_user_turn(mut messages: Vec<Value>, original_last_role: Option<&str>) -> Vec<Value> {
    if original_last_role == Some("assistant") {
        return messages;
    }
    if messages.last().map(|m| m["role"] != "assistant").unwrap_or(true) {
        return messages;
    }
    messages.push(json!({"role": "user", "content": [{"type": "text", "text": TRAILING_USER_PLACEHOLDER}]}));
    messages
}

/// normalizeClaudePassthrough(body, model)
pub fn normalize_claude_passthrough(body: &mut Value, model: &str) {
    if !body.is_object() {
        return;
    }
    let haiku = model.to_lowercase().contains("haiku");
    if body["thinking"]["type"] == "adaptive" && haiku {
        body["thinking"] = json!({"type": "enabled", "budget_tokens": 10000});
    }
    if haiku && !body["output_config"]["effort"].is_null() && body["output_config"].is_object() {
        del(&mut body["output_config"], "effort");
        if body["output_config"].as_object().map(|o| o.is_empty()).unwrap_or(false) {
            del(body, "output_config");
        }
    }
    let original_last_role = body["messages"].as_array().and_then(|m| m.last()).and_then(|m| m["role"].as_str()).map(str::to_owned);
    if let Some(msgs) = body["messages"].as_array_mut() {
        for m in msgs.iter_mut() {
            normalize_message_content(m);
        }
    }
    if let Some(msgs) = body["messages"].as_array().cloned() {
        let mut out: Vec<Value> = vec![];
        for msg in msgs {
            if msg["role"] != "system" {
                out.push(msg);
                continue;
            }
            let text = match &msg["content"] {
                Value::String(s) => s.clone(),
                Value::Array(a) => a
                    .iter()
                    .map(|b| match b {
                        Value::String(s) => s.clone(),
                        _ => b["text"].as_str().unwrap_or("").to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => String::new(),
            };
            if text.trim().is_empty() {
                continue;
            }
            let block = json!({"type": "text", "text": text});
            if let Some(prev) = out.last_mut() {
                if prev["role"] == "user" {
                    let mut content = match &prev["content"] {
                        Value::String(s) => vec![json!({"type": "text", "text": s})],
                        Value::Array(a) => a.clone(),
                        _ => vec![],
                    };
                    content.push(block);
                    prev["content"] = Value::Array(content);
                    continue;
                }
            }
            out.push(json!({"role": "user", "content": [block]}));
        }
        body["messages"] = Value::Array(out);
    }

    let thinking_enabled = body["thinking"]["type"] == "enabled";
    let mut dropped: Vec<String> = vec![];
    if let Some(msgs) = body["messages"].as_array_mut() {
        for msg in msgs.iter_mut() {
            if msg["role"] != "assistant" {
                continue;
            }
            let Some(content) = msg["content"].as_array().cloned() else { continue };
            let mut has_tool_use = false;
            let mut kept_thinking = false;
            let mut kept = vec![];
            for b in content {
                let ty = b["type"].as_str().unwrap_or("");
                if ty == "thinking" || ty == "redacted_thinking" {
                    if is_valid_claude_signature(&b["signature"]) {
                        kept_thinking = true;
                        kept.push(b);
                    }
                    continue;
                }
                if foreign_server_tool_use(&b) {
                    if !b["id"].is_null() {
                        dropped.push(crate::jsv::js_string(&b["id"]));
                    }
                    continue;
                }
                if ty == "tool_use" {
                    has_tool_use = true;
                }
                kept.push(b);
            }
            if thinking_enabled && !kept_thinking && has_tool_use {
                kept.insert(0, thinking_placeholder("claude", false));
            }
            msg["content"] = Value::Array(kept);
        }
    }
    if !dropped.is_empty() {
        if let Some(msgs) = body["messages"].as_array_mut() {
            for msg in msgs.iter_mut() {
                if let Some(c) = msg["content"].as_array() {
                    let kept: Vec<Value> = c
                        .iter()
                        .filter(|b| {
                            !((b["type"] == "tool_result" || b["type"] == "web_search_tool_result")
                                && dropped.contains(&crate::jsv::sn(&b["tool_use_id"])))
                        })
                        .cloned()
                        .collect();
                    if kept.len() != c.len() {
                        msg["content"] = Value::Array(kept);
                    }
                }
            }
        }
    }
    if let Some(msgs) = body["messages"].as_array().cloned() {
        let filtered: Vec<Value> = msgs
            .into_iter()
            .filter_map(|mut msg| {
                if let Some(s) = msg["content"].as_str() {
                    return (!s.trim().is_empty()).then_some(msg);
                }
                let Some(c) = msg["content"].as_array() else { return Some(msg) };
                let kept: Vec<Value> = c
                    .iter()
                    .filter(|b| !(b["type"] == "text" && crate::jsv::sn(&b["text"]).trim().is_empty()))
                    .cloned()
                    .collect();
                if kept.is_empty() {
                    return None;
                }
                msg["content"] = Value::Array(kept);
                Some(msg)
            })
            .collect();
        body["messages"] = Value::Array(ensure_trailing_user_turn(filtered, original_last_role.as_deref()));
    }
}

fn mark_last_cacheable_block(msg: &mut Value) -> bool {
    let Some(c) = msg["content"].as_array_mut() else { return false };
    for b in c.iter_mut().rev() {
        if !b.is_object() {
            continue;
        }
        if b["type"] == "thinking" || b["type"] == "redacted_thinking" {
            continue;
        }
        b["cache_control"] = cache_5m();
        return true;
    }
    false
}

fn mark_final_tool_results(body: &mut Value) -> bool {
    let count = count_cache_control_blocks(body);
    let Some(last) = body["messages"].as_array_mut().and_then(|m| m.last_mut()) else { return false };
    if last["role"] != "user" {
        return false;
    }
    let Some(c) = last["content"].as_array() else { return false };
    if !c.iter().any(|b| b["type"] == "tool_result") || c.iter().any(|b| truthy(&b["cache_control"])) {
        return false;
    }
    if count >= 4 {
        return false;
    }
    mark_last_cacheable_block(last)
}

/// anchorClaudeCache(body) — passthrough cache re-anchoring.
pub fn anchor_claude_cache(body: &mut Value) {
    if !body.is_object() {
        return;
    }
    if let Some(m) = body["messages"].as_array_mut() {
        for msg in m.iter_mut() {
            normalize_message_content(msg);
        }
    }
    if let Some(tools) = body["tools"].as_array_mut() {
        for t in tools.iter_mut() {
            if t["defer_loading"] == json!(true) {
                del(t, "cache_control");
            }
        }
    }
    if let Some(sys) = body["system"].as_array_mut() {
        let last = sys.len().saturating_sub(1);
        for (i, b) in sys.iter_mut().enumerate() {
            if !b.is_object() {
                continue;
            }
            if i == last {
                b["cache_control"] = cache_1h();
            } else {
                del(b, "cache_control");
            }
        }
    }
    let last_tool = last_cacheable_tool_index(&body["tools"]);
    if let Some(tools) = body["tools"].as_array_mut() {
        for (i, t) in tools.iter_mut().enumerate() {
            if Some(i) == last_tool {
                t["cache_control"] = cache_1h();
            } else {
                del(t, "cache_control");
            }
        }
    }
    if count_cache_control_blocks(body) >= 4 {
        cap_cache_control_blocks(body);
        return;
    }
    if let Some(msgs) = body["messages"].as_array_mut() {
        let mut anchored = false;
        for msg in msgs.iter_mut().rev() {
            let Some(c) = msg["content"].as_array_mut() else { continue };
            for b in c.iter_mut() {
                del(b, "cache_control");
            }
            if anchored || msg["role"] != "assistant" {
                continue;
            }
            anchored = mark_last_cacheable_block(msg);
        }
        if !anchored {
            for msg in msgs.iter_mut().rev() {
                if mark_last_cacheable_block(msg) {
                    break;
                }
            }
        }
    }
    mark_final_tool_results(body);
}

/// hoistToolResultImages(body)
pub fn hoist_tool_result_images(body: &mut Value) {
    let Some(msgs) = body["messages"].as_array_mut() else { return };
    for msg in msgs.iter_mut() {
        if msg["role"] != "user" {
            continue;
        }
        let Some(content) = msg["content"].as_array().cloned() else { continue };
        let mut hoisted = vec![];
        let new_content: Vec<Value> = content
            .into_iter()
            .map(|block| {
                if block["type"] != "tool_result" {
                    return block;
                }
                let Some(inner) = block["content"].as_array() else { return block };
                let images: Vec<Value> = inner.iter().filter(|c| c["type"] == "image").cloned().collect();
                if images.is_empty() {
                    return block;
                }
                let rest: Vec<Value> = inner.iter().filter(|c| c["type"] != "image").cloned().collect();
                hoisted.push(json!({"type": "text", "text": format!("[Image from tool result {}]", crate::jsv::js_string(&block["tool_use_id"]))}));
                hoisted.extend(images);
                let mut b = block.clone();
                b["content"] = if rest.is_empty() { json!([{"type": "text", "text": "(image attached below)"}]) } else { Value::Array(rest) };
                b
            })
            .collect();
        if hoisted.is_empty() {
            continue;
        }
        let mut c = new_content;
        c.extend(hoisted);
        msg["content"] = Value::Array(c);
    }
}

/// prepareClaudeRequest(body, provider, apiKey, connectionId, rawHeaders, sessionId)
pub fn prepare_claude_request(
    mut body: Value,
    provider: &str,
    api_key: Option<&str>,
    session_id: Option<&str>,
    headers: &Value,
    connection_id: Option<&str>,
) -> Value {
    let transport = REG.transport(provider);
    if transport["quirks"]["dropOutputConfig"] == json!(true) {
        del(&mut body, "output_config");
    }
    let model = body["model"].as_str().unwrap_or("").to_string();
    let caps = caps_for(Some(provider), &model);
    if let Some(off) = caps.get("thinkingOffType").as_str() {
        if body["thinking"]["type"] == "disabled" {
            body["thinking"] = json!({"type": off});
            let e = body["output_config"]["effort"].as_str();
            if e == Some("xhigh") || e == Some("max") {
                body["output_config"]["effort"] = json!("high");
            }
        }
    }
    if caps.get("forcedToolChoice") == &json!(false) {
        let t = body["tool_choice"]["type"].as_str();
        if t == Some("any") || t == Some("tool") {
            let dptu = body["tool_choice"]["disable_parallel_tool_use"].clone();
            let mut tc = json!({"type": "auto"});
            if !dptu.is_null() {
                tc["disable_parallel_tool_use"] = dptu;
            }
            body["tool_choice"] = tc;
        }
    }
    if let Some(max) = body["max_tokens"].as_i64().filter(|m| *m != 0) {
        let ceiling = caps.max_output().filter(|m| *m != 0).unwrap_or(DEFAULT_MAX_TOKENS);
        let mut max = max.min(ceiling);
        body["max_tokens"] = json!(max);
        if body["thinking"]["type"] == "enabled" {
            if let Some(budget) = body["thinking"]["budget_tokens"].as_i64().filter(|b| *b != 0) {
                if budget >= max {
                    max = (budget + 1024).min(ceiling);
                    body["max_tokens"] = json!(max);
                    if budget >= max {
                        body["thinking"]["budget_tokens"] = json!((max - 1024).max(1024));
                    }
                }
            }
        }
    }

    if let Some(sys) = body["system"].as_array().cloned() {
        let len = sys.len();
        body["system"] = Value::Array(
            sys.into_iter()
                .enumerate()
                .map(|(i, b)| {
                    let mut rest = omit(&b, &["cache_control"]);
                    if i == len - 1 {
                        rest["cache_control"] = cache_1h();
                    }
                    rest
                })
                .collect(),
        );
    }

    if let Some(msgs) = body["messages"].as_array().cloned() {
        let len = msgs.len();
        let original_last_role = msgs.last().and_then(|m| m["role"].as_str()).map(str::to_owned);
        let mut filtered = vec![];
        for (i, mut msg) in msgs.into_iter().enumerate() {
            normalize_message_content(&mut msg);
            if let Some(c) = msg["content"].as_array_mut() {
                for b in c.iter_mut() {
                    del(b, "cache_control");
                }
            }
            let final_assistant = i == len - 1 && msg["role"] == "assistant";
            if final_assistant || has_valid_content(&msg) {
                filtered.push(msg);
            }
        }
        let filtered = fix_tool_use_ordering(filtered);
        let mut filtered = ensure_trailing_user_turn(filtered, original_last_role.as_deref());

        let last_is_user = filtered.last().map(|m| m["role"] == "user").unwrap_or(false);
        let thinking_enabled = body["thinking"]["type"] == "enabled" && last_is_user;
        let deepseek_served = provider == "deepseek" || (provider == "opencode-go" && is_deepseek_model(&model));
        let mut last_assistant_done = false;
        for msg in filtered.iter_mut().rev() {
            if msg["role"] != "assistant" || !msg["content"].is_array() {
                continue;
            }
            if !last_assistant_done && msg["content"].as_array().map(|a| !a.is_empty()).unwrap_or(false) {
                if let Some(c) = msg["content"].as_array_mut() {
                    for b in c.iter_mut().rev() {
                        if b["type"] != "thinking" && b["type"] != "redacted_thinking" {
                            b["cache_control"] = cache_5m();
                            break;
                        }
                    }
                }
                last_assistant_done = true;
            }
            if handles_thinking_blocks(provider) || deepseek_served {
                let mut has_tool_use = false;
                let mut kept_thinking = false;
                let native = provider == "claude";
                let mut kept = vec![];
                for mut b in msg["content"].as_array().cloned().unwrap_or_default() {
                    let ty = b["type"].as_str().unwrap_or("").to_string();
                    if ty == "thinking" || ty == "redacted_thinking" {
                        if native {
                            if is_valid_claude_signature(&b["signature"]) {
                                kept_thinking = true;
                                kept.push(b);
                            }
                        } else if deepseek_served {
                            kept_thinking = true;
                            kept.push(b);
                        } else {
                            b["signature"] = json!(sig_claude());
                            kept_thinking = true;
                            kept.push(b);
                        }
                        continue;
                    }
                    if ty == "tool_use" {
                        has_tool_use = true;
                    }
                    kept.push(b);
                }
                if thinking_enabled && !kept_thinking && has_tool_use {
                    kept.insert(0, thinking_placeholder(provider, deepseek_served));
                }
                msg["content"] = Value::Array(kept);
            }
        }
        body["messages"] = Value::Array(filtered);
    }

    if let Some(tools) = body["tools"].as_array().cloned() {
        let mut tools = tools;
        if provider != "claude" {
            let whitelist: Option<Vec<String>> = transport["quirks"]["claudeSupportedToolTypes"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect());
            tools = tools
                .into_iter()
                .filter(|t| {
                    let ty = t["type"].as_str().unwrap_or("");
                    if !truthy(&t["type"]) || ty == "function" {
                        return true;
                    }
                    whitelist.as_ref().map(|w| w.iter().any(|x| x == ty)).unwrap_or(false)
                })
                .map(|t| {
                    if truthy(&t["function"]) {
                        let mut o = json!({"name": t["function"]["name"]});
                        if !t["function"]["description"].is_null() {
                            o["description"] = t["function"]["description"].clone();
                        }
                        if !t["function"]["parameters"].is_null() {
                            o["input_schema"] = t["function"]["parameters"].clone();
                        }
                        return o;
                    }
                    if whitelist.is_some() { t } else { omit(&t, &["type"]) }
                })
                .collect();
        }
        let last = last_cacheable_tool_index(&Value::Array(tools.clone()));
        let tools: Vec<Value> = tools
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                let mut r = omit(&t, &["cache_control"]);
                if Some(i) == last {
                    r["cache_control"] = cache_1h();
                }
                r
            })
            .collect();
        if tools.is_empty() {
            del(&mut body, "tools");
            del(&mut body, "tool_choice");
        } else {
            body["tools"] = Value::Array(tools);
        }
    }

    if provider != "claude" && !provider.starts_with("anthropic-compatible") {
        hoist_tool_result_images(&mut body);
    }
    mark_final_tool_results(&mut body);

    if (provider == "claude" || provider.starts_with("anthropic-compatible")) && api_key.map(|k| !k.is_empty()).unwrap_or(false) {
        let sid = session_id
            .map(str::to_owned)
            .unwrap_or_else(|| crate::session::resolve_session_id(headers, &body, connection_id, "claude"));
        body = apply_cloaking(body, api_key.unwrap(), Some(&sid));
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_and_merge() {
        let msgs = vec![
            json!({"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}, {"type": "text", "text": "after"}]}),
            json!({"role": "user", "content": [{"type": "text", "text": "x"}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "r"}]}),
        ];
        let out = fix_tool_use_ordering(msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["content"].as_array().unwrap().len(), 1);
        assert_eq!(out[1]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn trailing_user_added() {
        let body = json!({"model": "claude-sonnet-4-5", "max_tokens": 100, "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
            {"role": "user", "content": [{"type": "text", "text": "  "}]}
        ]});
        let out = prepare_claude_request(body, "anthropic", None, None, &json!({}), None);
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs.last().unwrap()["content"][0]["text"], "Continue.");
    }
}
