//! Format translation pipeline (port of 9router translator/index.js).
//! OpenAI chat-completions is the hub: source → openai → target for requests,
//! target → openai → source for responses, with direct routes where lossless.

pub mod claude_fmt;
pub mod concerns;
pub mod gemini_fmt;
pub mod nonstream;
pub mod req;
pub mod resp;
pub mod thinking;

use serde_json::{Map, Value};

use crate::jsv::truthy;
use crate::registry::REG;
pub use req::Ctx;

pub const OPENAI: &str = "openai";
pub const OPENAI_RESPONSES: &str = "openai-responses";
pub const CLAUDE: &str = "claude";
pub const GEMINI: &str = "gemini";
pub const GEMINI_CLI: &str = "gemini-cli";
pub const VERTEX: &str = "vertex";
pub const ANTIGRAVITY: &str = "antigravity";
pub const KIRO: &str = "kiro";
pub const CURSOR: &str = "cursor";
pub const OLLAMA: &str = "ollama";
pub const COMMANDCODE: &str = "commandcode";

/// Everything a request translation needs beyond the body.
#[derive(Default, Clone)]
pub struct ReqCtx {
    pub provider: String,
    pub api_key: Option<String>,
    pub connection_id: Option<String>,
    pub headers: Value,
    pub strip: Vec<String>,
    pub ctx: Ctx,
    /// Connection providerSpecificData (Kiro profile ARN / auth method, ...).
    pub psd: Value,
}

pub struct Translated {
    pub body: Value,
    pub tool_name_map: Option<Map<String, Value>>,
    pub custom_tool_names: Vec<String>,
    pub session_id: String,
}

fn client_last_role(body: &Value) -> Option<String> {
    if let Some(m) = body["messages"].as_array() {
        return m.last().and_then(|x| x["role"].as_str()).map(str::to_owned);
    }
    let items = body["contents"].as_array().or_else(|| body["input"].as_array())?;
    let role = items.last()?["role"].as_str()?;
    (role == "assistant" || role == "model").then(|| "assistant".to_string())
}

fn to_openai(source: &str, model: &str, body: &Value, stream: bool, custom: &mut Vec<String>) -> Option<Value> {
    Some(match source {
        CLAUDE => req::claude_to_openai(model, body, stream),
        GEMINI | GEMINI_CLI => req::gemini_to_openai(model, body, stream),
        ANTIGRAVITY => req::antigravity_to_openai(model, body, stream),
        OPENAI_RESPONSES => {
            let (b, c) = req::responses_to_openai(body);
            custom.extend(c);
            b
        }
        _ => return None,
    })
}

fn from_openai(target: &str, model: &str, body: &Value, stream: bool, rc: &ReqCtx) -> Option<Value> {
    Some(match target {
        CLAUDE => req::openai_to_claude(model, body, stream),
        GEMINI => req::openai_to_gemini(model, body, &rc.ctx),
        GEMINI_CLI => {
            let g = req::openai_to_gemini_cli(model, body, &rc.ctx);
            req::wrap_cloud_code_envelope(model, &g, &rc.ctx, false)
        }
        ANTIGRAVITY => req::openai_to_antigravity(model, body, stream, &rc.ctx),
        VERTEX => req::openai_to_vertex(model, body, &rc.ctx),
        OPENAI_RESPONSES => req::openai_to_responses(model, body),
        OLLAMA => req::openai_to_ollama(model, body, stream),
        KIRO => crate::providers::kiro::openai_to_kiro(model, body, stream, rc),
        CURSOR => crate::providers::cursor::openai_to_cursor(model, body, stream, rc),
        COMMANDCODE => crate::providers::commandcode::openai_to_commandcode(model, body, stream, rc),
        _ => return None,
    })
}

/// translateRequest(sourceFormat, targetFormat, model, body, stream, credentials, provider, ...)
pub fn translate_request(source: &str, target: &str, model: &str, body: &Value, stream: bool, rc: &mut ReqCtx) -> Translated {
    let mut result = body.clone();
    let last_role = client_last_role(&result);
    concerns::strip_content_types(&mut result, &rc.strip);
    concerns::normalize_thinking_config(&mut result);
    concerns::ensure_tool_call_ids(&mut result);
    if target != KIRO {
        concerns::fix_missing_tool_responses(&mut result);
    }
    let intent = thinking::capture_thinking(&result);
    let session_id = crate::session::resolve_session_id(&rc.headers, &result, rc.connection_id.as_deref(), target);
    rc.ctx.client_session_id = Some(session_id.clone());
    let mut custom = vec![];

    if source != target {
        if source == CLAUDE && target == KIRO {
            result = crate::providers::kiro::claude_to_kiro(model, &result, stream, rc);
        } else {
            if source != OPENAI {
                if let Some(r) = to_openai(source, model, &result, stream, &mut custom) {
                    result = r;
                }
            }
            if target != OPENAI {
                if let Some(r) = from_openai(target, model, &result, stream, rc) {
                    result = r;
                }
            }
        }
    }

    let kiro_mapped = target == KIRO && (source == OPENAI || source == CLAUDE);
    if !kiro_mapped {
        thinking::apply_thinking(target, model, &mut result, Some(&rc.provider), intent.as_ref());
    }
    let transport = REG.transport(&rc.provider);
    if target == OPENAI {
        concerns::filter_to_openai_format(&mut result, transport["quirks"]["preserveCacheControl"] == Value::Bool(true));
    }
    if target == CLAUDE {
        let key = rc.api_key.clone();
        result = claude_fmt::prepare_claude_request(result, &rc.provider, key.as_deref(), Some(&session_id), &rc.headers, rc.connection_id.as_deref());
        if let Some(m) = result["messages"].as_array().cloned() {
            result["messages"] = Value::Array(claude_fmt::ensure_trailing_user_turn(m, last_role.as_deref()));
        }
    }
    let mut tool_name_map = None;
    if let Some(Value::Object(m)) = result.as_object_mut().and_then(|o| o.shift_remove("_toolNameMap")) {
        tool_name_map = Some(m);
    }
    if transport["quirks"]["cloakToolsOnOAuth"] == Value::Bool(true) && rc.api_key.as_deref().map(|k| k.contains("sk-ant-oat")).unwrap_or(false) {
        let (b, m) = crate::cloak::cloak_claude_tools(result);
        result = b;
        if m.is_some() {
            tool_name_map = m;
        }
    }
    Translated { body: result, tool_name_map, custom_tool_names: custom, session_id }
}

fn has_response_translator(target: &str) -> bool {
    matches!(target, CLAUDE | GEMINI | GEMINI_CLI | ANTIGRAVITY | VERTEX | OPENAI_RESPONSES | KIRO | CURSOR | OLLAMA | COMMANDCODE)
}

fn target_to_openai(target: &str, chunk: Option<&Value>, state: &mut Value) -> Vec<Value> {
    match target {
        CLAUDE => chunk.map(|c| resp::claude_to_openai(c, state)).unwrap_or_default(),
        GEMINI | GEMINI_CLI | ANTIGRAVITY | VERTEX => chunk.map(|c| resp::gemini_to_openai(c, state)).unwrap_or_default(),
        OPENAI_RESPONSES => resp::responses_to_openai(chunk, state),
        OLLAMA => chunk.map(|c| resp::ollama_to_openai(c, state)).unwrap_or_default(),
        KIRO => crate::providers::kiro::kiro_to_openai(chunk, state),
        CURSOR => chunk.map(|c| crate::providers::cursor::cursor_to_openai(c, state)).unwrap_or_default(),
        COMMANDCODE => crate::providers::commandcode::commandcode_to_openai(chunk, state),
        _ => chunk.cloned().into_iter().collect(),
    }
}

fn openai_to_source(source: &str, chunk: Option<&Value>, state: &mut Value) -> Option<Vec<Value>> {
    Some(match source {
        CLAUDE => chunk.map(|c| resp::openai_to_claude(c, state)).unwrap_or_default(),
        OPENAI_RESPONSES => resp::openai_to_responses(chunk, state),
        ANTIGRAVITY => chunk.map(|c| resp::openai_to_antigravity(c, state)).unwrap_or_default(),
        _ => return None,
    })
}

fn restore(mut items: Vec<Value>, state: &Value) -> Vec<Value> {
    if let Some(map) = state["renamedToolNames"].as_object() {
        for it in items.iter_mut() {
            crate::providers::opencode::restore_tool_names(it, map);
        }
    }
    items
}

/// translateResponse(targetFormat, sourceFormat, chunk, state). `None` = flush.
pub fn translate_response(target: &str, source: &str, chunk: Option<&Value>, state: &mut Value) -> Vec<Value> {
    if source == target {
        let map = state["toolNameMap"].as_object().cloned();
        let out: Vec<Value> = chunk.map(|c| crate::cloak::decloak_stream_chunk(c.clone(), map.as_ref())).into_iter().collect();
        return restore(out, state);
    }
    if target == KIRO && source == CLAUDE {
        let out = crate::providers::kiro::kiro_to_claude(chunk, state);
        return restore(out, state);
    }
    let mut results: Vec<Option<Value>> = vec![chunk.cloned()];
    if target != OPENAI && has_response_translator(target) {
        results = target_to_openai(target, chunk, state).into_iter().map(Some).collect();
    }
    if source != OPENAI {
        let mut fin = vec![];
        let mut handled = false;
        for r in &results {
            if let Some(v) = openai_to_source(source, r.as_ref(), state) {
                handled = true;
                fin.extend(v);
            }
        }
        if handled || results.is_empty() {
            return restore(fin, state);
        }
    }
    restore(results.into_iter().flatten().collect(), state)
}

/// Whether a chunk carries anything worth sending (streamHelpers.hasValuableContent).
pub fn has_valuable_content(chunk: &Value, format: &str) -> bool {
    if format == OPENAI && chunk["choices"][0]["delta"].is_object() {
        let d = &chunk["choices"][0]["delta"];
        return (d["content"].as_str().map(|s| !s.is_empty()).unwrap_or(false))
            || (d["reasoning_content"].as_str().map(|s| !s.is_empty()).unwrap_or(false))
            || d["tool_calls"].as_array().map(|a| !a.is_empty()).unwrap_or(false)
            || truthy(&chunk["choices"][0]["finish_reason"])
            || truthy(&d["role"]);
    }
    if format == CLAUDE && chunk["type"] == "content_block_delta" {
        let d = &chunk["delta"];
        let any = |k: &str| d[k].as_str().map(|s| !s.is_empty()).unwrap_or(false);
        return any("text") || any("thinking") || any("partial_json");
    }
    true
}

/// formatSSE(data, sourceFormat)
pub fn format_sse(data: &Value, source: &str) -> String {
    if data.is_null() {
        return "data: null\n\n".into();
    }
    if data["done"] == Value::Bool(true) {
        return "data: [DONE]\n\n".into();
    }
    if truthy(&data["event"]) && truthy(&data["data"]) {
        return format!("event: {}\ndata: {}\n\n", crate::jsv::js_string(&data["event"]), clean_usage(&data["data"]));
    }
    let d = clean_usage(data);
    if source == CLAUDE && truthy(&d["type"]) {
        return format!("event: {}\ndata: {}\n\n", crate::jsv::js_string(&d["type"]), d);
    }
    format!("data: {d}\n\n")
}

fn clean_usage(v: &Value) -> Value {
    let mut out = v.clone();
    if out.is_object() {
        if out.get("usage").map(|u| u.is_null()).unwrap_or(false) {
            crate::jsv::del(&mut out, "usage");
        } else if out["usage"].is_object() && out["usage"].get("perf_metrics").map(|p| p.is_null()).unwrap_or(false) {
            crate::jsv::del(&mut out["usage"], "perf_metrics");
        }
        if out["response"].is_object() {
            out["response"] = clean_usage(&out["response"]);
        }
    }
    out
}
