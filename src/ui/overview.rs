use axum::extract::{Query, State};
use axum::http::HeaderMap;
use maud::{Markup, html};
use serde_json::{Value, json};

use super::{Flash, Nav, ago, bar_chart, base_url, copy_button, fmt_cost, fmt_num, live_pill, page_head, page_live, status_badge};
use crate::AppState;
use crate::db::{Db, now};

pub fn stats(db: &Db) -> Markup {
    let day = db.totals_since(now() - 86_400);
    let failed = day.requests - day.ok;
    let success = if day.requests > 0 { day.ok as f64 * 100.0 / day.requests as f64 } else { 100.0 };
    html! {
        div.stat {
            div.label { "Requests · 24h" }
            div.value data-k="req" { (fmt_num(day.requests)) }
            div.sub { span data-k="succ" { (format!("{success:.1}%")) } " success" @if failed > 0 { " · " span.down { (fmt_num(failed)) " failed" } } }
        }
        div.stat {
            div.label { "Input tokens" }
            div.value data-k="in" { (fmt_num(day.prompt_tokens)) }
            div.sub { "prompt" }
        }
        div.stat {
            div.label { "Output tokens" }
            div.value data-k="out" { (fmt_num(day.completion_tokens)) }
            div.sub { "completion" }
        }
        div.stat {
            div.label { "Est. cost" }
            div.value data-k="cost" { (fmt_cost(day.cost)) }
            div.sub { "at list prices" }
        }
    }
}

pub fn chart(db: &Db) -> Markup {
    let hist = db.histogram(24, 3600);
    bar_chart(&hist, 4, |ts| super::datetime(ts)[11..16].to_string())
}

pub fn recent(db: &Db, base: &str) -> Markup {
    let recent = db.recent_usage(10, 0);
    html! {
        @if recent.is_empty() {
            div.empty { "No traffic yet. Point a client at " code { (base) "/v1" } " and requests appear here as they happen." }
        } @else {
            div.table-wrap { table {
                thead { tr { th { "Model" } th { "Routed to" } th.num { "Tokens" } th.num { "Latency" } th { "Status" } th.num { "When" } } }
                tbody { @for r in &recent { tr data-id=(format!("{}-{}-{}", r.ts, r.latency_ms, r.requested_model)) title=[r.error.as_deref()] {
                    td.mono style="white-space:nowrap" { (r.requested_model) }
                    td.small { div { (r.connection) } div.mono.muted style="font-size:11px" { (r.model) } }
                    td.num { (fmt_num(r.prompt_tokens + r.completion_tokens)) }
                    td.num.muted { (fmt_ms(r.latency_ms)) }
                    td { (status_badge(r.status)) }
                    td.num.small.muted style="white-space:nowrap" { (ago(r.ts)) }
                } } }
            } }
        }
    }
}

pub fn fmt_ms(ms: i64) -> String {
    if ms >= 10_000 { format!("{:.0}s", ms as f64 / 1000.0) } else if ms >= 1000 { format!("{:.1}s", ms as f64 / 1000.0) } else { format!("{ms}ms") }
}

/// Connected accounts per provider, with their state.
pub fn health(db: &Db) -> Markup {
    let conns = db.list_connections();
    let mut rows: Vec<(String, usize, usize, usize)> = vec![];
    for c in &conns {
        let p = c["provider"].as_str().unwrap_or("").to_string();
        let name = super::providers::pinfo(db, &p).map(|i| i.name).unwrap_or_else(|| p.clone());
        let active = c["isActive"] != json!(false);
        let failing = active && (matches!(c["testStatus"].as_str(), Some("unavailable" | "error" | "expired")) || has_lock(c));
        match rows.iter_mut().find(|r| r.0 == name) {
            Some(r) => {
                r.1 += 1;
                r.2 += active as usize;
                r.3 += failing as usize;
            }
            None => rows.push((name, 1, active as usize, failing as usize)),
        }
    }
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    html! {
        @if rows.is_empty() {
            p.small.muted { "No accounts connected. " a href="/dashboard/providers" { "Connect a provider" } "." }
        } @else {
            div.table-wrap { table { tbody { @for (name, total, active, failing) in &rows { tr {
                td { (name) }
                td.num.small.muted { (active) "/" (total) " active" }
                td style="text-align:right" {
                    @if *active == 0 { span.badge { "off" } }
                    @else if *failing > 0 { span.badge.warn { span.dot {} (failing) " cooling down" } }
                    @else { span.badge.ok { span.dot {} "healthy" } }
                }
            } } } } }
        }
    }
}

fn has_lock(c: &Value) -> bool {
    let now = crate::jsv::now_ms();
    c.as_object().is_some_and(|o| {
        o.iter().any(|(k, v)| k.starts_with(crate::chat::accounts::MODEL_LOCK_PREFIX) && v.as_str().and_then(crate::jsv::parse_iso_ms).is_some_and(|t| t > now))
    })
}

pub async fn page(State(state): State<AppState>, headers: HeaderMap, Query(flash): Query<Flash>) -> Markup {
    let db = &state.db;
    let combos = db.list_combos();
    let base = base_url(&headers);
    let example_model = match combos.first() {
        Some(c) => c.name.clone(),
        None => crate::api::models::build_models_list(db, &["llm"], true).await.into_iter().find_map(|m| m["id"].as_str().map(str::to_owned)).unwrap_or_else(|| "openrouter/openai/gpt-5".into()),
    };
    let needs_key = crate::auth::api_key_required(&state);
    let key_hint = if needs_key { "<your FastRouter API key>" } else { "any-value" };
    let cc = format!("export ANTHROPIC_BASE_URL={base}\nexport ANTHROPIC_AUTH_TOKEN={key_hint}\nexport ANTHROPIC_MODEL={example_model}");
    let oa = format!("OPENAI_BASE_URL={base}/v1\nOPENAI_API_KEY={key_hint}\nmodel = \"{example_model}\"");
    let curl = format!("curl {base}/v1/chat/completions \\\n  -H 'Authorization: Bearer {key_hint}' \\\n  -H 'Content-Type: application/json' \\\n  -d '{{\"model\":\"{example_model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}]}}'");

    let body = html! {
        (page_head("Overview", "", html! { (live_pill()) a.btn href="/dashboard/providers" { "Add provider" } }))

        @if db.list_connections().is_empty() && combos.is_empty() {
            div.card style="margin-bottom:14px" {
                h2 { "Get started" }
                ol.small style="margin:0;padding-left:18px;color:var(--text-2);line-height:1.9" {
                    li { a href="/dashboard/providers" { "Connect a provider" } " — sign in with a subscription, paste an API key, or use a free one." }
                    li { a href="/dashboard/combos" { "Create a combo" } " — an ordered fallback chain like " code { "premium → cheap → free" } "." }
                    li { "Point your tool at " code { (base) "/v1" } " and use the combo name as the model." }
                }
            }
        }

        div.stats data-region="stats" { (stats(db)) }

        div.split style="margin-top:14px" {
            div.stack {
                div.card {
                    div.card-head { h2 { "Requests per hour" } a href="/dashboard/usage" { "Usage →" } }
                    div data-region="chart" { (chart(db)) }
                }
                div.card {
                    div.card-head { h2 { "Recent requests" } a href="/dashboard/usage" { "All logs →" } }
                    div data-region="recent" { (recent(db, &base)) }
                }
            }
            div.stack {
                div.card {
                    div.card-head { h2 { "Endpoint" } @if needs_key { span.badge.warn { "key required" } } @else { span.badge { "key optional" } } }
                    div.copy-row { code.endpoint { (base) "/v1" } (copy_button(&format!("{base}/v1"))) }
                    p.small.muted style="margin:10px 0 0" { "OpenAI, Anthropic, Responses, Gemini and Ollama clients all work against this URL. " a href="/dashboard/keys" { "Manage keys" } }
                }
                div.card {
                    div.card-head { h2 { "Accounts" } a href="/dashboard/providers" { "Providers →" } }
                    div data-region="health" { (health(db)) }
                }
                div.card {
                    h2 { "Connect a client" }
                    div.snippet { h3 { "Claude Code" } pre.code { (cc) } }
                    div.snippet { h3 { "OpenAI-compatible tools" } pre.code { (oa) } }
                    div.snippet { h3 { "curl" } pre.code { (curl) } }
                }
            }
        }
    };
    page_live("Overview", Nav::Overview, &flash, Some("/dashboard/live?view=overview".into()), body)
}
