use axum::extract::{Query, State};
use axum::http::HeaderMap;
use maud::{Markup, html};

use super::{
    Flash, Nav, bar_chart, base_url, copy_button, fmt_cost, fmt_num, page as layout, page_head,
    status_badge,
};
use crate::AppState;
use crate::db::now;

pub async fn page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(flash): Query<Flash>,
) -> Markup {
    let db = &state.db;
    let day = db.totals_since(now() - 86_400);
    let conns = db.list_connections();
    let active = conns.iter().filter(|c| c.enabled).count();
    let combos = db.list_combos();
    let hist = db.histogram(24, 3600);
    let recent = db.recent_usage(8, 0);
    let base = base_url(&headers);
    let success = if day.requests > 0 {
        day.ok as f64 * 100.0 / day.requests as f64
    } else {
        100.0
    };
    let example_model = combos
        .first()
        .map(|c| c.name.clone())
        .or_else(|| {
            crate::router::available_models(db)
                .into_iter()
                .map(|m| m.0)
                .next()
        })
        .unwrap_or_else(|| "openrouter/openai/gpt-5".into());
    let needs_key = crate::auth::api_key_required(&state);
    let key_hint = if needs_key {
        "<your FastRouter API key>"
    } else {
        "any-value"
    };

    let body = html! {
        (page_head("Overview", "Your local AI gateway at a glance — last 24 hours.", html! {
            a.btn href="/dashboard/providers" { "Add provider" }
            a.btn.primary href="/dashboard/combos" { "New combo" }
        }))

        @if conns.is_empty() {
            div.card style="margin-bottom:16px;border-color:var(--accent)" {
                h2 { "Get started in three steps" }
                ol style="margin:0;padding-left:18px" {
                    li { a href="/dashboard/providers" { "Connect a provider" } " — paste an API key for OpenRouter, Anthropic, GLM, Gemini…" }
                    li { a href="/dashboard/combos" { "Create a combo" } " — an ordered fallback chain such as " code { "premium → cheap → free" } "." }
                    li { "Point your tool at " code { (base) "/v1" } " and use the combo name as the model." }
                }
            }
        }

        div.grid.g4 {
            div.card.stat { div.label { "Requests" } div.value { (fmt_num(day.requests)) } div.sub { (format!("{success:.1}% success")) } }
            div.card.stat { div.label { "Input tokens" } div.value { (fmt_num(day.prompt_tokens)) } div.sub { "prompt" } }
            div.card.stat { div.label { "Output tokens" } div.value { (fmt_num(day.completion_tokens)) } div.sub { "completion" } }
            div.card.stat { div.label { "Est. API cost" } div.value { (fmt_cost(day.cost)) } div.sub { (active) " active connections · " (combos.len()) " combos" } }
        }

        div.split style="margin-top:16px" {
            div.stack {
                div.card {
                    div.card-head { h2 { "Requests per hour" } a.small href="/dashboard/usage" { "Usage details →" } }
                    (bar_chart(&hist, 3, |ts| super::datetime(ts)[11..16].to_string()))
                }
                div.card {
                    div.card-head { h2 { "Recent requests" } a.small href="/dashboard/usage" { "All logs →" } }
                    @if recent.is_empty() {
                        div.empty { "No traffic yet. Send a request to " code { (base) "/v1/chat/completions" } "." }
                    } @else {
                        div.table-wrap { table {
                            thead { tr { th { "When" } th { "Model" } th { "Routed to" } th.num { "Tokens" } th { "Status" } } }
                            tbody { @for r in &recent { tr {
                                td.muted.small style="white-space:nowrap" { (super::ago(r.ts)) }
                                td.mono { (r.requested_model) }
                                td.small { (r.connection) span.muted { " · " (r.model) } }
                                td.num { (fmt_num(r.prompt_tokens + r.completion_tokens)) }
                                td { (status_badge(r.status)) }
                            } } }
                        } }
                    }
                }
            }
            div.stack {
                div.card {
                    h2 { "Endpoint" }
                    p.small.muted { "OpenAI- and Anthropic-compatible. Use it as the base URL in any tool." }
                    div.copy-row { code { (base) "/v1" } (copy_button(&format!("{base}/v1"))) }
                    p.small style="margin-top:10px" {
                        "API key: "
                        @if needs_key { span.badge.warn { "required" } } @else { span.badge { "optional" } }
                        " · " a href="/dashboard/keys" { "manage keys" }
                    }
                }
                div.card {
                    h2 { "Quick setup" }
                    div.snippet {
                        h3 { "Claude Code" }
                        @let cc = format!("export ANTHROPIC_BASE_URL={base}\nexport ANTHROPIC_AUTH_TOKEN={key_hint}\nexport ANTHROPIC_MODEL={example_model}\nclaude");
                        pre.code { (cc) }
                    }
                    div.snippet {
                        h3 { "OpenAI SDK / Codex / Cline / Cursor" }
                        @let oa = format!("OPENAI_BASE_URL={base}/v1\nOPENAI_API_KEY={key_hint}\nmodel: {example_model}");
                        pre.code { (oa) }
                    }
                    div.snippet {
                        h3 { "curl" }
                        @let curl = format!("curl {base}/v1/chat/completions \\\n  -H 'Authorization: Bearer {key_hint}' \\\n  -H 'Content-Type: application/json' \\\n  -d '{{\"model\":\"{example_model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}]}}'");
                        pre.code { (curl) }
                    }
                }
            }
        }
    };
    layout("Overview", Nav::Overview, &flash, body)
}
