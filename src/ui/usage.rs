use axum::extract::{Query, State};
use axum::response::Response;
use maud::{Markup, html};
use serde::Deserialize;

use super::{
    Flash, Nav, bar_chart, datetime, fmt_cost, fmt_num, page as layout, page_head, redirect_err,
    redirect_ok, status_badge,
};
use crate::AppState;
use crate::db::{GroupRow, now};

#[derive(Deserialize, Default)]
pub struct UsageQuery {
    range: Option<String>,
    page: Option<i64>,
    ok: Option<String>,
    err: Option<String>,
}

const PAGE_SIZE: i64 = 50;

fn group_table(title: &str, rows: &[GroupRow]) -> Markup {
    html! {
        div.card {
            h2 { (title) }
            @if rows.is_empty() { div.empty { "No data" } } @else {
                div.table-wrap { table {
                    thead { tr { th { (title.trim_start_matches("By ")) } th.num { "Req" } th.num { "OK %" } th.num { "In" } th.num { "Out" } th.num { "Avg ms" } th.num { "Est. cost" } } }
                    tbody { @for r in rows { tr {
                        td.mono.small { (r.key) }
                        td.num { (fmt_num(r.requests)) }
                        td.num { (format!("{:.0}", r.ok as f64 * 100.0 / r.requests.max(1) as f64)) }
                        td.num { (fmt_num(r.prompt_tokens)) }
                        td.num { (fmt_num(r.completion_tokens)) }
                        td.num { (format!("{:.0}", r.avg_latency)) }
                        td.num { (fmt_cost(r.cost)) }
                    } } }
                } }
            }
        }
    }
}

pub async fn page(State(state): State<AppState>, Query(q): Query<UsageQuery>) -> Markup {
    let range = q.range.clone().unwrap_or_else(|| "24h".into());
    let (secs, buckets, bucket_size, label_every) = match range.as_str() {
        "7d" => (7 * 86_400, 28, 6 * 3600, 4),
        "30d" => (30 * 86_400, 30, 86_400, 5),
        _ => (86_400, 24, 3600, 3),
    };
    let since = now() - secs;
    let db = &state.db;
    let totals = db.totals_since(since);
    let hist = db.histogram(buckets, bucket_size);
    let by_model = db.group_since("model", since);
    let by_provider = db.group_since("provider", since);
    let by_key = db.group_since("api_key", since);
    let page_no = q.page.unwrap_or(0).max(0);
    let logs = db.recent_usage(PAGE_SIZE, page_no * PAGE_SIZE);
    let total_logs = db.usage_count();
    let flash = Flash {
        ok: q.ok.clone(),
        err: q.err.clone(),
    };
    let fmt_label = move |ts: i64| {
        let dt = datetime(ts);
        if bucket_size >= 86_400 {
            dt[5..10].to_string()
        } else if bucket_size > 3600 {
            dt[5..13].to_string() + "h"
        } else {
            dt[11..16].to_string()
        }
    };

    let body = html! {
        (page_head("Usage", "Token usage and estimated pay-as-you-go cost across all providers.", html! {
            div.tabs style="margin:0;border:0" {
                @for (r, label) in [("24h", "24 hours"), ("7d", "7 days"), ("30d", "30 days")] {
                    a href=(format!("/dashboard/usage?range={r}")) class=[(range == r).then_some("active")] { (label) }
                }
            }
        }))
        div.grid.g4 {
            div.card.stat { div.label { "Requests" } div.value { (fmt_num(totals.requests)) } div.sub { (fmt_num(totals.requests - totals.ok)) " failed attempts" } }
            div.card.stat { div.label { "Input tokens" } div.value { (fmt_num(totals.prompt_tokens)) } }
            div.card.stat { div.label { "Output tokens" } div.value { (fmt_num(totals.completion_tokens)) } }
            div.card.stat { div.label { "Est. API cost" } div.value { (fmt_cost(totals.cost)) } div.sub { "at list prices" } }
        }
        div.card style="margin-top:16px" {
            h2 { "Requests" }
            (bar_chart(&hist, label_every, fmt_label))
        }
        div.grid.g2 style="margin-top:16px" {
            (group_table("By model", &by_model))
            (group_table("By provider", &by_provider))
        }
        div style="margin-top:16px" { (group_table("By API key", &by_key)) }

        div.card style="margin-top:16px" {
            div.card-head {
                h2 { "Request log" }
                form.inline method="post" action="/dashboard/usage/clear" { button.btn.sm.danger type="submit" { "Clear logs" } }
            }
            @if logs.is_empty() { div.empty { "No requests logged." } } @else {
                div.table-wrap { table {
                    thead { tr { th { "Time (UTC)" } th { "Requested" } th { "Connection" } th { "Upstream model" } th.num { "In" } th.num { "Out" } th.num { "ms" } th { "Status" } } }
                    tbody { @for r in &logs { tr title=[r.error.as_deref()] {
                        td.small.mono { (datetime(r.ts)) }
                        td.mono.small { (r.requested_model) @if r.stream { " " span.badge { "stream" } } }
                        td.small { (r.connection) }
                        td.mono.small { (r.model) }
                        td.num { (fmt_num(r.prompt_tokens)) }
                        td.num { (fmt_num(r.completion_tokens)) }
                        td.num { (r.latency_ms) }
                        td {
                            (status_badge(r.status))
                            @if let Some(e) = &r.error { div.small.muted style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" { (e) } }
                        }
                    } } }
                } }
                div.pager {
                    span.small.muted { "Showing " (page_no * PAGE_SIZE + 1) "–" (page_no * PAGE_SIZE + logs.len() as i64) " of " (total_logs) }
                    div.actions {
                        @if page_no > 0 { a.btn.sm href=(format!("/dashboard/usage?range={range}&page={}", page_no - 1)) { "← Newer" } }
                        @if (page_no + 1) * PAGE_SIZE < total_logs { a.btn.sm href=(format!("/dashboard/usage?range={range}&page={}", page_no + 1)) { "Older →" } }
                    }
                }
            }
        }
    };
    layout("Usage", Nav::Usage, &flash, body)
}

pub async fn clear(State(state): State<AppState>) -> Response {
    match state.db.clear_usage() {
        Ok(()) => redirect_ok("/dashboard/usage", "Usage logs cleared"),
        Err(e) => redirect_err("/dashboard/usage", &e.to_string()),
    }
}
