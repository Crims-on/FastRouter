use axum::extract::{Query, State};
use axum::response::Response;
use maud::{Markup, html};
use serde::Deserialize;

use super::{Flash, Nav, bar_chart, datetime, fmt_cost, fmt_num, live_pill, page_head, page_live, redirect_err, redirect_ok, status_badge};
use crate::AppState;
use crate::db::{Db, GroupRow, now};

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
            @if rows.is_empty() { div.empty { "Nothing in this period" } } @else {
                div.table-wrap { table {
                    thead { tr { th { (title.trim_start_matches("By ")) } th.num { "Requests" } th.num { "Success" } th.num { "Tokens" } th.num { "Latency" } th.num { "Cost" } } }
                    tbody { @for r in rows { tr {
                        td.mono.small.key title=(r.key) { (r.key) }
                        td.num { (fmt_num(r.requests)) }
                        td.num { (format!("{:.0}%", r.ok as f64 * 100.0 / r.requests.max(1) as f64)) }
                        td.num title=(format!("{} in · {} out", fmt_num(r.prompt_tokens), fmt_num(r.completion_tokens))) { (fmt_num(r.prompt_tokens + r.completion_tokens)) }
                        td.num.muted { (super::overview::fmt_ms(r.avg_latency as i64)) }
                        td.num { (fmt_cost(r.cost)) }
                    } } }
                } }
            }
        }
    }
}

pub struct Range {
    pub key: String,
    secs: i64,
    buckets: i64,
    bucket_size: i64,
    label_every: usize,
}

pub fn range(r: Option<&str>) -> Range {
    let (key, secs, buckets, bucket_size, label_every) = match r.unwrap_or("24h") {
        "1h" => ("1h", 3600, 30, 120, 5),
        "7d" => ("7d", 7 * 86_400, 28, 6 * 3600, 4),
        "30d" => ("30d", 30 * 86_400, 30, 86_400, 5),
        _ => ("24h", 86_400, 24, 3600, 4),
    };
    Range { key: key.into(), secs, buckets, bucket_size, label_every }
}

pub fn stats(db: &Db, r: &Range) -> Markup {
    let t = db.totals_since(now() - r.secs);
    let failed = t.requests - t.ok;
    html! {
        div.stat { div.label { "Requests" } div.value data-k="req" { (fmt_num(t.requests)) } div.sub { @if failed > 0 { span.down { (fmt_num(failed)) " failed attempts" } } @else { "no failures" } } }
        div.stat { div.label { "Input tokens" } div.value data-k="in" { (fmt_num(t.prompt_tokens)) } div.sub { "prompt" } }
        div.stat { div.label { "Output tokens" } div.value data-k="out" { (fmt_num(t.completion_tokens)) } div.sub { "completion" } }
        div.stat { div.label { "Est. cost" } div.value data-k="cost" { (fmt_cost(t.cost)) } div.sub { "at list prices" } }
    }
}

pub fn chart(db: &Db, r: &Range) -> Markup {
    let hist = db.histogram(r.buckets, r.bucket_size);
    let size = r.bucket_size;
    bar_chart(&hist, r.label_every, move |ts| {
        let dt = datetime(ts);
        if size >= 86_400 {
            dt[5..10].to_string()
        } else if size > 3600 {
            format!("{} {}h", &dt[5..10], &dt[11..13])
        } else {
            dt[11..16].to_string()
        }
    })
}

pub fn groups(db: &Db, r: &Range) -> Markup {
    let since = now() - r.secs;
    html! {
        div.grid.g2 {
            (group_table("By model", &db.group_since("model", since)))
            (group_table("By provider", &db.group_since("provider", since)))
        }
        div style="margin-top:14px" { (group_table("By API key", &db.group_since("api_key", since))) }
    }
}

pub fn log(db: &Db, r: &Range, page_no: i64) -> Markup {
    let logs = db.recent_usage(PAGE_SIZE, page_no * PAGE_SIZE);
    let total_logs = db.usage_count();
    let range = &r.key;
    html! {
        @if logs.is_empty() { div.empty { "No requests logged." } } @else {
            div.table-wrap { table {
                thead { tr { th { "Time (UTC)" } th { "Requested" } th { "Account" } th { "Upstream model" } th.num { "In" } th.num { "Out" } th.num { "Latency" } th { "Status" } } }
                tbody { @for l in &logs { tr data-id=(format!("{}-{}-{}", l.ts, l.latency_ms, l.requested_model)) title=[l.error.as_deref()] {
                    td.small.mono.muted style="white-space:nowrap" { (datetime(l.ts)) }
                    td.mono { (l.requested_model) @if l.stream { " " span.badge { "stream" } } }
                    td.small { (l.connection) }
                    td.mono.small.muted { (l.model) }
                    td.num { (fmt_num(l.prompt_tokens)) }
                    td.num { (fmt_num(l.completion_tokens)) }
                    td.num.muted { (super::overview::fmt_ms(l.latency_ms)) }
                    td {
                        (status_badge(l.status))
                        @if let Some(e) = &l.error { div.small.muted style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:3px" { (e) } }
                    }
                } } }
            } }
            div.pager {
                span.small.muted { (page_no * PAGE_SIZE + 1) "–" (page_no * PAGE_SIZE + logs.len() as i64) " of " (fmt_num(total_logs)) }
                div.actions {
                    @if page_no > 0 { a.btn.sm href=(format!("/dashboard/usage?range={range}&page={}", page_no - 1)) { "← Newer" } }
                    @if (page_no + 1) * PAGE_SIZE < total_logs { a.btn.sm href=(format!("/dashboard/usage?range={range}&page={}", page_no + 1)) { "Older →" } }
                }
            }
        }
    }
}

pub async fn page(State(state): State<AppState>, Query(q): Query<UsageQuery>) -> Markup {
    let r = range(q.range.as_deref());
    let db = &state.db;
    let page_no = q.page.unwrap_or(0).max(0);
    let flash = Flash { ok: q.ok.clone(), err: q.err.clone() };
    let live = format!("/dashboard/live?view=usage&range={}&page={page_no}", r.key);
    let body = html! {
        (page_head("Usage", "Tokens, latency and estimated pay-as-you-go cost across every provider.", html! {
            (live_pill())
            div.tabs {
                @for (k, label) in [("1h", "1H"), ("24h", "24H"), ("7d", "7D"), ("30d", "30D")] {
                    a href=(format!("/dashboard/usage?range={k}")) class=[(r.key == k).then_some("active")] { (label) }
                }
            }
        }))
        div.stats data-region="stats" { (stats(db, &r)) }
        div.card style="margin-top:14px" {
            h2 { "Requests" }
            div data-region="chart" { (chart(db, &r)) }
        }
        div style="margin-top:14px" data-region="groups" { (groups(db, &r)) }
        div.card style="margin-top:14px" {
            div.card-head {
                h2 { "Request log" }
                form.inline method="post" action="/dashboard/usage/clear" { button.btn.sm.danger type="submit" { "Clear logs" } }
            }
            div data-region="log" { (log(db, &r, page_no)) }
        }
    };
    page_live("Usage", Nav::Usage, &flash, Some(live), body)
}

pub async fn clear(State(state): State<AppState>) -> Response {
    match state.db.clear_usage() {
        Ok(()) => redirect_ok("/dashboard/usage", "Usage logs cleared"),
        Err(e) => redirect_err("/dashboard/usage", &e.to_string()),
    }
}
