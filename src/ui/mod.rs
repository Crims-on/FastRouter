//! Server-rendered dashboard. Every page is fully rendered HTML produced by
//! compile-time `maud` templates; no client-side framework is involved.

mod combos;
mod keys;
mod oauth;
mod overview;
mod providers;
mod settings;
mod usage;

use axum::Router;
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use maud::{DOCTYPE, Markup, PreEscaped, html};
use serde::Deserialize;

use crate::AppState;

pub const CSS: &str = include_str!("style.css");

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/dashboard", get(overview::page))
        .route("/dashboard/providers", get(providers::list))
        .route("/dashboard/providers/new-node", post(providers::create_node))
        .route("/dashboard/providers/{id}", get(providers::detail))
        .route("/dashboard/providers/{id}/connections", post(providers::create))
        .route("/dashboard/providers/{id}/import/{method}", post(providers::import))
        .route("/dashboard/providers/{id}/custom-models", post(providers::save_custom_models))
        .route("/dashboard/providers/{id}/node", post(providers::update_node))
        .route("/dashboard/providers/{id}/node/delete", post(providers::delete_node))
        .route("/dashboard/connections/{id}/update", post(providers::update))
        .route("/dashboard/connections/{id}/toggle", post(providers::toggle))
        .route("/dashboard/connections/{id}/delete", post(providers::delete))
        .route("/dashboard/connections/{id}/test", post(providers::test))
        .route("/dashboard/connections/{id}/refresh", post(providers::refresh))
        .route("/dashboard/connections/{id}/unlock", post(providers::unlock))
        .route("/dashboard/oauth/{provider}/start", post(oauth::start))
        .route("/dashboard/oauth/{provider}/device", post(oauth::device_start))
        .route("/dashboard/oauth/flow/{state}", get(oauth::flow_page).post(oauth::flow_submit))
        .route("/dashboard/oauth/device/{id}", get(oauth::device_page))
        .route("/dashboard/combos", get(combos::page).post(combos::save))
        .route("/dashboard/combos/{name}/delete", post(combos::delete))
        .route("/dashboard/models", get(combos::models_page))
        .route("/dashboard/aliases", post(combos::save_alias))
        .route("/dashboard/aliases/{alias}/delete", post(combos::delete_alias))
        .route("/dashboard/keys", get(keys::page).post(keys::create))
        .route("/dashboard/keys/{id}/delete", post(keys::delete))
        .route("/dashboard/keys/require", post(keys::toggle_require))
        .route("/dashboard/usage", get(usage::page))
        .route("/dashboard/usage/clear", post(usage::clear))
        .route("/dashboard/settings", get(settings::page))
        .route("/dashboard/settings/password", post(settings::change_password))
        .route("/dashboard/settings/routing", post(settings::save_routing))
        .route("/dashboard/settings/oauth-clients", post(settings::save_oauth_clients))
}

pub fn public_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(oauth::root))
        .route("/callback", get(oauth::callback))
        .route("/login", get(settings::login_page).post(settings::login))
        .route("/logout", post(settings::logout))
        .route("/static/app.css", get(css))
}

async fn css() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/css; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        CSS,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Flash messages travel in the query string after POST/redirect/GET.
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
pub struct Flash {
    pub ok: Option<String>,
    pub err: Option<String>,
}

pub fn redirect_ok(path: &str, msg: &str) -> Response {
    Redirect::to(&format!("{path}?ok={}", urlencode(msg))).into_response()
}

pub fn redirect_err(path: &str, msg: &str) -> Response {
    Redirect::to(&format!("{path}?err={}", urlencode(msg))).into_response()
}

pub fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
pub enum Nav {
    Overview,
    Providers,
    Combos,
    Models,
    Keys,
    Usage,
    Settings,
}

const NAV: &[(Nav, &str, &str, &str)] = &[
    (
        Nav::Overview,
        "/dashboard",
        "Overview",
        r#"<path d="M3 13h8V3H3zm0 8h8v-6H3zm10 0h8V11h-8zm0-18v6h8V3z"/>"#,
    ),
    (
        Nav::Providers,
        "/dashboard/providers",
        "Providers",
        r#"<path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/><circle cx="8" cy="6" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="10" cy="18" r="2"/>"#,
    ),
    (
        Nav::Combos,
        "/dashboard/combos",
        "Combos",
        r#"<path d="M7 7h7a3 3 0 0 1 0 6H10a3 3 0 0 0 0 6h7" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/><circle cx="5" cy="7" r="2.4"/><circle cx="19" cy="19" r="2.4"/>"#,
    ),
    (
        Nav::Models,
        "/dashboard/models",
        "Models",
        r#"<path d="M12 2 3 7v10l9 5 9-5V7zm0 2.3L18.7 8 12 11.7 5.3 8zM5 9.7l6 3.3v6.7l-6-3.3zm8 10V13l6-3.3v6.7z"/>"#,
    ),
    (
        Nav::Keys,
        "/dashboard/keys",
        "API Keys",
        r#"<path d="M14 3a7 7 0 0 0-6.7 9.1L2 17.4V22h4.6l1-1v-2h2v-2h2l1.3-1.3A7 7 0 1 0 14 3m2.5 3a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3"/>"#,
    ),
    (
        Nav::Usage,
        "/dashboard/usage",
        "Usage",
        r#"<path d="M4 20V10h3v10zm6.5 0V4h3v16zM17 20v-7h3v7z"/>"#,
    ),
    (
        Nav::Settings,
        "/dashboard/settings",
        "Settings",
        r#"<path d="M19.4 13a7.5 7.5 0 0 0 0-2l2.1-1.6-2-3.5-2.5 1a7.4 7.4 0 0 0-1.7-1L15 3h-4l-.4 2.9a7.4 7.4 0 0 0-1.7 1l-2.5-1-2 3.5L6.6 11a7.5 7.5 0 0 0 0 2l-2.1 1.6 2 3.5 2.5-1c.5.4 1.1.7 1.7 1L11 21h4l.4-2.9c.6-.3 1.2-.6 1.7-1l2.5 1 2-3.5zM13 15.5a3.5 3.5 0 1 1 0-7 3.5 3.5 0 0 1 0 7"/>"#,
    ),
];

/// Tiny progressive enhancement: copy buttons. Every page works without it.
const COPY_JS: &str = r#"document.addEventListener('click',function(e){var b=e.target.closest('[data-copy]');if(!b)return;var t=b.getAttribute('data-copy');navigator.clipboard&&navigator.clipboard.writeText(t).then(function(){var o=b.textContent;b.textContent='Copied';setTimeout(function(){b.textContent=o},1200)})});"#;

pub fn page(title: &str, active: Nav, flash: &Flash, body: Markup) -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { (title) " · FastRouter" }
                link rel="stylesheet" href="/static/app.css";
                link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%235b5bf0'/><path d='M9 22 16 9l7 13' stroke='white' stroke-width='3' fill='none' stroke-linecap='round'/></svg>";
            }
            body {
                div.shell {
                    aside.side {
                        a.brand href="/dashboard" style="color:inherit;text-decoration:none" {
                            span.brand-mark { "⚡" } "FastRouter"
                        }
                        nav.nav {
                            @for (nav, href, label, icon) in NAV {
                                a href=(href) class=[(active == *nav).then_some("active")] {
                                    svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" { (PreEscaped(*icon)) }
                                    (label)
                                }
                            }
                        }
                        div.side-foot {
                            "v" (env!("CARGO_PKG_VERSION")) " · Rust + axum"
                            form method="post" action="/logout" { button.btn.sm type="submit" { "Log out" } }
                        }
                    }
                    main.main {
                        @if let Some(ok) = &flash.ok { div.flash.ok { (ok) } }
                        @if let Some(err) = &flash.err { div.flash.err { (err) } }
                        (body)
                    }
                }
                script { (PreEscaped(COPY_JS)) }
            }
        }
    }
}

pub fn page_head(title: &str, subtitle: &str, actions: Markup) -> Markup {
    html! {
        div.page-head {
            div { h1 { (title) } p { (subtitle) } }
            div.actions { (actions) }
        }
    }
}

pub fn copy_button(text: &str) -> Markup {
    html! { button.btn.sm type="button" data-copy=(text) { "Copy" } }
}

pub fn status_badge(status: i64) -> Markup {
    let class = match status {
        200..=299 => "badge ok",
        429 => "badge warn",
        _ => "badge err",
    };
    html! { span class=(class) { (status) } }
}

/// The public URL of this server, derived from the request's Host header.
pub fn base_url(headers: &HeaderMap) -> String {
    let host = headers
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("localhost:20128");
    let proto = headers
        .get("x-forwarded-proto")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("http");
    format!("{proto}://{host}")
}

pub fn mask(key: &str) -> String {
    let n = key.chars().count();
    if n <= 10 {
        return "•".repeat(n.max(4));
    }
    let start: String = key.chars().take(6).collect();
    let end: String = key.chars().skip(n - 4).collect();
    format!("{start}…{end}")
}

pub fn fmt_num(n: i64) -> String {
    let f = n as f64;
    if n.abs() >= 1_000_000_000 {
        format!("{:.2}B", f / 1e9)
    } else if n.abs() >= 1_000_000 {
        format!("{:.2}M", f / 1e6)
    } else if n.abs() >= 10_000 {
        format!("{:.1}K", f / 1e3)
    } else {
        let s = n.to_string();
        if n.abs() >= 1000 {
            format!("{},{}", &s[..s.len() - 3], &s[s.len() - 3..])
        } else {
            s
        }
    }
}

pub fn fmt_cost(c: f64) -> String {
    if c >= 100.0 {
        format!("${c:.0}")
    } else if c >= 0.01 || c == 0.0 {
        format!("${c:.2}")
    } else {
        format!("${c:.4}")
    }
}

pub fn ago(ts: i64) -> String {
    let d = crate::db::now() - ts;
    match d {
        i64::MIN..=4 => "just now".into(),
        5..=59 => format!("{d}s ago"),
        60..=3599 => format!("{}m ago", d / 60),
        3600..=86399 => format!("{}h ago", d / 3600),
        _ => format!("{}d ago", d / 86400),
    }
}

/// `YYYY-MM-DD HH:MM` in UTC.
pub fn datetime(ts: i64) -> String {
    let days = ts.div_euclid(86400);
    let secs = ts.rem_euclid(86400);
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}",
        secs / 3600,
        (secs % 3600) / 60
    )
}

/// A pre-rendered SVG bar chart.
pub fn bar_chart(
    buckets: &[(i64, i64, i64)],
    label_every: usize,
    label: impl Fn(i64) -> String,
) -> Markup {
    let w = 720.0_f64;
    let h = 160.0_f64;
    let pad_b = 20.0;
    let pad_t = 8.0;
    let max = buckets.iter().map(|b| b.1).max().unwrap_or(0).max(1) as f64;
    let n = buckets.len().max(1) as f64;
    let slot = w / n;
    let bw = (slot * 0.68).max(1.0);
    html! {
        div.chart {
            svg viewBox=(format!("0 0 {w} {h}")) preserveAspectRatio="none" role="img" aria-label="Requests over time" {
                @for i in 0..=3 {
                    @let y = pad_t + (h - pad_b - pad_t) * f64::from(i) / 3.0;
                    line.grid-line x1="0" x2=(w) y1=(y) y2=(y) {}
                }
                @for (i, (ts, count, tokens)) in buckets.iter().enumerate() {
                    @let bh = (h - pad_b - pad_t) * (*count as f64) / max;
                    @let x = slot * i as f64 + (slot - bw) / 2.0;
                    rect.bar x=(format!("{x:.1}")) y=(format!("{:.1}", h - pad_b - bh)) width=(format!("{bw:.1}")) height=(format!("{:.1}", bh.max(if *count > 0 { 1.5 } else { 0.0 }))) rx="2" {
                        title { (label(*ts)) " — " (count) " requests, " (fmt_num(*tokens)) " tokens" }
                    }
                    @if i % label_every == 0 {
                        text x=(format!("{:.1}", slot * i as f64 + slot / 2.0)) y=(h - 5.0) text-anchor="middle" { (label(*ts)) }
                    }
                }
            }
            div.small.muted style="display:flex;justify-content:space-between" {
                span { "peak " (max as i64) " req" }
                span { "UTC" }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datetime_formats() {
        assert_eq!(datetime(0), "1970-01-01 00:00");
        assert_eq!(datetime(1_700_000_000), "2023-11-14 22:13");
    }

    #[test]
    fn numbers() {
        assert_eq!(fmt_num(999), "999");
        assert_eq!(fmt_num(1234), "1,234");
        assert_eq!(fmt_num(12_345), "12.3K");
        assert_eq!(fmt_num(2_500_000), "2.50M");
        assert_eq!(mask("sk-abcdefghijklmnop"), "sk-abc…mnop");
    }
}
