//! Server-rendered dashboard. Every page is fully rendered HTML produced by
//! compile-time `maud` templates; no client-side framework is involved.

mod combos;
mod keys;
mod live;
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
        .route("/dashboard/live", get(live::stream))
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

/// Line icons (24px grid, drawn with `currentColor`).
const NAV: &[(&str, &[(Nav, &str, &str, &str)])] = &[
    (
        "Monitor",
        &[
            (Nav::Overview, "/dashboard", "Overview", r#"<rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="3.5" width="7" height="4.5" rx="1.5"/><rect x="13.5" y="11" width="7" height="9.5" rx="1.5"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.5"/>"#),
            (Nav::Usage, "/dashboard/usage", "Usage", r#"<path d="M4 20V13M10 20V6M16 20v-9M22 20H2"/>"#),
        ],
    ),
    (
        "Routing",
        &[
            (Nav::Providers, "/dashboard/providers", "Providers", r#"<circle cx="6" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="12" r="2.5"/><path d="M8.5 6h2c2 0 3 1 3.6 2.6l.4 1c.5 1.3 1 1.9 1.9 2.2M8.5 18h2c2 0 3-1 3.6-2.6l.4-1c.5-1.3 1-1.9 1.9-2.2"/>"#),
            (Nav::Combos, "/dashboard/combos", "Combos", r#"<rect x="3" y="4" width="18" height="4.5" rx="1.5"/><rect x="3" y="10.5" width="13" height="4.5" rx="1.5"/><rect x="3" y="17" width="8" height="4" rx="1.5"/>"#),
            (Nav::Models, "/dashboard/models", "Models", r#"<path d="M12 3 4 7.5v9L12 21l8-4.5v-9L12 3Z"/><path d="M4 7.5 12 12l8-4.5M12 12v9"/>"#),
        ],
    ),
    (
        "Access",
        &[
            (Nav::Keys, "/dashboard/keys", "API keys", r#"<circle cx="8" cy="15" r="4"/><path d="m10.8 12.2 8.7-8.7M16 7l2.5 2.5M14 9l1.8 1.8"/>"#),
            (Nav::Settings, "/dashboard/settings", "Settings", r#"<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>"#),
        ],
    ),
];

/// The FastRouter mark: three lanes merging into one.
pub fn mark() -> Markup {
    PreEscaped(r#"<svg class="mark" viewBox="0 0 26 26" aria-hidden="true"><rect width="26" height="26" rx="7" fill="var(--accent)"/><path d="M6.5 7.5c4 0 4.5 5.5 8.5 5.5M6.5 18.5c4 0 4.5-5.5 8.5-5.5M6.5 13H15M15 13h4.5" fill="none" stroke="var(--accent-ink)" stroke-width="2" stroke-linecap="round"/></svg>"#.to_string())
}

const FAVICON: &str = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 26 26'%3E%3Crect width='26' height='26' rx='7' fill='%23ff7a3d'/%3E%3Cpath d='M6.5 7.5c4 0 4.5 5.5 8.5 5.5M6.5 18.5c4 0 4.5-5.5 8.5-5.5M6.5 13H15M15 13h4.5' fill='none' stroke='%23160a04' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E";

/// Applies a saved theme before first paint.
const THEME_JS: &str = r#"try{var t=localStorage.getItem('fr-theme');if(t)document.documentElement.setAttribute('data-theme',t)}catch(e){}"#;

/// Progressive enhancement only — every page is complete HTML without it:
/// copy buttons, the theme toggle, and live region updates over SSE.
const APP_JS: &str = r#"(function(){
document.addEventListener('click',function(e){
var b=e.target.closest('[data-copy]');
if(b){var t=b.getAttribute('data-copy');if(navigator.clipboard)navigator.clipboard.writeText(t).then(function(){var o=b.textContent;b.textContent='Copied';setTimeout(function(){b.textContent=o},1200)});return}
if(e.target.closest('[data-theme-toggle]')){var r=document.documentElement,n=r.getAttribute('data-theme')==='light'?'dark':'light';r.setAttribute('data-theme',n);try{localStorage.setItem('fr-theme',n)}catch(_){}}
});
var src=document.body.getAttribute('data-live');
if(!src||!window.EventSource)return;
var pill=document.getElementById('live');
function busy(el){var a=document.activeElement;return(a&&el.contains(a)&&/^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName))||el.querySelector('details[open]')}
var es=new EventSource(src);
es.onopen=function(){if(pill)pill.classList.add('on')};
es.onerror=function(){if(pill)pill.classList.remove('on')};
es.addEventListener('region',function(ev){
var m=JSON.parse(ev.data),el=document.querySelector('[data-region="'+m.r+'"]');
if(!el||busy(el))return;
var old={},seen={},had=false;
el.querySelectorAll('[data-k]').forEach(function(n){old[n.getAttribute('data-k')]=n.textContent});
el.querySelectorAll('tr[data-id]').forEach(function(n){seen[n.getAttribute('data-id')]=1;had=true});
el.innerHTML=m.h;
el.querySelectorAll('[data-k]').forEach(function(n){var k=n.getAttribute('data-k');if(k in old&&old[k]!==n.textContent)n.classList.add('bump')});
if(had)el.querySelectorAll('tr[data-id]').forEach(function(n){if(!seen[n.getAttribute('data-id')])n.classList.add('fresh')});
});
es.addEventListener('reload',function(){if(!document.querySelector('details[open],input:focus,textarea:focus,select:focus'))location.reload()});
})();"#;

pub fn page(title: &str, active: Nav, flash: &Flash, body: Markup) -> Markup {
    page_live(title, active, flash, None, body)
}

/// A dashboard page; `live` is the SSE source that keeps its regions fresh.
pub fn page_live(title: &str, active: Nav, flash: &Flash, live: Option<String>, body: Markup) -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" data-theme="dark" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                meta name="color-scheme" content="dark light";
                title { (title) " · FastRouter" }
                script { (PreEscaped(THEME_JS)) }
                link rel="stylesheet" href="/static/app.css";
                link rel="icon" href=(FAVICON);
            }
            body data-live=[live.as_deref()] {
                div.shell {
                    aside.side {
                        a.brand href="/dashboard" { (mark()) "FastRouter" }
                        nav.nav {
                            @for (group, items) in NAV {
                                div.nav-label { (group) }
                                @for (nav, href, label, icon) in *items {
                                    a href=(href) class=[(active == *nav).then_some("active")] {
                                        svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" { (PreEscaped(*icon)) }
                                        (label)
                                    }
                                }
                            }
                        }
                        div.side-foot {
                            span.ver { "v" (env!("CARGO_PKG_VERSION")) }
                            button.icon-btn type="button" data-theme-toggle title="Toggle light / dark" {
                                svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" { (PreEscaped(r#"<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>"#)) }
                            }
                            form method="post" action="/logout" {
                                button.icon-btn type="submit" title="Sign out" {
                                    svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" { (PreEscaped(r#"<path d="M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3M10 17l5-5-5-5M15 12H4"/>"#)) }
                                }
                            }
                        }
                    }
                    main.main {
                        @if let Some(ok) = &flash.ok { div.flash.ok { (ok) } }
                        @if let Some(err) = &flash.err { div.flash.err { (err) } }
                        (body)
                    }
                }
                script { (PreEscaped(APP_JS)) }
            }
        }
    }
}

/// A standalone page (sign-in, OAuth results) without the sidebar.
pub fn bare_page(title: &str, body: Markup) -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" data-theme="dark" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { (title) " · FastRouter" }
                script { (PreEscaped(THEME_JS)) }
                link rel="stylesheet" href="/static/app.css";
                link rel="icon" href=(FAVICON);
            }
            body { div.login-wrap { div.login { (body) } } }
        }
    }
}

/// The "Live" pill shown on pages that update in place.
pub fn live_pill() -> Markup {
    html! { span.live #live title="Updates in real time" { i {} "Live" } }
}

pub fn page_head(title: &str, subtitle: &str, actions: Markup) -> Markup {
    html! {
        div.page-head {
            div { h1 { (title) } @if !subtitle.is_empty() { p { (subtitle) } } }
            div.actions { (actions) }
        }
    }
}

pub fn copy_button(text: &str) -> Markup {
    html! { button.btn.sm.ghost type="button" data-copy=(text) { "Copy" } }
}

pub fn status_badge(status: i64) -> Markup {
    let class = match status {
        200..=299 => "badge ok",
        429 => "badge warn",
        _ => "badge err",
    };
    html! { span class=(class) { span.code-pill { (status) } } }
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

/// A pre-rendered SVG bar chart: successful requests stacked under failures.
pub fn bar_chart(buckets: &[(i64, i64, i64, i64)], label_every: usize, label: impl Fn(i64) -> String) -> Markup {
    let w = 720.0_f64;
    let h = 150.0_f64;
    let max = buckets.iter().map(|b| b.1).max().unwrap_or(0).max(1) as f64;
    let n = buckets.len().max(1) as f64;
    let slot = w / n;
    let bw = (slot * 0.56).max(1.0);
    let total: i64 = buckets.iter().map(|b| b.1).sum();
    let failed: i64 = buckets.iter().map(|b| b.2).sum();
    html! {
        div.chart {
            svg viewBox=(format!("0 0 {w} {h}")) preserveAspectRatio="none" role="img" aria-label="Requests over time" {
                @for i in 1..=3 {
                    @let y = h * f64::from(i) / 4.0;
                    line.grid-line x1="0" x2=(w) y1=(y) y2=(y) vector-effect="non-scaling-stroke" {}
                }
                line.base x1="0" x2=(w) y1=(h - 0.5) y2=(h - 0.5) vector-effect="non-scaling-stroke" {}
                @for (i, (ts, count, errors, tokens)) in buckets.iter().enumerate() {
                    @let x = slot * i as f64 + (slot - bw) / 2.0;
                    @let total_h = (h - 4.0) * (*count as f64) / max;
                    @let err_h = if *count > 0 { total_h * (*errors as f64) / (*count as f64) } else { 0.0 };
                    @let ok_h = total_h - err_h;
                    @let tip = format!("{} — {} requests{}, {} tokens", label(*ts), count, if *errors > 0 { format!(" ({errors} failed)") } else { String::new() }, fmt_num(*tokens));
                    g {
                        title { (tip) }
                        @if ok_h > 0.0 { rect.bar x=(format!("{x:.1}")) y=(format!("{:.1}", h - ok_h.max(1.5))) width=(format!("{bw:.1}")) height=(format!("{:.1}", ok_h.max(1.5))) rx="1.5" {} }
                        @if err_h > 0.0 { rect.bar.err x=(format!("{x:.1}")) y=(format!("{:.1}", h - total_h.max(1.5))) width=(format!("{bw:.1}")) height=(format!("{:.1}", err_h.max(1.5))) rx="1.5" {} }
                    }
                }
            }
            div.chart-foot {
                @for (i, b) in buckets.iter().enumerate() {
                    @if i % label_every == 0 { span { (label(b.0)) } }
                }
            }
            div.chart-foot style="margin-top:12px" {
                span.legend { span { i {} "Requests " span.muted data-k="c-total" { (fmt_num(total)) } } span { i.err {} "Failed " span.muted data-k="c-fail" { (fmt_num(failed)) } } }
                span { "peak " (max as i64) "/bucket · UTC" }
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
