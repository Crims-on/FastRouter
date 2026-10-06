//! Browser and device-code logins for subscription providers.
//!
//! Pending logins live in memory. The browser flow redirects back to
//! `/callback` (or, for providers whose OAuth client only whitelists a fixed
//! loopback port, to a short-lived listener on that port). Every flow also
//! accepts the final redirect URL pasted by hand, which is what remote
//! deployments use.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use axum::Form;
use axum::extract::{Path, RawQuery, State};
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum_extra::extract::CookieJar;
use maud::{Markup, html};
use serde_json::{Value, json};

use super::{Flash, Nav, page, page_head, redirect_err, urlencode};
use crate::AppState;
use crate::oauth::flows::{self, AuthStart, DeviceStart, Poll};

const TTL: Duration = Duration::from_secs(15 * 60);

#[derive(Clone)]
enum Outcome {
    Pending,
    Done(String),
    Failed(String),
}

#[derive(Clone)]
struct Login {
    provider: String,
    auth: AuthStart,
    kiro_social: bool,
    created: Instant,
    outcome: Outcome,
}

#[derive(Clone)]
struct Device {
    provider: String,
    start: DeviceStart,
    created: Instant,
    last_poll: Option<Instant>,
    outcome: Outcome,
}

static LOGINS: LazyLock<Mutex<HashMap<String, Login>>> = LazyLock::new(Default::default);
static DEVICES: LazyLock<Mutex<HashMap<String, Device>>> = LazyLock::new(Default::default);

fn prune() {
    LOGINS.lock().unwrap().retain(|_, l| l.created.elapsed() < TTL);
    DEVICES.lock().unwrap().retain(|_, d| d.created.elapsed() < Duration::from_secs(d.start.expires_in.max(60) + 120));
}

fn provider_name(id: &str) -> String {
    crate::registry::REG.entry(id).map(|e| e["name"].clone()).unwrap_or_default().as_str().map(str::to_owned).unwrap_or_else(|| id.to_string())
}

/// The redirect URI a provider's OAuth client accepts. Most whitelist any
/// `http://localhost:<port>/callback`; when the dashboard is reached from
/// elsewhere we still use localhost and the user pastes the final URL.
fn redirect_uri(st: &AppState, headers: &HeaderMap, provider: &str) -> String {
    if let Some((port, path)) = flows::fixed_redirect(provider) {
        if provider == "zed" {
            return format!("http://127.0.0.1:{}", local_port(st, headers));
        }
        let host = if provider == "codex" { "localhost" } else { "127.0.0.1" };
        return format!("http://{host}:{port}{path}");
    }
    let base = super::base_url(headers);
    let host = base.split("://").nth(1).unwrap_or("").split(':').next().unwrap_or("");
    if matches!(host, "localhost" | "127.0.0.1" | "[::1]") {
        format!("{base}/callback")
    } else {
        format!("http://localhost:{}/callback", st.config.port)
    }
}

fn local_port(st: &AppState, headers: &HeaderMap) -> u16 {
    super::base_url(headers).rsplit(':').next().and_then(|p| p.parse().ok()).unwrap_or(st.config.port)
}

fn meta_from_form(f: &HashMap<String, String>) -> Value {
    let mut m = serde_json::Map::new();
    for (k, v) in f {
        if let Some(name) = k.strip_prefix("meta_") {
            if !v.trim().is_empty() {
                m.insert(name.to_string(), json!(v.trim()));
            }
        }
    }
    Value::Object(m)
}

// ---------------------------------------------------------------------------
// Browser flow
// ---------------------------------------------------------------------------

pub async fn start(State(st): State<AppState>, headers: HeaderMap, Path(provider): Path<String>, Form(f): Form<HashMap<String, String>>) -> Response {
    prune();
    let back = format!("/dashboard/providers/{}", urlencode(&provider));
    let meta = meta_from_form(&f);
    let (auth, kiro_social) = if provider == "kiro" {
        (flows::kiro_social_start(meta["idp"].as_str().unwrap_or("google")), true)
    } else {
        let uri = redirect_uri(&st, &headers, &provider);
        match flows::start_auth(&provider, &uri, meta).await {
            Ok(a) => (a, false),
            Err(e) => return redirect_err(&back, &e),
        }
    };
    let state = auth.state.clone();
    LOGINS.lock().unwrap().insert(state.clone(), Login { provider: provider.clone(), auth, kiro_social, created: Instant::now(), outcome: Outcome::Pending });
    if let Some((port, path)) = flows::fixed_redirect(&provider).filter(|_| provider != "zed") {
        spawn_loopback(st.clone(), port, path);
    }
    Redirect::to(&format!("/dashboard/oauth/flow/{}", urlencode(&state))).into_response()
}

/// Finishes a pending login with whatever the provider redirected back with.
async fn complete(st: &AppState, state: &str, code: &str) -> Result<String, String> {
    let login = LOGINS.lock().unwrap().get(state).cloned().ok_or("This login has expired — start it again.")?;
    if let Outcome::Done(id) = &login.outcome {
        return Ok(id.clone());
    }
    let res = async {
        let tokens = if login.kiro_social {
            let (c, _) = flows::parse_callback_input(code);
            flows::kiro_social_exchange(&c, &login.auth).await?
        } else {
            flows::exchange(&login.provider, code, &login.auth).await?
        };
        flows::save_connection(&st.db, &login.provider, &tokens).map_err(|e| e.to_string())
    }
    .await;
    if let Some(l) = LOGINS.lock().unwrap().get_mut(state) {
        l.outcome = match &res {
            Ok(id) => Outcome::Done(id.clone()),
            Err(e) => Outcome::Failed(e.clone()),
        };
    }
    res
}

/// Finds the pending login a redirect belongs to.
fn match_state(pairs: &[(String, String)]) -> Option<String> {
    let get = |k: &str| pairs.iter().find(|(a, _)| a == k).map(|(_, b)| b.clone());
    let logins = LOGINS.lock().unwrap();
    if let Some(s) = get("state") {
        return logins.contains_key(&s).then_some(s);
    }
    // Providers that don't echo `state` back: the newest pending login wins.
    let provider = if get("u").is_some() {
        Some("xiaomi-mimo")
    } else if get("user_id").is_some() {
        Some("zed")
    } else {
        None
    };
    logins
        .iter()
        .filter(|(_, l)| matches!(l.outcome, Outcome::Pending) && provider.is_none_or(|p| l.provider == p))
        .max_by_key(|(_, l)| l.created)
        .map(|(k, _)| k.clone())
}

/// What to hand `exchange` for a given redirect query.
fn code_from(provider: &str, raw_query: &str, pairs: &[(String, String)]) -> Option<String> {
    let get = |k: &str| pairs.iter().find(|(a, _)| a == k).map(|(_, b)| b.clone()).filter(|v| !v.is_empty());
    match provider {
        "zed" => get("user_id").map(|_| format!("?{raw_query}")),
        "xiaomi-mimo" => get("u"),
        "kimchi" => get("token").or_else(|| get("api_key")).or_else(|| get("apiKey")).or_else(|| get("access_token")).or_else(|| get("key")),
        _ => get("code"),
    }
}

fn result_page(ok: bool, title: &str, msg: &str, link: Option<String>) -> Html<String> {
    let m = super::bare_page(title, html! {
        div.brand { (super::mark()) "FastRouter" }
        div.card style="margin-top:18px" {
            div style="display:flex;align-items:center;gap:8px;margin-bottom:8px" {
                span class=(if ok { "badge ok" } else { "badge err" }) { span.dot {} (if ok { "Connected" } else { "Failed" }) }
            }
            h2 { (title) }
            p.small.muted { (msg) }
            @if let Some(l) = link { a.btn.primary href=(l) { "Back to dashboard" } }
        }
    });
    Html(m.into_string())
}

async fn handle_redirect(st: &AppState, raw: &str) -> Html<String> {
    prune();
    let pairs = flows::url_pairs(raw);
    if let Some(err) = pairs.iter().find(|(k, _)| k == "error").map(|(_, v)| v.clone()) {
        let desc = pairs.iter().find(|(k, _)| k == "error_description").map(|(_, v)| v.clone()).unwrap_or_default();
        if let Some(s) = match_state(&pairs) {
            if let Some(l) = LOGINS.lock().unwrap().get_mut(&s) {
                l.outcome = Outcome::Failed(format!("{err} {desc}"));
            }
        }
        return result_page(false, "Login failed", &format!("{err} {desc}"), Some("/dashboard/providers".into()));
    }
    let Some(state) = match_state(&pairs) else {
        return result_page(false, "Unknown login", "No pending login matches this redirect. Start the login again from the dashboard.", Some("/dashboard/providers".into()));
    };
    let provider = LOGINS.lock().unwrap().get(&state).map(|l| l.provider.clone()).unwrap_or_default();
    let Some(code) = code_from(&provider, raw, &pairs) else {
        return result_page(false, "Login failed", "The redirect did not include an authorization code.", Some(format!("/dashboard/oauth/flow/{}", urlencode(&state))));
    };
    match complete(st, &state, &code).await {
        Ok(_) => result_page(true, &format!("{} connected", provider_name(&provider)), "You can close this tab and return to FastRouter.", Some(format!("/dashboard/providers/{}?ok=Connected", urlencode(&provider)))),
        Err(e) => result_page(false, "Login failed", &e, Some(format!("/dashboard/oauth/flow/{}", urlencode(&state)))),
    }
}

/// `GET /callback` — the redirect target for most providers.
pub async fn callback(State(st): State<AppState>, RawQuery(q): RawQuery) -> Response {
    handle_redirect(&st, q.as_deref().unwrap_or("")).await.into_response()
}

/// `GET /` — dashboard entry; also Zed's native-app sign-in lands here with
/// `?user_id=…&access_token=…`.
pub async fn root(State(st): State<AppState>, RawQuery(q): RawQuery) -> Response {
    let q = q.unwrap_or_default();
    if q.contains("user_id=") && q.contains("access_token=") {
        return handle_redirect(&st, &q).await.into_response();
    }
    Redirect::to("/dashboard").into_response()
}

/// Short-lived listener for providers that only accept a fixed loopback
/// redirect (Codex → :1455, xAI → :56121). It stops once nothing is pending.
fn spawn_loopback(st: AppState, port: u16, path: &'static str) {
    static RUNNING: LazyLock<Mutex<std::collections::HashSet<u16>>> = LazyLock::new(Default::default);
    if !RUNNING.lock().unwrap().insert(port) {
        return;
    }
    tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("OAuth callback port {port} unavailable ({e}); paste the redirect URL instead");
                RUNNING.lock().unwrap().remove(&port);
                return;
            }
        };
        let app = axum::Router::new()
            .route(path, axum::routing::get(|State(st): State<AppState>, RawQuery(q): RawQuery| async move { handle_redirect(&st, q.as_deref().unwrap_or("")).await }))
            .with_state(st);
        let shutdown = async move {
            let started = Instant::now();
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                let pending = LOGINS.lock().unwrap().values().any(|l| matches!(l.outcome, Outcome::Pending) && flows::fixed_redirect(&l.provider).is_some_and(|(p, _)| p == port));
                if !pending || started.elapsed() > TTL {
                    break;
                }
            }
        };
        let _ = axum::serve(listener, app).with_graceful_shutdown(shutdown).await;
        RUNNING.lock().unwrap().remove(&port);
    });
}

pub async fn flow_page(Path(state): Path<String>) -> Response {
    let Some(login) = LOGINS.lock().unwrap().get(&state).cloned() else {
        return redirect_err("/dashboard/providers", "This login has expired — start it again.");
    };
    let name = provider_name(&login.provider);
    let back = format!("/dashboard/providers/{}", urlencode(&login.provider));
    if let Outcome::Done(_) = login.outcome {
        return super::redirect_ok(&back, &format!("{name} connected"));
    }
    let manual_hint = if login.kiro_social {
        "After signing in, the browser tries to open a kiro:// link. Copy that full link (from the address bar or the \"open app\" prompt) and paste it here."
    } else {
        match login.provider.as_str() {
            "zed" => "After approving, Zed redirects to this server. If that page doesn't load, paste the full URL from the address bar here.",
            "codex" | "xai" => "If the redirect page can't be reached (FastRouter isn't on this machine), paste the full URL from the address bar here.",
            _ => "If the redirect can't reach FastRouter (e.g. it runs on another machine), paste the full URL of the page you land on here.",
        }
    };
    let failed = match &login.outcome {
        Outcome::Failed(e) => Some(e.clone()),
        _ => None,
    };
    let flash = Flash { ok: None, err: failed };
    let body = html! {
        (page_head(&format!("Sign in to {name}"), "Complete the login in your browser", html! { a.btn href=(back) { "Cancel" } }))
        div.card {
            h2 { "1. Open the login page" }
            p { a.btn.primary href=(login.auth.auth_url) target="_blank" rel="noopener" { "Open " (name) " login ↗" } " " (super::copy_button(&login.auth.auth_url)) }
            p.small.muted { "This page refreshes automatically once the redirect arrives." }
        }
        div.card {
            h2 { "2. Or paste the redirect URL" }
            p.small.muted { (manual_hint) }
            form method="post" action=(format!("/dashboard/oauth/flow/{}", urlencode(&state))) {
                textarea name="callback" rows="3" required placeholder="http://localhost:…/callback?code=…" {}
                div style="margin-top:10px" { button.btn.primary type="submit" { "Finish login" } }
            }
        }
    };
    let mut html = page(&format!("Sign in · {name}"), Nav::Providers, &flash, body).into_string();
    html = html.replacen("<head>", "<head><meta http-equiv=\"refresh\" content=\"4\">", 1);
    Html(html).into_response()
}

pub async fn flow_submit(State(st): State<AppState>, Path(state): Path<String>, Form(f): Form<HashMap<String, String>>) -> Response {
    let input = f.get("callback").map(|s| s.trim().to_string()).unwrap_or_default();
    let Some(provider) = LOGINS.lock().unwrap().get(&state).map(|l| l.provider.clone()) else {
        return redirect_err("/dashboard/providers", "This login has expired — start it again.");
    };
    let here = format!("/dashboard/oauth/flow/{}", urlencode(&state));
    if input.is_empty() {
        return redirect_err(&here, "Paste the redirect URL");
    }
    let kiro_social = LOGINS.lock().unwrap().get(&state).is_some_and(|l| l.kiro_social);
    let code = if kiro_social || provider == "zed" {
        input.clone()
    } else {
        let raw = input.split_once('?').map(|x| x.1).unwrap_or(&input);
        let raw = raw.split('#').next().unwrap_or(raw);
        let pairs = flows::url_pairs(raw);
        match code_from(&provider, raw, &pairs) {
            Some(c) => c,
            // A bare code (Claude shows `code#state` on its own page).
            None => input.clone(),
        }
    };
    match complete(&st, &state, &code).await {
        Ok(_) => super::redirect_ok(&format!("/dashboard/providers/{}", urlencode(&provider)), &format!("{} connected", provider_name(&provider))),
        Err(e) => redirect_err(&here, &e),
    }
}

// ---------------------------------------------------------------------------
// Device-code flow
// ---------------------------------------------------------------------------

pub async fn device_start(Path(provider): Path<String>, Form(f): Form<HashMap<String, String>>) -> Response {
    prune();
    let back = format!("/dashboard/providers/{}", urlencode(&provider));
    let get = |k: &str| f.get(k).map(|s| s.trim()).filter(|s| !s.is_empty());
    let mut opts = json!({});
    if let Some(m) = get("auth_method") {
        opts["authMethod"] = json!(m);
    }
    if let Some(u) = get("start_url") {
        opts["startUrl"] = json!(u);
    }
    if let Some(r) = get("region") {
        opts["region"] = json!(r);
    }
    match flows::device_start(&provider, &opts).await {
        Ok(start) => {
            let id = uuid::Uuid::new_v4().simple().to_string();
            DEVICES.lock().unwrap().insert(id.clone(), Device { provider, start, created: Instant::now(), last_poll: None, outcome: Outcome::Pending });
            Redirect::to(&format!("/dashboard/oauth/device/{id}")).into_response()
        }
        Err(e) => redirect_err(&back, &e),
    }
}

async fn poll_device(st: &AppState, id: &str) {
    let Some(d) = DEVICES.lock().unwrap().get(id).cloned() else { return };
    if !matches!(d.outcome, Outcome::Pending) || d.last_poll.is_some_and(|t| t.elapsed() < Duration::from_secs(d.start.interval.max(1))) {
        return;
    }
    if let Some(x) = DEVICES.lock().unwrap().get_mut(id) {
        x.last_poll = Some(Instant::now());
    }
    let outcome = match flows::device_poll(&d.provider, &d.start).await {
        Poll::Pending => return,
        Poll::Done(tokens) => match flows::save_connection(&st.db, &d.provider, &tokens) {
            Ok(cid) => Outcome::Done(cid),
            Err(e) => Outcome::Failed(e.to_string()),
        },
        Poll::Error(e) => Outcome::Failed(e),
    };
    if let Some(x) = DEVICES.lock().unwrap().get_mut(id) {
        x.outcome = outcome;
    }
}

pub async fn device_page(State(st): State<AppState>, _jar: CookieJar, Path(id): Path<String>) -> Response {
    poll_device(&st, &id).await;
    let Some(d) = DEVICES.lock().unwrap().get(&id).cloned() else {
        return redirect_err("/dashboard/providers", "This login has expired — start it again.");
    };
    let name = provider_name(&d.provider);
    let back = format!("/dashboard/providers/{}", urlencode(if d.provider == "kimi-coding" { "kimi" } else { &d.provider }));
    match &d.outcome {
        Outcome::Done(_) => return super::redirect_ok(&back, &format!("{name} connected")),
        Outcome::Failed(e) => return redirect_err(&back, e),
        Outcome::Pending => {}
    }
    let remaining = d.start.expires_in.saturating_sub(d.created.elapsed().as_secs());
    if remaining == 0 {
        return redirect_err(&back, "The device code expired — start again.");
    }
    let link = if d.start.verification_uri_complete.is_empty() { d.start.verification_uri.clone() } else { d.start.verification_uri_complete.clone() };
    let body: Markup = html! {
        (page_head(&format!("Sign in to {name}"), "Approve this device in your browser", html! { a.btn href=(back) { "Cancel" } }))
        div.card {
            @if !d.start.user_code.is_empty() {
                p.muted { "Your code" }
                p style="font-size:2rem;font-weight:700;letter-spacing:.15em;font-family:ui-monospace,monospace" { (d.start.user_code) " " (super::copy_button(&d.start.user_code)) }
            }
            p { a.btn.primary href=(link) target="_blank" rel="noopener" { "Open verification page ↗" } }
            p.small.muted { "Waiting for approval… this page checks every " (d.start.interval.max(2)) "s. Code expires in " (remaining / 60) " min." }
        }
    };
    let refresh = d.start.interval.clamp(2, 10);
    let html = page(&format!("Sign in · {name}"), Nav::Providers, &Flash::default(), body).into_string().replacen("<head>", &format!("<head><meta http-equiv=\"refresh\" content=\"{refresh}\">"), 1);
    Html(html).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_from_redirects() {
        let p = flows::url_pairs("code=abc&state=xyz");
        assert_eq!(code_from("claude", "code=abc&state=xyz", &p).as_deref(), Some("abc"));
        let p = flows::url_pairs("user_id=1&access_token=t");
        assert_eq!(code_from("zed", "user_id=1&access_token=t", &p).as_deref(), Some("?user_id=1&access_token=t"));
        let p = flows::url_pairs("u=ENC");
        assert_eq!(code_from("xiaomi-mimo", "u=ENC", &p).as_deref(), Some("ENC"));
    }
}
