use axum::Form;
use axum::extract::{Query, State};
use axum::response::{IntoResponse, Redirect, Response};
use axum_extra::extract::CookieJar;
use axum_extra::extract::cookie::Cookie;
use maud::{Markup, html};
use serde::Deserialize;

use super::{Flash, Nav, page as layout, page_head, redirect_err, redirect_ok};
use crate::AppState;
use crate::auth;

pub async fn login_page(
    State(state): State<AppState>,
    jar: CookieJar,
    Query(flash): Query<Flash>,
) -> Response {
    if auth::valid_session(&state.db, &jar) {
        return Redirect::to("/dashboard").into_response();
    }
    let default_pw = auth::verify_password(&state.db, "123456");
    super::bare_page("Sign in", html! {
        div.brand { (super::mark()) "FastRouter" }
        p.tag { "One endpoint for every model you pay for." }
        @if let Some(err) = &flash.err { div.flash.err { (err) } }
        div.card {
            form method="post" action="/login" {
                label for="password" { "Password" }
                input #password type="password" name="password" autofocus required autocomplete="current-password";
                button.btn.primary type="submit" { "Sign in" }
            }
        }
        @if default_pw {
            p.small.muted style="text-align:center;margin-top:16px" { "Default password is " code { "123456" } " — change it in Settings." }
        }
    })
    .into_response()
}

#[derive(Deserialize)]
pub struct LoginForm {
    password: String,
}

pub async fn login(
    State(state): State<AppState>,
    jar: CookieJar,
    Form(f): Form<LoginForm>,
) -> Response {
    if auth::verify_password(&state.db, &f.password) {
        (
            jar.add(auth::session_cookie(&state.db)),
            Redirect::to("/dashboard"),
        )
            .into_response()
    } else {
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        redirect_err("/login", "Wrong password")
    }
}

pub async fn logout(jar: CookieJar) -> Response {
    (
        jar.remove(Cookie::build(auth::SESSION_COOKIE).path("/")),
        Redirect::to("/login"),
    )
        .into_response()
}

pub async fn page(State(state): State<AppState>, Query(flash): Query<Flash>) -> Markup {
    let st = crate::chat::accounts::settings(&state.db);
    let strategy = st["fallbackStrategy"].as_str().unwrap_or("fill-first").to_string();
    let combo_strategy = st["comboStrategy"].as_str().unwrap_or("fallback").to_string();
    let sticky = st["stickyRoundRobinLimit"].as_i64().unwrap_or(3);
    let combo_sticky = st["comboStickyRoundRobinLimit"].as_i64().unwrap_or(1);
    let cc_filter = st["ccFilterNaming"] == serde_json::json!(true);
    let clients = state.db.setting_json("oauthClients");
    let cfg = &state.config;
    let body = html! {
        (page_head("Settings", "Routing behaviour and dashboard access.", html! {}))
        div.grid.g2 {
            div.card {
                h2 { "Routing" }
                form method="post" action="/dashboard/settings/routing" {
                    div.stack {
                        p.small.muted { "Accounts of the same provider:" }
                        label.check { input type="radio" name="strategy" value="fill-first" checked[strategy != "round-robin"]; "Fill first — use the highest-priority account until it is rate-limited" }
                        label.check { input type="radio" name="strategy" value="round-robin" checked[strategy == "round-robin"]; "Round-robin — spread load across accounts" }
                        div { label { "Sticky requests per account (round-robin)" } input type="number" name="sticky" min="1" value=(sticky); }
                        p.small.muted { "Combos:" }
                        label.check { input type="radio" name="combo_strategy" value="fallback" checked[combo_strategy != "round-robin"]; "Fallback — always try models in order" }
                        label.check { input type="radio" name="combo_strategy" value="round-robin" checked[combo_strategy == "round-robin"]; "Round-robin — rotate the starting model" }
                        div { label { "Sticky requests per combo model (round-robin)" } input type="number" name="combo_sticky" min="1" value=(combo_sticky); }
                        label.check { input type="checkbox" name="cc_filter" value="1" checked[cc_filter]; "Skip Claude Code's title-generation / warm-up requests" }
                        div { button.btn.primary type="submit" { "Save" } }
                    }
                }
                p.small.muted style="margin-top:14px" {
                    "Failing accounts are cooled down per model with exponential backoff (2s → 5min); quota errors use the provider's reset time."
                }
            }
            div.card {
                h2 { "Dashboard password" }
                form method="post" action="/dashboard/settings/password" {
                    div.stack {
                        div { label for="current" { "Current password" } input #current type="password" name="current" required autocomplete="current-password"; }
                        div { label for="new" { "New password" } input #new type="password" name="new" required minlength="6" autocomplete="new-password"; }
                        div { label for="confirm" { "Confirm new password" } input #confirm type="password" name="confirm" required minlength="6" autocomplete="new-password"; }
                        div { button.btn.primary type="submit" { "Change password" } }
                    }
                }
            }
            div.card {
                h2 { "Google OAuth clients" }
                p.small.muted { "Gemini CLI and Antigravity sign in with those tools' own public OAuth clients. FastRouter finds them automatically at startup: Gemini CLI's from a local install or Google's " code { "@google/gemini-cli-core" } " npm package, Antigravity's from a local Antigravity app. You can also set them here or with the environment variables " code { "GOOGLE_OAUTH_CLIENT_ID" } ", " code { "GOOGLE_OAUTH_CLIENT_SECRET" } ", " code { "ANTIGRAVITY_OAUTH_CLIENT_ID" } ", " code { "ANTIGRAVITY_OAUTH_CLIENT_SECRET" } ". Changes apply after a restart." }
                form method="post" action="/dashboard/settings/oauth-clients" {
                    div.form-grid {
                        @for name in crate::secrets::NAMES {
                            div.full {
                                label { code { (name) } @if !crate::secrets::get(name).is_empty() { " " span.badge.ok { "set" } } }
                                input type=(if name.ends_with("SECRET") { "password" } else { "text" }) name=(name) autocomplete="off" value=(clients[*name].as_str().unwrap_or(""));
                            }
                        }
                    }
                    div style="margin-top:10px" { button.btn.primary type="submit" { "Save" } }
                }
            }
            div.card {
                h2 { "Server" }
                table { tbody {
                    tr { td.muted { "Listen" } td.mono { (cfg.host) ":" (cfg.port) } }
                    tr { td.muted { "Data directory" } td.mono { (cfg.data_dir.display()) } }
                    tr { td.muted { "Version" } td.mono { (env!("CARGO_PKG_VERSION")) } }
                } }
            }
        }
    };
    layout("Settings", Nav::Settings, &flash, body)
}

#[derive(Deserialize)]
pub struct PasswordForm {
    current: String,
    new: String,
    confirm: String,
}

pub async fn change_password(
    State(state): State<AppState>,
    Form(f): Form<PasswordForm>,
) -> Response {
    const BACK: &str = "/dashboard/settings";
    if !auth::verify_password(&state.db, &f.current) {
        return redirect_err(BACK, "Current password is wrong");
    }
    if f.new != f.confirm {
        return redirect_err(BACK, "New passwords do not match");
    }
    if f.new.len() < 6 {
        return redirect_err(BACK, "Use at least 6 characters");
    }
    match state
        .db
        .set_setting("password_hash", &auth::hash_password(&f.new))
    {
        Ok(()) => redirect_ok(BACK, "Password changed"),
        Err(e) => redirect_err(BACK, &e.to_string()),
    }
}

#[derive(Deserialize)]
pub struct RoutingForm {
    strategy: String,
    combo_strategy: String,
    sticky: Option<String>,
    combo_sticky: Option<String>,
    cc_filter: Option<String>,
}

pub async fn save_routing(State(state): State<AppState>, Form(f): Form<RoutingForm>) -> Response {
    let mut st = crate::chat::accounts::settings(&state.db);
    let num = |v: &Option<String>, d: i64| v.as_deref().and_then(|s| s.trim().parse::<i64>().ok()).filter(|n| *n >= 1).unwrap_or(d);
    st["fallbackStrategy"] = serde_json::json!(if f.strategy == "round-robin" { "round-robin" } else { "fill-first" });
    st["comboStrategy"] = serde_json::json!(if f.combo_strategy == "round-robin" { "round-robin" } else { "fallback" });
    st["stickyRoundRobinLimit"] = serde_json::json!(num(&f.sticky, 3));
    st["comboStickyRoundRobinLimit"] = serde_json::json!(num(&f.combo_sticky, 1));
    st["ccFilterNaming"] = serde_json::json!(f.cc_filter.is_some());
    match state.db.set_setting_json("settings", &st) {
        Ok(()) => redirect_ok("/dashboard/settings", "Routing settings saved"),
        Err(e) => redirect_err("/dashboard/settings", &e.to_string()),
    }
}

pub async fn save_oauth_clients(State(state): State<AppState>, Form(f): Form<std::collections::HashMap<String, String>>) -> Response {
    let mut m = serde_json::Map::new();
    for name in crate::secrets::NAMES {
        if let Some(v) = f.get(*name).map(|s| s.trim()).filter(|s| !s.is_empty()) {
            m.insert((*name).to_string(), serde_json::json!(v));
        }
    }
    match state.db.set_setting_json("oauthClients", &serde_json::Value::Object(m)) {
        Ok(()) => redirect_ok("/dashboard/settings", "OAuth clients saved — restart FastRouter to apply"),
        Err(e) => redirect_err("/dashboard/settings", &e.to_string()),
    }
}
