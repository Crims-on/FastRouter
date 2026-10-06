use axum::Form;
use axum::extract::{Query, State};
use axum::response::{IntoResponse, Redirect, Response};
use axum_extra::extract::CookieJar;
use axum_extra::extract::cookie::Cookie;
use maud::{DOCTYPE, Markup, html};
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
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { "Sign in · FastRouter" }
                link rel="stylesheet" href="/static/app.css";
            }
            body {
                div.login-wrap { div.login {
                    div.brand { span.brand-mark { "⚡" } "FastRouter" }
                    p.muted style="text-align:center;margin-bottom:18px" { "One endpoint for every AI provider." }
                    @if let Some(err) = &flash.err { div.flash.err { (err) } }
                    div.card {
                        form method="post" action="/login" {
                            label for="password" { "Password" }
                            input #password type="password" name="password" autofocus required autocomplete="current-password";
                            button.btn.primary type="submit" style="width:100%;margin-top:14px" { "Sign in" }
                        }
                    }
                    @if default_pw {
                        p.small.muted style="text-align:center;margin-top:14px" {
                            "Default password is " code { "123456" } " — change it in Settings."
                        }
                    }
                } }
            }
        }
    }
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
    let strategy = state
        .db
        .get_setting("strategy")
        .unwrap_or_else(|| "fallback".into());
    let cfg = &state.config;
    let body = html! {
        (page_head("Settings", "Routing behaviour and dashboard access.", html! {}))
        div.grid.g2 {
            div.card {
                h2 { "Routing strategy" }
                p.small.muted { "How FastRouter picks between multiple connections that share a prefix (e.g. several accounts of the same provider)." }
                form method="post" action="/dashboard/settings/routing" {
                    div.stack {
                        label.check { input type="radio" name="strategy" value="fallback" checked[strategy == "fallback"]; "Fallback — always try by priority order" }
                        label.check { input type="radio" name="strategy" value="round-robin" checked[strategy == "round-robin"]; "Round-robin — spread load across connections" }
                        div { button.btn.primary type="submit" { "Save" } }
                    }
                }
                p.small.muted style="margin-top:14px" {
                    "Failed connections are cooled down automatically: 60s after a 429, 15s after a 5xx, 5 minutes after an auth error."
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
}

pub async fn save_routing(State(state): State<AppState>, Form(f): Form<RoutingForm>) -> Response {
    let s = if f.strategy == "round-robin" {
        "round-robin"
    } else {
        "fallback"
    };
    match state.db.set_setting("strategy", s) {
        Ok(()) => redirect_ok("/dashboard/settings", "Routing strategy saved"),
        Err(e) => redirect_err("/dashboard/settings", &e.to_string()),
    }
}
