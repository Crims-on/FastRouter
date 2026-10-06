use axum::Form;
use axum::extract::{Path, Query, State};
use axum::response::Response;
use maud::{Markup, html};
use serde::Deserialize;

use super::{
    Flash, Nav, ago, copy_button, mask, page as layout, page_head, redirect_err, redirect_ok,
};
use crate::AppState;
use crate::db::{ApiKey, now};

pub async fn page(State(state): State<AppState>, Query(flash): Query<Flash>) -> Markup {
    let keys = state.db.list_api_keys();
    let required = crate::auth::api_key_required(&state);
    let env_locked = state.config.require_api_key.is_some();
    let body = html! {
        (page_head("API Keys", "Keys that clients present to FastRouter (not your upstream provider keys).", html! {}))
        div.split {
            div.card {
                div.card-head { h2 { "Keys" } span.muted.small { (keys.len()) " total" } }
                @if keys.is_empty() {
                    div.empty { "No keys yet." }
                } @else {
                    div.table-wrap { table {
                        thead { tr { th { "Name" } th { "Key" } th { "Created" } th { "Last used" } th {} } }
                        tbody { @for k in &keys { tr {
                            td { strong { (k.name) } }
                            td { div style="display:flex;gap:6px;align-items:center" { code { (mask(&k.key)) } (copy_button(&k.key)) } }
                            td.small.muted { (ago(k.created_at)) }
                            td.small.muted { @if let Some(t) = k.last_used { (ago(t)) } @else { "never" } }
                            td { form.inline method="post" action=(format!("/dashboard/keys/{}/delete", k.id)) {
                                button.btn.sm.danger type="submit" { "Revoke" }
                            } }
                        } } }
                    } }
                }
            }
            div.stack {
                div.card {
                    h2 { "Create key" }
                    form method="post" action="/dashboard/keys" {
                        label for="kname" { "Name" }
                        input #kname type="text" name="name" placeholder="laptop / claude-code / ci" required;
                        div style="margin-top:12px" { button.btn.primary type="submit" { "Create key" } }
                    }
                }
                div.card {
                    h2 { "Enforcement" }
                    p.small.muted { "When required, every /v1 request must send a valid key via " code { "Authorization: Bearer" } " or " code { "x-api-key" } "." }
                    form method="post" action="/dashboard/keys/require" {
                        input type="hidden" name="require" value=(if required { "0" } else { "1" });
                        p { "Currently: " @if required { span.badge.warn { "required" } } @else { span.badge { "optional" } } }
                        @if env_locked {
                            p.small.muted { "Set by the REQUIRE_API_KEY environment variable." }
                        } @else {
                            button.btn type="submit" { @if required { "Make optional" } @else { "Require API key" } }
                        }
                    }
                }
            }
        }
    };
    layout("API Keys", Nav::Keys, &flash, body)
}

#[derive(Deserialize)]
pub struct CreateForm {
    name: String,
}

pub async fn create(State(state): State<AppState>, Form(f): Form<CreateForm>) -> Response {
    let key = ApiKey {
        id: uuid::Uuid::new_v4().simple().to_string(),
        name: f.name.trim().to_string(),
        key: format!("fr-{}", crate::auth::random_token()),
        created_at: now(),
        last_used: None,
    };
    match state.db.insert_api_key(&key) {
        Ok(()) => redirect_ok(
            "/dashboard/keys",
            &format!("Created key “{}”: {}", key.name, key.key),
        ),
        Err(e) => redirect_err("/dashboard/keys", &e.to_string()),
    }
}

pub async fn delete(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.db.delete_api_key(&id) {
        Ok(()) => redirect_ok("/dashboard/keys", "Key revoked"),
        Err(e) => redirect_err("/dashboard/keys", &e.to_string()),
    }
}

#[derive(Deserialize)]
pub struct RequireForm {
    require: String,
}

pub async fn toggle_require(State(state): State<AppState>, Form(f): Form<RequireForm>) -> Response {
    let on = f.require == "1";
    if on && !state.db.has_api_keys() {
        return redirect_err(
            "/dashboard/keys",
            "Create a key first, or every client would be locked out",
        );
    }
    match state
        .db
        .set_setting("require_api_key", if on { "1" } else { "0" })
    {
        Ok(()) => redirect_ok(
            "/dashboard/keys",
            if on {
                "API key now required"
            } else {
                "API key now optional"
            },
        ),
        Err(e) => redirect_err("/dashboard/keys", &e.to_string()),
    }
}
