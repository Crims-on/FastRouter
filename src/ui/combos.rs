use std::collections::BTreeMap;

use axum::Form;
use axum::extract::{Path, Query, State};
use axum::response::Response;
use maud::{Markup, html};
use serde::Deserialize;

use super::{Flash, Nav, copy_button, page as layout, page_head, redirect_err, redirect_ok};
use crate::AppState;
use crate::router::available_models;

fn combo_form(name: &str, models: &[String], editing: bool) -> Markup {
    html! {
        form method="post" action="/dashboard/combos" {
            div.stack {
                div {
                    label for=(format!("cn-{name}")) { "Combo name " span.hint { "— used as the " code { "model" } " in requests" } }
                    @if editing {
                        input type="hidden" name="name" value=(name);
                        input id=(format!("cn-{name}")) type="text" value=(name) disabled;
                    } @else {
                        input id="cn-" type="text" name="name" placeholder="premium-coding" required pattern="[A-Za-z0-9_.:\\-]+";
                    }
                }
                div {
                    label for=(format!("cm-{name}")) { "Models, in fallback order " span.hint { "— one per line" } }
                    textarea id=(format!("cm-{name}")) name="models" required placeholder="anthropic/claude-sonnet-4-5\nglm/glm-4.6\nopenrouter/z-ai/glm-4.6" {
                        (models.join("\n"))
                    }
                }
                div { button.btn.primary type="submit" { @if editing { "Save combo" } @else { "Create combo" } } }
            }
        }
    }
}

pub async fn page(State(state): State<AppState>, Query(flash): Query<Flash>) -> Markup {
    let combos = state.db.list_combos();
    let models: Vec<String> = available_models(&state.db)
        .into_iter()
        .filter(|m| m.1 != "combo")
        .map(|m| m.0)
        .collect();
    let body = html! {
        (page_head("Combos", "Named fallback chains. When a model fails, hits a rate limit or runs out of quota, the next one is tried automatically.", html! {}))
        div.split {
            div.stack {
                @if combos.is_empty() {
                    div.card { div.empty { "No combos yet. Create your first one →" } }
                }
                @for c in &combos {
                    div.card {
                        div.card-head {
                            div style="display:flex;gap:8px;align-items:center" {
                                h2.mono { (c.name) }
                                (copy_button(&c.name))
                            }
                            form.inline method="post" action=(format!("/dashboard/combos/{}/delete", super::urlencode(&c.name))) {
                                button.btn.sm.danger type="submit" { "Delete" }
                            }
                        }
                        div.chain {
                            @for (i, m) in c.models.iter().enumerate() {
                                @if i > 0 { span.arrow { "→" } }
                                @let known = models.contains(m);
                                span.step title=(if known { "" } else { "Not found in your connected providers' model lists — will still be tried if its prefix matches." }) {
                                    span.n { (i + 1) }
                                    (m)
                                    @if !known { span.badge.warn { "?" } }
                                }
                            }
                        }
                        details.edit {
                            summary.small.muted { "Edit ▸" }
                            (combo_form(&c.name, &c.models, true))
                        }
                    }
                }
            }
            div.stack {
                div.card {
                    h2 { "New combo" }
                    (combo_form("", &[], false))
                }
                div.card {
                    h2 { "Available models" }
                    @if models.is_empty() {
                        p.small.muted { "Connect a provider to see models here." }
                    } @else {
                        div style="max-height:320px;overflow:auto" {
                            @for m in &models { div.mono.small style="padding:2px 0" { (m) } }
                        }
                    }
                }
            }
        }
    };
    layout("Combos", Nav::Combos, &flash, body)
}

#[derive(Deserialize)]
pub struct ComboForm {
    name: String,
    models: String,
}

pub async fn save(State(state): State<AppState>, Form(f): Form<ComboForm>) -> Response {
    let name = f.name.trim();
    if name.is_empty() || name.contains('/') || name.contains(char::is_whitespace) {
        return redirect_err(
            "/dashboard/combos",
            "Combo names cannot be empty or contain spaces or “/”",
        );
    }
    if state.db.list_connections().iter().any(|c| c.prefix == name) {
        return redirect_err(
            "/dashboard/combos",
            "That name is already used as a provider prefix",
        );
    }
    let models: Vec<String> = f
        .models
        .lines()
        .map(str::trim)
        .filter(|m| !m.is_empty() && *m != name)
        .map(str::to_owned)
        .collect();
    if models.is_empty() {
        return redirect_err("/dashboard/combos", "Add at least one model");
    }
    match state.db.upsert_combo(name, &models) {
        Ok(()) => redirect_ok("/dashboard/combos", &format!("Saved combo “{name}”")),
        Err(e) => redirect_err("/dashboard/combos", &e.to_string()),
    }
}

pub async fn delete(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    match state.db.delete_combo(&name) {
        Ok(()) => redirect_ok("/dashboard/combos", &format!("Deleted combo “{name}”")),
        Err(e) => redirect_err("/dashboard/combos", &e.to_string()),
    }
}

pub async fn models_page(State(state): State<AppState>, Query(flash): Query<Flash>) -> Markup {
    let all = available_models(&state.db);
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (id, owner) in &all {
        let key = if owner == "combo" {
            "Combos".to_string()
        } else {
            id.split_once('/')
                .map(|(p, _)| format!("{p}/"))
                .unwrap_or_default()
        };
        groups.entry(key).or_default().push(id.clone());
    }
    let body = html! {
        (page_head("Models", "Every model id your clients can request, as listed by GET /v1/models.", html! {}))
        @if all.is_empty() {
            div.card { div.empty { "Nothing yet — " a href="/dashboard/providers" { "connect a provider" } "." } }
        }
        div.grid.g2 {
            @for (group, ids) in &groups {
                div.card {
                    div.card-head { h2.mono { (group) } span.badge { (ids.len()) } }
                    div style="max-height:360px;overflow:auto" {
                        table { tbody { @for id in ids { tr {
                            td.mono.small style="padding-left:0" { (id) }
                            td style="text-align:right;padding-right:0" { (copy_button(id)) }
                        } } } }
                    }
                }
            }
        }
    };
    layout("Models", Nav::Models, &flash, body)
}
