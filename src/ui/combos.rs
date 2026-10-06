use std::collections::BTreeMap;

use axum::Form;
use axum::extract::{Path, Query, State};
use axum::response::Response;
use maud::{Markup, html};
use serde::Deserialize;

use super::{Flash, Nav, copy_button, page as layout, page_head, redirect_err, redirect_ok};
use crate::AppState;

pub const ALL_KINDS: &[&str] = &["llm", "image", "tts", "embedding", "stt", "imageToText", "video", "webSearch", "webFetch"];

/// (model id, owned_by, kind) for every model `/v1/models` would list.
pub async fn available_models(db: &crate::db::Db) -> Vec<(String, String, String)> {
    crate::api::models::build_models_list(db, ALL_KINDS, true)
        .await
        .into_iter()
        .filter_map(|m| {
            let id = m["id"].as_str()?.to_string();
            let owner = m["owned_by"].as_str().unwrap_or("").to_string();
            let kind = m["kind"].as_str().or_else(|| m["type"].as_str()).unwrap_or("llm").to_string();
            Some((id, owner, kind))
        })
        .collect()
}

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
        .await
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
    if crate::registry::REG.entry(name).is_some() || state.db.list_nodes(None).iter().any(|n| n["prefix"] == name) {
        return redirect_err(
            "/dashboard/combos",
            "That name is already used as a provider id or prefix",
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
    let all = available_models(&state.db).await;
    let mut groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for (id, owner, kind) in &all {
        let key = if owner == "combo" {
            "Combos".to_string()
        } else {
            id.split_once('/')
                .map(|(p, _)| format!("{p}/"))
                .unwrap_or_else(|| "(no prefix)".into())
        };
        groups.entry(key).or_default().push((id.clone(), kind.clone()));
    }
    let aliases = state.db.model_aliases();
    let body = html! {
        (page_head("Models", "Every model id your clients can request, as listed by GET /v1/models.", html! {}))
        div.card style="margin-bottom:16px" {
            h2 { "Model aliases" }
            p.small.muted { "Short names that map to a full " code { "provider/model" } " id (e.g. " code { "sonnet → cc/claude-sonnet-4-5" } ")." }
            @if !aliases.is_empty() {
                div.table-wrap { table {
                    thead { tr { th { "Alias" } th { "Target" } th {} } }
                    tbody { @for (a, t) in &aliases { tr {
                        td.mono { (a) }
                        td.mono { (t.as_str().unwrap_or("")) }
                        td style="text-align:right" {
                            form.inline method="post" action=(format!("/dashboard/aliases/{}/delete", super::urlencode(a))) { button.btn.sm.danger type="submit" { "Delete" } }
                        }
                    } } }
                } }
            }
            form method="post" action="/dashboard/aliases" style="margin-top:10px" {
                div.form-grid {
                    div { label { "Alias" } input type="text" name="alias" required placeholder="sonnet"; }
                    div { label { "Target model" } input type="text" name="target" required placeholder="cc/claude-sonnet-4-5" list="all-models"; }
                }
                datalist #all-models { @for (id, _, _) in &all { option value=(id) {} } }
                div style="margin-top:10px" { button.btn.primary type="submit" { "Save alias" } }
            }
        }
        @if all.is_empty() {
            div.card { div.empty { "Nothing yet — " a href="/dashboard/providers" { "connect a provider" } "." } }
        }
        div.grid.g2 {
            @for (group, ids) in &groups {
                div.card {
                    div.card-head { h2.mono { (group) } span.badge { (ids.len()) } }
                    div style="max-height:360px;overflow:auto" {
                        table { tbody { @for (id, kind) in ids { tr {
                            td.mono.small style="padding-left:0" { (id) }
                            td.small.muted { @if kind != "llm" { (kind) } }
                            td style="text-align:right;padding-right:0" { (copy_button(id)) }
                        } } } }
                    }
                }
            }
        }
    };
    layout("Models", Nav::Models, &flash, body)
}

#[derive(Deserialize)]
pub struct AliasForm {
    alias: String,
    target: String,
}

pub async fn save_alias(State(state): State<AppState>, Form(f): Form<AliasForm>) -> Response {
    let (a, t) = (f.alias.trim(), f.target.trim());
    if a.is_empty() || t.is_empty() || a.contains(char::is_whitespace) {
        return redirect_err("/dashboard/models", "Alias and target are required (no spaces)");
    }
    match state.db.set_model_alias(a, t) {
        Ok(()) => redirect_ok("/dashboard/models", &format!("Alias “{a}” saved")),
        Err(e) => redirect_err("/dashboard/models", &e.to_string()),
    }
}

pub async fn delete_alias(State(state): State<AppState>, Path(alias): Path<String>) -> Response {
    match state.db.delete_model_alias(&alias) {
        Ok(()) => redirect_ok("/dashboard/models", "Alias deleted"),
        Err(e) => redirect_err("/dashboard/models", &e.to_string()),
    }
}
