use axum::Form;
use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Response};
use maud::{Markup, html};
use serde::Deserialize;

use super::{Flash, Nav, ago, mask, page as layout, page_head, redirect_err, redirect_ok};
use crate::AppState;
use crate::catalog::{self, PROVIDERS, Provider, Tier};
use crate::db::{ProviderConn, now};
use crate::router::Target;

fn logo(p: &Provider) -> Markup {
    let initial: String = p
        .name
        .chars()
        .next()
        .unwrap_or('?')
        .to_uppercase()
        .collect();
    html! { span.logo style=(format!("background:{}", p.color)) { (initial) } }
}

pub async fn list(State(state): State<AppState>, Query(flash): Query<Flash>) -> Markup {
    let conns = state.db.list_connections();
    let tiers = [Tier::Subscription, Tier::Cheap, Tier::Free, Tier::Custom];
    let body = html! {
        (page_head("Providers", "Connect upstream accounts. Multiple connections per provider are load-balanced or used as fallbacks.", html! {}))
        @for tier in tiers {
            section.tier {
                h3 { (tier.label()) }
                div.prov-grid {
                    @for p in PROVIDERS.iter().filter(|p| p.tier == tier) {
                        @let n = conns.iter().filter(|c| c.provider == p.id).count();
                        @let on = conns.iter().filter(|c| c.provider == p.id && c.enabled).count();
                        a.prov href=(format!("/dashboard/providers/{}", p.id)) {
                            (logo(p))
                            div style="min-width:0;flex:1" {
                                div.name { (p.name) }
                                div.meta { (p.format.label()) " format" }
                            }
                            @if n > 0 {
                                span class=(if on > 0 { "badge ok" } else { "badge" }) { span.dot {} (on) "/" (n) }
                            }
                        }
                    }
                }
            }
        }
    };
    layout("Providers", Nav::Providers, &flash, body)
}

fn conn_form(p: &Provider, c: Option<&ProviderConn>, action: &str, submit: &str) -> Markup {
    let default_name = c
        .map(|c| c.name.clone())
        .unwrap_or_else(|| p.name.to_string());
    let default_prefix = c.map(|c| c.prefix.clone()).unwrap_or_else(|| {
        if p.custom {
            String::new()
        } else {
            p.id.to_string()
        }
    });
    html! {
        form method="post" action=(action) {
            div.form-grid {
                div {
                    label for="name" { "Name" }
                    input #name type="text" name="name" value=(default_name) required;
                }
                div {
                    label for="prefix" { "Model prefix " span.hint { "— clients call " code { "prefix/model" } } }
                    input #prefix type="text" name="prefix" value=(default_prefix) required pattern="[A-Za-z0-9_.\\-]+" placeholder="e.g. mylocal";
                }
                div.full {
                    label for="api_key" { "API key "
                        @if c.is_some() { span.hint { "(leave blank to keep the current key)" } }
                        @else if !p.needs_key { span.hint { "(optional)" } }
                    }
                    input #api_key type="password" name="api_key" autocomplete="off"
                        placeholder=(if p.id == "anthropic" { "sk-ant-api… or Claude Code OAuth token sk-ant-oat…" } else { "sk-…" })
                        required[c.is_none() && p.needs_key];
                    @if !p.key_url.is_empty() && c.is_none() {
                        div.small.muted style="margin-top:4px" { "Get a key at " a href=(p.key_url) target="_blank" rel="noopener" { (p.key_url) } }
                    }
                }
                div {
                    label for="base_url" { "Base URL " @if !p.custom { span.hint { "(optional override)" } } }
                    input #base_url type="url" name="base_url" value=(c.and_then(|c| c.base_url.clone()).unwrap_or_default())
                        placeholder=(if p.custom { "https://host/v1" } else { p.base_url }) required[p.custom];
                }
                div {
                    label for="priority" { "Priority " span.hint { "(lower is tried first)" } }
                    input #priority type="number" name="priority" value=(c.map(|c| c.priority).unwrap_or(0));
                }
            }
            div style="margin-top:14px" { button.btn.primary type="submit" { (submit) } }
        }
    }
}

pub async fn detail(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(flash): Query<Flash>,
) -> Response {
    let Some(p) = catalog::get(&id) else {
        return redirect_err("/dashboard/providers", "Unknown provider");
    };
    let conns: Vec<ProviderConn> = state
        .db
        .list_connections()
        .into_iter()
        .filter(|c| c.provider == p.id)
        .collect();

    let body = html! {
        div.page-head {
            div style="display:flex;gap:14px;align-items:center" {
                div.prov style="padding:0;border:0;box-shadow:none;background:none" { (logo(p)) }
                div {
                    h1 { (p.name) }
                    p { span.badge.accent { (p.format.label()) " API" } " "
                        @if p.base_url.is_empty() { span.muted.small { "custom endpoint" } }
                        @else { code.muted { (p.base_url) } } }
                }
            }
            div.actions { a.btn href="/dashboard/providers" { "← All providers" } }
        }

        div.card {
            div.card-head { h2 { "Connections" } span.muted.small { (conns.len()) " configured" } }
            @if conns.is_empty() {
                div.empty { "No connections yet. Add one below." }
            } @else {
                div.table-wrap { table {
                    thead { tr { th { "Name" } th { "Prefix" } th { "Key" } th.num { "Priority" } th { "Status" } th { "Models" } th {} } }
                    tbody { @for c in &conns {
                        @let cooling = state.router.cooling_until(&c.id);
                        tr {
                            td { strong { (c.name) } div.small.muted { "added " (ago(c.created_at)) } }
                            td.mono { (c.prefix) "/" }
                            td.mono.small { @if c.api_key.is_empty() { span.muted { "none" } } @else { (mask(&c.api_key)) } }
                            td.num { (c.priority) }
                            td {
                                @if !c.enabled { span.badge { "disabled" } }
                                @else if let Some(d) = cooling { span.badge.warn title="Temporarily skipped after errors" { "cooldown " (d.as_secs()) "s" } }
                                @else { span.badge.ok { span.dot {} "active" } }
                            }
                            td.small { @if c.models.is_empty() { span.muted { "defaults" } } @else { (c.models.len()) " fetched" } }
                            td { div.actions {
                                form.inline method="post" action=(format!("/dashboard/connections/{}/test", c.id)) { button.btn.sm type="submit" { "Test" } }
                                form.inline method="post" action=(format!("/dashboard/connections/{}/models", c.id)) { button.btn.sm type="submit" { "Fetch models" } }
                                form.inline method="post" action=(format!("/dashboard/connections/{}/toggle", c.id)) { button.btn.sm type="submit" { @if c.enabled { "Disable" } @else { "Enable" } } }
                                form.inline method="post" action=(format!("/dashboard/connections/{}/delete", c.id)) { button.btn.sm.danger type="submit" { "Delete" } }
                            } }
                        }
                        tr { td colspan="7" style="padding-top:0" {
                            details.edit {
                                summary.small.muted { "Edit connection ▸" }
                                (conn_form(p, Some(c), &format!("/dashboard/connections/{}/update", c.id), "Save changes"))
                            }
                        } }
                    } }
                } }
            }
        }

        div.split style="margin-top:16px" {
            div.card {
                h2 { "Add connection" }
                (conn_form(p, None, &format!("/dashboard/providers/{}/connections", p.id), "Add connection"))
            }
            div.card {
                h2 { "Models" }
                @let fetched: Vec<&String> = conns.iter().flat_map(|c| c.models.iter()).collect();
                @if fetched.is_empty() && p.models.is_empty() {
                    p.muted.small { "Use “Fetch models” on a connection to discover available models." }
                } @else {
                    p.small.muted { "Address these as " code { (conns.first().map(|c| c.prefix.as_str()).unwrap_or(p.id)) "/<model>" } "." }
                    div style="max-height:340px;overflow:auto" {
                        @if fetched.is_empty() {
                            @for m in p.models { div.mono.small style="padding:3px 0" { (m) } }
                        } @else {
                            @for m in fetched { div.mono.small style="padding:3px 0" { (m) } }
                        }
                    }
                }
            }
        }
    };
    layout(p.name, Nav::Providers, &flash, body).into_response()
}

#[derive(Deserialize)]
pub struct ConnForm {
    name: String,
    prefix: String,
    #[serde(default)]
    api_key: String,
    #[serde(default)]
    base_url: String,
    #[serde(default)]
    priority: String,
}

fn clean_prefix(s: &str) -> String {
    s.trim()
        .trim_end_matches('/')
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || "-_.".contains(*c))
        .collect()
}

pub async fn create(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Form(f): Form<ConnForm>,
) -> Response {
    let Some(p) = catalog::get(&id) else {
        return redirect_err("/dashboard/providers", "Unknown provider");
    };
    let back = format!("/dashboard/providers/{id}");
    let prefix = clean_prefix(&f.prefix);
    if prefix.is_empty() {
        return redirect_err(&back, "Prefix is required");
    }
    if let Some(other) = state
        .db
        .list_connections()
        .iter()
        .find(|c| c.prefix == prefix && c.provider != p.id)
    {
        return redirect_err(
            &back,
            &format!("Prefix `{prefix}` is already used by {}", other.name),
        );
    }
    if state.db.get_combo(&prefix).is_some() {
        return redirect_err(
            &back,
            &format!("Prefix `{prefix}` collides with a combo name"),
        );
    }
    let base_url = f.base_url.trim().to_string();
    if p.custom && base_url.is_empty() {
        return redirect_err(&back, "Base URL is required for custom providers");
    }
    let conn = ProviderConn {
        id: uuid::Uuid::new_v4().simple().to_string(),
        provider: p.id.to_string(),
        name: f.name.trim().to_string(),
        prefix,
        api_key: f.api_key.trim().to_string(),
        base_url: (!base_url.is_empty()).then_some(base_url),
        priority: f.priority.trim().parse().unwrap_or(0),
        enabled: true,
        models: vec![],
        created_at: now(),
    };
    match state.db.insert_connection(&conn) {
        Ok(()) => redirect_ok(
            &back,
            &format!("Added “{}”. Hit Test to verify it.", conn.name),
        ),
        Err(e) => redirect_err(&back, &format!("Could not save: {e}")),
    }
}

pub async fn update(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Form(f): Form<ConnForm>,
) -> Response {
    let Some(mut c) = state.db.get_connection(&id) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    let back = format!("/dashboard/providers/{}", c.provider);
    let prefix = clean_prefix(&f.prefix);
    if prefix.is_empty() {
        return redirect_err(&back, "Prefix is required");
    }
    if let Some(other) = state
        .db
        .list_connections()
        .iter()
        .find(|o| o.prefix == prefix && o.provider != c.provider)
    {
        return redirect_err(
            &back,
            &format!("Prefix `{prefix}` is already used by {}", other.name),
        );
    }
    c.name = f.name.trim().to_string();
    c.prefix = prefix;
    if !f.api_key.trim().is_empty() {
        c.api_key = f.api_key.trim().to_string();
    }
    let base_url = f.base_url.trim().to_string();
    c.base_url = (!base_url.is_empty()).then_some(base_url);
    c.priority = f.priority.trim().parse().unwrap_or(c.priority);
    state.router.clear_cooldown(&c.id);
    match state.db.update_connection(&c) {
        Ok(()) => redirect_ok(&back, "Connection updated"),
        Err(e) => redirect_err(&back, &format!("Could not save: {e}")),
    }
}

pub async fn toggle(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(mut c) = state.db.get_connection(&id) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    c.enabled = !c.enabled;
    let back = format!("/dashboard/providers/{}", c.provider);
    match state.db.update_connection(&c) {
        Ok(()) => redirect_ok(
            &back,
            if c.enabled {
                "Connection enabled"
            } else {
                "Connection disabled"
            },
        ),
        Err(e) => redirect_err(&back, &e.to_string()),
    }
}

pub async fn delete(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(c) = state.db.get_connection(&id) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    let back = format!("/dashboard/providers/{}", c.provider);
    match state.db.delete_connection(&id) {
        Ok(()) => redirect_ok(&back, &format!("Deleted “{}”", c.name)),
        Err(e) => redirect_err(&back, &e.to_string()),
    }
}

fn target_for(c: ProviderConn) -> Option<Target> {
    let provider = catalog::get(&c.provider)?;
    let model = c
        .models
        .iter()
        .find(|m| provider.models.contains(&m.as_str()))
        .cloned()
        .or_else(|| provider.models.first().map(|m| m.to_string()))
        .or_else(|| c.models.first().cloned())?;
    Some(Target {
        conn: c,
        provider,
        model,
    })
}

pub async fn test(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(c) = state.db.get_connection(&id) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    let back = format!("/dashboard/providers/{}", c.provider);
    let name = c.name.clone();
    let Some(target) = target_for(c) else {
        return redirect_err(&back, "No model to test with — fetch models first");
    };
    match crate::proxy::test_target(&state, &target).await {
        Ok(msg) => redirect_ok(&back, &format!("✓ {name}: {msg}")),
        Err(e) => redirect_err(&back, &format!("✗ {name}: {e}")),
    }
}

pub async fn refresh_models(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(mut c) = state.db.get_connection(&id) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    let back = format!("/dashboard/providers/{}", c.provider);
    let Some(provider) = catalog::get(&c.provider) else {
        return redirect_err(&back, "Unknown provider");
    };
    let target = Target {
        conn: c.clone(),
        provider,
        model: String::new(),
    };
    match crate::proxy::fetch_models(&state, &target).await {
        Ok(models) if models.is_empty() => redirect_err(&back, "Upstream returned no models"),
        Ok(models) => {
            let n = models.len();
            c.models = models;
            match state.db.update_connection(&c) {
                Ok(()) => redirect_ok(&back, &format!("Fetched {n} models for “{}”", c.name)),
                Err(e) => redirect_err(&back, &e.to_string()),
            }
        }
        Err(e) => redirect_err(&back, &format!("Could not fetch models: {e}")),
    }
}
