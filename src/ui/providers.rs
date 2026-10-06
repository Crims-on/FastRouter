//! Provider catalogue, provider detail and connection management pages.

use std::collections::HashMap;

use axum::Form;
use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Response};
use maud::{Markup, html};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Flash, Nav, mask, page as layout, page_head, redirect_err, redirect_ok, urlencode};
use crate::AppState;
use crate::chat::accounts;
use crate::exec::{is_anthropic_compatible, is_openai_compatible};
use crate::jsv::{js_string, now_ms, parse_iso_ms, truthy};
use crate::oauth::flows::{self, Flow};
use crate::registry::REG;

/// Display info for one provider (registry entry or custom node).
pub struct PInfo {
    pub id: String,
    pub name: String,
    pub color: String,
    pub icon_text: String,
    pub category: String,
    pub alias: String,
    pub website: String,
    pub notice: String,
    pub api_key_url: String,
    pub signup_url: String,
    pub auth_modes: Vec<String>,
    pub hint: String,
    pub regions: Vec<(String, String)>,
    pub hidden: bool,
    pub deprecated: bool,
    pub node: Option<Value>,
    pub kinds: Vec<String>,
}

pub fn pinfo(db: &crate::db::Db, id: &str) -> Option<PInfo> {
    if let Some(e) = REG.entry(id) {
        let d = &e["display"];
        let s = |v: &Value| v.as_str().unwrap_or("").to_string();
        let name = s(&d["name"]);
        return Some(PInfo {
            id: id.into(),
            icon_text: if truthy(&d["textIcon"]) { s(&d["textIcon"]) } else { name.chars().take(2).collect::<String>().to_uppercase() },
            name,
            color: if truthy(&d["color"]) { s(&d["color"]) } else { "#5b5bf0".into() },
            category: s(&e["category"]),
            alias: REG.alias_of(id),
            website: s(&d["website"]),
            notice: s(&d["notice"]["text"]),
            api_key_url: s(&d["notice"]["apiKeyUrl"]),
            signup_url: s(&d["notice"]["signupUrl"]),
            auth_modes: e["authModes"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default(),
            hint: s(&e["authHint"]),
            regions: e["regions"].as_array().map(|a| a.iter().map(|r| (s(&r["id"]), s(&r["label"]))).collect()).unwrap_or_default(),
            hidden: e["hidden"] == json!(true),
            deprecated: d["deprecated"] == json!(true),
            node: None,
            kinds: crate::api::models::provider_kinds(id),
        });
    }
    let n = db.get_node(id)?;
    let anth = n["type"] == "anthropic-compatible";
    let name = n["name"].as_str().filter(|s| !s.is_empty()).unwrap_or(id).to_string();
    Some(PInfo {
        id: id.into(),
        icon_text: name.chars().take(2).collect::<String>().to_uppercase(),
        name,
        color: if anth { "#c96442".into() } else { "#10a37f".into() },
        category: "custom".into(),
        alias: n["prefix"].as_str().unwrap_or(id).to_string(),
        website: String::new(),
        notice: format!("{} endpoint at {}", if anth { "Anthropic-compatible" } else { "OpenAI-compatible" }, n["baseUrl"].as_str().unwrap_or("")),
        api_key_url: String::new(),
        signup_url: String::new(),
        auth_modes: vec!["apikey".into()],
        hint: String::new(),
        regions: vec![],
        hidden: false,
        deprecated: false,
        node: Some(n),
        kinds: vec!["llm".into()],
    })
}

fn logo(p: &PInfo) -> Markup {
    html! { span.logo style=(format!("background:{}", p.color)) { (p.icon_text.chars().take(2).collect::<String>()) } }
}

fn category_label(c: &str) -> &'static str {
    match c {
        "oauth" => "Subscriptions & OAuth logins",
        "apikey" => "API key providers",
        "freeTier" => "Free tiers & local engines",
        "free" => "Free — no account needed",
        "webCookie" => "Web session (cookie)",
        "custom" => "Custom endpoints",
        _ => "Other",
    }
}

#[derive(Deserialize, Default)]
pub struct ListQuery {
    q: Option<String>,
    ok: Option<String>,
    err: Option<String>,
}

pub async fn list(State(state): State<AppState>, Query(q): Query<ListQuery>) -> Markup {
    let flash = Flash { ok: q.ok.clone(), err: q.err.clone() };
    let conns = state.db.list_connections();
    let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
    for c in &conns {
        let e = counts.entry(c["provider"].as_str().unwrap_or("").to_string()).or_default();
        e.0 += 1;
        if c["isActive"] != json!(false) {
            e.1 += 1;
        }
    }
    let needle = q.q.clone().unwrap_or_default().to_lowercase();
    let mut infos: Vec<PInfo> = REG.entries.iter().filter_map(|e| pinfo(&state.db, e["id"].as_str()?)).collect();
    for n in state.db.list_nodes(None) {
        if let Some(i) = n["id"].as_str().and_then(|id| pinfo(&state.db, id)) {
            infos.push(i);
        }
    }
    infos.retain(|p| needle.is_empty() || p.name.to_lowercase().contains(&needle) || p.id.contains(&needle) || p.alias.contains(&needle));
    let order = ["oauth", "free", "freeTier", "apikey", "webCookie", "custom"];
    let card = |p: &PInfo| {
        let (n, on) = counts.get(&p.id).copied().unwrap_or((0, 0));
        let free = accounts::is_free_no_auth(&p.id);
        html! {
            a.prov href=(format!("/dashboard/providers/{}", urlencode(&p.id))) {
                (logo(p))
                div style="min-width:0;flex:1" {
                    div.name { (p.name) @if p.deprecated { " " span.badge.warn { "deprecated" } } }
                    div.meta.mono { (p.alias) "/" }
                }
                @if n > 0 {
                    span class=(if on > 0 { "badge ok" } else { "badge" }) { span.dot {} (on) "/" (n) }
                } @else if free {
                    span.badge.ok { "ready" }
                }
            }
        }
    };
    let body = html! {
        (page_head("Providers", "Connect accounts and keys. Several connections per provider are rotated or used as fallbacks automatically.", html! {
            form method="get" action="/dashboard/providers" style="display:flex;gap:6px" {
                input type="search" name="q" value=(q.q.clone().unwrap_or_default()) placeholder="Search providers…" style="width:220px";
            }
        }))
        @for cat in order {
            @let items: Vec<&PInfo> = infos.iter().filter(|p| p.category == cat && !p.hidden).collect();
            @if !items.is_empty() || cat == "custom" {
                section.tier {
                    h3 { (category_label(cat)) " " span.muted { "(" (items.len()) ")" } }
                    div.prov-grid { @for p in &items { (card(p)) } }
                    @if cat == "custom" {
                        details.card style="margin-top:10px" {
                            summary { strong { "+ Add an OpenAI- or Anthropic-compatible endpoint" } }
                            form method="post" action="/dashboard/providers/new-node" style="margin-top:12px" {
                                div.form-grid {
                                    div { label { "Type" } select name="ty" {
                                        option value="openai-compatible" { "OpenAI-compatible (/chat/completions)" }
                                        option value="openai-compatible-responses" { "OpenAI Responses API (/responses)" }
                                        option value="anthropic-compatible" { "Anthropic-compatible (/messages)" }
                                    } }
                                    div { label { "Name" } input type="text" name="name" required placeholder="My LM Studio"; }
                                    div { label { "Model prefix " span.hint { "— clients call " code { "prefix/model" } } } input type="text" name="prefix" required pattern="[A-Za-z0-9_.\\-]+" placeholder="lmstudio"; }
                                    div { label { "Base URL" } input type="url" name="base_url" required placeholder="http://localhost:1234/v1"; }
                                    div.full { label { "API key " span.hint { "(optional)" } } input type="password" name="api_key" autocomplete="off"; }
                                }
                                div style="margin-top:12px" { button.btn.primary type="submit" { "Create endpoint" } }
                            }
                        }
                    }
                }
            }
        }
        @let hidden: Vec<&PInfo> = infos.iter().filter(|p| p.hidden).collect();
        @if !hidden.is_empty() {
            details style="margin-top:14px" {
                summary.muted { "More providers (" (hidden.len()) ")" }
                div.prov-grid style="margin-top:10px" { @for p in &hidden { (card(p)) } }
            }
        }
    };
    layout("Providers", Nav::Providers, &flash, body)
}

// ---------------------------------------------------------------------------
// Detail
// ---------------------------------------------------------------------------

fn conn_label(c: &Value) -> String {
    ["displayName", "name", "email"].iter().map(|k| &c[*k]).find(|v| truthy(v)).map(js_string).unwrap_or_else(|| c["id"].as_str().unwrap_or("").chars().take(8).collect())
}

fn auth_label(c: &Value) -> String {
    let m = c["providerSpecificData"]["authMethod"].as_str().unwrap_or("");
    let t = c["authType"].as_str().unwrap_or("apikey");
    if m.is_empty() { t.to_string() } else { format!("{t} · {m}") }
}

fn status_cell(c: &Value) -> Markup {
    let locks: Vec<(String, i64)> = c
        .as_object()
        .map(|o| o.iter().filter(|(k, _)| k.starts_with(accounts::MODEL_LOCK_PREFIX)).filter_map(|(k, v)| v.as_str().and_then(parse_iso_ms).filter(|t| *t > now_ms()).map(|t| (k[accounts::MODEL_LOCK_PREFIX.len()..].to_string(), t))).collect())
        .unwrap_or_default();
    let expired = c["expiresAt"].as_str().and_then(parse_iso_ms).map(|t| t < now_ms()).unwrap_or(false);
    html! {
        @if c["isActive"] == json!(false) { span.badge { "disabled" } }
        @else if !locks.is_empty() {
            span.badge.warn title=(locks.iter().map(|(m, t)| format!("{m}: {}", crate::chat::util::format_retry_after(*t))).collect::<Vec<_>>().join("\n")) { "cooling " (locks.len()) }
        }
        @else if c["testStatus"] == "unavailable" { span.badge.err { "error" } }
        @else if expired && !truthy(&c["refreshToken"]) { span.badge.warn { "expired" } }
        @else { span.badge.ok { span.dot {} "active" } }
        @if let Some(e) = c["lastError"].as_str().filter(|e| !e.is_empty()) {
            div.small.muted style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title=(e) { (e) }
        }
    }
}

/// Extra provider-specific fields shown in the add/edit form.
fn psd_fields(p: &PInfo) -> Vec<(&'static str, &'static str, &'static str)> {
    match p.id.as_str() {
        "azure" => vec![("azureEndpoint", "Azure endpoint", "https://myresource.openai.azure.com"), ("deployment", "Deployment name (optional)", "gpt-4o"), ("apiVersion", "API version (optional)", "2024-10-01-preview"), ("organization", "Organization (optional)", "")],
        "cloudflare-ai" => vec![("accountId", "Cloudflare Account ID", "")],
        "aws-polly" => vec![("accessKeyId", "AWS Access Key ID", "AKIA…"), ("region", "Region", "us-east-1")],
        "vertex" | "vertex-partner" => vec![("projectId", "Project ID (optional with SA JSON)", ""), ("location", "Location", "us-central1")],
        "google-pse" => vec![("cx", "Search engine ID (cx)", "")],
        "selfhosted-embedding" | "selfhosted-stt" | "selfhosted-tts" | "searxng" | "sdwebui" | "comfyui" | "coqui" | "tortoise" | "ollama-local" | "lm-studio" | "vllm" | "llamacpp" => vec![("baseUrl", "Server base URL", "http://localhost:8080/v1")],
        _ => vec![],
    }
}

fn key_label(p: &PInfo) -> (&'static str, String) {
    match (p.category.as_str(), p.id.as_str()) {
        (_, "vertex" | "vertex-partner") => ("Service account JSON or API key", "{ \"type\": \"service_account\", … }".into()),
        (_, "aws-polly") => ("AWS Secret Access Key", String::new()),
        ("webCookie", _) => ("Cookie value", p.hint.clone()),
        (_, "qoder" | "qoder-cn") => ("Personal Access Token", "pt-…".into()),
        _ => ("API key", if p.hint.is_empty() { "sk-…".into() } else { p.hint.clone() }),
    }
}

fn add_key_form(p: &PInfo) -> Markup {
    let (label, placeholder) = key_label(p);
    let textarea = matches!(p.id.as_str(), "vertex" | "vertex-partner");
    let optional = accounts::is_free_no_auth(&p.id) || REG.media(&p.id)["ttsConfig"]["authType"] == "none" || REG.media(&p.id)["searchConfig"]["authType"] == "none";
    html! {
        form method="post" action=(format!("/dashboard/providers/{}/connections", urlencode(&p.id))) {
            div.form-grid {
                div { label { "Name" } input type="text" name="name" placeholder=(p.name); }
                div { label { "Priority " span.hint { "(lower = tried first)" } } input type="number" name="priority" placeholder="auto"; }
                div.full {
                    label { (label) @if optional { span.hint { " (optional)" } } }
                    @if textarea { textarea name="api_key" rows="5" placeholder=(placeholder) required[!optional] {} }
                    @else { input type="password" name="api_key" autocomplete="off" placeholder=(placeholder) required[!optional]; }
                    @if !p.api_key_url.is_empty() { div.small.muted style="margin-top:4px" { "Get one at " a href=(p.api_key_url) target="_blank" rel="noopener" { (p.api_key_url) } } }
                }
                @if !p.regions.is_empty() {
                    div { label { "Region" } select name="psd_region" { @for (id, l) in &p.regions { option value=(id) { (l) } } } }
                }
                @for (k, l, ph) in psd_fields(p) {
                    div { label { (l) } input type="text" name=(format!("psd_{k}")) placeholder=(ph); }
                }
            }
            details style="margin-top:10px" {
                summary.small.muted { "Advanced ▸" }
                div.form-grid style="margin-top:8px" {
                    div.full { label { "HTTP(S)/SOCKS proxy for this connection " span.hint { "(optional)" } } input type="text" name="proxy" placeholder="http://user:pass@host:port"; }
                    div.full { label { "Extra providerSpecificData (JSON) " span.hint { "(optional)" } } textarea name="psd_json" rows="3" placeholder="{\"baseUrl\": \"https://…\"}" {} }
                }
            }
            div style="margin-top:12px" { button.btn.primary type="submit" { "Add connection" } }
        }
    }
}

fn oauth_section(p: &PInfo) -> Markup {
    let id = p.id.as_str();
    let f = flows::flow(id);
    let start = |label: &str, extra: Markup| html! {
        form method="post" action=(format!("/dashboard/oauth/{}/start", urlencode(id))) {
            (extra)
            button.btn.primary type="submit" { (label) }
        }
    };
    let device = |label: &str, extra: Markup| html! {
        form method="post" action=(format!("/dashboard/oauth/{}/device", urlencode(id))) {
            (extra)
            button.btn.primary type="submit" { (label) }
        }
    };
    let import = |method: &str, fields: &[(&str, &str, bool)], label: &str| html! {
        form method="post" action=(format!("/dashboard/providers/{}/import/{method}", urlencode(id))) {
            div.form-grid {
                @for (name, l, area) in fields {
                    div.full { label { (l) }
                        @if *area { textarea name=(name) rows="3" required {} } @else { input type="text" name=(name) autocomplete="off"; }
                    }
                }
            }
            div style="margin-top:10px" { button.btn type="submit" { (label) } }
        }
    };
    html! {
        @match f {
            Some(Flow::AuthCode { .. }) | Some(Flow::BrowserToken) => {
                div.card {
                    h2 { "Sign in" }
                    p.small.muted { "Opens the provider's login page. When it redirects back, FastRouter stores the tokens and refreshes them automatically. If the redirect can't reach this server (remote access), paste the final URL on the next page." }
                    @if id == "gitlab" {
                        (start("Sign in with GitLab", html! {
                            div.form-grid style="margin-bottom:10px" {
                                div { label { "GitLab URL" } input type="url" name="meta_baseUrl" placeholder="https://gitlab.com"; }
                                div { label { "OAuth application Client ID" } input type="text" name="meta_clientId" required; }
                                div.full { label { "Client secret " span.hint { "(confidential apps only)" } } input type="password" name="meta_clientSecret"; }
                            }
                        }))
                    } @else {
                        (start(&format!("Sign in with {}", p.name), html! {}))
                    }
                }
            }
            Some(Flow::Device) => {
                div.card {
                    h2 { "Sign in" }
                    p.small.muted { "You'll get a code to confirm in your browser; this page polls until it's approved." }
                    @if id == "kiro" {
                        (device("AWS Builder ID login", html! { input type="hidden" name="auth_method" value="builder-id"; }))
                        details style="margin-top:10px" {
                            summary.small.muted { "AWS IAM Identity Center (enterprise) ▸" }
                            (device("Identity Center login", html! {
                                input type="hidden" name="auth_method" value="idc";
                                div.form-grid style="margin:8px 0" {
                                    div { label { "Start URL" } input type="url" name="start_url" placeholder="https://d-xxxx.awsapps.com/start" required; }
                                    div { label { "Region" } input type="text" name="region" placeholder="us-east-1"; }
                                }
                            }))
                        }
                        div style="display:flex;gap:8px;margin-top:10px" {
                            form method="post" action="/dashboard/oauth/kiro/start" { input type="hidden" name="meta_idp" value="google"; button.btn type="submit" { "Google login" } }
                            form method="post" action="/dashboard/oauth/kiro/start" { input type="hidden" name="meta_idp" value="github"; button.btn type="submit" { "GitHub login" } }
                        }
                    } @else {
                        (device(&format!("Sign in with {}", p.name), html! {}))
                    }
                }
            }
            _ => {}
        }
        @match id {
            "cursor" => {
                div.card {
                    h2 { "Import from Cursor IDE" }
                    p.small.muted { "Read " code { "cursorAuth/accessToken" } " and " code { "storage.serviceMachineId" } " from Cursor's " code { "state.vscdb" } " (Linux: ~/.config/Cursor/User/globalStorage, macOS: ~/Library/Application Support/Cursor/User/globalStorage)." }
                    (import("token", &[("accessToken", "Access token", true), ("machineId", "Machine ID", false)], "Import"))
                }
            }
            "zed" => {
                div.card {
                    h2 { "Import existing Zed credentials" }
                    (import("token", &[("accessToken", "Access token", true), ("userId", "User ID", false), ("systemId", "System ID (optional)", false)], "Import"))
                }
            }
            "codex" => {
                details.card { summary { strong { "Import a ChatGPT access token" } }
                    p.small.muted style="margin-top:8px" { "Tokens imported this way are not refreshed; sign in instead for long-lived access." }
                    (import("token", &[("accessToken", "Access token (JWT)", true), ("name", "Name (optional)", false)], "Import")) }
            }
            "kiro" => {
                details.card { summary { strong { "Import refresh token / API key" } }
                    div style="margin-top:8px" {
                        h3 { "Refresh token" }
                        (import("import", &[("refreshToken", "Refresh token", true), ("clientId", "Client ID (Identity Center only)", false), ("clientSecret", "Client secret (Identity Center only)", false), ("region", "Region", false), ("profileArn", "Profile ARN (optional)", false)], "Import"))
                        h3 style="margin-top:14px" { "API key" }
                        (import("api-key", &[("apiKey", "Kiro API key", false), ("region", "Region (default us-east-1)", false)], "Validate & add"))
                    }
                }
            }
            "iflow" => {
                details.card { summary { strong { "Use a platform.iflow.cn cookie" } }
                    (import("cookie", &[("cookie", "Cookie (must contain BXAuth=…)", true)], "Import")) }
            }
            "gitlab" => {
                details.card { summary { strong { "Use a Personal Access Token" } }
                    (import("pat", &[("token", "Personal access token", false), ("baseUrl", "GitLab URL (optional)", false)], "Verify & add")) }
            }
            "xiaomi-mimo" => {
                details.card { summary { strong { "Add an API key (sk-…) or session pass token" } }
                    (import("api-key", &[("apiKey", "API key (sk-…)", false), ("baseUrl", "Base URL (optional)", false), ("region", "Region (cn/sgp/ams/ru/in)", false), ("mimoPassToken", "Pass token (optional, session login)", false), ("mimoUserId", "User ID (with pass token)", false), ("uid", "UID (optional)", false)], "Add")) }
            }
            _ => {}
        }
    }
}

fn wants_key_form(p: &PInfo) -> bool {
    if p.node.is_some() {
        return true;
    }
    if accounts::is_free_no_auth(&p.id) {
        return false;
    }
    match flows::flow(&p.id) {
        None => true,
        Some(_) => p.auth_modes.iter().any(|m| m == "apikey"),
    }
}

pub async fn detail(State(state): State<AppState>, Path(id): Path<String>, Query(flash): Query<Flash>) -> Response {
    let Some(p) = pinfo(&state.db, &id) else {
        return redirect_err("/dashboard/providers", "Unknown provider");
    };
    let conns = state.db.connections_for(&id, false);
    let models = REG.models_by_provider_id(&id);
    let st = accounts::settings(&state.db);
    let custom: Vec<String> = st["customModels"].as_array().into_iter().flatten().filter(|m| m["providerAlias"] == p.alias.as_str()).filter_map(|m| m["id"].as_str().map(str::to_owned)).collect();
    let free = accounts::is_free_no_auth(&id);
    let body = html! {
        div.page-head {
            div style="display:flex;gap:14px;align-items:center" {
                (logo(&p))
                div {
                    h1 { (p.name) }
                    p {
                        span.badge.accent { (category_label(&p.category)) } " "
                        span.small.muted { "model prefix " code { (p.alias) "/" } }
                        @if !p.website.is_empty() { " · " a.small href=(p.website) target="_blank" rel="noopener" { "website" } }
                    }
                }
            }
            div.actions { a.btn href="/dashboard/providers" { "← All providers" } }
        }
        @if !p.notice.is_empty() { div.flash style="background:var(--panel-2)" { (p.notice) } }
        @if free { div.flash.ok { "This provider needs no account — requests to " code { (p.alias) "/<model>" } " work out of the box." } }

        div.card {
            div.card-head { h2 { "Connections" } span.muted.small { (conns.len()) " configured" } }
            @if conns.is_empty() {
                div.empty { @if free { "Optional — add a connection only to route through a proxy." } @else { "No connections yet. Add one below." } }
            } @else {
                div.table-wrap { table {
                    thead { tr { th { "Account" } th { "Auth" } th { "Credential" } th.num { "Priority" } th { "Status" } th {} } }
                    tbody { @for c in &conns {
                        @let cid = c["id"].as_str().unwrap_or("");
                        tr {
                            td { strong { (conn_label(c)) } @if let Some(e) = c["email"].as_str().filter(|e| Some(*e) != c["name"].as_str()) { div.small.muted { (e) } } }
                            td.small { (auth_label(c)) }
                            td.mono.small {
                                @if let Some(k) = c["apiKey"].as_str().filter(|s| !s.is_empty()) { (mask(k)) }
                                @else if let Some(k) = c["accessToken"].as_str().filter(|s| !s.is_empty()) { (mask(k)) }
                                @else { span.muted { "—" } }
                                @if let Some(e) = c["expiresAt"].as_str() { div.small.muted { "expires " (e.get(..16).unwrap_or(e).replace('T', " ")) } }
                            }
                            td.num { (c["priority"]) }
                            td { (status_cell(c)) }
                            td { div.actions {
                                form.inline method="post" action=(format!("/dashboard/connections/{cid}/test")) { button.btn.sm type="submit" { "Test" } }
                                @if truthy(&c["refreshToken"]) || id == "github" || id.starts_with("vertex") { form.inline method="post" action=(format!("/dashboard/connections/{cid}/refresh")) { button.btn.sm type="submit" { "Refresh" } } }
                                form.inline method="post" action=(format!("/dashboard/connections/{cid}/unlock")) { button.btn.sm type="submit" title="Clear cooldowns and errors" { "Reset" } }
                                form.inline method="post" action=(format!("/dashboard/connections/{cid}/toggle")) { button.btn.sm type="submit" { @if c["isActive"] == json!(false) { "Enable" } @else { "Disable" } } }
                                form.inline method="post" action=(format!("/dashboard/connections/{cid}/delete")) { button.btn.sm.danger type="submit" { "Delete" } }
                            } }
                        }
                        tr { td colspan="6" style="padding-top:0" {
                            details.edit {
                                summary.small.muted { "Edit ▸" }
                                form method="post" action=(format!("/dashboard/connections/{cid}/update")) {
                                    div.form-grid {
                                        div { label { "Name" } input type="text" name="name" value=(c["name"].as_str().unwrap_or("")); }
                                        div { label { "Priority" } input type="number" name="priority" value=(c["priority"]); }
                                        div.full { label { "Replace key/token " span.hint { "(leave blank to keep)" } } input type="password" name="api_key" autocomplete="off"; }
                                        div.full { label { "Proxy" } input type="text" name="proxy" value=(c["providerSpecificData"]["connectionProxyUrl"].as_str().unwrap_or("")) placeholder="http://host:port"; }
                                        div.full { label { "Enabled models " span.hint { "(optional, one per line — restricts this account)" } } textarea name="enabled_models" rows="2" { (c["providerSpecificData"]["enabledModels"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join("\n")).unwrap_or_default()) } }
                                        div.full { label { "providerSpecificData (JSON)" } textarea.mono name="psd_json" rows="4" { (serde_json::to_string_pretty(&c["providerSpecificData"]).unwrap_or_default()) } }
                                    }
                                    div style="margin-top:10px" { button.btn.primary type="submit" { "Save" } }
                                }
                            }
                        } }
                    } }
                } }
            }
        }

        div.split style="margin-top:16px" {
            div.stack {
                (oauth_section(&p))
                @if wants_key_form(&p) {
                    div.card { h2 { @if p.category == "webCookie" { "Add session cookie" } @else { "Add API key" } } (add_key_form(&p)) }
                }
                @if let Some(n) = &p.node {
                    div.card {
                        h2 { "Endpoint settings" }
                        form method="post" action=(format!("/dashboard/providers/{}/node", urlencode(&id))) {
                            div.form-grid {
                                div { label { "Name" } input type="text" name="name" value=(n["name"].as_str().unwrap_or("")); }
                                div { label { "Prefix" } input type="text" name="prefix" value=(n["prefix"].as_str().unwrap_or("")) pattern="[A-Za-z0-9_.\\-]+"; }
                                div.full { label { "Base URL" } input type="url" name="base_url" value=(n["baseUrl"].as_str().unwrap_or("")); }
                                @if is_openai_compatible(&id) {
                                    div { label { "API" } select name="api_type" {
                                        option value="chat" selected[n["apiType"] != "responses"] { "Chat Completions" }
                                        option value="responses" selected[n["apiType"] == "responses"] { "Responses" }
                                    } }
                                }
                            }
                            div style="margin-top:10px;display:flex;gap:8px" { button.btn.primary type="submit" { "Save" } }
                        }
                        form method="post" action=(format!("/dashboard/providers/{}/node/delete", urlencode(&id))) style="margin-top:8px" {
                            button.btn.sm.danger type="submit" { "Delete endpoint and its connections" }
                        }
                    }
                }
            }
            div.stack {
                div.card {
                    h2 { "Models" }
                    p.small.muted { "Request them as " code { (p.alias) "/<model>" } ". Any model id the upstream accepts works, listed or not." }
                    div style="max-height:340px;overflow:auto" {
                        @if models.is_empty() && custom.is_empty() { p.small.muted { "No built-in list — the upstream catalogue is fetched live for " code { "/v1/models" } "." } }
                        @for m in models {
                            @let mid = m["id"].as_str().unwrap_or("");
                            div.copy-row style="padding:2px 0" {
                                span.mono.small { (p.alias) "/" (mid) }
                                @if let Some(k) = m["type"].as_str().or_else(|| m["kind"].as_str()) { span.badge { (k) } }
                                (super::copy_button(&format!("{}/{mid}", p.alias)))
                            }
                        }
                        @for m in &custom { div.copy-row style="padding:2px 0" { span.mono.small { (p.alias) "/" (m) } span.badge.accent { "custom" } (super::copy_button(&format!("{}/{m}", p.alias))) } }
                    }
                    details style="margin-top:10px" {
                        summary.small.muted { "Custom model ids ▸" }
                        form method="post" action=(format!("/dashboard/providers/{}/custom-models", urlencode(&id))) {
                            textarea name="models" rows="4" placeholder="one model id per line" { (custom.join("\n")) }
                            div style="margin-top:8px" { button.btn.sm type="submit" { "Save" } }
                        }
                    }
                }
            }
        }
    };
    layout(&p.name, Nav::Providers, &flash, body).into_response()
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

fn back(provider: &str) -> String {
    format!("/dashboard/providers/{}", urlencode(provider))
}

fn psd_from_form(f: &HashMap<String, String>) -> Result<Value, String> {
    let mut psd = match f.get("psd_json").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        Some(j) => serde_json::from_str::<Value>(j).map_err(|e| format!("providerSpecificData is not valid JSON: {e}"))?,
        None => json!({}),
    };
    if !psd.is_object() {
        return Err("providerSpecificData must be a JSON object".into());
    }
    for (k, v) in f {
        if let Some(key) = k.strip_prefix("psd_").filter(|k| *k != "json") {
            if !v.trim().is_empty() {
                psd[key] = json!(v.trim());
            }
        }
    }
    if let Some(p) = f.get("proxy").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        psd["connectionProxyEnabled"] = json!(true);
        psd["connectionProxyUrl"] = json!(p);
    }
    Ok(psd)
}

pub async fn create(State(state): State<AppState>, Path(id): Path<String>, Form(f): Form<HashMap<String, String>>) -> Response {
    let Some(p) = pinfo(&state.db, &id) else {
        return redirect_err("/dashboard/providers", "Unknown provider");
    };
    let b = back(&id);
    let key = f.get("api_key").map(|s| s.trim().to_string()).unwrap_or_default();
    let mut psd = match psd_from_form(&f) {
        Ok(p) => p,
        Err(e) => return redirect_err(&b, &e),
    };
    if let Some(n) = &p.node {
        for (a, k) in [("baseUrl", "baseUrl"), ("apiType", "apiType"), ("prefix", "prefix")] {
            if !truthy(&psd[k]) && truthy(&n[a]) {
                psd[k] = n[a].clone();
            }
        }
    }
    let name = f.get("name").map(|s| s.trim()).filter(|s| !s.is_empty()).map(str::to_owned).unwrap_or_else(|| format!("{} {}", p.name, state.db.connections_for(&id, false).len() + 1));
    let mut doc = json!({"provider": id, "authType": if p.category == "webCookie" { "cookie" } else { "apikey" }, "name": name, "providerSpecificData": psd, "testStatus": "untested"});
    if !key.is_empty() {
        doc["apiKey"] = json!(key);
    }
    if let Some(pr) = f.get("priority").and_then(|s| s.trim().parse::<i64>().ok()) {
        doc["priority"] = json!(pr);
    }
    match state.db.insert_connection(&doc) {
        Ok(cid) => {
            let msg = match test_connection(&state, &cid).await {
                Ok(m) => format!("Added “{name}” — ✓ {m}"),
                Err(e) => format!("Added “{name}”, but the test failed: {e}"),
            };
            redirect_ok(&b, &msg)
        }
        Err(e) => redirect_err(&b, &format!("Could not save: {e}")),
    }
}

pub async fn update(State(state): State<AppState>, Path(cid): Path<String>, Form(f): Form<HashMap<String, String>>) -> Response {
    let Some(c) = state.db.get_connection(&cid) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    let provider = c["provider"].as_str().unwrap_or("").to_string();
    let b = back(&provider);
    let mut psd = match psd_from_form(&f) {
        Ok(p) => p,
        Err(e) => return redirect_err(&b, &e),
    };
    if f.get("proxy").map(|s| s.trim().is_empty()).unwrap_or(false) {
        psd["connectionProxyEnabled"] = json!(false);
        psd["connectionProxyUrl"] = Value::Null;
    }
    let models: Vec<String> = f.get("enabled_models").map(|s| s.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned).collect()).unwrap_or_default();
    psd["enabledModels"] = if models.is_empty() { Value::Null } else { json!(models) };
    let mut patch = json!({"providerSpecificData": psd});
    // providerSpecificData is replaced, not merged, when edited as JSON.
    let _ = state.db.update_connection(&cid, &json!({"providerSpecificData": null}));
    if let Some(n) = f.get("name").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        patch["name"] = json!(n);
    }
    if let Some(p) = f.get("priority").and_then(|s| s.trim().parse::<i64>().ok()) {
        patch["priority"] = json!(p);
    }
    if let Some(k) = f.get("api_key").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        if truthy(&c["apiKey"]) || !truthy(&c["accessToken"]) {
            patch["apiKey"] = json!(k);
        } else {
            patch["accessToken"] = json!(k);
        }
    }
    match state.db.update_connection(&cid, &patch) {
        Ok(_) => redirect_ok(&b, "Connection updated"),
        Err(e) => redirect_err(&b, &e.to_string()),
    }
}

pub async fn toggle(State(state): State<AppState>, Path(cid): Path<String>) -> Response {
    let Some(c) = state.db.get_connection(&cid) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    let on = c["isActive"] == json!(false);
    let b = back(c["provider"].as_str().unwrap_or(""));
    match state.db.update_connection(&cid, &json!({"isActive": on})) {
        Ok(_) => redirect_ok(&b, if on { "Connection enabled" } else { "Connection disabled" }),
        Err(e) => redirect_err(&b, &e.to_string()),
    }
}

pub async fn delete(State(state): State<AppState>, Path(cid): Path<String>) -> Response {
    let Some(c) = state.db.get_connection(&cid) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    let b = back(c["provider"].as_str().unwrap_or(""));
    match state.db.delete_connection(&cid) {
        Ok(()) => redirect_ok(&b, &format!("Deleted “{}”", conn_label(&c))),
        Err(e) => redirect_err(&b, &e.to_string()),
    }
}

pub async fn unlock(State(state): State<AppState>, Path(cid): Path<String>) -> Response {
    let Some(c) = state.db.get_connection(&cid) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    let mut patch = json!({"testStatus": "active", "lastError": null, "errorCode": null, "lastErrorAt": null, "backoffLevel": 0});
    for k in c.as_object().into_iter().flatten().map(|(k, _)| k).filter(|k| k.starts_with(accounts::MODEL_LOCK_PREFIX)) {
        patch[k] = Value::Null;
    }
    let _ = state.db.update_connection(&cid, &patch);
    redirect_ok(&back(c["provider"].as_str().unwrap_or("")), "Cooldowns and errors cleared")
}

pub async fn refresh(State(state): State<AppState>, Path(cid): Path<String>) -> Response {
    let Some(c) = state.db.get_connection(&cid) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    let provider = c["provider"].as_str().unwrap_or("").to_string();
    let b = back(&provider);
    let creds = accounts::credentials_from_connection(&c);
    let ex = crate::providers::get_executor(&provider);
    match ex.refresh_credentials(&creds).await {
        Some(raw) if !crate::oauth::refresh::is_unrecoverable(&raw) => {
            let patch = crate::oauth::refresh::merge_refreshed(&provider, &creds, &raw);
            let mut persist = patch.clone();
            persist["testStatus"] = json!("active");
            accounts::update_provider_credentials(&state.db, &cid, &persist, &creds["providerSpecificData"]);
            redirect_ok(&b, "Token refreshed")
        }
        Some(raw) => redirect_err(&b, &format!("Refresh rejected ({}) — sign in again", raw["error"].as_str().unwrap_or("error"))),
        None => redirect_err(&b, "Refresh failed — sign in again"),
    }
}

/// Picks a model to test a connection with.
async fn test_model(state: &AppState, c: &Value) -> Option<String> {
    let provider = c["provider"].as_str().unwrap_or("");
    if let Some(m) = c["providerSpecificData"]["enabledModels"][0].as_str() {
        return Some(m.to_string());
    }
    let llm = REG.models_by_provider_id(provider).iter().find(|m| m["type"].is_null() && m["kind"].is_null()).and_then(|m| m["id"].as_str().map(str::to_owned));
    if llm.is_some() {
        return llm;
    }
    let alias = REG.alias_of(provider);
    let st = accounts::settings(&state.db);
    if let Some(m) = st["customModels"].as_array().into_iter().flatten().find(|m| m["providerAlias"] == alias.as_str()).and_then(|m| m["id"].as_str()) {
        return Some(m.to_string());
    }
    let prefix = if is_openai_compatible(provider) || is_anthropic_compatible(provider) {
        c["providerSpecificData"]["prefix"].as_str().map(str::to_owned).or_else(|| state.db.get_node(provider).and_then(|n| n["prefix"].as_str().map(str::to_owned))).unwrap_or_default()
    } else if REG.entry(provider).is_some_and(|e| truthy(&e["modelsFetcher"]) || truthy(&e["passthroughModels"])) {
        // Model catalogue is fetched live; a key check against /models is a better test.
        return None;
    } else {
        alias
    };
    let list = crate::api::models::build_models_list(&state.db, &[crate::api::models::LLM], false).await;
    list.iter().filter_map(|m| m["id"].as_str()).find_map(|id| id.strip_prefix(&format!("{prefix}/")).map(str::to_owned))
}

/// Key check for providers without a fixed model: `GET <base>/models`.
async fn probe_models(c: &Value) -> Result<String, String> {
    let provider = c["provider"].as_str().unwrap_or("");
    let base = c["providerSpecificData"]["baseUrl"].as_str().filter(|s| !s.is_empty()).map(str::to_owned).or_else(|| REG.transport(provider)["baseUrl"].as_str().map(str::to_owned)).ok_or("No model to test with — add a custom model id first")?;
    let base = base.trim_end_matches('/');
    let base = ["/chat/completions", "/messages", "/responses"].iter().find_map(|s| base.strip_suffix(s)).unwrap_or(base);
    let key = [&c["apiKey"], &c["accessToken"]].into_iter().find_map(|v| v.as_str().filter(|s| !s.is_empty())).unwrap_or("");
    let started = std::time::Instant::now();
    let r = crate::exec::http_client(None).get(format!("{base}/models")).bearer_auth(key).header("accept", "application/json").timeout(std::time::Duration::from_secs(20)).send().await.map_err(|e| e.to_string())?;
    let st = r.status().as_u16();
    if (200..300).contains(&st) {
        let v: Value = r.json().await.unwrap_or(Value::Null);
        let n = v["data"].as_array().or_else(|| v["models"].as_array()).map(|a| a.len()).unwrap_or(0);
        Ok(format!("key accepted in {} ms ({n} models listed)", started.elapsed().as_millis()))
    } else {
        let t: String = r.text().await.unwrap_or_default().chars().take(200).collect();
        Err(format!("models endpoint returned HTTP {st}: {t}"))
    }
}

/// Sends a tiny chat request through one specific connection.
pub async fn test_connection(state: &AppState, cid: &str) -> Result<String, String> {
    let c = state.db.get_connection(cid).ok_or("Connection not found")?;
    let provider = c["provider"].as_str().unwrap_or("").to_string();
    let kinds = crate::api::models::provider_kinds(&provider);
    if !kinds.iter().any(|k| k == "llm") && !is_openai_compatible(&provider) && !is_anthropic_compatible(&provider) {
        return Ok(format!("saved (no chat test for {} providers)", kinds.join("/")));
    }
    let model = match test_model(state, &c).await {
        Some(m) => m,
        None => return probe_models(&c).await,
    };
    let creds = accounts::credentials_from_connection(&c);
    let creds = accounts::check_and_refresh_token(&state.db, &provider, &creds).await;
    let mut creds = creds;
    if let Some(n) = state.db.get_node(&provider) {
        for k in ["baseUrl", "apiType"] {
            if !truthy(&creds["providerSpecificData"][k]) && truthy(&n[k]) {
                creds["providerSpecificData"][k] = n[k].clone();
            }
        }
    }
    let started = std::time::Instant::now();
    let res = crate::chat::core::handle_chat_core(crate::chat::core::CoreArgs {
        db: state.db.clone(),
        body: json!({"model": format!("{provider}/{model}"), "messages": [{"role": "user", "content": "Reply with the single word: pong"}], "max_tokens": 16, "stream": false}),
        provider: provider.clone(),
        model: model.clone(),
        creds,
        headers: json!({"accept": "application/json"}),
        endpoint: "dashboard-test".into(),
        source_override: Some("openai".into()),
        connection_id: cid.to_string(),
        api_key: None,
        requested_model: format!("{provider}/{model}"),
        provider_thinking: Value::Null,
        provider_overrides: Value::Null,
    })
    .await;
    let ms = started.elapsed().as_millis();
    if res.ok {
        accounts::clear_account_error(&state.db, cid, &c, Some(&model));
        let _ = state.db.update_connection(cid, &json!({"testStatus": "active"}));
        let bytes = axum::body::to_bytes(res.response.into_body(), 1 << 20).await.unwrap_or_default();
        let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        let text: String = v["choices"][0]["message"]["content"].as_str().unwrap_or("").chars().take(60).collect();
        Ok(format!("{model} answered in {ms} ms{}", if text.is_empty() { String::new() } else { format!(": “{}”", text.trim()) }))
    } else {
        let e = res.error.unwrap_or_else(|| format!("HTTP {}", res.status));
        let _ = state.db.update_connection(cid, &json!({"testStatus": "unavailable", "lastError": e.chars().take(200).collect::<String>(), "errorCode": res.status}));
        Err(format!("{model}: {e}"))
    }
}

pub async fn test(State(state): State<AppState>, Path(cid): Path<String>) -> Response {
    let Some(c) = state.db.get_connection(&cid) else {
        return redirect_err("/dashboard/providers", "Connection not found");
    };
    let b = back(c["provider"].as_str().unwrap_or(""));
    match test_connection(&state, &cid).await {
        Ok(m) => redirect_ok(&b, &format!("✓ {}: {m}", conn_label(&c))),
        Err(e) => redirect_err(&b, &format!("✗ {}: {e}", conn_label(&c))),
    }
}

pub async fn import(State(state): State<AppState>, Path((id, method)): Path<(String, String)>, Form(f): Form<HashMap<String, String>>) -> Response {
    let b = back(&id);
    let fields = json!(f);
    match flows::import(&id, &method, &fields).await {
        Ok(tokens) => match flows::save_connection(&state.db, &id, &tokens) {
            Ok(_) => redirect_ok(&b, "Imported — connection added"),
            Err(e) => redirect_err(&b, &e.to_string()),
        },
        Err(e) => redirect_err(&b, &e),
    }
}

pub async fn save_custom_models(State(state): State<AppState>, Path(id): Path<String>, Form(f): Form<HashMap<String, String>>) -> Response {
    let alias = pinfo(&state.db, &id).map(|p| p.alias).unwrap_or_else(|| id.clone());
    let mut st = accounts::settings(&state.db);
    let mut list: Vec<Value> = st["customModels"].as_array().cloned().unwrap_or_default();
    list.retain(|m| m["providerAlias"] != alias.as_str());
    for l in f.get("models").map(|s| s.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned).collect::<Vec<_>>()).unwrap_or_default() {
        list.push(json!({"id": l, "providerAlias": alias, "type": "llm"}));
    }
    st["customModels"] = json!(list);
    match state.db.set_setting_json("settings", &st) {
        Ok(()) => redirect_ok(&back(&id), "Custom models saved"),
        Err(e) => redirect_err(&back(&id), &e.to_string()),
    }
}

fn clean_prefix(s: &str) -> String {
    s.trim().trim_end_matches('/').chars().filter(|c| c.is_ascii_alphanumeric() || "-_.".contains(*c)).collect()
}

fn prefix_taken(state: &AppState, prefix: &str, except: Option<&str>) -> Option<String> {
    if REG.entries.iter().any(|e| e["id"] == prefix || e["alias"] == prefix || e["aliases"].as_array().map(|a| a.iter().any(|x| x == prefix)).unwrap_or(false)) {
        return Some(format!("`{prefix}` is a built-in provider prefix"));
    }
    if state.db.get_combo(prefix).is_some() {
        return Some(format!("`{prefix}` is a combo name"));
    }
    state.db.list_nodes(None).iter().find(|n| n["prefix"] == prefix && Some(n["id"].as_str().unwrap_or("")) != except).map(|n| format!("`{prefix}` is used by {}", n["name"].as_str().unwrap_or("another endpoint")))
}

pub async fn create_node(State(state): State<AppState>, Form(f): Form<HashMap<String, String>>) -> Response {
    let ty = f.get("ty").map(String::as_str).unwrap_or("openai-compatible");
    let (node_type, api_type, idp) = match ty {
        "anthropic-compatible" => ("anthropic-compatible", "chat", "anthropic-compatible"),
        "openai-compatible-responses" => ("openai-compatible", "responses", "openai-compatible-responses"),
        _ => ("openai-compatible", "chat", "openai-compatible"),
    };
    let prefix = clean_prefix(f.get("prefix").map(String::as_str).unwrap_or(""));
    let base = f.get("base_url").map(|s| s.trim().trim_end_matches('/').to_string()).unwrap_or_default();
    if prefix.is_empty() || base.is_empty() {
        return redirect_err("/dashboard/providers", "Prefix and base URL are required");
    }
    if let Some(e) = prefix_taken(&state, &prefix, None) {
        return redirect_err("/dashboard/providers", &e);
    }
    let id = format!("{idp}-{}", uuid::Uuid::new_v4().simple());
    let name = f.get("name").map(|s| s.trim()).filter(|s| !s.is_empty()).unwrap_or(&prefix).to_string();
    if let Err(e) = state.db.upsert_node(&id, node_type, &json!({"name": name, "prefix": prefix, "baseUrl": base, "apiType": api_type})) {
        return redirect_err("/dashboard/providers", &e.to_string());
    }
    let key = f.get("api_key").map(|s| s.trim().to_string()).unwrap_or_default();
    let mut doc = json!({"provider": id, "authType": "apikey", "name": name, "providerSpecificData": {"baseUrl": base, "prefix": prefix, "apiType": api_type, "nodeName": name}});
    if !key.is_empty() {
        doc["apiKey"] = json!(key);
    }
    let _ = state.db.insert_connection(&doc);
    redirect_ok(&back(&id), &format!("Created “{name}”. Use it as {prefix}/<model>."))
}

pub async fn update_node(State(state): State<AppState>, Path(id): Path<String>, Form(f): Form<HashMap<String, String>>) -> Response {
    let Some(mut n) = state.db.get_node(&id) else {
        return redirect_err("/dashboard/providers", "Endpoint not found");
    };
    let b = back(&id);
    let prefix = clean_prefix(f.get("prefix").map(String::as_str).unwrap_or(""));
    if !prefix.is_empty() {
        if let Some(e) = prefix_taken(&state, &prefix, Some(&id)) {
            return redirect_err(&b, &e);
        }
        n["prefix"] = json!(prefix);
    }
    if let Some(v) = f.get("name").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        n["name"] = json!(v);
    }
    if let Some(v) = f.get("base_url").map(|s| s.trim().trim_end_matches('/')).filter(|s| !s.is_empty()) {
        n["baseUrl"] = json!(v);
    }
    if let Some(v) = f.get("api_type") {
        n["apiType"] = json!(if v == "responses" { "responses" } else { "chat" });
    }
    let ty = n["type"].as_str().unwrap_or("openai-compatible").to_string();
    if let Err(e) = state.db.upsert_node(&id, &ty, &n) {
        return redirect_err(&b, &e.to_string());
    }
    for c in state.db.connections_for(&id, false) {
        let _ = state.db.update_connection(c["id"].as_str().unwrap_or(""), &json!({"providerSpecificData": {"baseUrl": n["baseUrl"], "prefix": n["prefix"], "apiType": n["apiType"]}}));
    }
    redirect_ok(&b, "Endpoint saved")
}

pub async fn delete_node(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.db.delete_node(&id) {
        Ok(()) => redirect_ok("/dashboard/providers", "Endpoint deleted"),
        Err(e) => redirect_err(&back(&id), &e.to_string()),
    }
}
