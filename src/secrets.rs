//! OAuth client credentials that must not live in the repository.
//!
//! Some providers (Gemini CLI, Antigravity) sign in through Google "installed
//! application" OAuth clients. Their IDs/secrets are referenced from the data
//! files as `{{NAME}}` placeholders and resolved at startup from, in order:
//! the environment (`NAME` or `FASTROUTER_NAME`), then the dashboard setting
//! `oauthClients` stored in the database.

use std::collections::HashMap;
use std::sync::{LazyLock, OnceLock};

use regex::Regex;

pub const NAMES: &[&str] = &["GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "ANTIGRAVITY_OAUTH_CLIENT_ID", "ANTIGRAVITY_OAUTH_CLIENT_SECRET"];

static OVERRIDES: OnceLock<HashMap<String, String>> = OnceLock::new();

/// Loads dashboard-stored values; call before the registry is first used.
pub fn init(db: &crate::db::Db) {
    let v = db.setting_json("oauthClients");
    let mut m = HashMap::new();
    for n in NAMES {
        if let Some(s) = v[*n].as_str().filter(|s| !s.trim().is_empty()) {
            m.insert(n.to_string(), s.trim().to_string());
        }
    }
    let _ = OVERRIDES.set(m);
}

pub fn get(name: &str) -> String {
    for k in [name.to_string(), format!("FASTROUTER_{name}")] {
        if let Ok(v) = std::env::var(&k) {
            if !v.trim().is_empty() {
                return v.trim().to_string();
            }
        }
    }
    OVERRIDES.get().and_then(|m| m.get(name).cloned()).unwrap_or_default()
}

pub fn is_set(name: &str) -> bool {
    !get(name).is_empty()
}

static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\{\{([A-Z0-9_]+)\}\}").unwrap());

/// Replaces `{{NAME}}` placeholders in embedded JSON text.
pub fn substitute(text: &str) -> String {
    PLACEHOLDER.replace_all(text, |c: &regex::Captures| get(&c[1])).into_owned()
}
