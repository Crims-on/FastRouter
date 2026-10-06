//! OAuth client credentials that must not live in the repository.
//!
//! Some providers (Gemini CLI, Antigravity) sign in through Google "installed
//! application" OAuth clients. Their IDs/secrets are referenced from the data
//! files as `{{NAME}}` placeholders and resolved at startup from, in order:
//! the environment (`NAME` or `FASTROUTER_NAME`), then the dashboard setting
//! `oauthClients` stored in the database, then (for values still missing)
//! auto-discovery from the user's own installed tools: a local Gemini CLI
//! install or Google's published `@google/gemini-cli-core` package, and a
//! local Antigravity app install. Discovered values are cached in the DB.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, RwLock};

use regex::Regex;
use serde_json::{Value, json};

pub const NAMES: &[&str] = &["GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "ANTIGRAVITY_OAUTH_CLIENT_ID", "ANTIGRAVITY_OAUTH_CLIENT_SECRET"];

static OVERRIDES: LazyLock<RwLock<HashMap<String, String>>> = LazyLock::new(Default::default);

/// Loads dashboard-stored values; call before the registry is first used.
pub fn init(db: &crate::db::Db) {
    let v = db.setting_json("oauthClients");
    let mut m = OVERRIDES.write().unwrap();
    for n in NAMES {
        if let Some(s) = v[*n].as_str().filter(|s| !s.trim().is_empty()) {
            m.insert(n.to_string(), s.trim().to_string());
        }
    }
}

pub fn get(name: &str) -> String {
    for k in [name.to_string(), format!("FASTROUTER_{name}")] {
        if let Ok(v) = std::env::var(&k) {
            if !v.trim().is_empty() {
                return v.trim().to_string();
            }
        }
    }
    OVERRIDES.read().unwrap().get(name).cloned().unwrap_or_default()
}

pub fn is_set(name: &str) -> bool {
    !get(name).is_empty()
}

static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\{\{([A-Z0-9_]+)\}\}").unwrap());

/// Replaces `{{NAME}}` placeholders in embedded JSON text.
pub fn substitute(text: &str) -> String {
    PLACEHOLDER.replace_all(text, |c: &regex::Captures| get(&c[1])).into_owned()
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

static ID_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"OAUTH_CLIENT_ID\s*=\s*['"]([^'"]+)['"]"#).unwrap());
static SECRET_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"OAUTH_CLIENT_SECRET\s*=\s*['"]([^'"]+)['"]"#).unwrap());
static GOOGLE_ID_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[0-9]{6,}-[a-z0-9]{20,}\.apps\.googleusercontent\.com").unwrap());
static GOOGLE_SECRET_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"GOCSPX-[A-Za-z0-9_-]{20,40}").unwrap());

/// `OAUTH_CLIENT_ID = '…'` / `OAUTH_CLIENT_SECRET = '…'` (Gemini CLI source).
pub fn parse_gemini_cli_source(js: &str) -> Option<(String, String)> {
    Some((ID_RE.captures(js)?[1].to_string(), SECRET_RE.captures(js)?[1].to_string()))
}

/// A Google client id with a client secret close to it (bundled app code).
pub fn parse_bundled_client(js: &str) -> Option<(String, String)> {
    for id in GOOGLE_ID_RE.find_iter(js) {
        let lo = id.start().saturating_sub(600);
        let hi = (id.end() + 600).min(js.len());
        let (lo, hi) = (floor_char(js, lo), floor_char(js, hi));
        if let Some(sec) = GOOGLE_SECRET_RE.find(&js[lo..hi]) {
            return Some((id.as_str().to_string(), sec.as_str().to_string()));
        }
    }
    None
}

fn floor_char(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn node_roots() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = vec![];
    if let Ok(o) = std::process::Command::new("npm").args(["root", "-g"]).output() {
        let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !p.is_empty() {
            v.push(p.into());
        }
    }
    for p in ["/usr/lib/node_modules", "/usr/local/lib/node_modules", "/opt/homebrew/lib/node_modules"] {
        v.push(p.into());
    }
    if let Some(h) = dirs::home_dir() {
        for p in [".npm-global/lib/node_modules", ".local/share/pnpm/global/5/node_modules", ".bun/install/global/node_modules", ".volta/tools/shared"] {
            v.push(h.join(p));
        }
    }
    if let Ok(a) = std::env::var("APPDATA") {
        v.push(Path::new(&a).join("npm").join("node_modules"));
    }
    v
}

fn read_small(p: &Path) -> Option<String> {
    let m = std::fs::metadata(p).ok()?;
    (m.is_file() && m.len() < 64 * 1024 * 1024).then(|| std::fs::read_to_string(p).ok()).flatten()
}

/// Looks for an installed Gemini CLI.
pub fn gemini_cli_local(roots: &[PathBuf]) -> Option<(String, String)> {
    let rels = [
        "@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
        "@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
        "@google/gemini-cli/bundle/gemini.js",
    ];
    for r in roots {
        for rel in rels {
            if let Some(js) = read_small(&r.join(rel)) {
                if let Some(c) = parse_gemini_cli_source(&js) {
                    return Some(c);
                }
            }
        }
    }
    None
}

/// Minimal ustar reader: returns the first entry whose path ends with `suffix`.
pub fn tar_find(tar: &[u8], suffix: &str) -> Option<Vec<u8>> {
    let mut pos = 0;
    while pos + 512 <= tar.len() {
        let h = &tar[pos..pos + 512];
        if h.iter().all(|b| *b == 0) {
            return None;
        }
        let field = |a: usize, b: usize| String::from_utf8_lossy(&h[a..b]).trim_end_matches('\0').to_string();
        let name = field(0, 100);
        let prefix = if &h[257..262] == b"ustar" { field(345, 500) } else { String::new() };
        let full = if prefix.is_empty() { name } else { format!("{prefix}/{name}") };
        let size = usize::from_str_radix(field(124, 136).trim().trim_end_matches('\0'), 8).unwrap_or(0);
        let start = pos + 512;
        if start + size > tar.len() {
            return None;
        }
        if full.ends_with(suffix) {
            return Some(tar[start..start + size].to_vec());
        }
        pos = start + size.div_ceil(512) * 512;
    }
    None
}

/// Reads the client from Google's published `@google/gemini-cli-core` package.
pub async fn gemini_cli_npm() -> Result<(String, String), String> {
    let http = crate::exec::http_client(None);
    let meta: Value = http.get("https://registry.npmjs.org/@google%2fgemini-cli-core/latest").timeout(std::time::Duration::from_secs(20)).send().await.map_err(|e| e.to_string())?.json().await.map_err(|e| e.to_string())?;
    let url = meta["dist"]["tarball"].as_str().filter(|u| u.starts_with("https://registry.npmjs.org/")).ok_or("npm metadata has no tarball")?;
    let tgz = http.get(url).timeout(std::time::Duration::from_secs(60)).send().await.map_err(|e| e.to_string())?.bytes().await.map_err(|e| e.to_string())?;
    let mut tar = vec![];
    flate2::read::GzDecoder::new(&tgz[..]).read_to_end(&mut tar).map_err(|e| e.to_string())?;
    let js = tar_find(&tar, "dist/src/code_assist/oauth2.js").ok_or("oauth2.js not found in package")?;
    parse_gemini_cli_source(&String::from_utf8_lossy(&js)).ok_or_else(|| "OAuth client not found in oauth2.js".into())
}

fn antigravity_dirs() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = vec!["/Applications/Antigravity.app/Contents/Resources/app".into(), "/usr/share/antigravity/resources/app".into(), "/opt/Antigravity/resources/app".into(), "/opt/antigravity/resources/app".into()];
    if let Some(h) = dirs::home_dir() {
        v.push(h.join("Applications/Antigravity.app/Contents/Resources/app"));
        v.push(h.join(".local/share/antigravity/resources/app"));
    }
    if let Ok(l) = std::env::var("LOCALAPPDATA") {
        v.push(Path::new(&l).join("Programs").join("Antigravity").join("resources").join("app"));
    }
    v
}

/// Scans a local Antigravity install's bundled JavaScript for its client.
pub fn antigravity_local(dirs: &[PathBuf]) -> Option<(String, String)> {
    for d in dirs.iter().filter(|d| d.is_dir()) {
        let mut stack = vec![(d.clone(), 0)];
        let mut scanned = 0;
        while let Some((dir, depth)) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if depth < 6 && p.file_name().is_some_and(|n| n != "node_modules") {
                        stack.push((p, depth + 1));
                    }
                } else if p.extension().is_some_and(|x| x == "js") && scanned < 4000 {
                    scanned += 1;
                    if let Some(js) = read_small(&p) {
                        if js.contains("apps.googleusercontent.com") {
                            if let Some(c) = parse_bundled_client(&js) {
                                return Some(c);
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn remember(db: &crate::db::Db, id_key: &str, secret_key: &str, (id, secret): (String, String)) {
    let mut v = db.setting_json("oauthClients");
    if !v.is_object() {
        v = json!({});
    }
    v[id_key] = json!(id);
    v[secret_key] = json!(secret);
    let _ = db.set_setting_json("oauthClients", &v);
    let mut m = OVERRIDES.write().unwrap();
    m.insert(id_key.into(), id);
    m.insert(secret_key.into(), secret);
}

/// Fills in missing Google OAuth clients from the user's installed tools
/// (or Google's npm package for Gemini CLI). Must run before the registry is
/// first used; returns a human-readable note per client.
pub async fn discover(db: &crate::db::Db) -> Vec<String> {
    let mut notes = vec![];
    if !is_set("GOOGLE_OAUTH_CLIENT_ID") || !is_set("GOOGLE_OAUTH_CLIENT_SECRET") {
        let found = match gemini_cli_local(&node_roots()) {
            Some(c) => Ok((c, "local Gemini CLI install")),
            None => gemini_cli_npm().await.map(|c| (c, "@google/gemini-cli-core on npm")),
        };
        match found {
            Ok((c, from)) => {
                remember(db, "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", c);
                notes.push(format!("Gemini CLI OAuth client loaded from {from}"));
            }
            Err(e) => notes.push(format!("Gemini CLI OAuth client not found ({e}); set GOOGLE_OAUTH_CLIENT_ID/SECRET")),
        }
    }
    if !is_set("ANTIGRAVITY_OAUTH_CLIENT_ID") || !is_set("ANTIGRAVITY_OAUTH_CLIENT_SECRET") {
        match antigravity_local(&antigravity_dirs()) {
            Some(c) => {
                remember(db, "ANTIGRAVITY_OAUTH_CLIENT_ID", "ANTIGRAVITY_OAUTH_CLIENT_SECRET", c);
                notes.push("Antigravity OAuth client loaded from the local Antigravity app".into());
            }
            None => notes.push("Antigravity OAuth client not configured (install Antigravity or set ANTIGRAVITY_OAUTH_CLIENT_ID/SECRET)".into()),
        }
    }
    notes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sources_and_tar() {
        let js = "const OAUTH_CLIENT_ID = '123456-abc.apps.googleusercontent.com';\nconst OAUTH_CLIENT_SECRET = 'S3CR3T';";
        assert_eq!(parse_gemini_cli_source(js), Some(("123456-abc.apps.googleusercontent.com".into(), "S3CR3T".into())));
        let bundled = format!("x={{clientId:\"1234567-{}.apps.googleusercontent.com\",clientSecret:\"GOCSPX-{}\"}}", "a".repeat(32), "b".repeat(28));
        let (id, sec) = parse_bundled_client(&bundled).unwrap();
        assert!(id.starts_with("1234567-") && sec.starts_with("GOCSPX-"));
        // ustar archive with one file.
        let mut tar = vec![0u8; 512];
        let name = b"package/dist/src/code_assist/oauth2.js";
        tar[..name.len()].copy_from_slice(name);
        let size = format!("{:011o}\0", js.len());
        tar[124..136].copy_from_slice(size.as_bytes());
        tar[257..262].copy_from_slice(b"ustar");
        tar.extend_from_slice(js.as_bytes());
        tar.resize(tar.len().div_ceil(512) * 512 + 1024, 0);
        assert_eq!(tar_find(&tar, "code_assist/oauth2.js").unwrap(), js.as_bytes());
    }
}
