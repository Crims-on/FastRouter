//! OAuth: token refresh for every OAuth provider and the interactive login
//! flows (authorization-code + PKCE, device code, import) used by the dashboard.

pub mod flows;
pub mod refresh;

use base64::Engine;
use serde_json::Value;

/// Decodes a JWT payload without verifying it (like 9router's decodeJwtPayload).
pub fn decode_jwt_payload(jwt: &str) -> Option<Value> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let p = parts[1].replace('-', "+").replace('_', "/");
    let pad = (4 - p.len() % 4) % 4;
    let b = base64::engine::general_purpose::STANDARD.decode(format!("{p}{}", "=".repeat(pad))).ok()?;
    serde_json::from_slice(&b).ok()
}

pub fn email_from_jwt(jwt: &str) -> Option<String> {
    let p = decode_jwt_payload(jwt)?;
    ["email", "preferred_username", "sub"].iter().find_map(|k| p[*k].as_str().filter(|s| !s.is_empty()).map(str::to_owned))
}

/// application/x-www-form-urlencoded body.
pub fn form(pairs: &[(&str, &str)]) -> String {
    pairs.iter().map(|(k, v)| format!("{}={}", enc(k), enc(v))).collect::<Vec<_>>().join("&")
}

/// URLSearchParams-style encoding (space → '+').
pub fn enc(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'*' | b'-' | b'.' | b'_' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
