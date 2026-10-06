//! Dashboard password + signed session cookies, and `/v1` API-key checks.

use argon2::Argon2;
use axum::extract::{Request, State};
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};
use axum_extra::extract::CookieJar;
use axum_extra::extract::cookie::Cookie;
use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

use crate::AppState;
use crate::db::{Db, now};

pub const SESSION_COOKIE: &str = "fr_session";
const SESSION_TTL: i64 = 7 * 24 * 3600;

pub fn random_token() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

fn hash_with_salt(password: &str, salt: &str) -> String {
    let mut out = [0u8; 32];
    Argon2::default()
        .hash_password_into(password.as_bytes(), salt.as_bytes(), &mut out)
        .expect("argon2 parameters are valid");
    hex::encode(out)
}

pub fn hash_password(password: &str) -> String {
    let salt = uuid::Uuid::new_v4().simple().to_string();
    format!("{salt}${}", hash_with_salt(password, &salt))
}

pub fn verify_password(db: &Db, password: &str) -> bool {
    let Some(stored) = db.get_setting("password_hash") else {
        return false;
    };
    let Some((salt, hash)) = stored.split_once('$') else {
        return false;
    };
    constant_eq(hash_with_salt(password, salt).as_bytes(), hash.as_bytes())
}

fn constant_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Ensures a password and session secret exist.
pub fn bootstrap(db: &Db, initial_password: &str) -> anyhow::Result<()> {
    if db.get_setting("password_hash").is_none() {
        db.set_setting("password_hash", &hash_password(initial_password))?;
    }
    if db.get_setting("session_secret").is_none() {
        db.set_setting("session_secret", &random_token())?;
    }
    Ok(())
}

fn sign(secret: &str, payload: &str) -> String {
    let mut mac =
        Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("hmac accepts any key length");
    mac.update(payload.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

pub fn session_cookie(db: &Db) -> Cookie<'static> {
    let secret = db.get_setting("session_secret").unwrap_or_default();
    let payload = format!("{}", now() + SESSION_TTL);
    let value = format!("{payload}.{}", sign(&secret, &payload));
    Cookie::parse(format!(
        "{SESSION_COOKIE}={value}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_TTL}"
    ))
    .expect("session cookie is well-formed")
}

pub fn valid_session(db: &Db, jar: &CookieJar) -> bool {
    let Some(c) = jar.get(SESSION_COOKIE) else {
        return false;
    };
    let Some((payload, sig)) = c.value().split_once('.') else {
        return false;
    };
    let Ok(exp) = payload.parse::<i64>() else {
        return false;
    };
    let secret = db.get_setting("session_secret").unwrap_or_default();
    exp > now() && constant_eq(sign(&secret, payload).as_bytes(), sig.as_bytes())
}

/// Middleware guarding the dashboard.
pub async fn require_session(
    State(state): State<AppState>,
    jar: CookieJar,
    req: Request,
    next: Next,
) -> Response {
    if valid_session(&state.db, &jar) {
        next.run(req).await
    } else {
        Redirect::to("/login").into_response()
    }
}

/// Extracts a client API key from `Authorization: Bearer` or `x-api-key`.
pub fn client_key(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-api-key")
        .or_else(|| headers.get("x-goog-api-key"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| {
                    v.strip_prefix("Bearer ")
                        .or_else(|| v.strip_prefix("bearer "))
                })
                .map(|s| s.trim().to_owned())
        })
        .filter(|k| !k.is_empty())
}

pub fn api_key_required(state: &AppState) -> bool {
    state
        .config
        .require_api_key
        .unwrap_or_else(|| state.db.get_setting("require_api_key").as_deref() == Some("1"))
}
