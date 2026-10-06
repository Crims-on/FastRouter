//! FastRouter — a local AI gateway: one OpenAI/Anthropic/Gemini/Ollama-compatible
//! endpoint in front of 120+ providers, with format translation, account
//! rotation, fallback combos, usage tracking and a server-rendered dashboard.

#![allow(dead_code)]

mod api;
mod auth;
mod caps;
mod chat;
mod cloak;
mod config;
mod consts;
mod db;
mod exec;
mod jsv;
mod media;
mod mediaapi;
mod oauth;
mod providers;
mod registry;
mod secrets;
mod session;
mod sse;
mod translate;
mod ui;

#[cfg(test)]
mod e2e_tests;

use std::sync::Arc;

use axum::Router;
use tower_http::trace::TraceLayer;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<db::Db>,
    pub config: Arc<config::Config>,
}

pub fn app(state: AppState) -> Router {
    let dashboard = ui::routes().route_layer(axum::middleware::from_fn_with_state(state.clone(), auth::require_session));
    Router::new().merge(api::routes()).merge(dashboard).merge(ui::public_routes()).layer(TraceLayer::new_for_http()).with_state(state)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "fastrouter=info,tower_http=warn".into()))
        .init();
    let config = config::Config::from_env();
    let db = db::Db::open(&config.data_dir.join("db").join("data.sqlite"))?;
    auth::bootstrap(&db, &config.initial_password)?;
    secrets::init(&db);
    let state = AppState { db: Arc::new(db), config: Arc::new(config.clone()) };
    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("FastRouter listening on http://{addr}");
    axum::serve(listener, app(state)).with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await?;
    Ok(())
}
