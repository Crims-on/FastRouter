//! FastRouter — a local AI gateway: one OpenAI/Anthropic-compatible endpoint
//! in front of many providers, with format translation, fallback combos,
//! usage tracking and a server-rendered dashboard.

mod auth;
mod catalog;
mod config;
mod db;
mod pricing;
mod proxy;
mod router;
mod translate;
mod ui;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tower_http::compression::CompressionLayer;
use tower_http::trace::TraceLayer;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<db::Db>,
    pub router: Arc<router::Router>,
    pub http: reqwest::Client,
    pub config: Arc<config::Config>,
}

pub fn app(state: AppState) -> Router {
    let api = Router::new()
        .route("/v1/chat/completions", post(proxy::chat_completions))
        .route("/chat/completions", post(proxy::chat_completions))
        .route("/v1/messages", post(proxy::messages))
        .route("/v1/messages/count_tokens", post(proxy::count_tokens))
        .route("/v1/models", get(proxy::models))
        .route("/models", get(proxy::models))
        .route("/health", get(|| async { "ok" }))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024));

    let dashboard = ui::routes()
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::require_session,
        ))
        .layer(CompressionLayer::new());

    Router::new()
        .merge(api)
        .merge(dashboard)
        .merge(ui::public_routes())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "fastrouter=info,tower_http=warn".into()),
        )
        .init();

    let config = config::Config::from_env();
    let db = db::Db::open(&config.data_dir.join("db").join("data.sqlite"))?;
    auth::bootstrap(&db, &config.initial_password)?;

    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .pool_idle_timeout(Duration::from_secs(90))
        .user_agent(concat!("FastRouter/", env!("CARGO_PKG_VERSION")))
        .build()?;

    let state = AppState {
        db: Arc::new(db),
        router: Arc::new(router::Router::default()),
        http,
        config: Arc::new(config.clone()),
    };

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("FastRouter listening on http://{addr}");
    tracing::info!("  dashboard: http://localhost:{}/dashboard", config.port);
    tracing::info!("  endpoint:  http://localhost:{}/v1", config.port);
    tracing::info!("  data:      {}", config.data_dir.display());

    axum::serve(listener, app(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
