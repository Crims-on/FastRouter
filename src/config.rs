use std::path::PathBuf;

/// Runtime configuration, read from environment variables.
#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub initial_password: String,
    /// When set, overrides the "require API key" toggle from the dashboard.
    pub require_api_key: Option<bool>,
}

impl Config {
    pub fn from_env() -> Self {
        let data_dir = std::env::var("DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".fastrouter")
            });
        Self {
            host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(20128),
            data_dir,
            initial_password: std::env::var("INITIAL_PASSWORD").unwrap_or_else(|_| "123456".into()),
            require_api_key: std::env::var("REQUIRE_API_KEY")
                .ok()
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")),
        }
    }
}
