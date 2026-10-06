//! Shared constants exported from 9router (`data/constants.json`).

use std::sync::LazyLock;

use serde_json::Value;

const CONSTANTS_JSON: &str = include_str!("../data/constants.json");

pub static C: LazyLock<Value> = LazyLock::new(|| serde_json::from_str(&crate::secrets::substitute(CONSTANTS_JSON)).expect("constants.json"));

pub fn s(key: &str) -> &'static str {
    C[key].as_str().unwrap_or("")
}

pub fn v(key: &str) -> &'static Value {
    &C[key]
}

pub const CLAUDE_SYSTEM_PROMPT: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";
pub const DEFAULT_MAX_TOKENS: i64 = 64000;
pub const DEFAULT_MIN_TOKENS: i64 = 32000;
pub const CLAUDE_TOOL_SUFFIX: &str = "_ide";

pub fn claude_cli_version() -> &'static str {
    s("CLAUDE_CLI_VERSION")
}

pub fn sig_claude() -> &'static str {
    s("DEFAULT_THINKING_CLAUDE_SIGNATURE")
}
pub fn sig_ag() -> &'static str {
    s("DEFAULT_THINKING_AG_SIGNATURE")
}
pub fn sig_gemini_cli() -> &'static str {
    s("DEFAULT_THINKING_GEMINI_CLI_SIGNATURE")
}
pub fn sig_vertex() -> &'static str {
    s("DEFAULT_THINKING_VERTEX_SIGNATURE")
}

pub fn cc_default_tools() -> Vec<&'static str> {
    C["CC_DEFAULT_TOOLS"].as_array().into_iter().flatten().filter_map(|x| x.as_str()).collect()
}

/// Map a Node-style platform to Stainless OS string.
pub fn stainless_os() -> &'static str {
    match std::env::consts::OS {
        "macos" => "MacOS",
        "windows" => "Windows",
        "linux" => "Linux",
        "freebsd" => "FreeBSD",
        _ => "Other",
    }
}

pub fn stainless_arch() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        "x86" => "x86",
        _ => "other",
    }
}

pub fn node_platform() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    }
}

pub fn node_arch() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        "x86" => "ia32",
        other => other,
    }
}
