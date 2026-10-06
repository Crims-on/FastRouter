//! Static catalog of upstream providers FastRouter knows how to talk to.

use serde::{Deserialize, Serialize};

/// Wire format spoken by an upstream (or by an inbound client).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    OpenAI,
    Claude,
    Gemini,
}

impl Format {
    pub fn label(self) -> &'static str {
        match self {
            Format::OpenAI => "OpenAI",
            Format::Claude => "Anthropic",
            Format::Gemini => "Gemini",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Subscription,
    Cheap,
    Free,
    Custom,
}

impl Tier {
    pub fn label(self) -> &'static str {
        match self {
            Tier::Subscription => "Premium API",
            Tier::Cheap => "Low-cost",
            Tier::Free => "Free / Local",
            Tier::Custom => "Custom",
        }
    }
}

pub struct Provider {
    pub id: &'static str,
    pub name: &'static str,
    pub format: Format,
    pub base_url: &'static str,
    pub tier: Tier,
    /// Whether an API key is required to connect.
    pub needs_key: bool,
    /// Custom providers require a base URL and a user-chosen prefix.
    pub custom: bool,
    pub color: &'static str,
    pub key_url: &'static str,
    pub models: &'static [&'static str],
}

pub static PROVIDERS: &[Provider] = &[
    Provider {
        id: "anthropic",
        name: "Anthropic",
        format: Format::Claude,
        base_url: "https://api.anthropic.com/v1",
        tier: Tier::Subscription,
        needs_key: true,
        custom: false,
        color: "#d97757",
        key_url: "https://console.anthropic.com/settings/keys",
        models: &["claude-opus-4-1", "claude-sonnet-4-5", "claude-haiku-4-5"],
    },
    Provider {
        id: "openai",
        name: "OpenAI",
        format: Format::OpenAI,
        base_url: "https://api.openai.com/v1",
        tier: Tier::Subscription,
        needs_key: true,
        custom: false,
        color: "#10a37f",
        key_url: "https://platform.openai.com/api-keys",
        models: &["gpt-5", "gpt-5-mini", "gpt-4.1", "o4-mini"],
    },
    Provider {
        id: "gemini",
        name: "Google Gemini",
        format: Format::Gemini,
        base_url: "https://generativelanguage.googleapis.com/v1beta",
        tier: Tier::Subscription,
        needs_key: true,
        custom: false,
        color: "#4285f4",
        key_url: "https://aistudio.google.com/apikey",
        models: &[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ],
    },
    Provider {
        id: "openrouter",
        name: "OpenRouter",
        format: Format::OpenAI,
        base_url: "https://openrouter.ai/api/v1",
        tier: Tier::Cheap,
        needs_key: true,
        custom: false,
        color: "#6467f2",
        key_url: "https://openrouter.ai/keys",
        models: &[
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-5",
            "google/gemini-2.5-pro",
            "z-ai/glm-4.6",
        ],
    },
    Provider {
        id: "glm",
        name: "GLM (Z.ai Coding)",
        format: Format::OpenAI,
        base_url: "https://api.z.ai/api/coding/paas/v4",
        tier: Tier::Cheap,
        needs_key: true,
        custom: false,
        color: "#3859ff",
        key_url: "https://z.ai/manage-apikey/apikey-list",
        models: &["glm-4.6", "glm-4.5", "glm-4.5-air"],
    },
    Provider {
        id: "minimax",
        name: "MiniMax",
        format: Format::Claude,
        base_url: "https://api.minimax.io/anthropic/v1",
        tier: Tier::Cheap,
        needs_key: true,
        custom: false,
        color: "#e2167e",
        key_url: "https://platform.minimax.io/user-center/basic-information/interface-key",
        models: &["MiniMax-M2"],
    },
    Provider {
        id: "kimi",
        name: "Kimi (Moonshot)",
        format: Format::OpenAI,
        base_url: "https://api.moonshot.ai/v1",
        tier: Tier::Cheap,
        needs_key: true,
        custom: false,
        color: "#16191e",
        key_url: "https://platform.moonshot.ai/console/api-keys",
        models: &["kimi-k2-0905-preview", "kimi-k2-turbo-preview"],
    },
    Provider {
        id: "deepseek",
        name: "DeepSeek",
        format: Format::OpenAI,
        base_url: "https://api.deepseek.com/v1",
        tier: Tier::Cheap,
        needs_key: true,
        custom: false,
        color: "#4d6bfe",
        key_url: "https://platform.deepseek.com/api_keys",
        models: &["deepseek-chat", "deepseek-reasoner"],
    },
    Provider {
        id: "qwen",
        name: "Qwen (DashScope)",
        format: Format::OpenAI,
        base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        tier: Tier::Cheap,
        needs_key: true,
        custom: false,
        color: "#615ced",
        key_url: "https://modelstudio.console.alibabacloud.com/",
        models: &["qwen3-coder-plus", "qwen3-max", "qwen-plus"],
    },
    Provider {
        id: "xai",
        name: "xAI",
        format: Format::OpenAI,
        base_url: "https://api.x.ai/v1",
        tier: Tier::Cheap,
        needs_key: true,
        custom: false,
        color: "#111111",
        key_url: "https://console.x.ai/",
        models: &["grok-4", "grok-code-fast-1"],
    },
    Provider {
        id: "mistral",
        name: "Mistral",
        format: Format::OpenAI,
        base_url: "https://api.mistral.ai/v1",
        tier: Tier::Cheap,
        needs_key: true,
        custom: false,
        color: "#fa520f",
        key_url: "https://console.mistral.ai/api-keys",
        models: &[
            "devstral-medium-latest",
            "mistral-large-latest",
            "codestral-latest",
        ],
    },
    Provider {
        id: "groq",
        name: "Groq",
        format: Format::OpenAI,
        base_url: "https://api.groq.com/openai/v1",
        tier: Tier::Free,
        needs_key: true,
        custom: false,
        color: "#f55036",
        key_url: "https://console.groq.com/keys",
        models: &[
            "moonshotai/kimi-k2-instruct-0905",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
        ],
    },
    Provider {
        id: "cerebras",
        name: "Cerebras",
        format: Format::OpenAI,
        base_url: "https://api.cerebras.ai/v1",
        tier: Tier::Free,
        needs_key: true,
        custom: false,
        color: "#f15a29",
        key_url: "https://cloud.cerebras.ai/",
        models: &["qwen-3-coder-480b", "gpt-oss-120b"],
    },
    Provider {
        id: "nvidia",
        name: "NVIDIA NIM",
        format: Format::OpenAI,
        base_url: "https://integrate.api.nvidia.com/v1",
        tier: Tier::Free,
        needs_key: true,
        custom: false,
        color: "#76b900",
        key_url: "https://build.nvidia.com/",
        models: &[
            "moonshotai/kimi-k2-instruct",
            "qwen/qwen3-coder-480b-a35b-instruct",
        ],
    },
    Provider {
        id: "together",
        name: "Together AI",
        format: Format::OpenAI,
        base_url: "https://api.together.xyz/v1",
        tier: Tier::Cheap,
        needs_key: true,
        custom: false,
        color: "#0f6fff",
        key_url: "https://api.together.ai/settings/api-keys",
        models: &[
            "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            "deepseek-ai/DeepSeek-V3.1",
        ],
    },
    Provider {
        id: "fireworks",
        name: "Fireworks",
        format: Format::OpenAI,
        base_url: "https://api.fireworks.ai/inference/v1",
        tier: Tier::Cheap,
        needs_key: true,
        custom: false,
        color: "#6720ff",
        key_url: "https://fireworks.ai/account/api-keys",
        models: &[
            "accounts/fireworks/models/kimi-k2-instruct-0905",
            "accounts/fireworks/models/glm-4p6",
        ],
    },
    Provider {
        id: "ollama",
        name: "Ollama (local)",
        format: Format::OpenAI,
        base_url: "http://localhost:11434/v1",
        tier: Tier::Free,
        needs_key: false,
        custom: false,
        color: "#555555",
        key_url: "https://ollama.com/",
        models: &["qwen3-coder:30b", "gpt-oss:20b"],
    },
    Provider {
        id: "openai-compatible",
        name: "OpenAI-compatible",
        format: Format::OpenAI,
        base_url: "",
        tier: Tier::Custom,
        needs_key: false,
        custom: true,
        color: "#7c8798",
        key_url: "",
        models: &[],
    },
    Provider {
        id: "anthropic-compatible",
        name: "Anthropic-compatible",
        format: Format::Claude,
        base_url: "",
        tier: Tier::Custom,
        needs_key: false,
        custom: true,
        color: "#a08872",
        key_url: "",
        models: &[],
    },
];

pub fn get(id: &str) -> Option<&'static Provider> {
    PROVIDERS.iter().find(|p| p.id == id)
}
