//! Rough list prices (USD per 1M tokens) used to estimate what traffic would
//! have cost on pay-as-you-go APIs. Matched by substring, first match wins.

const PRICES: &[(&str, f64, f64)] = &[
    ("opus", 15.0, 75.0),
    ("sonnet", 3.0, 15.0),
    ("haiku", 1.0, 5.0),
    ("gpt-5-nano", 0.05, 0.4),
    ("gpt-5-mini", 0.25, 2.0),
    ("gpt-5", 1.25, 10.0),
    ("gpt-4.1-mini", 0.4, 1.6),
    ("gpt-4.1", 2.0, 8.0),
    ("gpt-4o-mini", 0.15, 0.6),
    ("gpt-4o", 2.5, 10.0),
    ("o4-mini", 1.1, 4.4),
    ("o3", 2.0, 8.0),
    ("gpt-oss", 0.1, 0.5),
    ("gemini-2.5-pro", 1.25, 10.0),
    ("gemini-2.5-flash-lite", 0.1, 0.4),
    ("gemini-2.5-flash", 0.3, 2.5),
    ("glm", 0.6, 2.2),
    ("minimax", 0.3, 1.2),
    ("kimi", 0.6, 2.5),
    ("deepseek", 0.28, 0.42),
    ("qwen3-coder", 1.0, 5.0),
    ("qwen", 0.4, 1.2),
    ("grok-code", 0.2, 1.5),
    ("grok", 3.0, 15.0),
    ("devstral", 0.4, 2.0),
    ("mistral", 2.0, 6.0),
    ("llama", 0.6, 0.8),
];

pub fn estimate(model: &str, prompt: i64, completion: i64) -> f64 {
    let m = model.to_ascii_lowercase();
    PRICES
        .iter()
        .find(|(k, _, _)| m.contains(k))
        .map(|(_, i, o)| (prompt as f64 * i + completion as f64 * o) / 1_000_000.0)
        .unwrap_or(0.0)
}
