# FastRouter

![FastRouter dashboard](docs/dashboard.png)

A local AI gateway written in Rust, modelled on [9router](https://github.com/decolua/9router).
FastRouter gives Claude Code, Codex, Cursor, Cline and any OpenAI or Anthropic SDK client a single endpoint. It routes each request to one of many upstream providers, translates between API formats, falls back automatically when a provider fails, and tracks token usage and estimated cost.

- **Backend:** [axum](https://github.com/tokio-rs/axum) + tokio + reqwest. SQLite (bundled) is used for storage.
- **Dashboard:** rendered on the server by compile-time [maud](https://maud.lambda.xyz) templates. There's no SPA, no build step and no JS framework. Every page is complete HTML before it reaches the browser. A few lines of inline JS power the "Copy" buttons, and everything else works with JS disabled.
- **Single binary:** the CSS is embedded, so you don't need to deploy any assets.

## Features

| | |
|---|---|
| **Endpoints** | `POST /v1/chat/completions` (OpenAI), `POST /v1/messages` (Anthropic), `POST /v1/messages/count_tokens`, `GET /v1/models` |
| **Format translation** | OpenAI ⇄ Anthropic ⇄ Gemini for requests, responses and SSE streams. Covers system prompts, images, tool calls and tool results, `tool_choice`, stop sequences, reasoning/thinking and usage |
| **Providers** | Anthropic, OpenAI, Gemini, OpenRouter, GLM (Z.ai), MiniMax, Kimi, DeepSeek, Qwen, xAI, Mistral, Groq, Cerebras, NVIDIA NIM, Together, Fireworks, Ollama, plus any custom OpenAI- or Anthropic-compatible endpoint |
| **Combos** | Named fallback chains, e.g. `premium-coding = anthropic/claude-sonnet-4-5 → glm/glm-4.6 → groq/openai/gpt-oss-120b` |
| **Multi-account** | Multiple connections can share a prefix. They're tried in priority order, or round-robin |
| **Cooldowns** | A failing connection is skipped temporarily: 60s per model after a 429, 15s after a 5xx, 5 minutes for the whole connection after a 401/402/403 |
| **Usage** | Every attempt is logged, including fallbacks and aborted streams. You get token counts, latency, status and an estimated list-price cost, broken down by model, provider and API key |
| **Auth** | Password-protected dashboard with an HMAC-signed session cookie (argon2 password hash). `/v1` API keys are optional or required |
| **Claude Code OAuth** | Paste a `sk-ant-oat…` token as an Anthropic key and FastRouter sends the OAuth headers Claude subscriptions need |

## Run

```bash
cargo run --release
# dashboard: http://localhost:20128/dashboard   (default password: 123456)
# endpoint:  http://localhost:20128/v1
```

Or use Docker:

```bash
docker build -t fastrouter .
docker run -p 20128:20128 -v fastrouter-data:/data -e INITIAL_PASSWORD=change-me fastrouter
```

### Environment

| Variable | Default | |
|---|---|---|
| `PORT` | `20128` | |
| `HOST` | `0.0.0.0` | |
| `DATA_DIR` | `~/.fastrouter` | SQLite lives at `$DATA_DIR/db/data.sqlite` |
| `INITIAL_PASSWORD` | `123456` | Only applies on first start. Change it later in Settings |
| `REQUIRE_API_KEY` | (unset) | `true`/`false` overrides the dashboard toggle |
| `RUST_LOG` | `fastrouter=info` | |
| `HTTPS_PROXY` / `HTTP_PROXY` | | Honoured for upstream requests |

## Using it

1. Open **Providers**, pick one and add a connection: API key, an optional base-URL override, and a priority.
   - Each connection has a **prefix**, which is the provider id by default. Clients address models as `prefix/model`, for example `openrouter/z-ai/glm-4.6`.
   - **Test** sends a tiny request. **Fetch models** pulls the upstream model list.
2. Under **Combos**, create one with a model per line, in fallback order.
3. Point your tool at FastRouter:

```bash
# Claude Code
export ANTHROPIC_BASE_URL=http://localhost:20128
export ANTHROPIC_AUTH_TOKEN=<fastrouter key, or anything if keys are optional>
export ANTHROPIC_MODEL=premium-coding

# OpenAI-compatible tools (Codex, Cline, Cursor, SDKs…)
export OPENAI_BASE_URL=http://localhost:20128/v1
```

### Model resolution

1. If the name matches a **combo**, each entry is expanded in order.
2. `prefix/model` goes to the enabled connections with that prefix.
3. A bare model id goes to the first prefix whose known or fetched models include it.

Healthy targets are tried first. Targets in cooldown are kept as a last resort. When every target fails, the client gets the last upstream error in its own API's error format.

## Layout

```
src/
  main.rs            server setup, routes
  config.rs          env configuration
  db.rs              SQLite: connections, combos, keys, settings, usage
  catalog.rs         provider catalog (format, base URL, default models)
  router.rs          combo expansion, prefix routing, round-robin, cooldowns
  proxy.rs           /v1 handlers, fallback loop, upstream requests, streaming
  pricing.rs         list-price table for cost estimates
  auth.rs            password, sessions, API-key checks
  translate/
    request.rs       OpenAI ⇄ Claude ⇄ Gemini request conversion
    response.rs      non-streaming response conversion, error bodies
    stream.rs        SSE parser + streaming state machines
  ui/                server-rendered dashboard (maud) + embedded CSS
```

## Development

```bash
cargo test     # translation, streaming and formatting tests
cargo clippy
```

## Differences from 9router

FastRouter re-implements 9router's core: the OpenAI and Anthropic endpoints, format translation, combos and fallback, multiple accounts, usage tracking and the dashboard. It doesn't yet include:

- OAuth login flows for subscription providers (Copilot, Codex, Cursor, Kiro, Gemini CLI). An existing Claude Code OAuth token does work as an Anthropic key.
- Token-saving prompt rewrites ("caveman" mode, RTK tool-output compression).
- Cloud sync and Cloudflare Workers deployment.
- Gemini-format inbound requests (`/v1beta/models/...`). Gemini works as an upstream.
