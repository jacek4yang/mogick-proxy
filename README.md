# mogick-proxy

OAuth-aware reverse proxy that exposes a single OpenAI-compatible
`POST /chat/completions` endpoint and forwards to an upstream LLM using a
fresh access token obtained via OAuth 2.0 Device Authorization Grant (RFC 8628).

This mirrors the model-interaction layer Mogick uses internally
(`tongyuan.cc/ai/mogick/llmclient` + `modelprovider.OpenAICompatProvider`)
but is fully self-contained: it has no Anthropic dependency, no upstream SDK,
and no hidden state.

## Why

Mogick's `OpenAICompatProvider` does this:

```
POST {base_url}/chat/completions
Authorization: Bearer <api_key>
Content-Type: application/json
```

If `base_url` points at this proxy, and the proxy mints valid bearer tokens
via an interactive OAuth login + automatic refresh, then Mogick / Claude Code /
any other OpenAI-compatible client can keep using the same API surface while
the actual upstream keeps seeing fresh OAuth tokens.

## Layout

```
mogick-proxy/
├── Cargo.toml
├── config.example.json
└── src/
    ├── main.rs   — CLI (init / login / status / logout / serve)
    ├── config.rs — Config + TokenState (JSON on disk)
    ├── oauth.rs  — Device Authorization Grant client
    ├── token.rs  — TokenManager: cached + auto-refreshing
    └── server.rs — Axum HTTP server, /chat/completions passthrough
```

## Build

```bash
cargo build --release
```

## Configure

```bash
./target/release/mogick-proxy init
$EDITOR ~/.config/mogick-proxy/config.json   # or %APPDATA%\mogick-proxy\config.json on Windows
```

`config.json` example:

```json
{
  "server": { "bind": "127.0.0.1:8787", "api_key": "" },
  "oauth": {
    "client_id": "mogick-proxy",
    "device_authorization_endpoint": "https://login.tongyuan.cc/oauth2/device",
    "token_endpoint": "https://login.tongyuan.cc/oauth2/token",
    "scope": "openid profile email offline_access",
    "audience": null
  },
  "upstream": {
    "base_url": "https://api.tongyuan.cc/v1",
    "chat_path": "/chat/completions",
    "static_api_key": null,
    "extra_headers": {},
    "timeout_secs": 120
  },
  "tokens": {
    "access_token": "",
    "refresh_token": "",
    "expires_at": 0,
    "token_type": "",
    "scope": ""
  }
}
```

Notes:
- `server.api_key`: optional shared secret. If set, callers must present
  `Authorization: Bearer <server.api_key>` to use the proxy. If empty, only
  loopback callers are accepted.
- `upstream.static_api_key`: optional. When set, OAuth is bypassed entirely
  and this string is sent as the upstream bearer token.
- `upstream.extra_headers`: forwarded verbatim to the upstream.

## Login

```bash
./target/release/mogick-proxy login
```

The CLI:
1. POSTs to `oauth.device_authorization_endpoint` and receives a `device_code`.
2. Prints the `user_code` and `verification_uri`, and tries to open a browser.
3. Long-polls `oauth.token_endpoint` until the user authorises (or the
   device code expires).
4. Persists `access_token` + `refresh_token` to `config.json` atomically.

`mogick-proxy login --force` discards existing tokens and re-authenticates.

## Status

```bash
./target/release/mogick-proxy status
```

Prints token expiry, scopes, and remaining lifetime.

## Serve

```bash
./target/release/mogick-proxy serve
```

Starts the HTTP server. Calls:

```bash
curl -X POST http://127.0.0.1:8787/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [{"role":"user","content":"hello"}]
  }'
```

…are forwarded to `${upstream.base_url}${upstream.chat_path}` with
`Authorization: Bearer <fresh-access-token>`. Streaming (`"stream": true`) is
piped through verbatim — no buffering.

## Use with Mogick

Set Mogick's provider config to point at this proxy:

```json
{
  "llm": {
    "provider": "openai",
    "model": "claude-sonnet-4-5",
    "providers": [
      {
        "name": "local-proxy",
        "type": "openai",
        "base_url": "http://127.0.0.1:8787/v1",
        "models": [
          { "id": "claude-sonnet-4-5", "context_window": 200000 }
        ]
      }
    ]
  }
}
```

And either:
- put any non-empty string in `auth.json` (e.g. `{"providers":{"local-proxy":{"api_key":"anything"}}}`),
  Mogick will pass it as `Authorization: Bearer anything`; or
- start the proxy without `server.api_key` set so the loopback exemption applies.

The proxy substitutes the real OAuth access token on the way to the upstream.
