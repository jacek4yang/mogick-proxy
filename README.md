# mogick-provider

`mogick-provider` 是一个 Rust 多账户 OAuth 网关，对外同时提供 OpenAI 兼容的 `/v1/*` API 和 Anthropic Messages v1 API，对内转发到 Tongyuan Copilot `/api/v1/*`。模型 ID 不做白名单校验或改写，因此新增模型、`mm-*`、embedding 和 vision 模型会自动透传。

## 构建与首次登录

```bash
cargo build --release
./target/release/mogick-provider init
./target/release/mogick-provider login --account alice
./target/release/mogick-provider serve
```

登录采用 RFC 8628 设备码流程。终端会打印 authorization URL 和 user code；可在任意另一台机器完成授权，当前终端会自动轮询。无图形环境建议加 `--no-open`。access token、refresh token 和 JWT 不会输出。

增加第二个账户：

```bash
./target/release/mogick-provider login --account bob
./target/release/mogick-provider status
./target/release/mogick-provider logout --account alice
```

`login` 或 `logout` 未给 `--account` 时会交互读取账户名称。`status --account alice` 只显示一个账户；状态输出仅包含 enabled、有效期、scope 和脱敏错误摘要。

## 配置与凭据

默认使用当前目录的 `config.json`，凭据则保存在同目录的 `auth.json`：

```json
{
  "server": { "bind": "127.0.0.1:8787", "api_key": "" },
  "upstream": {
    "base_url": "https://copilot.tongyuan.cc",
    "api_prefix": "/api/v1",
    "timeout_secs": 120,
    "extra_headers": { "X-App-Id": "mogick" }
  },
  "runtime": {
    "refresh_skew_secs": 60,
    "balance_poll_secs": 180,
    "max_request_bytes": 33554432,
    "log_level": "info",
    "log_format": "pretty"
  }
}
```

OAuth client、scope 和端点是 provider 固有常量，不写入配置。`auth.json` 是版本化的多账户 map，在 Unix 上原子写入并设为 `0600`。两个本地文件都被 `.gitignore` 排除；仓库只提供无凭据的 `config.example.json`。

可用覆盖项：

- `--config PATH` / `MOGICK_PROVIDER_CONFIG`
- `--auth PATH` / `MOGICK_PROVIDER_AUTH`
- `RUST_LOG` 覆盖 `runtime.log_level`
- `runtime.log_format` 可设为 `pretty` 或 `json`

首次读取旧版 `config.json` 时，程序会先把 `oauth/tokens/accounts` 中的凭据原子写入并重新校验 `auth.json`，随后才重写无凭据的配置。任何步骤失败都会保留旧配置；已有同名但不同的账户凭据不会被覆盖。

## 入站鉴权

`server.api_key` 非空时，同时接受：

```text
Authorization: Bearer <server.api_key>
x-api-key: <server.api_key>
```

密钥为空时只接受 loopback 来源。上游认证头永远由网关重建，并强制使用 `X-App-Id: mogick`；调用者和 `extra_headers` 都不能覆盖 OAuth Authorization、cookie、Host、Content-Length 或 X-App-Id。

## OpenAI 兼容 API

所有 method、query、请求体和安全 header 按下列规则转发：

```text
/v1/chat/completions  -> /api/v1/chat/completions
/v1/models            -> /api/v1/models
/v1/embeddings        -> /api/v1/embeddings
/v1/files/...         -> /api/v1/files/...
/chat/completions     -> /api/v1/chat/completions（兼容旧客户端）
```

普通响应和 OpenAI SSE 均保持 OpenAI 格式。`GET /v1/models` 动态读取上游模型列表。

## Anthropic Messages v1

支持：

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- 带 `anthropic-version` 的 `GET /v1/models`

Messages 转换覆盖 system、text、image、document、tools、tool choice、并行工具控制、tool use/result、thinking、structured output、metadata、stop/采样参数，以及普通和流式 usage。无法无损处理的未知字段会返回 `invalid_request_error`，不会静默丢弃。

流式响应按 Anthropic 生命周期输出 `message_start`、content block start/delta/stop、`message_delta` 和 `message_stop`，支持 thinking、文本以及多个交错 tool call。解析器支持任意网络拆包、多行 `data:`、注释、CRLF 和 `[DONE]`，并保持背压，不缓存完整响应。

`/v1/messages/count_tokens` 会把同一请求转换成 `max_tokens=1` 的非流式最小上游调用，再返回实际模型 tokenizer 报告的 `usage.prompt_tokens`。因此该端点会产生一次最小上游调用和相应计费。

## 多账户和失败策略

网关从 enabled、无需重登且有凭据的账户中选择 `last_used` 最早者。每个账户有独立 refresh lock，并发请求只会触发一次 refresh。

- token 临近到期：只刷新选中的账户。
- 上游 401/403：强制刷新该账户并重试一次；仍失败或 refresh token 失效时标记需重登并切换账户。
- 上游 429：切换到下一账户。
- 网络错误和 5xx：不重放生成请求。
- 后台逐账户刷新和余额探活；`404 ACCOUNT_NOT_FOUND` 作为免费账户正常状态记录为 INFO。

## 日志与错误

请求日志包含 request ID、协议、模型、账户、路径、stream、状态、耗时、请求/响应字节、refresh 和 failover。不会记录请求正文。错误正文会截断，并递归清除 Authorization、cookie、API key、access/refresh token、Bearer 和 JWT。

Anthropic 错误使用：

```json
{"type":"error","error":{"type":"invalid_request_error","message":"..."},"request_id":"req_..."}
```

OpenAI 路由继续使用 OpenAI error envelope。健康检查只返回 `ok`。

## 测试

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo build --release
```

默认测试完全离线，使用本地 mock 覆盖 OAuth、迁移、多账户、failover、OpenAI/Anthropic 普通与 SSE 路径。

真实上游测试默认忽略且不会打印凭据。显式执行：

```bash
MOGICK_PROVIDER_REAL_TEST=1 \
MOGICK_REAL_CHAT_MODEL=deepseek-v4-flash \
MOGICK_REAL_EMBEDDING_MODEL=mm-embedding \
cargo test real_upstream_opt_in_smoke_suite -- --ignored
```

可选设置 `MOGICK_REAL_VISION_MODEL` 和 `MOGICK_REAL_VISION_DATA_URL` 覆盖 vision。真实测试始终检查 models、balance 和多账户轮询（若至少配置两个账户）；模型相关环境变量存在时再执行 chat、stream、embedding、vision，以避免默认产生调用费用。
