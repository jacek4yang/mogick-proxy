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
    "timeout_secs": 600,
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
- `runtime.log_format` 可设为 `pretty`（compact 单行文本）或 `json`（单行 JSON）

首次读取旧版 `config.json` 时，程序会先把 `oauth/tokens/accounts` 中的凭据原子写入并重新校验 `auth.json`，随后才重写无凭据的配置。任何步骤失败都会保留旧配置；已有同名但不同的账户凭据不会被覆盖。

## 入站鉴权

`server.api_key` 非空时，同时接受：

```text
Authorization: Bearer <server.api_key>
x-api-key: <server.api_key>
```

密钥为空时只接受 loopback 来源。上游认证头永远由网关重建，并强制使用 `X-App-Id: mogick`；调用者和 `extra_headers` 都不能覆盖 OAuth Authorization、cookie、Host、Content-Length 或 X-App-Id。

OAuth、余额和 Copilot 请求默认直连，不继承进程的 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`。这可避免本机 SOCKS 代理不支持或对 Tongyuan 域名 TLS 中断时导致刷新在收到响应前失败。

Tongyuan 的 refresh 请求使用 RFC 6749 表单，并明确发送 `grant_type=refresh_token`、`refresh_token`、`client_id=mogick` 和 `scope=openid profile email`，同时携带 `Accept: application/json` 与 `X-App-Id: mogick`。响应兼容标准 RFC 6749 JSON 和 `{code,data}` 信封（实际 IdP 成功码可能为 `0` 或 `200`）；服务端轮换 refresh token 时会与新 access token 一起原子保存。刷新被明确拒绝后账户会标记为需要重新登录，但尚未过期的 access token 不会因瞬态网络故障被提前清除。

## Claude Code 接入

启动 provider 后，可直接使用环境变量接入 Claude Code，不需要修改用户或项目的 `.claude` 目录：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
# 应与 server.api_key 相同；仅限 loopback 且 api_key 为空时可使用任意非空占位值。
export ANTHROPIC_AUTH_TOKEN=local-only

export ANTHROPIC_MODEL=sonnet
export ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4-flash
export ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4-flash
export ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-v4-flash
export ANTHROPIC_DEFAULT_OPUS_MODEL_SUPPORTED_CAPABILITIES='effort,thinking,adaptive_thinking,interleaved_thinking'
export ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES='effort,thinking,adaptive_thinking,interleaved_thinking'
export ANTHROPIC_DEFAULT_HAIKU_MODEL_SUPPORTED_CAPABILITIES='effort,thinking,adaptive_thinking,interleaved_thinking'

# 告知 Claude Code 自定义模型支持的请求能力，并从网关发现可选模型。
export ANTHROPIC_CUSTOM_MODEL_OPTION=deepseek-v4-flash
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME='DeepSeek V4 Flash via mogick'
export ANTHROPIC_CUSTOM_MODEL_OPTION_SUPPORTED_CAPABILITIES='effort,thinking,adaptive_thinking,interleaved_thinking'
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000
export API_TIMEOUT_MS=600000

claude
```

仓库中的 `claude-code.env.example` 提供同一套无凭据模板。若想让 `opus`、`sonnet`、`haiku` 使用不同上游模型，可分别修改三个 `ANTHROPIC_DEFAULT_*_MODEL`。网关不改写模型 ID，模型选择仍完全由 Claude Code 环境变量或 `--model` 控制。

Claude Code 不认识 `deepseek-v4-flash` 这类网关模型 ID 时会采用保守的 context window。DeepSeek 发布的 V4 Flash 上下文是 1,000,000 tokens，因此示例为该模型设置 `CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000`；切换模型或上游部署存在更小限制时必须同步改成真实值。对普通未知 ID，这个变量会保留 Claude Code 的主动 compaction 并修正窗口。不要仅为消除警告给模型名追加 `[1m]`：网关不会剥离后缀，上游必须真的接受该 ID 才能使用。`CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1` 仅适合无法取得真实窗口时作为后备，它会推迟到 API 报 context-limit 错误才触发恢复，不是推荐默认值。

长时间编程 turn 建议同时保持 `upstream.timeout_secs` 与 Claude Code 的 `API_TIMEOUT_MS` 一致；新配置默认分别为 600 秒和 600000 毫秒。已有 `config.json` 不会被静默改写，需要手动把旧的 120 秒值调大。

不需要设置 `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS`、`CLAUDE_CODE_DISABLE_THINKING` 或 `MAX_THINKING_TOKENS=0`。当前 Claude Code 的 beta header、adaptive thinking、effort、context management、prompt caching、mid-conversation system、严格/延迟工具 schema、普通与流式工具循环都由网关兼容处理。Anthropic 专属 header 不会继续泄漏给 OpenAI 上游。

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
- 带 `anthropic-version` 的 `GET /v1/models/{model_id}`

Messages 转换覆盖 system 与 mid-conversation system、text、image、document、tools、严格/延迟工具 schema、tool choice、并行工具控制、tool use/result、thinking、structured output、metadata、stop/采样参数，以及普通和流式 usage。adaptive thinking 不会被错误地原样发送给 OpenAI 上游；`output_config.effort` 会转换为上游 `reasoning_effort`，`xhigh/max` 在上游只支持三级 effort 时安全降为 `high`。

结构化输出和 `strict: true` 工具不只依赖提示词：网关先使用上游支持的 `json_object`，缓存完整响应并用原始 JSON Schema 本地验证；失败时携带错误输出进行最多两次修复采样，验证成功后再返回普通响应或合成完整 Anthropic SSE。因此会话标题、结构化最终输出和严格工具参数不会把已知不合规数据直接交给 Claude Code。代价是这类请求的首字节延迟高于普通流式请求，并且失败重试会产生额外上游用量。

Claude Code 当前使用的 `context_management` 会在网关本地执行：支持按最近 thinking turn 保留 reasoning，支持按网关估算的 input token 或精确 tool use 阈值清理旧 tool result，可选清理旧 tool input、排除指定工具并应用 `clear_at_least`。响应通过 `context_management.applied_edits` 报告实际执行的 thinking/tool 清理及估算的节省；token count 的 `input_tokens` 是编辑后上游 tokenizer 实测值，`original_input_tokens` 则以该实测值为基线加上网关估算的移除量。

`compact_20260112` 支持默认/自定义 trigger、instructions 和 `pause_after_compaction`；trigger 针对“应用历史 compaction block 且执行本地 thinking/tool edits 后”的有效上下文估算，而不是包含已作废历史或 context-management 配置本身的整个 HTTP body。触发后使用相同模型执行独立摘要采样，返回首个 `compaction` block，非暂停模式再用压缩后的有效上下文继续主采样。后续传回 compaction block 时，网关会丢弃其前方历史；同一长会话可再次 compaction。流式请求产生单个完整的 `compaction_delta`，usage 用 `compaction`/`message` 两类 iteration 分列。摘要和主响应的 usage 都取自各自真实上游调用。

thinking 响应包含非空的网关 opaque signature，流式响应在 block 结束前发送 `signature_delta`；`display: omitted` 不发送 thinking delta，`display: summarized` 只发送安全的兼容说明，不把上游原始 private reasoning 暴露给客户端。同一 provider 进程中，网关生成的 signature 可以从有界内存恢复上游 reasoning 以延续工具回合。它不是 Anthropic 的跨平台加密签名，provider 重启或旧签名被淘汰后无法恢复原 reasoning。

usage 会输出 `iterations`、缓存读写细分和 cache creation 明细；只报告上游真实提供的 cached token，未提供时保持为零，不伪造节省。OpenAI `x-ratelimit-*` 会映射成对应的 `anthropic-ratelimit-*`，`retry-after` 原样保留。模型列表和单模型详情输出 `max_input_tokens`、`max_tokens` 及 context management、effort、thinking、structured output、image/PDF 等 capabilities；embedding 模型不会进入 Claude Code 模型发现列表。

fast mode、diagnostics、task budget、fallback 和顶层缓存提示会被校验并作为 Anthropic advisory hint 消费，不会作为未知 OpenAI 字段导致上游 400。无法安全解释的未来字段仍返回明确的 `invalid_request_error`，不会无提示改变请求。

这是“在现有 OpenAI-compatible upstream 上实现的最大实用 Claude Code 兼容层”，不是完整 Anthropic 等价实现。Anthropic 原生 prompt-cache 计费/生命周期、托管 Web Search/Web Fetch、托管 Code Execution/Computer Use、Anthropic 内部 thinking 签名和限流基础设施无法由网关忠实重建；图像、PDF、reasoning 和工具质量也始终受所选上游模型及部署能力限制。模型发现因此只在上游明确声明或模型 ID 有保守证据时报告多模态/reasoning，不会把所有 chat model 标成支持。

流式响应按 Anthropic 生命周期输出 `message_start`、content block start/delta/stop、`message_delta` 和 `message_stop`，支持 thinking、文本以及多个交错 tool call。解析器支持任意网络拆包、多行 `data:`、注释、CRLF 和 `[DONE]`，并保持背压，不缓存完整响应。

`/v1/messages/count_tokens` 接受 Anthropic 标准的无 `max_tokens` 请求，并转换成 `max_tokens=1` 的非流式最小上游调用，再返回实际模型 tokenizer 报告的 `usage.prompt_tokens`。因此该端点会产生一次最小上游调用和相应计费。

## 多账户和失败策略

网关从 enabled、无需重登且有凭据的账户中选择 `last_used` 最早者。每个账户用 CAS、独立 refresh singleflight 和 `Notify` 协调等待者；同一旧 access token 引发的并发请求只会调用一次 refresh，完成后所有等待者共同使用新 token。

- 后台预刷新与余额探活使用独立节拍；在 `runtime.refresh_skew_secs` 窗口内提前刷新，使请求尽量总能取得热 token。
- 上游 401/403：强制刷新该账户并重试一次；SSE 在发送任何响应数据前的建连 401/403 同样用新 token 重建连接，并记录 `oauth stream retrying after forced refresh`。一旦已向客户端发送 SSE 数据便不重放，避免重复输出。
- refresh token 被明确拒绝或上游报告 `keystone_iam` 不支持刷新：标记需重登并提示运行 `mogick-provider login --account <name> --force`，随后切换其他账户。瞬态网络错误不会误标为必须重登。
- 上游 429：切换到下一账户。
- 网络错误和 5xx：不重放生成请求。
- 后台逐账户刷新和余额探活；`404 ACCOUNT_NOT_FOUND` 作为免费账户正常状态记录为 INFO。

## 日志与错误

所有入站响应和每次上游尝试都会生成单行日志，包括鉴权失败、请求大小错误、健康检查、401 重试、429 切换、OAuth 和余额响应。请求日志包含 request ID、协议、模型、账户、路径、stream、状态、耗时、请求/响应字节、refresh 和 failover。不会记录请求正文。错误正文会截断，并递归清除 Authorization、cookie、API key、access/refresh token、Bearer 和 JWT。

Anthropic 错误使用：

```json
{"type":"error","error":{"type":"invalid_request_error","message":"..."},"request_id":"req_..."}
```

OpenAI 路由继续使用 OpenAI error envelope。健康检查只返回 `ok`。

## 测试

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release
git diff --check
```

默认测试完全离线，使用本地 mock 覆盖 OAuth、迁移、多账户、failover、OpenAI/Anthropic 普通与 SSE 路径。

真实上游测试默认忽略且不会打印凭据。显式执行：

```bash
MOGICK_PROVIDER_REAL_TEST=1 \
MOGICK_REAL_FORCE_REFRESH=1 \
MOGICK_REAL_CHAT_MODEL=deepseek-v4-flash \
MOGICK_REAL_EMBEDDING_MODEL=mm-embedding \
cargo test real_upstream_opt_in_smoke_suite -- --ignored
```

可选设置 `MOGICK_REAL_VISION_MODEL` 和 `MOGICK_REAL_VISION_DATA_URL` 覆盖 vision。真实测试始终检查 models、balance 和多账户轮询（若至少配置两个账户）；模型相关环境变量存在时再执行 chat、stream、embedding、vision，以避免默认产生调用费用。
