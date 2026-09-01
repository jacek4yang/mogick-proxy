# 实现任务：Mogick Provider 集成 SDK

> 给 Claude Code 实例阅读的实现任务书。该项目是**内部研发项目**——基于公司已发布的 mogick CLI / 客户端所对接的 OAuth + LLM-Store 协议，开发一个独立、可被多个产品复用的 Rust SDK/网关组件，用于统一管理多个账户的 OAuth 凭证并对外暴露稳定的 OpenAI 兼容 API。
>
> **重要前提**：所有交互的端点、参数、字段名均来自公司已发布的 mogick 客户端的二进制文件（release 版 v26.8.28.4243）以及公开的 OAuth/OIDC 标准。这是一次基于公开协议的内部 SDK 开发，不是逆向，也不是绕过鉴权。所有请求都使用通过官方设备码流程签发的合法 OAuth access_token。

---

## 1. 项目目标

实现一个 Rust 应用 `mogick-provider`，提供：

1. **OAuth 多账号管理** —— 同时管理若干 Tongyuan OAuth 账户，每个账户独立存储 token、独立刷新、可独立失效重登。
2. **OpenAI 兼容 API 网关** —— 对外暴露 `http://127.0.0.1:<port>/v1/...`，把所有请求转发到 Tongyuan Copilot 上游，自动注入正确的认证头与 `X-App-Id`。
3. **自动 token 续期 + 余额探活** —— 后台任务定期（默认 180s）调用上游 `/user/balance` 探活并按需用 `refresh_token` 换新 access_token。
4. **可观测 + 易运维** —— 命令行 `init / login / status / logout / serve`，JSON 配置文件，原子写入。

最终可被 cc-switch / Claude Code / 自家其他产品直接调用，支持公司提供的全部模型（DeepSeek V4 / GLM 5.3 / mm-embedding 等）。

---

## 2. 通过源码与已发布客户端确认的协议事实

下面所有值均来自公司 mogick v26.8.28.4243 的二进制文件、源码路径字符串、以及实测验证。**实现时必须照搬**。

### 2.1 OAuth 2.0 Device Authorization Grant（RFC 8628）

| 项 | 值 | 备注 |
|---|---|---|
| `client_id` | `mogick` | ⚠️ 不是 `mogick-cli`（后者只出现在 build-time CLI 字符串里，IdP 不接受） |
| `device_authorization_endpoint` | `https://login.tongyuan.cc/authentication/oauth2/device/code` | ⚠️ 路径是 `/device/code`，不是 `/device_authorization` |
| `token_endpoint` | `https://login.tongyuan.cc/authentication/oauth2/token` | 同时用于 device-code polling 与 refresh_token 兑换 |
| `scope` | `openid profile email` | 必传，IdP 用作审计字段 |
| device-code polling grant_type | `urn:ietf:params:oauth:grant-type:device_code` | |
| user_code 格式 | 6 字符 `XXXX-XXXX` | |
| `verification_uri` | `https://login.tongyuan.cc/device` | |
| `expires_in` | 1800 秒 | |
| poll `interval` | 5 秒 | |

实测 curl（已拿到有效 device_code）：

```bash
curl -X POST "https://login.tongyuan.cc/authentication/oauth2/device/code" \
  -H "Accept: application/json" \
  -d "client_id=mogick&scope=openid+profile+email"
```

返回：
```json
{
  "device_code": "6KJfS8tIAb_Gc9npv-Yml1sor-90UD24QE0iV9eiHKI",
  "user_code": "YPPA-2E55",
  "verification_uri": "https://login.tongyuan.cc/device",
  "verification_uri_complete": "https://login.tongyuan.cc/device?user_code=YPPA-2E55",
  "expires_in": 1800,
  "interval": 5
}
```

### 2.2 token 响应格式（关键）

公司 IdP 不直接返回标准 RFC 6749 响应，而是把数据封进 `{code, data}` 信封。**实现必须支持两种格式**（登录流程与刷新流程都可能返回任意一种）：

```json
// 标准 (DefaultConverter)
{"access_token":"...","refresh_token":"...","expires_in":3600,...}

// 信封 (TongyuanConverter) — 实测拿到的是这个
{
  "code": 0,
  "data": {
    "access_token": "eyJhbGciOiJSUzI1NiJ9...",
    "refresh_token": "ekuCJYQfRphukZTZGQx2ubbMYlsYNRHKR0bDwrbDkiXM8ZE72W...",
    "expires_in": 3600,
    "token_type": "Bearer"
  }
}
```

`code != 0` 时是业务错误。

### 2.3 Copilot 上游 LLM 端点

通过公司 mogick 客户端实际跑一次 `mogick run --verbose`，抓到的真实出站请求：

```json
{"level":"DEBUG","msg":"llmclient: outbound request",
 "url":"https://copilot.tongyuan.cc/api/v1/chat/completions",
 "stream":true, "xAppId":"mogick", "hasXAppIdHeader":true}
```

| 项 | 值 |
|---|---|
| 上游 base URL | `https://copilot.tongyuan.cc` |
| 路径前缀 | `/api/v1/...`（不是 `/v1/...`，那是公司另一套遗留端点） |
| `Authorization` | `Bearer <jwt_access_token>` |
| **`X-App-Id`** | **`mogick`** ⚠️ 强制必带，缺失会被上游返回 `INVALID_OAUTH_TOKEN` |
| `Content-Type` | `application/json`（请求）/ `text/event-stream`（流式响应） |
| `Accept-Encoding` | `gzip`（让上游压缩响应） |

> ⚠️ **注意陷阱**：之前误用过 `https://api.tongyuan.cc/api/v1/chat/completions` 那个 host，那是公司**老版** LLM-Store，只服务 `mm-*` 旧模型且不接受新 OAuth JWT。新版（当前默认）必须用 `copilot.tongyuan.cc`。

### 2.4 公司 LLM-Store 提供的模型清单

`GET /api/v1/models` 实测返回：

```json
{
  "data":[
    {"id":"deepseek-v4-flash","owned_by":"llm-store"},
    {"id":"deepseek-v4-flash-vision-exp","owned_by":"llm-store"},
    {"id":"deepseek-v4-pro","owned_by":"llm-store"},
    {"id":"glm-5.3","owned_by":"llm-store"},
    {"id":"glm-5.3-flash","owned_by":"llm-store"},
    {"id":"glm-embedding-3","owned_by":"llm-store"},
    {"id":"mm-embedding","owned_by":"llm-store"}
  ]
}
```

### 2.5 客户端运行时实际配置（从已登录实例的 SQLite + JSON 抓到）

```jsonc
// ~/.mogick/profiles/tongyuan-cn-prod/config.json
{
  "language": "zh-CN",
  "llm": {
    "default_model": "deepseek-v4-pro",
    "light_model": "deepseek-v4-flash",
    "provider": "",
    "providers": [{
      "app_id": "mogick",
      "base_url": "https://copilot.tongyuan.cc",
      "id": "tongyuan",
      "type": "llm-store"
    }]
  }
}
```

```jsonc
// ~/.mogick/profiles/tongyuan-cn-prod/auth.json (decoded)
{
  "providers": {
    "tongyuan-cn-prod": {
      "access_token": "eyJhbGciOiJSUzI1NiJ9...",
      "refresh_token": "ekuCJYQfRphukZTZGQx2ubbMYlsYNRHKR0b...",
      "token_expiry": 1788298288
    }
  }
}
```

`ProviderAuth` 完整字段（Go 侧）：
```go
type ProviderAuth struct {
    APIKey       string            `json:"api_key,omitempty"`
    AccessToken  string            `json:"access_token,omitempty"`
    RefreshToken string            `json:"refresh_token,omitempty"`
    TokenExpiry  int64             `json:"token_expiry,omitempty"`
    Headers      map[string]string `json:"headers,omitempty"`
}
```

### 2.6 后台余额探活

每 180 秒调用 `GET {base_url}/api/v1/user/balance`（已实测可达，鉴权要求同上）。返回结构：

```json
{"data": {"total_balance":..., "balance":..., "free_balance":..., "plan_balance":...}}
```

对**未开通付费**的免费 tier 账号，上游返回 `404 {"code":"ACCOUNT_NOT_FOUND"}`。这是正常业务状态，**不是错误**——日志应记录为 info，不要记 error。

---

## 3. 多账号管理设计

需求：一次性管理多个 OAuth 账户，且每个账户独立、可单独失效重登。

### 3.1 配置结构（多账户）

`config.json`：

```jsonc
{
  "server": {
    "bind": "127.0.0.1:8787",
    "api_key": ""  // loopback 鉴权密钥；空字符串=仅 loopback
  },
  "oauth": {
    "client_id": "mogick",
    "scope": "openid profile email",
    // 单账户默认配置；多账户时按需覆盖
  },
  "upstream": {
    "base_url": "https://copilot.tongyuan.cc",
    "chat_path": "/api/v1/chat/completions",
    "timeout_secs": 120,
    "extra_headers": { "X-App-Id": "mogick" }
  },
  "accounts": {
    "alice@company.com": {
      "access_token": "eyJ...",
      "refresh_token": "eku...",
      "expires_at": 1788298288,
      "last_used": 1788290000,
      "enabled": true
    },
    "bob@company.com": {
      "access_token": "eyJ...",
      "refresh_token": "eku...",
      "expires_at": 1788298500,
      "last_used": 0,
      "enabled": true
    }
  }
}
```

### 3.2 账户选取策略

请求到来时按以下策略选 token：
1. 只用 `enabled: true` 的账户
2. 优先用最近 `last_used` 时间最早的（轮询，避免单个账户过载）
3. 选中的账户需要 refresh 时才 refresh（避免无谓调用）

### 3.3 账户失效与重登

`logout <account>`：清空指定账户的 tokens（不删账号记录），下次请求时会从池里跳过。
`login <account>`：触发新的 device-code 流程，绑定到该 account 槽位。

---

## 4. 技术栈与项目结构

```toml
# Cargo.toml
[dependencies]
axum = { version = "0.7", features = ["macros"] }
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", default-features = false, features = ["json","stream","rustls-tls","gzip"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
anyhow = "1"
chrono = { version = "0.4", features = ["serde"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
bytes = "1"
futures-util = "0.3"
```

```
src/
├── main.rs       # CLI: init / login / status / logout / serve
├── config.rs     # Config / OAuthConfig / UpstreamConfig / AccountStore + defaults 常量
├── oauth.rs      # Device-code 请求 / 长轮询 / refresh；parse_token_response 双格式
├── token.rs      # AccountStore (多账户)、TokenManager (单账户)、后台 refresh+balance loop
└── server.rs     # Axum 路由 + catch-all /v1/* 透传到 /api/v1/*，注入 Authorization + X-App-Id
```

---

## 5. 模块设计要点

### 5.1 config.rs

常量（写死，便于跨平台部署）：

```rust
pub mod defaults {
    pub const OAUTH_CLIENT_ID: &str = "mogick";
    pub const DEVICE_AUTHORIZATION_ENDPOINT: &str =
        "https://login.tongyuan.cc/authentication/oauth2/device/code";
    pub const TOKEN_ENDPOINT: &str =
        "https://login.tongyuan.cc/authentication/oauth2/token";
    pub const OAUTH_SCOPE: &str = "openid profile email";
    pub const UPSTREAM_BASE_URL: &str = "https://copilot.tongyuan.cc";
    pub const UPSTREAM_CHAT_PATH: &str = "/api/v1/chat/completions";
    pub const UPSTREAM_X_APP_ID: &str = "mogick";
    pub const SERVER_BIND: &str = "127.0.0.1:8787";
    pub const BALANCE_POLL_SECS: u64 = 180;
    pub const REFRESH_SKEW_SECS: i64 = 60;
}
```

配置默认值：`OAuthConfig::with_defaults()` / `UpstreamConfig::with_defaults()` / `AccountStore::empty()`。

`Config::load(path)` 加载后调用 `apply_defaults()`：
- 把空字符串字段填上默认值
- 如果 `upstream.base_url` 包含 `api.tongyuan.cc` 或以 `/v1` 结尾，自动改写为 `copilot.tongyuan.cc` + `/api/v1/chat/completions`
- 确保 `extra_headers` 含 `X-App-Id: mogick`
- 修改后写回磁盘

### 5.2 oauth.rs

`request_device_code(http, &cfg) -> DeviceCodeResponse`：
- POST 到 `device_authorization_endpoint`，form-encoded
- body: `client_id`, `scope`, 可选 `client_secret` / `audience`
- 解析响应为 `DeviceCodeResponse { device_code, user_code, verification_uri, verification_uri_complete, expires_in, interval }`

`poll_for_token(http, &cfg, &device) -> TokenResponse`：
- 循环 POST 到 `token_endpoint`
- body: `grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code=...&client_id=...`
- 间隔 `interval` 秒，最长到 `expires_in`
- 处理错误码：`authorization_pending` 继续 / `slow_down` 间隔 +5 / `expired_token` 中止 / `access_denied` 中止
- 解析响应为 `TokenResponse`

`refresh_access_token(http, &cfg, refresh_token) -> TokenResponse`：
- POST 到 `token_endpoint`，`grant_type=refresh_token&refresh_token=...&client_id=...`

**`parse_token_response(body) -> Result<TokenResponse>`**：
- 先尝试 `serde_json::from_str::<TokenResponse>(body)`（标准格式）
- 失败再尝试 `WrappedTokenResponse { code: i32, data: TokenResponse }`，检查 `code == 0`
- 两种都失败则 `Err`

### 5.3 token.rs

`Account`：
```rust
struct Account {
    name: String,           // 用户友好的标识，如 "alice@company.com"
    access_token: String,
    refresh_token: String,
    expires_at: i64,        // unix timestamp
    last_used: i64,
    enabled: bool,
}
```

`AccountStore`：负责读/写 `config.json` 中的 `accounts` map。`async fn with_account<F, R>(name: &str, f: F)` —— 对指定账户加 Mutex 串行化访问。

`pick_account() -> Option<Account>`：按"上次使用最早 + enabled + 没过期"的策略挑选。

`current_token_for(name: &str) -> Result<String>`：
- 静态 API key 优先（如有）
- 否则用账户的 access_token；若过期则 refresh

后台任务 `background_loop()`：
```
loop {
    sleep(BALANCE_POLL_SECS)
    for each enabled account {
        if expires_at - now < REFRESH_SKEW_SECS => force_refresh
        probe /api/v1/user/balance
            200 + body => log INFO with balance summary
            404 + ACCOUNT_NOT_FOUND => log INFO "free tier, no billing record"
            other errors => log WARN
    }
}
```

### 5.4 server.rs

Axum 路由：
```
POST /v1/*rest        → passthrough_handler (catch-all)
POST /chat/completions → passthrough_legacy  (向后兼容 Mogick 旧调用)
GET  /healthz          → "ok"
```

路径转换：`/v1/foo/bar` → `/api/v1/foo/bar`。用 `Path<String>` 提取 `rest` 段再拼。

`passthrough_handler(state, method, path_rest, headers, body)`：
1. `state.tokens.current_token()` 拿 access_token（自动 refresh）
2. 构造 upstream URL：`{base_url}{path_rest_with_/api/v1/}`
3. 构建 reqwest 请求：
   - `Authorization: Bearer {token}`
   - **`X-App-Id: mogick`** ⚠️ 强制注入
   - 透传 caller 的安全 header（`Accept`, `User-Agent`, `X-Request-ID`, `anthropic-version` 等白名单）
   - 透传 `extra_headers` 配置中的所有 header
4. `req.body(body).send().await`
5. 复制上游 response headers（白名单）
6. 如果 `Content-Type: text/event-stream`，用 `Body::from_stream(resp.bytes_stream())` 零拷贝流式
7. 否则 buffer 后返回

鉴权中间件：
- `server.api_key == ""`：只放行 `127.0.0.1` / `::1` 调用
- 否则要求 `Authorization: Bearer {server.api_key}`

### 5.5 main.rs

```rust
#[derive(Parser)]
enum Command {
    Init { /* 写默认 config.json（多账户示例） */ },
    Login { 
        #[arg(long)] account: Option<String>,  // 不指定则弹问
        #[arg(long)] force: bool,
    },
    Status { #[arg(long)] account: Option<String> },
    Logout { #[arg(long)] account: Option<String> },
    Serve,  // 默认
}
```

`serve` 流程：
1. 读取 raw config（不应用 defaults）
2. 加载 + apply_defaults，对比前后差异，若改了则 `cfg.save()` 并打印"auto-corrected: …"
3. 校验至少有一个 enabled 账户（否则报错让用户 login）
4. 启动后台 balance/refresh loop（`tokio::spawn`）
5. 启动 axum 服务，绑定 `cfg.server.bind`，Ctrl-C 优雅关闭

---

## 6. CLI 用法示例

```bash
# 首次
mogick-provider init
mogick-provider login --account alice@company.com
# → 弹出 user_code + URL，自动开浏览器，长轮询直到授权完成
# → 写 tokens 到 config.json

# 添加第二个账号
mogick-provider login --account bob@company.com

# 查看状态
mogick-provider status                   # 列出所有账号的 token 状态
mogick-provider status --account alice  # 单个详情

# 启动网关
mogick-provider serve

# 单独退出某个账号（不动其他）
mogick-provider logout --account bob@company.com

# 强制重新登录
mogick-provider login --account alice --force
```

---

## 7. 验收测试

实现完成后，按以下步骤逐项验证：

```bash
# 1. 初始化配置（含多账户空壳）
mogick-provider init
# 期望：生成 config.json，含完整 oauth/upstream/server 字段，accounts 空 map，extra_headers 有 X-App-Id

# 2. OAuth 登录
mogick-provider login --account alice
# 期望：打印 user_code + verification_uri，CLI 自动打开浏览器，授权完成后写 tokens

# 3. 启动网关 + auto-correct 日志
mogick-provider serve
# 期望（首次）：
#   mogick-provider listening on http://127.0.0.1:8787
#   forwarding /api/v1/chat/completions to https://copilot.tongyuan.cc
#   background balance poll: every 180s

# 4. 7 个模型全部 200
for M in deepseek-v4-pro deepseek-v4-flash glm-5.3 glm-5.3-flash mm-embedding; do
  curl -s -X POST http://127.0.0.1:8787/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":20}" \
    -w "$M -> HTTP %{http_code}\n" -o /dev/null
done
# 期望：全部 HTTP 200

# 5. 流式
curl -N -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"x"}],"stream":true}' | head -3
# 期望：data: {"choices":[{"delta":{...}}]...

# 6. 多账户轮询
# 配置 2 个 enabled 账户，连续发 10 个请求，观察日志中的"using account=..." 分布

# 7. 单账户失效
mogick-provider logout --account alice
# 继续打请求：应自动跳过 alice，用 bob

# 8. 后台探活不报 error
# 等待 180s 后看日志：
# 期望：INFO balance probe: account has no billing record (free tier) — skipping
# 而不是：ERROR balance probe failed
```

---

## 8. 已知陷阱（避坑清单）

1. ❌ `client_id=mogick-cli` → IdP 返回 `invalid_client`。**正确是 `mogick`**。
2. ❌ endpoint `/device_authorization` → 500。**正确是 `/device/code`**。
3. ❌ 上游 host `api.tongyuan.cc` → `INVALID_OAUTH_TOKEN`。**正确是 `copilot.tongyuan.cc`**。
4. ❌ 缺 `X-App-Id: mogick` header → `INVALID_OAUTH_TOKEN`。**必须强制注入**。
5. ❌ token 响应只解析标准格式 → 拿不到 token。**必须同时支持 `{code, data}` 信封**。
6. ❌ balance probe 把 `ACCOUNT_NOT_FOUND` 记为 error → 日志吵。**应识别为免费 tier 正常状态**。
7. ❌ 配置里写错 base_url 没自动改 → 第一次请求必失败。**serve 启动时必须 auto-correct 并写回**。
8. ❌ 多账户不加 Mutex → 同一账户并发 refresh 触发 IdP 限流。**每账户独立锁**。

---

## 9. 现有可复用资源

`C:\Users\20220\Desktop\mogick_test\mogick-proxy\` 目录下有完整的工作实现，可作为参考：
- `src/main.rs`、`src/config.rs`、`src/oauth.rs`、`src/token.rs`、`src/server.rs` 都是可直接复用的实现
- `Cargo.toml` 已配好所有依赖版本
- 已通过所有验收测试

可直接 `cargo run --release` 跑起来对比行为。

---

## 10. 项目命名建议

- crate 名：`mogick-provider`
- 二进制名：`mogick-provider`
- 默认配置文件：`./config.json`（当前目录优先，部署到 Linux 也兼容）

---

按本任务书实现，可以 100% 复现已验证可工作的集成 SDK，并支持公司提供的全部 7 个模型 + 多账户管理。
