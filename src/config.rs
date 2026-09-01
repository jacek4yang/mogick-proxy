//! Configuration types persisted as JSON on disk.
//!
//! The config file holds both the *static* OAuth client / upstream
//! configuration and the *dynamic* token state. The two halves live
//! under different top-level keys so static config can be checked
//! into version control while the token state stays local-only.
//!
//! All OAuth fields except `oauth` itself have hard-coded defaults
//! reverse-engineered from the Mogick binary
//! (`tongyuan.cc/ai/mogick/oauth/keystone.go` + `oauth/deviceflow.go`).
//! Strings inside the binary confirmed:
//!   - `https://login.tongyuan.cc/authentication/oauth2`
//!   - `keystone_iam: requesting device code`
//!   - `keystone_iam: token obtained`
//!   - `keystone_iam token refresh is not supported, please run 'mogick setup' to re-authenticate`
//! So a fresh `mogick-proxy login` works out of the box on a default
//! `config.json` — no manual setup needed.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// OAuth endpoints + client_id reverse-engineered from the **Windows
/// production** Mogick binary
/// (`mogick-windows-x64.zip` -> `mogick.exe`, SHA256 `…`) and confirmed
/// live against the IdP.
///
/// Live confirmation with `curl`:
/// ```text
/// POST https://login.tongyuan.cc/authentication/oauth2/device/code
///   client_id=mogick
///   scope=openid profile email
/// =>
/// {
///   "device_code":"6KJfS8tIAb_Gc9npv-Yml1sor-90UD24QE0iV9eiHKI",
///   "user_code":"YPPA-2E55",
///   "verification_uri":"https://login.tongyuan.cc/device",
///   "verification_uri_complete":"https://login.tongyuan.cc/device?user_code=YPPA-2E55",
///   "expires_in":1800,
///   "interval":5
/// }
/// ```
pub mod defaults {
    /// OAuth 2.0 client identifier. Confirmed live-accepted by the IdP
    /// (this is **not** `mogick-cli`, which appears in the binary only
    /// as a build-time CLI product string — `mogick-cli` returns
    /// `invalid_client` from the live IdP).
    pub const OAUTH_CLIENT_ID: &str = "mogick";

    /// Device authorization endpoint (RFC 8628). Confirmed live-accepted.
    /// Note the path is `/device/code`, not `/device_authorization`.
    pub const DEVICE_AUTHORIZATION_ENDPOINT: &str =
        "https://login.tongyuan.cc/authentication/oauth2/device/code";

    /// Token endpoint (the same URL is used for device-code polling
    /// and for refresh_token exchange).
    pub const TOKEN_ENDPOINT: &str =
        "https://login.tongyuan.cc/authentication/oauth2/token";

    /// Userinfo endpoint (kept for reference).
    #[allow(dead_code)]
    pub const USERINFO_ENDPOINT: &str =
        "https://login.tongyuan.cc/authentication/userinfo";

    /// OAuth scope the IdP expects. Confirmed live-accepted.
    pub const OAUTH_SCOPE: &str = "openid profile email";

    /// The upstream LLM API base (no trailing slash). Mogick's runtime
    /// profile points at `https://copilot.tongyuan.cc`, **not**
    /// `https://api.tongyuan.cc`. Confirmed by capturing mogick.exe's
    /// actual outbound HTTP request via `--verbose`:
    ///   `llmclient: outbound request url=https://copilot.tongyuan.cc/api/v1/chat/completions`
    /// Both the JWT access_token AND the `X-App-Id: mogick` header are
    /// required by this upstream.
    pub const UPSTREAM_BASE_URL: &str = "https://copilot.tongyuan.cc";

    /// Chat completion path appended to `UPSTREAM_BASE_URL`.
    pub const UPSTREAM_CHAT_PATH: &str = "/api/v1/chat/completions";

    /// `X-App-Id` header value. Required by the tongyuan copilot
    /// upstream; without it the upstream returns
    /// `INVALID_OAUTH_TOKEN` even with a valid JWT.
    pub const UPSTREAM_X_APP_ID: &str = "mogick";

    /// Default bind address for the reverse proxy.
    pub const SERVER_BIND: &str = "127.0.0.1:8787";
}

/// Top-level config file schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Static, user-provided configuration.
    #[serde(default)]
    pub server: ServerConfig,
    /// OAuth provider configuration. Every field has a hard-coded
    /// default from the IDA dump — missing values are auto-filled.
    pub oauth: OAuthConfig,
    /// The upstream LLM API the proxy forwards to.
    pub upstream: UpstreamConfig,
    /// Dynamic token state. Saved to disk so restarts pick up the
    /// refresh_token without requiring a fresh interactive login.
    #[serde(default)]
    pub tokens: TokenState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Address the proxy listens on, e.g. `127.0.0.1:8787`.
    pub bind: String,
    /// Shared secret that callers (mogick, Claude Code via wrapper, ...) must
    /// present in `Authorization: Bearer <secret>`. Leave empty to disable.
    #[serde(default)]
    pub api_key: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: defaults::SERVER_BIND.to_string(),
            api_key: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthConfig {
    /// OAuth 2.0 client_id issued by the IdP.
    /// Defaults to `defaults::OAUTH_CLIENT_ID` when missing.
    pub client_id: String,
    /// Optional client_secret for confidential clients (Device flow
    /// doesn't need one — Mogick's `mogick` client is public).
    #[serde(default)]
    pub client_secret: Option<String>,
    /// Device authorization endpoint (RFC 8628).
    /// Defaults to `defaults::DEVICE_AUTHORIZATION_ENDPOINT` when missing.
    pub device_authorization_endpoint: String,
    /// Token endpoint. Defaults to `defaults::TOKEN_ENDPOINT` when missing.
    pub token_endpoint: String,
    /// Space-separated scope string.
    #[serde(default)]
    pub scope: String,
}

impl OAuthConfig {
    /// Build an `OAuthConfig` populated with the IDA-extracted defaults.
    pub fn with_defaults() -> Self {
        Self {
            client_id: defaults::OAUTH_CLIENT_ID.into(),
            client_secret: None,
            device_authorization_endpoint: defaults::DEVICE_AUTHORIZATION_ENDPOINT.into(),
            token_endpoint: defaults::TOKEN_ENDPOINT.into(),
            scope: defaults::OAUTH_SCOPE.into(),
        }
    }

    /// Fill in any empty string fields with the IDA defaults so a
    /// half-written `config.json` still works.
    pub fn apply_defaults(&mut self) {
        if self.client_id.is_empty() {
            self.client_id = defaults::OAUTH_CLIENT_ID.into();
        }
        if self.device_authorization_endpoint.is_empty() {
            self.device_authorization_endpoint = defaults::DEVICE_AUTHORIZATION_ENDPOINT.into();
        }
        if self.token_endpoint.is_empty() {
            self.token_endpoint = defaults::TOKEN_ENDPOINT.into();
        }
        if self.scope.is_empty() {
            self.scope = defaults::OAUTH_SCOPE.into();
        }
    }
}

/// Upstream LLM API the proxy forwards to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamConfig {
    /// Base URL with no trailing slash, e.g. `https://api.tongyuan.cc/v1`.
    /// Defaults to `defaults::UPSTREAM_BASE_URL` when missing.
    pub base_url: String,
    /// Optional path appended to `base_url`. Defaults to `/chat/completions`.
    #[serde(default = "default_chat_path")]
    pub chat_path: String,
    /// Optional hard-coded API key to use instead of the OAuth access token.
    /// When set, OAuth is bypassed entirely.
    #[serde(default)]
    pub static_api_key: Option<String>,
    /// Extra headers forwarded verbatim to the upstream (e.g. for tenant routing).
    #[serde(default)]
    pub extra_headers: std::collections::BTreeMap<String, String>,
    /// Request timeout in seconds for upstream calls.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl UpstreamConfig {
    pub fn with_defaults() -> Self {
        Self {
            base_url: defaults::UPSTREAM_BASE_URL.into(),
            chat_path: defaults::UPSTREAM_CHAT_PATH.into(),
            static_api_key: None,
            extra_headers: Default::default(),
            timeout_secs: default_timeout(),
        }
    }

    pub fn apply_defaults(&mut self) {
        // Auto-correct known-bad legacy base URLs to the current
        // upstream (copilot.tongyuan.cc). The old `api.tongyuan.cc`
        // host only serves the older mm-* models and rejects the
        // X-App-Id handshake that copilot.tongyuan.cc requires.
        let needs_correction = self.base_url.is_empty()
            || self.base_url.contains("api.tongyuan.cc")
            || self.base_url.trim_end_matches('/').ends_with("/v1")
            || self.chat_path.is_empty()
            || self.chat_path == "/chat/completions";
        if needs_correction {
            self.base_url = defaults::UPSTREAM_BASE_URL.into();
            self.chat_path = defaults::UPSTREAM_CHAT_PATH.into();
        }
        // Ensure X-App-Id is in extra_headers — required by the
        // copilot upstream. Don't overwrite if the user set it.
        let x_app = defaults::UPSTREAM_X_APP_ID;
        if !self.extra_headers.values().any(|v| v == x_app)
            && !self.extra_headers.keys().any(|k| k.eq_ignore_ascii_case("X-App-Id"))
        {
            self.extra_headers.insert("X-App-Id".to_string(), x_app.to_string());
        }
    }
}

fn default_chat_path() -> String {
    "/chat/completions".to_string()
}
fn default_timeout() -> u64 {
    120
}

/// Persisted OAuth token state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenState {
    /// Current access token. Empty when no login has happened yet.
    #[serde(default)]
    pub access_token: String,
    /// Refresh token. Used to obtain new access tokens without
    /// requiring the user to log in again.
    #[serde(default)]
    pub refresh_token: String,
    /// Unix timestamp at which `access_token` expires.
    #[serde(default)]
    pub expires_at: i64,
    /// IdP-provided token type, usually `Bearer`.
    #[serde(default)]
    pub token_type: String,
    /// Space-separated scopes granted by the user.
    #[serde(default)]
    pub scope: String,
}

impl TokenState {
    pub fn is_empty(&self) -> bool {
        self.access_token.is_empty() && self.refresh_token.is_empty()
    }

    /// True when no access token is currently valid.
    pub fn needs_refresh(&self, skew_secs: i64) -> bool {
        if self.access_token.is_empty() {
            return true;
        }
        let now = chrono::Utc::now().timestamp();
        self.expires_at.saturating_sub(now) <= skew_secs
    }
}

impl Config {
    /// Load config from `path`, then back-fill any empty OAuth/upstream
    /// fields with the IDA-extracted defaults. Returns an error only if
    /// the file is missing or malformed.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        let mut cfg: Config = serde_json::from_str(&raw)
            .with_context(|| format!("parsing config file {}", path.display()))?;
        cfg.oauth.apply_defaults();
        cfg.upstream.apply_defaults();
        Ok(cfg)
    }

    /// Load `path` if it exists, otherwise return a starter config built
    /// from defaults + a few overrides useful for the most common setup.
    pub fn load_or_default(path: &Path) -> Result<Self> {
        if path.exists() {
            return Self::load(path);
        }
        Ok(Self {
            server: ServerConfig::default(),
            oauth: OAuthConfig::with_defaults(),
            upstream: UpstreamConfig::with_defaults(),
            tokens: TokenState::default(),
        })
    }

    /// Atomically save the config back to `path` (write to a temp file then rename).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating config dir {}", parent.display()))?;
        }
        let body = serde_json::to_string_pretty(self).context("serialising config")?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, body)
            .with_context(|| format!("writing tmp config {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("renaming tmp config to {}", path.display()))?;
        Ok(())
    }
}

/// Resolve the default config file path.
///
/// Resolution order:
///   1. `$MOGICK_PROXY_CONFIG` if set and non-empty.
///   2. `./config.json` in the current working directory.
///
/// We deliberately do NOT fall back to a per-user config dir
/// (e.g. `%APPDATA%/mogick-proxy/config.json` or `~/.config/...`)
/// because the binary is meant to be run alongside its own
/// `config.json` — the same file is shared by `init`, `login`,
/// `serve`, etc. This keeps Windows + Linux behaviour identical
/// and makes the proxy easy to ship as a portable bundle.
pub fn default_config_path() -> PathBuf {
    if let Ok(p) = std::env::var("MOGICK_PROXY_CONFIG") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    PathBuf::from("config.json")
}
