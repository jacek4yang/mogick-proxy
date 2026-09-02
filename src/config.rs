//! Runtime configuration, secure credential storage, and legacy migration.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

pub mod defaults {
    pub const OAUTH_CLIENT_ID: &str = "mogick";
    pub const DEVICE_AUTHORIZATION_ENDPOINT: &str =
        "https://login.tongyuan.cc/authentication/oauth2/device/code";
    pub const TOKEN_ENDPOINT: &str = "https://login.tongyuan.cc/authentication/oauth2/token";
    pub const OAUTH_SCOPE: &str = "openid profile email";
    pub const UPSTREAM_BASE_URL: &str = "https://copilot.tongyuan.cc";
    pub const UPSTREAM_API_PREFIX: &str = "/api/v1";
    pub const UPSTREAM_X_APP_ID: &str = "mogick";
    pub const UPSTREAM_TIMEOUT_SECS: u64 = 600;
    pub const SERVER_BIND: &str = "127.0.0.1:8787";
    pub const BALANCE_POLL_SECS: u64 = 180;
    pub const REFRESH_SKEW_SECS: i64 = 60;
    pub const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;
}

/// Provider-owned OAuth settings are deliberately not configurable or persisted.
#[derive(Debug, Clone)]
pub struct OAuthConfig {
    pub client_id: String,
    pub device_authorization_endpoint: String,
    pub token_endpoint: String,
    pub scope: String,
}

pub fn provider_oauth() -> OAuthConfig {
    OAuthConfig {
        client_id: defaults::OAUTH_CLIENT_ID.into(),
        device_authorization_endpoint: defaults::DEVICE_AUTHORIZATION_ENDPOINT.into(),
        token_endpoint: defaults::TOKEN_ENDPOINT.into(),
        scope: defaults::OAUTH_SCOPE.into(),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub upstream: UpstreamConfig,
    pub runtime: RuntimeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ServerConfig {
    pub bind: String,
    pub api_key: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: defaults::SERVER_BIND.into(),
            api_key: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct UpstreamConfig {
    pub base_url: String,
    pub api_prefix: String,
    pub timeout_secs: u64,
    pub extra_headers: BTreeMap<String, String>,
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        let mut extra_headers = BTreeMap::new();
        extra_headers.insert("X-App-Id".into(), defaults::UPSTREAM_X_APP_ID.into());
        Self {
            base_url: defaults::UPSTREAM_BASE_URL.into(),
            api_prefix: defaults::UPSTREAM_API_PREFIX.into(),
            timeout_secs: defaults::UPSTREAM_TIMEOUT_SECS,
            extra_headers,
        }
    }
}

impl UpstreamConfig {
    fn apply_defaults(&mut self) {
        let legacy_host = self.base_url.contains("api.tongyuan.cc");
        let legacy_suffix = self.base_url.trim_end_matches('/').ends_with("/v1");
        if self.base_url.is_empty() || legacy_host || legacy_suffix {
            self.base_url = defaults::UPSTREAM_BASE_URL.into();
        }
        if self.api_prefix.is_empty() {
            self.api_prefix = defaults::UPSTREAM_API_PREFIX.into();
        }
        self.api_prefix = format!("/{}", self.api_prefix.trim_matches('/'));
        if self.timeout_secs == 0 {
            self.timeout_secs = defaults::UPSTREAM_TIMEOUT_SECS;
        }
        if !self
            .extra_headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case("x-app-id"))
        {
            self.extra_headers
                .insert("X-App-Id".into(), defaults::UPSTREAM_X_APP_ID.into());
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Pretty,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RuntimeConfig {
    pub refresh_skew_secs: i64,
    pub balance_poll_secs: u64,
    pub max_request_bytes: usize,
    pub log_level: String,
    pub log_format: LogFormat,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            refresh_skew_secs: defaults::REFRESH_SKEW_SECS,
            balance_poll_secs: defaults::BALANCE_POLL_SECS,
            max_request_bytes: defaults::MAX_REQUEST_BYTES,
            log_level: "info".into(),
            log_format: LogFormat::Pretty,
        }
    }
}

impl RuntimeConfig {
    fn apply_defaults(&mut self) {
        if self.refresh_skew_secs <= 0 {
            self.refresh_skew_secs = defaults::REFRESH_SKEW_SECS;
        }
        if self.balance_poll_secs == 0 {
            self.balance_poll_secs = defaults::BALANCE_POLL_SECS;
        }
        if self.max_request_bytes == 0 {
            self.max_request_bytes = defaults::MAX_REQUEST_BYTES;
        }
        if self.log_level.trim().is_empty() {
            self.log_level = "info".into();
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        let mut config: Self = serde_json::from_str(&raw)
            .with_context(|| format!("parsing config file {}", path.display()))?;
        config.apply_defaults();
        Ok(config)
    }

    pub fn load_or_default(path: &Path) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            Ok(Self::default())
        }
    }

    pub fn apply_defaults(&mut self) {
        if self.server.bind.is_empty() {
            self.server.bind = defaults::SERVER_BIND.into();
        }
        self.upstream.apply_defaults();
        self.runtime.apply_defaults();
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let body = serde_json::to_vec_pretty(self).context("serializing config")?;
        atomic_write(path, &body, false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthStore {
    pub version: u32,
    #[serde(default)]
    pub accounts: BTreeMap<String, AccountAuth>,
}

impl Default for AuthStore {
    fn default() -> Self {
        Self {
            version: 1,
            accounts: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct AccountAuth {
    pub access_token: String,
    pub refresh_token: String,
    #[serde(alias = "expires_at")]
    pub token_expiry: i64,
    pub token_type: String,
    pub scope: String,
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    pub last_used: i64,
    pub reauth_required: bool,
    pub last_error: Option<AuthErrorRecord>,
}

fn enabled_by_default() -> bool {
    true
}

impl AccountAuth {
    pub fn has_credentials(&self) -> bool {
        !self.access_token.is_empty() || !self.refresh_token.is_empty()
    }

    pub fn usable(&self) -> bool {
        self.enabled
            && (self.has_valid_access()
                || (!self.reauth_required && !self.refresh_token.is_empty()))
    }

    pub fn has_valid_access(&self) -> bool {
        !self.access_token.is_empty() && self.token_expiry > chrono::Utc::now().timestamp()
    }

    pub fn needs_refresh(&self, skew_secs: i64) -> bool {
        self.access_token.is_empty()
            || self
                .token_expiry
                .saturating_sub(chrono::Utc::now().timestamp())
                <= skew_secs
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthErrorRecord {
    pub at: i64,
    pub message: String,
}

impl AuthStore {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path)
            .with_context(|| format!("reading auth file {}", path.display()))?;
        let store: Self = serde_json::from_str(&raw)
            .with_context(|| format!("parsing auth file {}", path.display()))?;
        if store.version != 1 {
            bail!("unsupported auth.json version {}", store.version);
        }
        Ok(store)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let body = serde_json::to_vec_pretty(self).context("serializing auth store")?;
        atomic_write(path, &body, true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationOutcome {
    NotNeeded,
    Migrated,
}

/// Securely move credentials out of legacy config formats.
pub fn migrate_legacy(config_path: &Path, auth_path: &Path) -> Result<MigrationOutcome> {
    let raw = fs::read_to_string(config_path)
        .with_context(|| format!("reading legacy config {}", config_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parsing legacy config {}", config_path.display()))?;
    let Some(object) = value.as_object() else {
        bail!("config root must be a JSON object");
    };
    let has_legacy = object.contains_key("oauth")
        || object.contains_key("tokens")
        || object.contains_key("accounts")
        || object
            .get("upstream")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|u| u.contains_key("static_api_key") || u.contains_key("chat_path"));
    if !has_legacy {
        return Ok(MigrationOutcome::NotNeeded);
    }

    let mut incoming = BTreeMap::new();
    if let Some(tokens) = object.get("tokens") {
        let account: AccountAuth =
            serde_json::from_value(tokens.clone()).context("parsing legacy tokens")?;
        if account.has_credentials() {
            incoming.insert("default".to_string(), account);
        }
    }
    if let Some(accounts) = object
        .get("accounts")
        .and_then(serde_json::Value::as_object)
    {
        for (name, value) in accounts {
            let account: AccountAuth = serde_json::from_value(value.clone())
                .with_context(|| format!("parsing legacy account {name}"))?;
            if account.has_credentials() {
                incoming.insert(name.clone(), account);
            }
        }
    }

    let mut auth = AuthStore::load(auth_path)?;
    for (name, account) in incoming {
        match auth.accounts.get(&name) {
            Some(existing)
                if existing.has_credentials() && !same_credentials(existing, &account) =>
            {
                bail!(
                    "auth account {name:?} already contains different credentials; legacy config was preserved"
                );
            }
            Some(existing) if existing.has_credentials() => {}
            _ => {
                auth.accounts.insert(name, account);
            }
        }
    }

    auth.save(auth_path)?;
    let verified = AuthStore::load(auth_path)?;
    if verified != auth {
        return Err(anyhow!("auth verification failed after atomic write"));
    }

    let mut clean: Config = serde_json::from_value(value).context("loading runtime config")?;
    clean.apply_defaults();
    clean.save(config_path)?;
    Ok(MigrationOutcome::Migrated)
}

fn same_credentials(left: &AccountAuth, right: &AccountAuth) -> bool {
    left.access_token == right.access_token && left.refresh_token == right.refresh_token
}

fn atomic_write(path: &Path, body: &[u8], secret: bool) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("state.json");
    let tmp = path.with_file_name(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(if secret { 0o600 } else { 0o644 });
    }
    let mut file = options
        .open(&tmp)
        .with_context(|| format!("opening temporary file {}", tmp.display()))?;
    file.write_all(body)
        .with_context(|| format!("writing temporary file {}", tmp.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing temporary file {}", tmp.display()))?;
    drop(file);
    fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    #[cfg(unix)]
    if secret {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("setting 0600 permissions on {}", path.display()))?;
    }
    Ok(())
}

pub fn default_config_path() -> PathBuf {
    std::env::var_os("MOGICK_PROVIDER_CONFIG")
        .or_else(|| std::env::var_os("MOGICK_PROXY_CONFIG"))
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config.json"))
}

pub fn default_auth_path(config_path: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("MOGICK_PROVIDER_AUTH").filter(|p| !p.is_empty()) {
        return PathBuf::from(path);
    }
    config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join("auth.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mogick-provider-{name}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn defaults_have_no_credentials() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("access_token"));
        assert!(!json.contains("refresh_token"));
        assert_eq!(config.upstream.api_prefix, "/api/v1");
        assert_eq!(
            config.upstream.extra_headers.get("X-App-Id").unwrap(),
            "mogick"
        );
    }

    #[test]
    fn reauth_required_keeps_only_unexpired_access_usable() {
        let mut account = AccountAuth {
            access_token: "access".into(),
            refresh_token: "refresh".into(),
            token_expiry: chrono::Utc::now().timestamp() + 60,
            enabled: true,
            reauth_required: true,
            ..AccountAuth::default()
        };
        assert!(account.usable());
        account.token_expiry = 1;
        assert!(!account.usable());
    }

    #[test]
    fn legacy_migration_is_secure_and_idempotent() {
        let dir = test_dir("migration");
        let config_path = dir.join("config.json");
        let auth_path = dir.join("auth.json");
        fs::write(
            &config_path,
            r#"{
              "server":{"bind":"127.0.0.1:9999","api_key":""},
              "oauth":{"client_id":"mogick"},
              "upstream":{"base_url":"https://copilot.tongyuan.cc","chat_path":"/api/v1/chat/completions"},
              "tokens":{"access_token":"access-secret","refresh_token":"refresh-secret","expires_at":42}
            }"#,
        )
        .unwrap();

        assert_eq!(
            migrate_legacy(&config_path, &auth_path).unwrap(),
            MigrationOutcome::Migrated
        );
        let clean = fs::read_to_string(&config_path).unwrap();
        assert!(!clean.contains("secret"));
        assert!(!clean.contains("oauth"));
        let auth = AuthStore::load(&auth_path).unwrap();
        assert_eq!(auth.accounts["default"].access_token, "access-secret");
        assert_eq!(
            migrate_legacy(&config_path, &auth_path).unwrap(),
            MigrationOutcome::NotNeeded
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&auth_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn migration_conflict_preserves_legacy_config() {
        let dir = test_dir("conflict");
        let config_path = dir.join("config.json");
        let auth_path = dir.join("auth.json");
        let legacy = r#"{"oauth":{},"tokens":{"access_token":"old","refresh_token":"old-r"}}"#;
        fs::write(&config_path, legacy).unwrap();
        let mut auth = AuthStore::default();
        auth.accounts.insert(
            "default".into(),
            AccountAuth {
                access_token: "new".into(),
                refresh_token: "new-r".into(),
                enabled: true,
                ..AccountAuth::default()
            },
        );
        auth.save(&auth_path).unwrap();
        assert!(migrate_legacy(&config_path, &auth_path).is_err());
        assert_eq!(fs::read_to_string(&config_path).unwrap(), legacy);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn migration_write_failure_does_not_remove_tokens() {
        let dir = test_dir("rollback");
        let config_path = dir.join("config.json");
        fs::write(
            &config_path,
            r#"{"tokens":{"access_token":"keep-me","refresh_token":"keep-me-too"}}"#,
        )
        .unwrap();
        let impossible_auth = dir.join("parent-is-a-file").join("auth.json");
        fs::write(dir.join("parent-is-a-file"), "x").unwrap();
        assert!(migrate_legacy(&config_path, &impossible_auth).is_err());
        assert!(fs::read_to_string(&config_path)
            .unwrap()
            .contains("keep-me"));
        fs::remove_dir_all(dir).unwrap();
    }
}
