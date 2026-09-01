//! Token manager: persists the active access/refresh tokens and refreshes
//! them transparently before they expire.
//!
//! All access from the HTTP server goes through [`TokenManager::current_token`].
//! If the cached access token is missing or about to expire, the manager
//! blocks on a single in-flight refresh so concurrent requests share one
//! round trip to the IdP.
//!
//! Also exposes [`TokenManager::background_loop`] which spawns a tokio task
//! that:
//!   1. Calls `/api/v1/user/balance` every `BALANCE_POLL_SECS` seconds
//!      (default 180 = 3 min, matching what Mogick's "balance gate" does).
//!      This both keeps the session warm on the IdP side and surfaces the
//!      remaining quota to the operator in the proxy logs.
//!   2. Triggers an explicit refresh whenever the cached access token is
//!      about to expire (`REFRESH_SKEW_SECS` seconds early), so callers
//!      served via [`current_token`] never have to wait.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use chrono::Utc;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::config::{Config, TokenState};
use crate::oauth::{refresh_access_token, TokenResponse};

/// Refresh tokens this many seconds early to absorb clock skew / network jitter.
const REFRESH_SKEW_SECS: i64 = 60;

/// Background balance poll cadence. Mogick's "balance gate" runs at the
/// same cadence (`mogick.network.balance.gate_*` metrics fire on this
/// schedule); matching it makes the proxy behave like a native client.
pub const BALANCE_POLL_SECS: u64 = 180;

/// Flat balance fields — matches the `subscription.UserBalance` shape
/// found in `mogick.exe`. Used as the `data` field of the upstream's
/// `{ "code": 0, "data": UserBalance }` envelope.
#[derive(Debug, Deserialize)]
struct BalanceData {
    #[serde(default)]
    total_balance: Option<serde_json::Value>,
    #[serde(default)]
    balance: Option<serde_json::Value>,
    #[serde(default)]
    free_balance: Option<serde_json::Value>,
    #[serde(default)]
    plan_balance: Option<serde_json::Value>,
}

/// `{ "code": 0, "data": ... }` envelope used by Mogick's keystone_iam
/// upstream responses.
#[derive(Debug, Deserialize)]
struct BalanceEnvelope {
    code: i32,
    data: BalanceData,
}

#[derive(Clone)]
pub struct TokenManager {
    inner: Arc<Inner>,
}

struct Inner {
    config_path: PathBuf,
    /// Held while a refresh is in progress so concurrent callers wait.
    refresh_lock: Mutex<()>,
    http: reqwest::Client,
}

impl TokenManager {
    pub fn new(config_path: PathBuf) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("building reqwest client");
        Self {
            inner: Arc::new(Inner {
                config_path,
                refresh_lock: Mutex::new(()),
                http,
            }),
        }
    }

    /// Return a valid access token, refreshing if needed.
    pub async fn current_token(&self) -> Result<String> {
        // First quick check: if the cached token is fine, skip the lock entirely.
        {
            let cfg = Config::load(&self.inner.config_path)?;
            if let Some(key) = cfg.upstream.static_api_key.clone() {
                if !key.is_empty() {
                    return Ok(key);
                }
            }
            if !cfg.tokens.needs_refresh(REFRESH_SKEW_SECS) {
                return Ok(cfg.tokens.access_token);
            }
        }

        // Slow path: serialise refresh attempts.
        let _guard = self.inner.refresh_lock.lock().await;
        // Re-check inside the lock — another worker may have refreshed already.
        {
            let cfg = Config::load(&self.inner.config_path)?;
            if !cfg.tokens.needs_refresh(REFRESH_SKEW_SECS) {
                return Ok(cfg.tokens.access_token);
            }
            if cfg.tokens.refresh_token.is_empty() {
                return Err(anyhow!(
                    "no refresh_token available — run `mogick-proxy login` first"
                ));
            }
            match refresh_access_token(
                &self.inner.http,
                &cfg.oauth,
                &cfg.tokens.refresh_token,
            )
            .await
            {
                Ok(new) => self.persist_new_tokens(&cfg.tokens, new)?,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("refresh")
                        || msg.contains("not_supported")
                        || msg.contains("invalid_grant")
                    {
                        return Err(anyhow!(
                            "refresh failed ({}). Re-run `mogick-proxy login --force` to re-authenticate",
                            msg
                        ));
                    }
                    return Err(e.context("refreshing access token"));
                }
            }
        }
        let cfg = Config::load(&self.inner.config_path)?;
        Ok(cfg.tokens.access_token)
    }

    /// Force a refresh now, regardless of expiry. Used by the background
    /// loop to refresh opportunistically when the balance probe indicates
    /// the token may have been rotated by the IdP.
    pub async fn force_refresh(&self) -> Result<()> {
        let _guard = self.inner.refresh_lock.lock().await;
        let cfg = Config::load(&self.inner.config_path)?;
        if cfg.tokens.refresh_token.is_empty() {
            return Err(anyhow!("no refresh_token to use"));
        }
        let new = refresh_access_token(
            &self.inner.http,
            &cfg.oauth,
            &cfg.tokens.refresh_token,
        )
        .await
        .context("refreshing access token")?;
        self.persist_new_tokens(&cfg.tokens, new)?;
        Ok(())
    }

    /// Persist the result of a brand-new device-code login. Replaces any
    /// previously stored tokens.
    pub fn store_initial_tokens(&self, resp: &TokenResponse) -> Result<()> {
        let mut cfg = Config::load(&self.inner.config_path)?;
        cfg.tokens = tokens_from_response(resp, &cfg.tokens.refresh_token);
        cfg.save(&self.inner.config_path)?;
        Ok(())
    }

    fn persist_new_tokens(
        &self,
        prev: &TokenState,
        resp: TokenResponse,
    ) -> Result<()> {
        let mut cfg = Config::load(&self.inner.config_path)?;
        let refresh_token = resp
            .refresh_token
            .clone()
            .unwrap_or_else(|| prev.refresh_token.clone());
        let new_expiry = compute_expires_at(resp.expires_in);
        cfg.tokens = TokenState {
            access_token: resp.access_token,
            refresh_token,
            expires_at: new_expiry,
            token_type: resp
                .token_type
                .unwrap_or_else(|| "Bearer".to_string()),
            scope: resp.scope.unwrap_or_else(|| prev.scope.clone()),
        };
        cfg.save(&self.inner.config_path)?;
        tracing::info!(
            expires_at = new_expiry,
            "access token refreshed and persisted"
        );
        Ok(())
    }

    /// Drop all stored tokens (logout).
    pub fn clear(&self) -> Result<()> {
        let mut cfg = Config::load(&self.inner.config_path)?;
        cfg.tokens = TokenState::default();
        cfg.save(&self.inner.config_path)?;
        Ok(())
    }

    /// Read-only view of the current token state (for `status` subcommand).
    pub fn snapshot(&self) -> Result<TokenState> {
        Ok(Config::load(&self.inner.config_path)?.tokens)
    }

    /// Run the background loop until the process is killed. The loop:
    ///  * sleeps for `BALANCE_POLL_SECS`
    ///  * fetches the user balance (logs it, no-op on errors)
    ///  * if the cached access token is near expiry, kicks off a refresh
    pub async fn background_loop(self) {
        tracing::info!(
            cadence_secs = BALANCE_POLL_SECS,
            "background balance+refresh loop starting"
        );
        loop {
            // First, opportunistically refresh if needed.
            match self.try_refresh_if_needed().await {
                Ok(true) => tracing::info!("background: refresh done"),
                Ok(false) => {}
                Err(e) => tracing::warn!(error=%e, "background: refresh failed"),
            }

            // Then probe the balance endpoint.
            match self.probe_balance().await {
                Ok(summary) => tracing::info!(balance=%summary, "balance probe ok"),
                Err(e) => tracing::debug!(error=%e, "balance probe failed"),
            }

            tokio::time::sleep(Duration::from_secs(BALANCE_POLL_SECS)).await;
        }
    }

    /// Returns Ok(true) when a refresh was performed.
    async fn try_refresh_if_needed(&self) -> Result<bool> {
        let needs = {
            let cfg = Config::load(&self.inner.config_path)?;
            cfg.tokens.needs_refresh(REFRESH_SKEW_SECS)
                && !cfg.tokens.refresh_token.is_empty()
        };
        if !needs {
            return Ok(false);
        }
        self.force_refresh().await?;
        Ok(true)
    }

    /// Probe `/api/v1/user/balance` and return a short human-readable
    /// summary. Errors are non-fatal — the operator can see the failure
    /// in the proxy logs and act on it.
    async fn probe_balance(&self) -> Result<String> {
        let cfg = Config::load(&self.inner.config_path)?;
        // Skip when using a static API key (no balance concept).
        if let Some(k) = cfg.upstream.static_api_key.as_ref() {
            if !k.is_empty() {
                return Ok("(static api key in use — balance probe skipped)".into());
            }
        }
        if cfg.tokens.access_token.is_empty() {
            return Err(anyhow!("no access token"));
        }
        let url = format!(
            "{}/api/v1/user/balance",
            cfg.upstream.base_url.trim_end_matches('/')
        );
        // The copilot upstream requires `X-App-Id: mogick` on every
        // authenticated request; without it the upstream returns
        // `INVALID_OAUTH_TOKEN` even with a valid JWT.
        let x_app_id = crate::config::defaults::UPSTREAM_X_APP_ID.to_string();
        let resp = self
            .inner
            .http
            .get(&url)
            .header(reqwest::header::AUTHORIZATION, format!("Bearer {}", cfg.tokens.access_token))
            .header("X-App-Id", x_app_id)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .context("GET user/balance")?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();

        if !status.is_success() {
            // Special-case: the upstream returns 404 ACCOUNT_NOT_FOUND
            // for free-tier accounts that have never activated billing.
            // That's a normal state, not an error — log it as info
            // instead of an `Err` to keep logs clean.
            if status.as_u16() == 404 && body.contains("ACCOUNT_NOT_FOUND") {
                tracing::info!(
                    "balance probe: account has no billing record (free tier) — skipping"
                );
                return Ok("(no billing record on this account)".into());
            }
            return Err(anyhow!(
                "user/balance HTTP {}: {}",
                status,
                body.chars().take(200).collect::<String>()
            ));
        }

        // The upstream wraps responses in `{ "code": 0, "data": ... }`.
        // Try the wrapped shape first, then fall back to flat.
        let summary = if let Ok(w) = serde_json::from_str::<BalanceEnvelope>(&body) {
            if w.code != 0 {
                return Err(anyhow!("user/balance code={}", w.code));
            }
            format_balance(&w.data)
        } else if let Ok(f) = serde_json::from_str::<BalanceData>(&body) {
            format_balance(&f)
        } else {
            return Err(anyhow!("user/balance: unrecognised response shape"));
        };
        Ok(summary)
    }
}

fn format_balance(f: &BalanceData) -> String {
    let total = f.total_balance.as_ref().map(|v| v.to_string()).unwrap_or_default();
    let bal = f.balance.as_ref().map(|v| v.to_string()).unwrap_or_default();
    let free = f.free_balance.as_ref().map(|v| v.to_string()).unwrap_or_default();
    let plan = f.plan_balance.as_ref().map(|v| v.to_string()).unwrap_or_default();
    let mut parts = Vec::new();
    if !bal.is_empty() { parts.push(format!("balance={}", bal)); }
    if !total.is_empty() { parts.push(format!("total={}", total)); }
    if !free.is_empty() { parts.push(format!("free={}", free)); }
    if !plan.is_empty() { parts.push(format!("plan={}", plan)); }
    if parts.is_empty() { "(empty)".into() } else { parts.join(" ") }
}

fn tokens_from_response(resp: &TokenResponse, prev_refresh: &str) -> TokenState {
    let refresh_token = resp
        .refresh_token
        .clone()
        .unwrap_or_else(|| prev_refresh.to_string());
    TokenState {
        access_token: resp.access_token.clone(),
        refresh_token,
        expires_at: compute_expires_at(resp.expires_in),
        token_type: resp.token_type.clone().unwrap_or_else(|| "Bearer".into()),
        scope: resp.scope.clone().unwrap_or_default(),
    }
}

fn compute_expires_at(expires_in: Option<i64>) -> i64 {
    let secs = expires_in.unwrap_or(3600).max(1);
    Utc::now().timestamp() + secs
}

use anyhow::Context as _;
