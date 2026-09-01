//! Multi-account selection, isolated refresh locks, and balance probing.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use serde::Deserialize;
use tokio::sync::{Mutex, Notify};

use crate::config::{provider_oauth, AccountAuth, AuthErrorRecord, AuthStore, Config, OAuthConfig};
use crate::oauth::{is_permanent_refresh_error, refresh_access_token, TokenResponse};

#[derive(Debug, Clone)]
pub struct SelectedAccount {
    pub name: String,
    pub access_token: String,
    pub refreshed: bool,
}

#[derive(Clone)]
pub struct AccountManager {
    inner: Arc<Inner>,
}

struct Inner {
    config: Config,
    oauth: OAuthConfig,
    auth_path: PathBuf,
    http: reqwest::Client,
    store_lock: Mutex<()>,
    account_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    refresh_flights: Mutex<HashMap<String, Arc<RefreshFlight>>>,
    last_used_clock: AtomicI64,
}

#[derive(Default)]
struct RefreshFlight {
    running: AtomicBool,
    generation: AtomicU64,
    notify: Notify,
}

struct RefreshLeader {
    flight: Arc<RefreshFlight>,
}

impl Drop for RefreshLeader {
    fn drop(&mut self) {
        self.flight.generation.fetch_add(1, Ordering::Release);
        self.flight.running.store(false, Ordering::Release);
        self.flight.notify.notify_waiters();
    }
}

impl RefreshFlight {
    fn try_claim(self: &Arc<Self>) -> std::result::Result<RefreshLeader, u64> {
        match self
            .running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => Ok(RefreshLeader {
                flight: self.clone(),
            }),
            Err(_) => Err(self.generation.load(Ordering::Acquire)),
        }
    }

    async fn wait(&self, generation: u64) {
        while self.running.load(Ordering::Acquire)
            && self.generation.load(Ordering::Acquire) == generation
        {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !self.running.load(Ordering::Acquire)
                || self.generation.load(Ordering::Acquire) != generation
            {
                break;
            }
            notified.await;
        }
    }
}

#[derive(Debug, Deserialize)]
struct BalanceEnvelope {
    #[serde(default)]
    code: serde_json::Value,
    #[serde(default)]
    data: serde_json::Value,
}

impl AccountManager {
    pub fn new(config: Config, auth_path: PathBuf) -> Result<Self> {
        Self::new_with_oauth(config, auth_path, provider_oauth())
    }

    pub(crate) fn new_with_oauth(
        config: Config,
        auth_path: PathBuf,
        oauth: OAuthConfig,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            // OAuth, balance, and provider APIs are intentionally direct.
            // This also prevents unsupported/broken SOCKS environment proxies
            // from turning every refresh into an immediate connect error.
            .no_proxy()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(config.upstream.timeout_secs))
            .build()
            .context("building account HTTP client")?;
        Ok(Self {
            inner: Arc::new(Inner {
                config,
                oauth,
                auth_path,
                http,
                store_lock: Mutex::new(()),
                account_locks: Mutex::new(HashMap::new()),
                refresh_flights: Mutex::new(HashMap::new()),
                last_used_clock: AtomicI64::new(Utc::now().timestamp_millis()),
            }),
        })
    }

    pub async fn pick_account(&self, excluded: &HashSet<String>) -> Result<SelectedAccount> {
        let name = {
            let _guard = self.inner.store_lock.lock().await;
            let mut auth = AuthStore::load(&self.inner.auth_path)?;
            let name = auth
                .accounts
                .iter()
                .filter(|(name, account)| account.usable() && !excluded.contains(*name))
                .min_by(|(left_name, left), (right_name, right)| {
                    (left.last_used, left_name.as_str())
                        .cmp(&(right.last_used, right_name.as_str()))
                })
                .map(|(name, _)| name.clone())
                .ok_or_else(|| anyhow!("no enabled authenticated accounts are available"))?;
            let timestamp = self.next_last_used();
            if let Some(account) = auth.accounts.get_mut(&name) {
                account.last_used = timestamp;
            }
            auth.save(&self.inner.auth_path)?;
            name
        };
        self.current_token_for(&name, false).await
    }

    pub async fn current_token_for(&self, name: &str, force: bool) -> Result<SelectedAccount> {
        self.current_token_for_observed(name, force, None).await
    }

    /// Force a refresh only if the rejected token is still current. Concurrent
    /// 401/403 responses therefore join the same refresh instead of rotating
    /// the refresh token repeatedly.
    pub async fn force_refresh_after_rejection(
        &self,
        name: &str,
        rejected_access_token: &str,
    ) -> Result<SelectedAccount> {
        self.current_token_for_observed(name, true, Some(rejected_access_token))
            .await
    }

    async fn current_token_for_observed(
        &self,
        name: &str,
        force: bool,
        observed_access_token: Option<&str>,
    ) -> Result<SelectedAccount> {
        let initial = self.load_account(name).await?;
        if !initial.usable() {
            bail!("account {name:?} is disabled or requires login");
        }
        if initial.reauth_required && initial.has_valid_access() && !force {
            return Ok(selected_account(name, initial.access_token, false));
        }
        if !force && !initial.needs_refresh(self.inner.config.runtime.refresh_skew_secs) {
            return Ok(selected_account(name, initial.access_token, false));
        }

        let observed_access_token = observed_access_token
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| initial.access_token.clone());
        let flight = self.refresh_flight(name).await;
        match flight.try_claim() {
            Ok(_leader) => {
                self.refresh_as_leader(name, force, &observed_access_token)
                    .await
            }
            Err(generation) => {
                tracing::info!(account = name, "waiting for in-flight token refresh");
                flight.wait(generation).await;
                self.result_after_shared_refresh(name, force, &observed_access_token)
                    .await
            }
        }
    }

    async fn refresh_as_leader(
        &self,
        name: &str,
        force: bool,
        observed_access_token: &str,
    ) -> Result<SelectedAccount> {
        let lock = self.account_lock(name).await;
        let _account_guard = lock.lock().await;
        let account = self.load_account(name).await?;
        if !account.usable() {
            bail!("account {name:?} is disabled or requires login");
        }
        if account.access_token != observed_access_token && account.has_valid_access() {
            return Ok(selected_account(name, account.access_token, true));
        }
        if account.reauth_required && account.has_valid_access() && !force {
            return Ok(selected_account(name, account.access_token, false));
        }
        if !force && !account.needs_refresh(self.inner.config.runtime.refresh_skew_secs) {
            return Ok(selected_account(name, account.access_token, false));
        }
        if account.refresh_token.is_empty() {
            self.mark_reauth_locked(name, "refresh token is unavailable")
                .await?;
            bail!("account {name:?} requires login; {}", reauth_command(name));
        }

        match refresh_access_token(&self.inner.http, &self.inner.oauth, &account.refresh_token)
            .await
        {
            Ok(response) => {
                let access_token = response.access_token.clone();
                self.persist_refreshed_locked(name, &account, response)
                    .await?;
                Ok(selected_account(name, access_token, true))
            }
            Err(error) => {
                let permanent = is_permanent_refresh_error(&error);
                self.record_refresh_error_locked(name, permanent, &error)
                    .await?;
                if permanent {
                    Err(error.context(format!(
                        "refreshing account {name:?}; {}",
                        reauth_command(name)
                    )))
                } else {
                    Err(error.context(format!("refreshing account {name:?}")))
                }
            }
        }
    }

    async fn result_after_shared_refresh(
        &self,
        name: &str,
        force: bool,
        observed_access_token: &str,
    ) -> Result<SelectedAccount> {
        let account = self.load_account(name).await?;
        if !account.usable() {
            bail!("account {name:?} requires login; {}", reauth_command(name));
        }
        if account.access_token != observed_access_token && account.has_valid_access() {
            return Ok(selected_account(name, account.access_token, true));
        }
        if !force && !account.needs_refresh(self.inner.config.runtime.refresh_skew_secs) {
            return Ok(selected_account(name, account.access_token, true));
        }
        let summary = account
            .last_error
            .as_ref()
            .map(|error| error.message.as_str())
            .unwrap_or("token refresh failed");
        bail!("{summary}")
    }

    #[cfg(test)]
    pub async fn force_refresh(&self, name: &str) -> Result<SelectedAccount> {
        self.current_token_for(name, true).await
    }

    async fn load_account(&self, name: &str) -> Result<AccountAuth> {
        let _guard = self.inner.store_lock.lock().await;
        AuthStore::load(&self.inner.auth_path)?
            .accounts
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("account {name:?} does not exist"))
    }

    pub async fn mark_unauthorized_if_current(
        &self,
        name: &str,
        rejected_access_token: &str,
    ) -> Result<()> {
        let lock = self.account_lock(name).await;
        let _account_guard = lock.lock().await;
        let _guard = self.inner.store_lock.lock().await;
        let mut auth = AuthStore::load(&self.inner.auth_path)?;
        if let Some(account) = auth.accounts.get_mut(name) {
            if account.access_token != rejected_access_token {
                return Ok(());
            }
            account.access_token.clear();
            account.token_expiry = 0;
            account.reauth_required = true;
            account.last_error = Some(AuthErrorRecord {
                at: Utc::now().timestamp(),
                message: "upstream rejected refreshed credentials".into(),
            });
        }
        auth.save(&self.inner.auth_path)
    }

    pub async fn store_login(&self, name: &str, response: &TokenResponse) -> Result<()> {
        let lock = self.account_lock(name).await;
        let _account_guard = lock.lock().await;
        let _guard = self.inner.store_lock.lock().await;
        let mut auth = AuthStore::load(&self.inner.auth_path)?;
        let previous_refresh = auth
            .accounts
            .get(name)
            .map(|account| account.refresh_token.as_str())
            .unwrap_or_default();
        let account = account_from_response(response, previous_refresh);
        auth.accounts.insert(name.into(), account);
        auth.save(&self.inner.auth_path)
    }

    pub async fn logout(&self, name: &str) -> Result<bool> {
        let lock = self.account_lock(name).await;
        let _account_guard = lock.lock().await;
        let _guard = self.inner.store_lock.lock().await;
        let mut auth = AuthStore::load(&self.inner.auth_path)?;
        let Some(account) = auth.accounts.get_mut(name) else {
            return Ok(false);
        };
        account.access_token.clear();
        account.refresh_token.clear();
        account.token_expiry = 0;
        account.enabled = false;
        account.reauth_required = false;
        account.last_error = None;
        auth.save(&self.inner.auth_path)?;
        Ok(true)
    }

    pub async fn snapshots(&self) -> Result<AuthStore> {
        let _guard = self.inner.store_lock.lock().await;
        AuthStore::load(&self.inner.auth_path)
    }

    pub async fn has_usable_account(&self) -> Result<bool> {
        Ok(self
            .snapshots()
            .await?
            .accounts
            .values()
            .any(AccountAuth::usable))
    }

    pub async fn background_loop(self) {
        let refresh = self.clone();
        tokio::join!(
            refresh.background_refresh_loop(),
            self.background_balance_loop()
        );
    }

    async fn background_refresh_loop(self) {
        let skew_secs = self.inner.config.runtime.refresh_skew_secs.max(1) as u64;
        let scan_secs = (skew_secs / 2).clamp(5, 60);
        let cadence = Duration::from_secs(scan_secs);
        tracing::info!(
            refresh_scan_secs = cadence.as_secs(),
            refresh_skew_secs = skew_secs,
            "background token refresh started"
        );
        loop {
            if let Err(error) = self.refresh_due_accounts().await {
                tracing::warn!(error = %safe_error(&error), "background refresh pass failed");
            }
            tokio::time::sleep(cadence).await;
        }
    }

    async fn background_balance_loop(self) {
        let cadence = Duration::from_secs(self.inner.config.runtime.balance_poll_secs);
        tracing::info!(
            balance_poll_secs = cadence.as_secs(),
            "background balance polling started"
        );
        loop {
            if let Err(error) = self.probe_all_balances().await {
                tracing::warn!(error = %safe_error(&error), "balance polling pass failed");
            }
            tokio::time::sleep(cadence).await;
        }
    }

    async fn refresh_due_accounts(&self) -> Result<()> {
        let accounts: Vec<(String, AccountAuth)> = self
            .snapshots()
            .await?
            .accounts
            .into_iter()
            .filter(|(_, account)| account.usable() && !account.reauth_required)
            .collect();
        for (name, account) in accounts {
            if account.needs_refresh(self.inner.config.runtime.refresh_skew_secs) {
                match self.current_token_for(&name, false).await {
                    Ok(selected) => {
                        tracing::info!(account = %name, refresh = selected.refreshed, "background refresh complete");
                    }
                    Err(error) => {
                        tracing::warn!(account = %name, error = %safe_error(&error), "background refresh failed");
                        continue;
                    }
                }
            }
        }
        Ok(())
    }

    async fn probe_all_balances(&self) -> Result<()> {
        let accounts: Vec<String> = self
            .snapshots()
            .await?
            .accounts
            .into_iter()
            .filter(|(_, account)| account.usable())
            .map(|(name, _)| name)
            .collect();
        for name in accounts {
            match self.probe_balance(&name).await {
                Ok(BalanceResult::FreeTier) => {
                    tracing::info!(account = %name, balance_result = "free_tier", "balance probe complete");
                }
                Ok(BalanceResult::Available(summary)) => {
                    tracing::info!(account = %name, balance_result = "available", balance = %summary, "balance probe complete");
                }
                Err(error) => {
                    tracing::warn!(account = %name, balance_result = "error", error = %safe_error(&error), "balance probe failed");
                }
            }
        }
        Ok(())
    }

    async fn probe_balance(&self, name: &str) -> Result<BalanceResult> {
        let started = std::time::Instant::now();
        let selected = self.current_token_for(name, false).await?;
        let url = format!(
            "{}{}/user/balance",
            self.inner.config.upstream.base_url.trim_end_matches('/'),
            self.inner.config.upstream.api_prefix
        );
        let mut request = self.inner.http.get(url).bearer_auth(selected.access_token);
        for (key, value) in &self.inner.config.upstream.extra_headers {
            if !is_reserved_upstream_header(key) {
                request = request.header(key, value);
            }
        }
        request = request.header("X-App-Id", crate::config::defaults::UPSTREAM_X_APP_ID);
        let response = request.send().await.context("requesting user balance")?;
        let status = response.status();
        tracing::info!(
            direction = "upstream",
            operation = "balance",
            account = name,
            path = "/api/v1/user/balance",
            status = status.as_u16(),
            duration_ms = started.elapsed().as_millis(),
            response_bytes = response.content_length().unwrap_or(0),
            "upstream response"
        );
        let body = response.text().await.unwrap_or_default();
        if status.as_u16() == 404 && body.contains("ACCOUNT_NOT_FOUND") {
            return Ok(BalanceResult::FreeTier);
        }
        if !status.is_success() {
            bail!("balance endpoint returned HTTP {status}");
        }
        let envelope: BalanceEnvelope =
            serde_json::from_str(&body).context("parsing balance response")?;
        if !envelope.code.is_null() && envelope.code.as_i64() != Some(0) {
            bail!("balance endpoint returned a business error");
        }
        Ok(BalanceResult::Available(balance_summary(&envelope.data)))
    }

    async fn persist_refreshed_locked(
        &self,
        name: &str,
        previous: &AccountAuth,
        response: TokenResponse,
    ) -> Result<()> {
        let _guard = self.inner.store_lock.lock().await;
        let mut auth = AuthStore::load(&self.inner.auth_path)?;
        let current = auth
            .accounts
            .get_mut(name)
            .ok_or_else(|| anyhow!("account disappeared during refresh"))?;
        current.access_token = response.access_token;
        current.refresh_token = response
            .refresh_token
            .unwrap_or_else(|| previous.refresh_token.clone());
        current.token_expiry = expires_at(response.expires_in);
        current.token_type = response.token_type.unwrap_or_else(|| "Bearer".into());
        current.scope = response.scope.unwrap_or_else(|| previous.scope.clone());
        current.enabled = true;
        current.reauth_required = false;
        current.last_error = None;
        auth.save(&self.inner.auth_path)
    }

    async fn record_refresh_error_locked(
        &self,
        name: &str,
        permanent: bool,
        error: &anyhow::Error,
    ) -> Result<()> {
        let _guard = self.inner.store_lock.lock().await;
        let mut auth = AuthStore::load(&self.inner.auth_path)?;
        if let Some(account) = auth.accounts.get_mut(name) {
            account.reauth_required |= permanent;
            let mut message = safe_error(error);
            if permanent {
                message.push_str("; ");
                message.push_str(&reauth_command(name));
            }
            account.last_error = Some(AuthErrorRecord {
                at: Utc::now().timestamp(),
                message,
            });
        }
        auth.save(&self.inner.auth_path)
    }

    async fn mark_reauth_locked(&self, name: &str, message: &str) -> Result<()> {
        let _guard = self.inner.store_lock.lock().await;
        let mut auth = AuthStore::load(&self.inner.auth_path)?;
        if let Some(account) = auth.accounts.get_mut(name) {
            account.reauth_required = true;
            account.last_error = Some(AuthErrorRecord {
                at: Utc::now().timestamp(),
                message: message.into(),
            });
        }
        auth.save(&self.inner.auth_path)
    }

    async fn account_lock(&self, name: &str) -> Arc<Mutex<()>> {
        let mut locks = self.inner.account_locks.lock().await;
        locks
            .entry(name.into())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    async fn refresh_flight(&self, name: &str) -> Arc<RefreshFlight> {
        let mut flights = self.inner.refresh_flights.lock().await;
        flights
            .entry(name.into())
            .or_insert_with(|| Arc::new(RefreshFlight::default()))
            .clone()
    }

    fn next_last_used(&self) -> i64 {
        let now = Utc::now().timestamp_millis();
        let mut previous = self.inner.last_used_clock.load(Ordering::Relaxed);
        loop {
            let next = now.max(previous.saturating_add(1));
            match self.inner.last_used_clock.compare_exchange_weak(
                previous,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return next,
                Err(actual) => previous = actual,
            }
        }
    }
}

fn selected_account(name: &str, access_token: String, refreshed: bool) -> SelectedAccount {
    SelectedAccount {
        name: name.into(),
        access_token,
        refreshed,
    }
}

fn reauth_command(name: &str) -> String {
    let safe_name: String = name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
        .take(64)
        .collect();
    format!(
        "run 'mogick-provider login --account {} --force' to re-authenticate",
        if safe_name.is_empty() {
            "default"
        } else {
            &safe_name
        }
    )
}

enum BalanceResult {
    FreeTier,
    Available(String),
}

fn account_from_response(response: &TokenResponse, previous_refresh: &str) -> AccountAuth {
    AccountAuth {
        access_token: response.access_token.clone(),
        refresh_token: response
            .refresh_token
            .clone()
            .unwrap_or_else(|| previous_refresh.into()),
        token_expiry: expires_at(response.expires_in),
        token_type: response
            .token_type
            .clone()
            .unwrap_or_else(|| "Bearer".into()),
        scope: response.scope.clone().unwrap_or_default(),
        enabled: true,
        last_used: 0,
        reauth_required: false,
        last_error: None,
    }
}

fn expires_at(expires_in: Option<i64>) -> i64 {
    Utc::now().timestamp() + expires_in.unwrap_or(3600).max(1)
}

fn safe_error(error: &anyhow::Error) -> String {
    let message = error
        .chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    if let Some(reason) = [
        "invalid_grant",
        "invalid_token",
        "expired_token",
        "refresh_token_expired",
        "access_denied",
        "not_supported",
        "unauthorized_client",
    ]
    .into_iter()
    .find(|reason| message.contains(reason))
    {
        format!("{reason}; login required")
    } else if let Some(status) = [400, 401, 403, 404, 429, 500, 502, 503]
        .into_iter()
        .find(|status| message.contains(&format!("http {status}")))
    {
        format!("token refresh rejected (HTTP {status})")
    } else if let Some(code) = message
        .split_whitespace()
        .find(|part| part.starts_with("token_business_code_") || part.starts_with("token_error_"))
    {
        format!(
            "token refresh rejected ({})",
            code.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        )
    } else if message.contains("timeout") {
        "request timed out".into()
    } else if message.contains("builder") {
        "OAuth request could not be built".into()
    } else if message.contains("connect") {
        "connection failed".into()
    } else if message.contains("refresh") {
        "token refresh failed".into()
    } else {
        "upstream operation failed".into()
    }
}

fn balance_summary(value: &serde_json::Value) -> String {
    let Some(object) = value.as_object() else {
        return "available".into();
    };
    ["balance", "total_balance", "free_balance", "plan_balance"]
        .into_iter()
        .filter_map(|key| object.get(key).map(|value| format!("{key}={value}")))
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_reserved_upstream_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "authorization" | "cookie" | "host" | "content-length" | "x-app-id"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::routing::post;
    use axum::{Json, Router};
    use tokio::sync::Barrier;

    fn auth_path() -> PathBuf {
        std::env::temp_dir().join(format!("mogick-auth-{}.json", uuid::Uuid::new_v4()))
    }

    fn fresh_account(token: &str, last_used: i64) -> AccountAuth {
        AccountAuth {
            access_token: token.into(),
            refresh_token: "refresh".into(),
            token_expiry: Utc::now().timestamp() + 3600,
            token_type: "Bearer".into(),
            enabled: true,
            last_used,
            ..AccountAuth::default()
        }
    }

    #[tokio::test]
    async fn fair_selection_and_logout_skip() {
        let path = auth_path();
        let mut auth = AuthStore::default();
        auth.accounts.insert("alice".into(), fresh_account("a", 0));
        auth.accounts.insert("bob".into(), fresh_account("b", 0));
        auth.save(&path).unwrap();
        let manager = AccountManager::new(Config::default(), path.clone()).unwrap();
        let excluded = HashSet::new();
        assert_eq!(manager.pick_account(&excluded).await.unwrap().name, "alice");
        assert_eq!(manager.pick_account(&excluded).await.unwrap().name, "bob");
        assert_eq!(manager.pick_account(&excluded).await.unwrap().name, "alice");
        assert!(manager.logout("alice").await.unwrap());
        assert_eq!(manager.pick_account(&excluded).await.unwrap().name, "bob");
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn late_unauthorized_cannot_clear_a_newer_token() {
        let path = auth_path();
        let mut auth = AuthStore::default();
        auth.accounts
            .insert("alice".into(), fresh_account("new-access", 0));
        auth.save(&path).unwrap();
        let manager = AccountManager::new(Config::default(), path.clone()).unwrap();
        manager
            .mark_unauthorized_if_current("alice", "old-rejected-access")
            .await
            .unwrap();
        let stored = AuthStore::load(&path).unwrap();
        assert_eq!(stored.accounts["alice"].access_token, "new-access");
        assert!(!stored.accounts["alice"].reauth_required);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn concurrent_requests_share_one_refresh_and_persist_rotation() {
        let calls = Arc::new(AtomicUsize::new(0));
        async fn refresh_handler(
            State(calls): State<Arc<AtomicUsize>>,
        ) -> (StatusCode, Json<serde_json::Value>) {
            calls.fetch_add(1, Ordering::SeqCst);
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "access_token":"new-access",
                    "refresh_token":"rotated-refresh",
                    "expires_in":3600
                })),
            )
        }
        let app = Router::new()
            .route("/token", post(refresh_handler))
            .with_state(calls.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let path = auth_path();
        let mut auth = AuthStore::default();
        auth.accounts.insert(
            "alice".into(),
            AccountAuth {
                access_token: "expired".into(),
                refresh_token: "old-refresh".into(),
                token_expiry: 1,
                enabled: true,
                ..AccountAuth::default()
            },
        );
        auth.save(&path).unwrap();
        let oauth = OAuthConfig {
            client_id: "mogick".into(),
            device_authorization_endpoint: format!("http://{address}/device"),
            token_endpoint: format!("http://{address}/token"),
            scope: "openid profile email".into(),
        };
        let manager =
            AccountManager::new_with_oauth(Config::default(), path.clone(), oauth).unwrap();
        let tasks: Vec<_> = (0..12)
            .map(|_| {
                let manager = manager.clone();
                tokio::spawn(
                    async move { manager.current_token_for("alice", false).await.unwrap() },
                )
            })
            .collect();
        for task in tasks {
            assert_eq!(task.await.unwrap().access_token, "new-access");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let stored = AuthStore::load(&path).unwrap();
        assert_eq!(stored.accounts["alice"].refresh_token, "rotated-refresh");
        server.abort();
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn concurrent_unauthorized_requests_join_one_forced_refresh() {
        let calls = Arc::new(AtomicUsize::new(0));
        async fn refresh_handler(
            State(calls): State<Arc<AtomicUsize>>,
        ) -> (StatusCode, Json<serde_json::Value>) {
            calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(25)).await;
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "code": 0,
                    "data": {
                        "access_token":"new-access",
                        "refresh_token":"rotated-refresh",
                        "expires_in":3600
                    }
                })),
            )
        }
        let app = Router::new()
            .route("/token", post(refresh_handler))
            .with_state(calls.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let path = auth_path();
        let mut auth = AuthStore::default();
        auth.accounts.insert(
            "alice".into(),
            AccountAuth {
                access_token: "rejected-access".into(),
                refresh_token: "old-refresh".into(),
                token_expiry: Utc::now().timestamp() + 3600,
                enabled: true,
                ..AccountAuth::default()
            },
        );
        auth.save(&path).unwrap();
        let oauth = OAuthConfig {
            client_id: "mogick".into(),
            device_authorization_endpoint: format!("http://{address}/device"),
            token_endpoint: format!("http://{address}/token"),
            scope: "openid profile email".into(),
        };
        let manager =
            AccountManager::new_with_oauth(Config::default(), path.clone(), oauth).unwrap();
        let barrier = Arc::new(Barrier::new(12));
        let tasks: Vec<_> = (0..12)
            .map(|_| {
                let manager = manager.clone();
                let barrier = barrier.clone();
                tokio::spawn(async move {
                    barrier.wait().await;
                    manager
                        .force_refresh_after_rejection("alice", "rejected-access")
                        .await
                        .unwrap()
                })
            })
            .collect();
        for task in tasks {
            assert_eq!(task.await.unwrap().access_token, "new-access");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            AuthStore::load(&path).unwrap().accounts["alice"].refresh_token,
            "rotated-refresh"
        );
        server.abort();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn stored_error_is_redacted() {
        let error = anyhow!("Bearer abc.def.ghi invalid_grant refresh_token=secret");
        assert_eq!(safe_error(&error), "invalid_grant; login required");
    }
}
