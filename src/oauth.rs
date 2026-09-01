//! OAuth 2.0 Device Authorization Grant (RFC 8628) client.
//!
//! Implements the same flow Mogick uses internally
//! (`keystone_iam: requesting device code` -> poll -> exchange tokens).
//!
//! Public entry points:
//! - [`request_device_code`] — get device_code + user_code + verification URI.
//! - [`poll_for_token`] — long-poll the token endpoint until the user approves.
//! - [`refresh_access_token`] — exchange a refresh_token for a new access_token.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::OAuthConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    #[serde(default)]
    pub verification_uri_complete: Option<String>,
    pub expires_in: i64,
    #[serde(default = "default_interval")]
    pub interval: i64,
}

fn default_interval() -> i64 {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_in: Option<i64>,
    #[serde(default)]
    pub token_type: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
}

/// Some IdPs (notably the `keystone_iam` provider used by Mogick) wrap
/// their OAuth responses in `{ "code": 0, "data": { ... } }`. We try the
/// flat shape first, then fall back to the wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrappedTokenResponse {
    pub code: i32,
    pub data: TokenResponse,
}

/// Try to parse a token endpoint body. Handles both the standard
/// RFC 6749 flat response and the `{ code, data }` envelope used by
/// `tongyuan.cc/ai/mogick/oauth/provider/tongyuan.go`.
pub fn parse_token_response(body: &str) -> Result<TokenResponse> {
    // Try flat shape first (matches the `oauth.(*DefaultConverter).*` paths).
    if let Ok(t) = serde_json::from_str::<TokenResponse>(body) {
        return Ok(t);
    }
    // Fall back to keystone_iam's `{ code, data }` envelope.
    if let Ok(w) = serde_json::from_str::<WrappedTokenResponse>(body) {
        if w.code == 0 {
            return Ok(w.data);
        }
        return Err(anyhow!(
            "token endpoint returned error code {}",
            w.code
        ));
    }
    Err(anyhow!(
        "could not parse token response as either flat or wrapped OAuth shape: {}",
        body
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthError {
    #[serde(default)]
    pub error: String,
    #[serde(default)]
    pub error_description: Option<String>,
}

/// Request a device code from the IdP. The returned object carries the
/// user-facing URL + code that the user must visit / enter.
pub async fn request_device_code(
    http: &reqwest::Client,
    cfg: &OAuthConfig,
) -> Result<DeviceCodeResponse> {
    let mut form: Vec<(String, String)> = vec![("client_id".into(), cfg.client_id.clone())];
    if let Some(secret) = &cfg.client_secret {
        form.push(("client_secret".into(), secret.clone()));
    }
    if !cfg.scope.is_empty() {
        form.push(("scope".into(), cfg.scope.clone()));
    }

    let resp = http
        .post(&cfg.device_authorization_endpoint)
        .form(&form)
        .send()
        .await
        .context("requesting device code")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!(
            "device code endpoint returned {}: {}",
            status,
            body
        ));
    }
    serde_json::from_str(&body).context("parsing device code response")
}

/// Poll the token endpoint until the user either approves or the device code
/// expires. Returns the [`TokenResponse`] on success.
pub async fn poll_for_token(
    http: &reqwest::Client,
    cfg: &OAuthConfig,
    device: &DeviceCodeResponse,
) -> Result<TokenResponse> {
    let deadline = std::time::Instant::now()
        + Duration::from_secs(device.expires_in.max(1) as u64);
    let mut interval_secs = device.interval.max(1);

    loop {
        if std::time::Instant::now() >= deadline {
            return Err(anyhow!(
                "device code expired before user authorised the request"
            ));
        }

        tokio::time::sleep(Duration::from_secs(interval_secs as u64)).await;

        let mut form: Vec<(String, String)> = vec![
            ("grant_type".into(), "urn:ietf:params:oauth:grant-type:device_code".into()),
            ("device_code".into(), device.device_code.clone()),
            ("client_id".into(), cfg.client_id.clone()),
        ];
        if let Some(secret) = &cfg.client_secret {
            form.push(("client_secret".into(), secret.clone()));
        }

        let resp = http
            .post(&cfg.token_endpoint)
            .form(&form)
            .send()
            .await
            .context("polling token endpoint")?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();

        if status.is_success() {
            return parse_token_response(&body);
        }

        // Standard RFC 8628 error codes:
        //   authorization_pending — slow down, keep polling
        //   slow_down              — server wants longer interval
        //   expired_token          — device code expired, abort
        //   access_denied          — user denied, abort
        if let Ok(err) = serde_json::from_str::<OAuthError>(&body) {
            match err.error.as_str() {
                "authorization_pending" => continue,
                "slow_down" => {
                    interval_secs += 5;
                    continue;
                }
                "expired_token" => {
                    return Err(anyhow!("device code expired during polling"));
                }
                "access_denied" => {
                    return Err(anyhow!(
                        "user denied authorisation ({})",
                        err.error_description.unwrap_or_default()
                    ));
                }
                _ => {
                    return Err(anyhow!(
                        "token endpoint error {}: {}",
                        err.error,
                        err.error_description.unwrap_or_default()
                    ));
                }
            }
        }

        return Err(anyhow!(
            "token endpoint returned HTTP {}: {}",
            status,
            body
        ));
    }
}

/// Exchange a refresh_token for a new access_token.
pub async fn refresh_access_token(
    http: &reqwest::Client,
    cfg: &OAuthConfig,
    refresh_token: &str,
) -> Result<TokenResponse> {
    let mut form: Vec<(String, String)> = vec![
        ("grant_type".into(), "refresh_token".into()),
        ("refresh_token".into(), refresh_token.to_string()),
        ("client_id".into(), cfg.client_id.clone()),
    ];
    if let Some(secret) = &cfg.client_secret {
        form.push(("client_secret".into(), secret.clone()));
    }
    if !cfg.scope.is_empty() {
        form.push(("scope".into(), cfg.scope.clone()));
    }

    let resp = http
        .post(&cfg.token_endpoint)
        .form(&form)
        .send()
        .await
        .context("requesting refresh")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        if let Ok(err) = serde_json::from_str::<OAuthError>(&body) {
            return Err(anyhow!(
                "refresh failed: {} ({})",
                err.error,
                err.error_description.unwrap_or_default()
            ));
        }
        return Err(anyhow!(
            "refresh endpoint returned HTTP {}: {}",
            status,
            body
        ));
    }
    parse_token_response(&body)
}
