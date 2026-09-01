//! RFC 8628 device authorization and refresh client.

use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Deserialize)]
struct WrappedTokenResponse {
    code: i64,
    data: TokenResponse,
}

#[derive(Debug, Deserialize)]
struct OAuthError {
    #[serde(default)]
    error: String,
    #[serde(default)]
    error_description: String,
}

pub fn parse_token_response(body: &str) -> Result<TokenResponse> {
    if let Ok(token) = serde_json::from_str::<TokenResponse>(body) {
        return Ok(token);
    }
    if let Ok(wrapped) = serde_json::from_str::<WrappedTokenResponse>(body) {
        if wrapped.code == 0 {
            return Ok(wrapped.data);
        }
        bail!(
            "token endpoint returned business error code {}",
            wrapped.code
        );
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(code) = value.get("code").and_then(serde_json::Value::as_i64) {
            bail!("token endpoint returned business error code {code}");
        }
    }
    Err(anyhow!("token endpoint returned an unrecognized response"))
}

pub async fn request_device_code(
    http: &reqwest::Client,
    config: &OAuthConfig,
) -> Result<DeviceCodeResponse> {
    let response = http
        .post(&config.device_authorization_endpoint)
        .form(&[
            ("client_id", config.client_id.as_str()),
            ("scope", config.scope.as_str()),
        ])
        .send()
        .await
        .context("requesting device code")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!(
            "device code endpoint returned HTTP {status}: {}",
            oauth_error_summary(&body)
        );
    }
    serde_json::from_str(&body).context("parsing device code response")
}

pub async fn poll_for_token(
    http: &reqwest::Client,
    config: &OAuthConfig,
    device: &DeviceCodeResponse,
) -> Result<TokenResponse> {
    let deadline = std::time::Instant::now() + Duration::from_secs(device.expires_in.max(1) as u64);
    let mut interval = device.interval.max(1) as u64;

    loop {
        if std::time::Instant::now() >= deadline {
            bail!("device code expired before authorization completed");
        }
        poll_sleep(interval).await;
        let response = http
            .post(&config.token_endpoint)
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("device_code", device.device_code.as_str()),
                ("client_id", config.client_id.as_str()),
            ])
            .send()
            .await
            .context("polling token endpoint")?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if status.is_success() {
            return parse_token_response(&body);
        }
        let error = serde_json::from_str::<OAuthError>(&body).unwrap_or(OAuthError {
            error: format!("http_{}", status.as_u16()),
            error_description: String::new(),
        });
        match error.error.as_str() {
            "authorization_pending" => continue,
            "slow_down" => {
                interval = interval.saturating_add(5);
            }
            "expired_token" => bail!("device code expired during polling"),
            "access_denied" => bail!("device authorization was denied"),
            other => bail!("device token endpoint error: {}", safe_oauth_code(other)),
        }
    }
}

async fn poll_sleep(interval_secs: u64) {
    #[cfg(not(test))]
    tokio::time::sleep(Duration::from_secs(interval_secs)).await;
    #[cfg(test)]
    tokio::time::sleep(Duration::from_millis(interval_secs.max(1))).await;
}

pub async fn refresh_access_token(
    http: &reqwest::Client,
    config: &OAuthConfig,
    refresh_token: &str,
) -> Result<TokenResponse> {
    let response = http
        .post(&config.token_endpoint)
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", config.client_id.as_str()),
            ("scope", config.scope.as_str()),
        ])
        .send()
        .await
        .context("requesting token refresh")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if status.is_success() {
        return parse_token_response(&body);
    }
    let summary = oauth_error_summary(&body);
    bail!("refresh_failed:{summary} (HTTP {status})")
}

fn oauth_error_summary(body: &str) -> String {
    serde_json::from_str::<OAuthError>(body)
        .ok()
        .map(|error| {
            let code = safe_oauth_code(&error.error);
            if error.error_description.is_empty() {
                code
            } else {
                format!("{code}: request rejected")
            }
        })
        .unwrap_or_else(|| "request rejected".into())
}

fn safe_oauth_code(code: &str) -> String {
    code.chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        .take(64)
        .collect()
}

pub fn is_permanent_refresh_error(error: &anyhow::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    [
        "invalid_grant",
        "invalid_token",
        "access_denied",
        "not_supported",
        "unauthorized_client",
    ]
    .iter()
    .any(|marker| message.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use axum::extract::{Form, State};
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::post;
    use axum::{Json, Router};
    use tokio::sync::Mutex;

    #[derive(Clone)]
    struct MockState {
        responses: Arc<Mutex<VecDeque<(StatusCode, serde_json::Value)>>>,
        calls: Arc<AtomicUsize>,
    }

    async fn device_handler() -> impl IntoResponse {
        Json(serde_json::json!({
            "device_code":"device-secret",
            "user_code":"ABCD-EFGH",
            "verification_uri":"https://example.test/device",
            "expires_in":30,
            "interval":1
        }))
    }

    async fn token_handler(
        State(state): State<MockState>,
        Form(form): Form<HashMap<String, String>>,
    ) -> impl IntoResponse {
        assert_eq!(form.get("client_id").map(String::as_str), Some("mogick"));
        state.calls.fetch_add(1, Ordering::SeqCst);
        let (status, body) = state.responses.lock().await.pop_front().unwrap();
        (status, Json(body))
    }

    async fn mock_server(
        responses: Vec<(StatusCode, serde_json::Value)>,
    ) -> (OAuthConfig, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let state = MockState {
            responses: Arc::new(Mutex::new(responses.into())),
            calls: calls.clone(),
        };
        let app = Router::new()
            .route("/device", post(device_handler))
            .route("/token", post(token_handler))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (
            OAuthConfig {
                client_id: "mogick".into(),
                device_authorization_endpoint: format!("http://{address}/device"),
                token_endpoint: format!("http://{address}/token"),
                scope: "openid profile email".into(),
            },
            calls,
            task,
        )
    }

    #[test]
    fn parses_flat_and_wrapped_token_responses() {
        let flat =
            parse_token_response(r#"{"access_token":"a","refresh_token":"r","expires_in":10}"#)
                .unwrap();
        assert_eq!(flat.access_token, "a");
        let wrapped =
            parse_token_response(r#"{"code":0,"data":{"access_token":"b","refresh_token":"rr"}}"#)
                .unwrap();
        assert_eq!(wrapped.access_token, "b");
        assert!(parse_token_response(r#"{"code":123,"data":{}}"#).is_err());
    }

    #[test]
    fn oauth_error_summary_never_echoes_description() {
        let summary = oauth_error_summary(
            r#"{"error":"invalid_grant","error_description":"Bearer secret.jwt.value"}"#,
        );
        assert_eq!(summary, "invalid_grant: request rejected");
        assert!(!summary.contains("secret"));
    }

    #[tokio::test]
    async fn device_poll_handles_pending_slow_down_and_wrapped_success() {
        let (config, calls, task) = mock_server(vec![
            (
                StatusCode::BAD_REQUEST,
                serde_json::json!({"error":"authorization_pending"}),
            ),
            (
                StatusCode::BAD_REQUEST,
                serde_json::json!({"error":"slow_down"}),
            ),
            (
                StatusCode::OK,
                serde_json::json!({"code":0,"data":{
                    "access_token":"access", "refresh_token":"refresh", "expires_in":3600
                }}),
            ),
        ])
        .await;
        let http = reqwest::Client::new();
        let device = request_device_code(&http, &config).await.unwrap();
        let token = poll_for_token(&http, &config, &device).await.unwrap();
        assert_eq!(token.access_token, "access");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        task.abort();
    }

    #[tokio::test]
    async fn device_poll_reports_denial_without_echoing_body() {
        let (config, _, task) = mock_server(vec![(
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error":"access_denied",
                "error_description":"Bearer do-not-print"
            }),
        )])
        .await;
        let device = DeviceCodeResponse {
            device_code: "device".into(),
            user_code: "CODE".into(),
            verification_uri: "https://example.test".into(),
            verification_uri_complete: None,
            expires_in: 30,
            interval: 1,
        };
        let error = poll_for_token(&reqwest::Client::new(), &config, &device)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("denied"));
        assert!(!error.contains("do-not-print"));
        task.abort();
    }

    #[tokio::test]
    async fn refresh_accepts_rotated_refresh_token() {
        let (config, calls, task) = mock_server(vec![(
            StatusCode::OK,
            serde_json::json!({
                "access_token":"new-access", "refresh_token":"new-refresh", "expires_in":120
            }),
        )])
        .await;
        let token = refresh_access_token(&reqwest::Client::new(), &config, "old-refresh")
            .await
            .unwrap();
        assert_eq!(token.refresh_token.as_deref(), Some("new-refresh"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        task.abort();
    }
}
