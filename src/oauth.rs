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
}

pub fn parse_token_response(body: &str) -> Result<TokenResponse> {
    if let Ok(token) = serde_json::from_str::<TokenResponse>(body) {
        return validate_token_response(token);
    }
    if let Ok(wrapped) = serde_json::from_str::<WrappedTokenResponse>(body) {
        if matches!(wrapped.code, 0 | 200) {
            return validate_token_response(wrapped.data);
        }
        bail!("token_business_code_{}", wrapped.code);
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(error_code) = value
            .get("data")
            .and_then(|data| data.get("errorCode"))
            .and_then(serde_json::Value::as_str)
        {
            bail!("token_error_{}", safe_oauth_code(error_code));
        }
        if let Some(code) = value.get("code").and_then(serde_json::Value::as_i64) {
            bail!("token_business_code_{code}");
        }
    }
    Err(anyhow!("token endpoint returned an unrecognized response"))
}

fn validate_token_response(token: TokenResponse) -> Result<TokenResponse> {
    if token.access_token.trim().is_empty() {
        bail!("token endpoint returned an empty access token");
    }
    Ok(token)
}

pub async fn request_device_code(
    http: &reqwest::Client,
    config: &OAuthConfig,
) -> Result<DeviceCodeResponse> {
    let response = http
        .post(&config.device_authorization_endpoint)
        .header(reqwest::header::USER_AGENT, &config.user_agent)
        .form(&[
            ("client_id", config.client_id.as_str()),
            ("scope", config.scope.as_str()),
        ])
        .send()
        .await
        .context("requesting device code")?;
    let status = response.status();
    tracing::info!(
        oauth_operation = "device_code",
        status = status.as_u16(),
        "OAuth response"
    );
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
            .header(reqwest::header::USER_AGENT, &config.user_agent)
            .header(reqwest::header::ACCEPT, "application/json")
            .header("X-App-Id", crate::config::defaults::UPSTREAM_X_APP_ID)
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
            let parsed = parse_token_response(&body);
            tracing::info!(
                oauth_operation = "device_poll",
                status = status.as_u16(),
                result = if parsed.is_ok() {
                    "token"
                } else {
                    "invalid_payload"
                },
                "OAuth response"
            );
            return parsed;
        }
        let error = serde_json::from_str::<OAuthError>(&body).unwrap_or(OAuthError {
            error: format!("http_{}", status.as_u16()),
        });
        tracing::info!(
            oauth_operation = "device_poll",
            status = status.as_u16(),
            oauth_error = %safe_oauth_code(&error.error),
            "OAuth response"
        );
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
    let started = std::time::Instant::now();
    let response = http
        .post(&config.token_endpoint)
        .header(reqwest::header::USER_AGENT, &config.user_agent)
        .header(reqwest::header::ACCEPT, "application/json")
        .header("X-App-Id", crate::config::defaults::UPSTREAM_X_APP_ID)
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", config.client_id.as_str()),
            ("scope", config.scope.as_str()),
        ])
        .send()
        .await;
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            tracing::warn!(
                oauth_operation = "refresh",
                result = "transport_error",
                oauth_error = classify_transport_error(&error),
                duration_ms = started.elapsed().as_millis(),
                "OAuth request failed"
            );
            return Err(error).context("requesting token refresh");
        }
    };
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if status.is_success() {
        let parsed = parse_token_response(&body);
        match &parsed {
            Ok(_) => tracing::info!(
                oauth_operation = "refresh",
                status = status.as_u16(),
                result = "token",
                duration_ms = started.elapsed().as_millis(),
                "OAuth response"
            ),
            Err(error) => tracing::warn!(
                oauth_operation = "refresh",
                status = status.as_u16(),
                result = "invalid_payload",
                oauth_error = %safe_parse_error(error),
                duration_ms = started.elapsed().as_millis(),
                "OAuth response"
            ),
        }
        return parsed;
    }
    let summary = oauth_error_summary(&body);
    tracing::warn!(
        oauth_operation = "refresh",
        status = status.as_u16(),
        oauth_error = %summary,
        duration_ms = started.elapsed().as_millis(),
        "OAuth response"
    );
    bail!("refresh_failed:{summary} (HTTP {status})")
}

fn safe_parse_error(error: &anyhow::Error) -> String {
    error
        .to_string()
        .split_whitespace()
        .find(|part| part.starts_with("token_business_code_") || part.starts_with("token_error_"))
        .map(|part| {
            part.trim_matches(|character: char| {
                !character.is_ascii_alphanumeric() && character != '_'
            })
            .to_string()
        })
        .unwrap_or_else(|| "unrecognized_token_payload".into())
}

fn classify_transport_error(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect_error"
    } else if error.is_builder() {
        "request_builder_error"
    } else if error.is_request() {
        "request_error"
    } else if error.is_decode() {
        "decode_error"
    } else {
        "transport_error"
    }
}

fn oauth_error_summary(body: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return "request_rejected".into();
    };
    let standard = value
        .get("error")
        .and_then(|error| {
            error
                .as_str()
                .or_else(|| error.get("code").and_then(serde_json::Value::as_str))
        })
        .or_else(|| {
            value
                .get("data")
                .and_then(|data| data.get("error"))
                .and_then(serde_json::Value::as_str)
        });
    if let Some(code) = standard {
        let code = safe_oauth_code(code);
        return if code.is_empty() {
            "request_rejected".into()
        } else {
            code
        };
    }
    if let Some(code) = value.get("code") {
        let code = code
            .as_str()
            .map(safe_oauth_code)
            .or_else(|| code.as_i64().map(|code| code.to_string()))
            .unwrap_or_else(|| "unknown".into());
        return format!("business_code_{code}");
    }
    let message = value
        .get("message")
        .or_else(|| value.get("msg"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    if message.contains("token refresh is not supported")
        || message.contains("refresh token is not supported")
        || message.contains("keystone_iam token refresh")
    {
        return "refresh_not_supported".into();
    }
    for marker in [
        "invalid_grant",
        "invalid_token",
        "expired_token",
        "refresh_token_expired",
        "unauthorized_client",
        "not_supported",
        "refresh_not_supported",
    ] {
        if message.contains(marker) {
            return marker.into();
        }
    }
    "request_rejected".into()
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
        "expired_token",
        "refresh_token_expired",
        "access_denied",
        "not_supported",
        "unauthorized_client",
        "token_business_code_400",
        "token_business_code_401",
        "token_business_code_403",
        "token_error_invalid_request",
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
    use axum::http::{header, HeaderMap, StatusCode};
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
        headers: HeaderMap,
        Form(form): Form<HashMap<String, String>>,
    ) -> impl IntoResponse {
        assert_eq!(
            headers
                .get(header::ACCEPT)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
        assert_eq!(
            headers
                .get("x-app-id")
                .and_then(|value| value.to_str().ok()),
            Some("mogick")
        );
        assert!(headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("application/x-www-form-urlencoded")));
        if form.get("grant_type").map(String::as_str)
            == Some("urn:ietf:params:oauth:grant-type:device_code")
        {
            assert_eq!(form.get("client_id").map(String::as_str), Some("mogick"));
        } else {
            assert_eq!(
                form.get("grant_type").map(String::as_str),
                Some("refresh_token")
            );
            assert_eq!(form.get("client_id").map(String::as_str), Some("mogick"));
            assert_eq!(
                form.get("scope").map(String::as_str),
                Some("openid profile email")
            );
        }
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
                user_agent: "Go-http-client/2.0".into(),
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
        let wrapped_http_code = parse_token_response(
            r#"{"code":200,"data":{"access_token":"c","refresh_token":"rrr"}}"#,
        )
        .unwrap();
        assert_eq!(wrapped_http_code.access_token, "c");
        assert!(parse_token_response(r#"{"code":123,"data":{}}"#).is_err());
        assert!(parse_token_response(r#"{"access_token":""}"#).is_err());
        let invalid_refresh = parse_token_response(
            r#"{"code":400,"data":{"errorCode":"invalid_request"},"message":"redacted"}"#,
        )
        .unwrap_err();
        assert_eq!(invalid_refresh.to_string(), "token_error_invalid_request");
        assert!(is_permanent_refresh_error(&invalid_refresh));
    }

    #[test]
    fn oauth_error_summary_never_echoes_description() {
        let summary = oauth_error_summary(
            r#"{"error":"invalid_grant","error_description":"Bearer secret.jwt.value"}"#,
        );
        assert_eq!(summary, "invalid_grant");
        assert!(!summary.contains("secret"));
    }

    #[test]
    fn keystone_refresh_limitation_is_permanent_and_redacted() {
        let summary = oauth_error_summary(
            r#"{"message":"keystone_iam token refresh is not supported, Bearer do-not-print"}"#,
        );
        assert_eq!(summary, "refresh_not_supported");
        let error = anyhow!(summary);
        assert!(is_permanent_refresh_error(&error));
        assert!(!error.to_string().contains("do-not-print"));
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
