//! Axum gateway for OpenAI-compatible passthrough and Anthropic Messages v1.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::rejection::BytesRejection;
use axum::extract::{ConnectInfo, DefaultBodyLimit, OriginalUri, Path, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};

use crate::anthropic::{self, ProtocolError};
use crate::config::Config;
use crate::token::{AccountManager, SelectedAccount};

#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub accounts: AccountManager,
    http: reqwest::Client,
}

impl AppState {
    pub fn new(config: Config, accounts: AccountManager) -> Result<Self> {
        let http = reqwest::Client::builder()
            // The configured provider is reached directly. Incoming gateway
            // traffic is unaffected by this outbound proxy policy.
            .no_proxy()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(config.upstream.timeout_secs))
            .build()
            .context("building upstream HTTP client")?;
        Ok(Self {
            config,
            accounts,
            http,
        })
    }
}

struct UpstreamResult {
    response: reqwest::Response,
    account: String,
    refreshed: bool,
    failover_count: usize,
}

pub fn router(state: AppState) -> Router {
    let max_request_bytes = state.config.runtime.max_request_bytes;
    Router::new()
        .route("/", get(healthz))
        .route("/healthz", get(healthz))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/models", any(models))
        .route("/v1/*rest", any(openai_passthrough))
        .route("/v1", any(openai_root))
        .route("/chat/completions", post(legacy_chat))
        .layer(DefaultBodyLimit::max(max_request_bytes))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(middleware::from_fn(response_log_middleware))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn openai_passthrough(
    State(state): State<AppState>,
    Path(rest): Path<String>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let body = match body_or_error(body, false, &request_id(&headers)) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let path = upstream_path(&state.config, &rest, uri.query());
    openai_forward(state, method, path, headers, body).await
}

async fn openai_root(
    State(state): State<AppState>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let body = match body_or_error(body, false, &request_id(&headers)) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let path = append_query(
        format!("{}/", state.config.upstream.api_prefix),
        uri.query(),
    );
    openai_forward(state, method, path, headers, body).await
}

async fn legacy_chat(
    State(state): State<AppState>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let body = match body_or_error(body, false, &request_id(&headers)) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let path = append_query(
        format!("{}/chat/completions", state.config.upstream.api_prefix),
        uri.query(),
    );
    openai_forward(state, method, path, headers, body).await
}

async fn models(
    State(state): State<AppState>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let is_anthropic = headers.contains_key("anthropic-version");
    let body = match body_or_error(body, is_anthropic, &request_id(&headers)) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let request_bytes = body.len();
    if !headers.contains_key("anthropic-version") {
        let path = append_query(
            format!("{}/models", state.config.upstream.api_prefix),
            uri.query(),
        );
        return openai_forward(state, method, path, headers, body).await;
    }
    let request_id = request_id(&headers);
    let start = Instant::now();
    let path = append_query(
        format!("{}/models", state.config.upstream.api_prefix),
        uri.query(),
    );
    let result = match send_with_failover(&state, method, &path, &headers, body).await {
        Ok(result) => result,
        Err(error) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                public_error(&error),
                &request_id,
            );
        }
    };
    let status = result.response.status();
    let response_bytes = result.response.content_length().unwrap_or(0);
    if !status.is_success() {
        return anthropic_upstream_error(result.response, &request_id).await;
    }
    let value: Value = match result.response.json().await {
        Ok(value) => value,
        Err(_) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "upstream returned invalid model data",
                &request_id,
            );
        }
    };
    tracing::info!(
        request_id,
        protocol = "anthropic",
        path = "/v1/models",
        account = %result.account,
        status = 200,
        duration_ms = start.elapsed().as_millis(),
        request_bytes,
        response_bytes,
        refresh = result.refreshed,
        failover = result.failover_count,
        "request complete"
    );
    json_response(
        StatusCode::OK,
        anthropic::convert_models(&value),
        &request_id,
    )
}

async fn openai_forward(
    state: AppState,
    method: Method,
    path: String,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let started = Instant::now();
    let request_id = request_id(&headers);
    let request_bytes = body.len();
    let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let model = parsed.get("model").and_then(Value::as_str).unwrap_or("");
    let stream = parsed
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let result = match send_with_failover(&state, method, &path, &headers, body).await {
        Ok(result) => result,
        Err(error) => {
            tracing::warn!(
                request_id,
                protocol = "openai",
                model,
                path,
                stream,
                status = 502,
                duration_ms = started.elapsed().as_millis(),
                request_bytes,
                error = %public_error(&error),
                "request failed"
            );
            return openai_error(StatusCode::BAD_GATEWAY, public_error(&error), &request_id);
        }
    };
    let status = result.response.status();
    let response_bytes = result.response.content_length().unwrap_or(0);
    tracing::info!(
        request_id,
        protocol = "openai",
        model,
        account = %result.account,
        path,
        stream,
        status = status.as_u16(),
        duration_ms = started.elapsed().as_millis(),
        request_bytes,
        response_bytes,
        refresh = result.refreshed,
        failover = result.failover_count,
        "request complete"
    );
    upstream_openai_response(result.response, &request_id).await
}

async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let started = Instant::now();
    let request_id = request_id(&headers);
    let body = match body_or_error(body, true, &request_id) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let request_bytes = body.len();
    let converted = match anthropic::convert_request(&body) {
        Ok(converted) => converted,
        Err(error) => return protocol_error(error, &request_id),
    };
    let upstream_body = match serde_json::to_vec(&converted.body) {
        Ok(body) => Bytes::from(body),
        Err(_) => {
            return anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "failed to serialize converted request",
                &request_id,
            );
        }
    };
    let path = format!("{}/chat/completions", state.config.upstream.api_prefix);
    let result =
        match send_with_failover(&state, Method::POST, &path, &headers, upstream_body).await {
            Ok(result) => result,
            Err(error) => {
                return anthropic_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    public_error(&error),
                    &request_id,
                );
            }
        };
    let status = result.response.status();
    let response_bytes = result.response.content_length().unwrap_or(0);
    tracing::info!(
        request_id,
        protocol = "anthropic",
        model = %converted.model,
        account = %result.account,
        path = "/v1/messages",
        stream = converted.stream,
        status = status.as_u16(),
        duration_ms = started.elapsed().as_millis(),
        request_bytes,
        response_bytes,
        refresh = result.refreshed,
        failover = result.failover_count,
        "request complete"
    );
    if !status.is_success() {
        return anthropic_upstream_error(result.response, &request_id).await;
    }
    if converted.stream {
        let mut response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("x-request-id", &request_id)
            .header("x-accel-buffering", "no")
            .body(anthropic::stream_body(
                result.response,
                request_id.clone(),
                converted.model,
            ))
            .unwrap_or_else(|_| Response::new(Body::empty()));
        insert_request_id(response.headers_mut(), &request_id);
        return response;
    }
    let value: Value = match result.response.json().await {
        Ok(value) => value,
        Err(_) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "upstream returned invalid JSON",
                &request_id,
            );
        }
    };
    match anthropic::convert_response(&value, &request_id, &converted.model) {
        Ok(value) => json_response(StatusCode::OK, value, &request_id),
        Err(_) => anthropic_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            "upstream returned an incompatible response",
            &request_id,
        ),
    }
}

async fn anthropic_count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let started = Instant::now();
    let request_id = request_id(&headers);
    let body = match body_or_error(body, true, &request_id) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let mut converted = match anthropic::convert_request(&body) {
        Ok(converted) => converted,
        Err(error) => return protocol_error(error, &request_id),
    };
    converted.body["stream"] = Value::Bool(false);
    converted.body["max_tokens"] = Value::from(1);
    converted
        .body
        .as_object_mut()
        .map(|body| body.remove("stream_options"));
    let upstream_body = match serde_json::to_vec(&converted.body) {
        Ok(body) => Bytes::from(body),
        Err(_) => {
            return anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "failed to serialize token count request",
                &request_id,
            );
        }
    };
    let path = format!("{}/chat/completions", state.config.upstream.api_prefix);
    let result =
        match send_with_failover(&state, Method::POST, &path, &headers, upstream_body).await {
            Ok(result) => result,
            Err(error) => {
                return anthropic_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    public_error(&error),
                    &request_id,
                );
            }
        };
    if !result.response.status().is_success() {
        return anthropic_upstream_error(result.response, &request_id).await;
    }
    let value: Value = match result.response.json().await {
        Ok(value) => value,
        Err(_) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "upstream returned invalid usage data",
                &request_id,
            );
        }
    };
    let input_tokens = value
        .get("usage")
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(Value::as_u64);
    match input_tokens {
        Some(input_tokens) => {
            tracing::info!(
                request_id,
                protocol = "anthropic",
                model = %converted.model,
                account = %result.account,
                path = "/v1/messages/count_tokens",
                stream = false,
                status = 200,
                duration_ms = started.elapsed().as_millis(),
                request_bytes = body.len(),
                response_bytes = 0,
                refresh = result.refreshed,
                failover = result.failover_count,
                "request complete"
            );
            json_response(
                StatusCode::OK,
                json!({"input_tokens":input_tokens}),
                &request_id,
            )
        }
        None => anthropic_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            "upstream response did not include prompt token usage",
            &request_id,
        ),
    }
}

async fn send_with_failover(
    state: &AppState,
    method: Method,
    path: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<UpstreamResult> {
    let mut excluded = HashSet::new();
    let mut failover_count = 0;
    let mut last_rate_limit = None;
    loop {
        let selected = match state.accounts.pick_account(&excluded).await {
            Ok(selected) => selected,
            Err(error) => {
                if let Some(response) = last_rate_limit {
                    return Ok(UpstreamResult {
                        response,
                        account: "exhausted".into(),
                        refreshed: false,
                        failover_count,
                    });
                }
                return Err(error);
            }
        };
        let account_name = selected.name.clone();
        let mut refreshed = selected.refreshed;
        let response = send_once(
            state,
            &selected,
            method.clone(),
            path,
            headers,
            body.clone(),
        )
        .await?;
        match response.status().as_u16() {
            401 | 403 => {
                let stream = request_is_stream(headers, &body);
                if stream {
                    tracing::warn!(
                        request_id = request_id(headers),
                        account = %account_name,
                        path = path.split('?').next().unwrap_or(path),
                        "oauth stream retrying after forced refresh"
                    );
                } else {
                    tracing::warn!(
                        request_id = request_id(headers),
                        account = %account_name,
                        path = path.split('?').next().unwrap_or(path),
                        "oauth request retrying after forced refresh"
                    );
                }
                match state
                    .accounts
                    .force_refresh_after_rejection(&account_name, &selected.access_token)
                    .await
                {
                    Ok(new_token) => {
                        refreshed = true;
                        let retry = send_once(
                            state,
                            &new_token,
                            method.clone(),
                            path,
                            headers,
                            body.clone(),
                        )
                        .await?;
                        if !matches!(retry.status().as_u16(), 401 | 403) {
                            return Ok(UpstreamResult {
                                response: retry,
                                account: account_name,
                                refreshed,
                                failover_count,
                            });
                        }
                        state
                            .accounts
                            .mark_unauthorized_if_current(&account_name, &new_token.access_token)
                            .await?;
                    }
                    Err(error) => {
                        tracing::warn!(account = %account_name, error = %public_error(&error), "account authorization refresh failed");
                    }
                }
                excluded.insert(account_name);
                failover_count += 1;
            }
            429 => {
                excluded.insert(account_name);
                last_rate_limit = Some(response);
                failover_count += 1;
            }
            _ => {
                return Ok(UpstreamResult {
                    response,
                    account: account_name,
                    refreshed,
                    failover_count,
                });
            }
        }
    }
}

fn request_is_stream(headers: &HeaderMap, body: &Bytes) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("text/event-stream"))
        || serde_json::from_slice::<Value>(body)
            .ok()
            .and_then(|value| value.get("stream").and_then(Value::as_bool))
            .unwrap_or(false)
}

async fn send_once(
    state: &AppState,
    account: &SelectedAccount,
    method: Method,
    path: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<reqwest::Response> {
    let started = Instant::now();
    let request_id = request_id(headers);
    let log_path = path.split('?').next().unwrap_or(path);
    let url = format!(
        "{}{}",
        state.config.upstream.base_url.trim_end_matches('/'),
        path
    );
    let mut request = state
        .http
        .request(method.clone(), url)
        .bearer_auth(&account.access_token);
    for (name, value) in headers {
        if is_forwardable_request_header(name.as_str()) {
            request = request.header(name, value);
        }
    }
    for (name, value) in &state.config.upstream.extra_headers {
        if !is_reserved_upstream_header(name) {
            request = request.header(name, value);
        }
    }
    request = request.header("X-App-Id", crate::config::defaults::UPSTREAM_X_APP_ID);
    if !headers.contains_key(header::CONTENT_TYPE) && !body.is_empty() {
        request = request.header(header::CONTENT_TYPE, "application/json");
    }
    let response = request.body(body).send().await;
    match &response {
        Ok(response) => tracing::info!(
            direction = "upstream",
            request_id,
            account = %account.name,
            method = %method,
            path = log_path,
            status = response.status().as_u16(),
            duration_ms = started.elapsed().as_millis(),
            response_bytes = response.content_length().unwrap_or(0),
            content_type = response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .unwrap_or(""),
            "upstream response"
        ),
        Err(error) => tracing::warn!(
            direction = "upstream",
            request_id,
            account = %account.name,
            method = %method,
            path = log_path,
            duration_ms = started.elapsed().as_millis(),
            error = safe_reqwest_error(error),
            "upstream request failed"
        ),
    }
    response.context("upstream request failed")
}

async fn upstream_openai_response(response: reqwest::Response, request_id: &str) -> Response {
    let status = response.status();
    let headers = response.headers().clone();
    if !status.is_success() {
        let value = sanitized_upstream_error(response).await;
        return json_response(
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            value,
            request_id,
        );
    }
    let mut builder = Response::builder().status(status.as_u16());
    for (name, value) in &headers {
        if is_forwardable_response_header(name.as_str()) {
            builder = builder.header(name, value);
        }
    }
    builder = builder.header("x-request-id", request_id);
    let is_stream = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("text/event-stream"));
    if is_stream {
        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(std::io::Error::other));
        builder
            .body(Body::from_stream(stream))
            .unwrap_or_else(|_| Response::new(Body::empty()))
    } else {
        match response.bytes().await {
            Ok(bytes) => builder
                .body(Body::from(bytes))
                .unwrap_or_else(|_| Response::new(Body::empty())),
            Err(_) => openai_error(
                StatusCode::BAD_GATEWAY,
                "failed to read upstream response",
                request_id,
            ),
        }
    }
}

async fn anthropic_upstream_error(response: reqwest::Response, request_id: &str) -> Response {
    let status =
        StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let value = sanitized_upstream_error(response).await;
    let message = value
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("upstream request failed");
    let error_type = match status.as_u16() {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        _ => "api_error",
    };
    anthropic_error(status, error_type, message, request_id)
}

async fn sanitized_upstream_error(response: reqwest::Response) -> Value {
    let text = response.text().await.unwrap_or_default();
    let mut value = serde_json::from_str::<Value>(&text).unwrap_or_else(
        |_| json!({"error":{"type":"upstream_error", "message":sanitize_text(&text)}}),
    );
    redact_value(&mut value);
    truncate_strings(&mut value, 1024);
    if !value.is_object() {
        json!({"error":{"type":"upstream_error", "message":"upstream request failed"}})
    } else {
        value
    }
}

fn redact_value(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                let key = key.to_ascii_lowercase();
                if [
                    "authorization",
                    "cookie",
                    "set-cookie",
                    "access_token",
                    "refresh_token",
                    "api_key",
                    "x-api-key",
                    "client_secret",
                ]
                .iter()
                .any(|sensitive| key == *sensitive || key.ends_with(sensitive))
                {
                    *value = Value::String("[REDACTED]".into());
                } else {
                    redact_value(value);
                }
            }
        }
        Value::Array(array) => array.iter_mut().for_each(redact_value),
        Value::String(text) => *text = sanitize_text(text),
        _ => {}
    }
}

fn sanitize_text(input: &str) -> String {
    let mut words = input.split_whitespace().peekable();
    let mut output = Vec::new();
    while let Some(word) = words.next() {
        if word.eq_ignore_ascii_case("bearer") {
            output.push("Bearer".to_string());
            if words.next().is_some() {
                output.push("[REDACTED]".into());
            }
        } else if looks_like_jwt(word) {
            output.push("[REDACTED]".into());
        } else {
            output.push(word.to_string());
        }
    }
    output.join(" ").chars().take(1024).collect()
}

fn looks_like_jwt(value: &str) -> bool {
    value.len() > 30
        && value.matches('.').count() == 2
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
}

fn truncate_strings(value: &mut Value, max_chars: usize) {
    match value {
        Value::String(text) => *text = text.chars().take(max_chars).collect(),
        Value::Array(array) => array
            .iter_mut()
            .for_each(|value| truncate_strings(value, max_chars)),
        Value::Object(object) => object
            .values_mut()
            .for_each(|value| truncate_strings(value, max_chars)),
        _ => {}
    }
}

async fn request_id_middleware(mut request: Request<Body>, next: Next) -> Response {
    let id = request
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 128
                && value.chars().all(|character| character.is_ascii_graphic())
        })
        .map(str::to_owned)
        .unwrap_or_else(|| format!("req_{}", uuid::Uuid::new_v4().simple()));
    if let Ok(value) = HeaderValue::from_str(&id) {
        request.headers_mut().insert("x-request-id", value);
    }
    let mut response = next.run(request).await;
    insert_request_id(response.headers_mut(), &id);
    response
}

async fn response_log_middleware(request: Request<Body>, next: Next) -> Response {
    let started = Instant::now();
    let request_id = request_id(request.headers());
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let protocol = if is_anthropic_request(&path, request.headers()) {
        "anthropic"
    } else {
        "openai"
    };
    let response = next.run(request).await;
    tracing::info!(
        direction = "client",
        request_id,
        protocol,
        method = %method,
        path,
        status = response.status().as_u16(),
        duration_ms = started.elapsed().as_millis(),
        response_bytes = response
            .headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0),
        content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or(""),
        "client response"
    );
    response
}

async fn auth_middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let request_id = request_id(request.headers());
    let anthropic = is_anthropic_request(request.uri().path(), request.headers());
    let authorized = if state.config.server.api_key.is_empty() {
        request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .is_some_and(|ConnectInfo(address)| address.ip().is_loopback())
    } else {
        let bearer = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "));
        let anthropic_key = request
            .headers()
            .get("x-api-key")
            .and_then(|value| value.to_str().ok());
        bearer
            .or(anthropic_key)
            .is_some_and(|candidate| constant_time_eq(candidate, &state.config.server.api_key))
    };
    if authorized {
        next.run(request).await
    } else if anthropic {
        anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid API key or non-loopback request",
            &request_id,
        )
    } else {
        openai_error(
            StatusCode::UNAUTHORIZED,
            "invalid API key or non-loopback request",
            &request_id,
        )
    }
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let mut difference = left.len() ^ right.len();
    let length = left.len().max(right.len());
    for index in 0..length {
        difference |= usize::from(
            left.get(index).copied().unwrap_or_default()
                ^ right.get(index).copied().unwrap_or_default(),
        );
    }
    difference == 0
}

fn is_anthropic_request(path: &str, headers: &HeaderMap) -> bool {
    path.starts_with("/v1/messages") || headers.contains_key("anthropic-version")
}

fn request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("req_unknown")
        .to_string()
}

fn insert_request_id(headers: &mut HeaderMap, request_id: &str) {
    if let Ok(value) = HeaderValue::from_str(request_id) {
        headers.insert("x-request-id", value);
    }
}

fn protocol_error(error: ProtocolError, request_id: &str) -> Response {
    anthropic_error(
        StatusCode::BAD_REQUEST,
        error.error_type,
        error.message,
        request_id,
    )
}

fn body_or_error(
    body: std::result::Result<Bytes, BytesRejection>,
    anthropic: bool,
    request_id: &str,
) -> std::result::Result<Bytes, Box<Response>> {
    body.map_err(|_| {
        if anthropic {
            Box::new(anthropic_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request_too_large",
                "request body exceeds the configured limit",
                request_id,
            ))
        } else {
            Box::new(openai_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body exceeds the configured limit",
                request_id,
            ))
        }
    })
}

fn anthropic_error(
    status: StatusCode,
    error_type: &str,
    message: impl Into<String>,
    request_id: &str,
) -> Response {
    json_response(
        status,
        anthropic::error_envelope(error_type, message, request_id),
        request_id,
    )
}

fn openai_error(status: StatusCode, message: impl Into<String>, request_id: &str) -> Response {
    json_response(
        status,
        json!({"error":{"type":"proxy_error", "message":message.into()}}),
        request_id,
    )
}

fn json_response(status: StatusCode, value: Value, request_id: &str) -> Response {
    let mut response = (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        value.to_string(),
    )
        .into_response();
    insert_request_id(response.headers_mut(), request_id);
    response
}

fn public_error(error: &anyhow::Error) -> &'static str {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("no enabled") {
        "no authenticated account is available"
    } else if message.contains("timeout") {
        "upstream request timed out"
    } else if message.contains("connect") {
        "could not connect to upstream"
    } else {
        "upstream request failed"
    }
}

fn safe_reqwest_error(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "upstream request timed out"
    } else if error.is_connect() {
        "could not connect to upstream"
    } else if error.is_request() {
        "upstream request could not be sent"
    } else {
        "upstream response failed"
    }
}

fn upstream_path(config: &Config, rest: &str, query: Option<&str>) -> String {
    let path = format!(
        "{}/{}",
        config.upstream.api_prefix.trim_end_matches('/'),
        rest.trim_start_matches('/')
    );
    append_query(path, query)
}

fn append_query(mut path: String, query: Option<&str>) -> String {
    if let Some(query) = query.filter(|query| !query.is_empty()) {
        path.push('?');
        path.push_str(query);
    }
    path
}

fn is_forwardable_request_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "accept"
            | "accept-encoding"
            | "accept-language"
            | "content-type"
            | "user-agent"
            | "x-request-id"
            | "x-client-type"
            | "x-session-id"
            | "anthropic-version"
            | "anthropic-beta"
    )
}

fn is_forwardable_response_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "content-type"
            | "cache-control"
            | "retry-after"
            | "openai-organization"
            | "openai-processing-ms"
            | "openai-version"
            | "x-accel-buffering"
    ) || lower.starts_with("x-ratelimit-")
        || lower.starts_with("anthropic-ratelimit-")
}

fn is_reserved_upstream_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "authorization" | "cookie" | "host" | "content-length" | "x-app-id"
    )
}

pub async fn serve(state: AppState) -> Result<()> {
    let bind = state.config.server.bind.clone();
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("binding to {bind}"))?;
    tracing::info!(bind, "mogick-provider listening");
    axum::serve(
        listener,
        router(state).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .context("serving HTTP")
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::path::PathBuf;
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::extract::OriginalUri;
    use axum::Json;
    use chrono::Utc;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use crate::config::{AccountAuth, AuthStore};

    #[derive(Clone, Default)]
    struct MockUpstream {
        seen: Arc<Mutex<Vec<SeenRequest>>>,
        rate_limit_first: bool,
        unauthorized_first: bool,
        server_error_first: bool,
    }

    #[derive(Clone)]
    struct SeenRequest {
        uri: String,
        authorization: String,
        x_app_id: String,
        request_id: String,
        body: Value,
    }

    async fn mock_upstream_handler(
        State(state): State<MockUpstream>,
        OriginalUri(uri): OriginalUri,
        headers: HeaderMap,
        body: Bytes,
    ) -> Response {
        let authorization = headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let value: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        state.seen.lock().await.push(SeenRequest {
            uri: uri.to_string(),
            authorization: authorization.clone(),
            x_app_id: headers
                .get("x-app-id")
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_string(),
            request_id: headers
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_string(),
            body: value.clone(),
        });
        if state.rate_limit_first && authorization == "Bearer token-a" {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({"error":{"message":"rate limited"}})),
            )
                .into_response();
        }
        if state.unauthorized_first && authorization == "Bearer token-a" {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error":{"message":"invalid token"}})),
            )
                .into_response();
        }
        if state.server_error_first && authorization == "Bearer token-a" {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error":{"message":"temporary failure"}})),
            )
                .into_response();
        }
        if uri.path().ends_with("/models") {
            return Json(json!({"data":[
                {"id":"mm-future-model","owned_by":"llm-store"},
                {"id":"embedding-anything","owned_by":"llm-store"}
            ]}))
            .into_response();
        }
        if uri.path().ends_with("/embeddings") {
            return Json(json!({"object":"list","model":value["model"],"data":[]})).into_response();
        }
        if value.get("stream").and_then(Value::as_bool) == Some(true) {
            let chunks: Vec<std::result::Result<Bytes, Infallible>> = vec![
                Ok(Bytes::from_static(
                    b"data: {\"id\":\"chat-s\",\"model\":\"mm-future-model\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
                )),
                Ok(Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n",
                )),
            ];
            return Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(futures_util::stream::iter(chunks)))
                .unwrap();
        }
        Json(json!({
            "id":"chat-1",
            "model":value.get("model").cloned().unwrap_or_else(|| Value::String("unknown".into())),
            "choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":42,"completion_tokens":1}
        }))
        .into_response()
    }

    async fn test_gateway(
        rate_limit_first: bool,
        server_error_first: bool,
        max_request_bytes: usize,
    ) -> (Router, MockUpstream, PathBuf, tokio::task::JoinHandle<()>) {
        let mock = MockUpstream {
            rate_limit_first,
            server_error_first,
            ..MockUpstream::default()
        };
        let upstream = Router::new()
            .route("/api/v1/*rest", any(mock_upstream_handler))
            .with_state(mock.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });

        let auth_path =
            std::env::temp_dir().join(format!("mogick-gateway-test-{}.json", uuid::Uuid::new_v4()));
        let mut auth = AuthStore::default();
        for (name, token) in [("alice", "token-a"), ("bob", "token-b")] {
            auth.accounts.insert(
                name.into(),
                AccountAuth {
                    access_token: token.into(),
                    refresh_token: format!("refresh-{name}"),
                    token_expiry: Utc::now().timestamp() + 3600,
                    token_type: "Bearer".into(),
                    enabled: true,
                    ..AccountAuth::default()
                },
            );
        }
        auth.save(&auth_path).unwrap();
        let mut config = Config::default();
        config.server.api_key = "gateway-secret".into();
        config.upstream.base_url = format!("http://{address}");
        config.runtime.max_request_bytes = max_request_bytes;
        config
            .upstream
            .extra_headers
            .insert("Authorization".into(), "Bearer configured-evil".into());
        config
            .upstream
            .extra_headers
            .insert("X-App-Id".into(), "configured-evil".into());
        let manager = AccountManager::new(config.clone(), auth_path.clone()).unwrap();
        let app = router(AppState::new(config, manager).unwrap());
        (app, mock, auth_path, task)
    }

    async fn invalid_refresh_handler() -> impl IntoResponse {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"invalid_grant"})),
        )
    }

    async fn valid_refresh_handler() -> impl IntoResponse {
        Json(json!({
            "code": 0,
            "data": {
                "access_token": "stream-refreshed-token",
                "refresh_token": "stream-rotated-refresh",
                "expires_in": 3600,
                "token_type": "Bearer"
            }
        }))
    }

    fn gateway_request(method: Method, uri: &str, body: impl Into<Body>) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header(header::AUTHORIZATION, "Bearer gateway-secret")
            .header(header::CONTENT_TYPE, "application/json")
            .header("x-request-id", "req-integration")
            .extension(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 43123))))
            .body(body.into())
            .unwrap()
    }

    #[test]
    fn redaction_covers_tokens_headers_and_jwts() {
        let mut value = json!({
            "authorization":"Bearer access-secret",
            "nested":{"refresh_token":"refresh-secret"},
            "message":"failed Bearer abc.def.ghi012345678901234567890"
        });
        redact_value(&mut value);
        let text = value.to_string();
        assert!(!text.contains("access-secret"));
        assert!(!text.contains("refresh-secret"));
        assert!(!text.contains("ghi012"));
    }

    #[test]
    fn path_rewrite_preserves_query_and_arbitrary_routes() {
        let config = Config::default();
        assert_eq!(
            upstream_path(&config, "embeddings", Some("dimensions=8")),
            "/api/v1/embeddings?dimensions=8"
        );
        assert_eq!(
            upstream_path(&config, "files/a/content", None),
            "/api/v1/files/a/content"
        );
    }

    #[test]
    fn incoming_secret_comparison_accepts_exact_only() {
        assert!(constant_time_eq("secret", "secret"));
        assert!(!constant_time_eq("secret", "secret2"));
        assert!(!constant_time_eq("", "secret"));
    }

    #[tokio::test]
    async fn openai_passthrough_preserves_query_model_and_forces_auth_headers() {
        let (app, mock, auth_path, task) = test_gateway(false, false, 1024 * 1024).await;
        let request = gateway_request(
            Method::POST,
            "/v1/embeddings?dimensions=8",
            Body::from(r#"{"model":"mm-arbitrary-new","input":"hello"}"#),
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-request-id"], "req-integration");
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["model"], "mm-arbitrary-new");
        let seen = mock.seen.lock().await;
        assert_eq!(seen[0].uri, "/api/v1/embeddings?dimensions=8");
        assert_eq!(seen[0].authorization, "Bearer token-a");
        assert_eq!(seen[0].x_app_id, "mogick");
        assert_eq!(seen[0].request_id, "req-integration");
        drop(seen);
        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn rate_limit_fails_over_to_next_account() {
        let (app, mock, auth_path, task) = test_gateway(true, false, 1024 * 1024).await;
        let response = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/chat/completions",
                Body::from(r#"{"model":"future-model","messages":[]}"#),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let seen = mock.seen.lock().await;
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].authorization, "Bearer token-a");
        assert_eq!(seen[1].authorization, "Bearer token-b");
        drop(seen);
        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn server_errors_are_not_replayed_on_another_account() {
        let (app, mock, auth_path, task) = test_gateway(false, true, 1024 * 1024).await;
        let response = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/chat/completions",
                Body::from(r#"{"model":"future-model","messages":[]}"#),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(mock.seen.lock().await.len(), 1);
        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn invalid_refresh_marks_account_and_fails_over_after_unauthorized() {
        let mock = MockUpstream {
            unauthorized_first: true,
            ..MockUpstream::default()
        };
        let upstream = Router::new()
            .route("/api/v1/*rest", any(mock_upstream_handler))
            .route("/oauth/token", post(invalid_refresh_handler))
            .with_state(mock.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
        let auth_path = std::env::temp_dir().join(format!(
            "mogick-unauthorized-test-{}.json",
            uuid::Uuid::new_v4()
        ));
        let mut auth = AuthStore::default();
        for (name, token) in [("alice", "token-a"), ("bob", "token-b")] {
            auth.accounts.insert(
                name.into(),
                AccountAuth {
                    access_token: token.into(),
                    refresh_token: format!("refresh-{name}"),
                    token_expiry: Utc::now().timestamp() + 3600,
                    enabled: true,
                    ..AccountAuth::default()
                },
            );
        }
        auth.save(&auth_path).unwrap();
        let mut config = Config::default();
        config.server.api_key = "gateway-secret".into();
        config.upstream.base_url = format!("http://{address}");
        let oauth = crate::config::OAuthConfig {
            client_id: "mogick".into(),
            device_authorization_endpoint: format!("http://{address}/device"),
            token_endpoint: format!("http://{address}/oauth/token"),
            scope: "openid profile email".into(),
        };
        let manager =
            AccountManager::new_with_oauth(config.clone(), auth_path.clone(), oauth).unwrap();
        let app = router(AppState::new(config, manager).unwrap());
        let response = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/chat/completions",
                Body::from(r#"{"model":"future-model","messages":[]}"#),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let seen = mock.seen.lock().await;
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].authorization, "Bearer token-a");
        assert_eq!(seen[1].authorization, "Bearer token-b");
        drop(seen);
        assert!(AuthStore::load(&auth_path).unwrap().accounts["alice"].reauth_required);
        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn stream_connection_unauthorized_forces_refresh_and_reconnects_once() {
        let mock = MockUpstream {
            unauthorized_first: true,
            ..MockUpstream::default()
        };
        let upstream = Router::new()
            .route("/api/v1/*rest", any(mock_upstream_handler))
            .route("/oauth/token", post(valid_refresh_handler))
            .with_state(mock.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
        let auth_path = std::env::temp_dir().join(format!(
            "mogick-stream-refresh-test-{}.json",
            uuid::Uuid::new_v4()
        ));
        let mut auth = AuthStore::default();
        auth.accounts.insert(
            "alice".into(),
            AccountAuth {
                access_token: "token-a".into(),
                refresh_token: "refresh-alice".into(),
                token_expiry: Utc::now().timestamp() + 3600,
                enabled: true,
                ..AccountAuth::default()
            },
        );
        auth.save(&auth_path).unwrap();
        let mut config = Config::default();
        config.server.api_key = "gateway-secret".into();
        config.upstream.base_url = format!("http://{address}");
        let oauth = crate::config::OAuthConfig {
            client_id: "mogick".into(),
            device_authorization_endpoint: format!("http://{address}/device"),
            token_endpoint: format!("http://{address}/oauth/token"),
            scope: "openid profile email".into(),
        };
        let manager =
            AccountManager::new_with_oauth(config.clone(), auth_path.clone(), oauth).unwrap();
        let app = router(AppState::new(config, manager).unwrap());
        let response = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/chat/completions",
                Body::from(r#"{"model":"future-model","messages":[],"stream":true}"#),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .contains("text/event-stream"));
        let response_body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert!(String::from_utf8_lossy(&response_body).contains("[DONE]"));
        let seen = mock.seen.lock().await;
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].authorization, "Bearer token-a");
        assert_eq!(seen[1].authorization, "Bearer stream-refreshed-token");
        drop(seen);
        let stored = AuthStore::load(&auth_path).unwrap();
        assert_eq!(
            stored.accounts["alice"].refresh_token,
            "stream-rotated-refresh"
        );
        assert!(!stored.accounts["alice"].reauth_required);
        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn anthropic_messages_count_models_and_stream_are_translated() {
        let (app, mock, auth_path, task) = test_gateway(false, false, 1024 * 1024).await;
        let message_body = r#"{"model":"mm-future-model","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#;
        let message_response = app
            .clone()
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(message_body),
            ))
            .await
            .unwrap();
        let message: Value = serde_json::from_slice(
            &to_bytes(message_response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(message["type"], "message");
        assert_eq!(message["content"][0]["text"], "hello");

        let count: Value = serde_json::from_slice(
            &to_bytes(
                app.clone()
                    .oneshot(gateway_request(
                        Method::POST,
                        "/v1/messages/count_tokens",
                        Body::from(message_body),
                    ))
                    .await
                    .unwrap()
                    .into_body(),
                usize::MAX,
            )
            .await
            .unwrap(),
        )
        .unwrap();
        assert_eq!(count["input_tokens"], 42);

        let mut model_request = gateway_request(Method::GET, "/v1/models", Body::empty());
        model_request
            .headers_mut()
            .insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let models: Value = serde_json::from_slice(
            &to_bytes(
                app.clone()
                    .oneshot(model_request)
                    .await
                    .unwrap()
                    .into_body(),
                usize::MAX,
            )
            .await
            .unwrap(),
        )
        .unwrap();
        assert_eq!(models["data"][0]["id"], "mm-future-model");

        let stream_body = r#"{"model":"mm-future-model","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        let stream = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(stream_body),
            ))
            .await
            .unwrap();
        assert_eq!(stream.headers()[header::CONTENT_TYPE], "text/event-stream");
        let stream_text = String::from_utf8(
            to_bytes(stream.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(stream_text.starts_with("event: message_start"));
        assert!(stream_text.contains("event: content_block_delta"));
        assert!(stream_text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));

        let seen = mock.seen.lock().await;
        assert!(seen.iter().any(|request| {
            request.uri == "/api/v1/chat/completions"
                && request.body["max_tokens"] == 1
                && request.body["stream"] == false
        }));
        drop(seen);
        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn anthropic_auth_and_body_limit_use_anthropic_errors() {
        let (app, _, auth_path, task) = test_gateway(false, false, 32).await;
        let mut x_api_key = gateway_request(Method::POST, "/v1/messages", Body::from("{}"));
        x_api_key.headers_mut().remove(header::AUTHORIZATION);
        x_api_key
            .headers_mut()
            .insert("x-api-key", HeaderValue::from_static("gateway-secret"));
        let accepted = app.clone().oneshot(x_api_key).await.unwrap();
        assert_eq!(accepted.status(), StatusCode::BAD_REQUEST);

        let mut unauthorized = gateway_request(Method::POST, "/v1/messages", Body::from("{}"));
        unauthorized.headers_mut().insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer wrong"),
        );
        let unauthorized = app.clone().oneshot(unauthorized).await.unwrap();
        let unauthorized_body: Value = serde_json::from_slice(
            &to_bytes(unauthorized.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(unauthorized_body["type"], "error");
        assert_eq!(unauthorized_body["error"]["type"], "authentication_error");

        let oversized = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from("x".repeat(128)),
            ))
            .await
            .unwrap();
        assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let oversized_body: Value =
            serde_json::from_slice(&to_bytes(oversized.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(oversized_body["error"]["type"], "request_too_large");
        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    #[ignore = "requires MOGICK_PROVIDER_REAL_TEST=1 and real credentials"]
    async fn real_upstream_opt_in_smoke_suite() {
        if std::env::var("MOGICK_PROVIDER_REAL_TEST").as_deref() != Ok("1") {
            return;
        }
        let config_path = crate::config::default_config_path();
        let auth_path = crate::config::default_auth_path(&config_path);
        let config = Config::load(&config_path).unwrap();
        let manager = AccountManager::new(config.clone(), auth_path).unwrap();
        let usable_accounts = manager
            .snapshots()
            .await
            .unwrap()
            .accounts
            .values()
            .filter(|account| account.usable())
            .count();
        let state = AppState::new(config.clone(), manager).unwrap();
        let headers = HeaderMap::from_iter([(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )]);

        if std::env::var("MOGICK_REAL_FORCE_REFRESH").as_deref() == Ok("1") {
            let account = state
                .accounts
                .snapshots()
                .await
                .unwrap()
                .accounts
                .into_iter()
                .find(|(_, account)| account.usable())
                .map(|(name, _)| name)
                .expect("a usable account is required");
            let refreshed = state.accounts.force_refresh(&account).await.unwrap();
            assert!(refreshed.refreshed);
        }

        let models = send_with_failover(
            &state,
            Method::GET,
            &format!("{}/models", config.upstream.api_prefix),
            &HeaderMap::new(),
            Bytes::new(),
        )
        .await
        .unwrap();
        assert!(models.response.status().is_success());

        let balance = send_with_failover(
            &state,
            Method::GET,
            &format!("{}/user/balance", config.upstream.api_prefix),
            &HeaderMap::new(),
            Bytes::new(),
        )
        .await
        .unwrap();
        assert!(
            balance.response.status().is_success() || balance.response.status().as_u16() == 404
        );

        if usable_accounts > 1 {
            let first = send_with_failover(
                &state,
                Method::GET,
                &format!("{}/models", config.upstream.api_prefix),
                &HeaderMap::new(),
                Bytes::new(),
            )
            .await
            .unwrap();
            let second = send_with_failover(
                &state,
                Method::GET,
                &format!("{}/models", config.upstream.api_prefix),
                &HeaderMap::new(),
                Bytes::new(),
            )
            .await
            .unwrap();
            assert_ne!(first.account, second.account);
        }

        if let Ok(model) = std::env::var("MOGICK_REAL_CHAT_MODEL") {
            let chat_body = Bytes::from(
                json!({"model":model,"messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8})
                    .to_string(),
            );
            let chat = send_with_failover(
                &state,
                Method::POST,
                &format!("{}/chat/completions", config.upstream.api_prefix),
                &headers,
                chat_body,
            )
            .await
            .unwrap();
            assert!(chat.response.status().is_success());
            let stream_body = Bytes::from(
                json!({"model":model,"messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8,"stream":true})
                    .to_string(),
            );
            let stream = send_with_failover(
                &state,
                Method::POST,
                &format!("{}/chat/completions", config.upstream.api_prefix),
                &headers,
                stream_body,
            )
            .await
            .unwrap();
            assert!(stream.response.status().is_success());
            assert!(stream
                .response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.contains("text/event-stream")));
        }

        if let Ok(model) = std::env::var("MOGICK_REAL_EMBEDDING_MODEL") {
            let embedding = send_with_failover(
                &state,
                Method::POST,
                &format!("{}/embeddings", config.upstream.api_prefix),
                &headers,
                Bytes::from(json!({"model":model,"input":"hello"}).to_string()),
            )
            .await
            .unwrap();
            assert!(embedding.response.status().is_success());
        }

        if let (Ok(model), Ok(data_url)) = (
            std::env::var("MOGICK_REAL_VISION_MODEL"),
            std::env::var("MOGICK_REAL_VISION_DATA_URL"),
        ) {
            let vision = send_with_failover(
                &state,
                Method::POST,
                &format!("{}/chat/completions", config.upstream.api_prefix),
                &headers,
                Bytes::from(
                    json!({"model":model,"messages":[{"role":"user","content":[
                        {"type":"text","text":"Describe this image briefly."},
                        {"type":"image_url","image_url":{"url":data_url}}
                    ]}],"max_tokens":16})
                    .to_string(),
                ),
            )
            .await
            .unwrap();
            assert!(vision.response.status().is_success());
        }
    }
}
