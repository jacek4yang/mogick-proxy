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
use crate::config::{defaults, Config};
use crate::fingerprint::MogickFingerprint;
use crate::token::{AccountManager, SelectedAccount};

const MAX_UPSTREAM_JSON_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub accounts: AccountManager,
    http: reqwest::Client,
    fingerprint: MogickFingerprint,
}

impl AppState {
    pub fn new(config: Config, accounts: AccountManager) -> Result<Self> {
        let http = reqwest::Client::builder()
            // The configured provider is reached directly. Incoming gateway
            // traffic is unaffected by this outbound proxy policy.
            .no_proxy()
            .connect_timeout(Duration::from_secs(10))
            // Apply the configured limit to each period without upstream
            // activity, rather than to the total lifetime of a streaming
            // response. A healthy long-running Claude Code turn may exceed
            // this duration while continuing to deliver chunks.
            .read_timeout(Duration::from_secs(config.upstream.timeout_secs))
            .build()
            .context("building upstream HTTP client")?;
        let fingerprint = MogickFingerprint::new(
            &config.headers.app_id,
            &config.headers.user_agent,
            defaults::UPSTREAM_CLIENT_TYPE,
            defaults::UPSTREAM_CLIENT_VERSION,
        )
        .context("building Mogick upstream fingerprint")?;
        Ok(Self {
            config,
            accounts,
            http,
            fingerprint,
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
        .route("/v1/models/:model_id", any(model))
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
    let upstream_headers = result.response.headers().clone();
    let value: Value = match parse_upstream_json(result.response).await {
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
    let mut response = json_response(
        StatusCode::OK,
        anthropic::convert_models(&value),
        &request_id,
    );
    copy_anthropic_rate_headers(response.headers_mut(), &upstream_headers);
    response
}

async fn model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let is_anthropic = headers.contains_key("anthropic-version");
    let request_id = request_id(&headers);
    let body = match body_or_error(body, is_anthropic, &request_id) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    if !is_anthropic {
        let path = append_query(
            format!("{}/models/{model_id}", state.config.upstream.api_prefix),
            uri.query(),
        );
        return openai_forward(state, method, path, headers, body).await;
    }

    // Some OpenAI-compatible providers expose only the list operation. Read
    // that stable endpoint and resolve the Anthropic model detail locally.
    let path = format!("{}/models", state.config.upstream.api_prefix);
    let result = match send_with_failover(&state, Method::GET, &path, &headers, Bytes::new()).await
    {
        Ok(result) => result,
        Err(error) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                public_error(&error),
                &request_id,
            )
        }
    };
    if !result.response.status().is_success() {
        return anthropic_upstream_error(result.response, &request_id).await;
    }
    let upstream_headers = result.response.headers().clone();
    let value: Value = match parse_upstream_json(result.response).await {
        Ok(value) => value,
        Err(_) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "upstream returned invalid model data",
                &request_id,
            )
        }
    };
    let Some(value) = anthropic::convert_model(&value, &model_id) else {
        return anthropic_error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "model not found",
            &request_id,
        );
    };
    let mut response = json_response(StatusCode::OK, value, &request_id);
    copy_anthropic_rate_headers(response.headers_mut(), &upstream_headers);
    response
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
    let mut converted = match anthropic::convert_request(&body) {
        Ok(converted) => converted,
        Err(error) => return protocol_error(error, &request_id),
    };
    log_anthropic_request(
        &converted,
        &request_id,
        request_bytes,
        state.config.runtime.log_prompt_preview_chars,
    );
    let path = format!("{}/chat/completions", state.config.upstream.api_prefix);
    let mut compaction_prelude = None;

    if let Some(config) = converted
        .compaction
        .as_ref()
        .filter(|config| config.should_compact())
        .cloned()
    {
        let summary_body = anthropic::compaction_request_body(&converted.body, &config);
        let summary_body = match serde_json::to_vec(&summary_body) {
            Ok(body) => Bytes::from(body),
            Err(_) => {
                return anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    "failed to serialize compaction request",
                    &request_id,
                )
            }
        };
        let summary_result =
            match send_with_failover(&state, Method::POST, &path, &headers, summary_body).await {
                Ok(result) => result,
                Err(error) => {
                    return anthropic_error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        public_error(&error),
                        &request_id,
                    )
                }
            };
        if !summary_result.response.status().is_success() {
            return anthropic_upstream_error(summary_result.response, &request_id).await;
        }
        let compaction_headers = summary_result.response.headers().clone();
        let summary_value: Value = match parse_upstream_json(summary_result.response).await {
            Ok(value) => value,
            Err(_) => {
                return anthropic_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    "upstream returned invalid compaction data",
                    &request_id,
                )
            }
        };
        let summary = match anthropic::extract_compaction_summary(&summary_value) {
            Ok(summary) => summary,
            Err(_) => {
                return anthropic_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    "upstream compaction did not return summary text",
                    &request_id,
                )
            }
        };
        let prelude = anthropic::CompactionPrelude {
            content: summary.clone(),
            usage: summary_value
                .get("usage")
                .cloned()
                .unwrap_or_else(|| json!({})),
        };
        if config.pause_after_compaction {
            let message = anthropic::compaction_pause_response(
                &request_id,
                &converted.model,
                &prelude,
                converted.context_management.as_ref(),
            );
            return anthropic_owned_message_response(
                message,
                converted.stream,
                &request_id,
                &compaction_headers,
            );
        }
        compaction_prelude = Some(prelude);
        anthropic::apply_compaction_summary(&mut converted.body, &summary);
    }

    // Strict structured outputs and strict tool calls must be buffered so the
    // complete JSON can be validated before any bytes become observable to a
    // streaming client. The client still receives Anthropic SSE afterward.
    let structured_format = converted.structured_output.clone();
    let strict_tools = converted.strict_tools.clone();
    if structured_format.is_some() || !strict_tools.is_empty() {
        converted.body["stream"] = Value::Bool(false);
        converted
            .body
            .as_object_mut()
            .map(|body| body.remove("stream_options"));
        let mut request_body = converted.body.clone();
        for attempt in 0..=2 {
            let serialized = match serde_json::to_vec(&request_body) {
                Ok(body) => Bytes::from(body),
                Err(_) => {
                    return anthropic_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "api_error",
                        "failed to serialize converted request",
                        &request_id,
                    )
                }
            };
            let result =
                match send_with_failover(&state, Method::POST, &path, &headers, serialized).await {
                    Ok(result) => result,
                    Err(error) => {
                        return anthropic_error(
                            StatusCode::BAD_GATEWAY,
                            "api_error",
                            public_error(&error),
                            &request_id,
                        )
                    }
                };
            if !result.response.status().is_success() {
                return anthropic_upstream_error(result.response, &request_id).await;
            }
            let upstream_headers = result.response.headers().clone();
            let value: Value = match parse_upstream_json(result.response).await {
                Ok(value) => value,
                Err(_) => {
                    return anthropic_error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        "upstream returned invalid JSON",
                        &request_id,
                    )
                }
            };
            let structured_validation = structured_format
                .as_ref()
                .map(|format| anthropic::validate_structured_response(&value, format))
                .unwrap_or(Ok(()));
            let tool_validation = anthropic::validate_strict_tool_response(&value, &strict_tools);
            let retry_tool_call = tool_validation.is_err();
            match structured_validation.and(tool_validation) {
                Ok(()) => {
                    let message = match anthropic::convert_response_with_context(
                        &value,
                        &request_id,
                        &converted.model,
                        converted.thinking_display,
                        compaction_prelude.as_ref(),
                        converted.context_management.as_ref(),
                    ) {
                        Ok(message) => message,
                        Err(_) => {
                            return anthropic_error(
                                StatusCode::BAD_GATEWAY,
                                "api_error",
                                "upstream returned an incompatible response",
                                &request_id,
                            )
                        }
                    };
                    return anthropic_owned_message_response(
                        message,
                        converted.stream,
                        &request_id,
                        &upstream_headers,
                    );
                }
                Err(error) if attempt < 2 => {
                    tracing::warn!(
                        request_id,
                        attempt = attempt + 1,
                        error,
                        "retrying invalid constrained output"
                    );
                    request_body = if retry_tool_call {
                        anthropic::strict_tool_retry_body(&request_body, &value)
                    } else {
                        anthropic::structured_retry_body(&request_body, &value)
                    };
                }
                Err(error) => {
                    tracing::warn!(
                        request_id,
                        attempts = 3,
                        error,
                        "constrained output validation exhausted"
                    );
                    return anthropic_error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        if retry_tool_call {
                            "upstream could not produce valid strict tool arguments after bounded retries"
                        } else {
                            "upstream could not produce output matching the requested JSON Schema after bounded retries"
                        },
                        &request_id,
                    );
                }
            }
        }
        return anthropic_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "constrained output retry loop ended unexpectedly",
            &request_id,
        );
    }

    let upstream_body = match serde_json::to_vec(&converted.body) {
        Ok(body) => Bytes::from(body),
        Err(_) => {
            return anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "failed to serialize converted request",
                &request_id,
            )
        }
    };
    let result =
        match send_with_failover(&state, Method::POST, &path, &headers, upstream_body).await {
            Ok(result) => result,
            Err(error) => {
                return anthropic_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    public_error(&error),
                    &request_id,
                )
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
        phase = if converted.stream { "stream_connected" } else { "complete" },
        "Anthropic response ready"
    );
    if !status.is_success() {
        return anthropic_upstream_error(result.response, &request_id).await;
    }
    let upstream_headers = result.response.headers().clone();
    if converted.stream {
        let mut response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("x-request-id", &request_id)
            .header("x-accel-buffering", "no")
            .body(anthropic::stream_body_with_context(
                result.response,
                request_id.clone(),
                converted.model,
                result.account,
                started,
                state.config.runtime.stream_progress_secs,
                converted.thinking_display,
                compaction_prelude,
                converted.context_management,
            ))
            .unwrap_or_else(|_| Response::new(Body::empty()));
        insert_request_id(response.headers_mut(), &request_id);
        copy_anthropic_rate_headers(response.headers_mut(), &upstream_headers);
        return response;
    }
    let value: Value = match parse_upstream_json(result.response).await {
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
    match anthropic::convert_response_with_context(
        &value,
        &request_id,
        &converted.model,
        converted.thinking_display,
        compaction_prelude.as_ref(),
        converted.context_management.as_ref(),
    ) {
        Ok(value) => {
            let mut response = json_response(StatusCode::OK, value, &request_id);
            copy_anthropic_rate_headers(response.headers_mut(), &upstream_headers);
            response
        }
        Err(_) => anthropic_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            "upstream returned an incompatible response",
            &request_id,
        ),
    }
}

fn anthropic_owned_message_response(
    message: Value,
    stream: bool,
    request_id: &str,
    upstream_headers: &HeaderMap,
) -> Response {
    let mut response = if stream {
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("x-accel-buffering", "no")
            .body(anthropic::synthetic_stream_body(message))
            .unwrap_or_else(|_| Response::new(Body::empty()))
    } else {
        json_response(StatusCode::OK, message, request_id)
    };
    insert_request_id(response.headers_mut(), request_id);
    copy_anthropic_rate_headers(response.headers_mut(), upstream_headers);
    response
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
    let mut converted = match anthropic::convert_count_request(&body) {
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
    let upstream_headers = result.response.headers().clone();
    let value: Value = match parse_upstream_json(result.response).await {
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
            let mut value = json!({"input_tokens":input_tokens});
            if let Some(context_management) = &converted.context_management {
                value["context_management"] = json!({
                    "original_input_tokens":context_management
                        .original_input_tokens_for(input_tokens)
                });
            }
            let mut response = json_response(StatusCode::OK, value, &request_id);
            copy_anthropic_rate_headers(response.headers_mut(), &upstream_headers);
            response
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
    let stream = request_is_stream(headers, &body);
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
    if !headers.contains_key(header::CONTENT_TYPE) && !body.is_empty() {
        request = request.header(header::CONTENT_TYPE, "application/json");
    }
    request = request.headers(state.fingerprint.headers(stream));
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
    let upstream_headers = response.headers().clone();
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
    let mut response = anthropic_error(status, error_type, message, request_id);
    copy_anthropic_rate_headers(response.headers_mut(), &upstream_headers);
    response
}

async fn parse_upstream_json(response: reqwest::Response) -> std::result::Result<Value, ()> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_UPSTREAM_JSON_BYTES as u64)
    {
        return Err(());
    }
    let bytes = response.bytes().await.map_err(|_| ())?;
    if bytes.len() > MAX_UPSTREAM_JSON_BYTES {
        return Err(());
    }
    serde_json::from_slice(&bytes).map_err(|_| ())
}

async fn sanitized_upstream_error(response: reqwest::Response) -> Value {
    let bytes = response.bytes().await.unwrap_or_default();
    let bytes = &bytes[..bytes.len().min(64 * 1024)];
    let text = String::from_utf8_lossy(bytes);
    let mut value = serde_json::from_slice::<Value>(bytes).unwrap_or_else(
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

fn log_anthropic_request(
    request: &anthropic::ConvertedRequest,
    request_id: &str,
    request_bytes: usize,
    prompt_preview_chars: usize,
) {
    let messages = request
        .body
        .get("messages")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let message_count = messages.len();
    let system_messages = messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
        .count();
    let tool_result_messages = messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("tool"))
        .count();
    let tool_count = request
        .body
        .get("tools")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let (last_user_chars, prompt_preview) = last_user_text_summary(
        messages,
        prompt_preview_chars.min(crate::config::defaults::MAX_PROMPT_PREVIEW_CHARS),
    );
    let prompt_preview = if prompt_preview_chars == 0 {
        "[disabled]".to_owned()
    } else {
        prompt_preview.unwrap_or_else(|| "[no user text]".into())
    };
    tracing::info!(
        direction = "client",
        request_id,
        protocol = "anthropic",
        model = %request.model,
        stream = request.stream,
        request_bytes,
        estimated_input_tokens = request.estimated_input_tokens,
        message_count,
        system_messages,
        tool_result_messages,
        tool_count,
        strict_tools = request.strict_tools.len(),
        structured_output = request.structured_output.is_some(),
        context_edits = request
            .context_management
            .as_ref()
            .map_or(0, |context| context.applied_edits.len()),
        compaction_requested = request.compaction.is_some(),
        compaction_triggered = request
            .compaction
            .as_ref()
            .is_some_and(anthropic::CompactionConfig::should_compact),
        last_user_chars,
        prompt_preview = %prompt_preview,
        "Anthropic request parsed"
    );
}

fn last_user_text_summary(messages: &[Value], preview_chars: usize) -> (usize, Option<String>) {
    let Some(content) = messages.iter().rev().find_map(|message| {
        (message.get("role").and_then(Value::as_str) == Some("user"))
            .then(|| message.get("content"))
            .flatten()
    }) else {
        return (0, None);
    };
    let mut text_chars = 0usize;
    let mut preview = String::new();
    let scan_limit = preview_chars.saturating_add(256);
    let mut append = |text: &str| {
        text_chars = text_chars.saturating_add(text.chars().count());
        if preview_chars == 0 || preview.chars().count() >= scan_limit {
            return;
        }
        if !preview.is_empty() {
            preview.push(' ');
        }
        let remaining = scan_limit.saturating_sub(preview.chars().count());
        preview.extend(text.chars().take(remaining));
    };
    match content {
        Value::String(text) => append(text),
        Value::Array(blocks) => {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        append(text);
                    }
                }
            }
        }
        _ => {}
    }
    let preview = (preview_chars > 0 && !preview.is_empty()).then(|| {
        sanitize_text(&preview)
            .chars()
            .take(preview_chars)
            .collect()
    });
    (text_chars, preview)
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
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let streaming = content_type.contains("text/event-stream");
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
        content_type,
        phase = if streaming { "stream_open" } else { "complete" },
        "client response ready"
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
    if let Some(error) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<reqwest::Error>())
    {
        return safe_reqwest_error(error);
    }
    let message = error
        .chain()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
        .to_ascii_lowercase();
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
        "accept-language" | "content-type"
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

fn copy_anthropic_rate_headers(target: &mut HeaderMap, source: &HeaderMap) {
    const MAPPINGS: &[(&str, &str)] = &[
        ("retry-after", "retry-after"),
        (
            "x-ratelimit-limit-requests",
            "anthropic-ratelimit-requests-limit",
        ),
        (
            "x-ratelimit-remaining-requests",
            "anthropic-ratelimit-requests-remaining",
        ),
        (
            "x-ratelimit-reset-requests",
            "anthropic-ratelimit-requests-reset",
        ),
        (
            "x-ratelimit-limit-tokens",
            "anthropic-ratelimit-tokens-limit",
        ),
        (
            "x-ratelimit-remaining-tokens",
            "anthropic-ratelimit-tokens-remaining",
        ),
        (
            "x-ratelimit-reset-tokens",
            "anthropic-ratelimit-tokens-reset",
        ),
        (
            "x-ratelimit-limit-input-tokens",
            "anthropic-ratelimit-input-tokens-limit",
        ),
        (
            "x-ratelimit-remaining-input-tokens",
            "anthropic-ratelimit-input-tokens-remaining",
        ),
        (
            "x-ratelimit-reset-input-tokens",
            "anthropic-ratelimit-input-tokens-reset",
        ),
        (
            "x-ratelimit-limit-output-tokens",
            "anthropic-ratelimit-output-tokens-limit",
        ),
        (
            "x-ratelimit-remaining-output-tokens",
            "anthropic-ratelimit-output-tokens-remaining",
        ),
        (
            "x-ratelimit-reset-output-tokens",
            "anthropic-ratelimit-output-tokens-reset",
        ),
    ];
    for (upstream, anthropic) in MAPPINGS {
        if let Some(value) = source.get(*upstream) {
            if let (Ok(name), Ok(value)) = (
                axum::http::HeaderName::from_bytes(anthropic.as_bytes()),
                HeaderValue::from_bytes(value.as_bytes()),
            ) {
                target.insert(name, value);
            }
        }
    }
    for (name, value) in source {
        if name.as_str().starts_with("anthropic-ratelimit-") {
            target.insert(name.clone(), value.clone());
        }
    }
}

fn is_reserved_upstream_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "authorization"
            | "cookie"
            | "host"
            | "content-length"
            | "accept"
            | "accept-encoding"
            | "user-agent"
            | "traceparent"
            | "x-app-id"
            | "x-client-type"
            | "x-client-version"
            | "x-llm-store-resumable"
            | "x-llm-store-stream-error-events"
            | "x-session-id"
    ) || lower.starts_with("x-mogick-")
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
        stall_compaction: bool,
        stall_stream: bool,
        fail_repair: bool,
    }

    #[derive(Clone)]
    struct SeenRequest {
        uri: String,
        authorization: String,
        x_app_id: String,
        request_id: String,
        headers: HeaderMap,
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
            headers: headers.clone(),
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
        let request_text = value.to_string();
        if state.fail_repair && request_text.contains("failed strict JSON Schema validation") {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error":{"message":"repair failed"}})),
            )
                .into_response();
        }
        if request_text.contains("Create the compaction summary now") {
            if state.stall_compaction {
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
            if request_text.contains("EMPTY_COMPACTION_FIXTURE") {
                return Json(json!({
                    "id":"compact-empty",
                    "model":value["model"],
                    "choices":[{"message":{"content":"   "},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":50001,"completion_tokens":1}
                }))
                .into_response();
            }
            return Json(json!({
                "id":"compact-1",
                "model":value["model"],
                "choices":[{"message":{"content":"<summary>preserved project state</summary>"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":50001,"completion_tokens":20}
            }))
            .into_response();
        }
        if request_text.contains("STRICT_TOOL_RETRY") {
            let repaired = request_text.contains("failed strict JSON Schema validation");
            let arguments = if repaired {
                r#"{"path":"src/main.rs"}"#
            } else {
                r#"{"wrong":1}"#
            };
            return Json(json!({
                "id":"strict-tool-1",
                "model":value["model"],
                "choices":[{"message":{"content":null,"tool_calls":[{
                    "id":"call_strict",
                    "type":"function",
                    "function":{"name":"Read","arguments":arguments}
                }]},"finish_reason":"tool_calls"}],
                "usage":{"prompt_tokens":42,"completion_tokens":8}
            }))
            .into_response();
        }
        if value.get("response_format").is_some() {
            let retry_fixture = request_text.contains("STRUCTURED_RETRY");
            let repaired = request_text.contains("failed strict JSON Schema validation");
            let content = if retry_fixture && !repaired {
                r#"{"wrong":1}"#
            } else {
                r#"{"title":"hello"}"#
            };
            return Json(json!({
                "id":"structured-1",
                "model":value["model"],
                "choices":[{"message":{"content":content},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":42,"completion_tokens":8}
            }))
            .into_response();
        }
        if value.get("stream").and_then(Value::as_bool) == Some(true) {
            let stall_stream = state.stall_stream;
            let chunks = async_stream::stream! {
                yield Ok::<Bytes, Infallible>(Bytes::from_static(
                    b"data: {\"id\":\"chat-s\",\"model\":\"mm-future-model\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
                ));
                if stall_stream {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                yield Ok::<Bytes, Infallible>(Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n",
                ));
            };
            return Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(chunks))
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
        test_gateway_with_mock(mock, max_request_bytes, 600).await
    }

    async fn test_gateway_with_mock(
        mock: MockUpstream,
        max_request_bytes: usize,
        timeout_secs: u64,
    ) -> (Router, MockUpstream, PathBuf, tokio::task::JoinHandle<()>) {
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
        config.upstream.timeout_secs = timeout_secs;
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
        assert!(!is_forwardable_request_header("anthropic-version"));
        assert!(!is_forwardable_request_header("anthropic-beta"));
        assert!(!is_forwardable_request_header("user-agent"));
        assert!(!is_forwardable_request_header("x-request-id"));
        assert!(!is_forwardable_request_header("x-session-id"));
        assert!(is_reserved_upstream_header("Traceparent"));
        assert!(is_reserved_upstream_header("X-Mogick-Run-Id"));
    }

    #[test]
    fn openai_rate_limit_headers_are_exposed_with_anthropic_names() {
        let mut upstream = HeaderMap::new();
        upstream.insert(
            "x-ratelimit-limit-requests",
            HeaderValue::from_static("100"),
        );
        upstream.insert(
            "x-ratelimit-remaining-tokens",
            HeaderValue::from_static("42000"),
        );
        upstream.insert("retry-after", HeaderValue::from_static("3"));
        let mut output = HeaderMap::new();
        copy_anthropic_rate_headers(&mut output, &upstream);
        assert_eq!(output["anthropic-ratelimit-requests-limit"], "100");
        assert_eq!(output["anthropic-ratelimit-tokens-remaining"], "42000");
        assert_eq!(output["retry-after"], "3");
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
        let mut request = gateway_request(
            Method::POST,
            "/v1/embeddings?dimensions=8",
            Body::from(r#"{"model":"mm-arbitrary-new","input":"hello"}"#),
        );
        request.headers_mut().insert(
            header::USER_AGENT,
            HeaderValue::from_static("spoof-client/1"),
        );
        request
            .headers_mut()
            .insert("x-client-type", HeaderValue::from_static("spoof"));
        request
            .headers_mut()
            .insert("x-session-id", HeaderValue::from_static("spoof-session"));
        request.headers_mut().insert(
            "traceparent",
            HeaderValue::from_static("00-11111111111111111111111111111111-2222222222222222-01"),
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
        assert_eq!(seen[0].request_id, "");
        assert_eq!(seen[0].headers[header::USER_AGENT], "mogick/26");
        assert_eq!(seen[0].headers[header::ACCEPT], "application/json");
        assert_eq!(seen[0].headers[header::ACCEPT_ENCODING], "gzip");
        assert_eq!(seen[0].headers["x-client-type"], "mogick");
        assert_eq!(seen[0].headers["x-client-version"], "26.8.28.4243");
        assert_eq!(seen[0].headers["x-llm-store-resumable"], "true");
        assert_eq!(seen[0].headers["x-llm-store-stream-error-events"], "true");
        let session = seen[0].headers["x-session-id"].to_str().unwrap();
        assert!(session.starts_with("ses_"));
        assert_eq!(session, seen[0].headers["x-mogick-session-id"]);
        assert!(seen[0].headers["x-mogick-run-id"]
            .to_str()
            .unwrap()
            .starts_with("run_"));
        assert!(seen[0].headers["x-mogick-turn-id"]
            .to_str()
            .unwrap()
            .starts_with("turn_"));
        assert!(seen[0].headers["x-mogick-step-id"]
            .to_str()
            .unwrap()
            .starts_with("step_"));
        assert!(seen[0].headers["x-mogick-llm-call-id"]
            .to_str()
            .unwrap()
            .starts_with("mc_"));
        let traceparent = seen[0].headers["traceparent"].to_str().unwrap();
        assert_eq!(traceparent.len(), 55);
        assert!(traceparent.starts_with("00-"));
        assert!(traceparent.ends_with("-01"));
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
        assert_eq!(
            seen[0].headers["x-session-id"],
            seen[1].headers["x-session-id"]
        );
        assert_ne!(
            seen[0].headers["x-mogick-run-id"],
            seen[1].headers["x-mogick-run-id"]
        );
        assert_ne!(
            seen[0].headers["traceparent"],
            seen[1].headers["traceparent"]
        );
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
            user_agent: "Go-http-client/2.0".into(),
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
            user_agent: "Go-http-client/2.0".into(),
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
                        Body::from(
                            json!({
                                "model":"mm-future-model",
                                "messages":[
                                    {"role":"user","content":"obsolete".repeat(1000)},
                                    {"role":"assistant","content":[{
                                        "type":"compaction","content":"short summary"
                                    }]},
                                    {"role":"user","content":"hi"}
                                ],
                                "context_management":{"edits":[{
                                    "type":"compact_20260112"
                                }]}
                            })
                            .to_string(),
                        ),
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
        assert!(
            count["context_management"]["original_input_tokens"]
                .as_u64()
                .unwrap()
                > 42
        );

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
        assert_eq!(
            models["data"][0]["capabilities"]["context_management"]["compact_20260112"]
                ["supported"],
            true
        );

        let mut model_request =
            gateway_request(Method::GET, "/v1/models/mm-future-model", Body::empty());
        model_request
            .headers_mut()
            .insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let model: Value = serde_json::from_slice(
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
        assert_eq!(model["id"], "mm-future-model");
        assert_eq!(
            model["capabilities"]["structured_outputs"]["supported"],
            true
        );

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
    async fn anthropic_stream_sends_pings_while_upstream_is_idle() {
        let mock = MockUpstream {
            stall_stream: true,
            ..MockUpstream::default()
        };
        let (app, _, auth_path, task) = test_gateway_with_mock(mock, 1024 * 1024, 600).await;
        let stream = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(
                    r#"{"model":"mm-future-model","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
                ),
            ))
            .await
            .unwrap();
        assert_eq!(stream.status(), StatusCode::OK);

        let stream_text = String::from_utf8(
            to_bytes(stream.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        let first_delta = stream_text.find("event: content_block_delta").unwrap();
        let ping = stream_text
            .find("event: ping\ndata: {\"type\":\"ping\"}")
            .unwrap();
        let message_stop = stream_text.find("event: message_stop").unwrap();
        assert!(first_delta < ping);
        assert!(ping < message_stop);
        assert!(stream_text.matches("event: ping").count() >= 2);

        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn compaction_runs_summary_then_continues_and_structured_stream_is_validated() {
        let (app, mock, auth_path, task) = test_gateway(false, false, 1024 * 1024).await;
        let large_history = format!("BEGIN_OLD_HISTORY{}END_OLD_HISTORY", "x".repeat(200_100));
        let compact_request = json!({
            "model":"mm-future-model",
            "max_tokens":100,
            "messages":[{"role":"user","content":large_history}],
            "context_management":{"edits":[{
                "type":"compact_20260112",
                "trigger":{"type":"input_tokens","value":50000}
            }]}
        });
        let response = app
            .clone()
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(compact_request.to_string()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(response["content"][0]["type"], "compaction");
        assert!(response["content"][0]["content"]
            .as_str()
            .unwrap()
            .contains("preserved project state"));
        assert_eq!(response["content"][1]["text"], "hello");
        assert_eq!(response["usage"]["iterations"][0]["type"], "compaction");
        assert_eq!(response["usage"]["iterations"][1]["type"], "message");

        let structured_request = json!({
            "model":"mm-future-model",
            "max_tokens":100,
            "stream":true,
            "messages":[{"role":"user","content":"make a title"}],
            "output_config":{"format":{"type":"json_schema","schema":{
                "type":"object",
                "properties":{"title":{"type":"string"}},
                "required":["title"],
                "additionalProperties":false
            }}}
        });
        let structured = app
            .clone()
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(structured_request.to_string()),
            ))
            .await
            .unwrap();
        assert_eq!(structured.status(), StatusCode::OK);
        assert_eq!(
            structured.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let structured = String::from_utf8(
            to_bytes(structured.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(structured.contains("{\\\"title\\\":\\\"hello\\\"}"));
        assert!(structured.contains("event: message_stop"));

        let seen = mock.seen.lock().await;
        assert_eq!(seen.len(), 3);
        assert!(seen[0].body.to_string().contains("BEGIN_OLD_HISTORY"));
        assert!(!seen[1].body.to_string().contains("BEGIN_OLD_HISTORY"));
        assert!(seen[1].body.to_string().contains("preserved project state"));
        assert_eq!(seen[2].body["stream"], false);
        drop(seen);
        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn constrained_outputs_retry_before_becoming_visible_to_claude_code() {
        let (app, mock, auth_path, task) = test_gateway(false, false, 1024 * 1024).await;
        let structured_request = json!({
            "model":"mm-future-model",
            "max_tokens":100,
            "stream":true,
            "messages":[{"role":"user","content":"STRUCTURED_RETRY"}],
            "output_config":{"format":{"type":"json_schema","schema":{
                "type":"object",
                "properties":{"title":{"type":"string"}},
                "required":["title"],
                "additionalProperties":false
            }}}
        });
        let response = app
            .clone()
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(structured_request.to_string()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let stream = String::from_utf8(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(stream.contains("{\\\"title\\\":\\\"hello\\\"}"));
        assert!(!stream.contains("wrong"));

        let strict_tool_request = json!({
            "model":"mm-future-model",
            "max_tokens":100,
            "stream":true,
            "messages":[{"role":"user","content":"STRICT_TOOL_RETRY"}],
            "tools":[{
                "name":"Read",
                "strict":true,
                "input_schema":{
                    "type":"object",
                    "properties":{"path":{"type":"string"}},
                    "required":["path"],
                    "additionalProperties":false
                }
            }]
        });
        let response = app
            .clone()
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(strict_tool_request.to_string()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let stream = String::from_utf8(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(stream.contains("src/main.rs"));
        assert!(!stream.contains("\\\"wrong\\\""));

        let seen = mock.seen.lock().await;
        assert_eq!(seen.len(), 4);
        assert!(seen[1]
            .body
            .to_string()
            .contains("failed strict JSON Schema validation"));
        assert!(seen[3]
            .body
            .to_string()
            .contains("invalid tool-call attempt"));
        assert!(seen[3].body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .all(|message| message.get("tool_calls").is_none()));
        drop(seen);
        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn repeated_compaction_streams_with_stable_block_indexes() {
        let (app, mock, auth_path, task) = test_gateway(false, false, 1024 * 1024).await;
        let request = json!({
            "model":"mm-future-model",
            "max_tokens":100,
            "stream":true,
            "messages":[
                {"role":"user","content":"OBSOLETE_BEFORE_FIRST_COMPACTION"},
                {"role":"assistant","content":[{
                    "type":"compaction","content":"first authoritative summary"
                }]},
                {"role":"user","content":format!(
                    "SECOND_EPOCH_BEGIN{}SECOND_EPOCH_END",
                    "x".repeat(210_000)
                )}
            ],
            "context_management":{"edits":[{
                "type":"compact_20260112",
                "trigger":{"type":"input_tokens","value":50000}
            }]}
        });
        let response = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(request.to_string()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let stream = String::from_utf8(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(stream
            .contains("\"content_block\":{\"content\":\"\",\"type\":\"compaction\"},\"index\":0"));
        assert!(stream.contains("\"content_block\":{\"text\":\"\",\"type\":\"text\"},\"index\":1"));
        assert!(stream.contains("\"type\":\"compaction_delta\""));
        assert!(stream.contains("\"iterations\":["));
        assert!(stream.contains("\"type\":\"compaction\""));
        assert!(stream.contains("\"type\":\"message\""));
        assert!(stream.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));

        let seen = mock.seen.lock().await;
        assert_eq!(seen.len(), 2);
        let summary_request = seen[0].body.to_string();
        assert!(summary_request.contains("first authoritative summary"));
        assert!(summary_request.contains("SECOND_EPOCH_BEGIN"));
        assert!(!summary_request.contains("OBSOLETE_BEFORE_FIRST_COMPACTION"));
        let continued_request = seen[1].body.to_string();
        assert_eq!(seen[1].body["stream"], true);
        assert!(continued_request.contains("preserved project state"));
        assert!(!continued_request.contains("SECOND_EPOCH_BEGIN"));
        drop(seen);
        task.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn orchestration_failures_stop_safely() {
        let timeout_mock = MockUpstream {
            stall_compaction: true,
            ..MockUpstream::default()
        };
        let (app, mock, auth_path, task) =
            test_gateway_with_mock(timeout_mock, 1024 * 1024, 1).await;
        let compaction_request = |marker: &str| {
            json!({
                "model":"mm-future-model",
                "max_tokens":100,
                "messages":[{"role":"user","content":format!(
                    "{marker}{}", "x".repeat(210_000)
                )}],
                "context_management":{"edits":[{
                    "type":"compact_20260112",
                    "trigger":{"type":"input_tokens","value":50000}
                }]}
            })
        };
        let response = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(compaction_request("TIMEOUT_COMPACTION_FIXTURE").to_string()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = String::from_utf8(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains("upstream request timed out"));
        assert_eq!(mock.seen.lock().await.len(), 1);
        task.abort();
        std::fs::remove_file(auth_path).unwrap();

        let (app, _, auth_path, task) = test_gateway(false, false, 1024 * 1024).await;
        let response = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(compaction_request("EMPTY_COMPACTION_FIXTURE").to_string()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = String::from_utf8(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains("did not return summary text"));
        task.abort();
        std::fs::remove_file(auth_path).unwrap();

        let repair_mock = MockUpstream {
            fail_repair: true,
            ..MockUpstream::default()
        };
        let (app, mock, auth_path, task) =
            test_gateway_with_mock(repair_mock, 1024 * 1024, 600).await;
        let response = app
            .oneshot(gateway_request(
                Method::POST,
                "/v1/messages",
                Body::from(
                    json!({
                        "model":"mm-future-model",
                        "max_tokens":100,
                        "messages":[{"role":"user","content":"STRUCTURED_RETRY"}],
                        "output_config":{"format":{"type":"json_schema","schema":{
                            "type":"object",
                            "properties":{"title":{"type":"string"}},
                            "required":["title"],
                            "additionalProperties":false
                        }}}
                    })
                    .to_string(),
                ),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(mock.seen.lock().await.len(), 2);
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
