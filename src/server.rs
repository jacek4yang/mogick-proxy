//! HTTP reverse proxy. Exposes an OpenAI-compatible API surface at
//! `/v1/...` and forwards to the upstream LLM, injecting a fresh OAuth
//! access token (or static API key) on the way out.
//!
//! ## Route mapping
//!
//! The upstream lives under `/api/v1/...`, but most OpenAI-compatible
//! callers (Claude Code via cc-switch, OpenAI SDK clients, curl, etc.)
//! expect the standard `/v1/...` path. We accept both shapes and rewrite
//! the leading `/v1` → `/api/v1` so a single base URL works for everyone.
//!
//!   client → proxy → upstream
//!   POST /v1/chat/completions         → POST {base}/api/v1/chat/completions
//!   GET  /v1/models                   → GET  {base}/api/v1/models
//!   POST /v1/embeddings               → POST {base}/api/v1/embeddings
//!   GET  /v1/files                    → GET  {base}/api/v1/files
//!   … any other /v1/* path           → {base}/api/v1/...
//!
//! For backwards compatibility we also accept the un-prefixed
//! `/chat/completions` (Mogick's original OpenAICompatProvider behaviour).
//!
//! Streaming responses (SSE) are piped through verbatim — we don't read or
//! buffer the body, so streaming latency stays the same as going direct.

use std::time::Duration;

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use bytes::Bytes;
use futures_util::StreamExt;

use crate::config::Config;
use crate::token::TokenManager;

#[derive(Clone)]
pub struct AppState {
    pub config_path: std::path::PathBuf,
    pub tokens: TokenManager,
}

/// Build the Axum router. Caller must `.with_state(state)` and serve.
pub fn router(state: AppState) -> Router {
    // Catch-all for `/v1/*` — pass-through for any OpenAI-compatible route.
    let v1 = Router::new()
        .route("/v1/*rest", any(passthrough))
        .route("/v1", any(passthrough_root));

    // Backwards-compat: bare `/chat/completions` (Mogick's original).
    // Implemented as a separate handler that hard-codes the upstream path.
    let bare = Router::new()
        .route("/chat/completions", post(passthrough_legacy_chat))
        .route("/healthz", get(healthz))
        .route("/", get(healthz));

    v1.merge(bare)
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

/// Legacy `/chat/completions` — Mogick's original OpenAICompatProvider
/// path. Forwards to `upstream.chat_path` (default `/api/v1/chat/completions`).
async fn passthrough_legacy_chat(
    State(state): State<AppState>,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let cfg = match Config::load(&state.config_path) {
        Ok(c) => c,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    match forward(&state, method, &cfg.upstream.chat_path, &headers, body).await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::error!(error = %e, "forwarding legacy chat failed");
            error_response(StatusCode::BAD_GATEWAY, format!("upstream error: {e}"))
        }
    }
}

/// Catch-all handler that forwards the request to the upstream LLM and
/// pipes the response back. The path on the upstream is derived from the
/// incoming path (rewrite `/v1/...` → `/api/v1/...`).
async fn passthrough(
    State(state): State<AppState>,
    method: Method,
    Path(rest): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // `rest` is the segment captured after `/v1/`, e.g. "chat/completions".
    // If the client hit `/v1` with no trailing path, `rest` is "".
    let upstream_path = rewrite_upstream_path(&rest);
    match forward(&state, method, &upstream_path, &headers, body).await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::error!(error = %e, %upstream_path, "forwarding failed");
            error_response(StatusCode::BAD_GATEWAY, format!("upstream error: {e}"))
        }
    }
}

/// Same as `passthrough` but for the bare `/v1` request (no trailing path).
async fn passthrough_root(
    State(state): State<AppState>,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match forward(&state, method, "/api/v1/", &headers, body).await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::error!(error = %e, "forwarding failed");
            error_response(StatusCode::BAD_GATEWAY, format!("upstream error: {e}"))
        }
    }
}

/// Map a client path captured by `/v1/*rest` onto the upstream path.
///   "chat/completions" → "/api/v1/chat/completions"
///   "models"           → "/api/v1/models"
///   ""                 → "/api/v1/"
fn rewrite_upstream_path(rest: &str) -> String {
    let trimmed = rest.trim_start_matches('/');
    if trimmed.is_empty() {
        "/api/v1/".to_string()
    } else {
        format!("/api/v1/{trimmed}")
    }
}

/// Bearer-token gate. When `server.api_key` is empty, only loopback
/// callers are accepted (the typical case for cc-switch running locally).
async fn auth_middleware(
    State(state): State<AppState>,
    req: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Response {
    let cfg = match Config::load(&state.config_path) {
        Ok(c) => c,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    if cfg.server.api_key.is_empty() {
        if let Some(addr) = req.extensions().get::<axum::extract::ConnectInfo<std::net::SocketAddr>>() {
            if !addr.ip().is_loopback() {
                return error_response(
                    StatusCode::FORBIDDEN,
                    "server.api_key is empty; only loopback callers allowed",
                );
            }
        }
        return next.run(req).await;
    }
    let auth_ok = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|t| t == cfg.server.api_key)
        .unwrap_or(false);
    if !auth_ok {
        return error_response(
            StatusCode::UNAUTHORIZED,
            "missing or invalid Authorization bearer token",
        );
    }
    next.run(req).await
}

/// Send the request to the upstream LLM and translate the response.
async fn forward(
    state: &AppState,
    method: Method,
    upstream_path: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<Response> {
    let cfg = Config::load(&state.config_path)?;
    if cfg.upstream.base_url.is_empty() {
        anyhow::bail!("upstream.base_url is not configured");
    }

    let access_token = state.tokens.current_token().await?;

    let url = format!(
        "{}{}",
        cfg.upstream.base_url.trim_end_matches('/'),
        upstream_path
    );

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(cfg.upstream.timeout_secs))
        .build()
        .context("building upstream client")?;

    let mut req = client
        .request(method, &url)
        .header(header::AUTHORIZATION, format!("Bearer {}", access_token))
        .header("X-App-Id", crate::config::defaults::UPSTREAM_X_APP_ID);

    // Forward caller-provided headers that are safe to forward.
    for (name, value) in headers.iter() {
        let lower = name.as_str().to_ascii_lowercase();
        if is_forwardable_header(&lower) {
            req = req.header(name.clone(), value.clone());
        }
    }

    // Inject configured extra headers.
    for (k, v) in &cfg.upstream.extra_headers {
        req = req.header(HeaderName::try_from(k)?, v.clone());
    }

    // Set a sensible default Content-Type for JSON requests.
    if !headers.contains_key(header::CONTENT_TYPE) && !body.is_empty() {
        req = req.header(header::CONTENT_TYPE, "application/json");
    }

    let resp = req.body(body).send().await.context("calling upstream")?;

    let status = resp.status();
    let mut builder = Response::builder().status(status.as_u16());

    for (name, value) in resp.headers().iter() {
        let lower = name.as_str().to_ascii_lowercase();
        if is_forwardable_response_header(&lower) {
            builder = builder.header(name, value);
        }
    }

    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let is_stream = content_type.contains("text/event-stream");

    if is_stream {
        let stream = resp.bytes_stream();
        let body = Body::from_stream(stream.map(|chunk| chunk.map_err(std::io::Error::other)));
        Ok(builder.body(body).unwrap_or_else(|e| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("building response: {e}"),
            )
        }))
    } else {
        let bytes = resp.bytes().await.context("reading upstream body")?;
        Ok(builder
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(bytes))
            .unwrap_or_else(|e| {
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("building response: {e}"),
                )
            }))
    }
}

fn is_forwardable_header(lower: &str) -> bool {
    matches!(
        lower,
        "accept"
            | "accept-language"
            | "user-agent"
            | "x-request-id"
            | "x-client-type"
            | "x-session-id"
            | "anthropic-version"
            | "anthropic-beta"
    )
}

fn is_forwardable_response_header(lower: &str) -> bool {
    matches!(
        lower,
        "content-type"
            | "cache-control"
            | "x-request-id"
            | "x-ratelimit-limit-requests"
            | "x-ratelimit-remaining-requests"
            | "x-ratelimit-limit-tokens"
            | "x-ratelimit-remaining-tokens"
            | "openai-organization"
            | "openai-processing-ms"
            | "openai-version"
            | "x-accel-buffer"
            | "retry-after"
            | "anthropic-ratelimit-*"
    ) || lower.starts_with("x-ratelimit-")
}

fn error_response(status: StatusCode, msg: impl Into<String>) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": msg.into(),
            "type": "proxy_error",
        }
    });
    (
        status,
        [(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        )],
        body.to_string(),
    )
        .into_response()
}

/// Run the HTTP server until Ctrl-C. Used by the `serve` CLI subcommand.
pub async fn serve(state: AppState) -> Result<()> {
    let cfg = Config::load(&state.config_path)?;
    let bind = cfg.server.bind.clone();
    let app = router(state);

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("binding to {bind}"))?;
    tracing::info!(%bind, "mogick-proxy listening");

    let make_service = app.into_make_service_with_connect_info::<std::net::SocketAddr>();
    axum::serve(listener, make_service)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum serve")?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}

#[allow(dead_code)]
fn _unused_imports() {
    let _ = Method::GET;
}
