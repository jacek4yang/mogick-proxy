//! Command-line entry point.
//!
//! Subcommands:
//!   - `init`    — write a starter config.json
//!   - `login`   — interactive OAuth device-code flow
//!   - `status`  — show current token expiry
//!   - `logout`  — discard stored tokens
//!   - `serve`   — start the reverse proxy (default)

mod config;
mod oauth;
mod server;
mod token;

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use crate::config::{default_config_path, Config};
use crate::oauth::{poll_for_token, request_device_code};
use crate::server::AppState;
use crate::token::TokenManager;

#[derive(Parser, Debug)]
#[command(
    name = "mogick-proxy",
    version,
    about = "OAuth-aware reverse proxy exposing OpenAI-compatible /chat/completions"
)]
struct Cli {
    /// Path to config.json. Defaults to $MOGICK_PROXY_CONFIG or
    /// the user's XDG/APPDATA config dir.
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Write a starter config.json (does not overwrite existing).
    Init,
    /// Run the interactive OAuth device-code flow and persist the tokens.
    Login {
        /// Force a fresh login even if a refresh_token is already present.
        #[arg(long)]
        force: bool,
    },
    /// Show current token expiry + scopes.
    Status,
    /// Discard stored tokens (forces a fresh `login` next time).
    Logout,
    /// Start the reverse proxy. This is the default if no subcommand is given.
    Serve,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(default_config_path);

    let cmd = cli.command.unwrap_or(Command::Serve);
    match cmd {
        Command::Init => cmd_init(&config_path).await,
        Command::Login { force } => cmd_login(&config_path, force).await,
        Command::Status => cmd_status(&config_path),
        Command::Logout => cmd_logout(&config_path),
        Command::Serve => cmd_serve(&config_path).await,
    }
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,mogick_proxy=debug"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

async fn cmd_init(path: &PathBuf) -> Result<()> {
    if path.exists() {
        anyhow::bail!(
            "config already exists at {} (delete it first if you want a fresh one)",
            path.display()
        );
    }
    let cfg = Config::load_or_default(path)?;
    cfg.save(path)?;
    println!("wrote starter config to {}", path.display());
    println!();
    println!("OAuth endpoints + client_id are hard-coded from the Mogick binary.");
    println!("You can run `mogick-proxy login` right away — no edits required.");
    println!();
    println!("Optional next steps:");
    println!("  - edit config.json to change upstream.base_url (default: https://api.tongyuan.cc/v1)");
    println!("  - edit config.json to set server.api_key for non-loopback callers");
    Ok(())
}

async fn cmd_login(path: &PathBuf, force: bool) -> Result<()> {
    let mut cfg = Config::load(path)
        .with_context(|| format!("loading config from {}", path.display()))?;
    // Apply IDA defaults to any empty OAuth field.
    cfg.oauth.apply_defaults();

    if !cfg.tokens.refresh_token.is_empty() && !force {
        println!(
            "refresh_token already present — use --force to discard and re-authenticate"
        );
        return Ok(());
    }

    println!("OAuth endpoints (hard-coded from Mogick binary):");
    println!("  client_id  : {}", cfg.oauth.client_id);
    println!("  device URL : {}", cfg.oauth.device_authorization_endpoint);
    println!("  token URL  : {}", cfg.oauth.token_endpoint);
    if !cfg.oauth.scope.is_empty() {
        println!("  scope      : {}", cfg.oauth.scope);
    }
    println!();

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .context("building HTTP client")?;

    println!("requesting device code...");
    let device = request_device_code(&http, &cfg.oauth).await?;

    let verify_url = device
        .verification_uri_complete
        .clone()
        .unwrap_or_else(|| device.verification_uri.clone());

    println!();
    println!("== Authorise this device ==");
    println!("user code : {}", device.user_code);
    println!("open URL  : {}", verify_url);
    println!(
        "expires in: {}s — waiting for you to complete the browser flow...",
        device.expires_in
    );
    println!();

    // Best-effort: open the URL in the user's default browser.
    if let Err(e) = open_browser(&verify_url) {
        tracing::debug!(error = %e, "could not auto-open browser");
    }

    let token = poll_for_token(&http, &cfg.oauth, &device).await?;
    let mgr = TokenManager::new(path.clone());
    mgr.store_initial_tokens(&token)?;
    println!();
    println!("login OK — tokens persisted to {}", path.display());
    println!("next: run `mogick-proxy serve` to start the reverse proxy");
    Ok(())
}

fn cmd_status(path: &PathBuf) -> Result<()> {
    let mgr = TokenManager::new(path.clone());
    let snap = mgr.snapshot()?;
    if snap.is_empty() {
        println!("no tokens stored — run `mogick-proxy login`");
        return Ok(());
    }
    let now = chrono::Utc::now().timestamp();
    let remaining = snap.expires_at.saturating_sub(now);
    println!("access_token : {}...{}", &snap.access_token[..8.min(snap.access_token.len())],
        if snap.access_token.len() > 8 { &snap.access_token[snap.access_token.len()-4..] } else { "" });
    println!("refresh_token: {}...", &snap.refresh_token[..8.min(snap.refresh_token.len())]);
    println!("token_type   : {}", if snap.token_type.is_empty() { "Bearer" } else { &snap.token_type });
    println!("scope        : {}", if snap.scope.is_empty() { "(none)" } else { &snap.scope });
    println!("expires_at   : {} ({} seconds from now)", snap.expires_at, remaining);
    Ok(())
}

fn cmd_logout(path: &PathBuf) -> Result<()> {
    let mgr = TokenManager::new(path.clone());
    mgr.clear()?;
    println!("tokens cleared");
    Ok(())
}

async fn cmd_serve(path: &PathBuf) -> Result<()> {
    // First read the raw file (without default-filling) so we can detect
    // when defaults corrected the user's config and persist the fix back.
    let raw_before = std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<Config>(&s).ok())
        .map(|c| (c.upstream.base_url, c.upstream.chat_path));

    let cfg = Config::load(path)
        .with_context(|| format!("loading config from {}", path.display()))?;

    if let Some((before_base, before_path)) = raw_before {
        if before_base != cfg.upstream.base_url || before_path != cfg.upstream.chat_path {
            let mut cfg = cfg.clone();
            cfg.save(path).context("persisting corrected upstream config")?;
            println!(
                "note: upstream config auto-corrected\n  base_url : {} -> {}\n  chat_path: {} -> {}\n  saved to {}",
                before_base, cfg.upstream.base_url,
                before_path, cfg.upstream.chat_path,
                path.display()
            );
        }
    }

    if cfg.upstream.base_url.is_empty() {
        anyhow::bail!("upstream.base_url is not configured");
    }

    // Verify we have something usable before binding the port. This catches
    // a missing login early instead of failing on the first request.
    let mgr = TokenManager::new(path.clone());
    let snap = mgr.snapshot()?;
    if snap.is_empty() && cfg.upstream.static_api_key.as_deref().unwrap_or("").is_empty() {
        anyhow::bail!(
            "no tokens stored and no upstream.static_api_key set — run `mogick-proxy login` first"
        );
    }

    println!("mogick-proxy listening on http://{}", cfg.server.bind);
    println!("forwarding {} to {}", cfg.upstream.chat_path, cfg.upstream.base_url);
    println!(
        "background balance poll: every {}s  →  {}",
        token::BALANCE_POLL_SECS,
        format!("{}/api/v1/user/balance", cfg.upstream.base_url.trim_end_matches('/'))
    );

    // Spawn the background balance + refresh loop. It lives for the
    // lifetime of the process; the HTTP server owns the foreground.
    let bg_mgr = mgr.clone();
    tokio::spawn(async move {
        bg_mgr.background_loop().await;
    });

    let state = AppState {
        config_path: path.clone(),
        tokens: mgr,
    };
    server::serve(state).await
}

/// Best-effort cross-platform "open URL in default browser".
fn open_browser(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(url)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::process::Command::new("xdg-open")
            .arg(url)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
    }
    Ok(())
}
