//! mogick-provider command-line entry point.

mod anthropic;
mod config;
mod oauth;
mod server;
mod token;

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use crate::config::{
    default_auth_path, default_config_path, migrate_legacy, provider_oauth, AuthStore, Config,
    LogFormat, MigrationOutcome,
};
use crate::oauth::{poll_for_token, request_device_code};
use crate::server::AppState;
use crate::token::AccountManager;

#[derive(Parser, Debug)]
#[command(
    name = "mogick-provider",
    version,
    about = "Multi-account OAuth gateway for OpenAI and Anthropic APIs"
)]
struct Cli {
    /// Runtime configuration path (or MOGICK_PROVIDER_CONFIG).
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Credential store path (or MOGICK_PROVIDER_AUTH; defaults beside config.json).
    #[arg(long, global = true)]
    auth: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Create config.json and a secure empty auth.json.
    Init,
    /// Authorize and store one OAuth account through RFC 8628 device flow.
    Login {
        #[arg(long)]
        account: Option<String>,
        #[arg(long)]
        force: bool,
        /// Print the remote authorization URL without opening a local browser.
        #[arg(long)]
        no_open: bool,
    },
    /// Show credential metadata without printing any token material.
    Status {
        #[arg(long)]
        account: Option<String>,
    },
    /// Disable and clear one account while leaving all other accounts untouched.
    Logout {
        #[arg(long)]
        account: Option<String>,
    },
    /// Run the API gateway (default command).
    Serve,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(default_config_path);
    let auth_path = cli.auth.unwrap_or_else(|| default_auth_path(&config_path));
    let command = cli.command.unwrap_or(Command::Serve);

    let logging_config = Config::load_or_default(&config_path).unwrap_or_default();
    init_tracing(&logging_config);

    if !matches!(command, Command::Init) {
        match migrate_legacy(&config_path, &auth_path)? {
            MigrationOutcome::Migrated => {
                println!(
                    "migrated legacy credentials to {} and removed them from {}",
                    auth_path.display(),
                    config_path.display()
                );
            }
            MigrationOutcome::NotNeeded => {}
        }
    }

    match command {
        Command::Init => cmd_init(&config_path, &auth_path),
        Command::Login {
            account,
            force,
            no_open,
        } => cmd_login(&config_path, &auth_path, account, force, no_open).await,
        Command::Status { account } => cmd_status(&config_path, &auth_path, account).await,
        Command::Logout { account } => cmd_logout(&config_path, &auth_path, account).await,
        Command::Serve => cmd_serve(&config_path, &auth_path).await,
    }
}

fn init_tracing(config: &Config) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&config.runtime.log_level));
    match config.runtime.log_format {
        LogFormat::Pretty => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_target(false)
                .pretty()
                .try_init();
        }
        LogFormat::Json => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_target(false)
                .json()
                .try_init();
        }
    }
}

fn cmd_init(config_path: &Path, auth_path: &Path) -> Result<()> {
    if config_path.exists() {
        bail!("config already exists at {}", config_path.display());
    }
    Config::default().save(config_path)?;
    if !auth_path.exists() {
        AuthStore::default().save(auth_path)?;
    }
    println!("wrote runtime config to {}", config_path.display());
    println!("wrote secure credential store to {}", auth_path.display());
    println!("next: mogick-provider login --account <name>");
    Ok(())
}

async fn cmd_login(
    config_path: &Path,
    auth_path: &Path,
    account: Option<String>,
    force: bool,
    no_open: bool,
) -> Result<()> {
    let account = resolve_account(account)?;
    let config = load_corrected_config(config_path)?;
    let manager = AccountManager::new(config, auth_path.to_path_buf())?;
    if let Some(existing) = manager.snapshots().await?.accounts.get(&account) {
        if existing.has_credentials() && !force {
            println!("account {account:?} already has credentials; use --force to re-authorize");
            return Ok(());
        }
    }

    let oauth = provider_oauth();
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("building OAuth client")?;
    println!("requesting a device code for account {account:?}...");
    let device = request_device_code(&http, &oauth).await?;
    let verification_url = device
        .verification_uri_complete
        .clone()
        .unwrap_or_else(|| device.verification_uri.clone());
    println!();
    println!("Authorization URL: {verification_url}");
    println!("User code: {}", device.user_code);
    println!(
        "Waiting up to {} seconds for remote authorization...",
        device.expires_in
    );
    if !no_open {
        if let Err(error) = open_browser(&verification_url) {
            tracing::debug!(error = %error, "browser open failed; remote authorization remains available");
        }
    }
    let response = poll_for_token(&http, &oauth, &device).await?;
    manager.store_login(&account, &response).await?;
    println!("login complete for account {account:?}; credentials saved securely");
    Ok(())
}

async fn cmd_status(config_path: &Path, auth_path: &Path, account: Option<String>) -> Result<()> {
    let config = load_corrected_config(config_path)?;
    let manager = AccountManager::new(config, auth_path.to_path_buf())?;
    let store = manager.snapshots().await?;
    let selected: Vec<_> = match account {
        Some(name) => {
            let account = store
                .accounts
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("account {name:?} does not exist"))?;
            vec![(name, account)]
        }
        None => store
            .accounts
            .iter()
            .map(|(name, account)| (name.clone(), account))
            .collect(),
    };
    if selected.is_empty() {
        println!("no accounts configured; run mogick-provider login --account <name>");
        return Ok(());
    }
    let now = chrono::Utc::now().timestamp();
    for (name, account) in selected {
        let remaining = account.token_expiry.saturating_sub(now);
        println!(
            "{name}: enabled={} authenticated={} reauth_required={} expires_in={}s scope={}",
            account.enabled,
            account.has_credentials(),
            account.reauth_required,
            remaining,
            if account.scope.is_empty() {
                "(none)"
            } else {
                account.scope.as_str()
            }
        );
        if let Some(error) = &account.last_error {
            println!("  last_error_at={} summary={}", error.at, error.message);
        }
    }
    Ok(())
}

async fn cmd_logout(config_path: &Path, auth_path: &Path, account: Option<String>) -> Result<()> {
    let account = resolve_account(account)?;
    let config = load_corrected_config(config_path)?;
    let manager = AccountManager::new(config, auth_path.to_path_buf())?;
    if manager.logout(&account).await? {
        println!("account {account:?} was logged out and disabled");
    } else {
        println!("account {account:?} does not exist; nothing changed");
    }
    Ok(())
}

async fn cmd_serve(config_path: &Path, auth_path: &Path) -> Result<()> {
    let config = load_corrected_config(config_path)?;
    let manager = AccountManager::new(config.clone(), auth_path.to_path_buf())?;
    if !manager.has_usable_account().await? {
        bail!("no enabled authenticated account; run mogick-provider login --account <name>");
    }
    println!("mogick-provider listening on http://{}", config.server.bind);
    println!(
        "forwarding /v1/* to {}{}/*",
        config.upstream.base_url, config.upstream.api_prefix
    );
    println!(
        "background balance poll: every {}s",
        config.runtime.balance_poll_secs
    );
    let background = manager.clone();
    tokio::spawn(async move { background.background_loop().await });
    server::serve(AppState::new(config, manager)?).await
}

fn load_corrected_config(path: &Path) -> Result<Config> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading config {}", path.display()))?;
    let before: Config =
        serde_json::from_str(&raw).with_context(|| format!("parsing config {}", path.display()))?;
    let mut after = before.clone();
    after.apply_defaults();
    if before != after {
        after.save(path)?;
        println!("auto-corrected runtime defaults in {}", path.display());
    }
    Ok(after)
}

fn resolve_account(account: Option<String>) -> Result<String> {
    if let Some(account) = account.map(|name| name.trim().to_string()) {
        if !account.is_empty() {
            return Ok(account);
        }
    }
    print!("Account name: ");
    io::stdout().flush().context("flushing account prompt")?;
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .context("reading account name")?;
    let account = input.trim().to_string();
    if account.is_empty() {
        bail!("account name cannot be empty");
    }
    Ok(account)
}

fn open_browser(url: &str) -> io::Result<()> {
    #[cfg(target_os = "windows")]
    std::process::Command::new("cmd")
        .args(["/C", "start", "", url])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    #[cfg(target_os = "macos")]
    std::process::Command::new("open")
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    #[cfg(all(unix, not(target_os = "macos")))]
    std::process::Command::new("xdg-open")
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    Ok(())
}
