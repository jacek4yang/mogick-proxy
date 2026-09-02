//! Mogick upstream HTTP-semantic fingerprint generation.
//!
//! This intentionally mirrors the observable request semantics of the
//! Mogick CLI without claiming TLS ClientHello or HTTP/2 frame-level
//! equivalence with Go's transport implementation.

use std::fmt::Write as _;

use anyhow::{Context, Result};
use rand::RngCore;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, ACCEPT_ENCODING, USER_AGENT};
use ulid::Ulid;

#[derive(Clone)]
pub struct MogickFingerprint {
    app_id: HeaderValue,
    user_agent: HeaderValue,
    client_type: HeaderValue,
    client_version: HeaderValue,
    session_id: HeaderValue,
}

impl MogickFingerprint {
    pub fn new(
        app_id: &str,
        user_agent: &str,
        client_type: &str,
        client_version: &str,
    ) -> Result<Self> {
        Ok(Self {
            app_id: header_value(app_id, "X-App-Id")?,
            user_agent: header_value(user_agent, "User-Agent")?,
            client_type: header_value(client_type, "X-Client-Type")?,
            client_version: header_value(client_version, "X-Client-Version")?,
            session_id: HeaderValue::from_str(&mogick_id("ses_"))
                .context("building Mogick session id header")?,
        })
    }

    /// Build headers for one upstream attempt. The session id stays stable
    /// for the process lifetime; trace/run/turn/step/call ids intentionally
    /// rotate for each retry/failover attempt.
    pub fn headers(&self, stream: bool) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, self.user_agent.clone());
        headers.insert(
            ACCEPT,
            HeaderValue::from_static(if stream {
                "text/event-stream"
            } else {
                "application/json"
            }),
        );
        headers.insert(ACCEPT_ENCODING, HeaderValue::from_static("gzip"));
        headers.insert("x-app-id", self.app_id.clone());
        headers.insert("x-client-type", self.client_type.clone());
        headers.insert("x-client-version", self.client_version.clone());
        headers.insert("x-llm-store-resumable", HeaderValue::from_static("true"));
        headers.insert(
            "x-llm-store-stream-error-events",
            HeaderValue::from_static("true"),
        );
        headers.insert("x-mogick-session-id", self.session_id.clone());
        headers.insert("x-session-id", self.session_id.clone());
        insert_generated(&mut headers, "traceparent", traceparent());
        insert_generated(&mut headers, "x-mogick-llm-call-id", mogick_id("mc_"));
        insert_generated(&mut headers, "x-mogick-run-id", mogick_id("run_"));
        insert_generated(&mut headers, "x-mogick-step-id", mogick_id("step_"));
        insert_generated(&mut headers, "x-mogick-turn-id", mogick_id("turn_"));
        headers
    }
}

fn header_value(value: &str, name: &str) -> Result<HeaderValue> {
    HeaderValue::from_str(value).with_context(|| format!("invalid {name} header value"))
}

fn insert_generated(headers: &mut HeaderMap, name: &'static str, value: String) {
    headers.insert(
        name,
        HeaderValue::from_str(&value).expect("generated Mogick headers are ASCII"),
    );
}

fn mogick_id(prefix: &str) -> String {
    format!("{prefix}{}", Ulid::new())
}

fn traceparent() -> String {
    format!("00-{}-{}-01", random_hex(16), random_hex(8))
}

fn random_hex(length: usize) -> String {
    let mut bytes = vec![0u8; length];
    rand::rng().fill_bytes(&mut bytes);
    if bytes.iter().all(|byte| *byte == 0) {
        bytes[0] = 1;
    }
    let mut output = String::with_capacity(length * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_is_stable_but_attempt_ids_rotate() {
        let fingerprint =
            MogickFingerprint::new("mogick", "mogick/26", "mogick", "26.8.28.4243").unwrap();
        let first = fingerprint.headers(true);
        let second = fingerprint.headers(true);

        assert_eq!(first["x-mogick-session-id"], second["x-mogick-session-id"]);
        assert_eq!(first["x-session-id"], first["x-mogick-session-id"]);
        assert_ne!(first["x-mogick-run-id"], second["x-mogick-run-id"]);
        assert_ne!(first["traceparent"], second["traceparent"]);
        assert_eq!(first[USER_AGENT], "mogick/26");
        assert_eq!(first[ACCEPT], "text/event-stream");
        assert_eq!(first[ACCEPT_ENCODING], "gzip");

        let session = first["x-session-id"].to_str().unwrap();
        assert!(session.starts_with("ses_"));
        assert_eq!(session.len(), 30);
        let trace = first["traceparent"].to_str().unwrap();
        assert_eq!(trace.len(), 55);
        assert!(trace.starts_with("00-"));
        assert!(trace.ends_with("-01"));
    }

    #[test]
    fn non_stream_requests_advertise_json() {
        let fingerprint =
            MogickFingerprint::new("mogick", "mogick/26", "mogick", "26.8.28.4243").unwrap();
        assert_eq!(fingerprint.headers(false)[ACCEPT], "application/json");
    }
}
