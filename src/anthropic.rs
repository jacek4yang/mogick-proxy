//! Anthropic Messages v1 to OpenAI Chat Completions protocol conversion.

use std::collections::{HashMap, HashSet};

use axum::body::Body;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Map, Value};

#[derive(Debug, Clone)]
pub struct ProtocolError {
    pub error_type: &'static str,
    pub message: String,
}

impl ProtocolError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            error_type: "invalid_request_error",
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ProtocolError {}

#[derive(Debug)]
pub struct ConvertedRequest {
    pub body: Value,
    pub model: String,
    pub stream: bool,
    pub compaction: Option<CompactionConfig>,
    pub structured_output: Option<StructuredOutputConfig>,
    pub thinking_display: ThinkingDisplay,
    pub strict_tools: HashMap<String, Value>,
    pub context_management: Option<ContextManagementResult>,
}

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    pub trigger_tokens: usize,
    pub pause_after_compaction: bool,
    pub instructions: Option<String>,
    /// Gateway estimate of the context that will actually reach the
    /// compaction trigger, after an existing compaction block and local edits.
    pub effective_input_tokens: usize,
}

impl CompactionConfig {
    pub fn should_compact(&self) -> bool {
        self.effective_input_tokens >= self.trigger_tokens
    }
}

#[derive(Debug, Clone)]
pub struct ContextManagementResult {
    /// Estimate before applying existing compaction blocks or requested edits.
    pub original_input_tokens: usize,
    /// Estimate after existing compaction blocks but before requested edits.
    pub effective_input_tokens_before_edits: usize,
    /// Estimate of the final prompt sent upstream after all local edits.
    pub effective_input_tokens: usize,
    pub applied_edits: Vec<Value>,
}

impl ContextManagementResult {
    /// Anchor the byte-based estimate to the upstream tokenizer's effective
    /// count. Only the removed portion remains an estimate.
    pub fn original_input_tokens_for(&self, actual_effective_tokens: u64) -> u64 {
        let removed_by_existing_compaction = self
            .original_input_tokens
            .saturating_sub(self.effective_input_tokens_before_edits);
        let removed_by_requested_edits = self
            .effective_input_tokens_before_edits
            .saturating_sub(self.effective_input_tokens);
        actual_effective_tokens
            .saturating_add(removed_by_existing_compaction as u64)
            .saturating_add(removed_by_requested_edits as u64)
    }

    fn response_value(&self) -> Value {
        json!({"applied_edits":self.applied_edits})
    }
}

#[derive(Debug, Clone)]
pub struct StructuredOutputConfig {
    pub schema: Value,
    pub strict: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingDisplay {
    Omitted,
    #[default]
    Summarized,
}

const SAFE_THINKING_SUMMARY: &str =
    "The upstream model performed internal reasoning; private details are omitted by the gateway.";
const MAX_UPSTREAM_SSE_EVENT_BYTES: usize = 8 * 1024 * 1024;
const STRUCTURED_OUTPUT_INSTRUCTION_PREFIX: &str =
    "Return exactly one JSON object for output format ";

#[derive(Debug, Default)]
struct ContextEdits {
    // `None` means all prior thinking turns are retained. This is both the
    // safest fallback and the explicit policy currently sent by Claude Code.
    thinking_turns_to_keep: Option<usize>,
    clear_thinking_requested: bool,
    clear_tool_uses: Option<ClearToolUses>,
    compaction: Option<CompactionConfig>,
}

#[derive(Debug)]
struct ClearToolUses {
    trigger: ContextCount,
    keep: usize,
    clear_at_least: Option<usize>,
    exclude_tools: HashSet<String>,
    clear_tool_inputs: bool,
}

#[derive(Debug)]
enum ContextCount {
    InputTokens(usize),
    ToolUses(usize),
}

pub fn convert_request(bytes: &[u8]) -> Result<ConvertedRequest, ProtocolError> {
    convert_request_inner(bytes, None)
}

pub fn convert_count_request(bytes: &[u8]) -> Result<ConvertedRequest, ProtocolError> {
    // Anthropic's count_tokens endpoint does not require max_tokens. A value
    // of one is sufficient for the minimal upstream tokenizer call.
    convert_request_inner(bytes, Some(1))
}

fn convert_request_inner(
    bytes: &[u8],
    default_max_tokens: Option<u64>,
) -> Result<ConvertedRequest, ProtocolError> {
    let input: Value = serde_json::from_slice(bytes)
        .map_err(|error| ProtocolError::invalid(format!("invalid JSON: {error}")))?;
    let object = input
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("request body must be a JSON object"))?;
    reject_unknown_keys(
        object,
        &[
            "model",
            "messages",
            "max_tokens",
            "system",
            "metadata",
            "stop_sequences",
            "stream",
            "temperature",
            "top_k",
            "top_p",
            "tools",
            "tool_choice",
            "thinking",
            "output_config",
            "output_format",
            "service_tier",
            "context_management",
            "speed",
            "diagnostics",
            "cache_control",
            "fallbacks",
            "betas",
            "anthropic_beta",
        ],
        "request",
    )?;
    let model = required_string(object, "model")?.to_string();
    let max_tokens = match object.get("max_tokens") {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| ProtocolError::invalid("max_tokens must be a positive integer"))?,
        None => default_max_tokens
            .ok_or_else(|| ProtocolError::invalid("max_tokens must be a positive integer"))?,
    };
    if max_tokens == 0 {
        return Err(ProtocolError::invalid(
            "max_tokens must be greater than zero",
        ));
    }
    let stream = object
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    validate_claude_code_advisory_fields(object)?;
    let mut output = Map::new();
    let mut strict_tools = HashMap::new();
    output.insert("model".into(), Value::String(model.clone()));
    output.insert("max_tokens".into(), Value::from(max_tokens));
    output.insert("stream".into(), Value::Bool(stream));

    let mut messages = Vec::new();
    if let Some(system) = object.get("system") {
        messages.push(json!({"role":"system", "content": convert_system(system)?}));
    }
    let context_management_requested = object.contains_key("context_management");
    let mut context_edits = object
        .get("context_management")
        .map(parse_context_management)
        .transpose()?
        .unwrap_or_default();
    let input_messages = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ProtocolError::invalid("messages must be an array"))?;
    let original_input_bytes = estimated_prompt_bytes(object, input_messages);
    let mut input_messages = apply_existing_compaction(input_messages)?;
    let effective_input_bytes_before_edits = estimated_prompt_bytes(object, &input_messages);

    let cleared_thinking_turns = apply_thinking_edit(&mut input_messages, &context_edits);
    let input_bytes_after_thinking = estimated_prompt_bytes(object, &input_messages);
    let cleared_tool_uses = apply_tool_use_edit(
        &mut input_messages,
        &context_edits,
        input_bytes_after_thinking,
    );
    let effective_input_bytes = estimated_prompt_bytes(object, &input_messages);
    let mut applied_edits = Vec::new();
    if cleared_thinking_turns > 0 {
        applied_edits.push(json!({
            "type":"clear_thinking_20251015",
            "cleared_thinking_turns":cleared_thinking_turns,
            "cleared_input_tokens":effective_input_bytes_before_edits
                .saturating_sub(input_bytes_after_thinking)
                .div_ceil(4)
        }));
    }
    if cleared_tool_uses > 0 {
        applied_edits.push(json!({
            "type":"clear_tool_uses_20250919",
            "cleared_tool_uses":cleared_tool_uses,
            "cleared_input_tokens":input_bytes_after_thinking
                .saturating_sub(effective_input_bytes)
                .div_ceil(4)
        }));
    }
    let context_management = context_management_requested.then(|| ContextManagementResult {
        original_input_tokens: original_input_bytes.div_ceil(4),
        effective_input_tokens_before_edits: effective_input_bytes_before_edits.div_ceil(4),
        effective_input_tokens: effective_input_bytes.div_ceil(4),
        applied_edits,
    });
    if let Some(compaction) = &mut context_edits.compaction {
        compaction.effective_input_tokens = effective_input_bytes.div_ceil(4);
    }
    for (index, message) in input_messages.iter().enumerate() {
        convert_message(message, index, true, &mut messages)?;
    }
    output.insert("messages".into(), Value::Array(messages));

    copy_probability(object, &mut output, "temperature")?;
    copy_probability(object, &mut output, "top_p")?;
    if let Some(top_k) = object.get("top_k") {
        let top_k = top_k
            .as_u64()
            .filter(|top_k| *top_k > 0)
            .ok_or_else(|| ProtocolError::invalid("top_k must be a positive integer"))?;
        output.insert("top_k".into(), Value::from(top_k));
    }
    if let Some(stop) = object.get("stop_sequences") {
        let sequences = stop
            .as_array()
            .ok_or_else(|| ProtocolError::invalid("stop_sequences must be an array"))?;
        if sequences
            .iter()
            .any(|sequence| sequence.as_str().map(str::is_empty).unwrap_or(true))
        {
            return Err(ProtocolError::invalid(
                "stop_sequences entries must be non-empty strings",
            ));
        }
        output.insert("stop".into(), stop.clone());
    }
    if let Some(metadata) = object.get("metadata") {
        if !metadata.is_object() {
            return Err(ProtocolError::invalid("metadata must be an object"));
        }
        output.insert("metadata".into(), metadata.clone());
    }
    // service_tier and speed use Anthropic-specific values. They are accepted
    // as client hints but not forwarded to an OpenAI-compatible provider.
    let thinking_display = if let Some(thinking) = object.get("thinking") {
        let display = validate_thinking(thinking, max_tokens)?;
        if thinking.get("type").and_then(Value::as_str) == Some("enabled") {
            let budget = thinking
                .get("budget_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(1024);
            let effort = if budget < 4_096 {
                "low"
            } else if budget < 16_384 {
                "medium"
            } else {
                "high"
            };
            output.insert("reasoning_effort".into(), Value::String(effort.into()));
        }
        display
        // Anthropic thinking configuration is not an OpenAI Chat Completions
        // field. Forwarding it makes otherwise valid Claude Code requests fail
        // against strict OpenAI-compatible providers. The upstream-native
        // reasoning control is populated from output_config.effort below.
    } else {
        ThinkingDisplay::Summarized
    };
    if let Some(tools) = object.get("tools") {
        output.insert("tools".into(), convert_tools(tools)?);
        if let Some(tools) = tools.as_array() {
            for tool in tools {
                if tool.get("strict").and_then(Value::as_bool) == Some(true) {
                    if let (Some(name), Some(schema)) = (
                        tool.get("name").and_then(Value::as_str),
                        tool.get("input_schema").filter(|schema| schema.is_object()),
                    ) {
                        strict_tools.insert(name.to_owned(), schema.clone());
                    }
                }
            }
        }
    }
    if let Some(choice) = object.get("tool_choice") {
        let (tool_choice, parallel) = convert_tool_choice(choice)?;
        output.insert("tool_choice".into(), tool_choice);
        if let Some(parallel) = parallel {
            output.insert("parallel_tool_calls".into(), Value::Bool(parallel));
        }
    }
    let mut structured_output = None;
    if let Some(config) = object.get("output_config") {
        structured_output = convert_output_config(config, &mut output)?;
    }
    if let Some(format) = object.get("output_format") {
        structured_output = Some(apply_output_format(format, &mut output)?);
    }

    if stream {
        output.insert("stream_options".into(), json!({"include_usage": true}));
    }
    Ok(ConvertedRequest {
        body: Value::Object(output),
        model,
        stream,
        compaction: context_edits.compaction,
        structured_output,
        thinking_display,
        strict_tools,
        context_management,
    })
}

fn estimated_prompt_bytes(object: &Map<String, Value>, messages: &[Value]) -> usize {
    let mut total = messages.iter().fold(2usize, |size, message| {
        size.saturating_add(1)
            .saturating_add(estimated_json_bytes(message))
    });
    for key in ["system", "tools", "output_format"] {
        if let Some(value) = object.get(key) {
            total = total.saturating_add(estimated_json_bytes(value));
        }
    }
    if let Some(format) = object
        .get("output_config")
        .and_then(|config| config.get("format"))
    {
        total = total.saturating_add(estimated_json_bytes(format));
    }
    total
}

fn estimated_json_bytes(value: &Value) -> usize {
    match value {
        Value::Null => 4,
        Value::Bool(true) => 4,
        Value::Bool(false) => 5,
        Value::Number(number) => number.to_string().len(),
        Value::String(text) => text.len().saturating_add(2),
        Value::Array(values) => values.iter().fold(2usize, |size, value| {
            size.saturating_add(1)
                .saturating_add(estimated_json_bytes(value))
        }),
        Value::Object(object) => {
            let base64_cap = if object.get("type").and_then(Value::as_str) == Some("base64") {
                Some(
                    if object
                        .get("media_type")
                        .and_then(Value::as_str)
                        .is_some_and(|media_type| media_type.starts_with("image/"))
                    {
                        8 * 1024
                    } else {
                        256 * 1024
                    },
                )
            } else {
                None
            };
            object.iter().fold(2usize, |size, (key, value)| {
                let value_size = if key == "data" {
                    base64_cap
                        .and_then(|cap| value.as_str().map(|data| data.len().min(cap) + 2))
                        .unwrap_or_else(|| estimated_json_bytes(value))
                } else {
                    estimated_json_bytes(value)
                };
                size.saturating_add(key.len())
                    .saturating_add(value_size)
                    .saturating_add(4)
            })
        }
    }
}

fn convert_system(value: &Value) -> Result<Value, ProtocolError> {
    if value.is_string() {
        return Ok(value.clone());
    }
    let blocks = value
        .as_array()
        .ok_or_else(|| ProtocolError::invalid("system must be a string or content block array"))?;
    blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            let object = block.as_object().ok_or_else(|| {
                ProtocolError::invalid(format!("system[{index}] must be an object"))
            })?;
            reject_unknown_keys(object, &["type", "text", "cache_control"], "system block")?;
            if object.get("type").and_then(Value::as_str) != Some("text") {
                return Err(ProtocolError::invalid("system only supports text blocks"));
            }
            let mut result = json!({"type":"text", "text": required_string(object, "text")?});
            preserve_cache_control(object, &mut result)?;
            Ok(result)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn apply_existing_compaction(messages: &[Value]) -> Result<Vec<Value>, ProtocolError> {
    let mut latest = None;
    for (message_index, message) in messages.iter().enumerate() {
        let Some(message_object) = message.as_object() else {
            continue;
        };
        let role = message_object.get("role").and_then(Value::as_str);
        let Some(blocks) = message_object.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (block_index, block) in blocks.iter().enumerate() {
            if block.get("type").and_then(Value::as_str) != Some("compaction") {
                continue;
            }
            if role != Some("assistant") {
                return Err(ProtocolError::invalid(
                    "compaction blocks are only valid in assistant messages",
                ));
            }
            let object = block
                .as_object()
                .ok_or_else(|| ProtocolError::invalid("compaction block must be an object"))?;
            reject_unknown_keys(
                object,
                &["type", "content", "cache_control"],
                "compaction block",
            )?;
            if let Some(cache) = object.get("cache_control") {
                if !cache.is_object() {
                    return Err(ProtocolError::invalid(
                        "compaction cache_control must be an object",
                    ));
                }
            }
            let summary = match object.get("content") {
                Some(Value::String(content)) => content.clone(),
                Some(Value::Null) | None => "[Compaction summary unavailable]".into(),
                _ => {
                    return Err(ProtocolError::invalid(
                        "compaction content must be a string or null",
                    ))
                }
            };
            latest = Some((message_index, block_index, summary));
        }
    }
    let Some((message_index, block_index, summary)) = latest else {
        return Ok(messages.to_vec());
    };

    // Anthropic ignores everything before the newest compaction block. The
    // OpenAI protocol has no equivalent block, so install its summary as a
    // high-priority system message and preserve only blocks/messages after it.
    let mut compacted = vec![json!({
        "role":"system",
        "content":format!("Conversation state restored from a compaction block:\n{summary}")
    })];
    if let Some(blocks) = messages[message_index]
        .get("content")
        .and_then(Value::as_array)
    {
        let remaining = blocks[block_index + 1..].to_vec();
        if !remaining.is_empty() {
            compacted.push(json!({"role":"assistant", "content":remaining}));
        }
    }
    compacted.extend_from_slice(&messages[message_index + 1..]);
    Ok(compacted)
}

fn apply_thinking_edit(messages: &mut [Value], edits: &ContextEdits) -> usize {
    if !edits.clear_thinking_requested {
        return 0;
    }
    let Some(keep) = edits.thinking_turns_to_keep else {
        return 0;
    };
    let retained: HashSet<usize> = messages
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, message)| message_contains_thinking(message))
        .take(keep)
        .map(|(index, _)| index)
        .collect();
    let mut cleared_turns = 0;
    for (index, message) in messages.iter_mut().enumerate() {
        if retained.contains(&index)
            || message.get("role").and_then(Value::as_str) != Some("assistant")
        {
            continue;
        }
        let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        let previous_len = blocks.len();
        blocks.retain(|block| {
            !matches!(
                block.get("type").and_then(Value::as_str),
                Some("thinking" | "redacted_thinking")
            )
        });
        if blocks.len() != previous_len {
            cleared_turns += 1;
        }
    }
    cleared_turns
}

fn apply_tool_use_edit(
    messages: &mut [Value],
    edits: &ContextEdits,
    request_bytes: usize,
) -> usize {
    let Some(policy) = &edits.clear_tool_uses else {
        return 0;
    };
    let mut tool_uses = Vec::new();
    for message in messages.iter() {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                continue;
            }
            let Some(id) = block.get("id").and_then(Value::as_str) else {
                continue;
            };
            let name = block.get("name").and_then(Value::as_str).unwrap_or("");
            if !policy.exclude_tools.contains(name) {
                tool_uses.push((id.to_owned(), name.to_owned()));
            }
        }
    }
    let triggered = match policy.trigger {
        ContextCount::InputTokens(threshold) => request_bytes.div_ceil(4) > threshold,
        ContextCount::ToolUses(threshold) => tool_uses.len() > threshold,
    };
    if !triggered || tool_uses.len() <= policy.keep {
        return 0;
    }
    let clear_count = tool_uses.len() - policy.keep;
    let clear_ids: HashSet<String> = tool_uses
        .into_iter()
        .take(clear_count)
        .map(|(id, _)| id)
        .collect();
    if let Some(minimum) = policy.clear_at_least {
        let clear_bytes: usize = messages
            .iter()
            .flat_map(|message| {
                message
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .filter(|block| {
                let kind = block.get("type").and_then(Value::as_str);
                let id = match kind {
                    Some("tool_use") => block.get("id"),
                    Some("tool_result") => block.get("tool_use_id"),
                    _ => None,
                }
                .and_then(Value::as_str);
                id.is_some_and(|id| clear_ids.contains(id))
                    && (kind == Some("tool_result") || policy.clear_tool_inputs)
            })
            .map(|block| {
                let before = serde_json::to_vec(block).map_or(0, |bytes| bytes.len());
                let mut cleared = block.clone();
                if let Some(object) = cleared.as_object_mut() {
                    match object.get("type").and_then(Value::as_str) {
                        Some("tool_use") => {
                            object.insert(
                                "input".into(),
                                json!({"_context_management":"tool input cleared"}),
                            );
                        }
                        Some("tool_result") => {
                            object.insert(
                                "content".into(),
                                Value::String(
                                    "[Old tool result cleared by context management]".into(),
                                ),
                            );
                        }
                        _ => {}
                    }
                }
                let after = serde_json::to_vec(&cleared).map_or(0, |bytes| bytes.len());
                before.saturating_sub(after)
            })
            .sum();
        if clear_bytes.div_ceil(4) < minimum {
            return 0;
        }
    }

    for message in messages {
        let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in blocks {
            let Some(object) = block.as_object_mut() else {
                continue;
            };
            match object.get("type").and_then(Value::as_str) {
                Some("tool_use") if policy.clear_tool_inputs => {
                    if object
                        .get("id")
                        .and_then(Value::as_str)
                        .is_some_and(|id| clear_ids.contains(id))
                    {
                        object.insert(
                            "input".into(),
                            json!({"_context_management":"tool input cleared"}),
                        );
                    }
                }
                Some("tool_result")
                    if object
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .is_some_and(|id| clear_ids.contains(id)) =>
                {
                    object.insert(
                        "content".into(),
                        Value::String("[Old tool result cleared by context management]".into()),
                    );
                }
                _ => {}
            }
        }
    }
    clear_count
}

fn message_contains_thinking(message: &Value) -> bool {
    let Some(object) = message.as_object() else {
        return false;
    };
    if object.get("role").and_then(Value::as_str) != Some("assistant") {
        return false;
    }
    object
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks.iter().any(|block| {
                matches!(
                    block.get("type").and_then(Value::as_str),
                    Some("thinking" | "redacted_thinking")
                )
            })
        })
}

fn convert_message(
    value: &Value,
    message_index: usize,
    retain_thinking: bool,
    output: &mut Vec<Value>,
) -> Result<(), ProtocolError> {
    let object = value.as_object().ok_or_else(|| {
        ProtocolError::invalid(format!("messages[{message_index}] must be an object"))
    })?;
    reject_unknown_keys(object, &["role", "content"], "message")?;
    let role = required_string(object, "role")?;
    if !matches!(role, "user" | "assistant" | "system") {
        return Err(ProtocolError::invalid(format!(
            "messages[{message_index}].role must be user, assistant, or system"
        )));
    }
    let content = object
        .get("content")
        .ok_or_else(|| ProtocolError::invalid("message content is required"))?;
    if role == "system" {
        output.push(json!({"role":"system", "content":convert_system(content)?}));
        return Ok(());
    }
    if content.is_string() {
        output.push(json!({"role":role, "content":content}));
        return Ok(());
    }
    let blocks = content
        .as_array()
        .ok_or_else(|| ProtocolError::invalid("message content must be a string or array"))?;
    if role == "assistant" {
        convert_assistant_blocks(blocks, retain_thinking, output)
    } else {
        convert_user_blocks(blocks, output)
    }
}

fn convert_assistant_blocks(
    blocks: &[Value],
    retain_thinking: bool,
    output: &mut Vec<Value>,
) -> Result<(), ProtocolError> {
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();
    let mut reasoning = String::new();
    let mut redacted_reasoning = Vec::new();
    for block in blocks {
        let object = block
            .as_object()
            .ok_or_else(|| ProtocolError::invalid("assistant content block must be an object"))?;
        match required_string(object, "type")? {
            "text" => content.push(convert_text_block(object)?),
            "thinking" => {
                reject_unknown_keys(object, &["type", "thinking", "signature"], "thinking block")?;
                let thinking = object
                    .get("thinking")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ProtocolError::invalid("thinking must be a string"))?;
                // Omitted-display thinking blocks legitimately contain an
                // empty string plus an encrypted signature.
                if retain_thinking {
                    if let Some(restored) = object
                        .get("signature")
                        .and_then(Value::as_str)
                        .and_then(restore_thinking_signature)
                    {
                        reasoning.push_str(&restored);
                    } else if !thinking.is_empty() {
                        reasoning.push_str(thinking);
                    }
                }
            }
            "redacted_thinking" => {
                reject_unknown_keys(object, &["type", "data"], "redacted thinking block")?;
                let data = required_string(object, "data")?;
                if retain_thinking {
                    redacted_reasoning.push(Value::String(data.into()));
                }
            }
            "tool_use" => {
                reject_unknown_keys(
                    object,
                    &["type", "id", "name", "input", "cache_control"],
                    "tool_use block",
                )?;
                let mut call = json!({
                    "id": required_string(object, "id")?,
                    "type": "function",
                    "function": {
                        "name": required_string(object, "name")?,
                        "arguments": serde_json::to_string(object.get("input").unwrap_or(&json!({})))
                            .map_err(|error| ProtocolError::invalid(error.to_string()))?
                    }
                });
                preserve_cache_control(object, &mut call)?;
                tool_calls.push(call);
            }
            kind => {
                return Err(ProtocolError::invalid(format!(
                    "unsupported assistant content block type {kind:?}"
                )));
            }
        }
    }
    let mut message = Map::new();
    message.insert("role".into(), Value::String("assistant".into()));
    message.insert("content".into(), Value::Array(content));
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    if !reasoning.is_empty() {
        message.insert("reasoning_content".into(), Value::String(reasoning));
    }
    if !redacted_reasoning.is_empty() {
        message.insert(
            "redacted_reasoning_content".into(),
            Value::Array(redacted_reasoning),
        );
    }
    output.push(Value::Object(message));
    Ok(())
}

fn convert_user_blocks(blocks: &[Value], output: &mut Vec<Value>) -> Result<(), ProtocolError> {
    let mut ordinary = Vec::new();
    for block in blocks {
        let object = block
            .as_object()
            .ok_or_else(|| ProtocolError::invalid("user content block must be an object"))?;
        match required_string(object, "type")? {
            "text" => ordinary.push(convert_text_block(object)?),
            "image" => ordinary.push(convert_image_block(object)?),
            "document" => ordinary.push(convert_document_block(object)?),
            "tool_result" => {
                flush_user_content(&mut ordinary, output);
                output.push(convert_tool_result(object)?);
            }
            kind => {
                return Err(ProtocolError::invalid(format!(
                    "unsupported user content block type {kind:?}"
                )));
            }
        }
    }
    flush_user_content(&mut ordinary, output);
    if blocks.is_empty() {
        output.push(json!({"role":"user", "content":[]}));
    }
    Ok(())
}

fn flush_user_content(content: &mut Vec<Value>, output: &mut Vec<Value>) {
    if !content.is_empty() {
        output.push(json!({"role":"user", "content":std::mem::take(content)}));
    }
}

fn convert_text_block(object: &Map<String, Value>) -> Result<Value, ProtocolError> {
    reject_unknown_keys(object, &["type", "text", "cache_control"], "text block")?;
    let mut result = json!({"type":"text", "text":required_string(object, "text")?});
    preserve_cache_control(object, &mut result)?;
    Ok(result)
}

fn convert_image_block(object: &Map<String, Value>) -> Result<Value, ProtocolError> {
    reject_unknown_keys(object, &["type", "source", "cache_control"], "image block")?;
    let source = object
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| ProtocolError::invalid("image source must be an object"))?;
    reject_unknown_keys(
        source,
        &["type", "media_type", "data", "url"],
        "image source",
    )?;
    let url = match required_string(source, "type")? {
        "base64" => format!(
            "data:{};base64,{}",
            required_string(source, "media_type")?,
            required_string(source, "data")?
        ),
        "url" => required_string(source, "url")?.into(),
        kind => {
            return Err(ProtocolError::invalid(format!(
                "unsupported image source {kind:?}"
            )))
        }
    };
    let mut result = json!({"type":"image_url", "image_url":{"url":url}});
    preserve_cache_control(object, &mut result)?;
    Ok(result)
}

fn convert_document_block(object: &Map<String, Value>) -> Result<Value, ProtocolError> {
    reject_unknown_keys(
        object,
        &[
            "type",
            "source",
            "title",
            "context",
            "citations",
            "cache_control",
        ],
        "document block",
    )?;
    let source = object
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| ProtocolError::invalid("document source must be an object"))?;
    let kind = required_string(source, "type")?;
    let mut result = match kind {
        "base64" => {
            reject_unknown_keys(source, &["type", "media_type", "data"], "document source")?;
            let media_type = required_string(source, "media_type")?;
            let data = required_string(source, "data")?;
            json!({"type":"file", "file":{
                "filename": object.get("title").and_then(Value::as_str).unwrap_or("document"),
                "file_data": format!("data:{media_type};base64,{data}")
            }})
        }
        "text" => {
            reject_unknown_keys(source, &["type", "media_type", "data"], "document source")?;
            json!({"type":"text", "text":required_string(source, "data")?})
        }
        "url" => {
            reject_unknown_keys(source, &["type", "url"], "document source")?;
            json!({"type":"file", "file":{"file_url":required_string(source, "url")?}})
        }
        "content" => {
            reject_unknown_keys(source, &["type", "content"], "document source")?;
            let content = source
                .get("content")
                .ok_or_else(|| ProtocolError::invalid("document content is required"))?;
            json!({"type":"text", "text":document_content_text(content)?})
        }
        _ => {
            return Err(ProtocolError::invalid(format!(
                "unsupported document source {kind:?}"
            )))
        }
    };
    if let Some(context) = object.get("context") {
        result["context"] = context.clone();
    }
    if let Some(citations) = object.get("citations") {
        result["citations"] = citations.clone();
    }
    preserve_cache_control(object, &mut result)?;
    Ok(result)
}

fn document_content_text(content: &Value) -> Result<String, ProtocolError> {
    if let Some(text) = content.as_str() {
        return Ok(text.into());
    }
    let blocks = content
        .as_array()
        .ok_or_else(|| ProtocolError::invalid("document content must be text or text blocks"))?;
    let mut text = String::new();
    for block in blocks {
        let object = block
            .as_object()
            .ok_or_else(|| ProtocolError::invalid("document content block must be an object"))?;
        reject_unknown_keys(object, &["type", "text"], "document content block")?;
        if required_string(object, "type")? != "text" {
            return Err(ProtocolError::invalid(
                "document content only supports text blocks",
            ));
        }
        text.push_str(required_string(object, "text")?);
    }
    Ok(text)
}

fn convert_tool_result(object: &Map<String, Value>) -> Result<Value, ProtocolError> {
    reject_unknown_keys(
        object,
        &[
            "type",
            "tool_use_id",
            "content",
            "is_error",
            "cache_control",
        ],
        "tool_result block",
    )?;
    let content = object
        .get("content")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    if !(content.is_string() || content.is_array()) {
        return Err(ProtocolError::invalid(
            "tool_result content must be a string or array",
        ));
    }
    let converted = if let Some(blocks) = content.as_array() {
        blocks
            .iter()
            .map(|block| {
                let object = block.as_object().ok_or_else(|| {
                    ProtocolError::invalid("tool_result content block must be an object")
                })?;
                match required_string(object, "type")? {
                    "text" => convert_text_block(object),
                    "image" => convert_image_block(object),
                    "document" => convert_document_block(object),
                    kind => Err(ProtocolError::invalid(format!(
                        "unsupported tool_result content type {kind:?}"
                    ))),
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array)?
    } else {
        content
    };
    let mut result = json!({
        "role":"tool",
        "tool_call_id":required_string(object, "tool_use_id")?,
        "content":converted
    });
    if let Some(is_error) = object.get("is_error") {
        if !is_error.is_boolean() {
            return Err(ProtocolError::invalid(
                "tool_result is_error must be boolean",
            ));
        }
        result["is_error"] = is_error.clone();
    }
    preserve_cache_control(object, &mut result)?;
    Ok(result)
}

fn convert_tools(value: &Value) -> Result<Value, ProtocolError> {
    let tools = value
        .as_array()
        .ok_or_else(|| ProtocolError::invalid("tools must be an array"))?;
    tools
        .iter()
        .map(|tool| {
            let object = tool
                .as_object()
                .ok_or_else(|| ProtocolError::invalid("tool must be an object"))?;
            reject_unknown_keys(
                object,
                &[
                    "name",
                    "description",
                    "input_schema",
                    "cache_control",
                    "strict",
                    "defer_loading",
                    "eager_input_streaming",
                ],
                "tool",
            )?;
            let schema = object
                .get("input_schema")
                .filter(|value| value.is_object())
                .ok_or_else(|| ProtocolError::invalid("tool input_schema must be an object"))?;
            let mut function = json!({
                "name":required_string(object, "name")?,
                "parameters":schema
            });
            if let Some(description) = object.get("description") {
                function["description"] = description.clone();
            }
            if let Some(strict) = object.get("strict") {
                if !strict.is_boolean() {
                    return Err(ProtocolError::invalid("tool strict must be boolean"));
                }
                if strict.as_bool() == Some(true) {
                    jsonschema::validator_for(schema).map_err(|error| {
                        ProtocolError::invalid(format!("invalid strict tool JSON Schema: {error}"))
                    })?;
                }
                // Strictness is enforced by the gateway after sampling. Do
                // not forward this optional OpenAI extension to providers
                // that reject it even though ordinary tool calls work.
            }
            for advisory in ["defer_loading", "eager_input_streaming"] {
                if object
                    .get(advisory)
                    .is_some_and(|value| !value.is_boolean())
                {
                    return Err(ProtocolError::invalid(format!(
                        "tool {advisory} must be boolean"
                    )));
                }
            }
            let mut result = json!({"type":"function", "function":function});
            preserve_cache_control(object, &mut result)?;
            Ok(result)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn convert_tool_choice(value: &Value) -> Result<(Value, Option<bool>), ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("tool_choice must be an object"))?;
    reject_unknown_keys(
        object,
        &["type", "name", "disable_parallel_tool_use"],
        "tool_choice",
    )?;
    let converted = match required_string(object, "type")? {
        "auto" => Value::String("auto".into()),
        "any" => Value::String("required".into()),
        "none" => Value::String("none".into()),
        "tool" => json!({"type":"function", "function":{"name":required_string(object, "name")?}}),
        kind => {
            return Err(ProtocolError::invalid(format!(
                "unsupported tool_choice type {kind:?}"
            )))
        }
    };
    let parallel = object
        .get("disable_parallel_tool_use")
        .map(|value| {
            value
                .as_bool()
                .map(|disabled| !disabled)
                .ok_or_else(|| ProtocolError::invalid("disable_parallel_tool_use must be boolean"))
        })
        .transpose()?;
    Ok((converted, parallel))
}

fn validate_thinking(value: &Value, max_tokens: u64) -> Result<ThinkingDisplay, ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("thinking must be an object"))?;
    reject_unknown_keys(object, &["type", "budget_tokens", "display"], "thinking")?;
    let kind = required_string(object, "type")?;
    let display = if let Some(display) = object.get("display") {
        if !matches!(display.as_str(), Some("summarized" | "omitted")) {
            return Err(ProtocolError::invalid(
                "thinking display must be summarized or omitted",
            ));
        }
        if kind == "disabled" {
            return Err(ProtocolError::invalid(
                "disabled thinking cannot specify display",
            ));
        }
        if display.as_str() == Some("omitted") {
            ThinkingDisplay::Omitted
        } else {
            ThinkingDisplay::Summarized
        }
    } else {
        ThinkingDisplay::Summarized
    };
    match kind {
        "enabled" => {
            let budget = object
                .get("budget_tokens")
                .and_then(Value::as_u64)
                .filter(|budget| *budget >= 1024)
                .ok_or_else(|| {
                    ProtocolError::invalid(
                        "enabled thinking requires budget_tokens of at least 1024",
                    )
                })?;
            if budget >= max_tokens {
                return Err(ProtocolError::invalid(
                    "thinking budget_tokens must be less than max_tokens",
                ));
            }
        }
        "disabled" | "adaptive" => {}
        kind => {
            return Err(ProtocolError::invalid(format!(
                "unsupported thinking type {kind:?}"
            )))
        }
    }
    Ok(display)
}

fn parse_context_management(value: &Value) -> Result<ContextEdits, ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("context_management must be an object"))?;
    reject_unknown_keys(object, &["edits"], "context_management")?;
    let edits = object
        .get("edits")
        .and_then(Value::as_array)
        .ok_or_else(|| ProtocolError::invalid("context_management.edits must be an array"))?;
    let mut result = ContextEdits::default();
    let mut saw_thinking_edit = false;
    let mut saw_non_thinking_edit = false;
    for (index, edit) in edits.iter().enumerate() {
        let edit = edit.as_object().ok_or_else(|| {
            ProtocolError::invalid(format!(
                "context_management.edits[{index}] must be an object"
            ))
        })?;
        match required_string(edit, "type")? {
            "clear_thinking_20251015" => {
                reject_unknown_keys(edit, &["type", "keep"], "clear_thinking edit")?;
                if saw_thinking_edit {
                    return Err(ProtocolError::invalid(
                        "context_management may contain only one clear_thinking edit",
                    ));
                }
                if saw_non_thinking_edit {
                    return Err(ProtocolError::invalid(
                        "clear_thinking_20251015 must be the first context management edit",
                    ));
                }
                saw_thinking_edit = true;
                result.clear_thinking_requested = true;
                result.thinking_turns_to_keep = parse_thinking_keep(edit.get("keep"))?;
            }
            "clear_tool_uses_20250919" => {
                saw_non_thinking_edit = true;
                if result.clear_tool_uses.is_some() {
                    return Err(ProtocolError::invalid(
                        "context_management may contain only one clear_tool_uses edit",
                    ));
                }
                result.clear_tool_uses = Some(parse_clear_tool_uses(edit)?);
            }
            "compact_20260112" => {
                saw_non_thinking_edit = true;
                if result.compaction.is_some() {
                    return Err(ProtocolError::invalid(
                        "context_management may contain only one compaction edit",
                    ));
                }
                result.compaction = Some(parse_compaction(edit)?);
            }
            kind => {
                return Err(ProtocolError::invalid(format!(
                    "unsupported context management edit type {kind:?}"
                )))
            }
        }
    }
    Ok(result)
}

fn parse_compaction(edit: &Map<String, Value>) -> Result<CompactionConfig, ProtocolError> {
    reject_unknown_keys(
        edit,
        &["type", "trigger", "pause_after_compaction", "instructions"],
        "compaction edit",
    )?;
    let trigger_tokens = match edit.get("trigger") {
        Some(trigger) => match parse_typed_count(trigger, &["input_tokens"], "compaction trigger")?
        {
            ContextCount::InputTokens(value) => value,
            ContextCount::ToolUses(_) => unreachable!("compaction only permits input_tokens"),
        },
        None => 150_000,
    };
    if trigger_tokens < 50_000 {
        return Err(ProtocolError::invalid(
            "compaction trigger must be at least 50000 input tokens",
        ));
    }
    let pause_after_compaction = edit
        .get("pause_after_compaction")
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| ProtocolError::invalid("pause_after_compaction must be a boolean"))
        })
        .transpose()?
        .unwrap_or(false);
    let instructions = match edit.get("instructions") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    ProtocolError::invalid("compaction instructions must be a non-empty string")
                })?,
        ),
    };
    Ok(CompactionConfig {
        trigger_tokens,
        pause_after_compaction,
        instructions,
        effective_input_tokens: 0,
    })
}

fn validate_claude_code_advisory_fields(object: &Map<String, Value>) -> Result<(), ProtocolError> {
    for key in ["service_tier", "speed"] {
        if object
            .get(key)
            .is_some_and(|value| value.as_str().map(str::is_empty).unwrap_or(true))
        {
            return Err(ProtocolError::invalid(format!(
                "{key} must be a non-empty string"
            )));
        }
    }
    for key in ["diagnostics", "cache_control"] {
        if object.get(key).is_some_and(|value| !value.is_object()) {
            return Err(ProtocolError::invalid(format!("{key} must be an object")));
        }
    }
    for key in ["betas", "anthropic_beta"] {
        if let Some(betas) = object.get(key) {
            let betas = betas
                .as_array()
                .ok_or_else(|| ProtocolError::invalid(format!("{key} must be an array")))?;
            if betas
                .iter()
                .any(|beta| beta.as_str().map(str::is_empty).unwrap_or(true))
            {
                return Err(ProtocolError::invalid(format!(
                    "{key} entries must be non-empty strings"
                )));
            }
        }
    }
    if let Some(fallbacks) = object.get("fallbacks") {
        if fallbacks.as_str() != Some("default") {
            let fallbacks = fallbacks
                .as_array()
                .ok_or_else(|| ProtocolError::invalid("fallbacks must be default or an array"))?;
            for fallback in fallbacks {
                let fallback = fallback
                    .as_object()
                    .ok_or_else(|| ProtocolError::invalid("fallback entries must be objects"))?;
                reject_unknown_keys(fallback, &["model"], "fallback")?;
                required_string(fallback, "model")?;
            }
        }
    }
    Ok(())
}

fn parse_thinking_keep(value: Option<&Value>) -> Result<Option<usize>, ProtocolError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.as_str() == Some("all") {
        return Ok(None);
    }
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("clear_thinking keep must be all or an object"))?;
    reject_unknown_keys(object, &["type", "value"], "clear_thinking keep")?;
    match required_string(object, "type")? {
        "all" => {
            if object.contains_key("value") {
                return Err(ProtocolError::invalid(
                    "clear_thinking keep all cannot specify value",
                ));
            }
            Ok(None)
        }
        "thinking_turns" => positive_usize(object, "value", "clear_thinking keep").map(Some),
        kind => Err(ProtocolError::invalid(format!(
            "unsupported clear_thinking keep type {kind:?}"
        ))),
    }
}

fn parse_clear_tool_uses(edit: &Map<String, Value>) -> Result<ClearToolUses, ProtocolError> {
    reject_unknown_keys(
        edit,
        &[
            "type",
            "trigger",
            "keep",
            "clear_at_least",
            "exclude_tools",
            "clear_tool_inputs",
        ],
        "clear_tool_uses edit",
    )?;
    let trigger = edit
        .get("trigger")
        .map(|value| parse_typed_count(value, &["input_tokens", "tool_uses"], "context trigger"))
        .transpose()?
        .unwrap_or(ContextCount::InputTokens(100_000));
    let keep = match edit.get("keep") {
        Some(value) => match parse_typed_count(value, &["tool_uses"], "context keep")? {
            ContextCount::ToolUses(value) => value,
            ContextCount::InputTokens(_) => unreachable!("keep only permits tool_uses"),
        },
        None => 3,
    };
    let clear_at_least = match edit.get("clear_at_least") {
        Some(value) => match parse_typed_count(value, &["input_tokens"], "context clear_at_least")?
        {
            ContextCount::InputTokens(value) => Some(value),
            ContextCount::ToolUses(_) => unreachable!("clear_at_least only permits input_tokens"),
        },
        None => None,
    };
    let mut exclude_tools = HashSet::new();
    if let Some(excluded) = edit.get("exclude_tools") {
        let excluded = excluded
            .as_array()
            .ok_or_else(|| ProtocolError::invalid("context exclude_tools must be an array"))?;
        if excluded
            .iter()
            .any(|name| name.as_str().map(str::is_empty).unwrap_or(true))
        {
            return Err(ProtocolError::invalid(
                "context exclude_tools entries must be non-empty strings",
            ));
        }
        exclude_tools.extend(excluded.iter().filter_map(Value::as_str).map(str::to_owned));
    }
    let clear_tool_inputs = edit
        .get("clear_tool_inputs")
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| ProtocolError::invalid("context clear_tool_inputs must be boolean"))
        })
        .transpose()?
        .unwrap_or(false);
    Ok(ClearToolUses {
        trigger,
        keep,
        clear_at_least,
        exclude_tools,
        clear_tool_inputs,
    })
}

fn parse_typed_count(
    value: &Value,
    allowed_types: &[&str],
    context: &str,
) -> Result<ContextCount, ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid(format!("{context} must be an object")))?;
    reject_unknown_keys(object, &["type", "value"], context)?;
    let kind = required_string(object, "type")?;
    if !allowed_types.contains(&kind) {
        return Err(ProtocolError::invalid(format!(
            "unsupported {context} type {kind:?}"
        )));
    }
    let value = positive_usize(object, "value", context)?;
    match kind {
        "input_tokens" => Ok(ContextCount::InputTokens(value)),
        "tool_uses" => Ok(ContextCount::ToolUses(value)),
        _ => unreachable!("kind was checked against allowed_types"),
    }
}

fn positive_usize(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<usize, ProtocolError> {
    let value = object
        .get(key)
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| ProtocolError::invalid(format!("{context} {key} must be positive")))?;
    usize::try_from(value)
        .map_err(|_| ProtocolError::invalid(format!("{context} {key} is too large")))
}

fn convert_output_config(
    value: &Value,
    output: &mut Map<String, Value>,
) -> Result<Option<StructuredOutputConfig>, ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("output_config must be an object"))?;
    reject_unknown_keys(
        object,
        &["format", "effort", "task_budget"],
        "output_config",
    )?;
    let structured_output = object
        .get("format")
        .map(|format| apply_output_format(format, output))
        .transpose()?;
    if let Some(effort) = object.get("effort") {
        let effort = effort
            .as_str()
            .ok_or_else(|| ProtocolError::invalid("output_config effort must be a string"))?;
        let upstream_effort = match effort {
            "low" | "medium" | "high" => effort,
            // Current Anthropic models expose deeper levels than the OpenAI
            // compatible upstream. Clamp instead of turning a valid Claude
            // Code request into an upstream 400.
            "xhigh" | "max" => "high",
            _ => {
                return Err(ProtocolError::invalid(
                    "output_config effort must be low, medium, high, xhigh, or max",
                ))
            }
        };
        output.insert(
            "reasoning_effort".into(),
            Value::String(upstream_effort.into()),
        );
    }
    if let Some(task_budget) = object.get("task_budget") {
        let task_budget = task_budget
            .as_object()
            .ok_or_else(|| ProtocolError::invalid("task_budget must be an object"))?;
        reject_unknown_keys(task_budget, &["type", "total", "remaining"], "task_budget")?;
        if required_string(task_budget, "type")? != "tokens" {
            return Err(ProtocolError::invalid("task_budget type must be tokens"));
        }
        positive_usize(task_budget, "total", "task_budget")?;
        if let Some(remaining) = task_budget.get("remaining") {
            remaining.as_u64().ok_or_else(|| {
                ProtocolError::invalid("task_budget remaining must be a non-negative integer")
            })?;
        }
        // Task budgets are advisory pacing controls rather than hard limits.
        // Validating and consuming them keeps Claude Code functional without
        // sending an unsupported object to the OpenAI upstream.
    }
    Ok(structured_output)
}

fn apply_output_format(
    value: &Value,
    output: &mut Map<String, Value>,
) -> Result<StructuredOutputConfig, ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("structured output format must be an object"))?;
    reject_unknown_keys(
        object,
        &["type", "schema", "name", "description", "strict"],
        "output format",
    )?;
    if required_string(object, "type")? != "json_schema" {
        return Err(ProtocolError::invalid(
            "only json_schema structured output is supported",
        ));
    }
    let schema = object
        .get("schema")
        .filter(|value| value.is_object())
        .ok_or_else(|| ProtocolError::invalid("output format schema must be an object"))?;
    jsonschema::validator_for(schema)
        .map_err(|error| ProtocolError::invalid(format!("invalid output JSON Schema: {error}")))?;
    if let Some(name) = object.get("name") {
        name.as_str()
            .ok_or_else(|| ProtocolError::invalid("output format name must be a string"))?;
    }
    if let Some(description) = object.get("description") {
        description
            .as_str()
            .ok_or_else(|| ProtocolError::invalid("output format description must be a string"))?;
    }
    let strict = if let Some(strict) = object.get("strict") {
        strict
            .as_bool()
            .ok_or_else(|| ProtocolError::invalid("output format strict must be a boolean"))?
    } else {
        true
    };

    // The Tongyuan OpenAI-compatible endpoint currently rejects the standard
    // json_schema response format, while accepting json_object. Keep the full
    // schema constraint in a system instruction and use the supported wire
    // format so Claude Code's structured auxiliary requests do not fail.
    let schema = serde_json::to_string(schema)
        .map_err(|error| ProtocolError::invalid(format!("invalid output schema: {error}")))?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("structured_output");
    let description = object
        .get("description")
        .and_then(Value::as_str)
        .map(|value| format!("\nFormat description: {value}"))
        .unwrap_or_default();
    let instruction = format!(
        "{STRUCTURED_OUTPUT_INSTRUCTION_PREFIX}{name:?}.{description}\n\
         It must conform to this JSON Schema: {schema}\n\
         Do not include Markdown fences, commentary, or any text outside the JSON object."
    );
    append_system_instruction(output, instruction)?;
    output.insert("response_format".into(), json!({"type":"json_object"}));
    Ok(StructuredOutputConfig {
        schema: object["schema"].clone(),
        strict,
    })
}

pub fn validate_structured_response(
    upstream: &Value,
    format: &StructuredOutputConfig,
) -> Result<(), String> {
    if upstream
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(Value::as_str)
        .is_some_and(|reason| matches!(reason, "tool_calls" | "function_call"))
    {
        return Ok(());
    }
    let text = completion_text(upstream)
        .ok_or_else(|| "upstream response did not contain structured output text".to_string())?;
    let instance: Value = serde_json::from_str(text)
        .map_err(|error| format!("structured output was not valid JSON: {error}"))?;
    if format.strict && !jsonschema::is_valid(&format.schema, &instance) {
        return Err("structured output did not match the requested JSON Schema".into());
    }
    Ok(())
}

pub fn validate_strict_tool_response(
    upstream: &Value,
    strict_tools: &HashMap<String, Value>,
) -> Result<(), String> {
    let calls = upstream
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array);
    let Some(calls) = calls else {
        return Ok(());
    };
    for call in calls {
        let function = call
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| "upstream returned a malformed tool call".to_string())?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| "upstream tool call did not include a function name".to_string())?;
        let Some(schema) = strict_tools.get(name) else {
            continue;
        };
        let arguments = match function
            .get("arguments")
            .ok_or_else(|| format!("strict tool {name:?} returned no arguments"))?
        {
            Value::String(arguments) => serde_json::from_str(arguments)
                .map_err(|error| format!("strict tool {name:?} returned invalid JSON: {error}"))?,
            Value::Object(_) => function["arguments"].clone(),
            _ => return Err(format!("strict tool {name:?} returned invalid arguments")),
        };
        if !jsonschema::is_valid(schema, &arguments) {
            return Err(format!(
                "strict tool {name:?} arguments did not match its JSON Schema"
            ));
        }
    }
    Ok(())
}

pub fn structured_retry_body(body: &Value, invalid_response: &Value) -> Value {
    let mut retried = body.clone();
    retried["stream"] = Value::Bool(false);
    retried
        .as_object_mut()
        .map(|body| body.remove("stream_options"));
    let invalid = truncate_utf8(
        completion_text(invalid_response).unwrap_or("[missing output]"),
        16 * 1024,
    );
    if let Some(messages) = retried.get_mut("messages").and_then(Value::as_array_mut) {
        messages.push(json!({"role":"assistant", "content":invalid}));
        messages.push(json!({
            "role":"user",
            "content":"The previous response failed strict JSON Schema validation. Return a corrected JSON object only. Do not use Markdown or explanatory text."
        }));
    }
    retried
}

pub fn strict_tool_retry_body(body: &Value, invalid_response: &Value) -> Value {
    let mut retried = body.clone();
    retried["stream"] = Value::Bool(false);
    retried
        .as_object_mut()
        .map(|body| body.remove("stream_options"));
    if let Some(messages) = retried.get_mut("messages").and_then(Value::as_array_mut) {
        if let Some(tool_calls) = invalid_response
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("tool_calls"))
        {
            let serialized_tool_calls = tool_calls.to_string();
            let attempted = truncate_utf8(&serialized_tool_calls, 16 * 1024);
            messages.push(json!({
                "role":"assistant",
                "content":format!("Previous invalid tool-call attempt: {attempted}")
            }));
        }
        messages.push(json!({
            "role":"user",
            "content":"The previous tool call failed strict JSON Schema validation. Call the intended tool again with corrected JSON arguments that exactly match its schema."
        }));
    }
    retried
}

fn truncate_utf8(value: &str, limit: usize) -> &str {
    if value.len() <= limit {
        return value;
    }
    let mut end = limit;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn completion_text(value: &Value) -> Option<&str> {
    let content = value
        .get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?;
    if let Some(text) = content.as_str() {
        return Some(text);
    }
    content
        .as_array()?
        .iter()
        .find_map(|part| part.get("text").and_then(Value::as_str))
}

pub fn compaction_request_body(body: &Value, config: &CompactionConfig) -> Value {
    let mut summary = body.clone();
    summary["stream"] = Value::Bool(false);
    summary["max_tokens"] = Value::from(4096);
    if let Some(object) = summary.as_object_mut() {
        for key in [
            "stream_options",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "metadata",
            "stop",
        ] {
            object.remove(key);
        }
    }
    let instructions = config.instructions.as_deref().unwrap_or(
        "Write a continuation summary that preserves the user's request, constraints, completed work, exact file names and code decisions, failures and lessons, current state, and concrete next steps. Wrap the summary in <summary></summary> tags. Do not call tools; return text only.",
    );
    if let Some(messages) = summary.get_mut("messages").and_then(Value::as_array_mut) {
        remove_structured_output_instructions(messages);
        messages.push(json!({
            "role":"user",
            "content":format!("Create the compaction summary now.\n\n{instructions}")
        }));
    }
    summary
}

fn remove_structured_output_instructions(messages: &mut Vec<Value>) {
    for message in messages
        .iter_mut()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
    {
        match message.get_mut("content") {
            Some(Value::String(content)) => {
                if let Some(start) = content.find(STRUCTURED_OUTPUT_INSTRUCTION_PREFIX) {
                    content.truncate(start);
                    *content = content.trim_end().to_owned();
                }
            }
            Some(Value::Array(blocks)) => blocks.retain(|block| {
                !block
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.starts_with(STRUCTURED_OUTPUT_INSTRUCTION_PREFIX))
            }),
            _ => {}
        }
    }
    messages.retain(|message| {
        if message.get("role").and_then(Value::as_str) != Some("system") {
            return true;
        }
        match message.get("content") {
            Some(Value::String(content)) => !content.is_empty(),
            Some(Value::Array(blocks)) => !blocks.is_empty(),
            _ => true,
        }
    });
}

pub fn extract_compaction_summary(value: &Value) -> Result<String, ProtocolError> {
    let content = completion_text(value)
        .filter(|content| !content.trim().is_empty())
        .ok_or_else(|| ProtocolError::invalid("upstream compaction did not return summary text"))?;
    let content = content.trim();
    match (content.find("<summary>"), content.find("</summary>")) {
        (Some(start), Some(end)) if end >= start + "<summary>".len() => {
            let summary = content[start + "<summary>".len()..end].trim();
            if summary.is_empty() {
                Err(ProtocolError::invalid(
                    "upstream compaction returned an empty summary",
                ))
            } else {
                Ok(summary.to_owned())
            }
        }
        (None, None) => Ok(content.to_owned()),
        _ => Err(ProtocolError::invalid(
            "upstream compaction returned malformed summary tags",
        )),
    }
}

pub fn apply_compaction_summary(body: &mut Value, summary: &str) {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    let mut compacted: Vec<Value> = messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
        .cloned()
        .collect();
    compacted.push(json!({
        "role":"user",
        "content":format!(
            "The earlier conversation was compacted by the gateway. Continue the task directly from this authoritative state summary:\n\n{summary}"
        )
    }));
    *messages = compacted;
}

#[derive(Debug, Clone)]
pub struct CompactionPrelude {
    pub content: String,
    pub usage: Value,
}

pub fn compaction_pause_response(
    request_id: &str,
    model: &str,
    prelude: &CompactionPrelude,
    context_management: Option<&ContextManagementResult>,
) -> Value {
    let mut response = json!({
        "id":request_id,
        "type":"message",
        "role":"assistant",
        "content":[{"type":"compaction", "content":prelude.content}],
        "model":model,
        "stop_reason":"compaction",
        "stop_sequence":null,
        "usage":usage_with_iterations(None, Some(prelude))
    });
    if let Some(context_management) = context_management {
        response["context_management"] = context_management.response_value();
    }
    response
}

fn append_system_instruction(
    output: &mut Map<String, Value>,
    instruction: String,
) -> Result<(), ProtocolError> {
    let messages = output
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProtocolError::invalid("converted messages must be an array"))?;
    if let Some(system) = messages
        .iter_mut()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("system"))
    {
        match system.get_mut("content") {
            Some(Value::String(content)) => {
                content.push_str("\n\n");
                content.push_str(&instruction);
            }
            Some(Value::Array(content)) => {
                content.push(json!({"type":"text", "text":instruction}));
            }
            _ => {
                return Err(ProtocolError::invalid(
                    "converted system content must be a string or array",
                ))
            }
        }
    } else {
        messages.insert(0, json!({"role":"system", "content":instruction}));
    }
    Ok(())
}

fn preserve_cache_control(
    source: &Map<String, Value>,
    destination: &mut Value,
) -> Result<(), ProtocolError> {
    if let Some(cache) = source.get("cache_control") {
        if !cache.is_object() {
            return Err(ProtocolError::invalid("cache_control must be an object"));
        }
        destination["cache_control"] = cache.clone();
    }
    Ok(())
}

fn copy_probability(
    input: &Map<String, Value>,
    output: &mut Map<String, Value>,
    key: &str,
) -> Result<(), ProtocolError> {
    if let Some(value) = input.get(key) {
        let number = value.as_f64().filter(|number| (0.0..=1.0).contains(number));
        if number.is_none() {
            return Err(ProtocolError::invalid(format!(
                "{key} must be a number between 0 and 1"
            )));
        }
        output.insert(key.into(), value.clone());
    }
    Ok(())
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
) -> Result<&'a str, ProtocolError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ProtocolError::invalid(format!("{key} must be a non-empty string")))
}

fn reject_unknown_keys(
    object: &Map<String, Value>,
    allowed: &[&str],
    context: &str,
) -> Result<(), ProtocolError> {
    let allowed: HashSet<&str> = allowed.iter().copied().collect();
    if let Some(key) = object.keys().find(|key| !allowed.contains(key.as_str())) {
        return Err(ProtocolError::invalid(format!(
            "unsupported {context} field {key:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
pub fn convert_response(
    value: &Value,
    request_id: &str,
    fallback_model: &str,
) -> Result<Value, ProtocolError> {
    convert_response_with_context(
        value,
        request_id,
        fallback_model,
        ThinkingDisplay::Summarized,
        None,
        None,
    )
}

pub fn convert_response_with_context(
    value: &Value,
    request_id: &str,
    fallback_model: &str,
    thinking_display: ThinkingDisplay,
    compaction: Option<&CompactionPrelude>,
    context_management: Option<&ContextManagementResult>,
) -> Result<Value, ProtocolError> {
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
        .ok_or_else(|| ProtocolError::invalid("upstream response has no choices"))?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| ProtocolError::invalid("upstream response has no message"))?;
    let mut content = Vec::new();
    if let Some(compaction) = compaction {
        content.push(json!({"type":"compaction", "content":compaction.content}));
    }
    if let Some(reasoning) = message
        .get("reasoning_content")
        .or_else(|| message.get("reasoning"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        content.push(json!({
            "type":"thinking",
            "thinking":if thinking_display == ThinkingDisplay::Omitted {
                ""
            } else {
                SAFE_THINKING_SUMMARY
            },
            "signature":thinking_signature(reasoning, request_id)
        }));
    }
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            content.push(json!({"type":"text", "text":text}));
        }
    } else if let Some(parts) = message.get("content").and_then(Value::as_array) {
        for part in parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                content.push(json!({"type":"text", "text":text}));
            } else if part.get("type").and_then(Value::as_str) == Some("reasoning") {
                if let Some(thinking) = part
                    .get("thinking")
                    .or_else(|| part.get("text"))
                    .and_then(Value::as_str)
                {
                    content.push(json!({
                        "type":"thinking",
                        "thinking":if thinking_display == ThinkingDisplay::Omitted {
                            ""
                        } else {
                            SAFE_THINKING_SUMMARY
                        },
                        "signature":thinking_signature(thinking, request_id)
                    }));
                }
            }
        }
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            let function = call
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| ProtocolError::invalid("upstream returned a malformed tool call"))?;
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| ProtocolError::invalid("upstream tool call has no id"))?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| ProtocolError::invalid("upstream tool call has no name"))?;
            let input = match function.get("arguments") {
                Some(Value::String(arguments)) => {
                    serde_json::from_str(arguments).map_err(|error| {
                        ProtocolError::invalid(format!(
                            "upstream tool call arguments were not valid JSON: {error}"
                        ))
                    })?
                }
                Some(Value::Object(_)) => function["arguments"].clone(),
                _ => {
                    return Err(ProtocolError::invalid(
                        "upstream tool call has invalid arguments",
                    ))
                }
            };
            content.push(json!({
                "type":"tool_use",
                "id":id,
                "name":name,
                "input":input
            }));
        }
    }
    let finish = choice.get("finish_reason").and_then(Value::as_str);
    let stop_sequence = choice.get("stop_sequence").cloned().unwrap_or(Value::Null);
    let mut response = json!({
        "id":value.get("id").and_then(Value::as_str).unwrap_or(request_id),
        "type":"message",
        "role":"assistant",
        "content":content,
        "model":value.get("model").and_then(Value::as_str).unwrap_or(fallback_model),
        "stop_reason":map_stop_reason(finish),
        "stop_sequence":stop_sequence,
        "usage":usage_with_iterations(value.get("usage"), compaction)
    });
    if let Some(context_management) = context_management {
        response["context_management"] = context_management.response_value();
    }
    Ok(response)
}

fn convert_usage(usage: Option<&Value>) -> Value {
    let usage = usage.and_then(Value::as_object);
    let prompt_tokens = usage
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(Value::as_u64);
    let output = usage
        .and_then(|usage| {
            usage
                .get("completion_tokens")
                .or_else(|| usage.get("output_tokens"))
        })
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = usage
        .and_then(|usage| {
            usage.get("cache_read_input_tokens").or_else(|| {
                usage
                    .get("prompt_tokens_details")
                    .and_then(|details| details.get("cached_tokens"))
            })
        })
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_creation = usage
        .and_then(|usage| usage.get("cache_creation_input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let input = prompt_tokens
        .map(|total| total.saturating_sub(cached).saturating_sub(cache_creation))
        .or_else(|| {
            usage
                .and_then(|usage| usage.get("input_tokens"))
                .and_then(Value::as_u64)
        })
        .unwrap_or(0);
    let mut converted = json!({
        "input_tokens":input,
        "output_tokens":output,
        "cache_creation_input_tokens":cache_creation,
        "cache_read_input_tokens":cached
    });
    if let Some(cache_creation_details) = usage.and_then(|usage| usage.get("cache_creation")) {
        converted["cache_creation"] = cache_creation_details.clone();
    } else {
        converted["cache_creation"] = json!({
            "ephemeral_5m_input_tokens":cache_creation,
            "ephemeral_1h_input_tokens":0
        });
    }
    if let Some(service_tier) = usage.and_then(|usage| usage.get("service_tier")) {
        converted["service_tier"] = service_tier.clone();
    }
    if let Some(inference_geo) = usage.and_then(|usage| usage.get("inference_geo")) {
        converted["inference_geo"] = inference_geo.clone();
    }
    converted
}

fn usage_iteration(usage: Option<&Value>, kind: &str) -> Value {
    let mut iteration = convert_usage(usage);
    iteration["type"] = Value::String(kind.into());
    iteration
}

fn usage_with_iterations(
    message_usage: Option<&Value>,
    compaction: Option<&CompactionPrelude>,
) -> Value {
    let mut usage = convert_usage(message_usage);
    let mut iterations = Vec::new();
    if let Some(compaction) = compaction {
        iterations.push(usage_iteration(Some(&compaction.usage), "compaction"));
    }
    if message_usage.is_some() {
        iterations.push(usage_iteration(message_usage, "message"));
    }
    usage["iterations"] = Value::Array(iterations);
    usage
}

fn thinking_signature(reasoning: &str, request_id: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    reasoning.hash(&mut hasher);
    request_id.hash(&mut hasher);
    let signature = format!("mogick-v1-{:016x}", hasher.finish());
    if !reasoning.is_empty() {
        let mut store = thinking_signature_store()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if store.len() >= 4096 {
            if let Some(oldest) = store.keys().next().cloned() {
                store.remove(&oldest);
            }
        }
        store.insert(
            signature.clone(),
            truncate_utf8(reasoning, 256 * 1024).to_owned(),
        );
    }
    signature
}

fn restore_thinking_signature(signature: &str) -> Option<String> {
    if !signature.starts_with("mogick-v1-") {
        return None;
    }
    thinking_signature_store()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(signature)
        .cloned()
}

fn thinking_signature_store() -> &'static std::sync::Mutex<HashMap<String, String>> {
    static STORE: std::sync::OnceLock<std::sync::Mutex<HashMap<String, String>>> =
        std::sync::OnceLock::new();
    STORE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn map_stop_reason(reason: Option<&str>) -> Value {
    match reason {
        Some("length") | Some("max_tokens") => Value::String("max_tokens".into()),
        Some("tool_calls") | Some("function_call") => Value::String("tool_use".into()),
        Some("stop") | Some("end_turn") => Value::String("end_turn".into()),
        Some("content_filter") | Some("refusal") => Value::String("refusal".into()),
        Some("pause_turn") => Value::String("pause_turn".into()),
        Some(_) => Value::String("end_turn".into()),
        None => Value::Null,
    }
}

pub fn convert_models(value: &Value) -> Value {
    let models = value
        .get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let data: Vec<Value> = models.iter().filter_map(model_info).collect();
    let first_id = data
        .first()
        .and_then(|model| model.get("id"))
        .cloned()
        .unwrap_or(Value::Null);
    let last_id = data
        .last()
        .and_then(|model| model.get("id"))
        .cloned()
        .unwrap_or(Value::Null);
    json!({
        "data":data,
        "has_more":false,
        "first_id":first_id,
        "last_id":last_id
    })
}

pub fn convert_model(value: &Value, model_id: &str) -> Option<Value> {
    value
        .get("data")
        .and_then(Value::as_array)?
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
        .and_then(model_info)
}

fn model_info(model: &Value) -> Option<Value> {
    let id = model.get("id").and_then(Value::as_str)?;
    if id.to_ascii_lowercase().contains("embedding") {
        return None;
    }
    let lower_id = id.to_ascii_lowercase();
    let image_input = declared_model_capability(model, "image_input")
        .unwrap_or_else(|| lower_id.starts_with("mm-") || lower_id.contains("vision"));
    let pdf_input = declared_model_capability(model, "pdf_input")
        .unwrap_or_else(|| lower_id.contains("pdf") || lower_id.contains("document"));
    let reasoning = declared_model_capability(model, "thinking")
        .or_else(|| declared_model_capability(model, "reasoning"))
        .unwrap_or_else(|| {
            lower_id.contains("deepseek")
                || lower_id.contains("reasoner")
                || lower_id.contains("reasoning")
                || lower_id.contains("r1")
        });
    let max_input_tokens = model
        .get("max_input_tokens")
        .or_else(|| model.get("context_window"))
        .or_else(|| model.get("max_context_length"))
        .cloned()
        .unwrap_or(Value::Null);
    let max_tokens = model
        .get("max_tokens")
        .or_else(|| model.get("max_output_tokens"))
        .cloned()
        .unwrap_or(Value::Null);
    Some(json!({
        "type":"model",
        "id":id,
        "display_name":model.get("display_name").and_then(Value::as_str).unwrap_or(id),
        "created_at":model_created_at(model),
        "max_input_tokens":max_input_tokens,
        "max_tokens":max_tokens,
        "capabilities":{
            "batch":{"supported":false},
            "citations":{"supported":false},
            "code_execution":{"supported":false},
            "context_management":{
                "supported":true,
                "clear_thinking_20251015":{"supported":true},
                "clear_tool_uses_20250919":{"supported":true},
                "compact_20260112":{"supported":true}
            },
            "effort":{
                "supported":reasoning,
                "low":{"supported":reasoning},
                "medium":{"supported":reasoning},
                "high":{"supported":reasoning},
                "xhigh":{"supported":false},
                "max":{"supported":false}
            },
            "image_input":{"supported":image_input},
            "pdf_input":{"supported":pdf_input},
            "structured_outputs":{"supported":true},
            "thinking":{
                "supported":reasoning,
                "types":{
                    "adaptive":{"supported":reasoning},
                    "enabled":{"supported":reasoning}
                }
            }
        }
    }))
}

fn declared_model_capability(model: &Value, name: &str) -> Option<bool> {
    model
        .get("capabilities")
        .and_then(|capabilities| capabilities.get(name))
        .and_then(|capability| {
            capability
                .as_bool()
                .or_else(|| capability.get("supported").and_then(Value::as_bool))
        })
        .or_else(|| model.get(name).and_then(Value::as_bool))
}

fn model_created_at(model: &Value) -> String {
    if let Some(created_at) = model.get("created_at").and_then(Value::as_str) {
        return created_at.to_owned();
    }
    model
        .get("created")
        .and_then(Value::as_i64)
        .and_then(|timestamp| chrono::DateTime::from_timestamp(timestamp, 0))
        .map(|created| created.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".into())
}

pub fn error_envelope(error_type: &str, message: impl Into<String>, request_id: &str) -> Value {
    json!({
        "type":"error",
        "error":{"type":error_type, "message":message.into()},
        "request_id":request_id
    })
}

pub fn stream_body_with_context(
    response: reqwest::Response,
    request_id: String,
    fallback_model: String,
    thinking_display: ThinkingDisplay,
    compaction: Option<CompactionPrelude>,
    context_management: Option<ContextManagementResult>,
) -> Body {
    let output = async_stream::stream! {
        let mut upstream = response.bytes_stream();
        let mut decoder = SseDecoder::default();
        let mut state = StreamState::new(
            request_id,
            fallback_model,
            thinking_display,
            compaction,
            context_management,
        );
        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(chunk) => {
                    let events = match decoder.push(&chunk) {
                        Ok(events) => events,
                        Err(()) => {
                            yield Ok(sse_frame("error", error_envelope(
                                "api_error",
                                "upstream SSE event exceeded the gateway limit",
                                &state.request_id,
                            )));
                            return;
                        }
                    };
                    for event in events {
                        for frame in state.handle(&event.data) {
                            yield Ok::<Bytes, std::io::Error>(frame);
                        }
                        if state.terminal {
                            return;
                        }
                    }
                }
                Err(_) => {
                    yield Ok(sse_frame("error", error_envelope(
                        "api_error",
                        "upstream stream was interrupted",
                        &state.request_id,
                    )));
                    return;
                }
            }
        }
        let trailing_events = match decoder.finish() {
            Ok(events) => events,
            Err(()) => {
                yield Ok(sse_frame("error", error_envelope(
                    "api_error",
                    "upstream SSE event exceeded the gateway limit",
                    &state.request_id,
                )));
                return;
            }
        };
        for event in trailing_events {
            for frame in state.handle(&event.data) {
                yield Ok(frame);
            }
        }
        if !state.terminal {
            if state.finish_reason.is_some() {
                for frame in state.finalize() {
                    yield Ok(frame);
                }
            } else {
                yield Ok(sse_frame("error", error_envelope(
                    "api_error",
                    "upstream stream ended unexpectedly",
                    &state.request_id,
                )));
            }
        }
    };
    Body::from_stream(output)
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
    event: Option<String>,
    data_lines: Vec<String>,
    data_bytes: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct SseEvent {
    #[allow(dead_code)]
    event: Option<String>,
    data: String,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>, ()> {
        if self.buffer.len().saturating_add(bytes.len()) > MAX_UPSTREAM_SSE_EVENT_BYTES {
            self.buffer.clear();
            self.data_lines.clear();
            self.data_bytes = 0;
            return Err(());
        }
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=position).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.process_line(&String::from_utf8_lossy(&line), &mut events);
            if self.event_size() > MAX_UPSTREAM_SSE_EVENT_BYTES {
                self.data_lines.clear();
                self.data_bytes = 0;
                return Err(());
            }
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<SseEvent>, ()> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let line = String::from_utf8_lossy(&std::mem::take(&mut self.buffer)).into_owned();
            self.process_line(&line, &mut events);
        }
        if self.event_size() > MAX_UPSTREAM_SSE_EVENT_BYTES {
            self.data_lines.clear();
            self.data_bytes = 0;
            return Err(());
        }
        self.dispatch(&mut events);
        Ok(events)
    }

    fn process_line(&mut self, line: &str, events: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(events);
        } else if line.starts_with(':') {
            // SSE comment / keepalive.
        } else if let Some(value) = line.strip_prefix("data:") {
            let value = value.strip_prefix(' ').unwrap_or(value);
            self.data_bytes = self.data_bytes.saturating_add(value.len());
            self.data_lines.push(value.to_string());
        } else if let Some(value) = line.strip_prefix("event:") {
            self.event = Some(value.strip_prefix(' ').unwrap_or(value).to_string());
        }
    }

    fn dispatch(&mut self, events: &mut Vec<SseEvent>) {
        if !self.data_lines.is_empty() {
            events.push(SseEvent {
                event: self.event.take(),
                data: self.data_lines.join("\n"),
            });
            self.data_lines.clear();
            self.data_bytes = 0;
        } else {
            self.event = None;
        }
    }

    fn event_size(&self) -> usize {
        self.data_bytes
    }
}

struct StreamState {
    request_id: String,
    fallback_model: String,
    started: bool,
    terminal: bool,
    blocks: Vec<BlockKind>,
    text_index: Option<usize>,
    thinking_index: Option<usize>,
    tools: HashMap<u64, usize>,
    finish_reason: Option<String>,
    stop_sequence: Value,
    usage: Value,
    thinking_display: ThinkingDisplay,
    thinking_content: String,
    thinking_summary_emitted: bool,
    compaction: Option<CompactionPrelude>,
    context_management: Option<ContextManagementResult>,
}

#[derive(Clone, Copy)]
enum BlockKind {
    Text,
    Thinking,
    Tool,
    Compaction,
}

impl StreamState {
    fn new(
        request_id: String,
        fallback_model: String,
        thinking_display: ThinkingDisplay,
        compaction: Option<CompactionPrelude>,
        context_management: Option<ContextManagementResult>,
    ) -> Self {
        Self {
            request_id,
            fallback_model,
            started: false,
            terminal: false,
            blocks: Vec::new(),
            text_index: None,
            thinking_index: None,
            tools: HashMap::new(),
            finish_reason: None,
            stop_sequence: Value::Null,
            usage: json!({}),
            thinking_display,
            thinking_content: String::new(),
            thinking_summary_emitted: false,
            compaction,
            context_management,
        }
    }

    fn handle(&mut self, data: &str) -> Vec<Bytes> {
        if data.trim() == "[DONE]" {
            if self.finish_reason.is_some() {
                return self.finalize();
            }
            self.terminal = true;
            return vec![sse_frame(
                "error",
                error_envelope(
                    "api_error",
                    "upstream stream ended without a stop reason",
                    &self.request_id,
                ),
            )];
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => {
                self.terminal = true;
                return vec![sse_frame(
                    "error",
                    error_envelope(
                        "api_error",
                        "upstream sent invalid SSE JSON",
                        &self.request_id,
                    ),
                )];
            }
        };
        if value.get("error").is_some() {
            self.terminal = true;
            return vec![sse_frame(
                "error",
                error_envelope("api_error", "upstream stream error", &self.request_id),
            )];
        }
        let mut frames = Vec::new();
        if !self.started {
            self.started = true;
            let id = value
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or(&self.request_id);
            let model = value
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(&self.fallback_model);
            frames.push(sse_frame(
                "message_start",
                json!({"type":"message_start", "message":{
                    "id":id, "type":"message", "role":"assistant", "content":[],
                    "model":model, "stop_reason":null, "stop_sequence":null,
                    "usage":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}
                }}),
            ));
            if let Some(content) = self
                .compaction
                .as_ref()
                .map(|compaction| compaction.content.clone())
            {
                let index = self.blocks.len();
                self.blocks.push(BlockKind::Compaction);
                frames.push(sse_frame(
                    "content_block_start",
                    json!({"type":"content_block_start", "index":index,
                        "content_block":{"type":"compaction", "content":""}}),
                ));
                frames.push(sse_frame(
                    "content_block_delta",
                    json!({"type":"content_block_delta", "index":index,
                        "delta":{"type":"compaction_delta", "content":content}}),
                ));
                frames.push(sse_frame(
                    "content_block_stop",
                    json!({"type":"content_block_stop", "index":index}),
                ));
            }
        }
        if let Some(usage) = value.get("usage") {
            self.usage = usage.clone();
        }
        if let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        {
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.finish_reason = Some(reason.into());
            }
            if let Some(sequence) = choice.get("stop_sequence") {
                self.stop_sequence = sequence.clone();
            }
            if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
                if let Some(thinking) = delta
                    .get("reasoning_content")
                    .or_else(|| delta.get("reasoning"))
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                {
                    self.thinking_content.push_str(thinking);
                    let index = self.ensure_thinking(&mut frames);
                    if self.thinking_display == ThinkingDisplay::Summarized
                        && !self.thinking_summary_emitted
                    {
                        self.thinking_summary_emitted = true;
                        frames.push(sse_frame(
                            "content_block_delta",
                            json!({"type":"content_block_delta", "index":index,
                                "delta":{"type":"thinking_delta",
                                    "thinking":SAFE_THINKING_SUMMARY}}),
                        ));
                    }
                }
                if let Some(text) = delta
                    .get("content")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                {
                    let index = self.ensure_text(&mut frames);
                    frames.push(sse_frame(
                        "content_block_delta",
                        json!({"type":"content_block_delta", "index":index,
                            "delta":{"type":"text_delta", "text":text}}),
                    ));
                }
                if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        let upstream_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
                        let function = call.get("function").and_then(Value::as_object);
                        let id = call
                            .get("id")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("toolu_{upstream_index}"));
                        let name = function
                            .and_then(|function| function.get("name"))
                            .and_then(Value::as_str)
                            .unwrap_or("unknown");
                        let index = self.ensure_tool(upstream_index, &id, name, &mut frames);
                        if let Some(arguments) = function
                            .and_then(|function| function.get("arguments"))
                            .and_then(Value::as_str)
                            .filter(|arguments| !arguments.is_empty())
                        {
                            frames.push(sse_frame(
                                "content_block_delta",
                                json!({"type":"content_block_delta", "index":index,
                                    "delta":{"type":"input_json_delta", "partial_json":arguments}}),
                            ));
                        }
                    }
                }
            }
        }
        frames
    }

    fn ensure_text(&mut self, frames: &mut Vec<Bytes>) -> usize {
        if let Some(index) = self.text_index {
            return index;
        }
        let index = self.blocks.len();
        self.blocks.push(BlockKind::Text);
        self.text_index = Some(index);
        frames.push(sse_frame(
            "content_block_start",
            json!({"type":"content_block_start", "index":index, "content_block":{"type":"text", "text":""}}),
        ));
        index
    }

    fn ensure_thinking(&mut self, frames: &mut Vec<Bytes>) -> usize {
        if let Some(index) = self.thinking_index {
            return index;
        }
        let index = self.blocks.len();
        self.blocks.push(BlockKind::Thinking);
        self.thinking_index = Some(index);
        frames.push(sse_frame(
            "content_block_start",
            json!({"type":"content_block_start", "index":index,
                "content_block":{"type":"thinking", "thinking":"", "signature":""}}),
        ));
        index
    }

    fn ensure_tool(
        &mut self,
        upstream_index: u64,
        id: &str,
        name: &str,
        frames: &mut Vec<Bytes>,
    ) -> usize {
        if let Some(index) = self.tools.get(&upstream_index) {
            return *index;
        }
        let index = self.blocks.len();
        self.blocks.push(BlockKind::Tool);
        self.tools.insert(upstream_index, index);
        frames.push(sse_frame(
            "content_block_start",
            json!({"type":"content_block_start", "index":index,
                "content_block":{"type":"tool_use", "id":id, "name":name, "input":{}}}),
        ));
        index
    }

    fn finalize(&mut self) -> Vec<Bytes> {
        if self.terminal {
            return Vec::new();
        }
        self.terminal = true;
        let mut frames = Vec::new();
        for index in 0..self.blocks.len() {
            if matches!(self.blocks[index], BlockKind::Compaction) {
                continue;
            }
            if matches!(self.blocks[index], BlockKind::Thinking) {
                frames.push(sse_frame(
                    "content_block_delta",
                    json!({"type":"content_block_delta", "index":index,
                        "delta":{"type":"signature_delta",
                            "signature":thinking_signature(&self.thinking_content, &self.request_id)}}),
                ));
            }
            frames.push(sse_frame(
                "content_block_stop",
                json!({"type":"content_block_stop", "index":index}),
            ));
        }
        let usage = usage_with_iterations(Some(&self.usage), self.compaction.as_ref());
        let mut message_delta = json!({"type":"message_delta", "delta":{
                "stop_reason":map_stop_reason(self.finish_reason.as_deref()),
                "stop_sequence":self.stop_sequence
            }, "usage":usage});
        if let Some(context_management) = &self.context_management {
            message_delta["context_management"] = context_management.response_value();
        }
        frames.push(sse_frame("message_delta", message_delta));
        frames.push(sse_frame("message_stop", json!({"type":"message_stop"})));
        frames
    }
}

fn sse_frame(event: &str, data: Value) -> Bytes {
    Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
}

pub fn synthetic_stream_body(message: Value) -> Body {
    let mut frames = Vec::new();
    let mut start_usage = message.get("usage").cloned().unwrap_or_else(|| json!({}));
    start_usage["output_tokens"] = Value::from(0);
    start_usage
        .as_object_mut()
        .map(|usage| usage.remove("iterations"));
    frames.push(sse_frame(
        "message_start",
        json!({"type":"message_start", "message":{
            "id":message.get("id").cloned().unwrap_or(Value::Null),
            "type":"message",
            "role":"assistant",
            "content":[],
            "model":message.get("model").cloned().unwrap_or(Value::Null),
            "stop_reason":null,
            "stop_sequence":null,
            "usage":start_usage
        }}),
    ));
    if let Some(content) = message.get("content").and_then(Value::as_array) {
        for (index, block) in content.iter().enumerate() {
            match block.get("type").and_then(Value::as_str) {
                Some("compaction") => {
                    frames.push(sse_frame(
                        "content_block_start",
                        json!({"type":"content_block_start", "index":index,
                            "content_block":{"type":"compaction", "content":""}}),
                    ));
                    frames.push(sse_frame(
                        "content_block_delta",
                        json!({"type":"content_block_delta", "index":index,
                            "delta":{"type":"compaction_delta",
                                "content":block.get("content").cloned().unwrap_or(Value::Null)}}),
                    ));
                }
                Some("thinking") => {
                    frames.push(sse_frame(
                        "content_block_start",
                        json!({"type":"content_block_start", "index":index,
                            "content_block":{"type":"thinking", "thinking":"", "signature":""}}),
                    ));
                    if let Some(thinking) = block
                        .get("thinking")
                        .and_then(Value::as_str)
                        .filter(|thinking| !thinking.is_empty())
                    {
                        frames.push(sse_frame(
                            "content_block_delta",
                            json!({"type":"content_block_delta", "index":index,
                                "delta":{"type":"thinking_delta", "thinking":thinking}}),
                        ));
                    }
                    if let Some(signature) = block.get("signature").and_then(Value::as_str) {
                        frames.push(sse_frame(
                            "content_block_delta",
                            json!({"type":"content_block_delta", "index":index,
                                "delta":{"type":"signature_delta", "signature":signature}}),
                        ));
                    }
                }
                Some("text") => {
                    frames.push(sse_frame(
                        "content_block_start",
                        json!({"type":"content_block_start", "index":index,
                            "content_block":{"type":"text", "text":""}}),
                    ));
                    frames.push(sse_frame(
                        "content_block_delta",
                        json!({"type":"content_block_delta", "index":index,
                            "delta":{"type":"text_delta",
                                "text":block.get("text").cloned().unwrap_or(Value::Null)}}),
                    ));
                }
                Some("tool_use") => {
                    frames.push(sse_frame(
                        "content_block_start",
                        json!({"type":"content_block_start", "index":index,
                            "content_block":{"type":"tool_use",
                                "id":block.get("id").cloned().unwrap_or(Value::Null),
                                "name":block.get("name").cloned().unwrap_or(Value::Null),
                                "input":{}}}),
                    ));
                    frames.push(sse_frame(
                        "content_block_delta",
                        json!({"type":"content_block_delta", "index":index,
                            "delta":{"type":"input_json_delta", "partial_json":
                                serde_json::to_string(block.get("input").unwrap_or(&json!({})))
                                    .unwrap_or_else(|_| "{}".into())}}),
                    ));
                }
                _ => continue,
            }
            frames.push(sse_frame(
                "content_block_stop",
                json!({"type":"content_block_stop", "index":index}),
            ));
        }
    }
    let mut message_delta = json!({"type":"message_delta", "delta":{
            "stop_reason":message.get("stop_reason").cloned().unwrap_or(Value::Null),
            "stop_sequence":message.get("stop_sequence").cloned().unwrap_or(Value::Null)
        }, "usage":message.get("usage").cloned().unwrap_or_else(|| json!({}))});
    if let Some(context_management) = message.get("context_management") {
        message_delta["context_management"] = context_management.clone();
    }
    frames.push(sse_frame("message_delta", message_delta));
    frames.push(sse_frame("message_stop", json!({"type":"message_stop"})));
    Body::from_stream(futures_util::stream::iter(
        frames.into_iter().map(Ok::<Bytes, std::io::Error>),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_golden_system_multimodal_tools_and_output() {
        let converted = convert_request(
            serde_json::to_vec(&json!({
                "model":"mm-any-future-model",
                "max_tokens":256,
                "system":[{"type":"text","text":"system"}],
                "messages":[
                    {"role":"user","content":[
                        {"type":"text","text":"look"},
                        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}},
                        {"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"}}
                    ]},
                    {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read","input":{"path":"x"}}]},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}
                ],
                "tools":[{"name":"read","description":"Read", "input_schema":{"type":"object"}}],
                "tool_choice":{"type":"any","disable_parallel_tool_use":true},
                "thinking":{"type":"adaptive","display":"omitted"},
                "context_management":{"edits":[
                    {"type":"clear_thinking_20251015","keep":"all"}
                ]},
                "output_config":{
                    "effort":"low",
                    "format":{"type":"json_schema","schema":{"type":"object"}}
                }
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        assert_eq!(converted.model, "mm-any-future-model");
        assert_eq!(converted.body["tool_choice"], "required");
        assert_eq!(converted.body["parallel_tool_calls"], false);
        assert_eq!(converted.body["response_format"]["type"], "json_object");
        assert!(converted.body["messages"][0]["content"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["text"]
            .as_str()
            .unwrap()
            .contains("JSON Schema"));
        let summary_body = compaction_request_body(
            &converted.body,
            &CompactionConfig {
                trigger_tokens: 50_000,
                pause_after_compaction: false,
                instructions: None,
                effective_input_tokens: 50_000,
            },
        );
        assert!(!summary_body
            .to_string()
            .contains(STRUCTURED_OUTPUT_INSTRUCTION_PREFIX));
        assert!(converted
            .body
            .to_string()
            .contains(STRUCTURED_OUTPUT_INSTRUCTION_PREFIX));
        assert_eq!(converted.body["reasoning_effort"], "low");
        assert!(converted.body.get("thinking").is_none());
        assert!(converted.body.get("context_management").is_none());
        assert_eq!(
            converted.body["messages"][2]["tool_calls"][0]["function"]["name"],
            "read"
        );
        assert_eq!(converted.body["messages"][3]["role"], "tool");
    }

    #[test]
    fn claude_code_title_schema_uses_supported_json_object_with_schema_instruction() {
        let converted = convert_request(
            serde_json::to_vec(&json!({
                "model":"deepseek-v4-flash",
                "max_tokens":32000,
                "stream":true,
                "system":[
                    {"type":"text","text":"Generate a concise title."},
                    {"type":"text","text":"Use the user's language."}
                ],
                "messages":[{"role":"user","content":[{"type":"text","text":"Fix it"}]}],
                "output_config":{
                    "effort":"high",
                    "format":{
                        "type":"json_schema",
                        "schema":{
                            "additionalProperties":false,
                            "properties":{"title":{"type":"string"}},
                            "required":["title"],
                            "type":"object"
                        }
                    }
                }
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();

        assert_eq!(
            converted.body["response_format"],
            json!({"type":"json_object"})
        );
        assert_eq!(converted.body["reasoning_effort"], "high");
        let system_blocks = converted.body["messages"][0]["content"]
            .as_array()
            .expect("system blocks");
        assert_eq!(system_blocks.len(), 3);
        let schema_instruction = system_blocks[2]["text"].as_str().unwrap();
        assert!(schema_instruction.contains("\"required\":[\"title\"]"));
        assert!(schema_instruction.contains("text outside the JSON object"));
    }

    #[test]
    fn unknown_request_fields_fail_explicitly() {
        let error =
            convert_request(br#"{"model":"x","max_tokens":1,"messages":[],"future_beta":true}"#)
                .unwrap_err();
        assert_eq!(error.error_type, "invalid_request_error");
        assert!(error.message.contains("future_beta"));
    }

    #[test]
    fn claude_code_context_fields_are_consumed_and_thinking_turns_are_retained() {
        let converted = convert_request(
            serde_json::to_vec(&json!({
                "model":"deepseek-v4-flash",
                "max_tokens":32000,
                "stream":true,
                "messages":[
                    {"role":"assistant","content":[
                        {"type":"thinking","thinking":"old reasoning","signature":"old"},
                        {"type":"text","text":"old answer"}
                    ]},
                    {"role":"user","content":"continue"},
                    {"role":"assistant","content":[
                        {"type":"thinking","thinking":"recent reasoning","signature":"recent"},
                        {"type":"text","text":"recent answer"}
                    ]},
                    {"role":"user","content":"continue again"}
                ],
                "tools":[],
                "thinking":{"type":"adaptive","display":"omitted"},
                "output_config":{"effort":"max"},
                "context_management":{"edits":[
                    {"type":"clear_thinking_20251015","keep":{
                        "type":"thinking_turns","value":1
                    }},
                    {"type":"clear_tool_uses_20250919","trigger":{
                        "type":"input_tokens","value":50000
                    },"keep":{"type":"tool_uses","value":5}}
                ]}
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        assert!(converted.body.get("thinking").is_none());
        assert!(converted.body.get("context_management").is_none());
        assert_eq!(converted.body["reasoning_effort"], "high");
        assert!(converted.body["messages"][0]
            .get("reasoning_content")
            .is_none());
        assert_eq!(
            converted.body["messages"][2]["reasoning_content"],
            "recent reasoning"
        );
        assert_eq!(
            converted.context_management.as_ref().unwrap().applied_edits[0]
                ["cleared_thinking_turns"],
            1
        );
    }

    #[test]
    fn unsupported_context_management_is_rejected_explicitly() {
        let error = convert_request(
            br#"{"model":"x","max_tokens":1,"messages":[],"context_management":{"edits":[{"type":"future_edit"}]}}"#,
        )
        .unwrap_err();
        assert!(error.message.contains("future_edit"));
    }

    #[test]
    fn count_tokens_and_current_claude_code_advisory_fields_are_accepted() {
        let converted = convert_count_request(
            serde_json::to_vec(&json!({
                "model":"model-x",
                "messages":[{"role":"user","content":"count me"}],
                "service_tier":"auto",
                "speed":"fast",
                "diagnostics":{"previous_message_id":"msg_previous"},
                "cache_control":{"type":"ephemeral","evict_on_complete":true},
                "fallbacks":"default",
                "betas":["context-management-2025-06-27"],
                "anthropic_beta":["effort-2025-11-24"],
                "output_config":{
                    "effort":"medium",
                    "task_budget":{"type":"tokens","total":20000,"remaining":10000}
                }
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        assert_eq!(converted.body["max_tokens"], 1);
        assert_eq!(converted.body["reasoning_effort"], "medium");
        for key in [
            "service_tier",
            "speed",
            "diagnostics",
            "cache_control",
            "fallbacks",
            "betas",
            "anthropic_beta",
            "task_budget",
        ] {
            assert!(converted.body.get(key).is_none(), "forwarded {key}");
        }
    }

    #[test]
    fn context_management_clears_old_tool_results_and_preserves_recent_ones() {
        let converted = convert_request(
            serde_json::to_vec(&json!({
                "model":"model-x",
                "max_tokens":100,
                "messages":[
                    {"role":"assistant","content":[{"type":"tool_use","id":"old","name":"Read","input":{"file_path":"old"}}]},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"old","content":"old contents"}]},
                    {"role":"system","content":[{"type":"text","text":"per-turn instruction"}]},
                    {"role":"assistant","content":[{"type":"tool_use","id":"recent","name":"Read","input":{"file_path":"recent"}}]},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"recent","content":"recent contents"}]}
                ],
                "context_management":{"edits":[{
                    "type":"clear_tool_uses_20250919",
                    "trigger":{"type":"tool_uses","value":1},
                    "keep":{"type":"tool_uses","value":1},
                    "clear_tool_inputs":true
                }]}
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        assert!(
            converted.body["messages"][0]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap()
                .contains("tool input cleared")
        );
        assert_eq!(
            converted.body["messages"][1]["content"],
            "[Old tool result cleared by context management]"
        );
        assert_eq!(converted.body["messages"][2]["role"], "system");
        assert!(
            converted.body["messages"][3]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap()
                .contains("recent")
        );
        assert_eq!(converted.body["messages"][4]["content"], "recent contents");
        let context = converted.context_management.as_ref().unwrap();
        assert_eq!(context.applied_edits.len(), 1);
        assert_eq!(context.applied_edits[0]["type"], "clear_tool_uses_20250919");
        assert_eq!(context.applied_edits[0]["cleared_tool_uses"], 1);
    }

    #[test]
    fn compaction_trigger_uses_post_edit_context_and_keeps_token_counts_distinct() {
        let obsolete_history = "o".repeat(220_000);
        let old_tool_result = "r".repeat(220_000);
        let converted = convert_request(
            serde_json::to_vec(&json!({
                "model":"model-x",
                "max_tokens":100,
                "messages":[
                    {"role":"user","content":obsolete_history},
                    {"role":"assistant","content":[
                        {"type":"compaction","content":"authoritative summary"}
                    ]},
                    {"role":"assistant","content":[
                        {"type":"tool_use","id":"old","name":"Read","input":{"path":"large"}},
                        {"type":"tool_use","id":"recent","name":"Read","input":{"path":"small"}}
                    ]},
                    {"role":"user","content":[
                        {"type":"tool_result","tool_use_id":"old","content":old_tool_result},
                        {"type":"tool_result","tool_use_id":"recent","content":"keep me"}
                    ]}
                ],
                "context_management":{"edits":[
                    {
                        "type":"clear_tool_uses_20250919",
                        "trigger":{"type":"tool_uses","value":1},
                        "keep":{"type":"tool_uses","value":1}
                    },
                    {
                        "type":"compact_20260112",
                        "trigger":{"type":"input_tokens","value":50000}
                    }
                ]}
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();

        let context = converted.context_management.as_ref().unwrap();
        assert!(context.original_input_tokens > context.effective_input_tokens_before_edits);
        assert!(context.effective_input_tokens_before_edits >= 50_000);
        assert!(context.effective_input_tokens < 50_000);
        assert!(context.original_input_tokens_for(100) > 100);
        assert_eq!(context.applied_edits[0]["cleared_tool_uses"], 1);
        assert!(!converted.compaction.as_ref().unwrap().should_compact());
        assert!(!converted.body.to_string().contains(&"o".repeat(1024)));
        assert!(!converted.body.to_string().contains(&"r".repeat(1024)));
    }

    #[test]
    fn conversation_can_compact_again_after_an_existing_compaction_block() {
        let converted = convert_request(
            serde_json::to_vec(&json!({
                "model":"model-x",
                "max_tokens":100,
                "messages":[
                    {"role":"user","content":"obsolete"},
                    {"role":"assistant","content":[
                        {"type":"compaction","content":"first summary"}
                    ]},
                    {"role":"user","content":"n".repeat(220_000)}
                ],
                "context_management":{"edits":[{
                    "type":"compact_20260112",
                    "trigger":{"type":"input_tokens","value":50000}
                }]}
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        let context = converted.context_management.as_ref().unwrap();
        assert!(context.original_input_tokens > context.effective_input_tokens_before_edits);
        assert_eq!(
            context.effective_input_tokens_before_edits,
            context.effective_input_tokens
        );
        assert!(converted.compaction.as_ref().unwrap().should_compact());
    }

    #[test]
    fn base64_image_transport_size_does_not_trigger_text_compaction() {
        let converted = convert_request(
            serde_json::to_vec(&json!({
                "model":"vision-model",
                "max_tokens":100,
                "messages":[{"role":"user","content":[{
                    "type":"image",
                    "source":{
                        "type":"base64",
                        "media_type":"image/png",
                        "data":"A".repeat(300_000)
                    }
                }]}],
                "context_management":{"edits":[{
                    "type":"compact_20260112",
                    "trigger":{"type":"input_tokens","value":50000}
                }]}
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        assert!(
            converted
                .context_management
                .as_ref()
                .unwrap()
                .effective_input_tokens
                < 50_000
        );
        assert!(!converted.compaction.as_ref().unwrap().should_compact());
    }

    #[test]
    fn compaction_configuration_and_existing_blocks_preserve_only_effective_context() {
        let converted = convert_request(
            serde_json::to_vec(&json!({
                "model":"model-x",
                "max_tokens":100,
                "messages":[
                    {"role":"user","content":"obsolete history"},
                    {"role":"assistant","content":[
                        {"type":"compaction","content":"authoritative summary","cache_control":{"type":"ephemeral"}},
                        {"type":"text","text":"post-summary answer"}
                    ]},
                    {"role":"user","content":"continue"}
                ],
                "context_management":{"edits":[{
                    "type":"compact_20260112",
                    "trigger":{"type":"input_tokens","value":50000},
                    "pause_after_compaction":true,
                    "instructions":"Preserve code details."
                }]}
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        let compaction = converted.compaction.as_ref().unwrap();
        assert_eq!(compaction.trigger_tokens, 50_000);
        assert!(compaction.pause_after_compaction);
        assert!(!compaction.should_compact());
        assert_eq!(converted.body["messages"].as_array().unwrap().len(), 3);
        assert!(converted.body["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("authoritative summary"));
        assert!(!converted.body.to_string().contains("obsolete history"));
        assert_eq!(
            converted.body["messages"][1]["content"][0]["text"],
            "post-summary answer"
        );

        let summary_request = compaction_request_body(&converted.body, compaction);
        assert_eq!(summary_request["stream"], false);
        assert_eq!(summary_request["max_tokens"], 4096);
        assert!(summary_request
            .to_string()
            .contains("Preserve code details"));
    }

    #[test]
    fn structured_output_is_locally_validated_and_repair_request_is_non_streaming() {
        let converted = convert_request(
            br#"{"model":"x","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"title"}],"output_config":{"format":{"type":"json_schema","schema":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"],"additionalProperties":false}}}}"#,
        )
        .unwrap();
        let format = converted.structured_output.as_ref().unwrap();
        let valid = json!({"choices":[{"message":{"content":"{\"title\":\"ok\"}"}}]});
        let invalid = json!({"choices":[{"message":{"content":"{\"wrong\":1}"}}]});
        assert!(validate_structured_response(&valid, format).is_ok());
        assert!(validate_structured_response(&invalid, format).is_err());
        let retry = structured_retry_body(&converted.body, &invalid);
        assert_eq!(retry["stream"], false);
        assert!(retry.get("stream_options").is_none());
        assert!(retry.to_string().contains("strict JSON Schema validation"));

        let strict_tools = HashMap::from([(
            "Read".into(),
            json!({
                "type":"object",
                "properties":{"file_path":{"type":"string"}},
                "required":["file_path"],
                "additionalProperties":false
            }),
        )]);
        let valid_call = json!({"choices":[{"message":{"tool_calls":[{
            "function":{"name":"Read","arguments":"{\"file_path\":\"x\"}"}
        }]}}]});
        let invalid_call = json!({"choices":[{"message":{"tool_calls":[{
            "function":{"name":"Read","arguments":"{\"path\":1}"}
        }]}}]});
        assert!(validate_strict_tool_response(&valid_call, &strict_tools).is_ok());
        assert!(validate_strict_tool_response(&invalid_call, &strict_tools).is_err());
        let tool_retry = strict_tool_retry_body(&converted.body, &invalid_call);
        let retry_messages = tool_retry["messages"].as_array().unwrap();
        assert!(retry_messages[retry_messages.len() - 2]
            .get("tool_calls")
            .is_none());
        assert!(retry_messages[retry_messages.len() - 2]["content"]
            .as_str()
            .unwrap()
            .contains("invalid tool-call attempt"));

        let malformed_call = json!({"choices":[{"message":{"tool_calls":[{
            "id":"broken","function":{"name":"Read","arguments":"not-json"}
        }]},"finish_reason":"tool_calls"}]});
        assert!(validate_strict_tool_response(&malformed_call, &strict_tools).is_err());
        assert!(convert_response(&malformed_call, "req", "model").is_err());
        assert!(extract_compaction_summary(&json!({
            "choices":[{"message":{"content":"   "}}]
        }))
        .is_err());
        assert!(extract_compaction_summary(&json!({
            "choices":[{"message":{"content":"<summary>unfinished"}}]
        }))
        .is_err());
        assert_eq!(
            extract_compaction_summary(&json!({
                "choices":[{"message":{"content":"<summary>state</summary>"}}]
            }))
            .unwrap(),
            "state"
        );
    }

    #[test]
    fn response_golden_includes_reasoning_tools_and_cache_usage() {
        let response = convert_response(
            &json!({
                "id":"chat_1","model":"model-x",
                "choices":[{"message":{"content":"answer","reasoning_content":"think","tool_calls":[
                    {"id":"call_1","function":{"name":"read","arguments":"{\"path\":\"x\"}"}}
                ]},"finish_reason":"tool_calls"}],
                "usage":{"prompt_tokens":12,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":5}}
            }),
            "req_1",
            "fallback",
        )
        .unwrap();
        assert_eq!(response["type"], "message");
        assert_eq!(response["content"][0]["type"], "thinking");
        assert_eq!(response["content"][0]["thinking"], SAFE_THINKING_SUMMARY);
        assert_ne!(response["content"][0]["thinking"], "think");
        assert_eq!(response["content"][2]["type"], "tool_use");
        assert_eq!(response["stop_reason"], "tool_use");
        assert_eq!(response["usage"]["cache_read_input_tokens"], 5);
        assert_eq!(response["usage"]["input_tokens"], 7);
        assert_eq!(response["usage"]["iterations"][0]["type"], "message");
        assert!(response["content"][0]["signature"]
            .as_str()
            .unwrap()
            .starts_with("mogick-v1-"));
        let signature = response["content"][0]["signature"].clone();
        let continued = convert_request(
            serde_json::to_vec(&json!({
                "model":"model-x",
                "max_tokens":10,
                "messages":[
                    {"role":"assistant","content":[{
                        "type":"thinking","thinking":SAFE_THINKING_SUMMARY,"signature":signature
                    }]},
                    {"role":"user","content":"continue"}
                ]
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        assert_eq!(continued.body["messages"][0]["reasoning_content"], "think");
    }

    #[test]
    fn sse_decoder_handles_splits_multiline_comments_and_crlf() {
        let mut decoder = SseDecoder::default();
        assert!(decoder
            .push(b": keepalive\r\nevent: x\r\ndata: {\"a\":")
            .unwrap()
            .is_empty());
        let events = decoder.push(b"1}\r\ndata: tail\r\n\r\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("x"));
        assert_eq!(events[0].data, "{\"a\":1}\ntail");

        let mut oversized = SseDecoder::default();
        assert!(oversized
            .push(&vec![b'x'; MAX_UPSTREAM_SSE_EVENT_BYTES + 1])
            .is_err());
    }

    #[test]
    fn stream_has_strict_lifecycle_and_interleaved_tools() {
        let mut state = StreamState::new(
            "req".into(),
            "model".into(),
            ThinkingDisplay::Summarized,
            None,
            None,
        );
        let mut frames = Vec::new();
        frames.extend(state.handle(
            r#"{"id":"chat","model":"m","choices":[{"delta":{"reasoning_content":"h"}}]}"#,
        ));
        frames.extend(state.handle(r#"{"choices":[{"delta":{"content":"hi","tool_calls":[{"index":1,"id":"call_b","function":{"name":"b","arguments":"{\"b\":"}},{"index":0,"id":"call_a","function":{"name":"a","arguments":"{\"a\":"}}]}}]}"#));
        frames.extend(state.handle(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}},{"index":1,"function":{"arguments":"2}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4}}"#));
        frames.extend(state.handle("[DONE]"));
        let text = frames
            .into_iter()
            .map(|frame| String::from_utf8(frame.to_vec()).unwrap())
            .collect::<String>();
        let names: Vec<&str> = text
            .lines()
            .filter_map(|line| line.strip_prefix("event: "))
            .collect();
        assert_eq!(names.first(), Some(&"message_start"));
        assert_eq!(names.last(), Some(&"message_stop"));
        assert_eq!(
            names
                .iter()
                .filter(|name| **name == "content_block_start")
                .count(),
            4
        );
        assert!(text.contains("thinking_delta"));
        assert!(!text.contains("\"thinking\":\"h\""));
        assert!(text.contains("input_json_delta"));
        assert!(text.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn abnormal_stream_emits_error() {
        let mut state = StreamState::new(
            "req".into(),
            "model".into(),
            ThinkingDisplay::Summarized,
            None,
            None,
        );
        let frames = state.handle("[DONE]");
        let text = String::from_utf8(frames[0].to_vec()).unwrap();
        assert!(text.starts_with("event: error"));
        assert!(text.contains("api_error"));
    }

    #[test]
    fn omitted_thinking_streams_only_an_opaque_signature() {
        let mut state = StreamState::new(
            "req".into(),
            "model".into(),
            ThinkingDisplay::Omitted,
            None,
            None,
        );
        let mut frames = state.handle(
            r#"{"choices":[{"delta":{"reasoning_content":"private reasoning"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2}}"#,
        );
        frames.extend(state.handle("[DONE]"));
        let text = frames
            .into_iter()
            .map(|frame| String::from_utf8(frame.to_vec()).unwrap())
            .collect::<String>();
        assert!(!text.contains("thinking_delta"));
        assert!(!text.contains("private reasoning"));
        assert!(text.contains("signature_delta"));
        assert!(text.contains("mogick-v1-"));
    }

    #[test]
    fn streaming_message_delta_reports_applied_context_edits() {
        let context = ContextManagementResult {
            original_input_tokens: 100,
            effective_input_tokens_before_edits: 100,
            effective_input_tokens: 80,
            applied_edits: vec![json!({
                "type":"clear_tool_uses_20250919",
                "cleared_tool_uses":2,
                "cleared_input_tokens":20
            })],
        };
        let mut state = StreamState::new(
            "req".into(),
            "model".into(),
            ThinkingDisplay::Summarized,
            None,
            Some(context),
        );
        let mut frames = state.handle(
            r#"{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":80,"completion_tokens":1}}"#,
        );
        frames.extend(state.handle("[DONE]"));
        let text = frames
            .into_iter()
            .map(|frame| String::from_utf8(frame.to_vec()).unwrap())
            .collect::<String>();
        assert!(text.contains("\"context_management\":{\"applied_edits\""));
        assert!(text.contains("\"cleared_tool_uses\":2"));
    }
}
