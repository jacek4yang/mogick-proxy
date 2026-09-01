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
}

pub fn convert_request(bytes: &[u8]) -> Result<ConvertedRequest, ProtocolError> {
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
        ],
        "request",
    )?;
    let model = required_string(object, "model")?.to_string();
    let max_tokens = object
        .get("max_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| ProtocolError::invalid("max_tokens must be a positive integer"))?;
    if max_tokens == 0 {
        return Err(ProtocolError::invalid(
            "max_tokens must be greater than zero",
        ));
    }
    let stream = object
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut output = Map::new();
    output.insert("model".into(), Value::String(model.clone()));
    output.insert("max_tokens".into(), Value::from(max_tokens));
    output.insert("stream".into(), Value::Bool(stream));

    let mut messages = Vec::new();
    if let Some(system) = object.get("system") {
        messages.push(json!({"role":"system", "content": convert_system(system)?}));
    }
    let input_messages = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ProtocolError::invalid("messages must be an array"))?;
    for (index, message) in input_messages.iter().enumerate() {
        convert_message(message, index, &mut messages)?;
    }
    output.insert("messages".into(), Value::Array(messages));

    copy_number(object, &mut output, "temperature")?;
    copy_number(object, &mut output, "top_p")?;
    copy_number(object, &mut output, "top_k")?;
    if let Some(stop) = object.get("stop_sequences") {
        if !stop.is_array() {
            return Err(ProtocolError::invalid("stop_sequences must be an array"));
        }
        output.insert("stop".into(), stop.clone());
    }
    if let Some(metadata) = object.get("metadata") {
        if !metadata.is_object() {
            return Err(ProtocolError::invalid("metadata must be an object"));
        }
        output.insert("metadata".into(), metadata.clone());
    }
    if let Some(service_tier) = object.get("service_tier") {
        output.insert("service_tier".into(), service_tier.clone());
    }
    if let Some(thinking) = object.get("thinking") {
        validate_thinking(thinking)?;
        output.insert("thinking".into(), thinking.clone());
    }
    if let Some(tools) = object.get("tools") {
        output.insert("tools".into(), convert_tools(tools)?);
    }
    if let Some(choice) = object.get("tool_choice") {
        let (tool_choice, parallel) = convert_tool_choice(choice)?;
        output.insert("tool_choice".into(), tool_choice);
        if let Some(parallel) = parallel {
            output.insert("parallel_tool_calls".into(), Value::Bool(parallel));
        }
    }
    if let Some(config) = object.get("output_config") {
        convert_output_config(config, &mut output)?;
    }
    if let Some(format) = object.get("output_format") {
        output.insert("response_format".into(), convert_output_format(format)?);
    }

    if stream {
        output.insert("stream_options".into(), json!({"include_usage": true}));
    }
    Ok(ConvertedRequest {
        body: Value::Object(output),
        model,
        stream,
    })
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

fn convert_message(
    value: &Value,
    message_index: usize,
    output: &mut Vec<Value>,
) -> Result<(), ProtocolError> {
    let object = value.as_object().ok_or_else(|| {
        ProtocolError::invalid(format!("messages[{message_index}] must be an object"))
    })?;
    reject_unknown_keys(object, &["role", "content"], "message")?;
    let role = required_string(object, "role")?;
    if !matches!(role, "user" | "assistant") {
        return Err(ProtocolError::invalid(format!(
            "messages[{message_index}].role must be user or assistant"
        )));
    }
    let content = object
        .get("content")
        .ok_or_else(|| ProtocolError::invalid("message content is required"))?;
    if content.is_string() {
        output.push(json!({"role":role, "content":content}));
        return Ok(());
    }
    let blocks = content
        .as_array()
        .ok_or_else(|| ProtocolError::invalid("message content must be a string or array"))?;
    if role == "assistant" {
        convert_assistant_blocks(blocks, output)
    } else {
        convert_user_blocks(blocks, output)
    }
}

fn convert_assistant_blocks(
    blocks: &[Value],
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
                reasoning.push_str(required_string(object, "thinking")?);
            }
            "redacted_thinking" => {
                reject_unknown_keys(object, &["type", "data"], "redacted thinking block")?;
                redacted_reasoning.push(Value::String(required_string(object, "data")?.into()));
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
                &["name", "description", "input_schema", "cache_control"],
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

fn validate_thinking(value: &Value) -> Result<(), ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("thinking must be an object"))?;
    reject_unknown_keys(object, &["type", "budget_tokens"], "thinking")?;
    match required_string(object, "type")? {
        "enabled" => {
            if object
                .get("budget_tokens")
                .and_then(Value::as_u64)
                .is_none()
            {
                return Err(ProtocolError::invalid(
                    "enabled thinking requires budget_tokens",
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
    Ok(())
}

fn convert_output_config(
    value: &Value,
    output: &mut Map<String, Value>,
) -> Result<(), ProtocolError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("output_config must be an object"))?;
    reject_unknown_keys(object, &["format", "effort"], "output_config")?;
    if let Some(format) = object.get("format") {
        output.insert("response_format".into(), convert_output_format(format)?);
    }
    if let Some(effort) = object.get("effort") {
        output.insert("reasoning_effort".into(), effort.clone());
    }
    Ok(())
}

fn convert_output_format(value: &Value) -> Result<Value, ProtocolError> {
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
    Ok(json!({
        "type":"json_schema",
        "json_schema":{
            "name":object.get("name").and_then(Value::as_str).unwrap_or("structured_output"),
            "description":object.get("description").cloned().unwrap_or(Value::Null),
            "schema":schema,
            "strict":object.get("strict").and_then(Value::as_bool).unwrap_or(true)
        }
    }))
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

fn copy_number(
    input: &Map<String, Value>,
    output: &mut Map<String, Value>,
    key: &str,
) -> Result<(), ProtocolError> {
    if let Some(value) = input.get(key) {
        if !value.is_number() {
            return Err(ProtocolError::invalid(format!("{key} must be a number")));
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

pub fn convert_response(
    value: &Value,
    request_id: &str,
    fallback_model: &str,
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
    if let Some(reasoning) = message
        .get("reasoning_content")
        .or_else(|| message.get("reasoning"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        content.push(json!({"type":"thinking", "thinking":reasoning}));
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
                    content.push(json!({"type":"thinking", "thinking":thinking}));
                }
            }
        }
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            let function = call.get("function").and_then(Value::as_object);
            let arguments = function
                .and_then(|function| function.get("arguments"))
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let input =
                serde_json::from_str(arguments).unwrap_or_else(|_| json!({"_raw":arguments}));
            content.push(json!({
                "type":"tool_use",
                "id":call.get("id").and_then(Value::as_str).unwrap_or("toolu_unknown"),
                "name":function.and_then(|function| function.get("name")).and_then(Value::as_str).unwrap_or("unknown"),
                "input":input
            }));
        }
    }
    let finish = choice.get("finish_reason").and_then(Value::as_str);
    let stop_sequence = choice.get("stop_sequence").cloned().unwrap_or(Value::Null);
    Ok(json!({
        "id":value.get("id").and_then(Value::as_str).unwrap_or(request_id),
        "type":"message",
        "role":"assistant",
        "content":content,
        "model":value.get("model").and_then(Value::as_str).unwrap_or(fallback_model),
        "stop_reason":map_stop_reason(finish),
        "stop_sequence":stop_sequence,
        "usage":convert_usage(value.get("usage"))
    }))
}

fn convert_usage(usage: Option<&Value>) -> Value {
    let usage = usage.and_then(Value::as_object);
    let input = usage
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .and_then(|usage| usage.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = usage
        .and_then(|usage| usage.get("prompt_tokens_details"))
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "input_tokens":input,
        "output_tokens":output,
        "cache_creation_input_tokens":0,
        "cache_read_input_tokens":cached
    })
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
    let data: Vec<Value> = models
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str))
        .map(|id| {
            json!({
                "type":"model",
                "id":id,
                "display_name":id,
                "created_at":"1970-01-01T00:00:00Z"
            })
        })
        .collect();
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

pub fn error_envelope(error_type: &str, message: impl Into<String>, request_id: &str) -> Value {
    json!({
        "type":"error",
        "error":{"type":error_type, "message":message.into()},
        "request_id":request_id
    })
}

pub fn stream_body(
    response: reqwest::Response,
    request_id: String,
    fallback_model: String,
) -> Body {
    let output = async_stream::stream! {
        let mut upstream = response.bytes_stream();
        let mut decoder = SseDecoder::default();
        let mut state = StreamState::new(request_id, fallback_model);
        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(chunk) => {
                    for event in decoder.push(&chunk) {
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
        for event in decoder.finish() {
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
}

#[derive(Debug, PartialEq, Eq)]
struct SseEvent {
    #[allow(dead_code)]
    event: Option<String>,
    data: String,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=position).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.process_line(&String::from_utf8_lossy(&line), &mut events);
        }
        events
    }

    fn finish(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let line = String::from_utf8_lossy(&std::mem::take(&mut self.buffer)).into_owned();
            self.process_line(&line, &mut events);
        }
        self.dispatch(&mut events);
        events
    }

    fn process_line(&mut self, line: &str, events: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(events);
        } else if line.starts_with(':') {
            // SSE comment / keepalive.
        } else if let Some(value) = line.strip_prefix("data:") {
            self.data_lines
                .push(value.strip_prefix(' ').unwrap_or(value).to_string());
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
        } else {
            self.event = None;
        }
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
}

#[derive(Clone, Copy)]
enum BlockKind {
    Text,
    Thinking,
    Tool,
}

impl StreamState {
    fn new(request_id: String, fallback_model: String) -> Self {
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
                    let index = self.ensure_thinking(&mut frames);
                    frames.push(sse_frame(
                        "content_block_delta",
                        json!({"type":"content_block_delta", "index":index,
                            "delta":{"type":"thinking_delta", "thinking":thinking}}),
                    ));
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
            json!({"type":"content_block_start", "index":index, "content_block":{"type":"thinking", "thinking":""}}),
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
            frames.push(sse_frame(
                "content_block_stop",
                json!({"type":"content_block_stop", "index":index}),
            ));
        }
        let usage = convert_usage(Some(&self.usage));
        frames.push(sse_frame(
            "message_delta",
            json!({"type":"message_delta", "delta":{
                "stop_reason":map_stop_reason(self.finish_reason.as_deref()),
                "stop_sequence":self.stop_sequence
            }, "usage":usage}),
        ));
        frames.push(sse_frame("message_stop", json!({"type":"message_stop"})));
        frames
    }
}

fn sse_frame(event: &str, data: Value) -> Bytes {
    Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
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
                "thinking":{"type":"enabled","budget_tokens":64},
                "output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}}
            }))
            .unwrap()
            .as_slice(),
        )
        .unwrap();
        assert_eq!(converted.model, "mm-any-future-model");
        assert_eq!(converted.body["tool_choice"], "required");
        assert_eq!(converted.body["parallel_tool_calls"], false);
        assert_eq!(converted.body["response_format"]["type"], "json_schema");
        assert_eq!(
            converted.body["messages"][2]["tool_calls"][0]["function"]["name"],
            "read"
        );
        assert_eq!(converted.body["messages"][3]["role"], "tool");
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
        assert_eq!(response["content"][2]["type"], "tool_use");
        assert_eq!(response["stop_reason"], "tool_use");
        assert_eq!(response["usage"]["cache_read_input_tokens"], 5);
    }

    #[test]
    fn sse_decoder_handles_splits_multiline_comments_and_crlf() {
        let mut decoder = SseDecoder::default();
        assert!(decoder
            .push(b": keepalive\r\nevent: x\r\ndata: {\"a\":")
            .is_empty());
        let events = decoder.push(b"1}\r\ndata: tail\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("x"));
        assert_eq!(events[0].data, "{\"a\":1}\ntail");
    }

    #[test]
    fn stream_has_strict_lifecycle_and_interleaved_tools() {
        let mut state = StreamState::new("req".into(), "model".into());
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
        assert!(text.contains("input_json_delta"));
        assert!(text.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn abnormal_stream_emits_error() {
        let mut state = StreamState::new("req".into(), "model".into());
        let frames = state.handle("[DONE]");
        let text = String::from_utf8(frames[0].to_vec()).unwrap();
        assert!(text.starts_with("event: error"));
        assert!(text.contains("api_error"));
    }
}
