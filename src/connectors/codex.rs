use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;
use walkdir::WalkDir;

use super::scan::{DiscoveredSourceFile, DiscoveredSourceRole, ScanContext, ScanRoot};
use super::utils::{
    add_encrypted_content, add_raw_role, dedupe_path_key, env_path_nonempty,
    set_tool_result_pairing,
};
use super::{
    Connector, extract_invocations_from_content_blocks, flatten_content,
    franken_detection_for_connector, parse_timestamp,
};
use crate::types::{
    DetectionResult, NormalizedConversation, NormalizedInvocation, NormalizedMessage,
};

pub struct CodexConnector;

const LARGE_SESSION_EXTRA_COMPACT_THRESHOLD_BYTES: u64 = 32 * 1024 * 1024;

enum FileScanMetadata {
    Process(Option<fs::Metadata>),
    Skip,
}

impl Default for CodexConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexConnector {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn is_under_codex_dir(path: &Path) -> bool {
        path.ancestors().any(|ancestor| {
            ancestor
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == ".codex")
        })
    }

    fn append_explicit_roots(roots: &mut Vec<PathBuf>, base: &Path) {
        if base.is_file() {
            roots.push(base.to_path_buf());
            return;
        }

        roots.push(base.to_path_buf());

        if !Self::is_under_codex_dir(base) {
            roots.push(base.join(".codex"));
        }
    }

    fn home() -> PathBuf {
        if let Some(explicit) = env_path_nonempty("CODEX_HOME") {
            return explicit;
        }
        dirs::home_dir().unwrap_or_default().join(".codex")
    }

    fn sessions_dir(home: &Path) -> PathBuf {
        let sessions = home.join("sessions");
        if sessions.exists() {
            sessions
        } else {
            home.to_path_buf()
        }
    }

    fn is_rollout_file(path: &Path) -> bool {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            return false;
        };
        if !name.starts_with("rollout-") {
            return false;
        }
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                ext.eq_ignore_ascii_case("jsonl") || ext.eq_ignore_ascii_case("json")
            })
    }

    fn sessions_dir_for_explicit_file(path: &Path) -> Option<PathBuf> {
        path.ancestors()
            .find(|ancestor| {
                ancestor.file_name().and_then(|name| name.to_str()) == Some("sessions")
            })
            .map(Path::to_path_buf)
    }

    fn rollout_files(root: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let sessions = Self::sessions_dir(root);
        if !sessions.exists() {
            return out;
        }
        for entry in WalkDir::new(sessions).into_iter().flatten() {
            if entry.file_type().is_file() {
                let name = entry.file_name().to_str().unwrap_or("");
                // Match both modern .jsonl and legacy .json formats
                if name.starts_with("rollout-")
                    && entry
                        .path()
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| {
                            ext.eq_ignore_ascii_case("jsonl") || ext.eq_ignore_ascii_case("json")
                        })
                {
                    out.push(entry.path().to_path_buf());
                }
            }
        }
        // Keep connector traversal deterministic across filesystems/runs.
        out.sort();
        out
    }

    fn is_token_usage_target_message(message: &NormalizedMessage) -> bool {
        // Attribute token_count usage to concrete assistant turns only.
        // This used to also require `author.is_none()` to exclude reasoning
        // messages, which were previously masquerading as `role="assistant"`
        // with `author=Some("reasoning")`. Now that reasoning has its own
        // `role="reasoning"` (task 1.3), checking `role` alone is sufficient
        // and correct even when a real model author is attached.
        message.role == "assistant"
    }

    fn legacy_token_usage_from_payload(payload: &Value) -> Option<Value> {
        let input_tokens = payload.get("input_tokens").and_then(Value::as_i64);
        let output_tokens = payload
            .get("output_tokens")
            .and_then(Value::as_i64)
            .or_else(|| payload.get("tokens").and_then(Value::as_i64));

        if input_tokens.is_none() && output_tokens.is_none() {
            return None;
        }

        let mut usage = serde_json::Map::new();
        if let Some(input) = input_tokens {
            usage.insert("input_tokens".to_string(), Value::from(input));
        }
        if let Some(output) = output_tokens {
            usage.insert("output_tokens".to_string(), Value::from(output));
        }
        usage.insert("data_source".to_string(), Value::String("api".to_string()));

        Some(Value::Object(usage))
    }

    fn token_usage_from_payload(payload: &Value) -> Result<Option<Value>> {
        let is_canonical = payload.get("info").is_some() || payload.get("rate_limits").is_some();
        if !is_canonical {
            let is_explicit_legacy = ["input_tokens", "output_tokens", "tokens"]
                .iter()
                .any(|key| payload.get(*key).is_some());
            if !is_explicit_legacy {
                anyhow::bail!("token_count payload matches neither canonical nor legacy shape");
            }
            return Ok(Self::legacy_token_usage_from_payload(payload));
        }

        let payload_object = payload
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("token_count payload must be an object"))?;
        validate_exact_keys(
            payload_object,
            &["type", "info"],
            &["rate_limits"],
            "token_count payload",
        )?;

        // `rate_limits` is optional: real-world rollouts from early codex
        // CLI builds (observed 2025-09, pre-dating this field) omit the key
        // entirely rather than setting it to `null`. Treat an absent key
        // the same as an explicit `null` -- both mean "no rate-limit info
        // attached to this event".
        let rate_limits = payload_object
            .get("rate_limits")
            .cloned()
            .unwrap_or(Value::Null);
        let info = payload_object
            .get("info")
            .context("validated token_count payload lost info")?;
        if info.is_null() {
            if !rate_limits.is_object() {
                anyhow::bail!("token_count rate_limits must be an object when info is null");
            }
            return Ok(None);
        }
        if !rate_limits.is_null() && !rate_limits.is_object() {
            anyhow::bail!("token_count rate_limits must be an object or null");
        }

        let info_object = info
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("token_count info must be an object or null"))?;
        validate_exact_keys(
            info_object,
            &[
                "last_token_usage",
                "model_context_window",
                "total_token_usage",
            ],
            &[],
            "token_count info",
        )?;

        let _context_window = info_object
            .get("model_context_window")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "token_count model_context_window must be a finite non-negative number"
                )
            })?;
        let mut last_usage = validate_token_usage_object(
            info_object
                .get("last_token_usage")
                .context("validated token_count info lost last_token_usage")?,
            "token_count last_token_usage",
        )?;
        validate_token_usage_object(
            info_object
                .get("total_token_usage")
                .context("validated token_count info lost total_token_usage")?,
            "token_count total_token_usage",
        )?;
        last_usage.insert("data_source".to_string(), Value::String("api".to_string()));
        Ok(Some(Value::Object(last_usage)))
    }

    fn should_compact_large_message_extra(file_size_bytes: Option<u64>) -> bool {
        file_size_bytes.is_some_and(|size| size >= LARGE_SESSION_EXTRA_COMPACT_THRESHOLD_BYTES)
    }

    fn file_metadata_if_modified(path: &Path, since_ts: Option<i64>) -> FileScanMetadata {
        let Ok(metadata) = fs::metadata(path) else {
            return FileScanMetadata::Process(None);
        };
        if Self::metadata_modified_since(&metadata, since_ts) {
            FileScanMetadata::Process(Some(metadata))
        } else {
            FileScanMetadata::Skip
        }
    }

    fn metadata_modified_since(metadata: &fs::Metadata, since_ts: Option<i64>) -> bool {
        since_ts.is_none_or(|ts| {
            let threshold = ts.saturating_sub(1_000);
            metadata.modified().map_or(true, |modified| {
                modified
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(true, |duration| {
                        i64::try_from(duration.as_millis()).unwrap_or(i64::MAX) >= threshold
                    })
            })
        })
    }

    fn source_roots(ctx: &ScanContext) -> Vec<ScanRoot> {
        let is_codex_dir = ctx.data_dir.to_str().is_some_and(|s| {
            s.contains(".codex") || s.ends_with("/codex") || s.ends_with("\\codex")
        }) && ctx.data_dir.join("sessions").exists();

        let mut roots: Vec<ScanRoot> =
            if ctx.use_default_detection() {
                if is_codex_dir {
                    vec![ScanRoot::local(ctx.data_dir.clone())]
                } else {
                    vec![ScanRoot::local(Self::home())]
                }
            } else {
                let mut explicit = Vec::new();
                for scan_root in &ctx.scan_roots {
                    Self::append_explicit_roots(&mut explicit, &scan_root.path);
                }
                explicit
                    .into_iter()
                    .map(|path| {
                        if let Some(root) = ctx.scan_roots.iter().find(|root| {
                            path.starts_with(&root.path) || root.path.starts_with(&path)
                        }) {
                            root.with_path(path)
                        } else {
                            ScanRoot::local(path)
                        }
                    })
                    .collect()
            };

        roots.sort_by(|a, b| a.path.cmp(&b.path));
        roots.dedup_by(|a, b| a.path == b.path);
        roots
    }

    fn discover_sources(ctx: &ScanContext) -> Vec<DiscoveredSourceFile> {
        let roots = Self::source_roots(ctx);
        let mut out = Vec::new();
        let mut seen_files: HashSet<PathBuf> = HashSet::new();

        for root in roots {
            let explicit_file = root
                .path
                .is_file()
                .then_some(root.path.clone())
                .filter(|path| Self::is_rollout_file(path));
            let home = explicit_file
                .as_ref()
                .and_then(|path| path.parent().map(Path::to_path_buf))
                .unwrap_or_else(|| root.path.clone());
            if !home.exists() {
                continue;
            }

            let files = explicit_file
                .clone()
                .map_or_else(|| Self::rollout_files(&home), |path| vec![path]);

            for file in files {
                if !seen_files.insert(dedupe_path_key(&file)) {
                    continue;
                }
                if matches!(
                    Self::file_metadata_if_modified(&file, ctx.since_ts),
                    FileScanMetadata::Skip
                ) {
                    continue;
                }
                out.push(
                    DiscoveredSourceFile::new(
                        "codex",
                        &root,
                        file,
                        DiscoveredSourceRole::PrimarySessionLog,
                        true,
                    )
                    .with_fs_metadata(),
                );
            }
        }

        out
    }

    fn compact_message_extra(raw: &Value) -> Value {
        let mut cass = serde_json::Map::new();

        if let Some(model) = raw
            .get("model")
            .or_else(|| raw.pointer("/response/model"))
            .and_then(|v| v.as_str())
            .filter(|value| !value.trim().is_empty())
        {
            cass.insert("model".to_string(), Value::String(model.to_string()));
        }

        if let Some(attachments) = raw
            .get("attachment_refs")
            .or_else(|| raw.get("attachments"))
            .cloned()
        {
            cass.insert("attachments".to_string(), attachments);
        }

        if cass.is_empty() {
            Value::Object(serde_json::Map::new())
        } else {
            let mut out = serde_json::Map::new();
            out.insert("cass".to_string(), Value::Object(cass));
            Value::Object(out)
        }
    }

    fn normalized_message_extra(
        raw_envelope: &Value,
        compact_message_extra: bool,
        raw_role: &str,
    ) -> Result<Value> {
        if raw_envelope.get("encrypted_content").is_some() {
            anyhow::bail!("raw envelope already contains reserved key encrypted_content");
        }

        let projected_extra = if compact_message_extra {
            Self::compact_message_extra(raw_envelope)
        } else {
            raw_envelope.clone()
        };
        add_raw_role(raw_envelope, projected_extra, raw_role)
    }

    fn attach_token_usage_to_latest_assistant(
        messages: &mut [NormalizedMessage],
        token_usage: Value,
        source_path: &Path,
        line_number: usize,
    ) {
        if let Some(target) = messages
            .iter_mut()
            .rev()
            .find(|m| Self::is_token_usage_target_message(m))
        {
            if !target.extra.is_object() {
                target.extra = Value::Object(serde_json::Map::new());
            }

            if let Some(extra) = target.extra.as_object_mut() {
                let cass = extra
                    .entry("cass".to_string())
                    .or_insert_with(|| Value::Object(serde_json::Map::new()));

                if !cass.is_object() {
                    *cass = Value::Object(serde_json::Map::new());
                }

                if let Some(cass_obj) = cass.as_object_mut() {
                    // Multiple token_count events for the same assistant turn:
                    // deterministic rule = last write wins.
                    cass_obj.insert("token_usage".to_string(), token_usage);
                }
            }
        } else {
            tracing::debug!(
                path = %source_path.display(),
                line_number,
                "codex token_count event had no preceding assistant message; skipping"
            );
        }
    }
}

fn update_time_bounds(started_at: &mut Option<i64>, ended_at: &mut Option<i64>, ts: Option<i64>) {
    if let Some(ts) = ts {
        *started_at = Some(started_at.map_or(ts, |curr| curr.min(ts)));
        *ended_at = Some(ended_at.map_or(ts, |curr| curr.max(ts)));
    }
}

fn validate_exact_keys(
    object: &serde_json::Map<String, Value>,
    required: &[&str],
    optional: &[&str],
    label: &str,
) -> Result<()> {
    let valid_len = object.len() >= required.len()
        && object.len() <= required.len().saturating_add(optional.len());
    let has_required = required.iter().all(|key| object.contains_key(*key));
    let only_allowed = object
        .keys()
        .all(|key| required.contains(&key.as_str()) || optional.contains(&key.as_str()));
    if !valid_len || !has_required || !only_allowed {
        anyhow::bail!("{label} has an unsupported key set");
    }
    Ok(())
}

fn validate_token_usage_object(
    value: &Value,
    label: &str,
) -> Result<serde_json::Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{label} must be an object"))?;
    validate_exact_keys(
        object,
        &[
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        ],
        &["cache_write_input_tokens"],
        label,
    )?;
    for (key, token_value) in object {
        if token_value.as_u64().is_none() {
            anyhow::bail!("{label}.{key} must be a non-negative integer");
        }
    }
    Ok(object.clone())
}

fn metadata_field_and_timestamp<'a>(
    envelope: &'a Value,
    entry_type: &str,
    field: &str,
) -> Result<(&'a str, Option<i64>)> {
    let timestamp = envelope
        .get("timestamp")
        .filter(|value| value.is_string())
        .ok_or_else(|| anyhow::anyhow!("{entry_type} timestamp must be a string"))?;
    let payload = envelope
        .get("payload")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("{entry_type} payload must be an object"))?;
    let field_value = payload
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("{entry_type} payload.{field} must be a string"))?;
    Ok((field_value, parse_timestamp(timestamp)))
}

/// Parse the arguments of a modern Codex `response_item` tool call.
///
/// `function_call` payloads carry `arguments` as a JSON-encoded string (e.g.
/// `"{\"cmd\":\"ls\"}"`); `custom_tool_call` payloads (e.g. `apply_patch`) carry
/// freeform `input`. Returns the decoded JSON when the string parses, otherwise
/// the raw string, so downstream consumers never lose the original payload.
fn parse_tool_call_arguments(payload: &Value) -> Option<Value> {
    let raw = payload.get("arguments").or_else(|| payload.get("input"))?;
    match raw {
        Value::String(s) if !s.is_empty() => {
            Some(serde_json::from_str::<Value>(s).unwrap_or_else(|_| Value::String(s.clone())))
        }
        other => Some(other.clone()),
    }
}

/// Extract the textual output of a modern Codex `response_item` tool result.
///
/// `output` is almost always a plain string, but the Responses API can also
/// nest it as `{"content":[{"text":...}]}`; handle both shapes and fall back to
/// flattening so structured results still surface as searchable text.
fn tool_output_text(payload: &Value) -> String {
    let Some(output) = payload.get("output") else {
        return String::new();
    };
    if let Some(text) = output.as_str() {
        return text.to_string();
    }
    if let Some(content) = output.get("content") {
        let flattened = flatten_content(content);
        if !flattened.trim().is_empty() {
            return flattened;
        }
    }
    flatten_content(output)
}

/// Render a tool call's own content for display: `<name>(<args JSON>)`, or
/// just `<name>` when there's no input. Mirrors `claude_code.rs`'s
/// `render_tool_call_content` -- the full untruncated arguments always live
/// in `extra["tool_call_args"]`/the invocation's `arguments` for exact
/// reconstruction (never truncated -- spec §3.2).
fn render_tool_call_content(name: &str, arguments: Option<&Value>) -> String {
    match arguments {
        Some(value) if !value.is_null() => format!("{name}({value})"),
        _ => name.to_string(),
    }
}

/// Extract plaintext reasoning text from a `response_item`/`reasoning`
/// payload's `summary` array (`[{"type":"summary_text","text":"..."}]`).
/// Returns empty when `summary` is absent/empty -- e.g. when the item is
/// fully encrypted with no plaintext summary. Never touches
/// `encrypted_content` (spec: do not attempt to decrypt it).
fn reasoning_summary_text(payload: &Value) -> Result<String> {
    let Some(summary) = payload.get("summary") else {
        return Ok(String::new());
    };
    let items = summary
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("reasoning summary must be an array"))?;
    let mut out = String::new();
    for item in items {
        let item = item
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("reasoning summary item must be an object"))?;
        let item_type = item
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("reasoning summary item.type must be a string"))?;
        if item_type != "summary_text" {
            anyhow::bail!("reasoning summary item.type must be summary_text");
        }
        let text = item
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("reasoning summary_text.text must be a string"))?;
        if text.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(text);
    }
    Ok(out)
}

fn parse_agent_message_content(payload: &Value) -> Result<(String, Option<Value>)> {
    let blocks = payload
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("agent_message content must be an array"))?;
    let mut visible = Vec::new();
    let mut encrypted_content = None;
    for block in blocks {
        let block = block
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("agent_message content block must be an object"))?;
        let block_type = block
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("agent_message content block.type must be a string"))?;
        match block_type {
            "text" | "input_text" | "output_text" => {
                let text = block.get("text").and_then(Value::as_str).ok_or_else(|| {
                    anyhow::anyhow!("agent_message visible text must be a string")
                })?;
                visible.push(text.to_string());
            }
            "encrypted_content" => {
                if encrypted_content.is_some() {
                    anyhow::bail!("agent_message has duplicate encrypted_content blocks");
                }
                let opaque = block.get("encrypted_content").ok_or_else(|| {
                    anyhow::anyhow!("agent_message encrypted_content block is missing its value")
                })?;
                if !opaque.is_string() {
                    anyhow::bail!("agent_message encrypted_content must be a string");
                }
                encrypted_content = Some(opaque.clone());
            }
            "input_image" | "refusal" => {}
            _ => anyhow::bail!("agent_message content block.type is unsupported"),
        }
    }
    Ok((visible.join("\n"), encrypted_content))
}

#[allow(clippy::too_many_lines)]
fn scan_codex_with_callback(
    ctx: &ScanContext,
    on_conversation: &mut dyn FnMut(NormalizedConversation) -> Result<()>,
) -> Result<()> {
    let roots: Vec<PathBuf> = CodexConnector::source_roots(ctx)
        .into_iter()
        .map(|root| root.path)
        .collect();

    if roots.is_empty() {
        return Ok(());
    }

    let mut seen_files: HashSet<PathBuf> = HashSet::new();

    for root in roots {
        let explicit_file = root
            .is_file()
            .then_some(root.clone())
            .filter(|path| CodexConnector::is_rollout_file(path));
        let home = explicit_file
            .as_ref()
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| root.clone());
        if !home.exists() {
            continue;
        }

        let files = explicit_file
            .clone()
            .map_or_else(|| CodexConnector::rollout_files(&home), |path| vec![path]);
        let sessions_dir = explicit_file
            .as_ref()
            .and_then(|path| CodexConnector::sessions_dir_for_explicit_file(path))
            .unwrap_or_else(|| CodexConnector::sessions_dir(&home));

        for file in files {
            if !seen_files.insert(dedupe_path_key(&file)) {
                continue;
            }
            let source_path = file.clone();
            let file_metadata = match CodexConnector::file_metadata_if_modified(&file, ctx.since_ts)
            {
                FileScanMetadata::Process(metadata) => metadata,
                FileScanMetadata::Skip => continue,
            };
            let file_size_bytes = file_metadata.as_ref().map(std::fs::Metadata::len);
            let compact_message_extra =
                CodexConnector::should_compact_large_message_extra(file_size_bytes);
            if compact_message_extra {
                tracing::debug!(
                    path = %file.display(),
                    size_bytes = file_size_bytes.unwrap_or_default(),
                    "codex compacting per-message extra payloads for large session"
                );
            }
            let external_id = source_path
                .strip_prefix(&sessions_dir)
                .ok()
                .and_then(|rel| {
                    rel.with_extension("")
                        .to_str()
                        .map(std::string::ToString::to_string)
                })
                .or_else(|| {
                    source_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(std::string::ToString::to_string)
                });
            let ext = file.extension().and_then(|e| e.to_str());
            let mut messages = Vec::new();
            let mut started_at = None;
            let mut ended_at = None;
            let mut session_cwd: Option<PathBuf> = None;
            // Model provenance for `author` on assistant/tool_call/reasoning
            // messages (spec: author = model name for those roles). Real
            // rollouts only ever carry `model` on `turn_context` payloads
            // (never on `session_meta` or individual `message` items), so
            // track the most recently seen one as turns progress.
            let mut current_model: Option<String> = None;

            if ext == Some("jsonl") {
                let f = std::fs::File::open(&file)
                    .with_context(|| format!("open rollout {}", file.display()))?;
                let reader = std::io::BufReader::new(f);

                for (line_idx, line_res) in std::io::BufRead::lines(reader).enumerate() {
                    let Ok(line) = line_res else {
                        continue;
                    };
                    if line.trim().is_empty() {
                        continue;
                    }
                    let Ok(val) = serde_json::from_str::<Value>(&line) else {
                        continue;
                    };

                    let entry_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    let created = val.get("timestamp").and_then(parse_timestamp);

                    match entry_type {
                        "session_meta" => {
                            let (cwd, metadata_created) =
                                metadata_field_and_timestamp(&val, "session_meta", "cwd")?;
                            session_cwd = Some(PathBuf::from(cwd));
                            update_time_bounds(&mut started_at, &mut ended_at, metadata_created);
                        }
                        "turn_context" => {
                            let (model, metadata_created) =
                                metadata_field_and_timestamp(&val, "turn_context", "model")?;
                            current_model = Some(model.to_string());
                            update_time_bounds(&mut started_at, &mut ended_at, metadata_created);
                        }
                        "response_item" => {
                            let Some(payload) = val.get("payload") else {
                                continue;
                            };
                            let payload_type = match payload.get("type") {
                                None => None,
                                Some(Value::String(payload_type)) => Some(payload_type.as_str()),
                                Some(_) => anyhow::bail!(
                                    "response_item payload.type must be a string when present"
                                ),
                            };
                            match payload_type {
                                Some("message") | None => {
                                    let Some(role @ ("user" | "assistant")) =
                                        payload.get("role").and_then(Value::as_str)
                                    else {
                                        continue;
                                    };
                                    let content = payload
                                        .get("content")
                                        .map(flatten_content)
                                        .unwrap_or_default();
                                    if content.trim().is_empty() {
                                        continue;
                                    }
                                    let extra = CodexConnector::normalized_message_extra(
                                        &val,
                                        compact_message_extra,
                                        role,
                                    )?;
                                    let invocations = payload.get("content").map_or_else(
                                        Vec::new,
                                        extract_invocations_from_content_blocks,
                                    );
                                    update_time_bounds(&mut started_at, &mut ended_at, created);
                                    messages.push(NormalizedMessage {
                                        idx: 0,
                                        role: role.to_string(),
                                        author: (role == "assistant")
                                            .then(|| current_model.clone())
                                            .flatten(),
                                        created_at: created,
                                        content,
                                        extra,
                                        invocations,
                                        snippets: Vec::new(),
                                    });
                                }
                                Some("agent_message") => {
                                    let (content, encrypted_content) =
                                        parse_agent_message_content(payload)?;
                                    if content.trim().is_empty() {
                                        if let Some(encrypted_content) = encrypted_content.as_ref()
                                        {
                                            let mut validation_extra = if compact_message_extra {
                                                CodexConnector::compact_message_extra(&val)
                                            } else {
                                                val.clone()
                                            };
                                            add_encrypted_content(
                                                &val,
                                                &mut validation_extra,
                                                encrypted_content,
                                            )?;
                                        }
                                        continue;
                                    }
                                    let mut extra = CodexConnector::normalized_message_extra(
                                        &val,
                                        compact_message_extra,
                                        "agent_message",
                                    )?;
                                    if let Some(encrypted_content) = encrypted_content.as_ref() {
                                        add_encrypted_content(&val, &mut extra, encrypted_content)?;
                                    }
                                    update_time_bounds(&mut started_at, &mut ended_at, created);
                                    messages.push(NormalizedMessage {
                                        idx: 0,
                                        role: "user".to_string(),
                                        author: None,
                                        created_at: created,
                                        content,
                                        extra,
                                        invocations: Vec::new(),
                                        snippets: Vec::new(),
                                    });
                                }
                                Some("reasoning") => {
                                    let content = reasoning_summary_text(payload)?;
                                    let encrypted_content = payload.get("encrypted_content");
                                    if content.trim().is_empty() && encrypted_content.is_none() {
                                        continue;
                                    }
                                    let mut extra = CodexConnector::normalized_message_extra(
                                        &val,
                                        compact_message_extra,
                                        "reasoning",
                                    )?;
                                    if let Some(encrypted_content) = encrypted_content {
                                        add_encrypted_content(&val, &mut extra, encrypted_content)?;
                                    }
                                    update_time_bounds(&mut started_at, &mut ended_at, created);
                                    messages.push(NormalizedMessage {
                                        idx: 0,
                                        role: "reasoning".to_string(),
                                        author: current_model.clone(),
                                        created_at: created,
                                        content,
                                        extra,
                                        invocations: Vec::new(),
                                        snippets: Vec::new(),
                                    });
                                }
                                Some(raw_role @ ("function_call" | "custom_tool_call")) => {
                                    let tool_name = payload
                                        .get("name")
                                        .and_then(Value::as_str)
                                        .unwrap_or("unknown")
                                        .to_string();
                                    let arguments = parse_tool_call_arguments(payload);
                                    let call_id = payload
                                        .get("call_id")
                                        .or_else(|| payload.get("id"))
                                        .and_then(Value::as_str)
                                        .map(String::from);
                                    let content =
                                        render_tool_call_content(&tool_name, arguments.as_ref());
                                    let mut extra = CodexConnector::normalized_message_extra(
                                        &val,
                                        compact_message_extra,
                                        raw_role,
                                    )?;
                                    let extra_object = extra
                                        .as_object_mut()
                                        .context("codex tool-call extra must be an object")?;
                                    if let Some(id) = call_id.as_ref() {
                                        extra_object.insert(
                                            "tool_call_id".to_string(),
                                            Value::String(id.clone()),
                                        );
                                    }
                                    extra_object.insert(
                                        "tool_call_args".to_string(),
                                        arguments.clone().unwrap_or(Value::Null),
                                    );
                                    update_time_bounds(&mut started_at, &mut ended_at, created);
                                    messages.push(NormalizedMessage {
                                        idx: 0,
                                        role: "tool_call".to_string(),
                                        author: current_model.clone(),
                                        created_at: created,
                                        content,
                                        extra,
                                        invocations: vec![NormalizedInvocation {
                                            kind: "tool".to_string(),
                                            name: tool_name,
                                            raw_name: None,
                                            call_id,
                                            arguments,
                                        }],
                                        snippets: Vec::new(),
                                    });
                                }
                                Some(
                                    raw_role @ ("function_call_output" | "custom_tool_call_output"),
                                ) => {
                                    let content = tool_output_text(payload);
                                    let call_id = payload
                                        .get("call_id")
                                        .and_then(Value::as_str)
                                        .map(String::from);
                                    let mut extra = CodexConnector::normalized_message_extra(
                                        &val,
                                        compact_message_extra,
                                        raw_role,
                                    )?;
                                    set_tool_result_pairing(&mut extra, call_id.as_deref())?;
                                    update_time_bounds(&mut started_at, &mut ended_at, created);
                                    messages.push(NormalizedMessage {
                                        idx: 0,
                                        role: "tool_result".to_string(),
                                        author: None,
                                        created_at: created,
                                        content,
                                        extra,
                                        invocations: Vec::new(),
                                        snippets: Vec::new(),
                                    });
                                }
                                Some(_) => {}
                            }
                        }
                        "event_msg" => {
                            let Some(payload) = val.get("payload") else {
                                continue;
                            };
                            let event_type = payload.get("type").and_then(Value::as_str);
                            // Event-layer agent messages duplicate the visible
                            // response item and are structural noise.
                            if event_type == Some("agent_message") {
                                continue;
                            }
                            match event_type {
                                Some("user_message") => {
                                    let text = payload
                                        .get("message")
                                        .and_then(Value::as_str)
                                        .unwrap_or("");
                                    if text.trim().is_empty() {
                                        continue;
                                    }
                                    let extra = CodexConnector::normalized_message_extra(
                                        &val,
                                        compact_message_extra,
                                        "user_message",
                                    )?;
                                    update_time_bounds(&mut started_at, &mut ended_at, created);
                                    messages.push(NormalizedMessage {
                                        idx: 0,
                                        role: "user".to_string(),
                                        author: None,
                                        created_at: created,
                                        content: text.to_string(),
                                        extra,
                                        invocations: Vec::new(),
                                        snippets: Vec::new(),
                                    });
                                }
                                Some("agent_reasoning") => {
                                    let text =
                                        payload.get("text").and_then(Value::as_str).unwrap_or("");
                                    if text.trim().is_empty() {
                                        continue;
                                    }
                                    let extra = CodexConnector::normalized_message_extra(
                                        &val,
                                        compact_message_extra,
                                        "agent_reasoning",
                                    )?;
                                    update_time_bounds(&mut started_at, &mut ended_at, created);
                                    messages.push(NormalizedMessage {
                                        idx: 0,
                                        role: "reasoning".to_string(),
                                        author: current_model.clone(),
                                        created_at: created,
                                        content: text.to_string(),
                                        extra,
                                        invocations: Vec::new(),
                                        snippets: Vec::new(),
                                    });
                                }
                                Some("tool_call") => {
                                    let tool_name = payload
                                        .get("name")
                                        .and_then(Value::as_str)
                                        .unwrap_or("unknown")
                                        .to_string();
                                    let arguments = payload
                                        .get("input")
                                        .or_else(|| payload.get("arguments"))
                                        .cloned();
                                    let call_id = payload
                                        .get("call_id")
                                        .or_else(|| payload.get("id"))
                                        .and_then(Value::as_str)
                                        .map(String::from);
                                    let content =
                                        render_tool_call_content(&tool_name, arguments.as_ref());
                                    let mut extra = CodexConnector::normalized_message_extra(
                                        &val,
                                        compact_message_extra,
                                        "tool_call",
                                    )?;
                                    let extra_object = extra
                                        .as_object_mut()
                                        .context("codex event tool-call extra must be an object")?;
                                    if let Some(id) = call_id.as_ref() {
                                        extra_object.insert(
                                            "tool_call_id".to_string(),
                                            Value::String(id.clone()),
                                        );
                                    }
                                    extra_object.insert(
                                        "tool_call_args".to_string(),
                                        arguments.clone().unwrap_or(Value::Null),
                                    );
                                    update_time_bounds(&mut started_at, &mut ended_at, created);
                                    messages.push(NormalizedMessage {
                                        idx: 0,
                                        role: "tool_call".to_string(),
                                        author: current_model.clone(),
                                        created_at: created,
                                        content,
                                        extra,
                                        invocations: vec![NormalizedInvocation {
                                            kind: "tool".to_string(),
                                            name: tool_name,
                                            raw_name: None,
                                            call_id,
                                            arguments,
                                        }],
                                        snippets: Vec::new(),
                                    });
                                }
                                Some("token_count") => {
                                    if let Some(token_usage) =
                                        CodexConnector::token_usage_from_payload(payload)?
                                    {
                                        CodexConnector::attach_token_usage_to_latest_assistant(
                                            &mut messages,
                                            token_usage,
                                            &source_path,
                                            line_idx + 1,
                                        );
                                    }
                                }
                                Some(_) | None => {}
                            }
                        }
                        _ => {}
                    }
                }
                crate::types::reindex_messages(&mut messages);
            } else if ext == Some("json") {
                let content = fs::read_to_string(&file)
                    .with_context(|| format!("read rollout {}", file.display()))?;
                let val: Value = match serde_json::from_str(&content) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                session_cwd = val
                    .get("session")
                    .and_then(|s| s.get("cwd"))
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from);

                if let Some(items) = val.get("items").and_then(|v| v.as_array()) {
                    for item in items {
                        let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("agent");
                        let content_str =
                            item.get("content").map(flatten_content).unwrap_or_default();

                        if content_str.trim().is_empty() {
                            continue;
                        }

                        let created = item.get("timestamp").and_then(parse_timestamp);
                        update_time_bounds(&mut started_at, &mut ended_at, created);

                        messages.push(NormalizedMessage {
                            idx: 0,
                            role: role.to_string(),
                            author: None,
                            created_at: created,
                            content: content_str,
                            extra: if compact_message_extra {
                                CodexConnector::compact_message_extra(item)
                            } else {
                                item.clone()
                            },
                            invocations: item
                                .get("content")
                                .map_or_else(Vec::new, extract_invocations_from_content_blocks),
                            snippets: Vec::new(),
                        });
                    }
                }
                crate::types::reindex_messages(&mut messages);
            }

            if messages.is_empty() {
                continue;
            }

            let title = messages
                .iter()
                .find(|m| m.role == "user")
                .map(|m| {
                    m.content
                        .lines()
                        .next()
                        .unwrap_or(&m.content)
                        .chars()
                        .take(100)
                        .collect::<String>()
                })
                .or_else(|| {
                    messages
                        .first()
                        .and_then(|m| m.content.lines().next())
                        .map(|s| s.chars().take(100).collect())
                });

            on_conversation(NormalizedConversation {
                agent_slug: "codex".to_string(),
                external_id,
                title,
                workspace: session_cwd,
                source_path: source_path.clone(),
                started_at,
                ended_at,
                metadata: serde_json::json!({"source": if ext == Some("json") { "rollout_json" } else { "rollout" }}),
                messages,
            })?;
        }
    }

    Ok(())
}

impl Connector for CodexConnector {
    fn detect(&self) -> DetectionResult {
        franken_detection_for_connector("codex").unwrap_or_else(DetectionResult::not_found)
    }

    fn scan(&self, ctx: &ScanContext) -> Result<Vec<NormalizedConversation>> {
        let mut convs = Vec::new();
        scan_codex_with_callback(ctx, &mut |conv| {
            convs.push(conv);
            Ok(())
        })?;
        Ok(convs)
    }

    fn supports_streaming_scan(&self) -> bool {
        true
    }

    fn discover_source_files(&self, ctx: &ScanContext) -> Result<Vec<DiscoveredSourceFile>> {
        Ok(Self::discover_sources(ctx))
    }

    fn scan_with_callback(
        &self,
        ctx: &ScanContext,
        on_conversation: &mut dyn FnMut(NormalizedConversation) -> Result<()>,
    ) -> Result<()> {
        scan_codex_with_callback(ctx, on_conversation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::scan::ScanRoot;
    use serde_json::json;
    use std::fs;
    use std::time::Instant;
    use tempfile::TempDir;

    fn scan_synthetic_jsonl(content: &str) -> Result<Vec<NormalizedConversation>> {
        let dir = TempDir::new()?;
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions)?;
        fs::write(sessions.join("rollout-synthetic.jsonl"), content)?;

        CodexConnector::new().scan(&ScanContext::local_default(codex_dir, None))
    }

    // =====================================================
    // Constructor Tests
    // =====================================================

    #[test]
    fn new_creates_connector() {
        let connector = CodexConnector::new();
        // Just verify it doesn't panic - struct has no fields
        let _ = connector;
    }

    #[test]
    fn default_creates_connector() {
        let connector = CodexConnector;
        let _ = connector;
    }

    // =====================================================
    // home() Tests
    // =====================================================

    #[test]
    fn home_returns_path_ending_with_codex() {
        // Note: We can't reliably test CODEX_HOME env var due to parallel test execution.
        // Testing that home() returns a valid path structure is sufficient.
        // The function uses CODEX_HOME if set, otherwise defaults to ~/.codex
        let home = CodexConnector::home();
        // Either the env var is set (ends with some path) or default (ends with .codex)
        let path_str = home.to_str().unwrap();
        let has_codex_dir = home
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                name.eq_ignore_ascii_case(".codex") || name.eq_ignore_ascii_case("codex")
            });
        assert!(
            has_codex_dir || path_str.to_ascii_lowercase().contains("codex"),
            "home() should return a path related to codex, got: {}",
            path_str
        );
    }

    // =====================================================
    // rollout_files() Tests
    // =====================================================

    #[test]
    fn rollout_files_finds_jsonl_files() {
        let dir = TempDir::new().unwrap();
        let sessions = dir.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let rollout = sessions.join("rollout-abc123.jsonl");
        fs::write(&rollout, "{}").unwrap();

        let files = CodexConnector::rollout_files(dir.path());
        assert_eq!(files.len(), 1);
        assert!(files[0].to_str().unwrap().contains("rollout-abc123.jsonl"));
    }

    #[test]
    fn rollout_files_finds_json_files() {
        let dir = TempDir::new().unwrap();
        let sessions = dir.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let rollout = sessions.join("rollout-legacy.json");
        fs::write(&rollout, "{}").unwrap();

        let files = CodexConnector::rollout_files(dir.path());
        assert_eq!(files.len(), 1);
        assert!(files[0].to_str().unwrap().contains("rollout-legacy.json"));
    }

    #[test]
    fn rollout_files_ignores_non_rollout_files() {
        let dir = TempDir::new().unwrap();
        let sessions = dir.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // Create various non-rollout files
        fs::write(sessions.join("config.json"), "{}").unwrap();
        fs::write(sessions.join("session.jsonl"), "{}").unwrap();
        fs::write(sessions.join("other.txt"), "test").unwrap();

        let files = CodexConnector::rollout_files(dir.path());
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn scan_with_explicit_home_root_finds_codex_sessions() {
        let dir = TempDir::new().unwrap();
        let home = dir.path().join("home");
        let sessions = home.join(".codex").join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Hello Codex"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"assistant","content":"Hi there!"}}
"#;
        let rollout = sessions.join("rollout-home.jsonl");
        fs::write(&rollout, content).unwrap();

        let connector = CodexConnector::new();
        let ctx =
            ScanContext::with_roots(dir.path().join("cass"), vec![ScanRoot::local(home)], None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 2);
        assert_eq!(convs[0].messages[0].content, "Hello Codex");
        assert_eq!(convs[0].messages[1].content, "Hi there!");
    }

    #[test]
    fn rollout_files_finds_nested_rollouts() {
        let dir = TempDir::new().unwrap();
        let nested = dir
            .path()
            .join("sessions")
            .join("2025")
            .join("12")
            .join("17");
        fs::create_dir_all(&nested).unwrap();

        let rollout = nested.join("rollout-nested.jsonl");
        fs::write(&rollout, "{}").unwrap();

        let files = CodexConnector::rollout_files(dir.path());
        assert_eq!(files.len(), 1);
        assert!(files[0].to_str().unwrap().contains("rollout-nested.jsonl"));
    }

    #[test]
    fn rollout_files_returns_sorted_order() {
        let dir = TempDir::new().unwrap();
        let sessions = dir.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        fs::write(sessions.join("rollout-z.jsonl"), "{}").unwrap();
        fs::write(sessions.join("rollout-a.jsonl"), "{}").unwrap();

        let files = CodexConnector::rollout_files(dir.path());
        assert_eq!(files.len(), 2);

        let names: Vec<_> = files
            .iter()
            .map(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        assert_eq!(names, vec!["rollout-a.jsonl", "rollout-z.jsonl"]);
    }

    #[test]
    fn rollout_files_returns_empty_when_no_sessions_dir() {
        let dir = TempDir::new().unwrap();
        let files = CodexConnector::rollout_files(dir.path());
        assert_eq!(files.len(), 0);
    }

    // =====================================================
    // scan() JSONL Format Tests
    // =====================================================

    #[test]
    fn scan_parses_jsonl_response_item_messages() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Hello Codex"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"assistant","content":"Hello! How can I help?"}}
"#;
        fs::write(sessions.join("rollout-test.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok());
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 2);
        assert_eq!(convs[0].messages[0].role, "user");
        assert_eq!(convs[0].messages[0].content, "Hello Codex");
        assert_eq!(convs[0].messages[1].role, "assistant");
    }

    #[test]
    fn scan_parses_modern_response_item_output_text_and_tool_calls() {
        // Regression test for #13: modern Codex rollout files encode assistant
        // text as `output_text` content blocks and encode tool calls/results as
        // `response_item` payloads (`function_call`, `function_call_output`,
        // `custom_tool_call`, `custom_tool_call_output`). None of these were
        // captured before, silently dropping assistant output and tool activity.
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("codex");

        let connector = CodexConnector::new();
        let ctx = ScanContext::with_roots(fixture.clone(), vec![ScanRoot::local(fixture)], None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1, "fixture is a single rollout");
        let conv = &convs[0];
        assert_eq!(conv.workspace, Some(PathBuf::from("/tmp/demo-project")));

        // user, assistant output_text, function_call, function_call_output,
        // custom_tool_call, custom_tool_call_output, reasoning = 7 messages.
        // The encrypted-only reasoning item is now EMITTED as an empty
        // structural reasoning message (completeness policy), not skipped --
        // its `encrypted_content` is preserved in `extra`.
        assert_eq!(
            conv.messages.len(),
            7,
            "all modern shapes captured (encrypted reasoning emitted): {:#?}",
            conv.messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect::<Vec<_>>()
        );

        // Encrypted-only reasoning: emitted with empty content, role
        // `reasoning`, and its opaque `encrypted_content` preserved in extra.
        let reasoning = conv
            .messages
            .iter()
            .find(|m| m.role == "reasoning")
            .expect("encrypted-only reasoning item is emitted, not dropped");
        assert!(
            reasoning.content.is_empty(),
            "encrypted-only reasoning has no plaintext content"
        );
        assert_eq!(
            reasoning
                .extra
                .pointer("/payload/encrypted_content")
                .and_then(|v| v.as_str()),
            Some("gAAAAA-opaque-no-plaintext"),
            "encrypted_content blob must be preserved in the reasoning message's extra"
        );

        // Assistant `output_text` is no longer dropped.
        let assistant = conv
            .messages
            .iter()
            .find(|m| {
                m.role == "assistant"
                    && m.content == "I will inspect the files and then apply a patch."
            })
            .expect("assistant output_text message captured");
        assert!(assistant.invocations.is_empty());

        // `function_call` -> its own `tool_call` message (not inlined into
        // assistant), with JSON-string arguments parsed and linkable via
        // `extra["tool_call_id"]`.
        let exec = conv
            .messages
            .iter()
            .find(|m| m.invocations.iter().any(|i| i.name == "exec_command"))
            .expect("exec_command function_call captured");
        assert_eq!(exec.role, "tool_call");
        let exec_inv = &exec.invocations[0];
        assert_eq!(exec_inv.kind, "tool");
        assert_eq!(exec_inv.call_id.as_deref(), Some("call_1"));
        assert_eq!(
            exec_inv
                .arguments
                .as_ref()
                .and_then(|a| a.get("cmd"))
                .and_then(|v| v.as_str()),
            Some("ls"),
            "JSON-string arguments are decoded into structured JSON"
        );
        assert_eq!(exec.extra["tool_call_id"].as_str(), Some("call_1"));

        // `function_call_output` -> `tool_result` (P0 rename from `tool`,
        // which the downstream adapter mistook for a tool call), linkable via
        // `extra["tool_call_id"]` -- the same key/value as its `tool_call`.
        let exec_out = conv
            .messages
            .iter()
            .find(|m| m.role == "tool_result" && m.content.contains("README.md"))
            .expect("function_call_output captured as tool_result");
        assert_eq!(
            exec_out.extra["tool_call_id"].as_str(),
            Some("call_1"),
            "tool_result remains linkable to its originating tool_call via extra[\"tool_call_id\"]"
        );
        assert_eq!(
            exec_out.extra["tool_call_id"], exec.extra["tool_call_id"],
            "tool_call and tool_result pair on the same tool_call_id"
        );

        // `custom_tool_call` (apply_patch) -> its own `tool_call` message;
        // freeform input kept.
        let patch = conv
            .messages
            .iter()
            .find(|m| m.invocations.iter().any(|i| i.name == "apply_patch"))
            .expect("apply_patch custom_tool_call captured");
        assert_eq!(patch.role, "tool_call");
        let patch_inv = &patch.invocations[0];
        assert_eq!(patch_inv.call_id.as_deref(), Some("call_2"));
        assert!(
            patch_inv
                .arguments
                .as_ref()
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.contains("Begin Patch")),
            "non-JSON tool input is retained as a raw string"
        );

        // `custom_tool_call_output` -> `tool_result` message.
        assert!(
            conv.messages
                .iter()
                .any(|m| m.role == "tool_result" && m.content.contains("A hello.txt")),
            "custom_tool_call_output captured as tool_result"
        );

        // No message uses a pre-6-role name.
        assert!(
            conv.messages
                .iter()
                .all(|m| !matches!(m.role.as_str(), "agent" | "tool" | "developer")),
            "roles: {:?}",
            conv.messages.iter().map(|m| &m.role).collect::<Vec<_>>()
        );
    }

    // =========================================================================
    // 6-role normalization tests (franken fork, spec §3.3 codex)
    // =========================================================================

    #[test]
    fn scan_codex_normalizes_to_six_role_messages() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // Real Codex rollout shapes (verified against ~/.codex/sessions/**/*.jsonl):
        // turn_context carries `model`; developer/user/assistant `message`
        // items; a `reasoning` item with plaintext `summary`; a paired
        // `function_call`/`function_call_output`.
        let content = concat!(
            r#"{"timestamp":"2026-01-01T00:00:00Z","type":"session_meta","payload":{"cwd":"/tmp/codex-demo"}}"#,
            "\n",
            r#"{"timestamp":"2026-01-01T00:00:01Z","type":"turn_context","payload":{"model":"gpt-5.5"}}"#,
            "\n",
            r#"{"timestamp":"2026-01-01T00:00:02Z","type":"response_item","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"You are Codex, a coding agent."}]}}"#,
            "\n",
            r#"{"timestamp":"2026-01-01T00:00:03Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"List the files in this repo."}]}}"#,
            "\n",
            r#"{"timestamp":"2026-01-01T00:00:04Z","type":"response_item","payload":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"I should run ls to see what's here."}],"encrypted_content":"gAAAAA-opaque"}}"#,
            "\n",
            r#"{"timestamp":"2026-01-01T00:00:05Z","type":"response_item","payload":{"type":"function_call","id":"fc_1","name":"exec_command","arguments":"{\"cmd\":\"ls -la\",\"workdir\":\"/tmp/codex-demo\"}","call_id":"call_1"}}"#,
            "\n",
            r#"{"timestamp":"2026-01-01T00:00:06Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call_1","output":"total 8\ndrwxr-xr-x  2 user user 4096 Jan  1 00:00 .\n-rw-r--r--  1 user user   12 Jan  1 00:00 README.md\n"}}"#,
            "\n",
            r#"{"timestamp":"2026-01-01T00:00:07Z","type":"response_item","payload":{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"I found README.md in the directory."}]}}"#,
            "\n",
        );
        fs::write(sessions.join("rollout-six-role.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        let conv = &convs[0];

        let developer_emitted_count = conv
            .messages
            .iter()
            .filter(|message| message.content == "You are Codex, a coding agent.")
            .count();
        assert_eq!(developer_emitted_count, 0);

        // function_call -> its own tool_call message, args non-empty.
        let tool_call = conv
            .messages
            .iter()
            .find(|m| m.role == "tool_call")
            .expect("tool_call message");
        assert!(
            tool_call.invocations[0].arguments.is_some(),
            "tool_call must carry non-empty args from function_call.arguments"
        );
        assert_eq!(
            tool_call.invocations[0]
                .arguments
                .as_ref()
                .and_then(|a| a.get("cmd"))
                .and_then(|v| v.as_str()),
            Some("ls -la")
        );
        let tool_call_id = tool_call
            .extra
            .get("tool_call_id")
            .and_then(|v| v.as_str())
            .expect("tool_call extra[\"tool_call_id\"] must be set");
        assert_eq!(tool_call_id, "call_1");

        // function_call_output -> role tool_result (P0 rename from "tool"),
        // full untruncated output, paired to the tool_call via tool_call_id.
        let tool_result = conv
            .messages
            .iter()
            .find(|m| m.role == "tool_result")
            .expect("tool_result message");
        assert_eq!(
            tool_result.content,
            "total 8\ndrwxr-xr-x  2 user user 4096 Jan  1 00:00 .\n-rw-r--r--  1 user user   12 Jan  1 00:00 README.md\n",
            "tool_result content must be the FULL output, never truncated/replaced with [tool call]"
        );
        assert_eq!(
            tool_result
                .extra
                .get("tool_call_id")
                .and_then(|v| v.as_str()),
            Some(tool_call_id),
            "tool_result pairs to its tool_call via extra[\"tool_call_id\"], not content order"
        );

        // reasoning -> role reasoning, author is the real model (not the
        // literal string "reasoning", and not empty since the model is known).
        let reasoning = conv
            .messages
            .iter()
            .find(|m| m.role == "reasoning")
            .expect("reasoning message");
        assert_eq!(reasoning.content, "I should run ls to see what's here.");
        assert_ne!(reasoning.author.as_deref(), Some("reasoning"));
        assert_eq!(reasoning.author.as_deref(), Some("gpt-5.5"));

        // No message anywhere uses a pre-6-role name.
        assert!(
            conv.messages
                .iter()
                .all(|m| !matches!(m.role.as_str(), "agent" | "tool" | "developer")),
            "roles: {:?}",
            conv.messages.iter().map(|m| &m.role).collect::<Vec<_>>()
        );

        // idx must be contiguous 0..N.
        assert!(
            conv.messages
                .iter()
                .enumerate()
                .all(|(i, m)| m.idx as usize == i)
        );
    }

    #[test]
    fn scan_emits_empty_tool_result_and_preserves_pairing() {
        // Completeness policy (uniform with claude_code.rs): a command that
        // succeeds with no stdout produces a legitimate EMPTY tool_result --
        // it must still be emitted (not dropped), so the tool_call<->tool_result
        // pairing chain via `extra["tool_call_id"]` survives.
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"type":"function_call","id":"fc_1","name":"exec_command","arguments":"{\"cmd\":\"true\"}","call_id":"call_empty"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"type":"function_call_output","call_id":"call_empty","output":""}}
"#;
        fs::write(sessions.join("rollout-empty-output.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        let tool_result = convs[0]
            .messages
            .iter()
            .find(|m| m.role == "tool_result")
            .expect("empty function_call_output must still be emitted as a tool_result");
        assert_eq!(
            tool_result.content, "",
            "empty output stays empty, not dropped"
        );
        assert_eq!(
            tool_result
                .extra
                .get("tool_call_id")
                .and_then(|v| v.as_str()),
            Some("call_empty"),
            "pairing to the tool_call must survive even with empty output"
        );
    }

    #[test]
    fn scan_emits_encrypted_only_reasoning_and_preserves_blob() {
        // Completeness policy: ~99.99% of real codex reasoning items are
        // encrypted-only (empty `summary`, opaque `encrypted_content`). They
        // must be emitted as empty structural reasoning messages, preserving
        // `encrypted_content` in `extra` (never decrypted).
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"turn_context","timestamp":"2025-12-01T09:59:59Z","payload":{"model":"gpt-5.5"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAAA-secret-blob"}}
"#;
        fs::write(sessions.join("rollout-encrypted-reasoning.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        let reasoning = convs[0]
            .messages
            .iter()
            .find(|m| m.role == "reasoning")
            .expect("encrypted-only reasoning must still be emitted, not dropped");
        assert!(
            reasoning.content.is_empty(),
            "encrypted-only reasoning has empty content"
        );
        assert_eq!(reasoning.author.as_deref(), Some("gpt-5.5"));
        // Non-compact path: the whole payload is carried, so the blob is
        // reachable under /payload.
        assert_eq!(
            reasoning
                .extra
                .pointer("/payload/encrypted_content")
                .and_then(|v| v.as_str()),
            Some("gAAAAA-secret-blob"),
            "encrypted_content must survive in the reasoning message's extra"
        );
        assert_eq!(
            reasoning
                .extra
                .get("encrypted_content")
                .and_then(|v| v.as_str()),
            Some("gAAAAA-secret-blob"),
            "encrypted_content must also use the normalized top-level field"
        );
    }

    #[test]
    fn scan_with_callback_matches_scan_for_jsonl_rollout() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Hello Codex"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"assistant","content":"Hello! How can I help?"}}
"#;
        fs::write(sessions.join("rollout-stream.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let scanned = connector.scan(&ctx).unwrap();
        let mut streamed = Vec::new();
        connector
            .scan_with_callback(&ctx, &mut |conversation| {
                streamed.push(conversation);
                Ok(())
            })
            .unwrap();

        assert_eq!(streamed.len(), scanned.len());
        assert_eq!(streamed[0].messages.len(), scanned[0].messages.len());
        assert_eq!(
            streamed[0].messages[0].content,
            scanned[0].messages[0].content
        );
        assert_eq!(
            streamed[0].messages[1].content,
            scanned[0].messages[1].content
        );
    }

    #[test]
    fn discover_source_files_matches_scanned_rollout_sources() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Hello Codex"}}"#;
        let rollout = sessions.join("rollout-discovery.jsonl");
        fs::write(&rollout, content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir, None);
        let discovered = connector.discover_source_files(&ctx).unwrap();
        let scanned = connector.scan(&ctx).unwrap();

        assert_eq!(scanned.len(), 1);
        assert_eq!(
            discovered
                .iter()
                .map(|source| source.source_path.clone())
                .collect::<Vec<_>>(),
            vec![scanned[0].source_path.clone()]
        );
        assert_eq!(discovered[0].role, DiscoveredSourceRole::PrimarySessionLog);
        assert!(discovered[0].required_for_reconstruction);
        assert_eq!(discovered[0].provider_slug, "codex");
        assert_eq!(discovered[0].source_path, rollout);
    }

    #[test]
    fn scan_with_explicit_rollout_file_only_reads_that_file_and_keeps_relative_external_id() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir
            .join("sessions")
            .join("2025")
            .join("12")
            .join("18");
        fs::create_dir_all(&sessions).unwrap();

        let first = sessions.join("rollout-one.jsonl");
        let second = sessions.join("rollout-two.jsonl");
        fs::write(
            &first,
            r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"first only"}}"#,
        )
        .unwrap();
        fs::write(
            &second,
            r#"{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"user","content":"second only"}}"#,
        )
        .unwrap();

        let connector = CodexConnector::new();
        let ctx =
            ScanContext::with_roots(first.clone(), vec![ScanRoot::local(first.clone())], None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].content, "first only");
        assert_eq!(
            convs[0].external_id.as_deref(),
            Some("2025/12/18/rollout-one")
        );
        assert_eq!(convs[0].source_path, first);
    }

    #[test]
    #[ignore = "release-mode performance harness; run explicitly for Codex scan wall-clock evidence"]
    fn perf_scan_large_codex_fixture() {
        let file_count = std::env::var("FAD_CODEX_SCAN_BENCH_FILES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2_000);
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");

        for i in 0..file_count {
            let day_dir = sessions
                .join("2026")
                .join("05")
                .join(format!("{:02}", (i % 28) + 1));
            fs::create_dir_all(&day_dir).unwrap();
            let ts = 1_746_265_600_000_u64 + u64::try_from(i).unwrap();
            fs::write(
                day_dir.join(format!("rollout-{i:06}.jsonl")),
                format!(
                    r#"{{"type":"event_msg","timestamp":{ts},"payload":{{"type":"user_message","message":"bench user {i}"}}}}
{{"type":"response_item","timestamp":{},"payload":{{"role":"assistant","content":"bench assistant {i}"}}}}
"#,
                    ts + 1
                ),
            )
            .unwrap();
        }

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir, None);
        let start = Instant::now();
        let convs = connector.scan(&ctx).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(convs.len(), file_count);
        eprintln!(
            "fad_codex_scan_large_fixture files={} elapsed_ms={} ns_per_file={}",
            file_count,
            elapsed.as_millis(),
            elapsed.as_nanos() / u128::try_from(file_count).unwrap()
        );
    }

    #[test]
    fn compact_message_extra_keeps_only_cass_metadata() {
        let raw = json!({
            "model": "gpt-5-codex",
            "attachments": [{"path": "/tmp/screenshot.png"}],
            "payload": {
                "content": "very large duplicated content"
            }
        });

        let compact = CodexConnector::compact_message_extra(&raw);
        assert_eq!(compact["cass"]["model"], "gpt-5-codex");
        assert_eq!(
            compact["cass"]["attachments"][0]["path"],
            "/tmp/screenshot.png"
        );
        assert!(compact.get("payload").is_none());
    }

    #[test]
    fn should_compact_large_message_extra_respects_threshold() {
        assert!(!CodexConnector::should_compact_large_message_extra(Some(
            LARGE_SESSION_EXTRA_COMPACT_THRESHOLD_BYTES - 1,
        )));
        assert!(CodexConnector::should_compact_large_message_extra(Some(
            LARGE_SESSION_EXTRA_COMPACT_THRESHOLD_BYTES,
        )));
        assert!(!CodexConnector::should_compact_large_message_extra(None));
    }

    #[test]
    fn file_metadata_if_modified_preserves_scan_fallbacks() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("rollout-meta.jsonl");
        fs::write(&file, "{}").unwrap();

        assert!(matches!(
            CodexConnector::file_metadata_if_modified(&file, None),
            FileScanMetadata::Process(Some(_))
        ));
        assert!(
            matches!(
                CodexConnector::file_metadata_if_modified(&file, Some(i64::MAX)),
                FileScanMetadata::Skip
            ),
            "future since_ts should skip older files"
        );

        let missing = dir.path().join("missing.jsonl");
        assert!(matches!(
            CodexConnector::file_metadata_if_modified(&missing, Some(i64::MAX)),
            FileScanMetadata::Process(None)
        ));
    }

    #[test]
    fn scan_parses_event_msg_user_message() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"event_msg","timestamp":"2025-12-01T10:00:00Z","payload":{"type":"user_message","message":"User typed this"}}
"#;
        fs::write(sessions.join("rollout-user.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].role, "user");
        assert_eq!(convs[0].messages[0].content, "User typed this");
        assert!(convs[0].started_at.is_some());
        assert!(convs[0].ended_at.is_some());
    }

    #[test]
    fn scan_parses_event_msg_agent_reasoning() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // `turn_context.model` precedes the reasoning event, like real
        // rollouts, so `author` reflects the real model instead of the old
        // literal `"reasoning"` mislabeling.
        let content = r#"{"type":"turn_context","timestamp":"2025-12-01T09:59:59Z","payload":{"model":"gpt-5.5"}}
{"type":"event_msg","timestamp":"2025-12-01T10:00:00Z","payload":{"type":"agent_reasoning","text":"Let me think about this..."}}
"#;
        fs::write(sessions.join("rollout-reasoning.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].role, "reasoning");
        assert_eq!(convs[0].messages[0].author, Some("gpt-5.5".to_string()));
        assert_eq!(convs[0].messages[0].content, "Let me think about this...");
        assert!(convs[0].started_at.is_some());
        assert!(convs[0].ended_at.is_some());
    }

    #[test]
    fn scan_extracts_workspace_from_session_meta() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"session_meta","timestamp":"2025-12-01T10:00:00Z","payload":{"cwd":"/home/user/project"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"user","content":"Test"}}
"#;
        fs::write(sessions.join("rollout-meta.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].workspace,
            Some(PathBuf::from("/home/user/project"))
        );
    }

    #[test]
    fn scan_skips_empty_lines_in_jsonl() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Message 1"}}

{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"user","content":"Message 2"}}
"#;
        fs::write(sessions.join("rollout-empty-lines.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 2);
    }

    #[test]
    fn scan_skips_invalid_json_lines() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Valid"}}
not valid json at all
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"user","content":"Also valid"}}
"#;
        fs::write(sessions.join("rollout-invalid.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 2);
    }

    #[test]
    fn scan_skips_empty_content_messages() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Has content"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"assistant","content":""}}
{"type":"response_item","timestamp":"2025-12-01T10:00:02Z","payload":{"role":"assistant","content":"   "}}
"#;
        fs::write(sessions.join("rollout-empty-content.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        // Only the message with actual content should be included
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].content, "Has content");
    }

    #[test]
    fn scan_skips_unknown_event_types() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Real message"}}
{"type":"event_msg","timestamp":"2025-12-01T10:00:01Z","payload":{"type":"token_count","tokens":100}}
{"type":"event_msg","timestamp":"2025-12-01T10:00:02Z","payload":{"type":"turn_aborted"}}
{"type":"turn_context","timestamp":"2025-12-01T10:00:03Z","payload":{"model":"gpt-synthetic"}}
"#;
        fs::write(sessions.join("rollout-unknown.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        // Only the response_item should be included
        assert_eq!(convs[0].messages.len(), 1);
    }

    #[test]
    fn scan_assigns_sequential_indices() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"First"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"assistant","content":"Second"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:02Z","payload":{"role":"user","content":"Third"}}
"#;
        fs::write(sessions.join("rollout-idx.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].messages[0].idx, 0);
        assert_eq!(convs[0].messages[1].idx, 1);
        assert_eq!(convs[0].messages[2].idx, 2);
    }

    #[test]
    fn scan_attaches_token_count_to_nearest_preceding_assistant() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Question"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"assistant","content":"First answer"}}
{"type":"event_msg","timestamp":"2025-12-01T10:00:02Z","payload":{"type":"token_count","input_tokens":10,"output_tokens":20}}
{"type":"response_item","timestamp":"2025-12-01T10:00:03Z","payload":{"role":"assistant","content":"Second answer"}}
{"type":"event_msg","timestamp":"2025-12-01T10:00:04Z","payload":{"type":"token_count","input_tokens":30,"output_tokens":40}}
"#;
        fs::write(sessions.join("rollout-attach-nearest.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].messages.len(),
            3,
            "no synthetic token_count messages"
        );

        let first = &convs[0].messages[1];
        assert_eq!(first.content, "First answer");
        assert_eq!(
            first
                .extra
                .pointer("/cass/token_usage/input_tokens")
                .and_then(Value::as_i64),
            Some(10)
        );
        assert_eq!(
            first
                .extra
                .pointer("/cass/token_usage/output_tokens")
                .and_then(Value::as_i64),
            Some(20)
        );

        let second = &convs[0].messages[2];
        assert_eq!(second.content, "Second answer");
        assert_eq!(
            second
                .extra
                .pointer("/cass/token_usage/input_tokens")
                .and_then(Value::as_i64),
            Some(30)
        );
        assert_eq!(
            second
                .extra
                .pointer("/cass/token_usage/output_tokens")
                .and_then(Value::as_i64),
            Some(40)
        );
        assert_eq!(
            second
                .extra
                .pointer("/cass/token_usage/data_source")
                .and_then(|v| v.as_str()),
            Some("api")
        );
    }

    #[test]
    fn scan_ignores_token_count_without_preceding_assistant() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Question"}}
{"type":"event_msg","timestamp":"2025-12-01T10:00:01Z","payload":{"type":"token_count","input_tokens":11,"output_tokens":22}}
{"type":"response_item","timestamp":"2025-12-01T10:00:02Z","payload":{"role":"assistant","content":"Answer later"}}
"#;
        fs::write(sessions.join("rollout-unmatched-token.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 2);
        assert!(
            convs[0].messages[1]
                .extra
                .pointer("/cass/token_usage")
                .is_none(),
            "token_count before first assistant must not attach to future message"
        );
    }

    #[test]
    fn scan_multiple_token_count_for_one_assistant_prefers_last() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Question"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"assistant","content":"Answer"}}
{"type":"event_msg","timestamp":"2025-12-01T10:00:02Z","payload":{"type":"token_count","input_tokens":5,"output_tokens":10}}
{"type":"event_msg","timestamp":"2025-12-01T10:00:03Z","payload":{"type":"token_count","input_tokens":7,"output_tokens":14}}
"#;
        fs::write(sessions.join("rollout-token-last-wins.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 2);
        let assistant = &convs[0].messages[1];
        assert_eq!(
            assistant
                .extra
                .pointer("/cass/token_usage/input_tokens")
                .and_then(Value::as_i64),
            Some(7)
        );
        assert_eq!(
            assistant
                .extra
                .pointer("/cass/token_usage/output_tokens")
                .and_then(Value::as_i64),
            Some(14)
        );
    }

    #[test]
    fn scan_attaches_token_count_to_assistant_with_model_author() {
        // Regression guard for the `is_token_usage_target_message` change
        // (`author.is_none()` -> `role == "assistant"`). Real rollouts carry
        // a `turn_context.model`, so real assistant messages get
        // `author = Some(model)`. The OLD guard required `author.is_none()`
        // and would therefore SKIP attaching token usage to every real
        // assistant turn -- yet every other token-usage test uses a fixture
        // with no `turn_context`, so `author` stays `None` and the old buggy
        // guard passes them. This test includes a `turn_context` line so the
        // assistant has a real model author, proving token usage still
        // attaches.
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"turn_context","timestamp":"2025-12-01T09:59:59Z","payload":{"model":"gpt-5.5"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"Question"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:01Z","payload":{"role":"assistant","content":"Answer"}}
{"type":"event_msg","timestamp":"2025-12-01T10:00:02Z","payload":{"type":"token_count","input_tokens":13,"output_tokens":21}}
"#;
        fs::write(sessions.join("rollout-token-model-author.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        let assistant = &convs[0].messages[1];
        assert_eq!(assistant.content, "Answer");
        // The assistant carries the real model as its author -- exactly the
        // case the old guard would have excluded.
        assert_eq!(assistant.author.as_deref(), Some("gpt-5.5"));
        assert_eq!(
            assistant
                .extra
                .pointer("/cass/token_usage/input_tokens")
                .and_then(Value::as_i64),
            Some(13),
            "token usage must still attach to a real assistant turn that has a model author"
        );
        assert_eq!(
            assistant
                .extra
                .pointer("/cass/token_usage/output_tokens")
                .and_then(Value::as_i64),
            Some(21)
        );
    }

    // =====================================================
    // scan() Legacy JSON Format Tests
    // =====================================================

    #[test]
    fn scan_parses_legacy_json_format() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = json!({
            "session": {"cwd": "/home/user/legacy"},
            "items": [
                {"role": "user", "content": "Legacy user message", "timestamp": "2025-12-01T10:00:00Z"},
                {"role": "assistant", "content": "Legacy assistant response", "timestamp": "2025-12-01T10:00:01Z"}
            ]
        });
        fs::write(sessions.join("rollout-legacy.json"), content.to_string()).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].workspace, Some(PathBuf::from("/home/user/legacy")));
        assert_eq!(convs[0].messages.len(), 2);
        assert_eq!(convs[0].messages[0].role, "user");
        assert_eq!(convs[0].messages[0].content, "Legacy user message");
        assert_eq!(convs[0].messages[1].role, "assistant");
    }

    #[test]
    fn scan_legacy_json_skips_empty_content() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = json!({
            "session": {},
            "items": [
                {"role": "user", "content": "Has content"},
                {"role": "assistant", "content": ""},
                {"role": "assistant", "content": "   "}
            ]
        });
        fs::write(
            sessions.join("rollout-empty-legacy.json"),
            content.to_string(),
        )
        .unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 1);
    }

    #[test]
    fn scan_legacy_json_handles_missing_items() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = json!({"session": {}});
        fs::write(sessions.join("rollout-no-items.json"), content.to_string()).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        // No messages = conversation is skipped
        assert_eq!(convs.len(), 0);
    }

    #[test]
    fn scan_skips_invalid_legacy_json() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        fs::write(sessions.join("rollout-bad.json"), "not valid json").unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 0);
    }

    // =====================================================
    // Title Extraction Tests
    // =====================================================

    #[test]
    fn scan_extracts_title_from_first_user_message() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","payload":{"role":"assistant","content":"I'm an assistant"}}
{"type":"response_item","payload":{"role":"user","content":"This should be the title"}}
"#;
        fs::write(sessions.join("rollout-title.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].title, Some("This should be the title".to_string()));
    }

    #[test]
    fn scan_truncates_long_titles() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let long_title = "x".repeat(200);
        let content = format!(
            r#"{{"type":"response_item","payload":{{"role":"user","content":"{}"}}}}"#,
            long_title
        );
        fs::write(sessions.join("rollout-long.jsonl"), content + "\n").unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].title.as_ref().unwrap().len(), 100);
    }

    #[test]
    fn scan_uses_first_line_for_multiline_title() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","payload":{"role":"user","content":"First line\nSecond line\nThird line"}}
"#;
        fs::write(sessions.join("rollout-multiline.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].title, Some("First line".to_string()));
    }

    #[test]
    fn scan_falls_back_to_first_message_for_title() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // No user messages, only assistant
        let content = r#"{"type":"response_item","payload":{"role":"assistant","content":"Assistant speaks first"}}
"#;
        fs::write(sessions.join("rollout-assistant-only.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].title, Some("Assistant speaks first".to_string()));
    }

    // =====================================================
    // External ID Tests
    // =====================================================

    #[test]
    fn scan_uses_relative_path_as_external_id() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir
            .join("sessions")
            .join("2025")
            .join("12")
            .join("17");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","payload":{"role":"user","content":"Test"}}
"#;
        fs::write(sessions.join("rollout-nested-id.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        // External ID should be the relative path from sessions dir
        assert!(convs[0].external_id.is_some());
        let ext_id = convs[0].external_id.as_ref().unwrap();
        assert!(ext_id.contains("2025") || ext_id.contains("rollout-nested-id"));
    }

    // =====================================================
    // Metadata Tests
    // =====================================================

    #[test]
    fn scan_sets_metadata_source_for_jsonl() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","payload":{"role":"user","content":"Test"}}
"#;
        fs::write(sessions.join("rollout-meta-jsonl.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].metadata["source"], "rollout");
    }

    #[test]
    fn scan_sets_metadata_source_for_json() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = json!({
            "session": {},
            "items": [{"role": "user", "content": "Test"}]
        });
        fs::write(sessions.join("rollout-meta-json.json"), content.to_string()).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].metadata["source"], "rollout_json");
    }

    // =====================================================
    // Agent Slug Tests
    // =====================================================

    #[test]
    fn scan_sets_agent_slug_to_codex() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","payload":{"role":"user","content":"Test"}}
"#;
        fs::write(sessions.join("rollout-slug.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].agent_slug, "codex");
    }

    // =====================================================
    // Timestamp Tests
    // =====================================================

    #[test]
    fn scan_parses_timestamps() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"First"}}
{"type":"response_item","timestamp":"2025-12-01T11:00:00Z","payload":{"role":"user","content":"Last"}}
"#;
        fs::write(sessions.join("rollout-ts.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert!(convs[0].started_at.is_some());
        assert!(convs[0].ended_at.is_some());
        assert!(convs[0].messages[0].created_at.is_some());
    }

    #[test]
    fn scan_tracks_timestamp_bounds_for_out_of_order_events() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","timestamp":"2025-12-01T11:00:00Z","payload":{"role":"assistant","content":"Second chronologically"}}
{"type":"response_item","timestamp":"2025-12-01T10:00:00Z","payload":{"role":"user","content":"First chronologically"}}
"#;
        fs::write(sessions.join("rollout-out-of-order.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        let conv = &convs[0];
        let expected_start = conv.messages.iter().filter_map(|m| m.created_at).min();
        let expected_end = conv.messages.iter().filter_map(|m| m.created_at).max();

        assert_eq!(conv.started_at, expected_start);
        assert_eq!(conv.ended_at, expected_end);
        assert!(conv.ended_at >= conv.started_at);
    }

    // =====================================================
    // Edge Cases
    // =====================================================

    #[test]
    fn scan_handles_empty_sessions_dir() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        // No files in sessions directory

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 0);
    }

    #[test]
    fn scan_handles_multiple_rollout_files() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content1 = r#"{"type":"response_item","payload":{"role":"user","content":"Session 1"}}
"#;
        let content2 = r#"{"type":"response_item","payload":{"role":"user","content":"Session 2"}}
"#;
        fs::write(sessions.join("rollout-1.jsonl"), content1).unwrap();
        fs::write(sessions.join("rollout-2.jsonl"), content2).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 2);
    }

    #[test]
    fn scan_skips_conversations_with_no_messages() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // Only metadata, no actual messages
        let content = r#"{"type":"session_meta","timestamp":"2025-12-01T10:00:00Z","payload":{"cwd":"/test"}}
{"type":"turn_context","timestamp":"2025-12-01T10:00:01Z","payload":{"model":"gpt-synthetic"}}
"#;
        fs::write(sessions.join("rollout-no-msgs.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        // Should be skipped because no actual messages
        assert_eq!(convs.len(), 0);
    }

    #[test]
    fn scan_handles_array_content_in_response_item() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // Content as array of text blocks (like Claude API format)
        let content = json!({
            "type": "response_item",
            "payload": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Part one."},
                    {"type": "text", "text": " Part two."}
                ]
            }
        });
        fs::write(
            sessions.join("rollout-array.jsonl"),
            content.to_string() + "\n",
        )
        .unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        // flatten_content should combine the parts
        assert!(convs[0].messages[0].content.contains("Part one"));
    }

    #[test]
    fn scan_drops_modern_response_item_without_type_or_role() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // No role specified in payload
        let content = r#"{"type":"response_item","payload":{"content":"No role specified"}}
"#;
        fs::write(sessions.join("rollout-no-role.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert!(convs.is_empty());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn scan_codex_response_item_and_event_msg_role_matrix() {
        let content = concat!(
            r#"{"type":"session_meta","timestamp":"2026-07-23T00:00:00Z","payload":{"cwd":"/tmp/synthetic-codex"}}"#,
            "\n",
            r#"{"type":"turn_context","timestamp":"2026-07-23T00:00:01Z","payload":{"model":"gpt-synthetic"}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:02Z","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"drop developer"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:03Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"modern user"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:04Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"modern assistant"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:05Z","payload":{"role":"user","content":"legacy user"}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:06Z","payload":{"role":"assistant","content":"legacy assistant"}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:07Z","payload":{"content":"unclassifiable"}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:08Z","payload":{"type":"agent_message","content":[{"type":"input_text","text":"agent input"},{"type":"text","text":"agent text"},{"type":"output_text","text":"agent output"},{"type":"encrypted_content","encrypted_content":"opaque-agent"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:09Z","payload":{"type":"reasoning","summary":[{"type":"summary_text","text":"visible reasoning"}],"content":[{"type":"text","text":"wrong reasoning source"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:10Z","payload":{"type":"function_call","name":"exec_command","arguments":"{\"cmd\":\"true\"}","call_id":"call-1"}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:11Z","payload":{"type":"custom_tool_call","name":"apply_patch","input":"synthetic patch","call_id":"call-2"}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:12Z","payload":{"type":"function_call_output","call_id":"call-1","output":""}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:13Z","payload":{"type":"custom_tool_call_output","output":"unpaired output"}}"#,
            "\n",
            r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:14Z","payload":{"type":"user_message","message":"event user"}}"#,
            "\n",
            r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:15Z","payload":{"type":"agent_reasoning","text":"event reasoning"}}"#,
            "\n",
            r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:16Z","payload":{"type":"tool_call","name":"event_tool","input":{"value":1},"call_id":"event-call"}}"#,
            "\n",
            r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:17Z","payload":{"type":"agent_message","message":"drop event agent"}}"#,
            "\n",
        );

        let convs = scan_synthetic_jsonl(content).unwrap();
        let conv = &convs[0];
        let developer_emitted_count = conv
            .messages
            .iter()
            .filter(|message| message.content == "drop developer")
            .count();
        assert_eq!(developer_emitted_count, 0);
        assert!(
            conv.messages
                .iter()
                .all(|message| message.content != "unclassifiable")
        );

        for (content, role, raw_role) in [
            ("modern user", "user", "user"),
            ("modern assistant", "assistant", "assistant"),
            ("legacy user", "user", "user"),
            ("legacy assistant", "assistant", "assistant"),
            ("visible reasoning", "reasoning", "reasoning"),
            ("event user", "user", "user_message"),
            ("event reasoning", "reasoning", "agent_reasoning"),
        ] {
            let message = conv
                .messages
                .iter()
                .find(|message| message.content == content)
                .unwrap_or_else(|| panic!("missing {content}"));
            assert_eq!(message.role, role);
            assert_eq!(message.extra["raw_role"], raw_role);
        }

        let agent_message = conv
            .messages
            .iter()
            .find(|message| message.content.contains("agent input"))
            .expect("visible response_item agent_message");
        assert_eq!(
            (
                agent_message.role.as_str(),
                agent_message.extra["raw_role"].as_str()
            ),
            ("user", Some("agent_message"))
        );
        assert!(agent_message.content.contains("agent text"));
        assert!(agent_message.content.contains("agent output"));
        assert_eq!(
            agent_message.extra["encrypted_content"].as_str(),
            Some("opaque-agent")
        );

        let reasoning = conv
            .messages
            .iter()
            .find(|message| message.content == "visible reasoning")
            .unwrap();
        assert!(!reasoning.content.contains("wrong reasoning source"));
        assert_eq!(reasoning.author.as_deref(), Some("gpt-synthetic"));

        for (name, raw_role) in [
            ("exec_command", "function_call"),
            ("apply_patch", "custom_tool_call"),
            ("event_tool", "tool_call"),
        ] {
            let message = conv
                .messages
                .iter()
                .find(|message| message.invocations.iter().any(|item| item.name == name))
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(message.role, "tool_call");
            assert_eq!(message.extra["raw_role"], raw_role);
            assert_eq!(message.author.as_deref(), Some("gpt-synthetic"));
        }

        let paired = conv
            .messages
            .iter()
            .find(|message| message.role == "tool_result" && message.content.is_empty())
            .expect("empty paired result retained");
        assert_eq!(paired.extra["raw_role"], "function_call_output");
        assert_eq!(paired.extra["tool_call_id"], "call-1");
        assert!(paired.extra.get("unpaired").is_none());

        let unpaired = conv
            .messages
            .iter()
            .find(|message| message.content == "unpaired output")
            .expect("missing-id result retained");
        assert_eq!(unpaired.extra["raw_role"], "custom_tool_call_output");
        assert_eq!(unpaired.extra["unpaired"], true);
        assert!(unpaired.extra.get("tool_call_id").is_none());

        assert!(
            conv.messages
                .iter()
                .all(|message| message.content != "drop event agent")
        );
        assert!(
            conv.messages
                .iter()
                .enumerate()
                .all(|(idx, message)| i64::try_from(idx) == Ok(message.idx))
        );
    }

    #[test]
    fn scan_codex_agent_message_metadata_only_blocks_do_not_emit() {
        let content = concat!(
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"agent_message","content":[{"type":"encrypted_content","encrypted_content":"opaque-only"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:01Z","payload":{"type":"agent_message","content":[{"type":"input_image","image_url":"synthetic://image"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:02Z","payload":{"type":"agent_message","content":[{"type":"refusal","refusal":"synthetic refusal"}]}}"#,
            "\n",
        );
        assert!(scan_synthetic_jsonl(content).unwrap().is_empty());
    }

    #[test]
    fn scan_codex_review_encrypted_only_agent_message_rejects_reserved_field_collision() {
        let content = r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"collision","payload":{"type":"agent_message","content":[{"type":"encrypted_content","encrypted_content":"opaque"}]}}"#;
        let error = scan_synthetic_jsonl(content).unwrap_err();
        assert!(error.to_string().contains("encrypted_content"), "{error:#}");
    }

    #[test]
    fn scan_codex_visible_agent_message_rejects_top_level_encrypted_content_collision() {
        let collision_value = "must-not-appear-in-errors-small-agent";
        let content = format!(
            r#"{{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"{collision_value}","payload":{{"type":"agent_message","content":[{{"type":"text","text":"visible agent"}}]}}}}"#
        );

        let error = scan_synthetic_jsonl(&content).unwrap_err();
        assert!(error.to_string().contains("encrypted_content"), "{error:#}");
        assert!(!error.to_string().contains(collision_value), "{error:#}");
    }

    #[test]
    fn scan_codex_visible_reasoning_rejects_top_level_encrypted_content_collision() {
        let collision_value = "must-not-appear-in-errors-reasoning";
        let content = format!(
            r#"{{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"{collision_value}","payload":{{"type":"reasoning","summary":[{{"type":"summary_text","text":"visible reasoning"}}]}}}}"#
        );

        let error = scan_synthetic_jsonl(&content).unwrap_err();
        assert!(error.to_string().contains("encrypted_content"), "{error:#}");
        assert!(!error.to_string().contains(collision_value), "{error:#}");
    }

    #[test]
    fn scan_codex_visible_agent_and_reasoning_without_collision_are_retained() {
        let content = concat!(
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"agent_message","content":[{"type":"text","text":"visible agent"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:01Z","payload":{"type":"reasoning","summary":[{"type":"summary_text","text":"visible reasoning"}]}}"#,
            "\n",
        );

        let convs = scan_synthetic_jsonl(content).unwrap();
        let messages = &convs[0].messages;
        assert_eq!(messages.len(), 2);
        assert_eq!(
            (
                messages[0].role.as_str(),
                messages[0].content.as_str(),
                messages[0].extra["raw_role"].as_str()
            ),
            ("user", "visible agent", Some("agent_message"))
        );
        assert_eq!(
            (
                messages[1].role.as_str(),
                messages[1].content.as_str(),
                messages[1].extra["raw_role"].as_str()
            ),
            ("reasoning", "visible reasoning", Some("reasoning"))
        );
    }

    #[test]
    fn scan_codex_all_modern_retained_branches_reject_top_level_encrypted_content_collision() {
        let collision_value = "must-not-appear-in-retained-branch-errors";
        let cases = [
            (
                "response user message",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"message","role":"user","content":"visible user"}}"#,
            ),
            (
                "response assistant message",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"message","role":"assistant","content":"visible assistant"}}"#,
            ),
            (
                "response agent message",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"agent_message","content":[{"type":"text","text":"visible agent"}]}}"#,
            ),
            (
                "response reasoning",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"reasoning","summary":[{"type":"summary_text","text":"visible reasoning"}]}}"#,
            ),
            (
                "response function call",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"function_call","name":"exec_command","arguments":"{}","call_id":"call-1"}}"#,
            ),
            (
                "response custom tool call",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"custom_tool_call","name":"apply_patch","input":"patch","call_id":"call-2"}}"#,
            ),
            (
                "response function output",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"function_call_output","call_id":"call-1","output":"result"}}"#,
            ),
            (
                "response custom tool output",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"custom_tool_call_output","call_id":"call-2","output":"result"}}"#,
            ),
            (
                "event user message",
                r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"user_message","message":"visible event user"}}"#,
            ),
            (
                "event agent reasoning",
                r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"agent_reasoning","text":"visible event reasoning"}}"#,
            ),
            (
                "event tool call",
                r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"must-not-appear-in-retained-branch-errors","payload":{"type":"tool_call","name":"event_tool","input":{"value":1},"call_id":"event-call"}}"#,
            ),
        ];

        let violations = cases
            .into_iter()
            .filter_map(|(case, content)| match scan_synthetic_jsonl(content) {
                Err(error)
                    if error.to_string().contains("encrypted_content")
                        && !error.to_string().contains(collision_value) =>
                {
                    None
                }
                Err(error) => Some(format!("{case}: wrong error: {error:#}")),
                Ok(_) => Some(format!("{case}: unexpectedly accepted")),
            })
            .collect::<Vec<_>>();
        assert!(violations.is_empty(), "{violations:#?}");
    }

    #[test]
    fn scan_codex_dropped_agent_messages_without_semantic_opaque_ignore_top_level_name() {
        let content = concat!(
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"not-normalized","payload":{"type":"agent_message","content":[{"type":"text","text":"  \n\t  "}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:01Z","encrypted_content":"not-normalized","payload":{"type":"agent_message","content":[{"type":"input_image","image_url":"synthetic://image"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:02Z","encrypted_content":"not-normalized","payload":{"type":"agent_message","content":[{"type":"refusal","refusal":"synthetic refusal"}]}}"#,
            "\n",
        );

        assert!(scan_synthetic_jsonl(content).unwrap().is_empty());
    }

    #[test]
    fn scan_codex_review_present_non_string_response_item_type_fails_closed() {
        let violations = [("null", "null"), ("number", "7")]
            .into_iter()
            .filter_map(|(case, payload_type)| {
                let content = format!(
                    r#"{{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{{"type":{payload_type},"role":"user","content":"must not enter legacy fallback"}}}}"#
                );
                match scan_synthetic_jsonl(&content) {
                    Err(error) if error.to_string().contains("payload.type") => None,
                    Err(error) => Some(format!("{case}: wrong error: {error:#}")),
                    Ok(_) => Some(format!("{case}: unexpectedly accepted")),
                }
            })
            .collect::<Vec<_>>();
        assert!(violations.is_empty(), "{violations:#?}");
    }

    #[test]
    fn scan_codex_review_reasoning_without_summary_or_encrypted_content_does_not_emit() {
        let content = concat!(
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"reasoning"}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:01Z","payload":{"type":"reasoning","summary":[]}}"#,
            "\n",
        );
        assert!(scan_synthetic_jsonl(content).unwrap().is_empty());
    }

    #[test]
    fn scan_codex_review_reasoning_summary_shape_fails_closed() {
        let cases = [
            (
                "wrong container",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"reasoning","summary":{}}}"#,
            ),
            (
                "non-object item",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"reasoning","summary":[7]}}"#,
            ),
            (
                "missing item type",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"reasoning","summary":[{"text":"hidden"}]}}"#,
            ),
            (
                "non-string item type",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"reasoning","summary":[{"type":null,"text":"hidden"}]}}"#,
            ),
            (
                "unknown item type",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"reasoning","summary":[{"type":"unknown","text":"hidden"}]}}"#,
            ),
            (
                "non-string summary text",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"reasoning","summary":[{"type":"summary_text","text":7}]}}"#,
            ),
        ];
        let violations = cases
            .into_iter()
            .filter_map(|(case, content)| match scan_synthetic_jsonl(content) {
                Err(error) if error.to_string().contains("reasoning summary") => None,
                Err(error) => Some(format!("{case}: wrong error: {error:#}")),
                Ok(_) => Some(format!("{case}: unexpectedly accepted")),
            })
            .collect::<Vec<_>>();
        assert!(violations.is_empty(), "{violations:#?}");
    }

    #[test]
    fn scan_codex_review_encrypted_only_reasoning_is_retained() {
        let content = r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"reasoning","encrypted_content":"opaque"}}"#;
        let convs = scan_synthetic_jsonl(content).unwrap();
        let message = &convs[0].messages[0];
        assert_eq!(message.role, "reasoning");
        assert!(message.content.is_empty());
        assert_eq!(message.extra["encrypted_content"], "opaque");
    }

    #[test]
    fn scan_codex_review_agent_message_block_shape_fails_closed() {
        let cases = [
            (
                "non-object block",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"agent_message","content":[7]}}"#,
            ),
            (
                "missing block type",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"agent_message","content":[{}]}}"#,
            ),
            (
                "non-string block type",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"agent_message","content":[{"type":null}]}}"#,
            ),
            (
                "unknown block type",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"agent_message","content":[{"type":"unknown"}]}}"#,
            ),
            (
                "non-string visible text",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"agent_message","content":[{"type":"text","text":7}]}}"#,
            ),
            (
                "visible plus malformed block",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"agent_message","content":[{"type":"text","text":"visible"},{"type":"unknown"}]}}"#,
            ),
        ];
        let violations = cases
            .into_iter()
            .filter_map(|(case, content)| match scan_synthetic_jsonl(content) {
                Err(error) if error.to_string().contains("agent_message") => None,
                Err(error) => Some(format!("{case}: wrong error: {error:#}")),
                Ok(_) => Some(format!("{case}: unexpectedly accepted")),
            })
            .collect::<Vec<_>>();
        assert!(violations.is_empty(), "{violations:#?}");
    }

    #[test]
    fn scan_codex_agent_message_encrypted_content_fails_loud() {
        let cases = [
            (
                "duplicate",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"agent_message","content":[{"type":"text","text":"visible"},{"type":"encrypted_content","encrypted_content":"one"},{"type":"encrypted_content","encrypted_content":"two"}]}}"#,
            ),
            (
                "wrong type",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"agent_message","content":[{"type":"text","text":"visible"},{"type":"encrypted_content","encrypted_content":7}]}}"#,
            ),
            (
                "raw collision",
                r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","encrypted_content":"collision","payload":{"type":"agent_message","content":[{"type":"text","text":"visible"},{"type":"encrypted_content","encrypted_content":"opaque"}]}}"#,
            ),
        ];

        for (case, content) in cases {
            let error = scan_synthetic_jsonl(content).unwrap_err();
            assert!(
                error.to_string().contains("encrypted_content"),
                "{case}: {error:#}"
            );
        }
    }

    #[test]
    fn scan_codex_reasoning_encrypted_content_requires_string() {
        let content = r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"reasoning","summary":[],"encrypted_content":{"opaque":"wrong"}}}"#;
        let error = scan_synthetic_jsonl(content).unwrap_err();
        assert!(error.to_string().contains("encrypted_content"));
    }

    #[test]
    fn scan_codex_event_plaintext_whitespace_is_dropped() {
        let content = concat!(
            r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"user_message","message":"  \n\t  "}}"#,
            "\n",
            r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:01Z","payload":{"type":"agent_reasoning","text":" \t "}}"#,
            "\n",
        );
        assert!(scan_synthetic_jsonl(content).unwrap().is_empty());
    }

    #[test]
    fn scan_codex_canonical_token_count_attaches_last_usage() {
        let content = concat!(
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}}"#,
            "\n",
            r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:01Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":7,"cached_input_tokens":1,"cache_write_input_tokens":3,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21},"model_context_window":128000,"total_token_usage":{"input_tokens":700,"cached_input_tokens":100,"output_tokens":1100,"reasoning_output_tokens":200,"total_tokens":2100}},"rate_limits":null}}"#,
            "\n",
        );

        let convs = scan_synthetic_jsonl(content).unwrap();
        let assistant = &convs[0].messages[0];
        assert_eq!(assistant.extra["cass"]["token_usage"]["input_tokens"], 7);
        assert_eq!(assistant.extra["cass"]["token_usage"]["output_tokens"], 11);
        assert_eq!(
            assistant.extra["cass"]["token_usage"]["cached_input_tokens"],
            1
        );
        assert_eq!(
            assistant.extra["cass"]["token_usage"]["cache_write_input_tokens"],
            3
        );
        assert_eq!(assistant.extra["cass"]["token_usage"]["total_tokens"], 21);
        assert_eq!(assistant.extra["cass"]["token_usage"]["data_source"], "api");
    }

    #[test]
    fn scan_codex_legacy_token_count_without_rate_limits_key_attaches_usage() {
        // Shape anchored to a real-world corpus sample, not invented from
        // reading this implementation (fad-fork EXEC discipline): a python
        // scan of 83 real codex rollout files found 7 files / 101
        // token_count events (all dated 2025-09-17, an early codex CLI
        // build) whose `payload` is exactly `{type, info}` -- `rate_limits`
        // is not present as a key at all, not even `null`. Values below are
        // fully synthetic; only the key-set shape is real.
        let content = concat!(
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}}"#,
            "\n",
            r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:01Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":7,"cached_input_tokens":1,"cache_write_input_tokens":3,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21},"model_context_window":128000,"total_token_usage":{"input_tokens":700,"cached_input_tokens":100,"output_tokens":1100,"reasoning_output_tokens":200,"total_tokens":2100}}}}"#,
            "\n",
        );

        let convs = scan_synthetic_jsonl(content).unwrap();
        let assistant = &convs[0].messages[0];
        assert_eq!(assistant.extra["cass"]["token_usage"]["input_tokens"], 7);
        assert_eq!(assistant.extra["cass"]["token_usage"]["output_tokens"], 11);
        assert_eq!(assistant.extra["cass"]["token_usage"]["total_tokens"], 21);
        assert_eq!(assistant.extra["cass"]["token_usage"]["data_source"], "api");
    }

    #[test]
    fn scan_codex_rate_limit_only_token_count_does_not_invent_usage() {
        let content = concat!(
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:00Z","payload":{"type":"message","role":"assistant","content":"answer"}}"#,
            "\n",
            r#"{"type":"event_msg","timestamp":"2026-07-23T00:00:01Z","payload":{"type":"token_count","info":null,"rate_limits":{"primary":{"used_percent":12.5}}}}"#,
            "\n",
        );
        let convs = scan_synthetic_jsonl(content).unwrap();
        assert!(
            convs[0].messages[0]
                .extra
                .pointer("/cass/token_usage")
                .is_none()
        );
    }

    #[test]
    fn scan_codex_canonical_token_count_rejects_malformed_shapes() {
        let valid_usage = r#"{"input_tokens":7,"cached_input_tokens":1,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21}"#;
        let cases = [
            (
                "neither canonical nor legacy",
                r#"{"type":"token_count"}"#.to_string(),
            ),
            (
                "payload extra key",
                format!(r#"{{"type":"token_count","info":{{"last_token_usage":{valid_usage},"model_context_window":128000,"total_token_usage":{valid_usage}}},"rate_limits":null,"extra":true}}"#),
            ),
            (
                "missing info field",
                format!(r#"{{"type":"token_count","info":{{"last_token_usage":{valid_usage},"model_context_window":128000}},"rate_limits":null}}"#),
            ),
            (
                "usage key drift",
                r#"{"type":"token_count","info":{"last_token_usage":{"input_tokens":7,"cached_input_tokens":1,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21,"unknown_tokens":1},"model_context_window":128000,"total_token_usage":{"input_tokens":7,"cached_input_tokens":1,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21}},"rate_limits":null}"#.to_string(),
            ),
            (
                "negative usage",
                r#"{"type":"token_count","info":{"last_token_usage":{"input_tokens":-1,"cached_input_tokens":1,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21},"model_context_window":128000,"total_token_usage":{"input_tokens":7,"cached_input_tokens":1,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21}},"rate_limits":null}"#.to_string(),
            ),
            (
                "boolean usage",
                r#"{"type":"token_count","info":{"last_token_usage":{"input_tokens":true,"cached_input_tokens":1,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21},"model_context_window":128000,"total_token_usage":{"input_tokens":7,"cached_input_tokens":1,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21}},"rate_limits":null}"#.to_string(),
            ),
            (
                "bad context window",
                format!(r#"{{"type":"token_count","info":{{"last_token_usage":{valid_usage},"model_context_window":-0.5,"total_token_usage":{valid_usage}}},"rate_limits":null}}"#),
            ),
            (
                "bad rate limits",
                format!(r#"{{"type":"token_count","info":{{"last_token_usage":{valid_usage},"model_context_window":128000,"total_token_usage":{valid_usage}}},"rate_limits":[]}}"#),
            ),
            (
                "rate only missing object",
                r#"{"type":"token_count","info":null,"rate_limits":null}"#.to_string(),
            ),
        ];

        for (case, payload) in cases {
            let content = format!(
                "{{\"type\":\"event_msg\",\"timestamp\":\"2026-07-23T00:00:00Z\",\"payload\":{payload}}}\n"
            );
            let error = scan_synthetic_jsonl(&content).unwrap_err();
            assert!(
                error.to_string().contains("token_count"),
                "{case}: {error:#}"
            );
        }
    }

    #[test]
    fn scan_codex_metadata_side_effects_require_current_shapes() {
        let content = concat!(
            r#"{"type":"session_meta","timestamp":"2026-07-23T00:00:00Z","payload":{"cwd":"/tmp/metadata-contract"}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:01Z","payload":{"type":"message","role":"assistant","content":"answer"}}"#,
            "\n",
            r#"{"type":"turn_context","timestamp":"2026-07-23T00:00:02Z","payload":{"model":"gpt-metadata"}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-07-23T00:00:03Z","payload":{"type":"message","role":"assistant","content":"authored answer"}}"#,
            "\n",
        );

        let convs = scan_synthetic_jsonl(content).unwrap();
        let conv = &convs[0];
        assert_eq!(
            conv.workspace,
            Some(PathBuf::from("/tmp/metadata-contract"))
        );
        assert_eq!(conv.messages[0].author, None);
        assert_eq!(conv.messages[1].author.as_deref(), Some("gpt-metadata"));
        assert!(conv.started_at < conv.messages[0].created_at);
        assert!(conv.ended_at > conv.messages[0].created_at);
    }

    #[test]
    fn scan_codex_metadata_records_fail_closed_before_side_effects() {
        let cases = [
            (
                "session missing timestamp",
                r#"{"type":"session_meta","payload":{"cwd":"/tmp/project"}}"#,
            ),
            (
                "session null timestamp",
                r#"{"type":"session_meta","timestamp":null,"payload":{"cwd":"/tmp/project"}}"#,
            ),
            (
                "session numeric timestamp",
                r#"{"type":"session_meta","timestamp":7,"payload":{"cwd":"/tmp/project"}}"#,
            ),
            (
                "session missing payload",
                r#"{"type":"session_meta","timestamp":"2026-07-23T00:00:00Z"}"#,
            ),
            (
                "session null payload",
                r#"{"type":"session_meta","timestamp":"2026-07-23T00:00:00Z","payload":null}"#,
            ),
            (
                "session string payload",
                r#"{"type":"session_meta","timestamp":"2026-07-23T00:00:00Z","payload":"bad"}"#,
            ),
            (
                "session array payload",
                r#"{"type":"session_meta","timestamp":"2026-07-23T00:00:00Z","payload":[]}"#,
            ),
            (
                "session missing cwd",
                r#"{"type":"session_meta","timestamp":"2026-07-23T00:00:00Z","payload":{}}"#,
            ),
            (
                "session wrong cwd",
                r#"{"type":"session_meta","timestamp":"2026-07-23T00:00:00Z","payload":{"cwd":7}}"#,
            ),
            (
                "turn missing timestamp",
                r#"{"type":"turn_context","payload":{"model":"gpt-synthetic"}}"#,
            ),
            (
                "turn null timestamp",
                r#"{"type":"turn_context","timestamp":null,"payload":{"model":"gpt-synthetic"}}"#,
            ),
            (
                "turn numeric timestamp",
                r#"{"type":"turn_context","timestamp":7,"payload":{"model":"gpt-synthetic"}}"#,
            ),
            (
                "turn missing payload",
                r#"{"type":"turn_context","timestamp":"2026-07-23T00:00:00Z"}"#,
            ),
            (
                "turn null payload",
                r#"{"type":"turn_context","timestamp":"2026-07-23T00:00:00Z","payload":null}"#,
            ),
            (
                "turn string payload",
                r#"{"type":"turn_context","timestamp":"2026-07-23T00:00:00Z","payload":"bad"}"#,
            ),
            (
                "turn array payload",
                r#"{"type":"turn_context","timestamp":"2026-07-23T00:00:00Z","payload":[]}"#,
            ),
            (
                "turn missing model",
                r#"{"type":"turn_context","timestamp":"2026-07-23T00:00:00Z","payload":{}}"#,
            ),
            (
                "turn wrong model",
                r#"{"type":"turn_context","timestamp":"2026-07-23T00:00:00Z","payload":{"model":7}}"#,
            ),
        ];

        for (case, content) in cases {
            let error = scan_synthetic_jsonl(content).unwrap_err();
            assert!(
                error.to_string().contains("session_meta")
                    || error.to_string().contains("turn_context"),
                "{case}: {error:#}"
            );
        }
    }

    #[test]
    fn scan_codex_compact_and_noncompact_preserve_same_contract_fields() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let records = vec![
            json!({
                "type": "response_item",
                "timestamp": "2026-07-23T00:00:00Z",
                "payload": {
                    "type": "agent_message",
                    "content": [
                        {"type": "text", "text": "visible agent"},
                        {"type": "encrypted_content", "encrypted_content": "opaque-agent"}
                    ]
                }
            }),
            json!({
                "type": "response_item",
                "timestamp": "2026-07-23T00:00:01Z",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"true\"}",
                    "call_id": "compact-call"
                }
            }),
            json!({
                "type": "response_item",
                "timestamp": "2026-07-23T00:00:02Z",
                "payload": {
                    "type": "custom_tool_call_output",
                    "output": "unpaired compact result"
                }
            }),
            json!({
                "type": "response_item",
                "timestamp": "2026-07-23T00:00:03Z",
                "payload": {
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "opaque-reasoning"
                }
            }),
        ];
        let small = records
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(sessions.join("rollout-small.jsonl"), small).unwrap();

        let mut large_records = records.clone();
        large_records[0].as_object_mut().unwrap().insert(
            "padding".to_string(),
            Value::String("x".repeat(32 * 1024 * 1024)),
        );
        let large = large_records
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(sessions.join("rollout-large.jsonl"), large).unwrap();

        let convs = CodexConnector::new()
            .scan(&ScanContext::local_default(codex_dir, None))
            .unwrap();
        let small = convs
            .iter()
            .find(|conv| conv.external_id.as_deref() == Some("rollout-small"))
            .unwrap();
        let large = convs
            .iter()
            .find(|conv| conv.external_id.as_deref() == Some("rollout-large"))
            .unwrap();

        let contract_snapshot = |conv: &NormalizedConversation| {
            conv.messages
                .iter()
                .map(|message| {
                    json!({
                        "idx": message.idx,
                        "role": message.role,
                        "content": message.content,
                        "raw_role": message.extra.get("raw_role"),
                        "tool_call_id": message.extra.get("tool_call_id"),
                        "tool_call_args": message.extra.get("tool_call_args"),
                        "unpaired": message.extra.get("unpaired"),
                        "encrypted_content": message.extra.get("encrypted_content"),
                    })
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(contract_snapshot(small), contract_snapshot(large));
        assert!(
            large
                .messages
                .iter()
                .all(|message| message.extra.get("payload").is_none())
        );
    }

    #[test]
    fn scan_codex_compact_path_rejects_raw_envelope_raw_role_collision() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let record = json!({
            "type": "response_item",
            "timestamp": "2026-07-23T00:00:00Z",
            "raw_role": "collision",
            "padding": "x".repeat(32 * 1024 * 1024),
            "payload": {
                "type": "message",
                "role": "user",
                "content": "visible"
            }
        });
        fs::write(
            sessions.join("rollout-compact-collision.jsonl"),
            record.to_string() + "\n",
        )
        .unwrap();

        let error = CodexConnector::new()
            .scan(&ScanContext::local_default(codex_dir, None))
            .unwrap_err();
        assert!(error.to_string().contains("raw_role"));
    }

    #[test]
    fn scan_codex_compact_visible_agent_rejects_top_level_encrypted_content_collision() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let collision_value = "must-not-appear-in-errors-compact-agent";
        let record = json!({
            "type": "response_item",
            "timestamp": "2026-07-23T00:00:00Z",
            "encrypted_content": collision_value,
            "padding": "x".repeat(32 * 1024 * 1024),
            "payload": {
                "type": "agent_message",
                "content": [{"type": "text", "text": "visible agent"}]
            }
        });
        fs::write(
            sessions.join("rollout-compact-encrypted-collision.jsonl"),
            record.to_string() + "\n",
        )
        .unwrap();

        let error = CodexConnector::new()
            .scan(&ScanContext::local_default(codex_dir, None))
            .unwrap_err();
        assert!(error.to_string().contains("encrypted_content"), "{error:#}");
        assert!(!error.to_string().contains(collision_value), "{error:#}");
    }

    #[test]
    fn scan_codex_compact_plain_message_rejects_top_level_encrypted_content_collision() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let collision_value = "must-not-appear-in-errors-compact-plain-message";
        let record = json!({
            "type": "response_item",
            "timestamp": "2026-07-23T00:00:00Z",
            "encrypted_content": collision_value,
            "padding": "x".repeat(32 * 1024 * 1024),
            "payload": {
                "type": "message",
                "role": "user",
                "content": "visible user"
            }
        });
        fs::write(
            sessions.join("rollout-compact-plain-encrypted-collision.jsonl"),
            record.to_string() + "\n",
        )
        .unwrap();

        let error = CodexConnector::new()
            .scan(&ScanContext::local_default(codex_dir, None))
            .unwrap_err();
        assert!(error.to_string().contains("encrypted_content"), "{error:#}");
        assert!(!error.to_string().contains(collision_value), "{error:#}");
    }

    #[test]
    fn scan_codex_legacy_json_matches_reviewed_literal_snapshot() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let legacy = json!({
            "session": {"cwd": "/tmp/legacy-contract"},
            "items": [
                {"content": "legacy default"},
                {"role": "assistant", "content": "legacy assistant"}
            ]
        });
        fs::write(
            sessions.join("rollout-legacy-snapshot.json"),
            legacy.to_string(),
        )
        .unwrap();

        let convs = CodexConnector::new()
            .scan(&ScanContext::local_default(codex_dir, None))
            .unwrap();
        assert_eq!(
            serde_json::to_value(&convs[0].messages).unwrap(),
            json!([
                {
                    "idx": 0,
                    "role": "agent",
                    "author": null,
                    "created_at": null,
                    "content": "legacy default",
                    "extra": {"content": "legacy default"},
                    "snippets": []
                },
                {
                    "idx": 1,
                    "role": "assistant",
                    "author": null,
                    "created_at": null,
                    "content": "legacy assistant",
                    "extra": {"role": "assistant", "content": "legacy assistant"},
                    "snippets": []
                }
            ])
        );
    }

    #[test]
    fn scan_stores_source_path() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = r#"{"type":"response_item","payload":{"role":"user","content":"Test"}}
"#;
        let file_path = sessions.join("rollout-path.jsonl");
        fs::write(&file_path, content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].source_path, file_path);
    }

    // =====================================================
    // Edge case tests — malformed input robustness (br-fiiv)
    // =====================================================

    #[test]
    fn truncated_jsonl_mid_json_returns_partial_results() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // First line valid, second truncated mid-JSON
        let content = b"{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"Valid\"}}\n{\"type\":\"response_item\",\"payload\":{\"role\":\"assistant\",\"con";
        fs::write(sessions.join("rollout-truncated.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "truncated file should not cause an error");
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].messages.len(),
            1,
            "should yield only the 1 valid message from truncated file"
        );
        assert_eq!(convs[0].messages[0].content, "Valid");
    }

    #[test]
    fn truncated_mid_utf8_does_not_panic() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            b"{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"OK\"}}\n",
        );
        // Incomplete 4-byte UTF-8 sequence (U+1F600 = F0 9F 98 80, only 2 bytes)
        bytes.extend_from_slice(b"\xF0\x9F");

        fs::write(sessions.join("rollout-utf8trunc.jsonl"), &bytes).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "truncated mid-UTF8 should not panic");
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages[0].content, "OK");
    }

    #[test]
    fn invalid_utf8_skips_corrupted_lines() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            b"{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"Before\"}}\n",
        );
        bytes.extend_from_slice(b"\xFF\xFE invalid utf8 line\n");
        bytes.extend_from_slice(
            b"{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"After\"}}\n",
        );

        fs::write(sessions.join("rollout-badbytes.jsonl"), &bytes).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "invalid UTF-8 should not cause a panic");
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].messages.len(),
            2,
            "should extract valid messages around invalid UTF-8"
        );
        assert_eq!(convs[0].messages[0].content, "Before");
        assert_eq!(convs[0].messages[1].content, "After");
    }

    #[test]
    fn empty_file_returns_no_conversations() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        fs::write(sessions.join("rollout-empty.jsonl"), b"").unwrap();
        fs::write(sessions.join("rollout-empty.json"), b"").unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "empty files should not cause errors");
        let convs = result.unwrap();
        assert!(
            convs.is_empty(),
            "empty files should produce no conversations"
        );
    }

    #[test]
    fn whitespace_only_file_returns_no_conversations() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        fs::write(sessions.join("rollout-ws.jsonl"), "  \n\n  \t\n").unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(
            result.is_ok(),
            "whitespace-only file should not cause errors"
        );
        let convs = result.unwrap();
        assert!(
            convs.is_empty(),
            "whitespace-only file should produce no conversations"
        );
    }

    #[test]
    fn json_type_mismatch_skips_gracefully() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = concat!(
            // payload is a string instead of object
            "{\"type\":\"response_item\",\"payload\":\"not an object\"}\n",
            // type is a number
            "{\"type\":123,\"payload\":{\"role\":\"user\",\"content\":\"num type\"}}\n",
            // content is a number
            "{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":99}}\n",
            // Correct entry
            "{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"Correct\"}}\n",
        );
        fs::write(sessions.join("rollout-types.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "type mismatches should not cause errors");
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert!(
            convs[0].messages.iter().any(|m| m.content == "Correct"),
            "should extract the correctly typed entry"
        );
    }

    #[test]
    fn deeply_nested_json_does_not_stack_overflow() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // serde_json has a recursion limit of 128; 200 levels will trigger parse error
        let mut nested = String::new();
        for _ in 0..200 {
            nested.push_str("{\"a\":");
        }
        nested.push('1');
        for _ in 0..200 {
            nested.push('}');
        }

        let content = format!(
            "{}\n{}\n",
            nested,
            r#"{"type":"response_item","payload":{"role":"user","content":"After nesting"}}"#
        );
        fs::write(sessions.join("rollout-deep.jsonl"), &content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(
            result.is_ok(),
            "deeply nested JSON should not cause stack overflow"
        );
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages[0].content, "After nesting");
    }

    #[test]
    fn large_message_body_handled_without_oom() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let large_content = "x".repeat(1_000_000);
        let line = format!(
            r#"{{"type":"response_item","payload":{{"role":"user","content":"{}"}}}}"#,
            large_content
        );
        fs::write(sessions.join("rollout-large.jsonl"), &line).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "large message body should not cause OOM");
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages[0].content.len(), 1_000_000);
    }

    #[test]
    fn null_bytes_embedded_in_content_handled() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = concat!(
            r#"{"type":"response_item","payload":{"role":"user","content":"before\u0000after"}}"#,
            "\n",
            r#"{"type":"response_item","payload":{"role":"user","content":"Clean"}}"#,
            "\n"
        );
        fs::write(sessions.join("rollout-null.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(
            result.is_ok(),
            "null bytes in content should not cause errors"
        );
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert!(!convs[0].messages.is_empty());
    }

    #[test]
    fn bom_marker_at_file_start_handled() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\xEF\xBB\xBF"); // UTF-8 BOM
        bytes.extend_from_slice(
            b"{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"BOM line\"}}\n",
        );
        bytes.extend_from_slice(
            b"{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"Second\"}}\n",
        );
        fs::write(sessions.join("rollout-bom.jsonl"), &bytes).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "BOM marker should not cause errors");
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert!(
            !convs[0].messages.is_empty(),
            "should extract at least the second line after BOM"
        );
        assert!(
            convs[0].messages.iter().any(|m| m.content == "Second"),
            "second line should parse correctly regardless of BOM"
        );
    }

    // =====================================================
    // Codex-specific edge cases (br-fiiv)
    // =====================================================

    #[test]
    fn missing_payload_field_skipped() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // response_item and event_msg without payload stay ignorable. Strict
        // metadata records are covered separately and fail closed.
        let content = concat!(
            "{\"type\":\"response_item\",\"timestamp\":\"2025-12-01T10:00:00Z\"}\n",
            "{\"type\":\"event_msg\",\"timestamp\":\"2025-12-01T10:00:01Z\"}\n",
            "{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"Has payload\"}}\n",
        );
        fs::write(sessions.join("rollout-nopayload.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "missing payload should not cause errors");
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].content, "Has payload");
    }

    #[test]
    fn timestamp_parsing_edge_cases() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = concat!(
            // ISO 8601 with milliseconds
            "{\"type\":\"response_item\",\"timestamp\":\"2025-12-01T10:00:00.123Z\",\"payload\":{\"role\":\"user\",\"content\":\"ms precision\"}}\n",
            // ISO 8601 with timezone offset
            "{\"type\":\"response_item\",\"timestamp\":\"2025-12-01T10:00:00+05:30\",\"payload\":{\"role\":\"user\",\"content\":\"tz offset\"}}\n",
            // Unix epoch milliseconds as number
            "{\"type\":\"response_item\",\"timestamp\":1700000000000,\"payload\":{\"role\":\"user\",\"content\":\"epoch millis\"}}\n",
            // No timestamp at all
            "{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"no timestamp\"}}\n",
            // Null timestamp
            "{\"type\":\"response_item\",\"timestamp\":null,\"payload\":{\"role\":\"user\",\"content\":\"null ts\"}}\n",
        );
        fs::write(sessions.join("rollout-timestamps.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(
            result.is_ok(),
            "varied timestamp formats should not cause errors"
        );
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].messages.len(),
            5,
            "all 5 messages should be extracted regardless of timestamp format"
        );
        // Messages with valid timestamps should have created_at set
        assert!(convs[0].messages[0].created_at.is_some());
        assert!(convs[0].messages[1].created_at.is_some());
        assert!(convs[0].messages[2].created_at.is_some());
    }

    #[test]
    fn workspace_path_encoding_edge_cases() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // Test various workspace path formats in session_meta
        let content = concat!(
            // Path with spaces
            "{\"type\":\"session_meta\",\"timestamp\":\"2025-12-01T10:00:00Z\",\"payload\":{\"cwd\":\"/home/user/my project/src\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"Spaces path\"}}\n",
        );
        fs::write(sessions.join("rollout-spaces.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].workspace,
            Some(PathBuf::from("/home/user/my project/src"))
        );

        // Unicode workspace path
        let content2 = concat!(
            "{\"type\":\"session_meta\",\"timestamp\":\"2025-12-01T10:00:00Z\",\"payload\":{\"cwd\":\"/home/\u{00FC}ser/projekt\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"Unicode path\"}}\n",
        );
        fs::write(sessions.join("rollout-unicode.jsonl"), content2).unwrap();

        let convs2 = connector.scan(&ctx).unwrap();
        assert!(
            !convs2.is_empty(),
            "unicode workspace paths should be handled"
        );
    }

    #[test]
    fn event_msg_with_unknown_subtypes_skipped() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // Various event_msg subtypes that should be gracefully skipped
        let content = concat!(
            "{\"type\":\"event_msg\",\"payload\":{\"type\":\"streaming_start\"}}\n",
            "{\"type\":\"event_msg\",\"payload\":{\"type\":\"streaming_delta\",\"delta\":\"partial\"}}\n",
            "{\"type\":\"event_msg\",\"payload\":{\"type\":\"streaming_end\"}}\n",
            "{\"type\":\"event_msg\",\"payload\":{\"type\":\"tool_call\",\"name\":\"bash\",\"input\":{\"cmd\":\"ls\"}}}\n",
            "{\"type\":\"event_msg\",\"payload\":{\"type\":\"tool_result\",\"output\":\"file.txt\"}}\n",
            "{\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"message\":\"Real user input\"}}\n",
        );
        fs::write(sessions.join("rollout-events.jsonl"), content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(
            result.is_ok(),
            "unknown event subtypes should not cause errors"
        );
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        // user_message + tool_call events should produce messages
        assert_eq!(convs[0].messages.len(), 2);

        // tool_call event should produce its own `tool_call` message (not
        // inlined into `assistant`), with invocation args intact.
        let tool_msg = &convs[0].messages[0];
        assert_eq!(tool_msg.role, "tool_call");
        assert_eq!(tool_msg.invocations.len(), 1);
        assert_eq!(tool_msg.invocations[0].kind, "tool");
        assert_eq!(tool_msg.invocations[0].name, "bash");
        assert!(tool_msg.invocations[0].arguments.is_some());
        assert!(
            tool_msg.content.contains("bash"),
            "content should render the tool name + args, not a bare marker: {}",
            tool_msg.content
        );

        // user_message event should still produce a user message
        let user_msg = &convs[0].messages[1];
        assert_eq!(user_msg.content, "Real user input");
        assert_eq!(user_msg.role, "user");
    }

    #[test]
    fn tool_call_format_variations() {
        let dir = TempDir::new().unwrap();
        let codex_dir = dir.path().join(".codex");
        let sessions = codex_dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();

        // response_item with tool_use content blocks (like Claude API format)
        let content = json!({
            "type": "response_item",
            "payload": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check that."},
                    {"type": "tool_use", "name": "read_file", "input": {"path": "/etc/hosts"}}
                ]
            }
        })
        .to_string()
            + "\n"
            + &json!({
                "type": "response_item",
                "payload": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "bash", "input": {"command": "ls -la"}},
                        {"type": "text", "text": "Here are the results."}
                    ]
                }
            })
            .to_string()
            + "\n";

        fs::write(sessions.join("rollout-tools.jsonl"), &content).unwrap();

        let connector = CodexConnector::new();
        let ctx = ScanContext::local_default(codex_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(
            result.is_ok(),
            "tool call format variations should not cause errors"
        );
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 2);
        // flatten_content should handle tool_use blocks
        assert!(convs[0].messages[0].content.contains("Let me check"));
        assert!(
            convs[0].messages[1]
                .content
                .contains("Here are the results")
        );
    }
}
