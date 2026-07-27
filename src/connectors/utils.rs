//! Shared utility functions used by all connectors.

use std::path::{Path, PathBuf};

use serde_json::Value;

const RAW_ROLE_KEY: &str = "raw_role";
const TOOL_CALL_ID_KEY: &str = "tool_call_id";
const UNPAIRED_KEY: &str = "unpaired";
const ENCRYPTED_CONTENT_KEY: &str = "encrypted_content";

fn object_for_read<'a>(
    value: &'a Value,
    field_name: &str,
) -> anyhow::Result<&'a serde_json::Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{field_name} must be an object"))
}

fn object_for_write<'a>(
    value: &'a mut Value,
    field_name: &str,
) -> anyhow::Result<&'a mut serde_json::Map<String, Value>> {
    value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("{field_name} must be an object"))
}

/// Add the source role to a normalized message's extra fields without losing
/// collision evidence from the original envelope.
pub(crate) fn add_raw_role(
    raw_envelope: &Value,
    projected_extra: Value,
    raw_role: &str,
) -> anyhow::Result<Value> {
    if raw_role.trim().is_empty() {
        anyhow::bail!("raw_role must be non-empty");
    }

    let raw_object = object_for_read(raw_envelope, "raw envelope")?;
    if raw_object.contains_key(RAW_ROLE_KEY) {
        anyhow::bail!("raw envelope already contains reserved key {RAW_ROLE_KEY}");
    }

    let mut projected_extra = projected_extra;
    let extra_object = object_for_write(&mut projected_extra, "projected extra")?;
    if extra_object.contains_key(RAW_ROLE_KEY) {
        anyhow::bail!("projected extra already contains reserved key {RAW_ROLE_KEY}");
    }

    extra_object.insert(
        RAW_ROLE_KEY.to_string(),
        Value::String(raw_role.to_string()),
    );
    Ok(projected_extra)
}

/// Store exactly one tool-result pairing state in normalized extra fields.
pub(crate) fn set_tool_result_pairing(
    extra: &mut Value,
    tool_call_id: Option<&str>,
) -> anyhow::Result<()> {
    let extra_object = object_for_write(extra, "projected extra")?;

    if let Some(tool_call_id) = tool_call_id.map(str::trim).filter(|id| !id.is_empty()) {
        extra_object.insert(
            TOOL_CALL_ID_KEY.to_string(),
            Value::String(tool_call_id.to_string()),
        );
        extra_object.remove(UNPAIRED_KEY);
    } else {
        extra_object.remove(TOOL_CALL_ID_KEY);
        extra_object.insert(UNPAIRED_KEY.to_string(), Value::Bool(true));
    }

    Ok(())
}

/// Add opaque encrypted content to normalized extra fields after validating
/// both the original envelope and the projected object for reserved-key use.
pub(crate) fn add_encrypted_content(
    raw_envelope: &Value,
    projected_extra: &mut Value,
    encrypted_content: &Value,
) -> anyhow::Result<()> {
    let raw_object = object_for_read(raw_envelope, "raw envelope")?;
    if raw_object.contains_key(ENCRYPTED_CONTENT_KEY) {
        anyhow::bail!("raw envelope already contains reserved key {ENCRYPTED_CONTENT_KEY}");
    }

    let extra_object = object_for_write(projected_extra, "projected extra")?;
    if extra_object.contains_key(ENCRYPTED_CONTENT_KEY) {
        anyhow::bail!("projected extra already contains reserved key {ENCRYPTED_CONTENT_KEY}");
    }

    let encrypted_content = encrypted_content
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("encrypted_content must be a string"))?;
    extra_object.insert(
        ENCRYPTED_CONTENT_KEY.to_string(),
        Value::String(encrypted_content.to_string()),
    );

    Ok(())
}

/// Read an environment variable, trimming whitespace and treating empty strings as unset.
pub(crate) fn env_var_nonempty(key: &str) -> Option<String> {
    dotenvy::var(key).ok().and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

/// Read an environment variable as a filesystem path, ignoring empty strings.
pub(crate) fn env_path_nonempty(key: &str) -> Option<PathBuf> {
    env_var_nonempty(key).map(PathBuf::from)
}

/// Read `CASS_EXCLUDE_PATHS` as a comma/newline-delimited list of exact files
/// or directory prefixes to skip during connector scans.
///
/// This is intentionally implemented in the connector crate rather than in CASS
/// so source discovery and parsing stay aligned: a path excluded here is neither
/// pre-mirrored nor parsed.
pub(crate) fn excluded_scan_paths_from_env() -> Vec<PathBuf> {
    env_var_nonempty("CASS_EXCLUDE_PATHS")
        .into_iter()
        .flat_map(|value| {
            value
                .split([',', '\n'])
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(PathBuf::from)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Return true when `path` should be skipped because it equals or is under one
/// of the configured exclusions.
#[must_use]
pub(crate) fn path_is_excluded(path: &Path, excluded_paths: &[PathBuf]) -> bool {
    excluded_paths
        .iter()
        .any(|excluded| path == excluded || path.starts_with(excluded))
}

/// Build a deduplication key for hot scan loops without paying the full
/// `canonicalize()` syscall cost on every ordinary file.
///
/// Most indexed session files are not symlinks. A full canonicalization on each
/// one walks every path component and triggers a storm of `readlink` probes that
/// dominate no-op incremental scans. We only resolve leaf symlinks here; callers
/// that need stronger root-level normalization should canonicalize the much
/// smaller root set separately.
#[must_use]
pub(crate) fn dedupe_path_key(path: &std::path::Path) -> PathBuf {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
        }
        _ => path.to_path_buf(),
    }
}

/// Check if a file was modified since the given timestamp.
/// Returns true if the file should be processed (modified since timestamp or no timestamp given).
#[must_use]
pub fn file_modified_since(path: &std::path::Path, since_ts: Option<i64>) -> bool {
    since_ts.is_none_or(|ts| {
        let threshold = ts.saturating_sub(1_000);
        std::fs::metadata(path)
            .and_then(|m| m.modified())
            .map_or(true, |mt| {
                mt.duration_since(std::time::UNIX_EPOCH).map_or(true, |d| {
                    i64::try_from(d.as_millis()).unwrap_or(i64::MAX) >= threshold
                })
            })
    })
}

/// Parse a timestamp from either i64 milliseconds or ISO-8601 string.
/// Returns milliseconds since Unix epoch, or None if unparseable.
#[must_use]
pub fn parse_timestamp(val: &serde_json::Value) -> Option<i64> {
    if let Some(ts) = val.as_i64() {
        let ts = if (0..100_000_000_000).contains(&ts) {
            ts.saturating_mul(1000)
        } else {
            ts
        };
        return Some(ts);
    }
    // Handle JSON float numbers (e.g., 1700000000.5) — serde_json's as_i64()
    // returns None for numbers with fractional parts, so check as_f64() too.
    // Note: as_f64() also succeeds for integer Numbers, but those are already
    // handled by as_i64() above.
    if val.is_number() {
        if let Some(f) = val.as_f64() {
            if f.is_finite() && f > 0.0 {
                #[allow(clippy::cast_possible_truncation)]
                let ts = if f < 100_000_000_000.0 {
                    (f * 1000.0).round() as i64
                } else {
                    f.round() as i64
                };
                return Some(ts);
            }
        }
    }
    if let Some(s) = val.as_str() {
        if let Ok(num) = s.parse::<i64>() {
            let ts = if (0..100_000_000_000).contains(&num) {
                num.saturating_mul(1000)
            } else {
                num
            };
            return Some(ts);
        }
        if let Ok(num) = s.parse::<f64>() {
            if !num.is_finite() {
                return None;
            }
            #[allow(clippy::cast_possible_truncation)]
            let ts = if (0.0..100_000_000_000.0).contains(&num) {
                (num * 1000.0).round() as i64
            } else {
                num.round() as i64
            };
            return Some(ts);
        }
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
            return Some(dt.timestamp_millis());
        }
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ") {
            return Some(dt.and_utc().timestamp_millis());
        }
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ") {
            return Some(dt.and_utc().timestamp_millis());
        }
    }
    None
}

#[cfg(test)]
mod dedupe_tests {
    use super::dedupe_path_key;

    #[test]
    fn dedupe_path_key_keeps_regular_file_paths_stable() {
        let dir = tempfile::TempDir::new().unwrap();
        let file = dir.path().join("session.jsonl");
        std::fs::write(&file, "hello").unwrap();

        assert_eq!(dedupe_path_key(&file), file);
    }

    #[cfg(unix)]
    #[test]
    fn dedupe_path_key_canonicalizes_symlink_leaf() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("target.jsonl");
        let link = dir.path().join("link.jsonl");
        std::fs::write(&target, "hello").unwrap();
        symlink(&target, &link).unwrap();

        assert_eq!(
            dedupe_path_key(&link),
            std::fs::canonicalize(&target).unwrap()
        );
    }
}

/// Flatten content that may be a string or array of content blocks.
/// Extracts text from text blocks and tool names from `tool_use` blocks.
#[must_use]
pub fn flatten_content(val: &serde_json::Value) -> String {
    if let Some(s) = val.as_str() {
        return s.to_string();
    }

    if let Some(arr) = val.as_array() {
        let mut result = String::new();
        for item in arr {
            if let Some(text) = extract_content_part(item) {
                if text.is_empty() {
                    continue;
                }
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(&text);
            }
        }
        return result;
    }

    String::new()
}

/// A single content-array element, split by block type rather than merged
/// into one flattened string.
///
/// Unlike [`flatten_content`] (which collapses everything into a single
/// display string and drops `tool_result`/`thinking` blocks entirely), this
/// preserves each block's full structure and type so callers can emit typed
/// 6-role messages. `ToolCall.input` and `ToolResult.content` always carry
/// the complete, untruncated value — truncation is an adapter-feed concern,
/// never a canonical/franken one.
pub(crate) enum TypedBlock {
    /// `{"type":"text","text":...}` (also covers `input_text`/`output_text`).
    Text(String),
    /// `{"type":"tool_use",...}` — full args preserved in `input`.
    ToolCall {
        name: String,
        input: Option<serde_json::Value>,
        id: Option<String>,
    },
    /// `{"type":"tool_result",...}` — full content preserved.
    ToolResult {
        content: Option<serde_json::Value>,
        tool_use_id: Option<String>,
    },
    /// `{"type":"thinking","text":...}`.
    Thinking(String),
}

/// Split a content array (or plain string) into typed blocks by block type,
/// instead of flattening everything into one string.
///
/// This is the typed counterpart to [`flatten_content`]: it keeps
/// `tool_result` and `thinking` blocks (which `flatten_content`'s whitelist
/// drops) and preserves full `tool_use`/`tool_result` payloads unmodified.
/// Malformed blocks (missing required fields) are skipped rather than
/// causing a panic.
#[must_use]
pub(crate) fn split_content_blocks(v: &serde_json::Value) -> Vec<TypedBlock> {
    let Some(arr) = v.as_array() else {
        return Vec::new();
    };

    let mut blocks = Vec::new();
    for item in arr {
        let item_type = item.get("type").and_then(|t| t.as_str());
        match item_type {
            Some("text") | Some("input_text") | Some("output_text") => {
                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                    blocks.push(TypedBlock::Text(text.to_string()));
                }
            }
            Some("tool_use") => {
                let Some(name) = item.get("name").and_then(|n| n.as_str()) else {
                    continue;
                };
                let id = item
                    .get("id")
                    .and_then(|i| i.as_str())
                    .map(std::string::ToString::to_string);
                blocks.push(TypedBlock::ToolCall {
                    name: name.to_string(),
                    input: item.get("input").cloned(),
                    id,
                });
            }
            Some("tool_result") => {
                let tool_use_id = item
                    .get("tool_use_id")
                    .and_then(|i| i.as_str())
                    .map(std::string::ToString::to_string);
                blocks.push(TypedBlock::ToolResult {
                    content: item.get("content").cloned(),
                    tool_use_id,
                });
            }
            Some("thinking") => {
                // Real signed Anthropic thinking blocks carry the reasoning in
                // the `thinking` key (with a `signature`), matching
                // `pi_agent.rs`. OpenClaw does the same -- its key is
                // `thinking` with a `thinkingSignature` beside it. (This
                // comment used to claim openclaw's normalized shape uses
                // `text`; that was wrong, and `openclaw.rs` was written to
                // match the wrong claim, so every OpenClaw thinking block was
                // dropped. Measured on live sessions: `thinking` in 3713 of
                // 3713 blocks, `text` in 0.) Read `thinking` first, fall back
                // to `text`. An empty-string value still yields a block (real
                // signed blocks can have empty text) rather than being
                // silently dropped.
                // Resolve each key to a string before falling through, rather
                // than picking the key first and stringifying after:
                // `{"thinking":null,"text":"body"}` would otherwise emit
                // nothing, because `get("thinking")` yields `Some(Null)` and
                // `or_else` only fires on `None`. Same for a non-string value.
                if let Some(text) = item
                    .get("thinking")
                    .and_then(|t| t.as_str())
                    .or_else(|| item.get("text").and_then(|t| t.as_str()))
                {
                    blocks.push(TypedBlock::Thinking(text.to_string()));
                }
            }
            _ => {}
        }
    }
    blocks
}

/// Extract text content from a single content block item.
fn extract_content_part(item: &serde_json::Value) -> Option<String> {
    if let Some(text) = item.as_str() {
        return Some(text.to_string());
    }

    let item_type = item.get("type").and_then(|v| v.as_str());

    if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
        // `output_text` is the modern Codex/Responses-API assistant text block;
        // `input_text` is the user/developer counterpart. Both, like a plain
        // `text` block, carry rendered text we want to surface.
        if item_type.is_none()
            || item_type == Some("text")
            || item_type == Some("input_text")
            || item_type == Some("output_text")
        {
            return Some(text.to_string());
        }
    }

    if item_type == Some("tool_use") {
        let name = item
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let desc = item
            .get("input")
            .and_then(|i| i.get("description"))
            .and_then(|v| v.as_str())
            .or_else(|| {
                item.get("input")
                    .and_then(|i| i.get("file_path"))
                    .and_then(|v| v.as_str())
            })
            .unwrap_or("");
        if desc.is_empty() {
            return Some(format!("[Tool: {name}]"));
        }
        return Some(format!("[Tool: {name} - {desc}]"));
    }

    None
}

/// Extract structured invocations from a Claude API-style content block array.
///
/// Emits every `tool_use` block as `kind: "tool"`. Connector-specific
/// unwrapping (e.g. Amp's skill wrapper) should be applied separately via
/// [`unwrap_skill_invocations`].
///
/// Works for any connector that stores content as an array of typed blocks:
/// amp, `claude_code`, codex, cline, factory.
#[must_use]
pub fn extract_invocations_from_content_blocks(
    val: &serde_json::Value,
) -> Vec<crate::types::NormalizedInvocation> {
    let Some(arr) = val.as_array() else {
        return Vec::new();
    };

    let mut invocations = Vec::new();
    for item in arr {
        let item_type = item.get("type").and_then(|v| v.as_str());
        if item_type != Some("tool_use") {
            continue;
        }

        let Some(raw_name) = item.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let call_id = item
            .get("id")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string);
        let input = item.get("input");

        invocations.push(crate::types::NormalizedInvocation {
            kind: "tool".to_string(),
            name: raw_name.to_string(),
            raw_name: None,
            call_id,
            arguments: input.cloned(),
        });
    }

    invocations
}

/// Amp-specific wrapper tools that should be unwrapped to their inner name.
const AMP_SKILL_WRAPPERS: &[(&str, &str)] = &[
    // (tool_name, input_key_for_real_name)
    ("skill", "name"),
    ("load_skill", "name"),
];

/// Unwrap Amp skill-wrapper invocations in place.
///
/// Tools like `skill` and `load_skill` are Amp-specific wrappers whose real
/// name lives inside the `input` object. This rewrites matching invocations
/// to `kind: "skill"` with the inner name, preserving `raw_name` for
/// traceability. Non-matching invocations are left unchanged.
pub fn unwrap_skill_invocations(invocations: &mut [crate::types::NormalizedInvocation]) {
    for inv in invocations.iter_mut() {
        if let Some((_, key)) = AMP_SKILL_WRAPPERS
            .iter()
            .find(|(name, _)| *name == inv.name)
        {
            if let Some(inner_name) = inv
                .arguments
                .as_ref()
                .and_then(|a| a.get(*key))
                .and_then(|v| v.as_str())
            {
                inv.raw_name = Some(inv.name.clone());
                inv.name = inner_name.to_string();
                inv.kind = "skill".to_string();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- parse_timestamp tests ---

    #[test]
    fn parse_timestamp_i64_milliseconds() {
        let val = json!(1_700_000_000_000_i64);
        assert_eq!(parse_timestamp(&val), Some(1_700_000_000_000));
    }

    #[test]
    fn parse_timestamp_i64_seconds() {
        let val = json!(1_700_000_000_i64);
        assert_eq!(parse_timestamp(&val), Some(1_700_000_000_000));
    }

    #[test]
    fn parse_timestamp_numeric_string_seconds() {
        let val = json!("1700000000");
        assert_eq!(parse_timestamp(&val), Some(1_700_000_000_000));
    }

    #[test]
    fn parse_timestamp_numeric_string_millis() {
        let val = json!("1700000000000");
        assert_eq!(parse_timestamp(&val), Some(1_700_000_000_000));
    }

    #[test]
    fn parse_timestamp_iso8601_with_fractional() {
        let val = json!("2025-11-12T18:31:32.217Z");
        let ts = parse_timestamp(&val).unwrap();
        assert!(ts > 0);
        // Verify it round-trips correctly through chrono
        let expected = chrono::DateTime::parse_from_rfc3339("2025-11-12T18:31:32.217Z")
            .unwrap()
            .timestamp_millis();
        assert_eq!(ts, expected);
    }

    #[test]
    fn parse_timestamp_iso8601_without_fractional() {
        let val = json!("2025-11-12T18:31:32Z");
        let ts = parse_timestamp(&val).unwrap();
        assert!(ts > 0);
    }

    #[test]
    fn parse_timestamp_rfc3339_with_offset() {
        let val = json!("2025-11-12T18:31:32+00:00");
        assert!(parse_timestamp(&val).is_some());
    }

    #[test]
    fn parse_timestamp_null_returns_none() {
        let val = json!(null);
        assert_eq!(parse_timestamp(&val), None);
    }

    #[test]
    fn parse_timestamp_invalid_string_returns_none() {
        let val = json!("not-a-timestamp");
        assert_eq!(parse_timestamp(&val), None);
    }

    #[test]
    fn parse_timestamp_empty_string_returns_none() {
        let val = json!("");
        assert_eq!(parse_timestamp(&val), None);
    }

    #[test]
    fn parse_timestamp_object_returns_none() {
        let val = json!({"time": 123});
        assert_eq!(parse_timestamp(&val), None);
    }

    #[test]
    fn parse_timestamp_negative_i64() {
        let val = json!(-1000);
        assert_eq!(parse_timestamp(&val), Some(-1000));
    }

    #[test]
    fn parse_timestamp_zero() {
        let val = json!(0);
        assert_eq!(parse_timestamp(&val), Some(0));
    }

    // --- flatten_content tests ---

    #[test]
    fn flatten_content_plain_string() {
        let val = json!("Hello, world!");
        assert_eq!(flatten_content(&val), "Hello, world!");
    }

    #[test]
    fn flatten_content_text_block_array() {
        let val = json!([
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"}
        ]);
        assert_eq!(flatten_content(&val), "Line 1\nLine 2");
    }

    #[test]
    fn flatten_content_tool_use_block() {
        let val = json!([
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/src/main.rs"}}
        ]);
        assert_eq!(flatten_content(&val), "[Tool: Read - /src/main.rs]");
    }

    #[test]
    fn flatten_content_mixed_blocks() {
        let val = json!([
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "name": "Write", "input": {"description": "writing file"}}
        ]);
        assert_eq!(flatten_content(&val), "Hello\n[Tool: Write - writing file]");
    }

    #[test]
    fn flatten_content_input_text_block() {
        let val = json!([{"type": "input_text", "text": "Codex input"}]);
        assert_eq!(flatten_content(&val), "Codex input");
    }

    #[test]
    fn flatten_content_output_text_block() {
        // Modern Codex assistant messages encode text as `output_text` blocks.
        let val = json!([{"type": "output_text", "text": "Codex assistant output"}]);
        assert_eq!(flatten_content(&val), "Codex assistant output");
    }

    #[test]
    fn flatten_content_null_returns_empty() {
        let val = json!(null);
        assert_eq!(flatten_content(&val), "");
    }

    #[test]
    fn flatten_content_empty_array() {
        let val = json!([]);
        assert_eq!(flatten_content(&val), "");
    }

    #[test]
    fn flatten_content_plain_string_array() {
        let val = json!(["Hello", "World"]);
        assert_eq!(flatten_content(&val), "Hello\nWorld");
    }

    #[test]
    fn flatten_content_empty_string() {
        let val = json!("");
        assert_eq!(flatten_content(&val), "");
    }

    #[test]
    fn flatten_content_number_returns_empty() {
        let val = json!(42);
        assert_eq!(flatten_content(&val), "");
    }

    #[test]
    fn flatten_content_whitespace_only() {
        let val = json!("   ");
        assert_eq!(flatten_content(&val), "   ");
    }

    // --- extract_invocations_from_content_blocks tests ---

    #[test]
    fn extract_invocations_plain_tool_use() {
        let val = json!([
            {"type": "text", "text": "Let me read that file."},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "/src/main.rs"}}
        ]);
        let invocations = extract_invocations_from_content_blocks(&val);
        assert_eq!(invocations.len(), 1);
        assert_eq!(invocations[0].kind, "tool");
        assert_eq!(invocations[0].name, "Read");
        assert!(invocations[0].raw_name.is_none());
        assert_eq!(invocations[0].call_id.as_deref(), Some("toolu_1"));
        assert_eq!(
            invocations[0].arguments.as_ref().unwrap()["path"],
            "/src/main.rs"
        );
    }

    #[test]
    fn extract_invocations_skill_not_unwrapped_by_shared_helper() {
        // The shared helper should NOT unwrap skill wrappers -- that's Amp-specific.
        let val = json!([
            {"type": "tool_use", "id": "toolu_2", "name": "skill", "input": {"name": "github-prs"}}
        ]);
        let invocations = extract_invocations_from_content_blocks(&val);
        assert_eq!(invocations.len(), 1);
        assert_eq!(invocations[0].kind, "tool");
        assert_eq!(invocations[0].name, "skill");
        assert!(invocations[0].raw_name.is_none());
    }

    #[test]
    fn extract_invocations_multiple_tools() {
        let val = json!([
            {"type": "tool_use", "name": "Read", "input": {"path": "a.rs"}},
            {"type": "text", "text": "Now editing..."},
            {"type": "tool_use", "name": "edit_file", "input": {"path": "a.rs", "old_str": "x", "new_str": "y"}},
            {"type": "tool_use", "name": "skill", "input": {"name": "git"}}
        ]);
        let invocations = extract_invocations_from_content_blocks(&val);
        assert_eq!(invocations.len(), 3);
        assert_eq!(invocations[0].name, "Read");
        assert_eq!(invocations[1].name, "edit_file");
        assert_eq!(invocations[2].name, "skill");
    }

    #[test]
    fn extract_invocations_no_tool_use_blocks() {
        let val = json!([
            {"type": "text", "text": "Just plain text."}
        ]);
        assert!(extract_invocations_from_content_blocks(&val).is_empty());
    }

    #[test]
    fn extract_invocations_string_content_returns_empty() {
        let val = json!("plain string");
        assert!(extract_invocations_from_content_blocks(&val).is_empty());
    }

    #[test]
    fn extract_invocations_null_returns_empty() {
        let val = json!(null);
        assert!(extract_invocations_from_content_blocks(&val).is_empty());
    }

    #[test]
    fn extract_invocations_tool_use_missing_name_skipped() {
        let val = json!([
            {"type": "tool_use", "input": {"path": "a.rs"}}
        ]);
        assert!(extract_invocations_from_content_blocks(&val).is_empty());
    }

    // --- unwrap_skill_invocations tests ---

    #[test]
    fn unwrap_skill_invocations_rewrites_skill_wrapper() {
        let mut invocations = vec![crate::types::NormalizedInvocation {
            kind: "tool".to_string(),
            name: "skill".to_string(),
            raw_name: None,
            call_id: Some("toolu_1".to_string()),
            arguments: Some(json!({"name": "github-prs"})),
        }];
        unwrap_skill_invocations(&mut invocations);
        assert_eq!(invocations[0].kind, "skill");
        assert_eq!(invocations[0].name, "github-prs");
        assert_eq!(invocations[0].raw_name.as_deref(), Some("skill"));
    }

    #[test]
    fn unwrap_skill_invocations_rewrites_load_skill_wrapper() {
        let mut invocations = vec![crate::types::NormalizedInvocation {
            kind: "tool".to_string(),
            name: "load_skill".to_string(),
            raw_name: None,
            call_id: None,
            arguments: Some(json!({"name": "git"})),
        }];
        unwrap_skill_invocations(&mut invocations);
        assert_eq!(invocations[0].kind, "skill");
        assert_eq!(invocations[0].name, "git");
        assert_eq!(invocations[0].raw_name.as_deref(), Some("load_skill"));
    }

    #[test]
    fn unwrap_skill_invocations_leaves_non_wrappers_unchanged() {
        let mut invocations = vec![crate::types::NormalizedInvocation {
            kind: "tool".to_string(),
            name: "Read".to_string(),
            raw_name: None,
            call_id: None,
            arguments: Some(json!({"path": "/src/main.rs"})),
        }];
        unwrap_skill_invocations(&mut invocations);
        assert_eq!(invocations[0].kind, "tool");
        assert_eq!(invocations[0].name, "Read");
        assert!(invocations[0].raw_name.is_none());
    }

    #[test]
    fn unwrap_skill_invocations_no_inner_name_leaves_unchanged() {
        let mut invocations = vec![crate::types::NormalizedInvocation {
            kind: "tool".to_string(),
            name: "skill".to_string(),
            raw_name: None,
            call_id: None,
            arguments: Some(json!({"arguments": "something"})),
        }];
        unwrap_skill_invocations(&mut invocations);
        // No "name" key in arguments -- should remain as tool "skill"
        assert_eq!(invocations[0].kind, "tool");
        assert_eq!(invocations[0].name, "skill");
        assert!(invocations[0].raw_name.is_none());
    }

    // --- split_content_blocks tests ---

    #[test]
    fn split_content_blocks_separates_text_tooluse_toolresult_thinking() {
        let v = json!([
            {"type":"text","text":"hi"},
            {"type":"tool_use","name":"Read","id":"tu_1","input":{"file_path":"/a"}},
            {"type":"tool_result","tool_use_id":"tu_1","content":"file body"},
            {"type":"thinking","text":"let me think"}
        ]);
        let blocks = split_content_blocks(&v);
        assert_eq!(blocks.len(), 4);
        assert!(matches!(blocks[0], TypedBlock::Text(ref t) if t=="hi"));
        assert!(
            matches!(&blocks[1], TypedBlock::ToolCall{name, id, ..} if name=="Read" && id.as_deref()==Some("tu_1"))
        );
        assert!(
            matches!(&blocks[2], TypedBlock::ToolResult{tool_use_id, ..} if tool_use_id.as_deref()==Some("tu_1"))
        );
        assert!(matches!(blocks[3], TypedBlock::Thinking(ref t) if t=="let me think"));
    }

    #[test]
    fn split_content_blocks_thinking_uses_anthropic_thinking_key() {
        let v = json!([{"type":"thinking","thinking":"real reasoning","signature":"sig"}]);
        let blocks = split_content_blocks(&v);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], TypedBlock::Thinking(ref t) if t=="real reasoning"));
    }

    #[test]
    fn add_raw_role_inserts_reserved_string_without_touching_other_fields() {
        let raw_envelope = json!({"source": "chatgpt"});
        let projected_extra = json!({"existing": true});

        let extra = add_raw_role(&raw_envelope, projected_extra, "assistant").unwrap();

        assert_eq!(extra, json!({"existing": true, "raw_role": "assistant"}));
        assert_eq!(raw_envelope, json!({"source": "chatgpt"}));
    }

    #[test]
    fn add_raw_role_rejects_collision_in_raw_or_projected_extra_and_non_objects() {
        let raw_collision = add_raw_role(
            &json!({"raw_role": "original"}),
            json!({"existing": true}),
            "assistant",
        )
        .unwrap_err();
        assert!(raw_collision.to_string().contains("raw_role"));

        let projected_collision = add_raw_role(
            &json!({"source": "chatgpt"}),
            json!({"raw_role": "projected"}),
            "assistant",
        )
        .unwrap_err();
        assert!(projected_collision.to_string().contains("raw_role"));

        assert!(add_raw_role(&json!([]), json!({}), "assistant").is_err());
        assert!(add_raw_role(&json!({}), json!([]), "assistant").is_err());
        assert!(add_raw_role(&json!({}), json!({}), "   ").is_err());
    }

    #[test]
    fn set_tool_result_pairing_enforces_exactly_one_branch() {
        let mut extra = json!({"tool_call_id": "old", "unpaired": true});

        set_tool_result_pairing(&mut extra, Some(" call_1 ")).unwrap();
        assert_eq!(
            extra
                .get("tool_call_id")
                .and_then(serde_json::Value::as_str),
            Some("call_1")
        );
        assert!(extra.get("unpaired").is_none());
        assert!(
            extra.get("tool_call_id").is_some() ^ (extra.get("unpaired") == Some(&json!(true)))
        );

        set_tool_result_pairing(&mut extra, None).unwrap();
        assert!(extra.get("tool_call_id").is_none());
        assert_eq!(extra.get("unpaired"), Some(&json!(true)));
        assert!(
            extra.get("tool_call_id").is_some() ^ (extra.get("unpaired") == Some(&json!(true)))
        );
    }

    #[test]
    fn add_encrypted_content_rejects_type_duplicate_and_top_level_collision() {
        let raw_envelope = json!({"source": "chatgpt"});
        let mut extra = json!({"existing": true});
        let encrypted = json!("opaque-secret-value");

        add_encrypted_content(&raw_envelope, &mut extra, &encrypted).unwrap();
        assert_eq!(extra["encrypted_content"], encrypted);

        let duplicate = add_encrypted_content(&raw_envelope, &mut extra, &encrypted).unwrap_err();
        assert!(duplicate.to_string().contains("encrypted_content"));

        let raw_collision = add_encrypted_content(
            &json!({"encrypted_content": "original"}),
            &mut json!({}),
            &encrypted,
        )
        .unwrap_err();
        assert!(raw_collision.to_string().contains("encrypted_content"));

        let projected_collision = add_encrypted_content(
            &raw_envelope,
            &mut json!({"encrypted_content": "projected"}),
            &encrypted,
        )
        .unwrap_err();
        assert!(
            projected_collision
                .to_string()
                .contains("encrypted_content")
        );

        let numeric =
            add_encrypted_content(&raw_envelope, &mut json!({}), &json!(123)).unwrap_err();
        assert!(numeric.to_string().contains("encrypted_content"));
        assert!(!numeric.to_string().contains("123"));

        let object =
            add_encrypted_content(&raw_envelope, &mut json!({}), &json!({"opaque": "secret"}))
                .unwrap_err();
        assert!(object.to_string().contains("encrypted_content"));
        assert!(!object.to_string().contains("secret"));
    }
}
