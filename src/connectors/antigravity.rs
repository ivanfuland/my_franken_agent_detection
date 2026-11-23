//! Antigravity (`agy`) CLI connector.
//!
//! [Antigravity](https://antigravity.google) ships the `agy` CLI, Google's
//! successor to the Gemini CLI (`gmi`). It stores CLI session history under
//! `~/.gemini/antigravity-cli/` (distinct from the legacy Gemini CLI under
//! `~/.gemini/tmp/` and separate from Antigravity IDE):
//!
//! - `brain/<uuid>/.system_generated/logs/transcript_full.jsonl` — untruncated
//!   line-per-step JSONL transcript (preferred primary read path).
//! - `brain/<uuid>/.system_generated/logs/transcript.jsonl` — compact fallback
//!   transcript (used when `transcript_full.jsonl` is absent, e.g. `agy 1.0.7`).
//! - `conversation_summaries.db` — optional read-only `SQLite` metadata sidecar
//!   (`workspace_uris`, `title`, `parent_conversation_id`, `agent_name`,
//!   `nesting_depth`), read strictly read-only under feature `antigravity-db`.
//! - `conversations/<uuid>.db` — per-conversation `SQLite`/protobuf archive,
//!   discovered only when located inside the active scan root.
//!
//! The `<uuid>` keying `conversations/<uuid>.db` and `brain/<uuid>/` is the same
//! id used by `agy --conversation <uuid>`.
//!
//! ## Normalized 6-role mapping (`agy 1.5.14` + `1.0.7` compatible)
//!
//! Every emitted message conforms to the canonical 6-role storage contract
//! (`user`, `assistant`, `reasoning`, `tool_call`, `tool_result`, `system`) and
//! carries a non-empty `extra["raw_role"]`:
//!
//! - `USER_INPUT` (`USER_EXPLICIT`) -> `role: "user"`, `raw_role: "user"`.
//!   Unwraps `<USER_REQUEST>` blocks and retains wrapper metadata in `extra`.
//! - `PLANNER_RESPONSE` (`MODEL`) -> ordered messages with `raw_role: "assistant"`
//!   and message-level `extra["model"]` (`thinking` -> `reasoning`, `content` ->
//!   `assistant`, `tool_calls[k]` -> `tool_call`, never dropped when `content`
//!   is absent). API token usage (`input_tokens`, `output_tokens`,
//!   `cache_read_tokens`) and `tool_call_count` are attached to `extra["cass"]`
//!   on exactly one emitted message per step to prevent double counting.
//! - `GENERIC` (`MODEL`) -> `role: "tool_result"`, `raw_role: "tool"`. Paired
//!   conservatively 1-to-1 with the preceding `PLANNER_RESPONSE.tool_calls`
//!   only when skipping interleaved `SYSTEM_MESSAGE` / `EPHEMERAL_MESSAGE` /
//!   `CONVERSATION_HISTORY` yields an exact count match (`len(tool_calls) ==
//!   len(GENERIC)`). Otherwise fail-closed (`extra["unpaired"] = true`).
//! - Legacy `agy 1.0.7` `MODEL` tool steps (`VIEW_FILE`, `RUN_COMMAND`,
//!   `EDIT_FILE`, ...) -> `tool_call` + paired `tool_result` when `tool_calls`
//!   is present, or unpaired `tool_result` otherwise.
//! - `ERROR_MESSAGE` -> `role: "system"`, `author: "error"`, preserving `error`.
//! - `CHECKPOINT` -> `role: "system"`, `author: "checkpoint"`.
//! - `SYSTEM_MESSAGE` / `EPHEMERAL_MESSAGE` -> filters two known high-volume
//!   boilerplate templates (`PreInvocation hook` with `<agent-contract-source `
//!   and `Reminder: work efficiently.` with `<SYSTEM_MESSAGE>`) while retaining
//!   subagent / background notifications as `role: "system"`.
//! - `CONVERSATION_HISTORY` -> skipped marker.

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{Map, Value};
use walkdir::WalkDir;

use super::scan::{DiscoveredSourceFile, DiscoveredSourceRole, ScanContext, ScanRoot};
use super::utils::{
    add_raw_role, dedupe_path_key, env_path_nonempty, excluded_scan_paths_from_env,
    path_is_excluded, set_tool_result_pairing,
};
use super::{Connector, file_modified_since, franken_detection_for_connector, parse_timestamp};
use crate::types::{
    DetectionResult, NormalizedConversation, NormalizedInvocation, NormalizedMessage,
    reindex_messages,
};

/// Normalized agent slug emitted for every Antigravity CLI conversation.
const AGENT_SLUG: &str = "antigravity";

/// Maximum transcript file size read in streaming mode (100 MiB).
const MAX_TRANSCRIPT_BYTES: u64 = 100 * 1024 * 1024;

/// Maximum character count kept in the inline `tool_call.content` summary.
/// Complete, untruncated arguments always live in `invocations[0].arguments`
/// and `extra["arguments"]`.
const MAX_TOOL_CALL_SUMMARY_CHARS: usize = 512;

/// Connector for the Antigravity (`agy`) CLI coding agent.
pub struct AntigravityConnector;

impl Default for AntigravityConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl AntigravityConnector {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Base directory holding `conversations/`, `brain/`, and
    /// `conversation_summaries.db`.
    ///
    /// Defaults to `~/.gemini/antigravity-cli`; overridable for tests / remote
    /// mirrors via `CASS_ANTIGRAVITY_DATA_ROOT`. Never probes IDE directories.
    fn base_root() -> PathBuf {
        if let Some(explicit) = env_path_nonempty("CASS_ANTIGRAVITY_DATA_ROOT") {
            return explicit;
        }
        dirs::home_dir()
            .unwrap_or_default()
            .join(".gemini")
            .join("antigravity-cli")
    }

    /// Compact transcript path for a `brain/<uuid>` conversation directory.
    fn transcript_path(conversation_dir: &Path) -> PathBuf {
        conversation_dir
            .join(".system_generated")
            .join("logs")
            .join("transcript.jsonl")
    }

    /// Full untruncated transcript path for a `brain/<uuid>` conversation directory.
    fn transcript_full_path(conversation_dir: &Path) -> PathBuf {
        conversation_dir
            .join(".system_generated")
            .join("logs")
            .join("transcript_full.jsonl")
    }

    /// Single primary transcript read path for a conversation directory:
    /// prefer `transcript_full.jsonl`, fall back to `transcript.jsonl`.
    fn primary_transcript_path(conversation_dir: &Path) -> Option<PathBuf> {
        let full = Self::transcript_full_path(conversation_dir);
        if full.is_file() {
            return Some(full);
        }
        let compact = Self::transcript_path(conversation_dir);
        compact.is_file().then_some(compact)
    }

    /// A conversation directory is recognized by the presence of its primary
    /// transcript (`transcript_full.jsonl` or `transcript.jsonl`).
    fn is_conversation_dir(path: &Path) -> bool {
        Self::primary_transcript_path(path).is_some()
    }

    fn is_transcript_filename(name: &OsStr) -> bool {
        name == "transcript_full.jsonl" || name == "transcript.jsonl"
    }

    /// Validate that `path` is an existing `transcript_full.jsonl` or
    /// `transcript.jsonl` under `<uuid>/.system_generated/logs/` and return
    /// `<uuid>` (the conversation directory).
    fn conversation_dir_for_transcript_file(path: &Path) -> Option<PathBuf> {
        if !path.is_file() {
            return None;
        }
        let name = path.file_name()?;
        if !Self::is_transcript_filename(name) {
            return None;
        }
        let conv_dir = path
            .parent()
            .filter(|logs| logs.file_name().is_some_and(|n| n == "logs"))
            .and_then(Path::parent)
            .filter(|sg| sg.file_name().is_some_and(|n| n == ".system_generated"))
            .and_then(Path::parent)?;
        Self::conversation_uuid(conv_dir)?;
        Some(conv_dir.to_path_buf())
    }

    /// Whether `path` looks like a hermetic Antigravity CLI root passed via
    /// `ScanContext::data_dir` in default-detection mode.
    fn looks_like_data_dir(path: &Path) -> bool {
        path.join("brain").is_dir()
            || (path.is_dir() && path.file_name().is_some_and(|n| n == "brain"))
            || Self::is_conversation_dir(path)
            || Self::conversation_dir_for_transcript_file(path).is_some()
    }

    /// The `<uuid>` for a conversation directory (its directory name).
    fn conversation_uuid(conversation_dir: &Path) -> Option<String> {
        conversation_dir
            .file_name()
            .and_then(|n| n.to_str())
            .filter(|s| !s.trim().is_empty())
            .map(String::from)
    }

    /// The sibling `conversations/<uuid>.db` for a `brain/<uuid>` directory, when
    /// the standard layout is intact (`<base>/brain/<uuid>` <->
    /// `<base>/conversations/<uuid>.db`).
    fn sibling_db_path(conversation_dir: &Path) -> Option<PathBuf> {
        let uuid = Self::conversation_uuid(conversation_dir)?;
        let base = conversation_dir.parent()?.parent()?;
        let db = base.join("conversations").join(format!("{uuid}.db"));
        db.is_file().then_some(db)
    }

    /// Locate `<base>/conversation_summaries.db` for a `<base>/brain/<uuid>` dir.
    #[cfg(feature = "antigravity-db")]
    fn summary_db_path(conversation_dir: &Path) -> Option<PathBuf> {
        let brain = conversation_dir.parent()?;
        if brain.file_name().is_none_or(|n| n != "brain") {
            return None;
        }
        let base = brain.parent()?;
        let db = base.join("conversation_summaries.db");
        db.is_file().then_some(db)
    }

    fn source_roots(ctx: &ScanContext) -> Vec<ScanRoot> {
        let mut roots = if ctx.use_default_detection() {
            env_path_nonempty("CASS_ANTIGRAVITY_DATA_ROOT").map_or_else(
                || {
                    if Self::looks_like_data_dir(&ctx.data_dir) {
                        vec![ScanRoot::local(ctx.data_dir.clone())]
                    } else {
                        vec![ScanRoot::local(Self::base_root())]
                    }
                },
                |explicit| vec![ScanRoot::local(explicit)],
            )
        } else {
            ctx.scan_roots.clone()
        };

        roots.sort_by(|a, b| a.path.cmp(&b.path));
        roots.dedup_by(|a, b| a.path == b.path);
        roots
    }

    /// Resolve every `brain/<uuid>` conversation directory reachable from a
    /// directory scan target. The target may be the conversation directory
    /// itself, the `brain/` directory, or an ancestor such as `antigravity-cli`
    /// or `~/.gemini`.
    fn conversation_dirs(scan_target: &Path) -> Vec<PathBuf> {
        if Self::is_conversation_dir(scan_target) {
            return vec![scan_target.to_path_buf()];
        }

        let mut out = Vec::new();
        for entry in WalkDir::new(scan_target)
            .min_depth(1)
            .max_depth(8)
            .into_iter()
            .flatten()
        {
            if !entry.file_type().is_file() || !Self::is_transcript_filename(entry.file_name()) {
                continue;
            }
            if let Some(dir) = Self::conversation_dir_for_transcript_file(entry.path()) {
                out.push(dir);
            }
        }

        out.sort();
        out.dedup();
        out
    }

    /// True when the conversation's sole primary transcript is newer than `since_ts`.
    fn conversation_modified_since(conversation_dir: &Path, since_ts: Option<i64>) -> bool {
        if since_ts.is_none() {
            return true;
        }
        Self::primary_transcript_path(conversation_dir)
            .is_some_and(|transcript| file_modified_since(&transcript, since_ts))
    }

    /// Stream-read a transcript JSONL file with a 100 MiB ceiling, UTF-8 BOM
    /// stripping, malformed-line skip, and full-conversation discard on mid-read
    /// I/O error.
    fn read_transcript_records(transcript: &Path) -> Vec<Value> {
        match fs::metadata(transcript) {
            Ok(meta) if meta.len() > MAX_TRANSCRIPT_BYTES => {
                tracing::warn!(
                    transcript = %transcript.display(),
                    bytes = meta.len(),
                    limit = MAX_TRANSCRIPT_BYTES,
                    "antigravity: skipping oversized transcript"
                );
                return Vec::new();
            }
            Ok(_) => {}
            Err(_) => return Vec::new(),
        }

        let Ok(file) = File::open(transcript) else {
            return Vec::new();
        };
        read_transcript_records_from_reader(
            BufReader::new(file.take(MAX_TRANSCRIPT_BYTES + 1)),
            transcript,
        )
    }
}

fn read_transcript_records_from_reader<R: BufRead>(mut reader: R, transcript: &Path) -> Vec<Value> {
    let mut records = Vec::new();
    let mut buf = String::new();
    let mut total_bytes: u64 = 0;
    let mut lineno: usize = 0;

    loop {
        buf.clear();
        match reader.read_line(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                lineno += 1;
                total_bytes = total_bytes.saturating_add(u64::try_from(n).unwrap_or(u64::MAX));
                if total_bytes > MAX_TRANSCRIPT_BYTES {
                    tracing::warn!(
                        transcript = %transcript.display(),
                        bytes = total_bytes,
                        limit = MAX_TRANSCRIPT_BYTES,
                        "antigravity: transcript exceeded streaming byte limit; discarding"
                    );
                    return Vec::new();
                }
                let trimmed = buf.trim_start_matches('\u{feff}').trim();
                if trimmed.is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(trimmed) {
                    Ok(value) if value.is_object() => records.push(value),
                    Ok(_) => tracing::warn!(
                        transcript = %transcript.display(),
                        line = lineno,
                        "antigravity: skipping non-object transcript line"
                    ),
                    Err(err) => tracing::warn!(
                        transcript = %transcript.display(),
                        line = lineno,
                        error = %err,
                        "antigravity: skipping malformed transcript line"
                    ),
                }
            }
            Err(err) => {
                tracing::warn!(
                    transcript = %transcript.display(),
                    line = lineno + 1,
                    error = %err,
                    "antigravity: I/O error while reading transcript; discarding partial conversation"
                );
                return Vec::new();
            }
        }
    }

    records
}

// ---------------------------------------------------------------------------
// Optional read-only conversation_summaries.db enrichment (antigravity-db)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct ConversationSummaryRow {
    title: Option<String>,
    workspace: Option<PathBuf>,
    parent_conversation_id: Option<String>,
    agent_name: Option<String>,
    nesting_depth: Option<i64>,
}

#[derive(Default)]
struct SummaryDbCache {
    #[cfg(feature = "antigravity-db")]
    by_db_path: HashMap<PathBuf, HashMap<String, ConversationSummaryRow>>,
}

impl SummaryDbCache {
    #[cfg(feature = "antigravity-db")]
    fn lookup_bulk(
        &mut self,
        conversation_dir: &Path,
        uuid: &str,
    ) -> Option<ConversationSummaryRow> {
        let db_path = AntigravityConnector::summary_db_path(conversation_dir)?;
        let rows = self
            .by_db_path
            .entry(db_path.clone())
            .or_insert_with(|| load_summary_rows_bulk(&db_path));
        rows.get(uuid).cloned()
    }

    #[cfg(not(feature = "antigravity-db"))]
    #[allow(clippy::unused_self, clippy::needless_pass_by_ref_mut)]
    const fn lookup_bulk(
        &mut self,
        _conversation_dir: &Path,
        _uuid: &str,
    ) -> Option<ConversationSummaryRow> {
        None
    }

    #[cfg(feature = "antigravity-db")]
    fn lookup_single(conversation_dir: &Path, uuid: &str) -> Option<ConversationSummaryRow> {
        let db_path = AntigravityConnector::summary_db_path(conversation_dir)?;
        load_summary_row_single(&db_path, uuid)
    }

    #[cfg(not(feature = "antigravity-db"))]
    const fn lookup_single(
        _conversation_dir: &Path,
        _uuid: &str,
    ) -> Option<ConversationSummaryRow> {
        None
    }
}

#[cfg(feature = "antigravity-db")]
fn parse_file_uri(uri: &str) -> Option<PathBuf> {
    let raw_path = uri.strip_prefix("file://")?;
    let decoded = urlencoding::decode(raw_path).ok()?;
    let mut path_str = decoded.as_ref();
    if cfg!(windows) && path_str.starts_with('/') && path_str.len() > 2 {
        let bytes = path_str.as_bytes();
        if bytes[2] == b':' && bytes[1].is_ascii_alphabetic() {
            path_str = &path_str[1..];
        }
    }
    if path_str.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(path_str))
}

#[cfg(feature = "antigravity-db")]
fn is_overlay_workspace(path: &Path) -> bool {
    let s = path.to_string_lossy();
    s.contains("/antigravity-overlays/") || s.contains("\\antigravity-overlays\\")
}

#[cfg(feature = "antigravity-db")]
fn parse_workspace_uris(raw_json: &str) -> Option<PathBuf> {
    let uris: Vec<String> = serde_json::from_str(raw_json).ok()?;
    let decoded: Vec<PathBuf> = uris.iter().filter_map(|u| parse_file_uri(u)).collect();
    decoded
        .iter()
        .find(|p| !is_overlay_workspace(p))
        .or_else(|| decoded.first())
        .cloned()
}

#[cfg(feature = "antigravity-db")]
fn build_summary_row(
    title: String,
    workspace_uris: &str,
    parent_conversation_id: String,
    agent_name: String,
    nesting_depth: i64,
) -> ConversationSummaryRow {
    let clean_opt = |s: String| {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    };
    ConversationSummaryRow {
        title: clean_opt(title),
        workspace: parse_workspace_uris(workspace_uris),
        parent_conversation_id: clean_opt(parent_conversation_id),
        agent_name: clean_opt(agent_name),
        nesting_depth: (nesting_depth > 0).then_some(nesting_depth),
    }
}

#[cfg(feature = "antigravity-db")]
fn open_summary_db_readonly(db_path: &Path) -> Option<frankensqlite::Connection> {
    use frankensqlite::compat::{OpenFlags, open_with_flags};

    // Strictly read-only: NEVER execute PRAGMA journal_mode or any write PRAGMA.
    match open_with_flags(
        db_path.to_string_lossy().as_ref(),
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) {
        Ok(conn) => Some(conn),
        Err(err) => {
            tracing::debug!(
                db = %db_path.display(),
                error = %err,
                "antigravity: unable to open conversation_summaries.db read-only; degrading gracefully"
            );
            None
        }
    }
}

#[cfg(feature = "antigravity-db")]
fn load_summary_rows_bulk(db_path: &Path) -> HashMap<String, ConversationSummaryRow> {
    use frankensqlite::compat::{ConnectionExt, RowExt};

    const QUERY: &str = "SELECT conversation_id, title, workspace_uris, \
                         parent_conversation_id, agent_name, nesting_depth \
                         FROM conversation_summaries";

    let Some(conn) = open_summary_db_readonly(db_path) else {
        return HashMap::new();
    };
    let result: Result<Vec<(String, ConversationSummaryRow)>, _> =
        conn.query_map_collect(QUERY, &[], |row| {
            let conv_id: String = row.get_typed(0)?;
            let title: String = row.get_typed::<Option<String>>(1)?.unwrap_or_default();
            let workspace_uris: String = row.get_typed::<Option<String>>(2)?.unwrap_or_default();
            let parent_id: String = row.get_typed::<Option<String>>(3)?.unwrap_or_default();
            let agent_name: String = row.get_typed::<Option<String>>(4)?.unwrap_or_default();
            let nesting_depth: i64 = row.get_typed::<Option<i64>>(5)?.unwrap_or(0);
            Ok((
                conv_id,
                build_summary_row(title, &workspace_uris, parent_id, agent_name, nesting_depth),
            ))
        });

    match result {
        Ok(rows) => rows.into_iter().collect(),
        Err(err) => {
            tracing::debug!(
                db = %db_path.display(),
                error = %err,
                "antigravity: failed bulk query on conversation_summaries.db; degrading gracefully"
            );
            HashMap::new()
        }
    }
}

#[cfg(feature = "antigravity-db")]
fn load_summary_row_single(db_path: &Path, uuid: &str) -> Option<ConversationSummaryRow> {
    use frankensqlite::compat::{ConnectionExt, ParamValue, RowExt};

    const QUERY: &str = "SELECT title, workspace_uris, parent_conversation_id, \
                         agent_name, nesting_depth \
                         FROM conversation_summaries WHERE conversation_id = ?1";

    let conn = open_summary_db_readonly(db_path)?;
    let params = [ParamValue::from(uuid.to_string())];
    let result: Result<Vec<ConversationSummaryRow>, _> =
        conn.query_map_collect(QUERY, &params, |row| {
            let title: String = row.get_typed::<Option<String>>(0)?.unwrap_or_default();
            let workspace_uris: String = row.get_typed::<Option<String>>(1)?.unwrap_or_default();
            let parent_id: String = row.get_typed::<Option<String>>(2)?.unwrap_or_default();
            let agent_name: String = row.get_typed::<Option<String>>(3)?.unwrap_or_default();
            let nesting_depth: i64 = row.get_typed::<Option<i64>>(4)?.unwrap_or(0);
            Ok(build_summary_row(
                title,
                &workspace_uris,
                parent_id,
                agent_name,
                nesting_depth,
            ))
        });

    match result {
        Ok(mut rows) => rows.pop(),
        Err(err) => {
            tracing::debug!(
                db = %db_path.display(),
                conversation_id = uuid,
                error = %err,
                "antigravity: failed point query on conversation_summaries.db; degrading gracefully"
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Text extraction, template filtering, and model detection
// ---------------------------------------------------------------------------

/// Extract the inner text of every `<TAG>...</TAG>` block in `content`.
fn extract_tagged_blocks(content: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = content;
    while let Some(start) = rest.find(&open) {
        let after = &rest[start + open.len()..];
        let Some(end) = after.find(&close) else {
            break;
        };
        out.push(after[..end].trim().to_string());
        rest = &after[end + close.len()..];
    }
    out
}

/// Split a wrapped `USER_INPUT` payload into `(request body, settings-change, metadata)`.
fn extract_user_request(content: &str) -> (String, Option<String>, Option<String>) {
    let requests = extract_tagged_blocks(content, "USER_REQUEST");
    let settings = extract_tagged_blocks(content, "USER_SETTINGS_CHANGE")
        .into_iter()
        .next();
    let metadata = extract_tagged_blocks(content, "ADDITIONAL_METADATA")
        .into_iter()
        .next();
    let body = if requests.is_empty() {
        content.trim().to_string()
    } else {
        requests.join("\n\n")
    };
    (body, settings, metadata)
}

/// Pull the model name out of a `<USER_SETTINGS_CHANGE>` body such as
/// "The user changed setting `Model Selection` from None to Gemini 3.1 Pro
/// (High). No need to comment..." -> `Some("Gemini 3.1 Pro (High)")`.
fn model_from_settings(settings: &str) -> Option<String> {
    let from_marker = " from ";
    let to_marker = " to ";
    let search_from = settings
        .find(from_marker)
        .map_or(0, |i| i + from_marker.len());
    let rel = settings[search_from..].find(to_marker)?;
    let start = search_from + rel + to_marker.len();
    let tail = &settings[start..];
    let end = tail.find(". ").unwrap_or(tail.len());
    let model = tail[..end].trim().trim_end_matches('.').trim();
    if model.is_empty() || model.eq_ignore_ascii_case("none") {
        return None;
    }
    Some(model.to_string())
}

/// Detect the primary model name for the conversation: prefer the first
/// `PLANNER_RESPONSE.model` (`agy 1.5.14`), falling back to `<USER_SETTINGS_CHANGE>`
/// (`agy 1.0.7`).
fn detect_model(ordered: &[&Value]) -> Option<String> {
    for rec in ordered {
        if rec.get("type").and_then(Value::as_str) == Some("PLANNER_RESPONSE")
            && let Some(model) = rec
                .get("model")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
        {
            return Some(model.to_string());
        }
    }
    for rec in ordered {
        if rec.get("type").and_then(Value::as_str) != Some("USER_INPUT") {
            continue;
        }
        let content = rec.get("content").and_then(Value::as_str).unwrap_or("");
        if let Some(settings) = extract_tagged_blocks(content, "USER_SETTINGS_CHANGE")
            .into_iter()
            .next()
            && let Some(model) = model_from_settings(&settings)
        {
            return Some(model);
        }
    }
    None
}

/// Filter only the two known high-volume boilerplate `SYSTEM_MESSAGE` templates.
fn is_boilerplate_system_message(content: &str) -> bool {
    (content.contains("The following system message was injected by a PreInvocation hook:")
        && content.contains("<agent-contract-source "))
        || (content.contains("Reminder: work efficiently.") && content.contains("<SYSTEM_MESSAGE>"))
}

// ---------------------------------------------------------------------------
// Tool call parsing and conservative turn-window pairing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ParsedToolCall {
    name: String,
    call_id: Option<String>,
    arguments: Option<Value>,
}

#[derive(Debug, Clone)]
struct GenericPairing {
    call_id: String,
    tool_name: String,
}

#[derive(Default)]
struct TurnPairingPlan {
    /// For each `PLANNER_RESPONSE` position with non-empty `tool_calls`:
    /// `(parsed_call, paired_call_id)` where `None` indicates an unpaired mismatch.
    planner_calls: HashMap<usize, Vec<(ParsedToolCall, Option<String>)>>,
    /// For each matched `GENERIC` position: `(call_id, tool_name)`.
    generic_matches: HashMap<usize, GenericPairing>,
}

fn parse_raw_tool_calls(arr: &[Value], fallback_name: &str) -> Vec<ParsedToolCall> {
    arr.iter()
        .map(|tc| {
            let name = tc
                .get("name")
                .or_else(|| tc.get("tool_name"))
                .or_else(|| tc.pointer("/function/name"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map_or_else(|| fallback_name.to_string(), String::from);
            let call_id = tc
                .get("id")
                .or_else(|| tc.get("call_id"))
                .or_else(|| tc.get("tool_call_id"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from);
            let arguments = tc
                .get("arguments")
                .or_else(|| tc.get("args"))
                .or_else(|| tc.get("input"))
                .or_else(|| tc.pointer("/function/arguments"))
                .cloned()
                .map(|raw| match raw {
                    Value::String(s) => {
                        serde_json::from_str::<Value>(&s).unwrap_or(Value::String(s))
                    }
                    other => other,
                });
            ParsedToolCall {
                name,
                call_id,
                arguments,
            }
        })
        .collect()
}

fn step_index_label(rec: &Value, fallback_pos: usize) -> String {
    rec.get("step_index")
        .and_then(Value::as_i64)
        .map_or_else(|| format!("pos{fallback_pos}"), |idx| idx.to_string())
}

fn compute_turn_pairings(ordered: &[&Value]) -> TurnPairingPlan {
    let mut plan = TurnPairingPlan::default();

    for (pos, rec) in ordered.iter().enumerate() {
        if rec.get("type").and_then(Value::as_str) != Some("PLANNER_RESPONSE") {
            continue;
        }
        let Some(raw_calls) = rec
            .get("tool_calls")
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
        else {
            continue;
        };

        let parsed = parse_raw_tool_calls(raw_calls, "unknown");
        let mut generic_positions = Vec::new();

        for (j, next_rec) in ordered.iter().enumerate().skip(pos + 1) {
            let next_type = next_rec.get("type").and_then(Value::as_str).unwrap_or("");
            let next_source = next_rec.get("source").and_then(Value::as_str).unwrap_or("");
            match next_type {
                "PLANNER_RESPONSE" | "USER_INPUT" | "CHECKPOINT" => break,
                "GENERIC" if next_source == "MODEL" => generic_positions.push(j),
                // Skip interleaved SYSTEM_MESSAGE / EPHEMERAL_MESSAGE / CONVERSATION_HISTORY
                // and any other non-boundary step within the turn window.
                _ => {}
            }
        }

        if generic_positions.len() == parsed.len() {
            let step_label = step_index_label(rec, pos);
            let mut entries = Vec::with_capacity(parsed.len());
            for (k, (tc, gen_pos)) in parsed.into_iter().zip(generic_positions).enumerate() {
                let pair_id = tc
                    .call_id
                    .clone()
                    .unwrap_or_else(|| format!("agy:step{step_label}:call{k}"));
                plan.generic_matches.insert(
                    gen_pos,
                    GenericPairing {
                        call_id: pair_id.clone(),
                        tool_name: tc.name.clone(),
                    },
                );
                entries.push((tc, Some(pair_id)));
            }
            plan.planner_calls.insert(pos, entries);
        } else {
            let entries = parsed.into_iter().map(|tc| (tc, None)).collect();
            plan.planner_calls.insert(pos, entries);
        }
    }

    plan
}

fn render_tool_call_content(name: &str, arguments: Option<&Value>) -> String {
    let Some(args) = arguments.filter(|v| !v.is_null()) else {
        return name.to_string();
    };
    let rendered = args.to_string();
    if rendered.chars().count() <= MAX_TOOL_CALL_SUMMARY_CHARS {
        format!("{name}({rendered})")
    } else {
        let prefix: String = rendered.chars().take(MAX_TOOL_CALL_SUMMARY_CHARS).collect();
        format!("{name}({prefix}...)")
    }
}

fn extract_error_text(rec: &Value) -> Option<String> {
    let val = rec.get("error").filter(|v| !v.is_null())?;
    let text = match val {
        Value::String(s) => s.trim().to_string(),
        other => other.to_string(),
    };
    (!text.is_empty()).then_some(text)
}

fn merge_content_and_error(content: &str, error_text: Option<&str>) -> String {
    let c = content.trim();
    match (
        c.is_empty(),
        error_text.map(str::trim).filter(|e| !e.is_empty()),
    ) {
        (false, Some(err)) if c != err => format!("{c}\n{err}"),
        (false, _) => content.to_string(),
        (true, Some(err)) => err.to_string(),
        (true, None) => String::new(),
    }
}

fn insert_step_index(extra: &mut Map<String, Value>, rec: &Value) {
    if let Some(step_index) = rec.get("step_index").and_then(Value::as_i64) {
        extra.insert("step_index".to_string(), Value::from(step_index));
    }
}

struct MessageMeta<'a> {
    role: &'a str,
    raw_role: &'a str,
    author: Option<&'a str>,
    created_at: Option<i64>,
}

fn build_message_with_raw_role(
    rec: &Value,
    meta: &MessageMeta<'_>,
    content: String,
    extra: Map<String, Value>,
    invocations: Vec<NormalizedInvocation>,
) -> Option<NormalizedMessage> {
    match add_raw_role(rec, Value::Object(extra), meta.raw_role) {
        Ok(extra_val) => Some(NormalizedMessage {
            idx: 0,
            role: meta.role.to_string(),
            author: meta.author.map(String::from),
            created_at: meta.created_at,
            content,
            extra: extra_val,
            invocations,
            snippets: Vec::new(),
        }),
        Err(err) => {
            tracing::warn!(
                role = meta.role,
                raw_role = meta.raw_role,
                error = %err,
                "antigravity: skipping message due to raw_role contract error"
            );
            None
        }
    }
}

fn build_tool_call_message(
    rec: &Value,
    step_type: &str,
    step_model: Option<&str>,
    created_at: Option<i64>,
    tc: &ParsedToolCall,
    pair_id: Option<&str>,
) -> Option<NormalizedMessage> {
    let mut extra = Map::new();
    extra.insert("agy_type".to_string(), Value::String(step_type.to_string()));
    insert_step_index(&mut extra, rec);
    if let Some(model) = step_model {
        extra.insert("model".to_string(), Value::String(model.to_string()));
    }
    extra.insert("name".to_string(), Value::String(tc.name.clone()));
    if let Some(args) = &tc.arguments {
        extra.insert("arguments".to_string(), args.clone());
    }
    if let Some(cid) = pair_id {
        extra.insert("tool_call_id".to_string(), Value::String(cid.to_string()));
    } else {
        extra.insert("unpaired".to_string(), Value::Bool(true));
    }

    let inv = NormalizedInvocation {
        kind: "tool".to_string(),
        name: tc.name.clone(),
        raw_name: None,
        call_id: pair_id.map(String::from),
        arguments: tc.arguments.clone(),
    };

    build_message_with_raw_role(
        rec,
        &MessageMeta {
            role: "tool_call",
            raw_role: "assistant",
            author: Some(tc.name.as_str()),
            created_at,
        },
        render_tool_call_content(&tc.name, tc.arguments.as_ref()),
        extra,
        vec![inv],
    )
}

fn attach_planner_cass_usage(
    step_messages: &mut [NormalizedMessage],
    rec: &Value,
    step_model: Option<&str>,
    tool_call_count: usize,
) {
    if step_messages.is_empty() {
        return;
    }
    let input_tokens = rec.get("input_tokens").and_then(Value::as_i64);
    let output_tokens = rec.get("output_tokens").and_then(Value::as_i64);
    let cache_read_tokens = rec.get("cache_read_tokens").and_then(Value::as_i64);
    let has_api_tokens =
        input_tokens.is_some() || output_tokens.is_some() || cache_read_tokens.is_some();

    let mut cass = Map::new();
    if let Some(model) = step_model {
        cass.insert("model".to_string(), Value::String(model.to_string()));
    }
    if has_api_tokens {
        let mut usage = Map::new();
        if let Some(v) = input_tokens {
            usage.insert("input_tokens".to_string(), Value::from(v));
        }
        if let Some(v) = output_tokens {
            usage.insert("output_tokens".to_string(), Value::from(v));
        }
        if let Some(v) = cache_read_tokens {
            usage.insert("cache_read_tokens".to_string(), Value::from(v));
        }
        usage.insert("data_source".to_string(), Value::String("api".to_string()));
        cass.insert("token_usage".to_string(), Value::Object(usage));
    }
    if tool_call_count > 0 {
        cass.insert("tool_call_count".to_string(), Value::from(tool_call_count));
    }
    if cass.is_empty() {
        return;
    }

    let target_idx = step_messages
        .iter()
        .position(|m| m.role == "assistant")
        .unwrap_or(0);
    if let Value::Object(map) = &mut step_messages[target_idx].extra {
        map.insert("cass".to_string(), Value::Object(cass));
    }
}

fn emit_planner_response(
    rec: &Value,
    pos: usize,
    created: Option<i64>,
    plan: &TurnPairingPlan,
    out: &mut Vec<NormalizedMessage>,
) {
    let step_model = rec
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let thinking = rec
        .get("thinking")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let content = rec.get("content").and_then(Value::as_str).unwrap_or("");
    let planned_calls = plan.planner_calls.get(&pos).map_or(&[][..], Vec::as_slice);

    let mut step_messages = Vec::new();
    let base_extra = || {
        let mut extra = Map::new();
        extra.insert(
            "agy_type".to_string(),
            Value::String("PLANNER_RESPONSE".to_string()),
        );
        insert_step_index(&mut extra, rec);
        if let Some(model) = step_model {
            extra.insert("model".to_string(), Value::String(model.to_string()));
        }
        extra
    };

    if let Some(t) = thinking
        && let Some(msg) = build_message_with_raw_role(
            rec,
            &MessageMeta {
                role: "reasoning",
                raw_role: "assistant",
                author: step_model,
                created_at: created,
            },
            t.to_string(),
            base_extra(),
            Vec::new(),
        )
    {
        step_messages.push(msg);
    }

    if !content.trim().is_empty()
        && let Some(msg) = build_message_with_raw_role(
            rec,
            &MessageMeta {
                role: "assistant",
                raw_role: "assistant",
                author: step_model,
                created_at: created,
            },
            content.to_string(),
            base_extra(),
            Vec::new(),
        )
    {
        step_messages.push(msg);
    }

    for (tc, pair_id) in planned_calls {
        if let Some(msg) = build_tool_call_message(
            rec,
            "PLANNER_RESPONSE",
            step_model,
            created,
            tc,
            pair_id.as_deref(),
        ) {
            step_messages.push(msg);
        }
    }

    attach_planner_cass_usage(&mut step_messages, rec, step_model, planned_calls.len());
    out.extend(step_messages);
}

fn emit_generic_tool_result(
    rec: &Value,
    pos: usize,
    created: Option<i64>,
    plan: &TurnPairingPlan,
    out: &mut Vec<NormalizedMessage>,
) {
    let matched = plan.generic_matches.get(&pos);
    let content = rec.get("content").and_then(Value::as_str).unwrap_or("");
    let error_text = extract_error_text(rec);
    let media = rec.get("media").filter(|v| !v.is_null());
    let body = if content.trim().is_empty() {
        error_text.unwrap_or_default()
    } else {
        content.to_string()
    };

    if body.trim().is_empty() && media.is_none() && matched.is_none() {
        return;
    }

    let author = matched.map_or("generic", |m| m.tool_name.as_str());
    let mut extra = Map::new();
    extra.insert("agy_type".to_string(), Value::String("GENERIC".to_string()));
    insert_step_index(&mut extra, rec);
    if let Some(err_val) = rec.get("error").filter(|v| !v.is_null()) {
        extra.insert("error".to_string(), err_val.clone());
    }
    if let Some(media_val) = media {
        extra.insert("media".to_string(), media_val.clone());
    }

    let Some(mut msg) = build_message_with_raw_role(
        rec,
        &MessageMeta {
            role: "tool_result",
            raw_role: "tool",
            author: Some(author),
            created_at: created,
        },
        body,
        extra,
        Vec::new(),
    ) else {
        return;
    };

    if let Err(err) = set_tool_result_pairing(&mut msg.extra, matched.map(|m| m.call_id.as_str())) {
        tracing::warn!(error = %err, "antigravity: failed to set GENERIC tool_result pairing");
        return;
    }
    out.push(msg);
}

fn push_legacy_tool_result(
    rec: &Value,
    step_type: &str,
    author: &str,
    created: Option<i64>,
    body: String,
    pair_id: Option<&str>,
    out: &mut Vec<NormalizedMessage>,
) {
    let thinking = rec
        .get("thinking")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let mut extra = Map::new();
    extra.insert("agy_type".to_string(), Value::String(step_type.to_string()));
    insert_step_index(&mut extra, rec);
    if let Some(t) = thinking {
        extra.insert("thinking".to_string(), Value::String(t.to_string()));
    }
    if let Some(mut msg) = build_message_with_raw_role(
        rec,
        &MessageMeta {
            role: "tool_result",
            raw_role: "tool",
            author: Some(author),
            created_at: created,
        },
        body,
        extra,
        Vec::new(),
    ) && set_tool_result_pairing(&mut msg.extra, pair_id).is_ok()
    {
        out.push(msg);
    }
}

fn emit_legacy_model_tool_step(
    rec: &Value,
    pos: usize,
    step_type: &str,
    created: Option<i64>,
    out: &mut Vec<NormalizedMessage>,
) {
    let content = rec.get("content").and_then(Value::as_str).unwrap_or("");
    let error_text = extract_error_text(rec);
    let body = merge_content_and_error(content, error_text.as_deref());
    let has_thinking = rec
        .get("thinking")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim().is_empty());
    let fallback_name = if step_type.trim().is_empty() {
        "unknown".to_string()
    } else {
        step_type.trim().to_ascii_lowercase()
    };

    if let Some(raw_calls) = rec
        .get("tool_calls")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
    {
        let calls = parse_raw_tool_calls(raw_calls, &fallback_name);
        if calls.len() == 1 {
            let tc = &calls[0];
            let step_label = step_index_label(rec, pos);
            let pair_id = tc
                .call_id
                .clone()
                .unwrap_or_else(|| format!("agy:step{step_label}:call0"));
            if let Some(call_msg) =
                build_tool_call_message(rec, step_type, None, created, tc, Some(pair_id.as_str()))
            {
                out.push(call_msg);
            }
            push_legacy_tool_result(
                rec,
                step_type,
                tc.name.as_str(),
                created,
                body,
                Some(pair_id.as_str()),
                out,
            );
        } else {
            for tc in &calls {
                if let Some(call_msg) =
                    build_tool_call_message(rec, step_type, None, created, tc, None)
                {
                    out.push(call_msg);
                }
            }
            push_legacy_tool_result(
                rec,
                step_type,
                fallback_name.as_str(),
                created,
                body,
                None,
                out,
            );
        }
        return;
    }

    if body.trim().is_empty() && !has_thinking {
        return;
    }
    push_legacy_tool_result(
        rec,
        step_type,
        fallback_name.as_str(),
        created,
        body,
        None,
        out,
    );
}

fn emit_user_input(
    rec: &Value,
    content: &str,
    created: Option<i64>,
    out: &mut Vec<NormalizedMessage>,
) {
    let (body, settings, metadata) = extract_user_request(content);
    if body.trim().is_empty() {
        return;
    }
    let mut extra = Map::new();
    extra.insert(
        "agy_type".to_string(),
        Value::String("USER_INPUT".to_string()),
    );
    insert_step_index(&mut extra, rec);
    if let Some(s) = settings {
        extra.insert("settings_change".to_string(), Value::String(s));
    }
    if let Some(m) = metadata {
        extra.insert("additional_metadata".to_string(), Value::String(m));
    }
    if let Some(msg) = build_message_with_raw_role(
        rec,
        &MessageMeta {
            role: "user",
            raw_role: "user",
            author: None,
            created_at: created,
        },
        body,
        extra,
        Vec::new(),
    ) {
        out.push(msg);
    }
}

fn emit_system_message(
    rec: &Value,
    step_type: &str,
    source: &str,
    author: Option<&str>,
    created: Option<i64>,
    body: String,
    out: &mut Vec<NormalizedMessage>,
) {
    if body.trim().is_empty() {
        return;
    }
    let mut extra = Map::new();
    extra.insert("agy_type".to_string(), Value::String(step_type.to_string()));
    insert_step_index(&mut extra, rec);
    if step_type == "ERROR_MESSAGE"
        && let Some(err_val) = rec.get("error").filter(|v| !v.is_null())
    {
        extra.insert("error".to_string(), err_val.clone());
    }
    if !source.is_empty() {
        extra.insert("agy_source".to_string(), Value::String(source.to_string()));
    }
    if let Some(msg) = build_message_with_raw_role(
        rec,
        &MessageMeta {
            role: "system",
            raw_role: "system",
            author,
            created_at: created,
        },
        body,
        extra,
        Vec::new(),
    ) {
        out.push(msg);
    }
}

fn emit_record_messages(
    rec: &Value,
    pos: usize,
    plan: &TurnPairingPlan,
    out: &mut Vec<NormalizedMessage>,
) {
    let source = rec.get("source").and_then(Value::as_str).unwrap_or("");
    let step_type = rec.get("type").and_then(Value::as_str).unwrap_or("");
    let created = rec.get("created_at").and_then(parse_timestamp);
    let content = rec.get("content").and_then(Value::as_str).unwrap_or("");

    match step_type {
        "USER_INPUT" => emit_user_input(rec, content, created, out),
        "PLANNER_RESPONSE" => emit_planner_response(rec, pos, created, plan, out),
        "GENERIC" if source == "MODEL" => emit_generic_tool_result(rec, pos, created, plan, out),
        "CONVERSATION_HISTORY" => {}
        "ERROR_MESSAGE" => {
            let error_text = extract_error_text(rec);
            let body = merge_content_and_error(content, error_text.as_deref());
            emit_system_message(rec, step_type, source, Some("error"), created, body, out);
        }
        "CHECKPOINT" => {
            emit_system_message(
                rec,
                step_type,
                source,
                Some("checkpoint"),
                created,
                content.to_string(),
                out,
            );
        }
        "EPHEMERAL_MESSAGE" | "SYSTEM_MESSAGE" => {
            if is_boilerplate_system_message(content) {
                return;
            }
            let author = if step_type == "EPHEMERAL_MESSAGE" {
                "ephemeral"
            } else {
                "system"
            };
            emit_system_message(
                rec,
                step_type,
                source,
                Some(author),
                created,
                content.to_string(),
                out,
            );
        }
        _ if source == "MODEL" => emit_legacy_model_tool_step(rec, pos, step_type, created, out),
        _ => {
            let error_text = extract_error_text(rec);
            let body = merge_content_and_error(content, error_text.as_deref());
            emit_system_message(rec, step_type, source, None, created, body, out);
        }
    }
}

fn parse_transcript_file(
    transcript: &Path,
    conversation_dir: &Path,
    summary_row: Option<ConversationSummaryRow>,
) -> Option<NormalizedConversation> {
    let records = AntigravityConnector::read_transcript_records(transcript);
    if records.is_empty() {
        return None;
    }

    let mut ordered: Vec<&Value> = records.iter().collect();
    ordered.sort_by_key(|r| {
        r.get("step_index")
            .and_then(Value::as_i64)
            .unwrap_or(i64::MAX)
    });

    let external_id = AntigravityConnector::conversation_uuid(conversation_dir);
    let model = detect_model(&ordered);
    let pairing_plan = compute_turn_pairings(&ordered);

    let mut messages: Vec<NormalizedMessage> = Vec::new();
    let mut started_at: Option<i64> = None;
    let mut ended_at: Option<i64> = None;

    for (pos, rec) in ordered.iter().enumerate() {
        if let Some(ts) = rec.get("created_at").and_then(parse_timestamp) {
            started_at = Some(started_at.map_or(ts, |cur| cur.min(ts)));
            ended_at = Some(ended_at.map_or(ts, |cur| cur.max(ts)));
        }
        emit_record_messages(rec, pos, &pairing_plan, &mut messages);
    }

    if messages.is_empty() {
        tracing::warn!(
            transcript = %transcript.display(),
            records = records.len(),
            "antigravity: transcript produced no messages (possible agy schema drift)"
        );
        return None;
    }

    reindex_messages(&mut messages);

    let mut title = messages
        .iter()
        .find(|m| m.role == "user")
        .or_else(|| messages.first())
        .and_then(|m| m.content.lines().find(|l| !l.trim().is_empty()))
        .map(|line| line.chars().take(100).collect::<String>());

    let mut metadata = Map::new();
    metadata.insert("source".to_string(), Value::String(AGENT_SLUG.to_string()));
    if let Some(model) = model {
        metadata.insert("model".to_string(), Value::String(model));
    }

    let mut workspace: Option<PathBuf> = None;
    if let Some(summary) = summary_row {
        if let Some(ws) = summary.workspace {
            workspace = Some(ws);
        }
        if let Some(db_title) = summary.title {
            title = Some(db_title);
            metadata.insert(
                "title_source".to_string(),
                Value::String("summary_db".to_string()),
            );
        }
        if let Some(parent_id) = summary.parent_conversation_id {
            metadata.insert(
                "parent_conversation_id".to_string(),
                Value::String(parent_id),
            );
        }
        if let Some(agent_name) = summary.agent_name {
            metadata.insert("agent_name".to_string(), Value::String(agent_name));
        }
        if let Some(depth) = summary.nesting_depth {
            metadata.insert("nesting_depth".to_string(), Value::from(depth));
        }
    }

    Some(NormalizedConversation {
        agent_slug: AGENT_SLUG.to_string(),
        external_id,
        title,
        workspace,
        source_path: transcript.to_path_buf(),
        started_at,
        ended_at,
        metadata: Value::Object(metadata),
        messages,
    })
}

fn scan_antigravity_with_callback_with_exclusions(
    ctx: &ScanContext,
    on_conversation: &mut dyn FnMut(NormalizedConversation) -> Result<()>,
    excluded_paths: &[PathBuf],
) -> Result<()> {
    let roots = AntigravityConnector::source_roots(ctx);
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut db_cache = SummaryDbCache::default();

    for root in roots {
        if !root.path.exists() {
            continue;
        }
        if root.path.is_file() {
            if path_is_excluded(&root.path, excluded_paths) {
                continue;
            }
            let Some(conv_dir) =
                AntigravityConnector::conversation_dir_for_transcript_file(&root.path)
            else {
                continue;
            };
            if path_is_excluded(&conv_dir, excluded_paths) {
                continue;
            }
            if !seen.insert(dedupe_path_key(&root.path)) {
                continue;
            }
            if !file_modified_since(&root.path, ctx.since_ts) {
                continue;
            }
            let summary = AntigravityConnector::conversation_uuid(&conv_dir)
                .and_then(|uuid| SummaryDbCache::lookup_single(&conv_dir, &uuid));
            if let Some(conversation) = parse_transcript_file(&root.path, &conv_dir, summary) {
                on_conversation(conversation).with_context(|| {
                    format!("emit antigravity conversation {}", root.path.display())
                })?;
            }
            continue;
        }

        for conv_dir in AntigravityConnector::conversation_dirs(&root.path) {
            let Some(transcript) = AntigravityConnector::primary_transcript_path(&conv_dir) else {
                continue;
            };
            if path_is_excluded(&conv_dir, excluded_paths)
                || path_is_excluded(&transcript, excluded_paths)
            {
                continue;
            }
            if !seen.insert(dedupe_path_key(&transcript)) {
                continue;
            }
            if !AntigravityConnector::conversation_modified_since(&conv_dir, ctx.since_ts) {
                continue;
            }
            let summary = AntigravityConnector::conversation_uuid(&conv_dir)
                .and_then(|uuid| db_cache.lookup_bulk(&conv_dir, &uuid));
            if let Some(conversation) = parse_transcript_file(&transcript, &conv_dir, summary) {
                on_conversation(conversation).with_context(|| {
                    format!("emit antigravity conversation {}", conv_dir.display())
                })?;
            }
        }
    }

    Ok(())
}

fn discover_sources_with_exclusions(
    ctx: &ScanContext,
    excluded_paths: &[PathBuf],
) -> Vec<DiscoveredSourceFile> {
    let roots = AntigravityConnector::source_roots(ctx);
    let mut out = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    for root in roots {
        if !root.path.exists() {
            continue;
        }
        if root.path.is_file() {
            if path_is_excluded(&root.path, excluded_paths) {
                continue;
            }
            let Some(conv_dir) =
                AntigravityConnector::conversation_dir_for_transcript_file(&root.path)
            else {
                continue;
            };
            if path_is_excluded(&conv_dir, excluded_paths) {
                continue;
            }
            if !file_modified_since(&root.path, ctx.since_ts) {
                continue;
            }
            if seen.insert(dedupe_path_key(&root.path)) {
                out.push(
                    DiscoveredSourceFile::new(
                        AGENT_SLUG,
                        &root,
                        root.path.clone(),
                        DiscoveredSourceRole::PrimarySessionLog,
                        true,
                    )
                    .with_fs_metadata(),
                );
            }
            continue;
        }

        for conv_dir in AntigravityConnector::conversation_dirs(&root.path) {
            let Some(transcript) = AntigravityConnector::primary_transcript_path(&conv_dir) else {
                continue;
            };
            if path_is_excluded(&conv_dir, excluded_paths)
                || path_is_excluded(&transcript, excluded_paths)
            {
                continue;
            }
            if !AntigravityConnector::conversation_modified_since(&conv_dir, ctx.since_ts) {
                continue;
            }

            if seen.insert(dedupe_path_key(&transcript)) {
                out.push(
                    DiscoveredSourceFile::new(
                        AGENT_SLUG,
                        &root,
                        transcript,
                        DiscoveredSourceRole::PrimarySessionLog,
                        true,
                    )
                    .with_fs_metadata(),
                );
            }

            // Only emit sibling conversations/<uuid>.db when it lives inside the active scan root.
            if let Some(db) = AntigravityConnector::sibling_db_path(&conv_dir)
                && db.starts_with(&root.path)
                && !path_is_excluded(&db, excluded_paths)
                && seen.insert(dedupe_path_key(&db))
            {
                out.push(
                    DiscoveredSourceFile::new(
                        AGENT_SLUG,
                        &root,
                        db,
                        DiscoveredSourceRole::SqliteDatabase,
                        false,
                    )
                    .with_fs_metadata(),
                );
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// agy version-compatibility guard (bd-47kjh.2.5)
// ---------------------------------------------------------------------------

/// The `agy` versions whose transcript schema this reader has been validated against.
pub const AGY_TESTED_VERSIONS: &[&str] = &["1.0.7", "1.5.14"];

/// Best-effort `agy --version`, parsed to a bare dotted-numeric string (e.g.
/// `1.5.14`). Returns `None` if agy is absent or the output is unparseable.
#[must_use]
pub fn installed_agy_version() -> Option<String> {
    let output = std::process::Command::new("agy")
        .arg("--version")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_agy_version(&text)
}

/// Pull the first dotted-numeric version token out of `agy --version` text such
/// as `agy version 1.5.14` or `v1.0.7`.
fn parse_agy_version(text: &str) -> Option<String> {
    text.split_whitespace()
        .map(|tok| tok.trim_start_matches('v'))
        .find(|t| {
            let parts: Vec<&str> = t.split('.').collect();
            parts.len() >= 2
                && parts
                    .iter()
                    .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        })
        .map(String::from)
}

/// Whether `version` is one this reader was validated against.
#[must_use]
pub fn agy_version_is_tested(version: &str) -> bool {
    let v = version.trim().trim_start_matches('v');
    AGY_TESTED_VERSIONS.contains(&v)
}

/// CI/diagnostic signal for agy version drift.
pub fn warn_if_untested_agy_version() -> Option<String> {
    let version = installed_agy_version()?;
    if !agy_version_is_tested(&version) {
        tracing::warn!(
            installed = %version,
            tested = ?AGY_TESTED_VERSIONS,
            "antigravity: installed agy version is outside the tested-compatible set; \
             re-validate the transcript reader (bd-47kjh.2.5)"
        );
    }
    Some(version)
}

impl Connector for AntigravityConnector {
    fn detect(&self) -> DetectionResult {
        franken_detection_for_connector(AGENT_SLUG).unwrap_or_else(DetectionResult::not_found)
    }

    fn scan(&self, ctx: &ScanContext) -> Result<Vec<NormalizedConversation>> {
        let mut convs = Vec::new();
        self.scan_with_callback(ctx, &mut |conv| {
            convs.push(conv);
            Ok(())
        })?;
        Ok(convs)
    }

    fn supports_streaming_scan(&self) -> bool {
        true
    }

    fn discover_source_files(&self, ctx: &ScanContext) -> Result<Vec<DiscoveredSourceFile>> {
        Ok(discover_sources_with_exclusions(
            ctx,
            &excluded_scan_paths_from_env(),
        ))
    }

    fn scan_with_callback(
        &self,
        ctx: &ScanContext,
        on_conversation: &mut dyn FnMut(NormalizedConversation) -> Result<()>,
    ) -> Result<()> {
        scan_antigravity_with_callback_with_exclusions(
            ctx,
            on_conversation,
            &excluded_scan_paths_from_env(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::assert_discovery_covers_scan_sources;
    use crate::connectors::token_extraction::{TokenDataSource, extract_tokens_for_agent};
    use serde_json::json;
    use std::fs;
    use std::io::{self, Cursor, Read};
    use tempfile::TempDir;

    const FIXTURE_UUID: &str = "f1e2d3c4-b5a6-4789-9abc-def012345678";

    /// The `antigravity-cli`-equivalent base directory of the checked-in 1.0.7 fixture.
    fn fixture_base() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("antigravity")
    }

    fn fixture_conversation_dir() -> PathBuf {
        fixture_base().join("brain").join(FIXTURE_UUID)
    }

    fn fixture_ctx() -> ScanContext {
        let base = fixture_base();
        ScanContext::with_roots(base.clone(), vec![ScanRoot::local(base)], None)
    }

    fn only_conversation() -> NormalizedConversation {
        let convs = AntigravityConnector::new().scan(&fixture_ctx()).unwrap();
        assert_eq!(convs.len(), 1, "fixture is a single conversation");
        convs.into_iter().next().unwrap()
    }

    fn write_jsonl_records(path: &Path, records: &[Value]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut text = records
            .iter()
            .map(|r| serde_json::to_string(r).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        text.push('\n');
        fs::write(path, text).unwrap();
    }

    fn make_conv_dir(base: &Path, uuid: &str) -> PathBuf {
        let dir = base.join("brain").join(uuid);
        fs::create_dir_all(dir.join(".system_generated").join("logs")).unwrap();
        dir
    }

    #[test]
    fn new_creates_connector() {
        let _ = AntigravityConnector::new();
        let _ = AntigravityConnector;
    }

    #[test]
    fn is_conversation_dir_recognizes_full_or_compact_transcript() {
        let dir = TempDir::new().unwrap();
        assert!(!AntigravityConnector::is_conversation_dir(dir.path()));
        let logs = dir.path().join(".system_generated").join("logs");
        fs::create_dir_all(&logs).unwrap();
        fs::write(logs.join("transcript_full.jsonl"), "{}\n").unwrap();
        assert!(AntigravityConnector::is_conversation_dir(dir.path()));
        assert_eq!(
            AntigravityConnector::primary_transcript_path(dir.path()),
            Some(logs.join("transcript_full.jsonl"))
        );

        fs::remove_file(logs.join("transcript_full.jsonl")).unwrap();
        fs::write(logs.join("transcript.jsonl"), "{}\n").unwrap();
        assert!(AntigravityConnector::is_conversation_dir(dir.path()));
        assert_eq!(
            AntigravityConnector::primary_transcript_path(dir.path()),
            Some(logs.join("transcript.jsonl"))
        );
    }

    #[test]
    fn extract_tagged_blocks_handles_multiple_and_missing() {
        assert_eq!(
            extract_tagged_blocks("<A>one</A> mid <A>two</A>", "A"),
            vec!["one".to_string(), "two".to_string()]
        );
        assert!(extract_tagged_blocks("no tags here", "A").is_empty());
        assert!(extract_tagged_blocks("<A>oops", "A").is_empty());
    }

    #[test]
    fn extract_user_request_unwraps_and_keeps_settings() {
        let raw = "<USER_REQUEST>\ndo the thing\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\ntime\n</ADDITIONAL_METADATA>\n<USER_SETTINGS_CHANGE>\nchanged to Gemini 3.1 Pro (High). ok\n</USER_SETTINGS_CHANGE>";
        let (body, settings, metadata) = extract_user_request(raw);
        assert_eq!(body, "do the thing");
        assert!(settings.unwrap().contains("Gemini 3.1 Pro (High)"));
        assert_eq!(metadata.as_deref(), Some("time"));
    }

    #[test]
    fn extract_user_request_falls_back_to_raw_when_unwrapped() {
        let (body, settings, metadata) = extract_user_request("plain prompt with no tags");
        assert_eq!(body, "plain prompt with no tags");
        assert!(settings.is_none() && metadata.is_none());
    }

    #[test]
    fn model_from_settings_extracts_human_name() {
        assert_eq!(
            model_from_settings(
                "The user changed setting `Model Selection` from None to Gemini 3.1 Pro (High). No need to comment."
            )
            .as_deref(),
            Some("Gemini 3.1 Pro (High)")
        );
        assert_eq!(model_from_settings("changed from X to None. done"), None);
        assert_eq!(
            model_from_settings(
                "changed setting `Auto to Stop` from None to Gemini 3.1 Pro (High). ok"
            )
            .as_deref(),
            Some("Gemini 3.1 Pro (High)")
        );
        assert_eq!(
            model_from_settings("set model to Gemini 3.1 Pro (High). thanks").as_deref(),
            Some("Gemini 3.1 Pro (High)")
        );
    }

    #[test]
    fn scan_parses_legacy_107_fixture_into_canonical_six_roles() {
        let conv = only_conversation();
        assert_eq!(conv.agent_slug, "antigravity");
        assert_eq!(conv.external_id.as_deref(), Some(FIXTURE_UUID));
        assert_eq!(conv.metadata["model"], "Gemini 3.1 Pro (High)");
        assert_eq!(conv.metadata["source"], "antigravity");
        assert!(conv.title.as_deref().unwrap().starts_with("Read data.txt"));
        assert!(conv.started_at.unwrap() <= conv.ended_at.unwrap());
        // Fixture contains both transcript_full.jsonl and transcript.jsonl; full is preferred.
        assert_eq!(
            conv.source_path,
            AntigravityConnector::transcript_full_path(&fixture_conversation_dir())
        );

        assert_eq!(conv.messages.len(), 12);
        for (i, m) in conv.messages.iter().enumerate() {
            assert_eq!(m.idx, i64::try_from(i).unwrap());
            assert!(
                m.extra
                    .get("raw_role")
                    .and_then(Value::as_str)
                    .is_some_and(|r| !r.is_empty()),
                "every message must carry non-empty raw_role"
            );
            assert!(
                matches!(
                    m.role.as_str(),
                    "user" | "assistant" | "reasoning" | "tool_call" | "tool_result" | "system"
                ),
                "unexpected role: {}",
                m.role
            );
        }
        assert!(conv.messages.iter().any(|m| m.role == "user"));
        assert!(conv.messages.iter().any(|m| m.role == "assistant"));
        assert!(conv.messages.iter().any(|m| m.role == "reasoning"));
        assert!(conv.messages.iter().any(|m| m.role == "tool_call"));
        assert!(conv.messages.iter().any(|m| m.role == "tool_result"));
        assert!(conv.messages.iter().any(|m| m.role == "system"));
    }

    #[test]
    fn user_message_unwraps_request_and_keeps_settings_extra() {
        let conv = only_conversation();
        let user = conv.messages.iter().find(|m| m.role == "user").unwrap();
        assert!(user.content.starts_with("Read data.txt"));
        assert!(!user.content.contains("USER_REQUEST"));
        assert!(
            user.extra["settings_change"]
                .as_str()
                .unwrap()
                .contains("Gemini 3.1 Pro (High)")
        );
        assert!(user.extra.get("additional_metadata").is_some());
        assert_eq!(user.extra["raw_role"], "user");
    }

    #[test]
    fn planner_response_splits_thinking_into_reasoning_role() {
        let conv = only_conversation();
        let reasoning = conv
            .messages
            .iter()
            .find(|m| m.role == "reasoning")
            .expect("reasoning message");
        assert!(reasoning.content.contains("Planning the steps"));
        assert_eq!(reasoning.extra["raw_role"], "assistant");
        let assistant = conv
            .messages
            .iter()
            .find(|m| m.role == "assistant" && m.content.contains("data.txt"))
            .expect("assistant message");
        assert_eq!(assistant.extra["raw_role"], "assistant");
    }

    #[test]
    fn legacy_view_file_becomes_unpaired_tool_result() {
        let conv = only_conversation();
        let view = conv
            .messages
            .iter()
            .find(|m| m.extra.get("agy_type").and_then(Value::as_str) == Some("VIEW_FILE"))
            .expect("VIEW_FILE message");
        assert_eq!(view.role, "tool_result");
        assert_eq!(view.author.as_deref(), Some("view_file"));
        assert_eq!(view.extra["raw_role"], "tool");
        assert_eq!(view.extra["unpaired"], true);
        assert!(view.content.contains("1234"));
    }

    #[test]
    fn legacy_run_command_and_edit_file_emit_paired_tool_call_and_result() {
        let conv = only_conversation();
        let run_call = conv
            .messages
            .iter()
            .find(|m| m.role == "tool_call" && m.author.as_deref() == Some("run_command"))
            .expect("run_command tool_call");
        assert_eq!(
            run_call.invocations[0].call_id.as_deref(),
            Some("call_run_0001")
        );
        assert_eq!(
            run_call.invocations[0].arguments.as_ref().unwrap()["command"],
            "echo HELLO_FROM_AGY"
        );
        let run_res = conv
            .messages
            .iter()
            .find(|m| {
                m.role == "tool_result"
                    && m.extra.get("tool_call_id").and_then(Value::as_str) == Some("call_run_0001")
            })
            .expect("run_command tool_result");
        assert!(run_res.content.contains("HELLO_FROM_AGY"));

        let edit_call = conv
            .messages
            .iter()
            .find(|m| m.role == "tool_call" && m.author.as_deref() == Some("edit_file"))
            .expect("edit_file tool_call");
        assert_eq!(
            edit_call.invocations[0].call_id.as_deref(),
            Some("call_edit_0002")
        );
        assert_eq!(
            edit_call.invocations[0].arguments.as_ref().unwrap()["path"],
            "out.txt"
        );
        assert_eq!(
            edit_call.invocations[0].arguments.as_ref().unwrap()["content"],
            "DONE"
        );
    }

    #[test]
    fn ephemeral_and_non_boilerplate_system_messages_preserved() {
        let conv = only_conversation();
        let ephemeral = conv
            .messages
            .iter()
            .find(|m| m.extra.get("agy_type").and_then(Value::as_str) == Some("EPHEMERAL_MESSAGE"))
            .expect("ephemeral message");
        assert_eq!(ephemeral.role, "system");
        assert_eq!(ephemeral.author.as_deref(), Some("ephemeral"));

        let system = conv
            .messages
            .iter()
            .find(|m| m.extra.get("agy_type").and_then(Value::as_str) == Some("SYSTEM_MESSAGE"))
            .expect("system message");
        assert_eq!(system.role, "system");
        assert_eq!(system.author.as_deref(), Some("system"));
    }

    #[test]
    fn unknown_model_step_type_is_preserved_not_dropped() {
        let conv = only_conversation();
        let unknown = conv
            .messages
            .iter()
            .find(|m| m.extra.get("agy_type").and_then(Value::as_str) == Some("BROWSER_PREVIEW"))
            .expect("unknown BROWSER_PREVIEW message preserved");
        assert_eq!(unknown.role, "tool_result");
        assert_eq!(unknown.author.as_deref(), Some("browser_preview"));
        assert_eq!(unknown.extra["unpaired"], true);
        assert!(unknown.content.contains("preview"));
    }

    #[test]
    fn discovery_lists_sole_primary_transcript_and_in_root_db() {
        let connector = AntigravityConnector::new();
        let discovered = connector.discover_source_files(&fixture_ctx()).unwrap();

        assert!(discovered.iter().all(|d| d.provider_slug == "antigravity"));
        let primaries: Vec<_> = discovered
            .iter()
            .filter(|d| d.role == DiscoveredSourceRole::PrimarySessionLog)
            .collect();
        assert_eq!(
            primaries.len(),
            1,
            "directory scan emits only the sole primary transcript"
        );
        assert!(primaries[0].source_path.ends_with("transcript_full.jsonl"));
        assert!(primaries[0].required_for_reconstruction);

        assert!(
            discovered
                .iter()
                .any(|d| d.role == DiscoveredSourceRole::SqliteDatabase
                    && d.source_path.extension().is_some_and(|e| e == "db")),
            "in-root conversations/<uuid>.db should be discovered"
        );
    }

    #[test]
    fn discovery_excludes_sibling_db_when_scanning_brain_subdir() {
        let brain_dir = fixture_base().join("brain");
        let ctx = ScanContext::with_roots(
            brain_dir.clone(),
            vec![ScanRoot::local(brain_dir.clone())],
            None,
        );
        let discovered = AntigravityConnector::new()
            .discover_source_files(&ctx)
            .unwrap();
        assert!(!discovered.is_empty());
        for item in &discovered {
            assert!(
                item.source_path.starts_with(&brain_dir),
                "discovered path {:?} escaped scan root {:?}",
                item.source_path,
                brain_dir
            );
            assert_ne!(
                item.role,
                DiscoveredSourceRole::SqliteDatabase,
                "sibling db outside brain/ root must not be emitted"
            );
        }
    }

    #[test]
    fn explicit_single_file_scan_root_is_100_percent_symmetric() {
        // Explicit transcript.jsonl when sibling transcript_full.jsonl exists.
        let compact = AntigravityConnector::transcript_path(&fixture_conversation_dir());
        assert!(AntigravityConnector::transcript_full_path(&fixture_conversation_dir()).is_file());
        let ctx = ScanContext::with_roots(
            compact.parent().unwrap().to_path_buf(),
            vec![ScanRoot::local(compact.clone())],
            None,
        );
        let connector = AntigravityConnector::new();
        let convs = connector.scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].source_path, compact);

        let discovered = connector.discover_source_files(&ctx).unwrap();
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].source_path, compact);
        assert_eq!(discovered[0].role, DiscoveredSourceRole::PrimarySessionLog);
        assert!(discovered[0].required_for_reconstruction);
        assert_discovery_covers_scan_sources(&connector, &ctx);

        // Explicit .db file yields empty in both scan and discover.
        let db = AntigravityConnector::sibling_db_path(&fixture_conversation_dir()).unwrap();
        let db_ctx = ScanContext::with_roots(
            db.parent().unwrap().to_path_buf(),
            vec![ScanRoot::local(db)],
            None,
        );
        assert!(connector.scan(&db_ctx).unwrap().is_empty());
        assert!(connector.discover_source_files(&db_ctx).unwrap().is_empty());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn agy_1514_six_roles_parallel_pairing_tokens_and_noise_filtering() {
        let tmp = TempDir::new().unwrap();
        let uuid = "11111111-2222-3333-4444-555555555555";
        let conv_dir = make_conv_dir(tmp.path(), uuid);
        let full = AntigravityConnector::transcript_full_path(&conv_dir);

        // Write a UTF-8 BOM on line 1 to verify BOM stripping.
        let records = vec![
            json!({
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-11-22T10:00:00Z",
                "content": "<USER_REQUEST>\nInspect project files\n</USER_REQUEST>"
            }),
            json!({
                "step_index": 1,
                "source": "SYSTEM_SDK",
                "type": "SYSTEM_MESSAGE",
                "created_at": "2026-11-22T10:00:01Z",
                "content": "The following system message was injected by a PreInvocation hook:\n<agent-contract-source id=\"global:USER.md\">boilerplate</agent-contract-source>"
            }),
            json!({
                "step_index": 2,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-11-22T10:00:02Z",
                "model": "gemini-4-neon-xhigh",
                "input_tokens": 1500,
                "cache_read_tokens": 900,
                "output_tokens": 250,
                "thinking": "Need to read Cargo.toml and run status.",
                "tool_calls": [
                    {"name": "read_file", "args": {"Path": "/tmp/demo/Cargo.toml"}},
                    {"name": "run_command", "args": {"CommandLine": "git status"}}
                ]
            }),
            // Interleaved boilerplate reminder between PLANNER_RESPONSE and GENERIC outputs.
            json!({
                "step_index": 3,
                "source": "SYSTEM",
                "type": "SYSTEM_MESSAGE",
                "created_at": "2026-11-22T10:00:03Z",
                "content": "The following is a <SYSTEM_MESSAGE> not actually sent by the user.\n<SYSTEM_MESSAGE>\nReminder: work efficiently. Batch tool calls.\n</SYSTEM_MESSAGE>"
            }),
            json!({
                "step_index": 4,
                "source": "MODEL",
                "type": "GENERIC",
                "created_at": "2026-11-22T10:00:04Z",
                "content": "Created At: 2026-11-22T10:00:03Z\nCompleted At: 2026-11-22T10:00:04Z\n[package]\nname = \"demo\"\n"
            }),
            json!({
                "step_index": 5,
                "source": "MODEL",
                "type": "GENERIC",
                "created_at": "2026-11-22T10:00:05Z",
                "content": "Created At: 2026-11-22T10:00:04Z\nCompleted At: 2026-11-22T10:00:05Z\nCommand exited with code 0.\n\nCommand output:\nOn branch main\n"
            }),
            // Meaningful subagent notification must be retained.
            json!({
                "step_index": 6,
                "source": "SYSTEM",
                "type": "SYSTEM_MESSAGE",
                "created_at": "2026-11-22T10:00:06Z",
                "content": "[Subagent completed] Codebase Researcher returned 3 findings."
            }),
            // Checkpoint must be retained.
            json!({
                "step_index": 7,
                "source": "SYSTEM",
                "type": "CHECKPOINT",
                "created_at": "2026-11-22T10:00:07Z",
                "content": "# Resuming from a compaction\nSummary preserved."
            }),
            // ERROR_MESSAGE carrying only `error` (no `content`) must be retained.
            json!({
                "step_index": 8,
                "source": "SYSTEM_SDK",
                "type": "ERROR_MESSAGE",
                "created_at": "2026-11-22T10:00:08Z",
                "error": "rate limit exceeded"
            }),
            // Final assistant reply with tokens.
            json!({
                "step_index": 9,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-11-22T10:00:09Z",
                "model": "gemini-4-neon-xhigh",
                "input_tokens": 2000,
                "output_tokens": 100,
                "content": "Everything looks clean."
            }),
        ];
        let mut raw = String::from("\u{feff}");
        for rec in &records {
            raw.push_str(&serde_json::to_string(rec).unwrap());
            raw.push('\n');
        }
        fs::write(&full, raw).unwrap();

        let ctx = ScanContext::with_roots(
            tmp.path().to_path_buf(),
            vec![ScanRoot::local(tmp.path().to_path_buf())],
            None,
        );
        let convs = AntigravityConnector::new().scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);
        let conv = &convs[0];
        assert_eq!(conv.metadata["model"], "gemini-4-neon-xhigh");
        assert_eq!(conv.source_path, full);

        // Verify boilerplate templates were filtered out.
        let serialized = serde_json::to_string(&conv.messages).unwrap();
        assert!(!serialized.contains("<agent-contract-source"));
        assert!(!serialized.contains("Reminder: work efficiently."));

        // Verify meaningful system, checkpoint, and error_message survived.
        assert!(conv.messages.iter().any(|m| {
            m.role == "system"
                && m.content
                    .contains("Codebase Researcher returned 3 findings")
        }));
        assert!(
            conv.messages
                .iter()
                .any(|m| { m.role == "system" && m.author.as_deref() == Some("checkpoint") })
        );
        assert!(conv.messages.iter().any(|m| {
            m.role == "system"
                && m.author.as_deref() == Some("error")
                && m.content.contains("rate limit exceeded")
        }));

        // Verify 1-to-1 conservative parallel pairing across interleaved SYSTEM_MESSAGE.
        let tool_calls: Vec<_> = conv
            .messages
            .iter()
            .filter(|m| m.role == "tool_call")
            .collect();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].author.as_deref(), Some("read_file"));
        assert_eq!(
            tool_calls[0].invocations[0].call_id.as_deref(),
            Some("agy:step2:call0")
        );
        assert_eq!(tool_calls[1].author.as_deref(), Some("run_command"));
        assert_eq!(
            tool_calls[1].invocations[0].call_id.as_deref(),
            Some("agy:step2:call1")
        );

        let tool_results: Vec<_> = conv
            .messages
            .iter()
            .filter(|m| m.role == "tool_result")
            .collect();
        assert_eq!(tool_results.len(), 2);
        assert_eq!(tool_results[0].author.as_deref(), Some("read_file"));
        assert_eq!(tool_results[0].extra["tool_call_id"], "agy:step2:call0");
        assert!(tool_results[0].extra.get("unpaired").is_none());
        assert_eq!(tool_results[1].author.as_deref(), Some("run_command"));
        assert_eq!(tool_results[1].extra["tool_call_id"], "agy:step2:call1");

        // Verify token attachment: step 2 has reasoning + 2 tool_calls (no assistant prose),
        // so API tokens attach to exactly one message (reasoning), while all 3 carry extra["model"].
        let step2_msgs: Vec<_> = conv
            .messages
            .iter()
            .filter(|m| m.extra.get("step_index").and_then(Value::as_i64) == Some(2))
            .collect();
        assert_eq!(step2_msgs.len(), 3);
        let api_attached: Vec<_> = step2_msgs
            .iter()
            .filter(|m| m.extra.pointer("/cass/token_usage").is_some())
            .collect();
        assert_eq!(api_attached.len(), 1, "no double counting of API tokens");
        for m in &step2_msgs {
            assert_eq!(m.extra["model"], "gemini-4-neon-xhigh");
            let extracted = extract_tokens_for_agent("antigravity", &m.extra, &m.content, &m.role);
            assert_eq!(extracted.model_name.as_deref(), Some("gemini-4-neon-xhigh"));
            assert_eq!(extracted.provider.as_deref(), Some("google"));
        }
        let step2_api = extract_tokens_for_agent(
            "antigravity",
            &api_attached[0].extra,
            &api_attached[0].content,
            &api_attached[0].role,
        );
        assert_eq!(step2_api.data_source, TokenDataSource::Api);
        assert_eq!(step2_api.input_tokens, Some(1500));
        assert_eq!(step2_api.cache_read_tokens, Some(900));
        assert_eq!(step2_api.output_tokens, Some(250));
        assert_eq!(step2_api.tool_call_count, 2);
    }

    #[test]
    fn mismatched_turn_window_fails_closed_unpaired() {
        let tmp = TempDir::new().unwrap();
        let conv_dir = make_conv_dir(tmp.path(), "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee");
        let full = AntigravityConnector::transcript_full_path(&conv_dir);
        write_jsonl_records(
            &full,
            &[
                json!({
                    "step_index": 0,
                    "source": "USER_EXPLICIT",
                    "type": "USER_INPUT",
                    "created_at": "2026-11-22T10:00:00Z",
                    "content": "<USER_REQUEST>Run two tools</USER_REQUEST>"
                }),
                json!({
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "created_at": "2026-11-22T10:00:01Z",
                    "model": "gemini-4-carbon-high",
                    "tool_calls": [
                        {"name": "read_file", "args": {"Path": "a.txt"}},
                        {"name": "run_command", "args": {"CommandLine": "ls"}}
                    ]
                }),
                // Only 1 GENERIC result instead of 2 -> mismatch -> all unpaired.
                json!({
                    "step_index": 2,
                    "source": "MODEL",
                    "type": "GENERIC",
                    "created_at": "2026-11-22T10:00:02Z",
                    "content": "Created At: 2026-11-22T10:00:02Z\npartial output"
                }),
            ],
        );

        let ctx = ScanContext::with_roots(
            tmp.path().to_path_buf(),
            vec![ScanRoot::local(tmp.path().to_path_buf())],
            None,
        );
        let convs = AntigravityConnector::new().scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);
        for tc in convs[0].messages.iter().filter(|m| m.role == "tool_call") {
            assert!(tc.invocations[0].call_id.is_none());
            assert_eq!(tc.extra["unpaired"], true);
            assert!(tc.extra.get("tool_call_id").is_none());
        }
        let tr = convs[0]
            .messages
            .iter()
            .find(|m| m.role == "tool_result")
            .unwrap();
        assert_eq!(tr.author.as_deref(), Some("generic"));
        assert_eq!(tr.extra["unpaired"], true);
        assert!(tr.extra.get("tool_call_id").is_none());
    }

    #[test]
    fn since_ts_uses_sole_existing_primary_transcript_not_missing_sibling() {
        let tmp = TempDir::new().unwrap();
        let conv_dir = make_conv_dir(tmp.path(), "22222222-3333-4444-5555-666666666666");
        // Only write transcript.jsonl (no transcript_full.jsonl).
        // Old `file_modified_since(a) || file_modified_since(b)` returned true
        // because missing `b` caused `metadata(b)` error -> `map_or(true, ...)`.
        write_jsonl_records(
            &AntigravityConnector::transcript_path(&conv_dir),
            &[json!({
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-11-22T10:00:00Z",
                "content": "<USER_REQUEST>only compact</USER_REQUEST>"
            })],
        );

        let ctx = ScanContext::with_roots(
            tmp.path().to_path_buf(),
            vec![ScanRoot::local(tmp.path().to_path_buf())],
            Some(i64::MAX),
        );
        assert!(
            AntigravityConnector::new().scan(&ctx).unwrap().is_empty(),
            "future since_ts must skip single-file compact conversation"
        );
        assert!(
            AntigravityConnector::new()
                .discover_source_files(&ctx)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn mid_read_io_error_discards_partial_conversation() {
        struct FailAfterFirstLine {
            first: Cursor<Vec<u8>>,
            failed: bool,
        }
        impl Read for FailAfterFirstLine {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                let n = self.first.read(buf)?;
                if n > 0 {
                    return Ok(n);
                }
                if !self.failed {
                    self.failed = true;
                    return Err(io::Error::other("simulated disk read error"));
                }
                Ok(0)
            }
        }

        let reader = BufReader::new(FailAfterFirstLine {
            first: Cursor::new(
                b"{\"step_index\":0,\"source\":\"USER_EXPLICIT\",\"type\":\"USER_INPUT\",\"content\":\"ok\"}\n"
                    .to_vec(),
            ),
            failed: false,
        });
        let records = read_transcript_records_from_reader(reader, Path::new("synthetic.jsonl"));
        assert!(
            records.is_empty(),
            "mid-read I/O error must discard the whole transcript"
        );
    }

    #[test]
    fn cass_exclude_paths_skips_matching_conversations() {
        let tmp = TempDir::new().unwrap();
        let keep_dir = make_conv_dir(tmp.path(), "aaaa0000-0000-0000-0000-000000000001");
        let drop_dir = make_conv_dir(tmp.path(), "bbbb0000-0000-0000-0000-000000000002");
        let rec = [json!({
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-11-22T10:00:00Z",
            "content": "<USER_REQUEST>hello</USER_REQUEST>"
        })];
        write_jsonl_records(&AntigravityConnector::transcript_full_path(&keep_dir), &rec);
        write_jsonl_records(&AntigravityConnector::transcript_full_path(&drop_dir), &rec);

        let ctx = ScanContext::with_roots(
            tmp.path().to_path_buf(),
            vec![ScanRoot::local(tmp.path().to_path_buf())],
            None,
        );
        let excluded = vec![drop_dir];
        let mut convs = Vec::new();
        scan_antigravity_with_callback_with_exclusions(
            &ctx,
            &mut |c| {
                convs.push(c);
                Ok(())
            },
            &excluded,
        )
        .unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].external_id.as_deref(),
            Some("aaaa0000-0000-0000-0000-000000000001")
        );

        let discovered = discover_sources_with_exclusions(&ctx, &excluded);
        assert_eq!(discovered.len(), 1);
        assert!(discovered[0].source_path.starts_with(&keep_dir));
    }

    #[test]
    fn default_detection_scopes_hermetically_when_data_dir_has_brain() {
        let tmp = TempDir::new().unwrap();
        let conv_dir = make_conv_dir(tmp.path(), "cccc0000-0000-0000-0000-000000000003");
        write_jsonl_records(
            &AntigravityConnector::transcript_full_path(&conv_dir),
            &[json!({
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-11-22T10:00:00Z",
                "content": "<USER_REQUEST>hermetic test</USER_REQUEST>"
            })],
        );

        // Only run hermetic assertion when env override is unset.
        if env_path_nonempty("CASS_ANTIGRAVITY_DATA_ROOT").is_none() {
            let ctx = ScanContext::local_default(tmp.path().to_path_buf(), None);
            let convs = AntigravityConnector::new().scan(&ctx).unwrap();
            assert_eq!(convs.len(), 1);
            assert_eq!(
                convs[0].external_id.as_deref(),
                Some("cccc0000-0000-0000-0000-000000000003")
            );
        }
    }

    #[cfg(feature = "antigravity-db")]
    #[test]
    #[allow(clippy::too_many_lines)]
    fn summary_db_enriches_workspace_title_subagent_readonly_and_degrades_on_corruption() {
        let tmp = TempDir::new().unwrap();
        let main_uuid = "dddd0000-0000-0000-0000-000000000001";
        let sub_uuid = "dddd0000-0000-0000-0000-000000000002";
        let main_dir = make_conv_dir(tmp.path(), main_uuid);
        let sub_dir = make_conv_dir(tmp.path(), sub_uuid);

        let rec = [json!({
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-11-22T10:00:00Z",
            "content": "<USER_REQUEST>fallback prompt line</USER_REQUEST>"
        })];
        write_jsonl_records(&AntigravityConnector::transcript_full_path(&main_dir), &rec);
        write_jsonl_records(&AntigravityConnector::transcript_full_path(&sub_dir), &rec);

        let db_path = tmp.path().join("conversation_summaries.db");
        {
            let conn = frankensqlite::Connection::open(db_path.to_string_lossy().as_ref()).unwrap();
            conn.execute(
                "CREATE TABLE conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    workspace_uris TEXT NOT NULL DEFAULT '[]',
                    parent_conversation_id TEXT NOT NULL DEFAULT '',
                    agent_name TEXT NOT NULL DEFAULT '',
                    nesting_depth INTEGER NOT NULL DEFAULT 0
                )",
            )
            .unwrap();
            conn.execute(&format!(
                "INSERT INTO conversation_summaries VALUES (
                    '{main_uuid}',
                    'Real DB Title',
                    '[\"file:///run/user/1000/antigravity-overlays/zzz\",\"file:///tmp/my%20repo/app\"]',
                    '',
                    '',
                    0
                )"
            ))
            .unwrap();
            conn.execute(&format!(
                "INSERT INTO conversation_summaries VALUES (
                    '{sub_uuid}',
                    '',
                    '[\"file:///run/user/1000/antigravity-overlays/only\"]',
                    '{main_uuid}',
                    'Codebase Researcher',
                    1
                )"
            ))
            .unwrap();
            // Flush WAL into the main DB file so setup starts clean before read-only scan.
            let _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE);");
            let _ = conn.execute("PRAGMA journal_mode = DELETE;");
        }
        let _ = fs::remove_file(tmp.path().join("conversation_summaries.db-wal"));
        let _ = fs::remove_file(tmp.path().join("conversation_summaries.db-shm"));
        assert!(!tmp.path().join("conversation_summaries.db-wal").exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&db_path, fs::Permissions::from_mode(0o444)).unwrap();
            fs::set_permissions(tmp.path(), fs::Permissions::from_mode(0o555)).unwrap();
        }
        let mtime_before = fs::metadata(&db_path).unwrap().modified().unwrap();

        let ctx = ScanContext::with_roots(
            tmp.path().to_path_buf(),
            vec![ScanRoot::local(tmp.path().to_path_buf())],
            None,
        );
        let convs = AntigravityConnector::new().scan(&ctx).unwrap();
        assert_eq!(convs.len(), 2);

        let main_conv = convs
            .iter()
            .find(|c| c.external_id.as_deref() == Some(main_uuid))
            .unwrap();
        assert_eq!(main_conv.title.as_deref(), Some("Real DB Title"));
        assert_eq!(main_conv.metadata["title_source"], "summary_db");
        assert_eq!(
            main_conv.workspace.as_deref(),
            Some(Path::new("/tmp/my repo/app"))
        );
        assert!(main_conv.metadata.get("parent_conversation_id").is_none());

        let sub_conv = convs
            .iter()
            .find(|c| c.external_id.as_deref() == Some(sub_uuid))
            .unwrap();
        assert_eq!(sub_conv.title.as_deref(), Some("fallback prompt line"));
        assert!(sub_conv.metadata.get("title_source").is_none());
        assert_eq!(
            sub_conv.workspace.as_deref(),
            Some(Path::new("/run/user/1000/antigravity-overlays/only"))
        );
        assert_eq!(sub_conv.metadata["parent_conversation_id"], main_uuid);
        assert_eq!(sub_conv.metadata["agent_name"], "Codebase Researcher");
        assert_eq!(sub_conv.metadata["nesting_depth"], 1);

        // Verify zero writes / no WAL sidecars created on read-only DB in 0555 dir.
        assert_eq!(
            fs::metadata(&db_path).unwrap().modified().unwrap(),
            mtime_before
        );
        assert!(!tmp.path().join("conversation_summaries.db-wal").exists());
        assert!(!tmp.path().join("conversation_summaries.db-shm").exists());

        // Single-file ScanRoot point query also enriches metadata.
        let single_path = AntigravityConnector::transcript_full_path(&sub_dir);
        let single_ctx = ScanContext::with_roots(
            single_path.parent().unwrap().to_path_buf(),
            vec![ScanRoot::local(single_path)],
            None,
        );
        let single_convs = AntigravityConnector::new().scan(&single_ctx).unwrap();
        assert_eq!(single_convs.len(), 1);
        assert_eq!(
            single_convs[0].metadata["parent_conversation_id"],
            main_uuid
        );

        // Corrupt DB must degrade silently without aborting transcript scan.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(tmp.path(), fs::Permissions::from_mode(0o755)).unwrap();
            fs::set_permissions(&db_path, fs::Permissions::from_mode(0o644)).unwrap();
        }
        fs::write(&db_path, b"not a valid sqlite file").unwrap();
        let degraded = AntigravityConnector::new().scan(&ctx).unwrap();
        assert_eq!(degraded.len(), 2);
        assert!(degraded.iter().all(|c| c.workspace.is_none()));
    }

    #[test]
    fn discovery_covers_scan_sources_for_fixture() {
        assert_discovery_covers_scan_sources(&AntigravityConnector::new(), &fixture_ctx());
    }

    #[test]
    fn scan_with_callback_matches_scan() {
        let connector = AntigravityConnector::new();
        let scanned = connector.scan(&fixture_ctx()).unwrap();
        let mut streamed = Vec::new();
        connector
            .scan_with_callback(&fixture_ctx(), &mut |c| {
                streamed.push(c);
                Ok(())
            })
            .unwrap();
        assert_eq!(streamed.len(), scanned.len());
        assert_eq!(streamed[0].messages.len(), scanned[0].messages.len());
    }

    #[test]
    fn scan_resolves_conversation_from_direct_dir() {
        let dir = fixture_conversation_dir();
        let ctx = ScanContext::with_roots(dir.clone(), vec![ScanRoot::local(dir)], None);
        let convs = AntigravityConnector::new().scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);
    }

    #[test]
    fn scan_returns_empty_for_missing_root() {
        let ctx = ScanContext::with_roots(
            PathBuf::from("/no/such/antigravity"),
            vec![ScanRoot::local(PathBuf::from("/no/such/antigravity"))],
            None,
        );
        assert!(AntigravityConnector::new().scan(&ctx).unwrap().is_empty());
    }

    #[test]
    fn since_ts_in_future_skips_conversation() {
        let base = fixture_base();
        let ctx =
            ScanContext::with_roots(base.clone(), vec![ScanRoot::local(base)], Some(i64::MAX));
        assert!(AntigravityConnector::new().scan(&ctx).unwrap().is_empty());
    }

    #[test]
    fn tested_versions_registry_and_version_check() {
        assert!(AGY_TESTED_VERSIONS.contains(&"1.0.7"));
        assert!(AGY_TESTED_VERSIONS.contains(&"1.5.14"));
        assert!(agy_version_is_tested("1.0.7"));
        assert!(agy_version_is_tested("v1.0.7"));
        assert!(agy_version_is_tested("1.5.14"));
        assert!(agy_version_is_tested("v1.5.14"));
        assert!(!agy_version_is_tested("1.0.8"));
        assert!(!agy_version_is_tested("2.0.0"));
    }

    #[test]
    fn parse_agy_version_extracts_dotted_numeric_token() {
        assert_eq!(
            parse_agy_version("agy version 1.5.14").as_deref(),
            Some("1.5.14")
        );
        assert_eq!(parse_agy_version("v1.0.7\n").as_deref(), Some("1.0.7"));
        assert_eq!(
            parse_agy_version("Antigravity CLI 1.2.34 (build x)").as_deref(),
            Some("1.2.34")
        );
        assert_eq!(parse_agy_version("no version here").as_deref(), None);
    }

    #[test]
    fn transcript_with_only_markers_yields_no_conversation() {
        let dir = TempDir::new().unwrap();
        let logs = dir.path().join(".system_generated").join("logs");
        fs::create_dir_all(&logs).unwrap();
        fs::write(
            logs.join("transcript.jsonl"),
            "{\"step_index\":0,\"source\":\"SYSTEM\",\"type\":\"CONVERSATION_HISTORY\",\"status\":\"DONE\",\"created_at\":\"2026-06-11T20:14:42Z\"}\n",
        )
        .unwrap();
        let ctx = ScanContext::with_roots(
            dir.path().to_path_buf(),
            vec![ScanRoot::local(dir.path().to_path_buf())],
            None,
        );
        assert!(AntigravityConnector::new().scan(&ctx).unwrap().is_empty());
    }

    #[test]
    fn malformed_transcript_lines_are_skipped_not_fatal() {
        let dir = TempDir::new().unwrap();
        let logs = dir.path().join(".system_generated").join("logs");
        fs::create_dir_all(&logs).unwrap();
        fs::write(
            logs.join("transcript.jsonl"),
            "{\"step_index\":0,\"source\":\"USER_EXPLICIT\",\"type\":\"USER_INPUT\",\"status\":\"DONE\",\"created_at\":\"2026-06-11T20:14:42Z\",\"content\":\"<USER_REQUEST>\\nhi there\\n</USER_REQUEST>\"}\nthis is not json\n{\"step_index\":1,\"source\":\"MODEL\",\"type\":\"PLANNER_RESPONSE\",\"status\":\"DONE\",\"created_at\":\"2026-06-11T20:14:43Z\",\"content\":\"hello!\"}\n",
        )
        .unwrap();
        let ctx = ScanContext::with_roots(
            dir.path().to_path_buf(),
            vec![ScanRoot::local(dir.path().to_path_buf())],
            None,
        );
        let convs = AntigravityConnector::new().scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].messages.len(),
            2,
            "valid records survive a junk line"
        );
    }

    #[test]
    fn ignores_legacy_gemini_cli_layout_under_shared_dot_gemini() {
        let tmp = TempDir::new().unwrap();
        let dot_gemini = tmp.path().join(".gemini");

        let chats = dot_gemini.join("tmp").join("deadbeef").join("chats");
        fs::create_dir_all(&chats).unwrap();
        fs::write(chats.join("session-1.json"), "{\"messages\":[]}").unwrap();

        let logs = dot_gemini
            .join("antigravity-cli")
            .join("brain")
            .join("11111111-2222-3333-4444-555555555555")
            .join(".system_generated")
            .join("logs");
        fs::create_dir_all(&logs).unwrap();
        fs::write(
            logs.join("transcript.jsonl"),
            "{\"step_index\":0,\"source\":\"USER_EXPLICIT\",\"type\":\"USER_INPUT\",\"status\":\"DONE\",\"created_at\":\"2026-06-11T20:14:42Z\",\"content\":\"<USER_REQUEST>\\nhi\\n</USER_REQUEST>\"}\n",
        )
        .unwrap();

        let ctx =
            ScanContext::with_roots(dot_gemini.clone(), vec![ScanRoot::local(dot_gemini)], None);
        let convs = AntigravityConnector::new().scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1, "only the agy conversation should be found");
        assert_eq!(convs[0].agent_slug, "antigravity");
        assert_eq!(
            convs[0].external_id.as_deref(),
            Some("11111111-2222-3333-4444-555555555555")
        );
    }
}
