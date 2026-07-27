//! Connector for `OpenClaw` session logs.
//!
//! `OpenClaw` stores JSONL sessions at:
//! - ~/.openclaw/agents/<agent-name>/sessions/*.jsonl
//!
//! Each line has a `type` discriminator: "session", "message", "`model_change`",
//! "`thinking_level_change`", "custom". Messages are wrapped:
//! {"type":"message","id":"...","message":{"role":"user","content":[...],...}}

use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::Value;
use walkdir::WalkDir;

use super::scan::{DiscoveredSourceFile, DiscoveredSourceRole, ScanContext, ScanRoot};
use super::utils::{add_raw_role, set_tool_result_pairing};
use super::{Connector, file_modified_since, flatten_content, parse_timestamp};
use crate::types::{
    DetectionResult, NormalizedConversation, NormalizedInvocation, NormalizedMessage,
};

pub struct OpenClawConnector;

impl Default for OpenClawConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenClawConnector {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn openclaw_home() -> Option<PathBuf> {
        dirs::home_dir().map(|home| home.join(".openclaw"))
    }

    fn agents_root() -> Option<PathBuf> {
        Self::openclaw_home().map(|home| home.join("agents"))
    }

    fn find_agent_session_dirs() -> Vec<PathBuf> {
        Self::agents_root().map_or_else(Vec::new, |agents_root| {
            Self::find_agent_session_dirs_at(&agents_root)
        })
    }

    fn find_agent_session_dirs_at(agents_root: &Path) -> Vec<PathBuf> {
        tracing::debug!(
            agents_root = %agents_root.display(),
            "openclaw: scanning agents root for sessions directories"
        );

        if !agents_root.exists() || !agents_root.is_dir() {
            return Vec::new();
        }

        let mut session_dirs: Vec<PathBuf> = Vec::new();
        let walker = WalkDir::new(agents_root)
            .follow_links(false)
            .min_depth(1)
            .max_depth(2);

        for entry_res in walker {
            let entry = match entry_res {
                Ok(entry) => entry,
                Err(err) => {
                    tracing::debug!(
                        agents_root = %agents_root.display(),
                        error = %err,
                        "openclaw: cannot read directory entry, continuing"
                    );
                    continue;
                }
            };

            if !entry.file_type().is_dir() || entry.depth() != 1 {
                continue;
            }

            let agent_name = entry.file_name().to_string_lossy().to_string();
            let sessions_dir = entry.path().join("sessions");
            let has_sessions = sessions_dir.is_dir();
            tracing::debug!(
                agent = %agent_name,
                has_sessions,
                "openclaw: found agent directory"
            );

            if has_sessions {
                session_dirs.push(sessions_dir);
            } else {
                tracing::debug!(
                    agent = %agent_name,
                    "openclaw: skipping agent directory without sessions/ subdirectory"
                );
            }
        }

        session_dirs.sort();
        session_dirs.dedup();

        let mut agent_names: Vec<String> = session_dirs
            .iter()
            .filter_map(|dir| {
                dir.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .map(String::from)
            })
            .collect();
        agent_names.sort();

        tracing::debug!(
            count = session_dirs.len(),
            agents = ?agent_names,
            "openclaw: discovered agent session directories"
        );

        session_dirs
    }

    fn detect_from_agents_root(agents_root: &Path) -> DetectionResult {
        let roots = Self::find_agent_session_dirs_at(agents_root);
        let mut evidence = vec![
            format!("found {}", agents_root.display()),
            format!("discovered {} agent session dirs", roots.len()),
        ];

        if !roots.is_empty() {
            let mut names: Vec<String> = roots
                .iter()
                .filter_map(|path| {
                    path.parent()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .map(String::from)
                })
                .collect();
            names.sort();
            evidence.push(format!("agents: {}", names.join(", ")));
        }

        DetectionResult {
            detected: true,
            evidence,
            root_paths: roots,
        }
    }

    fn looks_like_openclaw_storage(path: &Path) -> bool {
        let path_str = path.to_string_lossy().to_lowercase();
        path_str.contains("openclaw") && path_str.contains("sessions")
    }

    fn session_root_from_candidate(path: &Path) -> Option<PathBuf> {
        let dir = if path.is_file() {
            path.parent().unwrap_or(path)
        } else {
            path
        };

        if dir.file_name().and_then(|n| n.to_str()) == Some("sessions") && dir.is_dir() {
            return Some(dir.to_path_buf());
        }

        let sessions = dir.join("sessions");
        if sessions.is_dir() {
            Some(sessions)
        } else {
            None
        }
    }

    fn roots_from_scan_path(path: &Path) -> Vec<PathBuf> {
        let mut roots = Vec::new();

        if let Some(explicit) = Self::session_root_from_candidate(path)
            && Self::looks_like_openclaw_storage(&explicit)
        {
            roots.push(explicit);
        }

        let embedded_agents = path.join(".openclaw").join("agents");
        if embedded_agents.exists() {
            roots.extend(Self::find_agent_session_dirs_at(&embedded_agents));
        }

        if path.file_name().and_then(|n| n.to_str()) == Some(".openclaw") {
            roots.extend(Self::find_agent_session_dirs_at(&path.join("agents")));
        }

        if path.file_name().and_then(|n| n.to_str()) == Some("agents") {
            roots.extend(Self::find_agent_session_dirs_at(path));
        }

        roots.sort();
        roots.dedup();
        roots
    }

    fn agent_directory_from_sessions_root(path: &Path) -> String {
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("openclaw")
            .to_string()
    }

    fn agent_slug_for_directory(agent_dir: &str) -> String {
        if agent_dir == "openclaw" {
            "openclaw".to_string()
        } else {
            format!("openclaw/{agent_dir}")
        }
    }

    fn session_files(root: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if !root.exists() {
            return out;
        }

        for entry in WalkDir::new(root).into_iter().flatten() {
            if !entry.file_type().is_file() {
                continue;
            }
            if entry.path().extension().and_then(|s| s.to_str()) == Some("jsonl") {
                out.push(entry.path().to_path_buf());
            }
        }

        // Keep scan order deterministic across filesystems and runs.
        out.sort();
        out
    }

    fn source_roots(ctx: &ScanContext) -> Vec<ScanRoot> {
        let mut roots: Vec<ScanRoot> = Vec::new();
        if ctx.use_default_detection() {
            if let Some(explicit) = Self::session_root_from_candidate(&ctx.data_dir)
                && Self::looks_like_openclaw_storage(&explicit)
                && explicit.exists()
            {
                roots.push(ScanRoot::local(explicit));
            } else {
                roots.extend(
                    Self::find_agent_session_dirs()
                        .into_iter()
                        .map(ScanRoot::local),
                );
            }
        } else {
            for root in &ctx.scan_roots {
                roots.extend(
                    Self::roots_from_scan_path(&root.path)
                        .into_iter()
                        .map(|path| root.with_path(path)),
                );
            }
        }

        roots.sort_by(|a, b| a.path.cmp(&b.path));
        roots.dedup_by(|a, b| a.path == b.path);
        roots
    }

    fn discover_sources(ctx: &ScanContext) -> Vec<DiscoveredSourceFile> {
        let mut out = Vec::new();
        for mut root in Self::source_roots(ctx) {
            if root.path.is_file() {
                let parent = root.path.parent().unwrap_or(&root.path).to_path_buf();
                root = root.with_path(parent);
            }
            for file in Self::session_files(&root.path) {
                if !file_modified_since(&file, ctx.since_ts) {
                    continue;
                }
                out.push(
                    DiscoveredSourceFile::new(
                        "openclaw",
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

    /// Render a `toolCall` block's own content for display: `<name>(<args
    /// JSON>)`, or just `<name>` when there's no args. Mirrors
    /// `claude_code.rs`'s `render_tool_call_content` -- the full untruncated
    /// args always live in `extra["tool_call_args"]`/the invocation's
    /// `arguments` for exact reconstruction (never truncated -- spec §3.2).
    fn render_tool_call_content(name: &str, args: Option<&Value>) -> String {
        match args {
            Some(value) if !value.is_null() => format!("{name}({value})"),
            _ => name.to_string(),
        }
    }

    fn merge_openclaw_pairing_id(
        resolved: &mut Option<String>,
        candidate: Option<String>,
    ) -> Result<()> {
        let Some(candidate) = candidate else {
            return Ok(());
        };
        match resolved {
            Some(existing) if existing != &candidate => {
                anyhow::bail!("OpenClaw tool result pairing identifiers conflict");
            }
            Some(_) => {}
            None => *resolved = Some(candidate),
        }
        Ok(())
    }

    /// Resolve a `toolResult` pairing id from one structural object. All
    /// non-blank explicit id fields must agree; redundant equal fields are
    /// accepted, while distinct ids fail without being copied into the error.
    fn openclaw_pairing_id_from(v: &Value) -> Result<Option<String>> {
        let mut resolved = None;
        for key in ["toolCallId", "toolUseId", "tool_use_id", "id"] {
            let candidate = v
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(String::from);
            Self::merge_openclaw_pairing_id(&mut resolved, candidate)?;
        }
        Ok(resolved)
    }

    /// Merge the outer top-level `toolResult` message id with ids from only
    /// typed `toolResult` content blocks. Other block ids belong to other
    /// structures and must never participate in tool-result pairing.
    fn openclaw_top_level_tool_result_pairing_id(
        message: &Value,
        content: Option<&Value>,
    ) -> Result<Option<String>> {
        let mut resolved = Self::openclaw_pairing_id_from(message)?;
        if let Some(blocks) = content.and_then(Value::as_array) {
            for block in blocks
                .iter()
                .filter(|block| block.get("type").and_then(Value::as_str) == Some("toolResult"))
            {
                let candidate = Self::openclaw_pairing_id_from(block)?;
                Self::merge_openclaw_pairing_id(&mut resolved, candidate)?;
            }
        }
        Ok(resolved)
    }

    /// Extract a single content block's own result body: prefers `content`
    /// (the openclaw `toolResult` block's full body field), falling back to
    /// `text` (the plainer shape some real `toolResult` messages emit
    /// instead -- both verified against `~/.openclaw/agents/*/sessions/*.jsonl`).
    fn openclaw_block_body(block: &Value) -> Option<String> {
        let body = block.get("content").or_else(|| block.get("text"))?;
        Some(match body {
            Value::String(s) => s.clone(),
            other => flatten_content(other),
        })
    }

    /// Extract a `toolResult`'s full untruncated body from the
    /// message-level `content` value, which is a plain string, or an array
    /// containing a single block -- typed `toolResult` (many redundant id
    /// fields, body in `content`/`text`) or, in some real sessions, plain
    /// `text` (body only in `text`) -- whose own `content`/`text` carries
    /// the body. Never truncated (spec §3.2).
    fn openclaw_tool_result_text(content: &Value) -> String {
        match content {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(Self::openclaw_block_body)
                .collect::<Vec<_>>()
                .join("\n"),
            other => flatten_content(other),
        }
    }

    /// Split an `OpenClaw` `message.content` array into typed blocks, one per
    /// element, instead of flattening everything into one string. `OpenClaw`
    /// uses its own camelCase block vocabulary (`toolCall`/`toolResult`)
    /// rather than Anthropic's (`tool_use`/`tool_result`), so this doesn't
    /// reuse `split_content_blocks` (task 1.1) verbatim.
    fn split_openclaw_blocks(content: &Value) -> Result<Vec<OpenClawBlock>> {
        let Some(arr) = content.as_array() else {
            return Ok(Vec::new());
        };

        let mut blocks = Vec::new();
        for block in arr {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            // Image payloads are not canonical conversation messages.
            if block_type == "image" {
                continue;
            }
            match block_type {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        blocks.push(OpenClawBlock::Text(text.to_string()));
                    }
                }
                "toolCall" => {
                    let name = block
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let args = block
                        .get("arguments")
                        .or_else(|| block.get("input"))
                        .cloned();
                    let id = block.get("id").and_then(|v| v.as_str()).map(String::from);
                    blocks.push(OpenClawBlock::ToolCall { name, args, id });
                }
                "toolResult" => {
                    // Defensive: real ~/.openclaw sessions never nest a
                    // toolResult block inside a non-toolResult-role message
                    // (toolResult always arrives as its own top-level
                    // message -- handled separately in `scan`), but the
                    // spec calls for handling both shapes explicitly.
                    let id = Self::openclaw_pairing_id_from(block)?;
                    let text = Self::openclaw_block_body(block).unwrap_or_default();
                    blocks.push(OpenClawBlock::ToolResult { text, id });
                }
                "thinking" => {
                    // Real OpenClaw sessions carry the reasoning in `thinking`
                    // (alongside a `thinkingSignature`), not in `text`. Reading
                    // only `text` dropped every thinking block ever written --
                    // measured across 300 live sessions: 3713 thinking blocks,
                    // `thinking` present in all 3713 and `text` in 0 of them.
                    // Same read as `utils.rs`'s typed-block split: `thinking`
                    // first, fall back to `text`. An empty-string value still
                    // yields a block rather than being silently dropped -- 2114
                    // of those 3713 are signed-but-empty, and empty is a
                    // definite value, not a missing one.
                    if let Some(text) = block
                        .get("thinking")
                        .or_else(|| block.get("text"))
                        .and_then(|t| t.as_str())
                    {
                        blocks.push(OpenClawBlock::Thinking(text.to_string()));
                    }
                }
                _ => {}
            }
        }
        Ok(blocks)
    }
}

/// A single `OpenClaw` `message.content` array element, split by block type
/// rather than merged into one flattened string (the typed counterpart to
/// the old `flatten_openclaw_content`). Full `toolCall` args / `toolResult`
/// content are always preserved untruncated -- truncation is an
/// adapter-feed concern, never canonical/franken (spec §3.2).
enum OpenClawBlock {
    /// `{"type":"text","text":...}`.
    Text(String),
    /// `{"type":"toolCall","name":...,"arguments":...,"input":...,"id":...}`.
    ToolCall {
        name: String,
        args: Option<Value>,
        id: Option<String>,
    },
    /// A `toolResult` block nested in a non-toolResult-role message's
    /// content array (defensive path; see `split_openclaw_blocks`).
    ToolResult { text: String, id: Option<String> },
    /// `{"type":"thinking","text":...}`.
    Thinking(String),
}

impl Connector for OpenClawConnector {
    fn detect(&self) -> DetectionResult {
        // Use OpenClaw-specific multi-agent detection instead of the generic
        // franken probe, which only checks for directory existence and doesn't
        // walk the agents/<name>/sessions/ layout.
        match Self::agents_root() {
            Some(agents_root) if agents_root.exists() => {
                Self::detect_from_agents_root(&agents_root)
            }
            _ => DetectionResult::not_found(),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn scan(&self, ctx: &ScanContext) -> Result<Vec<NormalizedConversation>> {
        let roots: Vec<PathBuf> = Self::source_roots(ctx)
            .into_iter()
            .map(|root| root.path)
            .collect();

        if roots.is_empty() {
            return Ok(Vec::new());
        }

        let mut convs = Vec::new();
        let mut scanned_agents = 0usize;

        for mut root in roots {
            if root.is_file() {
                root = root.parent().unwrap_or(&root).to_path_buf();
            }

            let agent_directory = Self::agent_directory_from_sessions_root(&root);
            let agent_slug = Self::agent_slug_for_directory(&agent_directory);
            let files = Self::session_files(&root);
            let mut agent_file_count = 0usize;
            let mut agent_session_count = 0usize;
            let mut agent_error_count = 0usize;
            tracing::debug!(
                agent = %agent_directory,
                file_count = files.len(),
                "openclaw: scanning agent directory"
            );
            for file in files {
                agent_file_count += 1;
                if !file_modified_since(&file, ctx.since_ts) {
                    continue;
                }

                let source_path = file.clone();
                let external_id = source_path
                    .strip_prefix(&root)
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

                let external_id = if agent_directory == "openclaw" {
                    external_id
                } else {
                    external_id.map(|id| format!("{agent_directory}/{id}"))
                };

                let file_handle = match fs::File::open(&file) {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::debug!(path = %file.display(), error = %e, "openclaw: skipping unreadable session");
                        agent_error_count += 1;
                        continue;
                    }
                };
                let reader = std::io::BufReader::new(file_handle);

                let mut messages = Vec::new();
                let mut started_at: Option<i64> = None;
                let mut ended_at: Option<i64> = None;
                let mut session_cwd: Option<String> = None;

                for line_res in reader.lines() {
                    let Ok(line) = line_res else {
                        continue;
                    };
                    if line.trim().is_empty() {
                        continue;
                    }

                    let val: Value = match serde_json::from_str(&line) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let line_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("");

                    if matches!(
                        line_type,
                        "model_change"
                            | "thinking_level_change"
                            | "custom"
                            | "response.done"
                            | "turn.completion_idle_timeout"
                            | "turn.terminal_idle_timeout"
                            | "turn.client_closed"
                    ) {
                        continue;
                    }

                    match line_type {
                        "session" => {
                            // Extract session metadata
                            session_cwd = val.get("cwd").and_then(|v| v.as_str()).map(String::from);
                            if let Some(ts) = val.get("timestamp").and_then(parse_timestamp) {
                                started_at = Some(ts);
                            }
                        }
                        "message" => {
                            // Messages are wrapped: {type:"message", message:{role, content, ...}}
                            let Some(msg) = val.get("message") else {
                                continue;
                            };

                            let Some(raw_role @ ("user" | "assistant" | "toolResult")) =
                                msg.get("role").and_then(|v| v.as_str())
                            else {
                                continue;
                            };

                            // Timestamps can be on the wrapper or inner message
                            let created = val
                                .get("timestamp")
                                .and_then(parse_timestamp)
                                .or_else(|| msg.get("timestamp").and_then(parse_timestamp));

                            started_at = match (started_at, created) {
                                (Some(curr), Some(ts)) => Some(curr.min(ts)),
                                (None, Some(ts)) => Some(ts),
                                (other, None) => other,
                            };
                            ended_at = match (ended_at, created) {
                                (Some(curr), Some(ts)) => Some(curr.max(ts)),
                                (None, Some(ts)) => Some(ts),
                                (other, None) => other,
                            };

                            let author =
                                msg.get("model").and_then(|v| v.as_str()).map(String::from);
                            let content = msg.get("content");
                            let base_extra = add_raw_role(&val, val.clone(), raw_role)?;

                            // Top-level `toolResult` messages (role=="toolResult")
                            // carry the whole tool result body directly on the
                            // message -- pairing id lives on the message itself
                            // (`toolCallId`, falling back to
                            // `toolUseId`/`tool_use_id`/`id`), body lives in
                            // `content` (a plain string, or an array with a
                            // single block -- typed `toolResult`, or in some
                            // real sessions plain `text` -- both shapes
                            // verified against real ~/.openclaw sessions).
                            if raw_role == "toolResult" {
                                // Completeness policy (uniform w/ claude_code.rs
                                // + codex.rs): never drop a structural item for
                                // empty content. An empty toolResult body
                                // (real: 71 `content:[]` results, e.g.
                                // `update_plan`) is still emitted so the
                                // tool_call<->tool_result pairing chain
                                // survives.
                                let result_text = content
                                    .map(Self::openclaw_tool_result_text)
                                    .unwrap_or_default();
                                let pairing_id =
                                    Self::openclaw_top_level_tool_result_pairing_id(msg, content)?;
                                let mut extra = base_extra;
                                set_tool_result_pairing(&mut extra, pairing_id.as_deref())?;
                                messages.push(NormalizedMessage {
                                    idx: 0,
                                    role: "tool_result".to_string(),
                                    author: None,
                                    created_at: created,
                                    content: result_text,
                                    extra,
                                    invocations: Vec::new(),
                                    snippets: Vec::new(),
                                });
                                continue;
                            }

                            // user/assistant messages: content is a plain string
                            // or an array interleaving `text`, `toolCall`,
                            // `thinking` blocks. Split by type (mirrors
                            // claude_code.rs/task 1.2) instead of flattening
                            // everything into one string, so each block becomes
                            // its own typed 6-role message (spec §3.3) rather
                            // than an inline `[tool: name]` marker glued into
                            // prose. OpenClaw's role strings (`user`/`assistant`)
                            // already match the 6-role vocabulary verbatim --
                            // no rename needed (unlike codex's developer->system).
                            let canonical_role = raw_role;
                            let is_assistant = canonical_role == "assistant";

                            match content {
                                Some(Value::Array(_)) => {
                                    let blocks = Self::split_openclaw_blocks(content.unwrap())?;

                                    let mut prose = String::new();
                                    for block in &blocks {
                                        if let OpenClawBlock::Text(text) = block {
                                            if !prose.is_empty() {
                                                prose.push('\n');
                                            }
                                            prose.push_str(text);
                                        }
                                    }
                                    if !prose.trim().is_empty() {
                                        messages.push(NormalizedMessage {
                                            idx: 0,
                                            role: canonical_role.to_string(),
                                            author: if is_assistant {
                                                author.clone()
                                            } else {
                                                None
                                            },
                                            created_at: created,
                                            content: prose,
                                            extra: base_extra.clone(),
                                            invocations: Vec::new(),
                                            snippets: Vec::new(),
                                        });
                                    }

                                    for block in blocks {
                                        match block {
                                            OpenClawBlock::Text(_) => {}
                                            OpenClawBlock::ToolCall { name, args, id } => {
                                                let content_text = Self::render_tool_call_content(
                                                    &name,
                                                    args.as_ref(),
                                                );
                                                let mut extra = base_extra.clone();
                                                if let Value::Object(map) = &mut extra {
                                                    if let Some(ref call_id) = id {
                                                        map.insert(
                                                            "tool_call_id".to_string(),
                                                            Value::String(call_id.clone()),
                                                        );
                                                    }
                                                    map.insert(
                                                        "tool_call_args".to_string(),
                                                        args.clone().unwrap_or(Value::Null),
                                                    );
                                                }
                                                // tool_call/tool_result pairing is
                                                // via this explicit id
                                                // (extra["tool_call_id"]), never
                                                // content order (spec P-原则-3).
                                                messages.push(NormalizedMessage {
                                                    idx: 0,
                                                    role: "tool_call".to_string(),
                                                    author: author.clone(),
                                                    created_at: created,
                                                    content: content_text,
                                                    extra,
                                                    invocations: vec![NormalizedInvocation {
                                                        kind: "tool".to_string(),
                                                        name,
                                                        raw_name: None,
                                                        call_id: id,
                                                        arguments: args,
                                                    }],
                                                    snippets: Vec::new(),
                                                });
                                            }
                                            OpenClawBlock::ToolResult { text, id } => {
                                                // Completeness policy (uniform
                                                // w/ claude_code.rs + codex.rs):
                                                // an empty tool_result body is
                                                // still emitted, keeping its
                                                // pairing id.
                                                let mut extra = base_extra.clone();
                                                set_tool_result_pairing(&mut extra, id.as_deref())?;
                                                messages.push(NormalizedMessage {
                                                    idx: 0,
                                                    role: "tool_result".to_string(),
                                                    author: None,
                                                    created_at: created,
                                                    content: text,
                                                    extra,
                                                    invocations: Vec::new(),
                                                    snippets: Vec::new(),
                                                });
                                            }
                                            OpenClawBlock::Thinking(text) => {
                                                messages.push(NormalizedMessage {
                                                    idx: 0,
                                                    role: "reasoning".to_string(),
                                                    author: author.clone(),
                                                    created_at: created,
                                                    content: text,
                                                    extra: base_extra.clone(),
                                                    invocations: Vec::new(),
                                                    snippets: Vec::new(),
                                                });
                                            }
                                        }
                                    }
                                }
                                Some(other) => {
                                    // Not a content-block array (e.g. a plain
                                    // string) -- no toolCall/toolResult/thinking
                                    // blocks are possible here.
                                    let content_str = flatten_content(other);
                                    if !content_str.trim().is_empty() {
                                        messages.push(NormalizedMessage {
                                            idx: 0,
                                            role: canonical_role.to_string(),
                                            author: if is_assistant { author } else { None },
                                            created_at: created,
                                            content: content_str,
                                            extra: base_extra,
                                            invocations: Vec::new(),
                                            snippets: Vec::new(),
                                        });
                                    }
                                }
                                None => {}
                            }
                        }
                        "compaction" => {
                            // `compaction` events carry a substantive
                            // multi-paragraph `summary` (a running digest of
                            // the conversation), directly analogous to
                            // claude's `away_summary`. The normalized role is
                            // assistant while raw_role preserves compaction.
                            // Only emit when `summary` is a
                            // non-empty string; an absent/empty summary has no
                            // content -> drop. Routed through the same
                            // `messages` vector so the `reindex_messages` call
                            // below renumbers it into the contiguous idx run.
                            if let Some(summary) = val
                                .get("summary")
                                .and_then(|v| v.as_str())
                                .filter(|s| !s.trim().is_empty())
                            {
                                let created = val.get("timestamp").and_then(parse_timestamp);
                                started_at = match (started_at, created) {
                                    (Some(curr), Some(ts)) => Some(curr.min(ts)),
                                    (None, Some(ts)) => Some(ts),
                                    (other, None) => other,
                                };
                                ended_at = match (ended_at, created) {
                                    (Some(curr), Some(ts)) => Some(curr.max(ts)),
                                    (None, Some(ts)) => Some(ts),
                                    (other, None) => other,
                                };
                                messages.push(NormalizedMessage {
                                    idx: 0,
                                    role: "assistant".to_string(),
                                    author: None,
                                    created_at: created,
                                    content: summary.to_string(),
                                    extra: add_raw_role(&val, val.clone(), "compaction")?,
                                    invocations: Vec::new(),
                                    snippets: Vec::new(),
                                });
                            }
                        }
                        // Unknown future wrappers remain unclassifiable. Do
                        // not guess a canonical message mapping for them.
                        _ => {}
                    }
                }

                // Splitting one raw "message" line into several typed messages
                // (text/toolCall/thinking/toolResult) means idx must be
                // recomputed to stay contiguous 0..N (spec §3.4) -- required
                // for `UNIQUE(conversation_id,idx)`. OpenClaw previously never
                // called this (idx was assigned inline, 1 push per line); now
                // required.
                crate::types::reindex_messages(&mut messages);

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

                let workspace = session_cwd.as_ref().map(PathBuf::from);

                let metadata = serde_json::json!({
                    "source": "openclaw",
                    "cwd": session_cwd,
                    "agent_directory": agent_directory.clone(),
                });

                convs.push(NormalizedConversation {
                    agent_slug: agent_slug.clone(),
                    external_id,
                    title,
                    workspace,
                    source_path,
                    started_at,
                    ended_at,
                    metadata,
                    messages,
                });
                agent_session_count += 1;
            }

            scanned_agents += 1;
            tracing::debug!(
                agent = %agent_directory,
                files = agent_file_count,
                sessions = agent_session_count,
                errors = agent_error_count,
                "openclaw: completed agent scan"
            );
        }

        tracing::debug!(
            agents = scanned_agents,
            sessions = convs.len(),
            "openclaw: completed multi-agent scan"
        );

        Ok(convs)
    }

    fn discover_source_files(&self, ctx: &ScanContext) -> Result<Vec<DiscoveredSourceFile>> {
        Ok(Self::discover_sources(ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_session(root: &Path, name: &str, lines: &[&str]) -> PathBuf {
        let path = root.join(name);
        let content = lines.join("\n");
        fs::write(&path, content).unwrap();
        path
    }

    fn write_minimal_openclaw_session(
        sessions_root: &Path,
        file_name: &str,
        cwd: &str,
        user_text: &str,
    ) -> PathBuf {
        write_session(
            sessions_root,
            file_name,
            &[
                &format!(
                    r#"{{"type":"session","id":"s1","timestamp":"2026-02-01T16:00:00.000Z","cwd":"{cwd}"}}"#
                ),
                &format!(
                    r#"{{"type":"message","id":"m1","timestamp":"2026-02-01T16:00:01.000Z","message":{{"role":"user","content":[{{"type":"text","text":"{user_text}"}}]}}}}"#
                ),
            ],
        )
    }

    fn ctx_with_root(root: &Path) -> ScanContext {
        ScanContext::with_roots(
            root.to_path_buf(),
            vec![super::super::ScanRoot::local(root.to_path_buf())],
            None,
        )
    }

    #[test]
    fn scan_parses_openclaw_wrapped_messages() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"session","id":"abc","timestamp":"2026-02-01T16:00:00.000Z","cwd":"/home/user/project","version":"0.1.0"}"#,
                r#"{"type":"message","id":"m1","parentId":"abc","timestamp":"2026-02-01T16:00:00.828Z","message":{"role":"user","content":[{"type":"text","text":"Hello OpenClaw"}],"timestamp":1769961600827}}"#,
                r#"{"type":"message","id":"m2","parentId":"m1","timestamp":"2026-02-01T16:00:06.672Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi there!"},{"type":"toolCall","id":"tc1","name":"exec","arguments":{}}],"api":"anthropic-messages","provider":"anthropic","model":"claude-opus-4-5"}}"#,
            ],
        );

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::local_default(sessions.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].agent_slug, "openclaw");
        // 6-role normalization: the assistant's `text` + `toolCall` blocks
        // split into two messages (prose + tool_call) rather than one
        // combined message with an inline `[tool: exec]` marker.
        assert_eq!(convs[0].messages.len(), 3);
        assert_eq!(convs[0].title, Some("Hello OpenClaw".to_string()));
        assert_eq!(convs[0].messages[0].role, "user");
        assert_eq!(convs[0].messages[1].role, "assistant");
        assert!(convs[0].messages[1].content.contains("Hi there!"));
        assert!(
            !convs[0].messages[1].content.contains("[tool:"),
            "assistant prose must not carry an inline tool marker (spec §3.3): {:?}",
            convs[0].messages[1].content
        );
        assert_eq!(
            convs[0].messages[1].author,
            Some("claude-opus-4-5".to_string())
        );
        let tool_call = convs[0]
            .messages
            .iter()
            .find(|m| m.role == "tool_call")
            .expect("toolCall block must become its own tool_call message");
        assert!(tool_call.content.contains("exec"));
        assert_eq!(tool_call.extra["tool_call_id"].as_str(), Some("tc1"));
        assert!(convs[0].workspace.is_some());
        assert!(convs[0].started_at.is_some());
        crate::connectors::assert_discovery_covers_scan_sources(&connector, &ctx);
    }

    // =========================================================================
    // 6-role normalization tests (franken fork, spec §3.3 openclaw)
    // =========================================================================

    #[test]
    fn scan_openclaw_requires_an_explicit_whitelisted_message_role() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"message","id":"valid","message":{"role":"user","content":"kept"}}"#,
                r#"{"type":"message","id":"missing","message":{"content":"must not default to assistant"}}"#,
                r#"{"type":"message","id":"unknown","message":{"role":"system","content":"must not pass through"}}"#,
            ],
        );

        let convs = OpenClawConnector::new()
            .scan(&ScanContext::local_default(sessions, None))
            .unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].role, "user");
        assert_eq!(convs[0].messages[0].content, "kept");
    }

    #[test]
    fn scan_openclaw_copies_envelope_raw_role_to_every_retained_block() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"message","id":"m1","message":{"role":"user","content":[{"type":"text","text":"ask"}]}}"#,
                r#"{"type":"message","id":"m2","message":{"role":"assistant","model":"synthetic-model","content":[{"type":"text","text":"answer"},{"type":"toolCall","id":"call_1","name":"read","arguments":{"path":"fixture.txt"}},{"type":"thinking","thinking":"inspect first","thinkingSignature":"sig-1"},{"type":"toolResult","toolCallId":"call_1","content":""}]}}"#,
                r#"{"type":"message","id":"m3","message":{"role":"toolResult","toolCallId":"call_1","content":[]}}"#,
            ],
        );

        let convs = OpenClawConnector::new()
            .scan(&ScanContext::local_default(sessions, None))
            .unwrap();
        let messages = &convs[0].messages;
        assert_eq!(messages.len(), 6);

        for message in messages {
            let expected = match message.content.as_str() {
                "ask" => "user",
                "" if message.extra["message"]["role"] == "toolResult" => "toolResult",
                _ => "assistant",
            };
            assert_eq!(
                message.extra["raw_role"].as_str(),
                Some(expected),
                "role={} content={:?}",
                message.role,
                message.content
            );
        }

        let reasoning = messages
            .iter()
            .find(|message| message.role == "reasoning")
            .expect("thinking block");
        assert_eq!(reasoning.author.as_deref(), Some("synthetic-model"));
    }

    /// Locks the *real* OpenClaw thinking shape against the implementation.
    ///
    /// The shapes here were taken from live sessions, not from reading
    /// `split_openclaw_blocks`. That distinction is the whole point: the
    /// implementation, the `utils.rs` comment, two tests and a downstream
    /// fixture all previously encoded the same wrong belief (`text` carries
    /// the reasoning), so they agreed with each other and the gate stayed
    /// green while every thinking block was silently dropped. Measured on
    /// live sessions: 3713 thinking blocks, `thinking` present in all of
    /// them, `text` in none.
    ///
    /// Both live forms are covered -- signed with a body, and signed with an
    /// empty body (2114 of the 3713). Empty is a definite value: it must
    /// still produce a `reasoning` message so the block's position in the
    /// turn survives, exactly as an empty Claude thinking block does.
    #[test]
    fn scan_openclaw_reads_thinking_key_and_keeps_empty_signed_blocks() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"message","id":"m1","message":{"role":"user","content":[{"type":"text","text":"ask"}]}}"#,
                r#"{"type":"message","id":"m2","message":{"role":"assistant","model":"synthetic-model","content":[{"type":"thinking","thinking":"real body","thinkingSignature":"sig-a"},{"type":"thinking","thinking":"","thinkingSignature":"sig-b"}]}}"#,
            ],
        );

        let convs = OpenClawConnector::new()
            .scan(&ScanContext::local_default(sessions, None))
            .unwrap();
        let reasoning: Vec<_> = convs[0]
            .messages
            .iter()
            .filter(|message| message.role == "reasoning")
            .collect();

        assert_eq!(
            reasoning.len(),
            2,
            "both signed thinking blocks must survive, including the empty one"
        );
        assert_eq!(reasoning[0].content, "real body");
        assert_eq!(reasoning[1].content, "");
        for message in &reasoning {
            assert_eq!(message.author.as_deref(), Some("synthetic-model"));
            assert_eq!(message.extra["raw_role"].as_str(), Some("assistant"));
        }
    }

    /// The `text` fallback stays supported: `utils.rs` accepts both keys and
    /// this connector must not diverge from it. Dropping the fallback would
    /// be a second, opposite version of the same bug.
    #[test]
    fn scan_openclaw_still_accepts_legacy_thinking_text_key() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"message","id":"m1","message":{"role":"assistant","model":"m","content":[{"type":"thinking","text":"legacy body"}]}}"#,
            ],
        );

        let convs = OpenClawConnector::new()
            .scan(&ScanContext::local_default(sessions, None))
            .unwrap();
        let reasoning: Vec<_> = convs[0]
            .messages
            .iter()
            .filter(|message| message.role == "reasoning")
            .collect();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0].content, "legacy body");
    }

    #[test]
    fn scan_openclaw_tool_results_enforce_paired_unpaired_xor_and_scan_array_ids() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"message","id":"anchor","message":{"role":"user","content":"anchor"}}"#,
                r#"{"type":"message","id":"top-paired","message":{"role":"toolResult","content":[{"type":"text","text":"prefix"},{"type":"toolResult","toolCallId":"call_late","content":"paired"}]}}"#,
                r#"{"type":"message","id":"top-unpaired","message":{"role":"toolResult","content":[]}}"#,
                r#"{"type":"message","id":"embedded","message":{"role":"assistant","content":[{"type":"toolResult","toolUseId":"embedded_pair","content":""},{"type":"toolResult","content":""}]}}"#,
            ],
        );

        let convs = OpenClawConnector::new()
            .scan(&ScanContext::local_default(sessions, None))
            .unwrap();
        let results: Vec<_> = convs[0]
            .messages
            .iter()
            .filter(|message| message.role == "tool_result")
            .collect();
        assert_eq!(results.len(), 4, "empty results must remain structural");

        let paired: Vec<_> = results
            .iter()
            .filter(|message| message.extra.get("tool_call_id").is_some())
            .collect();
        let unpaired: Vec<_> = results
            .iter()
            .filter(|message| message.extra.get("unpaired") == Some(&Value::Bool(true)))
            .collect();
        assert_eq!(paired.len(), 2);
        assert_eq!(unpaired.len(), 2);
        assert!(
            paired
                .iter()
                .all(|message| !message.extra["unpaired"].is_boolean())
        );
        assert!(
            unpaired
                .iter()
                .all(|message| message.extra.get("tool_call_id").is_none())
        );
        assert!(
            paired
                .iter()
                .any(|message| message.extra["tool_call_id"] == "call_late"),
            "top-level content arrays must scan beyond the first block for a real id"
        );
        assert!(
            paired
                .iter()
                .any(|message| message.extra["tool_call_id"] == "embedded_pair")
        );
    }

    #[test]
    fn scan_openclaw_skips_blank_outer_pairing_id_and_finds_late_content_id() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"message","id":"anchor","message":{"role":"user","content":"anchor"}}"#,
                r#"{"type":"message","id":"result","message":{"role":"toolResult","toolCallId":"   ","content":[{"type":"text","text":"prefix"},{"type":"toolResult","toolCallId":" ","toolUseId":"valid_late","content":"paired"}]}}"#,
            ],
        );

        let convs = OpenClawConnector::new()
            .scan(&ScanContext::local_default(sessions, None))
            .unwrap();
        let result = convs[0]
            .messages
            .iter()
            .find(|message| message.role == "tool_result")
            .expect("tool result");

        assert_eq!(result.extra["tool_call_id"], "valid_late");
        assert!(result.extra.get("unpaired").is_none());
    }

    #[test]
    fn scan_openclaw_top_level_tool_result_ignores_ids_on_unrelated_blocks() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"message","id":"anchor","message":{"role":"user","content":"anchor"}}"#,
                r#"{"type":"message","id":"result","message":{"role":"toolResult","content":[{"type":"text","id":"unrelated-text-id","text":"prefix"},{"type":"toolResult","toolCallId":"real-call","content":"paired"}]}}"#,
            ],
        );

        let convs = OpenClawConnector::new()
            .scan(&ScanContext::local_default(sessions, None))
            .unwrap();
        let result = convs[0]
            .messages
            .iter()
            .find(|message| message.role == "tool_result")
            .expect("tool result");

        assert_eq!(result.extra["tool_call_id"], "real-call");
        assert_ne!(result.extra["tool_call_id"], "unrelated-text-id");
        assert!(result.extra.get("unpaired").is_none());
    }

    #[test]
    fn scan_openclaw_pairing_conflicts_fail_loud_without_echoing_ids() {
        let cases = [
            (
                "same object",
                r#"{"type":"message","id":"result","message":{"role":"toolResult","toolCallId":"same-object-a","toolUseId":"same-object-b","content":"result"}}"#,
                ["same-object-a", "same-object-b"],
            ),
            (
                "outer versus typed block",
                r#"{"type":"message","id":"result","message":{"role":"toolResult","toolCallId":"outer-a","content":[{"type":"toolResult","toolCallId":"typed-b","content":"result"}]}}"#,
                ["outer-a", "typed-b"],
            ),
            (
                "multiple typed blocks",
                r#"{"type":"message","id":"result","message":{"role":"toolResult","content":[{"type":"toolResult","toolCallId":"typed-first","content":"one"},{"type":"toolResult","toolCallId":"typed-second","content":"two"}]}}"#,
                ["typed-first", "typed-second"],
            ),
            (
                "embedded same object",
                r#"{"type":"message","id":"result","message":{"role":"assistant","content":[{"type":"toolResult","toolCallId":"embedded-a","tool_use_id":"embedded-b","content":"result"}]}}"#,
                ["embedded-a", "embedded-b"],
            ),
        ];

        let violations = cases
            .into_iter()
            .filter_map(|(case, record, forbidden_ids)| {
                let tmp = TempDir::new().unwrap();
                let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
                fs::create_dir_all(&sessions).unwrap();
                write_session(
                    &sessions,
                    "session.jsonl",
                    &[
                        r#"{"type":"message","id":"anchor","message":{"role":"user","content":"anchor"}}"#,
                        record,
                    ],
                );

                match OpenClawConnector::new()
                    .scan(&ScanContext::local_default(sessions, None))
                {
                    Err(error) => {
                        let error = error.to_string();
                        if error.contains("pairing")
                            && forbidden_ids.iter().all(|id| !error.contains(id))
                        {
                            None
                        } else {
                            Some(format!("{case}: wrong error: {error}"))
                        }
                    }
                    Ok(_) => Some(format!("{case}: unexpectedly accepted")),
                }
            })
            .collect::<Vec<_>>();
        assert!(violations.is_empty(), "{violations:#?}");
    }

    #[test]
    fn scan_openclaw_pairing_redundancy_blank_and_missing_ids_remain_valid() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"message","id":"anchor","message":{"role":"user","content":"anchor"}}"#,
                r#"{"type":"message","id":"redundant","message":{"role":"toolResult","toolCallId":"same-id","toolUseId":"same-id","content":[{"type":"toolResult","id":"same-id","toolCallId":"same-id","tool_use_id":"same-id","content":"redundant"}]}}"#,
                r#"{"type":"message","id":"blank","message":{"role":"toolResult","toolCallId":"   ","content":[{"type":"toolResult","toolUseId":"valid-typed","content":"blank outer"}]}}"#,
                r#"{"type":"message","id":"missing","message":{"role":"toolResult","content":[]}}"#,
            ],
        );

        let convs = OpenClawConnector::new()
            .scan(&ScanContext::local_default(sessions, None))
            .unwrap();
        let results = convs[0]
            .messages
            .iter()
            .filter(|message| message.role == "tool_result")
            .collect::<Vec<_>>();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].extra["tool_call_id"], "same-id");
        assert_eq!(results[1].extra["tool_call_id"], "valid-typed");
        assert_eq!(results[2].extra["unpaired"], true);
        assert!(results[2].extra.get("tool_call_id").is_none());
    }

    #[test]
    fn scan_openclaw_explicitly_drops_images_and_control_wrappers_without_idx_holes() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"session","id":"s1","timestamp":"2026-04-01T00:00:00.000Z","cwd":"/synthetic/workspace"}"#,
                r#"{"type":"message","id":"u1","message":{"role":"user","content":[{"type":"image","mimeType":"image/png","data":"synthetic"},{"type":"text","text":"first"}]}}"#,
                r#"{"type":"response.done","response":{"id":"r1"}}"#,
                r#"{"type":"turn.completion_idle_timeout","turn":{"id":"t1"}}"#,
                r#"{"type":"turn.terminal_idle_timeout","turn":{"id":"t1"}}"#,
                r#"{"type":"turn.client_closed","turn":{"id":"t1"}}"#,
                r#"{"type":"message","id":"a1","message":{"role":"assistant","content":[{"type":"image","mimeType":"image/png","data":"synthetic"},{"type":"text","text":"second"}]}}"#,
            ],
        );

        let convs = OpenClawConnector::new()
            .scan(&ScanContext::local_default(sessions, None))
            .unwrap();
        let conv = &convs[0];
        assert_eq!(
            conv.messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert!(
            conv.messages
                .iter()
                .enumerate()
                .all(|(idx, message)| usize::try_from(message.idx).unwrap() == idx)
        );
        assert_eq!(
            conv.workspace.as_deref(),
            Some(Path::new("/synthetic/workspace"))
        );
        assert_eq!(conv.metadata["cwd"], "/synthetic/workspace");
        assert_eq!(conv.started_at, Some(1_775_001_600_000));
    }

    #[test]
    fn scan_openclaw_propagates_raw_role_collision_errors() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session.jsonl",
            &[
                r#"{"type":"message","raw_role":"collision","message":{"role":"user","content":"must fail loud"}}"#,
            ],
        );

        let error = OpenClawConnector::new()
            .scan(&ScanContext::local_default(sessions, None))
            .expect_err("reserved raw_role collision must propagate through scan");
        assert!(error.to_string().contains("raw_role"));
    }

    #[test]
    fn scan_openclaw_splits_content_blocks_into_typed_6role_messages() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        // Real OpenClaw shapes (verified against
        // ~/.openclaw/agents/*/sessions/*.jsonl): an assistant `message`
        // record with text + toolCall + thinking blocks, followed by a
        // *top-level* `toolResult`-role `message` record (not nested inside
        // the assistant's content array) whose own content array holds a
        // single `toolResult`-typed block carrying many redundant id fields
        // that all equal the toolCall's id.
        let content = concat!(
            r#"{"type":"message","id":"m1","message":{"role":"user","content":[{"type":"text","text":"Please check that file"}]}}"#,
            "\n",
            r#"{"type":"message","id":"m2","timestamp":"2026-03-01T00:00:00.000Z","message":{"role":"assistant","model":"claude-opus-4-6","content":[{"type":"text","text":"Let me check that file."},{"type":"toolCall","id":"call_1","name":"bash","arguments":{"cmd":"cat file.txt"}},{"type":"thinking","thinking":"I should read the file first.","thinkingSignature":"sig-1"}]}}"#,
            "\n",
            r#"{"type":"message","id":"m3","timestamp":"2026-03-01T00:00:01.000Z","message":{"role":"toolResult","toolCallId":"call_1","toolName":"bash","content":[{"type":"toolResult","id":"call_1","toolCallId":"call_1","toolUseId":"call_1","tool_use_id":"call_1","toolName":"bash","name":"bash","content":"file contents here","text":"file contents here"}]}}"#,
            "\n",
        );
        write_session(&sessions, "session.jsonl", &[content]);

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::local_default(sessions.clone(), None);
        let convs = connector.scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);
        let conv = &convs[0];

        let roles: Vec<&str> = conv.messages.iter().map(|m| m.role.as_str()).collect();
        assert!(roles.contains(&"tool_call"), "roles: {roles:?}");
        assert!(roles.contains(&"tool_result"), "roles: {roles:?}");
        assert!(roles.contains(&"reasoning"), "roles: {roles:?}");
        assert!(
            !roles.iter().any(|r| matches!(*r, "agent" | "toolResult")),
            "no message may use a pre-6-role name: {roles:?}"
        );

        let assistant = conv
            .messages
            .iter()
            .find(|m| m.role == "assistant")
            .expect("assistant prose message");
        assert!(!assistant.content.contains("[tool:"));
        assert!(assistant.content.contains("Let me check that file."));
        assert_eq!(assistant.author.as_deref(), Some("claude-opus-4-6"));

        let tool_call = conv
            .messages
            .iter()
            .find(|m| m.role == "tool_call")
            .expect("tool_call message");
        assert!(tool_call.content.contains("bash"));
        let tool_call_id = tool_call
            .extra
            .get("tool_call_id")
            .and_then(|v| v.as_str())
            .expect("tool_call extra[\"tool_call_id\"] must be set");
        assert_eq!(tool_call_id, "call_1");
        assert_eq!(
            tool_call.extra["tool_call_args"]["cmd"], "cat file.txt",
            "full args must be preserved in extra, never truncated"
        );

        let tool_result = conv
            .messages
            .iter()
            .find(|m| m.role == "tool_result")
            .expect("tool_result message (from a *top-level* toolResult-role message)");
        assert_eq!(
            tool_result.content, "file contents here",
            "tool_result content must be the FULL result, never truncated"
        );
        assert_eq!(
            tool_result
                .extra
                .get("tool_call_id")
                .and_then(|v| v.as_str()),
            Some(tool_call_id),
            "tool_result pairs to its tool_call via extra[\"tool_call_id\"], not content order"
        );

        let reasoning = conv
            .messages
            .iter()
            .find(|m| m.role == "reasoning")
            .expect("thinking block must become its own reasoning message");
        assert_eq!(reasoning.content, "I should read the file first.");
        assert!(
            !assistant.content.contains("I should read the file first."),
            "thinking must not be inlined into the assistant message anymore"
        );

        // idx must be contiguous 0..N after splitting one raw message-line
        // into several typed messages (spec §3.4) -- required since
        // openclaw did not previously call `reindex_messages`.
        assert!(
            conv.messages
                .iter()
                .enumerate()
                .all(|(i, m)| m.idx as usize == i),
            "idx not contiguous: {:?}",
            conv.messages.iter().map(|m| m.idx).collect::<Vec<_>>()
        );
    }

    #[test]
    fn scan_openclaw_top_level_tool_result_text_block_variant() {
        // Real OpenClaw sessions sometimes emit a top-level `toolResult`
        // message whose content array holds a plain `text`-typed block
        // (not a `toolResult`-typed block) -- the pairing id then lives
        // only on the message itself (`toolCallId`), not in a nested
        // block's fields. Verified against a real
        // ~/.openclaw/agents/*/sessions/*.jsonl session.
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = concat!(
            r#"{"type":"message","id":"m1","message":{"role":"assistant","model":"gpt-5.5","content":[{"type":"toolCall","id":"call_2","name":"exec","arguments":{"cmd":"ls"}}]}}"#,
            "\n",
            r#"{"type":"message","id":"m2","message":{"role":"toolResult","toolCallId":"call_2","toolName":"exec","content":[{"type":"text","text":"file_a.txt\nfile_b.txt"}]}}"#,
            "\n",
        );
        write_session(&sessions, "session.jsonl", &[content]);

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::local_default(sessions.clone(), None);
        let convs = connector.scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);

        let tool_result = convs[0]
            .messages
            .iter()
            .find(|m| m.role == "tool_result")
            .expect("text-block-variant toolResult message must still become tool_result");
        assert_eq!(tool_result.content, "file_a.txt\nfile_b.txt");
        assert_eq!(
            tool_result
                .extra
                .get("tool_call_id")
                .and_then(|v| v.as_str()),
            Some("call_2"),
            "pairing id must come from the message-level toolCallId when the \
             nested block carries no id fields of its own"
        );
    }

    #[test]
    fn scan_openclaw_top_level_empty_tool_result_still_emitted_with_pairing() {
        // Completeness policy (uniform w/ claude_code.rs + codex.rs): never
        // drop a structural item for empty content. Real OpenClaw sessions
        // contain 71 empty toolResults (all `content:[]` with a
        // `toolCallId`, e.g. `update_plan` whose result body is empty) --
        // dropping them silently breaks the tool_call<->tool_result pairing
        // chain. The empty tool_result must still be emitted, keeping its
        // pairing id.
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = concat!(
            r#"{"type":"message","id":"m1","message":{"role":"assistant","model":"gpt-5.5","content":[{"type":"toolCall","id":"call_9","name":"update_plan","arguments":{"plan":"x"}}]}}"#,
            "\n",
            r#"{"type":"message","id":"m2","message":{"role":"toolResult","toolCallId":"call_9","toolName":"update_plan","content":[]}}"#,
            "\n",
        );
        write_session(&sessions, "session.jsonl", &[content]);

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::local_default(sessions.clone(), None);
        let convs = connector.scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);

        let tool_result = convs[0]
            .messages
            .iter()
            .find(|m| m.role == "tool_result")
            .expect("empty top-level toolResult must STILL be emitted (completeness policy)");
        assert_eq!(
            tool_result.content, "",
            "empty toolResult content is preserved as empty, not dropped"
        );
        assert_eq!(
            tool_result
                .extra
                .get("tool_call_id")
                .and_then(|v| v.as_str()),
            Some("call_9"),
            "empty tool_result keeps its pairing id so the tool_call<->tool_result chain survives"
        );
    }

    #[test]
    fn scan_openclaw_content_block_empty_tool_result_still_emitted_with_pairing() {
        // Same completeness policy for the content-block toolResult path (a
        // toolResult block nested inside a non-toolResult-role message): an
        // empty body must still emit a tool_result carrying its pairing id.
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = concat!(
            r#"{"type":"message","id":"m1","message":{"role":"assistant","model":"gpt-5.5","content":[{"type":"text","text":"done"},{"type":"toolResult","id":"call_7","toolCallId":"call_7","toolUseId":"call_7","tool_use_id":"call_7","content":""}]}}"#,
            "\n",
        );
        write_session(&sessions, "session.jsonl", &[content]);

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::local_default(sessions.clone(), None);
        let convs = connector.scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);

        let tool_result = convs[0]
            .messages
            .iter()
            .find(|m| m.role == "tool_result")
            .expect("empty content-block toolResult must STILL be emitted (completeness policy)");
        assert_eq!(tool_result.content, "");
        assert_eq!(
            tool_result
                .extra
                .get("tool_call_id")
                .and_then(|v| v.as_str()),
            Some("call_7"),
            "empty content-block tool_result keeps its pairing id"
        );
    }

    #[test]
    fn scan_openclaw_compaction_summary_becomes_assistant_with_compaction_raw_role() {
        // Real OpenClaw sessions emit `compaction` wrapper events carrying a
        // substantive multi-paragraph `summary` (a running digest of the
        // conversation), directly analogous to claude's `away_summary`.
        // It becomes assistant while preserving raw_role=compaction.
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = concat!(
            r#"{"type":"message","id":"m1","timestamp":"2026-03-01T00:00:00.000Z","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#,
            "\n",
            r###"{"type":"compaction","id":"c1","parentId":"m1","timestamp":"2026-03-01T00:00:01.000Z","summary":"## Decisions\n- Kept the plan.\n- Shipped it.","tokensBefore":12000}"###,
            "\n",
        );
        write_session(&sessions, "session.jsonl", &[content]);

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::local_default(sessions.clone(), None);
        let convs = connector.scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);

        let compaction = convs[0]
            .messages
            .iter()
            .find(|m| m.extra.get("raw_role") == Some(&Value::String("compaction".to_string())))
            .expect("compaction summary must become an assistant message, not be dropped");
        assert_eq!(
            compaction.content,
            "## Decisions\n- Kept the plan.\n- Shipped it."
        );
        assert_eq!(compaction.role, "assistant");
        assert!(
            compaction.author.is_none(),
            "assistant message from a compaction summary has no model author"
        );

        // idx stays contiguous 0..N after the compaction assistant message is
        // routed through the same vector as the message(s) (spec §3.4).
        assert!(
            convs[0]
                .messages
                .iter()
                .enumerate()
                .all(|(i, m)| usize::try_from(m.idx).unwrap() == i)
        );
    }

    #[test]
    fn scan_openclaw_compaction_without_summary_is_dropped() {
        // A compaction event with an absent/empty `summary` carries no
        // substantive content -> drop (no compaction message).
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        let content = concat!(
            r#"{"type":"message","id":"m1","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#,
            "\n",
            r#"{"type":"compaction","id":"c1","summary":"","tokensBefore":12000}"#,
            "\n",
            r#"{"type":"compaction","id":"c2","tokensBefore":12000}"#,
            "\n",
        );
        write_session(&sessions, "session.jsonl", &[content]);

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::local_default(sessions.clone(), None);
        let convs = connector.scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);
        assert!(
            convs[0]
                .messages
                .iter()
                .all(|m| m.extra.get("raw_role") != Some(&Value::String("compaction".to_string()))),
            "empty/absent compaction summary must not emit a compaction message"
        );
        assert_eq!(convs[0].messages.len(), 1);
    }

    #[test]
    fn scan_skips_non_message_types() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "session2.jsonl",
            &[
                r#"{"type":"session","id":"s1","timestamp":"2026-02-01T16:00:00.000Z","cwd":"/"}"#,
                r#"{"type":"model_change","model":"gpt-5"}"#,
                r#"{"type":"thinking_level_change","level":"high"}"#,
                r#"{"type":"message","id":"m1","timestamp":"2026-02-01T16:00:01.000Z","message":{"role":"user","content":"Only message"}}"#,
                r#"{"type":"custom","data":"something"}"#,
            ],
        );

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::local_default(sessions.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].content, "Only message");
    }

    #[test]
    fn scan_handles_empty_and_invalid_lines() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();

        write_session(
            &sessions,
            "bad.jsonl",
            &[
                "",
                "not-json",
                r#"{"type":"message","id":"m1","timestamp":"2026-02-01T16:00:00.000Z","message":{"role":"user","content":"Valid"}}"#,
                r#"{"type":"message","id":"m2","message":{"role":"assistant","content":""}}"#,
            ],
        );

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::local_default(sessions.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        // Only the valid non-empty message should appear
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].content, "Valid");
    }

    #[test]
    fn agents_root_path_construction() {
        if let Some(home) = dirs::home_dir() {
            assert_eq!(
                OpenClawConnector::agents_root().unwrap(),
                home.join(".openclaw").join("agents")
            );
        }
    }

    #[test]
    fn find_dirs_empty_root() {
        let tmp = TempDir::new().unwrap();
        let agents_root = tmp.path().join("agents");
        fs::create_dir_all(&agents_root).unwrap();
        tracing::debug!("Scanning agents root: {}", agents_root.display());
        let dirs = OpenClawConnector::find_agent_session_dirs_at(&agents_root);
        assert!(dirs.is_empty());
    }

    #[test]
    fn find_dirs_no_sessions_subdir() {
        let tmp = TempDir::new().unwrap();
        let agents_root = tmp.path().join("agents");
        fs::create_dir_all(agents_root.join("alice")).unwrap();
        let dirs = OpenClawConnector::find_agent_session_dirs_at(&agents_root);
        assert!(dirs.is_empty());
    }

    #[test]
    fn find_dirs_one_agent() {
        let tmp = TempDir::new().unwrap();
        let agents_root = tmp.path().join("agents");
        let alice = agents_root.join("alice").join("sessions");
        fs::create_dir_all(&alice).unwrap();

        let dirs = OpenClawConnector::find_agent_session_dirs_at(&agents_root);
        assert_eq!(dirs, vec![alice]);
    }

    #[test]
    fn find_dirs_multiple_agents_sorted() {
        let tmp = TempDir::new().unwrap();
        let agents_root = tmp.path().join("agents");
        fs::create_dir_all(agents_root.join("charlie").join("sessions")).unwrap();
        fs::create_dir_all(agents_root.join("alice").join("sessions")).unwrap();
        fs::create_dir_all(agents_root.join("bob").join("sessions")).unwrap();

        let dirs = OpenClawConnector::find_agent_session_dirs_at(&agents_root);
        let discovered: Vec<String> = dirs
            .iter()
            .filter_map(|p| {
                p.parent()
                    .and_then(|pp| pp.file_name())
                    .and_then(|n| n.to_str())
                    .map(String::from)
            })
            .collect();
        assert_eq!(
            discovered,
            vec![
                "alice".to_string(),
                "bob".to_string(),
                "charlie".to_string()
            ]
        );
    }

    #[test]
    fn find_dirs_max_depth_ignores_deep_nesting() {
        let tmp = TempDir::new().unwrap();
        let agents_root = tmp.path().join("agents");
        fs::create_dir_all(agents_root.join("alice").join("sessions")).unwrap();
        fs::create_dir_all(
            agents_root
                .join("nested")
                .join("too")
                .join("deep")
                .join("sessions"),
        )
        .unwrap();

        let dirs = OpenClawConnector::find_agent_session_dirs_at(&agents_root);
        assert_eq!(dirs.len(), 1);
        assert!(dirs[0].to_string_lossy().contains(&format!(
            "{}alice{}",
            std::path::MAIN_SEPARATOR,
            std::path::MAIN_SEPARATOR
        )));
    }

    #[test]
    fn session_files_are_sorted_for_deterministic_scan_order() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        write_session(
            &sessions,
            "z-last.jsonl",
            &[r#"{"type":"message","message":{"role":"user","content":"z"}}"#],
        );
        write_session(
            &sessions,
            "a-first.jsonl",
            &[r#"{"type":"message","message":{"role":"user","content":"a"}}"#],
        );

        let files = OpenClawConnector::session_files(&sessions);
        let file_names: Vec<String> = files
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
            .collect();

        assert_eq!(
            file_names,
            vec!["a-first.jsonl".to_string(), "z-last.jsonl".to_string()]
        );
    }

    #[cfg(unix)]
    #[test]
    fn find_dirs_symlink_skipped() {
        use std::os::unix::fs::symlink;

        let tmp = TempDir::new().unwrap();
        let agents_root = tmp.path().join("agents");
        let real_agent = tmp.path().join("real_alice");
        fs::create_dir_all(real_agent.join("sessions")).unwrap();
        fs::create_dir_all(&agents_root).unwrap();
        symlink(&real_agent, agents_root.join("alice_link")).unwrap();

        let dirs = OpenClawConnector::find_agent_session_dirs_at(&agents_root);
        assert!(dirs.is_empty());
    }

    #[test]
    fn detect_reports_agent_names() {
        let tmp = TempDir::new().unwrap();
        let agents_root = tmp.path().join("agents");
        fs::create_dir_all(agents_root.join("alice").join("sessions")).unwrap();
        fs::create_dir_all(agents_root.join("bob").join("sessions")).unwrap();

        let detection = OpenClawConnector::detect_from_agents_root(&agents_root);
        assert!(detection.detected);
        assert_eq!(detection.root_paths.len(), 2);
        let joined = detection.evidence.join(" | ");
        assert!(joined.contains("discovered 2 agent session dirs"));
        assert!(joined.contains("alice"));
        assert!(joined.contains("bob"));
    }

    #[test]
    fn detect_zero_agents() {
        let tmp = TempDir::new().unwrap();
        let agents_root = tmp.path().join("agents");
        fs::create_dir_all(&agents_root).unwrap();

        let detection = OpenClawConnector::detect_from_agents_root(&agents_root);
        assert!(detection.detected);
        assert!(detection.root_paths.is_empty());
        assert!(
            detection
                .evidence
                .iter()
                .any(|line| line.contains("discovered 0 agent session dirs"))
        );
    }

    #[test]
    fn scan_multiple_agents() {
        let tmp = TempDir::new().unwrap();
        let alice_sessions = tmp.path().join(".openclaw/agents/alice/sessions");
        let bob_sessions = tmp.path().join(".openclaw/agents/bob/sessions");
        fs::create_dir_all(&alice_sessions).unwrap();
        fs::create_dir_all(&bob_sessions).unwrap();
        write_minimal_openclaw_session(&alice_sessions, "alice.jsonl", "/tmp/alice", "hello alice");
        write_minimal_openclaw_session(&bob_sessions, "bob.jsonl", "/tmp/bob", "hello bob");

        let connector = OpenClawConnector::new();
        let ctx = ctx_with_root(tmp.path());
        let mut convs = connector.scan(&ctx).unwrap();
        convs.sort_by(|a, b| a.agent_slug.cmp(&b.agent_slug));

        assert_eq!(convs.len(), 2);
        assert_eq!(convs[0].agent_slug, "openclaw/alice");
        assert_eq!(convs[1].agent_slug, "openclaw/bob");
    }

    #[test]
    fn scan_agent_identity_preserved() {
        let tmp = TempDir::new().unwrap();
        let alice_sessions = tmp.path().join(".openclaw/agents/alice/sessions");
        fs::create_dir_all(&alice_sessions).unwrap();
        write_minimal_openclaw_session(&alice_sessions, "s1.jsonl", "/tmp/alice", "from alice");

        let connector = OpenClawConnector::new();
        let ctx = ctx_with_root(tmp.path());
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].agent_slug, "openclaw/alice");
        assert_eq!(convs[0].external_id.as_deref(), Some("alice/s1"));
    }

    #[test]
    fn scan_agent_metadata_present() {
        let tmp = TempDir::new().unwrap();
        let alice_sessions = tmp.path().join(".openclaw/agents/alice/sessions");
        fs::create_dir_all(&alice_sessions).unwrap();
        write_minimal_openclaw_session(
            &alice_sessions,
            "meta.jsonl",
            "/tmp/alice",
            "metadata check",
        );

        let connector = OpenClawConnector::new();
        let ctx = ctx_with_root(tmp.path());
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0]
                .metadata
                .get("agent_directory")
                .and_then(|v| v.as_str()),
            Some("alice")
        );
    }

    #[test]
    fn scan_mixed_valid_invalid_across_agents() {
        let tmp = TempDir::new().unwrap();
        let alice_sessions = tmp.path().join(".openclaw/agents/alice/sessions");
        let bob_sessions = tmp.path().join(".openclaw/agents/bob/sessions");
        fs::create_dir_all(&alice_sessions).unwrap();
        fs::create_dir_all(&bob_sessions).unwrap();
        write_session(
            &alice_sessions,
            "bad.jsonl",
            &["not-json", "still-not-json"],
        );
        write_minimal_openclaw_session(&bob_sessions, "good.jsonl", "/tmp/bob", "valid from bob");

        let connector = OpenClawConnector::new();
        let ctx = ctx_with_root(tmp.path());
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].agent_slug, "openclaw/bob");
    }

    #[test]
    fn scan_single_agent_unchanged_slug() {
        let tmp = TempDir::new().unwrap();
        let sessions = tmp.path().join(".openclaw/agents/openclaw/sessions");
        fs::create_dir_all(&sessions).unwrap();
        write_minimal_openclaw_session(&sessions, "single.jsonl", "/tmp/openclaw", "legacy mode");

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::local_default(sessions.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].agent_slug, "openclaw");
        assert_eq!(convs[0].external_id.as_deref(), Some("single"));
        assert_eq!(
            convs[0]
                .metadata
                .get("agent_directory")
                .and_then(|v| v.as_str()),
            Some("openclaw")
        );
    }

    #[test]
    fn scan_with_explicit_agent_root_path() {
        let tmp = TempDir::new().unwrap();
        let agent_root = tmp.path().join(".openclaw/agents/alice");
        let sessions = agent_root.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        write_minimal_openclaw_session(&sessions, "root.jsonl", "/tmp/alice", "explicit root");

        let connector = OpenClawConnector::new();
        let ctx = ScanContext::with_roots(
            tmp.path().to_path_buf(),
            vec![super::super::ScanRoot::local(agent_root)],
            None,
        );
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].agent_slug, "openclaw/alice");
    }
}
