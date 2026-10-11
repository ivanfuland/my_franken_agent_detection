#![cfg(feature = "connectors")]
use franken_agent_detection::*;
use sha2::{Digest, Sha256};
use std::fs;

// Input frames and expected literals are independent of the product parser.
fn context(bytes: &[u8], format: SourceFormat) -> (tempfile::TempDir, SealedInputContext) {
    let dir = tempfile::tempdir().unwrap();
    let sealed_locator = dir.path().join("blob");
    fs::write(&sealed_locator, bytes).unwrap();
    let mut framing = Sha256::new();
    framing.update(b"cass.source-framing/1");
    let mut offset = 0usize;
    let mut records = 0;
    let mut blanks = 0;
    let mut blank_bytes = 0;
    let frames: Vec<&[u8]> = match format {
        SourceFormat::Jsonl => bytes.split_inclusive(|byte| *byte == b'\n').collect(),
        SourceFormat::JsonDocument => vec![bytes],
    };
    for (line, frame) in frames.into_iter().enumerate() {
        let raw = if format == SourceFormat::Jsonl {
            frame
                .strip_suffix(b"\r\n")
                .or_else(|| frame.strip_suffix(b"\n"))
                .unwrap_or(frame)
        } else {
            frame
        };
        if format == SourceFormat::Jsonl && raw.iter().all(|b| b" \t\r\n".contains(b)) {
            blanks += 1;
            blank_bytes += raw.len() as u64;
        } else {
            records += 1;
            let digest = format!("{:x}", Sha256::digest(raw));
            let leaf = serde_json::to_vec(&[
                (line + 1).to_string(),
                String::new(),
                offset.to_string(),
                (offset + raw.len()).to_string(),
                digest,
            ])
            .unwrap();
            framing.update((leaf.len() as u64).to_be_bytes());
            framing.update(leaf);
        }
        offset += frame.len();
    }
    let origin = SourceOrigin {
        origin: Origin::local(),
        source_version: None,
    };
    let discovery = SourceInputDescriptor {
        connector: "claude_code".into(),
        scan_root: "/fixture/projects".into(),
        source_path: "/fixture/projects/session.jsonl".into(),
        relative_locator: "session.jsonl".into(),
        role: SourceInputRole::PrimarySessionLog,
        origin: origin.clone(),
        platform: None,
        size_bytes: None,
        modified_at_ms: None,
    };
    let parser_context = SourceParserContext {
        schema: "cass.source-parser-context/1".into(),
        connector: discovery.connector.clone(),
        roots: vec![SourceParserRoot {
            path: discovery.scan_root.clone(),
            origin,
            platform: None,
            workspace_rewrites: vec![],
        }],
        relative_locator: discovery.relative_locator.clone(),
        role: discovery.role,
        effective_config_digest: None,
    };
    let mut sealed = SealedInputContext {
        input_key: Sha256Digest::of(b""),
        identity_namespace: "fixture".into(),
        parser_context_digest: parser_context.digest().unwrap(),
        content_digest: ContentDigest::of(bytes),
        byte_len: bytes.len() as u64,
        format,
        normalizer_id: "test-normalizer/1".into(),
        safety_contract_id: "test-safety/1".into(),
        sealed_locator,
        sealed_ref: SealedSourceRef {
            transport: "frozen_file_unmirrored".into(),
            blob_algorithm: None,
            blob_digest: None,
            blob_relative_path: None,
        },
        expected_manifest: SourceExpectedManifest {
            verifier_version: "cass.source-verifier/1".into(),
            format,
            records,
            blank_lines: blanks,
            blank_bytes,
            framing_sha256: Sha256Digest::from_hex(&format!("{:x}", framing.finalize())).unwrap(),
            complete: true,
        },
        discovery,
        parser_context,
    };
    sealed.input_key = sealed.identity_for(sealed.content_digest).unwrap();
    (dir, sealed)
}

fn source(
    bytes: &[u8],
    format: SourceFormat,
    connector: &str,
) -> (tempfile::TempDir, SealedInputContext) {
    let (dir, mut sealed) = context(bytes, format);
    sealed.discovery.connector = connector.into();
    sealed.parser_context.connector = connector.into();
    sealed.parser_context_digest = sealed.parser_context.digest().unwrap();
    sealed.input_key = sealed.identity_for(sealed.content_digest).unwrap();
    (dir, sealed)
}

#[derive(Default)]
struct Sink {
    records: Vec<SourceRecordOutcome>,
    conversations: Vec<SourceMappedConversation>,
    claims: Vec<(SourceRecordLocator, Vec<SourceMappingClaim>)>,
    diagnostics: Vec<SourceMappingDiagnostic>,
    stop_at: Option<&'static str>,
    error: bool,
}

impl Sink {
    fn control(&self, stage: &str) -> Result<SourceSinkControl, SourceSinkFailure> {
        if self.stop_at == Some(stage) {
            if self.error {
                Err(SourceSinkFailure {
                    code: SourceSinkFailureCode::Rejected,
                    detail_digest: None,
                })
            } else {
                Ok(SourceSinkControl::Stop)
            }
        } else {
            Ok(SourceSinkControl::Continue)
        }
    }
    fn canonical(&self) -> Vec<&SourceMappingClaim> {
        self.claims
            .iter()
            .flat_map(|(_, claims)| claims)
            .filter(|claim| claim.mapping_state == "canonical")
            .collect()
    }
}

impl SourceRecordSink for Sink {
    fn on_record(
        &mut self,
        record: SourceRecordOutcome,
    ) -> Result<SourceSinkControl, SourceSinkFailure> {
        self.records.push(record);
        self.control("record")
    }
}
impl SourceMappingSink for Sink {
    fn on_record_claims(
        &mut self,
        locator: SourceRecordLocator,
        claims: Vec<SourceMappingClaim>,
    ) -> Result<SourceSinkControl, SourceSinkFailure> {
        self.claims.push((locator, claims));
        self.control("claims")
    }
    fn on_conversation(
        &mut self,
        item: SourceMappedConversation,
    ) -> Result<SourceSinkControl, SourceSinkFailure> {
        self.conversations.push(item);
        self.control("conversation")
    }
    fn on_diagnostic(
        &mut self,
        item: SourceMappingDiagnostic,
    ) -> Result<SourceSinkControl, SourceSinkFailure> {
        self.diagnostics.push(item);
        self.control("diagnostic")
    }
}

/// A scan context rooted at one explicit file, so the legacy path reads exactly
/// the bytes the sealed path is given.
fn scan_context(path: &std::path::Path) -> ScanContext {
    ScanContext {
        data_dir: path.parent().unwrap().to_path_buf(),
        scan_roots: vec![ScanRoot::local(path.to_path_buf())],
        since_ts: None,
    }
}

/// A sealed context whose discovery identity is the real file on disk.
fn source_at(
    bytes: &[u8],
    format: SourceFormat,
    connector: &str,
    path: &std::path::Path,
) -> (tempfile::TempDir, SealedInputContext) {
    source_with_locator(
        bytes,
        format,
        connector,
        path.parent().unwrap(),
        &path.file_name().unwrap().to_string_lossy(),
        path,
    )
}

/// A sealed context whose discovery mirror is an explicit root, locator and
/// original path — the only facts a source-aware read may use for identity.
fn source_with_locator(
    bytes: &[u8],
    format: SourceFormat,
    connector: &str,
    scan_root: &std::path::Path,
    relative_locator: &str,
    source_path: &std::path::Path,
) -> (tempfile::TempDir, SealedInputContext) {
    let (dir, mut sealed) = context(bytes, format);
    sealed.discovery.connector = connector.into();
    sealed.parser_context.connector = connector.into();
    sealed.discovery.scan_root = scan_root.to_path_buf();
    sealed.discovery.source_path = source_path.to_path_buf();
    sealed.discovery.relative_locator = relative_locator.into();
    sealed.parser_context.roots = vec![SourceParserRoot {
        path: sealed.discovery.scan_root.clone(),
        origin: sealed.discovery.origin.clone(),
        platform: None,
        workspace_rewrites: vec![],
    }];
    sealed.parser_context.relative_locator = relative_locator.into();
    sealed.parser_context_digest = sealed.parser_context.digest().unwrap();
    sealed.input_key = sealed.identity_for(sealed.content_digest).unwrap();
    (dir, sealed)
}

/// Run source-aware mapping against an already-built sealed context.
fn scan_sealed(connector: &dyn Connector, sealed: &SealedInputContext) -> Sink {
    let mut sink = Sink::default();
    let finish = connector.scan_source_aware(sealed, &mut sink).unwrap();
    assert!(finish.reached_eof);
    assert_eq!(finish.bytes_read, sealed.byte_len);
    assert_eq!(finish.records_seen, sink.records.len() as u64);
    let wrapper: Vec<_> = sink.conversations.iter().flat_map(|c| &c.claims).collect();
    assert_eq!(wrapper, sink.canonical());
    sink
}

fn scan(connector: &dyn Connector, bytes: &[u8], format: SourceFormat, name: &str) -> Sink {
    let (_dir, sealed) = source(bytes, format, name);
    let mut sink = Sink::default();
    let finish = connector.scan_source_aware(&sealed, &mut sink).unwrap();
    assert!(finish.reached_eof);
    assert_eq!(finish.bytes_read, bytes.len() as u64);
    assert_eq!(finish.records_seen, sink.records.len() as u64);
    let wrapper: Vec<_> = sink.conversations.iter().flat_map(|c| &c.claims).collect();
    assert_eq!(wrapper, sink.canonical());
    sink
}

#[test]
fn two_source_claims_and_legacy() {
    let sink = scan(
        &ClaudeCodeConnector::new(),
        include_bytes!("fixtures/source_mapping/claude-blocks.jsonl"),
        SourceFormat::Jsonl,
        "claude_code",
    );
    let messages = &sink.conversations[0].conversation.messages;
    assert_eq!(
        messages
            .iter()
            .map(|m| (m.idx, m.role.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (0, "user"),
            (1, "assistant"),
            (2, "reasoning"),
            (3, "tool_call"),
            (4, "assistant"),
            (5, "tool_result")
        ]
    );
    assert_eq!(messages[1].content, "first\nlast");
    assert_eq!(messages[2].content, "");
    assert_eq!(messages[4].extra["raw_role"], "system");
    let claims = sink.canonical();
    assert_eq!(
        claims[1]
            .source_block_pointers
            .iter()
            .map(SourcePointer::as_str)
            .collect::<Vec<_>>(),
        vec!["/message/content/0", "/message/content/3"]
    );
    assert_eq!(
        claims[2].source_block_pointers[0].as_str(),
        "/message/content/1"
    );
    assert_eq!(claims[2].projection_pointer.as_deref(), Some("/messages/2"));
    assert_eq!(sink.claims[3].1[0].mapping_state, "standalone");
    assert_eq!(sink.claims[4].1[0].conversation_ordinal, None);

    let sink = scan(
        &ClaudeCodeConnector::new(),
        include_bytes!("fixtures/source_mapping/claude-legacy.json"),
        SourceFormat::JsonDocument,
        "claude_code",
    );
    assert_eq!(
        sink.conversations[0].conversation.external_id.as_deref(),
        Some("session.jsonl")
    );
    assert_eq!(
        sink.canonical()
            .iter()
            .map(|c| c.source_block_pointers[0].as_str())
            .collect::<Vec<_>>(),
        vec!["/messages/0", "/messages/2"]
    );

    let modern = include_str!("fixtures/source_mapping/codex-modern.jsonl")
        .lines()
        .take(6)
        .collect::<Vec<_>>()
        .join("\n");
    let sink = scan(
        &CodexConnector::new(),
        modern.as_bytes(),
        SourceFormat::Jsonl,
        "codex",
    );
    let messages = &sink.conversations[0].conversation.messages;
    assert_eq!(
        messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
        vec!["assistant", "tool_result", "reasoning"]
    );
    assert_eq!(messages[0].extra["cass"]["token_usage"]["input_tokens"], 7);
    assert_eq!(sink.claims[3].1[0].mapping_state, "standalone");
    assert_eq!(sink.claims[3].1[0].final_message_idx, None);
    let sink = scan(
        &CodexConnector::new(),
        include_bytes!("fixtures/source_mapping/codex-legacy.json"),
        SourceFormat::JsonDocument,
        "codex",
    );
    assert!(sink.conversations.is_empty());
    assert!(sink.records[0].value.is_some());
    assert_eq!(sink.claims[0].1[0].mapping_state, "standalone");
}

#[test]
fn whole_input_rejection_retains_ir_and_publishes_no_claims() {
    let lines: Vec<_> = include_str!("fixtures/source_mapping/codex-modern.jsonl")
        .lines()
        .collect();
    let reasons = [
        "projection_reserved_raw_role",
        "projection_encrypted_content_invalid",
        "projection_payload_type",
        "projection_reasoning_shape",
        "projection_agent_message_shape",
        "projection_metadata_shape",
        "projection_token_count_shape",
    ];
    let reject = |line: &str, reason: &str| {
        let bytes = format!("{}\n{line}\n{}\n", lines[1], lines[1]);
        let (_dir, sealed) = source(bytes.as_bytes(), SourceFormat::Jsonl, "codex");
        let mut sink = Sink::default();
        let failure = CodexConnector::new()
            .scan_source_aware(&sealed, &mut sink)
            .unwrap_err();
        assert_eq!(failure.kind, SourceReadFailureKind::ProjectionRejected);
        assert!(failure.partial_finish.reached_eof);
        assert!(failure.next_cursor.is_none());
        assert_eq!(failure.partial_finish.mapping_diagnostics, 1);
        assert_eq!(sink.records.len(), 3);
        assert!(sink.conversations.is_empty());
        assert!(sink.claims.is_empty());
        assert_eq!(
            sink.records[1].diagnostics[0].detail.as_deref(),
            Some(reason)
        );
        assert!(
            sink.records
                .iter()
                .all(|r| r.parse_state == SourceParseState::Parsed && r.value.is_some())
        );
    };
    for (line, reason) in lines[6..].iter().zip(reasons) {
        reject(line, reason);
    }
    // Independent literal, deliberately not taken from the fixture: the same
    // `total_token_usage` shape with `cached_input_tokens` and
    // `reasoning_output_tokens` missing must stay a whole-input projection
    // rejection with the IR retained and no claims. Keeping it here is what makes
    // the codex-modern.jsonl calibration a calibrated-out *binding* shape rather
    // than a narrowing of the rejection domain.
    reject(
        r#"{"type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":7,"cached_input_tokens":1,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21},"model_context_window":128000,"total_token_usage":{"input_tokens":70,"output_tokens":110,"total_tokens":180}},"rate_limits":null}}"#,
        "projection_token_count_shape",
    );
}

#[test]
fn mapping_delivery_stop_reports_real_eof() {
    let bytes = include_bytes!("fixtures/source_mapping/claude-blocks.jsonl");
    let (_dir, sealed) = source(bytes, SourceFormat::Jsonl, "claude_code");
    for stage in ["record", "conversation", "claims"] {
        for error in [false, true] {
            let mut sink = Sink {
                stop_at: Some(stage),
                error,
                ..Sink::default()
            };
            let failure = ClaudeCodeConnector::new()
                .scan_source_aware(&sealed, &mut sink)
                .unwrap_err();
            assert_eq!(
                failure.kind,
                if error {
                    SourceReadFailureKind::Sink
                } else {
                    SourceReadFailureKind::Stopped
                }
            );
            assert_eq!(failure.partial_finish.reached_eof, stage != "record");
            assert!(failure.next_cursor.is_none());
            if stage != "record" {
                assert_eq!(failure.partial_finish.bytes_read, bytes.len() as u64);
            }
        }
    }
}

#[test]
fn sealed_transport_context_stability() {
    let bytes = include_bytes!("fixtures/source_mapping/claude-metadata.json");
    let (dir, mut sealed) = source(bytes, SourceFormat::JsonDocument, "claude_code");
    sealed.discovery.relative_locator = "claude-code-sessions/metadata.json".into();
    sealed.discovery.source_path = sealed
        .discovery
        .scan_root
        .join(&sealed.discovery.relative_locator);
    sealed.discovery.role = SourceInputRole::MetadataSidecar;
    sealed.parser_context.relative_locator = sealed.discovery.relative_locator.clone();
    sealed.parser_context.role = sealed.discovery.role;
    sealed.parser_context_digest = sealed.parser_context.digest().unwrap();
    sealed.input_key = sealed.identity_for(sealed.content_digest).unwrap();
    let mut first = Sink::default();
    ClaudeCodeConnector::new()
        .scan_source_aware(&sealed, &mut first)
        .unwrap();
    assert_eq!(
        first.conversations[0].conversation.external_id.as_deref(),
        Some("cli-a")
    );
    fs::rename(
        &sealed.sealed_locator,
        dir.path().join("different-name.data"),
    )
    .unwrap();
    sealed.sealed_locator = dir.path().join("different-name.data");
    let mut second = Sink::default();
    ClaudeCodeConnector::new()
        .scan_source_aware(&sealed, &mut second)
        .unwrap();
    assert_eq!(
        serde_json::to_value(&first.conversations[0].conversation).unwrap(),
        serde_json::to_value(&second.conversations[0].conversation).unwrap()
    );
    assert_eq!(first.canonical(), second.canonical());
    // A metadata role alone does not create a Desktop metadata message.
    sealed.discovery.relative_locator = "metadata.json".into();
    sealed.discovery.source_path = sealed.discovery.scan_root.join("metadata.json");
    sealed.parser_context.relative_locator = "metadata.json".into();
    sealed.parser_context_digest = sealed.parser_context.digest().unwrap();
    sealed.input_key = sealed.identity_for(sealed.content_digest).unwrap();
    let mut sink = Sink::default();
    ClaudeCodeConnector::new()
        .scan_source_aware(&sealed, &mut sink)
        .unwrap();
    assert!(sink.conversations.is_empty());
}

#[test]
fn default_source_capability_is_explicitly_unsupported() {
    struct Legacy;
    impl Connector for Legacy {
        fn detect(&self) -> DetectionResult {
            DetectionResult::not_found()
        }
        fn scan(&self, _ctx: &ScanContext) -> anyhow::Result<Vec<NormalizedConversation>> {
            panic!("must not call legacy scan")
        }
    }
    let (_dir, sealed) = source(b"{}", SourceFormat::JsonDocument, "claude_code");
    let connector: Box<dyn Connector + Send> = Box::new(Legacy);
    let failure = connector
        .scan_source_aware(&sealed, &mut Sink::default())
        .unwrap_err();
    assert_eq!(failure.kind, SourceReadFailureKind::Unsupported);
    assert!(!failure.partial_finish.reached_eof);
}

#[test]
fn projection_rejection_delivers_diagnostic_callback() {
    // The whole-input rejection has one registered, value-free reason. That same
    // reason must reach the mapping sink, and a Stop or error from that callback
    // must keep the real EOF, drop the cursor and leave the input partial.
    let lines: Vec<_> = include_str!("fixtures/source_mapping/codex-modern.jsonl")
        .lines()
        .collect();
    let bytes = format!("{}\n{}\n{}\n", lines[1], lines[12], lines[1]);
    for error in [false, true] {
        let (_dir, sealed) = source(bytes.as_bytes(), SourceFormat::Jsonl, "codex");
        let mut sink = Sink {
            stop_at: Some("diagnostic"),
            error,
            ..Sink::default()
        };
        let failure = CodexConnector::new()
            .scan_source_aware(&sealed, &mut sink)
            .unwrap_err();
        assert_eq!(sink.diagnostics.len(), 1);
        assert_eq!(sink.diagnostics[0].code, "source_projection_rejected");
        assert_eq!(
            failure.kind,
            if error {
                SourceReadFailureKind::Sink
            } else {
                SourceReadFailureKind::Stopped
            }
        );
        assert!(failure.partial_finish.reached_eof);
        assert!(failure.next_cursor.is_none());
        assert_eq!(failure.partial_finish.bytes_read, bytes.len() as u64);
        assert!(sink.conversations.is_empty() && sink.claims.is_empty());
    }
    // Continue: the rejection itself is still the typed failure.
    let (_dir, sealed) = source(bytes.as_bytes(), SourceFormat::Jsonl, "codex");
    let mut sink = Sink::default();
    let failure = CodexConnector::new()
        .scan_source_aware(&sealed, &mut sink)
        .unwrap_err();
    assert_eq!(sink.diagnostics.len(), 1);
    assert_eq!(sink.diagnostics[0].code, "source_projection_rejected");
    assert_eq!(failure.kind, SourceReadFailureKind::ProjectionRejected);
    assert!(failure.partial_finish.reached_eof);
    assert!(failure.next_cursor.is_none());
    assert_eq!(sink.records.len(), 3);
    assert!(sink.conversations.is_empty() && sink.claims.is_empty());
}

/// Structural equality over every serialized field of a projected conversation,
/// including message `snippets` and `invocations`.
fn conversations_identical(left: &NormalizedConversation, right: &NormalizedConversation) -> bool {
    serde_json::to_value(left).unwrap() == serde_json::to_value(right).unwrap()
}

/// Small inputs the legacy scan already accepted must project to the same
/// conversation through both paths, over the whole serialized structure. Shapes
/// the contract newly supports (oversized records, source-only inputs) are out of
/// scope: the legacy interface is not required to grow them.
#[test]
fn legacy_scan_matches_source_aware_on_shared_shapes() {
    let claude_modern = include_bytes!("fixtures/source_mapping/claude-blocks.jsonl");
    let claude_legacy = include_bytes!("fixtures/source_mapping/claude-legacy.json");
    let codex_modern_all = include_str!("fixtures/source_mapping/codex-modern.jsonl");
    let codex_modern = codex_modern_all
        .lines()
        .take(6)
        .collect::<Vec<_>>()
        .join("\n");
    let codex_legacy = include_bytes!("fixtures/source_mapping/codex-legacy.json");
    let codex_usage_with_cache = br#"{"type":"session_meta","timestamp":"2026-01-01T00:00:00Z","payload":{"id":"session-a","cwd":"/fixture/workspace"}}
{"type":"response_item","timestamp":"2026-01-01T00:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}}
{"type":"event_msg","timestamp":"2026-01-01T00:00:02Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":7,"cached_input_tokens":1,"cache_write_input_tokens":3,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21},"model_context_window":128000,"total_token_usage":{"input_tokens":70,"cached_input_tokens":10,"cache_write_input_tokens":30,"output_tokens":110,"reasoning_output_tokens":20,"total_tokens":180}},"rate_limits":null}}
"#;
    let codex_usage_without_cache = br#"{"type":"session_meta","timestamp":"2026-01-01T00:00:00Z","payload":{"id":"session-a","cwd":"/fixture/workspace"}}
{"type":"response_item","timestamp":"2026-01-01T00:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}}
{"type":"event_msg","timestamp":"2026-01-01T00:00:02Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":7,"cached_input_tokens":1,"output_tokens":11,"reasoning_output_tokens":2,"total_tokens":21},"model_context_window":128000,"total_token_usage":{"input_tokens":70,"cached_input_tokens":10,"output_tokens":110,"reasoning_output_tokens":20,"total_tokens":180}},"rate_limits":null}}
"#;

    let cases: [(&str, &[u8], SourceFormat, &str); 6] = [
        (
            "claude_code",
            claude_modern,
            SourceFormat::Jsonl,
            "session.jsonl",
        ),
        (
            "claude_code",
            claude_legacy,
            SourceFormat::JsonDocument,
            "session.json",
        ),
        (
            "codex",
            codex_modern.as_bytes(),
            SourceFormat::Jsonl,
            "rollout-2026-01-01T00-00-00-a.jsonl",
        ),
        (
            "codex",
            codex_legacy,
            SourceFormat::JsonDocument,
            "rollout-2026-01-01T00-00-00-a.json",
        ),
        // F6: the optional `cache_write_input_tokens` may be present or absent on
        // either usage object. Both sides must keep the message-projection
        // convention and preserve the original IR; neither changes the legacy
        // token strategy.
        (
            "codex",
            codex_usage_with_cache,
            SourceFormat::Jsonl,
            "rollout-2026-01-01T00-00-00-b.jsonl",
        ),
        (
            "codex",
            codex_usage_without_cache,
            SourceFormat::Jsonl,
            "rollout-2026-01-01T00-00-00-c.jsonl",
        ),
    ];

    let mut compared = 0usize;
    for (name, bytes, format, file_name) in cases {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(file_name);
        fs::write(&path, bytes).unwrap();

        let legacy = match name {
            "claude_code" => ClaudeCodeConnector::new()
                .scan(&scan_context(&path))
                .unwrap(),
            _ => CodexConnector::new().scan(&scan_context(&path)).unwrap(),
        };
        let (_tmp, sealed) = source_at(bytes, format, name, &path);
        let sink = match name {
            "claude_code" => scan_sealed(&ClaudeCodeConnector::new(), &sealed),
            _ => scan_sealed(&CodexConnector::new(), &sealed),
        };

        // A shape that yields no conversation on either path is equivalence too:
        // the legacy scan accepted the input and produced nothing, and so does
        // the source-aware path.
        assert_eq!(
            legacy.len(),
            sink.conversations.len(),
            "{name}/{file_name} conversation count"
        );
        for (index, (left, right)) in legacy.iter().zip(sink.conversations.iter()).enumerate() {
            let same = conversations_identical(left, &right.conversation);
            assert!(
                same,
                "{name}/{file_name} conversation {index} differs\n legacy = {}\n source = {}",
                serde_json::to_value(left).unwrap(),
                serde_json::to_value(&right.conversation).unwrap()
            );
            if !left.messages.is_empty() {
                compared += 1;
            }
        }
    }
    assert!(
        compared >= 2,
        "expected at least two non-empty conversations to be compared, compared {compared}"
    );
}

/// The parity comparison must reject a changed field, so a green parity run
/// cannot come from comparing nothing. This is a test-only control: no product
/// logic and no gold is involved.
#[test]
fn conversation_parity_comparison_is_not_vacuous() {
    let bytes = include_bytes!("fixtures/source_mapping/claude-blocks.jsonl");
    let (_dir, sealed) = source(bytes, SourceFormat::Jsonl, "claude_code");
    let sink = scan_sealed(&ClaudeCodeConnector::new(), &sealed);
    assert_eq!(sink.conversations.len(), 1);
    let base = sink.conversations[0].conversation.clone();
    assert!(conversations_identical(&base, &base));

    let mut changed_time = base.clone();
    changed_time.ended_at = Some(changed_time.ended_at.unwrap_or(0) + 1);
    assert!(!conversations_identical(&base, &changed_time), "ended_at");

    let mut changed_metadata = base.clone();
    changed_metadata.metadata = serde_json::json!({"source": "different"});
    assert!(
        !conversations_identical(&base, &changed_metadata),
        "metadata"
    );

    let mut changed_message = base.clone();
    changed_message.messages[0].content.push('!');
    assert!(!conversations_identical(&base, &changed_message), "message");
}

/// One rollout line the legacy Codex scan already accepts, so both paths must
/// produce the same single-message conversation.
const CODEX_ROLLOUT_LINE: &[u8] =
    br#"{"type":"response_item","timestamp":"2026-10-10T10:00:00Z","payload":{"role":"user","content":"hello"}}"#;

/// Compare legacy `scan` with the sealed source-aware read for one root shape:
/// both identities must equal `expected` and the whole conversation structure
/// must be identical. Returns the projected conversation.
fn codex_identity(
    legacy: &NormalizedConversation,
    sealed: &SealedInputContext,
    expected: &str,
    label: &str,
) -> NormalizedConversation {
    assert_eq!(
        legacy.external_id.as_deref(),
        Some(expected),
        "legacy {label}"
    );
    let mut sink = scan_sealed(&CodexConnector::new(), sealed);
    assert_eq!(sink.conversations.len(), 1, "{label}");
    assert_eq!(
        sink.conversations[0].conversation.external_id.as_deref(),
        Some(expected),
        "source-aware {label}"
    );
    assert!(conversations_identical(
        legacy,
        &sink.conversations[0].conversation
    ));
    sink.conversations.swap_remove(0).conversation
}

/// Legacy `scan` measures the Codex `external_id` from the sessions directory,
/// so a root whose locator carries a `sessions` component must drop it; a root
/// that already is the sessions directory, or that has none, keeps its full
/// locator. This case builds one `.codex/sessions/YYYY/MM/DD` layout to cover
/// all three root shapes the frozen discovery can express.
#[test]
fn codex_external_id_matches_legacy_for_sessions_root_shapes() {
    let stem = "rollout-2026-10-10T10-00-00-a";
    let file = format!("{stem}.jsonl");
    let expected = format!("2026/10/10/{stem}");

    let dir = tempfile::tempdir().unwrap();
    let codex_dir = dir.path().join(".codex");
    let day = codex_dir
        .join("sessions")
        .join("2026")
        .join("10")
        .join("10");
    fs::create_dir_all(&day).unwrap();
    let rollout = day.join(&file);
    fs::write(&rollout, CODEX_ROLLOUT_LINE).unwrap();

    // Default detection: the connector root is the `.codex` data directory, so
    // the frozen locator keeps the whole `sessions/YYYY/MM/DD/...` path.
    let default_legacy = CodexConnector::new()
        .scan(&ScanContext::local_default(codex_dir.clone(), None))
        .unwrap();
    assert_eq!(default_legacy.len(), 1);
    let (_default_dir, default_sealed) = source_with_locator(
        CODEX_ROLLOUT_LINE,
        SourceFormat::Jsonl,
        "codex",
        &codex_dir,
        &format!("sessions/2026/10/10/{file}"),
        &rollout,
    );
    codex_identity(
        &default_legacy[0],
        &default_sealed,
        &expected,
        "default root",
    );

    // Explicit sessions root: the locator already starts below `sessions`.
    let sessions = codex_dir.join("sessions");
    let legacy = CodexConnector::new()
        .scan(&ScanContext::with_roots(
            codex_dir,
            vec![ScanRoot::local(sessions.clone())],
            None,
        ))
        .unwrap();
    assert_eq!(legacy.len(), 1);
    let (_sessions_dir, sessions_sealed) = source_with_locator(
        CODEX_ROLLOUT_LINE,
        SourceFormat::Jsonl,
        "codex",
        &sessions,
        &format!("2026/10/10/{file}"),
        &rollout,
    );
    codex_identity(&legacy[0], &sessions_sealed, &expected, "sessions root");

    // Explicit root without a `sessions` directory: the locator is the path
    // below the root itself.
    let plain_dir = tempfile::tempdir().unwrap();
    let plain = plain_dir.path().join("plain");
    fs::create_dir_all(&plain).unwrap();
    let plain_rollout = plain.join(&file);
    fs::write(&plain_rollout, CODEX_ROLLOUT_LINE).unwrap();
    let legacy = CodexConnector::new()
        .scan(&ScanContext::with_roots(
            plain.clone(),
            vec![ScanRoot::local(plain.clone())],
            None,
        ))
        .unwrap();
    assert_eq!(legacy.len(), 1);
    let (_plain_dir, plain_sealed) = source_with_locator(
        CODEX_ROLLOUT_LINE,
        SourceFormat::Jsonl,
        "codex",
        &plain,
        &file,
        &plain_rollout,
    );
    codex_identity(&legacy[0], &plain_sealed, stem, "root without sessions");

    // The frozen identity depends on the sealed discovery, not on the original
    // live file. Move that file out of the scanned tree and read the same
    // sealed context again: the whole conversation and its identity must not
    // change.
    fs::rename(&rollout, dir.path().join("moved-away.jsonl")).unwrap();
    assert!(!rollout.exists());
    codex_identity(
        &default_legacy[0],
        &default_sealed,
        &expected,
        "after the live file moved",
    );
}

/// Records a limitation of the frozen discovery, not a fix. Legacy discovery
/// keeps the original root verbatim, so a rollout passed as an explicit file
/// has the file itself as its root, while the sealed contract requires
/// `scan_root.join(relative_locator) == source_path` with a non-empty,
/// dot-free locator. The same file therefore has two legacy roots with two
/// different identities but only one expressible sealed tuple.
#[test]
fn codex_explicit_file_root_identity_is_not_expressible_in_sealed_discovery() {
    let dir = tempfile::tempdir().unwrap();
    let codex_dir = dir.path().join(".codex");
    let day = codex_dir
        .join("sessions")
        .join("2026")
        .join("10")
        .join("10");
    fs::create_dir_all(&day).unwrap();
    let rollout = day.join("rollout-a.jsonl");
    fs::write(&rollout, CODEX_ROLLOUT_LINE).unwrap();

    // Legacy config X: the scan root is the rollout file itself.
    let file_root = CodexConnector::new()
        .scan(&ScanContext::with_roots(
            rollout.clone(),
            vec![ScanRoot::local(rollout.clone())],
            None,
        ))
        .unwrap();
    assert_eq!(file_root.len(), 1);
    assert_eq!(
        file_root[0].external_id.as_deref(),
        Some("2026/10/10/rollout-a")
    );

    // Legacy config Y: the scan root is the file's parent directory. This is
    // the only tuple the sealed contract can carry for the same file, because
    // it requires `scan_root.join(relative) == source_path` with a non-empty,
    // dot-free relative locator.
    let parent_root = CodexConnector::new()
        .scan(&ScanContext::with_roots(
            day.clone(),
            vec![ScanRoot::local(day)],
            None,
        ))
        .unwrap();
    assert_eq!(parent_root.len(), 1);
    assert_eq!(parent_root[0].external_id.as_deref(), Some("rollout-a"));

    // A descriptor faithful to config X cannot even validate: the empty
    // locator is rejected, so the file-root shape has no sealed representation.
    let (_dir, mut unrepresentable) = source_with_locator(
        CODEX_ROLLOUT_LINE,
        SourceFormat::Jsonl,
        "codex",
        &rollout,
        "rollout-a.jsonl",
        &rollout,
    );
    unrepresentable.discovery.relative_locator = String::new();
    unrepresentable.parser_context.relative_locator = String::new();
    unrepresentable.parser_context_digest = unrepresentable.parser_context.digest().unwrap();
    unrepresentable.input_key = unrepresentable
        .identity_for(unrepresentable.content_digest)
        .unwrap();
    assert_eq!(
        unrepresentable.validate().unwrap_err(),
        SourceReadFailureKind::InvalidContext
    );
}
