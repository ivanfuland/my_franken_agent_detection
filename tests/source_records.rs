#![cfg(feature = "connectors")]

use std::fs;

use franken_agent_detection::*;
use serde_json::json;
use sha2::{Digest, Sha256};

// Expected geometry and syntax literals are from the source wire contract,
// independently assembled here without invoking the product reader.
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

#[derive(Default)]
struct Sink {
    records: Vec<SourceRecordOutcome>,
    stop: bool,
    error: bool,
}
impl SourceRecordSink for Sink {
    fn on_record(
        &mut self,
        record: SourceRecordOutcome,
    ) -> Result<SourceSinkControl, SourceSinkFailure> {
        self.records.push(record);
        if self.error {
            Err(SourceSinkFailure {
                code: SourceSinkFailureCode::Rejected,
                detail_digest: None,
            })
        } else if self.stop {
            Ok(SourceSinkControl::Stop)
        } else {
            Ok(SourceSinkControl::Continue)
        }
    }
}

#[test]
fn number_text_before_value() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/source_records/numeric-literals.json"
    ))
    .unwrap();
    for text in fixture["accepted"].as_array().unwrap() {
        let text = text.as_str().unwrap();
        let value = LosslessSourceValue::parse(text.as_bytes()).unwrap();
        let LosslessSourceValue::Number(number) = &value else {
            panic!("wrong value kind")
        };
        assert_eq!(number.as_str(), text);
        let mut canonical = Vec::new();
        value.write_canonical(&mut canonical).unwrap();
        assert_eq!(canonical, text.as_bytes());
        assert_eq!(
            ValidatedJsonNumberText::new(text.into()).unwrap().as_str(),
            text
        );
    }
    for text in fixture["rejected"].as_array().unwrap() {
        assert!(ValidatedJsonNumberText::new(text.as_str().unwrap().into()).is_err());
        assert!(LosslessSourceValue::parse(text.as_str().unwrap().as_bytes()).is_err());
    }
    let raw = br#"{"$serde_json::private::Number":"ordinary","$serde_json::private::RawValue":{"x":-0},"array":[null,false,0,"",{},[]]}"#;
    let value = LosslessSourceValue::parse(raw).unwrap();
    let mut out = Vec::new();
    value.write_canonical(&mut out).unwrap();
    assert_eq!(out, raw);
    out.clear();
    value.write_tagged(&mut out).unwrap();
    let tagged: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(tagged[0], "obj");
    assert_eq!(tagged[1][1][1][1][0][1], json!(["num", "-0"]));
}

#[test]
fn framing_and_finish() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/source_records/jsonl-framing.json")).unwrap();
    let bytes = fixture["source"].as_str().unwrap().as_bytes();
    let (_dir, sealed) = context(bytes, SourceFormat::Jsonl);
    let mut sink = Sink::default();
    let finish = read_source_records(&sealed, None, &mut sink).unwrap();
    let locators: Vec<_> = sink
        .records
        .iter()
        .map(|r| {
            json!([
                r.locator.ordinal,
                r.locator.byte_start,
                r.locator.byte_end,
                r.terminator_len
            ])
        })
        .collect();
    assert_eq!(locators, fixture["locators"].as_array().unwrap().clone());
    assert_eq!(finish.records_seen, 3);
    assert_eq!(finish.blank_lines, 1);
    assert!(finish.reached_eof);
    assert_eq!(finish.bytes_read, bytes.len() as u64);
    assert_eq!(finish.content_digest, ContentDigest::of(bytes));
    for record in &sink.records {
        assert_eq!(
            record.source_record_sha256,
            Sha256Digest::of(
                &bytes[usize::try_from(record.locator.byte_start).unwrap()
                    ..usize::try_from(record.locator.byte_end).unwrap()]
            )
        );
    }
    for format in [SourceFormat::Jsonl, SourceFormat::JsonDocument] {
        let (_dir, sealed) = context(b"", format);
        let mut sink = Sink::default();
        let finish = read_source_records(&sealed, None, &mut sink).unwrap();
        assert!(finish.reached_eof);
        assert_eq!(
            finish.records_seen,
            u64::from(format == SourceFormat::JsonDocument)
        );
        assert_eq!(
            finish.parse_failed,
            u64::from(format == SourceFormat::JsonDocument)
        );
    }
}

#[test]
fn sink_acceptance_and_resume() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/source_records/cursor-and-stop.json")).unwrap();
    let bytes = fixture["source"].as_str().unwrap().as_bytes();
    let (_dir, sealed) = context(bytes, SourceFormat::Jsonl);
    for error in [false, true] {
        let mut sink = Sink {
            error,
            stop: !error,
            ..Sink::default()
        };
        let fail = read_source_records(&sealed, None, &mut sink).unwrap_err();
        assert_eq!(
            fail.kind,
            if error {
                SourceReadFailureKind::Sink
            } else {
                SourceReadFailureKind::Stopped
            }
        );
        assert!(!fail.partial_finish.reached_eof);
        assert_eq!(fail.partial_finish.bytes_read, 8);
        assert_eq!(
            fail.partial_finish.content_digest,
            ContentDigest::of(&bytes[..8])
        );
        let cursor = fail.next_cursor.unwrap();
        let expected = &fixture[if error {
            "error_on_first"
        } else {
            "stop_after_first"
        }];
        assert_eq!(
            cursor.next_ordinal,
            expected["next_ordinal"].as_u64().unwrap()
        );
        assert_eq!(
            cursor.next_byte_start,
            expected["next_byte_start"].as_u64().unwrap()
        );
        let mut resumed = Sink::default();
        let finish = read_source_records(&sealed, Some(&cursor), &mut resumed).unwrap();
        assert_eq!(resumed.records.len(), if error { 3 } else { 2 });
        assert_eq!(finish.records_seen, 3);
        assert_eq!(finish.parse_failed, 1);
        assert_eq!(finish.bytes_read, bytes.len() as u64);
    }
    let mut stopped = Sink {
        stop: true,
        ..Sink::default()
    };
    let cursor = read_source_records(&sealed, None, &mut stopped)
        .unwrap_err()
        .next_cursor
        .unwrap();
    let cursor = read_source_records(&sealed, Some(&cursor), &mut stopped)
        .unwrap_err()
        .next_cursor
        .unwrap();
    let finish = read_source_records(&sealed, Some(&cursor), &mut Sink::default()).unwrap();
    assert_eq!(finish.parse_failed, 1); // Prefix failures remain measured on resume.
}

#[test]
fn rejects_wrong_context_content_manifest_and_cursor() {
    let (_dir, sealed) = context(b"1\n2\n", SourceFormat::Jsonl);
    let mut changed = sealed.clone();
    changed.parser_context.relative_locator = "another.jsonl".into();
    let mut sink = Sink::default();
    assert_eq!(
        read_source_records(&changed, None, &mut sink)
            .unwrap_err()
            .kind,
        SourceReadFailureKind::InvalidContext
    );
    assert!(sink.records.is_empty());
    let mut stopped = Sink {
        stop: true,
        ..Sink::default()
    };
    let cursor = read_source_records(&sealed, None, &mut stopped)
        .unwrap_err()
        .next_cursor
        .unwrap();
    for mutation in 0..3 {
        let mut bad = cursor.clone();
        match mutation {
            0 => bad.next_byte_start = 1,
            1 => bad.input_key = Sha256Digest::of(b"changed"),
            _ => bad.next_ordinal += 1,
        }
        let mut sink = Sink::default();
        let failure = read_source_records(&sealed, Some(&bad), &mut sink).unwrap_err();
        assert_eq!(failure.kind, SourceReadFailureKind::InvalidCursor);
        assert!(!failure.partial_finish.reached_eof);
        assert!(sink.records.is_empty());
    }
    let mut changed = sealed.clone();
    changed.expected_manifest.records -= 1;
    assert_eq!(
        read_source_records(&changed, None, &mut Sink::default())
            .unwrap_err()
            .kind,
        SourceReadFailureKind::SourceIdentityMismatch
    );
    fs::write(&sealed.sealed_locator, b"1\n3\n").unwrap();
    let failure = read_source_records(&sealed, None, &mut Sink::default()).unwrap_err();
    assert_eq!(failure.kind, SourceReadFailureKind::SourceIdentityMismatch);
    assert_eq!(
        failure.partial_finish.content_digest,
        ContentDigest::of(b"1\n3\n")
    );
    assert!(failure.partial_finish.reached_eof);
}

#[test]
fn document_and_eof_cursor() {
    for bytes in [b"[]".as_slice(), b"".as_slice()] {
        let (_dir, sealed) = context(bytes, SourceFormat::JsonDocument);
        let fail = read_source_records(
            &sealed,
            None,
            &mut Sink {
                stop: true,
                ..Sink::default()
            },
        )
        .unwrap_err();
        assert!(!fail.partial_finish.reached_eof);
        let cursor = fail.next_cursor.unwrap();
        assert_eq!(cursor.next_ordinal, 2);
        assert_eq!(cursor.next_byte_start, bytes.len() as u64);
        let mut sink = Sink::default();
        let finish = read_source_records(&sealed, Some(&cursor), &mut sink).unwrap();
        assert!(sink.records.is_empty());
        assert_eq!(finish.records_seen, 1);
        assert!(finish.reached_eof);
    }
}

#[test]
fn bounded_diagnostics() {
    let literals = [
        (
            br#"{"private-member":1,"private-member":2}"#.as_slice(),
            "duplicate_object_key",
            "duplicate member name",
            20,
        ),
        (b"\xff".as_slice(), "invalid_utf8", "invalid UTF-8", 0),
        (b" ".as_slice(), "empty_input", "no JSON value", 0),
        (b"[".as_slice(), "unexpected_end", "input ended early", 1),
        (
            br#""\uD800""#.as_slice(),
            "lone_surrogate",
            "high surrogate without low surrogate",
            7,
        ),
    ];
    for (raw, code, detail, offset) in literals {
        let error = LosslessSourceValue::parse(raw).unwrap_err();
        assert_eq!(error.code, code);
        assert_eq!(error.detail.as_deref(), Some(detail));
        assert_eq!(error.offset, Some(offset));
        assert!(
            !serde_json::to_string(&error)
                .unwrap()
                .contains("private-member")
        );
    }
    let (_dir, sealed) = context(b"\xc2\xa0\n\xff\n{}\n", SourceFormat::Jsonl);
    let mut sink = Sink::default();
    let finish = read_source_records(&sealed, None, &mut sink).unwrap();
    assert_eq!(finish.records_seen, 3);
    assert_eq!(finish.parse_failed, 2);
    assert_eq!(finish.blank_lines, 0);
    assert!(finish.unprocessed_ranges.is_empty());
}

#[test]
fn deep_source_does_not_inherit_native_depth_limit() {
    let text = format!("{}1e99999{}", "[".repeat(2048), "]".repeat(2048));
    let value = LosslessSourceValue::parse(text.as_bytes()).unwrap();
    let mut canonical = Vec::new();
    value.write_canonical(&mut canonical).unwrap();
    assert_eq!(canonical, text.as_bytes());
    let mut tagged = Vec::new();
    value.write_tagged(&mut tagged).unwrap();
    assert!(!tagged.is_empty());
    drop(value);
}

#[test]
fn roles_keep_legacy_discovery_compatible() {
    assert_eq!(
        SourceInputRole::try_from(DiscoveredSourceRole::PrimarySessionLog).unwrap(),
        SourceInputRole::PrimarySessionLog
    );
    assert_eq!(
        SourceInputRole::try_from(DiscoveredSourceRole::MetadataSidecar).unwrap(),
        SourceInputRole::MetadataSidecar
    );
    assert_eq!(
        SourceInputRole::try_from(DiscoveredSourceRole::SqliteDatabase),
        Err(SourceReadFailureKind::Unsupported)
    );
    assert_eq!(
        serde_json::to_string(&SourceInputRole::Trace).unwrap(),
        "\"trace\""
    );
    assert!(Sha256Digest::from_hex(&"A".repeat(64)).is_err());
}
