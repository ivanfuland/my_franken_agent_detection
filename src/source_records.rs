//! Lossless records read from an explicitly sealed source, with verified identity.

use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::connectors::DiscoveredSourceRole;
use crate::{Origin, PathMapping, Platform};

/// A validated SHA-256 digest. Its text representation is lowercase hexadecimal.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Hash the supplied bytes.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Validate a digest without accepting uppercase or shortened encodings.
    pub fn from_hex(text: &str) -> Result<Self, SourceReadFailureKind> {
        if text.len() != 64
            || !text
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(SourceReadFailureKind::InvalidContext);
        }
        let mut bytes = [0; 32];
        for (index, pair) in text.as_bytes().chunks_exact(2).enumerate() {
            let digit = |b: u8| {
                if b.is_ascii_digit() {
                    b - b'0'
                } else {
                    b - b'a' + 10
                }
            };
            bytes[index] = digit(pair[0]) * 16 + digit(pair[1]);
        }
        Ok(Self(bytes))
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::from_hex(&String::deserialize(deserializer)?)
            .map_err(|_| serde::de::Error::custom("invalid sha256 digest"))
    }
}

/// Algorithm of an input content digest, independent of blob addressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigestAlgorithm {
    Sha256,
}

/// Content digest; the source wire uses its SHA-256 text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentDigest {
    pub algorithm: DigestAlgorithm,
    pub value: Sha256Digest,
}

impl ContentDigest {
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self {
            algorithm: DigestAlgorithm::Sha256,
            value: Sha256Digest::of(bytes),
        }
    }
}

impl Serialize for ContentDigest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ContentDigest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self {
            algorithm: DigestAlgorithm::Sha256,
            value: Sha256Digest::deserialize(deserializer)?,
        })
    }
}

/// Source role for the sealed JSON interface. The legacy discovery enum is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceInputRole {
    PrimarySessionLog,
    MetadataSidecar,
    Trace,
}

impl SourceInputRole {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PrimarySessionLog => "primary_session_log",
            Self::MetadataSidecar => "metadata_sidecar",
            Self::Trace => "trace",
        }
    }
}

impl TryFrom<DiscoveredSourceRole> for SourceInputRole {
    type Error = SourceReadFailureKind;
    fn try_from(role: DiscoveredSourceRole) -> Result<Self, Self::Error> {
        match role {
            DiscoveredSourceRole::PrimarySessionLog => Ok(Self::PrimarySessionLog),
            DiscoveredSourceRole::MetadataSidecar => Ok(Self::MetadataSidecar),
            DiscoveredSourceRole::SqliteDatabase => Err(SourceReadFailureKind::Unsupported),
        }
    }
}

/// Explicit provenance. Missing source versions remain missing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceOrigin {
    pub origin: Origin,
    pub source_version: Option<String>,
}

impl SourceOrigin {
    fn wire(&self) -> serde_json::Value {
        serde_json::json!({"kind": self.origin.kind, "host": self.origin.host, "source_version": self.source_version})
    }
}

/// Original logical location, never the transport filename.
#[derive(Debug, Clone)]
pub struct SourceInputDescriptor {
    pub connector: String,
    pub scan_root: PathBuf,
    pub source_path: PathBuf,
    pub relative_locator: String,
    pub role: SourceInputRole,
    pub origin: SourceOrigin,
    pub platform: Option<Platform>,
    pub size_bytes: Option<u64>,
    pub modified_at_ms: Option<i64>,
}

impl SourceInputDescriptor {
    /// Produce the closed descriptor wire, after verifying UTF-8 paths.
    pub fn to_wire(&self) -> Result<serde_json::Value, SourceReadFailureKind> {
        Ok(serde_json::json!({
            "connector": self.connector, "scan_root": path_text(&self.scan_root)?,
            "source_path": path_text(&self.source_path)?, "relative_locator": self.relative_locator,
            "role": self.role, "origin": self.origin.wire(), "platform": self.platform,
        }))
    }
}

/// A logical parser root without runtime caches or filesystem discovery.
#[derive(Debug, Clone)]
pub struct SourceParserRoot {
    pub path: PathBuf,
    pub origin: SourceOrigin,
    pub platform: Option<Platform>,
    pub workspace_rewrites: Vec<PathMapping>,
}

/// Versioned semantic parser context. Source-aware reads have no time filter.
#[derive(Debug, Clone)]
pub struct SourceParserContext {
    pub schema: String,
    pub connector: String,
    pub roots: Vec<SourceParserRoot>,
    pub relative_locator: String,
    pub role: SourceInputRole,
    pub effective_config_digest: Option<Sha256Digest>,
}

impl SourceParserContext {
    /// Encode only semantic fields, with stable key ordering and explicit nulls.
    pub fn to_wire(&self) -> Result<serde_json::Value, SourceReadFailureKind> {
        let first = self
            .roots
            .first()
            .ok_or(SourceReadFailureKind::InvalidContext)?;
        let roots: Result<Vec<_>, _> = self.roots.iter().map(|r| path_text(&r.path)).collect();
        if self.roots.iter().any(|r| {
            r.origin != first.origin
                || r.platform != first.platform
                || r.workspace_rewrites != first.workspace_rewrites
        }) {
            return Err(SourceReadFailureKind::InvalidContext);
        }
        Ok(serde_json::json!({
            "schema": self.schema, "connector": self.connector, "discovery_roots": roots?,
            "since_ts": null, "relative_locator": self.relative_locator, "role": self.role,
            "origin": first.origin.wire(), "platform": first.platform,
            "workspace_rewrites": first.workspace_rewrites,
            "effective_config_summary": { "effective_digest": self.effective_config_digest },
        }))
    }

    /// Hash the canonical semantic context, excluding the digest itself.
    pub fn digest(&self) -> Result<Sha256Digest, SourceReadFailureKind> {
        Ok(Sha256Digest::of(
            &serde_json::to_vec(&self.to_wire()?)
                .map_err(|_| SourceReadFailureKind::InvalidContext)?,
        ))
    }
}

/// Physical record format, independent of the transport extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFormat {
    Jsonl,
    JsonDocument,
}

/// Independent expected framing, never substituted for measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceExpectedManifest {
    pub verifier_version: String,
    pub format: SourceFormat,
    pub records: u64,
    pub blank_lines: u64,
    pub blank_bytes: u64,
    pub framing_sha256: Sha256Digest,
    pub complete: bool,
}

/// Address provenance; these optional blob fields never supply a content SHA.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedSourceRef {
    pub transport: String,
    pub blob_algorithm: Option<String>,
    pub blob_digest: Option<String>,
    pub blob_relative_path: Option<String>,
}

/// Immutable read contract supplied by the caller after capture.
#[derive(Debug, Clone)]
pub struct SealedInputContext {
    pub input_key: Sha256Digest,
    pub identity_namespace: String,
    pub parser_context_digest: Sha256Digest,
    pub content_digest: ContentDigest,
    pub byte_len: u64,
    pub format: SourceFormat,
    pub normalizer_id: String,
    pub safety_contract_id: String,
    pub sealed_locator: PathBuf,
    pub sealed_ref: SealedSourceRef,
    pub expected_manifest: SourceExpectedManifest,
    pub discovery: SourceInputDescriptor,
    pub parser_context: SourceParserContext,
}

impl SealedInputContext {
    /// Compute the input identity from semantic context and measured content.
    pub fn identity_for(
        &self,
        content: ContentDigest,
    ) -> Result<Sha256Digest, SourceReadFailureKind> {
        let d = &self.discovery;
        let identity = serde_json::json!([
            "cass.source-input/2",
            self.identity_namespace,
            d.origin.origin.source_id,
            d.connector,
            path_text(&d.scan_root)?,
            d.relative_locator,
            d.role,
            d.origin.origin.kind,
            d.origin.origin.host,
            d.origin.source_version,
            d.platform,
            self.parser_context.digest()?,
            content,
            self.normalizer_id,
            self.safety_contract_id,
        ]);
        Ok(Sha256Digest::of(
            &serde_json::to_vec(&identity).map_err(|_| SourceReadFailureKind::InvalidContext)?,
        ))
    }

    /// Validate the supplied semantic fields before delivering source records.
    pub fn validate(&self) -> Result<(), SourceReadFailureKind> {
        path_text(&self.sealed_locator)?;
        self.discovery.to_wire()?;
        let d = &self.discovery;
        let c = &self.parser_context;
        let root = c
            .roots
            .first()
            .ok_or(SourceReadFailureKind::InvalidContext)?;
        let relative = Path::new(&d.relative_locator);
        let blob_valid = match self.sealed_ref.transport.as_str() {
            "raw_blob" => {
                self.sealed_ref.blob_algorithm.as_deref() == Some("blake3")
                    && self.sealed_ref.blob_digest.as_ref().is_some_and(|s| {
                        s.len() == 64
                            && s.bytes()
                                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    })
                    && self
                        .sealed_ref
                        .blob_relative_path
                        .as_ref()
                        .is_some_and(|s| valid_relative(s))
            }
            "frozen_file_unmirrored" => {
                self.sealed_ref.blob_algorithm.is_none()
                    && self.sealed_ref.blob_digest.is_none()
                    && self.sealed_ref.blob_relative_path.is_none()
            }
            _ => false,
        };
        if !blob_valid
            || !valid_relative(&d.relative_locator)
            || d.scan_root.join(relative) != d.source_path
            || c.schema != "cass.source-parser-context/1"
            || c.connector != d.connector
            || c.relative_locator != d.relative_locator
            || c.role != d.role
            || root.path != d.scan_root
            || root.origin != d.origin
            || root.platform != d.platform
            || self.normalizer_id.is_empty()
            || self.safety_contract_id.is_empty()
            || self.identity_namespace.is_empty()
            || d.origin.origin.source_id.is_empty()
            || self.expected_manifest.verifier_version != "cass.source-verifier/1"
            || self.expected_manifest.format != self.format
            || !self.expected_manifest.complete
            || c.digest()? != self.parser_context_digest
            || self.identity_for(self.content_digest)? != self.input_key
        {
            return Err(SourceReadFailureKind::InvalidContext);
        }
        Ok(())
    }
}

fn valid_relative(text: &str) -> bool {
    !text.is_empty()
        && !text
            .split('/')
            .any(|s| s.is_empty() || matches!(s, "." | ".."))
        && Path::new(text)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

fn path_text(path: &Path) -> Result<&str, SourceReadFailureKind> {
    path.to_str()
        .ok_or(SourceReadFailureKind::PathNotRepresentable)
}

/// A JSON number token, validated before it can enter a source tree.
#[derive(Clone, PartialEq, Eq)]
pub struct ValidatedJsonNumberText(String);

impl ValidatedJsonNumberText {
    pub fn new(text: String) -> Result<Self, SourceDiagnostic> {
        {
            let mut parser = JsonParser::new(&text);
            parser.number()?;
            if parser.offset != text.len() {
                return Err(parser.error("invalid_number", "trailing number characters"));
            }
        }
        Ok(Self(text))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ValidatedJsonNumberText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidatedJsonNumberText")
            .field("bytes", &self.0.len())
            .finish()
    }
}

/// Original JSON kinds. Objects retain member order; numbers retain token text.
#[derive(Clone, PartialEq, Eq)]
pub enum LosslessSourceValue {
    Object(Vec<(String, Self)>),
    Array(Vec<Self>),
    String(String),
    Number(ValidatedJsonNumberText),
    Bool(bool),
    Null,
}

impl fmt::Debug for LosslessSourceValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Object(_) => "Object(..)",
            Self::Array(_) => "Array(..)",
            Self::String(_) => "String(..)",
            Self::Number(_) => "Number(..)",
            Self::Bool(_) => "Bool(..)",
            Self::Null => "Null",
        })
    }
}

/// A record's physical position. JSON document pointers are also empty.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRecordLocator {
    pub ordinal: u64,
    pub record_pointer: String,
    pub byte_start: u64,
    pub byte_end: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceParseState {
    Parsed,
    ParseFailed,
}

/// One original record, including failed parses. Diagnostics contain no source values.
#[derive(Debug)]
pub struct SourceRecordOutcome {
    pub locator: SourceRecordLocator,
    pub record_byte_len: u64,
    pub terminator_len: u8,
    pub source_record_sha256: Sha256Digest,
    pub value: Option<LosslessSourceValue>,
    pub parse_state: SourceParseState,
    pub diagnostics: Vec<SourceDiagnostic>,
}

/// A bounded suffix not yet read, or not yet accepted by the sink.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceUnprocessedRanges {
    pub byte_start: u64,
    pub byte_end: u64,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_pointer: Option<String>,
}

/// Measurements of the actual read, not a copy of the expected manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceInputFinish {
    pub content_digest: ContentDigest,
    pub bytes_read: u64,
    pub records_seen: u64,
    pub blank_lines: u64,
    pub parse_failed: u64,
    pub mapping_diagnostics: u64,
    pub unprocessed_ranges: Vec<SourceUnprocessedRanges>,
    pub reached_eof: bool,
}

impl Default for SourceInputFinish {
    fn default() -> Self {
        Self {
            content_digest: ContentDigest::of(&[]),
            bytes_read: 0,
            records_seen: 0,
            blank_lines: 0,
            parse_failed: 0,
            mapping_diagnostics: 0,
            unprocessed_ranges: Vec::new(),
            reached_eof: false,
        }
    }
}

/// Advisory record cursor. The caller decides when accepted data is committed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceCursor {
    pub version: u16,
    pub input_key: Sha256Digest,
    pub content_digest: ContentDigest,
    pub byte_len: u64,
    pub format: SourceFormat,
    pub normalizer_id: String,
    pub safety_contract_id: String,
    pub context_digest: Sha256Digest,
    pub next_ordinal: u64,
    pub next_byte_start: u64,
    pub next_record_pointer: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceSinkControl {
    Continue,
    Stop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceSinkFailureCode {
    Rejected,
    ExternalFailure,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct SourceSinkFailure {
    pub code: SourceSinkFailureCode,
    pub detail_digest: Option<Sha256Digest>,
}

/// Owned delivery: Stop accepts this record; Err does not confirm acceptance.
pub trait SourceRecordSink {
    fn on_record(
        &mut self,
        record: SourceRecordOutcome,
    ) -> Result<SourceSinkControl, SourceSinkFailure>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceReadFailureKind {
    Unsupported,
    InvalidContext,
    InvalidCursor,
    SourceIdentityMismatch,
    Io,
    Stopped,
    Sink,
    Resource,
    PathNotRepresentable,
    CounterOverflow,
    ProjectionRejected,
}

/// Safe diagnostic text. Source pointers are only exposed through explicit access.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceDiagnostic {
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pointer: Option<String>,
}

impl fmt::Debug for SourceDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceDiagnostic")
            .field("offset", &self.offset)
            .finish_non_exhaustive()
    }
}

impl SourceDiagnostic {
    #[must_use]
    pub fn code(code: &str) -> Self {
        Self {
            code: code.into(),
            detail: None,
            offset: None,
            pointer: None,
        }
    }
}

/// A failed read carries the measured prefix and, where supported, a retry cursor.
#[derive(Debug)]
pub struct SourceReadFailure {
    pub kind: SourceReadFailureKind,
    pub partial_finish: SourceInputFinish,
    pub next_cursor: Option<SourceCursor>,
    pub diagnostic: SourceDiagnostic,
}

impl fmt::Display for SourceReadFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sealed source read failed: {:?}", self.kind)
    }
}
impl std::error::Error for SourceReadFailure {}

impl SourceReadFailure {
    #[must_use]
    pub fn before_read(sealed: &SealedInputContext, kind: SourceReadFailureKind) -> Self {
        Self {
            kind,
            partial_finish: SourceInputFinish {
                unprocessed_ranges: vec![SourceUnprocessedRanges {
                    byte_start: 0,
                    byte_end: sealed.byte_len,
                    reason: "read_not_started".into(),
                    ordinal: None,
                    record_pointer: None,
                }],
                ..SourceInputFinish::default()
            },
            next_cursor: None,
            diagnostic: SourceDiagnostic::code("source_read_rejected"),
        }
    }
}

impl LosslessSourceValue {
    /// Parse original UTF-8 JSON before any native numeric conversion.
    pub fn parse(bytes: &[u8]) -> Result<Self, SourceDiagnostic> {
        let text = std::str::from_utf8(bytes).map_err(|error| SourceDiagnostic {
            code: "invalid_utf8".into(),
            detail: Some("invalid UTF-8".into()),
            offset: Some(error.valid_up_to() as u64),
            pointer: None,
        })?;
        JsonParser::new(text).parse()
    }

    /// Look up an original object member without interpreting its name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Self> {
        match self {
            Self::Object(members) => members
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    /// Write exact number tokens and original member order with no added spaces.
    pub fn write_canonical(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_json(writer, false)
    }

    /// Write the six-tag source wire without a recursive serializer.
    pub fn write_tagged(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_json(writer, true)
    }

    fn write_json(&self, writer: &mut dyn std::io::Write, tagged: bool) -> std::io::Result<()> {
        enum Part<'a> {
            Node(&'a LosslessSourceValue),
            Text(&'a str),
            String(&'a str),
        }
        let mut stack = vec![Part::Node(self)];
        while let Some(part) = stack.pop() {
            match part {
                Part::Text(text) => writer.write_all(text.as_bytes())?,
                Part::String(text) => serde_json::to_writer(&mut *writer, text)?,
                Part::Node(node) => match node {
                    Self::Object(members) => {
                        writer.write_all(if tagged { b"[\"obj\",[" } else { b"{" })?;
                        stack.push(Part::Text(if tagged { "]]" } else { "}" }));
                        for (index, (key, value)) in members.iter().enumerate().rev() {
                            if tagged {
                                stack.push(Part::Text("]"));
                            }
                            stack.push(Part::Node(value));
                            stack.push(Part::Text(if tagged { "," } else { ":" }));
                            stack.push(Part::String(key));
                            if tagged {
                                stack.push(Part::Text("["));
                            }
                            if index > 0 {
                                stack.push(Part::Text(","));
                            }
                        }
                    }
                    Self::Array(values) => {
                        writer.write_all(if tagged { b"[\"arr\",[" } else { b"[" })?;
                        stack.push(Part::Text(if tagged { "]]" } else { "]" }));
                        for (index, value) in values.iter().enumerate().rev() {
                            stack.push(Part::Node(value));
                            if index > 0 {
                                stack.push(Part::Text(","));
                            }
                        }
                    }
                    Self::String(text) => {
                        if tagged {
                            writer.write_all(b"[\"str\",")?;
                        }
                        serde_json::to_writer(&mut *writer, text)?;
                        if tagged {
                            writer.write_all(b"]")?;
                        }
                    }
                    Self::Number(text) => {
                        if tagged {
                            writer.write_all(b"[\"num\",")?;
                            serde_json::to_writer(&mut *writer, text.as_str())?;
                            writer.write_all(b"]")?;
                        } else {
                            writer.write_all(text.as_str().as_bytes())?;
                        }
                    }
                    Self::Bool(value) => {
                        if tagged {
                            writer.write_all(b"[\"bool\",")?;
                        }
                        writer.write_all(if *value { b"true" } else { b"false" })?;
                        if tagged {
                            writer.write_all(b"]")?;
                        }
                    }
                    Self::Null => writer.write_all(if tagged { b"[\"null\"]" } else { b"null" })?,
                },
            }
        }
        Ok(())
    }

    /// Retain fixed native JSON conversion behavior only for legacy projection.
    pub(crate) fn legacy_value(&self) -> Option<serde_json::Value> {
        #[cfg(test)]
        LEGACY_VALUE_CALLS.with(|calls| calls.set(calls.get() + 1));
        let mut bytes = Vec::new();
        self.write_canonical(&mut bytes).ok()?;
        serde_json::from_slice(&bytes).ok()
    }
}

// Test-only probe for the reader's conversion gate. It counts this thread's
// whole-document conversions so a test can show that a projector which declares
// it needs no native value never triggers one. It is not a product counter.
#[cfg(test)]
thread_local! {
    static LEGACY_VALUE_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn legacy_value_calls() -> u64 {
    LEGACY_VALUE_CALLS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn reset_legacy_value_calls() {
    LEGACY_VALUE_CALLS.with(|calls| calls.set(0));
}

// Destruction must not reintroduce a stack-depth limit into the source reader.
impl Drop for LosslessSourceValue {
    fn drop(&mut self) {
        fn detach(value: &mut LosslessSourceValue, pending: &mut Vec<LosslessSourceValue>) {
            match value {
                LosslessSourceValue::Object(members) => {
                    pending.extend(std::mem::take(members).into_iter().map(|(_, value)| value));
                }
                LosslessSourceValue::Array(values) => pending.append(values),
                _ => {}
            }
        }
        let mut pending = Vec::new();
        detach(self, &mut pending);
        while let Some(mut child) = pending.pop() {
            detach(&mut child, &mut pending);
        }
    }
}

struct JsonParser<'a> {
    text: &'a str,
    offset: usize,
}

/// What the first byte of a value turned out to be.
enum ValueStart {
    Value(LosslessSourceValue),
    /// A container frame was pushed; its first member comes next.
    Opened,
}

/// Whether the value just attached closed the whole document or left a
/// container open.
enum Attach {
    Complete(LosslessSourceValue),
    NextValue,
}

enum JsonFrame {
    Object {
        members: Vec<(String, LosslessSourceValue)>,
        key: String,
        seen: HashSet<String>,
    },
    Array(Vec<LosslessSourceValue>),
}

impl<'a> JsonParser<'a> {
    const fn new(text: &'a str) -> Self {
        Self { text, offset: 0 }
    }
    fn error_at(&self, code: &str, detail: &str, offset: usize) -> SourceDiagnostic {
        SourceDiagnostic {
            code: code.into(),
            detail: Some(detail.into()),
            offset: Some(self.text[..offset].chars().count() as u64),
            pointer: None,
        }
    }
    fn error(&self, code: &str, detail: &str) -> SourceDiagnostic {
        self.error_at(code, detail, self.offset)
    }
    fn whitespace(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.offset)
            .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
        {
            self.offset += 1;
        }
    }
    fn peek(&self) -> Result<u8, SourceDiagnostic> {
        self.text
            .as_bytes()
            .get(self.offset)
            .copied()
            .ok_or_else(|| self.error("unexpected_end", "input ended early"))
    }
    fn literal(&mut self, literal: &str) -> Result<(), SourceDiagnostic> {
        if !self.text[self.offset..].starts_with(literal) {
            return Err(self.error("invalid_literal", "expected a JSON literal"));
        }
        self.offset += literal.len();
        Ok(())
    }
    fn hex4(&mut self) -> Result<u32, SourceDiagnostic> {
        let end = self.offset.saturating_add(4);
        let Some(bytes) = self.text.as_bytes().get(self.offset..end) else {
            return Err(self.error("invalid_escape", "truncated \\u escape"));
        };
        let mut number = 0;
        for byte in bytes {
            number = number * 16
                + match byte {
                    b'0'..=b'9' => u32::from(byte - b'0'),
                    b'a'..=b'f' => u32::from(byte - b'a' + 10),
                    b'A'..=b'F' => u32::from(byte - b'A' + 10),
                    _ => return Err(self.error("invalid_escape", "bad hex in \\u escape")),
                };
        }
        self.offset = end;
        Ok(number)
    }
    fn string(&mut self) -> Result<String, SourceDiagnostic> {
        self.offset += 1;
        let mut segment = self.offset;
        let mut result = String::new();
        loop {
            let Some(byte) = self.text.as_bytes().get(self.offset).copied() else {
                return Err(self.error_at("unterminated_string", "no closing quote", segment));
            };
            match byte {
                b'"' => {
                    result.push_str(&self.text[segment..self.offset]);
                    self.offset += 1;
                    return Ok(result);
                }
                b'\\' => {
                    result.push_str(&self.text[segment..self.offset]);
                    self.offset += 1;
                    let Some(escape) = self.text.as_bytes().get(self.offset).copied() else {
                        return Err(self.error("unterminated_string", "escape at end"));
                    };
                    self.offset += 1;
                    let character = match escape {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut code = self.hex4()?;
                            if (0xd800..=0xdbff).contains(&code) {
                                if !self.text[self.offset..].starts_with("\\u") {
                                    return Err(self.error(
                                        "lone_surrogate",
                                        "high surrogate without low surrogate",
                                    ));
                                }
                                self.offset += 2;
                                let low = self.hex4()?;
                                if !(0xdc00..=0xdfff).contains(&low) {
                                    return Err(self.error(
                                        "lone_surrogate",
                                        "high surrogate not followed by low surrogate",
                                    ));
                                }
                                code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                            } else if (0xdc00..=0xdfff).contains(&code) {
                                return Err(self.error(
                                    "lone_surrogate",
                                    "low surrogate without high surrogate",
                                ));
                            }
                            char::from_u32(code).ok_or_else(|| {
                                self.error("invalid_escape", "invalid Unicode scalar")
                            })?
                        }
                        _ => {
                            return Err(self.error_at(
                                "invalid_escape",
                                "unknown escape",
                                self.offset - 2,
                            ));
                        }
                    };
                    result.push(character);
                    segment = self.offset;
                }
                0..=31 => return Err(self.error("unescaped_control_char", "control character")),
                _ => self.offset += 1,
            }
        }
    }
    fn number(&mut self) -> Result<ValidatedJsonNumberText, SourceDiagnostic> {
        let start = self.offset;
        let bytes = self.text.as_bytes();
        if bytes.get(self.offset) == Some(&b'-') {
            self.offset += 1;
        }
        match bytes.get(self.offset) {
            Some(b'0') => self.offset += 1,
            Some(b'1'..=b'9') => {
                while bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
                    self.offset += 1;
                }
            }
            _ => {
                self.offset = start;
                return Err(self.error("invalid_number", "no valid number"));
            }
        }
        if bytes.get(self.offset) == Some(&b'.')
            && bytes.get(self.offset + 1).is_some_and(u8::is_ascii_digit)
        {
            self.offset += 1;
            while bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
                self.offset += 1;
            }
        }
        if matches!(bytes.get(self.offset), Some(b'e' | b'E')) {
            let exponent_start = self.offset;
            let mut end = self.offset + 1;
            if matches!(bytes.get(end), Some(b'+' | b'-')) {
                end += 1;
            }
            if bytes.get(end).is_some_and(u8::is_ascii_digit) {
                self.offset = end;
                while bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
                    self.offset += 1;
                }
            } else {
                self.offset = exponent_start;
            }
        }
        if self.text[self.offset..]
            .chars()
            .next()
            .is_some_and(|c| c.is_alphanumeric() || matches!(c, '.' | '+' | '-'))
        {
            return Err(self.error("invalid_number", "trailing number characters"));
        }
        Ok(ValidatedJsonNumberText(
            self.text[start..self.offset].into(),
        ))
    }
    fn key(&mut self, seen: &mut HashSet<String>) -> Result<String, SourceDiagnostic> {
        self.whitespace();
        if self.peek()? != b'"' {
            return Err(self.error("object_key_not_string", "expected object key"));
        }
        let start = self.offset;
        let key = self.string()?;
        if !seen.insert(key.clone()) {
            return Err(self.error_at("duplicate_object_key", "duplicate member name", start));
        }
        self.whitespace();
        if self.peek()? != b':' {
            return Err(self.error("missing_colon", "expected ':'"));
        }
        self.offset += 1;
        Ok(key)
    }
    fn parse(mut self) -> Result<LosslessSourceValue, SourceDiagnostic> {
        self.whitespace();
        if self.offset == self.text.len() {
            return Err(self.error_at("empty_input", "no JSON value", 0));
        }
        let mut stack = Vec::new();
        loop {
            self.whitespace();
            let value = match self.value_start(&mut stack)? {
                ValueStart::Value(value) => value,
                ValueStart::Opened => continue,
            };
            if let Attach::Complete(value) = self.attach_value(&mut stack, value)? {
                return Ok(value);
            }
        }
    }

    /// Read the first byte of one value. Containers push a frame and ask the
    /// caller to restart at their first member.
    fn value_start(&mut self, stack: &mut Vec<JsonFrame>) -> Result<ValueStart, SourceDiagnostic> {
        let value = match self.peek()? {
            b'{' => {
                self.offset += 1;
                self.whitespace();
                if self.peek()? == b'}' {
                    self.offset += 1;
                    LosslessSourceValue::Object(Vec::new())
                } else {
                    let mut seen = HashSet::new();
                    let key = self.key(&mut seen)?;
                    stack.push(JsonFrame::Object {
                        members: Vec::new(),
                        key,
                        seen,
                    });
                    return Ok(ValueStart::Opened);
                }
            }
            b'[' => {
                self.offset += 1;
                self.whitespace();
                if self.peek()? == b']' {
                    self.offset += 1;
                    LosslessSourceValue::Array(Vec::new())
                } else {
                    stack.push(JsonFrame::Array(Vec::new()));
                    return Ok(ValueStart::Opened);
                }
            }
            b'"' => LosslessSourceValue::String(self.string()?),
            b't' => {
                self.literal("true")?;
                LosslessSourceValue::Bool(true)
            }
            b'f' => {
                self.literal("false")?;
                LosslessSourceValue::Bool(false)
            }
            b'n' => {
                self.literal("null")?;
                LosslessSourceValue::Null
            }
            b'-' | b'0'..=b'9' => LosslessSourceValue::Number(self.number()?),
            _ => return Err(self.error("unexpected_char", "not a value start")),
        };
        Ok(ValueStart::Value(value))
    }

    /// Close every container this value completes. `NextValue` means the input
    /// continues inside an open container.
    fn attach_value(
        &mut self,
        stack: &mut Vec<JsonFrame>,
        mut value: LosslessSourceValue,
    ) -> Result<Attach, SourceDiagnostic> {
        loop {
            let Some(frame) = stack.last_mut() else {
                self.whitespace();
                if self.offset != self.text.len() {
                    return Err(self.error(
                        "trailing_content",
                        &format!("offset {}", self.text[..self.offset].chars().count()),
                    ));
                }
                return Ok(Attach::Complete(value));
            };
            self.whitespace();
            match frame {
                JsonFrame::Object { members, key, seen } => {
                    members.push((std::mem::take(key), value));
                    match self.peek()? {
                        b',' => {
                            self.offset += 1;
                            *key = self.key(seen)?;
                            return Ok(Attach::NextValue);
                        }
                        b'}' => {
                            self.offset += 1;
                            value = LosslessSourceValue::Object(std::mem::take(members));
                        }
                        _ => {
                            return Err(self.error(
                                "expected_comma_or_close",
                                "expected comma or close in object",
                            ));
                        }
                    }
                }
                JsonFrame::Array(values) => {
                    values.push(value);
                    match self.peek()? {
                        b',' => {
                            self.offset += 1;
                            return Ok(Attach::NextValue);
                        }
                        b']' => {
                            self.offset += 1;
                            value = LosslessSourceValue::Array(std::mem::take(values));
                        }
                        _ => {
                            return Err(self.error(
                                "expected_comma_or_close",
                                "expected comma or close in array",
                            ));
                        }
                    }
                }
            }
            stack.pop();
        }
    }
}

struct ReadState {
    content: Sha256,
    framing: Sha256,
    finish: SourceInputFinish,
    blank_bytes: u64,
    next_ordinal: u64,
}

impl ReadState {
    fn new() -> Self {
        let mut framing = Sha256::new();
        framing.update(b"cass.source-framing/1");
        Self {
            content: Sha256::new(),
            framing,
            finish: SourceInputFinish::default(),
            blank_bytes: 0,
            next_ordinal: 1,
        }
    }
    fn measured_finish(&self) -> SourceInputFinish {
        let mut finish = self.finish.clone();
        finish.content_digest = ContentDigest {
            algorithm: DigestAlgorithm::Sha256,
            value: Sha256Digest(self.content.clone().finalize().into()),
        };
        finish
    }

    fn failure(
        &self,
        sealed: &SealedInputContext,
        kind: SourceReadFailureKind,
        start: u64,
        cursor: Option<SourceCursor>,
    ) -> SourceReadFailure {
        let mut partial_finish = self.measured_finish();
        partial_finish.unprocessed_ranges = vec![SourceUnprocessedRanges {
            byte_start: start.min(sealed.byte_len),
            byte_end: sealed.byte_len,
            reason: match kind {
                SourceReadFailureKind::Stopped => "sink_stopped",
                SourceReadFailureKind::Sink => "sink_failed",
                SourceReadFailureKind::SourceIdentityMismatch => "source_identity_mismatch",
                _ => "source_read_incomplete",
            }
            .into(),
            ordinal: None,
            record_pointer: None,
        }];
        SourceReadFailure {
            kind,
            partial_finish,
            next_cursor: cursor,
            diagnostic: SourceDiagnostic::code("source_read_incomplete"),
        }
    }

    /// Build the same typed failure, boxed so the internal helpers keep a small
    /// error type while the published signatures stay an inline recovery state.
    fn boxed_failure(
        &self,
        sealed: &SealedInputContext,
        kind: SourceReadFailureKind,
        start: u64,
        cursor: Option<SourceCursor>,
    ) -> Box<SourceReadFailure> {
        Box::new(self.failure(sealed, kind, start, cursor))
    }
    fn frame(
        &mut self,
        reader: &mut dyn BufRead,
        format: SourceFormat,
    ) -> Result<Vec<u8>, SourceReadFailureKind> {
        let mut frame = Vec::new();
        loop {
            let available = reader.fill_buf().map_err(|_| SourceReadFailureKind::Io)?;
            if available.is_empty() {
                return Ok(frame);
            }
            let length = match format {
                SourceFormat::Jsonl => available
                    .iter()
                    .position(|b| *b == b'\n')
                    .map_or(available.len(), |index| index + 1),
                SourceFormat::JsonDocument => available.len(),
            };
            frame
                .try_reserve(length)
                .map_err(|_| SourceReadFailureKind::Resource)?;
            let new_count = self
                .finish
                .bytes_read
                .checked_add(length as u64)
                .ok_or(SourceReadFailureKind::CounterOverflow)?;
            frame.extend_from_slice(&available[..length]);
            self.content.update(&available[..length]);
            self.finish.bytes_read = new_count;
            let line_end = format == SourceFormat::Jsonl && available[length - 1] == b'\n';
            reader.consume(length);
            if line_end {
                return Ok(frame);
            }
        }
    }
}

fn source_cursor(sealed: &SealedInputContext, ordinal: u64, offset: u64) -> SourceCursor {
    SourceCursor {
        version: 1,
        input_key: sealed.input_key,
        content_digest: sealed.content_digest,
        byte_len: sealed.byte_len,
        format: sealed.format,
        normalizer_id: sealed.normalizer_id.clone(),
        safety_contract_id: sealed.safety_contract_id.clone(),
        context_digest: sealed.parser_context_digest,
        next_ordinal: ordinal,
        next_byte_start: offset,
        next_record_pointer: String::new(),
    }
}

fn cursor_matches(sealed: &SealedInputContext, cursor: &SourceCursor) -> bool {
    cursor.version == 1
        && cursor.input_key == sealed.input_key
        && cursor.content_digest == sealed.content_digest
        && cursor.byte_len == sealed.byte_len
        && cursor.format == sealed.format
        && cursor.normalizer_id == sealed.normalizer_id
        && cursor.safety_contract_id == sealed.safety_contract_id
        && cursor.context_digest == sealed.parser_context_digest
        && cursor.next_record_pointer.is_empty()
        && cursor.next_ordinal > 0
        && cursor.next_byte_start <= sealed.byte_len
        && (sealed.format != SourceFormat::JsonDocument
            || matches!((cursor.next_ordinal, cursor.next_byte_start), (1, 0))
            || (cursor.next_ordinal == 2 && cursor.next_byte_start == sealed.byte_len))
}

/// Stream original records from byte zero, including during cursor recovery.
///
/// A successful return proves physical closure. Parse failures still prevent the
/// caller from declaring a complete source input. Stop and sink errors are typed
/// failures, and never imply that a database transaction was committed.
//
// Layout exception recorded under the 2026-10-10 clippy ruling: `SourceReadFailure`
// is the frozen inline recovery state, so the error type is large by contract and
// boxing it would change this published signature.
#[allow(clippy::result_large_err)]
pub fn read_source_records(
    sealed: &SealedInputContext,
    cursor: Option<&SourceCursor>,
    sink: &mut dyn SourceRecordSink,
) -> Result<SourceInputFinish, SourceReadFailure> {
    let (mut reader, mut state, mut resumed) = open_sealed_input(sealed, cursor)
        .map_err(|kind| SourceReadFailure::before_read(sealed, kind))?;
    read_records_into(sealed, &mut reader, &mut state, cursor, &mut resumed, sink)
        .map_err(|failure| *failure)?;
    if !resumed {
        return Err(state.failure(
            sealed,
            SourceReadFailureKind::InvalidCursor,
            state.finish.bytes_read,
            None,
        ));
    }
    state.finish.reached_eof = true;
    let finish = state.measured_finish();
    verify_manifest(sealed, &state, &finish).map_err(|failure| *failure)?;
    Ok(finish)
}

/// Validate the sealed context and cursor, then open the one permitted file.
fn open_sealed_input(
    sealed: &SealedInputContext,
    cursor: Option<&SourceCursor>,
) -> Result<(BufReader<File>, ReadState, bool), SourceReadFailureKind> {
    sealed.validate()?;
    if cursor.is_some_and(|c| !cursor_matches(sealed, c)) {
        return Err(SourceReadFailureKind::InvalidCursor);
    }
    let file = File::open(&sealed.sealed_locator).map_err(|_| SourceReadFailureKind::Io)?;
    Ok((BufReader::new(file), ReadState::new(), cursor.is_none()))
}

/// Read every frame from byte zero. `resumed` records that the cursor position
/// was reached; frames before it are verified but not delivered.
fn read_records_into(
    sealed: &SealedInputContext,
    reader: &mut BufReader<File>,
    state: &mut ReadState,
    cursor: Option<&SourceCursor>,
    resumed: &mut bool,
    sink: &mut dyn SourceRecordSink,
) -> Result<(), Box<SourceReadFailure>> {
    let mut document_seen = false;
    loop {
        let start = state.finish.bytes_read;
        let ordinal = state.next_ordinal;
        if let Some(c) = cursor {
            if c.next_byte_start == start && c.next_ordinal == ordinal {
                *resumed = true;
            } else if !*resumed && (c.next_byte_start < start || c.next_ordinal <= ordinal) {
                return Err(state.boxed_failure(
                    sealed,
                    SourceReadFailureKind::InvalidCursor,
                    start,
                    None,
                ));
            }
        }
        if document_seen {
            break;
        }
        let bytes = state
            .frame(reader, sealed.format)
            .map_err(|kind| state.boxed_failure(sealed, kind, start, None))?;
        if bytes.is_empty() && sealed.format == SourceFormat::Jsonl {
            break;
        }
        document_seen = sealed.format == SourceFormat::JsonDocument;
        consume_frame(
            sealed,
            state,
            cursor,
            *resumed,
            sink,
            Frame {
                bytes: &bytes,
                start,
                ordinal,
            },
        )?;
    }
    Ok(())
}

/// One physical frame handed to the record consumer.
#[derive(Clone, Copy)]
struct Frame<'a> {
    bytes: &'a [u8],
    start: u64,
    ordinal: u64,
}

/// Count one blank frame, reporting whether it held no record at all.
fn count_blank_frame(
    sealed: &SealedInputContext,
    state: &mut ReadState,
    frame: Frame<'_>,
    raw: &[u8],
) -> Result<bool, Box<SourceReadFailure>> {
    if sealed.format != SourceFormat::Jsonl
        || !raw
            .iter()
            .all(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
    {
        return Ok(false);
    }
    state.finish.blank_lines = state.finish.blank_lines.checked_add(1).ok_or_else(|| {
        state.boxed_failure(
            sealed,
            SourceReadFailureKind::CounterOverflow,
            frame.start,
            None,
        )
    })?;
    state.blank_bytes = state
        .blank_bytes
        .checked_add(raw.len() as u64)
        .ok_or_else(|| {
            state.boxed_failure(
                sealed,
                SourceReadFailureKind::CounterOverflow,
                frame.start,
                None,
            )
        })?;
    Ok(true)
}

/// Framed parts of one parsed record: source digest, value, state, diagnostics.
type FramedRecord = (
    Sha256Digest,
    Option<LosslessSourceValue>,
    SourceParseState,
    Vec<SourceDiagnostic>,
);

/// Count, frame and parse one record, returning everything the outcome needs.
fn frame_record_value(
    sealed: &SealedInputContext,
    state: &mut ReadState,
    frame: Frame<'_>,
    raw: &[u8],
    end: u64,
) -> Result<FramedRecord, Box<SourceReadFailure>> {
    state.finish.records_seen = state.finish.records_seen.checked_add(1).ok_or_else(|| {
        state.boxed_failure(
            sealed,
            SourceReadFailureKind::CounterOverflow,
            frame.start,
            None,
        )
    })?;
    let source_record_sha256 = Sha256Digest::of(raw);
    let leaf = serde_json::to_vec(&[
        frame.ordinal.to_string(),
        String::new(),
        frame.start.to_string(),
        end.to_string(),
        source_record_sha256.to_string(),
    ])
    .map_err(|_| state.boxed_failure(sealed, SourceReadFailureKind::Resource, frame.start, None))?;
    state.framing.update((leaf.len() as u64).to_be_bytes());
    state.framing.update(&leaf);
    let (value, parse_state, diagnostics) = match LosslessSourceValue::parse(raw) {
        Ok(value) => (Some(value), SourceParseState::Parsed, Vec::new()),
        Err(diagnostic) => {
            state.finish.parse_failed =
                state.finish.parse_failed.checked_add(1).ok_or_else(|| {
                    state.boxed_failure(
                        sealed,
                        SourceReadFailureKind::CounterOverflow,
                        frame.start,
                        None,
                    )
                })?;
            (None, SourceParseState::ParseFailed, vec![diagnostic])
        }
    };
    Ok((source_record_sha256, value, parse_state, diagnostics))
}

/// Turn one physical frame into counters, framing input and at most one record.
fn consume_frame(
    sealed: &SealedInputContext,
    state: &mut ReadState,
    cursor: Option<&SourceCursor>,
    resumed: bool,
    sink: &mut dyn SourceRecordSink,
    frame: Frame<'_>,
) -> Result<(), Box<SourceReadFailure>> {
    let term = if sealed.format == SourceFormat::Jsonl && frame.bytes.last() == Some(&b'\n') {
        if frame.bytes.len() >= 2 && frame.bytes[frame.bytes.len() - 2] == b'\r' {
            2u8
        } else {
            1u8
        }
    } else {
        0u8
    };
    let raw = &frame.bytes[..frame.bytes.len() - usize::from(term)];
    let end = frame.start.checked_add(raw.len() as u64).ok_or_else(|| {
        state.boxed_failure(
            sealed,
            SourceReadFailureKind::CounterOverflow,
            frame.start,
            None,
        )
    })?;
    state.next_ordinal = frame.ordinal.checked_add(1).ok_or_else(|| {
        state.boxed_failure(
            sealed,
            SourceReadFailureKind::CounterOverflow,
            frame.start,
            None,
        )
    })?;
    if let Some(c) = cursor
        && !resumed
        && c.next_byte_start < state.finish.bytes_read
    {
        return Err(state.boxed_failure(
            sealed,
            SourceReadFailureKind::InvalidCursor,
            frame.start,
            None,
        ));
    }
    if count_blank_frame(sealed, state, frame, raw)? {
        return Ok(());
    }
    let (source_record_sha256, value, parse_state, diagnostics) =
        frame_record_value(sealed, state, frame, raw, end)?;
    if !resumed {
        return Ok(());
    }
    let record = SourceRecordOutcome {
        locator: SourceRecordLocator {
            ordinal: frame.ordinal,
            record_pointer: String::new(),
            byte_start: frame.start,
            byte_end: end,
        },
        record_byte_len: end - frame.start,
        terminator_len: term,
        source_record_sha256,
        value,
        parse_state,
        diagnostics,
    };
    match sink.on_record(record) {
        Ok(SourceSinkControl::Continue) => Ok(()),
        Ok(SourceSinkControl::Stop) => Err(state.boxed_failure(
            sealed,
            SourceReadFailureKind::Stopped,
            state.finish.bytes_read,
            Some(source_cursor(
                sealed,
                state.next_ordinal,
                state.finish.bytes_read,
            )),
        )),
        Err(_) => Err(state.boxed_failure(
            sealed,
            SourceReadFailureKind::Sink,
            frame.start,
            Some(source_cursor(sealed, frame.ordinal, frame.start)),
        )),
    }
}

/// Close the physical counters against the sealed context and manifest.
fn verify_manifest(
    sealed: &SealedInputContext,
    state: &ReadState,
    finish: &SourceInputFinish,
) -> Result<(), Box<SourceReadFailure>> {
    let expected = &sealed.expected_manifest;
    let digest_mismatch = finish.content_digest != sealed.content_digest;
    let byte_mismatch = finish.bytes_read != sealed.byte_len;
    let record_mismatch = finish.records_seen != expected.records;
    let blank_line_mismatch = finish.blank_lines != expected.blank_lines;
    let blank_byte_mismatch = state.blank_bytes != expected.blank_bytes;
    let framing_mismatch =
        Sha256Digest(state.framing.clone().finalize().into()) != expected.framing_sha256;
    let manifest_mismatch = digest_mismatch
        || byte_mismatch
        || record_mismatch
        || blank_line_mismatch
        || blank_byte_mismatch
        || framing_mismatch;
    // The recomputed input identity is only built when the cheap physical
    // comparisons already agree, so a broken path never masks a real mismatch.
    let identity_mismatch = if manifest_mismatch {
        false
    } else {
        sealed
            .identity_for(finish.content_digest)
            .map_err(|kind| state.boxed_failure(sealed, kind, 0, None))?
            != sealed.input_key
    };
    if manifest_mismatch || identity_mismatch {
        return Err(state.boxed_failure(
            sealed,
            SourceReadFailureKind::SourceIdentityMismatch,
            0,
            None,
        ));
    }
    Ok(())
}
