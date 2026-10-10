//! Private acceptance adapter. Inputs contain capture descriptors, never expected records.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use franken_agent_detection::{
    ClaudeCodeConnector, CodexConnector, Connector, ContentDigest, GeminiConnector,
    LosslessSourceValue, OpenClawConnector, Origin, PathMapping, PiAgentConnector, Platform,
    SealedInputContext, SealedSourceRef, Sha256Digest, SourceExpectedManifest, SourceFormat,
    SourceInputDescriptor, SourceInputRole, SourceKind, SourceMappedConversation,
    SourceMappingClaim, SourceMappingDiagnostic, SourceMappingSink, SourceOrigin,
    SourceParserContext, SourceParserRoot, SourceRecordLocator, SourceRecordOutcome,
    SourceRecordSink, SourceSinkControl, SourceSinkFailure, SourceSinkFailureCode,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Contexts {
    schema: String,
    coverage: String,
    inputs: Vec<SelectedInput>,
    library_identity: Option<String>,
    normalizer_id: String,
    reader_version: u64,
    redaction_enabled: bool,
    safety_config_effective_sha256: String,
    safety_config_file_sha256: String,
    safety_contract_id: String,
    source_facts_revision: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireOrigin {
    kind: SourceKind,
    host: Option<String>,
    source_version: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Discovery {
    connector: String,
    scan_root: PathBuf,
    source_path: PathBuf,
    relative_locator: String,
    role: SourceInputRole,
    origin: WireOrigin,
    platform: Option<Platform>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigSummary {
    effective_digest: Option<Sha256Digest>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ParserContext {
    schema: String,
    connector: String,
    discovery_roots: Vec<PathBuf>,
    since_ts: Option<i64>,
    relative_locator: String,
    role: SourceInputRole,
    origin: WireOrigin,
    platform: Option<Platform>,
    workspace_rewrites: Vec<PathMapping>,
    effective_config_summary: ConfigSummary,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectedInput {
    connector: String,
    content_digest: ContentDigest,
    discovery: Discovery,
    expected_manifest: SourceExpectedManifest,
    format: SourceFormat,
    identity_namespace: String,
    input_key: Sha256Digest,
    logical_root: PathBuf,
    logical_source_path: PathBuf,
    origin_host: Option<String>,
    origin_kind: SourceKind,
    parser_context: ParserContext,
    parser_context_digest: Sha256Digest,
    platform: Option<Platform>,
    relative_locator: String,
    role: SourceInputRole,
    sealed_byte_len: u64,
    sealed_locator: PathBuf,
    sealed_ref: SealedSourceRef,
    shape_root: Option<String>,
    source_id: String,
    source_version: Option<String>,
    transport: String,
}

impl SelectedInput {
    fn sealed(&self, header: &Contexts) -> Result<SealedInputContext, &'static str> {
        let d = &self.discovery;
        let p = &self.parser_context;
        // The flattened descriptor fields must agree with the nested discovery
        // block; `logical_root` pairs with `scan_root`, not with a same-named
        // discovery key.
        let disagreements = [
            self.connector != d.connector,
            self.logical_root != d.scan_root,
            self.logical_source_path != d.source_path,
            self.origin_kind != d.origin.kind,
            self.origin_host != d.origin.host,
            self.source_version != d.origin.source_version,
            self.platform != d.platform,
            self.relative_locator != d.relative_locator,
            self.role != d.role,
            self.transport != self.sealed_ref.transport,
            p.since_ts.is_some(),
        ];
        if disagreements.iter().any(|disagreement| *disagreement) {
            return Err("descriptor disagreement");
        }
        let origin = |wire: &WireOrigin| SourceOrigin {
            origin: Origin {
                source_id: self.source_id.clone(),
                kind: wire.kind,
                host: wire.host.clone(),
            },
            source_version: wire.source_version.clone(),
        };
        Ok(SealedInputContext {
            input_key: self.input_key,
            identity_namespace: self.identity_namespace.clone(),
            parser_context_digest: self.parser_context_digest,
            content_digest: self.content_digest,
            byte_len: self.sealed_byte_len,
            format: self.format,
            normalizer_id: header.normalizer_id.clone(),
            safety_contract_id: header.safety_contract_id.clone(),
            sealed_locator: self.sealed_locator.clone(),
            sealed_ref: self.sealed_ref.clone(),
            expected_manifest: self.expected_manifest.clone(),
            discovery: SourceInputDescriptor {
                connector: d.connector.clone(),
                scan_root: d.scan_root.clone(),
                source_path: d.source_path.clone(),
                relative_locator: d.relative_locator.clone(),
                role: d.role,
                origin: origin(&d.origin),
                platform: d.platform,
                size_bytes: None,
                modified_at_ms: None,
            },
            parser_context: SourceParserContext {
                schema: p.schema.clone(),
                connector: p.connector.clone(),
                roots: p
                    .discovery_roots
                    .iter()
                    .map(|path| SourceParserRoot {
                        path: path.clone(),
                        origin: origin(&p.origin),
                        platform: p.platform,
                        workspace_rewrites: p.workspace_rewrites.clone(),
                    })
                    .collect(),
                relative_locator: p.relative_locator.clone(),
                role: p.role,
                effective_config_digest: p.effective_config_summary.effective_digest,
            },
        })
    }
}

const fn sink_failure() -> SourceSinkFailure {
    SourceSinkFailure {
        code: SourceSinkFailureCode::ExternalFailure,
        detail_digest: None,
    }
}

fn private_file(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

/// The Fact DTO is restricted to this adapter. It preserves registered original
/// IDs and metadata and never supplies standard usage, safety, or cost facts.
struct Fact<'a> {
    field: &'a str,
    value: &'a LosslessSourceValue,
    pointer: String,
    string_only: bool,
    nonempty: bool,
}

impl Fact<'_> {
    fn write(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        let mut object = json!({"field": self.field, "raw_ref": self.pointer});
        match self.value {
            LosslessSourceValue::Null => {
                object["status"] = json!("missing");
                object["reason"] = json!("null");
            }
            LosslessSourceValue::String(text) if !self.nonempty || !text.trim().is_empty() => {
                object["status"] = json!("present");
                object["value"] = json!(text);
            }
            LosslessSourceValue::String(_) => {
                object["status"] = json!("invalid");
                object["reason"] = json!("empty_id");
            }
            _ if self.string_only => {
                object["status"] = json!("invalid");
                object["reason"] = json!("bad_type");
            }
            LosslessSourceValue::Number(number) => {
                object["status"] = json!("present");
                object["value"] = json!(number.as_str());
                object["unit"] = json!("number_text");
            }
            LosslessSourceValue::Bool(value) => {
                object["status"] = json!("present");
                object["value"] = json!(value);
            }
            LosslessSourceValue::Object(_) | LosslessSourceValue::Array(_) => {
                object["status"] = json!("present");
                object["value_kind"] =
                    json!(if matches!(self.value, LosslessSourceValue::Object(_)) {
                        "object"
                    } else {
                        "array"
                    });
                let mut bytes = serde_json::to_vec(&object)?;
                bytes.pop();
                writer.write_all(&bytes)?;
                writer.write_all(b",\"value\":")?;
                self.value.write_tagged(writer)?;
                return writer.write_all(b"}");
            }
        }
        serde_json::to_writer(writer, &object)?;
        Ok(())
    }
}

fn add_fact<'a>(
    facts: &mut Vec<Fact<'a>>,
    value: Option<&'a LosslessSourceValue>,
    key: &'a str,
    field: &'a str,
    base: &str,
    string_only: bool,
    nonempty: bool,
) {
    if let Some(value) = value.and_then(|v| v.get(key)) {
        facts.push(Fact {
            field,
            value,
            pointer: format!("{base}/{key}"),
            string_only,
            nonempty,
        });
    }
}

fn claude_facts<'a>(
    facts: &mut Vec<Fact<'a>>,
    value: &'a LosslessSourceValue,
    msg: Option<&'a LosslessSourceValue>,
) {
    for (key, field) in [
        ("sessionId", "session_id"),
        ("uuid", "record_uuid"),
        ("requestId", "request_id"),
    ] {
        add_fact(facts, Some(value), key, field, "", true, true);
    }
    add_fact(
        facts,
        msg,
        "id",
        "provider_message_id",
        "/message",
        true,
        true,
    );
}

fn codex_facts<'a>(
    facts: &mut Vec<Fact<'a>>,
    payload: Option<&'a LosslessSourceValue>,
    event: &str,
) {
    match event {
        "session_meta" => add_fact(
            facts,
            payload,
            "id",
            "session_metadata_id",
            "/payload",
            true,
            true,
        ),
        "response_item" => {
            add_fact(facts, payload, "id", "item_id", "/payload", true, true);
            add_fact(facts, payload, "call_id", "call_id", "/payload", true, true);
        }
        "event_msg" | "token_usage_record" => {
            for key in ["call_id", "response_id", "thread_id", "turn_id"] {
                add_fact(facts, payload, key, key, "/payload", true, true);
            }
            if matches!(
                payload
                    .and_then(|p| p.get("type"))
                    .and_then(LosslessSourceValue::as_str),
                Some("tool_call" | "function_call")
            ) {
                add_fact(facts, payload, "id", "record_id", "/payload", true, true);
            }
        }
        _ => {}
    }
    add_fact(facts, payload, "cwd", "cwd", "/payload", true, true);
}

fn openclaw_pi_facts<'a>(
    facts: &mut Vec<Fact<'a>>,
    connector: &str,
    value: &'a LosslessSourceValue,
    msg: Option<&'a LosslessSourceValue>,
    event: &str,
) {
    add_fact(
        facts,
        Some(value),
        "id",
        if event == "session" {
            "session_id"
        } else {
            "record_id"
        },
        "",
        true,
        true,
    );
    if connector == "openclaw" {
        if msg
            .and_then(|m| m.get("role"))
            .and_then(LosslessSourceValue::as_str)
            == Some("assistant")
        {
            add_fact(
                facts,
                msg,
                "responseId",
                "provider_response_id",
                "/message",
                true,
                true,
            );
        }
        if event.contains('.') {
            for key in ["sessionId", "runId", "traceId"] {
                add_fact(facts, Some(value), key, key, "", true, true);
            }
        }
    }
}

fn native_facts<'a>(connector: &str, value: &'a LosslessSourceValue) -> Vec<Fact<'a>> {
    let event = value
        .get("type")
        .and_then(LosslessSourceValue::as_str)
        .unwrap_or("");
    let msg = value.get("message");
    let payload = value.get("payload");
    let mut facts = Vec::new();
    match connector {
        "claude_code" => claude_facts(&mut facts, value, msg),
        "codex" => codex_facts(&mut facts, payload, event),
        "openclaw" | "pi_agent" => openclaw_pi_facts(&mut facts, connector, value, msg, event),
        "gemini" => add_fact(
            &mut facts,
            Some(value),
            "sessionId",
            "session_id",
            "",
            true,
            true,
        ),
        _ => {}
    }
    facts
}

fn metadata_facts(value: &LosslessSourceValue) -> Vec<Fact<'_>> {
    let mut facts = Vec::new();
    for (base, object) in [
        ("", Some(value)),
        ("/payload", value.get("payload")),
        ("/message", value.get("message")),
    ] {
        for key in ["version", "cli_version", "cwd"] {
            add_fact(&mut facts, object, key, key, base, key == "cwd", false);
        }
    }
    facts
}

fn write_facts(writer: &mut dyn Write, facts: &[Fact<'_>], metadata: bool) -> std::io::Result<()> {
    writer.write_all(b"{")?;
    for (index, fact) in facts.iter().enumerate() {
        if index > 0 {
            writer.write_all(b",")?;
        }
        serde_json::to_writer(
            &mut *writer,
            if metadata {
                fact.pointer.as_str()
            } else {
                fact.field
            },
        )?;
        writer.write_all(b":")?;
        fact.write(writer)?;
    }
    writer.write_all(b"}")
}

struct ExportSink {
    connector: String,
    spool: BufWriter<File>,
    count: u64,
    claims: BTreeMap<u64, Vec<SourceMappingClaim>>,
    conversation_claims: Vec<SourceMappingClaim>,
}

impl ExportSink {
    fn write_record(&mut self, record: &SourceRecordOutcome) -> std::io::Result<()> {
        let physical = json!({"ordinal": record.locator.ordinal, "record_pointer": record.locator.record_pointer,
            "byte_start": record.locator.byte_start, "byte_end": record.locator.byte_end, "record_byte_len": record.record_byte_len,
            "terminator_len": record.terminator_len, "source_record_sha256": record.source_record_sha256,
            "parse_state": record.parse_state, "frame_state": "activated", "diagnostics": record.diagnostics});
        let mut bytes = serde_json::to_vec(&physical)?;
        bytes.pop();
        self.spool.write_all(&bytes)?;
        self.spool.write_all(b",\"raw_value\":")?;
        if let Some(value) = &record.value {
            value.write_tagged(&mut self.spool)?;
        } else {
            self.spool.write_all(b"null")?;
        }
        self.spool.write_all(b",\"native_ids\":")?;
        let native = record
            .value
            .as_ref()
            .map(|v| native_facts(&self.connector, v))
            .unwrap_or_default();
        write_facts(&mut self.spool, &native, false)?;
        self.spool.write_all(b",\"source_metadata\":")?;
        let metadata = record
            .value
            .as_ref()
            .map(metadata_facts)
            .unwrap_or_default();
        write_facts(&mut self.spool, &metadata, true)?;
        self.spool.write_all(b"}\n")?;
        self.count += 1;
        Ok(())
    }
}

impl SourceRecordSink for ExportSink {
    fn on_record(
        &mut self,
        record: SourceRecordOutcome,
    ) -> Result<SourceSinkControl, SourceSinkFailure> {
        self.write_record(&record).map_err(|_| sink_failure())?;
        Ok(SourceSinkControl::Continue)
    }
}

impl SourceMappingSink for ExportSink {
    fn on_record_claims(
        &mut self,
        locator: SourceRecordLocator,
        claims: Vec<SourceMappingClaim>,
    ) -> Result<SourceSinkControl, SourceSinkFailure> {
        if self.claims.insert(locator.ordinal, claims).is_some() {
            return Err(sink_failure());
        }
        Ok(SourceSinkControl::Continue)
    }
    fn on_conversation(
        &mut self,
        item: SourceMappedConversation,
    ) -> Result<SourceSinkControl, SourceSinkFailure> {
        self.conversation_claims.extend(item.claims);
        Ok(SourceSinkControl::Continue)
    }
    fn on_diagnostic(
        &mut self,
        _item: SourceMappingDiagnostic,
    ) -> Result<SourceSinkControl, SourceSinkFailure> {
        Ok(SourceSinkControl::Continue)
    }
}

fn connector(name: &str) -> Result<Box<dyn Connector>, &'static str> {
    Ok(match name {
        "claude_code" => Box::new(ClaudeCodeConnector::new()),
        "codex" => Box::new(CodexConnector::new()),
        "openclaw" => Box::new(OpenClawConnector::new()),
        "pi_agent" => Box::new(PiAgentConnector::new()),
        "gemini" => Box::new(GeminiConnector::new()),
        _ => return Err("unsupported connector"),
    })
}

fn join_claims(
    spool: &Path,
    output: &Path,
    claims: &BTreeMap<u64, Vec<SourceMappingClaim>>,
) -> Result<(String, u64), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(private_file(output)?);
    let mut hash = Sha256::new();
    let mut count = 0;
    for line in BufReader::new(File::open(spool)?).split(b'\n') {
        let mut line = line?;
        if line.pop() != Some(b'}') {
            return Err("invalid private spool".into());
        }
        // Physical metadata precedes the raw tree. Extract only that bounded prefix.
        let marker = b",\"raw_value\":";
        let at = line
            .windows(marker.len())
            .position(|part| part == marker)
            .ok_or("invalid spool metadata")?;
        let mut prefix = line[..at].to_vec();
        prefix.push(b'}');
        let metadata: Value = serde_json::from_slice(&prefix)?;
        let ordinal = metadata["ordinal"]
            .as_u64()
            .ok_or("invalid spool ordinal")?;
        line.extend_from_slice(b",\"claims\":");
        let entries: &[SourceMappingClaim] = claims.get(&ordinal).map_or(&[], Vec::as_slice);
        serde_json::to_writer(&mut line, &entries)?;
        line.extend_from_slice(b"}\n");
        hash.update(&line);
        writer.write_all(&line)?;
        count += 1;
    }
    writer.flush()?;
    Ok((format!("{:x}", hash.finalize()), count))
}

fn export_input(
    input: &SelectedInput,
    header: &Contexts,
    directory: &Path,
    index: usize,
) -> Result<(Value, bool), Box<dyn std::error::Error>> {
    let sealed = input.sealed(header)?;
    let spool_path = directory.join(format!("spool-{index}.jsonl"));
    let mut sink = ExportSink {
        connector: input.connector.clone(),
        spool: BufWriter::new(private_file(&spool_path)?),
        count: 0,
        claims: BTreeMap::new(),
        conversation_claims: Vec::new(),
    };
    let result = connector(&input.connector)?.scan_source_aware(&sealed, &mut sink);
    let (finish, succeeded) = match result {
        Ok(finish) => (finish, true),
        Err(failure) => (failure.partial_finish, false),
    };
    sink.spool.flush()?;
    let canonical: Vec<_> = sink
        .claims
        .values()
        .flatten()
        .filter(|claim| claim.mapping_state == "canonical")
        .cloned()
        .collect();
    if canonical != sink.conversation_claims {
        return Err("conversation and record claims disagree".into());
    }
    let record_name = format!("records-{index}.jsonl");
    let (digest, count) = join_claims(&spool_path, &directory.join(&record_name), &sink.claims)?;
    if count != sink.count || count != finish.records_seen {
        return Err("record transport count disagrees".into());
    }
    let complete = succeeded
        && finish.parse_failed == 0
        && finish.mapping_diagnostics == 0
        && finish.reached_eof;
    // The reader independently measured and checked this manifest. Its blank
    // byte count is usable only after full physical closure was verified.
    if !finish.reached_eof
        || finish.content_digest != sealed.content_digest
        || finish.bytes_read != sealed.byte_len
    {
        return Err("physical input incomplete".into());
    }
    let output = json!({
        "input_key": sealed.identity_for(finish.content_digest).map_err(|_| "input identity failure")?, "connector": input.connector, "format": input.format,
        "content_digest": finish.content_digest, "sealed_byte_len": finish.bytes_read, "state": if complete { "complete" } else { "partial" }, "reached_eof": finish.reached_eof,
        "counts": {"records": finish.records_seen, "blank_lines": finish.blank_lines, "blank_bytes": sealed.expected_manifest.blank_bytes},
        "records": [], "records_ref": record_name, "records_sha256": digest, "records_count": count, "records_complete": true,
        "source_id": input.source_id, "identity_namespace": input.identity_namespace, "role": input.role, "origin_kind": input.origin_kind, "origin_host": input.origin_host,
        "source_version": input.source_version, "platform": input.platform, "relative_locator": input.relative_locator, "logical_root": input.logical_root,
        "logical_source_path": input.logical_source_path, "shape_root": input.shape_root,
        "parser_context_digest": sealed.parser_context.digest().map_err(|_| "context digest failure")?, "parser_context": sealed.parser_context.to_wire().map_err(|_| "context encoding failure")?,
        "discovery": sealed.discovery.to_wire().map_err(|_| "descriptor encoding failure")?, "transport": input.transport,
        "finish": finish, "expected_manifest": sealed.expected_manifest,
    });
    Ok((output, complete))
}

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 || args[0] != "--inputs" || args[2] != "--output" {
        return Err("expected --inputs FILE --output FILE".into());
    }
    let header: Contexts = serde_json::from_reader(File::open(&args[1])?)?;
    if header.schema != "cass.source-oracle.export-contexts/1"
        || header.coverage != "source_five_v1"
        || header.reader_version != 1
        || !header.redaction_enabled
        || header.safety_config_file_sha256.len() != 64
        || header.safety_config_effective_sha256.len() != 64
    {
        return Err("unsupported export context".into());
    }
    let output = Path::new(&args[3]);
    let directory = output.parent().ok_or("missing output directory")?;
    fs::create_dir_all(directory)?;
    let mut inputs = Vec::new();
    let mut complete = true;
    for (index, input) in header.inputs.iter().enumerate() {
        let (record, done) = export_input(input, &header, directory, index)?;
        inputs.push(record);
        complete &= done;
    }
    let bundle = json!({"schema": "cass.source-oracle.fad-raw-ir/1", "role": "candidate", "surface": "fad_raw_ir", "reader_version": header.reader_version,
        "library_identity": header.library_identity, "source_facts_revision": header.source_facts_revision, "normalizer_id": header.normalizer_id,
        "safety_contract_id": header.safety_contract_id, "coverage": header.coverage, "inputs": inputs});
    let mut writer = BufWriter::new(private_file(output)?);
    serde_json::to_writer(&mut writer, &bundle)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(complete)
}

fn main() {
    // Never print filesystem, serde, or payload error text from private inputs.
    match run() {
        Ok(true) => println!("source export complete"),
        Ok(false) => {
            eprintln!("source export incomplete");
            std::process::exit(3);
        }
        Err(_) => {
            eprintln!("source export failed");
            std::process::exit(3);
        }
    }
}
