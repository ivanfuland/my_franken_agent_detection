//! Message projections whose source positions are recorded at each parser push.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::source_records::{
    SealedInputContext, SourceDiagnostic, SourceInputFinish, SourceParseState, SourceReadFailure,
    SourceReadFailureKind, SourceRecordLocator, SourceRecordOutcome, SourceRecordSink,
    SourceSinkControl, SourceSinkFailure, read_source_records,
};
use crate::{NormalizedConversation, NormalizedMessage};

/// An original JSON pointer. Debug output deliberately does not expose member names.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourcePointer(String);

impl SourcePointer {
    /// Construct a pointer from a parser-known source position.
    pub fn new(pointer: String) -> Result<Self, SourceReadFailureKind> {
        if !pointer.is_empty() && !pointer.starts_with('/') {
            return Err(SourceReadFailureKind::InvalidContext);
        }
        let bytes = pointer.as_bytes();
        for (index, byte) in bytes.iter().enumerate() {
            if *byte == b'~' && !matches!(bytes.get(index + 1), Some(b'0' | b'1')) {
                return Err(SourceReadFailureKind::InvalidContext);
            }
        }
        Ok(Self(pointer))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SourcePointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SourcePointer(..)")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceMappingClaim {
    pub record_ordinal: u64,
    pub record_pointer: String,
    pub conversation_ordinal: Option<u32>,
    pub final_message_idx: Option<i64>,
    pub source_block_pointers: Vec<SourcePointer>,
    pub projection_pointer: Option<String>,
    pub evidence_kind: String,
    pub mapping_state: String,
}

impl SourceMappingClaim {
    fn no_message(locator: &SourceRecordLocator, failed: bool) -> Self {
        Self {
            record_ordinal: locator.ordinal,
            record_pointer: locator.record_pointer.clone(),
            conversation_ordinal: None,
            final_message_idx: None,
            source_block_pointers: Vec::new(),
            projection_pointer: None,
            evidence_kind: if failed { "parse_failed" } else { "no_message" }.into(),
            mapping_state: if failed { "unmapped" } else { "standalone" }.into(),
        }
    }
}

#[derive(Debug)]
pub struct SourceMappedConversation {
    pub conversation_ordinal: u32,
    pub conversation: NormalizedConversation,
    pub claims: Vec<SourceMappingClaim>,
}

pub type SourceMappingDiagnostic = SourceDiagnostic;

/// Original records arrive first. Conversations and claims require whole-input
/// projection success. Callback failure never changes the physical EOF fact.
pub trait SourceMappingSink: SourceRecordSink {
    fn on_record_claims(
        &mut self,
        locator: SourceRecordLocator,
        claims: Vec<SourceMappingClaim>,
    ) -> Result<SourceSinkControl, SourceSinkFailure>;
    fn on_conversation(
        &mut self,
        item: SourceMappedConversation,
    ) -> Result<SourceSinkControl, SourceSinkFailure>;
    fn on_diagnostic(
        &mut self,
        item: SourceMappingDiagnostic,
    ) -> Result<SourceSinkControl, SourceSinkFailure>;
}

/// Internal parser state; implementations record claims at their message push.
pub(crate) trait SourceProjector {
    fn project(
        &mut self,
        value: &Value,
        locator: &SourceRecordLocator,
    ) -> Result<Vec<SourceMappingClaim>, &'static str>;
    fn finish(self) -> Option<NormalizedConversation>;
}

/// Append a message and its source proof in the same operation.
pub(crate) fn push_message(
    messages: &mut Vec<NormalizedMessage>,
    claims: &mut Vec<SourceMappingClaim>,
    locator: &SourceRecordLocator,
    pointers: Vec<String>,
    evidence: &str,
    mut message: NormalizedMessage,
) -> Result<(), &'static str> {
    let idx = i64::try_from(messages.len()).map_err(|_| "source_counter_overflow")?;
    message.idx = idx;
    let pointers = pointers
        .into_iter()
        .map(SourcePointer::new)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "source_pointer_invalid")?;
    messages.push(message);
    claims.push(SourceMappingClaim {
        record_ordinal: locator.ordinal,
        record_pointer: locator.record_pointer.clone(),
        conversation_ordinal: Some(0),
        final_message_idx: Some(idx),
        source_block_pointers: pointers,
        projection_pointer: Some(format!("/messages/{idx}")),
        evidence_kind: evidence.into(),
        mapping_state: "canonical".into(),
    });
    Ok(())
}

struct MappingReader<'a, P> {
    projector: P,
    sink: &'a mut dyn SourceMappingSink,
    records: Vec<(SourceRecordLocator, Vec<SourceMappingClaim>)>,
    rejections: u64,
}

impl<P: SourceProjector> SourceRecordSink for MappingReader<'_, P> {
    fn on_record(
        &mut self,
        mut record: SourceRecordOutcome,
    ) -> Result<SourceSinkControl, SourceSinkFailure> {
        let mut claims = Vec::new();
        if let Some(value) = record
            .value
            .as_ref()
            .and_then(crate::LosslessSourceValue::legacy_value)
        {
            match self.projector.project(&value, &record.locator) {
                Ok(projected) => claims = projected,
                Err(reason) => {
                    self.rejections += 1;
                    record.diagnostics.push(SourceDiagnostic {
                        code: "source_projection_rejected".into(),
                        detail: Some(reason.into()),
                        offset: None,
                        pointer: None,
                    });
                }
            }
        }
        if claims.is_empty() {
            claims.push(SourceMappingClaim::no_message(
                &record.locator,
                record.parse_state == SourceParseState::ParseFailed,
            ));
        }
        self.records.push((record.locator.clone(), claims));
        self.sink.on_record(record)
    }
}

// `SourceReadFailure` carries the frozen inline recovery state (partial `Finish`,
// next cursor and diagnostic), so the error type is large by contract; boxing it
// would change the published reader signatures. Layout exception recorded under the
// 2026-10-10 clippy ruling, not a blanket crate exemption.
#[allow(clippy::result_large_err)]
fn callback_result(
    result: &Result<SourceSinkControl, SourceSinkFailure>,
    finish: &SourceInputFinish,
) -> Result<(), SourceReadFailure> {
    let kind = match result {
        Ok(SourceSinkControl::Continue) => return Ok(()),
        Ok(SourceSinkControl::Stop) => SourceReadFailureKind::Stopped,
        Err(_) => SourceReadFailureKind::Sink,
    };
    Err(SourceReadFailure {
        kind,
        partial_finish: finish.clone(),
        next_cursor: None,
        diagnostic: SourceDiagnostic::code("mapping_delivery_incomplete"),
    })
}

// Layout exception recorded under the 2026-10-10 clippy ruling: `SourceReadFailure`
// is the frozen inline recovery state, so the error type is large by contract and
// boxing it would change the published reader signature.
#[allow(clippy::result_large_err)]
pub(crate) fn scan_projected<P: SourceProjector>(
    sealed: &SealedInputContext,
    sink: &mut dyn SourceMappingSink,
    projector: P,
) -> Result<SourceInputFinish, SourceReadFailure> {
    let mut reader = MappingReader {
        projector,
        sink,
        records: Vec::new(),
        rejections: 0,
    };
    let mut finish = match read_source_records(sealed, None, &mut reader) {
        Ok(finish) => finish,
        Err(mut failure) => {
            failure.next_cursor = None;
            failure.partial_finish.mapping_diagnostics = reader.rejections;
            return Err(failure);
        }
    };
    finish.mapping_diagnostics = reader.rejections;
    if reader.rejections > 0 {
        // The whole-input rejection already carries a value-free reason. Deliver
        // that same registered diagnostic to the mapping sink before the typed
        // failure, so the callback path is real and observes the physical EOF the
        // completed read actually reached. A sink Stop or error here keeps that
        // EOF and reports a partial input; it never rewrites the read facts.
        callback_result(
            &reader
                .sink
                .on_diagnostic(SourceDiagnostic::code("source_projection_rejected")),
            &finish,
        )?;
        return Err(SourceReadFailure {
            kind: SourceReadFailureKind::ProjectionRejected,
            partial_finish: finish,
            next_cursor: None,
            diagnostic: SourceDiagnostic::code("source_projection_rejected"),
        });
    }
    if let Some(conversation) = reader.projector.finish() {
        let claims = reader
            .records
            .iter()
            .flat_map(|(_, claims)| claims)
            .filter(|claim| claim.mapping_state == "canonical")
            .cloned()
            .collect();
        callback_result(
            &reader.sink.on_conversation(SourceMappedConversation {
                conversation_ordinal: 0,
                conversation,
                claims,
            }),
            &finish,
        )?;
    }
    for (locator, claims) in reader.records {
        callback_result(&reader.sink.on_record_claims(locator, claims), &finish)?;
    }
    Ok(finish)
}
