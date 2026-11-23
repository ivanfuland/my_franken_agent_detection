//! Lossless JSON value representation preserving member order and raw numeric tokens.

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

/// A JSON number token, validated before it can enter a lossless tree.
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

    /// Write the six-tag wire representation without a recursive serializer.
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
}

// Destruction must not reintroduce a stack-depth limit into the lossless parser.
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn number_text_before_value() {
        let accepted = [
            "-9223372036854775809",
            "-9223372036854775808",
            "9223372036854775807",
            "9223372036854775808",
            "18446744073709551615",
            "18446744073709551616",
            "9007199254740993",
            "1.0",
            "1e3",
            "1000e-3",
            "-0",
            "-0.0",
            "-0e5",
            "1e40000000",
            "1e-40000000",
            "0.1000000000000000000000000000000000000001",
        ];
        let rejected = [
            "01",
            "1.",
            ".5",
            "+1",
            "1e",
            "NaN",
            "Infinity",
            "-Infinity",
            "",
            "1 2",
        ];
        for text in accepted {
            let value = LosslessSourceValue::parse(text.as_bytes()).unwrap();
            let LosslessSourceValue::Number(number) = &value else {
                panic!("wrong value kind");
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
        for text in rejected {
            assert!(ValidatedJsonNumberText::new(text.into()).is_err());
            assert!(LosslessSourceValue::parse(text.as_bytes()).is_err());
        }
        let raw = br#"{"$serde_json::private::Number":"ordinary","$serde_json::private::RawValue":{"x":-0},"array":[null,false,0,"",{},[]]}"#;
        let value = LosslessSourceValue::parse(raw).unwrap();
        assert_eq!(
            value
                .get("$serde_json::private::Number")
                .and_then(LosslessSourceValue::as_str),
            Some("ordinary")
        );
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
    fn parse_errors_do_not_leak_source_keys() {
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
}
