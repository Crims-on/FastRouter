//! Line-oriented SSE / NDJSON parsing (port of utils/streamHelpers.js parseSSELine).

use serde_json::Value;

/// Splits a byte stream into lines, holding incomplete UTF-8 and partial lines.
#[derive(Default)]
pub struct LineParser {
    buf: String,
    pending: Vec<u8>,
}

impl LineParser {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(chunk);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        let text: Vec<u8> = self.pending.drain(..valid).collect();
        self.buf.push_str(&String::from_utf8_lossy(&text));
        let mut out = vec![];
        while let Some(pos) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=pos).collect();
            out.push(line.trim_end_matches(['\n', '\r']).to_string());
        }
        out
    }

    /// Remaining partial line at end of stream.
    pub fn finish(&mut self) -> Option<String> {
        if !self.pending.is_empty() {
            let rest: Vec<u8> = std::mem::take(&mut self.pending);
            self.buf.push_str(&String::from_utf8_lossy(&rest));
        }
        let rest = std::mem::take(&mut self.buf);
        (!rest.trim().is_empty()).then(|| rest.trim().to_string())
    }
}

pub enum Parsed {
    Done,
    Value(Value),
}

impl Parsed {
    pub fn is_done(&self) -> bool {
        matches!(self, Parsed::Done)
    }
    pub fn into_value(self) -> Value {
        match self {
            Parsed::Done => serde_json::json!({"done": true}),
            Parsed::Value(v) => v,
        }
    }
}

/// parseSSELine(line, format): `data: {...}` lines, or raw JSON lines for NDJSON (ollama).
pub fn parse_sse_line(line: &str, format: Option<&str>) -> Option<Parsed> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if format == Some("ollama") {
        if line.starts_with('{') {
            return serde_json::from_str(line).ok().map(Parsed::Value);
        }
        return None;
    }
    if !line.starts_with('d') {
        return None;
    }
    let data = line.get(5..)?.trim();
    if data == "[DONE]" {
        return Some(Parsed::Done);
    }
    serde_json::from_str(data).ok().map(Parsed::Value)
}

/// Collects `event:`/`data:` blocks (blank-line separated) — used where the
/// event name matters (Responses API passthrough, Claude).
#[derive(Default)]
pub struct EventParser {
    lines: LineParser,
    event: Option<String>,
    data: Vec<String>,
}

pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

impl EventParser {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        let mut out = vec![];
        for line in self.lines.push(chunk) {
            self.line(&line, &mut out);
        }
        out
    }

    fn line(&mut self, line: &str, out: &mut Vec<SseEvent>) {
        if line.trim().is_empty() {
            if !self.data.is_empty() {
                out.push(SseEvent { event: self.event.take(), data: self.data.join("\n") });
                self.data.clear();
            }
            self.event = None;
            return;
        }
        if let Some(v) = line.strip_prefix("event:") {
            self.event = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("data:") {
            self.data.push(v.strip_prefix(' ').unwrap_or(v).to_string());
        }
    }

    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut out = vec![];
        if let Some(l) = self.lines.finish() {
            self.line(&l, &mut out);
        }
        if !self.data.is_empty() {
            out.push(SseEvent { event: self.event.take(), data: self.data.join("\n") });
            self.data.clear();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_and_events() {
        let mut p = LineParser::default();
        assert!(p.push(b"data: {\"a\"").is_empty());
        let l = p.push(b":1}\r\n\r\ndata: [DONE]\n");
        assert_eq!(l[0], "data: {\"a\":1}");
        assert!(parse_sse_line(&l[2], None).unwrap().is_done());
        let mut e = EventParser::default();
        let evs = e.push(b"event: x\ndata: 1\n\nevent: y\ndata: 2");
        assert_eq!(evs.len(), 1);
        let rest = e.finish();
        assert_eq!(rest[0].event.as_deref(), Some("y"));
    }
}
