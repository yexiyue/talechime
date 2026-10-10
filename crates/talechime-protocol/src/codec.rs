use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashSet;
use std::io::{BufRead, Read};

/// Framing and protocol validation failure.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("message exceeds {0} bytes")]
    TooLarge(usize),
    #[error("unsupported protocol major version {0}")]
    Version(u32),
    #[error("request_id must not be empty")]
    EmptyRequestId,
    #[error("request id {0} was already used")]
    Duplicate(String),
    #[error("request limit reached; reconnect before sending more commands")]
    RequestLimit,
    #[error("invalid JSON message: {0}")]
    Json(#[from] serde_json::Error),
    #[error("protocol IO failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Read a single bounded frame. Oversize input is fatal to this connection.
/// EOF with no data means the parent disconnected; an unterminated frame is invalid.
pub fn read_frame<R: BufRead>(reader: &mut R) -> Result<Option<Vec<u8>>, ProtocolError> {
    let mut line = Vec::new();
    let count = reader
        .take((crate::MAX_MESSAGE_BYTES + 2) as u64)
        .read_until(b'\n', &mut line)?;
    if count == 0 {
        return Ok(None);
    }
    let payload = line.strip_suffix(b"\n").unwrap_or(&line);
    let payload = payload.strip_suffix(b"\r").unwrap_or(payload);
    if payload.len() > crate::MAX_MESSAGE_BYTES {
        return Err(ProtocolError::TooLarge(crate::MAX_MESSAGE_BYTES));
    }
    if !line.ends_with(b"\n") {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "unterminated protocol frame",
        )
        .into());
    }
    Ok(Some(line))
}

/// Serialize one message, including a newline. The limit excludes that newline.
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, ProtocolError> {
    let mut data = serde_json::to_vec(value)?;
    if data.len() > crate::MAX_MESSAGE_BYTES {
        return Err(ProtocolError::TooLarge(crate::MAX_MESSAGE_BYTES));
    }
    data.push(b'\n');
    Ok(data)
}

/// Decode one line. Readers must also bound buffering before calling this.
pub fn decode<T: DeserializeOwned>(line: &[u8]) -> Result<T, ProtocolError> {
    let data = line.strip_suffix(b"\n").unwrap_or(line);
    let data = data.strip_suffix(b"\r").unwrap_or(data);
    if data.len() > crate::MAX_MESSAGE_BYTES {
        return Err(ProtocolError::TooLarge(crate::MAX_MESSAGE_BYTES));
    }
    Ok(serde_json::from_slice(data)?)
}

/// Bounded deduplication for one connection. Never evicts IDs and re-executes them.
#[derive(Debug, Default)]
pub struct RequestTracker {
    seen: HashSet<String>,
}

impl RequestTracker {
    /// Validate the version/id before accepting a request for execution.
    pub fn accept(&mut self, request: &crate::Request) -> Result<(), ProtocolError> {
        if request.protocol_version != crate::PROTOCOL_VERSION {
            return Err(ProtocolError::Version(request.protocol_version));
        }
        if request.request_id.is_empty() {
            return Err(ProtocolError::EmptyRequestId);
        }
        if self.seen.contains(&request.request_id) {
            return Err(ProtocolError::Duplicate(request.request_id.clone()));
        }
        if self.seen.len() >= 65_536 {
            return Err(ProtocolError::RequestLimit);
        }
        self.seen.insert(request.request_id.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn request() -> Request {
        Request {
            protocol_version: PROTOCOL_VERSION,
            request_id: "1".into(),
            session_id: Some("chapter-1".into()),
            command: Command::Start(Box::new(PlanRequest {
                source: SourceId {
                    namespace: "reader".into(),
                    book: "书".into(),
                    chapter: "1".into(),
                },
                text: "你好🙂\r\n“再见。”".into(),
                text_hash: "digest".into(),
                backend: "fixture".into(),
                model: None,
                voices: vec!["A".into()],
                spans: vec![],
                sealed: false,
                playback: PlanPlayback::Streaming,
                resume_byte: None,
                restore_checkpoint: true,
                continuation: true,
                params: Default::default(),
                seed: None,
                verification: Default::default(),
            })),
        }
    }

    #[test]
    fn continuation_defaults_on_and_can_be_disabled_without_changing_v7() {
        let mut json = serde_json::to_value(request()).unwrap();
        json["payload"]
            .as_object_mut()
            .unwrap()
            .remove("continuation");
        let parsed: Request = serde_json::from_value(json.clone()).unwrap();
        assert!(matches!(parsed.command,Command::Start(plan) if plan.continuation));
        json["payload"]["continuation"] = serde_json::json!(false);
        let parsed: Request = serde_json::from_value(json).unwrap();
        assert!(matches!(parsed.command,Command::Start(plan) if !plan.continuation));
        assert_eq!(PROTOCOL_VERSION, 7);
    }

    #[test]
    fn params_and_seed_default_off_and_roundtrip_without_changing_v7() {
        let mut json = serde_json::to_value(request()).unwrap();
        json["payload"].as_object_mut().unwrap().remove("params");
        json["payload"].as_object_mut().unwrap().remove("seed");
        let parsed: Request = serde_json::from_value(json.clone()).unwrap();
        assert!(matches!(parsed.command, Command::Start(plan)
            if plan.params.is_empty() && plan.seed.is_none()));
        json["payload"]["params"] =
            serde_json::json!({"steps": 32, "temperature": 0.8, "language": "english"});
        json["payload"]["seed"] = serde_json::json!(7);
        let parsed: Request = serde_json::from_value(json).unwrap();
        assert!(matches!(parsed.command, Command::Start(plan)
            if plan.params.get("steps") == Some(&ParamValue::Int(32))
                && plan.params.get("temperature") == Some(&ParamValue::Float(0.8))
                && plan.params.get("language") == Some(&ParamValue::Text("english".into()))
                && plan.seed == Some(7)));
        assert_eq!(PROTOCOL_VERSION, 7);
    }
    #[test]
    fn removed_alignment_settings_are_rejected_in_config_updates() {
        for patch in [
            r#"{"expected_revision":0,"alignment_enabled":true}"#,
            r#"{"expected_revision":0,"alignment_device":"cpu"}"#,
        ] {
            assert!(serde_json::from_str::<crate::ConfigPatch>(patch).is_err());
        }
    }

    #[test]
    fn unicode_and_newlines_preserve_the_exact_snapshot() {
        let expected = request();
        let line = encode(&expected).unwrap();
        assert_eq!(line.iter().filter(|&&b| b == b'\n').count(), 1);
        assert_eq!(decode::<Request>(&line).unwrap(), expected);
    }

    #[test]
    fn invalid_and_oversized_messages_are_rejected() {
        assert!(matches!(
            decode::<Request>(b"bad"),
            Err(ProtocolError::Json(_))
        ));
        assert!(matches!(
            decode::<Request>(&vec![b' '; MAX_MESSAGE_BYTES + 1]),
            Err(ProtocolError::TooLarge(_))
        ));
    }

    #[test]
    fn duplicate_commands_and_other_major_versions_are_rejected() {
        let mut tracker = RequestTracker::default();
        let mut req = request();
        tracker.accept(&req).unwrap();
        assert!(matches!(
            tracker.accept(&req),
            Err(ProtocolError::Duplicate(_))
        ));
        req.protocol_version += 1;
        assert!(matches!(
            tracker.accept(&req),
            Err(ProtocolError::Version(_))
        ));
    }

    #[test]
    fn terminal_reasons_remain_distinct_on_the_wire() {
        for reason in [
            EndReason::Completed,
            EndReason::Cancelled,
            EndReason::Failed,
        ] {
            let event = Event::SessionEnded {
                reason,
                text_hash: "hash".into(),
            };
            assert_eq!(decode::<Event>(&encode(&event).unwrap()).unwrap(), event);
        }
    }

    #[test]
    fn ranges_check_unicode_boundaries_and_order() {
        let text = "中🙂\r\n";
        assert!(TextRange { start: 3, end: 7 }.is_valid(text));
        assert!(!TextRange { start: 4, end: 7 }.is_valid(text));
        assert!(!TextRange { start: 7, end: 3 }.is_valid(text));
    }

    #[test]
    fn preferences_keep_values_and_unknown_fields() {
        let config: Config =
            serde_json::from_str(include_str!("../tests/fixtures/preferences.json")).unwrap();
        assert_eq!(config.backend, Config::default().backend);
        assert_eq!(config.voice, "Weiguo");
        assert_eq!(config.speed, 1.3);
        assert_eq!(
            serde_json::from_slice::<Config>(&serde_json::to_vec(&config).unwrap()).unwrap(),
            config
        );
    }

    #[test]
    fn missing_config_fields_use_the_same_defaults_as_new_preferences() {
        assert_eq!(
            serde_json::from_str::<Config>("{}").unwrap(),
            Config::default()
        );
    }

    #[test]
    fn framing_bounds_memory_before_decoding_and_distinguishes_eof() {
        let mut data = std::io::Cursor::new(vec![b'x'; MAX_MESSAGE_BYTES + 100]);
        assert!(matches!(
            read_frame(&mut data),
            Err(ProtocolError::TooLarge(_))
        ));
        assert_eq!(data.position(), (MAX_MESSAGE_BYTES + 2) as u64);
        assert!(
            read_frame(&mut std::io::Cursor::new(b""))
                .unwrap()
                .is_none()
        );
        assert!(read_frame(&mut std::io::Cursor::new(b"{}")).is_err());
    }

    #[test]
    fn framing_does_not_consume_the_next_message() {
        let mut data = std::io::Cursor::new(b"{}\n{}\r\n");
        assert_eq!(read_frame(&mut data).unwrap().unwrap(), b"{}\n");
        assert_eq!(read_frame(&mut data).unwrap().unwrap(), b"{}\r\n");
    }
}

#[cfg(test)]
mod plan_tests {
    use crate::*;
    #[test]
    fn plan_commands_roundtrip_with_exact_unicode_snapshot() {
        let start = PlanRequest {
            source: SourceId {
                namespace: "test".into(),
                book: "书".into(),
                chapter: "一".into(),
            },
            text: "中🙂\r\n".into(),
            text_hash: text_hash("中🙂\r\n"),
            backend: "fixture".into(),
            model: None,
            voices: vec!["A".into()],
            spans: vec![VoiceSpan {
                range: TextRange { start: 0, end: 9 },
                voice: "A".into(),
                style: None,
            }],
            sealed: true,
            playback: PlanPlayback::AfterChapterReady,
            resume_byte: Some(0),
            restore_checkpoint: false,
            continuation: true,
            params: Default::default(),
            seed: None,
            verification: Default::default(),
        };
        for command in [
            Command::Start(Box::new(start)),
            Command::Append { spans: vec![] },
            Command::Seal,
            Command::FailInput {
                message: "failed".into(),
            },
            Command::GetProgress,
        ] {
            let request = Request {
                protocol_version: PROTOCOL_VERSION,
                request_id: "opaque".into(),
                session_id: Some("execution".into()),
                command,
            };
            assert_eq!(
                decode::<Request>(&encode(&request).unwrap()).unwrap(),
                request
            );
        }
        let hello = Request {
            protocol_version: PROTOCOL_VERSION,
            request_id: "hello".into(),
            session_id: None,
            command: Command::Hello,
        };
        assert_eq!(encode(&hello).unwrap(),b"{\"protocol_version\":7,\"request_id\":\"hello\",\"session_id\":null,\"type\":\"hello\"}\n");
    }
    #[test]
    fn typo_in_plan_control_fields_is_rejected() {
        let span = r#"{"range":{"start":0,"end":3},"voice":"A","style":null,"role_id":"ignored?"}"#;
        assert!(serde_json::from_str::<VoiceSpan>(span).is_err());
    }
}
