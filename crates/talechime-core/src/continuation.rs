//! Bounded, execution-local references made from complete raw generation.
use crate::backend::{BackendError, Pcm};
use std::sync::Arc;

pub(crate) const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_TEXT_BYTES: usize = 2048;
const MIN_SECONDS: u64 = 1;
const MAX_SECONDS: u64 = 15;
/// Previous complete utterance and its unprocessed PCM. Cloning shares immutable data.
#[derive(Debug, Clone)]
pub struct SpeechContext {
    text: Arc<str>,
    pcm: Arc<Pcm>,
}
impl SpeechContext {
    /// Validate a whole reference; partial audio must never carry a full transcript.
    pub fn new(text: impl Into<Arc<str>>, pcm: Pcm) -> Result<Self, BackendError> {
        let text = text.into();
        let duration = pcm.samples.len() as f64
            / f64::from(pcm.channels.max(1))
            / f64::from(pcm.sample_rate.max(1));
        pcm.duration_ms()?;
        if text.trim().is_empty()
            || text.len() > MAX_TEXT_BYTES
            || !(MIN_SECONDS as f64..=MAX_SECONDS as f64).contains(&duration)
            || pcm.samples.len() > MAX_BYTES / 4
            || !pcm.samples.iter().any(|v| v.abs() > 0.0001)
        {
            return Err(BackendError::Unsupported(
                "continuation reference requires full non-silent 1..15s PCM and 1..2048 text bytes"
                    .into(),
            ));
        }
        Ok(Self {
            text,
            pcm: Arc::new(pcm),
        })
    }
    /// Exact spoken text paired with the reference audio.
    pub fn text(&self) -> &str {
        &self.text
    }
    /// Raw audio before boundary trimming, volume or playback speed.
    pub fn pcm(&self) -> &Pcm {
        &self.pcm
    }
}

pub(crate) struct Candidate {
    pcm: Option<Pcm>,
    previous_bytes: usize,
    discarded: bool,
}
impl Candidate {
    pub(crate) fn new(previous: Option<&SpeechContext>, enabled: bool, text: &str) -> Self {
        if enabled && text.len() > MAX_TEXT_BYTES {
            eprintln!("continuation reset: complete reference text exceeds 2048 UTF-8 bytes");
        }
        Self {
            pcm: None,
            previous_bytes: previous.map_or(0, |p| p.pcm.samples.len() * 4),
            discarded: !enabled || text.len() > MAX_TEXT_BYTES,
        }
    }
    pub(crate) fn push(&mut self, chunk: &Pcm) -> Result<(), BackendError> {
        if self.discarded {
            return Ok(());
        }
        let pcm = self.pcm.get_or_insert_with(|| Pcm {
            samples: Vec::new(),
            sample_rate: chunk.sample_rate,
            channels: chunk.channels,
        });
        if (pcm.sample_rate, pcm.channels) != (chunk.sample_rate, chunk.channels) {
            return Err(BackendError::Synthesis(
                "PCM format changed inside continuation candidate".into(),
            ));
        }
        let samples = pcm.samples.len().saturating_add(chunk.samples.len());
        if samples > (MAX_BYTES - self.previous_bytes) / 4
            || samples as u64 > u64::from(pcm.sample_rate) * u64::from(pcm.channels) * MAX_SECONDS
        {
            self.pcm = None;
            self.discarded = true;
            eprintln!("continuation reset: complete reference exceeds duration or PCM budget");
        } else {
            // Exact reservation keeps the retained PCM allocation inside the combined budget.
            pcm.samples.reserve_exact(chunk.samples.len());
            pcm.samples.extend_from_slice(&chunk.samples);
        }
        Ok(())
    }
    pub(crate) fn finish(self, text: &str) -> Option<SpeechContext> {
        self.pcm
            .and_then(|pcm| match SpeechContext::new(text, pcm) {
                Ok(context) => Some(context),
                Err(error) => {
                    eprintln!("continuation reset: {error}");
                    None
                }
            })
    }
}

/// Examine complete source lines, so a soft newline at a span boundary stays soft.
pub(crate) fn hard_boundary(source: &str, start: usize, end: usize) -> bool {
    let anchor = if start > 0 && source.as_bytes().get(start - 1..start + 1) == Some(b"\r\n") {
        start - 1
    } else {
        start
    };
    let line_start = source[..anchor].rfind(['\r', '\n']).map_or(0, |i| i + 1);
    let mut at = line_start;
    while at < end {
        let rest = &source[at..];
        let length = rest.find(['\r', '\n']).unwrap_or(rest.len());
        let line = &rest[..length];
        let hard = crate::text::is_heading_line(line) || crate::text::is_non_spoken_line(line);
        if hard && (at >= start || !line.trim().is_empty()) {
            return true;
        }
        at += length;
        if source[at..].starts_with("\r\n") {
            at += 2;
        } else if at < source.len() {
            at += 1;
        } else {
            break;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pcm(seconds: usize, value: f32) -> Pcm {
        Pcm {
            samples: vec![value; seconds * 1000],
            sample_rate: 1000,
            channels: 1,
        }
    }
    #[test]
    fn full_references_only_and_shared_raw_pcm() {
        for (text, audio) in [
            ("a", pcm(0, 0.1)),
            ("a", pcm(16, 0.1)),
            ("a", pcm(1, 0.0)),
            ("", pcm(1, 0.1)),
            ("a", pcm(1, f32::NAN)),
        ] {
            assert!(SpeechContext::new(text, audio).is_err());
        }
        assert!(SpeechContext::new("字".repeat(683), pcm(1, 0.1)).is_err());
        for seconds in [1, 15] {
            let context = SpeechContext::new("a", pcm(seconds, 0.1)).unwrap();
            assert!(Arc::ptr_eq(&context.pcm, &context.clone().pcm));
        }
    }
    #[test]
    fn overflow_drops_entire_candidate_and_keeps_previous_snapshot() {
        let previous = SpeechContext::new("a", pcm(1, 0.2)).unwrap();
        let mut candidate = Candidate::new(Some(&previous), true, "b");
        candidate.push(&pcm(15, 0.1)).unwrap();
        candidate.push(&pcm(1, 0.1)).unwrap();
        assert!(candidate.finish("b").is_none());
        assert_eq!(previous.pcm().samples[0], 0.2);
        let high = Pcm {
            samples: vec![0.1; 2_400_000],
            sample_rate: 160000,
            channels: 1,
        };
        let previous = SpeechContext::new("a", high).unwrap();
        let mut candidate = Candidate::new(Some(&previous), true, "b");
        let chunk = Pcm {
            samples: vec![0.1; 2_000_000],
            sample_rate: 160000,
            channels: 1,
        };
        candidate.push(&chunk).unwrap();
        assert!(candidate.finish("b").is_none());
    }
    #[test]
    fn source_boundaries_distinguish_soft_lines_and_crlf() {
        for text in ["甲\n乙", "甲\r\n乙", "甲\r乙"] {
            let start = text.find('乙').unwrap();
            assert!(!hard_boundary(text, 0, text.len()));
            assert!(!hard_boundary(text, start - 1, start));
        }
        for text in [
            "甲\n\n乙",
            "甲\r\n\r\n乙",
            "甲\n---\n乙",
            "甲\n第二章\n乙",
            "甲\n\u{200b}\n乙",
        ] {
            assert!(hard_boundary(text, 0, text.len()), "{text:?}");
        }
    }
}
