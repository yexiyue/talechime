//! Serial synthesis, duration learning and boundary processing ahead of playback.
use super::*;
pub(super) enum Item {
    Verification(crate::verification::VerificationReport),
    Start(TextRange),
    Audio(Packet),
    End(TextRange),
    Skipped(TextRange),
    Finished,
}
pub(super) fn spawn(
    backend: Rc<dyn Backend>,
    source_text: Arc<str>,
    input: Rc<PlanInput>,
    byte: usize,
    tracked: Option<Arc<PendingWrites>>,
    completed: Option<tokio::sync::watch::Sender<bool>>,
) -> (AbortOnDrop, mpsc::Receiver<Result<Item, SessionError>>) {
    struct Completion(tokio::sync::watch::Sender<bool>);
    impl Drop for Completion {
        fn drop(&mut self) {
            self.0.send_replace(true);
        }
    }
    struct Lifetime {
        backend: Rc<dyn Backend>,
        text: Arc<str>,
        input: Rc<PlanInput>,
        _tracked: Option<WriteGuard>,
        _completed: Option<Completion>,
    }
    let lifetime = Lifetime {
        backend,
        text: source_text,
        input,
        _tracked: tracked.map(|pending| {
            pending.count.fetch_add(1, Ordering::SeqCst);
            WriteGuard(pending)
        }),
        _completed: completed.map(Completion),
    };
    let (tx, rx) = mpsc::channel(1);
    let task = AbortOnDrop(tokio::task::spawn_local(async move {
        let lifetime = lifetime;
        let backend = &lifetime.backend;
        let source_text = &lifetime.text;
        let input = &lifetime.input;
        let result: Result<(), SessionError> = async {
            let budget = Budget::default();
            let mut index = 0;
            while let Some(span) = input.next(index).await {
                index += 1;
                let assigned = span.range();
                if assigned.end <= byte {
                    continue;
                }
                backend.select_voice(span.voice());
                let mut at = assigned.start.max(byte);
                while at < assigned.end {
                    // The backend sees only this assignment: it cannot merge across voices.
                    let remaining = &source_text[at..assigned.end];
                    let Some(mut segment) = backend.next_segment(remaining).await? else {
                        send_skipped(&tx, source_text, at, assigned.end).await?;
                        break;
                    };
                    let relative = TextRange {
                        start: segment.start,
                        end: segment.end,
                    };
                    if !relative.is_valid(remaining)
                        || relative.start >= relative.end
                        || segment.text.trim().is_empty()
                    {
                        return Err(SessionError::Invalid("invalid backend source range".into()));
                    }
                    if segment.start > 0 {
                        send_skipped(&tx, source_text, at, at + segment.start).await?;
                        input.generated.set(at + segment.start);
                    }
                    segment.start += at;
                    segment.end += at;
                    let range = TextRange {
                        start: segment.start,
                        end: segment.end,
                    };
                    let paragraph_end =
                        backend.paragraph_end(&segment.text, &source_text[segment.end..]);
                    let mut silence = crate::audio::BoundarySilence::default();
                    let mut audio = if input.verification.policy
                        == crate::verification::VerificationPolicy::Off
                    {
                        backend
                            .stream_with_style(&segment.text, span.voice(), span.style())
                            .await?
                    } else {
                        verified_audio(backend, input, &segment, &span, &tx).await?
                    };
                    tx.send(Ok(Item::Start(range)))
                        .await
                        .map_err(|_| SessionError::Disconnected)?;
                    let mut ended = false;
                    let mut generated_seconds = 0.0;
                    while let Some(chunk) = audio.recv().await {
                        match chunk? {
                            crate::backend::AudioChunk::Pcm(pcm) => {
                                let (duration, _) = Budget::validate(&pcm)?;
                                generated_seconds += duration as f64 / 1000.0;
                                let Some(pcm) = silence.push(pcm)? else {
                                    continue;
                                };
                                let packet = budget.acquire(pcm).await?;
                                tx.send(Ok(Item::Audio(packet)))
                                    .await
                                    .map_err(|_| SessionError::Disconnected)?;
                            }
                            crate::backend::AudioChunk::End => {
                                if let Some(pcm) = silence.finish(paragraph_end)? {
                                    let packet = budget.acquire(pcm).await?;
                                    tx.send(Ok(Item::Audio(packet)))
                                        .await
                                        .map_err(|_| SessionError::Disconnected)?;
                                }
                                ended = true;
                                break;
                            }
                        }
                    }
                    if !ended {
                        return Err(SessionError::Invalid(
                            "synthesis stream ended without completion".into(),
                        ));
                    }
                    backend.observe_duration(&segment.text, span.voice(), generated_seconds);
                    tx.send(Ok(Item::End(range)))
                        .await
                        .map_err(|_| SessionError::Disconnected)?;
                    input.generated.set(range.end);
                    at = range.end;
                }
                input.generated.set(assigned.end);
            }
            if input.plan.borrow().state() == PlanState::Failed {
                return Err(PlanError::NotOpen(PlanState::Failed).into());
            }
            Ok(())
        }
        .await;
        match result {
            Ok(()) => {
                let _ = tx.send(Ok(Item::Finished)).await;
            }
            Err(error) => {
                let _ = tx.send(Err(error)).await;
            }
        }
    }));
    (task, rx)
}

async fn verified_audio(
    backend: &Rc<dyn Backend>,
    input: &PlanInput,
    segment: &crate::text::TextSegment,
    span: &SpeechSpan,
    tx: &mpsc::Sender<Result<Item, SessionError>>,
) -> Result<crate::backend::AudioStream, SessionError> {
    use crate::verification::*;
    let verifier = input
        .verifier
        .as_ref()
        .ok_or(VerificationError::NotPrepared)?;
    let caps = backend.capabilities();
    let source = input.plan.borrow().source().clone();
    let options = &input.verification;
    let mut attempt = 0;
    let pcm = loop {
        let stream = backend
            .stream_with_style(&segment.text, span.voice(), span.style())
            .await?;
        let pcm = collect_segment(stream, options).await?;
        let report = verifier
            .report(
                ReadbackRequest {
                    source: &source,
                    range: TextRange {
                        start: segment.start,
                        end: segment.end,
                    },
                    spoken_text: &segment.text,
                    backend: &caps.backend,
                    model: caps.model.as_deref(),
                    voice: span.voice(),
                    style: span.style(),
                    attempt,
                },
                &pcm,
                options,
            )
            .await?;
        let verdict = report.verdict;
        tx.send(Ok(Item::Verification(report)))
            .await
            .map_err(|_| SessionError::Disconnected)?;
        match options.policy {
            VerificationPolicy::ReportOnly => break pcm,
            VerificationPolicy::Gate {
                max_retries,
                strict_suspect,
            } => match verdict {
                VerificationVerdict::Passed => break pcm,
                VerificationVerdict::Suspect if !strict_suspect => break pcm,
                VerificationVerdict::ConfirmedError if attempt < max_retries => attempt += 1,
                _ => return Err(VerificationError::Rejected { verdict, attempt }.into()),
            },
            VerificationPolicy::Off => break pcm,
        }
    };
    // The bounded segment is already complete; no detached forwarding task is needed.
    let (sender, receiver) = mpsc::channel(2);
    sender
        .try_send(Ok(crate::backend::AudioChunk::Pcm(pcm)))
        .map_err(|_| SessionError::Disconnected)?;
    sender
        .try_send(Ok(crate::backend::AudioChunk::End))
        .map_err(|_| SessionError::Disconnected)?;
    Ok(receiver)
}

async fn collect_segment(
    mut stream: crate::backend::AudioStream,
    options: &crate::verification::VerificationOptions,
) -> Result<Pcm, SessionError> {
    use crate::verification::VerificationError;
    let mut audio: Option<Pcm> = None;
    let mut ended = false;
    while let Some(chunk) = stream.recv().await {
        match chunk? {
            crate::backend::AudioChunk::End => {
                ended = true;
                break;
            }
            crate::backend::AudioChunk::Pcm(pcm) => {
                pcm.duration_ms()?;
                let current = audio.get_or_insert_with(|| Pcm {
                    samples: Vec::new(),
                    sample_rate: pcm.sample_rate,
                    channels: pcm.channels,
                });
                if (current.sample_rate, current.channels) != (pcm.sample_rate, pcm.channels) {
                    return Err(SessionError::Invalid(
                        "PCM format changed inside segment".into(),
                    ));
                }
                let samples = current
                    .samples
                    .len()
                    .checked_add(pcm.samples.len())
                    .ok_or(VerificationError::Capacity)?;
                if samples > options.max_segment_bytes / 4
                    || (samples as u64 / pcm.channels as u64 * 1000)
                        .div_ceil(pcm.sample_rate as u64)
                        > options.max_segment_ms as u64
                {
                    return Err(VerificationError::Capacity.into());
                }
                current.samples.extend(pcm.samples);
            }
        }
    }
    if !ended {
        return Err(SessionError::Invalid(
            "synthesis stream ended without completion".into(),
        ));
    }
    audio.ok_or_else(|| SessionError::Invalid("empty synthesis segment".into()))
}

// A backend may omit only known non-spoken layout, never arbitrary prose.
async fn send_skipped(
    tx: &mpsc::Sender<Result<Item, SessionError>>,
    source: &str,
    start: usize,
    end: usize,
) -> Result<(), SessionError> {
    if !source[start..end]
        .lines()
        .all(|line| line.trim().is_empty() || crate::text::is_decoration_line(line))
    {
        return Err(SessionError::Invalid(
            "backend omitted spoken source text".into(),
        ));
    }
    tx.send(Ok(Item::Skipped(TextRange { start, end })))
        .await
        .map_err(|_| SessionError::Disconnected)
}
