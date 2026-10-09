use super::producer::Item;
use super::*;
pub(super) struct Runner {
    pub(super) id: String,
    pub(super) source: SourceSnapshot,
    pub(super) backend: Rc<dyn Backend>,
    pub(super) player: Rc<dyn Playback>,
    pub(super) checkpoints: Option<CheckpointStore>,
    pub(super) events: mpsc::Sender<SessionEvent>,
    pub(super) paused: Rc<Cell<bool>>,
    pub(super) phase: Rc<Cell<SessionState>>,
    pub(super) buffering: Rc<Cell<bool>>,
    pub(super) speed: Rc<Cell<f32>>,
    pub(super) position: Rc<Cell<usize>>,
    pub(super) text: Arc<str>,
    pub(super) writes: Arc<PendingWrites>,
    pub(super) staging: StagingOptions,
}

impl Runner {
    pub(super) async fn emit(&self, event: Event) -> Result<(), SessionError> {
        self.events
            .send(SessionEvent {
                session_id: self.id.clone(),
                event,
            })
            .await
            .map_err(|_| SessionError::Disconnected)
    }

    async fn save(&self, byte: usize, completed: bool) -> Result<(), SessionError> {
        let Some(checkpoints) = self.checkpoints.clone() else {
            return Ok(());
        };
        let source = self.source.source().clone();
        let text = self.text.clone();
        self.writes.count.fetch_add(1, Ordering::SeqCst);
        let pending = WriteGuard(self.writes.clone());
        tokio::task::spawn_blocking(move || {
            let _pending = pending;
            checkpoints.save(&source, &text, byte, completed)
        })
        .await
        .map_err(|error| SessionError::Invalid(error.to_string()))??;
        Ok(())
    }

    pub(super) async fn run(&self, byte: usize, input: Rc<PlanInput>) -> Result<(), SessionError> {
        self.emit(Event::SessionState {
            state: if self.paused.get() {
                SessionState::Paused
            } else {
                self.phase.get()
            },
        })
        .await?;
        self.save(byte, false).await?;
        let (mut producer, mut rx) = super::producer::spawn(
            self.backend.clone(),
            self.text.clone(),
            input.clone(),
            byte,
            Some(self.writes.clone()),
            None,
        );
        let mut staged = None;
        let after_chapter =
            input.plan.borrow().playback_policy() == PlaybackPolicy::AfterChapterReady;
        if after_chapter {
            let storage = super::staging::prepare(
                &mut rx,
                &self.staging,
                self.writes.clone(),
                &self.text,
                byte,
                Some((&self.id, &self.events)),
            )
            .await?;
            input.ready.set(true);
            let (reader, replay) = super::staging::replay(storage.clone(), self.writes.clone());
            producer = reader;
            rx = replay;
            staged = Some(storage);
        }
        let _producer = producer;
        // Keep the private directory alive until playback finishes or is cancelled.
        let _staged = staged;
        struct Marker {
            range: TextRange,
            start: Duration,
            end: Option<Duration>,
            started: bool,
            had_audio: bool,
            format: Option<(u32, u16)>,
        }
        const MAX_MARKERS: usize = 64;
        let mut completed = false;
        let mut buffering = super::buffering::Buffering::default();
        let mut last_report = tokio::time::Instant::now() - Duration::from_secs(1);
        let mut markers = std::collections::VecDeque::<Marker>::new();
        let mut pending = std::collections::VecDeque::new();
        let mut queued = Duration::ZERO;
        loop {
            let cursor = if self.player.is_empty() && !buffering.waiting {
                queued
            } else {
                self.player.position()
            };
            let buffered = queued.saturating_sub(cursor);
            let waiting = buffering.waiting;
            buffering.update(
                self.player.is_empty(),
                buffered,
                self.speed.get(),
                // Drain a finished input prefix while waiting for analysis, even
                // if it is shorter than the recovery buffer. This is not EOF.
                completed
                    || ((input.waiting.get() || markers.len() >= MAX_MARKERS)
                        && !buffered.is_zero()),
            );
            self.buffering.set(buffering.waiting);
            if waiting != buffering.waiting {
                if buffering.waiting {
                    self.player.pause();
                } else if !self.paused.get() {
                    self.player.resume();
                }
            }
            if waiting != buffering.waiting || last_report.elapsed() >= Duration::from_secs(1) {
                self.emit(Event::BufferStatus {
                    buffered_ms: buffered.as_millis() as u64,
                    target_ms: buffering.target(self.speed.get()).as_millis() as u64,
                    underruns: buffering.underruns,
                })
                .await?;
                last_report = tokio::time::Instant::now();
            }
            if !self.paused.get() {
                while pending.front().is_some_and(|(end, _)| *end <= cursor) {
                    pending.pop_front();
                }
                while let Some(marker) = markers.front_mut() {
                    if !marker.started
                        && (buffering.waiting || !marker.had_audio || marker.start > cursor)
                    {
                        break;
                    }
                    if marker.had_audio && !marker.started && marker.start <= cursor {
                        marker.started = true;
                        self.position.set(marker.range.start);
                        self.save(marker.range.start, false).await?;
                        self.phase.set(SessionState::Playing);
                        self.emit(Event::SegmentStarted {
                            range: marker.range,
                            text_hash: self.source.hash().into(),
                        })
                        .await?;
                        self.emit(Event::SessionState {
                            state: SessionState::Playing,
                        })
                        .await?;
                    }
                    if !marker.end.is_some_and(|end| end <= cursor) {
                        break;
                    }
                    let marker = markers.pop_front().expect("front exists");
                    self.position.set(marker.range.end);
                    self.save(marker.range.end, false).await?;
                    if marker.had_audio {
                        self.emit(Event::SegmentFinished {
                            range: marker.range,
                            text_hash: self.source.hash().into(),
                        })
                        .await?;
                    }
                }
            }
            if !self.paused.get() {
                let phase = if buffering.waiting {
                    SessionState::Buffering
                } else {
                    SessionState::Playing
                };
                if phase != self.phase.get() {
                    self.phase.set(phase);
                    self.emit(Event::SessionState { state: phase }).await?;
                }
            }
            if completed && markers.is_empty() {
                break;
            }
            let item = tokio::select! {
                item = rx.recv(), if !completed && markers.len() < MAX_MARKERS => item,
                _ = tokio::time::sleep(Duration::from_millis(10)) => continue,
            };
            let Some(item) = item else {
                return Err(SessionError::Invalid(
                    "audio producer disconnected before completion".into(),
                ));
            };
            match item? {
                Item::Verification(report) => self.emit(Event::Verification(report)).await?,
                Item::Finished => completed = true,
                Item::Skipped(range) => {
                    // Layout advances only after all preceding queued audio is consumed.
                    markers.push_back(Marker {
                        range,
                        start: queued,
                        end: Some(queued),
                        started: true,
                        had_audio: false,
                        format: None,
                    });
                }
                Item::Start(range) => {
                    if !range.is_valid(self.source.text()) || range.start >= range.end {
                        return Err(SessionError::Invalid("invalid source range".into()));
                    }
                    markers.push_back(Marker {
                        range,
                        start: queued,
                        end: None,
                        started: false,
                        had_audio: false,
                        format: None,
                    });
                }
                Item::Audio(packet) => {
                    let marker = markers.back_mut().ok_or_else(|| {
                        SessionError::Invalid("audio without source marker".into())
                    })?;
                    let format = (packet.audio.sample_rate, packet.audio.channels);
                    if marker.format.is_some_and(|old| old != format) {
                        return Err(SessionError::Invalid(
                            "PCM format changed inside segment".into(),
                        ));
                    }
                    marker.format = Some(format);
                    queued += Duration::from_secs_f64(
                        packet.audio.samples.len() as f64
                            / packet.audio.channels as f64
                            / packet.audio.sample_rate as f64,
                    );
                    let pcm = Arc::new(packet.audio);
                    self.player.append(pcm);
                    pending.push_back((queued, packet.lease));
                    marker.had_audio = true;
                }
                Item::End(range) => {
                    let marker = markers.back_mut().ok_or_else(|| {
                        SessionError::Invalid("completion without source marker".into())
                    })?;
                    if marker.range != range || !marker.had_audio {
                        return Err(SessionError::Invalid(
                            "backend produced no audio or invalid completion".into(),
                        ));
                    }
                    marker.end = Some(queued);
                }
            }
        }
        self.save(self.source.text().len(), true).await?;
        Ok(())
    }
}
