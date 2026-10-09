//! Player-free, checkpoint-free consumption of the shared serial producer.
use super::{producer::Item, *};
use std::{marker::PhantomData, sync::atomic::AtomicBool};
use tokio::sync::watch;

/// Generation state only: no audio device or playback completion is implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthesisState {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// An owned PCM block and source range. Its budget is released on drop.
/// Retaining blocks applies backpressure; copying PCM is the consumer's responsibility.
#[derive(Debug)]
pub struct SpeechAudio {
    range: TextRange,
    packet: Packet,
}
impl SpeechAudio {
    /// Immutable interleaved PCM. No inference or audio-device types escape.
    pub fn pcm(&self) -> &Pcm {
        &self.packet.audio
    }
    /// Original UTF-8 source range represented by this block.
    pub fn range(&self) -> TextRange {
        self.range
    }
}

/// Ordered synthesis output: reports precede the PCM they authorize.
#[derive(Debug)]
pub enum SynthesisItem {
    Audio(SpeechAudio),
    Verification(crate::verification::VerificationReport),
}

/// Send + Sync cancellation request; it does not move the local stream or backend.
#[derive(Clone)]
pub struct CancellationHandle {
    abort: tokio::task::AbortHandle,
    requested: Arc<AtomicBool>,
    completed: watch::Receiver<bool>,
    work: Arc<crate::verification::RequestCompletions>,
}
impl CancellationHandle {
    /// Request cancellation from any thread. Await stream.cancel() for cleanup.
    pub fn cancel(&self) {
        self.requested.store(true, Ordering::SeqCst);
        self.abort.abort();
    }
    /// Whether the producer has released its receiver and local resources.
    pub fn is_finished(&self) -> bool {
        *self.completed.borrow() && self.work.is_finished()
    }
    /// Wait for local producer teardown; the owning LocalSet must keep running.
    pub async fn closed(&self) {
        let mut completed = self.completed.clone();
        let _ = completed.wait_for(|done| *done).await;
        self.work.settled().await;
    }
}

/// Local bounded synthesis stream using the same segmentation and PCM rules as listening.
/// Consume to Completed, or explicitly cancel and await cleanup. Drop only requests it.
pub struct SynthesisStream {
    receiver: mpsc::Receiver<Result<Item, SessionError>>,
    producer: Option<AbortOnDrop>,
    cancel: CancellationHandle,
    state: SynthesisState,
    range: Option<TextRange>,
    report: Option<crate::verification::VerificationReport>,
    _local: PhantomData<Rc<()>>,
}
impl SynthesisStream {
    /// Validate text/voice/style and spawn on the caller's current LocalSet.
    /// Does not create a runtime, player, config or checkpoint file.
    pub fn start(
        backend: Rc<dyn Backend>,
        text: impl Into<Arc<str>>,
        voice: &str,
        style: Option<String>,
    ) -> Result<Self, SessionError> {
        Self::start_verified(
            backend,
            text,
            voice,
            style,
            None,
            crate::verification::VerificationOptions::default(),
        )
    }
    /// Optional readback before any segment PCM is delivered.
    pub fn start_verified(
        backend: Rc<dyn Backend>,
        text: impl Into<Arc<str>>,
        voice: &str,
        style: Option<String>,
        verifier: Option<Rc<crate::verification::Verifier>>,
        verification: crate::verification::VerificationOptions,
    ) -> Result<Self, SessionError> {
        crate::verification::validate_options(&verification)?;
        if verification.policy != crate::verification::VerificationPolicy::Off && verifier.is_none()
        {
            return Err(crate::verification::VerificationError::NotPrepared.into());
        }
        let verifier = if verification.policy == crate::verification::VerificationPolicy::Off {
            None
        } else {
            verifier.map(|verifier| verifier.scoped())
        };
        let work = verifier
            .as_ref()
            .map_or_else(Arc::default, |verifier| verifier.work());
        let text = text.into();
        let caps = backend.capabilities();
        let voices = VoiceSnapshot::new(
            &caps.backend,
            caps.model.as_deref(),
            &caps,
            vec![voice.into()],
        )?;
        let source = SourceSnapshot::new(
            tts_protocol::SourceId {
                namespace: "synthesis".into(),
                book: String::new(),
                chapter: String::new(),
            },
            text.clone(),
            &tts_protocol::text_hash(&text),
        )?;
        let plan =
            SpeechPlan::single_voice(source, voices, PlaybackPolicy::Streaming, voice, style)?;
        let mut input = PlanInput::new(plan, 0);
        input.verifier = verifier;
        input.verification = verification;
        let input = Rc::new(input);
        let (done, completed) = watch::channel(false);
        let (producer, receiver) =
            super::producer::spawn(backend, text, input, 0, None, Some(done));
        let cancel = CancellationHandle {
            abort: producer.0.abort_handle(),
            requested: Arc::new(AtomicBool::new(false)),
            completed,
            work,
        };
        Ok(Self {
            receiver,
            producer: Some(producer),
            cancel,
            state: SynthesisState::Running,
            range: None,
            report: None,
            _local: PhantomData,
        })
    }
    /// Most recent report, including rejected attempts; bounded to one report.
    pub fn last_report(&self) -> Option<&crate::verification::VerificationReport> {
        self.report.as_ref()
    }
    /// Cancellation requests can cross threads; this stream remains local.
    pub fn cancellation(&self) -> CancellationHandle {
        self.cancel.clone()
    }
    /// Completion means the shared generator ended validly, never that PCM was played.
    pub fn state(&self) -> SynthesisState {
        self.state
    }
    /// Receive bounded PCM. None follows an explicit valid Finished or cancellation.
    /// Errors are returned once, then state() remains Failed. Partial PCM is possible
    /// before a later synthesis failure; consumers must check the final state.
    pub async fn recv(&mut self) -> Option<Result<SpeechAudio, SessionError>> {
        while let Some(item) = self.next().await {
            match item {
                Ok(SynthesisItem::Audio(audio)) => return Some(Ok(audio)),
                Ok(SynthesisItem::Verification(_)) => {}
                Err(error) => return Some(Err(error)),
            }
        }
        None
    }
    /// Consume every attempt report, including failed/replaced audio, in bounded order.
    pub async fn next(&mut self) -> Option<Result<SynthesisItem, SessionError>> {
        if self.state != SynthesisState::Running {
            return None;
        }
        loop {
            if self.cancel.requested.load(Ordering::SeqCst) {
                self.cancel().await;
                return None;
            }
            let item = self.receiver.recv().await;
            if self.cancel.requested.load(Ordering::SeqCst) {
                self.cancel().await;
                return None;
            }
            match item {
                Some(Ok(Item::Verification(report))) => {
                    self.report = Some(report.clone());
                    return Some(Ok(SynthesisItem::Verification(report)));
                }
                Some(Ok(Item::Start(range))) => self.range = Some(range),
                Some(Ok(Item::Audio(packet))) => {
                    if let Some(range) = self.range {
                        return Some(Ok(SynthesisItem::Audio(SpeechAudio { range, packet })));
                    }
                    return self
                        .failed(SessionError::Invalid("audio without source range".into()))
                        .await;
                }
                Some(Ok(Item::End(_))) => self.range = None,
                Some(Ok(Item::Skipped(_))) => {}
                Some(Ok(Item::Finished)) => {
                    self.join().await;
                    self.state = SynthesisState::Completed;
                    return None;
                }
                Some(Err(error)) => return self.failed(error).await,
                None => {
                    return self
                        .failed(SessionError::Invalid(
                            "synthesis disconnected before completion".into(),
                        ))
                        .await;
                }
            }
        }
    }
    async fn failed(&mut self, error: SessionError) -> Option<Result<SynthesisItem, SessionError>> {
        self.cancel().await;
        self.state = SynthesisState::Failed;
        Some(Err(error))
    }
    async fn join(&mut self) {
        if let Some(mut producer) = self.producer.take() {
            let _ = (&mut producer.0).await;
        }
    }
    /// Release queued PCM, cancel the producer and await local teardown.
    pub async fn cancel(&mut self) {
        self.receiver.close();
        self.cancel.cancel();
        self.join().await;
        self.cancel.closed().await;
        while self.receiver.try_recv().is_ok() {}
        if self.state == SynthesisState::Running {
            self.state = SynthesisState::Cancelled;
        }
    }
}
impl Drop for SynthesisStream {
    fn drop(&mut self) {
        self.receiver.close();
        self.cancel.cancel();
    }
}
