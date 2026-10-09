use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::Arc,
    time::Duration,
};
use talechime::*;
use tokio::sync::mpsc;

#[derive(Clone, Copy)]
pub enum Mode {
    Normal,
    Long,
    Disconnect,
    Silent,
    Format,
    SegmentFormat,
    Oversized,
}
pub struct Fixture {
    pub calls: RefCell<Vec<(String, String, Option<String>)>>,
    pub chunks: Rc<Cell<usize>>,
    pub mode: Cell<Mode>,
}
impl Default for Fixture {
    fn default() -> Self {
        Self {
            calls: RefCell::new(vec![]),
            chunks: Rc::new(Cell::new(0)),
            mode: Cell::new(Mode::Normal),
        }
    }
}
impl Backend for Fixture {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            backend: "fixture".into(),
            model: Some("shared".into()),
            model_name: "No weights".into(),
            compiled_devices: vec![Device::Cpu],
            default_voice: "A".into(),
            voices: vec!["A".into(), "B".into()],
            voice_names: Default::default(),
            native_streaming: true,
            style: true,
            cloning: false,
            pronunciation: false,
        }
    }
    fn stream<'a>(&'a self, text: &'a str, voice: &'a str) -> Streaming<'a> {
        self.stream_with_style(text, voice, None)
    }
    fn stream_with_style<'a>(
        &'a self,
        text: &'a str,
        voice: &'a str,
        style: Option<&'a str>,
    ) -> Streaming<'a> {
        Box::pin(async move {
            self.calls
                .borrow_mut()
                .push((text.into(), voice.into(), style.map(str::to_owned)));
            let mode = self.mode.get();
            let segment_index = self.calls.borrow().len();
            let chunks = self.chunks.clone();
            let (tx, rx) = mpsc::channel(1);
            tokio::task::spawn_local(async move {
                let count = if matches!(mode, Mode::Long) { 100 } else { 2 };
                for i in 0..count {
                    let pcm = Pcm {
                        samples: vec![
                            if matches!(mode, Mode::Silent) {
                                0.0
                            } else {
                                0.1
                            };
                            if matches!(mode, Mode::Oversized) {
                                31_000
                            } else {
                                400
                            }
                        ],
                        sample_rate: if (i == 1 && matches!(mode, Mode::Format))
                            || (segment_index > 1 && matches!(mode, Mode::SegmentFormat))
                        {
                            2000
                        } else {
                            1000
                        },
                        channels: 1,
                    };
                    if tx.send(Ok(AudioChunk::Pcm(pcm))).await.is_err() {
                        return;
                    }
                    chunks.set(chunks.get() + 1);
                }
                if !matches!(mode, Mode::Disconnect) {
                    let _ = tx.send(Ok(AudioChunk::End)).await;
                }
            });
            Ok(rx)
        })
    }
}
#[derive(Default)]
pub struct Player {
    pub queued: Cell<Duration>,
    cursor: Cell<Duration>,
    pub paused: Cell<bool>,
}
impl Playback for Player {
    fn append(&self, pcm: Arc<Pcm>) {
        self.queued.set(
            self.queued.get()
                + Duration::from_secs_f64(
                    pcm.samples.len() as f64 / pcm.channels as f64 / pcm.sample_rate as f64,
                ),
        );
    }
    fn position(&self) -> Duration {
        if !self.paused.get() {
            self.cursor.set(self.queued.get());
        }
        self.cursor.get()
    }
    fn is_empty(&self) -> bool {
        self.position() >= self.queued.get()
    }
    fn pause(&self) {
        self.paused.set(true);
    }
    fn resume(&self) {
        self.paused.set(false);
    }
    fn stop(&self) {
        self.pause();
        self.queued.set(Duration::ZERO);
        self.cursor.set(Duration::ZERO);
    }
    fn configure(&self, _: f32, _: f32) {}
}
pub fn plan(engine: &Engine, text: &str, policy: PlaybackPolicy) -> SpeechPlan {
    let caps = engine.capabilities().unwrap();
    SpeechPlan::new(
        SourceSnapshot::new(
            SourceId {
                namespace: "fixture".into(),
                book: "book".into(),
                chapter: "one".into(),
            },
            text,
            &text_hash(text),
        )
        .unwrap(),
        VoiceSnapshot::new(
            &caps.backend,
            caps.model.as_deref(),
            &caps,
            caps.voices.clone(),
        )
        .unwrap(),
        policy,
    )
}
