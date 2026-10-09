#![allow(dead_code)] // Each standalone example uses a different subset of the fixtures.
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
    time::Duration,
};
use talechime::*;
use tokio::sync::mpsc;

pub struct Fixture {
    pub calls: RefCell<Vec<(String, String, Option<String>)>>,
}
impl Default for Fixture {
    fn default() -> Self {
        Self {
            calls: RefCell::new(vec![]),
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
            let (tx, rx) = mpsc::channel(1);
            tokio::task::spawn_local(async move {
                for _ in 0..2 {
                    let pcm = Pcm {
                        samples: vec![0.1; 400],
                        sample_rate: 1000,
                        channels: 1,
                    };
                    if tx.send(Ok(AudioChunk::Pcm(pcm))).await.is_err() {
                        return;
                    }
                }
                let _ = tx.send(Ok(AudioChunk::End)).await;
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
    engine
        .plan(
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
            vec!["A".into(), "B".into()],
            policy,
        )
        .unwrap()
}
