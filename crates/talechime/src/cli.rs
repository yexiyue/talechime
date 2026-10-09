use crate::resources::Resources;
use crossterm::{
    event::{Event as TerminalEvent, EventStream, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use std::{future::Future, io::IsTerminal, path::PathBuf, pin::Pin};
use tokio::sync::mpsc;
use tts_core::{checkpoint::CheckpointStore, config::ConfigStore, session::SessionEvent};
use tts_protocol::{EndReason, Event, SourceId, text_hash};

/// Keep acknowledgements in the event loop so reliable events cannot block them.
struct PendingControl {
    acknowledgement: Pin<Box<dyn Future<Output = Result<(), talechime::EngineError>>>>,
    stop: bool,
}
impl PendingControl {
    fn new(
        acknowledgement: impl Future<Output = Result<(), talechime::EngineError>> + 'static,
        stop: bool,
    ) -> Self {
        Self {
            acknowledgement: Box::pin(acknowledgement),
            stop,
        }
    }
    async fn wait(&mut self) -> Result<(), talechime::EngineError> {
        self.acknowledgement.as_mut().await
    }
}

struct TerminalGuard;
impl TerminalGuard {
    fn enter() -> std::io::Result<Self> {
        enable_raw_mode()?;
        Ok(Self)
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

pub async fn run(
    file: PathBuf,
    store: ConfigStore,
    checkpoints: CheckpointStore,
    resources: Resources,
    restore_checkpoint: bool,
    plan_file: Option<PathBuf>,
    after_chapter: bool,
) -> anyhow::Result<()> {
    // Read and validate before model download or opening the output device.
    let text = tokio::fs::read_to_string(&file)
        .await
        .map_err(|error| anyhow::anyhow!("cannot read UTF-8 file {}: {error}", file.display()))?;
    let canonical = tokio::fs::canonicalize(&file).await?;
    let config = store.load()?;
    let config =
        store.initialize(&resources.capabilities_for(&config.backend, config.model.as_deref())?)?;
    let source = SourceId {
        namespace: "cli-file".into(),
        book: canonical.to_string_lossy().into(),
        chapter: "file".into(),
    };
    let (plan, resume_byte, restore_checkpoint) = if let Some(path) = plan_file {
        // Bound file reads before deserialization. Full plans only; stdin worker handles increments.
        use tokio::io::AsyncReadExt;
        let mut data = Vec::new();
        tokio::fs::File::open(&path)
            .await?
            .take((tts_protocol::MAX_MESSAGE_BYTES + 1) as u64)
            .read_to_end(&mut data)
            .await?;
        if data.len() > tts_protocol::MAX_MESSAGE_BYTES {
            anyhow::bail!("plan file exceeds transport limit");
        }
        let input: tts_protocol::PlanRequest = serde_json::from_slice(&data)?;
        if input.text != text || !input.sealed {
            anyhow::bail!("CLI requires a sealed plan with the exact file text");
        }
        (
            crate::plan_input::build(
                &input,
                &resources.capabilities_for(&config.backend, config.model.as_deref())?,
            )?,
            if restore_checkpoint {
                input.resume_byte
            } else {
                Some(0)
            },
            restore_checkpoint && input.restore_checkpoint,
        )
    } else {
        let caps = resources.capabilities_for(&config.backend, config.model.as_deref())?;
        let source = talechime::SourceSnapshot::new(source, text.clone(), &text_hash(&text))?;
        let voices = talechime::VoiceSnapshot::new(
            &caps.backend,
            caps.model.as_deref(),
            &caps,
            vec![config.voice.clone()],
        )?;
        (
            talechime::SpeechPlan::single_voice(
                source,
                voices,
                if after_chapter {
                    talechime::PlaybackPolicy::AfterChapterReady
                } else {
                    talechime::PlaybackPolicy::Streaming
                },
                &config.voice,
                config.style.clone(),
            )?,
            if restore_checkpoint { None } else { Some(0) },
            restore_checkpoint,
        )
    };
    plan.validate_resume_byte(resume_byte.unwrap_or(0))?;
    let (progress, mut preparation) = mpsc::channel(16);
    let settings = config.clone();
    let mut task = tokio::task::spawn_local(async move {
        crate::preparation::prepare(resources, settings, progress).await
    });
    let prepared = loop {
        tokio::select! {
            result = &mut task => break result??,
            event = preparation.recv() => if let Some(event) = event { match event {
                Event::ModelProgress {resource,downloaded,total} => eprintln!("{resource}: {:.1}% · {:.1}/{:.1} MiB", if total==0 {0.0} else {downloaded as f64/total as f64*100.0},downloaded as f64/1048576.0,total as f64/1048576.0),
                Event::ResourceState {stage,resource}=>eprintln!("{stage} {resource}"),
                Event::DeviceStatus {component,selected,reason,..} => eprintln!("{component}: {selected:?} {}",reason.unwrap_or_default()),
                _=>{}
            } },
            signal = tokio::signal::ctrl_c() => { signal?; task.abort(); let _ = task.await; return Ok(()); }
        }
    };
    let (tx, mut events) = mpsc::channel::<SessionEvent>(64);
    let mut owner = crate::app_session::AppSession::open(prepared, &checkpoints, tx)?;
    let manager = owner.control.clone();
    let interactive = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
    let _terminal = if interactive {
        Some(TerminalGuard::enter()?)
    } else {
        None
    };
    let mut keys = if interactive {
        Some(EventStream::new())
    } else {
        None
    };
    let mut paused = false;
    let mut pending: Option<PendingControl> = None;
    let id = "cli-file";
    let started = manager
        .start(
            id,
            plan,
            talechime::PlanSessionOptions {
                volume: config.volume,
                speed: config.speed,
                resume_byte,
                restore_checkpoint,
                ..Default::default()
            },
        )
        .await;
    if let Err(error) = started {
        owner.close().await?;
        return Err(error.into());
    }
    if interactive {
        eprintln!("space: pause/resume; s: stop; q: exit\r");
    }
    let result: anyhow::Result<()> = async {
        loop {
            tokio::select! {
                event = events.recv() => {
                    let Some(event) = event else {break};
                    match event.event {
                        Event::BufferStatus{buffered_ms,target_ms,underruns} => eprintln!("缓冲 {:.1}/{:.1}s · 耗尽 {underruns} 次\r",buffered_ms as f64/1000.0,target_ms as f64/1000.0),
                        Event::SessionState{state} => eprintln!("{state:?}\r"),
                        Event::Error(error) => eprintln!("{}: {}\r",error.stage,error.message),
                        Event::SegmentStarted{range,..} => eprintln!("bytes {}..{}\r",range.start,range.end),
                        Event::SessionEnded{reason,..} => {
                            if reason == EndReason::Failed {return Err(anyhow::anyhow!("listening failed; run again to retry the unfinished segment"));}
                            break;
                        },
                        _ => {}
                    }
                }
                key = async {keys.as_mut().expect("guarded").next().await}, if interactive => {
                    if let Some(Ok(TerminalEvent::Key(key))) = key && key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => break,
                            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                            KeyCode::Char('s') if pending.is_none() => {
                                let manager = manager.clone();
                                pending = Some(PendingControl::new(async move { manager.stop().await }, true));
                            },
                            KeyCode::Char(' ') if pending.is_none() => {
                                let manager = manager.clone();
                                pending = Some(PendingControl::new(async move {
                                    if paused { manager.resume(id).await } else { manager.pause(id).await }
                                }, false));
                            }
                            _ => {}
                        }
                    }
                }
                acknowledgement = async { pending.as_mut().expect("guarded").wait().await }, if pending.is_some() => {
                    acknowledgement?;
                    if pending.take().expect("completed operation").stop { break; }
                    paused = !paused;
                }
                signal = tokio::signal::ctrl_c() => {signal?; break;}
            }
        }
        Ok(())
    }.await;
    owner.close().await?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn control_acknowledgement_allows_draining_a_full_event_queue() {
        let (sender, mut events) = mpsc::channel(1);
        sender.send(1).await.unwrap();
        let mut pending = PendingControl::new(
            async move {
                sender.send(2).await.unwrap();
                Ok(())
            },
            true,
        );
        let mut received = Vec::new();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                tokio::select! {
                    acknowledgement = pending.wait() => { acknowledgement.unwrap(); break; }
                    event = events.recv() => if let Some(event) = event { received.push(event); }
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(received.first(), Some(&1));
        assert!(pending.stop);
    }
}
