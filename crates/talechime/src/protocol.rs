use crate::{resources::Resources, runtime::Worker};
use std::{
    io::{BufReader, BufWriter, Write},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, oneshot};
use tts_core::{checkpoint::CheckpointStore, config::ConfigStore, session::SessionEvent};
use tts_protocol::{
    Command, ErrorInfo, Event, Message, PROTOCOL_VERSION, Request, RequestTracker, decode, encode,
    read_frame,
};

struct Output {
    instance: String,
    sequence: u64,
    messages: mpsc::Sender<Message>,
}
impl Output {
    // Commands can emit lifecycle events themselves. Keep draining their bounded
    // queue while awaiting a command, without accepting another command.
    async fn while_draining<T>(
        &mut self,
        events: &mut mpsc::Receiver<SessionEvent>,
        operation: impl std::future::Future<Output = T>,
    ) -> anyhow::Result<T> {
        tokio::pin!(operation);
        let mut events_open = true;
        loop {
            tokio::select! {
                result = &mut operation => return Ok(result),
                event = events.recv(), if events_open => {
                    if let Some(event) = event {
                        self.send(None, Some(event.session_id), event.event).await?;
                    } else {
                        events_open = false;
                    }
                }
            }
        }
    }

    async fn send(
        &mut self,
        request: Option<&Request>,
        session: Option<String>,
        event: Event,
    ) -> anyhow::Result<()> {
        self.sequence += 1;
        let message = Message {
            protocol_version: PROTOCOL_VERSION,
            instance_id: self.instance.clone(),
            session_id: session.or_else(|| request.and_then(|r| r.session_id.clone())),
            sequence: self.sequence,
            request_id: request.map(|r| r.request_id.clone()),
            event,
        };
        tokio::time::timeout(Duration::from_secs(5), self.messages.send(message)).await??;
        Ok(())
    }
    async fn error(
        &mut self,
        request: Option<&Request>,
        code: &str,
        stage: &str,
        message: impl ToString,
    ) -> anyhow::Result<()> {
        self.send(
            request,
            None,
            Event::Error(ErrorInfo {
                code: code.into(),
                stage: stage.into(),
                message: message.to_string(),
                retryable: code != "incompatible_version",
            }),
        )
        .await
    }
}

pub async fn run(
    store: ConfigStore,
    checkpoints: CheckpointStore,
    resources: Resources,
) -> anyhow::Result<()> {
    let (input_tx, mut input) = mpsc::channel(8);
    // A dedicated stdin thread can end with the process even if a broken parent
    // never closes its pipe; it is not a Tokio blocking task keeping the runtime alive.
    std::thread::Builder::new()
        .name("tts-stdin".into())
        .spawn(move || {
            let mut reader = BufReader::new(std::io::stdin());
            loop {
                let frame = read_frame(&mut reader);
                let end = !matches!(&frame, Ok(Some(_)));
                if input_tx.blocking_send(frame).is_err() || end {
                    break;
                }
            }
        })?;
    let (messages, mut outgoing) = mpsc::channel::<Message>(64);
    let (finished, writer) = oneshot::channel();
    // A blocked parent stdout must not keep Tokio's blocking pool alive on exit.
    std::thread::Builder::new()
        .name("tts-stdout".into())
        .spawn(move || {
            let result = (|| {
                let mut stdout = BufWriter::new(std::io::stdout());
                while let Some(message) = outgoing.blocking_recv() {
                    stdout.write_all(&encode(&message)?)?;
                    stdout.flush()?;
                }
                Ok::<_, anyhow::Error>(())
            })();
            let _ = finished.send(result);
        })?;
    let result = serve(store, checkpoints, resources, &mut input, messages).await;
    let written = tokio::time::timeout(Duration::from_secs(3), writer).await;
    match written {
        Ok(joined) => joined??,
        Err(_) => return Err(anyhow::anyhow!("protocol stdout did not drain")),
    }
    result
}

async fn serve(
    store: ConfigStore,
    checkpoints: CheckpointStore,
    resources: Resources,
    input: &mut mpsc::Receiver<Result<Option<Vec<u8>>, tts_protocol::ProtocolError>>,
    messages: mpsc::Sender<Message>,
) -> anyhow::Result<()> {
    let (events, session_events) = mpsc::channel::<SessionEvent>(64);
    let (progress, model_events) = mpsc::channel::<Event>(16);
    let worker = Worker::new(store, checkpoints, resources, events, progress);
    serve_worker(worker, session_events, model_events, input, messages).await
}

async fn serve_worker(
    mut worker: Worker,
    mut session_events: mpsc::Receiver<SessionEvent>,
    mut model_events: mpsc::Receiver<Event>,
    input: &mut mpsc::Receiver<Result<Option<Vec<u8>>, tts_protocol::ProtocolError>>,
    messages: mpsc::Sender<Message>,
) -> anyhow::Result<()> {
    let mut output = Output {
        instance: format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ),
        sequence: 0,
        messages,
    };
    let mut tracker = RequestTracker::default();
    let mut handshake = false;
    let deadline = tokio::time::sleep(Duration::from_secs(5));
    tokio::pin!(deadline);
    let result: anyhow::Result<()> = async {
        loop {
            let preparing = worker.is_preparing();
            tokio::select! {
                _ = &mut deadline, if !handshake => {
                    output.error(None, "handshake_timeout", "protocol", "hello required within five seconds").await?;
                    break;
                }
                event = session_events.recv() => {
                    if let Some(event) = event {
                        output.send(None, Some(event.session_id), event.event).await?;
                    }
                }
                event = model_events.recv(), if handshake => {
                    if let Some(event) = event {
                        output.send(None, None, event).await?;
                    }
                }
                response = worker.prepared(), if preparing => {
                    output.send(None, response.session, response.event).await?;
                }
                frame = input.recv() => {
                    let bytes = match frame {
                        Some(Ok(Some(bytes))) => bytes,
                        Some(Err(error)) => {
                            output.error(None, "invalid_message", "protocol", error).await?;
                            break;
                        }
                        _ => break,
                    };
                    let request: Request = match decode(&bytes) {
                        Ok(request) => request,
                        Err(error) => {
                            output.error(None, "invalid_message", "protocol", error).await?;
                            continue;
                        }
                    };
                    if let Err(error) = tracker.accept(&request) {
                        let incompatible = request.protocol_version != PROTOCOL_VERSION;
                        let code = if incompatible { "incompatible_version" } else { "invalid_request" };
                        output.error(Some(&request), code, "protocol", error).await?;
                        if incompatible { break; }
                        continue;
                    }
                    if !handshake && !matches!(request.command, Command::Hello) {
                        output.error(Some(&request), "hello_required", "protocol", "first command must be hello").await?;
                        continue;
                    }
                    match &request.command {
                        Command::Hello => {
                            handshake = true;
                            output.send(Some(&request), None, Event::Ready(worker.catalog()?)).await?;
                            #[cfg(any(feature="moss",feature="qwen",feature="voxcpm",feature="omnivoice"))]
                            output.send(None,None,worker.unprepared_device_status()?).await?;
                        }
                        Command::Shutdown => {
                            output.send(Some(&request), None, Event::Accepted).await?;
                            break;
                        }
                        command => {
                            let response = output.while_draining(&mut session_events, worker.command(&request)).await?;
                            let preparation_changed = matches!(command, Command::UpdateConfig(patch)
                                if patch.backend.is_some() || patch.model.is_some()
                                    || patch.tts_device.is_some());
                            if matches!(command, Command::CancelPrepare) || preparation_changed {
                                while model_events.try_recv().is_ok() {}
                            }
                            let config_changed = matches!(&response.event, Event::ConfigChanged(_))
                                && preparation_changed;
                            output.send(Some(&request), response.session, response.event).await?;
                            #[cfg(any(feature="moss",feature="qwen",feature="voxcpm",feature="omnivoice"))]
                            if config_changed && !worker.is_preparing() && !worker.has_prepared_model() {
                                output.send(None,None,worker.unprepared_device_status()?).await?;
                            }
                            #[cfg(not(any(feature="moss",feature="qwen",feature="voxcpm",feature="omnivoice")))]
                            let _ = config_changed;
                        }
                    }
                }
            }
        }
        Ok(())
    }.await;
    output
        .while_draining(&mut session_events, worker.shutdown())
        .await??;
    while let Ok(event) = session_events.try_recv() {
        output
            .send(None, Some(event.session_id), event.event)
            .await?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tts_protocol::{EndReason, SessionState};

    #[tokio::test]
    async fn command_lifecycle_events_drain_a_full_queue_before_response() {
        let (events, mut pending) = mpsc::channel(1);
        events
            .send(SessionEvent {
                session_id: "active".into(),
                event: Event::SessionState {
                    state: SessionState::Playing,
                },
            })
            .await
            .unwrap();
        let (messages, mut received) = mpsc::channel(8);
        let mut output = Output {
            instance: "test".into(),
            sequence: 0,
            messages,
        };
        // Stop and replacement commands must await this bounded terminal send.
        let command = async {
            events
                .send(SessionEvent {
                    session_id: "active".into(),
                    event: Event::SessionEnded {
                        reason: EndReason::Cancelled,
                        text_hash: "snapshot".into(),
                    },
                })
                .await
                .unwrap();
            Event::Accepted
        };
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            output.while_draining(&mut pending, command),
        )
        .await
        .expect("command must finish even when its event queue starts full")
        .unwrap();
        output.send(None, None, response).await.unwrap();
        // The transport resumes its event loop after dispatching the response.
        while let Ok(event) = pending.try_recv() {
            output
                .send(None, Some(event.session_id), event.event)
                .await
                .unwrap();
        }
        let first = received.recv().await.unwrap();
        let second = received.recv().await.unwrap();
        let third = received.recv().await.unwrap();
        assert!(matches!(first.event, Event::SessionState { .. }));
        assert!(
            matches!(second.event, Event::SessionEnded { .. })
                || matches!(third.event, Event::SessionEnded { .. })
        );
        assert!(matches!(second.event, Event::Accepted) || matches!(third.event, Event::Accepted));
        assert_eq!([first.sequence, second.sequence, third.sequence], [1, 2, 3]);
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    use crate::fixture::{Fixture, Player};
    use std::rc::Rc;
    use tts_protocol::{
        EndReason, PlanPlayback, PlanRequest, SourceId, TextRange, VoiceSpan, text_hash,
    };

    #[tokio::test]
    async fn json_plan_reaches_completed_only_after_seal_and_real_fixture_playback() {
        talechime::run_local(async {
            let root=tempfile::tempdir().unwrap();
            let backend=Rc::new(Fixture::default());
            let (events, pending)=mpsc::channel(2);
            let (progress, updates)=mpsc::channel(2);
            let mut worker=Worker::new(ConfigStore::new(root.path().join("config.json")),CheckpointStore::new(root.path().join("checkpoints")),Resources::new(Some(root.path().join("models"))).unwrap(),events,progress);
            worker.attach_test(backend.clone(),Rc::new(Player::default()));
            let (input,mut frames)=mpsc::channel(8);
            let (outgoing,mut received)=mpsc::channel(1);
            let task=tokio::task::spawn_local(async move {serve_worker(worker,pending,updates,&mut frames,outgoing).await});
            let span=|start,end,voice:&str| VoiceSpan {range:TextRange {start,end},voice:voice.into(),style:None};
            let plan=PlanRequest {source:SourceId {namespace:"test".into(),book:"book".into(),chapter:"chapter".into()},text:"甲乙丙".into(),text_hash:text_hash("甲乙丙"),backend:"fixture".into(),model:Some("shared".into()),voices:vec!["A".into(),"B".into()],spans:vec![span(0,3,"A")],sealed:false,playback:PlanPlayback::Streaming,resume_byte:Some(0),restore_checkpoint:false};
            for (id,session,command) in [("hello",None,Command::Hello),("start",Some("one"),Command::Start(Box::new(plan))),("early",Some("one"),Command::Seal),("invalid",Some("one"),Command::Append {spans:vec![span(4,6,"B")]}),("progress",Some("one"),Command::GetProgress),("append",Some("one"),Command::Append {spans:vec![span(3,6,"B"),span(6,9,"A")]}),("seal",Some("one"),Command::Seal)] {
                input.send(Ok(Some(encode(&Request {protocol_version:PROTOCOL_VERSION,request_id:id.into(),session_id:session.map(str::to_owned),command}).unwrap()))).await.unwrap();
            }
            let mut sequence=0;
            let mut instance=None;
            let mut responses=std::collections::BTreeSet::new();
            let mut completed=false;
            tokio::time::timeout(Duration::from_secs(3),async {
                loop {
                    let message=received.recv().await.unwrap();
                    assert_eq!(message.sequence,sequence+1);sequence=message.sequence;
                    assert_eq!(message.protocol_version,7);
                    if let Some(expected)=&instance {assert_eq!(&message.instance_id,expected);} else {instance=Some(message.instance_id.clone());}
                    // Exercise output serialization as the real stdout writer does.
                    assert_eq!(decode::<Message>(&encode(&message).unwrap()).unwrap(),message);
                    if let Some(id)=&message.request_id {
                        responses.insert(id.clone());
                        match id.as_str() {
                            "early"|"invalid" => assert!(matches!(message.event,Event::Error(_))),
                            "progress" => assert!(matches!(message.event,Event::Progress(ref p) if p.accepted_end==3 && p.input_state==tts_protocol::PlanInputState::Open)),
                            _ => {},
                        }
                    }
                    if matches!(message.event,Event::SessionEnded {reason:EndReason::Completed,..}) {assert_eq!(message.session_id.as_deref(),Some("one"));completed=true;}
                    if completed && responses.contains("seal") {break;}
                }
            }).await.unwrap();
            assert!(responses.contains("seal"));
            input.send(Ok(Some(encode(&Request {protocol_version:7,request_id:"shutdown".into(),session_id:None,command:Command::Shutdown}).unwrap()))).await.unwrap();
            while let Some(message)=received.recv().await {assert_eq!(message.sequence,sequence+1);sequence=message.sequence;}
            task.await.unwrap().unwrap();
            assert_eq!(backend.calls.borrow().iter().map(|c| c.1.as_str()).collect::<Vec<_>>(),vec!["A","B","A"]);
            assert!(!root.path().join("models").exists());
        }).await;
    }
}
