//! Local preparation owns its bounded progress channel without a forwarding task.
use crate::{EngineError, Event};
use std::future::Future;
use tokio::sync::mpsc;

pub(crate) async fn with_progress<T, F>(
    start: impl FnOnce(mpsc::Sender<Event>) -> F,
    mut progress: impl FnMut(Event),
) -> Result<T, EngineError>
where
    F: Future<Output = anyhow::Result<T>>,
{
    let (tx, mut rx) = mpsc::channel(16);
    let prepare = start(tx);
    tokio::pin!(prepare);
    let mut progress_open = true;
    loop {
        tokio::select! {
            result = &mut prepare => {
                while let Ok(event) = rx.try_recv() {
                    progress(event);
                }
                return result.map_err(EngineError::Prepare);
            }
            event = rx.recv(), if progress_open => {
                if let Some(event) = event {
                    progress(event);
                } else {
                    progress_open = false;
                }
            }
        }
    }
}
