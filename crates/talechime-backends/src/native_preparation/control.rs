use super::PreparationError;
use tokio::sync::{oneshot, watch};

/// Cooperative cancellation; cancellation acknowledgement waits for owned work.
#[derive(Clone, Debug)]
pub struct PreparationControl(std::sync::Arc<State>);
#[derive(Debug)]
struct State {
    cancel: watch::Sender<bool>,
    published: std::sync::Mutex<bool>,
}
impl Default for PreparationControl {
    fn default() -> Self {
        Self(std::sync::Arc::new(State {
            cancel: watch::channel(false).0,
            published: std::sync::Mutex::new(false),
        }))
    }
}
impl PreparationControl {
    /// Request cancellation without dropping a preparation future.
    pub fn cancel(&self) {
        if let Ok(published) = self.0.published.lock()
            && !*published
        {
            self.0.cancel.send_replace(true);
        }
    }
    /// Check at a bounded I/O, component or tensor boundary.
    pub fn check(&self) -> Result<(), PreparationError> {
        if *self.0.cancel.borrow() {
            Err(PreparationError::Cancelled)
        } else {
            Ok(())
        }
    }
    pub(super) fn commit<T>(
        &self,
        publish: impl FnOnce() -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let mut published = self
            .0
            .published
            .lock()
            .map_err(|_| anyhow::anyhow!("preparation publication unavailable"))?;
        self.check()?;
        let result = publish()?;
        *published = true;
        Ok(result)
    }
    pub(super) async fn cancelled(&self) {
        let mut receiver = self.0.cancel.subscribe();
        while !*receiver.borrow_and_update() {
            if receiver.changed().await.is_err() {
                break;
            }
        }
    }
}
struct Owner {
    thread: Option<std::thread::JoinHandle<()>>,
    control: PreparationControl,
}
impl Drop for Owner {
    fn drop(&mut self) {
        // Even an accidentally dropped future cannot detach a writer or unlock
        // its transaction while native work still owns it.
        if let Some(thread) = self.thread.take() {
            self.control.cancel();
            let _ = thread.join();
        }
    }
}
pub(super) async fn worker<T: Send + 'static>(
    control: PreparationControl,
    work: impl FnOnce(PreparationControl) -> anyhow::Result<T> + Send + 'static,
) -> anyhow::Result<T> {
    let (tx, mut rx) = oneshot::channel();
    let worker_control = control.clone();
    let thread = std::thread::Builder::new()
        .name("model-preparation".into())
        .spawn(move || {
            let result = work(worker_control);
            let _ = tx.send(result);
        })?;
    let mut owner = Owner {
        thread: Some(thread),
        control: control.clone(),
    };
    let result = tokio::select! {
        result = &mut rx => result?,
        _ = control.cancelled() => {
            let _ = rx.await;
            return Err(PreparationError::Cancelled.into());
        }
    };
    if let Some(thread) = owner.thread.take() {
        thread
            .join()
            .map_err(|_| anyhow::anyhow!("model preparation worker panicked"))?;
    }
    control.check()?;
    result
}
