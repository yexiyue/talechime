//! Owned blocking work for cache verification. Dropping the future cancels and joins it.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Default)]
pub(crate) struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    pub(crate) fn check(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.0.load(Ordering::Acquire),
            "resource verification cancelled"
        );
        Ok(())
    }
}

struct Owner {
    thread: Option<std::thread::JoinHandle<()>>,
    cancellation: Cancellation,
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.cancellation.0.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub(crate) async fn run<T: Send + 'static>(
    work: impl FnOnce(Cancellation) -> anyhow::Result<T> + Send + 'static,
) -> anyhow::Result<T> {
    let cancellation = Cancellation::default();
    let worker_cancellation = cancellation.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let thread = std::thread::Builder::new()
        .name("resource-verification".into())
        .spawn(move || {
            let _ = tx.send(work(worker_cancellation));
        })?;
    let mut owner = Owner {
        thread: Some(thread),
        cancellation,
    };
    let result = rx.await;
    if let Some(thread) = owner.thread.take() {
        thread
            .join()
            .map_err(|_| anyhow::anyhow!("resource verification worker panicked"))?;
    }
    result?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dropping_future_cancels_and_joins_worker() {
        let exited = Arc::new(AtomicBool::new(false));
        let worker_exited = exited.clone();
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(run(move |cancellation| {
            let _ = started.send(());
            while cancellation.check().is_ok() {
                std::thread::yield_now();
            }
            worker_exited.store(true, Ordering::Release);
            Ok(())
        }));
        ready.await.unwrap();
        task.abort();
        let _ = task.await;
        assert!(exited.load(Ordering::Acquire));
    }
}
