//! Design threads remain owned until cancellation and native cleanup complete.
use std::thread::JoinHandle;
use tokio::sync::oneshot;

pub(crate) struct DesignThread<T> {
    pub(crate) receiver: oneshot::Receiver<T>,
    thread: Option<JoinHandle<()>>,
}
impl<T> DesignThread<T> {
    pub(crate) fn new(receiver: oneshot::Receiver<T>, thread: JoinHandle<()>) -> Self {
        Self {
            receiver,
            thread: Some(thread),
        }
    }
}
impl<T> Drop for DesignThread<T> {
    fn drop(&mut self) {
        // Both kernels observe receiver closure at their cancellation boundaries.
        // Joining first would leave inference waiting on an unsignalled cancel.
        self.receiver.close();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::time::{Duration, Instant};

    struct Resource(Arc<AtomicBool>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn dropping_design_future_closes_receiver_then_waits_for_native_resources() {
        let released = Arc::new(AtomicBool::new(false));
        let native_released = released.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        let native_cancelled = cancelled.clone();
        let (sender, receiver) = oneshot::channel::<()>();
        let thread = std::thread::spawn(move || {
            let _resource = Resource(native_released);
            // A deadline makes a broken close-before-join order fail instead of hanging.
            let deadline = Instant::now() + Duration::from_secs(2);
            while !sender.is_closed() && Instant::now() < deadline {
                std::thread::yield_now();
            }
            native_cancelled.store(sender.is_closed(), Ordering::SeqCst);
            assert!(
                sender.is_closed(),
                "join ran before cancellation was signalled"
            );
        });
        {
            let future = async move {
                let mut owner = DesignThread::new(receiver, thread);
                (&mut owner.receiver).await
            };
            tokio::pin!(future);
            tokio::select! {
                biased;
                result = &mut future => panic!("design unexpectedly finished: {result:?}"),
                _ = std::future::ready(()) => {}
            }
        }
        assert!(
            released.load(Ordering::SeqCst),
            "future drop returned before cleanup"
        );
        assert!(
            cancelled.load(Ordering::SeqCst),
            "receiver was not closed before join"
        );
    }

    #[tokio::test]
    async fn successful_design_also_joins_before_owner_returns() {
        let released = Arc::new(AtomicBool::new(false));
        let native_released = released.clone();
        let (sender, receiver) = oneshot::channel();
        let thread = std::thread::spawn(move || {
            let _resource = Resource(native_released);
            let _ = sender.send(42);
        });
        let mut owner = DesignThread::new(receiver, thread);
        assert_eq!((&mut owner.receiver).await.unwrap(), 42);
        drop(owner);
        assert!(released.load(Ordering::SeqCst));
    }
}
