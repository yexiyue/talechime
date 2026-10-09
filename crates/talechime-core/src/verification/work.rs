use std::sync::{Mutex, MutexGuard};
use tokio::sync::watch;

/// Per-execution completion receipts. Native requests may outlive their futures.
#[derive(Default)]
pub(crate) struct RequestCompletions {
    requests: Mutex<Vec<watch::Receiver<bool>>>,
}

impl RequestCompletions {
    fn requests(&self) -> MutexGuard<'_, Vec<watch::Receiver<bool>>> {
        self.requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
    pub(super) fn register(&self, completion: watch::Receiver<bool>) {
        let mut requests = self.requests();
        requests.retain(|request| !finished(request));
        requests.push(completion);
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.requests().iter().all(finished)
    }

    pub(crate) async fn settled(&self) {
        let requests = self.requests().clone();
        for mut completion in requests {
            let _ = completion.wait_for(|done| *done).await;
        }
    }
}

fn finished(completion: &watch::Receiver<bool>) -> bool {
    *completion.borrow() || completion.has_changed().is_err()
}
