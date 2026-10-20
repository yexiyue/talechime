//! Console callbacks only set a flag; the executor owns cleanup and evidence.
use std::sync::atomic::{AtomicBool, Ordering};

pub static REQUESTED: AtomicBool = AtomicBool::new(false);

pub fn requested() -> bool {
    REQUESTED.load(Ordering::Acquire)
}

pub struct Handler;

#[cfg(windows)]
unsafe extern "system" fn signal(event: u32) -> windows::Win32::Foundation::BOOL {
    use windows::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT};
    if matches!(event, CTRL_C_EVENT | CTRL_BREAK_EVENT) {
        REQUESTED.store(true, Ordering::Release);
        true.into()
    } else {
        false.into()
    }
}

pub fn install() -> anyhow::Result<Handler> {
    REQUESTED.store(false, Ordering::Release);
    #[cfg(windows)]
    // SAFETY: Static callback touches only a static atomic; guard unregisters it.
    unsafe {
        windows::Win32::System::Console::SetConsoleCtrlHandler(Some(signal), true)?;
    }
    Ok(Handler)
}

impl Drop for Handler {
    fn drop(&mut self) {
        #[cfg(windows)]
        // SAFETY: Unregister the same static callback installed by this guard.
        unsafe {
            let _ = windows::Win32::System::Console::SetConsoleCtrlHandler(Some(signal), false);
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    #[test]
    fn console_callback_requests_cancellation_without_handling_other_events() {
        use windows::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT};
        // SAFETY: Invoke our callback directly; no process-wide console event is sent.
        unsafe {
            assert!(!signal(CTRL_CLOSE_EVENT).as_bool());
            assert!(signal(CTRL_BREAK_EVENT).as_bool());
        }
        assert!(requested());
        REQUESTED.store(false, Ordering::Release);
    }
}
