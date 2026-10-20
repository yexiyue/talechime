//! Console callbacks only set a flag; the executor owns cleanup and evidence.
use std::sync::atomic::{AtomicBool, Ordering};

pub static REQUESTED: AtomicBool = AtomicBool::new(false);

pub fn requested() -> bool {
    REQUESTED.load(Ordering::Acquire)
}

pub struct Handler {
    #[cfg(target_os = "macos")]
    previous: [libc::sigaction; 2],
}

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
    #[cfg(target_os = "macos")]
    {
        // SAFETY: Correct sigaction buffers; callback only stores a static atomic.
        unsafe {
            let mut previous: [libc::sigaction; 2] = std::mem::zeroed();
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = mac_signal as *const () as usize;
            libc::sigemptyset(&mut action.sa_mask);
            for (index, signal) in [libc::SIGINT, libc::SIGTERM].into_iter().enumerate() {
                if libc::sigaction(signal, &action, &mut previous[index]) != 0 {
                    if index > 0 {
                        libc::sigaction(libc::SIGINT, &previous[0], std::ptr::null_mut());
                    }
                    return Err(std::io::Error::last_os_error().into());
                }
            }
            Ok(Handler { previous })
        }
    }
    #[cfg(not(target_os = "macos"))]
    Ok(Handler {})
}

impl Drop for Handler {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        // SAFETY: Restore the exact actions saved by this CLI-owned guard.
        unsafe {
            for (index, signal) in [libc::SIGINT, libc::SIGTERM].into_iter().enumerate() {
                libc::sigaction(signal, &self.previous[index], std::ptr::null_mut());
            }
        }
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

#[cfg(target_os = "macos")]
extern "C" fn mac_signal(_signal: libc::c_int) {
    REQUESTED.store(true, Ordering::Release);
}
