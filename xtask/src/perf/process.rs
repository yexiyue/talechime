//! Platform-owned probe containment; unsupported platforms fail explicitly.
#[cfg(windows)]
mod windows_process;
#[cfg(windows)]
pub use windows_process::Process;

#[cfg(not(any(windows, target_os = "macos")))]
pub struct Process;

#[cfg(not(any(windows, target_os = "macos")))]
impl Process {
    pub fn spawn(_command: &mut std::process::Command) -> anyhow::Result<Self> {
        anyhow::bail!("deferred: native process lifecycle is implemented only on Windows and macOS")
    }
}

#[cfg(target_os = "macos")]
mod macos_process;
#[cfg(target_os = "macos")]
pub use macos_process::Process;

#[cfg(target_os = "macos")]
mod macos_pipe;
#[cfg(target_os = "macos")]
pub(crate) use macos_pipe::InterruptiblePipe;
