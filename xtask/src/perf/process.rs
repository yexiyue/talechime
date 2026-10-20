//! Windows process-tree containment, established while the process is suspended.
#[cfg(windows)]
mod windows_process;
#[cfg(windows)]
pub use windows_process::Process;

#[cfg(not(windows))]
pub struct Process;

#[cfg(not(windows))]
impl Process {
    pub fn spawn(_command: &mut std::process::Command) -> anyhow::Result<Self> {
        anyhow::bail!("deferred: native process lifecycle is currently verified only on Windows")
    }
}
