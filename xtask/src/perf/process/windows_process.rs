use anyhow::{Context, ensure};
use std::{
    mem::size_of,
    os::windows::{io::AsRawHandle, process::CommandExt},
    process::{Child, Command, ExitStatus},
    time::{Duration, Instant},
};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE},
        System::{
            Diagnostics::ToolHelp::{
                CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First,
                Thread32Next,
            },
            JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
                QueryInformationJobObject, SetInformationJobObject, TerminateJobObject,
            },
            Threading::{CREATE_SUSPENDED, OpenThread, ResumeThread, THREAD_SUSPEND_RESUME},
        },
    },
    core::PCWSTR,
};

struct Handle(HANDLE);
impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: Each Handle exclusively owns one successful Win32 acquisition.
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

pub struct Process {
    child: Child,
    job: Handle,
    cleaned: bool,
}

impl Process {
    pub fn spawn(command: &mut Command) -> anyhow::Result<Self> {
        // SAFETY: Null security/name use OS defaults; result is exclusively owned.
        let job = Handle(unsafe { CreateJobObjectW(None, PCWSTR::null())? });
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: Buffer type and byte size match JobObjectExtendedLimitInformation.
        unsafe {
            SetInformationJobObject(
                job.0,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as _,
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )?;
        }
        let child = command.creation_flags(CREATE_SUSPENDED.0).spawn()?;
        let mut process = Self {
            child,
            job,
            cleaned: false,
        };
        // SAFETY: Child owns a valid process handle; job stays alive through cleanup.
        let assigned = unsafe {
            AssignProcessToJobObject(
                process.job.0,
                HANDLE(process.child.as_raw_handle() as isize),
            )
        };
        if let Err(error) = assigned {
            process.child.kill()?;
            process.child.wait()?;
            return Err(error.into());
        }
        process.resume_primary()?;
        Ok(process)
    }

    fn resume_primary(&mut self) -> anyhow::Result<()> {
        // SAFETY: Snapshot is owned; THREADENTRY32 has its required size initialized.
        let snapshot = Handle(unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)? });
        let mut entry = THREADENTRY32 {
            dwSize: size_of::<THREADENTRY32>() as u32,
            ..Default::default()
        };
        unsafe {
            Thread32First(snapshot.0, &mut entry)?;
            loop {
                if entry.th32OwnerProcessID == self.child.id() {
                    let thread = Handle(OpenThread(
                        THREAD_SUSPEND_RESUME,
                        false,
                        entry.th32ThreadID,
                    )?);
                    ensure!(
                        ResumeThread(thread.0) != u32::MAX,
                        "resume suspended probe failed"
                    );
                    return Ok(());
                }
                if Thread32Next(snapshot.0, &mut entry).is_err() {
                    break;
                }
            }
        }
        anyhow::bail!("suspended probe primary thread not found")
    }

    pub fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn take_stdout(&mut self) -> Option<std::process::ChildStdout> {
        self.child.stdout.take()
    }
    pub fn take_stderr(&mut self) -> Option<std::process::ChildStderr> {
        self.child.stderr.take()
    }

    pub fn try_wait(&mut self) -> anyhow::Result<Option<ExitStatus>> {
        Ok(self.child.try_wait()?)
    }

    pub fn active_processes(&self) -> anyhow::Result<u32> {
        let mut accounting = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
        // SAFETY: Correct typed output buffer, valid owned job handle.
        unsafe {
            QueryInformationJobObject(
                self.job.0,
                JobObjectBasicAccountingInformation,
                &mut accounting as *mut _ as _,
                size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                None,
            )?;
        }
        Ok(accounting.ActiveProcesses)
    }

    /// Kill only this tool-owned job and verify no descendants remain active.
    pub fn cleanup(&mut self) -> anyhow::Result<()> {
        if self.cleaned {
            return Ok(());
        }
        // SAFETY: The job contains only the suspended child and its descendants.
        unsafe {
            TerminateJobObject(self.job.0, 1)?;
        }
        self.child.wait().context("reap probe")?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while self.active_processes()? != 0 {
            ensure!(
                Instant::now() < deadline,
                "probe job cleanup unverified; refuse next measurement"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        self.cleaned = true;
        Ok(())
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if !self.cleaned {
            // Cleanup errors cannot be returned by Drop; callers must explicitly
            // call cleanup and gate further measurement on its successful result.
            let _ = self.cleanup();
        }
    }
}
