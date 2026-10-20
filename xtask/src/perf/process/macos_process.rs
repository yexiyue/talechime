//! Process groups contain trusted probes, not adversarial processes that escape sessions.
use anyhow::{Context, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem::{size_of, zeroed},
    os::unix::process::{CommandExt, ExitStatusExt},
    process::{Child, Command, ExitStatus},
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
struct Identity {
    start: (u64, u64),
    parent: u32,
    group: u32,
    zombie: bool,
}

fn identity(pid: u32) -> anyhow::Result<Option<Identity>> {
    // SAFETY: Zero is valid for this C output structure; size and flavor match.
    let mut info: libc::proc_bsdinfo = unsafe { zeroed() };
    let length = unsafe {
        libc::proc_pidinfo(
            pid as i32,
            libc::PROC_PIDTBSDINFO,
            0,
            &mut info as *mut _ as _,
            size_of::<libc::proc_bsdinfo>() as i32,
        )
    };
    if length == 0 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(None);
        }
        return Err(error).context("query process identity");
    }
    ensure!(
        length as usize == size_of::<libc::proc_bsdinfo>(),
        "short process identity"
    );
    Ok(Some(Identity {
        start: (info.pbi_start_tvsec, info.pbi_start_tvusec),
        parent: info.pbi_ppid,
        group: info.pbi_pgid,
        zombie: info.pbi_status == 5,
    }))
}

fn snapshot() -> anyhow::Result<BTreeMap<u32, Identity>> {
    // A bounded, generously sized snapshot avoids allocation races with the PID count.
    let mut pids = vec![0i32; 131072];
    // SAFETY: Writable aligned PID buffer, byte length fits c_int.
    let count = unsafe {
        libc::proc_listpids(
            1, /* PROC_ALL_PIDS, sys/proc_info.h */
            0,
            pids.as_mut_ptr() as _,
            size_of_val(pids.as_slice()) as i32,
        )
    };
    ensure!(
        count > 0 && (count as usize) < size_of_val(pids.as_slice()),
        "process snapshot unavailable or full"
    );
    let mut result = BTreeMap::new();
    for pid in pids
        .into_iter()
        .take(count as usize / size_of::<i32>())
        .filter(|pid| *pid > 0)
    {
        // Other users' processes can be inaccessible. Owned identities are checked separately.
        if let Ok(Some(info)) = identity(pid as u32) {
            result.insert(pid as u32, info);
        }
    }
    Ok(result)
}

/// Owns a newly created probe group and the directly waitable child.
pub struct Process {
    child: Child,
    group: u32,
    known: BTreeMap<u32, Identity>,
    escaped: bool,
    uncertainty: BTreeSet<String>,
    cleaned: bool,
}

impl Process {
    pub fn spawn(command: &mut Command) -> anyhow::Result<Self> {
        let child = command.process_group(0).spawn()?;
        let group = child.id();
        let mut process = Self {
            child,
            group,
            known: BTreeMap::new(),
            escaped: false,
            uncertainty: BTreeSet::new(),
            cleaned: false,
        };
        // The child may already be a zombie, but remains identifiable until reaped.
        let info = identity(group)?.context("probe identity disappeared before wait")?;
        ensure!(info.group == group, "probe group not established at spawn");
        process.known.insert(group, info);
        Ok(process)
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

    fn members(&mut self) -> Vec<(u32, Identity)> {
        let all = match snapshot() {
            Ok(all) => all,
            Err(error) => {
                self.uncertainty.insert(error.to_string());
                BTreeMap::new()
            }
        };
        // The unreaped direct child pins the original PID/group number even
        // when libproc omits its zombie from the system-wide snapshot.
        let group_is_owned = !self.cleaned && self.known.contains_key(&self.group);
        let known = &mut self.known;
        loop {
            let previous = known.len();
            for (&pid, &info) in &all {
                let owned_parent = all.get(&info.parent).is_some_and(|parent| {
                    known
                        .get(&info.parent)
                        .is_some_and(|saved| saved.start == parent.start)
                });
                if (group_is_owned && info.group == self.group) || owned_parent {
                    match known.get(&pid) {
                        Some(saved) if saved.start != info.start => {
                            self.uncertainty
                                .insert(format!("probe PID {pid} identity changed"));
                        }
                        None => {
                            known.insert(pid, info);
                        }
                        _ => {}
                    }
                }
            }
            if previous == known.len() {
                break;
            }
        }
        let mut active = Vec::new();
        for (&pid, saved) in known.iter() {
            match identity(pid) {
                Ok(Some(info)) if info.start != saved.start => {
                    self.uncertainty
                        .insert(format!("observed descendant PID {pid} reused"));
                }
                Ok(Some(info)) if !info.zombie => active.push((pid, info)),
                Ok(_) => {}
                Err(error) => {
                    self.uncertainty
                        .insert(format!("query descendant {pid}: {error}"));
                }
            }
        }
        active
    }

    fn verify_identity(&self) -> anyhow::Result<()> {
        ensure!(
            self.uncertainty.is_empty(),
            "probe identities unverified: {}",
            self.uncertainty
                .iter()
                .take(16)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        );
        Ok(())
    }

    fn verify_containment(&self) -> anyhow::Result<()> {
        self.verify_identity()?;
        ensure!(
            !self.escaped,
            "descendant escaped probe group; refuse next measurement"
        );
        Ok(())
    }

    pub fn try_wait(&mut self) -> anyhow::Result<Option<ExitStatus>> {
        let members = self.members();
        self.escaped |= members.iter().any(|(_, info)| info.group != self.group);
        self.verify_identity()?;
        if self.cleaned {
            return Ok(self.child.try_wait()?);
        }
        // Observe exit without reaping: retaining the leader pins its PID/group
        // identity until all descendants have been inspected and cleaned.
        let mut info: libc::siginfo_t = unsafe { zeroed() };
        // SAFETY: A correctly sized output buffer and our unreaped direct child PID.
        if unsafe {
            libc::waitid(
                libc::P_PID,
                self.group,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        } != 0
        {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EINTR) {
                return Ok(None);
            }
            return Err(error).context("observe probe exit without reaping");
        }
        if info.si_pid == 0 {
            return Ok(None);
        }
        let status = match info.si_code {
            libc::CLD_EXITED => info.si_status << 8,
            libc::CLD_KILLED => info.si_status,
            libc::CLD_DUMPED => info.si_status | 0x80,
            _ => anyhow::bail!("unexpected probe wait status"),
        };
        Ok(Some(ExitStatus::from_raw(status)))
    }

    pub fn active_processes(&mut self) -> anyhow::Result<u32> {
        let count = self.members().len() as u32;
        self.verify_identity()?;
        Ok(count)
    }

    /// Terminate only identity-checked members; an observed escape still fails acceptance.
    pub fn cleanup(&mut self) -> anyhow::Result<()> {
        if self.cleaned {
            return self.verify_containment();
        }
        let start = Instant::now();
        loop {
            let members = self.members();
            self.escaped |= members.iter().any(|(_, info)| info.group != self.group);
            if members.is_empty() {
                self.child
                    .try_wait()?
                    .context("probe exited but is not yet waitable")?;
                self.cleaned = true;
                return self.verify_containment();
            }
            ensure!(
                start.elapsed() < Duration::from_secs(5),
                "probe cleanup unverified; refuse next measurement"
            );
            let signal = if start.elapsed() < Duration::from_millis(100) {
                libc::SIGTERM
            } else {
                libc::SIGKILL
            };
            for (pid, info) in members {
                let current = match identity(pid) {
                    Ok(Some(current)) if current.start == info.start => current,
                    Ok(None) => continue,
                    Ok(Some(_)) => {
                        self.uncertainty
                            .insert(format!("descendant PID {pid} changed before signal"));
                        continue;
                    }
                    Err(error) => {
                        self.uncertainty
                            .insert(format!("query descendant {pid} before signal: {error}"));
                        continue;
                    }
                };
                if !current.zombie {
                    // SAFETY: Positive identity-checked PID; never a process-name/global kill.
                    if unsafe { libc::kill(pid as i32, signal) } != 0 {
                        let error = std::io::Error::last_os_error();
                        if error.raw_os_error() != Some(libc::ESRCH) {
                            self.uncertainty
                                .insert(format!("signal probe {pid}: {error}"));
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if !self.cleaned && self.cleanup().is_err() {
            // If identity collection itself failed at admission, the unreaped
            // direct child is still ours. Never leave it running just because
            // descendant verification failed; the caller still gets a failure.
            if self.child.try_wait().is_ok_and(|status| status.is_none()) {
                let _ = self.child.kill();
                let deadline = Instant::now() + Duration::from_secs(1);
                while Instant::now() < deadline {
                    if !matches!(self.child.try_wait(), Ok(None)) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_identity_does_not_prevent_other_verified_cleanup() -> anyhow::Result<()> {
        let mut owned = Process::spawn(Command::new("/bin/sleep").arg("30"))?;
        let mut unrelated = Process::spawn(Command::new("/bin/sleep").arg("30"))?;
        let mut stale = identity(unrelated.id())?.context("unrelated fixture identity")?;
        stale.start.0 += 1;
        owned.known.insert(unrelated.id(), stale);
        assert!(owned.cleanup().is_err());
        assert!(
            owned.child.try_wait()?.is_some(),
            "verified child was left running"
        );
        assert!(unrelated.try_wait()?.is_none());
        unrelated.cleanup()?;
        Ok(())
    }
}
