//! Bounded escape from probe pipes when native cleanup cannot be verified.
/// Nonblocking pipes let failed cleanup stop readers rather than hang scoped joins.
pub(crate) struct InterruptiblePipe<'a, T> {
    pipe: T,
    abort: &'a std::sync::atomic::AtomicBool,
}

impl<'a, T: std::os::fd::AsRawFd> InterruptiblePipe<'a, T> {
    pub fn new(pipe: T, abort: &'a std::sync::atomic::AtomicBool) -> std::io::Result<Self> {
        // SAFETY: Descriptor belongs to the owned pipe and stays alive in this wrapper.
        unsafe {
            let flags = libc::fcntl(pipe.as_raw_fd(), libc::F_GETFL);
            if flags < 0
                || libc::fcntl(pipe.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) < 0
            {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(Self { pipe, abort })
    }
}

impl<T: std::io::Read> std::io::Read for InterruptiblePipe<'_, T> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if self.abort.load(std::sync::atomic::Ordering::Acquire) {
                return Err(std::io::Error::other(
                    "reader interrupted: cleanup unverified",
                ));
            }
            match self.pipe.read(bytes) {
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                result => return result,
            }
        }
    }
}
