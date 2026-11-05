//! Actual HTTP bodies against an owned loopback server. No external network.
use sha2::{Digest, Sha256};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use talechime_backends::downloads::{
    DownloadConfig, DownloadOptions, Downloads, FileProgressBridge,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
const REV: &str = "0123456789abcdef0123456789abcdef01234567";
const SIZE: usize = 10 * 1024 * 1024;
#[derive(Clone, Copy)]
enum Mode {
    Normal,
    Ignore,
    BadRange,
    Corrupt,
    Retry,
    Expired,
    Changed,
}
#[derive(Default)]
struct Counters {
    live: AtomicUsize,
    peak: AtomicUsize,
}
struct Active(Arc<Counters>);
impl Active {
    fn new(c: Arc<Counters>) -> Self {
        let n = c.live.fetch_add(1, Ordering::SeqCst) + 1;
        c.peak.fetch_max(n, Ordering::SeqCst);
        Self(c)
    }
}
impl Drop for Active {
    fn drop(&mut self) {
        self.0.live.fetch_sub(1, Ordering::SeqCst);
    }
}
struct Server {
    endpoint: String,
    payload: Arc<Vec<u8>>,
    ranges: Arc<Mutex<Vec<(usize, usize)>>>,
    heads: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Server {
    async fn new(mode: Mode, counters: Arc<Counters>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let payload = Arc::new((0..SIZE).map(|i| (i % 251) as u8).collect::<Vec<_>>());
        let hash = format!("{:x}", Sha256::digest(&*payload));
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let captures = ranges.clone();
        let body = payload.clone();
        let attempts = Arc::new(AtomicUsize::new(0));
        let heads = Arc::new(AtomicUsize::new(0));
        let head_counts = heads.clone();
        let base = endpoint.clone();
        let task = tokio::spawn(async move {
            let mut requests = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        let (mut socket, _) = accepted.unwrap();
                        let payload = body.clone(); let hash = hash.clone(); let ranges = captures.clone(); let c = counters.clone(); let attempts = attempts.clone(); let heads = head_counts.clone(); let base = base.clone();
                        requests.spawn(async move {
                            let mut request = Vec::new(); let mut buf = [0; 1024];
                            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                                let n = socket.read(&mut buf).await?; if n == 0 || request.len() > 16384 { return Ok::<_, std::io::Error>(()); } request.extend_from_slice(&buf[..n]);
                            }
                            let text = String::from_utf8_lossy(&request).to_ascii_lowercase();
                            if text.starts_with("head ") {
                                let count = heads.fetch_add(1, Ordering::SeqCst);
                                if matches!(mode, Mode::Expired | Mode::Changed) {
                                    let next = if count == 0 { "expired" } else { "fresh" };
                                    let identity = if count > 0 && matches!(mode, Mode::Changed) { "0".repeat(64) } else { hash.clone() };
                                    socket.write_all(format!("HTTP/1.1 302 Found\r\nContent-Length: 0\r\nX-Linked-Size: {SIZE}\r\nX-Linked-Etag: \"{identity}\"\r\nX-Repo-Commit: {REV}\r\nLocation: {base}/{next}\r\nConnection: close\r\n\r\n").as_bytes()).await?;
                                } else {
                                    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {SIZE}\r\nETag: \"{hash}\"\r\nX-Repo-Commit: {REV}\r\nConnection: close\r\n\r\n").as_bytes()).await?;
                                }
                                return Ok(());
                            }
                            if text.starts_with("get /expired ") {
                                socket.write_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await?;
                                return Ok(());
                            }
                            let active = Active::new(c);
                            let range = text.lines().find_map(|line| line.strip_prefix("range: bytes=")).map(|s| { let (a,b) = s.trim().split_once('-').unwrap(); (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap_or(SIZE-1)) });
                            if matches!(mode, Mode::Retry) && attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                                socket.write_all(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await?; return Ok(());
                            }
                            let ranged = range.is_some() && !matches!(mode, Mode::Ignore);
                            let (start, end) = if ranged { range.unwrap() } else { (0,SIZE-1) };
                            ranges.lock().unwrap().push((start,end));
                            let cr = if ranged { format!("Content-Range: bytes {}-{end}/{SIZE}\r\n", if matches!(mode, Mode::BadRange) { start+1 } else { start }) } else { String::new() };
                            socket.write_all(format!("HTTP/1.1 {} OK\r\nContent-Length: {}\r\n{cr}Connection: close\r\n\r\n", if ranged {206} else {200}, end-start+1).as_bytes()).await?;
                            for bytes in payload[start..=end].chunks(64*1024) {
                                if matches!(mode, Mode::Corrupt) { let mut b=bytes.to_vec(); b[0]^=1; socket.write_all(&b).await?; }
                                else { socket.write_all(bytes).await?; }
                                tokio::time::sleep(Duration::from_millis(3)).await;
                            }
                            drop(active); Ok(())
                        });
                    }
                    _ = requests.join_next(), if !requests.is_empty() => {},
                }
            }
        });
        Self {
            endpoint,
            payload,
            ranges,
            heads,
            task,
        }
    }
    fn client(&self, cache: &std::path::Path, connections: usize) -> Downloads {
        Downloads::with_options(
            &DownloadConfig {
                cache_dir: cache.into(),
                endpoint: Some(self.endpoint.clone()),
                token: None,
            },
            DownloadOptions { connections },
        )
        .unwrap()
    }
}
fn bridge() -> FileProgressBridge {
    FileProgressBridge::new(
        tokio::sync::mpsc::channel(1).0,
        "weights".into(),
        SIZE as u64,
    )
}
#[tokio::test]
async fn ranges_share_a_global_budget_across_clients_and_verify_bytes() {
    let counters = Arc::new(Counters::default());
    let a = Server::new(Mode::Normal, counters.clone()).await;
    let b = Server::new(Mode::Normal, counters.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ca = a.client(&root.path().join("a"), 4);
    let cb = b.client(&root.path().join("b"), 4);
    let da = root.path().join("a.bin");
    let db = root.path().join("b.bin");
    let (ra, rb) = tokio::join!(
        ca.ensure(&da, "team/model", REV, "weights.bin", bridge()),
        cb.ensure(&db, "team/model", REV, "weights.bin", bridge())
    );
    ra.unwrap();
    rb.unwrap();
    assert_eq!(std::fs::read(da).unwrap(), *a.payload);
    assert_eq!(std::fs::read(db).unwrap(), *b.payload);
    assert!((2..=4).contains(&counters.peak.load(Ordering::SeqCst)));
}
#[tokio::test]
async fn dropping_transfer_checkpoints_sparse_ranges_and_new_client_resumes() {
    let server = Server::new(Mode::Normal, Arc::new(Counters::default())).await;
    let root = tempfile::tempdir().unwrap();
    let cache = root.path().join("cache");
    let dest = root.path().join("engine.bin");
    let client = server.client(&cache, 4);
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut download = Box::pin(client.ensure(
        &dest,
        "team/model",
        REV,
        "weights.bin",
        FileProgressBridge::new(tx, "weights".into(), SIZE as u64),
    ));
    loop {
        tokio::select! { r=&mut download=>panic!("ended before pause: {r:?}"), event=rx.recv()=>if let Some(talechime_protocol::Event::ModelProgress {downloaded,..})=event && downloaded>=512*1024 {break;}, _=tokio::time::sleep(Duration::from_secs(10))=>panic!("no progress") }
    }
    drop(download);
    drop(client);
    assert!(!dest.exists());
    let blobs = cache.join("models--team--model/blobs");
    let checkpoint = std::fs::read_dir(&blobs)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.to_string_lossy().ends_with(".ranges.json"))
        .unwrap();
    let saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&checkpoint).unwrap()).unwrap();
    let completed: Vec<u64> = saved["completed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap())
        .collect();
    assert!(completed.iter().sum::<u64>() > 0 && completed.iter().sum::<u64>() < SIZE as u64);
    let before = server.ranges.lock().unwrap().len();
    server
        .client(&cache, 2)
        .ensure(&dest, "team/model", REV, "weights.bin", bridge())
        .await
        .unwrap();
    assert_eq!(std::fs::read(dest).unwrap(), *server.payload);
    assert!(!checkpoint.exists());
    let starts: Vec<_> = server.ranges.lock().unwrap()[before..]
        .iter()
        .map(|r| r.0 as u64)
        .collect();
    for (i, n) in completed.iter().enumerate() {
        let length = (4 * 1024 * 1024).min(SIZE as u64 - i as u64 * 4 * 1024 * 1024);
        if *n < length {
            assert!(starts.contains(&(i as u64 * 4 * 1024 * 1024 + n)));
        }
    }
}
#[tokio::test]
async fn unsupported_ranges_fall_back_and_invalid_or_corrupt_data_is_not_published() {
    for mode in [Mode::Ignore, Mode::BadRange, Mode::Corrupt, Mode::Retry] {
        let server = Server::new(mode, Arc::new(Counters::default())).await;
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("engine.bin");
        let result = server
            .client(&root.path().join("cache"), 4)
            .ensure(&dest, "team/model", REV, "weights.bin", bridge())
            .await;
        if matches!(mode, Mode::Ignore | Mode::Retry) {
            result.unwrap();
            assert_eq!(std::fs::read(dest).unwrap(), *server.payload);
        } else {
            assert!(result.is_err());
            assert!(!dest.exists());
        }
    }
}

#[tokio::test]
async fn expired_signed_address_refreshes_once_and_identity_changes_are_rejected() {
    for mode in [Mode::Expired, Mode::Changed] {
        let server = Server::new(mode, Arc::new(Counters::default())).await;
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("engine.bin");
        let result = server
            .client(&root.path().join("cache"), 4)
            .ensure(&dest, "team/model", REV, "weights.bin", bridge())
            .await;
        if matches!(mode, Mode::Expired) {
            result.unwrap();
            assert_eq!(std::fs::read(dest).unwrap(), *server.payload);
            assert_eq!(
                server.heads.load(Ordering::SeqCst),
                2,
                "one serialized refresh for all ranges"
            );
        } else {
            assert!(result.is_err());
            assert!(!dest.exists());
        }
    }
}

#[tokio::test]
async fn checkpoint_write_failure_is_visible_after_cancellation() {
    let server = Server::new(Mode::Normal, Arc::new(Counters::default())).await;
    let root = tempfile::tempdir().unwrap();
    let cache = root.path().join("cache");
    let dest = root.path().join("engine.bin");
    let client = server.client(&cache, 2);
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let mut download = Box::pin(client.ensure(
        &dest,
        "team/model",
        REV,
        "weights.bin",
        FileProgressBridge::new(tx, "weights".into(), SIZE as u64),
    ));
    loop {
        tokio::select! {
            result = &mut download => panic!("finished before failure injection: {result:?}"),
            event = rx.recv() => if let Some(talechime_protocol::Event::ModelProgress { downloaded, .. }) = event && downloaded >= 512*1024 { break; },
            _ = tokio::time::sleep(Duration::from_secs(10)) => panic!("no download progress"),
        }
    }
    let checkpoint = std::fs::read_dir(cache.join("models--team--model/blobs"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.to_string_lossy().ends_with(".ranges.json"))
        .unwrap();
    // Make atomic replacement fail deterministically inside our isolated directory.
    std::fs::remove_file(&checkpoint).unwrap();
    std::fs::create_dir(&checkpoint).unwrap();
    drop(download);
    assert!(client.checkpoint_failed());
    assert!(!dest.exists());
}
