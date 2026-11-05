//! Bounded real-network diagnostic. No model loading; never prints credentials or signed URLs.
//! cargo run -p talechime-backends --no-default-features --example download_progress_probe -- raw|bridge CACHE_DIR [ENDPOINT] [CONNECTIONS] [SECONDS] [BYTE_BUDGET]
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};
use talechime_backends::downloads::{
    DownloadConfig, DownloadOptions, Downloads, FileProgressBridge,
};
use talechime_protocol::Event;
const REPO: &str = "OpenMOSS-Team/MOSS-TTS-Realtime";
const REVISION: &str = "75682787d8e2fcc73faca37ba2931453ca9c4022";
const FILE: &str = "model.safetensors";
const SIZE: u64 = 4_663_931_664;
struct Raw {
    start: Instant,
    events: AtomicU64,
    bytes: Arc<AtomicU64>,
}
impl ProgressHandler for Raw {
    fn on_progress(&self, event: &ProgressEvent) {
        let count = self.events.fetch_add(1, Ordering::Relaxed);
        let (kind, bytes) = match event {
            ProgressEvent::Download(DownloadEvent::Progress { files }) => (
                "file",
                files.iter().map(|f| f.bytes_completed).max().unwrap_or(0),
            ),
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed, ..
            }) => ("aggregate", *bytes_completed),
            ProgressEvent::Download(DownloadEvent::Complete) => ("complete", SIZE),
            _ => ("lifecycle", 0),
        };
        self.bytes.fetch_max(bytes, Ordering::Relaxed);
        if count < 3 || count.is_multiple_of(20) || (bytes > 0 && count.is_multiple_of(5)) {
            eprintln!(
                "{:.2}s raw {kind}: {bytes} bytes",
                self.start.elapsed().as_secs_f64()
            );
        }
    }
}
// Summarize failures without printing tokens, signed URLs or request headers.
struct DiagnosticWarnings;
impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for DiagnosticWarnings {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        struct Failure {
            tls_eof: bool,
            timeout: bool,
        }
        impl tracing::field::Visit for Failure {
            fn record_debug(&mut self, _: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                let text = format!("{value:?}").to_lowercase();
                self.tls_eof |= text.contains("tls handshake eof");
                self.timeout |= text.contains("timed out") || text.contains("timeout");
            }
        }
        let mut failure = Failure {
            tls_eof: false,
            timeout: false,
        };
        event.record(&mut failure);
        eprintln!(
            "transport diagnostic: target={} tls_handshake_eof={} timeout={}",
            event.metadata().target(),
            failure.tls_eof,
            failure.timeout
        );
    }
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};
    let _ = tracing_subscriber::registry()
        .with(DiagnosticWarnings.with_filter(tracing_subscriber::filter::LevelFilter::WARN))
        .try_init();
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 3,
        "raw|bridge CACHE_DIR [ENDPOINT] [CONNECTIONS] [SECONDS] [BYTE_BUDGET]"
    );
    let cache_dir = std::path::PathBuf::from(&args[2]);
    let endpoint = args.get(3).cloned();
    let start = Instant::now();
    let bytes = Arc::new(AtomicU64::new(0));
    let observed = bytes.clone();
    let seconds: u64 = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(120)
        .clamp(5, 120);
    let budget: u64 = args
        .get(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64 * 1024 * 1024);
    let connections: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let timeout = std::time::Duration::from_secs(seconds);
    let operation = async {
        if args[1] == "raw" {
            let mut builder = hf_hub::HFClient::builder().cache_dir(cache_dir);
            if let Some(endpoint) = endpoint {
                builder = builder.endpoint(endpoint);
            }
            let client = builder.build()?;
            let result = client
                .model("OpenMOSS-Team", "MOSS-TTS-Realtime")
                .download_file()
                .filename(FILE)
                .revision(REVISION)
                .progress(hf_hub::progress::Progress::new(Raw {
                    start,
                    events: AtomicU64::new(0),
                    bytes,
                }))
                .send()
                .await;
            match result {
                Ok(_) => eprintln!("raw completed"),
                Err(error) => {
                    eprintln!(
                        "raw failed: {}",
                        format!("{error:#}")
                            .split('?')
                            .next()
                            .unwrap_or("download error")
                    );
                    anyhow::bail!("download failed; see sanitized diagnostic above");
                }
            }
        } else {
            let client = Downloads::with_options(
                &DownloadConfig {
                    cache_dir,
                    endpoint,
                    token: None,
                },
                DownloadOptions { connections },
            )?;
            let target = tempfile::tempdir()?;
            let (tx, mut rx) = tokio::sync::mpsc::channel(64);
            let destination = target.path().join(FILE);
            let download = client.ensure(
                &destination,
                REPO,
                REVISION,
                FILE,
                FileProgressBridge::new(tx, FILE.into(), SIZE),
            );
            let sink = async {
                while let Some(event) = rx.recv().await {
                    if let Event::ModelProgress {
                        downloaded, total, ..
                    } = event
                    {
                        bytes.fetch_max(downloaded, Ordering::Relaxed);
                        eprintln!(
                            "{:.2}s bridge {downloaded}/{total}",
                            start.elapsed().as_secs_f64()
                        );
                    }
                }
            };
            let (result, ()) = tokio::join!(download, sink);
            match result {
                Ok(()) => eprintln!("bridge completed"),
                Err(error) => {
                    eprintln!(
                        "bridge failed: {}",
                        format!("{error:#}")
                            .split('?')
                            .next()
                            .unwrap_or("download error")
                    );
                    anyhow::bail!("download failed; see sanitized diagnostic above");
                }
            }
        }
        Ok::<_, anyhow::Error>(())
    };
    tokio::select! {
        result = operation => result?,
        () = async {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if observed.load(Ordering::Relaxed) >= budget { break; }
            }
        } => eprintln!("stopped at progress budget; observed {} bytes", observed.load(Ordering::Relaxed)),
        _ = tokio::time::sleep(timeout) => eprintln!("stopped after {seconds}s; observed {} bytes", observed.load(Ordering::Relaxed)),
    }
    Ok(())
}
