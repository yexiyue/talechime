//! Explicit real-model ownership probe; ordinary tests never execute this.
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use talechime_backends::kokoro::{self, KokoroBackend};
use talechime_core::{
    backend::{AudioChunk, Backend, SegmentRequest},
    params::GenerationParams,
};
use talechime_protocol::Device;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: kokoro_lifecycle RESOURCE_ROOT"))?;
    let directory = kokoro::directory(&root);
    let (progress, _events) = tokio::sync::mpsc::channel(1);
    let started = Instant::now();
    let cancelled = tokio::time::timeout(
        Duration::from_millis(1),
        kokoro::resources::prepare(&directory, progress),
    )
    .await;
    anyhow::ensure!(
        cancelled.is_err(),
        "expected cancellation while verifying real resources"
    );
    println!(
        "preparation cancellation returned after verification join: {:?}",
        started.elapsed()
    );
    let started = Instant::now();
    let cancelled = tokio::time::timeout(
        Duration::from_millis(1),
        KokoroBackend::load_on(directory.clone(), Device::Cpu),
    )
    .await;
    anyhow::ensure!(
        cancelled.is_err(),
        "expected cancellation while loading real model"
    );
    println!(
        "load cancellation returned after owned thread join: {:?}",
        started.elapsed()
    );
    let backend = KokoroBackend::load_on(directory, Device::Cpu).await?;
    let params = GenerationParams::new();
    let mut receiver = backend
        .stream(SegmentRequest {
            text: "春天来了，院子里的花开了。",
            voice: "zf_001",
            style: None,
            seed: 7,
            params: &params,
        })
        .await?;
    tokio::time::timeout(Duration::from_secs(30), async {
        while receiver.is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;
    let started = Instant::now();
    drop(backend);
    println!(
        "drop with a live, full PCM receiver joined: {:?}",
        started.elapsed()
    );
    let mut pcm = 0;
    while let Some(value) = receiver.recv().await {
        match value? {
            AudioChunk::Pcm(_) => pcm += 1,
            AudioChunk::End => anyhow::bail!("cancelled owner published successful End"),
        }
    }
    anyhow::ensure!(pcm == 1, "expected exactly one buffered complete segment");
    println!("no false End after owner cancellation");
    Ok(())
}
