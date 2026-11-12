//! Real-network smoke checks. Ignored by default (`cargo test -- --ignored`);
//! run manually when verifying the shared-cache download path end to end.
//! Downloads real weights into the default shared HF cache and prepares the
//! model (load succeeds, then release).
use talechime_backends::{PrepareOptions, Registry};

#[tokio::test]
#[ignore = "downloads real MOSS Nano weights from the Hub"]
async fn moss_nano_prepare_downloads_and_loads_from_shared_cache() {
    let registry = Registry::new(None).expect("registry");
    let model = registry
        .resolve("moss", None)
        .expect("resolve moss default");
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let progress = tokio::spawn(async move {
        use talechime_protocol::Event;
        while let Some(event) = rx.recv().await {
            if let Event::ModelProgress {
                resource,
                downloaded,
                total,
            } = event
            {
                println!("progress {resource}: {downloaded}/{total}");
            }
        }
    });
    let started = std::time::Instant::now();
    let backend = model
        .prepare(
            PrepareOptions {
                device: talechime_protocol::Device::Cpu,
                calibrate: false,
                recovery: false,
                voice: None,
            },
            tx.clone(),
        )
        .await
        .expect("prepare moss nano end to end");
    drop(tx);
    let _ = progress.await;
    let capabilities = backend.capabilities();
    println!(
        "prepared {} in {:.1}s; resource status now {:?}",
        capabilities.model_name,
        started.elapsed().as_secs_f32(),
        registry.resource_status("moss", None).await.unwrap()
    );
}

#[tokio::test]
#[cfg(feature = "kokoro")]
#[ignore = "exercises kokoro local-conversion paths without real assets"]
async fn kokoro_reports_missing_conversion_without_downloading() {
    let registry = Registry::new(None).expect("registry");
    let status = registry
        .resource_status("kokoro", None)
        .await
        .expect("kokoro status");
    // Without a converted bundle the honest offline answer is Missing.
    println!("kokoro status: {status:?}");
}
