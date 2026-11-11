//! Explicit developer resource preparation; never runs inference or opens audio.
use std::path::PathBuf;
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 4 || args.len() == 7,
        "matrix_prepare ROOT BACKEND MODEL [REFERENCE.wav TRANSCRIPT VOICE_ID]"
    );
    let root = PathBuf::from(&args[1]);
    anyhow::ensure!(root.is_absolute(), "resource root must be absolute");
    if args.len() == 7 {
        let store = match args[2].as_str() {
            #[cfg(feature = "indextts")]
            "indextts" => talechime_backends::indextts::voice_store(&root)?,
            #[cfg(feature = "cosyvoice3")]
            "cosyvoice3" => talechime_backends::cosyvoice3::voice_store(&root)?,
            #[cfg(feature = "qwen")]
            "qwen" => {
                let model = talechime_backends::qwen::models::Model::parse(Some(&args[3]))?;
                talechime_backends::qwen::voice_store(&model.directory(&root), model)?
            }
            #[cfg(feature = "qwen")]
            "qwen-reference" => {
                let model = talechime_backends::qwen::models::Model::parse(Some(&args[3]))?;
                talechime_backends::qwen::voice_store(&root, model)?
            }
            #[cfg(feature = "voxcpm")]
            "voxcpm-reference" => talechime_backends::voxcpm::voice_store(&root)?,
            #[cfg(feature = "omnivoice")]
            "omnivoice" => talechime_backends::omnivoice::voice_store(
                &talechime_backends::omnivoice::directory(&root),
            )?,
            #[cfg(feature = "moss-candle")]
            "moss" => {
                let mode = talechime_backends::moss::candle::Mode::parse(&args[3])?;
                talechime_backends::moss::candle::voice_store(&mode.directory(&root), mode)?
            }
            backend => anyhow::bail!("reference import is not configured for {backend}"),
        };
        store.import(&args[6], &args[6], &PathBuf::from(&args[4]), &args[5], None)?;
        return Ok(());
    }
    let transport = match std::env::var("TALECHIME_PREPARE_TRANSPORT").as_deref() {
        Ok("http") => talechime_backends::downloads::DownloadTransport::Http,
        Ok("auto") | Err(_) => talechime_backends::downloads::DownloadTransport::Auto,
        Ok(_) => anyhow::bail!("TALECHIME_PREPARE_TRANSPORT must be auto or http"),
    };
    let registry = talechime_backends::Registry::with_download_options(
        Some(root.clone()),
        &Default::default(),
        talechime_backends::downloads::DownloadOptions {
            connections: 4,
            transport,
        },
    )?;
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let progress = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            println!("{}", serde_json::to_string(&event)?);
        }
        Ok::<_, anyhow::Error>(())
    });
    let result = match args[2].as_str() {
        #[cfg(feature = "qwen")]
        "qwen-design" => {
            let model = talechime_backends::qwen::models::Model::Design17;
            anyhow::ensure!(args[3] == model.id(), "Qwen design model mismatch");
            talechime_backends::qwen::resources::prepare_model(
                &model.directory(&root),
                model,
                registry.downloads(),
                tx,
            )
            .await
        }
        #[cfg(feature = "moss-candle")]
        "moss-design" => {
            anyhow::ensure!(args[3] == "voice-design-1.7b", "MOSS design model mismatch");
            talechime_backends::moss::candle::resources::prepare_design(
                &root,
                registry.downloads(),
                tx,
            )
            .await
        }
        backend => {
            registry
                .resolve(backend, Some(&args[3]))?
                .prepare_resources(tx)
                .await
        }
    };
    progress.await??;
    result
}
