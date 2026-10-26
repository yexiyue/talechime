//! Explicit real-model probe through the Talechime Engine; never runs in tests.
use talechime::*;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 4,
        "usage: cosyvoice3_backend RESOURCE_ROOT VOICE cpu|metal|cuda"
    );
    run_local(async {
        let mut options = ModelOptions::new("cosyvoice3", &args[1]);
        options.device = match args[3].as_str() {
            "cpu" => Device::Cpu,
            "metal" => Device::Metal,
            "cuda" => Device::Cuda,
            _ => anyhow::bail!("unsupported device"),
        };
        let mut engine = Engine::prepare(options, |_| {}).await?;
        let text = "你好，欢迎使用叙铃。";
        let mut first = None;
        for run in 0..2 {
            let mut stream = engine
                .synthesize_with_options(
                    text,
                    &args[2],
                    None,
                    SynthesisOptions {
                        seed: SeedPolicy::Pinned(42),
                        ..Default::default()
                    },
                )
                .await?;
            let mut samples = Vec::new();
            let mut blocks = 0;
            while let Some(block) = stream.recv().await {
                let block = block?;
                anyhow::ensure!(block.pcm().sample_rate == 24000 && block.pcm().channels == 1);
                samples.extend_from_slice(&block.pcm().samples);
                blocks += 1;
            }
            anyhow::ensure!(stream.state() == SynthesisState::Completed && !samples.is_empty());
            println!(
                "run {run}: {blocks} blocks, {} samples, completed",
                samples.len()
            );
            if let Some(first) = &first {
                anyhow::ensure!(first == &samples, "same-seed requests differ");
            } else {
                first = Some(samples);
            }
        }
        let mut stream = engine
            .synthesize(
                "今天天气真不错，我们一起去公园散步吧。顺便买点水果，晚上回家做沙拉。",
                &args[2],
                None,
            )
            .await?;
        anyhow::ensure!(stream.recv().await.transpose()?.is_some());
        stream.cancel().await;
        drop(stream);
        println!("cancel after first block: owner can be reused");
        let pcm = engine
            .synthesize_pcm(text, &args[2], None, 8 * 1024 * 1024)
            .await?;
        anyhow::ensure!(!pcm.samples.is_empty());
        engine.close().await?;
        println!("post-cancel request completed; engine closed");
        Ok(())
    })
    .await
}
