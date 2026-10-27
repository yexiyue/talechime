//! Explicit real-model Engine probe. Never invoked by ordinary tests.
use talechime::*;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 4 || args.len() == 5,
        "usage: kokoro_backend RESOURCE_ROOT VOICE cpu|metal|cuda [MODEL]"
    );
    run_local(async {
        let mut options = ModelOptions::new("kokoro", &args[1]);
        if let Some(model) = args.get(4) {
            options.model = Some(model.clone());
        }
        options.device = match args[3].as_str() {
            "cpu" => Device::Cpu,
            "metal" => Device::Metal,
            "cuda" => Device::Cuda,
            _ => anyhow::bail!("unsupported device"),
        };
        let mut engine = Engine::prepare(options, |_| {}).await?;
        let text = "春天来了，院子里的花开了。";
        let mut baseline: Option<Vec<f32>> = None;
        for run in 0..2 {
            let mut stream = engine
                .synthesize_with_options(
                    text,
                    &args[2],
                    None,
                    SynthesisOptions {
                        seed: SeedPolicy::Pinned(7),
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
            if let Some(first) = &baseline {
                anyhow::ensure!(same_bits(first, &samples), "same-seed requests differ");
            } else {
                baseline = Some(samples);
            }
        }
        let mut stream = engine
            .synthesize(
                "今天天气真不错，我们一起去公园散步吧。顺便买点水果，晚上回家做沙拉。",
                &args[2],
                None,
            )
            .await?;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        stream.cancel().await;
        anyhow::ensure!(stream.state() == SynthesisState::Cancelled);
        drop(stream);
        println!("cancelled before first PCM; checking owner reuse");
        // Exercise a different shape and another installed voice on this owner.
        for voice in engine
            .capabilities()?
            .voices
            .iter()
            .filter(|v| *v != &args[2])
            .take(1)
        {
            let pcm = engine
                .synthesize_pcm_with_options(
                    "今天天气很好。",
                    voice,
                    None,
                    8 * 1024 * 1024,
                    SynthesisOptions {
                        seed: SeedPolicy::Pinned(7),
                        ..Default::default()
                    },
                )
                .await?;
            anyhow::ensure!(!pcm.samples.is_empty());
        }
        let pcm = engine
            .synthesize_pcm_with_options(
                text,
                &args[2],
                None,
                8 * 1024 * 1024,
                SynthesisOptions {
                    seed: SeedPolicy::Pinned(7),
                    ..Default::default()
                },
            )
            .await?;
        anyhow::ensure!(
            same_bits(baseline.as_deref().unwrap(), &pcm.samples),
            "cancel/shape/voice changes altered the restored request"
        );
        println!(
            "post-cancel request completed: {} samples",
            pcm.samples.len()
        );
        engine.close().await?;
        println!("engine closed");
        Ok(())
    })
    .await
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
