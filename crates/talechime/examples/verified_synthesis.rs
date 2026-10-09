//! Explicit real TTS + native readback gate; no player/checkpoints. May download models.
use talechime::*;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: verified_synthesis BACKEND TTS_RESOURCE_DIR ASR_RESOURCE_DIR".into());
    }
    run_local(async {
        let verifier = prepare_readback(ReadbackModelOptions::new(&args[3]), |_| {}).await?;
        let mut engine = Engine::prepare(ModelOptions::new(&args[1], &args[2]), |_| {}).await?;
        engine.set_verifier(verifier)?;
        let voice = engine.capabilities()?.default_voice;
        let mut stream = engine.synthesize_verified(
            "你好，世界。",
            &voice,
            None,
            VerificationOptions {
                policy: VerificationPolicy::Gate {
                    max_retries: 1,
                    strict_suspect: false,
                },
                ..Default::default()
            },
        )?;
        let mut samples = 0;
        let mut reports = 0;
        while let Some(item) = stream.next().await {
            match item? {
                SynthesisItem::Verification(report) => {
                    reports += 1;
                    println!("{:?}: {:?}", report.verdict, report.evidence);
                }
                SynthesisItem::Audio(audio) => samples += audio.pcm().samples.len(),
            }
        }
        if stream.state() != SynthesisState::Completed || samples == 0 || reports == 0 {
            return Err("verified synthesis did not complete".into());
        }
        engine.close().await?;
        println!("completed: {samples} samples, {reports} reports, no playback");
        Ok::<_, Box<dyn std::error::Error>>(())
    })
    .await
}
