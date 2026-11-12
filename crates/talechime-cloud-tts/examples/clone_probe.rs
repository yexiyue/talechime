//! Explicit live probe. Creates a private remote voice and writes its ID locally.
use std::{env, fs, io::Write, path::PathBuf};
use talechime_cloud_tts::fish::{
    ApiKey, AudioFormat, Client, CreateVoice, Model, ReferenceAudio, SpeechRequest, Voice,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let reference = PathBuf::from(args.next().ok_or("expected reference WAV")?);
    let output = PathBuf::from(args.next().ok_or("expected output directory")?);
    fs::create_dir_all(&output)?;
    let transcript = env::var("FISH_REFERENCE_TEXT")?;
    let sample = ReferenceAudio {
        audio: fs::read(reference)?,
        transcript,
    };
    let client = Client::new(ApiKey::new(env::var("FISH_API_KEY")?)?)?;
    if env::var_os("FISH_CLONE_CREATE_ONLY").is_none() {
        let request = SpeechRequest {
            model: Model::S21ProFree,
            text: "这是即时克隆的语音测试。".into(),
            voice: Voice::References(vec![sample.clone()]),
            format: AudioFormat::Mp3,
            sample_rate: None,
            max_audio_bytes: 16 * 1024 * 1024,
        };
        let timer = std::time::Instant::now();
        let mut stream = client.synthesize(&request).await?;
        let mut audio = Vec::new();
        while let Some(chunk) = stream.next_chunk().await? {
            audio.extend_from_slice(&chunk);
        }
        fs::write(output.join("instant.mp3"), &audio)?;
        println!(
            "instant: {} bytes, {:.2}s",
            audio.len(),
            timer.elapsed().as_secs_f64()
        );
    }

    // Reserve a writable ID file before creating a voice; never overwrite a previous result.
    let mut id_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("remote-voice-id.txt"))?;
    let timer = std::time::Instant::now();
    let voice = client
        .create_voice(CreateVoice {
            title: "talechime-clone-probe".into(),
            samples: vec![sample],
        })
        .await?;
    id_file.write_all(voice.id.as_bytes())?;
    id_file.sync_all()?;
    println!(
        "create: state {:?}, {:.2}s; ID saved to output directory",
        voice.state,
        timer.elapsed().as_secs_f64()
    );
    let observed = client.get_voice(&voice.id).await?;
    println!("query: state {:?}", observed.state);
    Ok(())
}
