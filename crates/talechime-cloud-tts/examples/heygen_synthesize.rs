//! Set HEYGEN_API_KEY and HEYGEN_VOICE_ID, then supply an output path.
use std::{env, fs::File, io::Write};
use talechime_cloud_tts::heygen::{ApiKey, AudioFormat, Client, Model, SpeechRequest};

async fn generate(
    client: &Client,
    voice_id: String,
    mut output: impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let request = SpeechRequest {
        model: Model::TurboFree,
        text: "你好，欢迎使用语音合成。".into(),
        voice_id,
        format: AudioFormat::Mp3,
        language: Some("zh".into()),
        max_audio_bytes: 16 * 1024 * 1024,
    };
    let mut audio = client.synthesize(&request).await?;
    while let Some(bytes) = audio.next_chunk().await? {
        output.write_all(&bytes)?;
    }
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = ApiKey::new(env::var("HEYGEN_API_KEY")?)?;
    let voice = env::var("HEYGEN_VOICE_ID")?;
    let output = env::args_os().nth(1).ok_or("expected output path")?;
    generate(&Client::new(key)?, voice, File::create(output)?).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    #[tokio::test]
    async fn example_collects_mock_audio() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut data = [0; 8192];
            let _ = socket.read(&mut data).await.unwrap();
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: audio/mpeg\r\nX-Audio-Sample-Rate: 44100\r\nX-Audio-Channels: 1\r\nContent-Length: 3\r\n\r\nabc")
                .await
                .unwrap();
        });
        let client = Client::with_endpoint(ApiKey::new("test").unwrap(), &endpoint).unwrap();
        let mut output = Vec::new();
        generate(&client, "voice".into(), &mut output)
            .await
            .unwrap();
        assert_eq!(output, b"abc");
        server.await.unwrap();
    }
}
