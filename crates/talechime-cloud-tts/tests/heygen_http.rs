use talechime_cloud_tts::heygen::{
    ApiKey, AudioFormat, Client, Error, Model, SpeechRequest, StreamState, VoiceQuery, VoiceScope,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
};
async fn server(response: Vec<u8>) -> (String, oneshot::Receiver<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let n = socket.read(&mut buffer).await.unwrap();
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..n]);
            if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                if request.len() >= end + 4 + length {
                    break;
                }
            }
        }
        socket.write_all(&response).await.unwrap();
        let _ = tx.send(request);
    });
    (format!("http://{address}/"), rx)
}

fn client(endpoint: &str) -> Client {
    Client::with_endpoint(ApiKey::new("test-secret").unwrap(), endpoint).unwrap()
}

fn request() -> SpeechRequest {
    SpeechRequest {
        model: Model::TurboFree,
        text: "你好".into(),
        voice_id: "v".into(),
        format: AudioFormat::Pcm16,
        language: Some("zh".into()),
        max_audio_bytes: 8,
    }
}
fn query() -> VoiceQuery {
    VoiceQuery {
        scope: VoiceScope::Public,
        limit: 2,
        token: None,
        language: Some("zh".into()),
        gender: Some("female".into()),
        age: Some("young".into()),
        query: Some("warm narrator".into()),
    }
}
fn audio(headers: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 200 OK\r\nConnection: close\r\n{headers}\r\n").into_bytes();
    out.extend_from_slice(body);
    out
}
const PCM: &str = "Content-Type: audio/pcm; charset=binary\r\nX-Audio-Sample-Rate: 24000\r\nX-Audio-Channels: 1\r\n";
fn json(body: &str) -> Vec<u8> {
    audio("Content-Type: application/json\r\n", body.as_bytes())
}
async fn drain(stream: &mut talechime_cloud_tts::heygen::AudioStream) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next_chunk().await? {
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}
#[tokio::test]
async fn explicit_profiles_and_boundary_text() {
    for format in [AudioFormat::Pcm16, AudioFormat::Mp3] {
        let headers = if format == AudioFormat::Pcm16 {
            PCM
        } else {
            "Content-Type: audio/mpeg\r\nX-Audio-Sample-Rate: 44100\r\nX-Audio-Channels: 1\r\n"
        };
        let (endpoint, captured) = server(audio(headers, b"abcd")).await;
        let mut input = request();
        input.format = format;
        input.text = "中".repeat(5000);
        input.max_audio_bytes = 4;
        let mut stream = client(&endpoint).synthesize(&input).await.unwrap();
        assert_eq!(stream.format, format);
        assert_eq!(stream.metadata.channels, 1);
        assert_eq!(stream.metadata.sample_rate, format.sample_rate());
        assert_eq!(drain(&mut stream).await.unwrap(), b"abcd");
        assert_eq!(stream.state(), StreamState::Completed);
        stream.cancel();
        assert!(stream.next_chunk().await.unwrap().is_none());
        let sent = captured.await.unwrap();
        let end = sent.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let headers = String::from_utf8_lossy(&sent[..end]).to_lowercase();
        assert!(headers.starts_with("post /v3/models/audio/tts/stream "));
        assert!(headers.contains("x-api-key: test-secret"));
        assert!(!headers.contains("authorization:"));
        let body: serde_json::Value = serde_json::from_slice(&sent[end + 4..]).unwrap();
        assert_eq!(body["model"], "heygen-voice-1-turbo-free");
        assert_eq!(body["with_timestamps"], false);
        assert_eq!(body["language"], "zh");
        assert_eq!(body["text"].as_str().unwrap().chars().count(), 5000);
        assert_eq!(
            body["output_format"]["sample_rate_hertz"],
            format.sample_rate()
        );
        if format == AudioFormat::Mp3 {
            assert_eq!(body["output_format"]["encoding"], "mp3");
            assert_eq!(body["output_format"]["bitrate_bps"], 128000);
        } else {
            assert_eq!(body["output_format"]["encoding"], "pcm");
            assert!(body["output_format"].get("bitrate_bps").is_none());
        }
    }
}
#[tokio::test]
async fn invalid_inputs_never_connect() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let c = client(&format!("http://{}/", listener.local_addr().unwrap()));
    for case in 0..5 {
        let mut input = request();
        match case {
            0 => input.text = "中".repeat(5001),
            1 => input.text = " ".into(),
            2 => input.voice_id = " ".into(),
            3 => input.max_audio_bytes = 0,
            _ => input.language = Some("zh-CN".into()),
        };
        assert!(matches!(
            c.synthesize(&input).await,
            Err(Error::InvalidInput(_))
        ));
    }
    for case in 0..6 {
        let mut q = query();
        match case {
            0 => q.limit = 0,
            1 => q.limit = 51,
            2 => q.scope = VoiceScope::Private,
            3 => q.token = Some("".into()),
            4 => q.gender = Some("invalid".into()),
            _ => q.query = Some("x".repeat(257)),
        };
        assert!(matches!(
            c.search_voices(&q).await,
            Err(Error::InvalidInput(_))
        ));
    }
    for id in ["", ".", "..", " "] {
        assert!(matches!(c.get_voice(id).await, Err(Error::InvalidInput(_))));
    }
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(50), listener.accept())
            .await
            .is_err()
    );
    for url in [
        "http://example.com/",
        "https://user:pass@example.com/",
        "https://example.com/?secret=x",
        "https://example.com/#x",
        "https://example.com/path",
    ] {
        assert!(Client::with_endpoint(ApiKey::new("key").unwrap(), url).is_err());
    }
}
#[tokio::test]
async fn mismatched_and_missing_headers_are_rejected() {
    for headers in [
        "",
        "Content-Type: application/json\r\n",
        "Content-Type: text/event-stream\r\n",
    ]
    .into_iter()
    .map(String::from)
    .chain([
        PCM.replace("24000", "44100"),
        format!("{PCM}X-Audio-Sample-Rate: 44100\r\n"),
        format!("{PCM}X-Audio-Channels: 2\r\n"),
        format!("{PCM}Content-Type: application/json\r\n"),
        PCM.replace("Channels: 1", "Channels: 2"),
        PCM.replace("24000", "invalid"),
        PCM.replace("X-Audio-Channels: 1\r\n", ""),
    ]) {
        let (endpoint, _) = server(audio(&headers, b"secret")).await;
        assert!(matches!(
            client(&endpoint).synthesize(&request()).await,
            Err(Error::InvalidResponse)
        ));
    }
}
#[tokio::test]
async fn empty_oversized_truncated_and_cancelled_are_terminal() {
    for (body, extra, limit, transport) in [
        (b"".as_slice(), "", 8, false),
        (b"abcd".as_slice(), "", 3, false),
        (b"a".as_slice(), "Content-Length: 10\r\n", 8, true),
    ] {
        let (endpoint, _) = server(audio(&format!("{PCM}{extra}"), body)).await;
        let mut input = request();
        input.max_audio_bytes = limit;
        let mut stream = client(&endpoint).synthesize(&input).await.unwrap();
        let err = drain(&mut stream).await.unwrap_err();
        assert_eq!(matches!(err, Error::Transport), transport);
        assert_eq!(stream.state(), StreamState::Failed);
        stream.cancel();
        assert_eq!(stream.state(), StreamState::Failed);
        assert!(stream.next_chunk().await.is_err());
    }
    let (endpoint, _) = server(audio(PCM, b"abc")).await;
    let mut stream = client(&endpoint).synthesize(&request()).await.unwrap();
    stream.cancel();
    assert_eq!(stream.state(), StreamState::Cancelled);
    assert!(matches!(stream.next_chunk().await, Err(Error::Cancelled)));
}
#[tokio::test]
async fn service_errors_are_safe_and_do_not_follow_redirects() {
    let target = TcpListener::bind("127.0.0.1:0").await.unwrap();
    for code in [301, 401, 403, 402, 404, 409, 429, 500, 503] {
        let response=format!("HTTP/1.1 {code} Error\r\nLocation: http://{}/\r\nContent-Length: 999999999\r\nConnection: close\r\n\r\ntest-secret",target.local_addr().unwrap()).into_bytes();
        let (endpoint, _) = server(response).await;
        let err = client(&endpoint)
            .synthesize(&request())
            .await
            .err()
            .unwrap();
        match code {
            401 => assert!(matches!(err, Error::Authentication)),
            403 => assert!(matches!(err, Error::Forbidden)),
            429 => assert!(matches!(err, Error::RateLimited)),
            500 | 503 => assert!(matches!(err, Error::OutcomeUnknown)),
            _ => assert!(matches!(err,Error::Http(n) if n==code)),
        };
        assert!(!format!("{err:?}: {err}").contains("test-secret"));
    }
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(50), target.accept())
            .await
            .is_err()
    );
}
#[tokio::test]
async fn disconnect_does_not_retry() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut data = [0; 8192];
        let _ = socket.read(&mut data).await.unwrap();
        drop(socket);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), listener.accept())
                .await
                .is_err()
        );
    });
    assert!(matches!(
        client(&endpoint).synthesize(&request()).await,
        Err(Error::OutcomeUnknown)
    ));
    task.await.unwrap();
}
#[tokio::test]
async fn directory_pages_and_detail_preserve_provider_state() {
    for token in [None, Some("next & token".to_string())] {
        let (endpoint,captured)=server(json(r#"{"data":[{"voice_id":"v","name":"Name","mode":"future","status":"UNKNOWN","language":"zh","custom":42}],"has_more":true,"next_token":"next & token","extra":true}"#)).await;
        let mut q = query();
        q.token = token.clone();
        let page = client(&endpoint).search_voices(&q).await.unwrap();
        assert_eq!(page.data[0].status, "UNKNOWN");
        assert_eq!(page.data[0].metadata["custom"], 42);
        assert_eq!(page.metadata["extra"], true);
        assert!(page.has_more);
        assert_eq!(page.next_token.as_deref(), Some("next & token"));
        let sent = String::from_utf8(captured.await.unwrap()).unwrap();
        for field in [
            "type=public",
            "limit=2",
            "language=zh",
            "gender=female",
            "age=young",
            "query=warm+narrator",
        ] {
            assert!(sent.contains(field), "{sent}");
        }
        if token.is_some() {
            assert!(sent.contains("token=next+%26+token"));
        }
    }
    let (endpoint, captured) =
        server(json(r#"{"data":[],"has_more":false,"next_token":null}"#)).await;
    let q = VoiceQuery {
        scope: VoiceScope::Private,
        limit: 1,
        token: None,
        language: None,
        gender: None,
        age: None,
        query: None,
    };
    assert!(!client(&endpoint).search_voices(&q).await.unwrap().has_more);
    assert!(
        String::from_utf8(captured.await.unwrap())
            .unwrap()
            .contains("type=private")
    );
    for status in ["PENDING", "FAILED"] {
        let (endpoint,captured)=server(json(&format!(r#"{{"data":{{"voice_id":"a/b","name":"N","mode":"instant","status":"{status}","failure_reason":"test"}}}}"#))).await;
        let voice = client(&endpoint).get_voice("a/b").await.unwrap();
        assert_eq!(voice.status, status);
        assert_eq!(voice.metadata["failure_reason"], "test");
        assert!(
            String::from_utf8(captured.await.unwrap())
                .unwrap()
                .starts_with("GET /v3/models/audio/voices/a%2Fb ")
        );
    }
    for code in [403, 404] {
        let (endpoint, _) =
            server(format!("HTTP/1.1 {code} Error\r\nContent-Length: 0\r\n\r\n").into_bytes())
                .await;
        let err = client(&endpoint).get_voice("gone").await.unwrap_err();
        assert!(matches!(err, Error::Forbidden | Error::Http(404)));
    }
}
#[tokio::test]
async fn catalog_json_is_bounded_and_validated() {
    for body in ["invalid".to_string(), " ".repeat(4 * 1024 * 1024 + 1)] {
        let (endpoint, _) = server(json(&body)).await;
        assert!(matches!(
            client(&endpoint).search_voices(&query()).await,
            Err(Error::InvalidResponse)
        ));
    }
}
#[tokio::test]
async fn cancel_or_drop_closes_a_slow_stream() {
    for cancel in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/", listener.local_addr().unwrap());
        let (sent, ready) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 4096];
            let _ = socket.read(&mut buffer).await.unwrap();
            socket
                .write_all(
                    format!("HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\n{PCM}\r\na").as_bytes(),
                )
                .await
                .unwrap();
            let _ = sent.send(());
            loop {
                match socket.read(&mut buffer).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => continue,
                }
            }
        });
        let mut stream = client(&endpoint).synthesize(&request()).await.unwrap();
        ready.await.unwrap();
        assert_eq!(stream.next_chunk().await.unwrap().unwrap(), "a");
        if cancel {
            stream.cancel();
            assert_eq!(stream.state(), StreamState::Cancelled);
        }
        drop(stream);
        tokio::time::timeout(std::time::Duration::from_secs(3), server)
            .await
            .unwrap()
            .unwrap();
    }
}
