use talechime_cloud_tts::fish::{
    ApiKey, AudioFormat, Client, CreateVoice, Error, Model, ReferenceAudio, SpeechRequest,
    StreamState, Voice, VoiceQuery, VoiceScope,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
};

async fn server(response: &'static [u8]) -> (String, oneshot::Receiver<Vec<u8>>) {
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
        socket.write_all(response).await.unwrap();
        let _ = tx.send(request);
    });
    (format!("http://{address}/"), rx)
}

fn client(endpoint: &str) -> Client {
    Client::with_endpoint(ApiKey::new("test-secret").unwrap(), endpoint).unwrap()
}

fn request(voice: Voice) -> SpeechRequest {
    SpeechRequest {
        model: Model::S21ProFree,
        text: "hello".into(),
        voice,
        format: AudioFormat::Mp3,
        sample_rate: None,
        max_audio_bytes: 1024,
    }
}

#[tokio::test]
async fn synthesis_sends_explicit_model_and_streams() {
    let (endpoint, captured) = server(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc").await;
    let mut stream = client(&endpoint)
        .synthesize(&request(Voice::RemoteId("voice".into())))
        .await
        .unwrap();
    assert_eq!(stream.next_chunk().await.unwrap().unwrap(), "abc");
    assert!(stream.next_chunk().await.unwrap().is_none());
    assert_eq!(stream.state(), StreamState::Completed);
    let sent = String::from_utf8(captured.await.unwrap()).unwrap();
    assert!(sent.to_lowercase().contains("model: s2.1-pro-free"));
    assert!(sent.contains("\"reference_id\":\"voice\""));
    assert!(sent.contains("\"format\":\"mp3\""));
}

#[tokio::test]
async fn inline_reference_is_messagepack_with_binary_audio() {
    let (endpoint, captured) = server(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\na").await;
    let voice = Voice::References(vec![ReferenceAudio {
        audio: vec![0, 255, 42],
        transcript: "sample".into(),
    }]);
    let _stream = client(&endpoint).synthesize(&request(voice)).await.unwrap();
    let sent = captured.await.unwrap();
    let end = sent.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let headers = String::from_utf8_lossy(&sent[..end]).to_lowercase();
    assert!(headers.starts_with("post /v1/tts "));
    assert!(headers.contains("content-type: application/msgpack"));
    #[derive(serde::Deserialize)]
    struct Body {
        references: Vec<Reference>,
    }
    #[derive(serde::Deserialize)]
    struct Reference {
        #[serde(with = "serde_bytes")]
        audio: Vec<u8>,
        text: String,
    }
    let body: Body = rmp_serde::from_slice(&sent[end + 4..]).unwrap();
    assert_eq!(body.references[0].audio, vec![0, 255, 42]);
    assert_eq!(body.references[0].text, "sample");
}

#[tokio::test]
async fn interrupted_audio_reports_transport_failure() {
    let (endpoint, _) = server(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\na").await;
    let mut stream = client(&endpoint)
        .synthesize(&request(Voice::RemoteId("v".into())))
        .await
        .unwrap();
    let error = loop {
        match stream.next_chunk().await {
            Ok(Some(_)) => continue,
            Ok(None) => panic!("interrupted audio completed successfully"),
            Err(error) => break error,
        }
    };
    assert!(matches!(error, Error::Transport));
    assert_eq!(stream.state(), StreamState::Failed);
}

#[tokio::test]
async fn empty_truncated_oversized_and_cancelled_streams_are_not_success() {
    for (response, limit) in [
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n".as_slice(),
            2,
        ),
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\na".as_slice(),
            20,
        ),
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc".as_slice(),
            2,
        ),
    ] {
        let (endpoint, _) = server(response).await;
        let mut input = request(Voice::RemoteId("v".into()));
        input.max_audio_bytes = limit;
        let mut stream = client(&endpoint).synthesize(&input).await.unwrap();
        loop {
            if stream.next_chunk().await.is_err() {
                break;
            }
        }
        assert_eq!(stream.state(), StreamState::Failed);
    }
    let (endpoint, _) = server(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc").await;
    let mut stream = client(&endpoint)
        .synthesize(&request(Voice::RemoteId("v".into())))
        .await
        .unwrap();
    stream.cancel();
    assert!(matches!(stream.next_chunk().await, Err(Error::Cancelled)));
    assert_eq!(stream.state(), StreamState::Cancelled);
}

#[tokio::test]
async fn voice_search_and_private_creation_send_expected_fields() {
    let (endpoint, captured) = server(b"HTTP/1.1 200 OK\r\nContent-Length: 45\r\n\r\n{\"total\":1,\"items\":[{\"_id\":\"a\",\"title\":\"A\"}]}").await;
    let q = VoiceQuery {
        scope: VoiceScope::Own,
        page_number: 2,
        page_size: 20,
        title: Some("narration".into()),
        tag: Some("soft".into()),
        language: Some("zh".into()),
    };
    let page = client(&endpoint).search_voices(&q).await.unwrap();
    assert_eq!(page.items[0].id, "a");
    let sent = String::from_utf8(captured.await.unwrap()).unwrap();
    assert!(
        sent.contains("self=true") && sent.contains("language=zh") && sent.contains("tag=soft")
    );
    let (endpoint, captured) = server(b"HTTP/1.1 200 OK\r\nContent-Length: 41\r\n\r\n{\"_id\":\"v\",\"title\":\"A\",\"state\":\"created\"}").await;
    let record = client(&endpoint)
        .create_voice(CreateVoice {
            title: "A".into(),
            samples: vec![ReferenceAudio {
                audio: b"sample".to_vec(),
                transcript: "words".into(),
            }],
        })
        .await
        .unwrap();
    assert_eq!(record.state.as_deref(), Some("created"));
    let sent = String::from_utf8(captured.await.unwrap()).unwrap();
    assert!(
        sent.contains("name=\"visibility\"") && sent.contains("private") && sent.contains("sample")
    );
}

#[tokio::test]
async fn errors_do_not_expose_provider_bodies_and_mutation_outcome_is_uncertain() {
    for (response, expected) in [
        (
            b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 11\r\n\r\ntest-secret".as_slice(),
            "Fish authentication failed",
        ),
        (
            b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 11\r\n\r\ntest-secret".as_slice(),
            "Fish rate or concurrency limit reached",
        ),
        (
            b"HTTP/1.1 500 Error\r\nContent-Length: 11\r\n\r\ntest-secret".as_slice(),
            "request outcome is unknown",
        ),
    ] {
        let (endpoint, _) = server(response).await;
        let result = client(&endpoint)
            .synthesize(&request(Voice::RemoteId("v".into())))
            .await;
        let error = match result {
            Ok(_) => panic!("unexpected success"),
            Err(error) => error,
        };
        assert_eq!(error.to_string(), expected);
        assert!(!format!("{error:?}").contains("test-secret"));
    }
}

#[tokio::test]
async fn create_handles_multiple_samples_and_preserves_unknown_state() {
    let (endpoint, captured) = server(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"_id\":\"v\",\"title\":\"A\",\"state\":\"future-state\"}").await;
    let voice = CreateVoice {
        title: "A".into(),
        samples: vec![
            ReferenceAudio {
                audio: b"audio-one".to_vec(),
                transcript: "first".into(),
            },
            ReferenceAudio {
                audio: b"audio-two".to_vec(),
                transcript: "second".into(),
            },
        ],
    };
    let record = client(&endpoint).create_voice(voice).await.unwrap();
    assert_eq!(record.state.as_deref(), Some("future-state"));
    let sent = String::from_utf8(captured.await.unwrap()).unwrap();
    assert_eq!(sent.matches("name=\"voices\"").count(), 2);
    assert_eq!(sent.matches("name=\"texts\"").count(), 2);
    assert!(sent.contains("audio-one") && sent.contains("audio-two"));
}

#[tokio::test]
async fn dropping_a_slow_stream_closes_the_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/", listener.local_addr().unwrap());
    let (sent, ready) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = [0; 4096];
        let _ = socket.read(&mut buffer).await.unwrap();
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\n\r\na")
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
    let mut stream = client(&endpoint)
        .synthesize(&request(Voice::RemoteId("v".into())))
        .await
        .unwrap();
    ready.await.unwrap();
    assert_eq!(stream.next_chunk().await.unwrap().unwrap(), "a");
    drop(stream);
    tokio::time::timeout(std::time::Duration::from_secs(3), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn read_only_voice_query_distinguishes_http_errors_and_preserves_ids() {
    let (endpoint, captured) = server(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"_id\":\"a/b\",\"title\":\"A\",\"state\":\"mystery\"}").await;
    let voice = client(&endpoint).get_voice("a/b").await.unwrap();
    assert_eq!(voice.state.as_deref(), Some("mystery"));
    assert!(
        String::from_utf8(captured.await.unwrap())
            .unwrap()
            .starts_with("GET /model/a%2Fb ")
    );
    let (endpoint, _) = server(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n").await;
    assert!(matches!(
        client(&endpoint).get_voice("gone").await,
        Err(Error::Http(404))
    ));
}

#[tokio::test]
async fn creation_disconnect_reports_unknown_without_retrying() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = [0; 4096];
        let _ = socket.read(&mut buffer).await.unwrap();
        drop(socket);
        tokio::time::timeout(std::time::Duration::from_millis(300), listener.accept())
            .await
            .is_err()
    });
    let result = client(&endpoint)
        .create_voice(CreateVoice {
            title: "A".into(),
            samples: vec![ReferenceAudio {
                audio: vec![1],
                transcript: "sample".into(),
            }],
        })
        .await;
    assert!(matches!(result, Err(Error::OutcomeUnknown)));
    assert!(server.await.unwrap());
}

#[tokio::test]
async fn public_voice_search_retains_pagination_metadata() {
    let (endpoint, captured) = server(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"total\":2,\"has_more\":true,\"total_is_exact\":false,\"items\":[{\"_id\":\"v\",\"title\":\"A\",\"state\":\"trained\",\"language\":\"zh\"}]}").await;
    let q = VoiceQuery {
        scope: VoiceScope::Public,
        page_number: 1,
        page_size: 1,
        title: None,
        tag: None,
        language: Some("zh".into()),
    };
    let page = client(&endpoint).search_voices(&q).await.unwrap();
    assert_eq!(page.has_more, Some(true));
    assert_eq!(page.metadata["total_is_exact"], false);
    assert_eq!(page.items[0].metadata["language"], "zh");
    let sent = String::from_utf8(captured.await.unwrap()).unwrap();
    assert!(sent.contains("page_size=1") && sent.contains("page_number=1"));
    assert!(!sent.contains("self=true"));
    let (endpoint, _) = server(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n").await;
    assert!(matches!(
        client(&endpoint).get_voice("private").await,
        Err(Error::Forbidden)
    ));
}

#[tokio::test]
async fn error_response_body_is_never_read_or_exposed() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut data = [0; 4096];
        let _ = socket.read(&mut data).await.unwrap();
        socket
            .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 999999999\r\n\r\nsecret-body")
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(3), socket.read(&mut data)).await
    });
    let result = client(&endpoint)
        .synthesize(&request(Voice::RemoteId("v".into())))
        .await;
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("unexpected success"),
    };
    assert!(matches!(error, Error::Http(400)));
    assert!(!format!("{error:?}").contains("secret-body"));
    assert!(matches!(task.await.unwrap(), Ok(Ok(0)) | Ok(Err(_))));
}

#[tokio::test]
async fn clone_failed_state_is_reported_without_guessing() {
    let (endpoint, _) = server(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"_id\":\"v\",\"title\":\"A\",\"state\":\"failed\"}").await;
    let result = client(&endpoint)
        .create_voice(CreateVoice {
            title: "A".into(),
            samples: vec![ReferenceAudio {
                audio: vec![1],
                transcript: "text".into(),
            }],
        })
        .await
        .unwrap();
    assert_eq!(result.state.as_deref(), Some("failed"));
}
