//! End-to-end download tests against an in-process HTTP mock standing in for
//! the Hub. Never touches the network; pinned commit revisions avoid revision
//! resolution requests.
use talechime_backends::downloads::{
    DownloadConfig, DownloadFailure, DownloadOptions, DownloadTransport, Downloads,
    FileProgressBridge, HubResource,
};

const REPO: &str = "example-team/example-model";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const FILENAME: &str = "model.bin";

fn bridge(resource: &str, size: u64) -> FileProgressBridge {
    FileProgressBridge::new(tokio::sync::mpsc::channel(1).0, resource.into(), size)
}

fn resource() -> HubResource<'static> {
    HubResource {
        repo: REPO,
        revision: REVISION,
        filename: FILENAME,
        size: 13,
        sha256: "958fc29fabc266c3dbf13432936e396cfc6fe2b50c33a77457888896858e1463",
    }
}

fn downloads(endpoint: &str, cache_dir: std::path::PathBuf) -> Downloads {
    Downloads::with_options(
        &DownloadConfig {
            cache_dir,
            endpoint: Some(endpoint.to_owned()),
            token: None,
        },
        DownloadOptions {
            transport: DownloadTransport::Http,
            ..Default::default()
        },
    )
    .expect("local-hub downloads client")
}

#[tokio::test]
async fn auto_falls_back_when_mirror_advertises_xet_without_a_token_endpoint() {
    let payload = b"model-payload";
    let hash = sha256_hex(payload);
    let server = httpmock::MockServer::start();
    server.mock(|when, then| {
        when.method("HEAD").path_includes(FILENAME);
        then.status(200)
            .header("ETag", format!("\"{hash}\""))
            .header("X-Repo-Commit", REVISION)
            .header("X-Xet-Hash", "a".repeat(64))
            .header("Content-Length", "13");
    });
    let token = server.mock(|when, then| {
        when.method("GET").path_includes("/xet-read-token/");
        then.status(404);
    });
    let body = server.mock(|when, then| {
        when.method("GET").path_includes(FILENAME);
        then.status(200).body(payload.as_slice());
    });
    let root = tempfile::tempdir().unwrap();
    let client = Downloads::new(&DownloadConfig {
        cache_dir: root.path().join("cache"),
        endpoint: Some(server.base_url()),
        token: None,
    })
    .unwrap();
    let path = client
        .ensure_resource(resource(), bridge(FILENAME, 13))
        .await
        .unwrap();
    assert_eq!(std::fs::read(path).unwrap(), payload);
    token.assert_calls(1);
    body.assert_calls(1);
}

#[tokio::test]
async fn direct_cache_download_reuses_valid_and_repairs_corrupt_content() {
    let payload = b"model-payload".to_vec();
    let etag = format!("\"{}\"", sha256_hex(&payload));
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.path_includes(format!("/{REVISION}/{FILENAME}"));
        then.status(200)
            .header("ETag", &etag)
            .header("X-Repo-Commit", REVISION)
            .body(payload.clone());
    });
    let root = tempfile::tempdir().unwrap();
    let client = downloads(&server.base_url(), root.path().join("hf-cache"));
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let download = client.ensure_resource(
        resource(),
        FileProgressBridge::new(tx, "test/model.bin".into(), 13),
    );
    let collect = async move {
        let mut values = Vec::new();
        while let Some(talechime_protocol::Event::ModelProgress {
            downloaded, total, ..
        }) = rx.recv().await
        {
            assert_eq!(total, 13);
            values.push(downloaded);
        }
        values
    };
    let (result, values) = tokio::join!(download, collect);
    let path = result.unwrap();
    assert!(path.starts_with(root.path().join("hf-cache").canonicalize().unwrap()));
    assert_eq!(std::fs::read(&path).unwrap(), payload);
    assert_eq!(values.last(), Some(&13));
    assert!(values.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(mock.calls(), 2);
    let again = client
        .ensure_resource(resource(), bridge(FILENAME, 13))
        .await
        .unwrap();
    assert_eq!(again, path);
    assert_eq!(mock.calls(), 2);
    // Isolated cache only: corruption of the same size must be detected by hash.
    std::fs::write(&path, b"wrong-payload").unwrap();
    let repaired = client
        .ensure_resource(resource(), bridge(FILENAME, 13))
        .await
        .unwrap();
    assert_eq!(std::fs::read(repaired).unwrap(), payload);
    assert!(mock.calls() > 2);
    assert!(!root.path().join("engine").exists());
}

#[tokio::test]
async fn ensure_reports_server_errors() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.path_includes(format!("/{REVISION}/{FILENAME}"));
        then.status(404);
    });
    let root = tempfile::tempdir().expect("tempdir");
    let client = downloads(&server.base_url(), root.path().join("hf-cache"));
    let error = client
        .ensure_resource(resource(), bridge("test/missing.bin", 13))
        .await
        .expect_err("404 must fail");
    assert!(error.to_string().contains("download failed"), "{error}");
    assert!(!root.path().join("engine").exists());
    assert!(mock.calls() >= 1, "the server was asked");
}

#[tokio::test]
async fn gateway_failure_is_reported_as_network_error() {
    let server = httpmock::MockServer::start();
    server.mock(|when, then| {
        when.path_includes(format!("/{REVISION}/{FILENAME}"));
        then.status(502);
    });
    let root = tempfile::tempdir().unwrap();
    let error = downloads(&server.base_url(), root.path().join("hf-cache"))
        .ensure_resource(resource(), bridge("test/model.bin", 13))
        .await
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<DownloadFailure>(),
        Some(DownloadFailure::Network(_))
    ));
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

// Exercise the real HTTP cache path even when the Hub advertises Xet.
#[tokio::test]
async fn http_resume_validates_ranges_status_and_length_before_publication() {
    for case in [
        "resume",
        "ignored-range",
        "bad-range",
        "short-body",
        "forbidden",
    ] {
        let payload = b"model-payload";
        let hash = sha256_hex(payload);
        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.method("HEAD").path_includes(FILENAME);
            then.status(200)
                .header("ETag", format!("\"{hash}\""))
                .header("X-Repo-Commit", REVISION)
                .header("X-Xet-Hash", "unused")
                .header("Content-Length", "13");
        });
        let get = server.mock(|when, then| {
            when.method("GET")
                .path_includes(FILENAME)
                .header("range", "bytes=4-12");
            match case {
                "resume" => {
                    then.status(206)
                        .header("Content-Range", "bytes 4-12/13")
                        .body(&payload[4..]);
                }
                "ignored-range" => {
                    then.status(200).body(payload.as_slice());
                }
                "bad-range" => {
                    then.status(206)
                        .header("Content-Range", "bytes 0-8/13")
                        .body(&payload[4..]);
                }
                "short-body" => {
                    then.status(206)
                        .header("Content-Range", "bytes 4-12/13")
                        .body(&payload[..3]);
                }
                _ => {
                    then.status(403).body("forbidden");
                }
            }
        });
        let ignored = server.mock(|when, then| {
            when.method("GET")
                .path_includes(FILENAME)
                .header_missing("range");
            then.status(200).body(payload.as_slice());
        });
        let root = tempfile::tempdir().unwrap();
        let cache = root.path().join("hf-cache");
        let blobs = cache.join("models--example-team--example-model/blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let incomplete = blobs.join(format!("{hash}.incomplete"));
        std::fs::write(&incomplete, &payload[..4]).unwrap();
        std::fs::write(
            blobs.join(format!("{hash}.incomplete.ranges.json")),
            serde_json::to_vec(&serde_json::json!({
                "version": 1, "total": 13, "chunk": 4 * 1024 * 1024, "etag": hash, "completed": [4]
            }))
            .unwrap(),
        )
        .unwrap();
        let result = downloads(&server.base_url(), cache)
            .ensure_resource(resource(), bridge(FILENAME, 13))
            .await;
        if case == "resume" || case == "ignored-range" {
            assert_eq!(std::fs::read(result.unwrap()).unwrap(), payload);
            assert!(!incomplete.exists());
        } else {
            assert!(result.is_err(), "{case}");
            assert!(
                !root.path().join("engine").exists(),
                "{case}: must not publish"
            );
            assert!(
                !blobs.join(&hash).exists(),
                "{case}: must not finalize cache"
            );
            assert!(incomplete.exists());
        }
        get.assert_calls(1);
        ignored.assert_calls(usize::from(case == "ignored-range"));
    }
}

#[tokio::test]
async fn mirror_absolute_metadata_redirect_drops_credentials_across_origins() {
    let payload = b"model-payload";
    let hub = httpmock::MockServer::start();
    let mirror = httpmock::MockServer::start();
    let mirror_request = mirror.mock(|when, then| {
        when.path_includes(FILENAME)
            .header("authorization", "Bearer diagnostic-test-token");
        then.status(308)
            .header("location", format!("{}/{FILENAME}", hub.base_url()));
    });
    let hub_request = hub.mock(|when, then| {
        when.path(format!("/{FILENAME}"))
            .header_missing("authorization");
        then.status(200)
            .header("ETag", format!("\"{}\"", sha256_hex(payload)))
            .header("X-Repo-Commit", REVISION)
            .body(payload.as_slice());
    });
    let root = tempfile::tempdir().unwrap();
    let client = Downloads::new(&DownloadConfig {
        cache_dir: root.path().join("cache"),
        endpoint: Some(mirror.base_url()),
        token: Some("diagnostic-test-token".into()),
    })
    .unwrap();
    let path = client
        .ensure_resource(resource(), bridge(FILENAME, 13))
        .await
        .unwrap();
    assert_eq!(std::fs::read(path).unwrap(), payload);
    mirror_request.assert_calls(1);
    hub_request.assert_calls(2);
}

#[tokio::test]
async fn metadata_bearing_redirect_does_not_head_the_cdn() {
    let payload = b"model-payload";
    let hub = httpmock::MockServer::start();
    let cdn = httpmock::MockServer::start();
    let head = hub.mock(|when, then| {
        when.method("HEAD").path_includes(FILENAME);
        then.status(302)
            .header("location", format!("{}/{FILENAME}", cdn.base_url()))
            .header("X-Linked-Etag", format!("\"{}\"", sha256_hex(payload)))
            .header("X-Linked-Size", "13")
            .header("X-Repo-Commit", REVISION);
    });
    let cdn_head = cdn.mock(|when, then| {
        when.method("HEAD");
        then.status(500);
    });
    let get = cdn.mock(|when, then| {
        when.method("GET").path_includes(FILENAME);
        then.status(200).body(payload.as_slice());
    });
    let root = tempfile::tempdir().unwrap();
    let path = downloads(&hub.base_url(), root.path().join("cache"))
        .ensure_resource(resource(), bridge(FILENAME, 13))
        .await
        .unwrap();
    assert_eq!(std::fs::read(path).unwrap(), payload);
    head.assert_calls(1);
    get.assert_calls(1);
    cdn_head.assert_calls(0);
}
