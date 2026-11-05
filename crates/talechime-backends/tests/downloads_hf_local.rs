//! End-to-end download tests against an in-process HTTP mock standing in for
//! the Hub. Never touches the network; pinned commit revisions avoid revision
//! resolution requests.
use talechime_backends::downloads::{
    DownloadConfig, DownloadOptions, DownloadTransport, Downloads, FileProgressBridge,
};

const REPO: &str = "example-team/example-model";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const FILENAME: &str = "model.bin";

fn bridge(resource: &str, size: u64) -> FileProgressBridge {
    FileProgressBridge::new(tokio::sync::mpsc::channel(1).0, resource.into(), size)
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
    let dest = root.path().join("engine/model.bin");
    client
        .ensure(&dest, REPO, REVISION, FILENAME, bridge(FILENAME, 13))
        .await
        .unwrap();
    assert_eq!(std::fs::read(dest).unwrap(), payload);
    token.assert_calls(1);
    body.assert_calls(1);
}

#[tokio::test]
async fn ensure_downloads_links_and_skips_existing_dest() {
    let payload = b"model-payload".to_vec();
    let etag = format!("\"{}\"", sha256_hex(&payload));
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        // Any method: hf-hub probes with HEAD before the GET transfer.
        when.path_includes(format!("/{REVISION}/{FILENAME}"));
        then.status(200)
            .header("ETag", &etag)
            .header("X-Repo-Commit", REVISION)
            .header("X-Xet-Hash", "should-not-use-xet")
            .body(payload.clone());
    });
    let root = tempfile::tempdir().expect("tempdir");
    let client = downloads(&server.base_url(), root.path().join("hf-cache"));
    let dest = root.path().join("engine/moss/model.bin");

    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let ensure = client.ensure(
        &dest,
        REPO,
        REVISION,
        FILENAME,
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
    let (result, values) = tokio::join!(ensure, collect);
    result.expect("first ensure downloads from the local hub");
    assert_eq!(
        values.last(),
        Some(&13),
        "publication must flush exact file completion"
    );
    assert!(values.windows(2).all(|w| w[0] <= w[1]));

    assert!(
        dest.is_file(),
        "engine-layout entry must be a real file after linking"
    );
    assert_eq!(std::fs::read(&dest).unwrap(), b"model-payload");
    // One HEAD (etag) plus one GET (transfer).
    assert_eq!(mock.calls(), 2, "exactly one HEAD + one GET");
    // The blob landed in the shared cache, not the engine directory.
    assert!(root.path().join("hf-cache").is_dir());

    client
        .ensure(
            &dest,
            REPO,
            REVISION,
            FILENAME,
            bridge("test/model.bin", 13),
        )
        .await
        .expect("second ensure succeeds");
    assert_eq!(
        mock.calls(),
        2,
        "existing dest entries are never re-downloaded"
    );
    // Corrupt the hardlinked inode, including the cached blob. A retry must
    // ask the official client to refresh it rather than reuse the bad cache.
    std::fs::write(&dest, b"short").unwrap();
    client
        .ensure(
            &dest,
            REPO,
            REVISION,
            FILENAME,
            bridge("test/model.bin", 13),
        )
        .await
        .expect("truncated files must be repaired");
    assert_eq!(std::fs::read(&dest).unwrap(), payload);
    assert!(mock.calls() > 2);
    // Also heal a corrupt shared blob when the engine-layout entry is gone.
    std::fs::write(&dest, b"short").unwrap();
    std::fs::remove_file(&dest).unwrap();
    client
        .ensure(
            &dest,
            REPO,
            REVISION,
            FILENAME,
            bridge("test/model.bin", 13),
        )
        .await
        .expect("a corrupt cache hit must be refreshed");
    assert_eq!(std::fs::read(&dest).unwrap(), payload);
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
    let dest = root.path().join("engine/missing.bin");
    let error = client
        .ensure(
            &dest,
            REPO,
            REVISION,
            FILENAME,
            bridge("test/missing.bin", 13),
        )
        .await
        .expect_err("404 must fail");
    assert!(error.to_string().contains("download failed"), "{error}");
    assert!(!dest.exists(), "no partial destination on failure");
    assert!(mock.calls() >= 1, "the server was asked");
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
        let dest = root.path().join("engine/model.bin");
        let result = downloads(&server.base_url(), cache)
            .ensure(&dest, REPO, REVISION, FILENAME, bridge(FILENAME, 13))
            .await;
        if case == "resume" || case == "ignored-range" {
            result.unwrap();
            assert_eq!(std::fs::read(dest).unwrap(), payload);
            assert!(!incomplete.exists());
        } else {
            assert!(result.is_err(), "{case}");
            assert!(!dest.exists(), "{case}: must not publish");
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
    let dest = root.path().join("engine/model.bin");
    client
        .ensure(&dest, REPO, REVISION, FILENAME, bridge(FILENAME, 13))
        .await
        .unwrap();
    assert_eq!(std::fs::read(dest).unwrap(), payload);
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
    let dest = root.path().join("engine/model.bin");
    downloads(&hub.base_url(), root.path().join("cache"))
        .ensure(&dest, REPO, REVISION, FILENAME, bridge(FILENAME, 13))
        .await
        .unwrap();
    assert_eq!(std::fs::read(dest).unwrap(), payload);
    head.assert_calls(1);
    get.assert_calls(1);
    cdn_head.assert_calls(0);
}
