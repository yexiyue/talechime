//! End-to-end download tests against an in-process HTTP mock standing in for
//! the Hub. Never touches the network; pinned commit revisions avoid revision
//! resolution requests.
use talechime_backends::downloads::{DownloadConfig, Downloads, FileProgressBridge};

const REPO: &str = "example-team/example-model";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const FILENAME: &str = "model.bin";

fn bridge(resource: &str, size: u64) -> FileProgressBridge {
    FileProgressBridge::new(tokio::sync::mpsc::channel(1).0, resource.into(), size)
}

fn downloads(endpoint: &str, cache_dir: std::path::PathBuf) -> Downloads {
    Downloads::new(&DownloadConfig {
        cache_dir,
        endpoint: Some(endpoint.to_owned()),
        token: None,
    })
    .expect("local-hub downloads client")
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
                .header("range", "bytes=4-");
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
                    then.status(200).body(&payload[..3]);
                }
                _ => {
                    then.status(403).body("forbidden");
                }
            }
        });
        let root = tempfile::tempdir().unwrap();
        let cache = root.path().join("hf-cache");
        let blobs = cache.join("models--example-team--example-model/blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let incomplete = blobs.join(format!("{hash}.incomplete"));
        std::fs::write(&incomplete, &payload[..4]).unwrap();
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
    }
}
