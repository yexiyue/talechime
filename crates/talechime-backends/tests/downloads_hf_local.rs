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
            .body(payload.clone());
    });
    let root = tempfile::tempdir().expect("tempdir");
    let client = downloads(&server.base_url(), root.path().join("hf-cache"));
    let dest = root.path().join("engine/moss/model.bin");

    client
        .ensure(
            &dest,
            REPO,
            REVISION,
            FILENAME,
            bridge("test/model.bin", 13),
        )
        .await
        .expect("first ensure downloads from the local hub");

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
