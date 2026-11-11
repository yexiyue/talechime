//! Fixed non-Hub public assets. The preparation lease owns the partial prefix;
//! size and SHA-256 are verified by the existing recipe before publication.
use super::{Downloads, FileProgressBridge, ProgressRelay};
use std::{path::Path, sync::atomic::Ordering, time::Duration};
use tokio::io::AsyncWriteExt;

impl Downloads {
    pub(crate) async fn ensure_pinned_url(
        &self,
        destination: &Path,
        url: &str,
        progress: FileProgressBridge,
        refresh: bool,
    ) -> anyhow::Result<()> {
        let client = reqwest::Client::builder()
            .https_only(true)
            .connect_timeout(Duration::from_secs(15))
            .read_timeout(Duration::from_secs(45))
            .build()?;
        let relay = ProgressRelay {
            tx: progress.tx.clone(),
            resource: progress.resource.clone(),
            size: progress.size,
            latest: progress.latest.subscribe(),
        };
        relay
            .run(transfer(&client, destination, url, progress, refresh))
            .await
    }
}

async fn transfer(
    client: &reqwest::Client,
    destination: &Path,
    url: &str,
    progress: FileProgressBridge,
    refresh: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        progress.size > 0,
        "fixed resource must have a positive pinned size"
    );
    let mut partial_name = destination.as_os_str().to_os_string();
    partial_name.push(".part");
    let partial = std::path::PathBuf::from(partial_name);
    let mut offset = match tokio::fs::symlink_metadata(&partial).await {
        Ok(metadata) => {
            anyhow::ensure!(metadata.is_file(), "invalid fixed resource partial file");
            metadata.len()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    if refresh || offset > progress.size {
        offset = 0;
    }
    if offset == progress.size {
        tokio::fs::rename(partial, destination).await?;
        return Ok(());
    }
    let mut request = client.get(url);
    if offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let mut response = request.send().await?.error_for_status()?;
    if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
        let expected = format!("bytes {offset}-{}/{}", progress.size - 1, progress.size);
        anyhow::ensure!(
            response
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|value| value.to_str().ok())
                == Some(expected.as_str()),
            "fixed resource range mismatch"
        );
    } else {
        anyhow::ensure!(
            response.status() == reqwest::StatusCode::OK,
            "unexpected fixed resource response"
        );
        offset = 0;
    }
    if let Some(length) = response.content_length() {
        anyhow::ensure!(
            length == progress.size - offset,
            "fixed resource size mismatch"
        );
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(offset > 0)
        .truncate(offset == 0)
        .open(&partial)
        .await?;
    progress.latest.send_replace(Some(offset));
    while let Some(chunk) = response.chunk().await? {
        anyhow::ensure!(
            chunk.len() as u64 <= progress.size - offset,
            "fixed resource exceeds pinned size"
        );
        file.write_all(&chunk).await?;
        offset += chunk.len() as u64;
        progress.downloaded.store(offset, Ordering::Relaxed);
        progress.latest.send_replace(Some(offset));
    }
    anyhow::ensure!(offset == progress.size, "fixed resource response truncated");
    file.flush().await?;
    file.sync_all().await?;
    drop(file);
    tokio::fs::rename(partial, destination).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resumes_only_a_matching_range_and_keeps_partial_on_failure() {
        let server = httpmock::MockServer::start();
        let good = server.mock(|when, then| {
            when.path("/asset").header("range", "bytes=2-");
            then.status(206)
                .header("Content-Range", "bytes 2-4/5")
                .body("cde");
        });
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("asset");
        std::fs::write(root.path().join("asset.part"), "ab").unwrap();
        let (tx, _) = tokio::sync::mpsc::channel(1);
        transfer(
            &reqwest::Client::new(),
            &target,
            &server.url("/asset"),
            FileProgressBridge::new(tx, "fixture".into(), 5),
            false,
        )
        .await
        .unwrap();
        good.assert();
        assert_eq!(std::fs::read(&target).unwrap(), b"abcde");
        std::fs::write(root.path().join("asset.part"), "ab").unwrap();
        let bad = server.mock(|when, then| {
            when.path("/bad");
            then.status(206)
                .header("Content-Range", "bytes 1-3/5")
                .body("bcd");
        });
        let (tx, _) = tokio::sync::mpsc::channel(1);
        assert!(
            transfer(
                &reqwest::Client::new(),
                &target,
                &server.url("/bad"),
                FileProgressBridge::new(tx, "fixture".into(), 5),
                false
            )
            .await
            .is_err()
        );
        bad.assert();
        assert_eq!(std::fs::read(&target).unwrap(), b"abcde");
        assert_eq!(
            std::fs::read(root.path().join("asset.part")).unwrap(),
            b"ab"
        );
    }
}
