//! Resumable resource downloads. Dropping the future cancels the transfer.
use std::{fs::File, path::Path};
use tokio::{fs, io::AsyncWriteExt};

pub(super) async fn download_from_url<F>(
    url: &str,
    dest: &Path,
    mut on_progress: F,
) -> anyhow::Result<()>
where
    F: FnMut(u64, u64),
{
    if let Some(parent) = dest.parent()
        && !parent.exists()
    {
        fs::create_dir_all(parent).await?;
    }

    let _lock = lock(&dest.with_extension("download.lock"))?;
    if dest.exists() {
        return Ok(());
    }
    let path = format!("{}.download", dest.display());

    let (mut downloaded, mut file) = if let Ok(metadata) = std::fs::metadata(&path) {
        let file = fs::File::options().append(true).open(&path).await?;
        (metadata.len(), file)
    } else {
        (0, fs::File::create(&path).await?)
    };

    let mut request = reqwest::Client::new().get(url);
    if downloaded > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={}-", downloaded));
    }

    let mut res = request.send().await?.error_for_status()?;

    if downloaded > 0 {
        if res.status() == reqwest::StatusCode::PARTIAL_CONTENT {
            let prefix = format!("bytes {downloaded}-");
            if !res
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with(&prefix))
            {
                return Err(anyhow::anyhow!("invalid resume Content-Range"));
            }
        } else {
            // Servers may ignore Range. Restart, never append a complete file
            // to the existing partial download.
            file = fs::File::create(&path).await?;
            downloaded = 0;
        }
    }
    let content_length = res.content_length().map(|length| length + downloaded);

    on_progress(downloaded, content_length.unwrap_or(0));

    while let Some(data) = res.chunk().await? {
        file.write_all(&data).await?;
        downloaded += data.len() as u64;
        on_progress(downloaded, content_length.unwrap_or(0));
    }

    if content_length.is_some_and(|length| downloaded != length) {
        return Err(anyhow::anyhow!("Download failed"));
    }

    file.sync_all().await?;
    drop(file);
    fs::rename(path, dest).await?;
    Ok(())
}

fn lock(path: &Path) -> std::io::Result<File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)?;
    file.try_lock().map_err(std::io::Error::other)?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
    };

    fn server(
        response: &'static str,
        range: bool,
        hold: Option<std::sync::mpsc::Receiver<()>>,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/resource", listener.local_addr().unwrap());
        let task = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            loop {
                let mut byte = [0];
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            assert_eq!(
                String::from_utf8_lossy(&request)
                    .to_lowercase()
                    .contains("range: bytes=3-"),
                range
            );
            stream.write_all(response.as_bytes()).unwrap();
            if let Some(hold) = hold {
                hold.recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            }
        });
        (url, task)
    }

    #[tokio::test]
    async fn resumes_partial_content_and_restarts_when_range_is_ignored() {
        for response in [
            "HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 3-5/6\r\nConnection: close\r\n\r\ndef",
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabcdef",
        ] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("model");
            std::fs::write(format!("{}.download", path.display()), b"abc").unwrap();
            let (url, task) = server(response, true, None);
            download_from_url(&url, &path, |_, _| {}).await.unwrap();
            task.join().unwrap();
            assert_eq!(std::fs::read(path).unwrap(), b"abcdef");
        }
    }

    #[tokio::test]
    async fn cancellation_keeps_partial_file_without_claiming_success() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model");
        let (release, hold) = std::sync::mpsc::channel();
        let (url, task) = server(
            "HTTP/1.1 200 OK\r\nContent-Length: 600\r\nConnection: close\r\n\r\nabc",
            false,
            Some(hold),
        );
        let (cancelled, mut cancellation) = tokio::sync::watch::channel(false);
        let result = tokio::select! {
            biased;
            _ = cancellation.changed() => None,
            result = download_from_url(&url, &path, move |downloaded, _| {
                if downloaded > 0 { cancelled.send_replace(true); }
            }) => Some(result),
        };
        release.send(()).unwrap();
        task.join().unwrap();
        assert!(
            result.is_none(),
            "download completed instead of being cancelled"
        );
        assert!(!path.exists());
        assert!(std::path::PathBuf::from(format!("{}.download", path.display())).exists());
    }
}
