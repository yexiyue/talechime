//! List a page of public Chinese voices using the configured HeyGen account.
use std::env;
use talechime_cloud_tts::heygen::{ApiKey, Client, VoiceQuery, VoiceScope};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(ApiKey::new(env::var("HEYGEN_API_KEY")?)?)?;
    let page = list(&client).await?;
    println!(
        "Fetched {} voices; more pages: {}",
        page.data.len(),
        page.has_more
    );
    Ok(())
}

async fn list(
    client: &Client,
) -> Result<talechime_cloud_tts::heygen::VoicePage, talechime_cloud_tts::heygen::Error> {
    client
        .search_voices(&VoiceQuery {
            scope: VoiceScope::Public,
            token: None,
            limit: 1,
            age: None,
            query: None,
            gender: None,
            language: Some("zh".into()),
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };
    #[tokio::test]
    async fn example_query_uses_mock_catalog() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut data = [0; 8192];
            let _ = socket.read(&mut data).await.unwrap();
            socket.write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"data\":[],\"has_more\":false,\"next_token\":null}").await.unwrap();
        });
        let client = Client::with_endpoint(ApiKey::new("test").unwrap(), &endpoint).unwrap();
        assert!(list(&client).await.unwrap().data.is_empty());
        server.await.unwrap();
    }
}
