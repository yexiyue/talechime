//! List a page of public Chinese voices using the configured Fish account.
use std::env;
use talechime_cloud_tts::fish::{ApiKey, Client, VoiceQuery, VoiceScope};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(ApiKey::new(env::var("FISH_API_KEY")?)?)?;
    let page = client
        .search_voices(&VoiceQuery {
            scope: VoiceScope::Public,
            page_number: 1,
            page_size: 1,
            title: None,
            tag: None,
            language: Some("zh".into()),
        })
        .await?;
    println!(
        "Fetched {} voices; total reported: {}",
        page.items.len(),
        page.total
    );
    Ok(())
}
