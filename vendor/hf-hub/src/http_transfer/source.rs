use crate::{
    HFClient,
    error::{HFError, HFResult},
};

/// Resolve once per file; refresh an expired CDN signature with one serialized
/// metadata request. Credentials are removed after construction for other origins.
pub(crate) struct Source {
    pub(super) client: HFClient,
    original: String,
    url: tokio::sync::Mutex<(String, u64)>,
    pub(super) total: u64,
    pub(super) etag: String,
}
impl Source {
    pub(crate) fn new(
        client: HFClient,
        original: String,
        resolved: String,
        total: u64,
        etag: String,
    ) -> Self {
        Self {
            client,
            original,
            url: tokio::sync::Mutex::new((resolved, 0)),
            total,
            etag,
        }
    }
    pub(crate) async fn fetch(&self, range: Option<&str>) -> HFResult<reqwest::Response> {
        let (url, generation) = self.url.lock().await.clone();
        let response = self.send(&url, range).await?;
        if response.status() != reqwest::StatusCode::FORBIDDEN || url == self.original {
            return Ok(response);
        }
        drop(response);
        let mut current = self.url.lock().await;
        if current.1 == generation {
            let response = self
                .client
                .head_with_metadata_redirects(&self.original, &self.client.auth_headers())
                .await?;
            let etag = response
                .headers()
                .get("x-linked-etag")
                .or_else(|| response.headers().get(reqwest::header::ETAG))
                .and_then(|v| v.to_str().ok())
                .map(|v| v.strip_prefix("W/").unwrap_or(v).trim_matches('"'));
            let size = response
                .headers()
                .get("x-linked-size")
                .or_else(|| response.headers().get(reqwest::header::CONTENT_LENGTH))
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok());
            if etag != Some(self.etag.as_str()) || size != Some(self.total) {
                return Err(HFError::malformed_response(
                    "download identity changed during CDN refresh",
                ));
            }
            current.0 = body_url(&response)?;
            current.1 += 1;
        }
        let url = current.0.clone();
        drop(current);
        self.send(&url, range).await.map_err(Into::into)
    }
    async fn send(
        &self,
        url: &str,
        range: Option<&str>,
    ) -> Result<reqwest::Response, reqwest::Error> {
        let mut builder = self
            .client
            .http_client()
            .get(url)
            .headers(self.client.auth_headers())
            .header(reqwest::header::ACCEPT_ENCODING, "identity");
        if let Some(range) = range {
            builder = builder.header(reqwest::header::RANGE, range);
        }
        let mut request = builder.build()?;
        let same_origin = reqwest::Url::parse(self.client.endpoint())
            .ok()
            .is_some_and(|base| base.origin() == request.url().origin());
        if !same_origin {
            for name in [
                reqwest::header::AUTHORIZATION,
                reqwest::header::COOKIE,
                reqwest::header::PROXY_AUTHORIZATION,
            ] {
                request.headers_mut().remove(name);
            }
        }
        self.client.http_client().execute(request).await
    }
}
pub(crate) fn body_url(response: &reqwest::Response) -> HFResult<String> {
    let next = if response.status().is_redirection() {
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| HFError::malformed_response("missing download redirect"))?;
        response.url().join(location)?
    } else {
        response.url().clone()
    };
    if !matches!(next.scheme(), "http" | "https")
        || !next.username().is_empty()
        || next.password().is_some()
        || (response.url().scheme() == "https" && next.scheme() != "https")
    {
        return Err(HFError::malformed_response("unsafe download redirect"));
    }
    Ok(next.into())
}
