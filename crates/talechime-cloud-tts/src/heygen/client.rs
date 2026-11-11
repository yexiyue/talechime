use super::{ApiKey, Error};
use reqwest::{RequestBuilder, Response, StatusCode, Url};

/// HeyGen Audio client. It never follows redirects or retries requests automatically.
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base: Url,
    key: ApiKey,
}

impl Client {
    /// Create a client for the public HeyGen service.
    /// Credentials are explicit; proxy environment variables are managed by the host.
    pub fn new(key: ApiKey) -> Result<Self, Error> {
        Self::with_endpoint(key, "https://api.heygen.com/")
    }

    /// Use an HTTPS endpoint, or a loopback HTTP endpoint for local testing.
    pub fn with_endpoint(key: ApiKey, endpoint: &str) -> Result<Self, Error> {
        let base = Url::parse(endpoint).map_err(|_| Error::InvalidInput("endpoint"))?;
        let local = matches!(base.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
        if !(base.scheme() == "https" || base.scheme() == "http" && local)
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
            || base.path() != "/"
        {
            return Err(Error::InvalidInput("endpoint"));
        }
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(|_| Error::Transport)?;
        Ok(Self { http, base, key })
    }

    pub(super) fn request(&self, method: reqwest::Method, path: &str) -> RequestBuilder {
        let url = self.base.join(path).expect("static paths are valid URLs");
        self.http
            .request(method, url)
            .header("X-Api-Key", &self.key.0)
    }

    pub(super) fn voice_request(&self, id: &str) -> Result<RequestBuilder, Error> {
        if !super::valid_id(id) {
            return Err(Error::InvalidInput("voice id"));
        }
        let mut url = self.base.clone();
        url.path_segments_mut()
            .map_err(|_| Error::InvalidInput("endpoint"))?
            .clear()
            .push("v3")
            .push("models")
            .push("audio")
            .push("voices")
            .push(id);
        Ok(self.http.get(url).header("X-Api-Key", &self.key.0))
    }

    pub(super) async fn send(
        &self,
        request: RequestBuilder,
        side_effect: bool,
    ) -> Result<Response, Error> {
        let response = request.send().await.map_err(|_| {
            if side_effect {
                Error::OutcomeUnknown
            } else {
                Error::Transport
            }
        })?;
        match response.status() {
            status if status.is_success() => Ok(response),
            StatusCode::UNAUTHORIZED => Err(Error::Authentication),
            StatusCode::FORBIDDEN => Err(Error::Forbidden),
            StatusCode::TOO_MANY_REQUESTS => Err(Error::RateLimited),
            status if side_effect && status.is_server_error() => Err(Error::OutcomeUnknown),
            status => Err(Error::Http(status.as_u16())),
        }
    }
}

pub(super) async fn bounded_json<T: serde::de::DeserializeOwned>(
    mut response: Response,
) -> Result<T, Error> {
    const MAX: usize = 4 * 1024 * 1024;
    let mut data = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| Error::Transport)? {
        if chunk.len() > MAX.saturating_sub(data.len()) {
            return Err(Error::InvalidResponse);
        }
        data.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&data).map_err(|_| Error::InvalidResponse)
}
