#![allow(dead_code)]

use std::time::Duration;

use crate::embedding::{embedder_trait::Embedder, EmbedderError};
pub use async_openai::config::{AzureConfig, Config, OpenAIConfig};
use async_openai::{
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client,
};
use async_trait::async_trait;
use backoff::ExponentialBackoff;

#[derive(Debug)]
pub struct OpenAiEmbedder<C: Config> {
    config: C,
    model: String,
    timeout: Duration,
    retry_count: u32,
}

impl<C: Config + Send + Sync + 'static> Into<Box<dyn Embedder>> for OpenAiEmbedder<C> {
    fn into(self) -> Box<dyn Embedder> {
        Box::new(self)
    }
}

impl<C: Config> OpenAiEmbedder<C> {
    pub fn new(config: C) -> Self {
        OpenAiEmbedder {
            config,
            model: String::from("text-embedding-ada-002"),
            timeout: Duration::from_secs(30),
            retry_count: 3,
        }
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_config(mut self, config: C) -> Self {
        self.config = config;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_retry_count(mut self, retry_count: u32) -> Self {
        self.retry_count = retry_count;
        self
    }
}

impl Default for OpenAiEmbedder<OpenAIConfig> {
    fn default() -> Self {
        OpenAiEmbedder::new(OpenAIConfig::default())
    }
}

#[async_trait]
impl<C: Config + Send + Sync> Embedder for OpenAiEmbedder<C> {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        let backoff = ExponentialBackoff {
            max_elapsed_time: Some(self.timeout),
            max_interval: Duration::from_secs(30),
            ..ExponentialBackoff::default()
        };

        let client = Client::build(
            reqwest::Client::builder()
                .timeout(self.timeout)
                .build()
                .unwrap(),
            self.config.clone(),
            backoff,
        );

        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(EmbeddingInput::StringArray(documents.into()))
            .build()?;

        let response = client.embeddings().create(request).await?;

        let embeddings = response
            .data
            .into_iter()
            .map(|item| item.embedding)
            .map(|embedding| {
                embedding
                    .into_iter()
                    .map(|x| x as f64)
                    .collect::<Vec<f64>>()
            })
            .collect();

        Ok(embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        let backoff = ExponentialBackoff {
            max_elapsed_time: Some(self.timeout * (self.retry_count + 1)),
            max_interval: self.timeout,
            initial_interval: Duration::from_millis(100),
            multiplier: 2.0,
            ..ExponentialBackoff::default()
        };

        let client = Client::build(
            reqwest::Client::builder()
                .timeout(self.timeout)
                .build()
                .unwrap(),
            self.config.clone(),
            backoff,
        );

        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(text)
            .build()?;

        let mut response = client.embeddings().create(request).await?;

        let item = response.data.swap_remove(0);

        Ok(item
            .embedding
            .into_iter()
            .map(|x| x as f64)
            .collect::<Vec<f64>>())
    }
}
