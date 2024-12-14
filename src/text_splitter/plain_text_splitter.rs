use async_trait::async_trait;
use text_splitter::ChunkConfig;
use tiktoken_rs::tokenizer::Tokenizer;

use super::{SplitterOptions, TextSplitter, TextSplitterError};

pub struct PlainTextSplitter {
    splitter_options: SplitterOptions,
}

impl Default for PlainTextSplitter {
    fn default() -> Self {
        PlainTextSplitter::new(SplitterOptions::default())
    }
}

impl PlainTextSplitter {
    pub fn new(options: SplitterOptions) -> PlainTextSplitter {
        PlainTextSplitter {
            splitter_options: options,
        }
    }

    #[deprecated = "Use `SplitterOptions::get_tokenizer_from_str` instead"]
    pub fn get_tokenizer_from_str(&self, s: &str) -> Option<Tokenizer> {
        match s.to_lowercase().as_str() {
            "cl100k_base" => Some(Tokenizer::Cl100kBase),
            "p50k_base" => Some(Tokenizer::P50kBase),
            "r50k_base" => Some(Tokenizer::R50kBase),
            "p50k_edit" => Some(Tokenizer::P50kEdit),
            "gpt2" => Some(Tokenizer::Gpt2),
            _ => None,
        }
    }
}

#[async_trait]
impl TextSplitter for PlainTextSplitter {
    async fn split_text(&self, text: &str) -> Result<Vec<String>, TextSplitterError> {
        let chunk_config = ChunkConfig::try_from(&self.splitter_options)?;
        Ok(text_splitter::TextSplitter::new(chunk_config)
            .chunks(text)
            .map(|x| x.to_string())
            .collect())
    }
}
