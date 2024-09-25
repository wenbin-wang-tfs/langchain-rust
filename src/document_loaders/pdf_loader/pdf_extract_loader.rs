use std::{
    collections::HashMap,
    fmt,
    io::Read,
    path::Path,
    pin::Pin,
    sync::{Arc, Mutex},
};

use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use pdf_extract::{output_doc, ConvertToFmt, OutputDev, OutputError, PlainTextOutput};
use serde_json::Value;

use crate::{
    document_loaders::{process_doc_stream, Loader, LoaderError},
    schemas::Document,
    text_splitter::TextSplitter,
};

#[derive(Debug, Clone)]
pub struct PdfExtractLoader {
    document: pdf_extract::Document,
}

struct PagePlainTextOutput {
    inner: PlainTextOutput<OutputWrapper>,
    pages: HashMap<u32, String>,
    current_page: u32,
    reader: Arc<Mutex<String>>,
}

struct OutputWrapper(Arc<Mutex<String>>);

impl std::fmt::Write for OutputWrapper {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.lock().unwrap().write_str(s).map_err(|_| fmt::Error)
    }
}

impl ConvertToFmt for OutputWrapper {
    type Writer = OutputWrapper;

    fn convert(self) -> Self::Writer {
        self
    }
}

impl PagePlainTextOutput {
    fn new() -> Self {
        let s = Arc::new(Mutex::new(String::new()));
        let writer = Arc::clone(&s);
        Self {
            pages: HashMap::new(),
            current_page: 0,
            reader: s,
            inner: PlainTextOutput::new(OutputWrapper(writer)),
        }
    }
}

impl OutputDev for PagePlainTextOutput {
    fn begin_page(
        &mut self,
        page_num: u32,
        media_box: &pdf_extract::MediaBox,
        art_box: Option<(f64, f64, f64, f64)>,
    ) -> Result<(), OutputError> {
        self.current_page = page_num;
        self.inner.begin_page(page_num, media_box, art_box)
    }

    fn end_page(&mut self) -> Result<(), OutputError> {
        self.inner.end_page()?;

        let buf = self.reader.lock().unwrap().clone();
        self.pages.insert(self.current_page, buf);
        self.reader.lock().unwrap().clear();

        Ok(())
    }

    fn output_character(
        &mut self,
        trm: &pdf_extract::Transform,
        width: f64,
        spacing: f64,
        font_size: f64,
        char: &str,
    ) -> Result<(), OutputError> {
        self.inner
            .output_character(trm, width, spacing, font_size, char)
    }

    fn begin_word(&mut self) -> Result<(), OutputError> {
        self.inner.begin_word()
    }

    fn end_word(&mut self) -> Result<(), OutputError> {
        self.inner.end_word()
    }

    fn end_line(&mut self) -> Result<(), OutputError> {
        self.inner.end_line()
    }
}

impl PdfExtractLoader {
    /// Creates a new PdfLoader from anything that implements the Read trait.
    /// This is a generic constructor which can be used with any type of reader.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::io::Cursor;
    /// let data = Cursor::new(vec![...] /* some PDF data */);
    /// let loader = PdfExtractLoader::new(data)?;
    /// ```
    ///
    pub fn new<R: Read>(reader: R) -> Result<Self, LoaderError> {
        let document = pdf_extract::Document::load_from(reader)?;
        Ok(Self { document })
    }
    /// Creates a new PdfLoader from a path to a PDF file.
    /// This loads the PDF document and creates a PdfLoader from it.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = PdfExtractLoader::from_path("/path/to/my.pdf")?;
    /// ```
    ///
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, LoaderError> {
        let document = pdf_extract::Document::load(path)?;
        Ok(Self { document })
    }
}

#[async_trait]
impl Loader for PdfExtractLoader {
    async fn load(
        mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let stream = stream! {
            let mut output = PagePlainTextOutput::new();
            output_doc(&self.document, &mut output)?;
            for (page_num, text) in output.pages {
                let mut metadata = HashMap::new();
                metadata.insert("page_number".to_string(), Value::from(page_num));
                let doc = Document::new(text).with_metadata(metadata);
                yield Ok(doc);
            }
        };

        Ok(Box::pin(stream))
    }

    async fn load_and_split<TS: TextSplitter + 'static>(
        mut self,
        splitter: TS,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let doc_stream = self.load().await?;
        let stream = process_doc_stream(doc_stream, splitter).await;
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Cursor};

    use futures_util::StreamExt;

    use super::*;

    #[tokio::test]
    async fn test_lo_pdf_loader() {
        let path = "/Users/wenbing.wang/Downloads/introduction-to-counting-amp-probability-the-art-of-problem-solving-2nd.pdf";

        let loader = PdfExtractLoader::from_path(path).expect("Failed to create PdfExtractLoader");

        let docs = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;
        println!("{:?}", docs);
        assert_eq!(docs.len(), 256);
    }

    #[tokio::test]
    async fn test_lo_pdf_loader_reader() {
        let path = "./src/document_loaders/test_data/sample.pdf";
        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        let reader = Cursor::new(buffer);

        let loader = PdfExtractLoader::new(reader).expect("Failed to create PdfExtractLoader");

        let docs = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;

        assert_eq!(&docs[0].page_content[..100], "\n\nSample PDF Document\n\nRobert Maron\nGrzegorz Grudzi´nski\n\nFebruary 20, 1999\n\n2\n\nContents\n\n1 Templat");
        assert_eq!(docs.len(), 1);
    }
}
