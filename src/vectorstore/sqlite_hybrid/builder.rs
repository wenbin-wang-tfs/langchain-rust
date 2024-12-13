use std::{
    error::Error,
    sync::{Arc, Mutex},
};

use rusqlite::{ffi::sqlite3_auto_extension, Connection, Result};
use sqlite_vec::sqlite3_vec_init;

use super::Store;
use crate::{embedding::embedder_trait::Embedder, language_models::llm::LLM};

pub struct StoreBuilder {
    pool: Option<Arc<Mutex<rusqlite::Connection>>>,
    connection_url: Option<String>,
    table: String,
    vector_dimensions: i32,
    embedder: Option<Arc<dyn Embedder>>,
}

impl StoreBuilder {
    pub fn new() -> Self {
        StoreBuilder {
            pool: None,
            connection_url: None,
            table: "documents".to_string(),
            vector_dimensions: 0,
            embedder: None,
        }
    }

    pub fn pool(mut self, pool: Arc<Mutex<rusqlite::Connection>>) -> Self {
        self.pool = Some(pool);
        self.connection_url = None;
        self
    }

    pub fn connection_url<S: Into<String>>(mut self, connection_url: S) -> Self {
        self.connection_url = Some(connection_url.into());
        self.pool = None;
        self
    }

    pub fn table(mut self, table: &str) -> Self {
        self.table = table.into();
        self
    }

    pub fn vector_dimensions(mut self, vector_dimensions: i32) -> Self {
        self.vector_dimensions = vector_dimensions;
        self
    }

    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    pub async fn build(self) -> Result<Store, Box<dyn Error>> {
        if self.embedder.is_none() {
            return Err("Embedder is required".into());
        }

        Ok(Store {
            pool: self.get_pool().await?,
            table: self.table,
            vector_dimensions: self.vector_dimensions,
            embedder: self.embedder.unwrap(),
        })
    }

    async fn get_pool(&self) -> Result<Arc<Mutex<rusqlite::Connection>>, Box<dyn Error>> {
        if let Some(pool) = &self.pool {
            return Ok(pool.clone());
        }

        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
        }

        let connection_url = self
            .connection_url
            .as_ref()
            .ok_or_else(|| "Connection URL or DB is required")?;

        let pool: rusqlite::Connection = Connection::open(connection_url)
            .map_err(|e| format!("Failed to open SQLite connection: {}", e))?;

        let pool = Arc::new(Mutex::new(pool));

        Ok(pool)
    }
}
