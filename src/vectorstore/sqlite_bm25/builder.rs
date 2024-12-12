use std::{
    error::Error,
    sync::{Arc, Mutex},
};

use rusqlite::Result;

use super::Store;

pub struct StoreBuilder {
    connection_url: Option<String>,
    table: Option<String>,
}

impl StoreBuilder {
    pub fn new() -> Self {
        Self {
            connection_url: None,
            table: None,
        }
    }

    pub fn connection_url(mut self, url: impl Into<String>) -> Self {
        self.connection_url = Some(url.into());
        self
    }

    pub fn table(mut self, table: impl Into<String>) -> Self {
        self.table = Some(table.into());
        self
    }

    pub async fn build(self) -> Result<Store, Box<dyn Error>> {
        let connection_url = self.connection_url.ok_or("Connection URL is required")?;
        let table = self.table.ok_or("Table name is required")?;

        let conn = rusqlite::Connection::open(connection_url)?;
        let pool = Arc::new(Mutex::new(conn));

        Ok(Store { pool, table })
    }
}
