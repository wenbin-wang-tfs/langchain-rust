use std::{
    collections::HashMap,
    error::Error,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use rusqlite::params;
use serde_json::{json, Value};

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};

pub struct Store {
    pub(crate) pool: Arc<Mutex<rusqlite::Connection>>,
    pub(crate) table: String,
    pub(crate) vector_dimensions: i32,
    pub(crate) embedder: Arc<dyn Embedder>,
}

impl Store {
    pub async fn initialize(&self) -> Result<(), Box<dyn Error>> {
        self.create_table_if_not_exists().await?;
        Ok(())
    }

    async fn create_table_if_not_exists(&self) -> Result<(), Box<dyn Error>> {
        let table = &self.table;
        let db = &self.pool.lock().unwrap();

        db.execute(
            &format!(
                r#"
                CREATE TABLE IF NOT EXISTS {table}
                (
                  rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT,
                  metadata BLOB,
                  text_embedding BLOB
                )
                ;
                "#
            ),
            (),
        )?;

        let dimensions = self.vector_dimensions;

        db.execute(
            &format!(
                r#"
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_{table} USING vec0(
                text_embedding float[{dimensions}],
                );
                "#
            ),
            (),
        )?;

        db.execute(
            &format!(
                r#"
                CREATE TRIGGER IF NOT EXISTS embed_text_{table}
                AFTER INSERT ON {table}
                BEGIN
                    INSERT INTO vec_{table}(rowid, text_embedding)
                    VALUES (new.rowid, new.text_embedding)
                    ;
                END;
                "#
            ),
            (),
        )?;

        Ok(())
    }
}

#[async_trait]
impl VectorStore for Store {
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &VecStoreOptions,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();

        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);

        let vectors = embedder.embed_documents(&texts).await?;
        if vectors.len() != docs.len() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Number of vectors and documents do not match",
            )));
        }

        let table = &self.table;

        let mut db = self.pool.lock().unwrap();
        let tx = db.transaction()?;

        let mut ids = Vec::with_capacity(docs.len());

        for (doc, vector) in docs.iter().zip(vectors.iter()) {
            let text_embedding = json!(&vector).to_string();

            let id: i64 = tx
                .query_row(
                    &format!(
                        r#"
                    INSERT INTO {table}
                        (text, metadata, text_embedding)
                    VALUES
                        (?, ?, ?)
                    RETURNING rowid"#
                    ),
                    params![
                        &doc.page_content,
                        &json!(doc.metadata).to_string(),
                        &text_embedding
                    ],
                    |row| row.get::<_, i64>(0),
                )?
                .try_into()
                .unwrap();

            ids.push(id.to_string());
        }

        tx.commit()?;

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        _opt: &VecStoreOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let table = &self.table;

        let query_vector = self.embedder.embed_query(query).await?;
        let db = self.pool.lock().unwrap();
        let dimensions = self.vector_dimensions;

        let query = &format!(
            r#"WITH coarse_results AS (
                SELECT
                    rowid,
                    text_embedding,
                    distance
                FROM vec_{table}
                WHERE text_embedding MATCH vec_normalize(?)
                ORDER BY distance
                LIMIT 100
            )
            SELECT
                t.text,
                t.metadata,
                c.distance
            FROM {table} t
            INNER JOIN coarse_results c ON t.rowid = c.rowid
            ORDER BY vec_distance_l2(?, c.text_embedding)
            LIMIT ?;"#,
        );
        let mut stmt = db.prepare(query)?;
        let query_vector: Vec<f64> = query_vector
            .iter()
            .take(dimensions as usize)
            .cloned()
            .collect();
        let query_vector_string = json!(query_vector).to_string();
        let docs = stmt
            .query_map(
                params![query_vector_string, query_vector_string, limit.to_string()],
                |row| {
                    let page_content: String = row.get("text")?;
                    let metadata_json: String = row.get("metadata")?;
                    let score: f64 = row.get("distance")?;

                    let metadata: HashMap<String, Value> =
                        serde_json::from_str(&metadata_json).unwrap();

                    Ok(Document {
                        page_content,
                        metadata,
                        score,
                    })
                },
            )?
            .collect::<Result<Vec<Document>, rusqlite::Error>>()?;

        Ok(docs)
    }
}

impl Store {
    pub async fn delete_documents_by_ids(&self, ids: &[i64]) -> Result<(), Box<dyn Error>> {
        if ids.is_empty() {
            return Ok(());
        }

        let table = &self.table;
        let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");

        let mut db = self.pool.lock().unwrap();
        let tx = db.transaction()?;

        let query = format!(
            r#"
                    DELETE FROM {table}
                    WHERE rowid IN ({placeholders})
                "#
        );

        tx.execute(&query, rusqlite::params_from_iter(ids))?;

        let vss_table = format!("vec_{}", table);
        let vss_query = format!(
            r#"
                    DELETE FROM {vss_table}
                    WHERE rowid IN ({placeholders})
                "#
        );

        tx.execute(&vss_query, rusqlite::params_from_iter(ids))?;

        tx.commit()?;

        Ok(())
    }
}
