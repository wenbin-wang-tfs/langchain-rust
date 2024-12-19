use std::{
    collections::HashMap,
    error::Error,
    sync::{Arc, Mutex},
};

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};
use async_trait::async_trait;
use rusqlite::params;
use serde_json::{json, Value};

pub struct Store {
    pub(crate) pool: Arc<Mutex<rusqlite::Connection>>,
    pub(crate) table: String,
    pub(crate) vector_dimensions: i32,
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) batch_size: i32,
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

        db.execute(
            &format!(
                r#"
                CREATE VIRTUAL TABLE IF NOT EXISTS bm25_{table}
                USING fts5(
                  text,
                  metadata,
                );
                "#
            ),
            (),
        )?;

        db.execute(
            &format!(
                r#"
                CREATE TRIGGER IF NOT EXISTS bm25_{table}_insert_trigger
                AFTER INSERT ON {table}
                BEGIN
                    INSERT INTO bm25_{table} (rowid, text,metadata)
                    VALUES (new.rowid, new.text, new.metadata)
                    ;
                END;
                "#
            ),
            (),
        )?;

        db.execute(
            &format!(
                r#"
                CREATE TRIGGER IF NOT EXISTS bm25_{table}_delete_trigger
                AFTER DELETE ON {table}
                BEGIN
                    DELETE FROM bm25_{table} WHERE rowid = old.rowid;
                END;

                "#
            ),
            (),
        )?;

        db.execute(
            &format!(
                r#"
                CREATE TRIGGER IF NOT EXISTS vec_{table}_delete_trigger
                AFTER DELETE ON {table}
                BEGIN
                    DELETE FROM vec_{table} WHERE rowid = old.rowid;
                END;
                "#
            ),
            (),
        )?;
        Ok(())
    }

    fn get_filters(&self, opt: &VecStoreOptions) -> Result<HashMap<String, Value>, Box<dyn Error>> {
        match &opt.filters {
            Some(Value::Object(map)) => {
                let filters = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                Ok(filters)
            }
            None => Ok(HashMap::new()),
            _ => Err("Invalid filters format".into()),
        }
    }

    pub async fn delete_documents_by_metadata(
        &self,
        metadata_filters: &HashMap<String, Value>,
    ) -> Result<(), Box<dyn Error>> {
        if metadata_filters.is_empty() {
            return Ok(());
        }

        let table = &self.table;
        let mut db = self.pool.lock().unwrap();
        let tx = db.transaction()?;

        // Build metadata filter conditions
        let metadata_conditions = metadata_filters
            .iter()
            .map(|(k, v)| match v {
                Value::Array(arr) => {
                    let values: Vec<String> =
                        arr.iter().map(|val| json!(val).to_string()).collect();
                    format!(
                        "json_extract(metadata, '$.{}') IN ({})",
                        k,
                        values.join(",")
                    )
                }
                Value::String(s) => {
                    let json_value = json!(s).to_string();
                    format!("json_extract(metadata, '$.{}') = {}", k, json_value)
                }
                Value::Number(n) => {
                    format!("json_extract(metadata, '$.{}') = {}", k, n)
                }
                Value::Bool(b) => {
                    format!("json_extract(metadata, '$.{}') = {}", k, b)
                }
                _ => {
                    let json_value = json!(v).to_string();
                    format!("json_extract(metadata, '$.{}') = {}", k, json_value)
                }
            })
            .collect::<Vec<String>>()
            .join(" AND ");

        // Delete from main table
        tx.execute(
            &format!(
                r#"DELETE FROM {table}
                WHERE {}"#,
                metadata_conditions
            ),
            (),
        )?;

        tx.commit()?;
        Ok(())
    }

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
        tx.commit()?;

        Ok(())
    }

    pub async fn delete_all_documents(&self) -> Result<(), Box<dyn Error>> {
        let table = &self.table;

        let mut db = self.pool.lock().unwrap();
        let tx = db.transaction()?;

        tx.execute(
            &format!(
                r#"
                    DELETE FROM {table}
                "#
            ),
            (),
        )?;

        tx.commit()?;

        Ok(())
    }

    pub async fn keyword_search(
        &self,
        query: &str,
        limit: usize,
        opt: &VecStoreOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let table = format!("bm25_{}", self.table);
        let filter = self.get_filters(opt)?;
        let db = self.pool.lock().unwrap();

        let metadata_query = self.build_metadata_query(&filter, None);

        let mut stmt = db.prepare(&format!(
            r#"
            SELECT
                text,
                metadata,
                bm25({table}) as score
            FROM {table}
            WHERE {table} MATCH ?1 AND {metadata_query}
            ORDER BY score DESC
            LIMIT ?2
            "#
        ))?;

        let docs = stmt
            .query_map(params![query, limit as i64], |row| {
                let page_content: String = row.get(0)?;
                let metadata_json: String = row.get(1)?;
                let raw_score: f64 = row.get(2)?;

                // 将 BM25 分数转换为 0-1 范围
                // BM25 分数通常���正数，越大表示越相关
                // 使用 sigmoid 函数进行归一化: 1 / (1 + e^(-score))
                let score = 1.0 / (1.0 + (-raw_score).exp());

                let metadata: HashMap<String, Value> =
                    serde_json::from_str(&metadata_json).unwrap();

                Ok(Document {
                    page_content,
                    metadata,
                    score,
                })
            })?
            .collect::<Result<Vec<Document>, rusqlite::Error>>()?;

        Ok(docs)
    }

    fn build_metadata_query(
        &self,
        filter: &HashMap<String, Value>,
        table_prefix: Option<&str>,
    ) -> String {
        let prefix = table_prefix.unwrap_or("");
        let metadata_path = if prefix.is_empty() {
            "metadata".to_string()
        } else {
            format!("{}.metadata", prefix)
        };

        let query = filter
            .iter()
            .map(|(k, v)| match v {
                Value::Array(arr) => {
                    let values: Vec<String> =
                        arr.iter().map(|val| json!(val).to_string()).collect();
                    format!(
                        "json_extract({}, '$.{}') IN ({})",
                        metadata_path,
                        k,
                        values.join(",")
                    )
                }
                Value::String(s) => {
                    let json_value = json!(s).to_string();
                    format!(
                        "json_extract({}, '$.{}') = {}",
                        metadata_path, k, json_value
                    )
                }
                Value::Number(n) => {
                    format!("json_extract({}, '$.{}') = {}", metadata_path, k, n)
                }
                Value::Bool(b) => {
                    format!("json_extract({}, '$.{}') = {}", metadata_path, k, b)
                }
                _ => {
                    let json_value = json!(v).to_string();
                    format!(
                        "json_extract({}, '$.{}') = {}",
                        metadata_path, k, json_value
                    )
                }
            })
            .collect::<Vec<String>>()
            .join(" AND ");

        if query.is_empty() {
            "1=1".to_string()
        } else {
            query
        }
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

        let batch_size = self.batch_size as usize;
        let mut batches = texts.chunks(batch_size);

        let mut vectors = Vec::with_capacity(docs.len());

        while let Some(batch) = batches.next() {
            let vector = embedder.embed_documents(batch).await?;
            vectors.extend(vector);
        }

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
        opt: &VecStoreOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let table = &self.table;
        let query_vector_json = json!(self.embedder.embed_query(query).await?).to_string();
        let db = self.pool.lock().unwrap();

        let filter = self.get_filters(opt)?;
        let metadata_query = self.build_metadata_query(&filter, Some("e"));

        let mut stmt = db.prepare(&format!(
            r#"SELECT
                e.text,
                e.metadata,
                v.distance
            FROM {table} e
            INNER JOIN vec_{table} v on v.rowid = e.rowid
            WHERE v.text_embedding match ?1 AND k = ?2 AND {metadata_query}
            ORDER BY distance
            LIMIT ?3"#
        ))?;

        let doubled_limit = limit * 2;
        let docs = stmt
            .query_map(
                params![query_vector_json, limit as i32, doubled_limit as i32],
                |row| {
                    let page_content: String = row.get(0)?;
                    let metadata_json: String = row.get(1)?;
                    let distance: f64 = row.get(2)?;
                    let score = 1.0 / (1.0 + distance);
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

        let mut seen = std::collections::HashSet::new();
        let mut unique_docs: Vec<Document> = docs
            .into_iter()
            .filter(|doc| {
                let key = format!("{}{}", doc.page_content, json!(doc.metadata));
                seen.insert(key)
            })
            .collect();

        unique_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        unique_docs.truncate(limit);

        Ok(unique_docs)
    }
}
