use async_trait::async_trait;
use rusqlite::{params, params_from_iter};
use serde_json::{json, Value};
use sqlx::{Pool, Row, Sqlite};
use std::{
    collections::HashMap,
    error::Error,
    sync::{Arc, Mutex},
};

use crate::{
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};

pub struct Store {
    pub pool: Arc<Mutex<rusqlite::Connection>>,
    pub(crate) table: String,
}

impl Store {
    pub async fn initialize(&self) -> Result<(), Box<dyn Error>> {
        self.create_table_if_not_exists().await?;
        Ok(())
    }

    async fn create_table_if_not_exists(&self) -> Result<(), Box<dyn Error>> {
        let table = &self.table;
        let db = self.pool.lock().unwrap();

        db.execute(
            &format!(
                r#"
                CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5(
                    text,
                    metadata UNINDEXED
                );"#
            ),
            [],
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

    pub async fn delete_documents_by_ids(&self, ids: &[i64]) -> Result<(), Box<dyn Error>> {
        if ids.is_empty() {
            return Ok(());
        }

        let table = &self.table;
        let placeholders = (1..=ids.len())
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(",");

        let db = self.pool.lock().unwrap();
        db.execute(
            &format!(r#"DELETE FROM {table} WHERE rowid IN ({placeholders})"#),
            params_from_iter(ids),
        )?;

        Ok(())
    }

    pub async fn delete_documents_by_metadata(
        &self,
        metadata_filters: &HashMap<String, Value>,
    ) -> Result<(), Box<dyn Error>> {
        let table = &self.table;
        let db = self.pool.lock().unwrap();

        let metadata_query = metadata_filters
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

        let where_clause = if metadata_query.is_empty() {
            "1=1".to_string()
        } else {
            metadata_query
        };

        db.execute(&format!(r#"DELETE FROM {table} WHERE {where_clause}"#), [])?;

        Ok(())
    }

    pub async fn delete_all_documents(&self) -> Result<(), Box<dyn Error>> {
        let table = &self.table;
        let db = self.pool.lock().unwrap();
        db.execute(&format!(r#"DELETE FROM {table}"#), [])?;
        Ok(())
    }
}

#[async_trait]
impl VectorStore for Store {
    async fn add_documents(
        &self,
        docs: &[Document],
        _opt: &VecStoreOptions,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let table = &self.table;
        let mut db = self.pool.lock().unwrap();
        let tx = db.transaction()?;
        let mut ids = Vec::with_capacity(docs.len());

        for doc in docs {
            let id: i64 = tx.query_row(
                &format!(
                    r#"
                    INSERT INTO {table}
                        (text, metadata)
                    VALUES
                        (?1, ?2)
                    RETURNING rowid"#
                ),
                params![&doc.page_content, json!(&doc.metadata).to_string()],
                |row| row.get(0),
            )?;

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
        let filter = self.get_filters(opt)?;
        let db = self.pool.lock().unwrap();

        let mut metadata_query = filter
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

        if metadata_query.is_empty() {
            metadata_query = "1=1".to_string();
        }

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
                // BM25 分数通常是正数，越大表示越相关
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
}
