use std::{
    collections::HashMap,
    error::Error,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use rusqlite::params;
use serde_json::{json, Value};

use crate::{
    embedding::embedder_trait::Embedder, language_models::llm::LLM, schemas::Document, vectorstore::{VecStoreOptions, VectorStore}
};

pub struct Store {
    pub(crate) pool: Arc<Mutex<rusqlite::Connection>>,
    pub(crate) table: String,
    pub(crate) vector_dimensions: i32,
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) llm: Arc<dyn LLM>,
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
        Ok(())
    }

    fn get_filters(&self, opt: &VecStoreOptions) -> Result<HashMap<String, Value>, Box<dyn Error>> {
        match &opt.filters {
            Some(Value::Object(map)) => {
                // Convert serde_json Map to HashMap<String, Value>
                let filters = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                Ok(filters)
            }
            None => Ok(HashMap::new()), // No filters provided
            _ => Err("Invalid filters format".into()), // Filters provided but not in the expected format
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
        opt: &VecStoreOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let table = &self.table;
        let openai =  &self.llm;
        let prompt = format!(
            r#"
            {{
            "messages": [
                {{
                "role": "system",
                "content": "Extract keywords user question full-text search fewer keywords ensure full-text search data keywords multiple keywords space separate Note keywords language"
                }},
                {{
                "role": "user",
                "content": "{}"
                }}
            ],
            "temperature": 0.3,
            "max_tokens": 4096
            }}"#,
            query.to_string()
        );
        let ai_bm25_response  = openai.invoke(&prompt).await.unwrap_or_else(|err| {
            eprintln!("prepare sql error: {}", err);
            query.to_string()
        });
        let bm25_query = ai_bm25_response
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect::<String>();
        
        println!("bm25 key word: {}", bm25_query);

        let query_vector_json = json!(self.embedder.embed_query(query).await?).to_string();
        let db = self.pool.lock().unwrap();
        
        let filter = self.get_filters(opt)?;
        let mut metadata_query = filter
        .iter()
        .map(|(k, v)| format!("json_extract(e.metadata, '$.{}') = '{}'", k, v))
        .collect::<Vec<String>>()
        .join(" AND ");

        if metadata_query.is_empty() {
            metadata_query = "1 = 1".to_string();
        }
        
       

        let query_sql = &format!(
            r#"
            with vec_matches as (
                    select
                    rowid,
                    row_number() over (
                    order by distance) as rank_number,
                    distance
                from
                    vec_{table}
                where
                    text_embedding match ?
                    and k = {limit} and  {metadata_query}
                order by
                    distance
                ),
            fts_matches as (
                select
                    rowid,
                    row_number() OVER (ORDER BY rank) AS row_number,
                    rank AS score
                from
                    bm25_{table}
                where
                    bm25_{table} match ?
                    order by score
                    limit {limit}
                ),
            final as (
                    SELECT
                        items.rowid,
                        items.text,
                        items.metadata,
                        vec_matches.distance AS vec_score,
                        fts_matches.score AS bm25_score,
                        COALESCE(1.0 / (60 + fts_matches.row_number), 0.0) * 1.0 +
                        COALESCE(1.0 / (60 + vec_matches.rank_number), 0.0) * 1.0 AS combined_score
                    FROM
                        fts_matches
                    FULL OUTER JOIN vec_matches ON vec_matches.rowid = fts_matches.rowid
                    JOIN {table} items ON COALESCE(fts_matches.rowid, vec_matches.rowid) = items.rowid
                    ORDER BY combined_score 
                )
            select * from final order by combined_score DESC;
            "#,
        );
        let mut stmt = db.prepare(query_sql)?;
        
        let docs = stmt
            .query_map(
                params![query_vector_json,bm25_query],
                |row| {
                    let page_content: String = row.get("text")?;
                    let metadata_json: String = row.get("metadata")?;
                    let score: f64 = row.get::<_, Option<f64>>("combined_score")?.unwrap_or_default();
                    let vec_score: f64 = row.get::<_, Option<f64>>("vec_score")?.unwrap_or_default();
                    let bm25_score: f64 = row.get::<_, Option<f64>>("bm25_score")?.unwrap_or_default();
                    let mut metadata: HashMap<String, Value> =
                        serde_json::from_str(&metadata_json).unwrap();
                    metadata.insert("bm25_score".to_string(), json!(bm25_score));
                    metadata.insert("vec_score".to_string(), json!(vec_score));
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

        let vss_table = format!("vec_{}", table);

        tx.execute(
            &format!(
                r#"
                    DELETE FROM {vss_table}
                "#
            ),
            (),
        )?;

        tx.commit()?;

        Ok(())
    }
}
