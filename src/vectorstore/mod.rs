mod options;

#[cfg(feature = "postgres")]
pub mod pgvector;

#[cfg(feature = "sqlite-hybrid")]
pub mod sqlite_hybrid;

#[cfg(feature = "sqlite-vec")]
pub mod sqlite_vec;

#[cfg(feature = "sqlite-bm25")]
pub mod sqlite_bm25;

#[cfg(feature = "surrealdb")]
pub mod surrealdb;

#[cfg(feature = "opensearch")]
pub mod opensearch;

#[cfg(feature = "qdrant")]
pub mod qdrant;

mod vectorstore;

pub use options::*;
pub use vectorstore::*;
