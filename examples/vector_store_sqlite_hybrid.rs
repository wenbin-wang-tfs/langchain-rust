// To run this example execute: cargo run --example vector_store_sqlite_vss --features sqlite-vss
// Make sure vector0 and vss0 libraries are installed in the system or the path of the executable.
// Download the libraries from https://github.com/asg017/sqlite-vss
// For static compilation of sqlite-vss extension refer to the following link:
// https://github.com/launchbadge/sqlx/issues/3147.

#[cfg(feature = "sqlite-hybrid")]
use langchain_rust::{
    embedding::openai::openai_embedder::OpenAiEmbedder,
    schemas::Document,
    vectorstore::{sqlite_hybrid::StoreBuilder, VecStoreOptions, VectorStore},
};
#[cfg(feature = "sqlite-hybrid")]
use std::io::Write;

#[cfg(feature = "sqlite-hybrid")]
#[tokio::main]
async fn main() {
    // Initialize Embedder

    use langchain_rust::language_models::llm::LLM;
    use langchain_rust::llm::{AzureConfig, OpenAI};

    let azure_config = AzureConfig::default();

    let embedder = OpenAiEmbedder::new(azure_config);

    let azure_config = AzureConfig::default();

    let open_ai = OpenAI::new(azure_config);

    let database_url = std::env::var("DATABASE_URL").unwrap_or("sqlite::memory:".to_string());

    // Initialize the Sqlite Vector Store
    let store = StoreBuilder::new()
        .embedder(embedder)
        .llm(open_ai)
        .connection_url(database_url)
        .table("table_name")
        .vector_dimensions(3072)
        .build()
        .await
        .unwrap();

    // Initialize the tables in the database. This is required to be done only once.
    store.initialize().await.unwrap();

    // Add documents to the database
    let doc1 = Document::new(
        "langchain-rust is a port of the langchain python library to rust and was written in 2024.",
    );
    let doc2 = Document::new(
        "langchaingo is a port of the langchain python library to go language and was written in 2023."
    );
    let doc3 = Document::new(
        "Capital of United States of America (USA) is Washington D.C. and the capital of France is Paris."
    );
    let doc4 = Document::new("Capital of France is Paris.");

    store
        .add_documents(&vec![doc1, doc2, doc3, doc4], &VecStoreOptions::default())
        .await
        .unwrap();

    // Ask for user input
    print!("Query> ");
    // std::io::stdout().flush().unwrap();
    // let mut query = String::new();
    // std::io::stdin().read_line(&mut query).unwrap();

    let results = store
        .similarity_search(
            "how can i use langchain rust",
            4,
            &VecStoreOptions::default(),
        )
        .await
        .unwrap_or_else(|err| {
            eprintln!("Error during similarity search: {}", err);
            std::process::exit(1);
        });

    if results.is_empty() {
        println!("No results found.");
        return;
    } else {
        results.iter().for_each(|r| {
            println!("Document: {},combined_score:{}", r.page_content, r.score);
            if let Some(vec_score) = r.metadata.get("vec_score") {
                println!("combined_score: {}", vec_score);
            }
            if let Some(bm25_score) = r.metadata.get("bm25_score") {
                println!("bm25_score: {}", bm25_score);
            }
        });
    }
}

#[cfg(not(feature = "sqlite-hybrid"))]
fn main() {
    println!("This example requires the 'sqlite-vss' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example vector_store_sqlite_vss --features sqlite-vss");
}
