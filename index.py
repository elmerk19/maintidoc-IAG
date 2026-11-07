from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from extract import extract_all

# Modèle d'embeddings (open source, via Ollama)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = None  # pas de LLM nécessaire pour indexer

# Connexion à PGVector (Docker)
vector_store = PGVectorStore.from_params(
    database="ragdb",
    host="localhost",
    port=5433,
    user="postgres",
    password="postgres",
    table_name="maintenance",
    embed_dim=768,  # taille des vecteurs de nomic-embed-text
)

def main():
    text_docs, table_docs, _ = extract_all()
    docs = text_docs + table_docs
    print(f"{len(docs)} documents à indexer")

    storage = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage,
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50)],
        show_progress=True,
    )
    print("Indexation terminée")

if __name__ == "__main__":
    main()