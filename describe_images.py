import json, os
import ollama
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from extract import render_pages

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = None

vector_store = PGVectorStore.from_params(
    database="ragdb", host="localhost", port=5433,
    user="postgres", password="postgres",
    table_name="maintenance", embed_dim=768,
)

PROMPT = (
    "Tu analyses une page d'un manuel de maintenance de moteurs. "
    "Décris en français chaque figure et chaque tableau de la page : "
    "numéro de la figure, titre, composants visibles, valeurs et schémas. "
    "Sois précis et factuel."
)

CACHE = "data/descriptions.json"

def main():
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    pages = render_pages()
    docs = []
    for p in pages:
        key = str(p["page"])
        if key not in cache:
            print(f"Description de la page {key}...")
            r = ollama.chat(
                model="qwen2.5vl:3b",
                messages=[{"role": "user", "content": PROMPT, "images": [p["path"]]}],
            )
            cache[key] = r["message"]["content"]
            json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        docs.append(Document(
            text=cache[key],
            metadata={"page": p["page"], "type": "image", "image_path": p["path"]},
        ))

    storage = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    print("Descriptions indexées")

if __name__ == "__main__":
    main()