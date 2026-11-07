from importlib import import_module
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import    OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

Ollama = import_module("llama_index.llms.ollama").Ollama

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="qwen2.5:7b", request_timeout=300)

vector_store = PGVectorStore.from_params(
    database="ragdb", host="localhost", port=5433,
    user="postgres", password="postgres",
    table_name="maintenance", embed_dim=768,
)

index = VectorStoreIndex.from_vector_store(vector_store)

PROMPT = (
    "Réponds en français, uniquement à partir du contexte fourni. "
    "Si l'information n'y est pas, dis-le. "
    "Cite la page source pour chaque information."
)

query_engine = index.as_query_engine(similarity_top_k=5, system_prompt=PROMPT)

def ask(question):
    r = query_engine.query(question)
    pages = sorted({n.metadata.get("page") for n in r.source_nodes})
    return str(r), pages, r.source_nodes

if __name__ == "__main__":
    answer, pages, _ = ask("Quelle est la différence entre maintenance corrective et préventive ?")
    print(answer)
    print("Pages sources :", pages)