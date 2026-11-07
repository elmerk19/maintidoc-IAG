import torch
from transformers import CLIPModel, CLIPProcessor
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.postgres import PGVectorStore
#clip_search.py permet de rechercher des images dans un document PDF de maintenance en utilisant le modèle CLIP. 
#Il génère un vecteur de représentation pour une question textuelle et interroge la base de données PostgreSQL contenant les vecteurs d'images indexés pour trouver les images les plus pertinentes correspondant à la question.

MODEL = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL)
processor = CLIPProcessor.from_pretrained(MODEL)

def to_tensor(out, projection):
    # Selon la version de transformers, on reçoit un tenseur ou un objet
    if not isinstance(out, torch.Tensor):
        out = out.pooler_output
    if out.shape[-1] != 512:
        out = projection(out)
    return out

store = PGVectorStore.from_params(
    database="ragdb", host="localhost", port=5433,
    user="postgres", password="postgres",
    table_name="clip_images", embed_dim=512,
)

def search_images(question, k=3):
    inputs = processor(text=[question], return_tensors="pt", padding=True)
    with torch.no_grad():
         emb = to_tensor(model.get_text_features(**inputs), model.text_projection)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    res = store.query(VectorStoreQuery(query_embedding=emb[0].tolist(), similarity_top_k=k))
    return [(n.metadata["page"], n.metadata["image_path"]) for n in res.nodes]

if __name__ == "__main__":
    for page, path in search_images("a diesel engine with filters and pumps"):
        print("Page", page, "->", path)