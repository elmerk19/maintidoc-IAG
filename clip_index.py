import glob, re, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
#clip_index.py permet d'indexer les images extraites d'un document PDF de maintenance en utilisant le modèle CLIP. 
#Il génère des vecteurs de représentation pour chaque image et les stocke dans une base de données PostgreSQL pour permettre des recherches basées sur le contenu visuel des images.

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

# Table séparée : CLIP produit des vecteurs de taille 512
store = PGVectorStore.from_params(
    database="ragdb", host="localhost", port=5433,
    user="postgres", password="postgres",
    table_name="clip_images", embed_dim=512,
)

nodes = []
for path in sorted(glob.glob("data/images/*.png")):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = to_tensor(model.get_image_features(**inputs), model.visual_projection)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    page = int(re.search(r"page(\d+)", path).group(1))
    nodes.append(TextNode(
        text=path,
        metadata={"type": "clip", "image_path": path, "page": page},
        embedding=emb[0].tolist(),
    ))
    print("Image indexée :", path)

store.add(nodes)
print(f"{len(nodes)} images indexées avec CLIP")