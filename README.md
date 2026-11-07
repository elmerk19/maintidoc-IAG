# MaintiDoc : RAG agentique multimodal

Assistant pour techniciens de maintenance. Il répond aux questions sur un PDF
(texte, tableaux, schémas) avec les pages sources.

## Technologies
- LlamaIndex + PGVector (PostgreSQL dans Docker)
- Embeddings : nomic-embed-text (Ollama)
- LLM : qwen2.5:7b (Ollama)
- Descriptions des images : qwen2.5vl:3b (Ollama)
- CLIP pour la recherche d'images
- Agent LangChain (LangGraph) avec 3 outils
- Interface Streamlit

## Installation
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama pull qwen2.5vl:3b
docker run -d --name pgvector -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragdb -p 5433:5432 pgvector/pgvector:pg16
```

## Utilisation
1. Mettre le PDF dans `data/maintenance.pdf`
2. `python index.py` : indexe le texte et les tableaux
3. `python describe_images.py` : décrit et indexe les pages
4. `python clip_index.py` : indexe les images avec CLIP
5. `streamlit run app.py` : lance l'interface

## Fichiers
- `extract.py` : extraction du texte, des tableaux et des images
- `index.py`, `describe_images.py`, `clip_index.py` : indexation
- `rag.py` : pipeline RAG
- `clip_search.py` : recherche d'images
- `agent.py` : agent avec outils de recherche
- `app.py` : interface Streamlit
