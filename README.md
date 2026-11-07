# 🔧 MaintiDoc : assistant RAG agentique multimodal

MaintiDoc aide les techniciens de maintenance. Il répond à des questions techniques à partir d'un manuel PDF (texte, tableaux, schémas) et cite les pages sources.

Projet réalisé à l'EMSI Casablanca (4IADA), TP d'IA générative, étape 1 : RAG agentique multimodal.

## Fonctionnalités
- Extraction du texte, des tableaux et des images du PDF
- Indexation vectorielle dans PostgreSQL (PGVector)
- Descriptions automatiques des figures par un modèle de vision
- Recherche d'images avec CLIP
- Réponses en français, fondées sur le document, avec les pages sources
- Agent qui choisit lui-même l'outil de recherche adapté
- Mémoire de la conversation
- Interface web Streamlit qui affiche aussi les images trouvées

## Architecture

```text
                 PDF de maintenance
                        |
                  extract.py (PyMuPDF)
          +-------------+--------------+
          |             |              |
        Texte        Tableaux        Images
          |             |              |
          |             |      +-------+--------+
          |             |      |                |
          |             |   qwen2.5vl:3b      CLIP
          |             |   (descriptions)   (vecteurs 512)
          |             |      |                |
          v             v      v                v
     nomic-embed-text (768)                clip_images
          |
     PGVector : table "maintenance"
          |
          +-------- agent LangChain (qwen2.5:7b) --------+
          |   outils : chercher_texte                    |
          |            chercher_figures                  |
          |            chercher_images (CLIP)            |
          +----------------------+-----------------------+
                                 |
                          Interface Streamlit
```

## Technologies
| Rôle | Outil |
|---|---|
| Extraction et indexation | LlamaIndex, PyMuPDF |
| Base vectorielle | PostgreSQL + PGVector (Docker) |
| Embeddings texte | nomic-embed-text (Ollama) |
| LLM | qwen2.5:7b (Ollama) |
| Description des images | qwen2.5vl:3b (Ollama) |
| Recherche d'images | CLIP (openai/clip-vit-base-patch32) |
| Agent | LangChain / LangGraph |
| Interface | Streamlit |

## Prérequis
- Python 3.11 ou plus
- Docker Desktop
- Ollama
- Environ 10 Go d'espace disque pour les modèles

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

Le port 5433 évite un conflit avec un PostgreSQL déjà installé sur le PC.

## Utilisation

1. Placer le PDF dans `data/maintenance.pdf`.
2. Indexer le texte et les tableaux : `python index.py`
3. Décrire et indexer les pages : `python describe_images.py` (long sur CPU)
4. Indexer les images avec CLIP : `python clip_index.py`
5. Tester le RAG simple : `python rag.py`
6. Tester l'agent en console : `python agent.py`
7. Lancer l'interface : `streamlit run app.py`

## Exemples de questions
- Qu'est-ce que le MTBF ?
- Quelle est la différence entre maintenance corrective et préventive ?
- Que montre la figure 22 ?
- Montre-moi une image de pompe à combustible.

## Structure du projet

```text
projet_academique/
├── data/
│   ├── maintenance.pdf
│   ├── descriptions.json   # cache des descriptions d'images
│   └── images/             # images extraites (non versionnées)
├── extract.py              # extraction texte, tableaux, images
├── index.py                # indexation texte + tableaux
├── describe_images.py      # descriptions des figures
├── clip_index.py           # indexation CLIP
├── clip_search.py          # recherche d'images
├── rag.py                  # pipeline RAG
├── agent.py                # agent LangChain
├── app.py                  # interface Streamlit
├── requirements.txt
└── README.md
```

## Limites connues
- Les schémas techniques sont plus difficiles pour CLIP que les photos.
- Les descriptions d'images dépendent d'un petit modèle (3 milliards de paramètres) et peuvent contenir des erreurs.
- Les requêtes d'images fonctionnent mieux en anglais.
- Le chargement d'un nouveau PDF dans l'interface demande de relancer l'indexation à la main.

## Suite du projet MaintiDoc
- Étape 2 : fine-tuning
- Étape 3 : orchestration agentique et MCP (historique des interventions, préparation de tickets avec validation humaine)
- Étape 4 : client web, Telegram/Discord, sécurité et déploiement

## Auteur
Elmer, EMSI Casablanca, 4IADA.

## Source du document
« Maintenance des machines et des moteurs », G. Duchemin, Techniques de l'Ingénieur (BM 4 188). Document utilisé à titre pédagogique.
