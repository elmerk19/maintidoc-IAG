from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
#agent.py permet de créer un agent conversationnel qui utilise les outils de recherche pour répondre aux questions sur le document PDF de maintenance. Il combine les fonctionnalités de recherche dans le texte, les tableaux et les descriptions d'images pour fournir des réponses précises et contextualisées.
from rag import index
from clip_search import search_images

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

def _search(question, type_):
    filters = MetadataFilters(filters=[MetadataFilter(key="type", value=type_)])
    engine = index.as_retriever(similarity_top_k=4, filters=filters)
    nodes = engine.retrieve(question)
    return "\n\n".join(
        f"[Page {n.metadata.get('page')}] {n.text[:600]}" for n in nodes
    )

@tool
def chercher_texte(question: str) -> str:
    """Cherche dans le texte et les tableaux du manuel de maintenance."""
    return _search(question, "texte") + "\n\n" + _search(question, "tableau")

@tool
def chercher_figures(question: str) -> str:
    """Cherche dans les descriptions des figures et schémas du manuel."""
    return _search(question, "image")

@tool
def chercher_images(description: str) -> str:
    """Cherche des images du manuel proches d'une description. Écrire la description en anglais."""
    res = search_images(description, k=3)
    return "\n".join(f"Page {p} : {path}" for p, path in res)

SYSTEM = (
    "Tu es MaintiDoc, assistant pour techniciens de maintenance. "
    "Utilise les outils pour répondre, uniquement à partir de leurs résultats. "
    "Réponds en français et cite les pages sources. "
    "Si l'information est absente, dis-le."
)

agent = create_react_agent(llm, [chercher_texte, chercher_figures, chercher_images], prompt=SYSTEM)

if __name__ == "__main__":
    history = []
    while True:
        q = input("Question (vide pour quitter) : ")
        if not q:
            break
        history.append(("user", q))
        result = agent.invoke({"messages": history})
        answer = result["messages"][-1].content
        history.append(("assistant", answer))
        print("\n", answer, "\n")