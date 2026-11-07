import os, re
import streamlit as st
from agent import agent
#app.py permet de créer une interface web pour interagir avec le modèle de question-réponse basé sur le document PDF indexé. Il utilise Streamlit pour l'interface utilisateur et permet aux utilisateurs de poser des questions sur le contenu du PDF.

st.set_page_config(page_title="MaintiDoc", page_icon="🔧")
st.title("🔧 MaintiDoc")
st.caption("Assistant de maintenance : posez une question sur le document.")

# Chargement du PDF
os.makedirs("data", exist_ok=True)
uploaded = st.sidebar.file_uploader("Charger un PDF", type="pdf")
if uploaded:
    with open("data/maintenance.pdf", "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success("PDF enregistré. Relancez index.py et describe_images.py pour l'indexer.")

# Historique de la conversation
if "history" not in st.session_state:
    st.session_state.history = []   # liste de (role, texte, images)

for role, text, images in st.session_state.history:
    with st.chat_message(role):
        st.write(text)
        for path in images:
            st.image(path, caption=path)

# Question
question = st.chat_input("Votre question")

if question:
    with st.chat_message("user"):
        st.write(question)

    # Contexte envoyé à l'agent
    messages = [(r, t) for r, t, _ in st.session_state.history] + [("user", question)]

    with st.spinner("Recherche en cours..."):
        result = agent.invoke({"messages": messages})

    answer = result["messages"][-1].content

    # Images retrouvées par l'outil CLIP
    images = []
    for m in result["messages"]:
        if getattr(m, "type", "") == "tool":
            for p in re.findall(r"data/images/[^\s]+\.png", str(m.content)):
                if p not in images and os.path.exists(p):
                    images.append(p)

    with st.chat_message("assistant"):
        st.write(answer)
        for path in images:
            st.image(path, caption=path)

    st.session_state.history.append(("user", question, []))
    st.session_state.history.append(("assistant", answer, images))