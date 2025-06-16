import streamlit as st
from src.retrieval.retriever import Retriever
from src.generation.llm_service import OpenAIChatGenerator
from src.chatbot.rag_pipeline import RAGPipeline

# Initialisation (peut être déplacée dans un cache pour améliorer les performances)
@st.cache_resource
def init_pipeline():
    retriever = Retriever(model_name="sbert")
    generator = OpenAIChatGenerator(api_key="your-key", model="gpt-4")
    return RAGPipeline(retriever, generator)

pipeline = init_pipeline()

# Interface utilisateur
st.set_page_config(page_title="Chatbot Médical RAG")
st.title("🧠 Chatbot Médical basé sur RAG")

# Historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Champ de saisie utilisateur
user_input = st.chat_input("Posez votre question médicale...")

if user_input:
    # Ajout à l'historique
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des informations..."):
            try:
                response, debug_info = pipeline.run(user_input)
                
                # Affichage du debug
                with st.expander("🔍 Informations de debug"):
                    for info in debug_info:
                        st.text(info)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"ERREUR: {str(e)}")
                import traceback
                st.code(traceback.format_exc())