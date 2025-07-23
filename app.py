import streamlit as st
from src.retrieval.retriever import Retriever
from src.generation.llm_service import MISTRALChatGenerator
from src.chatbot.rag_pipeline import RAGPipeline
import numpy as np
import os
import sys
from dotenv import load_dotenv

# Empêcher Streamlit de surveiller les fichiers __pycache__
if "__pycache__" not in sys.path:
    sys.path.append("__pycache__")

# Initialisation (peut être déplacée dans un cache pour améliorer les performances)
@st.cache_resource
def init_pipeline():

	# Charger les variables d'environnement
	load_dotenv()

	# Récupérer la clé API
	api_key = os.getenv("MISTRAL_API_KEY")

	if not api_key:
	    raise ValueError("Clé API OpenAI non trouvée. Vérifiez votre fichier .env")

	retriever = Retriever(model_name="sbert")
	generator = MISTRALChatGenerator(api_key=api_key, model="mistral-small-latest")
	return RAGPipeline(retriever, generator)

pipeline = init_pipeline()

print(repr(os.getenv("MISTRAL_API_KEY")), flush=True)

# Interface utilisateur
st.set_page_config(page_title="Chatbot Médical RAG")
st.title("Chatbot Médical basé sur RAG")
st.warning("⚠️ This chatbot is available in English only.")


# Historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []


# Messages du chatbot (user + assistant)
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Message d'accueil assistant
with st.chat_message("assistant"):
    st.write("Hello I'm your medical assistant, How can I help you today?")



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
                with st.expander("Informations de debug"):
                    for info in debug_info:
                        st.text(info)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"ERREUR: {str(e)}")
                import traceback
                st.code(traceback.format_exc())



