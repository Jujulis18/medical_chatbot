import streamlit as st
from src.pipeline.retriever import Retriever
from src.pipeline.generation.llm_service import MISTRALChatGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.modules.sidebar_profile import show_filter_profile
from src.modules.chatbot import show_chatbot
import numpy as np
import os
import sys
from dotenv import load_dotenv

# Empêcher Streamlit de surveiller les fichiers __pycache__
if "__pycache__" not in sys.path:
    sys.path.append("__pycache__")


@st.cache_resource
def init_pipeline():
	# Charger les variables d'environnement
	load_dotenv()
	api_key = os.getenv("MISTRAL_API_KEY")
	if not api_key:
	    raise ValueError("Clé API OpenAI non trouvée. Vérifiez votre fichier .env")


	retriever = Retriever(model_name="sbert")
	generator = MISTRALChatGenerator(api_key=api_key, model="mistral-small-latest")
	return RAGPipeline(retriever, generator)





pipeline = init_pipeline()

my_profile = show_filter_profile()

my_chatbot = show_chatbot(pipeline)

