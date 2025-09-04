from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.huggingface_api import HuggingFaceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

def setup_llm():
    """Configure le LLM HuggingFace"""
    llm = HuggingFaceAPI(
        model_name="microsoft/DialoGPT-medium",
        token="your_hf_token_here",  # Remplacer par votre token HF
        max_new_tokens=256
    )
    return llm

def setup_embeddings():
    """Configure les embeddings"""
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embed_model

def create_index(data_path="./data/processed/"):
    """Crée l'index à partir des documents"""
    # Configuration des modèles
    llm = setup_llm()
    embed_model = setup_embeddings()
    
    # Configuration globale
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Chargement des documents
    docs = SimpleDirectoryReader(data_path).load_data()
    
    # Création de l'index
    index = VectorStoreIndex.from_documents(docs)
    
    return index.as_query_engine()

def query_llm(query_engine, question):
    """Execute une requête sur le LLM"""
    response = query_engine.query(question)
    return str(response)