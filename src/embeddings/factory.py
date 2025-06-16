# chatbot/embedding/factory.py

from src.embeddings.openai_embedder import OpenAIEmbedder
from src.embeddings.sbert_embedder import SBERTEmbedder
import numpy as np

def get_embedder(model_name: str = "sbert"):
    """
    Retourne une instance de l'embedder selon le modèle choisi.
    - model_name: "sbert" | "openai"
    """
    if model_name == "sbert":
        return SBERTEmbedder(model_name='all-MiniLM-L6-v2')
    elif model_name == "openai":
        return OpenAIEmbedder(model_name='text-embedding-3-small')  # ou autre
    else:
        raise ValueError(f"Embedder inconnu : {model_name}")
