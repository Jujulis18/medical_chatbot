from src.embeddings.openai_embedder import OpenAIEmbedder
from src.embeddings.sbert_embedder import SBERTEmbedder
import numpy as np

def get_embedder(model_name: str = "sbert"):
    
    if model_name == "sbert":
        return SBERTEmbedder(model_name='all-MiniLM-L6-v2')
    elif model_name == "openai":
        return OpenAIEmbedder(model_name='text-embedding-3-small')  # ou autre
    else:
        raise ValueError(f"Embedder inconnu : {model_name}")
