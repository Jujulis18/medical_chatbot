# chatbot/embedding/sbert_embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

class SBERTEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype('float32')  # pour FAISS
