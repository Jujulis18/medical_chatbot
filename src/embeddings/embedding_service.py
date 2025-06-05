from abc import ABC, abstractmethod
from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer
SENTENCE_TRANSFORMERS_AVAILABLE = True


class EmbeddingService(ABC):
    """Service abstrait pour générer des embeddings"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Générer l'embedding d'un texte"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Générer les embeddings d'une liste de textes"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Retourner la dimension des embeddings"""
        pass

class SentenceTransformerEmbedding(EmbeddingService):
    """Service d'embedding utilisant SentenceTransformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._dimension = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"SentenceTransformers non disponible, utilisation d'un embedding factice")
            self._dimension = 384
            return
            
        try:
            print(f"Chargement du modèle d'embedding: {model_name}")
            self.model = SentenceTransformer(model_name)
            # Tester pour obtenir la dimension
            test_embedding = self.model.encode("test")
            self._dimension = len(test_embedding)
            print(f"Modèle chargé avec succès. Dimension: {self._dimension}")
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {model_name}: {e}")
            print("Tentative avec le modèle par défaut...")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                test_embedding = self.model.encode("test")
                self._dimension = len(test_embedding)
                print(f"Modèle par défaut chargé. Dimension: {self._dimension}")
            except Exception as e2:
                print(f"Impossible de charger un modèle d'embedding: {e2}")
                self.model = None
                self._dimension = 384
    
    def embed_text(self, text: str) -> List[float]:
        """Générer l'embedding d'un texte"""
        if not text or not text.strip():
            return [0.0] * self._dimension
            
        try:
            if self.model is None:
                # Embedding factice pour les tests
                np.random.seed(hash(text) % 2**32)
                return np.random.random(self._dimension).tolist()
            
            embedding = self.model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            print(f"Erreur lors de l'embedding du texte: {e}")
            # Retourner un embedding factice basé sur le hash du texte
            np.random.seed(hash(text) % 2**32)
            return np.random.random(self._dimension).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Générer les embeddings d'une liste de textes"""
        if not texts:
            return []
            
        try:
            if self.model is None:
                # Embeddings factices pour les tests
                return [self.embed_text(text) for text in texts]
            
            embeddings = self.model.encode(texts)
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            print(f"Erreur lors de l'embedding batch: {e}")
            return [self.embed_text(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Retourner la dimension des embeddings"""
        return self._dimension

class DummyEmbeddingService(EmbeddingService):
    """Service d'embedding factice pour les tests"""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        print(f"Utilisation d'un service d'embedding factice (dimension: {dimension})")
    
    def embed_text(self, text: str) -> List[float]:
        """Générer un embedding factice basé sur le hash du texte"""
        if not text:
            return [0.0] * self._dimension
        
        # Utiliser le hash du texte comme seed pour la reproductibilité
        np.random.seed(hash(text) % 2**32)
        return np.random.random(self._dimension).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Générer des embeddings factices pour une liste de textes"""
        return [self.embed_text(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Retourner la dimension des embeddings"""
        return self._dimension

# Factory function pour créer le bon service
def create_embedding_service(model_name: str = None) -> EmbeddingService:
    """Créer un service d'embedding approprié"""
    if model_name and SENTENCE_TRANSFORMERS_AVAILABLE:
        return SentenceTransformerEmbedding(model_name)
    elif SENTENCE_TRANSFORMERS_AVAILABLE:
        return SentenceTransformerEmbedding()
    else:
        return DummyEmbeddingService()

# Test du service
if __name__ == "__main__":
    print("Test du service d'embedding...")
    
    service = create_embedding_service()
    
    # Test embedding simple
    text = "Bonjour, comment allez-vous ?"
    embedding = service.embed_text(text)
    print(f"Texte: {text}")
    print(f"Embedding: {len(embedding)} dimensions")
    print(f"Premiers 5 valeurs: {embedding[:5]}")
    
    # Test embedding batch
    texts = ["Bonjour", "Au revoir", "Comment ça va ?"]
    embeddings = service.embed_batch(texts)
    print(f"\nBatch de {len(texts)} textes:")
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        print(f"  {i+1}. '{text}' -> {len(emb)} dimensions")