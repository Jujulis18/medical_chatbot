from typing import List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
import json
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Retriever(ABC):
    """Classe abstraite pour les retrievers"""
    
    @abstractmethod
    def retrieve(self, query_embedding: List[float], query_text: str = "") -> List[Dict]:
        pass


class SimpleRetriever(Retriever):
    """Retriever simple basé sur la similarité cosinus"""
    
    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.7, embedding_service=None):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_service = embedding_service
        self.documents = []
        self.embeddings = []
        self.knowledge_base = None
        
        # Charger les documents médicaux par défaut
        self._load_default_medical_documents()
    
    def _load_default_medical_documents(self):
        """Charge les documents médicaux par défaut"""
        try:
            self.knowledge_base = self.load_default_medical_documents()
            if self.knowledge_base:
                # Extraire les documents et embeddings pour compatibilité
                self.documents = self.knowledge_base['documents']
                self.embeddings = self.knowledge_base['embeddings']
                print(f"Documents chargés: {len(self.documents)}")
            else:
                print("Impossible de charger les documents médicaux")
        except Exception as e:
            print(f"Erreur lors du chargement des documents: {e}")

    def load_default_medical_documents(self, 
                                     embeddings_path: str = "data/embeddings/embeddings.npy",
                                     chunks_path: str = "data/processed/chunks.csv") -> Dict[str, Any]:
        """
        Charge les documents médicaux depuis chunks.csv avec leurs embeddings pré-calculés.
        
        Args:
            embeddings_path: Chemin vers le fichier d'embeddings sauvegardé (.npy)
            chunks_path: Chemin vers le fichier CSV contenant les chunks de documents
            
        Returns:
            Dict contenant les documents et leurs embeddings
        """
        
        try:
            # Vérifier que les fichiers existent
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Fichier d'embeddings non trouvé: {embeddings_path}")
                
            if not os.path.exists(chunks_path):
                raise FileNotFoundError(f"Fichier chunks non trouvé: {chunks_path}")
            
            print(f"Chargement des embeddings depuis {embeddings_path}...")
            embeddings = np.load(embeddings_path)
            
            print(f"Chargement des chunks depuis {chunks_path}...")
            chunks_df = pd.read_csv(chunks_path)
            
            # Vérifier la cohérence
            if len(chunks_df) != embeddings.shape[0]:
                print(f"ATTENTION: Nombre de chunks ({len(chunks_df)}) != embeddings ({embeddings.shape[0]})")
                min_length = min(len(chunks_df), embeddings.shape[0])
                chunks_df = chunks_df.iloc[:min_length]
                embeddings = embeddings[:min_length]
                print(f"Ajustement: utilisation des {min_length} premiers éléments")
            
            # Convertir les chunks en format de documents
            documents_data = []
            for idx, row in chunks_df.iterrows():
                doc = {
                    'id': idx,
                    'chunk_id': row.get('chunk_id', idx),
                    'source': row.get('source', 'Unknown'),
                    'content': row.get('content', row.get('text', '')),
                    'title': row.get('title', f'Document {idx+1}'),
                    'embedding_index': idx,
                    'metadata': {}
                }
                
                # Ajouter d'autres colonnes comme métadonnées
                for col in chunks_df.columns:
                    if col not in ['chunk_id', 'source', 'content', 'text', 'title']:
                        doc['metadata'][col] = row[col]
                        
                documents_data.append(doc)
            
            # Structurer les données
            medical_knowledge_base = {
                'documents': documents_data,
                'embeddings': embeddings,
                'chunks_df': chunks_df,
                'metadata': {
                    'num_documents': len(documents_data),
                    'embedding_dimension': embeddings.shape[1],
                    'embeddings_source': embeddings_path,
                    'chunks_source': chunks_path,
                    'columns': list(chunks_df.columns)
                }
            }
            
            print(f"Base de connaissances chargée: {len(documents_data)} documents")
            return medical_knowledge_base
            
        except FileNotFoundError as e:
            print(f"Erreur: {e}")
            return None
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return None

    def get_document_by_index(self, index: int) -> Dict[str, Any]:
        """
        Récupère un document spécifique par son index.
        """
        if self.knowledge_base is None or index >= len(self.knowledge_base['documents']):
            return None
        
        return self.knowledge_base['documents'][index]

    def search_similar_documents(self, query_embedding: np.ndarray, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recherche les documents les plus similaires à un embedding de requête.
        """
        if self.knowledge_base is None:
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        # Reshape si nécessaire
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculer la similarité cosinus
        similarities = cosine_similarity(query_embedding, self.knowledge_base['embeddings'])[0]
        
        # Obtenir les indices des top_k documents les plus similaires
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.similarity_threshold:
                doc = self.knowledge_base['documents'][idx].copy()
                doc['similarity_score'] = float(similarities[idx])
                results.append(doc)
        
        return results

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculer la similarité cosinus entre deux vecteurs"""
        try:
            # Convertir en arrays numpy
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Éviter la division par zéro
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            # Calculer la similarité cosinus
            similarity = np.dot(a, b) / (norm_a * norm_b)
            return float(similarity)
            
        except Exception as e:
            print(f"Erreur lors du calcul de similarité: {e}")
            return 0.0
    
    def retrieve(self, query_embedding: List[float], query_text: str = "") -> List[Dict]:
        """Récupérer les documents pertinents"""
        if self.knowledge_base is None:
            print("Base de connaissances non chargée")
            return []
        
        if not query_embedding:
            print("Embedding de requête vide")
            return []
        
        # Convertir en numpy array si nécessaire
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        # Utiliser la nouvelle méthode de recherche
        similar_docs = self.search_similar_documents(query_embedding, self.top_k)
        
        # Convertir au format attendu par l'interface
        results = []
        for doc in similar_docs:
            results.append({
                "title": doc.get('title', f"Document {doc['id']}"),
                "content": doc.get('content', ''),
                "score": doc['similarity_score'],
                "metadata": doc.get('metadata', {}),
                "source": doc.get('source', 'Unknown'),
                "chunk_id": doc.get('chunk_id', doc['id'])
            })
        
        print(f"Documents trouvés: {len(results)} (seuil: {self.similarity_threshold})")
        return results


def initialize_medical_chatbot(embeddings_path: str = "data/embeddings/embeddings.npy",
                              chunks_path: str = "data/processed/chunks.csv"):
    """
    Initialise le chatbot médical avec les embeddings pré-calculés.
    
    Args:
        embeddings_path: Chemin vers les embeddings sauvegardés
        chunks_path: Chemin vers le fichier chunks
        
    Returns:
        Retriever initialisé
    """
    
    print("Initialisation du chatbot médical...")
    
    try:
        retriever = SimpleRetriever()
        if retriever.knowledge_base is None:
            print("Échec de l'initialisation du chatbot médical")
            return None
        
        print("Chatbot médical initialisé avec succès!")
        return retriever
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {e}")
        return None


# Test du retriever
if __name__ == "__main__":
    print("Test du retriever...")
    
    try:
        # Créer le retriever
        retriever = SimpleRetriever(
            top_k=3,
            similarity_threshold=0.3
        )
        
        if retriever.knowledge_base is None:
            print("Impossible de charger la base de connaissances")
            exit(1)
        
        # Test avec des embeddings factices (remplacez par votre service d'embedding)
        print("\nTest avec des embeddings aléatoires...")
        
        # Générer un embedding factice de la même dimension
        embedding_dim = retriever.knowledge_base['metadata']['embedding_dimension']
        fake_query_embedding = np.random.rand(embedding_dim)
        
        results = retriever.retrieve(fake_query_embedding.tolist(), "test query")
        
        if results:
            print(f"\nRésultats trouvés: {len(results)}")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']} (score: {result['score']:.3f})")
                content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                print(f"   {content_preview}")
        else:
            print("Aucun document pertinent trouvé")
            
    except Exception as e:
        print(f"Erreur lors du test: {e}")