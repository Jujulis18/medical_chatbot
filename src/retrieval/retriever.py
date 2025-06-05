from typing import List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
import json
import os
import pandas as pd

class SimpleRetriever(Retriever):
    """Retriever simple basé sur la similarité cosinus"""
    
    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.7, embedding_service=None):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_service = embedding_service
        self.documents = []
        self.embeddings = []
        
        # Charger les documents médicaux par défaut
        self._load_default_medical_documents()
    


    def load_default_medical_documents(embeddings_path: str = "data/embeddings/embeddings.npy",
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
                        
            # Charger les embeddings pré-calculés
            embeddings = np.load(embeddings_path)
            chunks_df = pd.read_csv(chunks_path)
                        
            # Convertir les chunks en format de documents
            documents_data = []
            for idx, row in chunks_df.iterrows():
                doc = {
                    'id': idx,
                    'chunk_id': row.get('chunk_id', idx),
                    'source': row.get('source', 'Unknown'),
                    'content': row.get('content', row.get('text', '')),
                    'embedding_index': idx
                }
                
                # Ajouter d'autres colonnes si elles existent
                for col in chunks_df.columns:
                    if col not in ['chunk_id', 'source', 'content', 'text']:
                        doc[col] = row[col]
                        
                documents_data.append(doc)
            
            # Structurer les données
            medical_knowledge_base = {
                'documents': documents_data,
                'embeddings': embeddings,
                'chunks_df': chunks_df,  # Garder le DataFrame original pour référence
                'metadata': {
                    'num_documents': len(documents_data),
                    'embedding_dimension': embeddings.shape[1],
                    'embeddings_source': embeddings_path,
                    'chunks_source': chunks_path,
                    'columns': list(chunks_df.columns)
                }
            }
            
                        
            return medical_knowledge_base
            
        except FileNotFoundError as e:
            print(f"Erreur: {e}")
            print("Assurez-vous que les fichiers d'embeddings et de chunks existent.")
            return None
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return None


    def get_document_by_index(knowledge_base: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Récupère un document spécifique par son index.
        
        Args:
            knowledge_base: Base de connaissances chargée
            index: Index du document à récupérer
            
        Returns:
            Document correspondant à l'index
        """
        if knowledge_base is None or index >= len(knowledge_base['documents']):
            return None
        
        return knowledge_base['documents'][index]


    def search_similar_documents(knowledge_base: Dict[str, Any], 
                               query_embedding: np.ndarray, 
                               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recherche les documents les plus similaires à un embedding de requête.
        
        Args:
            knowledge_base: Base de connaissances chargée
            query_embedding: Embedding de la requête
            top_k: Nombre de documents à retourner
            
        Returns:
            Liste des documents les plus similaires avec leurs scores
        """
        if knowledge_base is None:
            return []
        
        # Calculer la similarité cosinus
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity([query_embedding], knowledge_base['embeddings'])[0]
        
        # Obtenir les indices des top_k documents les plus similaires
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = knowledge_base['documents'][idx].copy()
            doc['similarity_score'] = float(similarities[idx])
            results.append(doc)
        
        return results


    def initialize_medical_chatbot(embeddings_path: str = "data/embeddings/embeddings.npy"):
        """
        Initialise le chatbot médical avec les embeddings pré-calculés.
        
        Args:
            embeddings_path: Chemin vers les embeddings sauvegardés
            
        Returns:
            Base de connaissances médicales initialisée
        """
        
        print("Initialisation du chatbot médical...")
        
        # Charger la base de connaissances avec les embeddings pré-calculés
        knowledge_base = load_default_medical_documents(embeddings_path)
        
        if knowledge_base is None:
            print("Échec de l'initialisation du chatbot médical")
            return None
        
        print("Chatbot médical initialisé avec succès!")
        return knowledge_base



    
    
    
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
        if not self.documents:
            print("Aucun document dans l'index")
            return []
        
        if not query_embedding:
            print("Embedding de requête vide")
            return []
        
        # Calculer les similarités
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            if not doc_embedding:
                similarity = 0.0
            else:
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filtrer par seuil et prendre les top_k
        results = []
        for doc_idx, similarity in similarities[:self.top_k]:
            if similarity >= self.similarity_threshold:
                doc = self.documents[doc_idx]
                results.append({
                    "title": doc.title,
                    "content": doc.content,
                    "score": similarity,
                    "metadata": doc.metadata
                })
        
        print(f"Documents trouvés: {len(results)} (seuil: {self.similarity_threshold})")
        return results
    
    

# Test du retriever
if __name__ == "__main__":
    from src.embeddings.embedding_service import create_embedding_service
    
    print("Test du retriever...")
    
    # Créer les services
    embedding_service = create_embedding_service()
    retriever = SimpleRetriever(
        top_k=3,
        similarity_threshold=0.3,
        embedding_service=embedding_service
    )
    
    # Test de recherche
    questions = [
        "Quels sont les symptômes de la grippe ?",
        "Comment traiter l'hypertension ?",
        "Que faire en cas de mal de gorge ?",
        "Symptômes du diabète"
    ]
    
    for question in questions:
        print(f"\n--- Question: {question} ---")
        query_embedding = embedding_service.embed_text(question)
        results = retriever.retrieve(query_embedding, question)
        
        if results:
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']} (score: {result['score']:.3f})")
                print(f"   {result['content'][:100]}...")
        else:
            print("Aucun document pertinent trouvé")

