from typing import List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
import json
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


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



def diagnose_retriever_issues(embeddings_path="data/embeddings/embeddings.npy", 
                             chunks_path="data/processed/chunks.csv"):
    """
    Diagnostic complet du problème de retrieval
    """
    
    print("=== DIAGNOSTIC DU RETRIEVER ===\n")
    
    try:
        # 1. Vérifier les fichiers
        print("1. Vérification des fichiers...")
        import os
        
        if not os.path.exists(embeddings_path):
            print(f"❌ ERREUR: {embeddings_path} n'existe pas")
            return
        else:
            print(f"✅ {embeddings_path} existe")
        
        if not os.path.exists(chunks_path):
            print(f"❌ ERREUR: {chunks_path} n'existe pas")
            return
        else:
            print(f"✅ {chunks_path} existe")
        
        # 2. Charger et analyser les embeddings
        print("\n2. Analyse des embeddings...")
        embeddings = np.load(embeddings_path)
        print(f"✅ Shape des embeddings: {embeddings.shape}")
        print(f"✅ Type: {embeddings.dtype}")
        print(f"✅ Min: {embeddings.min():.4f}, Max: {embeddings.max():.4f}")
        print(f"✅ Moyenne: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
        
        # Vérifier s'il y a des NaN ou des valeurs infinies
        if np.isnan(embeddings).any():
            print("❌ ATTENTION: Des valeurs NaN détectées dans les embeddings")
        if np.isinf(embeddings).any():
            print("❌ ATTENTION: Des valeurs infinies détectées dans les embeddings")
        
        # 3. Charger et analyser les chunks
        print("\n3. Analyse des chunks...")
        chunks_df = pd.read_csv(chunks_path)
        print(f"✅ Nombre de chunks: {len(chunks_df)}")
        print(f"✅ Colonnes: {list(chunks_df.columns)}")
        
        # Afficher quelques exemples de contenu
        print("\n📄 Exemples de contenu:")
        for i in range(min(3, len(chunks_df))):
            content = chunks_df.iloc[i].get('content', chunks_df.iloc[i].get('text', 'Pas de contenu'))
            content_preview = content[:100] + "..." if len(str(content)) > 100 else str(content)
            print(f"   {i+1}. {content_preview}")
        
        # 4. Test de similarité avec différents seuils
        print("\n4. Test de similarité...")
        
        # Créer un embedding de requête factice
        query_embedding = np.random.rand(embeddings.shape[1])
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        print(f"✅ Similarités calculées: {len(similarities)}")
        print(f"✅ Similarité max: {similarities.max():.4f}")
        print(f"✅ Similarité min: {similarities.min():.4f}")
        print(f"✅ Similarité moyenne: {similarities.mean():.4f}")
        
        # Tester différents seuils
        thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        print("\n📊 Résultats par seuil:")
        for threshold in thresholds:
            count = np.sum(similarities >= threshold)
            print(f"   Seuil {threshold:.1f}: {count} documents")
        
        # 5. Test avec embedding réel si possible
        print("\n5. Test avec des embeddings entre documents...")
        
        # Prendre 2 documents et calculer leur similarité
        if len(embeddings) >= 2:
            sim_between_docs = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            print(f"✅ Similarité entre doc 0 et doc 1: {sim_between_docs:.4f}")
            
            # Similarité d'un doc avec lui-même (devrait être 1.0)
            sim_self = cosine_similarity([embeddings[0]], [embeddings[0]])[0][0]
            print(f"✅ Similarité doc 0 avec lui-même: {sim_self:.4f}")
        
        # 6. Recommandations
        print("\n6. 🔧 RECOMMANDATIONS:")
        
        max_sim = similarities.max()
        if max_sim < 0.3:
            print("❌ Similarités très faibles - Problème probable avec les embeddings")
            print("   → Vérifiez que les embeddings correspondent aux bons documents")
            print("   → Utilisez un seuil très bas (0.1 ou moins) pour tester")
        elif max_sim < 0.5:
            print("⚠️  Similarités modérées")
            print(f"   → Utilisez un seuil de {max_sim * 0.7:.2f} ou moins")
        else:
            print("✅ Similarités normales")
            print(f"   → Utilisez un seuil de {max_sim * 0.8:.2f} ou moins")
        
        # Suggestions de questions à tester
        print("\n7. 💡 SUGGESTIONS DE TESTS:")
        content_examples = []
        for i in range(min(5, len(chunks_df))):
            content = str(chunks_df.iloc[i].get('content', chunks_df.iloc[i].get('text', '')))
            if content and len(content) > 20:
                # Extraire quelques mots-clés du contenu
                words = content.lower().split()[:5]
                content_examples.append(' '.join(words))
        
        if content_examples:
            print("   Essayez ces requêtes basées sur vos documents:")
            for i, example in enumerate(content_examples[:3]):
                print(f"   → \"{example}\"")
        
        return {
            'embeddings_shape': embeddings.shape,
            'num_chunks': len(chunks_df),
            'max_similarity': similarities.max(),
            'recommended_threshold': min(0.3, similarities.max() * 0.7)
        }
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_retriever_with_low_threshold():
    """
    Test rapide du retriever avec un seuil très bas
    """
    print("\n=== TEST RETRIEVER AVEC SEUIL BAS ===\n")
    
    try:
        # Importer votre classe (ajustez le chemin selon votre structure)
        from retriever import SimpleRetriever  # Ajustez selon votre import
        
        # Créer retriever avec seuil très bas
        retriever = SimpleRetriever(
            top_k=5,
            similarity_threshold=0.0  # Seuil très bas pour tester
        )
        
        if retriever.knowledge_base is None:
            print("❌ Impossible de charger la base de connaissances")
            return
        
        print(f"✅ Base chargée: {len(retriever.documents)} documents")
        
        # Test avec embedding factice
        embedding_dim = retriever.knowledge_base['metadata']['embedding_dimension']
        fake_embedding = np.random.rand(embedding_dim).tolist()
        
        results = retriever.retrieve(fake_embedding, "test")
        
        print(f"✅ Résultats trouvés: {len(results)}")
        for i, result in enumerate(results[:3]):
            print(f"   {i+1}. Score: {result['score']:.4f}")
            content = result['content'][:80] + "..." if len(result['content']) > 80 else result['content']
            print(f"       {content}")
            
    except Exception as e:
        print(f"❌ ERREUR lors du test: {e}")



  
    

# Test du retriever
if __name__ == "__main__":
    print("Test du retriever...")
    
    try:
        # Créer le retriever
        retriever = SimpleRetriever(
            top_k=10,
            similarity_threshold=0.1
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

          # Lancer le diagnostic
        result = diagnose_retriever_issues()
        
        # Test avec seuil bas
        test_retriever_with_low_threshold()
            
    except Exception as e:
        print(f"Erreur lors du test: {e}")