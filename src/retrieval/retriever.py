from typing import List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
import json
import os

class Document:
    """Classe représentant un document"""
    def __init__(self, content: str, title: str = "", metadata: Dict = None):
        self.content = content
        self.title = title
        self.metadata = metadata or {}
        self.embedding = None
    
    def __repr__(self):
        return f"Document(title='{self.title}', content_length={len(self.content)})"

class Retriever(ABC):
    """Interface abstraite pour les retrievers"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """Ajouter des documents à l'index"""
        pass
    
    @abstractmethod
    def retrieve(self, query_embedding: List[float], query_text: str = "") -> List[Dict]:
        """Récupérer les documents pertinents"""
        pass

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
    
    def _load_default_medical_documents(self):
        """Charger quelques documents médicaux de base"""
        medical_docs = [
            {
                "title": "Grippe - Symptômes",
                "content": """La grippe (influenza) est une infection virale qui affecte principalement le système respiratoire. 
                Les symptômes typiques incluent: fièvre élevée (38-40°C), maux de tête sévères, courbatures et douleurs musculaires, 
                fatigue intense, toux sèche, mal de gorge, écoulement nasal. Les symptômes apparaissent généralement brutalement 
                et durent entre 5 à 7 jours. La grippe est très contagieuse et se transmet par les gouttelettes respiratoires."""
            },
            {
                "title": "Hypertension artérielle",
                "content": """L'hypertension artérielle est définie par une pression artérielle systolique ≥ 140 mmHg et/ou 
                une pression diastolique ≥ 90 mmHg. C'est un facteur de risque majeur de maladies cardiovasculaires. 
                Souvent asymptomatique, elle peut causer maux de tête, vertiges, troubles visuels. Le traitement comprend 
                des mesures hygiéno-diététiques (réduction du sel, exercice, perte de poids) et des médicaments antihypertenseurs 
                si nécessaire."""
            },
            {
                "title": "Diabète de type 2",
                "content": """Le diabète de type 2 est caractérisé par une résistance à l'insuline et/ou un déficit relatif 
                en insuline. Les symptômes incluent: polyurie (urination fréquente), polydipsie (soif excessive), 
                polyphagie (faim excessive), perte de poids inexpliquée, fatigue, vision floue, cicatrisation lente. 
                Le diagnostic se fait par glycémie à jeun ≥ 1,26 g/L ou HbA1c ≥ 6,5%. Le traitement associe régime, 
                exercice et médicaments antidiabétiques."""
            },
            {
                "title": "Angine - Mal de gorge",
                "content": """L'angine est une inflammation des amygdales, le plus souvent d'origine virale (70%) ou bactérienne (30%). 
                Symptômes: mal de gorge intense, dysphagie, fièvre, adénopathies cervicales. L'angine virale guérit spontanément. 
                L'angine bactérienne (streptocoque A) nécessite un traitement antibiotique (amoxicilline). Le test de diagnostic 
                rapide (TDR) permet de différencier angine virale et bactérienne."""
            },
            {
                "title": "Migraine",
                "content": """La migraine est un type de céphalée primaire caractérisée par des crises récurrentes. 
                Symptômes typiques: céphalée unilatérale, pulsatile, d'intensité modérée à sévère, aggravée par l'effort, 
                accompagnée de nausées/vomissements, photophobie, phonophobie. Certaines migraines sont précédées d'aura 
                (troubles visuels, sensitifs). Le traitement comprend antalgiques pour la crise et traitements de fond 
                si crises fréquentes."""
            },
            {
                "title": "Asthme",
                "content": """L'asthme est une maladie inflammatoire chronique des voies respiratoires. Symptômes: dyspnée, 
                sifflements, oppression thoracique, toux surtout nocturne. Les crises peuvent être déclenchées par 
                allergènes, effort, stress, infections. Le diagnostic repose sur la spirométrie montrant un trouble 
                ventilatoire obstructif réversible. Traitement: bronchodilatateurs d'action rapide pour les crises, 
                corticoïdes inhalés comme traitement de fond."""
            },
            {
                "title": "Gastro-entérite",
                "content": """La gastro-entérite est une inflammation du tube digestif, souvent d'origine virale (norovirus, rotavirus). 
                Symptômes: diarrhée aqueuse, nausées, vomissements, douleurs abdominales, parfois fièvre. 
                La déshydratation est le principal risque, surtout chez les enfants et personnes âgées. 
                Traitement symptomatique: réhydratation orale, régime alimentaire adapté, antiémétiques si besoin. 
                Évolution favorable en 2-3 jours."""
            }
        ]
        
        documents = [Document(content=doc["content"], title=doc["title"]) for doc in medical_docs]
        self.add_documents(documents)
    
    def add_documents(self, documents: List[Document]):
        """Ajouter des documents à l'index"""
        print(f"Ajout de {len(documents)} documents à l'index...")
        
        for doc in documents:
            self.documents.append(doc)
            
            # Générer l'embedding si le service est disponible
            if self.embedding_service:
                try:
                    # Combiner titre et contenu pour l'embedding
                    text_to_embed = f"{doc.title}\n{doc.content}"
                    embedding = self.embedding_service.embed_text(text_to_embed)
                    self.embeddings.append(embedding)
                except Exception as e:
                    print(f"Erreur lors de l'embedding du document '{doc.title}': {e}")
                    # Utiliser un embedding zéro en cas d'erreur
                    dimension = self.embedding_service.get_dimension() if hasattr(self.embedding_service, 'get_dimension') else 384
                    self.embeddings.append([0.0] * dimension)
            else:
                self.embeddings.append([])
        
        print(f"Index mis à jour: {len(self.documents)} documents")
    
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
    
    def save_index(self, filepath: str):
        """Sauvegarder l'index sur disque"""
        try:
            data = {
                "documents": [
                    {
                        "title": doc.title,
                        "content": doc.content,
                        "metadata": doc.metadata
                    } for doc in self.documents
                ],
                "embeddings": self.embeddings,
                "config": {
                    "top_k": self.top_k,
                    "similarity_threshold": self.similarity_threshold
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Index sauvegardé dans {filepath}")
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
    
    def load_index(self, filepath: str):
        """Charger l'index depuis le disque"""
        try:
            if not os.path.exists(filepath):
                print(f"Fichier d'index non trouvé: {filepath}")
                return
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restaurer les documents
            self.documents = []
            for doc_data in data["documents"]:
                doc = Document(
                    content=doc_data["content"],
                    title=doc_data["title"],
                    metadata=doc_data.get("metadata", {})
                )
                self.documents.append(doc)
            
            # Restaurer les embeddings
            self.embeddings = data["embeddings"]
            
            # Restaurer la config
            config = data.get("config", {})
            self.top_k = config.get("top_k", self.top_k)
            self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)
            
            print(f"Index chargé: {len(self.documents)} documents")
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")

# Test du retriever
if __name__ == "__main__":
    from embedding_service import create_embedding_service
    
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