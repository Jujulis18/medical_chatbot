from typing import Dict, List
import os
import sys

# Ajouter le chemin du projet pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_service import SentenceTransformerEmbedding
from retriever import SimpleRetriever
from llm_service import SimpleLLMService

class RAGPipeline:
    def __init__(self, retriever, llm_service, embedding_service):
        self.retriever = retriever
        self.llm_service = llm_service
        self.embedding_service = embedding_service

    @classmethod
    def from_config(cls, settings: Dict):
        """Initialiser les services nécessaires à partir des paramètres"""
        
        # Créer le service d'embedding
        embedding_model = settings.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        embedding_service = SentenceTransformerEmbedding(embedding_model)
        
        # Créer le retriever
        retriever = SimpleRetriever(
            top_k=settings.get("top_k", 5),
            similarity_threshold=settings.get("similarity_threshold", 0.7),
            embedding_service=embedding_service
        )
        
        # Créer le service LLM
        llm_service = SimpleLLMService(
            temperature=settings.get("temperature", 0.7),
            max_tokens=settings.get("max_tokens", 500)
        )

        return cls(retriever, llm_service, embedding_service)

    def query(self, question: str) -> Dict:
        """Traiter une question et retourner une réponse avec sources"""
        try:
            print(f"[RAG] Traitement de la question: {question}")
            
            # Embedding de la question
            query_embedding = self.embedding_service.embed_text(question)
            print(f"[RAG] Embedding généré: {len(query_embedding)} dimensions")

            # Récupération des documents pertinents
            relevant_docs = self.retriever.retrieve(query_embedding, question)
            print(f"[RAG] Documents trouvés: {len(relevant_docs)}")

            # Génération de la réponse
            if relevant_docs:
                context = "\n\n".join([f"Document {i+1}: {doc['content']}" 
                                     for i, doc in enumerate(relevant_docs)])
                sources = relevant_docs
            else:
                context = "Aucun document pertinent trouvé."
                sources = []

            response = self.llm_service.generate(question, context)
            print(f"[RAG] Réponse générée: {len(response)} caractères")

            return {
                "answer": response,
                "sources": sources
            }
            
        except Exception as e:
            print(f"[RAG] Erreur: {str(e)}")
            return {
                "answer": f"Désolé, une erreur s'est produite lors du traitement de votre question: {str(e)}",
                "sources": []
            }

def main():
    """Test du pipeline RAG"""
    settings = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 3,
        "similarity_threshold": 0.5,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    pipeline = RAGPipeline.from_config(settings)
    
    # Test
    question = "Quels sont les symptômes de la grippe ?"
    result = pipeline.query(question)
    print(f"Question: {question}")
    print(f"Réponse: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")

if __name__ == "__main__":
    main()