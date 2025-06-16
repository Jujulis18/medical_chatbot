from src.vector_store.faiss_store import load_faiss_index
from src.embeddings.factory import get_embedder


class Retriever:
    def __init__(self, model_name):
        self.index, self.df_chunks = load_faiss_index()
        self.embedder = get_embedder(model_name)
        self.debug_info = []
    
    def retrieve(self, question: str, top_k: int = 5):
        self.debug_info = []
        self.debug_info.append(f"Question reçue: '{question}'")
        self.debug_info.append(f"Type: {type(question)}")
        
        # Encodage de la question
        question_embedding = self.embedder.encode(question)
        self.debug_info.append(f"Shape embedding: {question_embedding.shape}")
        
        # Reshape pour FAISS
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.reshape(1, -1)
        
        # Recherche
        distances, indices = self.index.search(question_embedding, top_k)
        self.debug_info.append(f"Indices trouvés: {indices[0]}")
        
        # Récupération des résultats
        results = []
        for idx in indices[0]:
            if idx < len(self.df_chunks):
                results.append(self.df_chunks.iloc[idx]["text"])
        
        self.debug_info.append(f"Nombre de résultats: {len(results)}")
        self.debug_info.append(f"Type du premier résultat: {type(results[0]) if results else 'Aucun résultat'}")
        return results