from src.vector_store.faiss_store import load_faiss_index
from src.embeddings.factory import get_embedder


class Retriever:
    def __init__(self, model_name):
        self.index, self.df_chunks = load_faiss_index()  # tu ignores metadata ici
        self.embedder = get_embedder(model_name)

    def retrieve(self, question: str, top_k: int = 5):
        question_embedding = self.embedder.encode(question)  # shape (1, dim), float32
        question_embedding = np.expand_dims(question_embedding, axis=0)  # (1, dim)
        distances, indices = self.index.search(question_embedding, top_k)
        results = [self.df_chunks.iloc[idx]["text"] for idx in indices[0]]
        return results