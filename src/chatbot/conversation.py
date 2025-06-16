from sentence_transformers import SentenceTransformer
import numpy as np

# Exemple : si tu utilises des embeddings SBERT
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_similar_chunks(question, top_k=5):
    question_emb = embedder.encode([question])
    distances, indices = index.search(np.array(question_emb), top_k)
    
    results = []
    for idx in indices[0]:
        chunk_info = metadata[idx]
        text = df_chunks.loc[chunk_info["row_id"], "text"]
        results.append(text)
    
    return results
