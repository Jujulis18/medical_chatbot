import faiss
import numpy as np
import pandas as pd

def load_faiss_index(index_path="data/embeddings/faiss_index.index",
                     chunks_path="data/processed/chunks.csv",
                     embeddings_path="data/embeddings/embeddings.npy"):
    """
    Charge l'index FAISS, les embeddings (optionnel), et le DataFrame des chunks.
    """
    # Charger l’index
    index = faiss.read_index(index_path)

    # Charger les chunks
    df_chunks = pd.read_csv(chunks_path)

    # Charger les embeddings pour vérification ou debug (optionnel)
    embeddings = np.load(embeddings_path)

    return index, df_chunks
