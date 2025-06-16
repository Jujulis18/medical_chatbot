import faiss
import numpy as np
import pandas as pd

def load_faiss_index(index_path="data/embeddings/faiss_index.index",
                     chunks_path="data/processed/chunks.csv",
                     embeddings_path="data/embeddings/embeddings.npy"):
    
    index = faiss.read_index(index_path)
    df_chunks = pd.read_csv(chunks_path)
    
    return index, df_chunks