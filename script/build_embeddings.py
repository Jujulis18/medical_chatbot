def build_embeddings_pipeline():
    # Chargement des documents
    documents = load_documents()
    
    # Préprocessing
    preprocessed_docs = preprocess_documents(documents)
    
    # Chunking
    chunks = chunk_documents(preprocessed_docs)
    
    # Embedding
    embedding_service = get_embedding_service()
    embeddings = embedding_service.embed_batch(chunks)
    
    # Stockage dans la base vectorielle
    vector_store = get_vector_store()
    vector_store.add_documents(chunks, embeddings)