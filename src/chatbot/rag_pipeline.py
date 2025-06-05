class RAGPipeline:
    def __init__(self, retriever, llm_service, embedding_service):
        self.retriever = retriever
        self.llm_service = llm_service
        self.embedding_service = embedding_service
    
    def query(self, question: str) -> str:
        # Embedding de la question
        query_embedding = self.embedding_service.embed_text(question)
        
        # Récupération des documents pertinents
        relevant_docs = self.retriever.retrieve(query_embedding)
        
        # Génération de la réponse
        context = "\n".join([doc.content for doc in relevant_docs])
        response = self.llm_service.generate(question, context)
        
        return response