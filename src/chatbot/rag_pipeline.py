class RAGPipeline:
    def __init__(self, retriever, llm_service, embedding_service):
        self.retriever = retriever
        self.llm_service = llm_service
        self.embedding_service = embedding_service

    @classmethod
    def from_config(cls, settings: Dict):
        # Initialiser les services nécessaires à partir des paramètres
        retriever = settings.get("retriever")
        llm_service = settings.get("llm_service")
        embedding_service = settings.get("embedding_service")

        return cls(retriever, llm_service, embedding_service)

    def query(self, question: str) -> str:
        # Embedding de la question
        query_embedding = self.embedding_service.embed_text(question)

        # Récupération des documents pertinents
        relevant_docs = self.retriever.retrieve(query_embedding)

        # Génération de la réponse
        context = "\n".join([doc.content for doc in relevant_docs])
        response = self.llm_service.generate(question, context)

        return response

if __name__ == "__main__":
    main()