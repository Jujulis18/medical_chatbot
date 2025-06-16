class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def run(self, question):
        # Récupération des chunks
        chunks = self.retriever.retrieve(question)
        
        if not chunks:
            return "Aucun document trouvé pour répondre à votre question.", []
        
        # Combinaison des chunks
        combined_text = "\n".join(chunks)
        
        # Génération de la réponse - passage des deux arguments requis
        response = self.generator.generate(question, chunks)
        return response, self.retriever.debug_info