class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, question):
        chunks = self.retriever.retrieve(question)
        combined_text = "\n".join(chunks)
        response = self.generator.generate(combined_text)
        return response