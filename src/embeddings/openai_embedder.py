# chatbot/embedding/openai_embedder.py

import openai
import numpy as np

class OpenAIEmbedder:
    def __init__(self, model_name='text-embedding-3-small', api_key=None):
        self.model_name = model_name
        self.api_key = api_key or "YOUR_OPENAI_API_KEY"
        openai.api_key = self.api_key

    def encode(self, text: str) -> np.ndarray:
        res = openai.Embedding.create(
            input=[text],
            model=self.model_name
        )
        embedding = np.array(res['data'][0]['embedding'], dtype='float32')
        return embedding.reshape(1, -1)
