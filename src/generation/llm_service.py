from openai import OpenAI
import numpy as np

class OpenAIChatGenerator:
    def __init__(self, api_key: str, model="gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(context_chunks)
        prompt = f"""Tu es un assistant médical. Ne réponds que selon ce contexte.
Contexte :
{context}
Question :
{question}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Tu es un assistant médical neutre et prudent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()