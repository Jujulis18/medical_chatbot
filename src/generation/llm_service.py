import openai
import numpy as np

class OpenAIChatGenerator:
    def __init__(self, api_key: str, model="gpt-4"):
        openai.api_key = api_key
        self.model = model

    def generate(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(context_chunks)
        prompt = f"""Tu es un assistant médical. Ne réponds que selon ce contexte.

Contexte :
{context}

Question :
{question}
"""
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Tu es un assistant médical neutre et prudent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
