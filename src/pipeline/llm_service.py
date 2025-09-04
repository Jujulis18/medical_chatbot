from llama_index.llms.huggingface import HuggingFaceLLM

class MISTRALChatGenerator:
    def __init__(self, api_key: str, model: str = "mistral-small-latest", debug: bool = False):
        print("API Key:", api_key, flush=True)
        self.client = HuggingFaceLLM(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1"  # point d'accès Mistral compatible OpenAI
        )
        self.model = model
        self.debug = debug

    def complete(self, prompt: str):
        if self.debug:
            print(f"[DEBUG] Prompt envoyé au LLM:\n{prompt}", flush=True)
        response = self.client.complete(
            prompt,
            model=self.model,
            temperature=0.2,
            max_tokens=1000
        )
        if self.debug:
            print(f"[DEBUG] Réponse brute:\n{response}", flush=True)
        return response
