import ollama
from google import genai
from core.config import Config


class LLMClient:

    def __init__(self):
        self.gemini_client = None

        if Config.GEMINI_API_KEY:
            self.gemini_client = genai.Client(
                api_key=Config.GEMINI_API_KEY
            )

    # -------- OLLAMA --------
    def chat_ollama(self, model, messages):
        response = ollama.chat(
            model=model,
            messages=messages
        )
        return response["message"]["content"]

    # -------- GEMINI --------
    def list_gemini_models(self):
        try:
            models = self.gemini_client.models.list()
            return [m.name for m in models]
        except:
            return ["gemini-1.5-flash"]

    def chat_gemini(self, model, messages):

        conversation = ""
        for m in messages:
            conversation += f"{m['role']}: {m['content']}\n"

        response = self.gemini_client.models.generate_content(
            model=model,
            contents=conversation
        )

        return response.text

