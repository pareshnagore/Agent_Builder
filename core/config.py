import os

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    OLLAMA_MODELS = [
        "gemma2:2b",
        "llama3.2",
        "mistral"
    ]

    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

