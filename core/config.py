# import os

# class Config:
#     GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#     OLLAMA_MODELS = [
#         "gemma2:2b",
#         "llama3.2",
#         "mistral"
#     ]

#     DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    print("Warning: python-dotenv not installed. Install it with: pip install python-dotenv")


class Config:
    """
    Centralized configuration for Agent_ng.
    Loads settings from environment variables and defaults.
    """

    # LLM Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    OLLAMA_MODELS: list[str] = [
        "gemma2:2b",
        "llama3.2",
        "mistral"
    ]
    
    DEFAULT_OLLAMA_MODEL: str = "gemma2:2b"
    DEFAULT_GEMINI_MODEL: str = "gemini-2.0-flash"

    # Embeddings Configuration
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "mxbai-embed-large")
    EMBEDDINGS_MODE: str = os.getenv("EMBEDDINGS_MODE", "ollama")  # "ollama" or "sentence-transformer"
    
    # Vector DB Configuration
    PERSIST_DIR: str = os.getenv("PERSIST_DIR", str(Path(__file__).parent.parent / "data" / "vector_store"))
    CHROMA_COLLECTION: str = "agent_ng_docs"

    # System Prompts
    DEFAULT_SYSTEM_PROMPT: str = "You are a helpful AI assistant."
    
    # Data paths
    UPLOADS_DIR: str = str(Path(__file__).parent.parent / "data" / "uploads")
    DB_DIR: str = str(Path(__file__).parent.parent / "db")

    @classmethod
    def validate(cls) -> bool:
        """
        Validate critical configuration settings.
        Returns True if all required configs are present, False otherwise.
        """
        if not cls.GEMINI_API_KEY and "gemini" in str(cls.DEFAULT_GEMINI_MODEL).lower():
            print("Warning: GEMINI_API_KEY not set. Gemini provider will not work.")
        return True

    @classmethod
    def to_dict(cls) -> dict:
        """Return all configuration as a dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and key.isupper()
        }