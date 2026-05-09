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
    OLLAMA_CLOUD_ENABLED = os.getenv("OLLAMA_CLOUD_ENABLED", "false").lower() == "true"
    OLLAMA_CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
    
    OLLAMA_MODELS: list[str] = [
        "gemma2:2b",
        "llama3.2",
        "mistral"
    ]
    
    DEFAULT_OLLAMA_MODEL: str = "gemma2:2b"
    DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
    
    # Context Length Configuration (in tokens)
    # Default context length for Ollama models
    DEFAULT_OLLAMA_CONTEXT_LENGTH: int = int(os.getenv("DEFAULT_OLLAMA_CONTEXT_LENGTH", "8192"))
    
    # Model-specific context windows (override defaults)
    OLLAMA_MODEL_CONTEXT_LENGTHS: dict[str, int] = {
        "llama2": 4096,
        "llama2:7b": 4096,
        "llama2:13b": 4096,
        "llama2:70b": 4096,
        "llama3": 8192,
        "llama3:8b": 8192,
        "llama3:70b": 8192,
        "llama3.2": 8192,
        "llama3.2:1b": 8192,
        "llama3.2:3b": 8192,
        "gemma": 8192,
        "gemma2": 8192,
        "gemma2:2b": 8192,
        "mistral": 8192,
        "neural-chat": 4096,
        "dolphin-mixtral": 4096,
    }

    # Embeddings Configuration
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "mxbai-embed-large")
    EMBEDDINGS_MODE: str = os.getenv("EMBEDDINGS_MODE", "ollama")  # "ollama" or "sentence-transformer"
    
    # Embedding Model Context Lengths (in tokens)
    # These models have limited context windows for embedding
    EMBEDDING_MODEL_CONTEXT_LENGTHS: dict[str, int] = {
        "mxbai-embed-large": 512,      # MixedBread's large embed model
        "mxbai-embed-small": 512,      # MixedBread's small embed model
        "all-minilm-l6-v2": 384,       # Sentence transformer model
        "nomic-embed-text": 2048,      # Nomic's embedding model
        "embeddinggemma:latest": 2048, # Google's embedding model
    }
    
    DEFAULT_EMBEDDING_CONTEXT_LENGTH: int = 512  # Safe default for most embedding models
    
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