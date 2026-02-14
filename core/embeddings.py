# import ollama


# class EmbeddingModel:

#     def __init__(self, model="mxbai-embed-large"):
#         self.model = model

#     def list_models(self):
#         return [
#             "mxbai-embed-large",
#             "embeddinggemma:latest"
#         ]

#     def embed(self, text):
#         response = ollama.embeddings(
#             model=self.model,
#             prompt=text
#         )

#         return response["embedding"]

"""
Embeddings Adapter for Agent_ng.
Supports both Ollama and sentence-transformers with a unified interface.
"""

from typing import Optional, Literal
from abc import ABC, abstractmethod
import numpy as np

from core.config import Config


class EmbeddingsException(Exception):
    """Base exception for embeddings-related errors."""
    pass


class EmbeddingsProvider(ABC):
    """Abstract base class for embeddings providers."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of texts with optional batch size control."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OllamaEmbeddingsProvider(EmbeddingsProvider):
    """Ollama Embeddings Provider."""

    def __init__(self, model: str = None, host: str = None):
        self.model = model or Config.EMBEDDINGS_MODEL
        self.host = host or Config.OLLAMA_HOST
        if not self.host.endswith("/"):
            self.host += "/"
        self._dimension = None

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        try:
            import requests
            payload = {"model": self.model, "prompt": text}
            response = requests.post(
                f"{self.host}api/embeddings",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", [])
            if not embedding:
                raise EmbeddingsException("No embedding returned from Ollama")
            return embedding
        except Exception as e:
            raise EmbeddingsException(f"Ollama embedding failed: {str(e)}")

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of texts."""
        try:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                for text in batch:
                    embedding = self.embed_text(text)
                    embeddings.append(embedding)
            return embeddings
        except Exception as e:
            raise EmbeddingsException(f"Batch embedding failed: {str(e)}")

    def get_dimension(self) -> int:
        """Get embedding dimension by embedding a test string."""
        if self._dimension is None:
            try:
                test_embedding = self.embed_text("test")
                self._dimension = len(test_embedding)
            except Exception as e:
                raise EmbeddingsException(f"Failed to determine embedding dimension: {str(e)}")
        return self._dimension


class SentenceTransformerEmbeddingsProvider(EmbeddingsProvider):
    """Sentence-Transformers Embeddings Provider (offline)."""

    def __init__(self, model: str = None):
        self.model = model or Config.EMBEDDINGS_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.model)
            self._dimension = self.embedder.get_sentence_embedding_dimension()
        except ImportError:
            raise EmbeddingsException(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingsException(f"Failed to load sentence-transformers model: {str(e)}")

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        try:
            embedding = self.embedder.encode(text, convert_to_numpy=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            raise EmbeddingsException(f"Sentence-transformers embedding failed: {str(e)}")

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of texts efficiently."""
        try:
            embeddings = self.embedder.encode(texts, batch_size=batch_size, convert_to_numpy=False)
            # Convert numpy arrays to lists
            return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
        except Exception as e:
            raise EmbeddingsException(f"Batch embedding failed: {str(e)}")

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


class EmbeddingsAdapter:
    """
    Unified embeddings adapter supporting multiple providers.
    Usage:
        adapter = EmbeddingsAdapter(mode="ollama")
        embedding = adapter.embed_text("Hello world")
        embeddings = adapter.embed_batch(["text1", "text2"])
    """

    def __init__(self, mode: str = None, model: str = None):
        """
        Initialize embeddings adapter.
        
        Args:
            mode: "ollama" or "sentence-transformer" (defaults to Config.EMBEDDINGS_MODE)
            model: Model name (defaults to Config.EMBEDDINGS_MODEL)
        """
        self.mode = mode or Config.EMBEDDINGS_MODE
        self.model = model or Config.EMBEDDINGS_MODEL

        if self.mode == "ollama":
            self.provider = OllamaEmbeddingsProvider(model=self.model)
        elif self.mode == "sentence-transformer":
            self.provider = SentenceTransformerEmbeddingsProvider(model=self.model)
        else:
            raise EmbeddingsException(f"Unknown embeddings mode: {self.mode}")

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        if not text or not isinstance(text, str):
            raise EmbeddingsException("Text must be a non-empty string")
        return self.provider.embed_text(text)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed a batch of texts with retry on failure.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
        
        Returns:
            List of embeddings (one per text)
        """
        if not texts or not isinstance(texts, list):
            raise EmbeddingsException("Texts must be a non-empty list")
        return self.provider.embed_batch(texts, batch_size=batch_size)

    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.provider.get_dimension()

    def get_safe_max_chars(self, model: str = None) -> int:
        """
        Return safe character limit for a model to avoid truncation.
        Estimates based on typical tokenization rates.
        
        Args:
            model: Model name (uses self.model if not provided)
        
        Returns:
            Max character count for safe embedding
        """
        # Rough estimate: 1 token ≈ 4 chars for most models
        # Most models support 512-2048 token limits
        MAX_TOKENS = 512
        CHARS_PER_TOKEN = 4
        return MAX_TOKENS * CHARS_PER_TOKEN  # 2048 chars

    def get_embedding_function(self):
        """
        Return a callable that can be used directly with Chroma.
        This allows passing the adapter to ChromaDB as an embedding function.
        """
        adapter = self
        
        class EmbeddingFunction:
            """Chroma-compatible embedding function wrapper."""
            
            def __call__(self, input):
                """Embed texts (Chroma expects 'input' parameter)."""
                if isinstance(input, str):
                    input = [input]
                return adapter.embed_batch(input)
            
            def name(self):
                """Return function name."""
                return f"{adapter.mode}_{adapter.model.replace('/', '_').replace(':', '_')}"
        
        return EmbeddingFunction()