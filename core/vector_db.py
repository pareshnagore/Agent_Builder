# import chromadb
# from chromadb.api.types import EmbeddingFunction
# import ollama


# class OllamaEmbeddingFunction(EmbeddingFunction):

#     def __init__(self, model="mxbai-embed-large"):
#         self.model = model

#     def __call__(self, input):
#         embeddings = []
#         for text in input:
#             response = ollama.embeddings(
#                 model=self.model,
#                 prompt=text
#             )
#             embeddings.append(response["embedding"])
#         return embeddings


# class VectorDB:

#     def __init__(self, persist_dir="data/vector_store", embed_model="mxbai-embed-large"):
#         self.client = chromadb.PersistentClient(path=persist_dir)
#         embedding_function = OllamaEmbeddingFunction(embed_model)
#         collection_name = f"docs_{embed_model.replace(':', '_')}"
#         self.collection = self.client.get_or_create_collection(
#             name=collection_name,
#             embedding_function=embedding_function
#         )

#     def add(self, documents, metadatas, ids):
#         self.collection.add(
#             documents=documents,
#             metadatas=metadatas,
#             ids=ids
#         )

#     def search(self, query_text, n_results=3):
#         return self.collection.query(
#             query_texts=[query_text],
#             n_results=n_results
#         )

#     def list_sources(self):
#         data = self.collection.get()
#         sources = set()
#         for m in data.get("metadatas", []):
#             if m and "source" in m:
#                 sources.add(m["source"])
#         return sorted(list(sources))

#     def reset(self):
#         self.client.reset()


"""
Vector Database Adapter for Agent_ng.
Wraps ChromaDB with a clean, reusable interface.
"""

import json
from typing import Optional, List, Dict, Any
import os
from pathlib import Path

from core.config import Config
from core.embeddings import EmbeddingsAdapter, EmbeddingsException


class VectorDBException(Exception):
    """Base exception for vector DB-related errors."""
    pass


class VectorDB:
    """
    Wrapper around ChromaDB for document storage and retrieval.
    Handles embeddings, persistence, and querying.
    """

    def __init__(
        self,
        persist_dir: str = None,
        embedding_adapter: EmbeddingsAdapter = None,
        collection_name: str = None
    ):
        """
        Initialize VectorDB.
        
        Args:
            persist_dir: Directory to persist Chroma data (defaults to Config.PERSIST_DIR)
            embedding_adapter: EmbeddingsAdapter instance (creates default if not provided)
            collection_name: Name of the collection. If not provided, automatically generated based on embedding model.
        """
        self.persist_dir = persist_dir or Config.PERSIST_DIR
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)

        # Initialize embeddings adapter if not provided
        if embedding_adapter is None:
            self.embedding_adapter = EmbeddingsAdapter()
        else:
            self.embedding_adapter = embedding_adapter

        # Generate collection name based on embedding model if not provided
        # This ensures separate collections for different embedding models
        if collection_name is None:
            # Create a safe collection name from the model and mode
            model_name = self.embedding_adapter.model.replace("/", "_").replace(":", "_").replace(".", "_")
            mode = self.embedding_adapter.mode.replace("-", "_")
            collection_name = f"docs_{mode}_{model_name}"
        
        self.collection_name = collection_name

        # Initialize Chroma client
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_adapter.get_embedding_function()
            )
        except ImportError:
            raise VectorDBException("chromadb not installed. Install with: pip install chromadb")
        except Exception as e:
            raise VectorDBException(f"Failed to initialize Chroma: {str(e)}")

    @staticmethod
    def _sanitize_metadata_for_chroma(meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metadata to ChromaDB-compatible format.
        ChromaDB accepts only str, int, float, bool — no None, no nested dicts.
        """
        if not meta:
            return {}
        result = {}
        for k, v in meta.items():
            if v is None:
                continue  # ChromaDB rejects None; omit the key
            if isinstance(v, dict):
                result[k] = json.dumps(v)
            elif isinstance(v, (str, int, float, bool)):
                result[k] = v
            else:
                result[k] = str(v)
        return result

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts (one per document)
            ids: List of document IDs (generated if not provided)
        
        Raises:
            VectorDBException: If add operation fails
        """
        try:
            if not documents:
                raise VectorDBException("Documents list cannot be empty")

            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]

            # Ensure metadatas has correct structure
            if metadatas is None:
                metadatas = [{} for _ in documents]
            elif len(metadatas) != len(documents):
                raise VectorDBException("Metadatas length must match documents length")

            # ChromaDB requires scalar metadata values (no nested dicts).
            # Sanitize for robustness across different callers.
            sanitized = [self._sanitize_metadata_for_chroma(m) for m in metadatas]

            self.collection.add(
                documents=documents,
                metadatas=sanitized,
                ids=ids
            )
        except VectorDBException:
            raise
        except Exception as e:
            raise VectorDBException(f"Failed to add documents: {str(e)}")

    def query_texts(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Query the vector database.
        Always returns a dict, even if ChromaDB returns a list (for compatibility).
        """
        try:
            if not query_texts:
                raise VectorDBException("Query texts cannot be empty")

            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where
            )
            # Defensive: if ChromaDB returns a list, wrap it in a dict
            if isinstance(results, list):
                return {"documents": results, "metadatas": [], "distances": [], "ids": []}
            if not isinstance(results, dict):
                raise VectorDBException(f"ChromaDB returned unexpected type: {type(results)}")
            return results
        except VectorDBException:
            raise
        except Exception as e:
            raise VectorDBException(f"Query failed: {str(e)}")

    def query_text(
        self,
        query_text: str,
        n_results: int = 5,
        where: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to query with a single text.
        Always returns a dict, even if ChromaDB returns a list.
        """
        return self.query_texts([query_text], n_results=n_results, where=where)

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Retrieve all document metadata from the collection.
        
        Returns:
            List of metadata dictionaries
        """
        try:
            # Get all items without limit
            results = self.collection.get()
            return results.get("metadatas", [])
        except Exception as e:
            raise VectorDBException(f"Failed to retrieve metadata: {str(e)}")

    def delete_by_source(self, source: str) -> int:
        """
        Delete all documents from a specific source.
        
        Args:
            source: Source identifier (typically filename or URL)
        
        Returns:
            Number of documents deleted
        """
        try:
            # Query documents with matching source metadata (indexer uses source_file)
            results = self.collection.get(
                where={"source_file": source} if source else None
            )
            if not results or not results.get("ids"):
                return 0

            # Delete the documents
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        except Exception as e:
            raise VectorDBException(f"Failed to delete by source: {str(e)}")

    def delete_by_id(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            doc_id: Document ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            raise VectorDBException(f"Failed to delete document: {str(e)}")

    def reset(self) -> None:
        """
        Delete all documents from the collection (WARNING: irreversible).
        """
        try:
            # Get all IDs and delete them
            results = self.collection.get()
            if results and results.get("ids"):
                self.collection.delete(ids=results["ids"])
        except Exception as e:
            raise VectorDBException(f"Failed to reset collection: {str(e)}")

    def count(self) -> int:
        """Return total number of documents in the collection."""
        try:
            results = self.collection.get()
            return len(results.get("ids", []))
        except Exception as e:
            raise VectorDBException(f"Failed to count documents: {str(e)}")

    def get_by_source(self, source: str) -> Dict[str, Any]:
        """
        Retrieve all documents from a specific source.
        
        Args:
            source: Source identifier
        
        Returns:
            Query results with matching documents
        """
        try:
            return self.collection.get(where={"source_file": source})
        except Exception as e:
            raise VectorDBException(f"Failed to get documents by source: {str(e)}")