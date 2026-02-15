"""
RAG Engine for Agent_ng - Phase 3
Orchestrates document indexing, retrieval, and context management.
Implements retrieval policies and prompt building with context trimming.
"""

import json
from typing import Optional, Literal, Tuple, List, Union
from datetime import datetime
from pathlib import Path
import tiktoken

from core.indexer import Indexer
from core.chunker import Chunker
from core.vector_db import VectorDB
from core.embeddings import EmbeddingsAdapter
from core.query_logger import QueryLogger


class RAGException(Exception):
    """Base exception for RAG-related errors."""
    pass


class RetrievalPolicy:
    """Data class for retrieval policy configuration."""
    
    def __init__(
        self,
        policy_type: Literal["strict", "relaxed", "hybrid"] = "strict",
        top_k: int = 3,
        similarity_threshold: float = 0.5,
        use_fallback: bool = False
    ):
        """
        Initialize retrieval policy.
        
        Args:
            policy_type: "strict" (top-k only), "relaxed" (threshold-based), "hybrid" (both)
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (for relaxed mode)
            use_fallback: Include BM25 fallback (if available)
        """
        self.policy_type = policy_type
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_fallback = use_fallback
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "policy_type": self.policy_type,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "use_fallback": self.use_fallback,
        }


class RAGEngine:
    """
    Main RAG Engine for document indexing and retrieval.
    
    Usage:
        rag = RAGEngine()
        
        # Index documents
        for progress in rag.index_file("document.pdf"):
            print(progress["message"])
        
        # Retrieve and build prompt
        contexts = rag.retrieve("What is AI?", top_k=3)
        prompt = rag.build_prompt(
            contexts=contexts,
            question="What is AI?",
            system_prompt="You are a helpful assistant."
        )
    """

    def __init__(
        self,
        embedding_mode: str = "ollama",
        embedding_model: str = "mxbai-embed-large",
        persist_dir: str = "data/vector_store",
        state_file: str = "data/ingestion_state.json",
        logs_dir: str = "data/logs",
        tokenizer_encoding: str = "cl100k_base"
    ):
        """
        Initialize RAG Engine.
        
        Args:
            embedding_mode: "ollama" or "sentence-transformer"
            embedding_model: Model to use for embeddings
            persist_dir: Directory for Chroma vectorDB
            state_file: File for ingestion state tracking
            logs_dir: Directory for query logs
            tokenizer_encoding: Tokenizer for token counting
        """
        # Initialize components
        try:
            self.embeddings_adapter = EmbeddingsAdapter(
                mode=embedding_mode,
                model=embedding_model
            )
        except Exception as e:
            raise RAGException(f"Failed to initialize embeddings: {e}")

        try:
            self.vectordb = VectorDB(
                persist_dir=persist_dir,
                embedding_adapter=self.embeddings_adapter
            )
        except Exception as e:
            raise RAGException(f"Failed to initialize vector DB: {e}")

        try:
            self.chunker = Chunker(strategy="sentence-aware")
        except Exception as e:
            raise RAGException(f"Failed to initialize chunker: {e}")

        try:
            self.indexer = Indexer(
                vectordb=self.vectordb,
                chunker=self.chunker,
                state_file=state_file
            )
        except Exception as e:
            raise RAGException(f"Failed to initialize indexer: {e}")

        try:
            self.query_logger = QueryLogger(logs_dir=logs_dir)
        except Exception as e:
            raise RAGException(f"Failed to initialize query logger: {e}")

        # Tokenizer for prompt building
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_encoding)
        except Exception as e:
            raise RAGException(f"Failed to initialize tokenizer: {e}")

        self.embedding_model = embedding_model
        self.embedding_mode = embedding_mode

    # ========== INDEXING ==========

    def index_file(
        self,
        file_path: str,
        force: bool = False,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
        document_title: Optional[str] = None,
        custom_tags: Optional[dict] = None,
    ):
        """
        Index a single document file.
        Yields progress updates.
        
        Args:
            file_path: Path to document
            force: Force re-indexing
            max_tokens: Max tokens per chunk
            overlap_tokens: Tokens to overlap
            document_title: Custom document title
            custom_tags: User-defined metadata
        
        Yields:
            Progress dicts
        """
        for progress in self.indexer.index_file(
            file_path=file_path,
            force=force,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            document_title=document_title,
            custom_tags=custom_tags,
        ):
            yield progress

    def index_folder(
        self,
        folder_path: str,
        pattern: str = "*",
        force: bool = False,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
    ):
        """
        Index all documents in a folder.
        
        Args:
            folder_path: Path to folder
            pattern: Glob pattern
            force: Force re-indexing
            max_tokens: Max tokens per chunk
            overlap_tokens: Tokens to overlap
        
        Yields:
            Progress dicts
        """
        for progress in self.indexer.index_folder(
            folder_path=folder_path,
            pattern=pattern,
            force=force,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ):
            yield progress

    # ========== RETRIEVAL ==========

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        policy: Optional[RetrievalPolicy] = None,
        filters: Optional[dict] = None,
    ) -> List[Tuple[str, dict]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of results (if not specified in policy)
            policy: RetrievalPolicy (defaults to strict top-k)
            filters: Metadata filters (e.g., {"source": "document.pdf"})
        
        Returns:
            List of (chunk_text, metadata) tuples
        """
        if not query or not query.strip():
            raise RAGException("Query cannot be empty")

        if policy is None:
            policy = RetrievalPolicy(policy_type="strict", top_k=top_k)

        try:
            # Query vector DB (use 'where' not 'filters')
            results = self.vectordb.query_text(
                query_text=query,
                n_results=policy.top_k * 2,  # Get extra for filtering
                where=filters
            )

            # Defensive: ChromaDB should return a dict, but if a list, raise a clear error
            if isinstance(results, list):
                raise RAGException("VectorDB returned a list, not a dict. This usually means the collection is empty or ChromaDB API changed. Please check your vector DB and document ingestion.")
            if not isinstance(results, dict):
                raise RAGException(f"VectorDB returned unexpected type: {type(results)}")

            chunks = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            distances = results.get("distances", [])

            if not chunks:
                return []

            # Apply retrieval policy
            if policy.policy_type == "strict":
                # Return top-k results
                results_list = [
                    (chunk, meta)
                    for chunk, meta in zip(chunks[:policy.top_k], metadatas[:policy.top_k])
                ]
            elif policy.policy_type == "relaxed":
                # Return results above threshold
                results_list = [
                    (chunk, meta)
                    for chunk, meta, distance in zip(chunks, metadatas, distances)
                    if (1 - distance) >= policy.similarity_threshold
                ]
            elif policy.policy_type == "hybrid":
                # Return top-k AND above threshold
                results_list = [
                    (chunk, meta)
                    for chunk, meta, distance in zip(chunks, metadatas, distances)
                    if ((1 - distance) >= policy.similarity_threshold) or (len(results_list) < policy.top_k)
                ]
            else:
                raise RAGException(f"Unknown policy type: {policy.policy_type}")

            return results_list

        except Exception as e:
            raise RAGException(f"Retrieval failed: {e}")

    def retrieve_with_provenance(
        self,
        query: str,
        top_k: int = 3,
        policy: Optional[RetrievalPolicy] = None,
    ) -> dict:
        """
        Retrieve results with full provenance information.
        
        Args:
            query: Query text
            top_k: Number of results
            policy: RetrievalPolicy
        
        Returns:
            Dict with results, sources, and metadata
        """
        results = self.retrieve(query, top_k=top_k, policy=policy)

        provenance = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "policy": (policy or RetrievalPolicy()).to_dict(),
            "results": [
                {
                    "text": chunk,
                    "source_file": meta.get("source_file", "unknown"),
                    "chunk_id": meta.get("chunk_id", "unknown"),
                    "page_number": meta.get("page_number"),
                    "custom_tags": meta.get("custom_tags", {}),
                }
                for chunk, meta in results
            ],
            "total_results": len(results),
        }

        return provenance

    # ========== PROMPT BUILDING ==========

    def build_prompt(
        self,
        contexts: Union[List[str], List[Tuple[str, dict]]],
        question: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_context_tokens: int = 2048,
    ) -> str:
        """
        Build a complete prompt with context and question.
        Automatically trims context to fit within token limit.
        
        Args:
            contexts: List of context texts OR (text, metadata) tuples
            question: User question
            system_prompt: System prompt
            max_context_tokens: Max tokens for context
        
        Returns:
            Complete prompt string
        """
        # Normalize contexts to text only
        context_texts = [
            ctx if isinstance(ctx, str) else ctx[0]
            for ctx in contexts
        ]

        # Trim context to token limit
        context = self._trim_context(
            context_texts,
            max_tokens=max_context_tokens
        )

        # Build prompt
        prompt = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""

        return prompt

    def build_prompt_with_citations(
        self,
        contexts: List[Tuple[str, dict]],
        question: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_context_tokens: int = 2048,
    ) -> Tuple[str, List[dict]]:
        """
        Build prompt with citations and return citation list.
        
        Args:
            contexts: List of (text, metadata) tuples
            question: User question
            system_prompt: System prompt
            max_context_tokens: Max tokens for context
        
        Returns:
            Tuple of (prompt, citations)
        """
        # Trim context while preserving metadata
        trimmed_contexts = self._trim_context_with_metadata(
            contexts,
            max_tokens=max_context_tokens
        )

        # Build citations
        citations = [
            {
                "id": meta.get("chunk_id", f"[{idx}]"),
                "source": meta.get("source_file", "unknown"),
                "page": meta.get("page_number"),
            }
            for idx, (_, meta) in enumerate(trimmed_contexts, 1)
        ]

        # Build context with citations
        context_parts = []
        for idx, (text, _) in enumerate(trimmed_contexts, 1):
            context_parts.append(f"[{idx}] {text}")

        context = "\n\n".join(context_parts)

        # Build prompt
        prompt = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer (cite sources):"""

        return prompt, citations

    # ========== CONTEXT TRIMMING ==========

    def _trim_context(
        self,
        context_texts: List[str],
        max_tokens: int = 2048,
    ) -> str:
        """
        Trim context to fit within token limit.
        Respects chunk boundaries.
        
        Args:
            context_texts: List of context chunks
            max_tokens: Max tokens
        
        Returns:
            Trimmed context string
        """
        combined = "\n\n".join(context_texts)
        tokens = self.tokenizer.encode(combined)

        if len(tokens) <= max_tokens:
            return combined

        # Trim to max tokens
        trimmed_tokens = tokens[:max_tokens]
        trimmed = self.tokenizer.decode(trimmed_tokens)

        return trimmed

    def _trim_context_with_metadata(
        self,
        contexts: List[Tuple[str, dict]],
        max_tokens: int = 2048,
    ) -> List[Tuple[str, dict]]:
        """
        Trim context while preserving metadata.
        Respects chunk boundaries.
        
        Args:
            contexts: List of (text, metadata) tuples
            max_tokens: Max tokens
        
        Returns:
            List of (text, metadata) tuples that fit within limit
        """
        result = []
        current_tokens = 0

        for text, meta in contexts:
            text_tokens = len(self.tokenizer.encode(text))

            if current_tokens + text_tokens <= max_tokens:
                result.append((text, meta))
                current_tokens += text_tokens
            else:
                # Can't fit this chunk; stop
                break

        return result

    # ========== UTILITIES ==========

    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        return len(self.tokenizer.encode(text))

    def log_query(
        self,
        query: str,
        results: List[Tuple[str, dict]],
        answer: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """
        Log a query and results for analysis.
        
        Args:
            query: Query text
            results: Retrieved results
            answer: LLM answer (optional)
            feedback: User feedback (optional)
        
        Returns:
            Log entry ID
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_results": len(results),
            "sources": [meta.get("source_file") for _, meta in results],
            "answer": answer,
            "feedback": feedback,
        }

        return self.query_logger.log(entry)

    def get_query_logs(self, limit: int = 100) -> List[dict]:
        """Get recent query logs."""
        return self.query_logger.read_logs(limit=limit)

    def export_logs(self, output_path: str = "query_logs_export.json"):
        """Export query logs to JSON."""
        logs = self.get_query_logs(limit=1000)
        with open(output_path, "w") as f:
            json.dump(logs, f, indent=2)
        return output_path

    def stats(self) -> dict:
        """Get RAG engine statistics."""
        return {
            "embedding_mode": self.embedding_mode,
            "embedding_model": self.embedding_model,
            "total_queries_logged": len(self.query_logger.read_logs(limit=10000)),
            "indexed_sources": self.indexer.state.state,
        }
