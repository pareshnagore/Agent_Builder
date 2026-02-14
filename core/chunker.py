"""
Chunker for Agent_ng.
Implements multiple chunking strategies with token-aware processing.
Designed for extensibility and use by agents.
"""

from typing import Literal, Optional, TypedDict
from abc import ABC, abstractmethod
import tiktoken
import nltk
from pathlib import Path


# Download tokenizers once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)


class ChunkMetadata(TypedDict, total=False):
    """Metadata attached to each chunk."""
    source_file: str  # Original file name/path
    source_type: str  # "pdf", "txt", "docx", etc.
    page_number: Optional[int]  # For multi-page documents
    chunk_index: int  # 0-based position in document
    chunk_id: str  # Unique identifier (e.g., "file_chunk_0001")
    upload_timestamp: str  # ISO timestamp
    document_title: Optional[str]
    custom_tags: dict  # User-defined metadata


class ChunkerException(Exception):
    """Base exception for chunking errors."""
    pass


class ChunkerStrategy(ABC):
    """Abstract base for chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        max_size: int,
        overlap: int = 0
    ) -> list[str]:
        """Split text into chunks."""
        pass


class SlidingWindowChunker(ChunkerStrategy):
    """
    Token-aware sliding window chunking.
    Simple, predictable, good for RAG.
    """

    def __init__(self, tokenizer_encoding: str = "cl100k_base"):
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_encoding)
        except Exception as e:
            raise ChunkerException(f"Failed to load tokenizer: {e}")

    def chunk(
        self,
        text: str,
        max_size: int,
        overlap: int = 0
    ) -> list[str]:
        """
        Chunk text using token-aware sliding window.
        
        Args:
            text: Text to chunk
            max_size: Max tokens per chunk
            overlap: Tokens to overlap between chunks
        
        Returns:
            List of chunk strings
        """
        if not text or not text.strip():
            return []

        if overlap >= max_size:
            raise ChunkerException(
                f"Overlap ({overlap}) must be < max_size ({max_size})"
            )

        # Encode full text
        tokens = self.tokenizer.encode(text)
        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Get next chunk of tokens
            end_idx = min(start_idx + max_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append(chunk_text)

            # Move start position (accounting for overlap)
            start_idx = end_idx - overlap
            if start_idx >= len(tokens):
                break

        return chunks


class SentenceAwareChunker(ChunkerStrategy):
    """
    Sentence-aware, token-safe chunking.
    Respects sentence boundaries, good for readability.
    """

    def __init__(self, tokenizer_encoding: str = "cl100k_base"):
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_encoding)
        except Exception as e:
            raise ChunkerException(f"Failed to load tokenizer: {e}")

    def chunk(
        self,
        text: str,
        max_size: int,
        overlap: int = 0
    ) -> list[str]:
        """
        Chunk text respecting sentence boundaries.
        
        Args:
            text: Text to chunk
            max_size: Max tokens per chunk
            overlap: Tokens to overlap (best-effort, respects sentences)
        
        Returns:
            List of chunk strings
        """
        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        overlap_text = ""

        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent))

            # If single sentence is too large, truncate it
            if sent_tokens > max_size:
                sent = self.tokenizer.decode(
                    self.tokenizer.encode(sent)[:max_size]
                )
                sent_tokens = len(self.tokenizer.encode(sent))

            # If adding this sentence exceeds limit, finalize chunk
            if current_tokens + sent_tokens > max_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Create overlap if requested
                if overlap > 0 and current_chunk:
                    overlap_tokens = self.tokenizer.encode(current_chunk)[-overlap:]
                    overlap_text = self.tokenizer.decode(overlap_tokens)
                    current_chunk = overlap_text + " " + sent
                else:
                    current_chunk = sent

                current_tokens = len(self.tokenizer.encode(current_chunk))
            else:
                current_chunk += " " + sent if current_chunk else sent
                current_tokens += sent_tokens

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks


class ParagraphChunker(ChunkerStrategy):
    """
    Paragraph-aware chunking.
    Groups paragraphs until reaching token limit.
    """

    def __init__(self, tokenizer_encoding: str = "cl100k_base"):
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_encoding)
        except Exception as e:
            raise ChunkerException(f"Failed to load tokenizer: {e}")

    def chunk(
        self,
        text: str,
        max_size: int,
        overlap: int = 0
    ) -> list[str]:
        """
        Chunk text respecting paragraph boundaries.
        
        Args:
            text: Text to chunk
            max_size: Max tokens per chunk
            overlap: Tokens to overlap (not applied to paragraphs)
        
        Returns:
            List of chunk strings
        """
        if not text or not text.strip():
            return []

        # Split into paragraphs (double newline or single for robustness)
        paragraphs = [
            p.strip() for p in text.split("\n\n") if p.strip()
        ]

        if not paragraphs:
            # Fallback: treat whole text as single paragraph
            paragraphs = [text]

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))

            # If single paragraph exceeds limit, add it anyway
            if para_tokens > max_size and not current_chunk:
                chunks.append(para)
                current_chunk = ""
                current_tokens = 0
                continue

            # If adding this paragraph exceeds limit, finalize chunk
            if current_tokens + para_tokens > max_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para
                current_tokens = para_tokens
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks


class Chunker:
    """
    Main chunker interface supporting multiple strategies.
    
    Usage:
        chunker = Chunker(strategy="sentence-aware")
        chunks = chunker.chunk_document(
            text="...",
            max_tokens=400,
            overlap_tokens=50,
            source_file="doc.pdf",
            page_number=1
        )
    """

    STRATEGIES = {
        "sliding-window": SlidingWindowChunker,
        "sentence-aware": SentenceAwareChunker,
        "paragraph": ParagraphChunker,
    }

    def __init__(
        self,
        strategy: Literal["sliding-window", "sentence-aware", "paragraph"] = "sentence-aware",
        tokenizer_encoding: str = "cl100k_base"
    ):
        """
        Initialize chunker.
        
        Args:
            strategy: Chunking strategy ("sliding-window", "sentence-aware", "paragraph")
            tokenizer_encoding: Tokenizer to use (default: cl100k_base for GPT models)
        """
        if strategy not in self.STRATEGIES:
            raise ChunkerException(
                f"Unknown strategy: {strategy}. Choose from: {list(self.STRATEGIES.keys())}"
            )

        self.strategy_name = strategy
        self.strategy = self.STRATEGIES[strategy](tokenizer_encoding=tokenizer_encoding)
        self.tokenizer = tiktoken.get_encoding(tokenizer_encoding)

    def chunk_text(
        self,
        text: str,
        max_tokens: int = 400,
        overlap_tokens: int = 50
    ) -> list[str]:
        """
        Chunk a text string.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Tokens to overlap between chunks
        
        Returns:
            List of chunks (strings)
        """
        return self.strategy.chunk(text, max_tokens, overlap_tokens)

    def chunk_document(
        self,
        text: str,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
        source_file: str = "unknown",
        source_type: str = "text",
        page_number: Optional[int] = None,
        document_title: Optional[str] = None,
        upload_timestamp: str = "",
        custom_tags: Optional[dict] = None
    ) -> list[tuple[str, ChunkMetadata]]:
        """
        Chunk a document and return chunks with metadata.
        
        Args:
            text: Document text
            max_tokens: Max tokens per chunk
            overlap_tokens: Tokens to overlap
            source_file: Source file name/path
            source_type: File type (pdf, txt, docx, etc.)
            page_number: Page number (if applicable)
            document_title: Document title
            upload_timestamp: ISO timestamp of upload
            custom_tags: User-defined metadata
        
        Returns:
            List of (chunk_text, metadata) tuples
        """
        chunks = self.chunk_text(text, max_tokens, overlap_tokens)

        result = []
        for idx, chunk in enumerate(chunks):
            # Generate unique chunk ID
            safe_filename = Path(source_file).stem.replace(" ", "_").lower()
            chunk_id = f"{safe_filename}_chunk_{idx:04d}"

            metadata: ChunkMetadata = {
                "source_file": source_file,
                "source_type": source_type,
                "page_number": page_number,
                "chunk_index": idx,
                "chunk_id": chunk_id,
                "upload_timestamp": upload_timestamp,
                "document_title": document_title,
                "custom_tags": custom_tags or {},
            }

            result.append((chunk, metadata))

        return result

    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        return len(self.tokenizer.encode(text))

    def shrink_and_clean(self, text: str, max_tokens: int) -> str:
        """
        Trim text to fit within token limit and clean whitespace.
        
        Args:
            text: Text to shrink
            max_tokens: Token limit
        
        Returns:
            Cleaned, trimmed text
        """
        # Clean whitespace
        text = " ".join(text.split())

        # Trim to token limit
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.tokenizer.decode(tokens)

        return text