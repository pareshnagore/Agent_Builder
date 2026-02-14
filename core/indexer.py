"""
Document indexer for Agent_ng.
Orchestrates loading, chunking, and vectorization of documents.
Supports idempotent ingestion with progress tracking.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Generator, Literal
from datetime import datetime
import uuid

from core.doc_loaders import DocumentLoaderFactory, LoaderException
from core.chunker import Chunker, ChunkMetadata
from core.vector_db import VectorDB


class IngestionState:
    """Track state of ingested documents for idempotency."""

    def __init__(self, state_file: str = "data/ingestion_state.json"):
        self.state_file = state_file
        self.state = {}
        self._load()

    def _load(self):
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    self.state = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load ingestion state: {e}")
                self.state = {}

    def _save(self):
        """Save state to file."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save ingestion state: {e}")

    def get_file_hash(self, file_path: str) -> str:
        """Compute hash of file contents."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            raise ValueError(f"Failed to compute hash: {e}")

    def is_ingested(self, file_path: str, force: bool = False) -> bool:
        """
        Check if file has already been ingested.
        
        Args:
            file_path: Path to file
            force: If True, ignore cached state
        
        Returns:
            True if already ingested (and not forced)
        """
        if force:
            return False

        file_hash = self.get_file_hash(file_path)
        key = Path(file_path).name

        if key in self.state:
            return self.state[key].get("file_hash") == file_hash

        return False

    def mark_ingested(self, file_path: str, doc_ids: list[str]):
        """
        Mark file as ingested.
        
        Args:
            file_path: Path to file
            doc_ids: List of document/chunk IDs generated
        """
        file_hash = self.get_file_hash(file_path)
        key = Path(file_path).name

        self.state[key] = {
            "file_path": str(file_path),
            "file_hash": file_hash,
            "timestamp": datetime.now().isoformat(),
            "doc_ids": doc_ids,
        }
        self._save()

    def get_doc_ids(self, file_path: str) -> list[str]:
        """Get document IDs for a previously ingested file."""
        key = Path(file_path).name
        return self.state.get(key, {}).get("doc_ids", [])


class IndexerException(Exception):
    """Base exception for indexer errors."""
    pass


class Indexer:
    """
    Main document indexer.
    Coordinates loading, chunking, and vectorization.
    
    Usage:
        indexer = Indexer(vectordb=vectordb, chunker=chunker)
        progress_gen = indexer.index_file(
            "document.pdf",
            force=False,
            max_tokens=400,
            overlap_tokens=50
        )
        for progress in progress_gen:
            print(progress['message'])
    """

    def __init__(
        self,
        vectordb: VectorDB,
        chunker: Optional[Chunker] = None,
        state_file: str = "data/ingestion_state.json"
    ):
        """
        Initialize indexer.
        
        Args:
            vectordb: VectorDB instance for storing embeddings
            chunker: Chunker instance (creates default if not provided)
            state_file: Path to ingestion state file for idempotency
        """
        self.vectordb = vectordb
        self.chunker = chunker or Chunker(strategy="sentence-aware")
        self.state = IngestionState(state_file=state_file)

    def index_file(
        self,
        file_path: str,
        force: bool = False,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
        document_title: Optional[str] = None,
        custom_tags: Optional[dict] = None,
        batch_size: int = 10,
    ) -> Generator[dict, None, None]:
        """
        Index a single document file.
        Yields progress updates.
        
        Args:
            file_path: Path to document
            force: Force re-indexing even if already ingested
            max_tokens: Max tokens per chunk
            overlap_tokens: Tokens to overlap
            document_title: Custom document title (uses filename if not provided)
            custom_tags: User-defined metadata tags
            batch_size: Number of chunks to embed at once
        
        Yields:
            Progress dicts with keys: status, message, progress, doc_ids (on completion)
        """
        file_path = Path(file_path)

        # Check if already ingested
        if self.state.is_ingested(str(file_path), force=force):
            yield {
                "status": "skip",
                "message": f"File already ingested: {file_path.name}",
                "progress": 1.0,
            }
            return

        try:
            # Step 1: Load document
            yield {
                "status": "loading",
                "message": f"Loading {file_path.name}...",
                "progress": 0.1,
            }

            text, file_type, supports_paging = DocumentLoaderFactory.load(str(file_path))

            if not text.strip():
                raise IndexerException(f"Document is empty: {file_path.name}")

            yield {
                "status": "loaded",
                "message": f"Loaded {len(text)} characters",
                "progress": 0.2,
            }

            # Step 2: Chunk document
            yield {
                "status": "chunking",
                "message": "Chunking document...",
                "progress": 0.3,
            }

            timestamp = datetime.now().isoformat()
            title = document_title or file_path.stem

            chunks_with_metadata = self.chunker.chunk_document(
                text=text,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                source_file=str(file_path),
                source_type=file_type,
                document_title=title,
                upload_timestamp=timestamp,
                custom_tags=custom_tags or {},
            )

            yield {
                "status": "chunked",
                "message": f"Created {len(chunks_with_metadata)} chunks",
                "progress": 0.4,
            }

            # Step 3: Add to vector DB
            yield {
                "status": "vectorizing",
                "message": "Vectorizing and storing chunks...",
                "progress": 0.5,
            }

            doc_ids = [metadata["chunk_id"] for _, metadata in chunks_with_metadata]
            chunk_texts = [text for text, _ in chunks_with_metadata]
            metadatas = [metadata for _, metadata in chunks_with_metadata]

            # Add to vector DB in batches
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]
                batch_ids = doc_ids[i : i + batch_size]

                self.vectordb.add_documents(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )

                progress = 0.4 + (0.5 * (i + len(batch_texts)) / len(chunk_texts))
                yield {
                    "status": "vectorizing",
                    "message": f"Vectorized {i + len(batch_texts)}/{len(chunk_texts)} chunks",
                    "progress": progress,
                }

            # Step 4: Mark as ingested
            self.state.mark_ingested(str(file_path), doc_ids)

            yield {
                "status": "complete",
                "message": f"Successfully ingested {file_path.name}",
                "progress": 1.0,
                "doc_ids": doc_ids,
            }

        except Exception as e:
            yield {
                "status": "error",
                "message": f"Failed to ingest {file_path.name}: {str(e)}",
                "progress": 0.0,
            }

    def index_folder(
        self,
        folder_path: str,
        pattern: str = "*",
        force: bool = False,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
        batch_size: int = 10,
    ) -> Generator[dict, None, None]:
        """
        Index all supported documents in a folder.
        
        Args:
            folder_path: Path to folder
            pattern: Glob pattern for files (default: "*" for all)
            force: Force re-indexing
            max_tokens: Max tokens per chunk
            overlap_tokens: Tokens to overlap
            batch_size: Batch size for vectorization
        
        Yields:
            Progress dicts for each file and overall progress
        """
        folder = Path(folder_path)
        supported_exts = set(DocumentLoaderFactory.supported_types())

        files = [
            f for f in folder.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_exts
        ]

        if not files:
            yield {
                "status": "skip",
                "message": f"No supported documents found in {folder_path}",
                "progress": 1.0,
            }
            return

        total_files = len(files)
        yield {
            "status": "starting",
            "message": f"Found {total_files} documents to index",
            "progress": 0.0,
        }

        for idx, file_path in enumerate(files):
            # Yield progress for current file
            for progress in self.index_file(
                str(file_path),
                force=force,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                batch_size=batch_size,
            ):
                # Adjust progress to overall folder progress
                file_progress = (idx / total_files) + (progress.get("progress", 0) / total_files)
                progress["overall_progress"] = file_progress
                yield progress

        yield {
            "status": "complete",
            "message": f"Indexed all {total_files} documents",
            "progress": 1.0,
        }