"""
Test Phase 2: Document Ingestion Pipeline
Tests chunker, doc_loaders, and indexer modules.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.chunker import Chunker, ChunkerException
from core.doc_loaders import DocumentLoaderFactory, LoaderException
from core.indexer import Indexer, IngestionState
from core.vector_db import VectorDB
from core.embeddings import EmbeddingsAdapter


def test_chunker():
    """Test chunking strategies."""
    print("\n" + "="*60)
    print("TEST 1: Chunker")
    print("="*60)
    
    test_text = """
    The quick brown fox jumps over the lazy dog. This is the first sentence.
    The second sentence is here. And now the third one appears.
    This paragraph has multiple sentences. Each one should be processed correctly.
    We test token-aware chunking with various strategies.
    The final sentence concludes this test text.
    """
    
    strategies = ["sliding-window", "sentence-aware", "paragraph"]
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy} strategy ---")
        try:
            chunker = Chunker(strategy=strategy)
            chunks = chunker.chunk_text(test_text, max_tokens=50, overlap_tokens=10)
            print(f"✓ Created {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:2]):  # Show first 2
                print(f"  Chunk {i}: {chunk[:60]}...")
            
            # Test with metadata
            chunks_with_meta = chunker.chunk_document(
                text=test_text,
                max_tokens=50,
                overlap_tokens=10,
                source_file="test.txt",
                source_type="txt",
                document_title="Test Document",
                custom_tags={"test": True}
            )
            print(f"✓ Chunks with metadata: {len(chunks_with_meta)}")
            if chunks_with_meta:
                text, meta = chunks_with_meta[0]
                print(f"  Metadata keys: {list(meta.keys())}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    return True


def test_doc_loaders():
    """Test document loaders."""
    print("\n" + "="*60)
    print("TEST 2: Document Loaders")
    print("="*60)
    
    # Create test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test TXT
        print("\n--- Testing TXT loader ---")
        txt_file = tmpdir / "test.txt"
        txt_file.write_text("Hello world! This is a test text file.")
        try:
            text, file_type, paging = DocumentLoaderFactory.load(str(txt_file))
            print(f"✓ Loaded TXT: {len(text)} chars, type={file_type}, paging={paging}")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
        
        # Test MD
        print("\n--- Testing MD loader ---")
        md_file = tmpdir / "test.md"
        md_file.write_text("# Heading\n\nThis is markdown content.\n\n## Subheading\n\nMore content.")
        try:
            text, file_type, paging = DocumentLoaderFactory.load(str(md_file))
            print(f"✓ Loaded MD: {len(text)} chars, type={file_type}, paging={paging}")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
        
        # Test CSV
        print("\n--- Testing CSV loader ---")
        csv_file = tmpdir / "test.csv"
        csv_file.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA")
        try:
            text, file_type, paging = DocumentLoaderFactory.load(str(csv_file))
            print(f"✓ Loaded CSV: {len(text)} chars, type={file_type}, paging={paging}")
            print(f"  Content: {text[:60]}...")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
        
        # Test that unsupported format is rejected
        print("\n--- Testing unsupported format rejection ---")
        unsupported = tmpdir / "test.xyz"
        unsupported.write_text("test")
        try:
            DocumentLoaderFactory.load(str(unsupported))
            print(f"✗ Should have rejected .xyz format")
            return False
        except LoaderException as e:
            print(f"✓ Correctly rejected: {e}")
        
        # Test supported types listing
        print("\n--- Testing supported types ---")
        supported = DocumentLoaderFactory.supported_types()
        print(f"✓ Supported types: {supported}")
    
    return True


def test_ingestion_state():
    """Test idempotency tracking."""
    print("\n" + "="*60)
    print("TEST 3: Ingestion State Tracking")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        state_file = tmpdir / "state.json"
        
        # Create test file
        test_file = tmpdir / "test.txt"
        test_file.write_text("Test content for hashing")
        
        try:
            state = IngestionState(state_file=str(state_file))
            
            # Should not be ingested initially
            if state.is_ingested(str(test_file)):
                print("✗ File marked as ingested before ingestion")
                return False
            print("✓ File correctly marked as not ingested")
            
            # Mark as ingested
            doc_ids = ["test_chunk_0001", "test_chunk_0002"]
            state.mark_ingested(str(test_file), doc_ids)
            print(f"✓ Marked as ingested with {len(doc_ids)} doc IDs")
            
            # Should be ingested now
            if not state.is_ingested(str(test_file)):
                print("✗ File not recognized as ingested after marking")
                return False
            print("✓ File correctly marked as ingested")
            
            # Get doc IDs back
            retrieved_ids = state.get_doc_ids(str(test_file))
            if retrieved_ids != doc_ids:
                print(f"✗ Doc IDs mismatch: {retrieved_ids} != {doc_ids}")
                return False
            print(f"✓ Retrieved correct doc IDs: {retrieved_ids}")
            
            # Force re-ingestion should skip state check
            if not state.is_ingested(str(test_file), force=True):
                print("✗ Force flag not working")
                return False
            print("✓ Force flag correctly bypasses state")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    return True


def test_chunker_token_count():
    """Test token counting utilities."""
    print("\n" + "="*60)
    print("TEST 4: Token Counting & Utilities")
    print("="*60)
    
    try:
        chunker = Chunker()
        
        # Test token counting
        test_text = "Hello world, this is a test."
        token_count = chunker.get_token_count(test_text)
        print(f"✓ Token count for '{test_text}': {token_count} tokens")
        
        # Test shrink and clean
        long_text = test_text * 100  # Repeat to make it longer
        cleaned = chunker.shrink_and_clean(long_text, max_tokens=50)
        cleaned_tokens = chunker.get_token_count(cleaned)
        print(f"✓ Cleaned text from {chunker.get_token_count(long_text)} → {cleaned_tokens} tokens")
        print(f"  Original: {len(long_text)} chars → Cleaned: {len(cleaned)} chars")
        
        if cleaned_tokens > 50:
            print(f"✗ Cleaned text exceeded max tokens: {cleaned_tokens} > 50")
            return False
        print("✓ Token limit respected")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def test_indexer_workflow():
    """Test indexer with actual file."""
    print("\n" + "="*60)
    print("TEST 5: Indexer Workflow")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test file
        test_file = tmpdir / "sample.txt"
        test_content = """
        This is a sample document for testing the indexer.
        It contains multiple paragraphs and sentences.
        The indexer should be able to load, chunk, and process this file.
        
        Here is another paragraph with different content.
        It tests the paragraph-aware chunking strategy.
        We want to ensure proper behavior across different text structures.
        """
        test_file.write_text(test_content)
        
        try:
            # Initialize vectordb and indexer
            vector_db_path = tmpdir / "vector_store"
            vectordb = VectorDB(persist_dir=str(vector_db_path))
            indexer = Indexer(
                vectordb=vectordb,
                state_file=str(tmpdir / "state.json")
            )
            
            print("\n--- Indexing file ---")
            progress_list = []
            for progress in indexer.index_file(
                str(test_file),
                max_tokens=100,
                overlap_tokens=20,
                document_title="Sample Document"
            ):
                progress_list.append(progress)
                status = progress.get("status")
                message = progress.get("message")
                print(f"  [{status}] {message}")
            
            # Verify completion
            final = progress_list[-1]
            if final["status"] != "complete":
                print(f"✗ Indexing did not complete: {final['status']}")
                return False
            
            doc_ids = final.get("doc_ids", [])
            print(f"✓ Indexing completed with {len(doc_ids)} chunks: {doc_ids[:3]}...")
            
            # Verify idempotency: re-indexing should skip
            print("\n--- Testing idempotency (re-index same file) ---")
            progress_list = []
            for progress in indexer.index_file(str(test_file)):
                progress_list.append(progress)
                print(f"  [{progress['status']}] {progress['message']}")
            
            if progress_list[0]["status"] != "skip":
                print(f"✗ Idempotency failed: expected 'skip', got '{progress_list[0]['status']}'")
                return False
            print("✓ Idempotency working: file skipped on second run")
            
            # Verify force re-index
            print("\n--- Testing force re-index ---")
            progress_list = []
            for progress in indexer.index_file(str(test_file), force=True):
                progress_list.append(progress)
            
            if progress_list[-1]["status"] != "complete":
                print(f"✗ Force re-index failed")
                return False
            print("✓ Force re-index working: file re-processed")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PHASE 2 TEST SUITE: Document Ingestion Pipeline")
    print("="*60)
    
    results = {
        "Chunker": test_chunker(),
        "Document Loaders": test_doc_loaders(),
        "Ingestion State": test_ingestion_state(),
        "Token Counting": test_chunker_token_count(),
        "Indexer Workflow": test_indexer_workflow(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    print("\n" + ("="*60))
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nPhase 2 is ready for Phase 3 integration!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
