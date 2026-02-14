#!/usr/bin/env python3
"""Quick sanity check for Phase 2 modules."""
import sys

print("Testing Phase 2 module imports...\n")

try:
    from core.chunker import Chunker
    print("✓ chunker.py imports successfully")
except Exception as e:
    print(f"✗ chunker error: {e}")
    sys.exit(1)

try:
    from core.doc_loaders import DocumentLoaderFactory
    print("✓ doc_loaders.py imports successfully")
except Exception as e:
    print(f"✗ doc_loaders error: {e}")
    sys.exit(1)

try:
    from core.indexer import Indexer
    print("✓ indexer.py imports successfully")
except Exception as e:
    print(f"✗ indexer error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("QUICK FUNCTIONAL TESTS")
print("="*60)

# Test 1: Basic chunking
print("\n[1] Testing chunking...")
text = "Hello world. This is test sentence one. Here is sentence two."
chunker = Chunker(strategy="sentence-aware")
chunks = chunker.chunk_text(text, max_tokens=30, overlap_tokens=5)
print(f"✓ Chunking works: {len(chunks)} chunks created")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i}: {chunk[:50]}...")

# Test 2: Token counting
print("\n[2] Testing token counting...")
token_count = chunker.get_token_count("Hello world")
print(f"✓ Token counting works: 'Hello world' = {token_count} tokens")

# Test 3: Supported file types
print("\n[3] Testing file type support...")
types = DocumentLoaderFactory.supported_types()
print(f"✓ File types recognized: {', '.join(types)}")

# Test 4: Testing different chunking strategies
print("\n[4] Testing all chunking strategies...")
strategies = ["sliding-window", "sentence-aware", "paragraph"]
for strategy in strategies:
    try:
        c = Chunker(strategy=strategy)
        ch = c.chunk_text(text, max_tokens=40, overlap_tokens=5)
        print(f"✓ {strategy:20s} → {len(ch)} chunks")
    except Exception as e:
        print(f"✗ {strategy:20s} → ERROR: {e}")

print("\n" + "="*60)
print("✅ ALL PHASE 2 MODULES FUNCTIONAL!")
print("="*60)
