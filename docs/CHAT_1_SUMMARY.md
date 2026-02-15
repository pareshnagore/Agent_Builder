# Agent_ng Development Summary - Chat 1
**Date:** 12 February 2026  
**Context:** Completed Phase 0 (Repo Setup) and Phase 1 (Core Primitives & API Surface)

---

## What Was Accomplished

### Phase 0: Repository & Environment Setup ✅
- Initialized git with main and dev branches
- Created .gitignore (excludes venv, data, db, secrets)
- Created comprehensive README.md with project vision
- Created requirements.txt with all dependencies
- Created .env and .env.example files
- Structured docs/ with complete 13-phase roadmap
- Updated .github/copilot-instructions.md with full vision and guidance for AI assistants

### Phase 1: Core Primitives & Stable API Surface ✅
Completed all three core adapters with robust error handling:

1. **core/config.py** - Centralized Configuration
   - Loads .env automatically using python-dotenv
   - Centralized LLM, embeddings, vector DB, and path configs
   - `validate()` method for checking critical settings
   - `to_dict()` for easy config inspection

2. **core/llm.py** - Unified LLM Adapter
   - `LLMClient` with support for Ollama and Gemini
   - `OllamaProvider` - connects to local Ollama instance
   - `GeminiProvider` - uses google-genai SDK
   - Dynamic model listing from APIs
   - Convenience methods: `chat_ollama()`, `chat_gemini()`

3. **core/embeddings.py** - Embeddings Adapter
   - `EmbeddingsAdapter` with support for Ollama and sentence-transformers
   - Batch processing with configurable batch sizes
   - `get_safe_max_chars()` helper to prevent token overflow
   - `get_embedding_function()` returns Chroma-compatible callable

4. **core/vector_db.py** - Chroma Wrapper
   - **KEY: Automatically creates separate collections per embedding model**
   - Collection naming: `docs_{mode}_{model_name}` (e.g., `docs_ollama_sentence_transformers_all_MiniLM_L6_v2`)
   - Methods: `add_documents()`, `query_text()`, `delete_by_source()`, `reset()`, etc.
   - Integrated with `EmbeddingsAdapter`

5. **test.py** - Testing Application
   - Streamlit UI with separate LLM and embeddings configuration sections
   - Ollama chat model dropdown (filters out embedding models)
   - Separate embedding model selection (ollama text input or sentence-transformer dropdown)
   - Session-based chat history
   - Clean error handling

---

## Key Issues Faced & Solutions

### Issue 1: Gemini API Message Format Error
**Problem:** Initial implementation passed messages in wrong format, causing 61 validation errors
```python
# WRONG: Tried to pass as-is
contents=messages  # {"role": "user", "content": "..."}
```

**Solution:** Convert to proper Gemini format with `parts`
```python
# RIGHT: Convert to Gemini format
{
    "role": "model",  # "assistant" must become "model"
    "parts": [{"text": content}]
}
```

**Learning:** Always check API documentation for exact input format. Gemini's `parts` structure is different from OpenAI's `content`.

---

### Issue 2: Ollama 400 Bad Request
**Problem:** Ollama was returning 400 errors without meaningful error messages

**Solution:** Improved error handling to return actual API response
```python
if response.status_code != 200:
    error_text = response.text
    try:
        error_json = response.json()
        error_text = error_json.get("error", error_text)
    except:
        pass
    raise LLMException(f"Ollama API error ({response.status_code}): {error_text}")
```

**Learning:** Always provide detailed error context in exceptions for easier debugging.

---

### Issue 3: Gemini Free Tier Quota Exceeded
**Problem:** gemini-2.0-flash no longer available in free tier (429 RESOURCE_EXHAUSTED)

**Solution:** 
- Changed default to `gemini-2.5-flash`
- Made model list dynamic using `client.models.list()` API
- Added fallback list for when API is unavailable

**Learning:** Free tier quotas change; dynamic model loading ensures future compatibility.

---

### Issue 4: Embedding & Chat Models Mixed in Dropdown ⚠️ CRITICAL
**Problem:** Ollama embedding models (mxbai-embed-large, nomic-embed-text) were appearing in chat model dropdown

**Solution:** 
- Filter chat models: `chat_models = [m for m in available_models if "embed" not in m.lower()]`
- Separate dropdowns for LLM and embedding selection
- Default embedding model to `mxbai-embed-large`

**Learning:** Models serve different purposes; mixing them in UI is confusing. Separate configuration is essential.

---

### Issue 5: Collection Incompatibility Between Embedding Models ⚠️ ARCHITECTURAL ISSUE
**Problem:** Using same Chroma collection for different embedding models breaks retrieval (embeddings not compatible)

**Solution:** 
**VectorDB automatically creates separate collections per embedding model:**
```python
# Generates collection names like:
# docs_ollama_mxbai_embed_large
# docs_sentence_transformer_all_MiniLM_L6_v2
# docs_ollama_nomic_embed_text

model_name = embedding_adapter.model.replace("/", "_").replace(":", "_").replace(".", "_")
mode = embedding_adapter.mode.replace("-", "_")
collection_name = f"docs_{mode}_{model_name}"
```

**WHY CRITICAL:** If this isn't done, switching embedding models breaks all vector searches (embeddings from different models are incompatible in same collection). This would cause silent failures in production.

---

### Issue 6: Package Dependencies
**Problem:** google-genai package name confusion (was looking for google-generativeai)

**Solution:** Updated requirements.txt with correct packages:
```
google-genai
requests
python-dotenv
sentence-transformers
chromadb
ollama
streamlit
```

---

## Architecture Decisions Made

1. **Adapter Pattern for Extensibility**
   - Each capability (LLM, embeddings, vector DB) has abstract base class
   - Easy to add new providers (e.g., OpenAI, Anthropic embeddings)

2. **Separate Collections Per Embedding Model**
   - Prevents future bugs when switching models
   - Allows A/B testing of different embedding models
   - Scalable design for production

3. **Centralized Configuration**
   - Single source of truth in core/config.py
   - Environment variables override defaults
   - `validate()` method catches missing configs early

4. **Unified Interface**
   - `LLMClient.chat(provider, model, messages)` works for all providers
   - Backward-compatible convenience methods: `chat_ollama()`, `chat_gemini()`

5. **Graceful Error Handling**
   - Custom exceptions with descriptive messages
   - Never crashes Streamlit UI
   - Falls back to sensible defaults

---

## Current Project State

✅ **Completed:**
- Phase 0: Full repo setup with git, docs, env configs
- Phase 1: All three core adapters (LLM, embeddings, vector DB)
- test.py: Working Streamlit app supporting Ollama and Gemini

✅ **Working:**
- Ollama chat (all models)
- Gemini chat (with dynamic model loading)
- Embeddings (ollama and sentence-transformers)
- Vector DB with proper collection segregation

⏭️ **Next:** Phase 2 - Document Ingestion Pipeline
- core/doc_loaders.py (PDF, docx, txt, csv, md)
- core/chunker.py (token-aware chunking)
- core/indexer.py (orchestration + streaming progress)

---

## Files Modified/Created in This Chat

```
✅ Created/Modified:
- .github/copilot-instructions.md (updated with full vision)
- docs/roadmap.md (13 phases with implementation details)
- core/config.py (enhanced with proper env loading)
- core/llm.py (2 providers: Ollama, Gemini)
- core/embeddings.py (2 providers: Ollama, sentence-transformer)
- core/vector_db.py (Chroma wrapper with per-model collections)
- test.py (Streamlit UI with separate LLM/embedding configs)
- test_gemini_direct.py (minimal Gemini API test)
- debug_test.py (configuration & connectivity test)
- requirements.txt (updated with all Phase 1 dependencies)
```

---

## Testing the Current Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Test Gemini API directly
python3 test_gemini_direct.py

# Run the full app
streamlit run test.py
```

---

## Things That Would Break If Not Done Correctly

1. **❌ Not separating collections by embedding model**
   - Vector search fails silently when switching models
   - Data becomes unusable with different embeddings

2. **❌ Not filtering embedding models from chat dropdown**
   - User confusion and potential crashes
   - Attempting to use embedding model for chat throws errors

3. **❌ Not handling Gemini message format correctly**
   - API validation errors
   - 60+ validation error messages make debugging impossible

4. **❌ Not centralizing config**
   - Hardcoded values scattered across files
   - Impossible to switch providers or models without code changes

5. **❌ Not using python-dotenv**
   - API keys hardcoded or forgotten
   - Security risk and deployment headaches

---

## Dependencies Summary
- **LLM:** ollama, google-genai
- **Embeddings:** sentence-transformers, ollama
- **Vector DB:** chromadb
- **Utilities:** requests, python-dotenv
- **UI:** streamlit

---

## Ready for Phase 2?
Once this chat ends and you start a new one, use the **Summary Generation Prompt** (below) to create continuity.

