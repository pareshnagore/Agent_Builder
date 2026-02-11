# Copilot Instructions for Agent_ng

## Project Overview
This project implements a modular Retrieval-Augmented Generation (RAG) system in Python. The architecture is organized around core components for embeddings, LLM interaction, data loading, vector database management, and configuration. The main entry points are `rag_app.py` and `app.py`.

## Key Components
- **core/**: Contains the main logic for RAG workflows:
  - `embeddings.py`: Embedding model logic
  - `llm.py`: Large Language Model interface
  - `loader.py`: Data/document loading
  - `vector_db.py`: Vector database (Chroma) integration
  - `rag.py`: Orchestrates RAG pipeline
  - `config.py`: Centralized configuration
- **agents/** and **tools/**: For agent/tool extensions (currently minimal)
- **data/uploads/**: User-uploaded data
- **data/vector_store/** and **db/**: Chroma vector DB storage

## Developer Workflows
- **Run the app**: `python rag_app.py` (main RAG pipeline)
- **Dependencies**: Install from `requirements.txt`
- **Data**: Place files in `data/uploads/` for ingestion
- **Vector DB**: Uses Chroma, data stored in `data/vector_store/` and `db/`

## Project Conventions
- All core logic is in `core/` and is imported by the main app
- Configuration is centralized in `core/config.py`
- Data flows: loader → embeddings/vector_db → rag orchestrator → LLM
- Use absolute imports within the project (e.g., `from core.llm import ...`)
- Avoid modifying files in `data/vector_store/` and `db/` directly

## Integration Points
- **Chroma**: Used for vector storage (see `vector_db.py`)
- **LLM**: Abstracted in `llm.py` (model details/config in `config.py`)
- **Embeddings**: Model selection/config in `embeddings.py` and `config.py`

## Examples
- To add a new embedding model, extend `core/embeddings.py` and update `core/config.py`
- To change LLM provider, update `core/llm.py` and `core/config.py`
- To add new data sources, extend `core/loader.py`

## References
- Main entry: [rag_app.py](../../rag_app.py)
- Core logic: [core/](../../core/)
- Configuration: [core/config.py](../../core/config.py)
- Vector DB: [core/vector_db.py](../../core/vector_db.py)

---
For questions or unclear patterns, review the main app and core modules, or ask for clarification.
