# Agent_ng

## Vision
Agent_ng aims to implement modular, multi-agent orchestration for Retrieval-Augmented Generation (RAG) workflows. The project is designed for extensibility, reproducibility, and easy integration of new agents, tools, and data sources.

## Features
- Modular RAG pipeline (embeddings, LLM, vector DB, loader)
- Multi-agent orchestration (planned)
- Streamlit UI for chat
- Chroma vector DB integration

## Usage
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the app:
   ```
   python rag_app.py
   ```
3. Place data files in `data/uploads/` for ingestion.

## Roadmap
- Phase 0: Repo & environment setup
- Phase 1: Agent orchestration
- Phase 2: Tool integration
- Phase 3: Advanced workflows

## Folder Structure
- `core/`: Main logic
- `agents/`: Agent extensions
- `tools/`: Tool extensions
- `data/`: Uploaded files and vector DB
- `db/`: Chroma DB storage

---

For architecture diagrams and roadmap, see the `docs/` folder.