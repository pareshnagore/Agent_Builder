
# Copilot Instructions for Agent_ng

## Vision & Objective
Agent_ng is a professional-grade platform for building advanced AI agents, systems, and workflows to automate and augment day-to-day tasks. The goal is to create robust, reusable building blocks for:
- Multi-agent orchestration (e.g., data scientist agents, automation agents, software builder agents, vision agents)
- Modular, extensible RAG (Retrieval-Augmented Generation) pipelines
- Structured, context-aware workflows that can be composed for new use cases

**Key principles:**
- Build for extensibility and reusability (plug-and-play agents/components)
- Professional quality, even for single-user scenarios
- Robust error handling, clear messaging, and best practices in agent orchestration and context engineering
- Step-by-step, incremental development for maximum clarity and maintainability

## Example Use Cases
- Data scientist agent: EDA, feature engineering, model building, comparison, reporting (with subagents for each step)
- Automation agent: Handles repetitive workflows (e.g., download, process, upload, notify)
- Software builder agent: Modular agents for frontend, backend, integration, etc.
- Vision/photo culling agent: Uses open-source models and modular software engineering

## How AI Assistants Should Help
- Always provide step-by-step, incremental guidance and code changes
- Prioritize robust, testable, and well-documented code
- Help design and implement reusable agent, tool, and workflow components
- Follow and refine the project plan below, adapting as needed to achieve the broader vision
- Maintain context across chats and phases, so progress is cumulative

## Project Plan & Phases
The following plan is the canonical roadmap for Agent_ng. AI assistants should use this as the basis for all guidance, code generation, and architectural decisions. Refine and adapt as needed to best serve the overall objective.

### PHASE 0 — Repo & environment (setup)
Establish a stable, reproducible workspace:
- Canonical repo structure
- Git with main/dev branches
- .gitignore (exclude venv, data, db, secrets)
- README.md (vision, usage)
- requirements.txt (pin versions later)
- .env.example for env vars (GEMINI_API_KEY, OLLAMA_HOST, etc.)
- scripts/ for helper scripts (optional)
- docs/ for architecture diagrams and roadmap

### PHASE 1 — Core primitives & stable API surface
Build clean, well-documented wrappers for config, LLM, embeddings, vector DB, with robust error handling and typing. Adapters must raise descriptive exceptions and not crash the UI.

### PHASE 2 — Document ingestion pipeline
Robust loaders and token-safe chunking. Idempotent ingestion, streaming progress, and metadata tracking.

### PHASE 3 — RAG & retrieval policies
Implement core RAG retrieval, context trimming, and query logging. Support strict/relaxed/hybrid retrieval policies.

### PHASE 4 — Speech (STT + TTS)
Optional audio support (online/offline), with clear toggles and fallbacks.

### PHASE 5 — Tools & Connectors (safe, sandboxed)
Design a tool registry and safe wrappers for file, code, web, email, and drive operations. All tool calls must be auditable and require explicit user approval for risky actions.

### PHASE 6 — Agent Framework (planner + executors)
Layered agent model: BaseAgent, PlannerAgent, ExecutorAgent, Subagents. Action spec system for tool calls. Planners can be rule-based or LLM-based.

### PHASE 7 — Workflow engine & recipes
Workflows as versioned recipes (YAML/JSON) that agents can execute. UI for creating/editing/running recipes interactively.

### PHASE 8 — Security, sandboxing & safety
Sandbox all code/web actions. Policy layer for tool/agent permissions. Audit logs for all tool calls and approvals.

### PHASE 9 — UI & Developer UX (VSCode & Streamlit)
Multi-page Streamlit app for chat, indexing, agent runner, and tools console. VSCode integration for developer playground and logs.

### PHASE 10 — Testing, CI, reproducibility
Unit and integration tests for all modules. GitHub Actions/local CI. Optional Docker dev env.

### PHASE 11 — Examples & templates
Ready-to-run agent templates (data scientist, automation, builder, photo culler). Sample datasets and docs.

### PHASE 12 — Observability, backup & ops
Periodic DB/config backups, health endpoint, metrics/log aggregation.

### PHASE 13 — Iterative polish & advanced features
Conversational RAG memory, agent meta-learning, plugin system, distributed agents.

## Project Structure (AI-Recommended)

my_ai_framework/
│
├── app.py                     → Streamlit UI
│
├── core/
│   ├── llm.py                 → Ollama/Gemini wrapper
│   ├── embeddings.py          → Embedding model handler
│   ├── vector_db.py           → Chroma wrapper
│   ├── rag.py                 → Retrieval logic
│   ├── speech.py              → STT / TTS
│   ├── memory.py              → Chat history manager
│   └── config.py              → Central config
│
├── agents/
│   ├── base_agent.py
│   ├── rag_agent.py
│   ├── coding_agent.py
│   └── tool_agent.py
│
├── tools/
│   ├── file_tools.py
│   ├── web_tools.py
│   └── system_tools.py
│
├── data/
│   └── vector_store/
│
└── requirements.txt

## Coding & Collaboration Conventions
- All core logic in core/, imported by main app
- Centralized config in core/config.py
- Data flows: loader → embeddings/vector_db → rag orchestrator → LLM
- Use absolute imports (e.g., from core.llm import ...)
- Do not modify files in data/vector_store/ or db/ directly
- Add new models, agents, or tools by extending the appropriate module and updating config
- Always write robust error handling and clear user/developer messages
- Document all public APIs and provide typing

## SDK Usage Guidelines
- Use documentation from https://ai.google.dev/gemini-api/ for Gemini integration
- For Ollama, refer to https://ollama.com/docs/api for API usage and best practices
- Follow the principle of least privilege when designing tool wrappers (e.g., file access should be sandboxed and require explicit user approval)
- Ensure all API calls are wrapped with error handling that provides clear feedback to the user and does not crash the app

## How to Use These Instructions
- AI assistants should always reference this file for context, even in new chats or when working on different phases
- Maintain awareness of the broader vision and incremental plan
- When in doubt, ask clarifying questions about objectives, use cases, or design preferences
- Help the user progress step by step, confirming each phase or subtask before moving on

---
For questions or unclear patterns, review the main app, core modules, and this plan, or ask for clarification.
