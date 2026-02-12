# Agent_ng Roadmap

## Project Vision
Agent_ng is a professional-grade platform for building advanced AI agents, systems, and workflows to automate and augment day-to-day tasks. The goal is to create robust, reusable building blocks for multi-agent orchestration, modular RAG pipelines, and structured workflows.

---

## PHASE 0 — Repo & environment (setup)

**Thought:** Establish a stable, reproducible workspace.

**Action:** Do these before coding agents.

- Create canonical repo `/Users/Paresh/Scripts/Agent_ng` (you already have).
- Initialize git, create main and dev branches.
  - `git init`
- Create `.gitignore` (exclude venv, data, db, secrets).
- Create `README.md` with top-level design vision and usage.
- Create `requirements.txt` (pin versions later).
- Add `.env.example` for env vars (GEMINI_API_KEY, OLLAMA_HOST if needed).
- Add a simple `scripts/` folder for helper scripts (activate, run).
- Create a `docs/` folder for architecture diagrams and this roadmap.

**Observation:** Keep dev branch for refactoring; keep old single-file apps on a legacy branch for reference.

---

## PHASE 1 — Core primitives & stable API surface

**Thought:** Build clean, small, well-documented wrappers that other modules call.

**Detailed Implementation:**

- **core/config.py** — Central config, load .env (use python-dotenv).
  - Read GEMINI_API_KEY, OLLAMA_URL, PERSIST_DIR, default model names.

- **core/llm.py** — LLMAdapter with two public methods:
  - `list_models(provider)`
  - `chat(provider, model, messages, **opts)` (returns assistant text, optional streaming hook)
  - Internally supports provider == "ollama" and provider == "gemini" (new SDK via `from google import genai`)

- **core/embeddings.py** — EmbeddingAdapter:
  - Two modes: mode="ollama" (call ollama.embeddings) and mode="sentence-transformer" (fallback)
  - Method: `embed_batch(list[str]) -> list[vector]` supporting batch size control and retry on failure.
  - Provide `get_safe_max_chars(model)` helper.

- **core/vector_db.py** — VectorDB wrapper:
  - Use `chromadb.PersistentClient(path=...)`
  - Constructor accepts embedding_function or embedding_adapter (embedding_function recommended so Chroma can call it directly)
  - Methods: `add_documents(docs, metadatas, ids)`, `query_texts(queries, n_results)`, `get_all_metadata()`, `delete_by_source(source)`, `reset()`.

- Add typing, docstrings, and small unit tests for these wrappers.

**Observation:** Adapters must raise descriptive exceptions and not crash Streamlit UI.

---

## PHASE 2 — Document ingestion pipeline

**Thought:** Robust loaders and token-safe chunking are essential — build once, reuse.

**Detailed Implementation:**

- **core/doc_loaders.py:**
  - `load_pdf(path)`: use pypdf to extract text; fallback to OCR via pdf2image + pytesseract for scanned pages.
  - `load_txt`, `load_docx` (python-docx), `load_md`, `load_csv` (read cells).

- **core/chunker.py:**
  - Implement token/char-aware chunking: `chunk_text(text, max_chars)` with optional overlap.
  - Implement `token_estimate(text)` if you want (approx chars->tokens).
  - Implement `shrink_and_clean(chunk)` to trim weird whitespace, remove long binary artifacts.

- **core/indexer.py:**
  - Orchestrates: for each doc -> loader -> chunker -> call `vectordb.add_documents` (batch)
  - Create unique doc_id and chunk_id convention (e.g. source_filename__chunk_0001)
  - Support per-model collections (e.g., docs_mxbai-embed-large vs docs_embeddinggemma) or a metadata field embed_model.
  - Add streaming/batching progress API for UI (yield progress events).

**Observation:** Keep ingestion idempotent — detect doc_id presence to avoid reindexing.

---

## PHASE 3 — RAG & retrieval policies

**Thought:** Implement core RAG retrieval and safety policies.

**Detailed Implementation:**

- **core/rag.py** (refined RAGEngine):
  - Method `index_documents(texts, source, embed_model)` — uses indexer and vectordb; ensure consistent with your earlier working design (Chroma embedding function used).
  - Method `retrieve(query, top_k, filters)` returns contexts + provenance.
  - Implement retrieval policies:
    - **strict:** return top-k with source citations
    - **relaxed:** expand search with semantic similarity threshold
    - **hybrid:** BM25-like fallback (if you add an inverted-index)
  - Implement context size trimming for LLM prompt assembly: `build_prompt(context_items, system_prompt, question, max_prompt_chars)`.
  - Add query_logger to record queries, times, and which docs used.

**Observation:** Always prefer to send small combined context to LLM (concatenate top N chunks but keep total chars under safe limit for chosen chat model).

---

## PHASE 4 — Speech (STT + TTS)

**Thought:** Make audio optional and toggleable; support offline-first.

**Detailed Implementation:**

- **core/speech.py:**
  - **STT adapters:**
    - Google STT (online)
    - whisper-local via Ollama or local whisper package (offline)
    - Wrapper method `transcribe(audio, mode="whisper"|"google")`
  - **TTS adapters:**
    - pyttsx3 (macOS native)
    - coqui/tts if you want custom voices later
    - Method `speak_text(text, engine="pyttsx3", rate=None, voice=None)`
  - Integrate STT/TTS in UI via SpeechService with clear enable toggles and fallback when offline.

**Observation:** Provide a "Use offline STT" flag. By default allow online STT for better accuracy until offline Whisper is configured.

---

## PHASE 5 — Tools & Connectors (safe, sandboxed)

**Thought:** Agents must call tools in a controlled way; design a small tool registry.

**Detailed Implementation:**

- **tools/file_tools.py:**
  - Safe file IO: `download_from_url`, `save_to_folder`, `list_folder`, `watch_folder`

- **tools/code_tools.py:**
  - `run_python_snippet(code, sandbox=True)` — use subprocess with temporary files; capture stdout/stderr
  - Add timeouts & resource limits

- **tools/web_tools.py:**
  - `http_get(url)`, `html_text(url)`, `scrape(url, selector)` — use requests, beautifulsoup
  - Rate-limiting & caching

- **tools/email_tools.py:**
  - `send_email(smtp_config, to, subject, body, attachments)` — secure via user-provided credentials

- **tools/drive_tools.py:**
  - Connectors for Google Drive (optional) or local sync (rclone)

- Provide tests and restrictive policies per tool (which agents can call which tools).

**Observation:** Tool calls must be auditable and require explicit user approval for risky operations.

---

## PHASE 6 — Agent Framework (planner + executors)

**Thought:** Implement a layered agent model: BaseAgent, PlannerAgent, ExecutorAgent, Subagents.

**Detailed Implementation:**

- **agents/base_agent.py:**
  - Defines: Agent interface: `plan(task)`, `decide(action)`, `execute(action)`, `observe(result)`.
  - Supports dry_run mode and logs actions.

- **agents/planner.py:**
  - Given goal, returns plan as list of atomic steps (use LLM to create steps)
  - Add deterministic fallback templates for known tasks (e.g., EDA pipeline)

- **agents/executor.py:**
  - Executes steps by calling Tools, RAG, LLM; each action is single responsibility
  - Enforce step authorization (user confirmation, auto-approve config)

- **agents/manager.py:**
  - Orchestrates subagents (EDA agent, Fetch agent, FE agent, Model builder agent)
  - Supports concurrency control and state persistence

- **agents/tool_agent.py:**
  - Wrapper that maps a requested tool call to the actual tools.* implementation and validates arguments (types, sizes)
  - Implement an action spec system: every tool action has a JSON schema describing expected inputs and outputs so agents can validate before invocation.

**Observation:** Keep planner outputs small and testable; start with rule-based planners and then allow LLM-based planners.

---

## PHASE 7 — Workflow engine & recipes

**Thought:** Represent workflows as versioned recipes (YAML/JSON) that agents can execute.

**Detailed Implementation:**

- Define recipe schema:
  - id, version, steps: each step references agent and action and params.

- **workflow/runner.py:**
  - Executes recipe steps, persists state, supports pause/resume, rollback

- Provide UI for creating/editing recipes and for running them interactively (step confirmation).

**Observation:** This enables the repetitive automation tasks you described (download-run-upload-mail).

---

## PHASE 8 — Security, sandboxing & safety

**Thought:** Agents may execute code and web actions — sandbox everything.

**Detailed Implementation:**

- Implement subprocess sandboxing for code execution:
  - Resource limits, network disabled by default, ephemeral containers preferred (or use local chroot)

- Implement a policy layer:
  - Which agent can call which tool
  - Maximum file sizes, allowed domains for web requests, allowed email recipients

- Audit logs:
  - Record all tool calls, inputs (redact secrets), outputs, and user approvals

**Observation:** This is critical before you allow any real-world automation (uploading, emailing, deleting).

---

## PHASE 9 — UI & Developer UX (VSCode & Streamlit)

**Thought:** Enable both interactive GUI for non-coders and a developer playground in VSCode.

**Detailed Implementation:**

- **Streamlit UI:** Multi-page app
  - Playground (chat + RAG)
  - Indexer (folder watch + manual)
  - Agent Runner (start/pause/resume workflows)
  - Tools console (run a tool manually)

- **VSCode integration:**
  - Add a simple extension or use the "Live Share / Local Webview" approach: focus on a dev-mode page that shows logs, plan steps, and allows inline code editing for code-run steps.
  - Implement comprehensive logs & downloadable transcripts.

**Observation:** Prioritize Streamlit to iterate fast; add VSCode integration in a follow-up sprint.

---

## PHASE 10 — Testing, CI, reproducibility

**Thought:** Each module must have unit + integration tests.

**Detailed Implementation:**

- Add `tests/` and write unit tests for:
  - llm adapter (mock LLM)
  - embeddings (mock)
  - vector DB (local ephemeral store)
  - doc loaders (small test docs)
  - tools (mock external calls)

- Add simple integration tests:
  - Index a small doc and query it end-to-end
  - Run a mini-workflow end-to-end

- Set up GitHub Actions or local CI:
  - Run tests, linting, static analysis

- Add reproducible docker dev env if needed (optional).

**Observation:** Tests prevent regressions as you expand with many agents.

---

## PHASE 11 — Examples & templates (the fun part)

**Thought:** Deliver ready-to-run agents so you can immediately use / extend.

**Detailed Implementation:**

- **Template: Data-Scientist Agent** (EDA → FE → Model → Compare → PPT)
  - Subagents: EDA, FeatEng, ModelTrainer, Evaluator, PPTExporter

- **Template: Automation Agent** (download → run → upload → email)

- **Template: Builder Agent** (project scaffold → frontend agent → backend agent → integration tests)

- **Template: Photo Culler** (image loader → vision model scorer → UI cull suggestions)

- Provide sample datasets and sample PDFs to test.

**Observation:** Templates accelerate real use and demonstrate how to compose agents.

---

## PHASE 12 — Observability, backup & ops

**Thought:** Ensure you can restore/inspect the system later.

**Detailed Implementation:**

- Periodic DB backups (chroma DB export)
- Config backup (encrypted)
- Add a health endpoint for checking service readiness (useful for local scripts)
- Add a simple metrics/log aggregator (local files + rotate)

---

## PHASE 13 — Iterative polish & advanced features

**Thought:** Once stable, add advanced capabilities.

**Detailed Implementation:**

- Conversational RAG memory with citations and provenance
- Agent meta-learning (teacher prompts, prompt templates, self-eval)
- Plugin system to easily add new tools
- Multi-agent orchestration: distributed agents on different machines (optional)

---

## PRACTICAL IMPLEMENTATION GUIDE (how to consume the list)

**Thought:** Implement in small sprints — each sprint = one phase or subphase.

**Action (recommended cadence):**

- **Sprint 1:** PHASE 0 + PHASE 1
- **Sprint 2:** PHASE 2 + PHASE 3
- **Sprint 3:** PHASE 4 + PHASE 5
- **Sprint 4:** PHASE 6 + PHASE 7
- **Sprint 5:** PHASE 8 + PHASE 9
- **Sprint 6:** PHASE 10 + PHASE 11 + monitoring
- **After that:** Advanced features iteratively

**Observation:** You can stop and use the system productively from end of Sprint 3 or 4.

---

## RISKS & MITIGATIONS (short)

- **Model limits / latency:** Add token-aware chunking and per-model safe sizes.
- **Data loss:** Backups & safe deletes.
- **Security:** Tool policies and sandboxed code execution.
- **Complexity creep:** Keep minimal viable features per sprint and document interfaces.

---

## Architecture Overview

```
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
│   ├── config.py              → Central config
│   ├── doc_loaders.py         → Document loading
│   ├── chunker.py             → Token-aware chunking
│   ├── indexer.py             → Ingestion orchestration
│   └── loader.py              → (existing)
│
├── agents/
│   ├── base_agent.py
│   ├── planner.py
│   ├── executor.py
│   ├── manager.py
│   ├── tool_agent.py
│   ├── rag_agent.py
│   ├── coding_agent.py
│   └── __init__.py
│
├── tools/
│   ├── file_tools.py
│   ├── code_tools.py
│   ├── web_tools.py
│   ├── email_tools.py
│   ├── drive_tools.py
│   └── __init__.py
│
├── workflow/
│   ├── runner.py
│   └── recipes/
│
├── tests/
│   ├── test_llm.py
│   ├── test_embeddings.py
│   ├── test_vector_db.py
│   ├── test_loaders.py
│   └── test_tools.py
│
├── data/
│   ├── uploads/
│   └── vector_store/
│
├── db/
│
├── docs/
│   ├── roadmap.md             → This file
│   └── architecture/
│
└── requirements.txt
```

---

## Notes for Developers

- Always follow step-by-step, incremental development — one phase at a time.
- Each phase should be testable and usable independently where possible.
- Maintain backward compatibility within phases.
- Document all public APIs with docstrings and type hints.
- Keep error handling robust and user-friendly.

---

**Last Updated:** 12 February 2026
