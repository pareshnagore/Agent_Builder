
# FINAL DETAILED STEP LIST (incremental — grouped into phases)

Below are actionable steps. I include sub-steps where important. You said you don't mind many steps — I keep them concrete and ordered so you can implement and confirm after each group.

## PHASE 0 — Repo & environment (setup)

**Thought:** establish a stable, reproducible workspace.
**Action:** do these before coding agents.

- Create canonical repo `/Users/Paresh/Scripts/Agent_ng` (you already have).
- Initialize git, create main and dev branches.
	- `git init`
- Create `.gitignore` (exclude venv, data, db, secrets).
- Create `README.md` with top-level design vision and usage.
- Create `requirements.in` and `requirements.txt` (pin versions later).
- Add `.env.example` for env vars (`GEMINI_API_KEY`, `OLLAMA_HOST` if needed).
- Add a simple `scripts/` folder for helper scripts (activate, run).
- Create a `docs/` folder for architecture diagrams and this roadmap.

**Observation:** keep dev branch for refactoring; keep old single-file apps on a legacy branch for reference.

## PHASE 1 — Core primitives & stable API surface

**Thought:** build clean, small, well-documented wrappers that other modules call.

- `core/config.py` — central config, load .env (use python-dotenv).
	- read `GEMINI_API_KEY`, `OLLAMA_URL`, `PERSIST_DIR`, default model names.
- `core/llm.py` — LLMAdapter with two public methods:
	- `list_models(provider)`
	- `chat(provider, model, messages, **opts)` (returns assistant text, optional streaming hook)
	- internally supports `provider == "ollama"` and `provider == "gemini"` (new SDK via `from google import genai`)
- `core/embeddings.py` — EmbeddingAdapter:
	- two modes: `mode="ollama"` (call ollama.embeddings) and `mode="sentence-transformer"` (fallback)
	- method: `embed_batch(list[str]) -> list[vector]` supporting batch size control and retry on failure.
	- provide `get_safe_max_chars(model)` helper.
- `core/vector_db.py` — VectorDB wrapper:
	- use `chromadb.PersistentClient(path=...)`
	- constructor accepts `embedding_function` or `embedding_adapter` (`embedding_function` recommended so Chroma can call it directly)
	- methods: `add_documents(docs, metadatas, ids)`, `query_texts(queries, n_results)`, `get_all_metadata()`, `delete_by_source(source)`, `reset()`.
- Add typing, docstrings, and small unit tests for these wrappers.

**Observation:** adapters must raise descriptive exceptions and not crash streamlit UI.

## PHASE 2 — Document ingestion pipeline

**Thought:** robust loaders and token-safe chunking are essential — build once, reuse.

- `core/doc_loaders.py`:
	- `load_pdf(path)`: use pypdf to extract text; fallback to OCR via pdf2image + pytesseract for scanned pages.
	- `load_txt`, `load_docx` (python-docx), `load_md`, `load_csv` (read cells).
- `core/chunker.py`:
	- implement token/char-aware chunking: `chunk_text(text, max_chars)` with optional overlap.
	- implement `token_estimate(text)` if you want (approx chars->tokens).
	- implement `shrink_and_clean(chunk)` to trim weird whitespace, remove long binary artifacts.
- `core/indexer.py`:
	- orchestrates: for each doc -> loader -> chunker -> call `vectordb.add_documents` (batch)
	- create unique `doc_id` and `chunk_id` convention (e.g. `source_filename__chunk_0001`)
	- support per-model collections (e.g., `docs_mxbai-embed-large` vs `docs_embeddinggemma`) or a metadata field `embed_model`.
	- Add streaming/batching progress API for UI (yield progress events).

**Observation:** keep ingestion idempotent — detect `doc_id` presence to avoid reindexing.

## PHASE 3 — RAG & retrieval policies

**Thought:** implement core RAG retrieval and safety policies.

- `core/rag.py` (refined RAGEngine):
	- method `index_documents(texts, source, embed_model)` — uses indexer and vectordb; ensure consistent with your earlier working design (Chroma embedding function used).
	- method `retrieve(query, top_k, filters)` returns contexts + provenance.
- Implement retrieval policies:
	- strict: return top-k with source citations
	- relaxed: expand search with semantic similarity threshold
	- hybrid: BM25-like fallback (if you add an inverted-index)
- Implement context size trimming for LLM prompt assembly: `build_prompt(context_items, system_prompt, question, max_prompt_chars)`.
- Add `query_logger` to record queries, times, and which docs used.

**Observation:** always prefer to send small combined context to LLM (concatenate top N chunks but keep total chars under safe limit for chosen chat model).

## PHASE 4 — Speech (STT + TTS)

**Thought:** make audio optional and toggleable; support offline-first.

- `core/speech.py`:
	- STT adapters:
		- Google STT (online)
		- whisper-local via Ollama or local whisper package (offline)
	- Wrapper method `transcribe(audio, mode="whisper"|"google")`
	- TTS adapters:
		- pyttsx3 (macOS native)
		- coqui/tts if you want custom voices later
	- `speak_text(text, engine="pyttsx3", rate=None, voice=None)`
- Integrate STT/TTS in UI via SpeechService with clear enable toggles and fallback when offline.

**Observation:** Provide a “Use offline STT” flag. By default allow online STT for better accuracy until offline Whisper is configured.

## PHASE 5 — Tools & Connectors (safe, sandboxed)

**Thought:** agents must call tools in a controlled way; design a small tool registry.

- `tools/file_tools.py`:
	- safe file IO: `download_from_url`, `save_to_folder`, `list_folder`, `watch_folder`
- `tools/code_tools.py`:
	- `run_python_snippet(code, sandbox=True)` — use subprocess with temporary files; capture stdout/stderr
	- add timeouts & resource limits
- `tools/web_tools.py`:
	- `http_get(url)`, `html_text(url)`, `scrape(url, selector)` — use requests, beautifulsoup
	- rate-limiting & caching
- `tools/email_tools.py`:
	- `send_email(smtp_config, to, subject, body, attachments)` — secure via user-provided credentials
- `tools/drive_tools.py`:
	- connectors for Google Drive (optional) or local sync (rclone)
- Provide tests and restrictive policies per tool (which agents can call which tools).

**Observation:** tool calls must be auditable and require explicit user approval for risky operations.

## PHASE 6 — Agent Framework (planner + executors)

**Thought:** implement a layered agent model: BaseAgent, PlannerAgent, ExecutorAgent, Subagents.

- `agents/base_agent.py`:
	- defines: Agent interface: `plan(task)`, `decide(action)`, `execute(action)`, `observe(result)`.
	- supports dry_run mode and logs actions.
- `agents/planner.py`:
	- given goal, returns plan as list of atomic steps (use LLM to create steps)
	- add deterministic fallback templates for known tasks (e.g., EDA pipeline)
- `agents/executor.py`:
	- executes steps by calling Tools, RAG, LLM; each action is single responsibility
	- enforce step authorization (user confirmation, auto-approve config)
- `agents/manager.py`:
	- orchestrates subagents (EDA agent, Fetch agent, FE agent, Model builder agent)
	- supports concurrency control and state persistence
- `agents/tool_agent.py`:
	- wrapper that maps a requested tool call to the actual `tools.*` implementation and validates arguments (types, sizes)
- Implement an action spec system: every tool action has a JSON schema describing expected inputs and outputs so agents can validate before invocation.

**Observation:** Keep planner outputs small and testable; start with rule-based planners and then allow LLM-based planners.

## PHASE 7 — Workflow engine & recipes

**Thought:** represent workflows as versioned recipes (YAML/JSON) that agents can execute.

- Define recipe schema:
	- id, version, steps: each step references agent and action and params.
- `workflow/runner.py`:
	- executes recipe steps, persists state, supports pause/resume, rollback
- Provide UI for creating/editing recipes and for running them interactively (step confirmation).

**Observation:** this enables the repetitive automation tasks you described (download-run-upload-mail).

## PHASE 8 — Security, sandboxing & safety

**Thought:** agents may execute code and web actions — sandbox everything.

- Implement subprocess sandboxing for code execution:
	- resource limits, network disabled by default, ephemeral containers preferred (or use local chroot)
- Implement a policy layer:
	- which agent can call which tool
	- maximum file sizes, allowed domains for web requests, allowed email recipients
- Audit logs:
	- record all tool calls, inputs (redact secrets), outputs, and user approvals

**Observation:** this is critical before you allow any real-world automation (uploading, emailing, deleting).

## PHASE 9 — UI & Developer UX (VSCode & Streamlit)

**Thought:** enable both interactive GUI for non-coders and a developer playground in VSCode.

- Streamlit UI: multi-page app
	- Playground (chat + RAG)
	- Indexer (folder watch + manual)
	- Agent Runner (start/pause/resume workflows)
	- Tools console (run a tool manually)
- VSCode integration:
	- Add a simple extension or use the “Live Share / Local Webview” approach: focus on a dev-mode page that shows logs, plan steps, and allows inline code editing for code-run steps.
	- Implement comprehensive logs & downloadable transcripts.

**Observation:** prioritize Streamlit to iterate fast; add VSCode integration in a follow-up sprint.

## PHASE 10 — Testing, CI, reproducibility

**Thought:** each module must have unit + integration tests.

- Add `tests/` and write unit tests for:
	- llm adapter (mock LLM)
	- embeddings (mock)
	- vector DB (local ephemeral store)
	- doc loaders (small test docs)
	- tools (mock external calls)
- Add simple integration tests:
	- index a small doc and query it end-to-end
	- run a mini-workflow end-to-end
- Set up GitHub Actions or local CI:
	- run tests, linting, static analysis
- Add reproducible docker dev env if needed (optional).

**Observation:** tests prevent regressions as you expand with many agents.

## PHASE 11 — Examples & templates (the fun part)

**Thought:** deliver ready-to-run agents so you can immediately use / extend.

- Template: Data-Scientist Agent (EDA → FE → Model → Compare → PPT)
	- subagents: EDA, FeatEng, ModelTrainer, Evaluator, PPTExporter
- Template: Automation Agent (download → run → upload → email)
- Template: Builder Agent (project scaffold → frontend agent → backend agent → integration tests)
- Template: Photo Culler (image loader → vision model scorer → UI cull suggestions)
- Provide sample datasets and sample PDFs to test.

**Observation:** templates accelerate real use and demonstrate how to compose agents.

## PHASE 12 — Observability, backup & ops

**Thought:** ensure you can restore/inspect the system later.

- Periodic DB backups (chroma DB export)
- Config backup (encrypted)
- Add a health endpoint for checking service readiness (useful for local scripts)
- Add a simple metrics/log aggregator (local files + rotate)

## PHASE 13 — Iterative polish & advanced features

**Thought:** once stable, add advanced capabilities.

- Conversational RAG memory with citations and provenance
- Agent meta-learning (teacher prompts, prompt templates, self-eval)
- Plugin system to easily add new tools
- Multi-agent orchestration: distributed agents on different machines (optional)

---

## PRACTICAL IMPLEMENTATION GUIDE (how to consume the list)

**Thought:** implement in small sprints — each sprint = one phase or subphase.

**Action (recommended cadence):**

- Sprint 1: PHASE 0 + PHASE 1 (steps 1–12)
- Sprint 2: PHASE 2 + PHASE 3 (13–20)
- Sprint 3: PHASE 4 + PHASE 5 (21–28)
- Sprint 4: PHASE 6 + PHASE 7 (29–37)
- Sprint 5: PHASE 8 + PHASE 9 (38–43)
- Sprint 6: PHASE 10 + PHASE 11 + monitoring (44–56)
- After that, advanced features iteratively (57—60+)

**Observation:** you can stop and use the system productively from end of Sprint 3 or 4.

---

## RISKS & MITIGATIONS (short)

- Model limits / latency — add token-aware chunking and per-model safe sizes.
- Data loss — backups & safe deletes.
- Security — tool policies and sandboxed code execution.
- Complexity creep — keep minimal viable features per sprint and document interfaces.

