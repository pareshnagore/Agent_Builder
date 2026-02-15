"""
Test application for Agent_ng Phases 0-3: Complete RAG Pipeline.
Comprehensive Streamlit app with document ingestion, indexing, and RAG retrieval.

Phases tested:
  - Phase 0: Configuration & environment setup
  - Phase 1: LLM, Embeddings, Vector DB adapters
  - Phase 2: Document loaders, chunking, idempotent indexing
  - Phase 3: RAG engine, retrieval policies, query logging
"""

import streamlit as st
import json
from pathlib import Path
from typing import Optional, List, Tuple

# Phase 1 imports
from core.config import Config
from core.llm import LLMClient, LLMException
from core.embeddings import EmbeddingsAdapter, EmbeddingsException
from core.vector_db import VectorDB, VectorDBException

# Phase 2 imports
from core.doc_loaders import DocumentLoaderFactory, LoaderException
from core.chunker import Chunker
from core.indexer import Indexer, IngestionState

# Phase 3 imports
from core.rag import RAGEngine, RetrievalPolicy
from core.query_logger import QueryLogger


# ========== INITIALIZATION HELPERS ==========
def initialize_llm() -> Optional[LLMClient]:
    """Initialize and return LLM client."""
    try:
        return LLMClient()
    except LLMException as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None


def initialize_embeddings(mode: str) -> Optional[EmbeddingsAdapter]:
    """Initialize and return embeddings adapter."""
    try:
        return EmbeddingsAdapter(mode=mode)
    except EmbeddingsException as e:
        st.error(f"Failed to initialize embeddings: {str(e)}")
        return None


def initialize_vector_db(embedding_adapter: EmbeddingsAdapter) -> Optional[VectorDB]:
    """Initialize and return vector database."""
    try:
        return VectorDB(embedding_adapter=embedding_adapter)
    except VectorDBException as e:
        st.error(f"Failed to initialize vector DB: {str(e)}")
        return None


def initialize_rag_engine(
    embedding_mode: str,
    embedding_model: str
) -> Optional[RAGEngine]:
    """Initialize and return RAG engine (creates its own components internally)."""
    try:
        return RAGEngine(
            embedding_mode=embedding_mode,
            embedding_model=embedding_model,
            persist_dir="data/vector_store",
            state_file="data/ingestion_state.json",
            logs_dir="data/logs"
        )
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {str(e)}")
        return None


# ========== FILE HANDLING ==========
def save_uploaded_file(uploaded_file) -> Optional[Path]:
    """Save uploaded file to data/uploads and return path."""
    try:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Failed to save file: {str(e)}")
        return None


# ========== STREAMLIT PAGE CONFIG ==========
st.set_page_config(
    page_title="Agent_ng: Phase 0-3 Full RAG Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Agent_ng: Phases 0-3 Complete RAG Pipeline")
st.markdown("**Full workflow:** Document upload → Chunking → Indexing → RAG Retrieval → Chat")

# ========== SESSION STATE INITIALIZATION ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
def refresh_indexed_documents(rag_engine):
    """Refresh indexed documents from ingestion state."""
    docs = []
    try:
        # Force reload from disk to get latest state
        rag_engine.indexer.state._load()
        state = rag_engine.indexer.state.state
        for filename, file_info in state.items():
            docs.append({
                "name": filename,
                "path": file_info.get("file_path", ""),
                "documents": len(file_info.get("doc_ids", []))
            })
    except Exception:
        pass
    return docs

if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None


# ========== SIDEBAR: PHASE 1 & 3 CONFIGURATION ==========
st.sidebar.header("⚙️ Configuration")

# === PHASE 1: LLM Configuration ===
st.sidebar.subheader("📌 Phase 1: LLM Configuration")
provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    ["Ollama", "Gemini"],
    key="llm_provider"
)

llm = initialize_llm()
if llm is None:
    st.stop()

try:
    if provider == "Ollama":
        available_models = llm.list_ollama_models()
        chat_models = [m for m in available_models if "embed" not in m.lower()]
        llm_model = st.sidebar.selectbox(
            "Choose Ollama Model (Chat)",
            chat_models if chat_models else ["gemma2:2b"],
            key="ollama_llm_model"
        )
    else:
        available_models = llm.list_gemini_models()
        llm_model = st.sidebar.selectbox(
            "Choose Gemini Model",
            available_models if available_models else ["gemini-2.5-flash"],
            key="gemini_llm_model"
        )
except LLMException as e:
    st.sidebar.error(f"Error loading LLM models: {str(e)}")
    llm_model = Config.DEFAULT_OLLAMA_MODEL if provider == "Ollama" else "gemini-2.5-flash"

system_prompt = st.sidebar.text_area(
    "System Prompt",
    "You are a helpful AI assistant. Use the provided context to answer questions accurately. Cite your sources.",
    key="system_prompt",
    height=80
)

# === PHASE 1: Embeddings Configuration ===
st.sidebar.subheader("📌 Phase 1: Embeddings Configuration")
embeddings_mode = st.sidebar.selectbox(
    "Choose Embeddings Mode",
    ["ollama", "sentence-transformer"],
    key="embeddings_mode"
)

if embeddings_mode == "ollama":
    ollama_embedding_models = ["mxbai-embed-large", "nomic-embed-text", "all-minilm"]
    try:
        all_models = llm.list_ollama_models()
        available_embedding_models = [m for m in all_models if "embed" in m.lower()]
        embedding_models = list(set(ollama_embedding_models + available_embedding_models))
        embedding_models.sort()
    except:
        embedding_models = ollama_embedding_models
    
    embeddings_model = st.sidebar.selectbox(
        "Ollama Embedding Model",
        embedding_models,
        index=embedding_models.index("mxbai-embed-large") if "mxbai-embed-large" in embedding_models else 0,
        key="ollama_embeddings_model"
    )
else:
    embeddings_model = st.sidebar.selectbox(
        "Sentence-Transformer Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
        ],
        key="sentence_transformer_model"
    )

# === PHASE 2: Chunking Configuration ===
st.sidebar.subheader("📌 Phase 2: Chunking Configuration")
chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    ["sliding-window", "sentence-aware", "paragraph"],
    help="sliding-window: Fixed token chunks\nsentence-aware: Respects sentence boundaries\nparagraph: Respects paragraph boundaries"
)

if chunking_strategy == "sliding-window":
    chunk_size = st.sidebar.slider("Chunk Size (tokens)", 100, 1000, 512, 50)
    chunk_overlap = st.sidebar.slider("Chunk Overlap (tokens)", 0, 200, 50, 10)
else:
    chunk_size = st.sidebar.slider("Target Chunk Size (tokens)", 100, 1000, 512, 50)
    chunk_overlap = 0

# === PHASE 3: RAG Configuration ===
st.sidebar.subheader("📌 Phase 3: RAG Configuration")
retrieval_mode = st.sidebar.selectbox(
    "Retrieval Policy",
    ["strict", "relaxed", "hybrid"],
    help="strict: Top-k results\nrelaxed: Similarity threshold\nhybrid: Both methods combined"
)

top_k = st.sidebar.slider("Top-K Results", 1, 10, 3)
similarity_threshold = st.sidebar.slider("Similarity Threshold (for relaxed/hybrid)", 0.0, 1.0, 0.6, 0.05)
max_context_tokens = st.sidebar.slider("Max Context Tokens", 100, 2000, 500, 50)

# === DEBUG & RESET ===
st.sidebar.divider()
st.sidebar.subheader("🔧 Debug & Reset")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("Reset Indexer State"):
    try:
        state_file = Path("data/ingestion_state.json")
        if state_file.exists():
            state_file.unlink()
            st.sidebar.success("Indexer state reset!")
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to reset: {str(e)}")

# Display current settings
with st.sidebar.expander("📋 Current Settings", expanded=False):
    st.write("**LLM:**")
    st.write(f"  • Provider: {provider}")
    st.write(f"  • Model: {llm_model}")
    st.write("\n**Embeddings:**")
    st.write(f"  • Mode: {embeddings_mode}")
    st.write(f"  • Model: {embeddings_model}")
    st.write("\n**Chunking:**")
    st.write(f"  • Strategy: {chunking_strategy}")
    st.write(f"  • Chunk Size: {chunk_size}")
    st.write(f"  • Overlap: {chunk_overlap}")
    st.write("\n**RAG:**")
    st.write(f"  • Policy: {retrieval_mode}")
    st.write(f"  • Top-K: {top_k}")
    st.write(f"  • Threshold: {similarity_threshold}")
    st.write(f"  • Max Context: {max_context_tokens}")


# ========== MAIN CONTENT: TABS ==========
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload & Index (Phase 2)",
    "🔍 RAG Retrieval (Phase 3)",
    "💬 Chat with RAG (Phase 1-3)",
    "📊 Query Logs (Phase 3)"
])

# ========== TAB 1: DOCUMENT UPLOAD & INDEXING (PHASE 2) ==========
with tab1:
    st.subheader("Document Upload & Indexing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Step 1: Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX, XLSX, CSV, MD)",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "xlsx", "csv", "md"]
        )
        
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) selected**")
            for f in uploaded_files:
                st.write(f"  • {f.name} ({f.size / 1024:.1f} KB)")
    
    with col2:
        st.markdown("### Step 2: Initialize & Index")
        
        if st.button("🚀 Initialize Components & Index", use_container_width=True):
            with st.spinner("Initializing RAG Engine..."):
                rag_engine = initialize_rag_engine(
                    embedding_mode=embeddings_mode,
                    embedding_model=embeddings_model
                )
                if rag_engine is None:
                    st.stop()
                st.session_state.rag_engine = rag_engine
                st.success("✅ RAG Engine initialized with internal components!")

            if not uploaded_files:
                st.warning("No documents to index.")
                st.stop()

            with st.spinner("Processing and indexing documents..."):
                indexed_count = 0
                failed_count = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                for idx, uploaded_file in enumerate(uploaded_files):
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path is None:
                        failed_count += 1
                        continue
                    status_text.text(f"Processing: {uploaded_file.name}")
                    try:
                        # Use rag_engine.index_file (not direct indexer)
                        progress_updates = rag_engine.index_file(
                            file_path=str(file_path),
                            force=False,
                            max_tokens=chunk_size,
                            overlap_tokens=chunk_overlap
                        )
                        doc_ids = None
                        for update in progress_updates:
                            status_text.text(f"{update.get('message', '')}")
                            if update.get("status") == "skip":
                                st.info(f"⏭️ {uploaded_file.name} already indexed")
                                indexed_count += 1
                                break
                            if "doc_ids" in update:
                                doc_ids = update["doc_ids"]
                        if doc_ids is not None or update.get("status") != "skip":
                            indexed_count += 1
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                        failed_count += 1
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                status_text.text(f"✅ Indexed {indexed_count}/{len(uploaded_files)} files")
                if indexed_count > 0:
                    st.success(f"📊 Successfully indexed {indexed_count} document(s)")
                if failed_count > 0:
                    st.warning(f"⚠️ Failed to index {failed_count} document(s)")
            # Always refresh indexed docs from backend
            st.session_state.indexed_documents = refresh_indexed_documents(st.session_state.rag_engine)

    # Display indexed documents (from backend state)
    st.markdown("### Indexed Documents")
    st.session_state.indexed_documents = refresh_indexed_documents(st.session_state.rag_engine) if st.session_state.rag_engine else []
    if st.session_state.indexed_documents:
        for doc in st.session_state.indexed_documents:
            st.write(f"✅ **{doc['name']}** ({doc['documents']} docs)")
    else:
        st.info("No documents indexed yet.")
    
    # Display Ingestion State
    st.markdown("### Ingestion State (Idempotency Check)")
    try:
        state_file = Path("data/ingestion_state.json")
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            st.write(f"**Total files processed:** {len(state)}")
            with st.expander("View processed files"):
                for filename, file_info in state.items():
                    st.write(f"  • {filename}: {len(file_info.get('doc_ids', []))} docs, Hash: {file_info.get('file_hash', 'N/A')[:8]}...")
        else:
            st.info("No ingestion state yet.")
    except Exception as e:
        st.error(f"Failed to read ingestion state: {str(e)}")


# ========== TAB 2: RAG RETRIEVAL (PHASE 3) ==========
with tab2:
    st.subheader("RAG Retrieval Testing (Phase 3)")
    
    if st.session_state.rag_engine is None:
        st.warning("⚠️ Please initialize components in Tab 1 first.")
    else:
        rag = st.session_state.rag_engine
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query = st.text_input("Enter a query to test retrieval:", placeholder="e.g., What is machine learning?")
        
        with col2:
            retrieve_button = st.button("🔍 Retrieve", use_container_width=True)
        
        if retrieve_button and query:
            with st.spinner("Retrieving documents..."):
                try:
                    policy = RetrievalPolicy(
                        policy_type=retrieval_mode,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold
                    )
                    results = rag.retrieve(
                        query=query,
                        top_k=top_k,
                        policy=policy
                    )
                    st.markdown("### Retrieval Results")
                    if results:
                        st.success(f"✅ Retrieved {len(results)} results")
                        for idx, (text, metadata) in enumerate(results, 1):
                            with st.expander(f"📄 Result {idx}: {metadata.get('source_file', 'Unknown')}"):
                                st.write(f"**Source:** {metadata.get('source_file', 'Unknown')}")
                                st.write(f"**Page:** {metadata.get('page_number', 'N/A')}")
                                st.write(f"**Text Preview:**")
                                st.text(text[:500] + "..." if len(text) > 500 else text)
                    else:
                        st.info("No results found. Try adjusting the threshold or query.")
                except Exception as e:
                    st.error(f"Retrieval failed: {str(e)}")
        
        # Display retrieval statistics
        st.markdown("### Retrieval Policy Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mode", retrieval_mode.upper())
        with col2:
            st.metric("Top-K", top_k)
        with col3:
            st.metric("Threshold", f"{similarity_threshold:.2f}")


# ========== TAB 3: CHAT WITH RAG (PHASE 1-3) ==========
with tab3:
    st.subheader("Chat with RAG Pipeline")
    
    if st.session_state.rag_engine is None:
        st.warning("⚠️ Please initialize components in Tab 1 first.")
    else:
        rag = st.session_state.rag_engine
        
        # Enable/disable RAG
        use_rag = st.checkbox("Use RAG context", value=True, help="When enabled, uses retrieved documents to inform responses")
        
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("retrieved_docs"):
                    with st.expander(f"📚 Sources ({len(msg['retrieved_docs'])} docs)"):
                        for doc in msg["retrieved_docs"]:
                            st.write(f"• {doc['source']} (score: {doc['score']:.3f})")

        # Chat input
        prompt = st.chat_input("Ask a question...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.spinner("Processing your question..."):
                retrieved_docs = []
                context_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
                results = []
                if use_rag:
                    try:
                        policy = RetrievalPolicy(
                            policy_type=retrieval_mode,
                            top_k=top_k,
                            similarity_threshold=similarity_threshold
                        )
                        results = rag.retrieve(query=prompt, policy=policy)
                        retrieved_docs = [
                            {
                                "source": meta.get("source_file", "Unknown"),
                                "score": meta.get("similarity_score", 0),
                                "text": text[:200] + "..." if len(text) > 200 else text
                            }
                            for text, meta in results
                        ]
                        if results:
                            context_prompt, _ = rag.build_prompt_with_citations(
                                contexts=results,
                                question=prompt,
                                system_prompt=system_prompt,
                                max_context_tokens=max_context_tokens
                            )
                    except Exception as e:
                        st.error(f"RAG retrieval failed: {str(e)}")
                # Get LLM response
                try:
                    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                    if provider == "Ollama":
                        reply = llm.chat_ollama(llm_model, messages)
                    else:
                        if not Config.GEMINI_API_KEY:
                            reply = "Error: GEMINI_API_KEY not set."
                        else:
                            reply = llm.chat_gemini(llm_model, messages)
                    # Log query using rag.log_query
                    try:
                        rag.log_query(
                            query=prompt,
                            results=results,
                            answer=reply
                        )
                    except Exception as e:
                        st.warning(f"Failed to log query: {str(e)}")
                except LLMException as e:
                    reply = f"Error: Failed to get response from {provider}: {str(e)}"
                except Exception as e:
                    reply = f"Unexpected error: {str(e)}"
            msg_data = {"role": "assistant", "content": reply}
            if retrieved_docs:
                msg_data["retrieved_docs"] = retrieved_docs
            st.session_state.messages.append(msg_data)
            with st.chat_message("assistant"):
                st.write(reply)
                if retrieved_docs:
                    with st.expander(f"📚 Sources ({len(retrieved_docs)} docs)"):
                        for doc in retrieved_docs:
                            st.write(f"**{doc['source']}** (score: {doc['score']:.3f})")
                            st.caption(doc['text'])


# ========== TAB 4: QUERY LOGS (PHASE 3) ==========
with tab4:
    st.subheader("Query Logs & Analytics (Phase 3)")
    
    if st.session_state.rag_engine is None:
        st.warning("⚠️ Please initialize components in Tab 1 first.")
    else:
        rag = st.session_state.rag_engine
        query_logger = rag.query_logger
        
        col1, col2, col3 = st.columns(3)
        
        # Get stats
        try:
            stats = query_logger.get_stats()
            with col1:
                st.metric("Total Queries", stats.get("total_queries", 0))
            with col2:
                st.metric("Avg Results/Query", f"{stats.get('avg_results_per_query', 0):.2f}")
            with col3:
                st.metric("Top Sources Count", len(stats.get("top_sources", [])))
        except Exception as e:
            st.error(f"Failed to fetch stats: {str(e)}")

        st.markdown("### Recent Queries")
        try:
            logs = query_logger.read_logs(limit=20)
            if logs:
                for entry in logs:
                    with st.expander(f"🔍 {entry.get('query', 'N/A')[:60]} | {entry.get('timestamp', 'N/A')}"):
                        st.json(entry)
            else:
                st.info("No queries logged yet.")
        except Exception as e:
            st.error(f"Failed to read logs: {str(e)}")

        # Search logs
        st.markdown("### Search Logs")
        search_term = st.text_input("Search in queries:")
        if search_term:
            try:
                results = query_logger.search_logs(search_term)
                if results:
                    st.write(f"Found {len(results)} matching queries:")
                    for entry in results:
                        st.write(f"  • {entry.get('query')} ({entry.get('timestamp')})")
                else:
                    st.info("No matching queries found.")
            except Exception as e:
                st.error(f"Search failed: {str(e)}")


# ========== FOOTER ==========
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Agent_ng: Phases 0-3 Complete Pipeline")
with col2:
    st.caption(f"💬 Chat messages: {len(st.session_state.messages)}")
with col3:
    st.caption(f"📚 Indexed docs: {sum(d['documents'] for d in st.session_state.indexed_documents)}")
