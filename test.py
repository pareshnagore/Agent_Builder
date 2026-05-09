"""
Test application for Agent_ng using the new architecture.
Basic chat application supporting both Ollama and Gemini LLM providers.
"""

import streamlit as st
from core.config import Config
from core.llm import LLMClient, LLMException
# from core.embeddings import EmbeddingsAdapter, EmbeddingsException
# from core.vector_db import VectorDB, VectorDBException

@st.cache_resource
def initialize_llm():
    """Initialize and return LLM client."""
    try:
        return LLMClient()
    except LLMException as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None


# def initialize_embeddings(mode: str):
#     """Initialize and return embeddings adapter."""
#     try:
#         return EmbeddingsAdapter(mode=mode)
#     except EmbeddingsException as e:
#         st.error(f"Failed to initialize embeddings: {str(e)}")
#         return None


# def initialize_vector_db(embedding_adapter: EmbeddingsAdapter):
#     """Initialize and return vector database."""
#     try:
#         return VectorDB(embedding_adapter=embedding_adapter)
#     except VectorDBException as e:
#         st.error(f"Failed to initialize vector DB: {str(e)}")
#         return None


# ========== STREAMLIT UI ==========
st.title("Agent_ng: Testing Chat with New Architecture")

# ========== SIDEBAR SETTINGS ==========
st.sidebar.header("Settings")

# LLM Provider Selection
st.sidebar.subheader("LLM Configuration")
provider_options = ["Ollama"]
if Config.OLLAMA_CLOUD_ENABLED:
    provider_options.append("Ollama Cloud")
provider_options.append("Gemini")

provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    provider_options,
    key="llm_provider"
)

# Initialize LLM client
llm = initialize_llm()
if llm is None:
    st.stop()

# LLM Model Selection
try:
    if provider == "Ollama":
        available_models = llm.list_ollama_models()
        # Filter out embedding models (exclude models with "embed" in the name)
        chat_models = [m for m in available_models if "embed" not in m.lower()]
        llm_model = st.sidebar.selectbox(
            "Choose Ollama Model (Chat)",
            chat_models if chat_models else ["gemma2:2b"],
            key="ollama_llm_model"
        )
    elif provider == "Ollama Cloud":
        available_models = llm.list_models("ollama-cloud")
        # Filter out embedding models
        chat_models = [m for m in available_models if "embed" not in m.lower()]
        llm_model = st.sidebar.selectbox(
            "Choose Ollama Cloud Model (Chat)",
            chat_models if chat_models else ["mistral"],
            key="ollama_cloud_llm_model"
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

# System Prompt
system_prompt = st.sidebar.text_area(
    "System Prompt",
    Config.DEFAULT_SYSTEM_PROMPT,
    key="system_prompt"
)

# ========== EMBEDDINGS SETTINGS ==========
st.sidebar.subheader("Embeddings Configuration")

# Embeddings Mode Selection
embeddings_mode = st.sidebar.selectbox(
    "Choose Embeddings Mode",
    ["ollama", "sentence-transformer"],
    key="embeddings_mode"
)

# Embeddings Model Selection
if embeddings_mode == "ollama":
    # Common Ollama embedding models
    ollama_embedding_models = [
        "mxbai-embed-large",
        "nomic-embed-text",
        "all-minilm",
        "neural-rerank",
    ]
    try:
        # Try to fetch available Ollama models and filter for embeddings
        all_models = llm.list_ollama_models()
        available_embedding_models = [m for m in all_models if "embed" in m.lower()]
        # Combine with common models, avoiding duplicates
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
    # Sentence-transformer models
    embeddings_model = st.sidebar.selectbox(
        "Sentence-Transformer Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
        ],
        key="sentence_transformer_model"
    )

# Clear Chat Button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Display current settings
with st.sidebar.expander("Current Settings"):
    st.write(f"**LLM Provider:** {provider}")
    st.write(f"**LLM Model:** {llm_model}")
    st.write(f"**Embeddings Mode:** {embeddings_mode}")
    st.write(f"**Embeddings Model:** {embeddings_model}")


# ========== CHAT HISTORY ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ========== USER INPUT ==========
prompt = st.chat_input("Ask something...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Prepare messages for LLM
    messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    # Get response from LLM
    try:
        with st.spinner(f"Getting response from {provider}..."):
            if provider == "Ollama" :
                reply = llm.chat_ollama(llm_model, messages)
            elif provider == "Ollama Cloud":
                reply = llm.chat_ollama_cloud(llm_model, messages)
            else:
                if not Config.GEMINI_API_KEY:
                    reply = "Error: GEMINI_API_KEY not set. Please configure it in .env file."
                else:
                    reply = llm.chat_gemini(llm_model, messages)
    except LLMException as e:
        reply = f"Error: Failed to get response from {provider}: {str(e)}"
    except Exception as e:
        reply = f"Unexpected error: {str(e)}"

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": reply})
    
    # Display assistant message
    with st.chat_message("assistant"):
        st.write(reply)

# ========== FOOTER ==========
st.divider()
st.caption("Agent_ng - Testing Application with New Architecture")
st.caption(f"Session Messages: {len(st.session_state.messages)}")
