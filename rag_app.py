"""
Agent_ng RAG Assistant
Complete application with multi-provider LLM support and document RAG functionality.
"""

import streamlit as st
import os
from core.llm import LLMClient, LLMException
from core.rag import RAGEngine
from core.config import Config
from core.loader import DocumentLoader
from core.embeddings import EmbeddingModel

st.set_page_config(page_title="Agent_ng RAG Assistant", layout="wide")
st.title("Agent_ng: RAG Assistant with Multi-Provider LLM")

# ========== INITIALIZE LLM CLIENT ==========
@st.cache_resource
def initialize_llm():
    """Initialize and return LLM client."""
    try:
        return LLMClient()
    except LLMException as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

llm = initialize_llm()
if llm is None:
    st.stop()

# ========== SIDEBAR SETTINGS ==========
st.sidebar.header("Settings")

# ========== LLM CONFIGURATION ==========
st.sidebar.subheader("LLM Configuration")

# LLM Provider Selection
provider_options = ["Gemini"]
if Config.OLLAMA_CLOUD_ENABLED:
    provider_options.append("Ollama Cloud")
provider_options.append("Ollama")

provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    provider_options,
    key="llm_provider"
)

# LLM Model Selection
try:
    if provider == "Ollama":
        available_models = llm.list_ollama_models()
        chat_models = [m for m in available_models if "embed" not in m.lower()]
        llm_model = st.sidebar.selectbox(
            "Choose Ollama Model (Chat)",
            chat_models if chat_models else ["gemma2:2b"],
            key="ollama_llm_model"
        )
    elif provider == "Ollama Cloud":
        available_models = llm.list_models("ollama-cloud")
        chat_models = [m for m in available_models if "embed" not in m.lower()]
        llm_model = st.sidebar.selectbox(
            "Choose Ollama Cloud Model (Chat)",
            chat_models if chat_models else ["mistral"],
            key="ollama_cloud_llm_model"
        )
    else:  # Gemini
        available_models = llm.list_gemini_models()
        llm_model = st.sidebar.selectbox(
            "Choose Gemini Model",
            available_models if available_models else ["gemini-2.5-flash"],
            key="gemini_llm_model"
        )
except LLMException as e:
    st.sidebar.error(f"Error loading LLM models: {str(e)}")
    llm_model = Config.DEFAULT_OLLAMA_MODEL if provider in ["Ollama", "Ollama Cloud"] else "gemini-2.5-flash"

# System Prompt
system_prompt = st.sidebar.text_area(
    "System Prompt",
    Config.DEFAULT_SYSTEM_PROMPT,
    key="system_prompt"
)

# Context Length (for Ollama models)
context_length = None
if provider in ["Ollama", "Ollama Cloud"]:
    try:
        # Get recommended context length for the selected model
        get_context_fn = (
            llm.get_ollama_context_length if provider == "Ollama"
            else llm.get_ollama_cloud_context_length
        )
        default_context = get_context_fn(llm_model)
        
        context_length = st.sidebar.slider(
            "Context Length (tokens)",
            min_value=512,
            max_value=32768,
            value=default_context,
            step=512,
            help=f"Maximum tokens for model context. Model default: {default_context}",
            key="context_length"
        )
    except LLMException as e:
        st.sidebar.warning(f"Could not determine model context length: {str(e)}")
        context_length = 8192  # Fallback default
    except Exception as e:
        st.sidebar.warning(f"Context length setup error: {str(e)}")
        context_length = 8192

# Enable/Disable Streaming
enable_streaming = st.sidebar.checkbox(
    "Enable Streaming Response",
    value=True,
    key="enable_streaming"
)

# ========== EMBEDDINGS CONFIGURATION ==========
st.sidebar.subheader("Embeddings Configuration")

# Initialize EmbeddingModel for compatibility with RAGEngine
embedder = EmbeddingModel()

embed_model = st.sidebar.selectbox(
    "Embedding Model",
    embedder.list_models(),
    key="embedding_model"
)

# Initialize RAG Engine with selected embedding model
rag = RAGEngine(embed_model=embed_model)

# ========== DOCUMENT MANAGEMENT ==========
st.sidebar.subheader("Document Management")

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    accept_multiple_files=True,
    type=["pdf", "txt", "docx", "md"]
)

if st.sidebar.button("Index Documents") and uploaded_files:
    os.makedirs("data/uploads", exist_ok=True)
    for file in uploaded_files:
        save_path = os.path.join("data/uploads", file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        
        text = DocumentLoader.load_file(save_path)
        if not text.strip():
            st.sidebar.error(f"No readable text: {file.name}")
        else:
            result = rag.index_documents([text], source=file.name)
            if result:
                st.sidebar.success(f"Indexed: {file.name}")
            else:
                st.sidebar.info(f"{file.name} already indexed")

# Clear Chat Button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Display current settings
with st.sidebar.expander("Current Settings"):
    st.write(f"**LLM Provider:** {provider}")
    st.write(f"**LLM Model:** {llm_model}")
    st.write(f"**Embedding Model:** {embed_model}")
    st.write(f"**System Prompt:** {system_prompt[:50]}...")

# ========== CHAT INTERFACE ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
prompt = st.chat_input("Ask from your documents...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Retrieve context from RAG
    context = rag.retrieve(prompt)

    # Build final prompt with context
    system_with_context = f"""{system_prompt}
    You have access to the following relevant documents:

    {context}

    Use this information to answer questions about the conversation if relevant, else answer based on your general knowledge."""

    # Prepare messages for LLM
    messages = [
        {"role": "system", "content": system_with_context}
    ]
    
    for msg in st.session_state.messages:  
        messages.append(msg)

    # Get response from selected LLM provider
    reply = ""  # Initialize reply to avoid unbound variable errors
    try:
        with st.spinner(f"Getting response from {provider}..."):
            # Prepare additional options for Ollama models
            llm_options = {}
            if context_length is not None and provider in ["Ollama", "Ollama Cloud"]:
                llm_options["num_ctx"] = context_length
            
            if enable_streaming:
                # Streaming response
                response_placeholder = st.empty()
                full_response = ""
                
                if provider == "Ollama":
                    stream = llm.chat_ollama_stream(llm_model, messages, **llm_options)
                elif provider == "Ollama Cloud":
                    stream = llm.chat_ollama_cloud_stream(llm_model, messages, **llm_options)
                else:  # Gemini
                    if not Config.GEMINI_API_KEY:
                        reply = "Error: GEMINI_API_KEY not set. Please configure it in .env file."
                        stream = None
                    else:
                        stream = llm.chat_gemini_stream(llm_model, messages, **llm_options)
                
                if stream:
                    with st.chat_message("assistant"):
                        for chunk in stream:
                            full_response += chunk
                            response_placeholder.write(full_response)
                    reply = full_response
            else:
                # Non-streaming response (original behavior)
                if provider == "Ollama":
                    reply = llm.chat_ollama(llm_model, messages, **llm_options)
                elif provider == "Ollama Cloud":
                    reply = llm.chat_ollama_cloud(llm_model, messages, **llm_options)
                else:  # Gemini
                    if not Config.GEMINI_API_KEY:
                        reply = "Error: GEMINI_API_KEY not set. Please configure it in .env file."
                    else:
                        reply = llm.chat_gemini(llm_model, messages, **llm_options)
                
                with st.chat_message("assistant"):
                    st.write(reply)
                    
    except LLMException as e:
        reply = f"Error: Failed to get response from {provider}: {str(e)}"
        st.error(reply)
    except Exception as e:
        reply = f"Unexpected error: {str(e)}"
        st.error(reply)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": reply})
    
    # Display assistant message
    # with st.chat_message("assistant"):
    #     st.write(reply)

# ========== FOOTER ==========
st.divider()
st.caption("Agent_ng - RAG Assistant with Multi-Provider LLM Support")
st.caption(f"Session Messages: {len(st.session_state.messages)}")