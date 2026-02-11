import streamlit as st
import os
from core.llm import LLMClient
from core.rag import RAGEngine
from core.config import Config
from core.loader import DocumentLoader
from core.embeddings import EmbeddingModel

st.title("Agent_ng : RAG Assistant")

llm = LLMClient()

st.sidebar.header("Settings")

provider = st.sidebar.selectbox(
    "Provider",
    ["Ollama", "Gemini"]
)

if provider == "Ollama":
    model = st.sidebar.selectbox(
        "Model",
        Config.OLLAMA_MODELS
    )
else:
    model = st.sidebar.selectbox(
        "Model",
        llm.list_gemini_models()
    )

embedder = EmbeddingModel()

embed_model = st.sidebar.selectbox(
    "Embedding Model",
    embedder.list_models()
)
rag = RAGEngine(embed_model=embed_model)

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    accept_multiple_files=True
)

# if st.sidebar.button("Index Documents"):
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
            # rag.index_documents([text], source=file.name)
            # st.sidebar.success(f"Indexed: {file.name}")
            result = rag.index_documents([text], source=file.name)
            if result:
                st.sidebar.success(f"Indexed: {file.name}")
            else:
                st.sidebar.info(f"{file.name} already indexed")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Ask from your documents...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    st.chat_message("user").write(prompt)

    context = rag.retrieve(prompt)

    final_prompt = f"""
    Use the following context to answer:
    {context}
    Question: {prompt}
    """

    messages = [
        {"role": "system", "content": Config.DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": final_prompt}
    ]

    if provider == "Ollama":
        reply = llm.chat_ollama(model, messages)
    else:
        reply = llm.chat_gemini(model, messages)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
    st.chat_message("assistant").write(reply)
