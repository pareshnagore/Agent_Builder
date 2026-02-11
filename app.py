import streamlit as st

from core.llm import LLMClient
from core.config import Config

st.title("Agent_ng : AI Chat")

# Initialize LLM client
llm = LLMClient()

# -------- Sidebar --------
st.sidebar.header("Settings")

provider = st.sidebar.selectbox(
    "Choose Provider",
    ["Ollama", "Gemini"]
)

if provider == "Ollama":
    model = st.sidebar.selectbox(
        "Choose Model",
        Config.OLLAMA_MODELS
    )
else:
    model = st.sidebar.selectbox(
        "Choose Gemini Model",
        llm.list_gemini_models()
    )

system_prompt = st.sidebar.text_area(
    "System Prompt",
    Config.DEFAULT_SYSTEM_PROMPT
)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# -------- Chat History --------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------- User Input --------
prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    st.chat_message("user").write(prompt)

    messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    if provider == "Ollama":
        reply = llm.chat_ollama(model, messages)

    else:
        if not Config.GEMINI_API_KEY:
            reply = "Please set GEMINI_API_KEY environment variable"
        else:
            reply = llm.chat_gemini(model, messages)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

    st.chat_message("assistant").write(reply)

