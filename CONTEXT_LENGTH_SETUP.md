# Context Length Configuration for Ollama & Ollama Cloud

## Overview
Context length (number of tokens the model can process) has been added as a configurable parameter for both Ollama and Ollama Cloud models in the Agent_ng RAG application.

## Changes Made

### 1. **core/config.py**
Added context length configuration:
```python
DEFAULT_OLLAMA_CONTEXT_LENGTH: int = 8192  # Default context length
OLLAMA_MODEL_CONTEXT_LENGTHS: dict = {
    "llama3": 8192,
    "llama3.2": 8192,
    "gemma2:2b": 8192,
    "mistral": 8192,
    # ... more models
}
```

- `DEFAULT_OLLAMA_CONTEXT_LENGTH`: Default context window (8192 tokens)
- `OLLAMA_MODEL_CONTEXT_LENGTHS`: Model-specific context windows for accurate defaults

Can override via environment variable:
```bash
export DEFAULT_OLLAMA_CONTEXT_LENGTH=4096
```

### 2. **core/llm.py (OllamaProvider class)**
Added three new methods:

#### `get_model_info(model: str) -> dict`
Retrieves detailed information about a model from the Ollama API.

#### `get_context_length(model: str) -> int`
Returns the context length for a given model:
- Checks model-specific mappings first
- Falls back to base model name (e.g., "gemma2:2b" → "gemma2")
- Returns default if not found

#### LLMClient convenience methods:
- `get_ollama_context_length(model)` - Get context for Ollama model
- `get_ollama_cloud_context_length(model)` - Get context for Ollama Cloud model
- `get_ollama_model_info(model)` - Get detailed model info

### 3. **rag_app.py (Streamlit UI)**
Added context length slider control in the sidebar:

```python
# Shows only for Ollama and Ollama Cloud providers
context_length = st.sidebar.slider(
    "Context Length (tokens)",
    min_value=512,
    max_value=32768,
    value=default_context,  # Auto-loads model's default
    step=512
)
```

**Features:**
- Slider appears only for Ollama/Ollama Cloud providers
- Auto-loads model's default context length
- Range: 512 to 32,768 tokens
- Step size: 512 tokens
- Passes context length to Ollama API via `num_ctx` parameter

## How It Works

1. **User selects a model** → Auto-loads model's default context length
2. **User adjusts slider** → Changes context length value
3. **Messages sent to Ollama** → Includes `num_ctx` parameter with selected value

### Example API Call
```python
# Ollama API call includes context length
payload = {
    "model": "llama3.2",
    "messages": [...],
    "num_ctx": 8192  # Context length parameter
}
```

## Running the App

```bash
streamlit run rag_app.py
```

The context length control will appear in the sidebar below the system prompt for Ollama models.

## Model Context Defaults

Current configured models:
- **Llama 2**: 4,096 tokens
- **Llama 3**: 8,192 tokens  
- **Llama 3.2**: 8,192 tokens
- **Gemma/Gemma2**: 8,192 tokens
- **Mistral**: 8,192 tokens

To add more models, update `OLLAMA_MODEL_CONTEXT_LENGTHS` in `core/config.py`.

## Troubleshooting

1. **Context length slider not appearing**
   - Ensure you're using Ollama or Ollama Cloud provider
   - Check that `llm` client initialized successfully

2. **Model context info unavailable**
   - App will use default 8,192 tokens
   - Check Ollama API connectivity

3. **Out of memory errors**
   - Reduce context length value
   - Check available GPU/CPU memory

## Environment Variables

Optional configuration:
```bash
# Set default context length
export DEFAULT_OLLAMA_CONTEXT_LENGTH=4096

# Ollama configuration
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_CLOUD_ENABLED=false
export OLLAMA_CLOUD_HOST=https://api.example.com
export OLLAMA_API_KEY=your-key
```
