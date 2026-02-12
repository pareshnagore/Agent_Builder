"""
Debug script to test basic connectivity and configuration.
"""

import sys
from core.config import Config
from core.llm import LLMClient, LLMException

print("=" * 60)
print("Agent_ng Configuration & Connectivity Test")
print("=" * 60)

# 1. Check config
print("\n1. Configuration Check:")
print(f"   OLLAMA_HOST: {Config.OLLAMA_HOST}")
print(f"   GEMINI_API_KEY: {'SET' if Config.GEMINI_API_KEY else 'NOT SET'}")
print(f"   EMBEDDINGS_MODE: {Config.EMBEDDINGS_MODE}")
print(f"   EMBEDDINGS_MODEL: {Config.EMBEDDINGS_MODEL}")

# 2. Test Ollama connectivity
print("\n2. Testing Ollama Connectivity:")
try:
    llm = LLMClient()
    print("   ✓ LLMClient initialized")
    
    try:
        models = llm.list_ollama_models()
        print(f"   ✓ Ollama is running. Available models:")
        for model in models:
            print(f"      - {model}")
    except LLMException as e:
        print(f"   ✗ Failed to get Ollama models: {e}")
        print("   → Make sure Ollama is running: ollama serve")
        
except LLMException as e:
    print(f"   ✗ Failed to initialize LLMClient: {e}")
    sys.exit(1)

# 3. Test Gemini (if API key is set)
print("\n3. Testing Gemini:")
if Config.GEMINI_API_KEY:
    try:
        models = llm.list_gemini_models()
        print(f"   ✓ Gemini is configured. Available models:")
        for model in models:
            print(f"      - {model}")
    except LLMException as e:
        print(f"   ✗ Gemini error: {e}")
else:
    print("   ℹ GEMINI_API_KEY not set (optional)")

# 4. Test simple Ollama chat (if models available)
print("\n4. Testing Ollama Chat (if available):")
try:
    models = llm.list_ollama_models()
    if models:
        model = models[0]
        print(f"   Testing model: {model}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello' only."}
        ]
        response = llm.chat_ollama(model, messages)
        print(f"   ✓ Ollama chat works!")
        print(f"   Response: {response[:100]}...")
    else:
        print("   ℹ No Ollama models available")
except LLMException as e:
    print(f"   ✗ Ollama chat error: {e}")

print("\n" + "=" * 60)
print("Debug test complete!")
print("=" * 60)
