"""
Minimal test for Gemini API to debug issues.
"""

import os
from core.config import Config

print("=" * 60)
print("Testing Gemini API")
print("=" * 60)

# Check API key
if not Config.GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not set in .env")
    exit(1)

print("✓ GEMINI_API_KEY is set")

# Test basic API call
try:
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=Config.GEMINI_API_KEY)
    print("✓ Gemini client initialized")
    
    # Simple single-turn test
    print("\n1. Testing simple text generation...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'Hello' only."
    )
    print(f"✓ Response: {response.text}")
    
    # Multi-turn test with messages
    print("\n2. Testing multi-turn conversation...")
    messages = [
        {"role": "user", "parts": [{"text": "What is 2+2?"}]},
        {"role": "model", "parts": [{"text": "2+2 equals 4."}]},
        {"role": "user", "parts": [{"text": "What is 3+3?"}]},
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages
    )
    print(f"✓ Response: {response.text}")
    
    # With system instruction
    print("\n3. Testing with system instruction...")
    config = types.GenerateContentConfig(
        system_instruction="You are a helpful assistant. Keep responses brief."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Tell me about AI in one sentence.",
        config=config
    )
    print(f"✓ Response: {response.text}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

# from google import genai
# client = genai.Client()

# response = client.models.generate_content(
#     model='gemini-2.5-flash',
#     contents='Tell me a story in 30 words.'
# )
# print(response.text)

# print(response.model_dump_json(
#     exclude_none=True, indent=4))

# from google import genai
# client = genai.Client()

# for m in client.models.list():
#     print(m.name)
