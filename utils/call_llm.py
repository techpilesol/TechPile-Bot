from google import genai
import os
import hashlib
import json

# In-memory cache to avoid repeated API calls for identical prompts
_cache = {}

def call_llm(prompt: str, use_cache: bool = True) -> str:
    # Check cache first
    if use_cache:
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in _cache:
            print("[LLM] Cache hit — skipping API call")
            return _cache[cache_key]

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY", ""),
    )

    # gemini-2.0-flash: no thinking mode, significantly faster than 2.5-flash
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config={
            "temperature": 0.1,       # Lower temp = more focused, slightly faster
            "max_output_tokens": 1024, # Limit output length to reduce generation time
        }
    )

    result = response.text

    # Store in cache
    if use_cache:
        _cache[cache_key] = result

    return result

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
