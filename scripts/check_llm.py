"""Diagnostic: test OpenRouter plain message, then tool-calling. Never prints the API key."""
from __future__ import annotations
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from app.config import settings

TOOL_DEF = [{
    "type": "function",
    "function": {
        "name": "extract_info",
        "description": "Extract information from the message.",
        "parameters": {"type": "object", "properties": {"intent": {"type": "string"}}, "required": []},
    },
}]

def _headers():
    h = {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}
    if "openrouter" in settings.llm_base_url:
        h["HTTP-Referer"] = "https://wakeel.ai"
        h["X-Title"] = "Wakeel AI"
    return h

def _safe(d: dict) -> dict:
    """Recursively redact the API key from any dict."""
    key = settings.llm_api_key
    return json.loads(json.dumps(d).replace(key, "***REDACTED***"))

async def test_plain():
    print(f"\n{'='*60}")
    print(f"BASE_URL : {settings.llm_base_url}")
    print(f"MODEL    : {settings.llm_model}")
    print(f"API_KEY  : {'set (' + str(len(settings.llm_api_key)) + ' chars)' if settings.llm_api_key and settings.llm_api_key != 'sk-placeholder' else 'NOT SET / placeholder'}")
    print(f"{'='*60}")

    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 20,
    }
    print(f"\n[1/2] Plain message → {url}")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(url, headers=_headers(), json=payload)
            print(f"  Status : {r.status_code}")
            try:
                body = r.json()
                if r.is_success:
                    print(f"  Reply  : {body['choices'][0]['message']['content']!r}")
                else:
                    print(f"  Error  : {json.dumps(_safe(body), indent=2)}")
            except Exception:
                print(f"  Raw    : {r.text[:500]}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")

async def test_tools():
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": "I want to buy an apartment."}],
        "tools": TOOL_DEF,
        "tool_choice": "auto",
        "max_tokens": 100,
    }
    print(f"\n[2/2] Tool-calling → {url}")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(url, headers=_headers(), json=payload)
            print(f"  Status : {r.status_code}")
            try:
                body = r.json()
                if r.is_success:
                    msg = body["choices"][0]["message"]
                    if msg.get("tool_calls"):
                        print(f"  ✅ Tool calls supported! Got: {[tc['function']['name'] for tc in msg['tool_calls']]}")
                    else:
                        print(f"  ⚠️  No tool calls returned (model may not support tools). Content: {msg.get('content','')!r}")
                else:
                    print(f"  ❌ Error: {json.dumps(_safe(body), indent=2)}")
            except Exception:
                print(f"  Raw: {r.text[:500]}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")

asyncio.run(test_plain())
asyncio.run(test_tools())
print()
