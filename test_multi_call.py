"""
Test 3 consecutive ollama subprocess calls to reproduce empty response on iter 2/3.
Run: python test_multi_call.py
"""
from agent_llm.llm import call_llm

PROMPT = "Reply with a single JSON object: {\"hello\": \"world\"}"

for i in range(1, 4):
    print(f"\n{'='*50}  CALL {i}  {'='*50}")
    raw, cpu_seconds = call_llm(PROMPT, local=True)
    print(f"Result [{i}]: empty={not bool(raw)}  cpu_s={cpu_seconds:.3f}")
    if raw:
        print(f"Content (first 200): {raw[:200]}")
    else:
        print("!!! EMPTY RESPONSE !!!")
