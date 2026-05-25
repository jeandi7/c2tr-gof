# agent_llm/llm.py

import subprocess
import requests
import time
import threading
import psutil

OLLAMA_BASE_URL   = "https://ollama.iboo.ovh"
OLLAMA_MODEL_LOCAL  = "llama3.1-gof"  # custom Modelfile: num_ctx=8192 to fit 3400-token prompt
OLLAMA_MODEL_REMOTE = "devstral-small-2:24b"

# MLX — Apple Silicon (M-series Neural Engine + GPU)
MLX_MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

_mlx_cache: tuple | None = None  # (model, tokenizer) — loaded once per process


def _ensure_mlx() -> tuple:
    global _mlx_cache
    if _mlx_cache is None:
        from mlx_lm import load
        print(f"  Loading MLX model {MLX_MODEL_ID} (first call — cached for the session)...")
        _mlx_cache = load(MLX_MODEL_ID)
        print("  MLX model ready.")
    return _mlx_cache


def _call_mlx(prompt: str) -> tuple[str, float]:
    """Local inference via mlx-lm on Apple Silicon Neural Engine + GPU."""
    from mlx_lm import generate
    model, tokenizer = _ensure_mlx()

    # Apply the model's chat template so the instruct model behaves correctly.
    # The full benchmark prompt (SYSTEM + TASK + FORMAT) is passed as a user message.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = prompt

    wall_start = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=2000,
        verbose=False,
        temp=0.2,
        repetition_penalty=1.1,
        top_p=0.9,
    )
    elapsed = time.perf_counter() - wall_start

    print("MLX inference (M-series) = OK")
    print(f"MLX Model                = {MLX_MODEL_ID}")
    print(f"Elapsed wall time        = {elapsed:.2f} s")
    return response.strip(), elapsed


def _apply_ansi_corrections(text: str) -> str:
    """Clean ollama CLI stdout for JSON parsing.

    Handles three issues produced by ollama's streaming terminal output:
    1. ESC[ND + ESC[K (cursor-back + erase) — reconstruct corrected text
    2. Bare \\r characters — strip them
    3. Literal newlines inside JSON string values — replace with space
       (LLM wraps long lines mid-string, producing invalid control chars)
    """
    # Pass 1: apply ANSI cursor-back + erase, strip \r
    buf: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\r':
            i += 1
            continue
        if c == '\x1b' and i + 1 < len(text) and text[i + 1] == '[':
            j = i + 2
            while j < len(text) and text[j] not in 'ABCDEFGHIJKLMSTfhlHmnpqrsu':
                j += 1
            if j < len(text):
                cmd = text[j]
                param = text[i + 2:j]
                if cmd == 'D':  # Cursor Back N — pop last N chars
                    n = int(param) if param.isdigit() else 1
                    del buf[-n:]
                # ESC[K (Erase to End): buffer already truncated by ESC[D; skip
                i = j + 1
            else:
                buf.append(c)
                i += 1
        else:
            buf.append(c)
            i += 1

    # Pass 2: replace literal newlines inside JSON strings with a space
    result: list[str] = []
    in_string = False
    escaped = False
    for ch in buf:
        if escaped:
            result.append(ch)
            escaped = False
        elif ch == '\\' and in_string:
            result.append(ch)
            escaped = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif ch == '\n' and in_string:
            result.append(' ')
        else:
            result.append(ch)
    return ''.join(result)


def call_llm(prompt: str, local: bool = True, use_mlx: bool = False) -> tuple[str, float]:
    """
    Returns:
        raw_output (str)
        elapsed (float)  — wall_time × avg_cpu% for Ollama local,
                           wall_time for MLX or remote
    """

    # -------- MLX MODE (Apple Silicon — Neural Engine + GPU) --------
    if local and use_mlx:
        return _call_mlx(prompt)

    # -------- LOCAL MODE (Ollama subprocess) --------
    if local:
        cpu_samples = []
        stop_event = threading.Event()

        def _monitor():
            while not stop_event.is_set():
                cpu_samples.append(psutil.cpu_percent(interval=0.5))

        monitor_thread = threading.Thread(target=_monitor, daemon=True)
        wall_start = time.perf_counter()
        monitor_thread.start()

        try:
            result = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL_LOCAL],
                input=prompt,
                encoding="utf-8",
                capture_output=True,
                timeout=7200,
            )
        except Exception as e:
            print(f"LLM local error: {e}")
            stop_event.set()
            return "", 0.0
        finally:
            stop_event.set()
            monitor_thread.join(timeout=2)

        elapsed = time.perf_counter() - wall_start
        avg_cpu_pct = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 50.0
        cpu_seconds = elapsed * (avg_cpu_pct / 100.0)

        stdout_clean = _apply_ansi_corrections(result.stdout).strip()
        print("LLM Ollama (local)      = OK")
        print(f"LLM Model               = {OLLAMA_MODEL_LOCAL}")
        print(f"Elapsed wall time       = {elapsed:.2f} s")
        print(f"Avg CPU utilization     = {avg_cpu_pct:.1f} %")
        print(f"Return code             = {result.returncode}")
        print(f"stdout length           = {len(result.stdout)} chars  cleaned={len(stdout_clean)}")
        if result.stderr:
            print(f"stderr                  = {result.stderr[:400]!r}")
        if not stdout_clean:
            print("[WARN] stdout is empty — checking ollama state...")
        return stdout_clean, cpu_seconds

    # -------- REMOTE MODE (API) --------
    else:
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL_REMOTE,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 2000,
                        "temperature": 0.2,
                        "repeat_penalty": 1.1,
                        "top_p": 0.9,
                    },
                },
                timeout=1500,
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            cpu_seconds = time.perf_counter() - start

            print("LLM Ollama (remote)     = OK")
            print(f"LLM URL                 = {OLLAMA_BASE_URL}")
            print(f"LLM Model               = {OLLAMA_MODEL_REMOTE}")
            return result, cpu_seconds

        except Exception as e:
            print(f"LLM remote error: {e}")
            return "", 0.0
