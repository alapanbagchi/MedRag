#!/usr/bin/env python3
"""Build ``kaggle/medrag_inference.ipynb`` from ``kaggle/inference_server.py``.

The notebook embeds the thin server code verbatim (Kaggle has no repo to
check out), so the single source of truth stays ``kaggle/inference_server.py``.
Run:  python3 scripts/build_kaggle_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "kaggle" / "inference_server.py"
OUT = ROOT / "kaggle" / "medrag_inference.ipynb"

INTRO_MD = """# MedRAG — Kaggle Inference Server (thin)

Kaggle hosts **only** the model. It is an **OpenAI-compatible chat-completions
endpoint**: it receives a system prompt + user prompt and returns the model
output. Nothing else.

Everything else — planning, retrieval, UMLS, evidence extraction, verification,
synthesis, orchestration — runs in the **local application** (PydanticAI + the
`app/` package in this repository).

```
LOCAL APP  ──(OpenAI-compatible API)──▶  KAGGLE
                                          ├─ receive request
                                          ├─ extract system prompt + user prompt
                                          ├─ run model
                                          └─ return response
```

Endpoints exposed:

- `GET  /`                    health
- `GET  /v1/models`           model list
- `POST /v1/chat/completions` chat completion (system + user -> output)

Config via environment / Kaggle secrets:

- `MEDGEMMA_MODEL`       .gguf basename (default `medgemma-27b-it-Q8_0.gguf`)
- `MEDGEMMA_MODEL_PATH`  explicit path to a .gguf
- `MEDGEMMA_MODEL_NAME`  reported model id (default `medgemma`)
- `N_CTX`                context size (default 32768)

The code below is the same module as `kaggle/inference_server.py` in the repo.
"""

INSTALL = '''# Install llama.cpp with CUDA (the only heavy dependency — no torch needed).
import os
os.environ.setdefault("CMAKE_ARGS", "-DGGML_CUDA=on")
os.environ.setdefault("FORCE_CMAKE", "1")

!pip install -q \\
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \\
  llama-cpp-python

print("llama-cpp-python installed")'''

DOWNLOAD = '''# Download the MedGemma GGUF once (cache in /kaggle/temp so re-runs reuse it).
from huggingface_hub import hf_hub_download
from pathlib import Path

MODEL_DIR = Path("/kaggle/temp/medgemma")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "unsloth/medgemma-27b-it-GGUF"
FILENAME = os.environ.get(
    "MEDGEMMA_GGUF",
    "medgemma-27b-it-IQ4_XS.gguf",   # ~14 GB; Q8_0 is ~28 GB
)

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=str(MODEL_DIR),
)
print("Model:", model_path)
print("Size:", round(Path(model_path).stat().st_size / 1024**3, 2), "GB")'''

SERVER_MD = """## The server (thin)

This cell defines the FastAPI app. It contains **no** retrieval, UMLS,
query-planning, decomposition, reranking or evidence logic — only the
OpenAI-compatible wire contract.

Structured output is supported through the OpenAI `response_format`
parameter (json_schema / json_object), which llama.cpp honours with its
grammar engine. The local PydanticAI agents use exactly this.
"""

LAUNCH = '''# Launch the server in a background thread and verify health.
import json
import os
import requests
import threading
import time

port = int(os.environ.get("KAGGLE_API_PORT", "8083"))
host = os.environ.get("KAGGLE_API_HOST", "0.0.0.0")

def _run():
    uvicorn.run(app, host=host, port=port, log_level="info", reload=False)

threading.Thread(target=_run, daemon=True, name="inference-server").start()

ok = False
for _ in range(60):
    time.sleep(1)
    try:
        r = requests.get(f"http://127.0.0.1:{port}/", timeout=2)
        if r.status_code == 200:
            ok = True
            print("✓ SERVER READY", json.dumps(r.json(), indent=2))
            break
    except Exception:
        pass

if not ok:
    raise RuntimeError("inference server failed to start")'''

TUNNEL = '''# Expose the server to the internet with a Cloudflare quick tunnel.
import os
import re
import subprocess

!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \\
    -O /tmp/cloudflared && chmod +x /tmp/cloudflared

subprocess.run(["pkill", "-f", "cloudflared"], capture_output=True)

proc = subprocess.Popen(
    ["/tmp/cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}", "--loglevel", "info"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)

public_url = None
for line in proc.stdout:
    line = line.rstrip()
    print("[cloudflared]", line)
    m = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", line)
    if m and public_url is None:
        public_url = m.group(0)
        print()
        print("=" * 70)
        print("KAGGLE INFERENCE SERVER READY")
        print("=" * 70)
        print()
        print("Health:          " + public_url + "/")
        print("Models:          " + public_url + "/v1/models")
        print("Chat:            " + public_url + "/v1/chat/completions")
        print()
        print("Local .env (copy to your machine):")
        print(f'  LLM_PROVIDER="kaggle"')
        print(f'  KAGGLE_BASE_URL="{public_url}/v1"')
        print(f'  KAGGLE_API_KEY="dummy"')
        print(f'  KAGGLE_MODEL="medgemma"')
        print()
        print("=" * 70)'''


def server_source() -> str:
    text = SERVER.read_text(encoding="utf-8")
    # Strip the __main__ tail (launched from the notebook cell instead).
    marker = 'if __name__ == "__main__":'
    if marker in text:
        text = text.split(marker, 1)[0]
    return text.rstrip()


def make_cell(cell_type: str, source: str, md: bool = False) -> dict:
    lines = source.splitlines()
    if source and not source.endswith("\n"):
        lines.append("")
    cell = {
        "cell_type": "markdown" if cell_type == "markdown" else "code",
        "metadata": {"trusted": True},
        "source": [line + "\n" for line in lines],
    }
    if cell_type != "markdown":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


notebook = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python",
            "version": "3.12.13",
            "mimetype": "text/x-python",
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
        "kaggle": {
            "accelerator": "GPU T4 x2",
            "dataSources": [],
            "dockerImageVersionId": 28755,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True,
        },
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": [
        make_cell("markdown", INTRO_MD, md=True),
        make_cell("code", INSTALL),
        make_cell("code", DOWNLOAD),
        make_cell("markdown", SERVER_MD, md=True),
        make_cell("code", server_source()),
        make_cell("code", LAUNCH),
        make_cell("code", TUNNEL),
    ],
}


def main() -> None:
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()