"""Clean LLM client abstraction for the MedRAG pipeline.

Supports any OpenAI-compatible API (OpenAI, Ollama, vLLM, LM Studio, etc.).
Configured entirely through environment variables — no hard-coded keys.

Environment variables:
    LLM_API_KEY       API key (required for OpenAI, optional for local servers)
    LLM_BASE_URL      Base URL (default: https://api.openai.com/v1)
    LLM_MODEL         Model name (default: gpt-4o-mini)
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completions API.

    Defaults to a local DeepSeek/Ollama instance at localhost:11434.
    Configure via env vars: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or _env("LLM_API_KEY")
        self.base_url = (base_url or _env("LLM_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or _env("LLM_MODEL", "deepseek")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str = "You are a biomedical research assistant.",
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> str:
        """Send a chat completion request. Returns the assistant's text."""
        from medrag.trace import get_trace
        trace = get_trace()

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        resp = self._session.post(url, json=payload, headers=headers, timeout=60)
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "").strip()

        trace.log(
            "llm_generate",
            params={
                "url": url,
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            result={
                "response": content,
                "usage": data.get("usage", {}),
                "finish_reason": data["choices"][0].get("finish_reason", ""),
            },
            duration_ms=latency_ms,
        )

        return content

    def generate_json(
        self,
        prompt: str,
        system: str = "You are a biomedical research assistant. Return only valid JSON.",
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> Any:
        """Generate and parse JSON from the LLM response."""
        text = self.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        return _parse_json(text)

    def is_available(self) -> bool:
        """Check if the LLM is configured (has a base URL)."""
        return bool(self.base_url)


def _parse_json(text: str) -> Any:
    """Robustly extract JSON from LLM output."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding any JSON object or array
    for pattern in [r"\{(?:[^{}]|\{[^{}]*\})*\}", r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return text  # Return raw text as fallback
