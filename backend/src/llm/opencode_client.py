"""OpenCode (Muse Spark) LLM client adapter.

OpenCode uses the OpenAI Responses API format (/v1/responses), NOT the
Chat Completions API (/v1/chat/completions). LangChain's ChatOpenAI expects
Chat Completions, so this module provides a wrapper that translates between
the two formats.

Key gotcha: Muse Spark has an internal reasoning step that consumes most
of the token budget. We set reasoning.effort="low" and a large max_output_tokens
to ensure the actual answer text is not starved.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class OpenCodeClient:
    """Sync client for OpenCode's Responses API."""

    def __init__(
        self,
        base_url: str = "https://opencode.ai/zen/v1",
        api_key: str = "",
        model: str = "muse-spark-1.3-contributor-free",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _convert_messages_to_input(self, messages: list[dict]) -> str | list[dict]:
        """Convert Chat Completions messages to Responses API input format."""
        if not messages:
            return ""
        if len(messages) == 1 and messages[0].get("role") == "user":
            content = messages[0].get("content", "")
            if isinstance(content, str):
                return content
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            result.append({"role": role, "content": str(content)})
        return result

    def complete_sync(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float | None = None,
        reasoning_effort: str = "low",
    ) -> dict:
        """Synchronous non-streaming chat completion. Returns Chat Completions format."""
        body = {
            "model": self.model,
            "input": self._convert_messages_to_input(messages),
            "max_output_tokens": max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False,
            "reasoning": {"effort": reasoning_effort},
        }
        url = f"{self.base_url}/responses"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self._headers(), json=body)
            resp.raise_for_status()
            data = resp.json()
        return self._to_chat_completion(data)

    def _to_chat_completion(self, data: dict) -> dict:
        """Convert Responses API response to Chat Completions format."""
        text = ""
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text += content.get("text", "")

        usage = data.get("usage", {})
        status = data.get("status", "")
        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "created": data.get("created_at", int(time.time())),
            "model": data.get("model", self.model),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop" if status == "completed" else "length",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }


class OpenCodeChatModel(BaseChatModel):
    """LangChain-compatible ChatModel for OpenCode (Muse Spark).

    Wraps the Responses API and presents it as a BaseChatModel so it works
    with deepagents, LangChain agents, and the xdeep pipeline.
    """

    model_name: str = "muse-spark-1.3-contributor-free"
    reasoning_effort: str = "low"
    _client: OpenCodeClient = PrivateAttr()

    def __init__(self, client: OpenCodeClient, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "opencode-muse-spark"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model_name,
            "base_url": self._client.base_url,
        }

    def _to_langchain_messages(self, messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to plain dicts."""
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation."""
        plain_msgs = self._to_langchain_messages(messages)
        max_tokens = kwargs.get("max_tokens", 2048)
        data = self._client.complete_sync(
            messages=plain_msgs,
            max_tokens=max_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=text))
            ],
            llm_output={
                "model": self.model_name,
                "token_usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            },
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Synchronous streaming — falls back to non-streaming."""
        result = self._generate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            yield gen


def build_opencode_model(
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> OpenCodeChatModel:
    """Build an OpenCode model compatible with the xdeep pipeline."""
    client = OpenCodeClient(
        base_url=base_url or "https://opencode.ai/zen/v1",
        api_key=api_key or "dummy",
        model=model or "muse-spark-1.3-contributor-free",
        temperature=temperature,
        timeout=timeout,
    )
    return OpenCodeChatModel(client=client, model_name=client.model)
