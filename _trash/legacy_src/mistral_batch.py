"""Mistral Batch API (Jobs API) client — ``/v1/batch/jobs``.

Contract validated empirically against ``mistralai`` SDK 2.9.4 + live probes:

  create    POST /v1/batch/jobs
            {endpoint: "/v1/chat/completions", model, requests: [
              {"custom_id": "...", "body": {chat-completion payload}}],
             timeout_hours}
  get       GET  /v1/batch/jobs/{job_id}   -> BatchJob {id, status,
            total_requests, completed_requests, succeeded_requests,
            failed_requests, outputs, ...}
  status    QUEUED | RUNNING | SUCCESS | FAILED | TIMEOUT_EXCEEDED |
            CANCELLATION_REQUESTED | CANCELLED

NOTE: the Batch API requires a Mistral plan with BILLING ENABLED — creating a
job on a free/credit plan returns HTTP 402 ("enable billing via the console").
Callers MUST fall back to sequential calls whenever create/poll fails.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.config import AppConfig

logger = logging.getLogger("src.llm.mistral_batch")

DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_TIMEOUT_HOURS = 2
TERMINAL_STATUSES = ("SUCCESS", "FAILED", "TIMEOUT_EXCEEDED",
                     "CANCELLATION_REQUESTED", "CANCELLED")


class MistralBatchError(Exception):
    """Batch API failure (auth, billing, transport, terminal job state)."""


class MistralBatch:
    """Thin async client for Mistral's Jobs/Batch API (no new dependency)."""

    def __init__(self, config: Any = None, *, base_url: str = "",
                 api_key: str = "", model: str = ""):
        cfg = config or AppConfig()
        self.base_url = (base_url or cfg.mistral_base_url
                         or "https://api.mistral.ai/v1").rstrip("/")
        self.api_key = api_key or cfg.mistral_api_key
        self.model = model or cfg.mistral_model

    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, path: str, body: Optional[dict] = None,
                       timeout: float = 60.0) -> Any:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, f"{self.base_url}{path}",
                                        headers=self._headers(),
                                        json=body)
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400:
            detail = (data.get("detail") or data.get("message")
                      or resp.text[:200]) if isinstance(data, dict) else resp.text[:200]
            raise MistralBatchError(
                f"mistral batch {method} {path} -> HTTP {resp.status_code}: {detail}")
        return data

    # ------------------------------------------------------------------
    async def create_job(self, requests: List[dict], *,
                         endpoint: str = DEFAULT_ENDPOINT,
                         model: str = "",
                         timeout_hours: int = DEFAULT_TIMEOUT_HOURS,
                         metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """POST /v1/batch/jobs with inline requests; returns the job dict."""
        payload: Dict[str, Any] = {
            "endpoint": endpoint,
            "model": model or self.model,
            "requests": list(requests),
            "timeout_hours": timeout_hours,
        }
        if metadata:
            payload["metadata"] = metadata
        return await self._request("POST", "/batch/jobs", payload)

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/batch/jobs/{job_id}")

    async def run(
        self,
        requests: List[dict],
        *,
        model: str = "",
        poll_seconds: float = 5.0,
        timeout_seconds: float = 900.0,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a batch job and poll until SUCCESS or a terminal state.

        Returns the final job dict (``outputs`` holds the per-request
        results). Raises MistralBatchError on any failure.
        """
        import math

        job = await self.create_job(requests, model=model, metadata=metadata)
        job_id = job.get("id")
        if not job_id:
            raise MistralBatchError(f"batch create returned no job id: {job}")
        deadline = time.monotonic() + timeout_seconds
        sleeps = max(0.5, float(poll_seconds))
        while True:
            job = await self.get_job(job_id)
            status = (job.get("status") or "").upper()
            if status == "SUCCESS":
                return job
            if status in TERMINAL_STATUSES:
                raise MistralBatchError(
                    f"mistral batch job {job_id} ended {status}: "
                    f"succeeded={job.get('succeeded_requests')} "
                    f"failed={job.get('failed_requests')} "
                    f"errors={job.get('errors')}")
            if time.monotonic() + sleeps > deadline:
                raise MistralBatchError(
                    f"mistral batch job {job_id} not finished after "
                    f"{timeout_seconds:.0f}s (status={status})")
            await self._sleep(sleeps)
            # small linear backoff so we don't hammer the status endpoint
            sleeps = min(sleeps * 1.3, 30.0)

    @staticmethod
    async def _sleep(seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)

    # ------------------------------------------------------------------
    @staticmethod
    def extract_outputs(job: Dict[str, Any], custom_ids: List[str],
                        ) -> Dict[str, Optional[str]]:
        """Map custom_id -> assistant text from a finished job.

        Tolerates several server shapes for ``outputs`` (list of dicts with
        ``custom_id``/``id`` + ``response``/``body``). Missing/failed entries
        map to None so the caller can treat them as critic failures.
        """
        outputs = job.get("outputs") or []
        if not isinstance(outputs, list):
            try:
                outputs = list(outputs.values()) if isinstance(outputs, dict) else []
            except AttributeError:
                outputs = []

        index: Dict[str, List[dict]] = {}
        ordered: List[dict] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            cid = item.get("custom_id") or item.get("id") or ""
            if cid:
                index.setdefault(str(cid), []).append(item)
            else:
                ordered.append(item)

        def _text(item: dict) -> str:
            resp = item.get("response") if isinstance(item.get("response"), dict) else None
            if resp is None and isinstance(item.get("body"), dict):
                resp = item["body"].get("response")
            if isinstance(resp, dict):
                for choice in resp.get("choices") or []:
                    msg = choice.get("message") or {}
                    content = msg.get("content")
                    if content:
                        return str(content).strip()
            out = item.get("output") or item.get("content") or item.get("text")
            return str(out).strip() if out else ""

        result: Dict[str, Optional[str]] = {}
        for cid in custom_ids:
            items = index.get(cid)
            if items:
                result[cid] = _text(items[0]) or None
        # fall back to positional order for entries without ids
        missing = [c for c in custom_ids if c not in result]
        for cid, item in zip(missing, ordered):
            if cid not in result:
                result[cid] = _text(item) or None
        return result