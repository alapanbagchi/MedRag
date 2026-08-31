"""Evidence extraction workers: bounded concurrency + quota-aware retries."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from src.config import AppConfig
from src.llm.ratelimit import is_rate_limit_error, retry_after_seconds

logger = logging.getLogger("src.workers")


@dataclass
class EvidenceTask:
    subquery: Any
    document: Any


@dataclass
class EvidenceWorkerResult:
    task: EvidenceTask
    evidence: list = field(default_factory=list)
    failed: bool = False
    error: str = ""
    stats: dict = field(default_factory=dict)


class EvidenceWorkerPool:
    def __init__(self, worker: Callable, config: Optional[AppConfig] = None):
        self.worker = worker
        cfg = config or AppConfig()
        self.max_workers = cfg.max_workers
        self.max_retries = cfg.max_retries
        self.timeout = cfg.request_timeout
        self.llm_max_attempts = getattr(cfg, "max_llm_retries", 4)
        self.retry_base_s = float(getattr(cfg, "llm_retry_base_s", 2.0))
        self._semaphore = asyncio.Semaphore(self.max_workers)

    async def run(self, tasks: List[EvidenceTask]) -> List[EvidenceWorkerResult]:
        if not tasks:
            return []
        return list(await asyncio.gather(*(self._run_one(t) for t in tasks)))

    def _delay_for(self, exc: Exception, attempt: int) -> float:
        """Backoff that survives per-minute quota windows.

        Rate-limit errors honor the server's Retry-After hint (with jitter);
        everything else uses exponential backoff.
        """
        if is_rate_limit_error(exc):
            hint = retry_after_seconds(exc)
            delay = hint if hint is not None else self.retry_base_s * (2 ** attempt)
        else:
            delay = self.retry_base_s * (2 ** attempt)
        return min(delay, 90.0) + random.uniform(0.0, 1.5)

    async def _run_one(self, task: EvidenceTask) -> EvidenceWorkerResult:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    outcome = await asyncio.wait_for(
                        self.worker(task.subquery, task.document), timeout=self.timeout
                    )
                # Worker may return plain evidence list or (evidence, stats).
                if isinstance(outcome, tuple):
                    evidence, stats = outcome
                else:
                    evidence, stats = outcome, {}
                return EvidenceWorkerResult(task=task, evidence=evidence, stats=stats)
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self._delay_for(exc, attempt))
        logger.error("worker failed after %d attempts: %s", self.max_retries, last_exc)
        return EvidenceWorkerResult(
            task=task, failed=True,
            error=str(last_exc)[:300] if last_exc else "unknown",
        )
