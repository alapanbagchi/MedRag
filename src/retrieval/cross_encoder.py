"""Two-tier candidate filtering before LLM verification.

Tier 1 (always on): deterministic intent scoring (see retrieval.reranker).
Tier 2 (opt-in via ENABLE_CROSS_ENCODER=1): MedCPT cross-encoder scoring of
(subquery query, unit text) pairs.

Neither tier REPLACES the LLM verifier; they ORDER candidates so that when the
verifier budget is capped (VERIFY_BATCH_MAX_DOCS / VERIFY_BATCH_MAX_TOKENS),
the strongest units are verified first. Neural scoring degrades gracefully:
if the model can't be loaded (offline, no weights), we fall back to tier 1.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger("src.cross_encoder")


class MedCPTCrossEncoder:
    """Lazy-loading wrapper around ncbi/MedCPT-Cross-Encoder."""

    def __init__(self, model_name: str = "ncbi/MedCPT-Cross-Encoder") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("loading cross-encoder %s ...", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
            self._torch = torch
            return True
        except Exception as exc:
            logger.warning("cross-encoder unavailable (%s); falling back to intent scores", exc)
            self._model = False  # sentinel: failed load, don't retry
            return False

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    def score(self, query: str, texts: List[str]) -> List[float]:
        """Relevance logits for (query, text) pairs; [] when unavailable."""
        if not self._ensure_loaded():
            return []
        torch = self._torch
        scores: List[float] = []
        batch = 16
        with torch.no_grad():
            for start in range(0, len(texts), batch):
                chunk = texts[start : start + batch]
                enc = self._tokenizer(
                    [query] * len(chunk), chunk, truncation=True, max_length=512,
                    padding=True, return_tensors="pt",
                )
                logits = self._model(**enc).logits.squeeze(-1)
                scores.extend(float(x) for x in logits.tolist())
        return scores


def _unit_as_namespace(unit: dict) -> Any:
    class _NS:  # minimal duck-type for reranker.intent_score
        text = unit.get("unit_text", "")
        node_type = unit.get("name", "paragraph")
        section = ""
    return _NS()


class UnitPrefilter:
    """Orders structural units before LLM verification."""

    def __init__(self, config: Any = None, encoder: Optional[MedCPTCrossEncoder] = None) -> None:
        self.config = config
        self.enabled = bool(getattr(config, "enable_cross_encoder", False)) if config else False
        self._encoder = encoder

    def _get_encoder(self) -> Optional[MedCPTCrossEncoder]:
        if not self.enabled:
            return None
        if self._encoder is None:
            from src.config import AppConfig

            cfg = self.config or AppConfig()
            self._encoder = MedCPTCrossEncoder(cfg.cross_encoder_model)
        return self._encoder

    def order_units(self, units: List[dict], subquery_query: str) -> List[dict]:
        """Return units sorted best-first, each annotated with prefilter_score."""
        if len(units) <= 1:
            return list(units)

        from src.retrieval.reranker import intent_score

        base_scores = {
            u["chunk_id"]: intent_score(subquery_query_dummy(subquery_query), _unit_as_namespace(u))
            for u in units
        }

        encoder = self._get_encoder()
        if encoder is not None:
            enc = encoder.score(
                subquery_query,
                [u.get("unit_text", "")[:4000] for u in units],
            )
            if enc:
                for u, s in zip(units, enc):
                    u["prefilter_score"] = float(s)
                ordered = sorted(units, key=lambda u: -u["prefilter_score"])
                return ordered
            # Encoder unavailable -> keep deterministic scores.

        for u in units:
            u["prefilter_score"] = float(base_scores.get(u["chunk_id"], 0.0))
        return sorted(units, key=lambda u: -u["prefilter_score"])


def subquery_query_dummy(query: str) -> Any:
    """Minimal object exposing .query/.target/.evidence_required for intent_score."""
    from src.agents.planner import SubQuery

    return SubQuery(id="P", target=query, focus="evidence", query=query)


__all__ = ["MedCPTCrossEncoder", "UnitPrefilter"]
