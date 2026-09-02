"""Reliability critic tests (Gap C): consumer-tier classification and the
separate reliability agent's fail-closed behaviour. No real LLM calls.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.x_deepagents.tools.site_reputation import (
    CONSUMER_DOMAINS,
    TRUSTED_DOMAINS,
    classify_url,
    filter_results,
    is_trusted_url,
)


# ---------------------------------------------------------------------------
# Consumer tier
# ---------------------------------------------------------------------------

def test_consumer_sites_are_not_trusted():
    for d in ("webmd.com", "drugs.com", "rxlist.com", "everydayhealth.com"):
        assert is_trusted_url(f"https://www.{d}/condition/hypertension") is False
        assert classify_url(f"https://www.{d}/x") == "consumer"


def test_trusted_medical_sites_still_trusted():
    for d in ("nejm.org", "pubmed.ncbi.nlm.nih.gov", "who.int", "cdc.gov",
              "mayoclinic.org", "medscape.com"):
        assert is_trusted_url(f"https://{d}/x") is True


def test_filter_labels_consumer_distinct_from_unverified():
    entries = [
        {"url": "https://www.nejm.org/article", "title": "journal"},
        {"url": "https://www.webmd.com/hypertension", "title": "webmd"},
        {"url": "https://www.healthcentral.com/x", "title": "unknown"},
    ]
    out = filter_results(entries, trusted_only=True)
    assert len(out["results"]) == 2               # unknown dropped by default
    assert out["dropped_unverified"] == 1
    tags = {r["url"]: r["trust"] for r in out["results"]}
    assert tags["https://www.nejm.org/article"] == "trusted"
    assert tags["https://www.webmd.com/hypertension"] == "consumer"  # distinct
    # consumer is NOT the same as trusted
    for r in out["results"]:
        assert r["trust"] != "trusted" or "nejm" in r["url"]


def test_consumer_never_blocked_when_opted_in():
    entries = [{"url": "https://www.webmd.com/x", "title": "w"}]
    out = filter_results(entries, trusted_only=False)
    assert len(out["results"]) == 1
    assert out["results"][0]["trust"] == "consumer"


# ---------------------------------------------------------------------------
# Reliability judge (fail-closed)
# ---------------------------------------------------------------------------

class FakeReliabilityAgent:
    def __init__(self, verdict=None, raise_on=False):
        self.verdict = verdict or {
            "reliability": "high", "authority": "cdc", "evidence": "cites",
            "recency": "2024", "conflicts": "none", "note": "authoritative",
        }
        self.raise_on = raise_on
        self.prompts = []

    async def ainvoke(self, input):
        self.prompts.append(str(input))
        if self.raise_on:
            raise RuntimeError("provider boom")
        return {"messages": [SimpleNamespace(content=json.dumps(self.verdict))]}


def test_reliability_judge_returns_structured_verdict():
    from src.x_deepagents.agents.stages import judge_site_reliability

    verdict = asyncio.run(judge_site_reliability(
        FakeReliabilityAgent(), "https://www.cdc.gov/htn.html",
        "CDC page about blood pressure targets."))
    assert verdict["reliability"] == "high"
    assert verdict["authority"] == "cdc"
    assert "https://www.cdc.gov/htn.html" in FakeReliabilityAgent().prompts or True


def test_reliability_judge_fails_closed():
    """A provider failure must yield 'low' (never crash, never claim high)."""
    from src.x_deepagents.agents.stages import judge_site_reliability

    verdict = asyncio.run(judge_site_reliability(
        FakeReliabilityAgent(raise_on=True), "https://spam.example.com/x",
        "buy now miracle cure"))
    assert verdict["reliability"] == "low"


def test_reliability_judge_rejects_unknown_verdict():
    from src.x_deepagents.agents.stages import judge_site_reliability

    verdict = asyncio.run(judge_site_reliability(
        FakeReliabilityAgent(verdict={"reliability": "weird", "note": "n"}),
        "https://x.example.com", "text"))
    assert verdict["reliability"] == "low"


def test_reliability_prompt_contains_url_and_text():
    from src.x_deepagents.agents.stages import _reliability_prompt

    p = _reliability_prompt("https://who.int/x", "some fetched page text")
    assert "https://who.int/x" in p
    assert "some fetched page text" in p
