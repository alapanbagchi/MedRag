"""Unit tests for the end-to-end agentic pipeline (fakes, no live LLM)."""
import pytest

from src.agentic.planner import PlannedEntity, SubQueryPlan, Decomposition
from src.agentic.loop import EvidenceReport
from src.agentic.pipeline import AgenticPipeline


class FakePlanner:
    def __init__(self, decomposition):
        self.decomposition = decomposition

    async def plan(self, query):
        return self.decomposition


class FakeEnricher:
    def __init__(self):
        self.enriched = 0

    async def enrich_subquery(self, sub):
        self.enriched += 1
        return []

    def apply_to_query(self, sub, terms):
        return sub.query or sub.target


class FakeLoop:
    def __init__(self, reports):
        self.reports = reports

    async def run(self, sub, base_query=""):
        return self.reports.get(sub.id, EvidenceReport(subquery_id=sub.id, succeeded=False, summary="none"))


def _decomposition():
    s1 = SubQueryPlan(id="H1", target="vasospasm IR", query="vasospasm IR access",
                      entities=[PlannedEntity(text="vasospasm")])
    s2 = SubQueryPlan(id="H2", target="KID-ACS", query="KID-ACS PENK AKI",
                      entities=[PlannedEntity(text="proenkephalin")])
    return Decomposition(question_type="comprehensive", subqueries=[s1, s2])


def _make(loop_reports):
    pipeline = AgenticPipeline.__new__(AgenticPipeline)
    pipeline.config = object()
    pipeline.planner = FakePlanner(_decomposition())
    pipeline.enricher = FakeEnricher()
    pipeline.loop = FakeLoop(loop_reports)
    return pipeline


@pytest.mark.asyncio
async def test_pipeline_merge_shape():
    reports = {
        "H1": EvidenceReport(subquery_id="H1", succeeded=True, summary="found vasospasm",
                             evidence_excerpts=["excerpt A"], citations=["PMC1/c1"],
                             searches_performed=["q1"]),
        "H2": EvidenceReport(subquery_id="H2", succeeded=False, summary="gave up",
                             searches_performed=["q2"]),
    }
    pipeline = _make(reports)
    out = await pipeline.answer("test query")
    assert out["num_subqueries"] == 2
    assert out["succeeded_subqueries"] == 1
    assert "PMC1/c1" in out["citations"]
    assert out["subqueries"][0]["succeeded"] is True
    assert out["subqueries"][1]["succeeded"] is False
    assert pipeline.enricher.enriched == 2  # both subqueries enriched


@pytest.mark.asyncio
async def test_pipeline_loop_failure_is_captured():
    class BoomLoop:
        async def run(self, sub, base_query=""):
            raise RuntimeError("boom")
    pipeline = AgenticPipeline.__new__(AgenticPipeline)
    pipeline.config = object()
    pipeline.planner = FakePlanner(_decomposition())
    pipeline.enricher = FakeEnricher()
    pipeline.loop = BoomLoop()
    out = await pipeline.answer("test query")
    # no exception propagates; every subquery reports failure gracefully
    assert out["succeeded_subqueries"] == 0
    assert all(not s["succeeded"] for s in out["subqueries"])
    assert any("loop error" in s["summary"] for s in out["subqueries"])
