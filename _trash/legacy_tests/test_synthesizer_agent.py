"""Synthesizer tests: provenance-rich prompts and citation repair."""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.agents.evidence import Evidence
from src.agents.planner import QueryPlan, SubQuery
from src.agents.synthesizer import Synthesizer, format_evidence_line
from src.config import AppConfig

from tests.conftest import native_test_model

PLAN = QueryPlan(
    original_query="Which repair techniques were associated with recurrent coarctation?",
    question_type="association",
    subqueries=[SubQuery(id="H1", target="repair techniques", focus="association")],
)

EVIDENCE = Evidence(
    subquery_id="H1",
    document_id="PMC11743609",
    chunk_id="c42",
    source="PMC11743609",
    claim="End-to-end anastomosis was associated with recurrent coarctation in 23.1% (p=0.04)",
    supporting_text="End-to-end anastomosis was associated with recurrent coarctation in 23.1% of patients (p = 0.04)",
    supports_claim=True,
    confidence=0.95,
    section="Results",
    evidence_id="E-H1-1",
)


def _answer_json(document_id="PMC11743609", chunk_id="c42") -> str:
    return json.dumps({
        "summary": "End-to-end anastomosis was associated with 23.1% recurrence (p=0.04).",
        "sections": [{
            "heading": "Repair techniques",
            "body": "End-to-end anastomosis showed 23.1% recurrence (p = 0.04).",
            "citations": [{
                "subquery_id": "H1",
                "document_id": document_id,
                "chunk_id": chunk_id,
                "section": "Results",
                "evidence_id": "E-H1-1",
                "quote": "23.1% of patients (p = 0.04)",
            }],
        }],
        "limitations": [],
        "coverage_note": "H1 covered",
    })


async def test_synthesis_uses_verified_evidence():
    synth = Synthesizer(model=native_test_model(_answer_json()), config=AppConfig())
    report = SimpleNamespace(verdicts=[SimpleNamespace(group_id="H1", status="covered")])
    ans = await synth.synthesize(PLAN.original_query, PLAN, report, [EVIDENCE],
                                 ["H1: covered"])
    assert ans.summary
    citation = ans.sections[0].citations[0]
    assert citation.document_id == "PMC11743609"
    assert citation.chunk_id == "c42"


async def test_prompt_includes_provenance(monkeypatch):
    """The prompt must carry doc/chunk ids, otherwise the model invents them."""
    from src.agents.synthesizer import FinalAnswer

    captured = {}

    async def fake_ask(agent, prompt, output_type, **kwargs):
        captured["prompt"] = prompt
        return FinalAnswer.model_validate(json.loads(_answer_json()))

    # synthesizer imports ask_structured inside the call, so patch the source
    # module the import resolves against.
    import src.llm.run as run_mod

    monkeypatch.setattr(run_mod, "ask_structured", fake_ask)
    synth = Synthesizer(model=None, config=AppConfig())
    await synth.synthesize("q", PLAN, SimpleNamespace(verdicts=[]), [EVIDENCE])
    prompt = captured["prompt"]
    assert "document=PMC11743609" in prompt
    assert "chunk=c42" in prompt
    assert "[E-H1-1]" in prompt


async def test_mangled_citations_repaired_from_evidence_ids():
    """The observed production bug: model cites document_id='9' (a reference
    number). Citation repair must overwrite it from the real evidence item."""
    synth = Synthesizer(model=native_test_model(_answer_json(document_id="9", chunk_id="unknown")),
                        config=AppConfig())
    report = SimpleNamespace(verdicts=[])
    ans = await synth.synthesize("q", PLAN, report, [EVIDENCE], [])
    citation = ans.sections[0].citations[0]
    assert citation.document_id == "PMC11743609"
    assert citation.chunk_id == "c42"


def test_format_evidence_line_carries_ids():
    line = format_evidence_line(EVIDENCE)
    assert "PMC11743609" in line
    assert "E-H1-1" in line
    assert "Results" in line
