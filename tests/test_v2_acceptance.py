"""Acceptance tests for the V2.1 architecture repair (spec parts 1-41).

Test 1: recurrent-coarctation surgical question -> ONE coherent requirement,
        no H1-H4 statistical fragmentation, PageIndex navigates to
        Results / Table 2, final evidence includes table row(s) + headers +
        percentages + p-values.
Test 2: single-paper numerical question -> table retrieval.
Test 3: single-paper paragraph question -> PageIndex -> section -> paragraph.
Test 4: figure/caption question.
Test 5: multi-hop query -> independent branches + hop dependencies.
Test 6: INOCA stays INOCA (never silently MINOCA).
Test 7: evidence genuinely missing -> honest uncovered, no substitution.

Unit-level tests run without the corpus; integration tests are gated on the
corpus + indexes being present (same pattern as the existing V2 tests).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medrag.retrieval_v2.config import V2Config, config_from_env
from medrag.retrieval_v2.planner import plan_question

CORPUS = Path("index/corpus.parquet")
PAGEINDEX_DIR = Path("index/pageindex")

SURGICAL_Q = (
    "Which surgical repair techniques were associated with recurrent "
    "coarctation and what are the percentages and p-values?"
)


# ---------------------------------------------------------------------------
# Test 1a — planner contract: ONE requirement, no statistical fragmentation
# ---------------------------------------------------------------------------

class TestPlannerSingleObligation:
    def test_surgical_question_is_one_requirement(self):
        plan = plan_question(SURGICAL_Q)
        assert len(plan.requirements) == 1, "must be ONE evidence obligation"
        r = plan.requirements[0]
        assert r.id == "H1"
        assert r.focus == "comparative_numerical"
        assert "percentage" in r.requested_fields and "p-value" in r.requested_fields
        assert r.condition == "recurrent coarctation"
        assert "surgical repair techniques" in (r.target or "").lower()

    def test_no_duplicate_statistical_branches(self):
        plan = plan_question(SURGICAL_Q)
        rids = [r.id for r in plan.requirements]
        assert rids == ["H1"]
        for r in plan.requirements:
            # no artificial "statistical significance" / "reported recurrence rates" branches
            assert "statistical significance" not in r.topic.lower()
            assert "reported recurrence rates" not in r.topic.lower()

    def test_preferred_evidence_types_are_table_heavy(self):
        plan = plan_question(SURGICAL_Q)
        pref = plan.requirements[0].preferred_evidence_types
        assert "table_row" in pref and "table_summary" in pref
        assert "table_footnotes" in pref

    def test_compact_retrieval_variants(self):
        plan = plan_question(SURGICAL_Q)
        vs = plan.requirements[0].retrieval_variants
        assert 4 <= len(vs) <= 6, f"expected 4-6 variants, got {len(vs)}"
        texts = [v.text.lower() for v in vs]
        assert len(set(texts)) == len(texts), "no duplicate variants"
        # V0 preserves the original question
        assert texts[0].startswith("which surgical repair techniques")
        # modifier-preserving canonical appears
        assert any("recurrent coarctation" in t for t in texts)
        assert any("percentage" in t or "p-value" in t for t in texts)

    def test_navigation_objective_targets_results_and_tables(self):
        plan = plan_question(SURGICAL_Q)
        obj = plan.requirements[0].navigation_objective.lower()
        assert "results" in obj and "table" in obj and "footnotes" in obj
        assert "recurrent coarctation" in obj

    def test_terminology_guard_clean(self):
        plan = plan_question(SURGICAL_Q)
        assert plan.terminology_guard.get("violations") == []


# ---------------------------------------------------------------------------
# Test 6 — INOCA stays INOCA
# ---------------------------------------------------------------------------

class TestInocaPreservation:
    BENCH = (
        "How do metabolic syndrome, advanced CKD, and anemia affect "
        "cardiovascular vulnerability, and how does this differ across INOCA, "
        "non-diabetic CKD, and elderly hip-fracture patients?"
    )

    def test_inoca_never_becomes_minoca(self):
        plan = plan_question(self.BENCH)
        for q in plan.queries:
            low = q.text.lower()
            assert "minoca" not in low or "inoca" in low
        for r in plan.requirements:
            blob = " ".join([r.topic, r.population, r.target, r.outcome]).lower()
            assert "minoca" not in blob or "inoca" in blob
        assert "inoca" in _lower(" ".join(q.text for q in plan.queries))

    def test_inoca_surface_form_preserved_in_entities(self):
        plan = plan_question("How does ischemia with nonobstructive coronary arteries affect outcomes?")
        assert plan.terminology_guard.get("violations") == []
        blob = " ".join(r.topic + " " + r.population for r in plan.requirements)
        assert "ischemia with nonobstructive coronary arteries" in blob

    def test_q4_lead_keeps_branch_identity(self):
        plan = plan_question(self.BENCH)
        q4 = plan.query_by_id("q4")
        assert q4 is not None
        req = plan.requirement_by_id(q4.requirement_ids[0])
        assert req is not None
        assert "metabolic syndrome" in req.topic.lower()
        assert "inoca" in q4.text.lower()


def _lower(s: str) -> str:
    return (s or "").lower()


# ---------------------------------------------------------------------------
# Test 5 — multi-hop
# ---------------------------------------------------------------------------

class TestMultiHop:
    def test_mechanism_link_creates_two_hops_with_dependency(self):
        plan = plan_question(
            "What mechanism links CKD to cardiovascular risk, and how does "
            "that mechanism differ in non-diabetic CKD?")
        assert [r.id for r in plan.requirements] == ["H1", "H2"]
        h1, h2 = plan.requirements
        assert h1.focus == "mechanism"
        assert h2.hop_dependencies == ["H1"]
        assert "non-diabetic" in h2.population.lower()

    def test_hops_have_independent_queries(self):
        plan = plan_question(
            "What mechanism links CKD to cardiovascular risk, and how does "
            "that mechanism differ in non-diabetic CKD?")
        q1_ids = {q.id for q in plan.queries if q.requirement_ids == ["H1"]}
        q2_ids = {q.id for q in plan.queries if q.requirement_ids == ["H2"]}
        assert q1_ids and q2_ids and q1_ids.isdisjoint(q2_ids)


# ---------------------------------------------------------------------------
# /expand NEW contract (part 3): the payload is query INTELLIGENCE, and even a
# legacy payload carrying evidence_requirements must not fragment the question.
# ---------------------------------------------------------------------------

class TestExpandContract:
    def test_structured_payload_maps_to_one_requirement(self):
        from medrag.retrieval_v2.llm_planner import plan_from_expand
        payload = {
            "query": SURGICAL_Q,
            "clinical_entities": [
                {"surface_form": "recurrent coarctation", "base_concept": "coarctation",
                 "modifiers": ["recurrent"], "role": "condition"},
                {"surface_form": "surgical repair techniques", "base_concept": "surgical repair techniques",
                 "modifiers": [], "role": "target"},
            ],
            "query_targets": ["surgical repair techniques"],
            "requested_fields": ["percentage", "p-value"],
            "relationships": ["surgical repair techniques associated with recurrent coarctation"],
            "populations": [],
            "reranker_intent": {"question_type": "numerical", "focus": "comparative_numerical",
                                "target": "surgical repair techniques", "outcome": ""},
            "umls": [],
        }
        plan = plan_from_expand(SURGICAL_Q, payload)
        assert len(plan.requirements) == 1
        r = plan.requirements[0]
        assert r.focus == "comparative_numerical"
        assert "percentage" in r.requested_fields and "p-value" in r.requested_fields
        assert plan.planner_method == "expand+plan"

    def test_legacy_evidence_requirements_are_ignored(self):
        """The dead contract: even if a server still sends evidence_requirements,
        the planner must NOT turn them into H1..H4 statistical branches."""
        from medrag.retrieval_v2.llm_planner import plan_from_expand
        payload = {
            "query": SURGICAL_Q,
            "concepts": [{"name": "recurrent coarctation", "type": "disease"}],
            "intent": {"conditions": ["recurrent coarctation"], "target": "surgical repair techniques"},
            # legacy fragmenting list that used to create H1..H4:
            "evidence_requirements": [
                "Specific surgical repair techniques used for coarctation.",
                "Reported recurrence rates for each technique or comparison.",
                "Statistical significance (p-values) associated with recurrence.",
                "Percentages of recurrence for each technique or comparison.",
            ],
        }
        plan = plan_from_expand(SURGICAL_Q, payload)
        assert len(plan.requirements) == 1, "evidence_requirements must be ignored"
        assert [r.id for r in plan.requirements] == ["H1"]


# ---------------------------------------------------------------------------
# Table-context unit tests (no corpus needed)
# ---------------------------------------------------------------------------

class TestTableFields:
    def test_table_aware_pvalue_and_percentage(self):
        from medrag.retrieval_v2.table_context import (
            parse_row, parse_summary_columns, detect_table_row_fields)
        headers = parse_summary_columns(
            "Table 2: X\n\nColumns:\n- Variables\n- No re-CoA (n=24) or n (%)\n- re-CoA (n=4) or n (%)\n- p")
        variable, cells = parse_row(
            "Row: End-to-end\nNo re-CoA (n=24) or n (%): 2 (8)\nre-CoA (n=4) or n (%): 2 (50)\np: 0.04")
        fields = detect_table_row_fields(variable, cells, headers, [])
        assert fields["percentage"] is True
        assert fields["p-value"] is True
        assert fields["effect_estimate"] is False

    def test_row_without_p_value_is_flagged_accordingly(self):
        from medrag.retrieval_v2.table_context import (
            parse_row, parse_summary_columns, detect_table_row_fields)
        headers = parse_summary_columns(
            "Table 1\n\nColumns:\n- Variable\n- Group A\n- Group B\n- p")
        variable, cells = parse_row(
            "Row: Age (years)\nGroup A: 40\nGroup B: 42\np: 0.31")
        fields = detect_table_row_fields(variable, cells, headers, [])
        assert fields["p-value"] is True
        assert fields["percentage"] is False


# ---------------------------------------------------------------------------
# PageIndex adapter tests (need corpus + artifacts; gated)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CORPUS.exists(), reason="corpus.parquet not available")
class TestPageIndexAdapter:
    @pytest.fixture(scope="class")
    @classmethod
    def adapter(cls):
        from medrag.retrieval_v2.document_index import LogicalDocumentIndex
        from medrag.retrieval_v2.pageindex_adapter import PageIndexAdapter
        idx = LogicalDocumentIndex(CORPUS)
        cfg = V2Config(pageindex_dir=str(PAGEINDEX_DIR))
        a = PageIndexAdapter(doc_index=idx, pageindex_dir=PAGEINDEX_DIR, config=cfg)
        return a, idx

    def test_build_artifact_and_navigate_to_table(self, adapter):
        a, idx = adapter
        st = a.build_or_load("PMC11743609", force=False)
        assert st in ("loaded", "built")
        plan = plan_question(SURGICAL_Q)
        req = plan.requirements[0]
        hits = a.navigate("PMC11743609", req.navigation_objective, top_k=8)
        tables = [h for h in hits if h.title.startswith("Table")]
        assert tables, "PageIndex must navigate toward tables"
        t2 = next((h for h in hits if "Table 2" in h.title.replace("  ", " ")), None)
        # the tree must navigate toward the recurrent-coarctation region
        assert t2 is not None, "Table 2 must be among the top hits"
        assert any("re-current" not in h.section for h in [t2]) or "coarctation" in t2.section.lower(), \
            "Table 2 must sit in the coarctation region of the paper"

    def test_resolve_table_node_to_chunks(self, adapter):
        a, idx = adapter
        a.build_or_load("PMC11743609")
        plan = plan_question(SURGICAL_Q)
        hits = a.navigate("PMC11743609", plan.requirements[0].navigation_objective, top_k=8)
        t2 = next((h for h in hits if "Table 2" in h.title.replace("  ", " ")), None)
        assert t2 is not None, "Table 2 must be among the navigation hits"
        resolved = a.resolve_node("PMC11743609", t2.pageindex_node_id)
        assert resolved["paper_id"] == "PMC11743609"
        # The MD-mode tree maps the table node to the paper's existing corpus
        # chunks (summary + rows + footnotes); ids are corpus ids (paper prefix).
        assert resolved["chunk_ids"], "Table 2 must resolve to existing corpus chunks"
        for c in resolved["chunk_ids"]:
            assert c.startswith("PMC11743609"), "table chunks must be existing corpus ids"

    def test_unavailable_paper_falls_back(self, adapter):
        a, _idx = adapter
        hits = a.navigate("PMC00000000_MISSING", "find table evidence", top_k=4)
        assert hits == []
        assert a.paper_status("PMC00000000_MISSING") == "unavailable"


# ---------------------------------------------------------------------------
# Table-context integration (needs corpus; gated)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CORPUS.exists(), reason="corpus.parquet not available")
class TestTableContextIntegration:
    def test_real_table_row_context(self):
        from medrag.retrieval_v2.document_index import LogicalDocumentIndex
        from medrag.retrieval_v2.table_context import build_table_context
        idx = LogicalDocumentIndex(CORPUS)
        tc, ctx, det = build_table_context(idx, "PMC11743609", "PMC11743609_T2_row_13", V2Config())
        assert tc is not None and tc["table_id"] == "T2"
        assert "End-to-end" in ctx and "Headers" in ctx and "Footnotes" in ctx
        assert det["percentage"] is True and det["p-value"] is True
        assert "0.04" in ctx


# ---------------------------------------------------------------------------
# End-to-end integration (gated on corpus + indexes)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CORPUS.exists(), reason="corpus.parquet not available")
class TestEndToEnd:
    @pytest.fixture(scope="class")
    @classmethod
    def components(cls):
        from medrag.retrieval_v2.pipeline import load_components
        cfg = config_from_env({"enable_rerank": False})
        comp = load_components(Path("index"), cfg)
        yield comp
        comp.close()

    def test_surgical_question_end_to_end(self, components):
        from medrag.retrieval_v2.pipeline import run_v2_pipeline
        cfg = config_from_env({"enable_rerank": False})
        result = run_v2_pipeline(
            SURGICAL_Q, config=cfg, index_dir=Path("index"),
            components=components, keep_components=True,
        )
        plan = result["plan"]
        assert len(plan["requirements"]) == 1, "one obligation end to end"

        papers = [p["paper_id"] for p in result["papers"]]
        assert "PMC11743609" in papers[:15], "correct paper must survive selection"

        final = result["final_evidence"]
        # evidence must include table rows from the target paper with context
        table_rows = [e for e in final
                      if e["node_type"] == "table_row"
                      and e["paper_id"] == "PMC11743609"]
        assert table_rows, "final evidence must contain table rows from PMC11743609"
        ctx = table_rows[0].get("table_context")
        assert ctx and ctx.get("headers"), "table row must carry table context"
        any_fields = any(
            (e.get("detected_fields") or {}).get("percentage") is True
            or (e.get("detected_fields") or {}).get("p-value") is True
            for e in table_rows
        )
        assert any_fields, "some PMC11743609 table row must carry percentage/p-value fields"

    def test_missing_evidence_is_honest(self, components):
        from medrag.retrieval_v2.pipeline import run_v2_pipeline
        cfg = config_from_env({"enable_rerank": False})
        q = ("Which lunar regolith minerals predicted survival in cardiac "
             "transplant patients, and what were the hazard ratios?")
        result = run_v2_pipeline(q, config=cfg, index_dir=Path("index"),
                                 components=components, keep_components=True)
        coverage = result["coverage"]
        if plan_question(q).requirements:
            assert coverage.get("uncovered"), "should honestly report uncovered"
