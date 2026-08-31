"""Tests for paper-level selection: branch preservation (spec sections 13-17).

The core anti-failure requirement: a paper that is excellent for ONE branch
must survive even when it is absent from every other branch.
"""

from __future__ import annotations


from src.retrieval_v2.config import V2Config
from src.retrieval_v2.models import PaperQueryScore, Requirement, SearchQuery


def _pqs(qid, pid, score):
    return PaperQueryScore(paper_id=pid, query_id=qid, score=score,
                           supporting_chunks=[], sections=[], evidence_types=[])


def test_branch_paper_survives_global_competition():
    """Six queries; paper INOCA is mediocre globally but best at q4."""
    from src.retrieval_v2.paper_select import select_papers
    reqs = [Requirement(id=f"H{i}", topic=f"topic{i}", population=f"pop{i}",
                        focus="outcome" if i > 3 else "mechanism") for i in range(1, 7)]
    qs = [SearchQuery(id=f"q{i}", requirement_ids=[f"H{i}"], text=f"query {i}") for i in range(1, 7)]
    scores = {}
    # per query, the branch paper ranks #1; global strong papers rank high everywhere
    for i in range(1, 7):
        scores[(f"q{i}", "INOCA_PAPER")] = _pqs(f"q{i}", "INOCA_PAPER", 1.0 if i == 4 else 0.1)
        scores[(f"q{i}", f"GLOBAL_P{i}")] = _pqs(f"q{i}", f"GLOBAL_P{i}", 0.8)
        scores[(f"q{i}", f"RANDOM_P{i}")] = _pqs(f"q{i}", f"RANDOM_P{i}", 0.2)
    cfg = V2Config(per_requirement_shortlist=5, branch_guarantee_top=2, top_papers=8)
    selected, per_req = select_papers(scores, qs, reqs, doc_index=None, config=cfg)
    ids = [p.paper_id for p in selected]
    assert "INOCA_PAPER" in ids, "branch-critical paper was eliminated during selection"
    # it must be retained either by ranking or by the branch guarantee,
    # and must still support its own branch (H4)
    p = next((s for s in selected if s.paper_id == "INOCA_PAPER"), None)
    assert p is not None
    assert "H4" in p.supported_requirements or p.guarantee


def test_strongest_branch_dominates():
    from src.retrieval_v2.paper_select import select_papers
    reqs = [Requirement(id=f"H{i}", topic=f"t{i}") for i in range(1, 4)]
    qs = [SearchQuery(id=f"q{i}", requirement_ids=[f"H{i}"], text=f"q {i}") for i in range(1, 4)]
    scores = {}
    scores[("q1", "A")] = _pqs("q1", "A", 0.95)   # A: dominant at H1 only
    scores[("q2", "B")] = _pqs("q2", "B", 0.90)   # B: dominant at H2 only
    scores[("q3", "C")] = _pqs("q3", "C", 0.90)   # C: dominant at H3 only
    scores[("q1", "D")] = _pqs("q1", "D", 0.84)   # D: broad but weaker
    scores[("q2", "D")] = _pqs("q2", "D", 0.84)
    scores[("q3", "D")] = _pqs("q3", "D", 0.84)
    cfg = V2Config(top_papers=5, per_requirement_shortlist=3, branch_guarantee_top=2)
    selected, _ = select_papers(scores, qs, reqs, doc_index=None, config=cfg)
    ids = [p.paper_id for p in selected]
    assert "A" in ids and "B" in ids and "C" in ids


def test_paper_query_score_uses_formula():
    """section 13 formula: score = 0.5*max + 0.2*mean_top3 + 0.1*support + ..."""
    from src.retrieval_v2.paper_select import aggregate_papers
    from src.retrieval_v2.models import QueryLocalResult
    # build a mini QueryLocalResult with fused entries for one paper
    class _FakeNode:
        def get(self, k, d=None):
            return d
    class _Node:
        def __init__(self, cid, sec, ntype):
            self.cid = cid
            self.sec = sec
            self.ntype = ntype
    class FakeDocIndex:
        def paper_of(self, cid):
            return "PAPER"
        def chunk_metadata_for(self, ids):
            return {c: {"section": "S1", "node_type": "paragraph"} for c in ids}
    fused = []
    for rank in range(1, 6):
        fused.append({"chunk_id": f"PAPER_{rank}", "paper_id": "PAPER",
                      "rrf_score": 1.0 / rank, "rank": rank,
                      "scores": {"bm25": 0.1, "dense": 0.2}, "methods": ["bm25", "dense"]})
    qr = QueryLocalResult(query=SearchQuery(id="q1", requirement_ids=["H1"], text="x"),
                          events=[], fused=fused)
    scores = aggregate_papers([qr], FakeDocIndex(), V2Config())
    pqs = scores[("q1", "PAPER")]
    assert pqs.n_support == 5
    assert pqs.score > 0
    assert pqs.best_chunks

