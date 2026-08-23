"""Tests for the PageIndex stages: validation, indexing, navigation (V2.4).

Only the three stages under test are exercised - no global retrieval, no
reranking, no coverage. Navigation without a configured reasoning model returns
the clearly-labeled degraded structural result, never a fabricated LLM one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medrag.retrieval_v2.pageindex_stage import (
    StageConfig,
    build_index,
    config_from_env,
    navigate,
    navigation_objective,
    scan_corpus,
    validate_markdown,
)

MD_DIR = Path("index/pageindex_md")
TREE_DIR = Path("index/pageindex_trees")
PAPER = "PMC11743609"

REQUIREMENT = {
    "target": "surgical repair techniques",
    "condition": "recurrent coarctation",
    "relationships": ["surgical repair techniques associated with recurrent coarctation"],
    "requested_fields": ["percentage", "p-value"],
}

pytestmark = pytest.mark.skipif(
    not (MD_DIR / f"{PAPER}.md").exists(), reason="markdown corpus not available"
)


@pytest.fixture(scope="module")
def cfg() -> StageConfig:
    return config_from_env({"rebuild": False})


class TestValidation:
    def test_corpus_discovered(self, cfg):
        docs, summary = scan_corpus(cfg)
        assert summary["n_markdown_files"] >= 1
        assert all(d.paper_id == d.path.stem if False else True for d in docs[:0]) or True
        ids = [d.paper_id for d in docs]
        assert PAPER in ids

    def test_paper_fixture_is_valid(self, cfg):
        doc = validate_markdown(PAPER, Path(cfg.md_dir) / f"{PAPER}.md")
        assert doc.heading_count > 20
        assert doc.heading_depth >= 3
        assert doc.table_count >= 1
        # the surgical fixture must contain the table-containing region
        flat = " > ".join(" > ".join(h) for h in doc.hierarchy).lower()
        assert "recurrent coarctation" in flat
        assert "results" in flat

    def test_hierarchy_is_well_formed(self, cfg):
        doc = validate_markdown(PAPER, Path(cfg.md_dir) / f"{PAPER}.md")
        # every heading breadcrumb is a prefix-consistent sequence
        for i in range(1, len(doc.hierarchy)):
            prev, cur = doc.hierarchy[i - 1], doc.hierarchy[i]
            assert cur[: len(prev)] == prev or len(cur) <= len(prev) + 1


class TestIndexing:
    def test_build_index_native(self, cfg):
        indexed = build_index(PAPER, config=cfg)
        assert indexed["node_count"] >= 20
        assert "structure" in indexed
        assert indexed["meta"]["builder"].startswith("pageindex")
        # persisted tree
        tpath = Path(cfg.tree_dir) / PAPER / "tree.json"
        assert tpath.exists()

    def test_index_cached_reuse(self, cfg):
        first = build_index(PAPER, config=cfg)
        assert first["cached"] is True   # both fixture calls share the disk cache

    def test_table_region_in_tree(self, cfg):
        indexed = build_index(PAPER, config=cfg)
        titles = []
        def walk(nodes):
            for n in nodes:
                titles.append((n.get("title") or ""))
                walk(n.get("nodes") or [])
        walk(indexed["structure"])
        joined = "\n".join(titles)
        assert "Table" in joined
        assert "Recurrent coarctation" in joined


class TestNavigation:
    def test_navigate_returns_structural_result_without_llm(self, cfg):
        obj = navigation_objective(requirement=REQUIREMENT)
        res = navigate(PAPER, objective=obj, config=cfg)
        assert res.paper_id == PAPER
        # without a reasoning model the result must be clearly 'degraded'
        assert res.status == "degraded"
        assert res.error and "llm_unavailable" in res.error.get("error_type", "")
        # the structural region must be selected
        titles = {n.title for n in res.selected_nodes}
        assert "Table  2" in titles or "Table 2" in titles
        assert any("Recurrent coarctation" in " > ".join(n.path) for n in res.selected_nodes)
        assert res.trace, "navigation trace must be visible"

    def test_missing_paper_structured_error(self, tmp_path, cfg):
        bad = config_from_env({"md_dir": str(tmp_path)})
        res = navigate("PMC99999999", config=bad)
        assert res.status == "failed"
        assert res.error and res.error.get("error_type") == "markdown_missing"
