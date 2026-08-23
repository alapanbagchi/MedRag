"""Tests for the logical document structure index (spec sections 4-5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from medrag.retrieval_v2.document_index import LogicalDocumentIndex

CORPUS = Path("index/corpus.parquet")

pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus.parquet not available")


@pytest.fixture(scope="module")
def idx():
    return LogicalDocumentIndex(CORPUS)


def test_index_basics(idx):
    assert idx.n_chunks > 100_000
    assert idx.n_papers > 1_000
    assert idx.paper_size("PMC11743015") > 10


def test_get_paper_nodes_ordered(idx):
    nodes = idx.get_paper_nodes("PMC11743015")
    assert nodes
    positions = [n["position"] for n in nodes]
    assert positions == sorted(positions)


def test_table_assembly(idx):
    tbl = idx.get_table("PMC10327125", "T1")
    assert tbl["summary"] is not None
    assert len(tbl["rows"]) >= 5
    assert len(tbl["footnotes"]) >= 1
    for cid in tbl["all_node_ids"]:
        assert idx.get_node(cid) is not None


def test_figure_lookup(idx):
    f = idx.get_figure("PMC10327125", "F6")
    assert f is not None and f["node_type"] == "figure"


def test_section_and_neighbors(idx):
    nodes = idx.get_section("PMC11743015", "Results")
    assert nodes
    neigh = idx.get_neighbors("PMC11743015", nodes[0]["chunk_id"], 1, 1)
    assert neigh


def test_children_of_table_summary(idx):
    tbl = idx.get_table("PMC10327125", "T2")
    summary = tbl["summary"]
    children = idx.get_children("PMC10327125", summary["chunk_id"])
    assert len(children) >= len(tbl["rows"])

