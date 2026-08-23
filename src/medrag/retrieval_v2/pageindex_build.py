"""Offline PageIndex artifact builder (V2.1 parts 10, 25-26).

Builds a persistent PageIndex tree artifact for every paper ONCE, offline:

    index/pageindex/{paper_id}.json

The existing XML-derived corpus structure (LogicalDocumentIndex) is converted
into structured Markdown, then the OFFICIAL PageIndex library builds the
hierarchical tree via pageindex.page_index_md.md_to_tree (VectifyAI/PageIndex).
A deterministic node -> chunk_id mapping is derived from the emitted markdown
order, so every PageIndex node resolves onto the existing XML corpus chunk
identifiers (part 24): PageIndex decides WHERE to look, the
LogicalDocumentIndex decides HOW to retrieve the exact evidence object.

Do NOT construct PageIndex trees at query time and do NOT store the trees in
the corpus parquet - this is an ADDITIONAL index (pageindex artifacts) built
only for new/changed papers on incremental updates.

Usage:
    python -m medrag.retrieval_v2.pageindex_build --paper PMC11743609
    python -m medrag.retrieval_v2.pageindex_build --papers PMC11743609 PMC12009809
    python -m medrag.retrieval_v2.pageindex_build --limit 50
    python -m medrag.retrieval_v2.pageindex_build --all --rebuild
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medrag.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from medrag.retrieval_v2.document_index import LogicalDocumentIndex, TABLE_TYPES

_TITLE_SLICE = 52


def _clean_title(text: str) -> str:
    text = " ".join((text or "").split())
    if len(text) > _TITLE_SLICE:
        text = text[:_TITLE_SLICE].rstrip() + "..."
    return text or "(untitled)"


def _paragraph_leaf_title(node: Dict[str, Any]) -> str:
    """Compact heading used for a paragraph/leaf node in the markdown."""
    prefix = node.get("text") or ""
    title = _clean_title(prefix)
    ntype = node.get("node_type", "paragraph")
    if ntype == "list":
        return "List: " + title
    if ntype == "equation":
        return "Equation: " + title
    return title or f"Paragraph {node.get('position', 0)}"


def paper_to_markdown(doc_index: Any, paper_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert one paper's XML-derived structure into PageIndex-compatible Markdown.

    Returns (markdown_text, ordered_leaf_records). Every corpus chunk of the
    paper maps to exactly one markdown leaf; tables are SINGLE nodes whose text
    covers summary + rows + footnotes and whose record carries ALL member chunk
    ids (part 9 tree: Tables -> Table N; part 15: the table is one semantic
    evidence object).
    """
    nodes = doc_index.get_paper_nodes(paper_id)
    if not nodes:
        return "", []

    lines: List[str] = ["# " + paper_id]
    # one leaf record per emitted markdown leaf; tables carry ALL member chunk
    # ids so the tree mirrors the logical table object (part 15 / 24).
    leaf_records: List[Dict[str, Any]] = []
    emitted_tables: set = set()

    # group text resolution once
    ids = [n["chunk_id"] for n in nodes]
    texts = doc_index.get_texts(ids)
    for n in nodes:
        cid = n["chunk_id"]
        n["text"] = texts.get(cid, "")

    cur_section = None
    cur_subsection = None

    def emit_section(node: Dict[str, Any]) -> None:
        nonlocal cur_section, cur_subsection
        section = node.get("section") or ""
        subsection = node.get("subsection") or ""
        if section != cur_section:
            cur_section = section
            cur_subsection = None
            lines.append("")
            lines.append("## " + (section or "General"))
        if subsection != cur_subsection:
            cur_subsection = subsection
            if subsection:
                lines.append("")
                lines.append("### " + subsection)

    for node in nodes:
        ntype = node.get("node_type", "paragraph")
        cid = node["chunk_id"]
        if ntype in TABLE_TYPES:
            # tables: ONE node covering summary + rows + footnotes. The table
            # members (summary / rows / footnotes) are emitted together so the
            # PageIndex tree mirrors the logical table object (part 15).
            table_id = node.get("table_id")
            if not table_id:
                continue
            if table_id in emitted_tables:
                continue
            emitted_tables.add(table_id)
            emit_section(node)
            tbl = doc_index.get_table(paper_id, table_id)
            members: List[Dict[str, Any]] = []
            if tbl["summary"]:
                members.append(tbl["summary"])
            members.extend(tbl["rows"])
            members.extend(tbl["footnotes"])
            member_texts = doc_index.get_texts([m["chunk_id"] for m in members])
            title = "Table " + str(table_id)
            lines.append("")
            lines.append("#### " + title)
            blocks: List[str] = []
            for m in members:
                t = member_texts.get(m["chunk_id"], "") or ""
                t = t.strip()
                if t:
                    blocks.append(t)
            lines.append("\n\n".join(blocks))
            leaf_records.append({"chunk_ids": [m["chunk_id"] for m in members]})
        elif ntype == "figure":
            emit_section(node)
            fig_id = node.get("figure_id") or cid
            lines.append("")
            lines.append("#### Figure " + str(fig_id))
            lines.append((node.get("text") or "").strip())
            leaf_records.append({"chunk_ids": [cid]})
        else:
            # paragraphs / lists / equations -> per-node leaves
            emit_section(node)
            if node.get("subsection"):
                lines.append("")
            title = _paragraph_leaf_title(node)
            lines.append("#### " + title)
            lines.append((node.get("text") or "").strip())
            leaf_records.append({"chunk_ids": [cid]})

    md = "\n".join(lines)
    return md, leaf_records


# ---------------------------------------------------------------------------
# tree helpers (mirror pageindex write_node_id DFS order for deterministic ids)
# ---------------------------------------------------------------------------


def _iter_tree_preorder(nodes: Sequence[Dict[str, Any]]) -> Any:
    for node in nodes:
        yield node
        children = node.get("nodes") or []
        yield from _iter_tree_preorder(children)


def _iter_leaves(nodes: Sequence[Dict[str, Any]]) -> Any:
    for node in nodes:
        children = node.get("nodes") or []
        if children:
            yield from _iter_leaves(children)
        else:
            yield node


def _breadcrumb(ancestors: Sequence[Dict[str, Any]]) -> str:
    return " > ".join([a.get("title", "") for a in ancestors])


def build_pageindex_artifact(
    doc_index: Any,
    paper_id: str,
    pageindex_dir: Path,
    config: Optional[V2Config] = None,
    pageindex_lib: Any = None,
) -> Dict[str, Any]:
    """Build (or rebuild) the PageIndex artifact for one paper; returns it."""
    cfg = config or DEFAULT_CONFIG
    pageindex_dir = Path(pageindex_dir)
    pageindex_dir.mkdir(parents=True, exist_ok=True)
    out_path = pageindex_dir / f"{paper_id}.json"

    md, leaf_records = paper_to_markdown(doc_index, paper_id)
    if not leaf_records:
        raise RuntimeError(f"paper {paper_id} has no searchable nodes")

    # ── build the tree with the OFFICIAL PageIndex library ──────────
    lib = pageindex_lib
    if lib is None:
        try:
            from pageindex.page_index_md import md_to_tree as _md_to_tree
            lib = _md_to_tree
        except Exception:  # noqa: BLE001
            lib = None
    if lib is None:
        raise RuntimeError("pageindex library is unavailable; cannot build artifacts")

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as fh:
        fh.write(md)
        md_path = fh.name
    try:
        tree = asyncio.run(lib(
            md_path=md_path,
            if_thinning=False,
            min_token_threshold=None,
            if_add_node_summary="no",
            if_add_node_text="yes",
            if_add_node_id="yes",
        ))
    finally:
        try:
            Path(md_path).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    structure = tree.get("structure") or []
    leaves = list(_iter_leaves(structure))
    if len(leaves) != len(leaf_records):
        raise RuntimeError(
            f"leaf mismatch for {paper_id}: tree leaves={len(leaves)} "
            f"vs emitted leaves={len(leaf_records)} (md_to_tree parse drift)"
        )

    # ── deterministic node -> chunk mapping (pre-order == markdown order) ──
    node_map: Dict[str, Dict[str, Any]] = {}
    chunk_to_node: Dict[str, str] = {}
    record_iter = iter(leaf_records)

    def assign(nodes: Sequence[Dict[str, Any]], ancestors: List[Dict[str, Any]]) -> None:
        for node in nodes:
            node_id = str(node.get("node_id", ""))
            children = node.get("nodes") or []
            node_map[node_id] = {
                "title": node.get("title", ""),
                "section": _breadcrumb(ancestors),
                "level": int(node.get("level", len(ancestors) + 1)) if node.get("level") is not None else len(ancestors) + 1,
                "chunk_ids": [],
                "page_refs": None,
            }
            if children:
                assign(children, ancestors + [node])
            else:
                rec = next(record_iter, {"chunk_ids": []})
                cids = list(rec.get("chunk_ids", []))
                node_map[node_id]["chunk_ids"] = cids
                for cid in cids:
                    chunk_to_node[cid] = node_id
    structure = tree.get("structure") or []
    assign(structure, [])
    # for non-leaf nodes, chunk_ids = descendant leaf chunk ids
    def fill_desc(nodes: Sequence[Dict[str, Any]]) -> None:
        for node in nodes:
            children = node.get("nodes") or []
            if children:
                fill_desc(children)
                desc_chunks: List[str] = []
                for ch in children:
                    nid = str(ch.get("node_id", ""))
                    rec = node_map.get(nid, {})
                    desc_chunks.extend(rec.get("chunk_ids", []))
                node_id = str(node.get("node_id", ""))
                node_map[node_id]["chunk_ids"] = list(dict.fromkeys(desc_chunks))
    fill_desc(structure)

    artifact: Dict[str, Any] = {
        "paper_id": paper_id,
        "metadata": {
            "source": "index/corpus.parquet (XML-derived logical structure)",
            "n_chunks": sum(len(rec["chunk_ids"]) for rec in leaf_records),
            "n_tree_nodes": len(node_map),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pageindex_version": getattr(pageindex_lib, "__module__", "pageindex") if pageindex_lib else "pageindex.md_to_tree",
            "format": "pageindex md_to_tree (hierarchy-first)",
        },
        "tree": structure,
        "node_map": node_map,
        "chunk_to_node": chunk_to_node,
    }
    out_path.write_text(json.dumps(artifact, indent=1, ensure_ascii=False), encoding="utf-8")
    return artifact


def load_doc_index(index_dir: Path) -> LogicalDocumentIndex:
    from medrag.retrieval.corpus import CORPUS_FILENAME
    index_dir = Path(index_dir)
    return LogicalDocumentIndex(index_dir / CORPUS_FILENAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline PageIndex artifact builder")
    parser.add_argument("--paper", default=None, help="single paper id")
    parser.add_argument("--papers", nargs="*", default=[], help="multiple paper ids")
    parser.add_argument("--limit", type=int, default=None, help="build first N papers")
    parser.add_argument("--all", action="store_true", help="build every paper")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if artifact exists")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--pageindex-dir", type=Path, default=None)
    parser.add_argument("--md-dir", type=Path, default=Path("index/pageindex_md"),
                        help="folder with jats-converted Markdown (pageindex_build per paper)")
    parser.add_argument("--from-md", action="store_true",
                        help="build the tree from the jats-converted Markdown folder "
                             "(XML -> jats -> MD -> official pageindex md_to_tree)")
    parser.add_argument("--from-chunks", action="store_true",
                        help="build the legacy chunk-derived tree (comparison only)")
    args = parser.parse_args(argv)

    cfg = DEFAULT_CONFIG
    pageindex_dir = Path(args.pageindex_dir) if args.pageindex_dir else Path(cfg.pageindex_dir)
    pageindex_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading logical document index from {args.index_dir} ...")
    doc_index = load_doc_index(args.index_dir)
    print(f"  {doc_index.n_chunks:,} chunks, {doc_index.n_papers:,} papers")

    if args.paper:
        paper_ids = [args.paper]
    elif args.papers:
        paper_ids = list(args.papers)
    elif args.all or args.limit is not None:
        paper_ids = doc_index.paper_ids()
        if args.limit is not None:
            paper_ids = paper_ids[: args.limit]
    else:
        parser.error("provide --paper, --papers, --limit, or --all")

    built, skipped, failed = 0, 0, 0
    t0 = time.perf_counter()
    for pid in paper_ids:
        try:
            if args.from_md:
                artifact = build_from_md(
                    pid, args.md_dir, pageindex_dir,
                    Path(args.index_dir) / "corpus.parquet", cfg)
                built += 1
                print(f"  built(md) {pid}  nodes={len(artifact['node_map'])}  mapped_chunks={artifact['metadata']['mapped_chunks']}/{artifact['metadata']['total_existing_chunks']}")
                continue
            if args.from_chunks:
                from medrag.retrieval_v2.document_index import LogicalDocumentIndex
                from medrag.retrieval.corpus import CORPUS_FILENAME
                doc_index2 = LogicalDocumentIndex(Path(args.index_dir) / CORPUS_FILENAME)
                artifact = build_pageindex_artifact(doc_index2, pid, pageindex_dir, cfg)
                built += 1
                print(f"  built(chunks) {pid}  nodes={len(artifact['node_map'])}")
                continue
            from medrag.retrieval_v2.pageindex_adapter import PageIndexAdapter
            adapter = PageIndexAdapter(doc_index, pageindex_dir=pageindex_dir, config=cfg)
            existed = adapter.build_or_load(pid, force=args.rebuild)
            if existed == "built":
                built += 1
                print(f"  built  {pid}")
            else:
                skipped += 1
                print(f"  exists {pid}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAILED {pid}: {exc}")
    dt = time.perf_counter() - t0
    print(f"Done: built={built} skipped={skipped} failed={failed} in {dt:.1f}s")
    print(f"Artifacts: {pageindex_dir}")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Build from the JATS->Markdown folder (V2.3): XML -> jats -> MD -> md_to_tree
# ---------------------------------------------------------------------------
def build_from_md(
    paper_id: str,
    md_dir: Path,
    pageindex_dir: Path,
    corpus_path: Path,
    config: Optional[V2Config] = None,
) -> Dict[str, Any]:
    """Build the PageIndex artifact from the jats-converted Markdown using the
    OFFICIAL PageIndex library (pageindex.page_index_md.md_to_tree).

    The Markdown was produced from the ORIGINAL ARTICLE XML by the installed
    jats package (see medrag.retrieval_v2.jats_convert), so the tree hierarchy
    comes from the XML structure; existing chunk ids are re-attached to leaf
    nodes by exact text matching (expectation: paragraphs match 1:1; table HTML
    blocks and rows are reported as md-unmapped in this mode).
    """
    import asyncio

    from medrag.retrieval_v2.xml_tree import load_paper_chunks, normalize_text

    cfg = config or DEFAULT_CONFIG
    md_path = Path(md_dir) / f"{paper_id}.md"
    if not md_path.exists():
        raise RuntimeError(f"no markdown for {paper_id} at {md_path} (run medrag.retrieval_v2.jats_convert first)")

    pageindex_dir = Path(pageindex_dir)
    pageindex_dir.mkdir(parents=True, exist_ok=True)

    from pageindex.page_index_md import md_to_tree as _md_to_tree
    tree = asyncio.run(_md_to_tree(
        md_path=str(md_path),
        if_thinning=False,
        min_token_threshold=None,
        if_add_node_summary="no",
        if_add_node_text="yes",
        if_add_node_id="yes",
    ))

    chunks = load_paper_chunks(paper_id, corpus_path)
    structure = tree.get("structure") or []

    node_map: Dict[str, Dict[str, Any]] = {}
    chunk_to_node: Dict[str, str] = {}
    counter = [0]
    used_chunks: set = set()

    def ident() -> str:
        i = counter[0]
        counter[0] += 1
        return f"{i:04d}"

    def leaf_text(node: Dict[str, Any]) -> str:
        text = node.get("text") or ""
        lines = text.split("\n")
        if lines and lines[0].startswith("#"):
            lines = lines[1:]
        return normalize_text("\n".join(lines))

    def walk(nodes, ancestors) -> None:
        for node in nodes:
            nid = node.get("node_id", "")
            title = node.get("title") or ""
            children = node.get("nodes") or []
            crumbs = [a.get("title") or "" for a in ancestors] + ([title] if title else [])
            is_leaf = not children
            low = title.lower()
            if low.startswith("table"):
                ntype = "table"
            elif low.startswith("figure"):
                ntype = "figure"
            elif is_leaf:
                ntype = "paragraph"
            elif len(ancestors) == 0:
                ntype = "document"
            elif len(ancestors) == 1:
                ntype = "section"
            else:
                ntype = "subsection"
            rec = {
                "node_id": nid,
                "node_type": ntype,
                "title": title,
                "section": " > ".join([paper_id] + crumbs),
                "breadcrumb": crumbs,
                "level": len(ancestors) + 1,
                "chunk_ids": [],
                "page_refs": None,
                "text": node.get("text") or "",
                "mapping_status": "md-unmapped",
            }
            node_map[nid] = rec
            if is_leaf:
                want = leaf_text(node)
                if want:
                    for cid in sorted(chunks, key=lambda c: chunks[c]["position"]):
                        if cid in used_chunks:
                            continue
                        if chunks[cid]["text"] == want:
                            rec["chunk_ids"] = [cid]
                            rec["mapping_status"] = "mapped-text"
                            used_chunks.add(cid)
                            chunk_to_node[cid] = nid
                            break
            walk(children, ancestors + [node])

    walk(structure, [])

    artifact = {
        "paper_id": paper_id,
        "metadata": {
            "source_md": str(md_path),
            "source_xml": "originated from ORIGINAL ARTICLE XML (jats converter: medrag.retrieval_v2.jats_convert)",
            "builder_version": "xml-md-tree-v2.3",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_existing_chunks": len(chunks),
            "mapped_chunks": len(chunk_to_node),
            "unmapped_chunks": len(chunks) - len(chunk_to_node),
            "format": "pageindex md_to_tree over jats-converted markdown",
        },
        "tree": structure,
        "node_map": node_map,
        "chunk_to_node": chunk_to_node,
        "mapping_log": [
            {"node_id": nid, "node_type": rec["node_type"], "chunk_ids": [],
             "status": "md-unmapped",
             "note": "leaf text did not exactly match any corpus chunk (tables/rows are HTML in MD mode)"}
            for nid, rec in node_map.items()
            if rec["node_type"] in ("table", "figure") and rec["mapping_status"] == "md-unmapped"
        ],
    }
    out_path = pageindex_dir / f"{paper_id}.json"
    out_path.write_text(json.dumps(artifact, indent=1, ensure_ascii=False), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    raise SystemExit(main())
