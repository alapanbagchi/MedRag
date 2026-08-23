"""Stage 2 - Markdown -> PageIndex tree indexing (V2.4).

The existing Markdown paper is fed DIRECTLY into the installed PageIndex
package's Markdown indexing API (pageindex.page_index_md.md_to_tree). That is
the heuristic tree-generation path: Markdown headings already contain the
hierarchy, so NO LLM is spent on tree generation (Step 2B).

The resulting native tree JSON is persisted separately from the Markdown
({tree_dir}/{paper_id}/tree.json) and registered in the PageIndex local
document store so the SDK's query-time navigation surfaces can read it.

Stage 1 (validation) utilities live here too (validate_markdown / scan_corpus).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medrag.retrieval_v2.pageindex_stage.config import StageConfig, config_from_env
from medrag.retrieval_v2.pageindex_stage.cache import (
    is_cached,
    load_tree,
    save_tree,
)
from medrag.retrieval_v2.pageindex_stage.models import NavigationError, PaperDocument


# ---------------------------------------------------------------------------
# Stage 1 - Markdown validation
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_TABLE_HTML_RE = re.compile(r"<table")
_PIPE_TABLE_RE = re.compile(r"^\s*\|")
_CAPTION_RE = re.compile(
    r"^(?:\*\*|####)\s*(Table|Figure|Supplementary Table|Supplementary Figure|Extended Data Figure)\b")
_FOOTNOTE_RE = re.compile(r"^_|^Footnotes?\s*:?|^\*Data are presented|^Table footnotes?")


def validate_markdown(paper_id: str, path: Path) -> PaperDocument:
    """Structural validation of one Markdown paper (Step 1A).

    Reports heading count/depth, table count, caption/footnote presence,
    heading hierarchy and MALFORMED headings (level jumps > 1) - validation
    only, never modifies the file.
    """
    path = Path(path)
    doc = PaperDocument(paper_id=paper_id, path=str(path))
    if not path.exists():
        return doc
    lines = path.read_text(encoding="utf-8").splitlines()
    doc.file_size = path.stat().st_size

    stack: List[Tuple[int, str]] = []          # (level, title)
    tables = 0
    pipe_run = False
    captions = 0
    footnotes = 0
    prev_level = 0

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        m = _HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            title = (m.group(2) or "").strip()
            doc.heading_count += 1
            doc.heading_depth = max(doc.heading_depth, level)
            # malformed: skipping heading levels (> prev+1 without an ancestor)
            if level > prev_level and level > prev_level + 1 and prev_level > 0:
                doc.malformed_headings += 1
                doc.suspicious.append(f"heading level jump {prev_level}->{level} near '{title[:40]}'")
            elif level > prev_level + 1 and prev_level == 0 and level > 2:
                doc.suspicious.append(f"unexpected top-level depth {level} near '{title[:40]}'")
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            doc.hierarchy.append([t for _l, t in stack])
            prev_level = level
            continue
        if _TABLE_HTML_RE.search(stripped):
            tables += 1
        if _PIPE_TABLE_RE.match(stripped) and "|" in stripped:
            pipe_run = True
        elif pipe_run:
            pipe_run = False
        if pipe_run and stripped.startswith("|") and stripped.count("|") >= 3:
            tables += 0  # markdown tables are contiguous ranges; count blocks below
        if _CAPTION_RE.match(stripped):
            captions += 1
        if _FOOTNOTE_RE.match(stripped):
            footnotes += 1

    # count contiguous markdown table blocks (a run of |-prefixed lines with >=2 pipes)
    md_tables = 0
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("|") and s.count("|") >= 3:
            md_tables += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                i += 1
        else:
            i += 1
    doc.table_count = tables + md_tables
    doc.caption_count = captions
    doc.footnote_count = footnotes
    return doc


def scan_corpus(config: Optional[StageConfig] = None) -> Tuple[List[PaperDocument], Dict[str, Any]]:
    """Validate the whole Markdown corpus directory (Step 1)."""
    cfg = config or config_from_env()
    md_dir = Path(cfg.md_dir)
    docs: List[PaperDocument] = []
    if not md_dir.exists():
        return docs, {"error": f"no markdown directory at {md_dir}"}
    for path in sorted(md_dir.glob("*.md")):
        paper_id = path.stem
        docs.append(validate_markdown(paper_id, path))
    summary = {
        "directory": str(md_dir),
        "n_markdown_files": len(docs),
        "with_tables": sum(1 for d in docs if d.table_count > 0),
        "with_captions": sum(1 for d in docs if d.caption_count > 0),
        "with_footnotes": sum(1 for d in docs if d.footnote_count > 0),
        "malformed_total": sum(d.malformed_headings for d in docs),
        "total_headings": sum(d.heading_count for d in docs),
    }
    return docs, summary


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------
def _count_nodes(nodes: Sequence[Dict[str, Any]]) -> int:
    n = 0
    for node in nodes:
        n += 1
        n += _count_nodes(node.get("nodes") or [])
    return n


def _leaves_and_paths(nodes: Sequence[Dict[str, Any]], path: List[str]) -> List[Tuple[str, str, List[str], int, str]]:
    """Flatten native tree into (node_id, title, breadcrumb, level, text_head)."""
    out: List[Tuple[str, str, List[str], int, str]] = []
    for node in nodes:
        title = node.get("title") or ""
        nid = str(node.get("node_id", ""))
        text = node.get("text") or ""
        crumbs = path + ([title] if title else [])
        children = node.get("nodes") or []
        if children:
            out.extend(_leaves_and_paths(children, crumbs))
        else:
            out.append((nid, title, crumbs, len(crumbs), text[:160]))
    return out


def _register_in_sdk_store(config: StageConfig, paper_id: str, structure: List[Dict[str, Any]],
                           description: str = "") -> Dict[str, Any]:
    """Persist the native tree into the installed PageIndex local document
    store so the SDK's query-time surfaces (chat_completions with doc_id,
    get_document / get_document_structure tools) can navigate it.

    NOTE: the SDK's local submit_document only accepts PDFs; for a Markdown
    paper we persist the md_to_tree output with the SDK's own DocStore class -
    the SDK-native storage format.
    """
    from pageindex.local_store import DocStore

    store = DocStore(str(config.sdk_storage))
    doc_id = paper_id if re.fullmatch(r"[A-Za-z0-9._-]+", paper_id) else "pi-" + paper_id
    meta = {
        "id": doc_id,
        "name": f"{paper_id}.md",
        "description": description or f"PageIndex tree for {paper_id} (md_to_tree)",
        "status": "completed",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pageNum": 0,
        "folderId": None,
        "metadata": {"source": "md_to_tree", "paper_id": paper_id},
        "mode": "flash-md",
    }
    store.save_document(doc_id, meta, structure, [])
    return {"doc_id": doc_id, "storage": str(config.sdk_storage)}


# ---------------------------------------------------------------------------
# Stage 2 - indexing
# ---------------------------------------------------------------------------
def build_index(paper_id: str, markdown_path: Optional[Path] = None,
                config: Optional[StageConfig] = None, rebuild: bool = False) -> Dict[str, Any]:
    """Feed the Markdown paper DIRECTLY into the installed PageIndex Markdown
    indexing API and persist the native tree (Step 2 / 2A / 2C / 2F).

    Returns:
        {"paper_id", "tree" (native dict), "structure", "native",
         "meta", "latency_ms", "node_count", "cached", "sdk"}
    Raises NavigationError (structured) on failure.
    """
    cfg = config or config_from_env()
    md_path = Path(markdown_path) if markdown_path is not None else Path(cfg.md_dir) / f"{paper_id}.md"
    if not md_path.exists():
        from medrag.retrieval_v2.pageindex_stage.models import TREE_LOAD_ERROR
        raise NavigationError(paper_id=paper_id, error_type=TREE_LOAD_ERROR,
                              message=f"markdown missing for {paper_id} at {md_path} (tree cannot be loaded)")

    if not rebuild and is_cached(cfg, paper_id):
        tree, meta = load_tree(cfg, paper_id) or (None, {})
        if tree is not None:
            return {
                "paper_id": paper_id,
                "tree": tree,
                "structure": tree.get("structure", []),
                "native": tree,
                "meta": meta,
                "latency_ms": 0.0,
                "node_count": _count_nodes(tree.get("structure", [])),
                "cached": True,
            }

    # heuristic Markdown indexing through the installed PageIndex package (no LLM)
    try:
        from pageindex.page_index_md import md_to_tree as _md_to_tree
    except Exception as exc:  # noqa: BLE001
        from medrag.retrieval_v2.pageindex_stage.models import PAGEINDEX_CLIENT_ERROR
        raise NavigationError(paper_id=paper_id, error_type=PAGEINDEX_CLIENT_ERROR,
                              message=f"pageindex package import failed: {exc}") from exc

    t0 = time.perf_counter()
    try:
        import asyncio
        native = asyncio.run(_md_to_tree(
            md_path=str(md_path),
            if_thinning=False,
            min_token_threshold=None,
            if_add_node_summary="no",
            if_add_node_text="yes",
            if_add_node_id="yes",
        ))
    except Exception as exc:  # noqa: BLE001
        from medrag.retrieval_v2.pageindex_stage.models import TREE_LOAD_ERROR
        raise NavigationError(paper_id=paper_id, error_type=TREE_LOAD_ERROR,
                              message=f"md_to_tree failed: {exc}") from exc
    latency_ms = (time.perf_counter() - t0) * 1000

    structure = native.get("structure") or []
    node_count = _count_nodes(structure)

    meta = save_tree(cfg, paper_id, native, extra_meta={
        "markdown_path": str(md_path),
        "indexing_ms": round(latency_ms, 1),
        "node_count": node_count,
    })

    # register in the SDK local store so query-time navigation can use it
    sdk: Dict[str, Any] = {}
    try:
        sdk = _register_in_sdk_store(cfg, paper_id, structure,
                                     description=f"nodes={node_count} source={md_path.name}")
    except Exception as exc:  # noqa: BLE001
        sdk = {"error": str(exc), "doc_id": paper_id}

    return {
        "paper_id": paper_id,
        "tree": native,
        "structure": structure,
        "native": native,
        "meta": meta,
        "latency_ms": round(latency_ms, 1),
        "node_count": node_count,
        "cached": False,
        "sdk": sdk,
    }


def load_index(paper_id: str, config: Optional[StageConfig] = None) -> Dict[str, Any]:
    """Load a previously indexed tree (error if absent - never rebuilds here)."""
    cfg = config or config_from_env()
    return build_index(paper_id, config=cfg, rebuild=False)


# ---------------------------------------------------------------------------
# Tree rendering
# ---------------------------------------------------------------------------
def print_tree(structure: Sequence[Dict[str, Any]], max_depth: int = 7,
               max_children: int = 64) -> None:
    """Print the native PageIndex tree structure (Step 2D)."""
    def walk(nodes, prefix: str, depth: int) -> None:
        for i, node in enumerate(nodes):
            last = i == len(nodes) - 1
            branch = "└── " if last else "├── "
            title = node.get("title") or node.get("node_id", "")
            nid = node.get("node_id", "")
            children = node.get("nodes") or []
            label = f"{title}  [{nid}]"
            if not children:
                chunk = ""
                text = (node.get("text") or "").strip().replace("\n", " ")
                if text and node.get("title"):
                    pass  # text-only leaves simply show the title
            print(("" if depth == 0 else prefix + branch) + label)
            ext = "    " if last else "│   "
            if children and depth < max_depth:
                walk(children, prefix + ext, depth + 1)
            elif children:
                print(prefix + ext + "…")
    walk(structure, "", 0)
