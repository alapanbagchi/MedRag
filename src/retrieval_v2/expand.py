"""Bounded structural expansion for V2 local search (spec sections 21-24).

The document structure index turns structural information into an active
retrieval tool: a retrieved table row immediately identifies its table, its
neighboring rows and footnotes; a retrieved paragraph identifies its
subsection context; a figure identifies its caption and surrounding text.

Expansion stays bounded:
    max_tokens_per_hit, max_expansions_per_hit, and an overall per-paper
    token budget are enforced so we never expand an entire paper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from src.retrieval_v2.document_index import TABLE_TYPES, tokens


def _estimate_tokens(node: Dict[str, Any]) -> int:
    text = node.get("text") or ""
    return tokens(text)


def _dedupe(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out = []
    for n in nodes:
        if n["chunk_id"] in seen:
            continue
        seen.add(n["chunk_id"])
        out.append(n)
    return out


def expand_anchor(
    doc_index: Any,
    paper_id: str,
    node: Dict[str, Any],
    config: Optional[V2Config] = None,
    texts: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Expand one anchor node into its structural neighborhood.

    Returns a deduplicated, text-enriched list of node records.
    """
    cfg = config or DEFAULT_CONFIG
    ntype = node.get("node_type", "paragraph")
    node_with_text = dict(node)
    node_with_text["text"] = (texts or {}).get(node["chunk_id"], "")
    out: List[Dict[str, Any]] = [node_with_text]

    # ── table row / footnote / summary -> assemble the whole table ──
    if ntype in TABLE_TYPES:
        table_id = node.get("table_id")
        if not table_id:
            # fall back to chunk-id prefix convention
            prefix = node.get("chunk_id", "")
            if "_" in prefix:
                import re as _re
                m = _re.match(r"^(.+)_(?:summary|row_\d+|footnotes)$", prefix)
                if m:
                    prefix = m.group(1)
            # strip paper prefix to recover the table id
            if prefix.startswith(paper_id + "_"):
                table_id = prefix[len(paper_id) + 1:]
        if table_id:
            tbl = doc_index.get_table(paper_id, table_id)
            members: List[Dict[str, Any]] = []
            if tbl["summary"]:
                members.append(tbl["summary"])
            members.extend(tbl["rows"])
            members.extend(tbl["footnotes"])
            member_ids = [m["chunk_id"] for m in members]
            t_texts = doc_index.get_texts(member_ids)
            for m in members:
                rec = dict(m)
                rec["text"] = t_texts.get(m["chunk_id"], "")
                rec["expanded_from"] = node["chunk_id"]
                out.append(rec)
        return _dedupe(out)

    # ── figure -> figure + surrounding text (section 22) ─────────
    if ntype == "figure":
        neighbors = doc_index.get_neighbors(paper_id, node["chunk_id"], before=1, after=1)
        nb_ids = [n["chunk_id"] for n in neighbors]
        nb_texts = doc_index.get_texts(nb_ids)
        for n in neighbors:
            if n["chunk_id"] == node["chunk_id"]:
                continue
            rec = dict(n)
            rec["text"] = nb_texts.get(n["chunk_id"], "")
            rec["expanded_from"] = node["chunk_id"]
            out.append(rec)
        return _dedupe(out)

    # ── paragraph -> same-subsection neighbors (section 24) ──────
    if ntype == "paragraph":
        win = cfg.neighbor_window
        neighbors = doc_index.get_neighbors(paper_id, node["chunk_id"], before=win, after=win)
        nb_ids = [n["chunk_id"] for n in neighbors]
        nb_texts = doc_index.get_texts(nb_ids)
        for n in neighbors:
            if n["chunk_id"] == node["chunk_id"]:
                continue
            rec = dict(n)
            rec["text"] = nb_texts.get(n["chunk_id"], "")
            rec["expanded_from"] = node["chunk_id"]
            out.append(rec)
        return _dedupe(out)

    # generic: just the node itself
    return _dedupe(out)


def expand_anchors(
    doc_index: Any,
    paper_id: str,
    anchors: Sequence[Dict[str, Any]],
    config: Optional[V2Config] = None,
) -> List[Dict[str, Any]]:
    """Expand a set of anchor hits with bounded budget.

    Budget enforcement:
        - at most cfg.max_expansions_per_hit nodes per anchor
        - at most cfg.max_tokens_per_hit characters-equivalent per anchor
        - the whole result is capped by cfg.max_total_tokens
    """
    cfg = config or DEFAULT_CONFIG
    anchor_ids = [a["chunk_id"] for a in anchors]
    texts = doc_index.get_texts(anchor_ids)
    all_nodes: List[Dict[str, Any]] = []
    total_tokens = 0
    for anchor in anchors:
        if total_tokens >= cfg.max_total_tokens:
            break
        expanded = expand_anchor(doc_index, paper_id, anchor, cfg, texts)
        budget = cfg.max_tokens_per_hit
        seen_anchor: set = set()
        for node in expanded:
            if len(seen_anchor) >= cfg.max_expansions_per_hit:
                break
            node_tok = _estimate_tokens(node)
            if budget - node_tok < 0 and seen_anchor:
                break
            if total_tokens + node_tok > cfg.max_total_tokens:
                break
            if node["chunk_id"] in seen_anchor:
                continue
            seen_anchor.add(node["chunk_id"])
            budget -= node_tok
            total_tokens += node_tok
            node["token_count"] = node_tok
            all_nodes.append(node)
    return _dedupe(all_nodes)

