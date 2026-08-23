"""Tree cache for persisted PageIndex trees (V2.4).

The PageIndex tree is built ONCE per paper and cached on disk:
    {tree_dir}/{paper_id}/tree.json      - the native md_to_tree JSON
    {tree_dir}/{paper_id}/meta.json      - paper_id / source md / timing / node count

Trees are never rebuilt on every query (only with rebuild=True).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from medrag.retrieval_v2.pageindex_stage.config import StageConfig


def tree_path(config: StageConfig, paper_id: str) -> Path:
    return Path(config.tree_dir) / paper_id / "tree.json"


def meta_path(config: StageConfig, paper_id: str) -> Path:
    return Path(config.tree_dir) / paper_id / "meta.json"


def is_cached(config: StageConfig, paper_id: str) -> bool:
    return tree_path(config, paper_id).exists() and meta_path(config, paper_id).exists()


def save_tree(config: StageConfig, paper_id: str, tree: Any,
              extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist the native PageIndex tree + metadata; returns the meta dict."""
    tpath = tree_path(config, paper_id)
    mpath = meta_path(config, paper_id)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tpath.write_text(json.dumps(tree, indent=1, ensure_ascii=False), encoding="utf-8")
    extra = extra_meta if isinstance(extra_meta, dict) else {}
    meta = {
        "paper_id": paper_id,
        "source": str(extra.get("markdown_path", "")),
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "indexing_ms": float(extra.get("indexing_ms", 0.0)),
        "node_count": int(extra.get("node_count", 0)),
        "tree_size_bytes": tpath.stat().st_size,
        "builder": "pageindex.page_index_md.md_to_tree (heuristic, no LLM)",
    }
    if isinstance(extra_meta, dict):
        meta.update({k: v for k, v in extra_meta.items() if k not in meta})
    mpath.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


def load_tree(config: StageConfig, paper_id: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Load (tree, meta) from cache; None when not cached."""
    if not is_cached(config, paper_id):
        return None
    tree = json.loads(tree_path(config, paper_id).read_text(encoding="utf-8"))
    meta = json.loads(meta_path(config, paper_id).read_text(encoding="utf-8"))
    return tree, meta
