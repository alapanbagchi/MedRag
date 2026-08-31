#!/usr/bin/env python
"""Embedding-FREE evaluation gate for chunker v2 (chunking A/B + fidelity).

Deliberately requires NO encoder: retrieval is pure BM25 (the SAME sparse
implementation the hybrid pipeline uses, ``src.retrieval.sparse.BM25Index``),
so chunking choices are compared on their lexical retrievability and on
fidelity metrics that need no vectors:

  * recall@k / MRR across chunking variants (paragraph vs window strategy,
    ``max_tokens``, global dedup on/off),
  * fidelity: chunk-type mix, dedup suppression rate, citation-resolution
    rate, embedding-budget violations, truncation markers, unit coverage,
    determinism, empty-chunk guard.

Queries are SELF-SUPERVISED (the same pattern the old 50-doc benchmark used):
the corpus itself provides gold chunks; a query is a truncated window of a
gold chunk's text. A hit means the gold TEXT appears inside a retrieved
chunk (containment, not id-equality), which keeps the comparison fair when a
variant merges/splits/suppresses chunks (window packing, paragraph merging,
dedup) — the gold paragraph still lives inside whatever chunk it landed in.

This is Tier-1 of a two-tier gate: cheap, CPU-only, run on every chunking
change. Tier-2 (dense/hybrid recall + end-to-end answer faithfulness) needs
embeddings and is a separate, slower step.

Usage:
    python scripts/benchmark_chunking.py --xml-dir data/chunked --sample 300 \
        --variants paragraph-320,window-320,paragraph-160,paragraph-320-dedup \
        --report eval/reports/chunking_benchmark.md
    # FULL CORPUS (all documents, no sample):
    python scripts/benchmark_chunking.py --md-dir data/md --sample 0 \
        --variants paragraph-320,paragraph-320-dedup --queries-per-doc 1 \
        --max-queries 4000 --workers 16 --report eval/reports/chunking_benchmark_full.md
    python scripts/benchmark_chunking.py --md-dir data/md --sample 200 \
        --workers 8 --k 5,10,20
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm  # noqa: E402

from src.retrieval.sparse import BM25Index  # noqa: E402
from src.chunking.md_chunker import MDChunker, _chunks_to_df, _global_dedup_pass  # noqa: E402

STOP = frozenset({
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "with", "on",
    "as", "by", "at", "is", "are", "was", "were", "be", "been", "we", "our",
    "it", "this", "that", "these", "those", "from", "between", "among",
})


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", (text or ""))
            if w.lower() not in STOP and len(w) > 1]


def _spec(s: str) -> Dict[str, Any]:
    """'window-320-dedup' -> {strategy, max_tokens, dedup}."""
    parts = s.split("-")
    dedup = "dedup" in parts
    tokens = 320
    for p in parts:
        if p.isdigit():
            tokens = int(p)
    strategy = "window" if "window" in parts else "paragraph"
    return {"prose_strategy": strategy, "max_tokens": tokens, "global_dedup": dedup}


# ---------------------------------------------------------------------------
# Corpus materialization
# ---------------------------------------------------------------------------

def _materialize(xml_dir: Path, md_dir: Path, sample: int, workers: int) -> List[Path]:
    """Convert a deterministic sample of JATS files to Markdown (if needed)."""
    sys.path.insert(0, str(ROOT))
    from src.processing.jats_to_md import convert_one

    # --md-dir is the PRIMARY input: use its Markdown whenever present
    # (sample==0 = ALL of it). Only fall back to converting from --xml-dir
    # when the Markdown dir is empty/missing.
    mds = sorted(md_dir.rglob("*.md"))
    if mds:
        return mds if sample <= 0 else mds[:sample]
    xmls = sorted(xml_dir.rglob("*.xml"))
    if sample:
        xmls = xmls[:sample]
    md_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    from concurrent.futures import ThreadPoolExecutor

    def run_one(x: str) -> Optional[Path]:
        md = md_dir / (Path(x).stem + ".md")
        res = convert_one(Path(x), md, overwrite=True, verbose=False)
        return md if res["status"] == "converted" else None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for md in tqdm(pool.map(run_one, xmls), total=len(xmls),
                       desc="Converting sample", unit="xml"):
            if md is not None:
                out.append(md)
    return sorted(out)


# ---------------------------------------------------------------------------
# Per-variant chunking (parquet out, like the real CLI)
# ---------------------------------------------------------------------------

def _chunk_variant(mds: List[Path], spec: Dict[str, Any], chunks_out: Path,
                   workers: int) -> List[Dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor

    chunks_out.mkdir(parents=True, exist_ok=True)

    def run_one(md: Path) -> None:
        text = md.read_text(encoding="utf-8")
        chunker = MDChunker(max_tokens=spec["max_tokens"],
                            prose_strategy=spec["prose_strategy"])
        chunks, units, report = chunker.chunk_md(text, doc_id=md.stem)
        if chunks:
            _chunks_to_df(chunks).to_parquet(chunks_out / f"{md.stem}.parquet",
                                             index=False)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(tqdm(pool.map(run_one, mds), total=len(mds),
                  desc=f"Chunking {spec['prose_strategy']}-{spec['max_tokens']}"
                       f"({'dedup' if spec['global_dedup'] else ''})",
                  unit="doc"))

    if spec["global_dedup"]:
        _global_dedup_pass(chunks_out)

    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for pq in sorted(chunks_out.glob("*.parquet")):
        df = pd.read_parquet(pq)
        cols = {c: df[c] for c in df.columns}
        serie_id = cols.get("id"); serie_doc = cols.get("document_id")
        serie_txt = cols.get("text"); serie_ct = cols.get("chunk_type")
        serie_el = cols.get("retrieval_eligible"); serie_md = cols.get("metadata")
        serie_ci = cols.get("citation_refs"); serie_tk = cols.get("embedding_token_count")
        serie_pos = cols.get("document_position")
        n = len(df)
        for i in range(n):
            raw_meta = serie_md.iloc[i]
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else {}
            raw_cite = serie_ci.iloc[i]
            rows.append({
                "id": str(serie_id.iloc[i]), "document_id": str(serie_doc.iloc[i]),
                "text": str(serie_txt.iloc[i]), "chunk_type": str(serie_ct.iloc[i]),
                "retrieval_eligible": bool(serie_el.iloc[i]),
                "metadata": meta,
                "citation_refs": json.loads(raw_cite) if isinstance(raw_cite, str) and raw_cite else [],
                "embedding_token_count": int(serie_tk.iloc[i] or 0),
                "position": int(serie_pos.iloc[i]),
            })
    return rows


# ---------------------------------------------------------------------------
# Self-supervised queries (built once from the baseline variant)
# ---------------------------------------------------------------------------

def _build_queries(rows: List[Dict[str, Any]], queries_per_doc: int,
                   seed: int, max_queries: int) -> List[Dict[str, Any]]:
    import random

    rng = random.Random(seed)
    eligible_pool = [r for r in rows if r["retrieval_eligible"]
                     and r["chunk_type"] in ("paragraph", "list", "table_row",
                                             "table_summary", "figure", "equation")]
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in eligible_pool:
        by_doc[r["document_id"]].append(r)

    queries: List[Dict[str, Any]] = []
    doc_order = sorted(by_doc)
    rng.shuffle(doc_order)   # corpus-wide spread, deterministic via seed
    for doc_id in doc_order:
        pool = sorted(by_doc[doc_id], key=lambda r: r["id"])
        picked = rng.sample(pool, min(queries_per_doc, len(pool)))
        for r in picked:
            words = _words(r["text"])
            if len(words) < 6:
                continue
            gold_text = " ".join((r["text"] or "").split())
            n = len(words)
            queries.append({
                "gold_doc": r["document_id"],
                "gold_text": gold_text,
                "q_verbatim": " ".join(words[:min(40, n)]),
                "q_needle": " ".join(words[n // 2:min(n, n // 2 + 18)]),
            })
            if len(queries) >= max_queries:
                return queries
    return queries


def _contained(text: str, gold_text: str) -> bool:
    return gold_text in " ".join((text or "").split())


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def _bm25_search(rows: List[Dict[str, Any]], queries: List[str],
                 k_vals: List[int], tmp_dir: Path
                 ) -> Tuple[Dict[str, Any], Dict[int, List[List[str]]]]:
    """Build BM25 over eligible chunks; return (meta, {k: [[chunk_id...], ...]})."""
    eligible = [r for r in rows if r["retrieval_eligible"]]

    def factory():
        for r in eligible:
            yield r["id"], r["text"]

    t0 = time.time()
    index = BM25Index.build(factory, tmp_dir, k1=1.5, b=0.75)
    build_s = time.time() - t0
    k_max = max(k_vals)
    results: Dict[int, List[List[str]]] = {}
    t1 = time.time()
    ranked = index.search(queries, top_k=k_max)
    search_s = time.time() - t1
    for k in k_vals:
        results[k] = [[hit.chunk_id for hit in q[:k]] for q in ranked]
    meta = {"n_docs": index.n_docs, "vocab_size": index.vocab_size,
            "avgdl": round(index.avgdl, 1), "build_s": round(build_s, 1),
            "search_s": round(search_s, 1)}
    return meta, results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _fidelity(rows: List[Dict[str, Any]], mds: List[Path]) -> Dict[str, Any]:
    by_type: Dict[str, int] = Counter()
    n_eligible = n_reference = n_admin = n_dedup = 0
    cite_harvest = cite_resolved = 0
    budget_viol = trunc = unit_orphan = 0
    empty_guard = 0
    total_tokens = 0
    n_eligible_count = 0
    n_unit = 0
    for r in rows:
        by_type[r["chunk_type"]] += 1
        if r["chunk_type"] in ("paragraph", "list") and r["retrieval_eligible"]:
            total_tokens += r["embedding_token_count"]
            n_eligible_count += 1
        m = r["metadata"]
        if r["retrieval_eligible"]:
            n_eligible += 1
        if r["chunk_type"] == "reference":
            n_reference += 1
        if r["chunk_type"] == "administrative":
            n_admin += 1
        if m.get("dedup_of") and not r["retrieval_eligible"]:
            n_dedup += 1
        if r["chunk_type"] in ("paragraph", "list"):
            if m.get("citation_numbers"):
                cite_harvest += 1
            if r.get("citation_refs"):
                cite_resolved += 1
        if m.get("embedding_tokens_within_budget") is False:
            budget_viol += 1
        if "[...]" in r["text"] or "[...]" in str(m):
            trunc += 1
        if m.get("unit_id"):
            n_unit += 1
        else:
            unit_orphan += 1
        if r["text"] == r["id"]:
            empty_guard += 1
    # determinism: rechunk 8 docs, compare id sequences
    det_ok = True
    det_checked = 0
    ch = MDChunker()
    for md in mds[:8]:
        a = [c.id for c in ch.chunk_md(md.read_text(encoding="utf-8"),
                                       doc_id=md.stem)[0]]
        b = [c.id for c in ch.chunk_md(md.read_text(encoding="utf-8"),
                                       doc_id=md.stem)[0]]
        det_checked += 1
        if a != b:
            det_ok = False
            break
    return {
        "n_docs": len(set(r["document_id"] for r in rows)),
        "n_chunks": len(rows),
        "n_eligible": n_eligible,
        "avg_tokens": round(total_tokens / max(1, n_eligible_count), 1),
        "by_type": dict(sorted(by_type.items())),
        "dedup_suppressed": n_dedup,
        "citation_harvest_pct": round(100 * cite_harvest / max(1, by_type["paragraph"] + by_type["list"]), 1),
        "citation_resolved_pct": round(100 * cite_resolved / max(1, cite_harvest), 1),
        "budget_violations": budget_viol,
        "truncation_markers": trunc,
        "unit_orphans": unit_orphan,
        "empty_guard": empty_guard,
        "deterministic": det_ok and det_checked > 0,
    }


def _mrr(rank_of_gold: List[int]) -> float:
    hits = [1.0 / r for r in rank_of_gold if r > 0]
    return round(sum(hits) / max(1, len(rank_of_gold)), 4)


def _recall(ranks: List[int], k: int) -> float:
    return round(sum(1 for r in ranks if 0 < r <= k) / max(1, len(ranks)), 4)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _report(config: Dict[str, Any], qs: List[Dict[str, Any]],
            results: Dict[str, Dict[str, Any]], baseline: str) -> str:
    L: List[str] = []
    L.append("# Chunker v2 — Embedding-Free Evaluation Gate (Tier-1)\n")
    L.append(f"- generated: `{time.strftime('%Y-%m-%d %H:%M')}` | seed: `{config['seed']}`")
    L.append(f"- documents: `{config['n_docs']}` | queries: `{len(qs)}` "
             f"({config['queries_per_doc']}/doc, verbatim + mid-paragraph needle)")
    L.append(f"- retrieval: **BM25** (k1=1.5, b=0.75) — `src.retrieval.sparse`, "
             f"no embeddings involved")
    L.append(f"- gold = source paragraph text; a hit = gold text contained in a "
             f"retrieved chunk (fair across merge/split/dedup)")
    L.append(f"- variants: `{'` vs `'.join(config['variants'])}` "
             f"(baseline: `{baseline}`)\n")

    L.append("## 1. Fidelity by variant\n")
    L.append("| metric | " + " | ".join(config["variants"]) + " |")
    L.append("|---" * (1 + len(config["variants"])) + "|")
    fidelity_rows = [
        ("chunks", lambda v: f"{v['fidelity']['n_chunks']:,}"),
        ("eligible", lambda v: f"{v['fidelity']['n_eligible']:,}"),
        ("avg embedding tokens", lambda v: str(v["fidelity"]["avg_tokens"])),
        ("dedup suppressed", lambda v: str(v["fidelity"]["dedup_suppressed"])),
        ("citation harvested %", lambda v: f"{v['fidelity']['citation_harvest_pct']}%"),
        ("citation resolved %", lambda v: f"{v['fidelity']['citation_resolved_pct']}%"),
        ("embed-budget violations", lambda v: str(v["fidelity"]["budget_violations"])),
        ("truncation markers", lambda v: str(v["fidelity"]["truncation_markers"])),
        ("unit orphans", lambda v: str(v["fidelity"]["unit_orphans"])),
        ("empty-chunk guard hits", lambda v: str(v["fidelity"]["empty_guard"])),
        ("deterministic", lambda v: "yes" if v["fidelity"]["deterministic"] else "NO"),
    ]
    for name, fn in fidelity_rows:
        L.append(f"| {name} | " + " | ".join(fn(v) for v in results.values()) + " |")
    L.append("\nChunk-type mix per variant:\n")
    L.append("| type | " + " | ".join(config["variants"]) + " |")
    L.append("|---" * (1 + len(config["variants"])) + "|")
    for t in sorted({t for v in results.values() for t in v["fidelity"]["by_type"]}):
        L.append(f"| {t} | " + " | ".join(
            str(v["fidelity"]["by_type"].get(t, 0)) for v in results.values()) + " |")

    L.append("\n## 2. Retrieval — recall@k (chunk-level, gold contained)\n")
    for qtype in ("verbatim", "needle"):
        L.append(f"\n### queries: {qtype}\n")
        L.append("| k | " + " | ".join(config["variants"]) + " |")
        L.append("|---" * (1 + len(config["variants"])) + "|")
        for k in config["k_vals"]:
            L.append(f"| {k} | " + " | ".join(
                _fmt_pct(v["recall"][qtype][k]) for v in results.values()) + " |")
        L.append("\nMRR (mean reciprocal rank of the gold chunk):\n")
        L.append("| variant | MRR |")
        L.append("|---|---|")
        for name in config["variants"]:
            L.append(f"| {name} | {results[name]['mrr'][qtype]} |")

    L.append("\n## 3. Retrieval — document-level recall@k\n")
    L.append("| k | " + " | ".join(config["variants"]) + " |")
    L.append("|---" * (1 + len(config["variants"])) + "|")
    for k in config["k_vals"]:
        L.append(f"| {k} | " + " | ".join(
            _fmt_pct(v["doc_recall"][k]) for v in results.values()) + " |")

    L.append("\n## 4. Summary (relative to baseline `" + baseline + "`, recall@10)\n")
    L.append("| variant | chunk recall@10 Δ | doc recall@10 Δ | fidelity notes |")
    L.append("|---|---|---|---|")
    for name in config["variants"]:
        v = results[name]
        dr = v["recall"]["verbatim"][10] - results[baseline]["recall"]["verbatim"][10]
        dd = v["doc_recall"][10] - results[baseline]["doc_recall"][10]
        notes = []
        if v["fidelity"]["dedup_suppressed"]:
            notes.append(f"{v['fidelity']['dedup_suppressed']:,} suppressed")
        if v["fidelity"]["budget_violations"]:
            notes.append(f"{v['fidelity']['budget_violations']} budget violations")
        L.append(f"| {name} | {dr:+.1%} | {dd:+.1%} | {', '.join(notes) or '—'} |")

    L.append("\n## 5. Failure examples (gold NOT found @k=10, verbatim queries)\n")
    v = results[baseline]
    shown = 0
    for i, q in enumerate(qs):
        if v["rank"][i] == 0 or v["rank"][i] > 10:
            top1 = v["top1_text"][i]
            L.append(f"- q: `{q['q_verbatim'][:70]}…`  gold: `{q['gold_doc']}`")
            L.append(f"  → top-1: `{top1[:90]}…`")
            shown += 1
            if shown >= 8:
                break
    if shown == 0:
        L.append("- (none at k=10 for the baseline)")

    L.append("\n## 6. Interpretation & caveats\n")
    L.append("- Tier-1 is **lexical (BM25)** — it measures retrievability of evidence "
             "under chunking changes, not semantic quality. A variant that wins here "
             "may still lose with dense/hybrid (Tier-2) and vice-versa; run Tier-2 "
             "before locking a config.")
    L.append("- Queries are self-supervised (chunk-as-gold). The verbatim query is an "
             "upper bound; the mid-paragraph needle measures fragmentation impact.")
    L.append("- Gold = text containment (not id), so merging (window), splitting "
             "(max_tokens) and dedup do not bias the comparison.")
    L.append("- Fidelity columns are regression walls: budget violations / truncation "
             "markers / unit orphans should stay ~0; dedup suppression is a policy "
             "choice you must eyeball.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", type=Path, default=Path("data/chunked"),
                        help="JATS XML corpus (converted to MD for the sample).")
    parser.add_argument("--md-dir", type=Path, default=Path("data/md"),
                        help="Existing Markdown dir (bypasses conversion).")
    parser.add_argument("--sample", type=int, default=300,
                        help="0 = ALL documents (full corpus).")
    parser.add_argument("--variants", default="paragraph-320,window-320,"
                                              "paragraph-160,paragraph-320-dedup")
    parser.add_argument("--queries-per-doc", type=int, default=2)
    parser.add_argument("--max-queries", type=int, default=800)
    parser.add_argument("--k", default="5,10,20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", type=Path,
                        default=Path("eval/reports/chunking_benchmark.md"))
    parser.add_argument("--tmp", type=Path, default=Path(".bench_tmp"))
    args = parser.parse_args(argv)

    k_vals = [int(k) for k in args.k.split(",") if k]
    variants = [s.strip() for s in args.variants.split(",") if s.strip()]
    specs = {s: _spec(s) for s in variants}

    mds = _materialize(args.xml_dir, args.md_dir, args.sample, args.workers)
    if not mds:
        print("No Markdown available; set --xml-dir with the corpus or --md-dir.",
              file=sys.stderr)
        return 1

    print(f"MD directory: {args.md_dir.resolve()}", flush=True)
    print(f"Found .md files: {len(list(args.md_dir.rglob('*.md'))):,}", flush=True)
    print(f"Documents selected for benchmark: {len(mds):,}", flush=True)

    tmp_root = args.tmp
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # 1) chunk each variant to parquet + collect rows
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    for s in variants:
        chunk_dir = tmp_root / s / "chunks"
        all_rows[s] = _chunk_variant(mds, specs[s], chunk_dir, args.workers)

    # 2) build queries once from the BASELINE variant (first on the list)
    baseline = variants[0]
    qs = _build_queries(all_rows[baseline], args.queries_per_doc, args.seed,
                        args.max_queries)
    if not qs:
        print("No queries could be built (sample too small / no eligible chunks).",
              file=sys.stderr)
        return 1
    print(f"Queries: {len(qs)}", flush=True)

    # 3) per-variant BM25 + metrics
    q_verb = [q["q_verbatim"] for q in qs]
    q_need = [q["q_needle"] for q in qs]
    results: Dict[str, Dict[str, Any]] = {}
    for s in variants:
        rows = all_rows[s]
        row_by_id = {r["id"]: r for r in rows}
        v = {"fidelity": _fidelity(rows, mds), "rank": [], "mrr": {},
             "recall": {}, "doc_recall": {}, "top1_text": []}
        for qtype, queries in (("verbatim", q_verb), ("needle", q_need)):
            bdir = tmp_root / s / "bm25"
            meta, res = _bm25_search(rows, queries, k_vals, bdir)
            v["_bm25" + qtype] = meta
            ranks: List[int] = []
            doc_ranks: List[int] = []
            for i, q in enumerate(qs):
                found_rank = 0
                doc_found = 0
                for rk, cid in enumerate(res[k_vals[-1]][i], start=1):
                    row = row_by_id.get(cid)
                    if row is None:
                        continue
                    if _contained(row["text"], q["gold_text"]) and found_rank == 0:
                        found_rank = rk
                    if row["document_id"] == q["gold_doc"] and doc_found == 0:
                        doc_found = rk
                    if found_rank and doc_found:
                        break
                ranks.append(found_rank)
                doc_ranks.append(doc_found)
                if qtype == "verbatim":
                    v["top1_text"].append(res[k_vals[0]][i][0] if res[k_vals[0]][i]
                                          else "")
            v["rank"] = ranks if qtype == "verbatim" else v["rank"]
            v["recall"][qtype] = {k: _recall(ranks, k) for k in k_vals}
            v["doc_recall"] = {k: _recall(doc_ranks, k) for k in k_vals}
            v["mrr"][qtype] = _mrr(ranks)
        results[s] = v
        print(f"[{s}] recall@10 verbatim: "
              f"{_fmt_pct(v['recall']['verbatim'][10])} | doc: "
              f"{_fmt_pct(v['doc_recall'][10])} | chunks: {v['fidelity']['n_chunks']:,}",
              flush=True)

    # 4) report
    config = {
        "seed": args.seed, "n_docs": len(mds), "queries_per_doc": args.queries_per_doc,
        "variants": variants, "k_vals": k_vals,
    }
    report = _report(config, qs, results, baseline)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"\nReport -> {args.report}", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
