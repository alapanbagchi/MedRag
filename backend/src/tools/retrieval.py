"""Hybrid evidence retrieval tool: BM25 + MedCPT dense -> RRF -> MedCPT cross-encoder.

Pipeline (all in one call):
  1. BM25 leg takes top 60, dense leg takes top 60 (MedCPT query encoder + pgvector).
  2. Reciprocal-rank fusion merges the two rankings, top 60 survive.
  3. MedCPT cross-encoder reranks those 60 in one batch; top 10 go to the LLM.
  4. Each survivor is expanded to its closest boundary (whole paragraph, whole
     table with caption) and returned as JSON for the agent to quote from.

Heavy models (query encoder, cross-encoder, BM25 files) load once per process
via the module singleton; tests swap it with a stub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from collections.abc import Callable
from pydantic import BaseModel
from pydantic_ai import RunContext

from src.lib import narrate
from src.lib.pretty import style
from src.lib.trace import get_trace

logger = logging.getLogger(__name__)
from src.retrieval.pgvector_store import PgConfig, PgVectorStore, current_schema
from src.retrieval.query import MedCPTQueryEncoder
from src.retrieval.pg_search import PgSearchBM25
from src.retrieval.sparse import BM25Index
from src.tools.umls import DeepDeps

BM25_TOPK = 60
DENSE_TOPK = 60
# Rerank the full fused set (60) into the top 10. Keep 60: measured CPU
# rerank cost is worth the recall, so this is deliberately NOT env-tunable.
CROSS_RERANK_TOPK = 60
FINAL_TOPK = 10
RRF_K = 60
PROSE_TYPES = ("paragraph", "list")

# Exact-repeat retrieval cache (normalized query + leg/top-k). The pipeline
# re-runs the encoder, both DB legs and the CPU cross-encoder (~7 s), so a
# leg that retries a query verbatim gets the same passages without paying it
# again. Bounded; RETRIEVAL_CACHE=0 disables.
try:
    SEARCH_CACHE_MAX = max(0, int(os.environ.get("RETRIEVAL_CACHE_SIZE", "32")))
except (TypeError, ValueError):
    SEARCH_CACHE_MAX = 32


class EvidenceHit(BaseModel):
    chunk_id: str
    document_id: str
    chunk_type: str
    section: str
    score: float
    text: str
    title: str = ""


# 1. Rank fusion (pure): reciprocal-rank scores over each leg's rank order.
def rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: -scores[cid])


# 2. Cross-encoder reranker (lazy): ncbi/MedCPT-Cross-Encoder scores query/passage pairs.
class MedCPTCrossEncoder:
    # Small default batch: 60 x 512-token pairs in one forward pass OOMs
    # small GPUs (observed on a 3.7 GiB card); override via env when VRAM allows.
    def __init__(self, model_name: str | None = None, batch_size: int | None = None) -> None:
        self.model_name = model_name or os.environ.get(
            "CROSS_ENCODER_MODEL", "ncbi/MedCPT-Cross-Encoder"
        )
        if batch_size is None:
            try:
                batch_size = max(1, int(os.environ.get("CROSS_ENCODER_BATCH_SIZE", "16")))
            except (TypeError, ValueError):
                batch_size = 16
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from src.lib._torch import resolve_device

        self._device = resolve_device(None)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()
        self._torch = torch

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        self._load()
        batch = max(1, self.batch_size)
        while True:
            try:
                return self._score_batched(pairs, batch)
            except Exception as exc:
                if not _is_oom(exc) or batch <= 1:
                    raise
                batch //= 2
                _free_cuda()
                narrate.say(f"[local_search] cross-encoder OOM at batch"
                            f" — retrying with batch size {max(1, batch)}…")

    def _score_batched(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        out: list[float] = []
        # inference_mode skips autograd bookkeeping outright; the reranker is
        # inference-only, so there is nothing to track.
        with self._torch.inference_mode():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                enc = self._tokenizer(
                    [q for q, _ in batch],
                    [d for _, d in batch],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt",
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(**enc).logits
                probs = self._torch.sigmoid(logits[:, 0])
                out.extend(float(p) for p in probs.cpu().tolist())
        return out


def _is_oom(exc: Exception) -> bool:
    return (type(exc).__name__ == "OutOfMemoryError"
            or "out of memory" in str(exc).lower())


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# 3. Orchestrator: legs -> fusion -> rerank -> boundary expansion.
class EvidenceRetriever:
    def __init__(self, *, store=None, encoder=None, bm25=None, reranker=None) -> None:
        self._store = store
        self._encoder = encoder
        self._bm25 = bm25
        self._reranker = reranker
        self._search_cache: "OrderedDict[tuple, list[EvidenceHit]]" = OrderedDict()
        # Serializes heavy builds (torch/CUDA loads) so a background warmup
        # and a retrieval can never load the same weights twice at once.
        self._build_lock = threading.Lock()

    def _store_or_build(self, progress: Callable[[str], None] | None = None):
        if self._store is None:
            with self._build_lock:
                if self._store is None:
                    if progress is not None:
                        progress("connecting pgvector…")
                    store = PgVectorStore(PgConfig.from_env())
                    store.connect()
                    self._store = store
        return self._store

    def _encoder_or_build(self, progress: Callable[[str], None] | None = None):
        if self._encoder is None:
            with self._build_lock:
                if self._encoder is None:
                    if progress is not None:
                        progress("loading MedCPT query encoder (first run downloads weights)…")
                    model = os.environ.get("EMBEDDING_MODEL", "ncbi/MedCPT-Query-Encoder")
                    self._encoder = MedCPTQueryEncoder(model)
        return self._encoder

    def _bm25_or_build(self, progress: Callable[[str], None] | None = None):
        if self._bm25 is None:
            with self._build_lock:
                if self._bm25 is None:
                    backend = os.environ.get("BM25_BACKEND", "pgsearch")
                    if backend == "pgsearch":
                        try:
                            if progress is not None:
                                progress("connecting ParadeDB pg_search (BM25)…")
                            leg = PgSearchBM25()
                            leg.ping()
                            self._bm25 = leg
                        except Exception as exc:  # noqa: BLE001
                            # ParadeDB unreachable: fall back to the on-disk
                            # Rank-BM25 index so old environments keep working.
                            if progress is not None:
                                progress(f"pg_search unavailable ({exc}); "
                                         f"falling back to on-disk BM25")
                            logger.warning("pg_search BM25 fallback: %s", exc)
                            self._bm25 = BM25Index.load(
                                Path(os.environ.get("BM25_INDEX_DIR", "index/bm25_v2")))
                    else:
                        if progress is not None:
                            progress("loading BM25 index…")
                        index_dir = Path(os.environ.get("BM25_INDEX_DIR", "index/bm25_v2"))
                        self._bm25 = BM25Index.load(index_dir)
        return self._bm25

    def _reranker_or_build(self, progress: Callable[[str], None] | None = None):
        if self._reranker is None:
            with self._build_lock:
                if self._reranker is None:
                    if progress is not None:
                        progress("loading MedCPT cross-encoder (first run downloads weights)…")
                    self._reranker = MedCPTCrossEncoder()
        return self._reranker

    def warmup(self, progress: Callable[[str], None] | None = None) -> bool:
        """Load all heavy weights up front (idempotent, never raises).

        Call once per query run in the background while the agent plans, so
        the first retrieval finds hot weights instead of stalling on torch
        downloads/loads. Returns True when everything is resident.
        """
        ok = True
        for name, build in (("query encoder", self._encoder_or_build),
                            ("pgvector store", self._store_or_build),
                            ("BM25 index", self._bm25_or_build),
                            ("cross-encoder", self._reranker_or_build)):
            try:
                build(progress)
            except Exception as exc:  # noqa: BLE001 — retrieval reports for real later
                ok = False
                if progress is not None:
                    progress(f"warmup: {name} not ready ({exc})")
                logger.exception("retriever warmup failed for %s", name)
        # The wrappers are lazy: MedCPTQueryEncoder / MedCPTCrossEncoder only
        # load their torch weights on the first encode / score. Force both here
        # so a successful warmup really has the models resident (and their files
        # downloaded/cached at boot) instead of deferring it to the first query.
        if self._encoder is not None:
            try:
                self._encoder.encode_single("warmup")
            except Exception as exc:  # noqa: BLE001
                ok = False
                if progress is not None:
                    progress(f"warmup: query encoder weights failed ({exc})")
                logger.exception("retriever warmup failed loading query encoder")
        if self._reranker is not None:
            try:
                self._reranker.score([("warmup", "warmup")])
            except Exception as exc:  # noqa: BLE001
                ok = False
                if progress is not None:
                    progress(f"warmup: cross-encoder weights failed ({exc})")
                logger.exception("retriever warmup failed loading cross-encoder")
        return ok

    def status(self) -> dict:
        """Which heavy components are resident (for the server health check).

        Tells a warm process from a fresh one: all True means the weights are
        loaded once and being reused.
        """
        return {
            "encoder": bool(self._encoder is not None
                            and getattr(self._encoder, "loaded", False)),
            "reranker": bool(self._reranker is not None
                             and getattr(self._reranker, "loaded", False)),
            "store": self._store is not None,
            "bm25": self._bm25 is not None,
        }

    def search(
        self,
        query: str,
        per_leg: int = BM25_TOPK,
        top_k: int = FINAL_TOPK,
        progress: Callable[[str], None] | None = None,
    ) -> list[EvidenceHit]:
        """Run the hybrid pipeline, reporting each stage via progress (if given).

        progress receives one short human-readable line per stage, e.g.
        'dense leg: fetched 60 (top 60) in 1.2s'. None keeps search silent.
        """

        def _say(stage: str) -> None:
            if progress is not None:
                progress(stage)

        query = " ".join((query or "").split())
        if not query:
            return []
        cache_key = (query, per_leg, top_k)
        if SEARCH_CACHE_MAX:
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                self._search_cache.move_to_end(cache_key)
                _say(f"retrieval cache hit: {len(cached)} passage(s)"
                     " (no encoder / DB / rerank)")
                get_trace().log("retrieval_cache_hit", query=query[:200])
                return [hit.model_copy() for hit in cached]
        # 3a. Dense leg: MedCPT query vector against pgvector.
        started = time.perf_counter()
        qvec = self._encoder_or_build(progress).encode_single(query)
        dense_ids = [cid for cid, _ in self._store_or_build(progress).search(qvec, top_k=per_leg)]
        _say(f"dense leg: fetched {len(dense_ids)} (top {per_leg})"
             f" in {time.perf_counter() - started:.1f}s")
        # 3b. Sparse leg: BM25 file index.
        started = time.perf_counter()
        sparse_ids = [h.chunk_id for h in self._bm25_or_build(progress).search_single(query, per_leg)]
        _say(f"bm25 leg: fetched {len(sparse_ids)} (top {per_leg})"
             f" in {time.perf_counter() - started:.1f}s")
        # 3c. Fuse both rankings, keep 60 for the cross-encoder.
        fused_all = rrf_fuse([dense_ids, sparse_ids])
        fused = fused_all[:CROSS_RERANK_TOPK]
        _say(f"rrf fusion: {len(fused_all)} unique candidates"
             f" -> keeping top {len(fused)}")
        if not fused:
            return []
        # 3d. Candidate text for the rerank: the fused chunk rows only.
        started = time.perf_counter()
        metas = self._store_or_build(progress).get_chunks(fused)
        _say(f"candidate rows: fetched {len(metas)}"
             f" in {time.perf_counter() - started:.1f}s")
        # 3e. Cross-encoder rerank of the FULL fused set (60), keep top 10.
        pairs, pair_ids = [], []
        for cid in fused:
            text = (metas.get(cid) or {}).get("text") or ""
            if text.strip():
                pairs.append((query, text))
                pair_ids.append(cid)
        started = time.perf_counter()
        try:
            scores = self._reranker_or_build(progress).score(pairs)
        except Exception as exc:
            # A dead reranker must not kill the run: keep fusion order
            # (scores 0.0 = unranked) so the agent still gets passages.
            _say(f"cross-encoder rerank failed ({exc}); keeping fusion order")
            get_trace().log("cross_encoder_failed", error=str(exc), pairs=len(pairs))
            ranked = [(cid, 0.0) for cid in pair_ids[:top_k]]
        else:
            ranked = sorted(zip(pair_ids, scores), key=lambda kv: -kv[1])[:top_k]
            _say(f"cross-encoder rerank: scored {len(scores)} pairs"
                 f" -> top {len(ranked)} in {time.perf_counter() - started:.1f}s")
        if not ranked:
            return []
        # 3f. Expansion rows for the SURVIVORS only. Fetching every sibling of
        # all 60 candidates downloaded whole tables/paragraphs that the rerank
        # was about to throw away; the 60 chunk rows above are all the rerank
        # needs.
        _say(f"fetching expansion rows for {len(ranked)} survivor(s)…")
        started = time.perf_counter()
        metas, table_groups, unit_groups, child_groups, by_id = (
            self._fetch_expansion_rows([cid for cid, _ in ranked]))
        _say(f"expansion rows: fetched in {time.perf_counter() - started:.1f}s")
        # 3g. Expand survivors to boundary units and shape the hits.
        started = time.perf_counter()
        hits = assemble_passages(ranked, metas, table_groups, unit_groups, child_groups, by_id)
        _say(f"expanded {len(hits)} passages (whole paragraph/table)"
             f" in {time.perf_counter() - started:.1f}s")
        if SEARCH_CACHE_MAX and hits:
            self._search_cache[cache_key] = list(hits)
            self._search_cache.move_to_end(cache_key)
            while len(self._search_cache) > SEARCH_CACHE_MAX:
                self._search_cache.popitem(last=False)
        return hits

    def _fetch_expansion_rows(self, candidate_ids: list[str]):
        # Candidates first.
        store = self._store_or_build()
        metas = store.get_chunks(candidate_ids)
        # Then, in batched queries, every sibling needed for expansion:
        # same-table chunks and same-parent children (whole tables), plus
        # same-parent prose (whole paragraphs) and any parent summaries.
        table_ids = {m.get("table_id") for m in metas.values() if m.get("table_id")}
        table_sids = {
            (cid if (m.get("chunk_type") == "table_summary") else (m.get("parent_id") or ""))
            for cid, m in metas.items()
            if (m.get("chunk_type") or "") in ("table_summary", "table_row", "table_footnotes")
        } - {""}
        prose_pids = {
            m.get("parent_id")
            for m in metas.values()
            if m.get("parent_id") and (m.get("chunk_type") in PROSE_TYPES)
        }
        cur = store.connect().cursor()
        table_groups = _group_by(_fetch_by(cur, "table_id", table_ids), "table_id")
        child_groups = _group_by(_fetch_by(cur, "parent_id", table_sids), "parent_id")
        unit_groups = _group_by(_fetch_by(cur, "parent_id", prose_pids), "parent_id")
        have = set(metas) | {m.get("id") for rows in child_groups.values() for m in rows}
        by_id = dict(metas)
        for cid, meta in _fetch_by(cur, "id", table_sids - have).items():
            by_id[cid] = meta
        cur.close()
        return metas, table_groups, unit_groups, child_groups, by_id


def _fetch_by(cur, column: str, values: set[str]) -> dict[str, dict[str, Any]]:
    # One batched metadata fetch for a set of table_id / parent_id values.
    if not values:
        return {}
    cur.execute(
        ("SELECT id, document_id, chunk_type, section, subsection, breadcrumb,"
         " parent_id, table_id, figure_id, document_position, text"
         " FROM {SCHEMA}.chunks WHERE " + column + " = ANY(%s)"
         " ORDER BY document_position").format(SCHEMA=current_schema()),
        (sorted(values),),
    )
    rows: dict[str, dict[str, Any]] = {}
    for row in cur.fetchall():
        bc = row[5]
        if isinstance(bc, (list, dict)):
            breadcrumb = bc
        elif bc:
            breadcrumb = json.loads(bc)
        else:
            breadcrumb = []
        rows[row[0]] = {
            "id": row[0], "document_id": row[1], "chunk_type": row[2],
            "section": row[3], "subsection": row[4], "breadcrumb": breadcrumb,
            "parent_id": row[6], "table_id": row[7], "figure_id": row[8],
            "document_position": row[9], "text": row[10],
        }
    return rows


def _group_by(rows: dict[str, dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for meta in rows.values():
        groups.setdefault(meta.get(key) or "", []).append(meta)
    return groups


def assemble_passages(
    ranked: list[tuple[str, float]],
    metas: dict[str, dict[str, Any]],
    table_groups: dict[str, list[dict[str, Any]]],
    unit_groups: dict[str, list[dict[str, Any]]],
    child_groups: dict[str, list[dict[str, Any]]],
    by_id: dict[str, dict[str, Any]],
) -> list[EvidenceHit]:
    # Expand each survivor to its closest boundary, then shape the hit.
    hits: list[EvidenceHit] = []
    for cid, score in ranked:
        meta = metas.get(cid) or {}
        ctype = meta.get("chunk_type") or ""
        tid = meta.get("table_id") or ""
        pid = meta.get("parent_id") or ""
        if ctype in ("table_summary", "table_row", "table_footnotes"):
            # Whole table: summary (label + caption + columns) plus all rows
            # in order. Link via table_id, falling back to the summary parent
            # (rows always point at their summary chunk).
            sid = cid if ctype == "table_summary" else (pid or "")
            pool = list(table_groups.get(tid, [])) if tid else []
            pool += child_groups.get(sid, []) if sid else []
            ordered = sorted(pool, key=lambda m: m.get("document_position") or 0)
            parts, seen = [], set()
            summary = by_id.get(sid) if sid else None
            if summary is not None and summary.get("chunk_type") == "table_summary":
                parts.append(summary)
                seen.add(summary.get("id"))
            for m in ordered:
                if m.get("id") not in seen and m.get("chunk_type") in (
                    "table_summary", "table_row", "table_footnotes",
                ):
                    parts.append(m)
                    seen.add(m.get("id"))
            text = "\n".join(m.get("text") or "" for m in parts if (m.get("text") or "").strip())
            if not text:
                text = meta.get("text") or ""
        elif ctype in PROSE_TYPES and pid and pid in unit_groups:
            # Whole paragraph: same-parent prose siblings in document order.
            parts = [m for m in unit_groups[pid] if m.get("chunk_type") in PROSE_TYPES]
            text = "\n".join(m.get("text") or "" for m in parts if (m.get("text") or "").strip())
        else:
            # Figures, equations and lone chunks are already self-contained.
            text = meta.get("text") or ""
        hits.append(EvidenceHit(
            chunk_id=cid,
            document_id=meta.get("document_id") or "",
            chunk_type=ctype,
            section=meta.get("section") or "",
            score=round(float(score), 5),
            text=text,
            title=meta.get("title") or "",
        ))
    return hits


# Module singleton: heavy models load once per process; tests swap this.
_retriever: EvidenceRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> EvidenceRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = EvidenceRetriever()
    return _retriever


async def local_search(
    ctx: RunContext[DeepDeps],
    query: str,
    evidence_requirements: list[dict[str, str] | str] | str | None = None,
) -> str:
    """Search the local PMC corpus for passages relevant to a research question.

    Runs a hybrid search (BM25 top 60 + MedCPT dense top 60 over pgvector),
    fuses the rankings, cross-reranks the top 60 fused candidates with the
    MedCPT cross-encoder down to the top 10, and expands each hit to its
    closest boundary (whole paragraph, whole table with caption). Returns a
    JSON list of passages with chunk_id, document_id, chunk_type, section,
    score and text. Ground every claim in verbatim quotes from the passage texts.

    evidence_requirements lists what the caller is looking for — items shaped
    {"id": "E1", "description": "..."} (bare "E1: ..." strings also work). It
    is not used for ranking; it is carried for the verifier middleware, which
    judges each passage against these requirements before the result reaches
    the agent.
    """
    cleaned = " ".join((query or "").split())
    if not cleaned:
        return "[]"
    retriever = get_retriever()

    def _report(stage: str) -> None:
        narrate.say(f"{style('🔎', '36')} [local_search] {stage}")

    reqs = evidence_requirements or []
    if isinstance(reqs, (str, dict)):
        reqs = [reqs]
    _report(f"query: {cleaned[:160]} ({len(reqs)} requirement(s))…")
    get_trace().log("local_search_start", query=cleaned[:200],
                    requirements=len(reqs))
    hits = await asyncio.to_thread(retriever.search, cleaned, BM25_TOPK, FINAL_TOPK, _report)
    _report(f"done: {len(hits)} passage(s)")
    get_trace().retrieved("hybrid-rrf-crossencoder", cleaned,
                          [(h.chunk_id, h.score) for h in hits])
    return json.dumps([h.model_dump() for h in hits])

