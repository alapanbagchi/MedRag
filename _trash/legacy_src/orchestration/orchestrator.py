"""Pipeline orchestrator: parallel per-subquery paper-finding + evidence flow.

Per subquery (all subqueries run in PARALLEL):
  1. Hybrid-retrieve CHUNKS (BM25+dense union -> intent rerank). Chunks are
     only pointers to their structural unit (paragraph / table / figure).
  2. Load each candidate chunk's ENTIRE containing unit.
  3. Verify units against the subquery intent IN BATCHED LLM CALLS.
     - relevant/partially_relevant -> kept
     - not_relevant                -> excluded, rejection reason remembered
     - unknown (quota/parse/infra) -> RETRIED on a later round; never treated
       as a rejection.
  4. When insufficient DISTINCT papers were found, REWRITE the query with an
     LLM call conditioned on the verifier's rejection reasons + UMLS synonyms
     (+ deterministic fallback), and re-search excluding already-seen chunks.

Cross-round / cross-subquery verification MEMO prevents re-verifying the same
(chunk_id, criteria) pair. The loop stops on DISTINCT papers, not raw units,
so one paper cannot satisfy MIN_PAPERS alone.

After retrieval: evidence extraction from verified units (bounded worker
pool), deterministic aggregation, and synthesis over a per-subquery-balanced
evidence selection with full provenance for citations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from src.agents.planner import PlannerAgent, EnrichedPlan
from src.agents.verifier_new import VerifierAgent
from src.agents.evidence import EvidenceExtractor, Evidence, EvidenceAggregator
from src.agents.synthesizer import Synthesizer
from src.config import AppConfig

logger = logging.getLogger("src.orchestrator")

CHUNK_TOP_K = 20           # chunk candidates per search round (unit pointers)
MAX_UNITS_PER_SUBQUERY = 10


class _Funnel:
    """Per-run attrition counters: how many candidates survive each stage."""

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {
            "subqueries": 0,
            "search_rounds": 0,
            "chunks_retrieved": 0,
            "new_chunks": 0,
            "units_loaded": 0,
            "units_verified": 0,
            "units_relevant": 0,
            "units_not_relevant": 0,
            "units_unknown": 0,
            "verifier_llm_calls": 0,
            "papers_found": 0,
            "evidence_tasks": 0,
            "evidence_failed_tasks": 0,
            "evidence_raw_items": 0,
            "evidence_kept": 0,
            "quotes_dropped_ungrounded": 0,
            "evidence_groups": 0,
            "evidence_synthesized": 0,
        }

    def inc(self, key: str, n: int = 1) -> None:
        self.data[key] = self.data.get(key, 0) + n

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def merge(self, other: Dict[str, Any]) -> None:
        for k, v in other.items():
            if isinstance(v, int):
                self.inc(k, v)
            else:
                self.data[k] = v


class Orchestrator:
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        *,
        planner: Optional[PlannerAgent] = None,
        verifier: Optional[VerifierAgent] = None,
        extractor: Optional[EvidenceExtractor] = None,
        aggregator: Optional[EvidenceAggregator] = None,
        synthesizer: Optional[Synthesizer] = None,
        retrieval_service: Optional[Any] = None,
        umls_client: Optional[Any] = None,
    ):
        self.config = config or AppConfig()
        self.planner = planner or PlannerAgent(config=self.config)
        self.verifier = verifier or VerifierAgent(config=self.config)
        self.extractor = extractor or EvidenceExtractor(config=self.config)
        self.aggregator = aggregator or EvidenceAggregator()
        self.synthesizer = synthesizer or Synthesizer(config=self.config)
        self.retrieval_service = retrieval_service
        if umls_client is None:
            try:
                from src.umls.client import UMLSClient

                umls_client = UMLSClient(
                    api_key=self.config.umls_api_key,
                    base_url=self.config.umls_base_url,
                    sabs=self.config.umls_sabs,
                    max_synonyms=self.config.umls_max_synonyms,
                    timeout=self.config.umls_timeout,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("UMLS client unavailable: %s", exc)
                umls_client = None
        self.umls = umls_client
        self._rewriter_prompt = self._load_rewriter_prompt()
        self._rewrite_agent: Any = None

    def _load_rewriter_prompt(self) -> str:
        try:
            from src.prompts.load import load_prompt
            return load_prompt("legacy", "rewriter.txt")
        except Exception:
            return ""

    def _get_rewrite_agent(self) -> Optional[Any]:
        """Lazily build the rewriter Agent (None when unavailable -> fallback)."""
        if self._rewrite_agent is not None:
            return self._rewrite_agent or None
        if not self._rewriter_prompt:
            self._rewrite_agent = False
            return None
        try:
            from pydantic_ai import Agent
            from src.llm import build_model

            self._rewrite_agent = Agent(build_model(self.config), system_prompt=self._rewriter_prompt,
                                        name="rewriter")
            return self._rewrite_agent
        except Exception as exc:
            logger.warning("rewriter agent unavailable (%s); deterministic rewrites only", exc)
            self._rewrite_agent = False
            return None

    # ------------------------------------------------------------------
    async def answer(self, query: str) -> dict:
        from src.trace import get_trace

        trace = get_trace()
        timings: Dict[str, float] = {}
        warnings: List[str] = []
        funnel = _Funnel()

        trace.stage("STAGE 1/5 - PLANNER")
        t0 = time.perf_counter()
        enriched = await self.planner.plan(query)
        timings["plan_ms"] = round((time.perf_counter() - t0) * 1000)
        funnel.inc("subqueries", len(enriched.plan.subqueries))
        trace.bullet(f"plan: question_type={enriched.plan.question_type} subqueries={[s.id for s in enriched.plan.subqueries]}")

        # Deterministic UMLS terminology enrichment (plain HTTP, no LLM):
        # preferred names + synonyms are folded INTO the round-1 retrieval
        # queries, and kept on the subquery for later rewrite rounds.
        t0 = time.perf_counter()
        term_map = await self._enrich_terminology(enriched, warnings)
        if term_map:
            for sub in enriched.plan.subqueries:
                expanded = self._apply_terminology(sub, term_map)
                sub.query = expanded
            trace.bullet(
                "terminology expansion applied to "
                f"{[s.id for s in enriched.plan.subqueries if s.synonyms]}: "
                + "; ".join(f"{s.id}: {s.query}" for s in enriched.plan.subqueries if s.synonyms)
            )
        timings["umls_ms"] = round((time.perf_counter() - t0) * 1000)

        trace.stage("STAGE 2/5 - FIND RELEVANT PAPERS (parallel per subquery)")
        t0 = time.perf_counter()
        memo: Dict[tuple, dict] = {}   # (chunk_id, criteria) -> kept-unit info
        per_subquery: Dict[str, List[dict]] = {}
        tasks = [self._find_relevant_papers(sub, memo, warnings, funnel) for sub in enriched.plan.subqueries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for sub, res in zip(enriched.plan.subqueries, results):
            if isinstance(res, BaseException):
                logger.exception("subquery %s failed", sub.id)
                trace.bullet(f"subquery {sub.id} FAILED: {res}")
                warnings.append(f"subquery {sub.id} failed: {str(res)[:160]}")
                per_subquery[sub.id] = []
            else:
                per_subquery[sub.id] = res
                trace.bullet(f"subquery {sub.id}: {len(res)} relevant papers")
        timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000)

        trace.stage("STAGE 3/5 - EVIDENCE EXTRACTION (verified units, parallel workers)")
        t0 = time.perf_counter()
        evidence = await self._extract_evidence(enriched.plan, per_subquery, funnel)
        timings["extract_ms"] = round((time.perf_counter() - t0) * 1000)
        trace.bullet(f"evidence items: {len(evidence)}")

        trace.stage("STAGE 4/5 - AGGREGATION")
        groups = self.aggregator.aggregate(evidence)
        funnel.set("evidence_groups", len(groups))
        trace.bullet(f"groups: {len(groups)}")

        trace.stage("STAGE 5/5 - SYNTHESIS")
        t0 = time.perf_counter()
        selected, coverage_notes, report = self._prepare_synthesis_input(
            enriched.plan, per_subquery, evidence
        )
        funnel.set("evidence_synthesized", len(selected))
        answer = await self.synthesizer.synthesize(query, enriched.plan, report, selected, coverage_notes)
        timings["synthesize_ms"] = round((time.perf_counter() - t0) * 1000)
        trace.bullet(f"timings: {timings}")

        return {
            "plan": enriched.plan,
            "papers": per_subquery,
            "evidence": evidence,
            "groups": groups,
            "answer": answer,
            "timings": timings,
            "warnings": warnings,
            "funnel": funnel.data,
        }

    # ------------------------------------------------------------------
    # Terminology enrichment
    # ------------------------------------------------------------------
    async def _enrich_terminology(self, enriched: EnrichedPlan, warnings: List[str]) -> Dict[str, Dict[str, Any]]:
        """Resolve plan entities against UMLS/MeSH.

        Returns surface_form -> {"preferred": str|None, "synonyms": [...],
        "all_terms": [preferred + synonyms]}. The preferred names/synonyms are
        then folded INTO the retrieval queries (see _apply_terminology), so
        round-1 searches already use canonical keywords instead of waiting
        for a rewrite round to discover them.
        """
        if self.umls is None or not getattr(self.umls, "enabled", False):
            return {}
        try:
            entities = await self.umls.enrich_entities(list(enriched.clinical_entities))
        except Exception as exc:
            logger.warning("UMLS enrichment failed: %s", exc)
            warnings.append(f"UMLS enrichment failed: {str(exc)[:120]}")
            return {}
        term_map: Dict[str, Dict[str, Any]] = {}
        max_terms = max(2, self.config.umls_max_synonyms)
        for e in entities:
            synonyms = [s for s in e.synonyms if s]
            all_terms: List[str] = []
            if e.preferred_name:
                all_terms.append(e.preferred_name)
            all_terms.extend(synonyms)
            # dedupe, drop echo of the surface form, drop MeSH-inverted forms
            seen_cf = {e.surface_form.casefold()}
            clean: List[str] = []
            for t in all_terms:
                cf = t.casefold()
                if cf in seen_cf or "," in t or len(t) > 40:
                    continue
                seen_cf.add(cf)
                clean.append(t)
            if clean:
                term_map[e.surface_form.casefold()] = {
                    "preferred": (e.preferred_name or "").strip() or None,
                    "synonyms": synonyms[:max_terms],
                    "all_terms": clean[: max_terms + 1],
                }
        return term_map

    @staticmethod
    def _matching_terms(sub: Any, term_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        haystack = f"{sub.target} {sub.query}".casefold()
        return [entry for surface, entry in term_map.items() if surface and surface in haystack]

    @classmethod
    def _apply_terminology(cls, sub: Any, term_map: Dict[str, Dict[str, Any]]) -> str:
        """Fold UMLS preferred names/synonyms into the subquery's round-1 query.

        Returns the expanded query; also fills ``sub.synonyms`` for later
        rewrite rounds. Expansion is capped so BM25 stays meaningful.
        """
        base = sub.query or sub.target
        matches = cls._matching_terms(sub, term_map)
        flat: List[str] = []
        base_cf = f" {base.casefold()} "
        for entry in matches:
            flat.extend(entry["all_terms"][:2])
        additions: List[str] = []
        seen: set = set()
        for t in flat:
            t = " ".join(t.split())
            if not t or t.casefold() in seen:
                continue
            if f" {t.casefold()} " in base_cf:
                continue
            seen.add(t.casefold())
            additions.append(t)
            if len(additions) >= 5:
                break
        sub.synonyms = additions[:6]
        if not additions:
            return base
        return f"{base} {' '.join(additions)}"

    # ------------------------------------------------------------------
    # Paper finding
    # ------------------------------------------------------------------
    async def _find_relevant_papers(
        self,
        sub: Any,
        memo: Dict[tuple, dict],
        warnings: List[str],
        funnel: "_Funnel",
    ) -> List[dict]:
        """Per subquery: retrieve chunks -> load structural units -> verify in
        batches -> rewrite & re-search until enough DISTINCT papers."""
        from src.trace import get_trace

        trace = get_trace()
        trace.stage(f"SUBQUERY {sub.id}: {sub.target}")
        trace.bullet(f"query={sub.query!r} | evidence_required={sub.evidence_required}")

        base_query = sub.query or sub.target
        criteria = tuple(sorted(sub.evidence_required))
        relevant_pool: Dict[str, dict] = {}
        excluded_ids: List[str] = []      # verdicted not_relevant (never re-fetch)
        rejected_reasons: List[str] = []  # feeds the rewriter
        tried_queries: List[str] = [base_query]
        unused_synonyms = list(getattr(sub, "synonyms", []) or [])
        unknown_units: Dict[str, dict] = {}

        for round_num in range(1, self.config.max_query_rounds + 1):
            funnel.inc("search_rounds")
            if round_num == 1:
                query_text = base_query
            else:
                query_text = await self._rewrite_query(
                    sub, round_num, rejected_reasons, tried_queries, unused_synonyms, funnel
                )
                tried_queries.append(query_text)
            trace.bullet(f"round {round_num}: query={query_text[:90]!r}")

            # 1) retrieve chunk pointers, skipping everything already excluded
            chunk_ids = await self._retrieve_chunk_ids(query_text, exclude=excluded_ids)
            fresh = [c for c in chunk_ids
                     if c not in relevant_pool and (c, criteria) not in memo]
            funnel.inc("chunks_retrieved", len(chunk_ids))
            funnel.inc("new_chunks", len(fresh))
            if not fresh:
                trace.bullet("no new candidate chunks this round")
                continue
            trace.bullet(f"  candidate chunks: {fresh[:10]}")

            # 2) load entire structural units (parallel, shared index)
            units = await self._load_structural_units(fresh)
            if not units:
                continue
            funnel.inc("units_loaded", len(units))

            # strongest candidates verified first when budgets cap the batch
            units = self.prefilter_order(units, query_text)

            # memo hits skip the LLM entirely
            to_verify: List[dict] = []
            for u in units:
                hit = memo.get((u["chunk_id"], criteria))
                if hit is not None:
                    relevant_pool.setdefault(u["chunk_id"], {**hit, "via": "memo"})
                else:
                    to_verify.append(u)

            # 3) BATCHED verification of remaining units
            new_unknown: Dict[str, dict] = {}
            if to_verify:
                verdicts = await self.verifier.verify_papers(
                    [{"document_id": u["chunk_id"], "full_text": u["unit_text"]} for u in to_verify],
                    base_query,
                    sub.evidence_required,
                )
                funnel.inc("verifier_llm_calls", max(1, (len(to_verify) + self.config.verify_batch_max_docs - 1) // self.config.verify_batch_max_docs))
                by_id = {v.document_id: v for v in verdicts.results}
                for u in to_verify:
                    v = by_id.get(u["chunk_id"])
                    if v is None:
                        continue
                    funnel.inc("units_verified")
                    if v.relevance in ("relevant", "partially_relevant"):
                        funnel.inc("units_relevant")
                        info = {
                            "chunk_id": u["chunk_id"],
                            "document_id": u["document_id"],
                            "unit_name": u["name"],
                            "full_text": u["unit_text"],
                            "relevance": v.relevance,
                            "reason": v.reason,
                            "confidence": v.confidence,
                        }
                        relevant_pool[u["chunk_id"]] = info
                        memo[(u["chunk_id"], criteria)] = info
                    elif v.relevance == "unknown":
                        # Infrastructure/quota failure: retryable, NOT a rejection.
                        funnel.inc("units_unknown")
                        new_unknown[u["chunk_id"]] = u
                        unknown_units[u["chunk_id"]] = u
                        trace.bullet(f"  unit {u['chunk_id']} UNKNOWN ({v.reason[:80]}) - will retry")
                    else:
                        funnel.inc("units_not_relevant")
                        reason = (v.reason or "").strip()
                        if reason:
                            rejected_reasons.append(reason)
                        if u["chunk_id"] not in excluded_ids:
                            excluded_ids.append(u["chunk_id"])
                        unknown_units.pop(u["chunk_id"], None)

            distinct_papers = len({p.get("document_id") for p in relevant_pool.values() if p.get("document_id")})
            trace.bullet(
                f"  relevant units so far: {len(relevant_pool)} "
                f"({distinct_papers} distinct papers / min {self.config.min_papers})"
            )
            if distinct_papers >= self.config.min_papers:
                break
            if not to_verify and not chunk_ids and not unused_synonyms:
                break  # nothing left to try deterministically

        leftover_unknown = [c for c in unknown_units if c not in relevant_pool]
        if leftover_unknown:
            msg = (f"subquery {sub.id}: {len(leftover_unknown)} units could not be "
                   f"verified (quota/parse failures); they are EXCLUDED from evidence")
            warnings.append(msg)
            trace.bullet(msg)
        funnel.inc("papers_found", len({p.get("document_id") for p in relevant_pool.values()}))
        trace.bullet(f"=> subquery {sub.id}: {len(relevant_pool)} relevant structural units")
        return list(relevant_pool.values())[:MAX_UNITS_PER_SUBQUERY]

    def prefilter_order(self, units: List[dict], query_text: str) -> List[dict]:
        """Order units best-first before budget-capped verification."""
        try:
            from src.retrieval.cross_encoder import UnitPrefilter

            prefilter = UnitPrefilter(self.config)
            return prefilter.order_units(units, query_text)
        except Exception as exc:  # never let ordering break the pipeline
            logger.warning("prefilter ordering failed: %s", exc)
            return units

    async def _rewrite_query(
        self,
        sub: Any,
        round_num: int,
        rejected_reasons: List[str],
        tried_queries: List[str],
        unused_synonyms: List[str],
        funnel: "_Funnel",
    ) -> str:
        """Coverage-driven query rewrite; LLM when enabled, deterministic fallback."""
        base_query = sub.query or sub.target
        if not self.config.rewrite_enabled or not self._rewriter_prompt:
            return self._deterministic_rewrite(base_query, tried_queries, unused_synonyms)

        from pydantic import BaseModel

        class RewrittenQuery(BaseModel):
            rewritten_query: str
            rationale: str = ""

        reasons_block = "\n".join(f"- {r}" for r in rejected_reasons[-6:]) or "(none recorded)"
        tried_block = "\n".join(f"- {q}" for q in tried_queries) or "(none)"
        syn_block = ", ".join(unused_synonyms[:6]) or "(none)"
        prompt = (
            f"ORIGINAL QUESTION INTENT (subquery {sub.id}): {sub.target}\n"
            f"EVIDENCE REQUIRED: {', '.join(sub.evidence_required) if sub.evidence_required else '(unspecified)'}\n\n"
            f"QUERIES ALREADY TRIED:\n{tried_block}\n\n"
            f"WHY UNITS WERE REJECTED:\n{reasons_block}\n\n"
            f"AVAILABLE TERMINOLOGY VARIANTS (UMLS/MeSH): {syn_block}\n\n"
            "Produce the next search query."
        )

        agent = self._get_rewrite_agent()
        if agent is None:
            return self._deterministic_rewrite(base_query, tried_queries, unused_synonyms)
        from src.llm.run import ask_structured

        funnel.inc("rewrite_llm_calls")
        try:
            result = await ask_structured(
                agent, prompt, RewrittenQuery,
                label="rewriter",
                max_tokens=min(512, self.config.agent_max_tokens),
            )
            candidate = " ".join(result.rewritten_query.split()).strip()
            lowered = {q.strip().lower() for q in tried_queries}
            if candidate and candidate.lower() not in lowered:
                if candidate.lower() != base_query.strip().lower():
                    return candidate
        except Exception as exc:
            logger.warning("LLM rewrite failed (%s); using deterministic fallback", exc)
        return self._deterministic_rewrite(base_query, tried_queries, unused_synonyms)

    @staticmethod
    def _deterministic_rewrite(base_query: str, tried_queries: List[str], synonyms: List[str]) -> str:
        """Fallback: append unused terminology variants (better than static seeds)."""
        used = {q.lower() for q in tried_queries}
        for syn in synonyms:
            candidate = f"{base_query} {syn}"
            if candidate.strip().lower() not in used:
                if syn in synonyms:
                    synonyms.remove(syn)
                return candidate
        return base_query

    # ------------------------------------------------------------------
    async def _retrieve_chunk_ids(self, query_text: str, exclude: Optional[List[str]] = None) -> List[str]:
        service = self.retrieval_service
        if service is None:
            from src.retrieval.retriever import get_retrieval_service

            service = self.retrieval_service = get_retrieval_service(self.config)
        from src.agents.planner import SubQuery

        sub = SubQuery(id=f"H:{query_text[:24]}", target=query_text, focus="evidence", query=query_text)
        docs = await service.search_subquery(sub, exclude_chunk_ids=list(exclude or []))
        ids = [(d.chunk_id or d.document_id) for d in docs[:CHUNK_TOP_K]]
        return list(dict.fromkeys(i for i in ids if i))

    async def _load_structural_units(self, chunk_ids: List[str]) -> List[dict]:
        """For each chunk, load the ENTIRE containing structural unit (shared,
        build-once index; capped at MAX_PAPER_TOKENS)."""
        loop = asyncio.get_event_loop()
        svc = self.retrieval_service
        if svc is None:
            from src.retrieval.retriever import get_retrieval_service

            svc = self.retrieval_service = get_retrieval_service(self.config)
        corpus = svc._components()["corpus"]
        from src.retrieval.fullpaper import get_unit_index

        units_index = get_unit_index(corpus)
        max_tokens = self.config.max_paper_tokens
        out = []
        for cid in chunk_ids:
            text = await loop.run_in_executor(None, units_index.get, cid)
            if not text or not text.strip():
                continue
            kind = await loop.run_in_executor(None, units_index.unit_kind, cid)
            if max_tokens and len(text.split()) > max_tokens:
                words = text.split()
                tail = "\n[...unit truncated to %d tokens for context]\n" % max_tokens
                text = " ".join(words[:max_tokens]) + tail
            out.append({
                "chunk_id": cid,
                "document_id": self._unit_doc(corpus, cid),
                "name": kind,
                "unit_text": text,
            })
        return out

    @staticmethod
    def _unit_doc(corpus: Any, chunk_id: str) -> str:
        try:
            return corpus.document_id(chunk_id) or ""
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Evidence extraction
    # ------------------------------------------------------------------
    async def _extract_evidence(self, plan, per_subquery: Dict[str, List[dict]], funnel: "_Funnel") -> List[Evidence]:
        from src.orchestration.workers import EvidenceWorkerPool, EvidenceTask
        from src.trace import get_trace

        trace = get_trace()
        tasks = []
        for sub in plan.subqueries:
            for unit in per_subquery.get(sub.id, []):
                doc = type("Doc", (), {
                    "document_id": unit.get("document_id") or unit["chunk_id"],
                    "chunk_id": unit["chunk_id"],  # the retrieved chunk = pointer
                    "subquery_id": sub.id,
                    "section": unit.get("unit_name", ""),
                    "subsection": "",
                    "breadcrumb": [],
                    "table_id": None,
                    "figure_id": None,
                    "node_type": unit.get("name", "paragraph"),
                    "text": unit.get("full_text", ""),  # ENTIRE containing unit
                })()
                tasks.append(EvidenceTask(subquery=sub, document=doc))

        if not tasks:
            return []

        async def _worker(sq, doc):
            return await self.extractor.extract_with_stats(sq, doc)

        pool = EvidenceWorkerPool(_worker, self.config)
        funnel.inc("evidence_tasks", len(tasks))
        trace.wake(len(tasks), kind="evidence-extraction (verified units)")
        trace.bullet(f"workers: {self.config.max_workers} max | tasks: {len(tasks)}")
        results = await pool.run(tasks)
        evidence = [e for r in results for e in r.evidence]
        failed = sum(1 for r in results if r.failed)
        funnel.inc("evidence_failed_tasks", failed)
        for r in results:
            stats = r.stats or {}
            funnel.inc("evidence_raw_items", int(stats.get("raw_items", 0)))
            funnel.inc("quotes_dropped_ungrounded", int(stats.get("quotes_dropped", 0)))
        funnel.inc("evidence_kept", len(evidence))
        trace.bullet(f"evidence extracted: {len(evidence)} items ({failed} failed tasks)")

        counts: Dict[str, int] = {}
        for item in evidence:
            if item.evidence_id:
                continue
            counts[item.subquery_id] = counts.get(item.subquery_id, 0) + 1
            item.evidence_id = f"E-{item.subquery_id}-{counts[item.subquery_id]}"

        return evidence

    # ------------------------------------------------------------------
    # Synthesis input preparation
    # ------------------------------------------------------------------
    def _prepare_synthesis_input(
        self, plan, per_subquery: Dict[str, List[dict]], evidence: List[Evidence]
    ):
        """Balance evidence ACROSS subqueries and pass real verification status.

        The old code took the first N evidence items in insertion order, which
        (insertion being subquery-major) starved every subquery after the
        first. Round-robin here guarantees each subquery is represented up to
        its share of the budget.
        """
        budget = max(1, self.config.synthesizer_max_evidence)
        by_sub: Dict[str, List[Evidence]] = {}
        order: List[str] = []
        for item in evidence:
            sid = item.subquery_id or "-"
            if sid not in by_sub:
                by_sub[sid] = []
                order.append(sid)
            by_sub[sid].append(item)
        for sid in by_sub:
            by_sub[sid].sort(key=lambda e: -float(e.confidence or 0.0))

        selected: List[Evidence] = []
        while len(selected) < budget:
            progressed = False
            for sid in order:
                if by_sub.get(sid) and len(selected) < budget:
                    selected.append(by_sub[sid].pop(0))
                    progressed = True
            if not progressed:
                break

        coverage_notes: List[str] = []
        verdict_rows = []
        for sub in plan.subqueries:
            n_ev = sum(1 for e in selected if e.subquery_id == sub.id)
            n_units = len(per_subquery.get(sub.id, []))
            papers = len({u.get("document_id") for u in per_subquery.get(sub.id, []) if u.get("document_id")})
            covered = n_ev > 0
            status = f"covered ({n_ev} evidence items, {n_units} units, {papers} papers)" if covered \
                else f"UNCOVERED ({n_units} relevant units but no grounded evidence)"
            verdict_rows.append(SimpleNamespace(group_id=sub.id, status=status))
            coverage_notes.append(f"{sub.id} '{sub.target}': {status}")
        report = SimpleNamespace(verdicts=verdict_rows)
        return selected, coverage_notes, report
