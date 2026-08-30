"""Synthesis agent: final answer from verified evidence only."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("src.synthesizer")


class Citation(BaseModel):
    subquery_id: str
    document_id: str
    chunk_id: Optional[str] = None
    section: str = ""
    evidence_id: Optional[str] = None
    quote: str = ""


class AnswerSection(BaseModel):
    heading: str = ""
    body: str
    citations: List[Citation] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    summary: str = Field(default="")
    sections: List[AnswerSection] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    coverage_note: str = ""


def format_evidence_line(e: Any) -> str:
    """One evidence item WITH full provenance so the model can cite honestly.

    Without document_id/chunk_id in the prompt the model invents citation
    metadata (observed: document_id="9" — a reference number from the quote).
    """
    doc = getattr(e, "document_id", "") or "unknown"
    chunk = getattr(e, "chunk_id", "") or "unknown"
    section = getattr(e, "section", "") or "unknown"
    sub = getattr(e, "subquery_id", "") or "-"
    eid = getattr(e, "evidence_id", "") or "-"
    quote = (getattr(e, "supporting_text", "") or "").strip()
    claim = (getattr(e, "claim", "") or "").strip()
    return (
        f"[{eid}] (subquery={sub}, document={doc}, chunk={chunk}, section={section})\n"
        f"    claim: {claim}\n"
        f'    quote: "{quote}"'
    )


class Synthesizer:
    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        from src.prompts.load import load_prompt

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="synthesizer")
        self._prompt = load_prompt("legacy", "synthesizer.txt")
        self.agent = Agent(self.model, system_prompt=self._prompt, name="synthesizer")

    def _build_prompt(
        self,
        query: str,
        plan: Any,
        report: Any,
        evidence: list,
        coverage_notes: Optional[List[str]] = None,
    ) -> str:
        verdicts = list(getattr(report, "verdicts", []) or [])
        verification_block = (
            "; ".join(f"{v.group_id}: {v.status}" for v in verdicts) if verdicts else "(no per-subquery verification summary)"
        )
        coverage_block = "\n".join(coverage_notes or []) or "(none)"
        return (
            f"ORIGINAL QUERY: {query}\n"
            f"QUESTION TYPE: {plan.question_type}\n"
            f"SUBQUERIES:\n"
            + "\n".join(f"  {s.id}: {s.target} (focus={s.focus})" for s in plan.subqueries)
            + "\n\nVERIFICATION (per subquery): "
            + verification_block
            + "\n\nCOVERAGE NOTES:\n"
            + coverage_block
            + "\n\nVERIFIED EVIDENCE:\n"
            + "\n".join(format_evidence_line(e) for e in evidence)
            + "\n\nWrite the final answer using ONLY the verified evidence above. "
            "In every citation, copy subquery_id, document_id, chunk_id, section and "
            "evidence_id EXACTLY as they appear in the evidence lines — never invent ids."
        )

    async def synthesize(
        self,
        query: str,
        plan: Any,
        report: Any,
        evidence: list,
        coverage_notes: Optional[List[str]] = None,
    ) -> FinalAnswer:
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        trace.agent("synthesizer", output_type="FinalAnswer",
                    meta={"evidence_items": len(evidence), "subqueries": [s.id for s in plan.subqueries]})

        prompt = self._build_prompt(query, plan, report, evidence, coverage_notes)
        ans = await ask_structured(
            self.agent, prompt, FinalAnswer,
            label="synthesizer",
            max_tokens=self.config.agent_max_tokens,
        )
        return self._repair_citations(ans, evidence)

    @staticmethod
    def _repair_citations(ans: FinalAnswer, evidence: list) -> FinalAnswer:
        """Fix citation ids by resolving evidence_ids against known items.

        The model still occasionally mangles document/chunk ids; evidence_id is
        a strong key, so when present we overwrite the rest of the citation
        from the real evidence item.
        """
        if not ans.sections:
            return ans
        by_eid = {e.evidence_id: e for e in evidence if getattr(e, "evidence_id", None)}
        changed = False
        sections = []
        for section in ans.sections:
            citations = []
            for c in section.citations:
                match = by_eid.get(c.evidence_id) if c.evidence_id else None
                if match is not None:
                    fixed = c.model_copy(update={
                        "subquery_id": match.subquery_id or c.subquery_id,
                        "document_id": match.document_id or c.document_id,
                        "chunk_id": match.chunk_id if match.chunk_id else c.chunk_id,
                        "section": match.section or c.section,
                    })
                    if fixed != c:
                        changed = True
                    citations.append(fixed)
                else:
                    citations.append(c)
            sections.append(section.model_copy(update={"citations": citations}))
        if changed:
            logger.info("repaired citation metadata from evidence ids")
        return ans.model_copy(update={"sections": sections})
