"""UMLS terminology tool: enrich queries with medical terms and synonyms."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pydantic_ai import RunContext

from src.umls.client import UMLSClient, UMLSConcept

from src.budget.budget import BudgetManager


@dataclass
class DeepDeps:
    umls: UMLSClient
    # UI-visible notes stashed by tools mid-run (e.g. fail-open fallbacks).
    # The stream adapter drains these into the thought stream.
    notes: list[str] = field(default_factory=list)
    # Judge-kept passages accumulated run-wide (populated by the judge
    # middleware, consumed by the gap checker so it always sees the full
    # accepted set even when the agent passes a subset).
    kept_passages: list[dict] = field(default_factory=list)
    # Hierarchical budget for this run/task (None = enforcement off;
    # the budget capability is inert without one). Read by
    # BudgetCapability at the tool boundary; the agent itself only ever
    # sees read-only snapshots via the budget_status tool.
    budget: "BudgetManager | None" = None
    # Run-state ledger identity (the per-chat JSON ledger; writers at
    # the tool/execution boundary resolve it through these ids). Writes
    # land in the chat's current turn under task_id; the agent only ever
    # reads compact progress views, never raw stored evidence.
    chat_id: str | None = None
    task_id: str | None = None


def build_umls_client() -> UMLSClient:
    return UMLSClient(
        api_key=os.environ.get("UMLS_API_KEY", ""),
        base_url=os.environ.get("UMLS_BASE_URL", "https://uts-ws.nlm.nih.gov/rest"),
    )


async def lookup_medical_term(ctx: RunContext[DeepDeps], term: str) -> str:
    """Look up a medical term in UMLS to enrich the query before researching.

    Call this for each medical term in your task. Returns a JSON object with
    "term", "found", "cui", "preferred_name", "synonyms", "semantic_types"
    and "source". When "found" is false the term has no UMLS match — carry on
    with the original term.
    """
    cleaned = " ".join((term or "").split())
    if not cleaned:
        return UMLSConcept(term="").model_dump_json()
    return (await ctx.deps.umls.search_concept(cleaned)).model_dump_json()
