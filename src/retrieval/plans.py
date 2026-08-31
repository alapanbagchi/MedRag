"""Shared retrieval plan models (relocated from the legacy agentic planners).

These pydantic models are the data contract between the v3 worker's
evidence requirements and the shared retrieval service. They were
historically defined inside the now-removed legacy planner modules
(src.agentic, src.agents); they are pure data, so they live here.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """One entity of the original question (legacy shape)."""
    text: str
    role: str = "condition"
    terminology: bool = True


class SubQuery(BaseModel):
    """One retrieval obligation (what the shared service searches for)."""
    id: str
    target: str
    focus: str = "evidence"
    query: str = ""  # natural language query for retrieval
    evidence_required: List[str] = Field(default_factory=list)
    terminology: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)  # UMLS-enriched terms


class QueryPlan(BaseModel):
    """The full decomposition of a question into subqueries (legacy shape)."""
    original_query: str = ""
    question_type: str = "factual"
    entities: List[Entity] = Field(default_factory=list)
    targets: List[str] = Field(default_factory=list)
    subqueries: List[SubQuery] = Field(default_factory=list)


class PlannedEntity(BaseModel):
    """A medical entity extracted for a specific evidence requirement."""
    text: str
    role: str = "condition"
    synonyms: List[str] = Field(default_factory=list)


class ClinicalEntity(BaseModel):
    """A UMLS-resolved entity (used by the UMLS client)."""
    surface_form: str
    base_concept: str = ""
    role: str = "condition"
    preferred_name: str = ""
    ontology: str = ""
    cui: str = ""
    synonyms: List[str] = Field(default_factory=list)


class SubQueryPlan(BaseModel):
    """One evidence obligation derived from the question (v3 shape)."""
    id: str
    target: str = ""
    intent: str = ""
    query: str = ""
    focus: str = "evidence"
    evidence_required: List[str] = Field(default_factory=list)
    entities: List[PlannedEntity] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)  # UMLS-enriched terms
    enriched_query: str = ""  # query after synonym folding


__all__ = ["ClinicalEntity", "Entity", "PlannedEntity", "QueryPlan",
           "SubQuery", "SubQueryPlan"]