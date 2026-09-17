"""Shared source-trust verdict models.

The LLM trust gate that used to live here was replaced on the web path by the
MyBib credibility gate (src.middleware.mybib), which imports these models to
keep the tool output shape (trust: [...]) stable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UrlVerdict(BaseModel):
    url: str
    trustworthy: bool = False
    category: str = ""
    reason: str = ""


class TrustReport(BaseModel):
    results: list[UrlVerdict] = Field(default_factory=list)


__all__ = ["TrustReport", "UrlVerdict"]
