"""Structured-output parsing tests (fences, prose, think-blocks, salvage)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from src.lib.utils import decode_structured, strip_think


class Plan(BaseModel):
    question_type: str = "factual"
    original_query: str = ""
    subqueries: list = []


class Sub(BaseModel):
    id: str
    question: str = ""


def test_decode_plain_json():
    plan = decode_structured('{"question_type":"factual","subqueries":[]}', Plan)
    assert plan.question_type == "factual"


def test_decode_strips_code_fences():
    fenced = "```json\n" + json.dumps({"question_type": "definition", "subqueries": []}) + "\n```"
    plan = decode_structured(fenced, Plan)
    assert plan.question_type == "definition"


def test_decode_strips_fences_and_prose():
    plan = decode_structured(
        'Sure! Here you go:\n```json\n{"question_type":"factual",'
        '"subqueries":[]}\n```\nHope that helps.',
        Plan,
    )
    assert plan.question_type == "factual"
    assert plan.original_query == ""  # omitted optional fields default cleanly


def test_decode_think_wrapped_payload():
    """Gemma-style <thought> reasoning must not break JSON extraction."""
    raw = (
        "<thought>Let me analyze the question step by step.\n"
        "- identify entities\n- build plan</thought>"
        '{"question_type":"association","subqueries":[{"id":"H1"}]}'
    )
    plan = decode_structured(raw, Plan)
    assert plan.subqueries[0]["id"] == "H1" if isinstance(plan.subqueries[0], dict) else plan.subqueries[0].id == "H1"


def test_decode_reports_validation_detail():
    with pytest.raises(ValueError):
        decode_structured('{"id": 123}', Sub)  # wrong types, no salvageable object


def test_decode_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        decode_structured("", Plan)


def test_strip_think_removes_reasoning_blocks():
    assert strip_think('<think>hidden</think>{"a":1}') == '{"a":1}'
    assert strip_think('<thinking>abc</thinking>tail') == 'tail'
    assert strip_think('<thought>x</thought><thought>y</thought>ok') == 'ok'
    assert strip_think("no blocks here") == "no blocks here"
