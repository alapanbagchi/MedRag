"""Medical evidence verifier backed by TypeSafe System One (Jev).

Scores every retrieved passage against every evidence requirement as a set of
independent typed judgments:

- one **Noul** (yes/no probability) for the passage's overall intent, and
- one **Noul** per evidence requirement for coverage, and
- one **Choice** per requirement (web passages only) that selects the exact
  span of the passage carrying that evidence.

Code owns the thresholds, ids, keep gate, and cache; Jev supplies the semantic
probability. There is no prompt-and-parse step: the response is a typed answer
keyed by question name, so there is no free-text JSON to repair and no
hallucinated passage ids to reconcile. Coverage is a code-side threshold on a
calibrated probability ([0, 1]) rather than a model-authored list.

Contract (unchanged for callers):
    verify_passages(question, evidence_requirements, passages) -> VerdictResponse

PassageVerdict keeps intent_score (0..1), coverage (requirement ids), reason
(a deterministic probability breakdown for the UI), and verbatim (an exact span
copied out of the passage by code, never paraphrased). The keep gate rejects a
passage iff its coverage list is empty. On any failure the call degrades to an
empty (or cache-only) response instead of raising, so retrieval tools stay
alive when the verifier is down.

Configuration (environment):
    TYPESAFE_API_KEY              required; read by the SDK
    TYPESAFE_MODEL                model id (default "jev-latest")
    TYPESAFE_COVERAGE_THRESHOLD   Noul probability at/above which a requirement
                                  is covered (default 0.5)
    TYPESAFE_MAX_INFLIGHT         concurrent System One calls (default 16)
    VERIFIER_TIMEOUT_S            per-call timeout in seconds (default 180)
    VERIFIER_MAX_CHARS            per-passage text cap (default 2000; <=0 = no cap)
    VERIFIER_VERBATIM_CANDIDATES  max span options per passage (default 12)
    VERIFIER_VERDICT_CACHE        "0" disables the in-process verdict cache
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import textwrap
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import AsyncExitStack

from pydantic import BaseModel, Field

from src.lib import narrate
from src.lib.pretty import style
from src.lib.trace import get_trace

logger = logging.getLogger(__name__)

VERIFIER_TIMEOUT_S = "VERIFIER_TIMEOUT_S"
DEFAULT_VERIFIER_TIMEOUT_S = 180.0

# Passage text handed to Jev. The judgment quality is flat past a few hundred
# tokens of context and the call is input-token-bound, so this is a real cost
# lever. VERIFIER_MAX_CHARS overrides; a positive value caps the text and -1
# (or 0) sends the whole passage untruncated.
DEFAULT_VERIFIER_MAX_CHARS = 2000

DEFAULT_TYPESAFE_MODEL = "jev-latest"
# Noul returns P(yes). 0.5 is the calibrated decision boundary; raise it to be
# more conservative about coverage, lower it to recall more.
DEFAULT_COVERAGE_THRESHOLD = 0.5
DEFAULT_MAX_INFLIGHT = 16
DEFAULT_VERBATIM_CANDIDATES = 12

def _env_seconds(name: str, default: float) -> float:
    try:
        return max(1.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _max_text_chars() -> int:
    """Effective per-passage cap for Jev.

    A positive value caps the text; 0 and every negative value (including the
    documented -1) mean "send the whole passage, untruncated".
    """
    try:
        cap = int(os.environ.get("VERIFIER_MAX_CHARS",
                                 DEFAULT_VERIFIER_MAX_CHARS))
    except (TypeError, ValueError):
        return DEFAULT_VERIFIER_MAX_CHARS
    return cap if cap > 0 else 0


def _coverage_threshold() -> float:
    try:
        return min(1.0, max(0.0, float(os.environ.get(
            "TYPESAFE_COVERAGE_THRESHOLD", DEFAULT_COVERAGE_THRESHOLD))))
    except (TypeError, ValueError):
        return DEFAULT_COVERAGE_THRESHOLD


def _max_inflight() -> int:
    return _env_int("TYPESAFE_MAX_INFLIGHT", DEFAULT_MAX_INFLIGHT)


def _verbatim_candidates() -> int:
    return _env_int("VERIFIER_VERBATIM_CANDIDATES",
                    DEFAULT_VERBATIM_CANDIDATES)


def _typesafe_model() -> str:
    return (os.environ.get("TYPESAFE_MODEL", DEFAULT_TYPESAFE_MODEL).strip()
            or DEFAULT_TYPESAFE_MODEL)


class EvidenceRequirement(BaseModel):
    id: str
    description: str


class PassageVerdict(BaseModel):
    passage_id: str
    # One intent score per passage (overall question), plus the ids of the
    # evidence requirements this passage meets.
    intent_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage: list[str] = Field(default_factory=list)
    reason: str = ""
    # The most relevant sentence(s) of the passage, copied VERBATIM out of the
    # source by code from the span Jev selected (never model-generated, so it
    # always passes is_verbatim_excerpt).
    verbatim: str = ""


class VerdictResponse(BaseModel):
    evidence_results: list[PassageVerdict] = []
    # True when the judge actually ran (even if every passage was rejected);
    # False on a hard failure. Lets the UI tell "nothing relevant" apart from
    # "verification failed".
    judged: bool = False
    # Total input+output tokens spent judging. Jev bills input only; output is
    # free but still reported.
    tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    # Per-passage critic reason for kept AND rejected passages, keyed by
    # passage id. The keep gate drops rejected verdicts from evidence_results,
    # so this is how their reasons still reach the UI.
    reasons: dict[str, str] = {}


_VERBATIM_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "that", "this", "it", "its", "not", "but", "have", "has", "had", "do",
    "does", "did",
})


def _excerpt_tokens(text: str) -> list[str]:
    """Whitespace tokens of an excerpt/passage (punctuation kept attached so
    exactness survives). Elision ellipses are dropped."""
    import re as _re

    cleaned = _re.sub(r"[.…]{2,}|…", " ", text or "")
    return [t for t in cleaned.split() if t]


def is_verbatim_excerpt(excerpt: str | None, passage: str | None) -> bool:
    """Deterministic check: the excerpt tokens must appear IN ORDER as a
    subsequence of the passage tokens (verbatim spans, elisions allowed).
    Requires at least one non-stopword token so "the of and" never passes.
    """
    if not excerpt or not passage:
        return False
    source = _excerpt_tokens(passage)
    want = _excerpt_tokens(excerpt)
    if not want:
        return False
    if not any(t not in _VERBATIM_STOPWORDS for t in want):
        return False
    i = 0
    for token in source:
        if token.lower() == want[i].lower():
            i += 1
            if i == len(want):
                return True
    return False


# Cheap boilerplate signal: very short texts built only from these tokens are
# navigation/chrome, never evidence.
_BOILERPLATE_TOKENS = frozenset({
    "menu", "navigation", "nav", "cookie", "cookies", "subscribe", "login",
    "signup", "sign-up", "advertisement", "footer", "header", "sidebar",
    "404", "not-found",
})


def _coerce_requirement(
    item: EvidenceRequirement | dict | str, index: int
) -> EvidenceRequirement:
    import json

    if isinstance(item, EvidenceRequirement):
        return item
    if isinstance(item, str):
        text = " ".join(item.split())
        # Agents often hand over one JSON-encoded requirement (or list);
        # parse it before the "id: desc" heuristic misreads JSON syntax.
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            return _coerce_requirement(parsed, index)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], (str, dict)):
            return _coerce_requirement(parsed[0], index)
        if ":" in text and len(text.split(":", 1)[0]) <= 12:
            rid, desc = text.split(":", 1)
            return EvidenceRequirement(
                id=rid.strip() or f"E{index + 1}",
                description=desc.strip() or text,
            )
        return EvidenceRequirement(id=f"E{index + 1}", description=text)
    if not isinstance(item, dict):
        item = {}
    return EvidenceRequirement(
        id=str(item.get("id") or item.get("requirement_id") or f"E{index + 1}"),
        description=" ".join(str(
            item.get("description") or item.get("desc") or item.get("text") or ""
        ).split()),
    )


def _coerce_passage(item: dict, index: int) -> dict[str, str]:
    # Accept retrieval passages ({chunk_id/passage_id, text}), web results
    # ({url, snippet}), scraped pages ({url, markdown}), and middleware
    # string entries ({citation_id, text}). Markdown last: it is the rawest
    # form, but a page with nothing else must still reach the judge.
    if not isinstance(item, dict):
        item = {}
    text = (item.get("text") or item.get("snippet") or item.get("content")
            or item.get("markdown") or "")
    text = " ".join(str(text).split())
    cap = _max_text_chars()
    if cap and len(text) > cap:
        text = text[:cap] + "… [truncated]"
    return {
        "id": str(item.get("id") or item.get("passage_id") or item.get("chunk_id")
                  or item.get("citation_id") or item.get("url") or f"P{index + 1}"),
        "text": text,
    }


def _is_boilerplate(text: str) -> bool:
    if not text.strip():
        return True
    words = [w.strip(""".,:;!?()[]{}"'""").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return True
    if len(words) > 12:
        return False
    return all(w in _BOILERPLATE_TOKENS for w in words)


def _clamp01(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _clean_reason(reason: object) -> str:
    """Whitespace-normalize a judge reason; length is never capped."""
    return " ".join(str(reason or "").split())


def _clean_verbatim(text: object) -> str:
    """Trim a verbatim excerpt WITHOUT touching inner whitespace: the
    deterministic check compares tokens, so only leading/trailing whitespace
    is removed."""
    return str(text or "").strip()


def normalize_response(
    response: VerdictResponse,
    requirements: list[EvidenceRequirement],
    passages: list[dict[str, str]],
    check_verbatim: bool = True,
) -> VerdictResponse:
    """Deterministic hygiene on the judge output; same shape in and out.

    - Verdicts realign to the judged passages in order (unknown ids dropped,
      missing ones padded with zero intent and empty coverage).
    - Coverage ids are filtered to the requested requirements (unknown ids
      dropped); intent clamps to [0, 1]; empty or boilerplate text zeroes the
      intent and empties coverage.
    """
    known = {req.id for req in requirements}
    by_pid: dict[str, PassageVerdict] = {}
    for verdict in response.evidence_results or []:
        pid = str(verdict.passage_id)
        if pid and pid not in by_pid:
            by_pid[pid] = verdict
    results = []
    for passage in passages:
        pid = passage["id"]
        raw = by_pid.get(pid)
        if _is_boilerplate(passage["text"]):
            results.append(PassageVerdict(
                passage_id=pid, reason="empty or boilerplate"))
        elif raw is None:
            results.append(PassageVerdict(
                passage_id=pid, reason=_REASON_UNSCORED))
        else:
            coverage = [str(c) for c in (raw.coverage or []) if str(c) in known]
            excerpt = _clean_verbatim(raw.verbatim) if check_verbatim else ""
            if excerpt and not is_verbatim_excerpt(excerpt, passage["text"]):
                # Paraphrased or invented excerpt: never accepted on trust.
                excerpt = ""
            results.append(PassageVerdict(
                passage_id=pid,
                intent_score=_clamp01(raw.intent_score),
                coverage=coverage,
                reason=_clean_reason(raw.reason) or "no reason given",
                verbatim=excerpt,
            ))
    return VerdictResponse(evidence_results=results)


def _apply_keep_gate(response: VerdictResponse) -> tuple[VerdictResponse, set[str]]:
    """Deterministically split kept vs rejected passages.

    A passage is rejected iff its coverage list is empty (it meets none
    of the evidence requirements); everything else is kept. Intent is
    informational only. Returns the kept-only response plus the rejected
    passage ids.
    """
    rejected = {v.passage_id for v in response.evidence_results
                if not v.coverage}
    kept = [v for v in response.evidence_results if v.passage_id not in rejected]
    return VerdictResponse(evidence_results=kept), rejected


# --------------------------------------------------------------------------
# Span candidates: code finds the quotable spans, Jev selects among them.
# --------------------------------------------------------------------------

_SENTENCE_SPLIT = None  # compiled lazily to keep import cheap


def _span_candidates(text: str) -> list[str]:
    """Split a passage into at most VERIFIER_VERBATIM_CANDIDATES spans.

    Sentences when there are few enough; otherwise contiguous groups of
    sentences so the whole passage stays represented (a long page must not
    lose its ending to a head-only cut). Every span is a contiguous slice of
    the already whitespace-normalized passage, so a selected span is exact by
    construction.
    """
    import re as _re

    global _SENTENCE_SPLIT
    if _SENTENCE_SPLIT is None:
        _SENTENCE_SPLIT = _re.compile(r"(?<=[.!?]) ")
    text = " ".join((text or "").split())
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return [text]
    cap = _verbatim_candidates()
    if len(sentences) <= cap:
        return sentences
    spans = []
    n = len(sentences)
    for i in range(cap):
        lo = i * n // cap
        hi = (i + 1) * n // cap
        spans.append(" ".join(sentences[lo:hi]))
    return spans


# --------------------------------------------------------------------------
# Question construction: one Noul per decision, one Choice per verbatim span.
# --------------------------------------------------------------------------

INTENT_KEY = "intent"
NONE_OPTION = "none"

_INTENT_CRITERIA = {
    "true": ("The passage contains information that directly helps answer the "
             "research question, even if only partially."),
    "false": ("The passage is unrelated to the research question, is boilerplate, "
              "or only mentions the topic in passing."),
}
_COVERAGE_CRITERIA = {
    "true": ("The passage genuinely provides substantive information that "
             "addresses the requirement."),
    "false": ("The passage does not address the requirement, is off-topic, or "
              "only mentions it in passing."),
}


def coverage_key(requirement_id: str) -> str:
    return f"coverage::{requirement_id}"


def verbatim_key(requirement_id: str) -> str:
    return f"verbatim::{requirement_id}"


def _intent_instructions(question: str) -> str:
    return (
        "The overall research question is:\n"
        f"{question.strip()}\n\n"
        "Does the passage in 'passage.text' contain information that helps "
        "answer that question? Answer yes if it supplies useful substantive "
        "information, even partially. Answer no if it is unrelated, "
        "boilerplate/navigation, or only mentions the topic in passing."
    )


def _coverage_instructions(requirement: EvidenceRequirement) -> str:
    return (
        "Does the passage in 'passage.text' provide substantive evidence for "
        "the following evidence requirement?\n\n"
        f"Requirement {requirement.id}: {requirement.description}\n\n"
        "Answer yes only if the passage genuinely addresses this requirement "
        "with substantive information. A passing mention, a definition of an "
        "unrelated term, or off-topic text is no."
    )


def _verbatim_instructions(requirement: EvidenceRequirement) -> str:
    return (
        "The options are contiguous spans of the passage in 'passage.text', "
        "in order. Select the span that most directly provides the substantive "
        "evidence for the following requirement:\n\n"
        f"Requirement {requirement.id}: {requirement.description}\n\n"
        f"Select the exact supporting span, or the option named {NONE_OPTION!r} "
        "when the passage provides no substantive evidence for it."
    )


def build_questions(
    question: str,
    requirements: list[EvidenceRequirement],
    spans: list[str],
    ask_verbatim: bool,
) -> dict:
    """The typed question map sent to System One for one passage.

    All questions are independent judgments over the same state, so they run
    in parallel in a single request. Question names are code-only identifiers
    (never sent to the model); the full meaning lives in each instructions
    string.
    """
    from typesafe_sdk import Choice, Noul

    questions: dict = {
        INTENT_KEY: Noul(
            instructions=_intent_instructions(question),
            criteria=_INTENT_CRITERIA,
        )
    }
    for req in requirements:
        questions[coverage_key(req.id)] = Noul(
            instructions=_coverage_instructions(req),
            criteria=_COVERAGE_CRITERIA,
        )
    if ask_verbatim and spans:
        criteria: dict = {f"s{i}": span for i, span in enumerate(spans)}
        criteria[NONE_OPTION] = "The passage provides no substantive evidence."
        for req in requirements:
            questions[verbatim_key(req.id)] = Choice(
                instructions=_verbatim_instructions(req),
                criteria=criteria,
            )
    return questions


def build_state(
    question: str,
    requirements: list[EvidenceRequirement],
    passage: dict[str, str],
) -> dict:
    """Structured state for one passage: the question, the contract, the text."""
    return {
        "question": question.strip(),
        "evidence_requirements": [
            {"id": r.id, "description": r.description} for r in requirements
        ],
        "passage": {"id": passage["id"], "text": passage["text"]},
    }


# --------------------------------------------------------------------------
# System One call per passage.
# --------------------------------------------------------------------------


def _make_client():
    """Fresh async TypeSafe client for one verification pass.

    load_env_file runs first so TYPESAFE_API_KEY in backend/.env is honored
    even when the verifier is exercised without the full app startup. Raises
    TypeSafeError when the key is missing; callers degrade.
    """
    from src.config import load_env_file

    load_env_file()
    from typesafe_sdk import AsyncTypeSafeClient

    return AsyncTypeSafeClient(model=_typesafe_model())


def _noul(response: object, key: str) -> float | None:
    answer = getattr(response, "nouls", {}).get(key)
    if answer is None:
        return None
    try:
        return min(1.0, max(0.0, float(answer.noul)))
    except (TypeError, ValueError):
        return None


def _choice(response: object, key: str) -> str | None:
    answer = getattr(response, "choices", {}).get(key)
    return getattr(answer, "choice", None)


def _span_index(label: str | None, spans: list[str]) -> int | None:
    if not label or label == NONE_OPTION:
        return None
    if label.startswith("s"):
        try:
            idx = int(label[1:])
        except ValueError:
            return None
        if 0 <= idx < len(spans):
            return idx
    return None


def _join_unique(spans: list[str]) -> str:
    seen: OrderedDict[str, None] = OrderedDict()
    for span in spans:
        span = span.strip()
        if span:
            seen.setdefault(span, None)
    return " ... ".join(seen)


def _format_reason(scores: list[tuple[str, float]], threshold: float) -> str:
    if not scores:
        return "no verdict received"
    parts = []
    for rid, probability in scores:
        mark = ">=" if probability >= threshold else "<"
        parts.append(f"{rid} {probability:.2f}{mark}{threshold:.2f}")
    return "; ".join(parts)


def verdict_from_response(
    response: object,
    requirements: list[EvidenceRequirement],
    passage: dict[str, str],
    spans: list[str],
    threshold: float,
    ask_verbatim: bool,
) -> PassageVerdict:
    """Turn one typed System One response into a PassageVerdict.

    Coverage is the set of requirements whose Noul probability is at or above
    the threshold. The verbatim span is copied out of the passage, never
    generated: a selected span is a contiguous slice of passage.text.
    """
    intent = _noul(response, INTENT_KEY)
    coverage: list[str] = []
    scores: list[tuple[str, float]] = []
    picked: list[str] = []
    for req in requirements:
        probability = _noul(response, coverage_key(req.id))
        if probability is None:
            continue
        scores.append((req.id, probability))
        if probability >= threshold:
            coverage.append(req.id)
            if ask_verbatim and spans:
                idx = _span_index(_choice(response, verbatim_key(req.id)), spans)
                if idx is not None:
                    picked.append(spans[idx])
    verbatim = ""
    if ask_verbatim and coverage and spans:
        # A covered passage must carry quotable text: fall back to the first
        # span when no explicit span was selected, so web evidence never
        # silently reaches the agent with its body stripped.
        verbatim = _join_unique(picked) or spans[0]
    return PassageVerdict(
        passage_id=passage["id"],
        intent_score=intent if intent is not None else 0.0,
        coverage=coverage,
        reason=_format_reason(scores, threshold),
        verbatim=verbatim,
    )


async def _judge_passage(
    client: object,
    label: str,
    question: str,
    requirements: list[EvidenceRequirement],
    passage: dict[str, str],
    ask_verbatim: bool,
    threshold: float,
    timeout: float,
    semaphore: asyncio.Semaphore,
) -> tuple[PassageVerdict, dict]:
    """Judge one passage; returns (verdict, usage). Raises on a hard failure."""
    spans = _span_candidates(passage["text"]) if ask_verbatim else []
    questions = build_questions(question, requirements, spans, ask_verbatim)
    state = build_state(question, requirements, passage)
    started = time.perf_counter()
    async with semaphore:
        response = await asyncio.wait_for(
            client.system_one(state, questions), timeout)  # type: ignore[attr-defined]
    latency_ms = round((time.perf_counter() - started) * 1000)
    verdict = verdict_from_response(
        response, requirements, passage, spans, threshold, ask_verbatim)
    usage = getattr(response, "usage", None)
    prompt = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    completion = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
    return verdict, {"prompt": prompt, "completion": completion,
                     "latency_ms": latency_ms}


# --------------------------------------------------------------------------
# Verdict cache (content-addressed, bounded).
# --------------------------------------------------------------------------

_VERDICT_CACHE: "OrderedDict[tuple, PassageVerdict]" = OrderedDict()
VERDICT_CACHE_MAX = 512


def _verdict_cache_enabled() -> bool:
    return os.environ.get("VERIFIER_VERDICT_CACHE", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _verdict_cache_key(
    question: str,
    requirements: list[EvidenceRequirement],
    passage: dict[str, str],
    ask_verbatim: bool,
    threshold: float,
) -> tuple:
    req_sig = tuple((r.id, r.description) for r in requirements)
    digest = hashlib.sha1(
        passage["text"].encode("utf-8", "ignore")).hexdigest()
    return (question.strip(), req_sig, bool(ask_verbatim), round(threshold, 4),
            _typesafe_model(), digest)


def _verdict_cache_lookup(key: tuple) -> PassageVerdict | None:
    verdict = _VERDICT_CACHE.get(key)
    if verdict is not None:
        _VERDICT_CACHE.move_to_end(key)
    return verdict


def _verdict_cache_store(key: tuple, verdict: PassageVerdict) -> None:
    _VERDICT_CACHE[key] = verdict
    _VERDICT_CACHE.move_to_end(key)
    while len(_VERDICT_CACHE) > VERDICT_CACHE_MAX:
        _VERDICT_CACHE.popitem(last=False)


_REASON_UNSCORED = "not scored"
_REASON_FAILED = "judge request failed"


# --------------------------------------------------------------------------
# Public entry point.
# --------------------------------------------------------------------------


def _upsert(response: VerdictResponse, verdict: PassageVerdict) -> VerdictResponse:
    """Replace-or-append one verdict, preserving first-seen passage order."""
    order = [v.passage_id for v in response.evidence_results]
    by_pid = {v.passage_id: v for v in response.evidence_results}
    if verdict.passage_id not in by_pid:
        order.append(verdict.passage_id)
    by_pid[verdict.passage_id] = verdict
    return VerdictResponse(evidence_results=[by_pid[pid] for pid in order])


async def verify_passages(
    question: str,
    evidence_requirements: list[EvidenceRequirement | dict | str],
    passages: list[dict[str, str]],
    client: object | None = None,
    label: str = "verifier",
    on_thinking: Callable[[str], None] | None = None,
    ask_verbatim: bool = True,
) -> VerdictResponse:
    """Score every passage: one intent probability plus covered requirement ids.

    One System One request per passage, fanned out in parallel (bounded by
    TYPESAFE_MAX_INFLIGHT). Each request carries every independent judgment for
    that passage - intent, one coverage Noul per requirement, and (web
    passages) one span-selection Choice per requirement - so a passage costs a
    single round trip. Coverage is a code-side threshold on the returned
    probabilities; the verbatim excerpt is copied out of the passage by code.

    client injects a System One client (tests pass a fake); when omitted a real
    AsyncTypeSafeClient is created for the duration of the call. on_thinking is
    accepted for caller compatibility but is a no-op: System One returns
    decisions, not a token stream.

    On any failure (missing key, transport error, timeout) the affected
    passages stay unscored and are rejected by the keep gate; the call never
    raises for a judge problem.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    requirements = [_coerce_requirement(r, i)
                    for i, r in enumerate(evidence_requirements or [])]
    clean_passages = [_coerce_passage(p, i)
                      for i, p in enumerate(passages or [])]
    if not requirements or not clean_passages:
        return VerdictResponse(evidence_results=[])

    threshold = _coverage_threshold()
    timeout = _env_seconds(VERIFIER_TIMEOUT_S, DEFAULT_VERIFIER_TIMEOUT_S)
    semaphore = asyncio.Semaphore(_max_inflight())
    use_cache = client is None and _verdict_cache_enabled()

    narrate.say(
        f"{style('⚖', '33')} [{label}] judging {len(clean_passages)} "
        f"passage(s) against {len(requirements)} requirement(s) with "
        f"{_typesafe_model()}: {question.strip()}"
    )
    narrate.say(f"  [{label}] evidence requirements:")
    for req in requirements:
        narrate.say_folded(f"    {req.id}: ", req.description)
    narrate.say(f"  [{label}] passages (showing up to 10 expanded):",
                forward=False)
    for n, passage in enumerate(clean_passages[:10], start=1):
        narrate.say(f"    [{n}/{len(clean_passages)}] {passage['id']}:",
                    forward=False)
        for ln in passage["text"].splitlines() or [""]:
            narrate.say(f"      | {ln}", forward=False)
    if len(clean_passages) > 10:
        narrate.say(f"  [{label}] … +{len(clean_passages) - 10} more "
                    f"passage(s) judged but not shown", forward=False)

    # Seed from the verdict cache: a passage already judged against this exact
    # question + requirement set + threshold + model is not re-scored.
    cache_keys: list[tuple | None] = []
    seeded: list[PassageVerdict] = []
    for passage in clean_passages:
        key = (_verdict_cache_key(question, requirements, passage, ask_verbatim,
                                  threshold) if use_cache else None)
        cache_keys.append(key)
        hit = _verdict_cache_lookup(key) if key is not None else None
        if hit is not None:
            seeded.append(hit.model_copy(update={"passage_id": passage["id"]}))
    if seeded:
        narrate.say(f"  [{label}] {len(seeded)} passage(s) already judged - "
                    "reusing cached verdicts")
        get_trace().log("evidence_judgment_cache_hit", label=label,
                        hits=len(seeded))

    merged = VerdictResponse(evidence_results=seeded)
    judged_any = bool(seeded)
    prompt_total = completion_total = 0
    calls: list[dict] = []

    by_pid = {v.passage_id: v for v in merged.evidence_results}
    todo = [p for p in clean_passages if p["id"] not in by_pid]

    async def _run(passage: dict[str, str]):
        # The SDK retries rate limits, 5xx, and transport failures with
        # backoff; a failure here is terminal for that passage and the keep
        # gate rejects it. One attempt keeps a hard outage from stalling the
        # tool call for a whole backoff budget.
        return await _judge_passage(
            active_client, label, question, requirements, passage,
            ask_verbatim, threshold, timeout, semaphore)

    active_client: object
    try:
        async with AsyncExitStack() as stack:
            active_client = client
            if active_client is None:
                active_client = await stack.enter_async_context(_make_client())
            if todo:
                started = time.perf_counter()
                results = await asyncio.gather(
                    *[_run(p) for p in todo], return_exceptions=True)
                elapsed = time.perf_counter() - started
            else:
                results = []
                elapsed = 0.0
    except Exception as exc:  # noqa: BLE001 - judge construction/transport
        logger.warning("verifier unavailable for question %r: %s",
                       question[:120], exc)
        narrate.say(f"  [{label}] judge unavailable ({exc}); continuing "
                    f"unverified")
        get_trace().log("evidence_judgment_unavailable", label=label,
                        error=str(exc)[:200])
        response = normalize_response(merged, requirements, clean_passages,
                                      check_verbatim=ask_verbatim)
        return _finish(label, question, response, judged=judged_any,
                       prompt_tokens=prompt_total,
                       completion_tokens=completion_total)

    for passage, result in zip(todo, results):
        if isinstance(result, BaseException):
            logger.warning("verifier failed for passage %s: %s",
                           passage["id"], result)
            narrate.say(f"  [{label}] judge failed for {passage['id']}: "
                        f"{result}")
            get_trace().log("evidence_judgment_error", label=label,
                            passage=passage["id"], error=str(result)[:200])
            continue
        verdict, usage = result
        judged_any = True
        prompt_total += usage["prompt"]
        completion_total += usage["completion"]
        calls.append(usage)
        get_trace().log("judge_call", label=label,
                        passage=passage["id"],
                        latency_ms=usage["latency_ms"],
                        prompt_tokens=usage["prompt"],
                        completion_tokens=usage["completion"])
        merged = _upsert(merged, verdict)

    if calls:
        narrate.say(
            f"  [{label}] judge usage: {len(calls)} call(s) in {elapsed:.1f}s | "
            f"input {prompt_total:,} | output {completion_total:,}"
        )
        get_trace().log(
            "evidence_judgment_usage", label=label, calls=len(calls),
            prompt_tokens=prompt_total, completion_tokens=completion_total,
            elapsed_ms=round(elapsed * 1000),
        )

    response = normalize_response(merged, requirements, clean_passages,
                                  check_verbatim=ask_verbatim)
    if use_cache:
        by_pid = {v.passage_id: v for v in response.evidence_results}
        for passage, key in zip(clean_passages, cache_keys):
            if key is None:
                continue
            verdict = by_pid.get(passage["id"])
            if verdict is None or verdict.reason == _REASON_UNSCORED:
                continue
            _verdict_cache_store(key, verdict)

    if not judged_any:
        for verdict in response.evidence_results:
            if verdict.reason == _REASON_UNSCORED:
                verdict.reason = _REASON_FAILED

    return _finish(label, question, response, judged=judged_any,
                   prompt_tokens=prompt_total,
                   completion_tokens=completion_total)


def _finish(
    label: str,
    question: str,
    response: VerdictResponse,
    *,
    judged: bool = True,
    tokens: int = 0,
    prompt_tokens: int = 0,
    cached_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> VerdictResponse:
    """Narrate the full grid, log it, apply the keep gate, return kept-only."""
    gated, rejected = _apply_keep_gate(response)
    gated.judged = judged
    gated.tokens = tokens or (prompt_tokens + completion_tokens)
    gated.prompt_tokens = prompt_tokens
    gated.cached_tokens = cached_tokens
    gated.completion_tokens = completion_tokens
    gated.reasoning_tokens = reasoning_tokens
    # Reasons for every judged passage (the rejected ones are not in
    # evidence_results any more, but the UI shows their verdict too). A hard
    # failure has no critic verdict, so it reports no reasons.
    if judged:
        gated.reasons = {
            v.passage_id: v.reason
            for v in response.evidence_results
            if v.passage_id and v.reason
        }
    _narrate_response(label, response, rejected)
    get_trace().log(
        "evidence_judgment", label=label, question=question[:160],
        verdicts={v.passage_id: {"intent": round(v.intent_score, 2),
                                 "coverage": v.coverage}
                  for v in response.evidence_results},
        kept=[v.passage_id for v in gated.evidence_results],
        rejected=sorted(rejected),
    )
    return gated


def _narrate_response(
    label: str, response: VerdictResponse, rejected: set[str] | None = None,
) -> None:
    """Render the verdicts as a terminal table: passage, reason, scores, status.

    Always prints every judged passage (unscored ones show zero intent with
    their reason), so a failed or all-zero judgment is still visible as a
    table instead of silence.
    """
    dropped = rejected or set()
    rows = []
    for scored in response.evidence_results:
        rows.append((
            scored.passage_id,
            scored.reason or "-",
            scored.intent_score,
            ",".join(scored.coverage) or "-",
            "REJECTED" if scored.passage_id in dropped else "kept",
        ))
    narrate.say(f"  [{label}] verdicts:")
    if rows:
        _print_verdict_table(rows)
    else:
        narrate.say("    (no verdicts)")
    if any(reason in (_REASON_UNSCORED, _REASON_FAILED)
           for _, reason, _, _, _ in rows):
        narrate.say(f"  [{label}] note: '{_REASON_UNSCORED}'/"
                    f"'{_REASON_FAILED}' = no verdict received for that passage;"
                    f" 0.00 with a reason = judged irrelevant")
    narrate.say(
        f"  [{label}] judgment: {len(rows) - len(dropped)} kept,"
        f" {len(dropped)} rejected of {len(rows)} passage(s)"
        f" (reject = empty coverage)"
    )


def _print_verdict_table(rows: list[tuple[str, str, float, str, str]]) -> None:
    """Fixed-width ASCII grid; the reason column wraps to terminal width."""
    term_w = shutil.get_terminal_size(fallback=(100, 24)).columns - 4
    cell_w = min(28, max(len("passage"), max(len(cell) for cell, _, _, _, _ in rows)))
    intent_w, status_w = len("intent"), len("REJECTED")
    coverage_w = min(24, max(len("coverage"),
                             max(len(cov) for _, _, _, cov, _ in rows)))
    reason_w = max(20, term_w - cell_w - intent_w - coverage_w - status_w - 15)
    widths = (cell_w, reason_w, intent_w, coverage_w, status_w)
    bar = "  +" + "+".join("-" * (w + 2) for w in widths) + "+"
    header = (f"  | {'passage':<{cell_w}} | {'reason':<{reason_w}}"
              f" | {'intent':>{intent_w}} | {'coverage':<{coverage_w}}"
              f" | {'status':<{status_w}} |")
    narrate.say(bar, forward=False)
    narrate.say(header, forward=False)
    narrate.say(bar, forward=False)
    for cell, reason, intent, coverage, status in rows:
        if len(cell) > cell_w:
            cell = cell[: max(1, cell_w - 1)] + "…"
        if len(coverage) > coverage_w:
            coverage = coverage[: max(1, coverage_w - 1)] + "…"
        lines = textwrap.wrap(reason, width=reason_w) or [""]
        narrate.say(f"  | {cell:<{cell_w}} | {lines[0]:<{reason_w}}"
                    f" | {intent:>{intent_w}.2f} | {coverage:<{coverage_w}}"
                    f" | {status:<{status_w}} |", forward=False)
        for extra in lines[1:]:
            narrate.say(f"  | {'':<{cell_w}} | {extra:<{reason_w}}"
                        f" | {'':>{intent_w}} | {'':<{coverage_w}}"
                        f" | {'':<{status_w}} |", forward=False)
    narrate.say(bar, forward=False)
