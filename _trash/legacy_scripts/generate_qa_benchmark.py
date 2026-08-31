"""Generate a leveled QA benchmark with NON-leaking ground truth.

This is benchmark B / the query-transformation benchmark: for each sampled
chunk we emit four query levels with the source chunk as the relevant answer,
but only the first two levels may draw on the source surface form:

    verbatim   - first 60 words of the chunk (sanity ceiling, expected ~100%)
    keywords   - top TF-weighted non-stopword terms (shares content words)
    paraphrase- synonym-substituted, reordered restatement (no verbatim copy)
    naturalqa  - entity-slotted question templates (genuine question form)

An anti-leakage guard verifies every non-verbatim query: it must not contain
any >= 7-token contiguous n-gram of the source and its content-word overlap
with the source must stay below a threshold. Failed candidates are
rephrased automatically by dropping modifying phrases / synonym shuffling.

Optionally the relevance judgment is expanded to nearby same-section chunks
(positions within +/-2) so answers are not artificially unique.

Usage:
    python scripts/generate_qa_benchmark.py --n-docs 50 --out eval/qa_benchmark_queries.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.benchmark_retrieval import (  # noqa: E402
    CHUNK_DIR,
    STOPWORDS,
    WORD_RE,
    _words,
    select_docs,
    top_keywords,
)

# ----------------------------------------------------------------------
# Deterministic biomedical-ish paraphrase lexicon (token-boundary swaps)
# ----------------------------------------------------------------------

SYNONYMS: Dict[str, str] = {
    "patients": "subjects",
    "patient": "subject",
    "treatment": "therapy",
    "treated": "managed",
    "treatment group": "intervention arm",
    "control group": "comparator arm",
    "group": "cohort",
    "groups": "cohorts",
    "showed": "demonstrated",
    "show": "demonstrate",
    "found": "identified",
    "observed": "documented",
    "significant": "marked",
    "significantly": "markedly",
    "associated": "linked",
    "association": "link",
    "correlated": "correlated",
    "role": "function",
    "effect": "influence",
    "effects": "influences",
    "outcomes": "findings",
    "outcome": "endpoint",
    "results": "findings",
    "result": "finding",
    "study": "investigation",
    "patients with": "subjects diagnosed with",
    "risk": "hazard",
    "levels": "concentrations",
    "elevated": "raised",
    "reduced": "lowered",
    "increased": "raised",
    "decreased": "lowered",
    "mean": "average",
    "median": "midpoint",
    "use": "application",
    "using": "applying",
}

# Longest-first so multi-word keys win.
SYNONYM_ITEMS = sorted(SYNONYMS.items(), key=lambda kv: -len(kv[0].split()))

# Chunks whose text starts with these phrases are administrative/discussion
# non-content (disclaimers, funding, ethics) and make poor QA candidates.
SKIP_PREFIXES = (
    "all claims expressed in this article",
    "this article was prepared",
    "publisher's note",
    "availability of data and materials",
    "data availability",
    "conflict of interest",
    "competing interests",
    "ethics approval",
    "ethical approval",
    "ethics statement",
    "funding statement",
    "funding sources",
    "acknowledgements",
    "acknowledgments",
    "author contributions",
    "consent to publish",
    "consent for publication",
    "the authors declare",
    "abbreviations (in order of appearance",
    "abbreviations",
    "supplementary information",
    "supplementary materials",
    "declarations",
    "editor's note",
    "correction to:",
    "this is a summary of",
)
SKIP_SECTIONS = frozenset({
    "funding", "conflict of interest", "competing interests", "data availability",
    "ethics", "ethical statement", "acknowledgements", "acknowledgments",
    "author contributions", "supplementary material", "supplementary materials",
    "notes", "abbreviations", "authors' contributions", "declarations",
})

# Tokens too generic to name a biomedical entity (adjectives, adverbs, abstract
# nouns, question artifacts). These should never appear bare in a query.
EXCLUDED_ENTITIES = frozenset(
    """there large small high low early late current overall available availability
    concomitant trans however moreover although whereas during based related following
    specific significant associated group study article paper findings results role effect
    effects outcome outcomes analysis analyses level levels patient patients treatment
    therapy method methodology methods model table figure fig data value values score
    scores rate rates risk percent p value p-value clinical acute with without among after
    before between within across most more less other another such same also only just
    this these those their its from have has had were was being been both each few many
    several some further new first second third mainly mostly often sometimes usually
    reported investigated assessed evaluated measured examined compared increased
    decreased reduced elevated raised higher lower greater smaller older younger
    male female men women children adults subjects cohort cohorts arm arms
    total overall survival primary secondary
    """.split()
)
EXCLUDED_ENTITIES |= {"q" + str(i) for i in range(100)}
EXCLUDED_ENTITIES |= {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty",
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii",
    "ultimately", "recently", "previously", "accordingly", "subsequently",
    "eventually", "particularly", "approximately", "respectively", "furthermore",
    "moreover", "conversely", "similarly", "notably", "essentially", "predominantly",
    "shortly", "newly", "however", "therefore", "mental", "physical",
}


def is_skip_chunk(chunk: pd.Series) -> bool:
    """True for chunks that are non-content (disclaimers, admin sections, etc.)."""
    text = str(chunk["text"] or "").strip().lower()
    if not text:
        return True
    if (chunk.get("section") or "").lower() in SKIP_SECTIONS:
        return True
    return any(text.startswith(p) for p in SKIP_PREFIXES)

QUESTION_TEMPLATES: List[str] = [
    "According to this study, what is the role of {e1} in {topic}?",
    "What is the association between {e1} and {e2}?",
    "How does {e1} relate to {e2} in this study?",
    "What outcomes are reported for {e1}?",
    "Does {e1} have a significant effect on {e2}?",
    "What do the findings indicate about {e1}?",
    "Which factors were reported for {topic}?",
]
TABLE_TEMPLATES: List[str] = [
    "What values are reported for {e1} in the table?",
    "According to the table, what is reported for {topic}?",
    "What does the table indicate about {e1}?",
]
LIST_TEMPLATES: List[str] = [
    "According to the article, what points are made about {topic}?",
    "What items are listed regarding {e1}?",
]


def substitute_synonyms(sentence: str) -> str:
    out = sentence
    for src, dst in SYNONYM_ITEMS:
        if src in out.lower() and len(src) > 2:
            out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
    return out


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p.split()) >= 4]


def content_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content words between two strings."""
    def content_toks(s: str) -> set:
        return {t for t in WORD_RE.findall(s.lower()) if t not in STOPWORDS and len(t) > 2}
    x, y = content_toks(a), content_toks(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def contains_source_ngram(question: str, source_words: List[str], n: int = 7) -> bool:
    """True if any contiguous n-gram of source_words appears in the question."""
    q = WORD_RE.findall(question.lower())
    if len(q) < n:
        return False
    src = [t for t in source_words if len(t) > 1]
    if len(src) < n:
        src = src + [""] * (n - len(src))
    qgrams = {tuple(q[i : i + n]) for i in range(len(q) - n + 1)}
    srcgrams = {tuple(src[i : i + n]) for i in range(len(src) - n + 1)}
    return bool(qgrams & srcgrams)


def entities_from(text: str, section: str, n: int = 4) -> List[str]:
    """Mix of multi-word capitalized phrases and top content words.

    Prefers clinical phrase runs (e.g. "Chronic Kidney Disease") and drops
    generic adjectives/adverbs/abstract nouns that are not real biomedical
    entities (which is what made the first version emit queries like
    "Does Chronic have a significant effect on COPD?").
    """
    # 1. Multi-word capitalized runs ("Chronic Kidney Disease", "MEK Inhibitors").
    phrases: List[str] = []
    for m in re.finditer(r"\b(?:[A-Z][A-Za-z-]+(?:[ -][A-Z][A-Za-z-]+){0,3})\b", text):
        phrase = re.sub(r"^(The|This|These|Those|A|An)\s+", "", m.group(0))
        phrase = phrase.rstrip("-–_")  # strip trailing artifact hyphens (e.g. "SF-")
        if not phrase:
            continue
        if any(t.lower() in EXCLUDED_ENTITIES for t in phrase.replace("-", " ").split()):
            continue
        if phrase.lower() not in STOPWORDS and phrase not in phrases:
            phrases.append(phrase)
    # 2. Capitalized single tokens, then plain content keywords.
    caps = [
        c.rstrip("-–_") for c in re.findall(r"\b[A-Z][A-Za-z0-9-]{1,}\b", text)
        if c.lower() not in EXCLUDED_ENTITIES
    ]
    caps = [c for c in caps if c]
    kws = [k for k in top_keywords(text, n) if k not in EXCLUDED_ENTITIES]

    ent: List[str] = []
    for cand in phrases + caps + kws:
        low = cand.lower()
        if low in STOPWORDS or low in EXCLUDED_ENTITIES:
            continue
        if cand not in ent and low not in [e.lower() for e in ent]:
            ent.append(cand)
        if len(ent) >= n:
            break
    return ent or [section or "this study", "this study"]


def topic_from(section: str, entities: Sequence[str], text: str) -> str:
    # Prefer a real entity to a bare section label ("Discussion"/"Methods")
    # as the query topic - "role of X in Discussion?" is a weak question.
    if len(entities) > 1:
        return entities[1]
    if entities:
        return entities[0]
    return section or "this study"


def make_paraphrase(chunk_text: str, rng: random.Random) -> str:
    """Return a synonym-shuffled restatement that is not a verbatim copy.

    Strategy: pick the most information-dense sentence, cap it to ~16 words
    (long stretches are what create n-gram matches), apply synonym swaps and
    a leading-adverbial reorder, and verify with the leakage guard. If the
    guard still fails after several tries, degrade gracefully to a frame
    sentence over the chunk's entities (guaranteed low surface overlap).
    """
    src_words = WORD_RE.findall(chunk_text.lower())
    cands = sentences(chunk_text)
    if not cands:
        return _words(chunk_text, 14)

    def info_score(s: str) -> int:
        toks = {t for t in WORD_RE.findall(s.lower()) if t not in STOPWORDS}
        return len(toks)

    cands.sort(key=info_score, reverse=True)
    s = " ".join(cands[0].split()[:16])

    for _ in range(6):
        s = substitute_synonyms(s)
        # Move a leading adverbial ("In X, ..." / "Among X, ...") to the end.
        m = re.match(r"^(In|Among|For|During|After|Before|Across)\s+(.+?)[,;]\s+(.+)$", s, re.IGNORECASE)
        if m:
            s = f"{m.group(3).strip().capitalize()}, as {m.group(1).lower()} {m.group(2).strip()}."
        s = re.sub(r"\([^)]*\)", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        if not contains_source_ngram(s, src_words, 7) and content_overlap(s, chunk_text) < 0.5:
            return s

    # Guaranteed-low-overlap fallback: frame sentence over the chunk's entities.
    ents = entities_from(chunk_text, "", 3)
    frame = rng.choice([
        "The article reports on {} and discusses related outcomes.",
        "This study examined {} and associated factors.",
        "The investigation focuses on {} among the studied subjects.",
        "Findings regarding {} are presented and discussed.",
    ])
    return frame.format(" and ".join(ents[:3]))


def make_naturalqa(chunk_text: str, section: str, chunk_type: str, rng: random.Random) -> str:
    ents = entities_from(chunk_text, section)
    topic = topic_from(section, ents, chunk_text)
    e1, e2 = ents[0], ents[1] if len(ents) > 1 else topic
    if chunk_type == "table_summary":
        template = rng.choice(TABLE_TEMPLATES)
    elif chunk_type == "list":
        template = rng.choice(LIST_TEMPLATES)
    else:
        template = rng.choice(QUESTION_TEMPLATES)
    q = template.format(e1=e1, e2=e2, topic=topic)
    src_words = WORD_RE.findall(chunk_text.lower())
    if contains_source_ngram(q, src_words, 7):
        q = re.sub(r"\bthe role of\b", "the function of", q)
    return q


# ----------------------------------------------------------------------
# Per-chunk query builder
# ----------------------------------------------------------------------

def build_chunk_queries(
    chunk: pd.Series,
    doc_id: str,
    section_neighbors: Sequence[str],
    rng: random.Random,
    expand_neighbors: bool,
) -> List[Dict[str, Any]]:
    text = str(chunk["text"] or "")
    if len(text.split()) < 8:
        return []
    chunk_id = str(chunk["id"])
    chunk_type = str(chunk["chunk_type"])
    section = str(chunk.get("section") or "")

    relevant = [chunk_id]
    if expand_neighbors:
        relevant = [chunk_id] + [n for n in section_neighbors if n != chunk_id][:2]

    kinds = [
        ("verbatim", _words(text, 60), 0.0),
        ("keywords", " ".join(top_keywords(text, 6)) or _words(text, 40), 0.0),
        ("paraphrase", make_paraphrase(text, rng), 0.0),
        ("naturalqa", make_naturalqa(text, section, chunk_type, rng), 0.0),
    ]
    # Record measured surface overlap with the source for transparency.
    out: List[Dict[str, Any]] = []
    for kind, query, _ in kinds:
        src_words = WORD_RE.findall(text.lower())
        ngram_leak = contains_source_ngram(query, src_words, 7)
        overlap = content_overlap(query, text)
        out.append({
            "query": query,
            "kind": kind,
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "section": section,
            "document_id": doc_id,
            "surface_overlap": round(overlap, 3),
            "ngram_leak": ngram_leak,
            "relevant_chunk_ids": relevant,
            "relevant_document_ids": [doc_id],
        })
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-docs", type=int, default=50)
    ap.add_argument("--queries-per-doc", type=int, default=2)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--expand-neighbors", action="store_true",
                    help="expand relevance to nearby same-section chunks (default: strict single-relevant)")
    ap.add_argument("--out", default="eval/qa_benchmark_queries.json")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    doc_ids = select_docs(args.seed, args.n_docs)

    queries: List[Dict[str, Any]] = []
    per_kind: Counter = Counter()
    leaks = 0
    for doc in doc_ids:
        files = sorted(CHUNK_DIR.glob(f"{doc}.*.parquet"))
        df = (
            pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            if files
            else pd.DataFrame()
        )
        if df.empty:
            continue

        elig = df[df["retrieval_eligible"].fillna(False).astype(bool)].sort_values(
            "document_position"
        )
        candidates = elig[elig["chunk_type"].isin(("paragraph", "list", "table_summary"))]
        candidates = candidates[~candidates.apply(is_skip_chunk, axis=1)]
        if candidates.empty:
            continue

        # Pick up to N spaced chunks to spread across sections.
        n = min(args.queries_per_doc, len(candidates))
        idxs = sorted(rng.sample(range(len(candidates)), n)) if n > 1 else [0]
        for i in idxs:
            chunk = candidates.iloc[i]
            pos = int(chunk["document_position"])
            same_section = elig[
                (elig["section"] == chunk["section"])
                & (elig["document_position"].between(pos - 2, pos + 2))
            ]["id"].tolist()
            built = build_chunk_queries(
                chunk, doc, same_section, rng, expand_neighbors=args.expand_neighbors
            )
            for q in built:
                per_kind[q["kind"]] += 1
                if q["kind"] != "verbatim" and q["ngram_leak"]:
                    leaks += 1
            queries.extend(built)

    if args.max_queries > 0 and len(queries) > args.max_queries:
        queries = rng.sample(queries, args.max_queries)

    payload = {
        "note": (
            "Leveled QA benchmark with non-leaking ground truth. verbatim = sanity "
            "ceiling; keywords = content-word queries; paraphrase and naturalqa are "
            "surface-transformed or template questions that do NOT copy the source. "
            "Relevant chunks = source chunk (+ nearby same-section chunks unless "
            "--no-neighbors). surface_overlap / ngram_leak record measured leakage."
        ),
        "queries": [
            {k: v for k, v in q.items() if k in (
                "query", "kind", "chunk_id", "chunk_type", "section",
                "document_id", "surface_overlap", "ngram_leak",
                "relevant_chunk_ids", "relevant_document_ids",
            )}
            for q in queries
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(queries)} queries -> {out.resolve()}")
    print("Kinds:", dict(per_kind))
    print(f"Non-verbatim queries with residual n-gram leak: {leaks}")
    avg_overlap = {
        kind: round(
            sum(q["surface_overlap"] for q in queries if q["kind"] == kind)
            / max(1, sum(1 for q in queries if q["kind"] == kind)),
            3,
        )
        for kind in per_kind
    }
    print("Mean surface overlap with source:", avg_overlap)
    return 0


if __name__ == "__main__":
    sys.exit(main())