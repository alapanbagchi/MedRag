"""Bridge citation reconciliation tests (repair_citations semantics)."""

from __future__ import annotations

import asyncio

from src.agents.bridge import _cited_text, _doc_to_index
from src.agents.agents.stages import _answer_to_prose


SOURCES = [{"id": "PMC1"}, {"id": "PMC2"}, {"id": "PMC3"}]


def test_markers_resolve_to_verified_source_numbers():
    d2i = _doc_to_index(SOURCES)
    ans = "Vitamin D reduced SBP.\u3014cite:PMC2\u3015 Another claim.\u3014cite:PMC1\u3015"
    out = _cited_text(ans, [], ["PMC1", "PMC2", "PMC3"], doc_to_index=d2i)
    assert out == "Vitamin D reduced SBP. [2] Another claim. [1]"


def test_unverified_and_hallucinated_markers_dropped():
    d2i = _doc_to_index(SOURCES)
    ans = "Claim.\u3014cite:PMC9\u3015 Claim2.\u3014cite:PMC2\u3015"
    out = _cited_text(ans, [], ["PMC1", "PMC2", "PMC3"], doc_to_index=d2i)
    # PMC9 is not a verified source -> dropped; PMC2 -> [2]
    assert "PMC9" not in out
    assert "[2]" in out


def test_answer_to_prose_embeds_markers_from_citations():
    data = {
        "summary": "s",
        "sections": [{
            "heading": "BP",
            "body": "Vitamin D lowered BP.",
            "citations": [{"requirement_id": "R1", "document_id": "PMC77",
                           "support": "supports"},
                          {"requirement_id": "R1", "document_id": "PMC77",
                           "support": "supports"}],  # dup deduped
        }],
        "limitations": [],
        "unresolved_gaps": [],
    }
    prose = _answer_to_prose(data)
    assert "\u3014cite:PMC77\u3015" in prose
    # duplicate documents collapse to a single marker
    assert prose.count("\u3014cite:PMC77\u3015") == 1


def test_answer_to_prose_dedups_multiple_citations():
    data = {
        "sections": [{
            "heading": "X",
            "body": "Text.",
            "citations": [{"document_id": "PMC1"}, {"document_id": "PMC2"},
                          {"document_id": "PMC1"}],
        }],
    }
    prose = _answer_to_prose(data)
    assert "\u3014cite:PMC1\u3015" in prose
    assert "\u3014cite:PMC2\u3015" in prose
    assert prose.count("\u3014cite:PMC1\u3015") == 1




# ---------------------------------------------------------------------------
# Web-source highlightable URLs (Text Fragments) + inline resolved contradictions
# ---------------------------------------------------------------------------

def test_web_source_url_carries_text_fragment():
    """A web source's URL must open the REAL site with the cited passage
    scroll-to + highlighted via the browser-native:#:~:text= fragment."""
    from src.agents.bridge import _src, _text_fragment_url

    url, frag = _text_fragment_url(
        "https://www.who.int/factsheet",
        "The DASH diet reduced systolic blood pressure by 11 mmHg in "
        "hypertensive adults in an 8-week randomized controlled trial.",
    )
    assert url.startswith("https://www.who.int/factsheet#:~:text=")
    assert "%20" in url.split("#:~:text=")[1]  # spaces percent-encoded
    assert "mmHg" in frag


def test_web_source_dict_exposes_highlight_and_isWeb():
    from types import SimpleNamespace

    from src.agents.bridge import _src

    it = SimpleNamespace(
        document_id="web:who", source_url="https://www.who.int/x",
        text="The DASH diet lowered systolic blood pressure by 11 mmHg.",
        trust="trusted", journal="WHO", confidence_float=0.9,
    )
    s = _src(it)
    assert s["isWeb"] is True
    assert s["pmcid"] in ("", None)
    assert "#:~:text=" in s["url"]
    assert s["highlight"]


def test_pmc_source_url_unchanged():
    from types import SimpleNamespace

    from src.agents.bridge import _src

    it = SimpleNamespace(document_id="PMC10262995", source_url="",
                         text="x" * 100, trust="", journal="", confidence_float=None)
    s = _src(it)
    assert s["isWeb"] is False
    assert s["url"] == "https://pmc.ncbi.nlm.nih.gov/articles/PMC10262995/"
    assert not s.get("highlight")


def test_synthesize_prompt_tells_resolved_contradictions_inline():
    """The synthesize prompt instructs the model to WEAVE resolved
    contradictions into the relevant section body, not a separate list."""
    from src.agents.agents.stages import _synthesize_prompt
    from src.agents.state import (
        Contradiction,
        ContradictionKind,
        ResolutionOutcome,
        ResolutionStatus,
        XDeepRunState,
    )

    c1 = Contradiction(id="C1", claim="Dose conflict",
                       kind=ContradictionKind.DIRECT_CONFLICT)
    c1.resolution = ResolutionOutcome(
        status=ResolutionStatus.RESOLVED,
        explanation="benefit only at low baseline vitamin D")
    run = XDeepRunState(run_id="r", question="q", contradictions=[c1])
    p = _synthesize_prompt(run)
    # the instruction to fold INLINE is present
    assert "INLINE" in p
    assert "separate resolved_contradictions list" in p
    # the resolved explanation is fed to the model so it can write it inline
    assert "benefit only at low baseline vitamin D" in p
