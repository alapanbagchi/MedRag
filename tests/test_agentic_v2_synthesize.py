"""Unit tests for agentic v2 final synthesis evidence selection (no LLM)."""
from src.agentic_v2.state import (
    EvidenceQuality,
    ResearchState,
    VerifiedEvidence,
)
from src.agentic_v2.synthesize import _synthesis_prompt, usable_evidence


def _ev(quality, chunk):
    return VerifiedEvidence(
        id=f"E-H1-{chunk}", objective_id="H1", document_id="PMC1",
        chunk_id=chunk, section="Results", excerpt="supporting text",
        quality=quality, support="supports", confidence=0.8,
    )


def _state_with_all_qualities() -> ResearchState:
    state = ResearchState(question="q")
    state.evidence = [
        _ev(EvidenceQuality.DIRECT, "c-relevant"),
        _ev(EvidenceQuality.INDIRECT, "c-partial"),
        _ev(EvidenceQuality.BACKGROUND, "c-rejected"),
        _ev(EvidenceQuality.UNKNOWN, "c-unknown"),
    ]
    return state


def test_usable_evidence_keeps_direct_and_indirect_only():
    usable = usable_evidence(_state_with_all_qualities())
    assert [e.chunk_id for e in usable] == ["c-relevant", "c-partial"]


def test_synthesis_prompt_excludes_rejected_and_unknown():
    prompt = _synthesis_prompt(_state_with_all_qualities())
    # relevant + partially relevant feed the answer...
    assert "c-relevant" in prompt
    assert "c-partial" in prompt
    # ...rejected (BACKGROUND) and failed (UNKNOWN) verifications do not
    assert "c-rejected" not in prompt
    assert "c-unknown" not in prompt
    assert "partially relevant only" in prompt


def test_usable_evidence_empty_when_nothing_verified():
    state = ResearchState(question="q")
    state.evidence = [_ev(EvidenceQuality.BACKGROUND, "c1")]
    assert usable_evidence(state) == []


# -- regression: a definition buried deep in a table unit must reach the
#    synthesizer (previously `_evidence_line` head-sliced excerpts at 400 chars,
#    so "No definition of cardiac arrest was found" despite relevant units). --

_TABLE_WITH_DEFINITION = (
    "[Table ehae724-T2]\n"
    "Table 2: Acute coronary syndrome/percutaneous coronary intervention—clinical "
    "outcomes and their definitions\n"
    "Row: Acute coronary syndrome/PCI: Level 1 variables\n"
    "column_1: Acute coronary syndrome/PCI: Level 1 variables\n"
    "Row: Acute kidney injury requiring renal replacement therapy\n"
    "column_1: Renal replacement therapy includes ultrafiltration (haemofiltration), "
    "haemodialysis or peritoneal dialysis.50\n"
    "Row: Cardiac arrest\n"
    "column_1: Cardiac arrest is defined as a verified sudden cessation of cardiac "
    "activity causing unresponsiveness, absence of normal breathing and no signs of "
    "circulation (excluding syncope or profound vagally mediated bradycardia) with "
    "ventricular fibrillation, rapid ventricular tachycardia or bradycardia resulting "
    "in loss of consciousness, pulseless electrical activity, or asystole as the "
    "major causes.\n"
    "Row: Heart failure hospitalisation\n"
    "column_1: Hospital admission primarily due to heart failure.\n"
)


def test_synthesis_prompt_keeps_deep_anchored_excerpt():
    """Evidence whose supporting text starts past the old 400-char head slice
    must still reach the synthesizer verbatim (bounded)."""
    # the definition really sits past the old 400-char slice
    assert _TABLE_WITH_DEFINITION.find("Cardiac arrest is defined as") > 400

    state = ResearchState(question="What is a cardiac arrest?")
    state.evidence = [VerifiedEvidence(
        id="E-H1-1", objective_id="H1", document_id="PMC11704390",
        chunk_id="c-def", section="Results", excerpt=_TABLE_WITH_DEFINITION,
        quality=EvidenceQuality.DIRECT, support="supports", confidence=1.0,
    )]
    prompt = _synthesis_prompt(state)
    assert "Cardiac arrest is defined as" in prompt
    assert "verified sudden cessation of cardiac activity" in prompt


def test_display_excerpt_marks_truncation_never_silent_cut():
    from src.agentic_v2.synthesize import _display_excerpt

    short = _display_excerpt("a b " * 20)
    assert "..." not in short
    assert short == ("a b " * 20).strip()

    long = _display_excerpt("word " * 2000)
    assert long.endswith("... [excerpt truncated]")
    assert len(long) <= 1600 + len("... [excerpt truncated]")
