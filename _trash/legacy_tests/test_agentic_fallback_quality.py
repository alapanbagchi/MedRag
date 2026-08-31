"""Fallback entity quality: abstract nouns and generic words are excluded."""
from src.agentic.planner import _fallback_entities


def test_fallback_excludes_abstract_nouns():
    ents = _fallback_entities("comparison of PENK and serum creatinine for AKI prediction")
    texts = [e.text.casefold() for e in ents]
    # PENK and AKI (acronyms) survive; generic abstracts do not.
    assert "penk" in " ".join(texts) or "PENK" in [e.text for e in ents]
    assert not any("prediction" in t for t in texts)
    assert not any(t == "serum" for t in texts)


def test_fallback_keeps_real_concepts():
    ents = _fallback_entities("radial artery vasospasm during interventional radiology")
    texts = " ".join(e.text.lower() for e in ents)
    assert "radial artery vasospasm" in texts
