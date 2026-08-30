"""Unit tests for UMLS term filtering (_usable_term)."""
from src.agentic.umls_tool import _usable_term


def test_keeps_good_terms():
    assert _usable_term("Coronary Vasospasm")
    assert _usable_term("pro-enkephalin", "proenkephalin")
    assert _usable_term("Acute Kidney Injury", "AKI")


def test_drops_code_like_terms():
    assert not _usable_term("AKI 001", "AKI")
    assert not _usable_term("AKI-001", "AKI")
    assert not _usable_term("AKI001", "AKI")


def test_drops_generic_words():
    assert not _usable_term("Active Site")
    assert not _usable_term("Social Stratification")
    assert not _usable_term("site")


def test_drops_inverted_and_echo():
    assert not _usable_term("Arterial spasm, radial")
    assert not _usable_term("radial artery vasospasm", "radial artery vasospasm")


def test_drops_empty_and_short():
    assert not _usable_term("")
    assert not _usable_term("ak")
    assert not _usable_term("x y z " * 20)  # > 40 chars
