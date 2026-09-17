'''Skill-library tests: the synthesizer's writing skills.

Skills are plain SKILL.md prompt files under backend/skills/writing/.
A skill must ALWAYS be used (default concise-clinical) and the humanizer
pass is mandatory on every composed prompt.
'''

from __future__ import annotations

import pytest

from src.lib.skills import (
    DEFAULT_MODE,
    MODES,
    build_synthesis_prompt,
    detect_writing_mode,
    list_skills,
    load_skill,
)


def test_list_skills_has_expected_set():
    names = set(list_skills())
    assert {'base', 'humanizer'} <= names
    assert {'concise-clinical', 'structured-review',
            'patient-facing', 'evidence-critique'} <= names


def test_load_skill_strips_frontmatter():
    text = load_skill('humanizer')
    assert 'HUMANIZER' in text
    assert 'description:' not in text


def test_load_skill_unknown_raises():
    with pytest.raises(FileNotFoundError):
        load_skill('does-not-exist')


def test_detect_writing_mode_picks_styles():
    assert detect_writing_mode('How does diet affect hypertension?') == 'concise-clinical'
    assert (detect_writing_mode('Review the evidence on DASH diet')
            == 'structured-review')
    assert (detect_writing_mode('I have high blood pressure, what should I know')
            == 'patient-facing')
    assert (detect_writing_mode('How strong is the evidence behind statins?')
            == 'evidence-critique')
    assert detect_writing_mode('') == DEFAULT_MODE
    assert detect_writing_mode(None) == DEFAULT_MODE


def test_build_synthesis_prompt_always_composes_all_layers():
    for mode in MODES:
        p = build_synthesis_prompt(mode, '')
        assert 'MEDRAG EVIDENCE DISCIPLINE' in p        # base always
        assert 'HUMANIZER' in p                          # humanizer mandatory
        assert 'WRITING STYLE' in p                      # one style always
    # Unknown mode falls back to the default style, never fails.
    p = build_synthesis_prompt('nonsense-mode', '')
    assert 'CONCISE CLINICAL' in p


def test_build_synthesis_prompt_defaults_from_question():
    p = build_synthesis_prompt(None, 'I have high blood pressure')
    assert 'PATIENT-FACING' in p
    p2 = build_synthesis_prompt()  # no args at all
    assert 'CONCISE CLINICAL' in p2


# --- Contract pins: robustness rules every composed prompt must carry ---

def test_every_composed_prompt_carries_the_presentation_contract():
    for mode in MODES:
        p = build_synthesis_prompt(mode, '')
        assert '## Presentation contract (markdown, every style)' in p
        assert '## Drafting procedure (do this in order)' in p
        # each style contributes its own concrete presentation line
        assert '- Presentation:' in p


def test_base_pins_the_markdown_presentation_rules():
    base = load_skill('base')
    for needle in ('GFM table', 'wall of prose', 'No raw HTML',
                   'One idea per paragraph'):
        assert needle in base
    # the References example and the guard agree on the separator
    assert '[Pn] Title' in base
    assert 'em dash' in base.lower()


def test_humanizer_pins_banned_tells_and_worked_rewrites():
    h = load_skill('humanizer').lower()
    for needle in ('no em dashes in prose', 'worked rewrites',
                   'not only x but also y', 'stacked qualifiers',
                   'read it aloud'):
        assert needle in h


def test_prompt_layers_are_ordered_base_then_style_then_humanizer():
    p = build_synthesis_prompt('structured-review', '')
    i_base = p.index('MEDRAG EVIDENCE DISCIPLINE')
    i_style = p.index('WRITING STYLE')
    i_human = p.index('HUMANIZER')
    assert i_base < i_style < i_human

