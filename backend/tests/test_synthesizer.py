'''Synthesizer unit tests: tool-free agent, question + all passages.

The synthesizer no longer reads the ledger through tools: the runner
collects every judge-kept passage and renders them (with [Pn] refs)
into the single user message. These tests pin that contract.
'''

from __future__ import annotations

import pytest

from src.agents.synthesizer import (
    build_agent,
    build_synthesis_message,
    collect_evidence,
    render_proofs,
)
from src.runstate.models import EvidenceRecord
from src.runstate.store import RunStore, get_store, set_store


@pytest.fixture()
def store(tmp_path):
    return RunStore(tmp_path / 'chats')


@pytest.fixture(autouse=True)
def _reset_global_store():
    set_store(None)
    yield
    set_store(None)


async def _seed(store):
    await store.create_turn('c1', 't1', 'Does the DASH diet help?')
    await store.record_plan('c1', [
        {'id': 'T1', 'question': 'dash evidence?',
         'evidence_requirements': [
             {'id': 'E1', 'description': 'blood pressure effect'}]},
        {'id': 'T2', 'question': 'sodium?',
         'evidence_requirements': [
             {'id': 'E2', 'description': 'sodium and hypertension'}]},
    ])
    await store.record_evidence('c1', [
        EvidenceRecord(passage_id='chunk-a', requirement_ids=['E1'],
                       title='Sodium paper', url='https://a.example/',
                       text='Sodium reduction lowers blood pressure.')
    ], task_id='T1')
    await store.record_evidence('c1', [
        EvidenceRecord(passage_id='chunk-b', requirement_ids=['E1'],
                       title='DASH trial', section='Methods',
                       text='The DASH diet markedly lowers systolic pressure.')
    ], task_id='T1')


async def test_collect_evidence_returns_all_passages_sorted(store):
    set_store(store)
    await _seed(store)
    evidence, uncovered = collect_evidence('c1')
    assert [e['ref'] for e in evidence] == ['P1', 'P2']
    assert evidence[0]['title'] == 'Sodium paper'
    assert evidence[0]['url'] == 'https://a.example/'
    assert 'Sodium reduction lowers blood pressure' in evidence[0]['text']
    assert evidence[1]['section'] == 'Methods'
    # E2 has zero verified support: declared uncovered, never hidden.
    assert uncovered == ['E2: sodium and hypertension']


async def test_collect_evidence_missing_chat_or_turn(store):
    set_store(store)
    assert collect_evidence('nope') == ([], [])
    await store.create_turn('c1', 't1', 'Q?')
    assert collect_evidence('c1') == ([], [])


def test_render_proofs_question_plus_all_passages():
    evidence = [
        {'ref': 'P1', 'title': 'Sodium paper', 'url': 'https://a.example/',
         'document_id': '', 'section': '', 'text': 'Sodium lowers pressure.',
         'requirement_ids': ['E1']},
        {'ref': 'P2', 'title': 'DASH trial', 'url': '', 'document_id': 'pmc123',
         'section': 'Results', 'text': 'DASH lowers systolic pressure.',
         'requirement_ids': ['E1']},
    ]
    msg = render_proofs('Does DASH help?', evidence, ['E2: sodium dose'])
    assert 'QUESTION' in msg and 'Does DASH help?' in msg
    assert '[P1] Sodium paper — https://a.example/' in msg
    assert 'Sodium lowers pressure.' in msg
    # No url -> the document id resolves the reference.
    assert '[P2] DASH trial — pmc123' in msg
    assert 'section: Results' in msg
    assert '- E2: sodium dose' in msg


def test_render_proofs_empty_evidence_and_no_gaps():
    msg = render_proofs('Q?', [], [])
    assert '(no verified evidence was collected)' in msg
    assert '(none — every requirement has verified support)' in msg


async def test_build_synthesis_message_wires_collect_and_render(store):
    set_store(store)
    await _seed(store)
    msg = build_synthesis_message('Does the DASH diet help?', 'c1')
    assert 'Does the DASH diet help?' in msg
    assert '[P1] Sodium paper — https://a.example/' in msg
    assert 'DASH diet markedly lowers systolic pressure' in msg
    assert '- E2: sodium and hypertension' in msg


def test_collect_evidence_never_raises_on_bad_store():
    assert collect_evidence('') == ([], [])


def test_render_proofs_caps_passage_text(monkeypatch):
    """The prompt is bounded per passage and in total, but every ref and its
    head line survives so the citation pool is never lost."""
    monkeypatch.setenv('SYNTH_MAX_PASSAGE_CHARS', '100')
    monkeypatch.setenv('SYNTH_MAX_TOTAL_CHARS', '150')
    evidence = [
        {'ref': 'P1', 'title': 't1', 'url': '', 'document_id': 'd1',
         'section': 's1', 'text': 'x' * 500, 'requirement_ids': ['E1']},
        {'ref': 'P2', 'title': 't2', 'url': '', 'document_id': 'd2',
         'section': 's2', 'text': 'y' * 500, 'requirement_ids': ['E1']},
        {'ref': 'P3', 'title': 't3', 'url': '', 'document_id': 'd3',
         'section': 's3', 'text': 'z' * 500, 'requirement_ids': ['E1']},
    ]
    msg = render_proofs('q', evidence, [])
    assert '[P1] t1' in msg and '[P2] t2' in msg and '[P3] t3' in msg
    assert 'x' * 101 not in msg
    assert 'y' * 51 not in msg
    assert 'z' * 11 not in msg
    assert 'evidence budget reached' in msg

