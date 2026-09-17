'''Human-in-the-loop clarification tests: the ask_user tool contract.

Covers the wire contract (normalize_questions/validate_answer), the
park-and-resume broker (ask_user awaits a submit_answer-resolution), the
ledger writes, and the timeout fallback. No LLM involved.
'''

from __future__ import annotations

import asyncio
import types

import pytest

import src.agents.clarify as clarify
from src.runstate.models import QuestionRecord
from src.runstate.store import RunStore, get_store, set_store
from src.tools.umls import DeepDeps


@pytest.fixture()
def store(tmp_path):
    return RunStore(tmp_path / 'chats')


@pytest.fixture(autouse=True)
def _reset_global_store():
    set_store(None)
    yield
    set_store(None)


def _deps(chat_id='c1', task_id=None):
    return DeepDeps(umls=object(), chat_id=chat_id, task_id=task_id)


def _ctx(chat_id='c1', task_id=None):
    return types.SimpleNamespace(deps=_deps(chat_id, task_id))


def _questions():
    return [
        {'id': 'q1',
         'text': 'How does diet affect hypertension?',
         'options': ['Physiological mechanisms', 'Named diets (DASH, sodium)',
                     'Management recommendations'],
         'multi_select': True},
        {'id': 'q2',
         'text': 'Do you want local or web sources?',
         'options': ['Local corpus only', 'Web plus local'],
         'multi_select': False},
    ]


# --- normalize_questions ------------------------------------------------------
def test_normalize_list_of_dicts():
    qs = clarify.normalize_questions(_questions())
    assert [q.id for q in qs] == ['q1', 'q2']
    assert qs[0].multi_select is True
    assert qs[1].multi_select is False
    assert qs[0].allow_find_all is True  # implicit defaults


def test_normalize_json_string_and_missing_ids():
    import json
    raw = json.dumps([{'text': 'Question A', 'options': ['x', 'y']},
                      {'text': 'Question B', 'options': ['z']}])
    qs = clarify.normalize_questions(raw)
    assert [q.id for q in qs] == ['q1', 'q2']
    assert [q.text for q in qs] == ['Question A', 'Question B']


def test_normalize_dict_with_questions_key_and_garbage():
    qs = clarify.normalize_questions({'questions': [{'text': 'Only'}]})
    assert len(qs) == 1 and qs[0].text == 'Only'
    assert clarify.normalize_questions(None) == []
    assert clarify.normalize_questions('not json') == []
    assert clarify.normalize_questions(42) == []
    assert clarify.normalize_questions([{'options': []}]) == []  # no text


def test_normalize_tolerates_ragged_option_types():
    """Model tool payloads are untrusted: ragged option/multi_select types
    must degrade to a usable question, never raise ValidationError."""
    qs = clarify.normalize_questions([
        {'id': 'q1', 'text': 'Pick', 'options': 'not-a-list'},
        {'id': 'q2', 'text': 'Pick two', 'options': [1, 2]},
    ])
    assert [q.id for q in qs] == ['q1', 'q2']
    assert [q.text for q in qs] == ['Pick', 'Pick two']
    assert all(q.options == [] for q in qs)


# --- validate_answer -----------------------------------------------------------
def test_validate_answer_requires_something():
    assert clarify.validate_answer({}) == {}
    assert clarify.validate_answer({'selections': []}) == {}
    got = clarify.validate_answer(
        {'selections': ['DASH diet'], 'other': '', 'find_all': False})
    assert got['selections'] == ['DASH diet']
    got2 = clarify.validate_answer({'selections': [], 'other': '  anything else '})
    assert got2['other'] == 'anything else'
    got3 = clarify.validate_answer({'find_all': True})
    assert got3['find_all'] is True


# --- park / resume broker -------------------------------------------------------
async def _park(task, chat_id, *qids):
    """Yield until ask_user has registered its questions.

    The tool persists each question to the ledger before it parks, and that
    write now runs off the event loop, so a single sleep(0) is not enough.
    """
    for _ in range(400):
        if task.done():
            break
        if all(clarify.has_pending(chat_id, q) for q in qids):
            return
        await asyncio.sleep(0.005)
    raise AssertionError('ask_user never parked')


async def test_ask_user_parks_until_answer(store):
    set_store(store)
    await store.create_turn('c1', 't1', 'How does diet affect hypertension?')
    task = asyncio.create_task(clarify.ask_user(
        _ctx(), questions=_questions()))
    await _park(task, 'c1', 'q1', 'q2')
    assert clarify.submit_answer(
        'c1', 'q1',
        {'selections': ['Named diets (DASH, sodium)'], 'other': '', 'find_all': False})
    assert clarify.submit_answer(
        'c1', 'q2',
        {'selections': ['Web plus local'], 'other': '', 'find_all': False})
    result = await asyncio.wait_for(task, timeout=5)
    assert 'Named diets (DASH, sodium)' in result
    assert 'Web plus local' in result


async def test_ask_user_find_all_and_other(store):
    set_store(store)
    await store.create_turn('c1', 't1', 'q')
    task = asyncio.create_task(clarify.ask_user(_ctx(), questions=
        [{'id': 'q1', 'text': 'Scope?', 'options': ['Narrow']}]))
    await _park(task, 'c1', 'q1')
    clarify.submit_answer('c1', 'q1',
                          {'selections': [], 'other': 'mechanisms only',
                           'find_all': True})
    result = await asyncio.wait_for(task, timeout=5)
    assert 'Find all you can find' in result
    assert 'mechanisms only' in result


async def test_ask_user_timeout_falls_back(store, monkeypatch):
    set_store(store)
    await store.create_turn('c1', 't1', 'q')
    monkeypatch.setattr(clarify, '_FALLBACK_TIMEOUT', 0.05)
    result = await clarify.ask_user(_ctx(), questions=
        [{'id': 'q1', 'text': 'Scope?', 'options': ['Narrow']}])
    assert 'no user answers' in result or 'best interpretation' in result


async def test_ask_user_unavailable_without_chat():
    result = await clarify.ask_user(_ctx(chat_id=''), questions=_questions())
    assert 'unavailable' in result


# --- ledger ---------------------------------------------------------------------
async def test_store_records_question_and_answer(store):
    set_store(store)
    await store.create_turn('c1', 't1', 'q')
    await store.record_question('c1', QuestionRecord(
        id='q1', text='Scope?', options=['A', 'B'], by_task='T2'))
    ok = await store.record_question_answer(
        'c1', 'q1', {'selections': ['A'], 'other': '', 'find_all': False})
    assert ok is True
    chat = get_store().get_chat('c1')
    q = chat.questions['q1']
    assert q.answered is True
    assert q.answer == {'selections': ['A'], 'other': '', 'find_all': False}
    assert q.by_task == 'T2'
    # unknown question -> False, no crash
    assert await store.record_question_answer('c1', 'nope', {}) is False


async def test_ask_user_persists_question(store):
    set_store(store)
    await store.create_turn('c1', 't1', 'q')
    task = asyncio.create_task(clarify.ask_user(
        _ctx(task_id='T3'), questions=_questions()))
    await _park(task, 'c1', 'q1', 'q2')
    chat = get_store().get_chat('c1')
    assert 'q1' in chat.questions
    assert chat.questions['q1'].by_task == 'T3'
    clarify.submit_answer('c1', 'q1', {'selections': ['x'], 'other': '', 'find_all': False})
    clarify.submit_answer('c1', 'q2', {'selections': ['y'], 'other': '', 'find_all': False})
    await asyncio.wait_for(task, timeout=5)


async def test_wait_answers_keeps_later_answers_when_earlier_is_unanswered():
    """An unanswered q1 must not cost the run the answers it did get for q2."""
    chat = 'c-wait'
    questions = [clarify.ClarifyQuestion(id='q1', text='first?'),
                 clarify.ClarifyQuestion(id='q2', text='second?')]

    async def _answer_q2():
        await asyncio.sleep(0.05)
        with clarify._broker_lock:
            fut = clarify._broker.get((chat, 'q2'))
        if fut is not None and not fut.done():
            fut.set_result({'selections': ['B'], 'other': '', 'find_all': False})

    task = asyncio.create_task(_answer_q2())
    try:
        answers = await clarify._wait_answers(chat, questions, timeout=0.4)
    finally:
        await task
    assert answers.get('q2', {}).get('selections') == ['B']
    assert 'q1' not in answers

