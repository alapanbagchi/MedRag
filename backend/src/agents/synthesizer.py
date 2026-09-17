'''Final-answer synthesizer: question + ALL verified passages, one prompt.

The runner hands the agent everything in a single user message: the
question, every judge-kept proof passage with its [Pn] ref (title,
url, section, full text), and the uncovered-requirement list. No ledger
tools, no iteration - the research is finished and coverage-reviewed,
so the model just writes. The deterministic citation guard rebuilds
'## References' afterwards; nothing here re-checks coverage.

The system prompt is composed by the SKILL LIBRARY (backend/src/lib/
skills.py): base evidence discipline + one writing-style skill (always
used; concise-clinical by default) + the humanizer pass (mandatory on
every answer). synthesize.txt remains as the fallback prompt only.
'''

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from src.llm.models import agent_endpoint, make_model

SYNTH_TEMPERATURE = 0.0

_FALLBACK_PROMPT = (Path(__file__).resolve().parent.parent
                    / 'prompts' / 'synthesize.txt')

logger = logging.getLogger(__name__)


def load_prompt(mode: str | None = None, question: str = '') -> str:
    '''Compose the synthesizer's system prompt from the skill library.

    A skill MUST always be used: the prompt is always base (evidence
    discipline) + one writing-style skill (default concise-clinical when
    the question gives no style signal) + the humanizer pass (mandatory,
    applied to every answer). Unknown modes fall back to the default
    style. If the skill tree is missing (fresh checkout), the classic
    synthesize.txt prompt is used so synthesis never breaks.
    '''
    try:
        from src.lib.skills import build_synthesis_prompt
        return build_synthesis_prompt(mode, question)
    except Exception:  # noqa: BLE001 - fallback keeps synthesis alive
        logger.warning("skill prompt load failed; using synthesize.txt",
                       exc_info=True)
        return _FALLBACK_PROMPT.read_text(encoding='utf-8').strip()


def build_agent(mode: str | None = None,
                question: str = '') -> Agent[None, str]:
    '''Build the synthesizer on the big thinking lane (pinned greedy).

    Pure text-to-text: the question and every proof passage ride in the
    user message, so there are no tools and no deps. 'mode' selects the
    writing-style skill (concise-clinical | structured-review |
    patient-facing | evidence-critique); None picks it from the question.
    '''
    endpoint = agent_endpoint()
    model = make_model(endpoint, settings=ModelSettings(
        temperature=SYNTH_TEMPERATURE))
    return Agent(model, system_prompt=load_prompt(mode, question))


def collect_evidence(chat_id: str) -> tuple[list[dict], list[str]]:
    '''All proof passages across the turn, plus the uncovered list.

    Returns (evidence, uncovered): evidence is every judge-kept
    supporting-evidence passage with ref/title/url/document_id/section/
    text/requirement_ids, sorted by ref number; uncovered is the
    requirement descriptions with zero verified support. Best-effort: a
    missing chat or turn yields empty lists, never an exception.
    '''
    try:
        from src.runstate.store import get_store

        chat = get_store().get_chat(chat_id)
    except Exception:  # noqa: BLE001 - reading never breaks synthesis
        logger.warning("evidence ledger read failed for chat %s", chat_id,
                       exc_info=True)
        return [], []
    turn = chat.current if chat is not None else None
    if turn is None:
        return [], []
    evidence = []  # list[dict]
    uncovered = []  # list[str]
    for task in turn.plan:
        for req in task.evidence_requirements:
            if not req.supporting_evidence:
                label = req.description or req.id
                uncovered.append(req.id + ': ' + str(label))
            for ev in req.supporting_evidence:
                ref = str(getattr(ev, 'ref', '') or '')
                if not ref:
                    continue
                evidence.append({
                    'ref': ref,
                    'title': str(getattr(ev, 'title', '') or ''),
                    'url': str(getattr(ev, 'url', '') or ''),
                    'document_id': str(getattr(ev, 'document_id', '') or ''),
                    'section': str(getattr(ev, 'section', '') or ''),
                    'text': str(getattr(ev, 'text', '') or ''),
                    'requirement_ids': list(getattr(ev, 'requirement_ids', None) or []),
                })

    def _num(r: str) -> int:
        digits = r[1:] if r[:1].lower() == 'p' else ''
        return int(digits) if digits.isdigit() else 0

    evidence.sort(key=lambda r: _num(r['ref']))
    return evidence, uncovered


# The synthesizer is the last and most important call, and it receives every
# kept passage in ONE user message. Stored evidence is clipped at 12k chars
# each, so an evidence-heavy run can otherwise push a quarter-megabyte of
# unique (non-prefix-cacheable) text into a single prefill. Cap per passage
# and in total; refs, titles and sections are never dropped.
def _synth_caps() -> tuple[int, int]:
    def _env(name: str, default: int) -> int:
        try:
            return max(0, int(os.environ.get(name, default)))
        except (TypeError, ValueError):
            return default

    return (_env('SYNTH_MAX_PASSAGE_CHARS', 3000),
            _env('SYNTH_MAX_TOTAL_CHARS', 80_000))


def render_proofs(question, evidence, uncovered) -> str:
    '''The synthesizer user message: question + every proof passage
    (with refs) + declared gaps. The model never fetches.

    Passage text is capped per passage and in total; when the budget runs
    out the remaining passages keep their refs (still citable) but carry
    no body, so the prompt stays bounded without losing the citation pool.
    '''
    per_passage, total_cap = _synth_caps()
    lines = ['QUESTION', str(question or '').strip(), '']
    lines.append('EVIDENCE — all verified proof passages. Every claim')
    lines.append('must cite exactly one of these [Pn] refs, copied as written:')
    lines.append('')
    if not evidence:
        lines.append('(no verified evidence was collected)')
        lines.append('')
    used = 0
    omitted = 0
    for ev in evidence:
        head = '[' + ev['ref'] + '] ' + (ev['title'] or ev['document_id'] or 'source')
        src = ev['url'] or ev['document_id']
        lines.append(head + ' — ' + src if src else head)
        if ev['section']:
            lines.append('section: ' + ev['section'])
        text = str(ev.get('text') or '')
        if per_passage and len(text) > per_passage:
            text = text[:per_passage] + ' … [truncated]'
        room = total_cap - used if total_cap else None
        if room is not None and room <= 0:
            omitted += 1
            lines.append('(text omitted — evidence budget reached)')
        else:
            if room is not None and len(text) > room:
                text = text[:room] + ' … [truncated]'
            used += len(text)
            lines.append(text)
        lines.append('')
    if omitted:
        lines.append('(' + str(omitted) + ' passage(s) kept their [Pn] ref but '
                     'had their text omitted to bound this prompt.)')
        lines.append('')
    lines.append('UNCOVERED — requirements with NO verified evidence.')
    lines.append('State them plainly as gaps; never fill from general knowledge:')
    lines.append('')
    if not uncovered:
        lines.append('(none — every requirement has verified support)')
        lines.append('')
    for item in uncovered:
        lines.append('- ' + item)
    lines.append('')
    lines.append('Answer the question now, in structured prose, using only')
    lines.append('the EVIDENCE above, with inline [Pn] citations on every claim.')
    return '\n'.join(lines)


def build_synthesis_message(question: str, chat_id: str) -> str:
    '''Question + all proof passages + uncovered, in one user message.
    '''
    evidence, uncovered = collect_evidence(chat_id)
    return render_proofs(question, evidence, uncovered)


__all__ = ['build_agent', 'build_synthesis_message', 'collect_evidence',
           'load_prompt', 'render_proofs']
