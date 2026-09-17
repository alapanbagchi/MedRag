# Critique: Autonomous Research Runtime proposal

Grounded against `backend/src/agents/` as of 2026-09-04
(`graph.py`, `deep.py`, `state.py`, `rules.py`, `worker.py`,
`verify_queue.py`, `synthesis_gate.py`, `MERGE_PLAN.md`).

## Verdict

The proposal's core boundary — **agent decides strategy, runtime decides
admissibility** — is correct and already half-built here. `rules.py` +
`state.py` already enforce "retrieved text is not evidence until verified",
`worker.py` already lets the LLM pick one tool per step inside a
code-enforced guardrail, and `deep.py` already has a zero-subtask direct
path. The proposal should therefore be read as an **evolution**, not a
rewrite.

The main objection: the proposal removes the fixed DAG before its
replacements (plan validator, claim graph, promotion chain, claim checker)
exist and are tested. That inversion is the single biggest risk. Keep the
current DAG as the validated fallback until each replacement earns its
place behind a test.

## What to adopt as stated

1. **Tool contracts owned by runtime (§9–§11, §13–§16, §25).** This is the
   strongest section. `worker.py:1-17` already documents per-tool guardrails
   (corpus → verifier; firecrawl → trust-gate + reliability + verifier; umls →
   never evidence). Promote that docstring into real `ToolContract` objects
   with `output_class` (`DISCOVERY` vs `EVIDENCE_CANDIDATE` vs
   `CONTEXT_ONLY`). UMLS-as-`CONTEXT_ONLY` (§15) is a one-line runtime
   enforcement of what today is prompt + worker discipline.
2. **Claim-centered epistemics (§4–§5 of §1, §17–§23).** Requirement
   coverage (`satisfied()`) is a proxy for "we looked"; claims with
   support/contradict/qualify relations are a model of "what we know".
   Adopt the relation set (§21) and `underlying_studies` independence
   tracking (§22) — the latter fixes a real overcounting bug class
   (cf. `MERGE_PLAN.md` P0-2/P2-3 on web ids and collisions).
3. **Honest terminal states (§52–§53) and `no evidence` semantics.**
   `BUDGET_EXHAUSTED ≠ NO_EVIDENCE ≠ EVIDENCE_OF_NO_EFFECT`. The current
   `EXHAUSTED`-with-gap-note path in `deep.py:92-100` is honest but coarse;
   splitting it per §52 is cheap and high-value for the UI wire (§89).
4. **Final claim checker (§47–§50).** Strength-mismatch detection
   ("associated with" → "causes", "may improve" → "improves") is the
   highest-ROI correctness addition. Nothing equivalent exists today;
   `synthesis_gate.py` only checks freshness, and `bridge._cited_text`
   only repairs citation identity, not entailment or wording strength.
5. **Typed tool results (§67) and failure-vs-empty distinction (§68).**
   `DiscoveryResult` must not be assignable where `EvidenceResult` is
   expected; tool timeout must not surface as "no results". Both are small
   type-level changes with outsized debugging payoff.
6. **Structured event log (§58, §73–§74).** `emit_event` + `xdeep_log`
   already exist (`graph.py:57-80`); formalizing the event enum and
   JSONL-serializing it is uncontroversial.

## What to adopt with modifications

7. **Dynamic plans (§4–§7, §83) — yes, but validator-first.** The proposal
   lists validator checks (tool exists/enabled/allowed, args, budget,
   DAG validity, depth, parallelism) without specifying rejection semantics.
   Required before any dynamic executor ships: unknown capability → reject
   with `ActionRejected` event; invalid args → reject, never coerce; budget
   shortfall → reject with `BUDGET_EXHAUSTED`, never partial-execute. The
   current planner allow-list scope (`deep.py:75`, `tools_for_strategy`)
   is the seed of this validator — grow it, don't replace it.
8. **Promotion chain (§12, §20) — collapse to what `state.py` has.**
   The 7-state chain (DISCOVERY → … → EVIDENCE_ACCEPTED) overlaps the
   existing 4-state chain
   (`RETRIEVED → UNDER_REVIEW → ACCEPTED / REJECTED / CONTRADICTORY`).
   Two parallel state machines will cause exactly the orphan bugs
   `MERGE_PLAN.md` P0-3 documents. Recommendation: keep the 4 states,
   add `DISCOVERY` (unfetched/unverified lead) and `SOURCE_VERIFIED`
   (identity confirmed, content not yet judged) only, and forbid any
   assignment outside transition helpers — the proposal's §20 rule is
   right, enforce it by making state fields private with transition
   methods (as `EvidenceItem.submit_to_verifier()/set_verdict()` already
   does; cf. `verify_queue.py:67-73`).
9. **Batched verification (§27, §96–§97) — yes, but keep fail-closed
   per-item semantics.** The current FIFO (`verify_queue.py`) is the only
   serialized lane, with dedup cache and fail-closed timeout behavior
   (item stays `UNDER_REVIEW`, invisible to synthesis). A batch verifier
   must preserve all three properties: every input gets exactly one result
   (proposal says this — good), a batch-level failure must leave every
   item `UNDER_REVIEW` (proposal is silent — specify it), and the dedup
   key `(chunk_id_or_url_hash, requirement_hash)` must survive batching.
   Priority by claim importance (§97) is fine once claims exist; until
   then, preserve FIFO.
10. **Capability registry (§8) — merge into `KNOWN_TOOLS` + worker
    guardrails, don't fork.** `rules.py:52-53` already distinguishes known
    vs available-now. Extend that record with `evidence_class`,
    `verification_policy`, `cost_class` rather than building a parallel
    registry. Unused-spec risk: `paper_inspect` was advertised with no
    implementation (`MERGE_PLAN.md` P1-1, since removed) — every registry
    entry must resolve to a live implementation or fail closed.
11. **Research depth (§28) — keep as internal hint, never a model-visible
    enum the answer depends on.** Emergent depth is fine as a prompt
    heuristic, but budgets (`RunBudget`, `RunDeadline`, per-call timeouts
    in `timeouts.py`) stay authoritative. An agent-declared "DEPTH 4"
    must never extend a budget.
12. **Blackboard + dedup (§36–§38) — events first, semantic dedup
    second.** The append-only event fold (§57) is the right concurrency
    model and fixes real duplicate-inspection waste. But semantic
    research-intent dedup (§37) is LLM-judged and will both over-merge
    (distinct questions, similar words) and under-merge. Ship exact +
    canonicalized-string dedup first (already partially in `seen_chunks`
    and the verify cache), add semantic dedup only with a false-merge
    metric (§75's `duplicate research rate` cuts both ways).
13. **Contradiction handling (§32–§33) — keep mandatory analysis, make
    *investigation depth* adaptive.** Today contradiction runs on the final
    verified set and every detection goes through resolution (`rules.py`
    rule 3, `synthesis_gate.py` freshness bounce). The proposal makes even
    detection optional ("runtime does not require a mandatory contradiction
    phase"). That is a regression: silent contradiction-dropping is worse
    than wasted investigation. Compromise: detection + classification stay
    mandatory; the three-branch deep investigation (§33) becomes
    agent-invoked. The 9-way taxonomy
    (TRUE_CONTRADICTION … UNRESOLVED) is good — adopt it as the
    `ResolutionStatus` vocabulary.
14. **Statistics capability (§16) — scope it.** Structured
    metric/unit/time/geography/population validation is correct for
    WHO-style lookups, but most biomedical claims are prose. Build it as
    one capability (`statistics.lookup`), not as the model for all
    evidence.

## What to reject or defer

15. **"Synthesis must not invent strategy; checker loops back to research"
    (§86, §80 loop).** An unbounded
    synthesize → check-fail → research → synthesize loop needs a hard
    iteration cap and a terminal `UNRESOLVED`-with-qualified-answer state,
    or it becomes an infinite budget drain on hard questions. Specify
    max 1–2 repair rounds, then ship the qualified answer. The current
    one-shot `synthesize_gated` + freshness bounce is the safe base.
16. **Expected-information-gain stopping (§35).** Unfalsifiable without a
    calibration set. The stopping hierarchy (§85) is good prompt guidance,
    but ship it as prompt text, not as a numeric gain/cost comparator.
    Measure `useful-evidence / cost` (§76) offline first; only promote to
    runtime when it predicts answer-change on held-out runs.
17. **Recursive child tasks (§81–§82).** Real need (definition-change
    example is convincing), but recursion + parallelism + shared budget is
    where deadlocks and budget leaks live. Defer until the event-sourced
    blackboard (§57) and per-run budget threading (`MERGE_PLAN.md` P1-3)
    are done and tested. Until then, subtasks are flat (as today).
18. **Multidimensional source authority (§39).** Five scored dimensions
    with no calibration will produce confident-looking numbers the system
    cannot justify. Keep the deterministic study-type hint
    (`state.py:30-37`, best-effort, never replaces verifier) and add at
    most one authority tier per domain profile (§40) with documented
    ordering (e.g. WHO primary dataset > secondary article).
19. **Question-model revision (§41–§42) as separate machinery.** A mutable
    `QuestionModel` that replans research is a second planner by another
    name, with the same writeback-bug risk `MERGE_PLAN.md` Phase C
    documents for v3 (`search.py:133-149` bound queries to the wrong
    requirement). If adopted, bind every revision to requirement ids and
    test that binding explicitly.
20. **Directory restructure (§60).** Premature. The merge plan just
    finished unifying two engines; a 30-directory reshuffle now destroys
    `git blame` and invalidates every pending test path. Evolve in place:
    `runtime/validator.py` ← `rules.py` extension, `graph/claims.py` ← new,
    `capabilities/` ← `tools/` + `strategies.py` migration (§61 is right
    about the direction, wrong about the timing).

## Missing from the proposal (needed for implementation)

21. **Rejection UX and observability.** Every runtime rejection
    (`ActionRejected`, `DISCOVERY_ONLY` downgrade, budget refusal) must be
    a first-class event the agent *and* the UI log can see. Silent
    downgrades cause the "worker satisfied but requirement unsatisfied"
    class of bug (cf. P0-2). §5's example shows the right shape; make it
    a contract: no rejection without an event + reason string.
22. **Concurrency control for parallel branches (§56–§57).** The proposal
    names max parallelism, shared budget, cancellation — but not the
    atomicity unit. Specify: budget decrement is atomic per action
    admission (validator holds the lock); event append is lock-free;
    state fold is single-threaded per run. Test: two branches racing the
    last web-search slot → exactly one admitted, one `ActionRejected`.
23. **Citation identity across the claim migration.** Today citations
    resolve markers → verified ids (`bridge._cited_text`). With claims,
    citations attach to claims (§46) which map to evidence ids. Specify
    the join (claim → evidence → source → rendered `[n]`) and the repair
    rule when evidence is rejected post-synthesis — otherwise the
    id-mismatch bug class from v3 `_repair_citations` (documented in
    `MERGE_PLAN.md` Phase C) recurs one layer up.
24. **Freshness/versioning interaction (§71–§72).** `retrieved_at` +
    `content_hash` versioning is right, but the proposal doesn't say what
    invalidates what: a new content hash for the same URL must re-run
    source verification *and* entailment, and must emit
    `SourceRevalidated` — otherwise cached verdicts (cf. `verify_queue.py`
    `_verdict_cache`) bless stale content.
25. **Benchmarks before adaptivity (§75–§77).** The A–E benchmark is good
    but underspecified: each category needs expected tool-call ceilings
    (A: 0, B: ≤2, C: ≤5, D/E: bounded by budget) and a grading rubric
    (claim support rate, citation entailment rate — not source count).
    Build the benchmark *before* the adaptive loop, or "efficiency" is
    unmeasurable.

## Concrete sequencing (post-merge)

1. `respond` fast path + terminal-state split (§52–§53) — small, safe.
2. `ToolContract` + `output_class` enforcement (UMLS CONTEXT_ONLY first).
3. Claim graph v0: claim text/status/relations + `underlying_studies` —
   alongside `EvidenceItem`, dual-written, read by contradiction analysis.
4. Batch verifier preserving FIFO fail-closed + dedup semantics.
5. Claim checker (entailment, scope, strength) as hard pre-render gate.
6. Plan validator + dynamic executor; current DAG becomes fallback.
7. Retire mandatory gap/conflict stages only after 3–6 hold green on
   the A–E benchmark.

## Adversarial tests (§78) — keep all six, add four

The six listed tests are correct and should be written exactly as stated
(model self-verification, memory-as-evidence, blocked source, budget,
causal overclaim, independence count). Add:

7. Batch verifier drops one candidate → run fails loud, no partial
   evidence set silently promoted.
8. Two branches race the last budget slot → exactly one admitted.
9. Same URL, new content hash → verdict cache invalidated, re-verified.
10. Checker-rejected synthesis loops twice → run terminates with
    qualified `UNRESOLVED` answer, budget intact.

## One-line summary

Adopt the invariants (contracts, promotion discipline, claim relations,
checker, honest states); earn the freedoms (dynamic plans, adaptive
stopping, recursion) behind validators, benchmarks, and caps.
