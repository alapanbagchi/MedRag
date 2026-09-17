# Run-state ledger (`src/runstate`)

One JSON object per chat — the statefulness layer. The orchestrator
fills it within its budget and steers from it, instead of re-reading
full tool outputs. This is the stateful layer, and there is no
stateless fallback.

## The contract

- **Writers are code, not the model.** Plan, task lifecycle,
  requirements, judge-kept evidence (nested per requirement), gap
  verdicts, and per-task budget ledgers land in the store at the
  tool/execution boundary (judge middleware, gap/planner tools,
  spawner). The model never hand-writes ledger JSON.
- **Evidence per requirement is the deliverable.** No findings blobs:
  each requirement carries its `supporting_evidence` and `gaps`, and
  each task carries `budget_allocated`, `budget_expenditure_history`,
  and `budget_remaining` snapshotted from its budget manager — so a
  null budget field means the task ran unbudgeted, visibly.
- **Readers get arrays, not text.** Spawn returns carry ONLY the
  `[RUN STATE]` receipt (verified counts, gaps, run total) — evidence
  texts never enter orchestrator context. `run_progress` defaults to
  the same lean shape; `run_progress(task_id)` pulls full
  supporting-evidence texts for the tasks being synthesized, and only
  then. Massive coverage + empty gaps = stop.

## Shape (`chats/{chat}.json`, `MEDRAG_RUNSTATE_DIR` overrides)

`{chat_id, turns: {turn_id: {question, status, plan[], budget,
error}}}` where each plan item is:

```json
{
  "id": "T1",
  "task": "Which dietary factors affect blood pressure?",
  "depth": "deep",
  "state": "done",
  "evidence_requirements": [
    {
      "id": "E1",
      "description": "Evidence concerning dietary factors…",
      "supporting_evidence": [
        {"passage_id": "PMC1_p3", "requirement_ids": ["E1]",
         "intent": 0.92, "reason": "…", "tool": "retrieve_evidence",
         "document_id": "PMC1", "chunk_id": "PMC1_p3",
         "url": "", "section": "Results", "text": "…"}
      ],
      "gaps": [{"evidence_id": "E2", "missing": "potassium dose-response"}]
    }
  ],
  "budget_allocated": {"max_tool_calls": 20, …},
  "budget_expenditure_history": [{"event": "budget_consumed", …}, …],
  "budget_remaining": {"tool_calls": 14, …}
}
```

The plan IS the task list — no separate `tasks` object, no findings
blobs (evidence per requirement is the deliverable). Turns accumulate
oldest-first (capped at 20); `GET /v1/chats/{chat_id}` serves the file.
States distinguish `done`, `budget_exhausted`, and `failed`.

## Wiring

- `DeepDeps.chat_id / task_id` identify the ledger and owning task;
  writes land in the chat's current turn.
- Judge middleware records only verdict-backed passages ("valid
  evidence" = judged, not merely retrieved), deduped by passage id.
- `stream_adapter` (orchestrated path, `conversation_id` selects the
  chat) and `run_deep_task` + `__main__` (CLI path, one turn per run)
  record plan → task started → finished lifecycles.

## Tests

`tests/test_runstate.py` (21 tests): findings parsing, schema, store
CRUD, disk round-trip, concurrent writes, writer hooks, progress views,
receipts, builder wiring, plus two integration proofs — a real judge
run landing in the ledger, and a scripted plan→spawn→done run filling
the on-disk chat JSON with structured findings.

## Deliberate V1 limits

- Orchestrator direct tool calls still return full outputs (needed for
  its "quick single checks"); receipts + `run_progress` cover steering.
- No cross-chat queries or compaction — the JSON is per chat by design.
