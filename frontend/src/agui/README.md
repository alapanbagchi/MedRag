# AG-UI surface

AG-UI is the **transport** over MedRAG's existing pipeline — not a second
pipeline. The agentic flow is unchanged.

- **Backend** — `POST /v1/ag-ui` (`backend/src/agents/ag_ui_endpoint.py`) parses an
  AG-UI `RunAgentInput` and runs `stream_adapter.stream_deep_agent` **unchanged**:
  the exact generator `POST /v1/chat/stream` uses, with the same orchestrator,
  `_spawn`, `_synthesize_impl`, budgets, ledger, `_requirements_block`
  injection, and `enforce_citations` guard.
  `backend/src/agents/ag_ui_transport.py` re-encodes that generator's event
  stream as AG-UI: standard `TEXT_MESSAGE_*` / `TOOL_CALL_*` / `THINKING_*` /
  `RUN_*` events, plus `CUSTOM` events for MedRAG's richer events
  (`plan`, `step`, `task`, `sources`, `status`, `question`, …).
- **HITL** — the existing broker is preserved. `ask_user` parks the run and the
  stream stays open; the transport surfaces a `question` CUSTOM event carrying
  the chat id, and the dock posts each answer to
  `/v1/chats/{chatId}/questions/{questionId}/answer`, which releases the parked
  run and lets the same stream continue.
- **Frontend** — `@assistant-ui/react-ag-ui`'s `useAgUiRuntime` over an
  `HttpAgent` pointed at `/v1/ag-ui` (override with `VITE_AG_UI_URL`).

## Layout

| File | Purpose |
| --- | --- |
| `AgUiApp.tsx` | Shell; remounts the runtime per thread |
| `AgUiRuntimeProvider.tsx` | `useAgUiRuntime` + history adapter |
| `AgUiThread.tsx` | Thread, composer, title sync |
| `AgUiSidebar.tsx` | Thread list (new/switch/delete) |
| `renderers.tsx` | `data.by_name` renderers for CUSTOM events |
| `AgUiInterruptDock.tsx` | `question` form → broker answer POST |
| `AgUiInspector.tsx` | Step/tool-call inspector |
| `aguiStore.ts` / `aguiHistory.ts` | Thread metadata + per-thread message persistence |

## Flags

- `?legacy=1` or `VITE_AG_UI=0` — use the previous NDJSON surface.
- `VITE_AG_UI_URL` — override the AG-UI endpoint (default `/v1/ag-ui`).

## Not yet at UI parity

Trajectory view and the run-stats panel are still NDJSON-only.

## Reused rich rendering

The AG-UI surface reuses the app's rich components rather than a skeleton:
`AgUiAssistantMessage` renders the message through the same
`AssistantMessage` → `ThinkingPanel`-style stream, `ToolCalls`,
`SubagentSection`, plan card, sources, verdict table, and run-stats card.

The part readers (`frontend/src/lib/parts.ts` `partName`/`partArgs`, plus the
`AssistantMessage`/`SubagentSection`/`ThinkingPanel` hooks) accept **both**
storage shapes: the legacy `tool-call` parts (`toolName`/`args`) and the AG-UI
`data` parts (`name`/`data`). The transport emits the same `StepArgs` payloads
(`step`, `status`, `plan`, `task`, `sources`, …) as `data` parts, so one
component tree serves both transports.
