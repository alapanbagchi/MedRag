# MedRAG frontend (assistant-ui · Gemini-style)

Chat frontend for the MedRAG/MedPat backend, built entirely from
[assistant-ui](https://www.assistant-ui.com) components
(`@assistant-ui/react` primitives + tool UIs) with a minimal Gemini-inspired
theme. Talks to the **xdeep** research pipeline over the NDJSON stream API.

## Stack

- Vite + React 19 + TypeScript + Tailwind CSS v4
- `@assistant-ui/react` — `ExternalStoreRuntime` (zustand store owns threads +
  messages, localStorage persistence), `Thread` / `Composer` / `Message`
  primitives, `ThreadList` sidebar, `GroupedParts` thinking panel
- `@assistant-ui/react-markdown` — streamed markdown answers with `[n]`
  citation chips
- Custom tool UIs (`defineToolkit` + `AuiConfig`) for every research step:
  searching, web search, fetches, verdicts, reliability, contradictions, gaps

## Running

```bash
# 1. backend (from repo root / backend/)
cd backend && .venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000

# 2. frontend
cd frontend && npm install && npm run dev        # http://localhost:5174
```

`BACKEND_URL` env overrides the proxied API target (default
`http://127.0.0.1:8000`); `/v1` is proxied in `vite.config.ts`.

## What the UI shows

- **Sidebar** — thread list (new/switch/archive/delete via
  `ThreadListPrimitive`), auto-titled conversations, xdeep badge, theme toggle.
- **Thinking panel** — auto-opens while a run is in flight: animated dots, live
  stage ("Searching the literature", "Verifying evidence", …), elapsed seconds,
  and grouped **tool cards** per step kind (icon + label + count + per-row
  spinner/check). Collapses to "View thoughts · N" when done.
- **Research plan** — the decomposed task list (`decompose_done`), standalone
  above the answer.
- **Sources** — verified evidence grid, numbered to match `[n]` citation chips
  in the answer; web sources carry a globe badge and snippet.
- **Streaming markdown** — answers render incrementally with GFM tables,
  citations, action bar (copy / regenerate), and a stop button while running.

## Scripts

- `npm run dev` / `npm run build` (typecheck + vite build) / `npm run preview`
- `scripts/test-ui.mjs` — headless Chromium (puppeteer-core) smoke test:
  empty state → send question → live panel → completion → sidebar
- `scripts/test-complete.mjs` — long-wait E2E completion check
- screenshots land in `.ui-shots/`

## API contract (from `backend/api.py`)

`POST /v1/chat/stream` `{question, conversation_id, engine:"xdeep"}` →
newline-delimited JSON events: `status` (stage), `pipeline`
(`progress`/`decompose_done`/`query_start`/`retrieved`/`web_search_*`/
`web_fetch`/`verdict`/`reliability_verdict`/`synthesis_*`/…),
`sources`, `token`, `done`, `error`. See `src/lib/xdeep.ts` + `src/lib/run.ts`
for the event→step mapping.