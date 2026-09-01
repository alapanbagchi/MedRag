# MedPat — frontend

The MedPat research interface: a Next.js (App Router) + TypeScript + Tailwind CSS v4 UI for a medical RAG research assistant. Brutalist × clinical × instrument-console design system (nixie laboratory counter grammar), dark-first with light mode.

## Run

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

Other scripts: `npm run build`, `npm run start`, `npm run typecheck`.

## Layout

```
frontend/
├── app/                 # pages & root layout
│   ├── page.tsx         # home / landing
│   └── chat/[id]/page.tsx
├── components/
│   ├── layout/          # Sidebar, AppShell
│   ├── home/            # HomeView (hero + search + telemetry)
│   ├── chat/            # ChatView, MessageList, AssistantMessage, ThinkingStatus, Markdown, ChatInput
│   ├── sources/         # SourcePanel (drawer / bottom sheet)
│   └── ui/              # primitives (Dropdown, Kbd, Led, …)
└── lib/
    ├── types.ts         # Conversation/Message/Source/StreamEvent
    ├── rag-client.ts    # ← the only file that talks to a backend
    ├── mock-rag.ts      # simulated engine (default)
    ├── mock-data.ts     # seed conversations + corpus telemetry
    └── store.tsx        # app state (localStorage-backed)
```

## Connecting your real RAG backend

All backend I/O is isolated behind `lib/rag-client.ts`. It consumes a
**newline-delimited JSON stream** from:

```text
POST  {NEXT_PUBLIC_RAG_API_URL}/v1/chat/stream
```

```json
{"type":"status","stage":"retrieving","message":"Searching PMC…","count":42}
{"type":"sources","sources":[{"id":"PMC123456","pmcid":"PMC123456","pmid":"12345678","title":"…","authors":["…"],"journal":"…","year":2024,"score":0.94,"snippet":"…","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/"}]}
{"type":"token","content":"The"}
{"type":"done","timingMs":3127}
```

Wire it up:

1. Create `frontend/.env.local`:
   ```
   NEXT_PUBLIC_RAG_API_URL=http://localhost:8000
   NEXT_PUBLIC_USE_MOCK=false
   ```
2. The UI now streams from the real endpoint. Status stages the engine can
   emit: `understanding` `decomposing` `retrieving` `reranking` `verifying` `synthesizing`.
   Unknown stages are tolerated.

While `NEXT_PUBLIC_USE_MOCK` is `true` (default) or no URL is set, the app runs
fully on the simulated engine — nothing else in the UI changes.

## Shortcuts

- `⌘/Ctrl + K` — focus the search / question input
- `⌘/Ctrl + N` — new research conversation
- `Esc` — close menus and drawers

Data (conversations, pins, theme) persists in localStorage under `medpat:*`;
swap `lib/store.tsx` for an API when you have one.
