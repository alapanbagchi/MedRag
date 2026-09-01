# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack
Delegated to the implementing session (user: "just create the web ui"): Next.js (App Router) + TypeScript + Tailwind CSS v4, living in `frontend/`; the existing Python RAG pipeline lives untouched in `backend/`. Stack choice was inferred from the brief, not asked.

## Users
Medical researchers, clinicians, and clinical research associates who need trustworthy, evidence-grounded answers from the biomedical literature (PMC-centric RAG). They work at a desk, have little patience for hype, and will not trust an answer they cannot trace to a source.

## Product Purpose
MedPat is a medical research RAG interface. It turns the retrieval pipeline into a fast, legible research conversation: ask a question, watch the research pipeline work at a high level, read a synthesized answer anchored to citations, and inspect every source. Success is being able to run a full literature conversation without leaving the surface and to trust what it reports.

## Positioning
Evidence-transparency is the product: every answer resolves to retrievable sources, the research process is surfaced as explicit status events (understanding → decomposing → retrieving → reranking → verifying → synthesizing), and the interface behaves like an instrument, not a chatbot. The visual language (brutalist + clinical + technical) is a product feature for a serious research audience.

## Operating Context
Desktop-first research workstation: persistent sidebar, wide chat column, optional right-side source panel. Tablet collapses the sidebar; mobile turns it into a drawer and sources into a bottom sheet. The backend is the Python RAG pipeline in `backend/`; the UI talks to it through a replaceable streaming client (`lib/rag-client.ts`), URL configurable via `NEXT_PUBLIC_RAG_API_URL`. Until wired, `lib/mock-rag.ts` simulates the full pipeline and token streaming so the UI is fully usable offline. Conversation history, pins, rename, delete persist to localStorage now, shaped for a later database.

## Capabilities and Constraints
- Streaming answers (newline-delimited JSON events: `status`, `sources`, `token`, `done`), progressive markdown rendering, blinking cursor, stop generation, retry, regenerate, copy, thumbs up/down.
- Live research-pipeline status component driven by stream events; no fabricated progress — the UI renders what the stream emits (the mock emits realistic events of its own).
- Sources panel: citation numbers, title, authors, journal, year, relevance score, snippet, PMCID/PMID, expand; citation chips inline in answers open the panel.
- Conversation history: pin, search, rename, delete, grouped by time (today / yesterday / previous 7 days / older).
- Dark mode primary, light mode supported, persisted. Keyboard shortcuts (Cmd/Ctrl+K search, Cmd/Ctrl+N new research).
- Responsive down to phones; reduced-motion support; semantic HTML + aria + visible focus.

## Brand Commitments
- Name: **MedPat**. Tagline: "Evidence-aware medical research, accelerated."
- Visual world pinned by brief: new-gen brutalist + futuristic + clinical + technical. Near-monochrome foundation, sharp edges, strong 1px/2px borders, grid lines, monospaced technical metadata, dense-but-clear hierarchy, one strong accent. Explicit bans: generic purple AI gradients, glassmorphism, rounded-everything SaaS chrome, glowing blobs, stock illustrations, "Bootstrap dashboard" look.
- Logo: abstract medical/molecular/neural mark drawn in CSS/SVG (no external assets), reusable as an app icon.
- Typography: strong display/UI face + clinical monospace for metadata, source IDs, statuses. Fonts in the brief's spirit: Archivo + IBM Plex Mono.

## Evidence on Hand
The backend implementation, tests, papers and data live in `backend/` (not shipped into the UI). The UI has no real backend screenshots yet; demo conversations and mock source metadata in the frontend's mock layer are authored at realistic fidelity and are synthetic.

## Product Principles
- Ground every answer in literature and make the grounding visible and inspectable.
- Behave like an instrument: show the research process, stay dense but legible, never block on the backend (mock today, real endpoint tomorrow, zero UI rewrite).
- The technical visual identity carries the product's seriousness — it is a feature for the intended audience.
- Respect attention: keyboard-driven, flicker-free, responsive, reduced-motion aware, no fake progress.

## Accessibility & Inclusion
Standard web a11y: semantic elements, keyboard operability, visible focus states, aria labelling, sufficient contrast in both themes, prefers-reduced-motion respected.
