# MedPat — Design System (DESIGN.md)

Written from the built frontend (frontend/), the ground truth of what shipped.
Surface primary: home (/). Related: chat (frontend/app/chat/[id]/page.tsx).

## World / voice
A clinical **research instrument console** — blackened steel under hairline rule
grid, engraved monospace labels and caps, chamfered square controls, one warm
phosphor-orange accent that lights only live readings. Grammar derived from the
nixie laboratory counter. Brutalist × futuristic × clinical. Never a SaaS
dashboard; no purple gradients, no glassmorphism, no glow beyond a lit reading.

## Tokens
Defined as CSS custom properties in app/globals.css (:root = light default,
.dark = dark primary), consumed via Tailwind v4 @theme inline vars
(bg-ground, text-ink, border-line, text-accent, …). Two-tier neutrals: base
* ground / panel / ink at both ends, plus an emphasized tier *2 (ground-2,
panel-2) and de-emphasized ink-2 / ink-3 for secondary text.
- Ground / panel / ink (light then dark), base + *2:
  - light: ground #f1f0eb · ground-2 #ebe9e2 · panel #faf9f4 · panel-2 #ffffff · ink #171a1f
  - dark (primary): ground #0a0b0d · ground-2 #0d0f12 · panel #121417 · panel-2 #171a1f · ink #e8eaee
- Ink tiers (secondary/tertiary text): light #4c525c / #747b86 · dark #a8b0ba / #78818c
- Accent (single): light #c93300 · dark #ff6a1a (phosphor orange); on-accent text
  --accent-ink: light #ffffff · dark #160a02
- Status: ok #13795c/#3fd08f · warn #8a5a00/#f0b54a · err #c22b2b/#ff5656
- Lines: --line / --line-strong (hairline 1px) · grid 64px with --grid-line
  color-mix(ink 7% / 5% transparent, light/dark)
- Type scale: base 15px/1.6; display clamp(3rem,9vw,6.5rem); H caps 20–30px;
  mono 9.5–13px meta (caps-label 10px/0.16em track)
- Fonts: var(--font-archivo) (Archivo) display+UI, var(--font-plex) (IBM Plex Mono) metadata/code/status/source IDs
- Utilities: .mono (mono family), .tnum (tabular-nums); scrollbars themed to
  --line-strong; carets + ::selection + :focus-visible all accent-on-ink

## Components (component conventions)
- **panel / panel-raised**: 1px bordered chassis surfaces (background + border-line).
- **grid-surface**: 64px hairline rule background; **noise** overlay — low-opacity
  fractal-noise texture (inline SVG feTurbulence, opacity ~0.035) adding film grain.
- **caps-label**: 10px mono, uppercase, 0.16em tracking (metadata labels).
- **led** (nixie lamp): 8px square; states dim / accent / ok / err; pulse animates a live reading.
- **cursor-blink**: square phosphor block, steps blink — streaming / readiness indicator.
- **btn**: square control with 1px strong border, uppercase 0.08em, hover border, active translateY(1px); **btn--accent** phosphor fill using --accent-ink text (btn--square = square padding).
- **icon-btn**: 30px square, hover border+ground; **icon-btn--on** = lit accent.
- **input-frame**: bordered console input; focus-within → accent border + 1px ring.
- **corner-ticks**: 2px corner registration ticks (brutalist detailing) on panels.
- **cite-ref**: inline citation chip (numbered source tube) — mono, hairline border, accent on hover.
- **kbd**: mono keycap, 1px bottom-heavy border.
- **rule / rule--accent**: signature hairline (1px / 2px accent) under section heads.
- **motion**: .anim-rise (staggered home), .anim-fade, .anim-slide-left,
  .anim-drawer (drawers), .anim-sheet (mobile sheets), .pulse-soft; dedicated
  @keyframes + @media (prefers-reduced-motion: reduce) collapses all to ~0ms.
- Sidebar: 264px (desktop) / 64px collapsed rail; grouped history buckets
  (PINNED / TODAY / YESTERDAY / PREVIOUS 7 DAYS / OLDER); search filters live;
  3-dot menu per row (pin/rename/delete); bottom footer: engine LED + theme +
  settings/help + user chip. SVG MedPat logo (instrument frame + ECG + nodes).
- Home: giant spaced-caps MEDPAT (clamp display), top bar (mobile nav button,
  logo, theme toggle, engine badge + LED, DR chip), tagline, one search console
  (Enter run / Shift+Enter newline, blinking cursor, submit arrow), 4
  suggested-readout rows, right-side telemetry column (xl+, hidden), bottom
  status strip; suggestions/telemetry/status caps are mono metadata.
- Chat: header (title inline-rename, pin, RAG-active LED + timer, engine badge,
  sources toggle, more menu — pin/rename/delete); browsing pane with grid
  surface; research-pipeline panel (LED stage lamps, distinct-stage ordering);
  assistant markdown readout with inline citation chips, sources strip
  (Sources · NN + numbered tubes), footer actions (copy / thumbs /
  regenerate); footer input dock (Enter run, Shift+Enter newline, stop button
  while streaming). Empty state = "Ask the literature" + suggestions; missing
  conversation = mono "Err · 404" panel.

## State / feedback
- ResearchStatus: idle · understanding · decomposing · retrieving · reranking ·
  verifying · synthesizing · complete · error. Lamps derive from the distinct
  emitted stage sequence (never a future stage active — no fake progress).
- Message states: queued · streaming · complete · stopped · error (error panel
  with Retry / Copy partial; stopped shows a partial-output banner).
- Stream events drive status/sources/token/done; source panel scopes to the
  active assistant's sources (numbering matches its inline citations).
- Markdown (react-markdown + remark-gfm) renders headings, tables, lists,
  blockquote, inline code, links, themed to the console grammar; [n] tokens in
  text become cite-ref chips (onCite opens+highlights that source card).

## Theme
Dark primary (default), light secondary. class="dark"/"light" on <html>, theme
init script prevents flash, persisted in localStorage (medpat:theme).
prefers-reduced-motion collapses all animation.

## Responsive
Sidebar → collapsed rail (lg) → drawer (mobile, 290px, closes via overlay click
or X button — no Escape handler in the shipped source). Source panel → right
drawer (desktop, 420px / max-w 92vw) → bottom sheet (mobile, max-h 64vh,
closes via overlay click or X). Telemetry column fills unused xl width.
No horizontal overflow at 1440 or 390. Keyboard: ⌘K focus, ⌘N new research
(via useHotkey).

## Raster provenance
Shipping rasters (build captures) live under /home/alapanbagchi/PycharmProjects/MedRag/.impeccable/review/: desktop-home.png, desktop-streaming.png, desktop-chat-complete.png, desktop-sources.png, desktop-seed-chat.png, desktop-light.png, desktop-collapsed.png, mobile-home.png, mobile-nav.png, mobile-chat.png, mobile-sources-sheet.png. The nixie direction quality-bar references: .impeccable/review/nixie-board.webp, nixie-hero.webp. No UI illustration/photo assets are used; the SVG logo is authored source.
