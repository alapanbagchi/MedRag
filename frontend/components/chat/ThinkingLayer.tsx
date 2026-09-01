// ── Research thinking layer ─────────────────────────────────────────
// A chat-style, collapsible "thinking" section attached to each MedPat
// response. While the pipeline runs it streams open and live; when done it
// collapses to a summary header (like ChatGPT's reasoning block). The log
// renders EVERY backend event verbatim — stages, retrievals, verdicts,
// evidence, replans, contradictions, resolutions — including LLM call
// attempts/retries/rate-limit failures, so nothing the engine does is
// hidden. No fabricated progress: the log is whatever the stream emitted.
"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { Message, ResearchStatus, TraceEntry } from "@/lib/types";
import { cn, formatDuration } from "@/lib/utils";
import { CapsLabel, Led } from "@/components/ui/primitives";

const STAGE_ORDER: ResearchStatus[] = [
  "understanding", "decomposing", "retrieving", "reranking", "verifying", "synthesizing",
];

export const STAGE_LABEL: Record<ResearchStatus, string> = {
  idle: "Idle",
  understanding: "Understanding query",
  decomposing: "Decomposing question",
  retrieving: "Retrieving evidence",
  reranking: "Reranking evidence",
  verifying: "Verifying claims",
  synthesizing: "Synthesizing answer",
  complete: "Complete",
  error: "Error",
};

type Kind = "info" | "ok" | "warn" | "err";

interface LogRow {
  key: string;
  kind: Kind;
  glyph: string;
  title: string;
  detail?: string;
  chip?: string; // small right-side mono chip, e.g. "429"
  fields?: Record<string, unknown>; // full payload for the expandable body
}

// -- tiny helpers -----------------------------------------------------

const cap = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const str = (v: unknown) => (v == null ? "" : String(v));
const trunc = (s: string, n: number) => (s.length > n ? s.slice(0, n).trimEnd() + "…" : s);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

function queryCount(q: unknown): number {
  if (Array.isArray(q)) return q.length;
  if (q && typeof q === "object") {
    let n = 0;
    for (const v of Object.values(q as Record<string, unknown>)) {
      n += Array.isArray(v) ? v.length : 1;
    }
    return n;
  }
  return num(q) ?? 0;
}

// -- event -> log row -------------------------------------------------

function describeTrace(entry: TraceEntry): LogRow {
  const { event, fields, at } = entry;
  const f = (k: string) => fields[k];
  let row: LogRow = {
    key: `${event}-${at}-${Math.random().toString(36).slice(2, 6)}`,
    kind: "info", glyph: "•", title: event,
  };
  switch (event) {
    case "run_start":
      row = { ...row, glyph: "▶", title: "Run started", detail: trunc(str(f("question")), 90) };
      break;
    case "master_plan": {
      const tasks = arr(f("tasks"));
      row = {
        ...row, glyph: "▸", title: `Master plan — ${tasks.length} task${tasks.length === 1 ? "" : "s"}`,
        detail: trunc(tasks.map((t) => str((t as Record<string, unknown>).title)).filter(Boolean).join(" · "), 110),
      };
      break;
    }
    case "fallback_plan":
      row = { ...row, kind: "warn", glyph: "!", title: "Master unavailable — single-hop fallback", detail: trunc(str(f("reason")), 130), chip: "FALLBACK" };
      break;
    case "task_start":
      row = { ...row, glyph: "▣", title: `Task ${str(f("task_id"))} · ${str(f("title"))}` };
      break;
    case "task_done":
      row = { ...row, kind: "ok", glyph: "▣", title: `Task ${str(f("task_id"))} — ${str(f("status"))}`, detail: trunc(str(f("summary")), 110) };
      break;
    case "search_round":
      row = { ...row, glyph: "∿", title: `Search round ${str(f("round_no"))}`, detail: `${queryCount(f("queries"))} quer${queryCount(f("queries")) === 1 ? "y" : "ies"} dispatched` };
      break;
    case "retrieved": {
      const papers = arr(f("papers"));
      const ids = papers.map((p) => str((p as Record<string, unknown>).document_id)).filter(Boolean);
      row = {
        ...row, glyph: "●", title: `Retrieved ${papers.length} candidate${papers.length === 1 ? "" : "s"}`,
        detail: `req ${str(f("requirement_id"))}${ids.length ? ` · ${trunc(ids.join(", "), 90)}` : ""}`,
      };
      break;
    }
    case "verdict": {
      const accepted = f("accepted") === true;
      const note = str(f("note"));
      row = {
        ...row, kind: accepted ? "ok" : "info",
        glyph: accepted ? "✓" : "✗",
        title: `${accepted ? "Verified" : "Rejected"} ${str(f("document_id"))}`,
        detail: [
          str(f("support")),
          num(f("confidence")) != null ? `conf ${Number(f("confidence")).toFixed(2)}` : "",
          note ? trunc(note, 120) : "",
        ].filter(Boolean).join(" · "),
      };
      break;
    }
    case "evidence_added":
      row = { ...row, kind: "ok", glyph: "+", title: `Evidence ${str(f("evidence_id"))} · ${str(f("document_id"))} [${str(f("support"))}]` };
      break;
    case "evidence_state":
      row = { ...row, glyph: "→", title: `Evidence ${str(f("evidence_id"))} ${str(f("old_state"))} → ${str(f("new_state"))}` };
      break;
    case "requirement_state":
      row = {
        ...row, glyph: "→", title: `Requirement ${str(f("requirement_id"))} ${str(f("old_state"))} → ${str(f("new_state"))}`,
        detail: `coverage ${str(f("coverage"))}/${str(f("target_n"))}`,
      };
      break;
    case "deep_inspection":
      row = {
        ...row, glyph: "§", title: `Deep inspection ${str(f("document_id"))} — ${str(f("status"))}`,
        detail: [num(f("findings")) != null ? `${f("findings")} findings` : "", f("verified") === true ? "verified" : ""].filter(Boolean).join(" · "),
      };
      break;
    case "worker_report":
      row = {
        ...row, kind: str(f("status")) === "ok" ? "ok" : "warn", glyph: "▤",
        title: `Worker ${str(f("task_id"))} ${str(f("status"))}`,
        detail: [num(f("searches_used")) != null ? `${f("searches_used")} searches` : "", num(f("deep_inspections_used")) != null ? `${f("deep_inspections_used")} deep inspections` : "", str(f("stop_reason"))].filter(Boolean).join(" · "),
      };
      break;
    case "replan":
      row = { ...row, kind: "warn", glyph: "↻", title: `Replanning ${str(f("requirement_id"))}`, detail: trunc(str(f("diagnosis")), 130), chip: "RETRY" };
      break;
    case "contradiction":
      row = { ...row, kind: "warn", glyph: "◇", title: `Contradiction ${str(f("contradiction_id"))} — ${str(f("kind"))}`, detail: trunc(str(f("claim")), 120), chip: "CONFLICT" };
      break;
    case "resolution":
      row = { ...row, glyph: "◇", title: `Resolution ${str(f("contradiction_id"))}: ${str(f("status"))}`, detail: trunc(str(f("explanation")), 130) };
      break;
    case "final_evidence":
      row = {
        ...row, glyph: "Σ", title: `Final evidence — ${str(f("evidence"))} item${f("evidence") === 1 ? "" : "s"}`,
        detail: [num(f("contradictions")) != null ? `${f("contradictions")} contradictions (${f("resolved")} resolved)` : "", Array.isArray(f("gaps")) && arr(f("gaps")).length ? `${arr(f("gaps")).length} gaps` : ""].filter(Boolean).join(" · "),
      };
      break;
    case "answer":
      row = { ...row, glyph: "≈", title: "Answer drafted", detail: trunc(str(f("summary")), 100) };
      break;
    case "run_end":
      row = { ...row, glyph: "■", title: `Run ${str(f("stop_reason"))}`, detail: num(f("confidence")) != null ? `confidence ${Number(f("confidence")).toFixed(2)}` : undefined };
      break;
    case "agent_spawn":
      row = { ...row, glyph: "◈", title: `Agent ${str(f("agent"))} · ${str(f("action"))}` };
      break;
    case "agent_output":
      row = { ...row, glyph: "◈", title: `Agent ${str(f("agent"))} ${str(f("status"))}`, detail: trunc(JSON.stringify(f("output") ?? {}), 90) };
      break;
    case "llm_call": {
      const status = str(f("status"));
      const role = str(f("role"));
      const model = str(f("model"));
      const attempt = num(f("attempt"));
      if (status === "failed") {
        const code = num(f("status_code"));
        row = {
          ...row, kind: "err", glyph: "✕",
          title: `LLM ${role} — failed${attempt != null ? ` (attempt ${attempt})` : ""}`,
          detail: `${model} · ${trunc(str(f("error")), 160)}`,
          chip: code != null ? String(code) : "FAILED",
        };
      } else if (status === "ok") {
        row = { ...row, kind: "ok", glyph: "✓", title: `LLM ${role} — ok${attempt != null ? ` (attempt ${attempt})` : ""}`, detail: model };
      } else {
        row = { ...row, glyph: "⇄", title: `LLM ${role} — calling${attempt != null ? ` (attempt ${attempt})` : ""}`, detail: model };
      }
      break;
    }
    default:
      // every event type is visible, even ones the formatter does not know
      row = { ...row, title: cap(event), detail: trunc(JSON.stringify(fields), 160) };
  }
  // the expanded body shows the VERBATIM payload ("literally everything")
  row.fields = fields;
  return row;
}

// -- expandable post-mortem: every field of the event ----------------

const FIELD_ORDER = [
  "task_id", "requirement_id", "run_id", "question", "query", "queries",
  "papers", "document_id", "chunk_id", "section", "relevance",
  "answers_task", "support", "confidence", "accepted", "note",
  "reason", "diagnosis", "missing_evidence", "evidence_id",
  "old_state", "new_state", "claim", "kind", "explanation",
  "evidence_a", "evidence_b", "additional_papers", "status", "model",
  "role", "attempt", "error", "status_code", "coverage", "target_n",
  "evidence", "gaps", "contradictions", "resolved", "unresolved",
  "summary", "stop_reason", "searches_used", "deep_inspections_used",
  "findings", "verified", "title", "rationale", "tasks", "stop_criteria",
  "entities", "agent", "action", "iteration", "input", "output",
];

const FIELD_LABEL: Record<string, string> = {
  task_id: "Task", requirement_id: "Requirement", run_id: "Run",
  question: "Question", query: "Query", queries: "Queries",
  papers: "Papers fetched", document_id: "Document", chunk_id: "Chunk",
  section: "Section", relevance: "Relevance", answers_task: "Answers task",
  support: "Support", confidence: "Confidence", accepted: "Accepted",
  note: "Critic reason", reason: "Reason", diagnosis: "Diagnosis",
  missing_evidence: "Missing evidence", evidence_id: "Evidence",
  old_state: "From state", new_state: "To state", claim: "Claim",
  kind: "Kind", explanation: "Explanation", evidence_a: "Evidence A",
  evidence_b: "Evidence B", additional_papers: "Additional papers",
  status: "Status", model: "Model", role: "Role", attempt: "Attempt",
  error: "Error", status_code: "HTTP status", coverage: "Coverage",
  target_n: "Target N", evidence: "Evidence items", gaps: "Gaps",
  contradictions: "Contradictions", resolved: "Resolved",
  unresolved: "Unresolved", summary: "Summary", stop_reason: "Stop reason",
  searches_used: "Searches", deep_inspections_used: "Deep inspections",
  findings: "Findings", verified: "Verified", title: "Title",
  rationale: "Rationale", tasks: "Tasks", stop_criteria: "Stop criteria",
  entities: "Entities", agent: "Agent", action: "Action",
  iteration: "Iteration", input: "Input", output: "Output",
};

function InlineValue({ v }: { v: unknown }) {
  const s = String(v);
  if (s.length > 260) {
    return <p className="max-h-[10rem] overflow-y-auto whitespace-pre-wrap text-[11.5px] leading-relaxed text-ink2">{s}</p>;
  }
  return <span className={s ? "text-[11.5px] leading-relaxed text-ink2" : "mono italic text-[10px] text-ink3"}>
    {s || "(empty)"}
  </span>;
}

function ValueView({ v }: { v: unknown }) {
  if (Array.isArray(v)) {
    if (v.length === 0) return <span className="mono text-[10px] italic text-ink3">(none)</span>;
    return (
      <ul className="space-y-1">
        {v.map((item, i) => (
          <li key={i} className="pb-0.5">
            {typeof item === "object" && item !== null ? (
              <div className="border border-line/60 bg-ground/60 p-1.5">
                {Object.entries(item as Record<string, unknown>).map(([k, val]) => (
                  <div key={k} className="flex gap-2 py-0.5">
                    <span className="mono w-28 flex-none truncate text-[9.5px] uppercase tracking-[0.08em] text-ink3">{k}</span>
                    <div className="min-w-0 flex-1"><InlineValue v={val} /></div>
                  </div>
                ))}
              </div>
            ) : (
              <InlineValue v={item} />
            )}
          </li>
        ))}
      </ul>
    );
  }
  if (v !== null && typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>);
    if (entries.length === 0) return <span className="mono text-[10px] italic text-ink3">(empty)</span>;
    return (
      <div className="border border-line/60 bg-ground/60 p-1.5">
        {entries.map(([k, val]) => (
          <div key={k} className="flex gap-2 py-0.5">
            <span className="mono w-28 flex-none truncate text-[9.5px] uppercase tracking-[0.08em] text-ink3">{k}</span>
            <div className="min-w-0 flex-1"><ValueView v={val} /></div>
          </div>
        ))}
      </div>
    );
  }
  return <InlineValue v={v} />;
}

function FieldsPanel({ fields }: { fields: Record<string, unknown> }) {
  const keys = FIELD_ORDER.filter((k) => k in fields).concat(
    Object.keys(fields).filter((k) => !FIELD_ORDER.includes(k))
  );
  return (
    <div>
      {keys.map((k) => (
        <div key={k} className="border-b border-line/50 py-1 last:border-0">
          <span className="mono text-[9px] uppercase tracking-[0.14em] text-ink3">
            {FIELD_LABEL[k] ?? k}
          </span>
          <div className="mt-0.5"><ValueView v={fields[k]} /></div>
        </div>
      ))}
    </div>
  );
}

function RawJson({ fields }: { fields: Record<string, unknown> }) {
  return (
    <details className="group mt-2">
      <summary className="mono flex cursor-pointer list-none items-center gap-1 text-[9px] uppercase tracking-[0.14em] text-ink3 transition-colors hover:text-ink [&::-webkit-details-marker]:hidden">
        Raw event payload
        <ChevronDown size={10} className="transition-transform group-open:rotate-180" />
      </summary>
      <pre className="mono mt-1.5 max-h-[16rem] overflow-auto border border-line bg-ground p-2 text-[10px] leading-relaxed text-ink2">
        {JSON.stringify(fields, null, 2)}
      </pre>
    </details>
  );
}

// One expandable log row. Failures/retries start open (so the trouble is
// immediately visible); the user is free to collapse/expand without the
// parent re-renders (elapsed ticker) forcing it back.
function LogRowDetails({ row, children }: { row: LogRow; children: React.ReactNode }) {
  const [open, setOpen] = useState(row.kind === "warn" || row.kind === "err");
  return (
    <details
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
      className={cn("group", row.kind === "warn" && "bg-warn/[0.07]", row.kind === "err" && "bg-err/[0.07]")}
    >
      {children}
    </details>
  );
}

// -- component --------------------------------------------------------

export function ThinkingLayer({ message }: { message: Message }) {
  const [open, setOpen] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const toggledRef = useRef(false);
  const listRef = useRef<HTMLDivElement>(null);
  const startedAt = message.startedAt ?? message.createdAt;

  const working = message.status === "queued" || message.status === "streaming";
  const trace = message.trace ?? [];
  const stages = message.stages ?? [];
  const rowCount = trace.length;

  // live elapsed ticker while working
  useEffect(() => {
    if (!working) {
      setElapsed(Date.now() - startedAt);
      return;
    }
    const t = window.setInterval(() => setElapsed(Date.now() - startedAt), 250);
    return () => window.clearInterval(t);
  }, [working, startedAt]);

  // ChatGPT-style: auto-open while running, auto-collapse when done
  useEffect(() => {
    if (working) setOpen(true);
    else if (!toggledRef.current && rowCount > 0) setOpen(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [working]);

  // keep the live log pinned to the newest entry while running
  useEffect(() => {
    if (working && listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [rowCount, working]);

  const toggle = () => {
    toggledRef.current = true;
    setOpen((v) => !v);
  };

  const distinctStages: ResearchStatus[] = [];
  for (const e of stages) if (distinctStages[distinctStages.length - 1] !== e.stage) distinctStages.push(e.stage);
  const currentStage = distinctStages[distinctStages.length - 1];
  const stageDoneCount = distinctStages.length;
  const hasLog = rowCount > 0 || stages.length > 0;
  if (!hasLog) return null;

  const durationMs = message.finishedAt && message.startedAt ? message.finishedAt - message.startedAt : elapsed;
  const rows: LogRow[] =
    rowCount > 0
      ? trace.map(describeTrace)
      : stages.map((s, i) => ({
          key: `stage-${i}`,
          kind: "info" as Kind,
          glyph: "▸",
          title: s.label,
          detail: s.message ? (num(s.count) != null ? `${s.message} · ${s.count}` : s.message) : undefined,
        }));

  const busyLabel = currentStage ? STAGE_LABEL[currentStage] ?? currentStage : "starting";
  const statusLed = working ? "accent" : message.status === "error" ? "err" : "ok";
  const statusText = working ? "WORKING" : message.status === "error" ? "INTERRUPTED" : "COMPLETE";

  return (
    <div className="panel corner-ticks mb-4 overflow-hidden">
      {/* header — toggles the thinking log, visible even when collapsed */}
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="flex w-full items-center gap-2.5 border-b border-line px-3 py-2 text-left transition-colors hover:bg-ground2/60"
      >
        <Led state={statusLed} pulse={working} />
        <span className="flex min-w-0 items-center gap-2">
          <CapsLabel>Research process</CapsLabel>
          <span className={`mono text-[9.5px] uppercase tracking-[0.14em] ${working ? "text-accent" : message.status === "error" ? "text-err" : "text-ok"}`}>
            {statusText}
          </span>
        </span>
        <span className="mono ml-auto hidden text-[10px] text-ink3 tnum sm:inline">
          {working ? busyLabel : `${stageDoneCount}/6 stages`} · {rowCount} events
        </span>
        <span className="mono text-[10px] text-ink3 tnum">{formatDuration(durationMs)}</span>
        <ChevronDown size={13} className={`flex-none text-ink2 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div>
          {/* compact stage strip (linear, not a grid) */}
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 border-b border-line bg-ground2/40 px-3 py-1.5">
            <span className="mono text-[9px] uppercase tracking-[0.18em] text-ink3">Stage</span>
            {STAGE_ORDER.map((stage) => {
              const emitted = distinctStages.includes(stage);
              const isCurrent = stage === currentStage && working;
              return (
                <span
                  key={stage}
                  className={`mono text-[9px] uppercase tracking-[0.12em] ${
                    isCurrent ? "text-accent" : emitted ? "text-ok" : "text-ink3/60"
                  }`}
                >
                  {isCurrent ? "▸" : emitted ? "✓ " : ""}
                  {STAGE_LABEL[stage]}
                </span>
              );
            })}
          </div>

          {/* the verbatim event log — every row expands to its full data */}
          <div ref={listRef} className="max-h-[46vh] overflow-y-auto">
            <ol className="divide-y divide-line">
              {rows.map((row) => (
                <LogRowDetails key={row.key} row={row}>
                  <summary className="flex cursor-pointer list-none items-start gap-2.5 px-3 py-1.5 transition-colors hover:bg-ground2/40 [&::-webkit-details-marker]:hidden">
                    <span
                      aria-hidden
                      className={`mono mt-0.5 w-3.5 flex-none text-center text-[11px] leading-5 ${
                        row.kind === "ok" ? "text-ok" : row.kind === "warn" ? "text-warn" : row.kind === "err" ? "text-err" : "text-ink3"
                      }`}
                    >
                      {row.glyph}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline gap-2">
                        <span className="mono text-[11.5px] font-semibold leading-5 text-ink">{row.title}</span>
                        {row.chip && (
                          <span
                            className={`mono flex-none border px-1 py-px text-[8.5px] leading-3 tracking-[0.1em] ${
                              row.kind === "err"
                                ? "border-err/50 text-err"
                                : "border-warn/50 text-warn"
                            }`}
                          >
                            {row.chip}
                          </span>
                        )}
                      </div>
                      {row.detail && (
                        <p className="mono truncate text-[10px] leading-4 text-ink3" title={row.detail}>
                          {row.detail}
                        </p>
                      )}
                    </div>
                    {row.fields && Object.keys(row.fields).length > 0 && (
                      <ChevronDown
                        size={12}
                        className="mt-1 flex-none text-ink3 transition-transform group-open:rotate-180"
                      />
                    )}
                  </summary>
                  {row.fields && Object.keys(row.fields).length > 0 && (
                    <div className="border-t border-line/70 bg-ground2/40 px-3 pb-3 pt-2">
                      <FieldsPanel fields={row.fields} />
                      <RawJson fields={row.fields} />
                    </div>
                  )}
                </LogRowDetails>
              ))}
            </ol>
          </div>
        </div>
      )}
    </div>
  );
}