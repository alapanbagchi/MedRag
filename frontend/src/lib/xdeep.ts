/**
 * xdeep streaming model.
 *
 * The x_deepagents bridge (`backend/src/x_deepagents/bridge.py`) streams two
 * kinds of pipeline observations:
 *   - structured events  {"type":"pipeline","event":<name>,"fields":<dict>}
 *     (decompose_done, query_start, retrieved, web_search_*, web_fetch,
 *      verdict, reliability_verdict, contradiction, gap_*, synthesis_*, …)
 *   - raw progress lines {"type":"pipeline","event":"progress",
 *     "fields":{"msg":"<line>"}}  ("retrieve: …", "[research:R1] …",
 *      "[web] …", "[verify] …")
 *
 * These are folded into a small set of *step* kinds so the UI can render each
 * as a proper tool card (icon + label + detail + status) instead of a flat
 * bullet log.
 */

export type StepKind =
  | "retrieve"
  | "web_search"
  | "web_fetch"
  | "reliability"
  | "verdict"
  | "decompose"
  | "research"
  | "search_round"
  | "contradiction"
  | "resolution"
  | "gap_probe"
  | "gap_resolution"
  | "evidence"
  | "synthesize"
  | "planner"
  | "join"
  | "done"
  | "thought"
  | "error";

/** Everything stored in a step tool-call part's `args`. */
export interface StepArgs {
  kind: StepKind;
  /** false = still in flight (spinner), true = finished (check). */
  done?: boolean;
  label: string;
  detail?: string;
  sub?: string;
  query?: string;
  url?: string;
  error?: string;
  /** Backend tool-call id (opens the inspector); raw payloads for it. */
  callId?: string;
  rawArgs?: unknown;
  rawResult?: unknown;
  startedTs?: string;
  finishedTs?: string;
  /** Streamed processing timeline for the inspector (append-only). */
  timeline?: { t?: string; text: string }[];
  [key: string]: unknown;
}

/** Parse one raw progress line into a step, or null for unclassifiable lines. */
export function parseProgress(msg: string): StepArgs | null {
  const line = (msg ?? "").trim();
  if (!line) return null;

  const retrieve = /^retrieve:\s*(.+)$/i.exec(line);
  if (retrieve) {
    return { kind: "retrieve", label: "Searching literature", query: retrieve[1]!.trim(), detail: retrieve[1]!.trim() };
  }
  if (/SEARCH PLANNER/i.test(line) || /REPLANNER/i.test(line)) {
    return { kind: "planner", label: "Planning research", detail: line.replace(/\s+/g, " ").slice(0, 96) };
  }
  if (/\[decompose\]/i.test(line)) {
    return { kind: "decompose", label: "Decomposing the question", detail: line.replace(/\s+/g, " ").slice(0, 96) };
  }
  const research = /^(\s*)\[(research:)?R?([0-9A-Z_.-]+)\]\s*(.*)$/i.exec(line);
  if (research && (research[2] || /research/i.test(research[0]))) {
    return {
      kind: "research",
      label: `Task ${research[3]}`,
      detail: (research[4] ?? "").trim() || "researching…",
      sub: "R" + research[3],
    };
  }
  if (line.startsWith("[web]") || /\[web\]/i.test(line)) {
    return { kind: "web_search", label: "Web search", detail: line.replace(/\s+/g, " ").slice(0, 110) };
  }
  const fetch = /\[web-fetch:?([^\]]*)\]/i.exec(line);
  if (fetch) {
    return { kind: "web_fetch", label: "Fetching source", url: fetch[1]!.trim(), detail: fetch[1]!.trim() };
  }
  if (/\[verify\]/i.test(line)) {
    return { kind: "verdict", label: "Verifying evidence", detail: line.replace(/\s+/g, " ").slice(0, 120) };
  }
  if (/\[conflict\]/i.test(line) || /contradiction/i.test(line)) {
    return { kind: "contradiction", label: "Contradiction detected", detail: line.replace(/\s+/g, " ").slice(0, 120) };
  }
  if (/\[gap-resolution\]/i.test(line) || /\[gap\]/i.test(line)) {
    return { kind: "gap_resolution", label: "Resolving evidence gaps", detail: line.replace(/\s+/g, " ").slice(0, 120) };
  }
  if (/\[synthesis|\[synthesize\]/i.test(line)) {
    return { kind: "synthesize", label: "Synthesizing answer", detail: line.replace(/\s+/g, " ").slice(0, 120) };
  }
  if (/\[join\]/i.test(line)) {
    return { kind: "join", label: "Joining findings", detail: line.replace(/\s+/g, " ").slice(0, 120) };
  }
  if (/\[done\]/i.test(line)) {
    return { kind: "done", label: "Finished", detail: line.replace(/\s+/g, " ").slice(0, 120) };
  }
  return { kind: "thought", label: "Thought", detail: line.replace(/\s+/g, " ").slice(0, 140) };
}

/** First-line summary for structured-event fields (many are dicts). */
export function summaryOf(value: unknown, max = 110): string {
  if (typeof value === "string") return value.replace(/\s+/g, " ").slice(0, max);
  if (!value || typeof value !== "object") return "";
  const fields = value as Record<string, unknown>;
  for (const key of ["query", "url", "note", "message", "reason", "text", "error", "title", "id", "document_id"]) {
    const v = fields[key];
    if (typeof v === "string" && v.trim()) return v.replace(/\s+/g, " ").slice(0, max);
  }
  return "";
}

export function firstArray(fields: Record<string, unknown> | undefined): unknown[] | null {
  if (!fields) return null;
  for (const v of Object.values(fields)) {
    if (Array.isArray(v) && v.length) return v;
  }
  return null;
}