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
  | "umls"
  | "web_search"
  | "web_fetch"
  | "reliability"
  | "verdict"
  | "decompose"
  | "research"
  | "search_round"
  | "contradiction"
  | "resolution"
  | "delegate"
  | "gap_check"
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
  /** Owning agent for thought steps: task id (T1, …) or "orchestrator". */
  agent?: string;
  [key: string]: unknown;
}

export function firstArray(fields: Record<string, unknown> | undefined): unknown[] | null {
  if (!fields) return null;
  for (const v of Object.values(fields)) {
    if (Array.isArray(v) && v.length) return v;
  }
  return null;
}