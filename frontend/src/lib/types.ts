/** Wire types for the MedPat/MedRAG streaming API (`backend/api.py`). */

export type Stage =
  | "understanding"
  | "decomposing"
  | "retrieving"
  | "reranking"
  | "verifying"
  | "synthesizing"
  | "complete"
  | (string & {});

export interface Source {
  id?: string;
  pmcid?: string | null;
  pmid?: string | null;
  title?: string;
  authors?: string[];
  journal?: string;
  year?: number;
  score?: number | null;
  snippet?: string;
  url?: string | null;
}

/**
 * A single newline-delimited JSON event of `POST /v1/chat/stream`.
 * `pipeline` mirrors every internal pipeline / LLM observation verbatim.
 * The `thinking` / `tool_call` / `tool_result` / `verdict_table` / `answer`
 * variants are emitted by the stream adapter (`backend/src/agents/stream_adapter.py`).
 * `plan` is the task-list planner's output, streamed first so the plan UI
 * can render before any research streams.
 */
export type BackendEvent =
  | { type: "status"; stage?: string; state?: string; message?: string; count?: number; ts?: string }
  | { type: "thinking"; delta?: string; done?: boolean; ts?: string }
  | { type: "tool_call"; call_id?: string; name?: string; args?: Record<string, unknown>; ts?: string }
  | { type: "tool_result"; call_id?: string; name?: string; ok?: boolean; result?: unknown; truncated?: boolean; ts?: string }
  | { type: "verdict_table"; columns?: string[]; rows?: unknown[] }
  | {
      type: "plan";
      items?: Array<{ id?: string; question?: string; text?: string; deep_research?: boolean }>;
    }
  | { type: "answer"; delta?: string; done?: boolean }
  | { type: "sources"; sources: Source[] }
  | { type: "token"; content: string }
  | { type: "done"; timingMs?: number; usage?: Record<string, unknown>; citations?: unknown[] }
  | { type: "error"; message?: string; code?: string }
  | {
      type: "memory";
      kind: string;
      session_id?: string;
      session_title?: string;
      prior_claims?: number;
      prior_contradictions?: number;
      prior_gaps?: number;
      stats?: Record<string, unknown>;
    }
  | { type: "pipeline"; event: string; fields?: Record<string, unknown> };

export interface ResearchRequest {
  question: string;
  conversationId?: string;
  engine?: "" | "xdeep";
}