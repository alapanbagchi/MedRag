// ── MedPat domain types ────────────────────────────────────────────────

export type Role = "user" | "assistant";

/** High-level research-pipeline states surfaced to the UI.
 *  These mirror the backend's status events; the UI never fabricates them. */
export type ResearchStatus =
  | "idle"
  | "understanding"
  | "decomposing"
  | "retrieving"
  | "reranking"
  | "verifying"
  | "synthesizing"
  | "complete"
  | "error";

export interface Source {
  id: string;         // stable identifier, e.g. PMCID
  pmcid?: string;
  pmid?: string;
  title: string;
  authors: string[];
  journal: string;
  year: number;
  /** 0..1 relevance reported by the ranker, when available */
  score?: number;
  snippet?: string;
  url?: string;
}

export interface StageEvent {
  stage: ResearchStatus;
  label: string;      // display label (caps)
  message?: string;   // engine detail line
  count?: number;     // e.g. documents found / passages retained
  at: number;         // epoch ms
}

/** One raw event from the backend's verbose trace (the thinking log).
 *  Mirrors the pipeline's own event vocabulary (task_start, retrieved,
 *  verdict, llm_call, contradiction, ...) verbatim. */
export interface TraceEntry {
  at: number;                       // epoch ms
  event: string;                    // backend event type
  fields: Record<string, unknown>;
}

export type MessageStatus = "queued" | "streaming" | "complete" | "stopped" | "error";

export interface Message {
  id: string;
  role: Role;
  content: string;
  status: MessageStatus;
  sources?: Source[];
  stages?: StageEvent[];
  /** Chronological pipeline / LLM trace for the thinking layer. */
  trace?: TraceEntry[];
  createdAt: number;
  startedAt?: number;
  finishedAt?: number;
  error?: string;
  truncated?: boolean;
  rating?: "up" | "down" | null;
}

export interface Conversation {
  id: string;
  title: string;
  pinned: boolean;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
}

/** Wire events emitted by the RAG client (real backend or mock). */
export type StreamEvent =
  | { type: "status"; stage: ResearchStatus; message?: string; count?: number }
  | { type: "sources"; sources: Source[] }
  | { type: "token"; content: string }
  | { type: "done"; timingMs?: number }
  | { type: "error"; message: string }
  // verbose backend trace line (thinking log) — any event type verbatim
  | { type: "pipeline"; event: string; fields: Record<string, unknown> };

export interface RAGStreamResult {
  aborted: boolean;
  error?: string;
  timingMs?: number;
}

export interface RAGClient {
  /** Stream a research run for one question. Resolves when the stream ends. */
  streamResearch(input: {
    question: string;
    conversationId: string;
    signal?: AbortSignal;
    onEvent: (event: StreamEvent) => void;
  }): Promise<RAGStreamResult>;
}
