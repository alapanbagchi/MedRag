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
  /** Web source: true when this came from a website (not PMC); its url then
   *  carries a #:~:text= fragment that scrolls to + highlights the cited
   *  passage on the actual site. */
  isWeb?: boolean;
  /** The quoted passage that the Text-Fragment URL highlights (web sources). */
  highlight?: string;
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

/** Persistent research memory attached to one assistant response.
 *  Advisory context only — never evidence (the strip is labeled as such). */
export interface MemoryInfo {
  sessionId: string;
  sessionTitle: string;
  /** Prior research surfaced into the planner for this run (L2 memory). */
  priorClaims: number;
  priorContradictions: number;
  priorGaps: number;
  /** What this run persisted back into memory (0 before the commit event). */
  committed?: {
    claimsCommitted: number;
    claimsDeduped: number;
    contradictions: number;
    gaps: number;
    questions: number;
  };
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  status: MessageStatus;
  sources?: Source[];
  stages?: StageEvent[];
  /** Chronological pipeline / LLM trace for the thinking layer. */
  trace?: TraceEntry[];
  /** Research-memory linkage for this response (when the memory layer is on). */
  memory?: MemoryInfo;
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
  // research-memory linkage: emitted before (prepare) and after (commit) the run
  | {
      type: "memory";
      kind: "prepare" | "commit";
      sessionId?: string;
      sessionTitle?: string;
      priorClaims?: number;
      priorContradictions?: number;
      priorGaps?: number;
      /** commit-only: what the run persisted (snake_case from the backend). */
      stats?: Record<string, unknown>;
    }
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
