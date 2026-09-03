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
 */
export type BackendEvent =
  | { type: "status"; stage: string; message?: string; count?: number }
  | { type: "sources"; sources: Source[] }
  | { type: "token"; content: string }
  | { type: "done"; timingMs?: number }
  | { type: "error"; message?: string }
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