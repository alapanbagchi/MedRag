// ── RAG client boundary ─────────────────────────────────────────────
// Everything outside this module treats the backend as a typed StreamEvent
// feed. Point NEXT_PUBLIC_RAG_API_URL at your MedPat backend and the real
// protocol below starts being used; keep NEXT_PUBLIC_USE_MOCK=true (default)
// to run on the simulated engine instead.
//
// Wire contract (newline-delimited JSON, POST to {base}/v1/chat/stream):
//   {"type":"status","stage":"retrieving","message":"...","count":42}
//   {"type":"sources","sources":[{id,pmcid,pmid,title,authors,journal,year,score,snippet,url}]}
//   {"type":"pipeline","event":"task_start","task_id":"T1",...}   <- verbose trace
//   {"type":"pipeline","event":"llm_call","role":"critic","status":"failed","status_code":429,...}
//   {"type":"token","content":"The"}
//   {"type":"done","timingMs":3127}
// Every line that is not one of status|sources|token|done|error is preserved
// verbatim as a "pipeline" event so the thinking layer shows EVERYTHING the
// backend does (mirroring its Logfire trace).
import type { RAGClient, ResearchStatus, Source, StreamEvent } from "@/lib/types";
import { mockRagEvents } from "@/lib/mock-rag";

const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL?.replace(/\/$/, "") ?? "";
/** The backend base URL ("" in mock mode) — shared with the source panel. */
export const RAG_API_URL = API_URL;
const USE_MOCK = API_URL === "" || process.env.NEXT_PUBLIC_USE_MOCK === "true";

const KNOWN_STAGES: ResearchStatus[] = [
  "understanding", "decomposing", "retrieving", "reranking", "verifying", "synthesizing",
];

function normalizeEvent(raw: unknown): StreamEvent | null {
  if (typeof raw !== "object" || raw === null) return null;
  const e = raw as Record<string, unknown>;
  switch (e.type) {
    case "status": {
      const stage = String(e.stage ?? "idle");
      return {
        type: "status",
        stage: (KNOWN_STAGES as string[]).includes(stage) ? (stage as ResearchStatus) : "idle",
        message: typeof e.message === "string" ? e.message : undefined,
        count: typeof e.count === "number" ? e.count : undefined,
      };
    }
    case "sources":
      return { type: "sources", sources: (Array.isArray(e.sources) ? e.sources : []) as Source[] };
    case "token":
      return { type: "token", content: String(e.content ?? "") };
    case "done":
      return { type: "done", timingMs: typeof e.timingMs === "number" ? e.timingMs : undefined };
    case "error":
      return { type: "error", message: String(e.message ?? "Unknown engine error") };
    case "memory": {
      // research-memory linkage: session resume (prepare) + persistence stats
      // (commit). Backend sends snake_case; the UI speaks camelCase.
      const kind = e.kind === "commit" ? "commit" : "prepare";
      return {
        type: "memory",
        kind,
        sessionId: typeof e.session_id === "string" ? e.session_id : undefined,
        sessionTitle: typeof e.session_title === "string" ? e.session_title : undefined,
        priorClaims: typeof e.prior_claims === "number" ? e.prior_claims : 0,
        priorContradictions: typeof e.prior_contradictions === "number" ? e.prior_contradictions : 0,
        priorGaps: typeof e.prior_gaps === "number" ? e.prior_gaps : 0,
        stats: e.stats && typeof e.stats === "object" ? (e.stats as Record<string, unknown>) : undefined,
      };
    }
    case "pipeline": {
      // verbose trace line from the backend (thinking log)
      const fields = e.fields && typeof e.fields === "object"
        ? (e.fields as Record<string, unknown>)
        : { ...e, event: undefined, type: undefined };
      return { type: "pipeline", event: String(e.event ?? "unknown"), fields };
    }
    default: {
      // ANY other event type the backend emits is preserved for the trace —
      // nothing is silently dropped.
      const name = typeof e.type === "string" ? e.type : "";
      if (!name) return null;
      const { type: _t, ...fields } = e;
      return { type: "pipeline", event: name, fields };
    }
  }
}

async function streamFromBackend(input: {
  question: string;
  conversationId: string;
  signal?: AbortSignal;
  /** Backend research engine: "" (server default/v3) | "xdeep" */
  engine?: "xdeep" | "v3";
  onEvent: (e: StreamEvent) => void;
}): Promise<{ aborted: boolean; error?: string; timingMs?: number }> {
  const started = Date.now();
  let res: Response;
  try {
    res = await fetch(`${API_URL}/v1/chat/stream`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        question: input.question,
        conversation_id: input.conversationId,
        engine: input.engine ?? "",
      }),
      signal: input.signal,
    });
  } catch (err) {
    const aborted = input.signal?.aborted === true;
    return {
      aborted,
      error: aborted ? undefined : `Network error: ${err instanceof Error ? err.message : "request failed"}`,
    };
  }

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    return { aborted: false, error: `HTTP ${res.status} ${detail.slice(0, 200)}` };
  }
  if (!res.body) return { aborted: false, error: "Empty response body" };

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawAny = false;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        let line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (line.startsWith("data:")) line = line.slice(5).trim();
        if (!line) continue;
        sawAny = true;
        try {
          const event = normalizeEvent(JSON.parse(line));
          if (event) input.onEvent(event);
        } catch {
          // skip malformed frame
        }
      }
    }
  } catch (err) {
    if (input.signal?.aborted) return { aborted: true, timingMs: Date.now() - started };
    return { aborted: false, error: `Stream interrupted: ${err instanceof Error ? err.message : "read failed"}` };
  }

  const aborted = input.signal?.aborted === true;
  if (!aborted && !sawAny) {
    input.onEvent({ type: "error", message: "Empty response — the engine returned no events." });
    return { aborted: false, error: "Empty response" };
  }
  return { aborted, timingMs: Date.now() - started };
}

export const ragClient: RAGClient = {
  async streamResearch(input) {
    if (USE_MOCK) {
      const started = Date.now();
      for await (const event of mockRagEvents(input.question, input.signal)) {
        if (input.signal?.aborted) break;
        input.onEvent(event);
      }
      return { aborted: input.signal?.aborted === true, timingMs: Date.now() - started };
    }
    return streamFromBackend(input);
  },
};

export function engineMode(): "MOCK" | "LIVE" {
  return USE_MOCK ? "MOCK" : "LIVE";
}
