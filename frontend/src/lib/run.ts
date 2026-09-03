import type { ThreadMessageLike } from "@assistant-ui/react";
import { streamResearch } from "./backend";
import {
  contentOf,
  pushMemory,
  pushPlan,
  pushStep,
  type Content,
  updateStep,
  upsertSources,
  upsertStatus,
  upsertText,
} from "./parts";
import { parseProgress, summaryOf, type StepArgs } from "./xdeep";
import { useChatStore, uid } from "./store";

const controllers = new Map<string, AbortController>();

export interface RunOptions {
  /** Pre-existing message list to continue from (reload / edit). */
  preMessages?: ThreadMessageLike[];
  /** Append the question as a new user message (default true). */
  appendUser?: boolean;
  question: string;
}

type StepPatch = Partial<StepArgs>;

/**
 * Runs the xdeep research pipeline against the backend and streams results
 * into the assistant message identified by `assistantId` inside `threadId`.
 */
export async function startRun(threadId: string, options: RunOptions): Promise<void> {
  const st = useChatStore.getState();
  const existing = st.messages[threadId] ?? [];
  const base = options.preMessages ?? existing;
  const question = options.question.trim();

  let messages = base;
  if (options.appendUser ?? true) {
    messages = [
      ...base,
      { role: "user", content: [{ type: "text", text: question }], id: uid(), createdAt: new Date() },
    ];
    const thread = st.threads.find((t) => t.id === threadId);
    if (thread && !thread.title) st.renameThread(threadId, question.slice(0, 56));
  } else if (base.length === 0) {
    messages = [
      {
        role: "user",
        content: [{ type: "text", text: question }],
        id: uid(),
        createdAt: new Date(),
      },
    ];
  }

  const assistantId = uid();
  const assistant: ThreadMessageLike = {
    role: "assistant",
    content: [],
    id: assistantId,
    createdAt: new Date(),
    status: { type: "running" },
  };
  st.patchMessages(threadId, () => [...messages, assistant]);

  const patch = (updater: (message: ThreadMessageLike) => ThreadMessageLike) => {
    useChatStore.getState().patchMessages(threadId, (list) =>
      list.map((m) => (m.id === assistantId ? updater(m) : m)),
    );
  };
  const patchContent = (fn: (content: Content) => Content) => {
    patch((m) => ({ ...m, content: fn(contentOf(m)) }));
  };

  const controller = new AbortController();
  controllers.set(threadId, controller);
  let cancelled = false;
  controller.signal.addEventListener("abort", () => {
    cancelled = true;
  });

  const finish = (status: ThreadMessageLike["status"]) => {
    patch((m) => ({ ...m, status }));
    controllers.delete(threadId);
  };

  // ---- xdeep event -> step helpers ------------------------------------
  const step = (s: StepArgs) => patchContent((c) => pushStep(c, s));
  const patchStep = (pred: (s: StepArgs) => boolean, patchTo: StepPatch) =>
    patchContent((c) => updateStep(c, pred, patchTo));

  try {
    await streamResearch(
      { question, conversationId: threadId, engine: "xdeep" },
      controller.signal,
      (event) => {
        switch (event.type) {
          case "status":
            patchContent((c) => upsertStatus(c, event.stage, 0));
            break;

          case "pipeline": {
            const name = event.event;
            const fields = event.fields;

            // raw progress line -> step
            if (name === "progress") {
              const parsed = parseProgress((fields as { msg?: string } | undefined)?.msg ?? "");
              if (parsed) step({ ...parsed, done: true });
              break;
            }

            const f = (fields ?? {}) as Record<string, unknown>;

            switch (name) {
              // --- planner ---
              case "decompose_done": {
                const reqs = Array.isArray(f.requirements) ? f.requirements : [];
                step({ kind: "decompose", label: "Decomposing the question", detail: `${reqs.length} research task${reqs.length === 1 ? "" : "s"}`, done: true });
                step({ kind: "research", label: "Dispatching workers", detail: `${reqs.map((r) => (r as { id?: string }).id ?? "?").join(", ")}`, done: true });
                patchContent((c) => pushPlan(c, f));
                break;
              }
              case "master_plan": { // legacy event
                patchContent((c) => pushPlan(c, f));
                break;
              }

              // --- local retrieval ---
              case "query_start": {
                const q = summaryOf(f.query) || "searching…";
                step({ kind: "retrieve", label: "Searching literature", query: q, done: false });
                break;
              }
              case "retrieved": {
                const q = summaryOf(f.query);
                patchStep((s) => s.kind === "retrieve" && !s.done && (!q || s.query === q), {
                  done: true,
                  detail: `${Number(f.count) ?? 0} candidate${Number(f.count) === 1 ? "" : "s"} · ${summaryOf(f.method) || "hybrid"}`,
                });
                break;
              }
              case "search_round": {
                step({ kind: "search_round", label: "Search round", detail: `round ${f.round_no ?? "?"}`, done: true });
                break;
              }

              // --- web gap-fill ---
              case "web_search_started": {
                const q = summaryOf(f.query) || "web search…";
                step({ kind: "web_search", label: "Web search", query: q, done: false });
                break;
              }
              case "web_search_done": {
                const q = summaryOf(f.query);
                patchStep((s) => s.kind === "web_search" && !s.done && (!q || s.query === q), {
                  done: true,
                  detail: `${Number(f.count) ?? 0} trusted result${Number(f.count) === 1 ? "" : "s"}`,
                });
                break;
              }
              case "web_search_failed":
              case "web_search_config_error": {
                const q = summaryOf(f.query);
                patchStep((s) => s.kind === "web_search" && (!q || s.query === q), {
                  done: true,
                  error: summaryOf(f.error) || "search failed",
                });
                break;
              }
              case "web_fetch": {
                const url = summaryOf(f.url) || "";
                step({
                  kind: "web_fetch",
                  label: "Fetching source",
                  url,
                  detail: url.replace(/^https?:\/\/(www\.)?/, "").slice(0, 64) || "web source",
                  sub: f.ok ? `${Number(f.chars) ?? 0} chars` : "failed",
                  done: true,
                });
                break;
              }

              // --- verification (verdicts) ---
              case "verdict": {
                const status = summaryOf(f.status) || "?";
                const support = summaryOf(f.support);
                step({
                  kind: "verdict",
                  label: "Evidence verdict",
                  detail: `${summaryOf(f.evidence_id, 40) || "passage"} → ${status}${support ? ` (${support})` : ""}`,
                  sub: summaryOf(f.note, 80) || undefined,
                  done: true,
                });
                break;
              }
              case "reliability_verdict": {
                step({
                  kind: "reliability",
                  label: "Reliability check",
                  detail: `${summaryOf(f.document_id || f.id, 40) || "source"} · ${summaryOf(f.reliability) || "?"}`,
                  sub: summaryOf(f.note, 80) || undefined,
                  done: true,
                });
                break;
              }
              case "evidence_state": {
                step({ kind: "evidence", label: "Verified evidence", detail: `${Number(f.verified) ?? 0} item${Number(f.verified) === 1 ? "" : "s"}`, done: true });
                break;
              }

              // --- contradictions / gaps ---
              case "contradiction":
                step({ kind: "contradiction", label: "Contradiction detected", detail: summaryOf(f, 96) || "conflicting claims found", done: true });
                break;
              case "resolution":
                step({ kind: "resolution", label: "Resolved contradiction", detail: summaryOf(f, 96) || "resolved", done: true });
                break;
              case "gap_probe":
                step({ kind: "gap_probe", label: "Probing evidence gap", detail: `requirement ${summaryOf(f.requirement_id) || "?"}`, done: true });
                break;
              case "gap_resolution":
                step({ kind: "gap_resolution", label: "Resolving evidence gap", detail: summaryOf(f, 96) || "gap addressed", done: true });
                break;
              case "gaps_reconciled":
                step({ kind: "gap_resolution", label: "Gaps reconciled", detail: `${Number(f.synthesizer_listed) ?? 0} listed`, done: true });
                break;

              // --- synthesis ---
              case "synthesis_start":
                step({ kind: "synthesize", label: "Synthesizing answer", detail: `${Number(f.verified) ?? 0} verified item${Number(f.verified) === 1 ? "" : "s"}`, done: false });
                break;
              case "synthesis_fallback":
                step({ kind: "synthesize", label: "Synthesizing answer", detail: summaryOf(f, 110) || "fallback synthesis", done: false });
                break;
              case "synthesis_done":
                patchStep((s) => s.kind === "synthesize" && !s.done, { done: true, detail: `answer · ${Number(f.answer_len) ?? 0} chars` });
                break;

              default:
                step({ kind: "thought", label: "Thought", detail: `${name}${summaryOf(f, 60) ? ` — ${summaryOf(f, 60)}` : ""}`, done: true });
            }
            break;
          }

          case "sources":
            patchContent((c) => upsertSources(c, event.sources));
            break;

          case "memory":
            patchContent((c) => pushMemory(c, event.kind, event));
            break;

          case "token":
            patchContent((c) => upsertText(c, event.content));
            break;

          case "error":
            step({ kind: "error", label: "Error", detail: event.message ?? "Unknown error", done: true });
            finish({ type: "incomplete", reason: "error", error: event.message ?? "Unknown error" });
            break;

          case "done":
            finish({ type: "complete", reason: "stop" });
            break;
        }
      },
    );
  } catch (error) {
    const isAbort = (error as Error)?.name === "AbortError";
    if (cancelled || isAbort) {
      finish({ type: "incomplete", reason: "cancelled" });
    } else {
      const message = (error as Error)?.message ?? String(error);
      step({ kind: "error", label: "Error", detail: message, done: true });
      finish({ type: "incomplete", reason: "error", error: message });
    }
  }
}

export function cancelRun(threadId: string): void {
  const controller = controllers.get(threadId);
  controller?.abort();
}