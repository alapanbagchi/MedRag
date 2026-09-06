import type { ThreadMessageLike } from "@assistant-ui/react";
import { DEMO_ANSWER, DEMO_SOURCES, DEMO_STEPS, DEMO_THOUGHTS } from "./demo";
import {
  contentOf,
  pushPlan,
  pushStep,
  upsertSources,
  upsertStatus,
  upsertText,
  type Content,
} from "./parts";
import { useChatStore, uid } from "./store";

const timers = new Map<string, ReturnType<typeof setTimeout>[]>();
const liveRuns = new Set<string>();

/** Lead-in so the submit slide lands before the thinking stream begins. */
const LEAD_MS = 750;

function later(threadId: string, ms: number, fn: () => void): void {
  const list = timers.get(threadId) ?? [];
  list.push(setTimeout(fn, ms));
  timers.set(threadId, list);
}

export function cancelDemoRun(threadId: string): void {
  const list = timers.get(threadId) ?? [];
  for (const t of list) clearTimeout(t);
  timers.delete(threadId);
  liveRuns.delete(threadId);
  useChatStore.getState().patchMessages(threadId, (messages) =>
    messages.map((m) =>
      m.role === "assistant" && m.status?.type === "running"
        ? { ...m, status: { type: "complete", reason: "stop" } }
        : m,
    ),
  );
}

const STAGES = ["understanding", "decomposing", "retrieving", "verifying", "synthesizing"];

/**
 * Offline demo run — no backend. Streams the master-agent thinking trace,
 * tool-call steps, the live source list, then the answer token by token,
 * using the same message-part shapes the real backend stream produces.
 */
export async function startDemoRun(threadId: string, question: string): Promise<void> {
  cancelDemoRun(threadId);
  const st = useChatStore.getState();
  const q = question.trim() || "Can AI replace radiologists?";
  const base = st.messages[threadId] ?? [];
  const withUser: ThreadMessageLike[] = [
    ...base,
    { role: "user", content: [{ type: "text", text: q }], id: uid(), createdAt: new Date() },
  ];
  const thread = st.threads.find((t) => t.id === threadId);
  if (thread && !thread.title) st.renameThread(threadId, q.slice(0, 56));

  const assistantId = uid();
  const assistant: ThreadMessageLike = {
    role: "assistant",
    content: [],
    id: assistantId,
    createdAt: new Date(),
    status: { type: "running" },
  };
  st.patchMessages(threadId, () => [...withUser, assistant]);

  liveRuns.add(threadId);
  await new Promise((r) => setTimeout(r, LEAD_MS));
  if (!liveRuns.has(threadId)) return;

  const patchContent = (fn: (content: Content) => Content) => {
    useChatStore.getState().patchMessages(threadId, (list) =>
      list.map((m) => (m.id === assistantId ? { ...m, content: fn(contentOf(m)) } : m)),
    );
  };
  const finish = () => {
    timers.delete(threadId);
    liveRuns.delete(threadId);
    useChatStore.getState().patchMessages(threadId, (list) =>
      list.map((m) =>
        m.id === assistantId ? { ...m, status: { type: "complete", reason: "stop" } } : m,
      ),
    );
  };

  // 1. Thinking trace + stage lamps stream first.
  DEMO_THOUGHTS.forEach((thought, i) => {
    later(threadId, 350 * (i + 1), () => {
      patchContent((c) => upsertStatus(c, STAGES[Math.min(i, STAGES.length - 1)]!, 0));
      patchContent((c) =>
        pushStep(c, { kind: "thought", label: "Master agent", detail: thought, done: true }),
      );
    });
  });

  // 2. Research plan card.
  later(threadId, 350 * (DEMO_THOUGHTS.length + 1), () => {
    patchContent((c) =>
      pushPlan(c, {
        requirements: [
          { id: "T1", text: "Diagnostic accuracy of AI vs radiologists", target_n: 4 },
          { id: "T2", text: "Workflow and workforce effects", target_n: 3 },
          { id: "T3", text: "Limits, failure modes, and liability", target_n: 3 },
        ],
      }),
    );
  });

  // 3. Tool-call arrangement streams in.
  const stepBase = 350 * (DEMO_THOUGHTS.length + 1) + 500;
  DEMO_STEPS.forEach((step, i) => {
    later(threadId, stepBase + 450 * (i + 1), () => {
      patchContent((c) => upsertStatus(c, "retrieving", 0));
      patchContent((c) => pushStep(c, step));
    });
  });

  // 4. Live source list.
  const sourcesAt = stepBase + 450 * (DEMO_STEPS.length + 1);
  later(threadId, sourcesAt, () => {
    patchContent((c) => upsertStatus(c, "verifying", 0));
    patchContent((c) => upsertSources(c, DEMO_SOURCES));
  });

  // 5. Answer streams token by token (word chunks).
  const chunks = DEMO_ANSWER.split(/(\s+)/).filter((s) => s.length > 0);
  const textStart = sourcesAt + 900;
  chunks.forEach((chunk, i) => {
    later(threadId, textStart + 28 * i, () => {
      if (i === 0) patchContent((c) => upsertStatus(c, "synthesizing", 0));
      patchContent((c) => upsertText(c, chunk));
      if (i === chunks.length - 1) {
        patchContent((c) => upsertStatus(c, "complete", 0));
        finish();
      }
    });
  });
}
