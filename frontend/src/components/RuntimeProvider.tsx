import {
  AuiConfig,
  AssistantRuntimeProvider,
  Tools,
  useExternalStoreRuntime,
  type AppendMessage,
  type ExternalStoreAdapter,
  type ThreadMessageLike,
  type ThreadSuggestion,
} from "@assistant-ui/react";
import { useCallback, useMemo, type ReactNode } from "react";
import { toolkit } from "../lib/toolkit";
import { cancelRun, startRun } from "../lib/run";
import { DEMO_QUESTION } from "../lib/demo";
import { EMPTY_MESSAGES, useChatStore, uid } from "../lib/store";

const SUGGESTIONS: ThreadSuggestion[] = [
  {
    title: "AI in radiology",
    label: "thinking, sources, cited answer",
    prompt: DEMO_QUESTION,
  },
  {
    title: "Radial artery vs CABG",
    label: "vasospasm risk by harvest technique",
    prompt:
      "Compare radial artery harvesting for CABG with saphenous vein grafts — does harvest technique affect vasospasm risk?",
  },
  {
    title: "PENK in KID-ACS",
    label: "proenkephalin for AKI prediction",
    prompt: "Is proenkephalin (PENK) predictive of acute kidney injury in KID-ACS patients?",
  },
  {
    title: "Evidence map",
    label: "one answer across many papers",
    prompt:
      "Map the current evidence on perioperative acute kidney injury prediction models in cardiac surgery.",
  },
];

function textOf(message: AppendMessage): string {
  const content = message.content;
  if (typeof content === "string") return content;
  for (const part of content) {
    if (part?.type === "text" && "text" in part) return (part as { text: string }).text;
  }
  return "";
}

function textOfMessage(message: ThreadMessageLike): string {
  const content = message.content;
  if (typeof content === "string") return content;
  for (const part of content) {
    if (part?.type === "text" && "text" in part) return (part as { text: string }).text;
  }
  return "";
}

/**
 * Bridges the zustand store to assistant-ui's ExternalStoreRuntime.
 * Live mode: every submit streams from POST /v1/chat/stream (no demo).
 */
export function RuntimeProvider({ children }: { children: ReactNode }) {
  const currentThreadId = useChatStore((s) => s.currentThreadId);
  const threads = useChatStore((s) => s.threads);
  const messages = useChatStore((s) =>
    s.currentThreadId ? (s.messages[s.currentThreadId] ?? EMPTY_MESSAGES) : EMPTY_MESSAGES,
  );

  // `Tools(...)` is a hook (registers tool-call renderers + model context),
  // so it must be called inside the component — the toolkit itself is static.
  const config = AuiConfig({ tools: Tools({ toolkit }) });

  const onNew = useCallback(async (message: AppendMessage) => {
    const text = textOf(message);
    if (!text.trim()) return;
    const threadId = useChatStore.getState().currentThreadId;
    if (!threadId) return;
    await startRun(threadId, { question: text });
  }, []);

  const onEdit = useCallback(async (message: AppendMessage) => {
    const text = textOf(message);
    if (!text.trim()) return;
    const threadId = useChatStore.getState().currentThreadId;
    if (!threadId) return;
    const st = useChatStore.getState();
    const list = st.messages[threadId] ?? [];
    const parentId = message.parentId;
    const index = parentId ? list.findIndex((m) => m.id === parentId) : -1;
    const edited: ThreadMessageLike = {
      role: "user",
      content: [{ type: "text", text }],
      id: uid(),
      createdAt: new Date(),
    };
    const pre = index >= 0 ? [...list.slice(0, index), edited] : [...list, edited];
    st.patchMessages(threadId, () => pre);
    await startRun(threadId, { question: text });
  }, []);

  const onReload = useCallback(async (parentId: string | null) => {
    const threadId = useChatStore.getState().currentThreadId;
    if (!threadId) return;
    const st = useChatStore.getState();
    const list = st.messages[threadId] ?? [];
    const index = parentId ? list.findIndex((m) => m.id === parentId) : -1;
    const pre = index >= 0 ? list.slice(0, index + 1) : list.filter((m) => m.role !== "assistant");
    const last = pre[pre.length - 1];
    const question = last && last.role === "user" ? textOfMessage(last) : "";
    if (!question) return;
    st.patchMessages(threadId, () => pre);
    await startRun(threadId, { question });
  }, []);

  const onCancel = useCallback(async () => {
    const threadId = useChatStore.getState().currentThreadId;
    if (threadId) cancelRun(threadId);
  }, []);

  // The adapter must keep a STABLE identity unless its data actually changed:
  // the external-store runtime cores compare snapshot references (messages,
  // threads, …) and notify subscribers on every change — a fresh object per
  // render would loop forever.
  const adapter = useMemo<ExternalStoreAdapter<ThreadMessageLike>>(
    () => ({
      messages,
      suggestions: SUGGESTIONS,
      // messages are already ThreadMessageLike — identity conversion.
      convertMessage: (message) => message,
      onNew,
      onEdit,
      onReload,
      onCancel,
      setMessages: (next) => {
        const id = useChatStore.getState().currentThreadId;
        if (id) useChatStore.getState().patchMessages(id, () => [...next]);
      },
      adapters: {
        threadList: {
          threadId: currentThreadId ?? "",
          threads: threads
            .filter((t) => !t.archived)
            .map((t) => ({ status: "regular", id: t.id, title: t.title })),
          archivedThreads: threads
            .filter((t) => t.archived)
            .map((t) => ({ status: "archived", id: t.id, title: t.title })),
          onSwitchToNewThread: () => {
            useChatStore.getState().createThread();
          },
          onSwitchToThread: (id) => {
            useChatStore.getState().selectThread(id);
          },
          onRename: (id, title) => {
            useChatStore.getState().renameThread(id, title);
          },
          onArchive: (id) => {
            useChatStore.getState().archiveThread(id);
          },
          onUnarchive: (id) => {
            useChatStore.getState().unarchiveThread(id);
          },
          onDelete: (id) => {
            useChatStore.getState().deleteThread(id);
          },
        },
      },
    }),
    [messages, threads, currentThreadId, onNew, onEdit, onReload, onCancel],
  );

  const runtime = useExternalStoreRuntime(adapter);

  return (
    <AssistantRuntimeProvider runtime={runtime} config={config}>
      {children}
    </AssistantRuntimeProvider>
  );
}
