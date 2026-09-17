/** Thread metadata + current selection for the AG-UI surface (localStorage-backed). */

import { create } from "zustand";
import { uid } from "../lib/uid";

export interface AgUiThread {
  id: string;
  title: string;
  createdAt: number;
}

export const HISTORY_KEY_PREFIX = "medrag:agui:history:v1:";

const KEY = "medrag:agui:threads:v1";

interface State {
  threads: AgUiThread[];
  currentThreadId: string | null;
  createThread: () => string;
  selectThread: (id: string) => void;
  deleteThread: (id: string) => void;
  renameThread: (id: string, title: string) => void;
}

function load(): { threads: AgUiThread[]; currentThreadId: string | null } {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as { threads?: AgUiThread[]; currentThreadId?: string | null };
      const threads = (parsed.threads ?? []).filter(
        (t): t is AgUiThread => !!t && typeof t === "object" && typeof t.id === "string",
      );
      if (threads.length) {
        return {
          threads,
          currentThreadId:
            parsed.currentThreadId && threads.some((t) => t.id === parsed.currentThreadId)
              ? parsed.currentThreadId
              : threads[0]!.id,
        };
      }
    }
  } catch {
    // storage unavailable — in-memory only
  }
  const id = uid();
  return { threads: [{ id, title: "", createdAt: Date.now() }], currentThreadId: id };
}

const boot = load();

export const useAgUiStore = create<State>()((set) => ({
  threads: boot.threads,
  currentThreadId: boot.currentThreadId,
  createThread: () => {
    const id = uid();
    set((s) => ({
      threads: [...s.threads, { id, title: "", createdAt: Date.now() }],
      currentThreadId: id,
    }));
    return id;
  },
  selectThread: (id) => set({ currentThreadId: id }),
  deleteThread: (id) =>
    set((s) => {
      const remaining = s.threads.filter((t) => t.id !== id);
      const threads = remaining.length ? remaining : [{ id: uid(), title: "", createdAt: Date.now() }];
      const currentThreadId = s.currentThreadId === id ? threads[0]!.id : s.currentThreadId;
      try {
        localStorage.removeItem(HISTORY_KEY_PREFIX + id);
      } catch {
        // ignore
      }
      return { threads, currentThreadId };
    }),
  renameThread: (id, title) =>
    set((s) => ({ threads: s.threads.map((t) => (t.id === id ? { ...t, title } : t)) })),
}));

interface UiState {
  inspectedCallId: string | null;
  inspect: (callId: string) => void;
  closeInspector: () => void;
  /** The left-sliding agentic-flow panel. */
  flowOpen: boolean;
  openFlow: () => void;
  closeFlow: () => void;
  /** Clarification call ids already answered this session. */
  answeredQuestions: Record<string, boolean>;
  markQuestionAnswered: (callId: string) => void;
  clearAnswered: () => void;
}

/** Ephemeral UI state (never persisted): inspector, flow panel, answered ids. */
export const useAgUiUiStore = create<UiState>()((set) => ({
  inspectedCallId: null,
  inspect: (callId) => set({ inspectedCallId: callId }),
  closeInspector: () => set({ inspectedCallId: null }),
  flowOpen: false,
  openFlow: () => set({ flowOpen: true }),
  closeFlow: () => set({ flowOpen: false }),
  answeredQuestions: {},
  markQuestionAnswered: (callId) =>
    set((s) => ({ answeredQuestions: { ...s.answeredQuestions, [callId]: true } })),
  clearAnswered: () => set({ answeredQuestions: {} }),
}));

useAgUiStore.subscribe((s) => {
  try {
    localStorage.setItem(KEY, JSON.stringify({ threads: s.threads, currentThreadId: s.currentThreadId }));
  } catch {
    // storage unavailable
  }
});
