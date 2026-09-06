import { create } from "zustand";
import type { ThreadMessageLike } from "@assistant-ui/react";

export interface ChatThread {
  id: string;
  title: string;
  createdAt: number;
  archived?: boolean;
}

const STORAGE_KEY = "medrag:state:v3";

/** Stable empty-message reference (never reallocate: ExternalStoreRuntime
 * compares adapter snapshot references and notifies subscribers on change). */
export const EMPTY_MESSAGES: ThreadMessageLike[] = Object.freeze([]) as unknown as ThreadMessageLike[];

export const uid = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `id-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

export interface InspectorSelection {
  callId: string;
}

interface ChatState {
  threads: ChatThread[];
  currentThreadId: string | null;
  messages: Record<string, ThreadMessageLike[]>;
  /** Singular research pipeline (always xdeep; kept for persisted shape). */
  engine: "xdeep";
  /** Open tool-call inspector (ephemeral; never persisted). */
  inspector: InspectorSelection | null;
  openInspector: (callId: string) => void;
  closeInspector: () => void;

  createThread: () => string;
  selectThread: (id: string) => void;
  renameThread: (id: string, title: string) => void;
  archiveThread: (id: string) => void;
  unarchiveThread: (id: string) => void;
  deleteThread: (id: string) => void;
  patchMessages: (
    threadId: string,
    updater: (messages: ThreadMessageLike[]) => ThreadMessageLike[],
  ) => void;
  setEngine: (engine: "xdeep") => void;
}

interface PersistedState {
  threads: ChatThread[];
  currentThreadId: string | null;
  messages: Record<string, ThreadMessageLike[]>;
  engine: "xdeep";
}

function loadPersisted(): PersistedState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PersistedState;
    if (!Array.isArray(parsed.threads)) return null;
    return { threads: parsed.threads, currentThreadId: parsed.currentThreadId, messages: parsed.messages ?? {}, engine: "xdeep" };
  } catch {
    return null;
  }
}

function initialState(): PersistedState {
  const persisted = typeof localStorage !== "undefined" ? loadPersisted() : null;
  if (persisted && persisted.threads.length > 0) {
    return {
      ...persisted,
      currentThreadId:
        persisted.currentThreadId && persisted.threads.some((t) => t.id === persisted.currentThreadId)
          ? persisted.currentThreadId
          : persisted.threads[0]!.id,
    };
  }
  // Fresh start: mock conversation history so the sidebar shows grouped
  // chats right away; the first (today) thread hosts the auto-played demo.
  const now = Date.now();
  const H = 3600_000;
  const D = 24 * H;
  const mock: Array<{ title: string; createdAt: number }> = [
    { title: "", createdAt: now - 2 * H },
    { title: "PENK for AKI prediction in KID-ACS", createdAt: now - 5 * H },
    { title: "Radial artery vs vein graft vasospasm", createdAt: now - D - 3 * H },
    { title: "Perioperative AKI prediction models", createdAt: now - D - 6 * H },
    { title: "Vasoplegia management evidence gaps", createdAt: now - 3 * D },
    { title: "Contrast-induced nephropathy prophylaxis", createdAt: now - 12 * D },
  ];
  const threads: ChatThread[] = mock.map((m) => ({ id: uid(), ...m }));
  return { threads, currentThreadId: threads[0]!.id, messages: {}, engine: "xdeep" };
}

const boot = initialState();

export const useChatStore = create<ChatState>()((set) => ({
  threads: boot.threads,
  currentThreadId: boot.currentThreadId,
  messages: boot.messages,
  engine: boot.engine,

  createThread: () => {
    const id = uid();
    set((s) => ({
      threads: [...s.threads, { id, title: "", createdAt: Date.now() }],
      currentThreadId: id,
    }));
    return id;
  },
  selectThread: (id) => set({ currentThreadId: id }),
  renameThread: (id, title) =>
    set((s) => ({ threads: s.threads.map((t) => (t.id === id ? { ...t, title } : t)) })),
  archiveThread: (id) =>
    set((s) => ({ threads: s.threads.map((t) => (t.id === id ? { ...t, archived: true } : t)) })),
  unarchiveThread: (id) =>
    set((s) => ({ threads: s.threads.map((t) => (t.id === id ? { ...t, archived: false } : t)) })),
  deleteThread: (id) =>
    set((s) => {
      const threads = s.threads.filter((t) => t.id !== id);
      const messages = { ...s.messages };
      delete messages[id];
      let currentThreadId = s.currentThreadId;
      if (currentThreadId === id) {
        const next = threads.find((t) => !t.archived) ?? threads[0];
        currentThreadId = next?.id ?? null;
      }
      return { threads, messages, currentThreadId };
    }),
  patchMessages: (threadId, updater) =>
    set((s) => ({ messages: { ...s.messages, [threadId]: updater(s.messages[threadId] ?? []) } })),
  setEngine: (engine) => set({ engine }),
  inspector: null,
  openInspector: (callId) => set({ inspector: { callId } }),
  closeInspector: () => set({ inspector: null }),
}));

// Persist to localStorage (best-effort).
useChatStore.subscribe((state) => {
  try {
    const snapshot: PersistedState = {
      threads: state.threads,
      currentThreadId: state.currentThreadId,
      messages: state.messages,
      engine: state.engine,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
  } catch {
    // storage unavailable — in-memory only
  }
});