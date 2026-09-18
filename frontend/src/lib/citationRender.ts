/** On-demand citation rendering for styles outside the answer's precomputed set. */

import { create } from "zustand";
import type { CitationRenderResult } from "./citations";

function cacheKey(chatId: string, runId: string, styleId: string): string {
  return chatId + ":" + runId + ":" + styleId;
}

interface State {
  cache: Record<string, CitationRenderResult>;
  loading: Record<string, boolean>;
  error: Record<string, string>;
  /** Render one style for a turn if it is not already cached. */
  ensure: (chatId: string, runId: string, styleId: string) => void;
}

const inflight = new Map<string, Promise<void>>();

export const useCitationRender = create<State>()((set, get) => ({
  cache: {},
  loading: {},
  error: {},
  ensure: (chatId, runId, styleId) => {
    if (!chatId || !styleId) return;
    const key = cacheKey(chatId, runId, styleId);
    if (get().cache[key] || inflight.has(key)) return;
    set((state) => ({
      loading: { ...state.loading, [key]: true },
      error: { ...state.error, [key]: "" },
    }));
    const params = new URLSearchParams({ style: styleId });
    if (runId) params.set("run_id", runId);
    const promise = fetch(
      "/v1/chats/" + encodeURIComponent(chatId) + "/citations?" + params.toString(),
      { method: "POST" },
    )
      .then(async (response) => {
        if (!response.ok) {
          const body = (await response.json().catch(() => ({}))) as { detail?: string };
          throw new Error(body.detail || response.statusText || "render failed");
        }
        return (await response.json()) as CitationRenderResult;
      })
      .then((data) => {
        set((state) => ({
          cache: { ...state.cache, [key]: data },
          loading: { ...state.loading, [key]: false },
        }));
      })
      .catch((error: unknown) => {
        set((state) => ({
          error: {
            ...state.error,
            [key]: error instanceof Error ? error.message : String(error),
          },
          loading: { ...state.loading, [key]: false },
        }));
      })
      .finally(() => {
        inflight.delete(key);
      });
    inflight.set(key, promise);
  },
}));
