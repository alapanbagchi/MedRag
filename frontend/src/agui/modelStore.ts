/** OpenAI-compatible model list for the composer's model picker. */

import { create } from "zustand";

interface State {
  models: string[];
  defaultModel: string;
  loading: boolean;
  loaded: boolean;
  error: string;
  load: () => Promise<void>;
}

let inflight: Promise<void> | null = null;

export const useModelsStore = create<State>()((set, get) => ({
  models: [],
  defaultModel: "",
  loading: false,
  loaded: false,
  error: "",
  load: () => {
    if (get().loaded) return Promise.resolve();
    if (inflight) return inflight;
    set({ loading: true, error: "" });
    inflight = fetch("/v1/models")
      .then(async (response) => {
        if (!response.ok) {
          const body = (await response.json().catch(() => ({}))) as { detail?: string };
          throw new Error(body.detail || response.statusText || "model list failed");
        }
        return response.json();
      })
      .then((data: { models?: unknown; default?: unknown }) => {
        const models = Array.isArray(data?.models)
          ? (data.models as unknown[]).filter((m): m is string => typeof m === "string")
          : [];
        set({
          models,
          defaultModel: typeof data?.default === "string" ? data.default : "",
          loaded: true,
          loading: false,
        });
      })
      .catch((error: unknown) => {
        set({
          error: error instanceof Error ? error.message : String(error),
          loading: false,
          loaded: true,
        });
      })
      .finally(() => {
        inflight = null;
      });
    return inflight;
  },
}));
