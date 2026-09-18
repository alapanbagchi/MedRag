/** Composer options: model, web search, deep search (localStorage-backed). */

import { create } from "zustand";

const KEY = "medrag:composer:v1";

export interface ComposerOptions {
  model: string;
  webSearch: boolean;
  deepSearch: boolean;
}

/** The model selected on a fresh load (and whenever the stored value is blank). */
export const DEFAULT_MODEL = "deepseek-v4.1-flash";

const DEFAULTS: ComposerOptions = {
  model: DEFAULT_MODEL,
  webSearch: true,
  deepSearch: true,
};

function load(): ComposerOptions {
  try {
    if (typeof window === "undefined") return DEFAULTS;
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    const parsed = JSON.parse(raw) as Partial<ComposerOptions>;
    return {
      // A blank stored model is the pre-change default; pin it to the house
      // model so the picker is always preselected.
      model:
        typeof parsed.model === "string" && parsed.model
          ? parsed.model
          : DEFAULT_MODEL,
      webSearch: parsed.webSearch !== false,
      deepSearch: parsed.deepSearch !== false,
    };
  } catch {
    return DEFAULTS;
  }
}

function save(state: ComposerOptions): void {
  try {
    if (typeof window !== "undefined") localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // storage unavailable - in-memory only
  }
}

interface State extends ComposerOptions {
  setModel: (model: string) => void;
  setWebSearch: (on: boolean) => void;
  setDeepSearch: (on: boolean) => void;
}

export const useComposerStore = create<State>()((set, get) => ({
  ...load(),
  setModel: (model) => {
    set({ model });
    const { webSearch, deepSearch } = get();
    save({ model, webSearch, deepSearch });
  },
  setWebSearch: (webSearch) => {
    set({ webSearch });
    const { model, deepSearch } = get();
    save({ model, webSearch, deepSearch });
  },
  setDeepSearch: (deepSearch) => {
    set({ deepSearch });
    const { model, webSearch } = get();
    save({ model, webSearch, deepSearch });
  },
}));

/** Snapshot used when a run starts (read outside React). */
export function composerForwardedProps(): Record<string, unknown> {
  const { model, webSearch, deepSearch } = useComposerStore.getState();
  return { model, web_search: webSearch, deep_search: deepSearch };
}
