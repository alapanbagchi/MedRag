/** The full CSL style catalog, fetched once and cached in memory. */

import { create } from "zustand";
import type { CatalogStyle } from "./citations";

interface State {
  styles: CatalogStyle[];
  loading: boolean;
  loaded: boolean;
  error: string;
  /** Fetch the catalog (idempotent; safe to call on every picker open). */
  load: () => Promise<void>;
}

let inflight: Promise<void> | null = null;

export const useCslCatalog = create<State>()((set, get) => ({
  styles: [],
  loading: false,
  loaded: false,
  error: "",
  load: () => {
    if (get().loaded) return Promise.resolve();
    if (inflight) return inflight;
    set({ loading: true, error: "" });
    inflight = fetch("/v1/citations/styles")
      .then((response) => {
        if (!response.ok) throw new Error(response.statusText || "catalog failed");
        return response.json();
      })
      .then((data: { styles?: unknown }) => {
        const list = Array.isArray(data?.styles) ? data.styles : [];
        set({
          styles: list.filter(
            (entry): entry is CatalogStyle =>
              !!entry &&
              typeof entry === "object" &&
              typeof (entry as CatalogStyle).id === "string",
          ),
          loaded: true,
          loading: false,
        });
      })
      .catch((error: unknown) => {
        set({
          error: error instanceof Error ? error.message : String(error),
          loaded: true,
          loading: false,
        });
      })
      .finally(() => {
        inflight = null;
      });
    return inflight;
  },
}));
