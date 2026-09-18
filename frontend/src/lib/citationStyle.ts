/** The reader's citation-style preference (localStorage-backed, app-wide). */

import { create } from "zustand";

const KEY = "medrag:citation-style:v1";

function load(): string {
  try {
    if (typeof window === "undefined") return "";
    return localStorage.getItem(KEY) ?? "";
  } catch {
    return "";
  }
}

interface State {
  /** Selected style id, or "" to use the answer's default (APA). */
  styleId: string;
  setStyleId: (styleId: string) => void;
}

export const useCitationStyle = create<State>()((set) => ({
  styleId: load(),
  setStyleId: (styleId) => {
    try {
      if (typeof window !== "undefined") localStorage.setItem(KEY, styleId);
    } catch {
      // storage unavailable - in-memory only
    }
    set({ styleId });
  },
}));
