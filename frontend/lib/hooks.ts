// ── small client hooks ───────────────────────────────────────────────
"use client";

import { useEffect, useRef, useState } from "react";

export function useMounted(): boolean {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () => typeof window !== "undefined" && window.matchMedia(query).matches
  );
  useEffect(() => {
    const mq = window.matchMedia(query);
    const update = () => setMatches(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, [query]);
  return matches;
}

/** Register a key handler. The handler may return true to stop propagation. */
export function useHotkey(
  combo: string,
  handler: (e: KeyboardEvent) => void,
  enabled = true
): void {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      const key = e.key.toLowerCase();
      const parts = combo.toLowerCase().split("+");
      const wantsMod = parts.includes("mod");
      const wantsShift = parts.includes("shift");
      const want = parts[parts.length - 1];
      if (wantsMod !== mod) return;
      if (wantsShift !== e.shiftKey) return;
      if (key !== want) return;
      ref.current(e);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [combo, enabled]);
}
