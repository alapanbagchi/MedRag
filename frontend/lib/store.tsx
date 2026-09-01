// ── App store: conversations, theme, shell state ─────────────────────
// localStorage-backed today; the reducer shape is the future API boundary.
"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { Conversation, Message } from "@/lib/types";
import { seedConversations } from "@/lib/mock-data";
import { uid } from "@/lib/utils";

export type ThemeMode = "dark" | "light";

const CONV_KEY = "medpat:conversations:v1";
const THEME_KEY = "medpat:theme";

function readStoredConversations(): Conversation[] | null {
  try {
    const raw = localStorage.getItem(CONV_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Conversation[];
    return Array.isArray(parsed) && parsed.length >= 0 ? parsed : null;
  } catch {
    return null;
  }
}

interface AppState {
  theme: ThemeMode;
  hydrated: boolean;
  conversations: Conversation[];
  sidebarCollapsed: boolean;
  mobileNavOpen: boolean;
  setTheme: (t: ThemeMode) => void;
  toggleTheme: () => void;
  setSidebarCollapsed: (v: boolean) => void;
  setMobileNavOpen: (v: boolean) => void;
  createConversation: (title?: string) => string;
  renameConversation: (id: string, title: string) => void;
  togglePin: (id: string) => void;
  deleteConversation: (id: string) => void;
  commitConversation: (id: string, messages: Message[], title?: string) => void;
  getConversation: (id: string) => Conversation | undefined;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<ThemeMode>("dark");
  // start empty on both server and client; hydrate from storage (or seed) after mount
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [hydrated, setHydrated] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  // hydrate from storage, else seed once so first-time visitors see history
  useEffect(() => {
    const stored = readStoredConversations();
    setConversations(stored && stored.length > 0 ? stored : seedConversations());
    const storedTheme = localStorage.getItem(THEME_KEY);
    applyTheme(storedTheme === "light" ? "light" : "dark");
    setThemeState(storedTheme === "light" ? "light" : "dark");
    setHydrated(true);
  }, []);

  // persist conversations (cheap: writes happen on discrete actions)
  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(CONV_KEY, JSON.stringify(conversations));
    } catch {
      // storage full or blocked — non-fatal
    }
  }, [conversations, hydrated]);

  const setTheme = useCallback((t: ThemeMode) => {
    applyTheme(t);
    setThemeState(t);
    try { localStorage.setItem(THEME_KEY, t); } catch { /* noop */ }
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark");
  }, [theme, setTheme]);

  const createConversation = useCallback((title?: string) => {
    const id = uid("conv");
    const conv: Conversation = {
      id,
      title: title ?? "Untitled research",
      pinned: false,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    setConversations((prev) => [conv, ...prev]);
    return id;
  }, []);

  const renameConversation = useCallback((id: string, title: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title: title || c.title, updatedAt: Date.now() } : c))
    );
  }, []);

  const togglePin = useCallback((id: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, pinned: !c.pinned, updatedAt: Date.now() } : c))
    );
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
  }, []);

  const commitConversation = useCallback((id: string, messages: Message[], title?: string) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === id
          ? { ...c, messages, title: title ?? c.title, updatedAt: Date.now() }
          : c
      )
    );
  }, []);

  const getConversation = useCallback(
    (id: string) => conversations.find((c) => c.id === id),
    [conversations]
  );

  const value = useMemo<AppState>(
    () => ({
      theme, hydrated, conversations, sidebarCollapsed, mobileNavOpen,
      setTheme, toggleTheme, setSidebarCollapsed, setMobileNavOpen,
      createConversation, renameConversation, togglePin, deleteConversation,
      commitConversation, getConversation,
    }),
    [theme, hydrated, conversations, sidebarCollapsed, mobileNavOpen, setTheme, toggleTheme, setSidebarCollapsed, setMobileNavOpen, createConversation, renameConversation, togglePin, deleteConversation, commitConversation, getConversation]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

function applyTheme(t: ThemeMode) {
  const root = document.documentElement;
  root.classList.toggle("dark", t === "dark");
  root.classList.toggle("light", t === "light");
}

export function useApp(): AppState {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside <AppProvider>");
  return ctx;
}
