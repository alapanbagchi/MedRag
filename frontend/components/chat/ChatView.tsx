// ── Chat / research view ─────────────────────────────────────────────
"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowUp, Menu, MoreHorizontal, PanelRight, Pencil, Pin, PinOff, Trash2 } from "lucide-react";
import { useApp } from "@/lib/store";
import { useHotkey, useMediaQuery } from "@/lib/hooks";
import { engineMode, ragClient } from "@/lib/rag-client";
import { SUGGESTED_QUERIES } from "@/lib/mock-data";
import { formatClock, titleFromText, uid } from "@/lib/utils";
import type { Message } from "@/lib/types";
import { Logo } from "@/components/Logo";
import { CapsLabel, Dropdown, Led } from "@/components/ui/primitives";
import { ChatInput } from "@/components/chat/ChatInput";
import { MessageList } from "@/components/chat/MessageList";
import { SourcePanel } from "@/components/sources/SourcePanel";
import { STAGE_LABEL } from "@/components/chat/ThinkingStatus";

export function ChatView({
  conversationId,
  initialQuestion,
}: {
  conversationId: string;
  initialQuestion?: string;
}) {
  const {
    hydrated, getConversation, commitConversation, togglePin,
    deleteConversation, renameConversation, setMobileNavOpen, createConversation,
  } = useApp();
  const router = useRouter();
  const isDesktop = useMediaQuery("(min-width: 1024px)");

  const [messages, setMessages] = useState<Message[]>([]);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [title, setTitle] = useState("");
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleValue, setTitleValue] = useState("");

  const msgRef = useRef<Message[]>([]);
  const busyRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);
  const loadedIdRef = useRef<string | null>(null);
  const bootedRef = useRef(false);
  const inputFocusRef = useRef<(() => void) | null>(null);

  const conv = getConversation(conversationId);

  // hydrate once per conversation id
  useEffect(() => {
    if (!hydrated) return;
    if (loadedIdRef.current === conversationId) return;
    loadedIdRef.current = conversationId;
    const c = getConversation(conversationId);
    if (!c) {
      setNotFound(true);
      return;
    }
    setNotFound(false);
    setTitle(c.title);
    msgRef.current = c.messages;
    setMessages(c.messages);
  }, [hydrated, conversationId]);

  // boot: auto-run a question arriving from the home search
  useEffect(() => {
    if (!hydrated || bootedRef.current) return;
    if (!initialQuestion) return;
    bootedRef.current = true;
    void startStream(initialQuestion);
  }, [hydrated, initialQuestion]);

  // live elapsed ticker while working
  useEffect(() => {
    if (!busyId) return;
    const started = Date.now();
    const t = window.setInterval(() => setElapsed(Date.now() - started), 250);
    return () => window.clearInterval(t);
  }, [busyId]);

  useHotkey("mod+k", () => inputFocusRef.current?.(), true);
  useHotkey("mod+n", () => {
    router.push(`/chat/${createConversation()}`);
  }, true);

  const mutate = (next: Message[]) => {
    msgRef.current = next;
    setMessages(next);
  };

  const patch = (id: string, updater: (m: Message) => Message) => {
    const next = msgRef.current.map((m) => (m.id === id ? updater(m) : m));
    mutate(next);
  };

  const stop = () => {
    abortRef.current?.abort();
  };

  const startStream = async (question: string) => {
    if (busyRef.current) return;
    busyRef.current = true;
    const t0 = Date.now();
    const asstId = uid("m");
    const userMsg: Message = { id: uid("m"), role: "user", content: question, status: "complete", createdAt: t0 };
    const asst: Message = {
      id: asstId, role: "assistant", content: "", status: "queued",
      sources: [], stages: [], createdAt: t0, startedAt: t0,
    };
    mutate([...msgRef.current, userMsg, asst]);
    setBusyId(asstId);

    const controller = new AbortController();
    abortRef.current = controller;
    let engineError: string | undefined;

    const result = await ragClient.streamResearch({
      question,
      conversationId,
      signal: controller.signal,
      onEvent: (e) => {
        if (e.type === "status") {
          patch(asstId, (m) => ({
            ...m,
            status: "streaming",
            stages: [
              ...(m.stages ?? []),
              { stage: e.stage, label: STAGE_LABEL[e.stage] ?? e.stage, message: e.message, count: e.count, at: Date.now() },
            ],
          }));
        } else if (e.type === "sources") {
          patch(asstId, (m) => ({ ...m, sources: e.sources }));
        } else if (e.type === "token") {
          patch(asstId, (m) => ({ ...m, content: m.content + e.content }));
        } else if (e.type === "error") {
          engineError = e.message;
        }
      },
    });

    // finalize
    let finalStatus: Message["status"] = "complete";
    let finalError: string | undefined;
    if (controller.signal.aborted) {
      finalStatus = "stopped";
    } else if (result.error || engineError) {
      finalStatus = "error";
      finalError = result.error ?? engineError;
    }
    patch(asstId, (m) => ({
      ...m,
      status: finalStatus,
      error: finalError,
      truncated: finalStatus === "stopped" && m.content.length > 0,
      finishedAt: Date.now(),
    }));
    busyRef.current = false;
    abortRef.current = null;
    setBusyId(null);
    const nextTitle = titleFromText(question);
    commitConversation(conversationId, msgRef.current, nextTitle);
    setTitle((prev) => (prev === "Untitled research" ? nextTitle : prev));
  };

  const regenerate = (msgId: string) => {
    const msgs = msgRef.current;
    const idx = msgs.findIndex((m) => m.id === msgId);
    if (idx < 0) return;
    const lastUser = [...msgs.slice(0, idx)].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    const cut = msgs.slice(0, msgs.indexOf(lastUser) + 1);
    mutate(cut);
    void startStream(lastUser.content);
  };

  const retry = (msgId: string) => {
    const msgs = msgRef.current;
    const idx = msgs.findIndex((m) => m.id === msgId);
    if (idx < 0) return;
    const lastUser = [...msgs.slice(0, idx)].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    const cut = msgs.slice(0, msgs.indexOf(lastUser) + 1);
    mutate(cut);
    void startStream(lastUser.content);
  };

  const rate = (msgId: string, rating: "up" | "down" | null) => {
    patch(msgId, (m) => ({ ...m, rating }));
    commitConversation(conversationId, msgRef.current);
  };

  const cite = (n: number) => {
    setActiveCitation(n);
    setSourcesOpen(true);
  };

  const commitTitle = () => {
    const v = titleValue.trim();
    if (v) {
      renameConversation(conversationId, v);
      setTitle(v);
    }
    setEditingTitle(false);
  };

  const busy = busyId != null;
  const activeAssistant = [...messages].reverse().find((m) => m.role === "assistant" && m.sources && m.sources.length > 0);

  // auto-scroll while anchored to the bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  });

  if (notFound) {
    return (
      <div className="flex min-h-dvh flex-col items-center justify-center px-6 text-center">
        <p className="mono text-[10px] uppercase tracking-[0.3em] text-ink3">Err · 404</p>
        <h1 className="mt-3 text-[28px] font-black uppercase tracking-tight">Conversation not found</h1>
        <p className="mt-3 max-w-sm text-[14px] text-ink2">
          This research session doesn&apos;t exist or was deleted.
        </p>
        <Link href="/" className="btn btn--accent mt-6">
          Back to MedPat
        </Link>
      </div>
    );
  }

  return (
    <div className="flex h-dvh flex-col">
      {/* header */}
      <header className="flex h-14 flex-none items-center gap-2 border-b border-line px-3 sm:gap-3 sm:px-4">
        <button type="button" onClick={() => setMobileNavOpen(true)} aria-label="Open navigation" className="icon-btn lg:hidden">
          <Menu size={17} />
        </button>
        <Logo size={20} className="hidden text-ink sm:block" accent="var(--accent)" />
        <div className="flex min-w-0 items-center">
          {editingTitle ? (
            <input
              autoFocus
              value={titleValue}
              onChange={(e) => setTitleValue(e.target.value)}
              onBlur={commitTitle}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitTitle();
                if (e.key === "Escape") setEditingTitle(false);
              }}
              aria-label="Rename conversation"
              className="mono h-8 w-64 max-w-[52vw] border border-accent bg-panel2 px-2 text-[12.5px] text-ink outline-none"
            />
          ) : (
            <button
              type="button"
              onClick={() => {
                setTitleValue(title);
                setEditingTitle(true);
              }}
              title="Rename conversation"
              className="max-w-[38vw] truncate text-left text-[13px] font-semibold hover:text-ink2 sm:max-w-[46vw]"
            >
              {title}
            </button>
          )}
        </div>

        <div className="ml-auto flex items-center gap-1 sm:gap-2">
          <span className="mono hidden items-center gap-1.5 border border-line px-2 py-1 text-[9.5px] uppercase tracking-[0.14em] text-ink2 md:inline-flex">
            <Led state={busy ? "accent" : "dim"} pulse={busy} className="!h-1.5 !w-1.5" />
            {busy ? "RAG Active" : "Idle"}
          </span>
          {busy && <span className="mono text-[10px] text-accent tnum">{(elapsed / 1000).toFixed(1)}s</span>}
          <span className="mono hidden items-center border border-line px-2 py-1 text-[9.5px] uppercase tracking-[0.14em] text-ink3 sm:inline-flex">
            {engineMode()}
          </span>
          <button
            type="button"
            onClick={() => setSourcesOpen((v) => !v)}
            aria-label="Toggle sources panel"
            aria-pressed={sourcesOpen}
            className="icon-btn"
          >
            <PanelRight size={15} />
          </button>
          <Dropdown
            label="Conversation options"
            trigger={<MoreHorizontal size={15} />}
            items={[
              {
                key: "pin",
                label: conv?.pinned ? "Unpin" : "Pin",
                icon: conv?.pinned ? <PinOff size={13} /> : <Pin size={13} />,
                onSelect: () => togglePin(conversationId),
              },
              {
                key: "rename",
                label: "Rename",
                icon: <Pencil size={13} />,
                onSelect: () => {
                  setTitleValue(title);
                  setEditingTitle(true);
                },
              },
              {
                key: "delete",
                label: "Delete",
                danger: true,
                icon: <Trash2 size={13} />,
                onSelect: () => {
                  deleteConversation(conversationId);
                  router.push("/");
                },
              },
            ]}
          />
        </div>
      </header>

      {/* messages */}
      <div
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 96;
        }}
        className="grid-surface noise relative min-h-0 flex-1 overflow-y-auto"
      >
        <div className="mx-auto w-full max-w-3xl px-4 pb-8 pt-6 sm:px-6">
          {messages.length === 0 ? (
            <div className="anim-rise flex min-h-[52vh] flex-col items-center justify-center text-center">
              <p className="mono text-[10px] uppercase tracking-[0.3em] text-ink3">
                {engineMode()} engine online
              </p>
              <h2 className="mt-3 text-[30px] font-black uppercase tracking-tight">Ask the literature</h2>
              <p className="mt-2 max-w-md text-[13.5px] text-ink2">
                Research conversations are grounded in retrieved sources — every claim resolves to an
                inspectable paper.
              </p>
              <ul className="mt-6 flex w-full max-w-lg flex-col gap-1.5">
                {SUGGESTED_QUERIES.map((q, i) => (
                  <li key={q}>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void startStream(q)}
                      className="group flex w-full items-center gap-3 border border-transparent px-3 py-2 text-left transition-colors hover:border-line hover:bg-ground2/60 disabled:opacity-40"
                    >
                      <span className="mono w-6 flex-none text-[11px] text-ink3 tnum group-hover:text-accent">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="truncate text-[13px] text-ink2 group-hover:text-ink">{q}</span>
                      <ArrowUp size={12} className="ml-auto flex-none -rotate-45 text-ink3 opacity-0 group-hover:opacity-100" />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <MessageList
              messages={messages}
              onCite={cite}
              onOpenSources={() => setSourcesOpen(true)}
              onRegenerate={(id) => regenerate(id)}
              onRetry={(id) => retry(id)}
              onRate={(id, rating) => rate(id, rating)}
            />
          )}
        </div>
      </div>

      {/* input dock */}
      <div className="flex-none border-t border-line bg-ground px-3 pb-3 pt-3 sm:px-6">
        <div className="mx-auto w-full max-w-3xl">
          <ChatInput
            onSend={(q) => void startStream(q)}
            onStop={stop}
            busy={busy}
            autoFocus={messages.length === 0}
            shortcutRef={inputFocusRef}
          />
        </div>
      </div>

      <SourcePanel
        open={sourcesOpen}
        onClose={() => {
          setSourcesOpen(false);
          setActiveCitation(null);
        }}
        sources={activeAssistant?.sources ?? []}
        activeCitation={activeCitation}
        isDesktop={isDesktop}
        contextualLabel={activeAssistant?.startedAt ? `response · ${formatClock(activeAssistant.startedAt)}` : undefined}
      />
    </div>
  );
}
