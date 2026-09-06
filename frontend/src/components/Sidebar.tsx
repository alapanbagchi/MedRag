import {
  MessageCirclePlusIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
  SearchIcon,
} from "lucide-react";
import { useMemo, useState } from "react";
import { useChatStore } from "../lib/store";

const COLLAPSE_KEY = "medrag:sidebar-collapsed";

function startOfDay(t: number): number {
  const d = new Date(t);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

function groupOf(createdAt: number, now: number): string {
  const day = 24 * 3600_000;
  const today = startOfDay(now);
  if (createdAt >= today) return "Today";
  if (createdAt >= today - day) return "Yesterday";
  if (createdAt >= today - 7 * day) return "Previous 7 days";
  return "Older";
}

function LogoMark({ size = "size-9" }: { size?: string }) {
  return (
    <span
      className={`flex ${size} shrink-0 items-center justify-center rounded-full bg-white text-[#18AE95] ring-1 ring-border`}
    >
      <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round">
        <path d="M12 3v18M4.2 7.5l15.6 9M19.8 7.5l-15.6 9" />
      </svg>
    </span>
  );
}

/**
 * Expandable sidebar: logo + collapse toggle, new-chat button, search,
 * grouped conversation history, and the user row pinned to the bottom.
 */
export function Sidebar({ defaultCollapsed = false }: { defaultCollapsed?: boolean }) {
  const [collapsed, setCollapsed] = useState(() => {
    if (defaultCollapsed) return false;
    try {
      return localStorage.getItem(COLLAPSE_KEY) === "1";
    } catch {
      return false;
    }
  });
  const [query, setQuery] = useState("");
  const threads = useChatStore((s) => s.threads);
  const currentThreadId = useChatStore((s) => s.currentThreadId);
  const createThread = useChatStore((s) => s.createThread);
  const selectThread = useChatStore((s) => s.selectThread);

  const toggle = () => {
    setCollapsed((v) => {
      const next = !v;
      try {
        localStorage.setItem(COLLAPSE_KEY, next ? "1" : "0");
      } catch {
        // ignore
      }
      return next;
    });
  };

  const groups = useMemo(() => {
    const now = Date.now();
    const q = query.trim().toLowerCase();
    const visible = [...threads]
      .filter((t) => !t.archived)
      .filter((t) => !q || (t.title || "New chat").toLowerCase().includes(q))
      .sort((a, b) => b.createdAt - a.createdAt);
    const out: Array<{ label: string; ids: typeof visible }> = [];
    for (const t of visible) {
      const label = groupOf(t.createdAt, now);
      const g = out.find((g) => g.label === label);
      if (g) g.ids.push(t);
      else out.push({ label, ids: [t] });
    }
    return out;
  }, [threads, query]);

  if (collapsed) {
    return (
      <aside className="flex h-full w-20 shrink-0 flex-col items-center border-r border-border bg-white py-4">
        <LogoMark size="size-11" />
        <div className="mt-5 flex flex-col items-center gap-1.5">
          <button
            type="button"
            title="New chat"
            aria-label="New chat"
            onClick={createThread}
            className="flex size-10 items-center justify-center rounded-full text-white shadow-[0_6px_18px_rgba(24,174,149,0.4)] transition hover:brightness-105"
            style={{ background: "linear-gradient(135deg, #1883AE 0%, #18AE95 100%)" }}
          >
            <MessageCirclePlusIcon className="size-5" />
          </button>
          <button
            type="button"
            title="Search chats"
            aria-label="Search chats"
            onClick={toggle}
            className="flex size-10 items-center justify-center rounded-full text-[#0D0E1A] transition hover:bg-muted"
          >
            <SearchIcon className="size-5" />
          </button>
          <button
            type="button"
            title="Expand sidebar"
            aria-label="Expand sidebar"
            onClick={toggle}
            className="flex size-10 items-center justify-center rounded-full text-[#0D0E1A] transition hover:bg-muted"
          >
            <PanelLeftOpenIcon className="size-5" />
          </button>
        </div>
        <div className="mt-auto">
          <span className="flex size-8 items-center justify-center rounded-full bg-[#0D0E1A] text-[11px] font-semibold text-white">
            M
          </span>
        </div>
      </aside>
    );
  }

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-border bg-white">
      <div className="flex items-center gap-2.5 px-4 pb-1 pt-4">
        <LogoMark />
        <span className="font-greeting text-lg font-semibold tracking-tight text-[#0D0E1A]">
          MedRAG
        </span>
        <button
          type="button"
          title="Collapse sidebar"
          aria-label="Collapse sidebar"
          onClick={toggle}
          className="ml-auto flex size-8 items-center justify-center rounded-lg text-muted-foreground transition hover:bg-muted hover:text-foreground"
        >
          <PanelLeftCloseIcon className="size-4" />
        </button>
      </div>

      <div className="px-3 pt-2">
        <button
          type="button"
          onClick={createThread}
          className="flex h-10 w-full items-center justify-center gap-2 rounded-full text-sm font-medium text-white shadow-sm transition hover:brightness-105"
          style={{ background: "linear-gradient(135deg, #1883AE 0%, #18AE95 100%)" }}
        >
          <MessageCirclePlusIcon className="size-4" />
          Start new chat
        </button>
        <label className="mt-2.5 flex items-center gap-2 rounded-full border border-border bg-[#F7FAFB] px-3 py-2 transition focus-within:border-[#1883AE]/50">
          <SearchIcon className="size-3.5 shrink-0 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search chats"
            aria-label="Search chats"
            className="w-full bg-transparent text-[13px] outline-none placeholder:text-[#9aa6b1]"
          />
        </label>
      </div>

      <nav aria-label="Conversation history" className="mt-2 min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {groups.length === 0 ? (
          <p className="px-3 py-6 text-center text-[13px] text-muted-foreground">
            No chats match “{query.trim()}”.
          </p>
        ) : (
          groups.map((g) => (
            <div key={g.label} className="mt-3 first:mt-1">
              <p className="px-3 pb-1 font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                {g.label}
              </p>
              <ul className="space-y-0.5">
                {g.ids.map((t) => {
                  const active = t.id === currentThreadId;
                  return (
                    <li key={t.id}>
                      <button
                        type="button"
                        onClick={() => selectThread(t.id)}
                        className={`flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-[13.5px] transition ${
                          active
                            ? "bg-[#EAF4F6] font-medium text-[#0D0E1A]"
                            : "text-[#3c4450] hover:bg-muted"
                        }`}
                      >
                        <span className="min-w-0 flex-1 truncate">{t.title || "New chat"}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))
        )}
      </nav>

      <div className="border-t border-border p-3">
        <div className="flex items-center gap-2.5 rounded-xl px-1.5 py-1">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-[#0D0E1A] text-[11px] font-semibold text-white">
            M
          </span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-[13px] font-semibold leading-tight text-[#0D0E1A]">
              Researcher
            </span>
            <span className="block truncate text-[11px] text-muted-foreground">local workspace</span>
          </span>
        </div>
      </div>
    </aside>
  );
}
