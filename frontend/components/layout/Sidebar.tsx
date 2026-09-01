// ── Sidebar: history + pinned + search + shell footer ───────────────
"use client";

import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  ChevronLeft, ChevronsLeft, ExternalLink, HelpCircle, Moon, MoreHorizontal,
  Pencil, Pin, PinOff, Plus, Search, Settings, Sun, Trash2, X,
} from "lucide-react";
import { useApp } from "@/lib/store";
import { ageGroup, cn, formatClock, formatDuration } from "@/lib/utils";
import { Logo } from "@/components/Logo";
import { CapsLabel, Dropdown, Kbd, Led } from "@/components/ui/primitives";
import { engineMode } from "@/lib/rag-client";
import type { Conversation } from "@/lib/types";

function ConvRow({
  conv,
  active,
  collapsed,
  onNavigate,
}: {
  conv: Conversation;
  active: boolean;
  collapsed: boolean;
  onNavigate: () => void;
}) {
  const { togglePin, deleteConversation, renameConversation } = useApp();
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(conv.title);
  const inputRef = useRef<HTMLInputElement>(null);

  const commit = () => {
    renameConversation(conv.id, value.trim());
    setEditing(false);
  };

  if (collapsed) {
    return (
      <Link
        href={`/chat/${conv.id}`}
        onClick={onNavigate}
        title={conv.title}
        aria-label={conv.title}
        className={cn(
          "flex h-9 w-9 items-center justify-center border text-[11px] font-bold",
          active ? "border-accent bg-accent/10 text-accent" : "border-transparent text-ink2 hover:border-line hover:text-ink"
        )}
      >
        {conv.title.charAt(0).toUpperCase()}
      </Link>
    );
  }

  return (
    <div
      className={cn(
        "group relative flex items-center border text-left",
        active ? "border-line-strong bg-ground2" : "border-transparent hover:border-line hover:bg-ground2/60"
      )}
    >
      <span
        aria-hidden
        className={cn("h-full w-[3px] flex-none", active ? "bg-accent" : "bg-transparent")}
      />
      {editing ? (
        <input
          ref={inputRef}
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") setEditing(false);
          }}
          className="h-9 flex-1 border border-accent bg-panel2 px-2.5 text-[13px] text-ink outline-none"
          aria-label="Conversation title"
        />
      ) : (
        <Link
          href={`/chat/${conv.id}`}
          onClick={onNavigate}
          className="h-9 min-w-0 flex-1 px-2.5 py-1"
          title={conv.title}
        >
          <span className={cn("block truncate text-[13px] leading-tight", active ? "font-semibold text-ink" : "text-ink2")}>
            {conv.title}
          </span>
          <span className="mono mt-0.5 block truncate text-[10px] leading-none text-ink3">
            {formatClock(conv.updatedAt)}
          </span>
        </Link>
      )}
      {conv.pinned && !editing && <Pin size={13} className="mx-1 flex-none text-accent" aria-label="Pinned" />}
      {!editing && (
        <div className="flex flex-none pr-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
          <Dropdown
            label={`Options for ${conv.title}`}
            trigger={<MoreHorizontal size={14} />}
            items={[
              {
                key: "pin",
                label: conv.pinned ? "Unpin" : "Pin",
                icon: conv.pinned ? <PinOff size={13} /> : <Pin size={13} />,
                onSelect: () => togglePin(conv.id),
              },
              {
                key: "rename",
                label: "Rename",
                icon: <Pencil size={13} />,
                onSelect: () => {
                  setValue(conv.title);
                  setEditing(true);
                  requestAnimationFrame(() => inputRef.current?.focus());
                },
              },
              {
                key: "delete",
                label: "Delete",
                danger: true,
                icon: <Trash2 size={13} />,
                onSelect: () => deleteConversation(conv.id),
              },
            ]}
          />
        </div>
      )}
    </div>
  );
}

interface SidebarInnerProps {
  collapsed: boolean;
  onNavigate: () => void;
}

function SidebarInner({ collapsed, onNavigate }: SidebarInnerProps) {
  const {
    conversations, sidebarCollapsed, setSidebarCollapsed,
    theme, toggleTheme, createConversation, hydrated,
  } = useApp();
  const router = useRouter();
  const pathname = usePathname();
  const [query, setQuery] = useState("");

  const newResearch = () => {
    const id = createConversation();
    onNavigate();
    router.push(`/chat/${id}`);
  };

  const filter = query.trim().toLowerCase();
  const grouped = useMemo(() => {
    const list = [...conversations].sort(
      (a, b) => Number(b.pinned) - Number(a.pinned) || b.updatedAt - a.updatedAt
    );
    const pinned: Conversation[] = [];
    const buckets: Record<string, Conversation[]> = {
      "TODAY": [], "YESTERDAY": [], "PREVIOUS 7 DAYS": [], "OLDER": [],
    };
    for (const c of list) {
      if (filter && !(c.title + " " + c.messages.map((m) => m.content).join(" ")).toLowerCase().includes(filter)) continue;
      if (c.pinned) pinned.push(c);
      else buckets[ageGroup(c.updatedAt)].push(c);
    }
    return { pinned, buckets };
  }, [conversations, filter]);

  const renderBucket = (label: string, items: Conversation[]) => {
    if (items.length === 0) return null;
    return (
      <div>
        <div className="mb-1 flex items-center justify-between px-1">
          <CapsLabel>{label}</CapsLabel>
          <span className="mono text-[10px] text-ink3 tnum">{String(items.length).padStart(2, "0")}</span>
        </div>
        <div className="flex flex-col gap-[3px]">
          {items.map((c) => (
            <ConvRow key={c.id} conv={c} active={pathname === `/chat/${c.id}`} collapsed={collapsed} onNavigate={onNavigate} />
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-full flex-col">
      {/* brand */}
      <div className={cn("flex items-center border-b border-line px-3", collapsed ? "h-14 justify-center" : "h-14 gap-2")}>
        <Link href="/" onClick={onNavigate} aria-label="MedPat home" className={cn("flex items-center gap-2", collapsed && "justify-center")}>
          <Logo size={24} className="text-ink" />
          {!collapsed && <span className="text-[15px] font-extrabold tracking-[0.18em]">MEDPAT</span>}
        </Link>
        {!collapsed && <span className="mono ml-auto text-[9px] tracking-[0.14em] text-ink3">v0.1</span>}
        {collapsed && (
          <button
            type="button"
            onClick={() => setSidebarCollapsed(false)}
            aria-label="Expand sidebar"
            className="icon-btn absolute -right-4 top-14 hidden border border-line bg-ground lg:inline-flex"
          >
            <ChevronLeft size={14} />
          </button>
        )}
      </div>

      {/* new research */}
      <div className="px-3 pt-3">
        <button type="button" onClick={newResearch} className="btn btn--accent w-full">
          <Plus size={14} strokeWidth={2.5} />
          {!collapsed && "New research"}
        </button>
      </div>

      {/* search */}
      {!collapsed && (
        <div className="px-3 pt-3">
          <div className="input-frame flex h-9 items-center gap-2 px-2.5">
            <Search size={13} className="flex-none text-ink3" aria-hidden />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search conversations…"
              aria-label="Search conversations"
              className="h-full w-full bg-transparent text-[13px] text-ink placeholder:text-ink3 focus:outline-none"
            />
          </div>
        </div>
      )}

      {/* lists */}
      <div className="mt-4 flex-1 space-y-4 overflow-y-auto px-3 pb-4">
        {!hydrated && (
          <p className="mono px-1 pt-2 text-[11px] tracking-[0.14em] text-ink3">SCANNING HISTORY…</p>
        )}
        {grouped.pinned.length > 0 && (
          <div>
            <div className="mb-1 flex items-center justify-between px-1">
              <CapsLabel className="flex items-center gap-1.5"><Pin size={10} className="text-accent" /> Pinned</CapsLabel>
              <span className="mono text-[10px] text-ink3 tnum">{String(grouped.pinned.length).padStart(2, "0")}</span>
            </div>
            <div className="flex flex-col gap-[3px]">
              {grouped.pinned.map((c) => (
                <ConvRow key={c.id} conv={c} active={pathname === `/chat/${c.id}`} collapsed={collapsed} onNavigate={onNavigate} />
              ))}
            </div>
          </div>
        )}
        {!query && conversations.length === 0 && (
          <p className="mono px-1 pt-2 text-[11px] leading-relaxed text-ink3">
            NO CONVERSATIONS YET.
            <br />
            RUN YOUR FIRST RESEARCH.
          </p>
        )}
        {query && grouped.pinned.length === 0 && !["TODAY","YESTERDAY","PREVIOUS 7 DAYS","OLDER"].some((k) => grouped.buckets[k].length > 0) && (
          <p className="mono px-1 pt-2 text-[11px] text-ink3">NO MATCHES.</p>
        )}
        {renderBucket("Today", grouped.buckets["TODAY"])}
        {renderBucket("Yesterday", grouped.buckets["YESTERDAY"])}
        {renderBucket("Previous 7 days", grouped.buckets["PREVIOUS 7 DAYS"])}
        {renderBucket("Older", grouped.buckets["OLDER"])}
      </div>

      {/* footer */}
      <div className="border-t border-line">
        {!collapsed && (
          <div className="flex items-center justify-between px-3 py-2">
            <div className="flex items-center gap-2">
              <Led state={engineMode() === "MOCK" ? "accent" : "ok"} pulse className={cn("!h-1.5 !w-1.5")} />
              <span className="mono text-[10px] uppercase tracking-[0.14em] text-ink3">{engineMode()} ENGINE</span>
            </div>
            <span className="mono text-[10px] text-ink3">PMC · 1.84M</span>
          </div>
        )}
        <div className={cn("flex items-center border-t border-line px-2 py-2", collapsed ? "flex-col gap-1" : "gap-1")}>
          <button type="button" onClick={toggleTheme} aria-label="Toggle theme" className="icon-btn" title="Toggle theme">
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          </button>
          <button type="button" className="icon-btn" aria-label="Settings — coming soon" title="Settings — coming soon" disabled>
            <Settings size={15} />
          </button>
          <button type="button" className="icon-btn" aria-label="Help — coming soon" title="Help — coming soon" disabled>
            <HelpCircle size={15} />
          </button>
          {!collapsed && (
            <button
              type="button"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              title="Collapse sidebar"
              className="icon-btn ml-auto"
            >
              <ChevronsLeft size={15} />
            </button>
          )}
          <div className={cn("mt-1 flex items-center gap-2", collapsed ? "flex-col" : "w-full mt-1 border-t border-line pt-2")}>
            <div className={cn("flex h-7 w-7 flex-none items-center justify-center border border-line-strong text-[10px] font-bold", collapsed && "mt-1")} aria-hidden>
              DR
            </div>
            {!collapsed && (
              <div className="min-w-0">
                <p className="truncate text-[12px] font-semibold leading-tight">D. Researcher</p>
                <p className="mono truncate text-[10px] leading-tight text-ink3">clinical · oncology</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function Sidebar() {
  const { sidebarCollapsed, mobileNavOpen, setMobileNavOpen } = useApp();

  return (
    <>
      {/* desktop rail */}
      <aside
        aria-label="Conversations"
        className={cn(
          "fixed inset-y-0 left-0 z-40 hidden border-r border-line bg-ground transition-[width] duration-300 ease-out lg:block",
          sidebarCollapsed ? "w-16" : "w-[264px]"
        )}
      >
        <SidebarInner collapsed={sidebarCollapsed} onNavigate={() => {}} />
      </aside>

      {/* mobile drawer */}
      {mobileNavOpen && (
        <div className="fixed inset-0 z-50 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <button
            type="button"
            aria-label="Close navigation"
            className="anim-fade absolute inset-0 bg-black/60"
            onClick={() => setMobileNavOpen(false)}
          />
          <div className="anim-drawer absolute inset-y-0 left-0 w-[290px] border-r border-line bg-ground shadow-xl">
            <button
              type="button"
              aria-label="Close navigation"
              className="icon-btn absolute -right-11 top-3 border border-line bg-ground"
              onClick={() => setMobileNavOpen(false)}
            >
              <X size={16} />
            </button>
            <SidebarInner collapsed={false} onNavigate={() => setMobileNavOpen(false)} />
          </div>
        </div>
      )}
    </>
  );
}
