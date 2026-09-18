/** Thread sidebar for the AG-UI surface. */

import { useState } from "react";
import { PlusIcon, SearchIcon, Trash2Icon } from "lucide-react";
import { useAgUiStore, useAgUiUiStore } from "./aguiStore";
import { ThemeToggle } from "../components/ThemeToggle";
import { cn } from "../lib/utils";

export function AgUiSidebar() {
 const threads = useAgUiStore((s) => s.threads);
 const currentThreadId = useAgUiStore((s) => s.currentThreadId);
 const createThread = useAgUiStore((s) => s.createThread);
 const selectThread = useAgUiStore((s) => s.selectThread);
 const deleteThread = useAgUiStore((s) => s.deleteThread);
 const [query, setQuery] = useState("");

 const q = query.trim().toLowerCase();
 const shown = q
 ? threads.filter((t) => (t.title ?? "").toLowerCase().includes(q))
 : threads;

 const resetUi = () => {
 useAgUiUiStore.getState().closeInspector();
 useAgUiUiStore.getState().clearAnswered();
 };

 return (
 <aside className="bg-sidebar hidden h-full w-[260px] shrink-0 flex-col border-r border-border/60 px-3 py-4 md:flex">
 <div className="flex items-center gap-2.5 px-2 pb-5">
 <img src="/favicon.png" alt="MedRAG" className="size-10 rounded-[10px] " />
 <span className="text-[17px] font-semibold tracking-tight text-foreground">MEDPAT</span>
 <span className="ml-auto rounded-full border border-border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
 beta
 </span>
 </div>

 <nav className="flex flex-col gap-0.5">
 <button
 type="button"
 onClick={() => {
 createThread();
 resetUi();
 }}
 className="flex items-center gap-3 rounded-full px-3 py-2 text-left text-[13.5px] font-medium text-foreground/85 transition-colors hover:bg-foreground/[0.06] hover:text-foreground"
 >
 <PlusIcon className="size-4 shrink-0" strokeWidth={2} />
 New chat
 </button>
 <label className="mt-0.5 flex items-center gap-3 rounded-full px-3 py-2 text-[13.5px] text-muted-foreground transition-colors focus-within:bg-foreground/[0.06] focus-within:text-foreground">
 <SearchIcon className="size-4 shrink-0" strokeWidth={2} />
 <input
 value={query}
 onChange={(e) => setQuery(e.target.value)}
 placeholder="Search chats"
 className="min-w-0 flex-1 bg-transparent outline-none placeholder:text-muted-foreground"
 />
 </label>
 </nav>

 <div className="mt-6 flex min-h-0 flex-1 flex-col">
 <p className="px-3 pb-1.5 text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground/60">
 Recent
 </p>
 <nav aria-label="Chats" className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto pb-2">
 {shown.length === 0 ? (
 <p className="px-3 py-2 text-[13px] text-muted-foreground/60">No chats yet</p>
 ) : (
 shown.map((thread) => {
 const active = thread.id === currentThreadId;
 return (
 <div
 key={thread.id}
 className={cn(
 "group flex items-center gap-1 rounded-full pr-1.5 transition-colors",
 active
 ? "bg-foreground/[0.09] text-foreground"
 : "text-muted-foreground hover:bg-foreground/[0.05] hover:text-foreground",
 )}
 >
 <button
 type="button"
 onClick={() => {
 selectThread(thread.id);
 resetUi();
 }}
 className="min-w-0 flex-1 truncate px-3 py-2 text-left text-[13.5px]"
 >
 {thread.title || "New chat"}
 </button>
 <button
 type="button"
 aria-label="Delete chat"
 title="Delete chat"
 onClick={() => {
 if (!window.confirm("Delete this chat?")) return;
 deleteThread(thread.id);
 resetUi();
 }}
 className="shrink-0 rounded-full p-1 opacity-0 transition hover:bg-foreground/10 focus-visible:opacity-100 group-hover:opacity-100 group-focus-within:opacity-100"
 >
 <Trash2Icon className="size-3.5" />
 </button>
 </div>
 );
 })
 )}
 </nav>
 </div>

 <div className="mt-2 flex items-center gap-2.5 rounded-full px-2 py-2">
 <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-brand-gradient text-[11px] font-semibold text-brand-foreground">
 M
 </span>
 <span className="min-w-0 flex-1">
 <span className="block truncate text-[13px] font-medium leading-tight text-foreground">
 Researcher
 </span>
 <span className="block truncate text-[11px] text-muted-foreground">local workspace</span>
 </span>
 <ThemeToggle />
 </div>
 </aside>
 );
}
