import {
  ThreadListItemMorePrimitive,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
} from "@assistant-ui/react";
import {
  ArchiveIcon,
  BotIcon,
  ChevronRightIcon,
  MoreHorizontalIcon,
  MoonIcon,
  PlusIcon,
  SunIcon,
  TrashIcon,
} from "lucide-react";
import { useState } from "react";

function ThreadItem() {
  return (
    <ThreadListItemPrimitive.Root className="group relative flex min-h-9 items-center rounded-xl text-sm transition data-active:bg-muted hover:bg-muted">
      <ThreadListItemPrimitive.Trigger className="min-w-0 flex-1 truncate px-3 py-2 text-left outline-none data-active:font-medium">
        <ThreadListItemPrimitive.Title fallback="New chat" />
      </ThreadListItemPrimitive.Trigger>
      <ThreadListItemMorePrimitive.Root sharedFocusGroup>
        <ThreadListItemMorePrimitive.Trigger className="absolute right-1.5 top-1/2 flex size-7 -translate-y-1/2 items-center justify-center rounded-lg text-muted-foreground opacity-0 transition group-hover:opacity-100 group-has-focus-visible:opacity-100 data-[state=open]:opacity-100">
          <MoreHorizontalIcon className="size-4" />
        </ThreadListItemMorePrimitive.Trigger>
        <ThreadListItemMorePrimitive.Content className="z-50 min-w-40 rounded-xl border border-border bg-popover p-1 shadow-lg">
          <ThreadListItemPrimitive.Archive asChild>
            <ThreadListItemMorePrimitive.Item className="flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm hover:bg-accent">
              <ArchiveIcon className="size-3.5" />
              Archive
            </ThreadListItemMorePrimitive.Item>
          </ThreadListItemPrimitive.Archive>
          <ThreadListItemMorePrimitive.Separator className="my-1 h-px bg-border" />
          <ThreadListItemPrimitive.Delete asChild>
            <ThreadListItemMorePrimitive.Item className="flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm text-destructive hover:bg-destructive/10">
              <TrashIcon className="size-3.5" />
              Delete
            </ThreadListItemMorePrimitive.Item>
          </ThreadListItemPrimitive.Delete>
        </ThreadListItemMorePrimitive.Content>
      </ThreadListItemMorePrimitive.Root>
    </ThreadListItemPrimitive.Root>
  );
}

export function Sidebar() {
  const [dark, setDark] = useState(() =>
    typeof document !== "undefined" ? document.documentElement.classList.contains("dark") : false,
  );

  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("medrag:theme", next ? "dark" : "light");
    } catch {
      // ignore
    }
  };

  return (
    <aside className="flex h-full w-[280px] shrink-0 flex-col border-r border-border bg-[var(--sidebar)]">
      <div className="px-3 pb-2 pt-4">
        <ThreadListPrimitive.New className="flex h-11 w-full items-center justify-center gap-2 rounded-full bg-gradient-to-r from-[#4b8cf5] to-[#9d7bfb] px-4 text-sm font-medium text-white shadow-sm transition hover:brightness-110 data-active:outline data-active:outline-2 data-active:outline-offset-2 data-active:outline-primary">
          <PlusIcon className="size-4" />
          New chat
        </ThreadListPrimitive.New>
      </div>

      <ThreadListPrimitive.Root className="flex min-h-0 flex-1 flex-col px-2">
        <ThreadListPrimitive.Items>
          {() => (
            <div className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-1 py-1.5">
              <ThreadItem />
            </div>
          )}
        </ThreadListPrimitive.Items>
      </ThreadListPrimitive.Root>

      <div className="space-y-3 border-t border-border p-3">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-[#4b8cf5] to-[#9d7bfb] text-white">
            <BotIcon className="size-4" />
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-[13px] font-semibold leading-tight">xdeep</p>
            <p className="truncate text-[11px] text-muted-foreground">deep research</p>
          </div>
          <button
            type="button"
            onClick={toggleTheme}
            className="flex h-8 w-8 items-center justify-center rounded-full text-muted-foreground transition hover:bg-muted hover:text-foreground"
            aria-label="Toggle theme"
          >
            {dark ? <SunIcon className="size-4" /> : <MoonIcon className="size-4" />}
          </button>
        </div>
      </div>
    </aside>
  );
}