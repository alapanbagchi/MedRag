import { AuiIf, ComposerPrimitive } from "@assistant-ui/react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";

export function Composer() {
  return (
    <ComposerPrimitive.Root className="flex w-full items-end gap-2 rounded-3xl border border-border bg-card px-3 py-2.5 shadow-xl shadow-black/[0.04] transition focus-within:border-primary/50 focus-within:ring-2 focus-within:ring-primary/20">
      <ComposerPrimitive.Input
        rows={1}
        autoFocus
        placeholder="Ask the medical literature…"
        className="max-h-40 min-h-[28px] flex-1 resize-none bg-transparent px-2 py-1 text-[15px] leading-6 outline-none placeholder:text-muted-foreground"
      />
      <AuiIf condition={(s) => !s.thread.isRunning}>
        <ComposerPrimitive.Send className="flex size-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-[#4b8cf5] to-[#9d7bfb] text-white shadow-md shadow-indigo-500/25 transition hover:brightness-105 disabled:opacity-40">
          <ArrowUpIcon className="size-4" />
        </ComposerPrimitive.Send>
      </AuiIf>
      <AuiIf condition={(s) => s.thread.isRunning}>
        <ComposerPrimitive.Cancel className="flex size-9 shrink-0 items-center justify-center rounded-full border border-border bg-muted text-foreground transition hover:bg-accent">
          <SquareIcon className="size-3.5" />
        </ComposerPrimitive.Cancel>
      </AuiIf>
    </ComposerPrimitive.Root>
  );
}