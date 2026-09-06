import { AuiIf, ComposerPrimitive } from "@assistant-ui/react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { ComposerPlan } from "./ComposerPlan";

/** Roomy multi-line prompt box plus send / stop, with the live plan above it. */
export function Composer() {
  return (
    <div className="w-full">
      <ComposerPlan />
      <div className="w-full overflow-hidden rounded-2xl border border-border bg-[#F7FAFB] shadow-[0_18px_50px_rgba(13,14,26,0.10)] transition focus-within:border-[#1883AE]/50 focus-within:ring-2 focus-within:ring-[#1883AE]/20">
      <ComposerPrimitive.Root className="flex w-full items-end gap-2 px-4 py-3.5">
        <ComposerPrimitive.Input
          rows={4}
          autoFocus
          placeholder="What do you want to know…"
          className="max-h-64 min-h-[112px] flex-1 resize-none bg-transparent py-1 text-[15px] leading-6 outline-none placeholder:text-[#9aa6b1]"
        />
        <AuiIf condition={(s) => !s.thread.isRunning}>
          <ComposerPrimitive.Send
            aria-label="Send"
            className="flex size-10 shrink-0 items-center justify-center rounded-full text-white shadow-[0_6px_18px_rgba(24,174,149,0.45)] transition hover:brightness-105 disabled:opacity-40"
            style={{ background: "linear-gradient(135deg, #1883AE 0%, #18AE95 100%)" }}
          >
            <ArrowUpIcon className="size-4" />
          </ComposerPrimitive.Send>
        </AuiIf>
        <AuiIf condition={(s) => s.thread.isRunning}>
          <ComposerPrimitive.Cancel
            aria-label="Stop"
            className="flex size-10 shrink-0 items-center justify-center rounded-full border border-border bg-white text-[#0D0E1A] transition hover:bg-muted"
          >
            <SquareIcon className="size-4" />
          </ComposerPrimitive.Cancel>
        </AuiIf>
      </ComposerPrimitive.Root>
      </div>
    </div>
  );
}
