"use client";

import { CheckIcon, ChevronRightIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { mono, ShimmerLabel, SwapLabel } from "@/lib/surfaces";

export interface ToolCallProps {
  label: string;
  activeLabel: string;
  query: string;
  running: boolean;
  onOpenChange: () => void;
  className?: string;
}

export function ToolCall({
  label,
  activeLabel,
  query,
  running,
  onOpenChange,
  className,
}: ToolCallProps) {
  return (
    <button
      type="button"
      data-slot="tool-call"
      onClick={onOpenChange}
      title="Open tool-call inspector"
      className={cn("group/trigger text-foreground/55 hover:text-foreground/90 flex w-full max-w-sm items-center gap-2 rounded-md py-1 text-left text-[13.5px] transition-colors outline-none", className)}
    >
      <ChevronRightIcon className="size-3.5 shrink-0 opacity-60" />
      <SwapLabel active={running ? 0 : 1} className="text-start">
        <ShimmerLabel active={running} className="relative inline-block leading-none">
          {activeLabel}
        </ShimmerLabel>
        <>{label}</>
      </SwapLabel>
      <span className={cn(mono, "bg-foreground/[0.06] text-foreground/70 rounded-md px-1.5 py-0.5")}>
        {query}
      </span>
      <span className="ms-auto flex w-4 items-center justify-end">
        {!running && (
          <CheckIcon className="fade-in zoom-in-90 animate-in size-3.5 text-emerald-500 duration-200" />
        )}
      </span>
    </button>
  );
}
