import { ActionBarPrimitive, AuiIf, MessagePrimitive, useAuiState } from "@assistant-ui/react";
import { CopyIcon, RotateCcwIcon, SparklesIcon } from "lucide-react";
import type { StepArgs } from "../lib/xdeep";
import { MarkdownText } from "./MarkdownText";
import { StepCard, ThinkingPanel } from "./ThinkingPanel";

/**
 * Tool-call parts that belong inside the collapsible thinking panel:
 *  - "step"   → one research step (tool-card row)
 *  - "status" → the live pipeline stage (pinned to the top)
 *  - "memory" → memory-layer events
 * "plan" and "sources" are standalone and render on their own.
 */
function groupPath(part: {
  type: string;
  toolName?: string;
  args?: unknown;
}): readonly `group-${string}`[] | null {
  if (part.type !== "tool-call") return null;
  if (part.toolName === "status") return ["group-work", "group-status-pin"];
  if (part.toolName === "step") {
    const kind = (part.args as StepArgs | undefined)?.kind ?? "thought";
    return [`group-work`, `group-tool-${kind}`];
  }
  if (part.toolName === "memory") return ["group-work", "group-tool-memory"];
  return null; // plan + sources stay standalone
}

export function AssistantMessage() {
  // Primitive selector: CSV of step kinds currently in flight (stable string).
  const runningKinds = useAuiState((s) => {
    const out: string[] = [];
    for (const p of s.message.content) {
      if (p?.type === "tool-call" && p.toolName === "step") {
        const a = (p.args as StepArgs | undefined);
        if (a && !a.done) out.push(a.kind ?? "thought");
      }
    }
    return out.join(",");
  });

  return (
    <MessagePrimitive.Root className="group flex w-full gap-3.5">
      <div className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-[#4b8cf5] to-[#9d7bfb] text-white shadow-sm">
        <SparklesIcon className="size-4" />
      </div>

      <div className="min-w-0 flex-1 space-y-2.5">
        <MessagePrimitive.GroupedParts groupBy={(part) => groupPath(part as never)}>
          {({ part, children }) => {
            if (part.type.startsWith("group-")) {
              const group = part as never as { type: string; indices: number[] };
              if (group.type === "group-work") {
                return <ThinkingPanel count={group.indices.length}>{children}</ThinkingPanel>;
              }
              if (group.type === "group-status-pin") {
                return <div className="mx-1 mb-1.5 rounded-xl bg-muted/30">{children}</div>;
              }
              if (group.type.startsWith("group-tool-")) {
                const kind = group.type.replace("group-tool-", "");
                const running = runningKinds.split(",").includes(kind);
                return (
                  <StepCard kind={kind} count={group.indices.length} running={running}>
                    {children}
                  </StepCard>
                );
              }
            }
            if (part.type === "tool-call") return part.toolUI ?? null;
            if (part.type === "text") return <MarkdownText />;
            return null;
          }}
        </MessagePrimitive.GroupedParts>

        {/* While running with no answer text yet, show a working indicator. */}
        <AuiIf
          condition={(s) =>
            s.thread.isRunning &&
            !s.message.content.some((p) => p.type === "text" && !!p.text)
          }
        >
          <div className="flex items-center gap-2 px-1 text-sm text-muted-foreground">
            <span className="relative flex size-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
              <span className="relative inline-flex size-2 rounded-full bg-primary" />
            </span>
            Working…
          </div>
        </AuiIf>

        <ActionBarPrimitive.Root hideWhenRunning className="mt-1 flex gap-1">
          <ActionBarPrimitive.Copy
            className="flex size-8 items-center justify-center rounded-lg text-muted-foreground opacity-0 transition group-hover:opacity-100 hover:bg-muted hover:text-foreground focus:opacity-100"
            aria-label="Copy answer"
          >
            <CopyIcon className="size-3.5" />
          </ActionBarPrimitive.Copy>
          <ActionBarPrimitive.Reload
            className="flex size-8 items-center justify-center rounded-lg text-muted-foreground opacity-0 transition group-hover:opacity-100 hover:bg-muted hover:text-foreground focus:opacity-100"
            aria-label="Regenerate answer"
          >
            <RotateCcwIcon className="size-3.5" />
          </ActionBarPrimitive.Reload>
        </ActionBarPrimitive.Root>
      </div>
    </MessagePrimitive.Root>
  );
}