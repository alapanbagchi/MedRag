/**
 * AG-UI assistant message: the app's rich rendering (thinking stream, tool-call
 * rows, sub-agent cards) plus the standalone cards (plan, sources, verdict,
 * run stats) that the old surface rendered through `MessagePrimitive.Parts`.
 *
 * Standalone cards read the message's AG-UI `data` parts directly; the rich
 * tree handles `step`/`status`/`thinking` through the shared part readers.
 */

import type { ComponentType } from "react";
import { useAuiState } from "@assistant-ui/react";
import { AssistantMessage } from "../components/AssistantMessage";
import { AGUI_DATA_BY_NAME } from "./renderers";

function RenderData({ name, data }: { name: string; data: unknown }) {
  const Renderer = AGUI_DATA_BY_NAME[name] as unknown as
    | ComponentType<{ data: unknown }>
    | undefined;
  if (!Renderer) return null;
  return <Renderer data={data} />;
}

function Standalone({ names, position }: { names: string[]; position: "before" | "after" }) {
  const content = useAuiState((s) => s.message.content);
  const parts = content.filter(
    (part) => part.type === "data" && names.includes((part as { name?: string }).name ?? ""),
  );
  if (parts.length === 0) return null;
  return (
    <div className={position === "before" ? "mb-3 flex flex-col gap-2" : "mt-3 flex flex-col gap-2"}>
      {parts.map((part, index) => (
        <RenderData
          key={index}
          name={(part as { name?: string }).name ?? ""}
          data={(part as { data?: unknown }).data}
        />
      ))}
    </div>
  );
}

export function AgUiAssistantMessage() {
  return (
    <div className="mb-5 flex w-full min-w-0 flex-1 flex-col">
      <AssistantMessage />
      <Standalone
        names={["sources", "verdict_table", "run_stats", "tool_progress", "error_detail"]}
        position="after"
      />
    </div>
  );
}
