import { useMemo } from "react";
import { AgentPlan } from "./assistant-ui/elements/agent-plan";
import { extractPlanSteps } from "../lib/toolkit";
import { EMPTY_MESSAGES, useChatStore } from "../lib/store";

/**
 * Pipeline stage order, earliest first — maps a live run to plan progress.
 * Covers both the live backend states (started/planning/retrieving/
 * verifying/answering) and the legacy xdeep stages.
 */
const STAGE_ORDER = [
  "understanding",
  "started",
  "decomposing",
  "planning",
  "retrieving",
  "reranking",
  "verifying",
  "synthesizing",
  "answering",
];

interface PlanPart {
  type?: string;
  toolName?: string;
  args?: { fields?: Record<string, unknown>; stage?: string };
  result?: { fields?: Record<string, unknown> };
}

/**
 * Live research plan above the composer. Reads the latest assistant
 * message's `plan` tool part (written when the planning tool finishes) and
 * renders it as a task list: checks for done tasks, spinner on the active
 * one. Progress is the pipeline stage as a share of the plan while running,
 * all-done once the run completes. Renders nothing until a plan exists.
 */
export function ComposerPlan() {
  const messages = useChatStore((s) =>
    s.currentThreadId ? (s.messages[s.currentThreadId] ?? EMPTY_MESSAGES) : EMPTY_MESSAGES,
  );

  const plan = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i]!;
      if (message.role !== "assistant" || !Array.isArray(message.content)) continue;
      const parts = message.content as PlanPart[];
      const planPart = parts.find((p) => p?.type === "tool-call" && p.toolName === "plan");
      if (!planPart) continue;
      const fields = planPart.args?.fields ?? planPart.result?.fields;
      const steps = extractPlanSteps(fields).map((s) => s.text);
      if (steps.length === 0) return null;
      const running = (message.status as { type?: string } | undefined)?.type === "running";
      if (!running) return { steps, activeIndex: steps.length };
      const stage = parts.find((p) => p?.type === "tool-call" && p.toolName === "status")?.args?.stage ?? "";
      const position = STAGE_ORDER.indexOf(stage) + 1;
      const activeIndex = Math.min(steps.length - 1, Math.floor((position / (STAGE_ORDER.length + 1)) * steps.length));
      return { steps, activeIndex: Math.max(0, activeIndex) };
    }
    return null;
  }, [messages]);

  if (!plan) return null;
  return (
    <div className="mb-2 overflow-hidden rounded-2xl border border-border bg-white px-4 py-3 shadow-[0_8px_30px_rgba(13,14,26,0.06)]">
      <AgentPlan steps={plan.steps} activeIndex={plan.activeIndex} className="max-w-none" />
    </div>
  );
}
