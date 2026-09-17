/**
 * Sub-agent view model for the Grok-style agent stream.
 *
 * Every agent's thoughts stream into the agent sheet as paragraph chat bubbles
 * — the master's and each spawned leg's alike — separated by tool calls and
 * closed by the leg's `response`. The orchestrator's reasoning also feeds the
 * main message (`AssistantMessage`).
 *
 * Input is the assistant message content (AG-UI `data` parts or the NDJSON
 * `tool-call` parts), read through the shared `partName`/`partArgs` helpers so
 * the exact same view model serves both transports.
 */

import { useMemo } from "react";
import { useAuiState } from "@assistant-ui/react";
import { partArgs, partName, taskStatesOf, type Content } from "./parts";
import {
  parseJudgmentReasons,
  parseJudgmentTokens,
  parseVerdicts,
  splitJudgment,
  type EvidencePassage,
} from "./evidence";
import { extractPlanSteps, type PlanStep } from "./toolkit";
import type { StepArgs } from "./xdeep";

export interface SubagentToolItem {
  kind: "tool";
  key: string;
  /** Step kind (drives the icon/label via `stepMeta`). */
  stepKind: string;
  /** Human label — the backend tool name. */
  label: string;
  detail: string;
  done: boolean;
  error?: string;
  /** Raw call args (delegations read `task`/`task_id`/`depth` from here). */
  rawArgs?: Record<string, unknown>;
}

export interface UmlsRow {
  term: string;
  cui?: string;
  concept?: string;
  synonyms: string[];
  semanticTypes: string[];
}

/**
 * UMLS expansion is conversational: a "searching" bubble while it runs, then a
 * "result" bubble that either reports no match or tables the concept found.
 */
export interface SubagentUmlsItem {
  kind: "umls";
  key: string;
  phase: "searching" | "result";
  query: string;
  found: boolean;
  rows: UmlsRow[];
  /** Stream time of this phase (search start / result), for the message stamp. */
  ts?: string;
}

/** One retrieved passage, shown as an attachment box on the tool side. */
export interface RetrievalPassage {
  id: string;
  documentId?: string;
  section?: string;
  chunkType?: string;
  score?: number;
  text: string;
  /** Paper title (from the chunk's document metadata). */
  title?: string;
  journal?: string;
  /** Web source URL (web search only). */
  url?: string;
}

/**
 * Document retrieval as a conversation: the agent asks the local database, the
 * tool answers with article attachments, the agent takes a beat to judge
 * relevance, and the irrelevant attachments then gray out.
 */
export interface SubagentRetrievalItem {
  kind: "retrieval";
  key: string;
  /** Local corpus search or web search. */
  source: "local" | "web";
  query: string;
  done: boolean;
  passages: RetrievalPassage[];
  /** Passage ids the judge kept (everything else dims once verified). */
  relevantIds: string[];
  /** Verdicts were present — only then can anything be dimmed. */
  judged: boolean;
  /** Total judge tokens spent verifying these passages (when reported). */
  tokens?: number;
  /** Critic reason per passage id (kept and rejected). */
  reasons?: Record<string, string>;
}

/**
 * One paragraph of an agent's own narration, shown as an assistant-side chat
 * bubble. A tool call closes the open block, and each block splits on blank
 * lines, so thoughts read as separate messages around the tools.
 */
export interface SubagentSayItem {
  kind: "say";
  key: string;
  text: string;
  /** Stream time the block started (only the first bubble shows it). */
  ts?: string;
}

/**
 * A sub-agent's closing reply (its completion note). The pipeline emits it as
 * a `thinking` delta tagged `origin="response"`; it closes the open thought
 * block first, so it reads as the agent's last word rather than narration.
 */
export interface SubagentResponseItem {
  kind: "response";
  key: string;
  text: string;
  ts?: string;
}

/** One entry in an agent's transcript. */
export type SubagentItem =
  | SubagentSayItem
  | SubagentResponseItem
  | SubagentToolItem
  | SubagentUmlsItem
  | SubagentRetrievalItem;

export interface Subagent {
  id: string;
  /** Full task question (used as the agent's display name). */
  title: string;
  /** Short label for the compact rail card. */
  shortName: string;
  model?: string;
  /**
   * How this leg was delegated to: `plan` for the planner, `deep`/`shallow`
   * for research. Undefined for the master itself. Drives the handoff copy.
   */
  depth?: SubagentDepth;
  color: string;
  /** Stable seed for the agent's randomized face (see `AgentFace`). */
  avatarSeed: number;
  startedTs?: string;
  finishedTs?: string;
  /** Grok-style rail timestamp: "14:43", "Yesterday", "Mar 4". */
  timeLabel: string;
  started: boolean;
  done: boolean;
  /** True while this leg is still producing output. */
  live: boolean;
  /** Interleaved transcript: thoughts (bubbles) + tool calls (cards). */
  items: SubagentItem[];
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Derivation
// ---------------------------------------------------------------------------

function hashOf(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i++) hash = (hash * 31 + value.charCodeAt(i)) | 0;
  return Math.abs(hash);
}

/** HSL → hex for the generated per-agent accent colour. */
function hslToHex(h: number, s: number, l: number): string {
  const sat = s / 100;
  const light = l / 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = sat * Math.min(light, 1 - light);
  const f = (n: number) => light - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  const to = (v: number) => Math.round(v * 255).toString(16).padStart(2, "0");
  return `#${to(f(0))}${to(f(8))}${to(f(4))}`;
}

/**
 * Per-agent visual identity. `slot` is the agent's stable position in the
 * roster (its index in the deterministic id set): golden-angle hue stepping
 * keeps every slot far apart on the colour wheel, so no two agents get the
 * same colour, while the id hash jitters the shade and seeds the face. Both
 * values stay stable for an id across renders, even as new agents appear.
 */
export function agentVisual(
  id: string,
  slot: number,
): { color: string; avatarSeed: number } {
  const jitter = hashOf(id);
  const hue = (slot * 137.508 + ((jitter % 41) - 20) + 360) % 360;
  const color = hslToHex(hue, 60 + (jitter % 32), 54 + ((jitter >> 5) % 16));
  const mixed = (jitter ^ Math.imul(slot + 1, 0x9e3779b1)) >>> 0;
  const avatarSeed = (Math.imul(mixed, 0x85ebca6b) >>> 0) || 1;
  return { color, avatarSeed };
}

/**
 * How a leg was delegated to. The planner is spawned with `depth="plan"`;
 * research legs are `deep` or `shallow`. Defaults to `deep` when the spawn
 * args cannot be recovered.
 */
export type SubagentDepth = "plan" | "deep" | "shallow";

interface MasterSpawn {
  taskId: string;
  task: string;
  depth: SubagentDepth;
}

/**
 * The master's own `spawn_subagent` calls, in order. Only un-namespaced calls
 * count — a research leg's nested delegations don't describe how *it* was
 * spawned. Used to recover each leg's depth, which the `task` event itself
 * does not carry.
 */
function masterSpawns(content: readonly unknown[]): MasterSpawn[] {
  const out: MasterSpawn[] = [];
  for (const part of content) {
    if (partName(part) !== "step") continue;
    const args = partArgs(part) as StepArgs | undefined;
    if (!args || args.kind !== "delegate") continue;
    if (taskIdOfCall(args.callId)) continue;
    const raw = (args.rawArgs ?? {}) as Record<string, unknown>;
    const depth = String(raw.depth ?? "deep").trim().toLowerCase();
    out.push({
      taskId: String(raw.task_id ?? "").trim(),
      task: String(raw.task ?? "").trim(),
      depth: depth === "plan" ? "plan" : depth === "shallow" ? "shallow" : "deep",
    });
  }
  return out;
}

/** The plan's items by id (T1…), from the message's single `plan` part. */
function planItemsOf(content: readonly unknown[]): Map<string, PlanStep> {
  for (const part of content) {
    if (partName(part) !== "plan") continue;
    const args = partArgs(part) as { fields?: Record<string, unknown> } | undefined;
    const result = (part as { result?: { fields?: Record<string, unknown> } }).result;
    const steps = extractPlanSteps(args?.fields ?? result?.fields);
    if (steps.length > 0) return new Map(steps.map((step) => [step.id, step]));
  }
  return new Map();
}

/** Tool-call ids are namespaced `taskId:callId` for delegated legs. */
function taskIdOfCall(callId: string | undefined): string {
  if (!callId) return "";
  const at = callId.indexOf(":");
  return at > 0 ? callId.slice(0, at) : "";
}

interface ParsedUmls {
  term?: string;
  found?: boolean;
  cui?: string;
  preferred_name?: string;
  synonyms?: string[];
  semantic_types?: string[];
}

/** The UMLS tool returns a JSON string — parse it leniently. */
function parseUmlsResult(raw: unknown): ParsedUmls | null {
  try {
    const value = typeof raw === "string" ? (raw.trim() ? JSON.parse(raw) : null) : raw;
    if (!value || typeof value !== "object") return null;
    return value as ParsedUmls;
  } catch {
    return null;
  }
}

/** The looked-up term, from the call args (falls back to the card detail). */
function umlsQueryOf(args: StepArgs): string {
  const raw = args.rawArgs as { term?: unknown } | undefined;
  if (raw && typeof raw.term === "string" && raw.term.trim()) return raw.term.trim();
  return (args.detail || "").replace(/[{}[\]"]/g, " ").replace(/\bterm\b:?/i, "").trim();
}

/** The retrieval query, from the call args. */
function retrieveQueryOf(args: StepArgs): string {
  const raw = args.rawArgs as { query?: unknown } | undefined;
  if (raw && typeof raw.query === "string" && raw.query.trim()) return raw.query.trim();
  return (args.query || args.detail || "").trim();
}

/** Parse a retrieval result: passages plus the judge's kept ids and cost. */
function parseRetrieval(rawResult: unknown): {
  passages: RetrievalPassage[];
  relevantIds: string[];
  judged: boolean;
  tokens?: number;
  reasons?: Record<string, string>;
} {
  const text = typeof rawResult === "string" ? rawResult : "";
  const judgment = text ? splitJudgment(text) : null;
  let value: unknown = rawResult;
  if (text) {
    try {
      value = judgment?.passagesText ? JSON.parse(judgment.passagesText) : null;
    } catch {
      value = null;
    }
  }
  const hits = retrievalHits(value);
  const verdicts = parseVerdicts(judgment?.verdictsText ?? null);
  const passages: RetrievalPassage[] = hits.map((hit, index) => ({
    id: String(hit.chunk_id ?? hit.document_id ?? hit.url ?? `p${index}`),
    documentId: hit.document_id,
    section: hit.section,
    chunkType: hit.chunk_type,
    score: typeof hit.score === "number" ? hit.score : undefined,
    text: hit.text ?? "",
    title: hit.title,
    journal: hit.journal,
    url: hit.url,
  }));
  const relevantIds = verdicts
    ? verdicts.map((verdict) => String(verdict.passage_id ?? "")).filter(Boolean)
    : [];
  const tokens = parseJudgmentTokens(judgment?.verdictsText ?? null);
  const reasons = parseJudgmentReasons(judgment?.verdictsText ?? null);
  return { passages, relevantIds, judged: verdicts !== null, tokens, reasons };
}

/**
 * Pull passages out of a search tool: `local_search` returns a bare array;
 * `web_search` returns `{results, pages, ...}` where the scraped `pages` are
 * what the judge scores (fall back to the listings when no page survived).
 */
function retrievalHits(value: unknown): EvidencePassage[] {
  if (Array.isArray(value)) return value as EvidencePassage[];
  if (value && typeof value === "object") {
    const local = (value as { local?: unknown }).local;
    if (Array.isArray(local)) return local as EvidencePassage[];
    const pages = (value as { pages?: unknown }).pages;
    if (Array.isArray(pages)) {
      const usable = pages.filter(
        (p) => p && typeof p === "object" && (p as { success?: boolean }).success !== false,
      );
      if (usable.length > 0) return usable as EvidencePassage[];
    }
    const results = (value as { results?: unknown }).results;
    if (Array.isArray(results)) return results as EvidencePassage[];
  }
  return [];
}

/**
 * One agent's transcript in stream order: each namespaced tool call lands as
 * its own step (UMLS/retrieval get conversational renderers, everything else a
 * tool card). A running step is replaced in place by its completed version
 * (same call id), so it never renders twice.
 *
 * When `includeThoughts` is set (the master), the agent's own thoughts stream
 * in too: consecutive thought deltas/narration accumulate into one block, every
 * tool call closes it, and each block splits on blank lines into paragraph
 * bubbles. Keys carry the block id, so a growing trailing paragraph updates in
 * place instead of remounting and a tool call starts a fresh message.
 */
function buildAgentItems(
  content: readonly unknown[],
  agentId: string,
  /** Surface this agent's own thoughts as paragraph bubbles. */
  includeThoughts = false,
): SubagentItem[] {
  const items: SubagentItem[] = [];
  const toolIndex = new Map<string, number>();
  const umlsResultsDone = new Set<string>();
  let counter = 0;
  let responseCount = 0;

  // The open thought block: consecutive deltas/narration until a tool call.
  let blockText = "";
  let blockIndex = 0;
  let blockTs: string | undefined;
  const flushThoughts = () => {
    const text = blockText;
    const ts = blockTs;
    blockText = "";
    blockTs = undefined;
    text
      .split(/\n\s*\n/)
      .map((paragraph) => paragraph.trim())
      .filter(Boolean)
      .forEach((paragraph, index) => {
        items.push({ kind: "say", key: `say-${blockIndex}-${index}`, text: paragraph, ts });
      });
    blockIndex += 1;
  };

  for (const part of content) {
    const name = partName(part);
    if (name === "thinking") {
      // Raw streamed deltas: surfaced only for the agent that owns them.
      if (!includeThoughts) continue;
      const thought = partArgs(part) as
        | { agent?: string; delta?: string; origin?: string; ts?: string }
        | undefined;
      const owner = thought?.agent && thought.agent !== "orchestrator" ? thought.agent : "";
      if (!thought?.delta || owner !== agentId) continue;
      // Judge/middleware reasoning is internal, never shown.
      if (thought.origin === "judge") continue;
      // The agent's closing reply: close the open block so it lands as the
      // last message instead of merging into the narration.
      if (thought.origin === "response") {
        const text = thought.delta.trim();
        if (!text) continue;
        flushThoughts();
        items.push({
          kind: "response",
          key: `${agentId || "master"}-response-${responseCount++}`,
          text,
          ts: thought.ts,
        });
        continue;
      }
      blockTs ??= thought.ts;
      blockText += thought.delta;
      continue;
    }
    if (name !== "step") {
      // The retrieval tool fires a progress event with the raw passages
      // *before* the judge runs, so the attachments can appear live.
      if (name === "tool_progress") {
        const progress = partArgs(part) as { callId?: string; result?: unknown } | undefined;
        if (!progress?.callId || taskIdOfCall(progress.callId) !== agentId) continue;
        const index = toolIndex.get(progress.callId);
        if (index === undefined || items[index]!.kind !== "retrieval") continue;
        const current = items[index] as SubagentRetrievalItem;
        const parsed = parseRetrieval(progress.result);
        if (parsed.passages.length > 0) {
          items[index] = { ...current, passages: parsed.passages };
        }
      }
      continue;
    }
    const args = partArgs(part) as StepArgs | undefined;
    if (!args) continue;
    if (args.kind === "thought") {
      // Accumulated narration for the agent's own turn. Tool-middleware
      // (judge) reasoning carries a callId and stays internal.
      if (!includeThoughts || args.callId) continue;
      const owner = args.agent && args.agent !== "orchestrator" ? args.agent : "";
      if (owner !== agentId) continue;
      const text = (args.detail || args.label || "").trim();
      if (!text) continue;
      if (blockText) blockText += "\n\n";
      blockText += text;
      blockTs ??= args.startedTs;
      continue;
    }
    if (taskIdOfCall(args.callId) !== agentId) continue;
    // A tool call closes the open thought block: its paragraphs are emitted
    // before the tool, so the next thoughts start a fresh message.
    if (includeThoughts) flushThoughts();
    const key = args.callId ?? `${agentId}-x${counter++}`;

    // UMLS expansion reads as two conversational bubbles instead of a card:
    // an "alright, searching" line and then a found/not-found result with a
    // table. The running step keeps its search bubble; the completed step
    // appends the result.
    if (args.kind === "umls") {
      if (toolIndex.get(key) === undefined) {
        items.push({
          kind: "umls",
          key: `${key}-find`,
          phase: "searching",
          query: umlsQueryOf(args),
          found: false,
          rows: [],
          ts: args.startedTs,
        });
        toolIndex.set(key, items.length - 1);
      }
      if (args.done && !umlsResultsDone.has(key)) {
        umlsResultsDone.add(key);
        const parsed = parseUmlsResult(args.rawResult);
        const found = parsed?.found === true;
        const rows: UmlsRow[] = found
          ? [
              {
                term: parsed?.term?.trim() || umlsQueryOf(args),
                cui: parsed?.cui,
                concept: parsed?.preferred_name,
                synonyms: parsed?.synonyms ?? [],
                semanticTypes: parsed?.semantic_types ?? [],
              },
            ]
          : [];
        items.push({
          kind: "umls",
          key: `${key}-result`,
          phase: "result",
          query: umlsQueryOf(args),
          found,
          rows,
          ts: args.finishedTs ?? args.startedTs,
        });
      }
      continue;
    }

    // Document retrieval: the agent asks the local database; the tool answers
    // with article attachments and the agent then judges their relevance (the
    // graying out is handled in the UI). One item, updated in place as the
    // tool call goes running → done.
    if (args.kind === "retrieve" || args.kind === "web_search") {
      const parsed = args.done ? parseRetrieval(args.rawResult) : null;
      const existing = toolIndex.get(key);
      const current =
        existing !== undefined && items[existing]!.kind === "retrieval"
          ? (items[existing] as SubagentRetrievalItem)
          : null;
      // Prefer the progress payload's passages (fuller metadata/text) and fall
      // back to the judged result when progress never arrived.
      const passages =
        current && current.passages.length > 0 ? current.passages : (parsed?.passages ?? []);
      const retrieval: SubagentRetrievalItem = {
        kind: "retrieval",
        key,
        source: args.kind === "web_search" ? "web" : "local",
        query: current?.query || retrieveQueryOf(args),
        done: args.done === true,
        passages,
        relevantIds: parsed?.relevantIds ?? current?.relevantIds ?? [],
        judged: parsed?.judged ?? current?.judged ?? false,
        tokens: parsed?.tokens ?? current?.tokens,
        reasons: parsed?.reasons ?? current?.reasons,
      };
      if (existing !== undefined) {
        items[existing] = retrieval;
        continue;
      }
      toolIndex.set(key, items.length);
      items.push(retrieval);
      continue;
    }

    const tool: SubagentToolItem = {
      kind: "tool",
      key,
      stepKind: args.kind,
      label: args.label || args.kind,
      detail: (args.detail || "").replace(/\s+/g, " ").trim(),
      done: args.done === true,
      error: args.error,
      rawArgs: (args.rawArgs as Record<string, unknown> | undefined) ?? undefined,
    };
    const existing = toolIndex.get(key);
    if (existing !== undefined) {
      items[existing] = tool;
      continue;
    }
    toolIndex.set(key, items.length);
    items.push(tool);
  }
  if (includeThoughts) flushThoughts();
  return coalesceUmls(items);
}

/**
 * Fold an agent's UMLS calls into the two bubbles the UI wants: one "alright,
 * searching" line and one result card whose table holds every concept found.
 * Both are emitted at the first UMLS position so the pair reads together even
 * when other tools stream in between the parallel lookups.
 */
function coalesceUmls(items: SubagentItem[]): SubagentItem[] {
  const umls = items.filter((item): item is SubagentUmlsItem => item.kind === "umls");
  if (umls.length === 0) return items;

  const searching = umls.filter((item) => item.phase === "searching");
  const results = umls.filter((item) => item.phase === "result");

  const pair: SubagentItem[] = [];
  if (searching.length > 0) {
    pair.push({ ...searching[0]!, found: false, rows: [] });
  }
  if (results.length > 0) {
    pair.push({
      kind: "umls",
      key: results[0]!.key,
      phase: "result",
      query: results.map((item) => item.query).filter(Boolean).join(", "),
      found: results.some((item) => item.found),
      rows: results.flatMap((item) => item.rows),
      ts: results[0]!.ts,
    });
  }

  const out: SubagentItem[] = [];
  let placed = false;
  for (const item of items) {
    if (item.kind !== "umls") {
      out.push(item);
      continue;
    }
    if (!placed) {
      out.push(...pair);
      placed = true;
    }
  }
  return out;
}

function displayName(id: string, question: string | undefined): string {
  if (question && question.trim()) return question.trim();
  if (/^S\d+$/i.test(id)) return "Planner";
  return id;
}

/** Grok-style relative/clock label for a rail card. */
function timeLabelOf(ts: string | undefined): string {
  if (!ts) return "";
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return "";
  const now = new Date();
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false });
  }
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (date.toDateString() === yesterday.toDateString()) return "Yesterday";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** Build the ordered list of started legs (plan order first). */
/** Owning agent of a part ("" = master, null = not owned by any agent). */
function ownerOfPart(part: unknown): string | null {
  const name = partName(part);
  if (name === "thinking") {
    const thought = partArgs(part) as { agent?: string } | undefined;
    return thought?.agent && thought.agent !== "orchestrator" ? thought.agent : "";
  }
  if (name === "step") {
    const args = partArgs(part) as StepArgs | undefined;
    if (!args) return null;
    if (args.kind === "thought") {
      return args.agent && args.agent !== "orchestrator" ? args.agent : "";
    }
    return taskIdOfCall(args.callId);
  }
  if (name === "tool_progress") {
    const progress = partArgs(part) as { callId?: string } | undefined;
    return progress?.callId ? taskIdOfCall(progress.callId) : "";
  }
  return null;
}

export function buildSubagents(content: readonly unknown[], isRunning: boolean): Subagent[] {
  const plan = planItemsOf(content);

  let taskStates: ReturnType<typeof taskStatesOf>;
  try {
    taskStates = taskStatesOf(content as unknown as Content);
  } catch {
    taskStates = new Map();
  }

  const ids = new Set<string>([...plan.keys(), ...taskStates.keys()]);
  // One pass to bucket every part under its owning agent (or the master), so
  // buildAgentItems scans only that agent's parts instead of all of content.
  const slices = new Map<string, unknown[]>();
  for (const part of content) {
    const owner = ownerOfPart(part);
    if (owner === null) continue;
    const bucket = slices.get(owner);
    if (bucket) bucket.push(part);
    else slices.set(owner, [part]);
    if (owner) ids.add(owner);
  }

  const spawns = masterSpawns(content);

  const agents: Subagent[] = [];

  for (const id of ids) {
    const item = plan.get(id);
    const task = taskStates.get(id);
    const title = displayName(id, item?.text ?? task?.question);
    const items = buildAgentItems(slices.get(id) ?? [], id, true);

    const hasContent = items.length > 0;
    const done =
      item?.done === true ||
      task?.state === "done" ||
      task?.state === "failed" ||
      // A leg outside the plan whose terminal event never arrived: once the
      // run has ended its card must settle rather than spin forever.
      (!isRunning && !item && (task !== undefined || hasContent));
    const started =
      done || item?.started === true || task !== undefined || hasContent;
    if (!started) continue;

    const finishedTs = task?.finishedTs;
    const startedTs = task?.startedTs;
    // Recover the leg's depth from the master's spawn call: by task id for
    // planned research, else by question text (the planner spawn carries no
    // task id).
    const spawn =
      spawns.find((s) => s.taskId && s.taskId === id) ??
      spawns.find((s) => s.task && s.task === title);
    agents.push({
      id,
      title,
      shortName: "",
      model: item?.model ?? task?.model,
      depth: spawn?.depth ?? "deep",
      color: "#8b93a7",
      avatarSeed: 1,
      startedTs,
      finishedTs,
      timeLabel: timeLabelOf(finishedTs ?? startedTs),
      started,
      done,
      live: isRunning && started && !done,
      items,
    });
  }

  // Chronological spawn order: "Agent N" and the colour slot then stay put as
  // later legs append (a leg that starts later only ever lands after the ones
  // already running).
  agents.sort((a, b) => {
    const ta = a.startedTs ? Date.parse(a.startedTs) : Number.NaN;
    const tb = b.startedTs ? Date.parse(b.startedTs) : Number.NaN;
    if (Number.isNaN(ta) || Number.isNaN(tb)) return 0;
    return ta - tb;
  });
  let researchIndex = 0;
  agents.forEach((agent, index) => {
    const visual = agentVisual(agent.id, index + 1);
    agent.color = visual.color;
    agent.avatarSeed = visual.avatarSeed;
    if (agent.depth === "plan") {
      agent.shortName = "Agent Planner";
    } else if (agent.id === "synthesizer") {
      // The synthesizer is a pipeline leg, not a spawned research sub-agent:
      // give it a name instead of numbering it as "Agent N".
      agent.shortName = "Agent Synth";
    } else {
      researchIndex += 1;
      agent.shortName = `Agent ${researchIndex}`;
    }
  });

  // The orchestrator is surfaced as the "Master" agent so the window shows
  // the whole cast: the master plus every spawned leg. It is always part of
  // the roster while a run is in flight (or once any leg exists), even before
  // it has produced any renderable step.
  const masterItems = buildAgentItems(slices.get("") ?? [], "", true);
  if (isRunning || agents.length > 0 || masterItems.length > 0) {
    const masterVisual = agentVisual("master", 0);
    const master: Subagent = {
      id: "master",
      title: "Master",
      shortName: "Master",
      color: masterVisual.color,
      avatarSeed: masterVisual.avatarSeed,
      timeLabel: "",
      started: true,
      done: !isRunning,
      live: isRunning,
      items: masterItems,
    };
    return [master, ...agents];
  }

  return agents;
}

/**
 * Sub-agents for the full-height agent sheet, which lives outside the message
 * component: read the active thread's latest assistant turn instead of the
 * current message context.
 */
export function useThreadSubagents(): Subagent[] {
  const messages = useAuiState((s) => s.thread.messages);
  const isRunning = useAuiState((s) => s.thread.isRunning);
  return useMemo(() => {
    let assistant: (typeof messages)[number] | undefined;
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i]!.role === "assistant") {
        assistant = messages[i];
        break;
      }
    }
    if (!assistant || !Array.isArray(assistant.content)) return [];
    return buildSubagents(assistant.content as readonly unknown[], isRunning);
  }, [messages, isRunning]);
}
