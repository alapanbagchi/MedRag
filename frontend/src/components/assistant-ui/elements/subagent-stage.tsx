"use client";

/**
 * Grok-style sub-agent window.
 *
 * The orchestrator's own thoughts render in the main stream (see
 * `AssistantMessage`). The moment the first leg is spawned this window opens
 * from zero height: the agent roster slides into the left rail and the selected
 * agent's thoughts arrive as chunked message bubbles on the right.
 *
 * Tool rendering per agent is intentionally out of scope here — this surface
 * only owns the agent roster + their thought bubbles.
 */

import {
  memo,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import { CheckIcon, FileTextIcon, GlobeIcon, Loader2Icon, XIcon } from "lucide-react";
import {
  useThreadSubagents,
  type RetrievalPassage,
  type Subagent,
  type SubagentItem,
  type SubagentRetrievalItem,
  type SubagentToolItem,
  type SubagentUmlsItem,
  type UmlsRow,
} from "../../../lib/subagents";
import { useAgUiUiStore } from "../../../agui/aguiStore";
import { stepMeta } from "../../../lib/toolkit";
import { AgentFace } from "./agent-face";
import { domainOf } from "@/lib/url";
import { Message } from "./message";

/** Clock label ("3:07 AM") for a stream timestamp. */
function clockOf(ts?: string): string {
  if (!ts) return "";
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

/** Animated blob face for one agent (chatters while it is working). */
function AgentAvatar({ agent, size = 52 }: { agent: Subagent; size?: number }) {
  return (
    <AgentFace
      color={agent.color}
      seed={agent.avatarSeed}
      mood={agent.live ? "talking" : "idle"}
      size={size}
      // Settled agents read as finished: desaturated and faded.
      className={agent.done ? "shrink-0 opacity-50 grayscale" : "shrink-0"}
    />
  );
}

function AgentStatus({ agent, className }: { agent: Subagent; className?: string }) {
  if (agent.done) {
    return <CheckIcon className={className ?? "size-3 shrink-0 text-emerald-500"} />;
  }
  if (agent.live) {
    return (
      <Loader2Icon
        className={`${className ?? "size-3"} shrink-0 animate-spin motion-reduce:animate-none`}
        style={{ color: agent.color }}
      />
    );
  }
  return null;
}

/** Last thing the agent said — the one-line preview under its name. */
function previewOf(agent: Subagent): string {
  if (agent.depth === "plan") return agent.done ? "Plan delivered" : "Planning…";
  const last = agent.items.at(-1);
  if (!last) return agent.live ? "Working…" : "No steps recorded.";
  if (last.kind === "response" || last.kind === "say") return last.text;
  if (last.kind === "tool") {
    const raw = last.label || last.stepKind;
    // Backend step labels are raw snake_case tool names; prefer the friendly
    // step label (e.g. "spawn_subagent" → "Delegation").
    return /\s/.test(raw) ? raw : stepMeta(last.stepKind).label;
  }
  if (last.kind === "umls") {
    if (last.phase === "searching") return "Finding medical terms…";
    return last.found ? "Found medical terms" : "No medical terms found";
  }
  const docs = new Set(last.passages.map((p) => p.url || p.documentId || p.id)).size;
  const noun = last.source === "web" ? "source" : "document";
  if (!last.done) {
    return docs > 0 ? `Found ${docs} ${noun}${docs === 1 ? "" : "s"}` : (last.source === "web" ? "Searching the web…" : "Searching the corpus…");
  }
  return last.judged ? "Verified sources" : "Found sources";
}

/** One row in the left-hand agent roster — Grok sidebar card. */
function AgentRailItem({
  agent,
  index,
  selected,
  onSelect,
}: {
  agent: Subagent;
  index: number;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const unread = agent.live && !selected;
  return (
    <button
      type="button"
      onClick={() => onSelect(agent.id)}
      aria-current={selected ? "true" : undefined}
      title={agent.title}
      className={`anim-agent-in relative flex w-[220px] shrink-0 items-center gap-2.5 rounded-xl px-2.5 py-2 text-left outline-none transition-[color,background-color,opacity] duration-150 sm:w-full ${
        selected ? "bg-foreground/[0.08]" : "hover:bg-foreground/[0.045]"
      } ${agent.done ? "opacity-60" : ""}`}
      style={{ animationDelay: `${index * 70}ms` }}
    >
      <span className="relative shrink-0">
        <AgentAvatar agent={agent} size={36} />
        {unread ? (
          <span className="absolute -top-0.5 -right-0.5 size-3 rounded-full border-2 border-popover bg-[#ff453a]" />
        ) : null}
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-baseline gap-2">
          <span className="min-w-0 flex-1 truncate text-[15px] leading-5 font-medium text-foreground">
            {agent.shortName}
          </span>
          {agent.timeLabel ? (
            <span className="shrink-0 text-[13px] leading-5 text-muted-foreground">
              {agent.timeLabel}
            </span>
          ) : null}
        </span>
        <span className="mt-1 block truncate text-[13px] leading-5 text-muted-foreground">
          {previewOf(agent)}
        </span>
      </span>
    </button>
  );
}

/**
 * One chunked thought as a gray chat bubble (avatar on the left). New chunks
 * mount with the slide-in animation; the trailing live chunk grows in place
 * behind a caret.
 */
/** One agent-side bubble (gray, with the agent avatar and a timestamp). */
function AgentSay({
  agent,
  text,
  live,
  time,
  delayMs = 0,
  label,
}: {
  agent: Subagent;
  text: string;
  live?: boolean;
  time?: string;
  delayMs?: number;
  /** Optional caption above the text (e.g. "Response"). */
  label?: string;
}) {
  return (
    <Message
      side="in"
      avatar={<AgentAvatar agent={agent} size={36} />}
      time={time}
      delayMs={delayMs}
      className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]"
    >
      {label ? (
        <span className="mb-1 block text-[10.5px] font-semibold tracking-[0.08em] text-muted-foreground/70 uppercase">
          {label}
        </span>
      ) : null}
      {text}
      {live ? (
        <span
          aria-hidden
          className="ml-1 inline-block h-[15px] w-[2px] translate-y-[2px] animate-pulse rounded-full bg-chat-ai-ink align-baseline"
        />
      ) : null}
    </Message>
  );
}

/**
 * Tool args arrive as a (often truncated) JSON blob. Pull the values out so the
 * card reads like prose instead of raw JSON.
 */
function cleanDetail(raw: string): string {
  const text = raw.trim();
  if (!text.startsWith("{")) return text;
  const values: string[] = [];
  for (const match of text.matchAll(/:\s*"([^"]*)(?:"|$)/g)) {
    const value = match[1]?.trim();
    if (value) values.push(value);
  }
  if (values.length > 0) return values.join(" · ");
  return text.replace(/^\{|\}$/g, "").trim();
}

/** The concept table inside a UMLS result card (tool side). */
function UmlsTable({ rows }: { rows: UmlsRow[] }) {
  const synonyms = rows[0]?.synonyms ?? [];
  return (
    <div className="mt-2 overflow-hidden rounded-lg border border-chat-tool-ink/15">
      <table className="w-full border-collapse text-[13px] leading-5">
        <thead>
          <tr className="text-left text-chat-tool-ink/55">
            <th className="border-b border-chat-tool-ink/15 px-2.5 py-1 font-medium">Term</th>
            <th className="border-b border-chat-tool-ink/15 px-2.5 py-1 font-medium">CUI</th>
            <th className="border-b border-chat-tool-ink/15 px-2.5 py-1 font-medium">Concept</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-chat-tool-ink/10">
          {rows.map((row, index) => (
            <tr key={`${row.cui ?? row.term}-${index}`} className="align-top">
              <td className="px-2.5 py-1.5 break-words">{row.term}</td>
              <td className="px-2.5 py-1.5 font-mono text-[12px] whitespace-nowrap">
                {row.cui || "—"}
              </td>
              <td className="px-2.5 py-1.5 break-words">{row.concept || "—"}</td>
            </tr>
          ))}
        </tbody>
        {synonyms.length > 0 ? (
          <tfoot>
            <tr>
              <td
                colSpan={3}
                className="border-t border-chat-tool-ink/10 px-2.5 py-1.5 text-[12px] text-chat-tool-ink/60"
              >
                Synonyms: {synonyms.slice(0, 6).join(", ")}
                {synonyms.length > 6 ? ` +${synonyms.length - 6}` : ""}
              </td>
            </tr>
          </tfoot>
        ) : null}
      </table>
    </div>
  );
}

/**
 * UMLS enrichment as a short conversation: the agent asks for the medical
 * terms tied to the question, the tool acknowledges, a “…” bubble holds
 * the wait, and the concept table lands in a final bubble.
 */
const AgentUmlsFlow = memo(function AgentUmlsFlow({
  agent,
  query,
  done,
  found,
  rows,
  startedTs,
  finishedTs,
  delayMs,
}: {
  agent: Subagent;
  query: string;
  done: boolean;
  found: boolean;
  rows: UmlsRow[];
  startedTs?: string;
  finishedTs?: string;
  delayMs: number;
}) {
  const term = query.trim() || "this";
  const settled = done || !agent.live;
  return (
    <>
      <AgentSay
        agent={agent}
        text={`Can you find medical terms and concepts related to ${term}`}
        time={clockOf(startedTs)}
        delayMs={delayMs}
      />
      <ToolSay text="Alright I’m on it" delayMs={delayMs + 90} />
      {!settled ? (
        <TypingBubble side="out" delayMs={delayMs + 180} />
      ) : found && rows.length > 0 ? (
        <ToolSay
          text="Here’s what I found"
          time={clockOf(finishedTs)}
          delayMs={delayMs + 180}
        >
          <UmlsTable rows={rows} />
        </ToolSay>
      ) : (
        <ToolSay
          text="I couldn’t find any medical terms or concepts for that."
          time={clockOf(finishedTs)}
          delayMs={delayMs + 180}
        />
      )}
    </>
  );
});

/**
 * One tool call as a white card, aligned to the right (the conversation's
 * "user" side). The running step is replaced in place by its completed
 * version, so it never renders twice.
 */
const AgentToolCard = memo(function AgentToolCard({
  item,
  delayMs,
}: {
  item: SubagentToolItem;
  delayMs: number;
}) {
  const meta = stepMeta(item.stepKind);
  const Icon = meta.icon;
  const detail = cleanDetail(item.detail);
  return (
    <div className="anim-agent-in flex justify-end" style={{ animationDelay: `${delayMs}ms` }}>
      <div className="min-w-0 max-w-[74%] rounded-2xl bg-chat-tool px-4 py-3 text-[14px] leading-[21px] text-chat-tool-ink">
        <div className="mb-1 flex items-center gap-1.5 opacity-65">
          <Icon className="size-3.5 shrink-0" />
          <span className="truncate text-[12px] font-medium tracking-tight">{meta.label}</span>
          {item.done ? (
            <CheckIcon className="size-3 shrink-0 text-emerald-600" />
          ) : (
            <Loader2Icon className="size-3 shrink-0 animate-spin motion-reduce:animate-none" />
          )}
        </div>
        {item.error ? (
          <p className="whitespace-pre-wrap break-words [overflow-wrap:anywhere] text-red-600">
            {item.error}
          </p>
        ) : detail ? (
          <p className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]">{detail}</p>
        ) : null}
      </div>
    </div>
  );
});

/**
 * iOS-style “…” typing bubble: three dots bouncing in sequence while a tool
 * is working. Carries the agent's avatar for agent-side waits.
 */
function TypingBubble({
  agent,
  side = "out",
  delayMs = 0,
}: {
  agent?: Subagent;
  side?: "in" | "out";
  delayMs?: number;
}) {
  return (
    <Message
      side={side}
      avatar={agent ? <AgentAvatar agent={agent} size={36} /> : undefined}
      delayMs={delayMs}
    >
      <span className="flex items-center gap-1.5 opacity-60">
        {[0, 1, 2].map((i) => (
          <span key={i} className="ios-typing-dot" style={{ animationDelay: `${i * 0.16}s` }} />
        ))}
      </span>
    </Message>
  );
}

/** Waiting state for a leg that has spawned but produced no output yet. */
function AgentPending({ agent }: { agent: Subagent }) {
  return <TypingBubble agent={agent} side="in" />;
}

/** One source document/page, grouping every retrieved passage that belongs to it. */
interface DocumentGroup {
  key: string;
  title: string;
  pmcid?: string;
  journal?: string;
  section?: string;
  url?: string;
  passages: RetrievalPassage[];
  score?: number;
  /** At least one passage was kept by the verifier. */
  relevant: boolean;
  /** Web source (globe icon + domain) vs local-corpus document. */
  web: boolean;
  /** Critic verdict + reason per grouped passage (same order as `passages`). */
  critique: Array<{ kept: boolean; reason: string }>;
}

/** Collapse the retrieved passages into one card per source document/page. */
function groupDocuments(item: SubagentRetrievalItem): DocumentGroup[] {
  const groups = new Map<string, DocumentGroup>();
  const relevant = new Set(item.relevantIds);
  const web = item.source === "web";
  for (const passage of item.passages) {
    const key = passage.url || passage.documentId || passage.id;
    let group = groups.get(key);
    if (!group) {
      group = {
        key,
        title:
          passage.title?.trim() ||
          (web ? domainOf(passage.url) : passage.documentId) ||
          "Untitled source",
        pmcid: passage.documentId,
        journal: passage.journal,
        section: passage.section?.trim() || undefined,
        url: passage.url,
        passages: [],
        relevant: false,
        web,
        critique: [],
      };
      groups.set(key, group);
    }
    group.passages.push(passage);
    group.critique.push({
      kept: relevant.has(passage.id),
      reason: item.reasons?.[passage.id] ?? "",
    });
    if (
      typeof passage.score === "number" &&
      (group.score === undefined || passage.score > group.score)
    ) {
      group.score = passage.score;
    }
    if (relevant.has(passage.id)) group.relevant = true;
  }
  return [...groups.values()];
}

function DocChip({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-md bg-chat-tool-ink/[0.06] px-1.5 py-0.5 font-mono text-[10px] leading-none text-chat-tool-ink/60">
      {children}
    </span>
  );
}

/**
 * One source document as a high-fidelity card: paper title, PMC id, section,
 * best score and passage count. Plain white; rejected cards dim once the judge
 * has ruled.
 */
function DocumentCard({
  doc,
  verified,
  index,
}: {
  doc: DocumentGroup;
  verified: boolean;
  index: number;
}) {
  const rejected = verified && !doc.relevant;
  const domain = domainOf(doc.url);
  const Icon = doc.web ? GlobeIcon : FileTextIcon;
  return (
    <div
      title={(doc.passages[0]?.text ?? "").slice(0, 300)}
      style={{ animationDelay: `${index * 55}ms` }}
      className={`anim-agent-in group relative flex flex-col overflow-hidden rounded-2xl border p-3 transition-all duration-500 ease-out ${
        rejected
          ? "border-chat-tool-ink/10 bg-chat-tool/70 opacity-45"
          : "border-chat-tool-ink/12 bg-chat-tool shadow-[0_1px_2px_rgba(13,14,26,0.05),0_12px_32px_-20px_rgba(13,14,26,0.55)] hover:-translate-y-0.5 hover:border-chat-tool-ink/20 hover:shadow-[0_2px_6px_rgba(13,14,26,0.08),0_20px_44px_-22px_rgba(13,14,26,0.65)]"
      }`}
    >
      <span
        aria-hidden
        className="absolute inset-y-0 left-0 w-[3px] bg-chat-tool-ink/[0.08]"
      />
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-xl bg-chat-tool-ink/[0.06] text-chat-tool-ink/50">
          <Icon className="size-[18px]" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="line-clamp-2 text-[13.5px] leading-[1.35] font-semibold tracking-[-0.01em] text-chat-tool-ink">
            {doc.title}
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-1.5 gap-y-1">
            {doc.web ? (
              <>
                {domain ? <DocChip>{domain}</DocChip> : null}
                {typeof doc.score === "number" ? <DocChip>{doc.score.toFixed(2)}</DocChip> : null}
              </>
            ) : (
              <>
                {doc.pmcid ? <DocChip>{doc.pmcid}</DocChip> : null}
                {doc.section ? <DocChip>{doc.section}</DocChip> : null}
                {typeof doc.score === "number" ? <DocChip>{doc.score.toFixed(2)}</DocChip> : null}
              </>
            )}
            <span className="text-[11px] text-chat-tool-ink/45">
              {doc.passages.length} passage{doc.passages.length === 1 ? "" : "s"}
            </span>
          </div>
        </div>
        <span className="mt-0.5 shrink-0">
          {!verified ? (
            <span className="flex size-5 items-center justify-center">
              <Loader2Icon className="size-3.5 animate-spin text-chat-tool-ink/40 motion-reduce:animate-none" />
            </span>
          ) : doc.relevant ? (
            <span className="flex size-5 items-center justify-center rounded-full bg-chat-tool-ink/[0.06] text-chat-tool-ink/60">
              <CheckIcon className="size-3.5" />
            </span>
          ) : (
            <span className="flex size-5 items-center justify-center rounded-full bg-chat-tool-ink/[0.06] text-chat-tool-ink/40">
              <XIcon className="size-3.5" />
            </span>
          )}
        </span>
      </div>
      {doc.passages.length > 0 ? (
        <div className="mt-2.5 rounded-lg border border-chat-tool-ink/[0.06] bg-chat-tool-ink/[0.045] px-2.5 py-2 shadow-[inset_0_1px_2px_rgba(13,14,26,0.07)]">
          <p className="text-[10.5px] font-medium tracking-[0.01em] text-chat-tool-ink/50">
            {doc.passages.length === 1 ? "Chunk" : "Chunks"} that might be relevant:
          </p>
          <div className="mt-1 space-y-1.5">
            {doc.passages.slice(0, 3).map((passage, i) => (
              <p
                key={i}
                className="line-clamp-4 whitespace-pre-wrap break-words text-[12px] leading-[1.5] text-chat-tool-ink/70"
              >
                {passage.text}
              </p>
            ))}
          </div>
          {doc.passages.length > 3 ? (
            <p className="mt-1 text-[11px] text-chat-tool-ink/45">
              +{doc.passages.length - 3} more chunk{doc.passages.length - 3 === 1 ? "" : "s"}
            </p>
          ) : null}
        </div>
      ) : null}
      {verified && doc.critique.length > 0 ? (
        <div className="mt-2 rounded-lg border border-chat-tool-ink/[0.06] bg-chat-tool-ink/[0.03] px-2.5 py-2 shadow-[inset_0_1px_2px_rgba(13,14,26,0.05)]">
          <p className="text-[10.5px] font-medium tracking-[0.01em] text-chat-tool-ink/50">
            Critic’s reason:
          </p>
          <div className="mt-1 space-y-1">
            {doc.critique.map((c, i) => (
              <p
                key={i}
                className="whitespace-pre-wrap break-words text-[11.5px] leading-[1.45] text-chat-tool-ink/70"
              >
                <span className="font-semibold text-chat-tool-ink/60">
                  {c.kept ? "Accepted" : "Rejected"}
                </span>
                <span className="text-chat-tool-ink/45"> — </span>
                {c.reason || "no reason provided"}
              </p>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Document retrieval in three beats: the agent asks the local database, the
 * tool answers with article attachments, then the agent takes a beat to judge
 * relevance and the irrelevant attachments gray out.
 */
/** One white tool-side message (right-aligned), optionally holding attachments. */
function ToolSay({
  text,
  live,
  time,
  delayMs = 0,
  children,
}: {
  text: string;
  live?: boolean;
  time?: string;
  delayMs?: number;
  children?: ReactNode;
}) {
  return (
    <Message
      side="out"
      time={time}
      delayMs={delayMs}
      className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]"
    >
      <p>
        {text}
        {live ? (
          <span
            aria-hidden
            className="ml-1 inline-block h-[15px] w-[2px] translate-y-[2px] animate-pulse rounded-full bg-chat-tool-ink align-baseline"
          />
        ) : null}
      </p>
      {children}
    </Message>
  );
}

/**
 * Local corpus search as a conversation: the agent asks, the tool
 * acknowledges, a “…” bubble holds the wait, the found documents land as
 * attachment cards while the judge runs, then the rejected ones gray out and
 * the tool reports how many were actually relevant.
 */
const AgentRetrievalFlow = memo(function AgentRetrievalFlow({
  agent,
  item,
  delayMs,
}: {
  agent: Subagent;
  item: SubagentRetrievalItem;
  delayMs: number;
}) {
  const query = item.query.trim() || "this";
  const web = item.source === "web";
  const searching = !item.done && item.passages.length === 0;
  const found = item.passages.length > 0;
  const verified = item.done;
  const docs = useMemo(
    () => groupDocuments(item),
    [item.key, item.passages.length, item.judged, item.done],
  );
  const relevantCount = item.judged ? docs.filter((doc) => doc.relevant).length : docs.length;
  const noun = web ? "source" : "document";

  return (
    <>
      <AgentSay
        agent={agent}
        text={
          web
            ? `Can you search the web for \u201c${query}\u201d?`
            : `Can you search for \u201c${query}\u201d locally?`
        }
        delayMs={delayMs}
      />
      <ToolSay text="Of course. On it" delayMs={delayMs + 90} />
      {searching ? <TypingBubble side="out" delayMs={delayMs + 180} /> : null}
      {found ? (
        <>
          <ToolSay
            text={
              web
                ? "I found some sources online. Let me check their credibility"
                : "I found some documents that might be relevant. Wait for me to verify them"
            }
            delayMs={delayMs + 180}
          />
          <div className="ios-pop ios-pop--out flex justify-end">
            <div className="flex w-full flex-col gap-2">
              {docs.map((doc, index) => (
                <DocumentCard
                  key={doc.key}
                  doc={doc}
                  verified={verified && item.judged}
                  index={index}
                />
              ))}
            </div>
          </div>
          {verified && typeof item.tokens === "number" ? (
            <div className="flex justify-end pr-1">
              <span className="text-[11.5px] leading-4 text-muted-foreground/70">
                {item.tokens.toLocaleString()} tokens to verify
              </span>
            </div>
          ) : null}
        </>
      ) : item.done ? (
        <ToolSay
          text={
            web
              ? "I couldn't find anything online for that."
              : "I couldn't find anything in the local corpus for that."
          }
          delayMs={delayMs + 180}
        />
      ) : null}
      {verified ? (
        <ToolSay
          text={
            relevantCount > 0
              ? `Alright, I found ${relevantCount} ${noun}${relevantCount === 1 ? "" : "s"} relevant to the question`
              : `Alright, none of the ${noun}s were relevant to the question`
          }
        />
      ) : null}
    </>
  );
});

/** The right pane: the selected agent's interleaved thought/tool transcript. */
/**
 * A delegation divider: the handoff between the master and a sub-agent, shown
 * whenever a `spawn_subagent` step appears in an agent's transcript. The copy
 * follows the delegation's depth (planner vs deep vs shallow) and settles into
 * a "received" state once the leg returns. Styled after the reference
 * "Messages from X and Y" divider.
 */
function DelegationDivider({
  item,
  agents,
  onSelect,
}: {
  item: SubagentToolItem;
  agents: readonly Subagent[];
  onSelect: (id: string) => void;
}) {
  const raw = (item.rawArgs ?? {}) as { task?: unknown; task_id?: unknown; depth?: unknown };
  const taskText = String(raw.task ?? "").trim();
  const taskId = String(raw.task_id ?? "").trim();
  const target =
    (taskId ? agents.find((candidate) => candidate.id === taskId) : undefined) ??
    agents.find((candidate) => {
      const title = candidate.title.trim();
      return (
        !!title &&
        !!taskText &&
        (title === taskText || taskText.startsWith(title) || title.startsWith(taskText))
      );
    }) ??
    null;
  const depth = String(raw.depth ?? "").toLowerCase();
  const isPlan = depth === "plan";
  const name =
    target?.shortName ??
    (isPlan ? "Planner" : depth === "shallow" ? "Quick answer" : "Deep research");
  const color = target?.color ?? "#8b93a7";

  // Planner handoffs name the agent inside the sentence (and show no chip);
  // research handoffs keep the target in the clickable chip that follows.
  const label = isPlan
    ? item.done
      ? "Plan received from the planning agent"
      : "Planning the task using the planning agent"
    : depth === "shallow"
      ? item.done
        ? "Answer received from"
        : "Asking for a quick answer"
      : item.done
        ? "Findings received from"
        : "Delegating deep research to";

  const chip = (
    <span className="flex min-w-0 items-center gap-1.5">
      <AgentFace color={color} seed={target?.avatarSeed ?? 1} size={18} mood="idle" />
      <span className="min-w-0 truncate font-medium" style={{ color }}>
        {name}
      </span>
    </span>
  );

  const body = (
    <span className="flex min-w-0 max-w-full items-center gap-2">
      <span className="min-w-0 truncate">{label}</span>
      {isPlan ? null : chip}
    </span>
  );

  return (
    <div className="anim-agent-in flex flex-wrap items-center justify-center gap-x-2 gap-y-1 px-2 py-3 text-center text-[13px] text-muted-foreground">
      {target ? (
        // The badge is the agent's chat entry point: click selects them.
        <button
          type="button"
          onClick={() => onSelect(target.id)}
          title={`Open ${target.title}`}
          className="flex min-w-0 max-w-full cursor-pointer items-center rounded-full border border-border/60 bg-card px-2.5 py-1 outline-none transition-colors hover:bg-foreground/[0.06] focus-visible:ring-1 focus-visible:ring-foreground/20"
        >
          {body}
        </button>
      ) : (
        <span className="flex min-w-0 max-w-full items-center rounded-full border border-border/60 bg-card px-2.5 py-1">
          {body}
        </span>
      )}
    </div>
  );
}

/** The mirror of a delegation, seen from the sub-agent's own side. */
function IncomingHandoff({ text }: { text: string }) {
  return (
    <div className="anim-agent-in flex items-center justify-center px-2 py-3">
      <span className="rounded-full border border-border/60 bg-card px-3 py-1 text-[13px] text-muted-foreground">
        {text}
      </span>
    </div>
  );
}

type ThreadEntry =
  | { type: "response"; key: string; text: string; ts?: string }
  | { type: "say"; key: string; text: string; ts?: string }
  | { type: "delegate"; key: string; item: SubagentToolItem }
  | { type: "retrieval"; key: string; item: SubagentRetrievalItem }
  | {
      type: "umls";
      key: string;
      query: string;
      done: boolean;
      found: boolean;
      rows: UmlsRow[];
      startedTs?: string;
      finishedTs?: string;
    };

/**
 * Collapse an agent's items into the conversation the pane renders: its own
 * thought paragraphs and closing response, delegated handoffs, retrieval flows
 * and the UMLS enrichment exchange (agent ask → tool ack → result table).
 */
function threadEntries(items: readonly SubagentItem[]): ThreadEntry[] {
  const out: ThreadEntry[] = [];
  let umlsSeen = false;
  for (const item of items) {
    if (item.kind === "response") {
      // The agent's closing reply — its own bubble, after the narration.
      out.push({ type: "response", key: item.key, text: item.text, ts: item.ts });
      continue;
    }
    if (item.kind === "say") {
      // The agent's narration, already split into paragraph bubbles.
      out.push({ type: "say", key: item.key, text: item.text, ts: item.ts });
      continue;
    }
    if (item.kind === "umls") {
      // All of an agent's UMLS calls are coalesced into one ask/result pair.
      if (umlsSeen) continue;
      umlsSeen = true;
      const searching = items.find(
        (candidate): candidate is SubagentUmlsItem =>
          candidate.kind === "umls" && candidate.phase === "searching",
      );
      const result = items.find(
        (candidate): candidate is SubagentUmlsItem =>
          candidate.kind === "umls" && candidate.phase === "result",
      );
      out.push({
        type: "umls",
        key: "umls",
        query: searching?.query || result?.query || "",
        done: !!result,
        found: result?.found ?? false,
        rows: result?.rows ?? [],
        startedTs: searching?.ts ?? result?.ts,
        finishedTs: result?.ts,
      });
      continue;
    }
    if (item.kind === "retrieval") {
      out.push({ type: "retrieval", key: item.key, item });
      continue;
    }
    if (item.kind === "tool" && item.stepKind === "delegate") {
      out.push({ type: "delegate", key: item.key, item });
    }
  }
  return out;
}

function AgentThread({
  agent,
  agents,
  onSelect,
  incoming,
}: {
  agent: Subagent;
  agents: readonly Subagent[];
  onSelect: (id: string) => void;
  incoming: string | null;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep the newest item in view while the agent streams.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [agent.id, agent.items]);

  const entries = useMemo(() => threadEntries(agent.items), [agent.items]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-3 overflow-x-hidden overflow-y-auto px-4 py-3.5">
        {agent.timeLabel ? (
          <div className="anim-agent-in pt-0.5 pb-1 text-center text-[12px] font-medium text-muted-foreground/70">
            {agent.timeLabel}
          </div>
        ) : null}
        {incoming ? <IncomingHandoff text={incoming} /> : null}
        {entries.length > 0 ? (
          entries.map((entry, index) =>
            entry.type === "response" ? (
              <AgentSay
                key={entry.key}
                agent={agent}
                label="Response"
                text={entry.text}
                time={clockOf(entry.ts)}
              />
            ) : entry.type === "say" ? (
              <AgentSay
                key={entry.key}
                agent={agent}
                text={entry.text}
                // Group a paragraph run under one stamp: only the first bubble
                // of consecutive thoughts keeps the time.
                time={entries[index - 1]?.type === "say" ? undefined : clockOf(entry.ts)}
                live={agent.live && index === entries.length - 1}
              />
            ) : entry.type === "delegate" ? (
              <DelegationDivider
                key={entry.key}
                item={entry.item}
                agents={agents}
                onSelect={onSelect}
              />
            ) : entry.type === "retrieval" ? (
              <AgentRetrievalFlow
                key={entry.key}
                agent={agent}
                item={entry.item}
                delayMs={index * 60}
              />
            ) : (
              <AgentUmlsFlow
                key={entry.key}
                agent={agent}
                query={entry.query}
                done={entry.done}
                found={entry.found}
                rows={entry.rows}
                startedTs={entry.startedTs}
                finishedTs={entry.finishedTs}
                delayMs={index * 60}
              />
            ),
          )
        ) : agent.live && !incoming ? (
          <AgentPending agent={agent} />
        ) : null}
      </div>
    </div>
  );
}

/** Agent roster rail sizing (desktop): wider by default, drag to resize. */
const RAIL_MIN_W = 220;
const RAIL_MAX_W = 460;
const RAIL_DEFAULT_W = 300;

/** The window chrome: left agent roster + right activity pane, full height. */
function SubagentWindow({
  agents,
  onClose,
}: {
  agents: readonly Subagent[];
  onClose?: () => void;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [railWidth, setRailWidth] = useState(RAIL_DEFAULT_W);
  const [resizing, setResizing] = useState(false);
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const seenRef = useRef<string[]>([]);

  // Follow a freshly spawned agent; otherwise keep the user's pick.
  useEffect(() => {
    const ids = agents.map((a) => a.id);
    const fresh = ids.filter((id) => !seenRef.current.includes(id));
    seenRef.current = ids;
    if (fresh.length > 0) setSelectedId(fresh[fresh.length - 1]!);
  }, [agents]);

  // Clear the drag cursor/selection if the sheet unmounts mid-drag.
  useEffect(
    () => () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    },
    [],
  );

  const startResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { startX: event.clientX, startWidth: railWidth };
    setResizing(true);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };

  const moveResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const next = drag.startWidth + (event.clientX - drag.startX);
    setRailWidth(Math.min(RAIL_MAX_W, Math.max(RAIL_MIN_W, next)));
  };

  const endResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    setResizing(false);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const nudgeResize = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const delta = event.key === "ArrowLeft" ? -16 : 16;
    setRailWidth((width) => Math.min(RAIL_MAX_W, Math.max(RAIL_MIN_W, width + delta)));
  };

  const selected = useMemo(
    () => agents.find((a) => a.id === selectedId) ?? agents.at(-1) ?? null,
    [agents, selectedId],
  );

  if (!selected) return null;

  // The sub-agent's side of the handoff: it received the task from the master
  // and, once settled, sends its plan or findings back.
  const incoming =
    selected.id === "master"
      ? null
      : selected.done
        ? selected.depth === "plan"
          ? "Plan sent to Master"
          : "Findings sent to Master"
        : "Task received from Master";

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {/* Title bar */}
      <div className="flex items-center gap-3 border-b border-border/60 px-5 py-4">
        <span
          className="min-w-0 flex-1 truncate text-[17px] leading-6 font-semibold text-foreground"
          title={selected.title}
        >
          {selected.shortName}
        </span>
        <span className="flex shrink-0 items-center justify-end gap-2">
          {selected.live ? (
            <AgentStatus agent={selected} />
          ) : (
            <span className="font-mono text-[12px] text-muted-foreground/60">
              {agents.filter((a) => a.done).length}/{agents.length} done
            </span>
          )}
          {onClose ? (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close agent flow"
              className="flex size-9 items-center justify-center rounded-full border border-border/70 bg-foreground/[0.04] text-foreground/80 outline-none transition-colors hover:bg-foreground/[0.1] hover:text-foreground focus-visible:ring-1 focus-visible:ring-foreground/20"
            >
              <XIcon className="size-5" />
            </button>
          ) : null}
        </span>
      </div>

      {/* Body: roster + activity. The roster is a resizable rail on desktop —
          drag the divider to widen it, or focus it and use the arrow keys.
          On small screens it stays a horizontal scroller. */}
      <div
        className="flex min-h-0 flex-1 flex-col sm:flex-row"
        style={{ "--agent-rail-w": `${railWidth}px` } as CSSProperties}
      >
        <div className="flex gap-1.5 overflow-x-auto border-b border-border/50 p-2 sm:w-[var(--agent-rail-w)] sm:shrink-0 sm:flex-col sm:overflow-y-auto sm:border-b-0">
          {agents.map((agent, index) => (
            <AgentRailItem
              key={agent.id}
              agent={agent}
              index={index}
              selected={agent.id === selected.id}
              onSelect={setSelectedId}
            />
          ))}
        </div>
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize agent list"
          aria-valuenow={railWidth}
          aria-valuemin={RAIL_MIN_W}
          aria-valuemax={RAIL_MAX_W}
          tabIndex={0}
          onPointerDown={startResize}
          onPointerMove={moveResize}
          onPointerUp={endResize}
          onPointerCancel={endResize}
          onKeyDown={nudgeResize}
          onDoubleClick={() => setRailWidth(RAIL_DEFAULT_W)}
          className={`group relative hidden w-px shrink-0 cursor-col-resize touch-none bg-border/60 outline-none transition-colors hover:bg-foreground/30 focus-visible:bg-foreground/40 sm:block ${
            resizing ? "bg-foreground/40" : ""
          }`}
        >
          {/* Fat invisible hit area around the 1px divider. */}
          <span aria-hidden className="absolute inset-y-0 -right-1.5 -left-1.5" />
        </div>
        <AgentThread
          agent={selected}
          agents={agents}
          onSelect={setSelectedId}
          incoming={incoming}
        />
      </div>
    </div>
  );
}

/**
 * Agent sheet: covers the whole viewport — over the sidebar and the composer —
 * sliding in from the right. The outer fixed layer clips the off-screen state
 * so the slide never adds a horizontal scrollbar.
 */
export function AgentPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const agents = useThreadSubagents();
  if (agents.length === 0) return null;
  return (
    <div
      data-slot="agent-panel"
      aria-hidden={!open}
      inert={!open}
      className={`fixed inset-0 z-50 overflow-hidden ${open ? "" : "pointer-events-none"}`}
    >
      <div
        className={`h-full w-full app-canvas transition-transform duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <SubagentWindow agents={agents} onClose={onClose} />
      </div>
    </div>
  );
}

/**
 * In-message "Thinking · Click to see the agentic flow" row. Rendered where
 * the typing dots used to sit; `fallback` shows until a run has agents.
 */
export function AgentFlowTrigger() {
  const agents = useThreadSubagents();
  const running = agents.some((agent) => agent.live);
  const flowOpen = useAgUiUiStore((s) => s.flowOpen);
  const openFlow = useAgUiUiStore((s) => s.openFlow);
  if (agents.length === 0) return null;
  const content = (
    <>
      <span
        className={
          "size-1.5 shrink-0 rounded-full " +
          (running ? "medrag-dot bg-[#4b8cf5]" : "bg-emerald-500")
        }
        aria-hidden
      />
      <span>{running ? "Thinking" : "Agent flow"}</span>
      <span aria-hidden>·</span>
      <span>
        {flowOpen
          ? agents.filter((agent) => agent.done).length + "/" + agents.length + " done"
          : "Click to see the agentic flow (" + agents.length + ")"}
      </span>
    </>
  );
  return (
    <div className="flex justify-start px-1 py-1.5">
      {flowOpen ? (
        <span className="flex items-center gap-1.5 text-[13px] text-muted-foreground">
          {content}
        </span>
      ) : (
        <button
          type="button"
          onClick={openFlow}
          className="pill-hover flex cursor-pointer items-center gap-1.5 rounded-full px-2.5 py-1 text-[13px] text-muted-foreground outline-none focus-visible:ring-1 focus-visible:ring-foreground/20"
        >
          {content}
        </button>
      )}
    </div>
  );
}


