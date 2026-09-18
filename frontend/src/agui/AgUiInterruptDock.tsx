/**
 * Human-in-the-loop dock for the AG-UI surface.
 *
 * The pipeline keeps its existing broker: \`ask_user\` parks the run and the AG-UI
 * stream stays open. The transport surfaces the clarification as a \`question\`
 * CUSTOM event; this dock posts each answered question to the existing
 * \`/v1/chats/{threadId}/questions/{questionId}/answer\` endpoint, which releases
 * the parked run and lets the same stream continue.
 *
 * "Answered" is derived from the thread itself (the completed step for the
 * ask_user call) as well as this session, so a persisted question block never
 * reappears after reload; a late 404 (the run already moved on) is treated as
 * resolved rather than retried forever.
 */

import { useMemo, useState } from "react";
import { useAuiState } from "@assistant-ui/react";
import { useAgUiUiStore } from "./aguiStore";

interface AskQuestion {
 id?: string;
 text?: string;
 options?: string[];
 multi_select?: boolean;
 allow_other?: boolean;
 allow_find_all?: boolean;
 context?: string;
}

interface QuestionBlock {
 callId: string;
 threadId: string;
 questions: AskQuestion[];
}

interface Answer {
 selections: string[];
 other: string;
 find_all: boolean;
}

const EMPTY_ANSWER: Answer = { selections: [], other: "", find_all: false };

function isAnswered(a: Answer): boolean {
 return a.selections.length > 0 || a.other.trim().length > 0 || a.find_all;
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

const EMPTY_BLOCKS: QuestionBlock[] = [];
const questionCache = new WeakMap<readonly unknown[], QuestionBlock[]>();

/**
 * Every unanswered question data part in the thread, in stream order. Cached
 * by the messages array so a stream with no question returns the same array
 * and never re-renders the dock on a thinking/status token.
 */
function questionBlocksFor(messages: readonly unknown[]): QuestionBlock[] {
 const hit = questionCache.get(messages);
 if (hit) return hit;
 const blocks: QuestionBlock[] = [];
 const resolved = new Set<string>();
 for (const message of messages as { role?: string; content?: unknown }[]) {
 if (message.role !== "assistant" || !Array.isArray(message.content)) continue;
 for (const part of message.content as { type?: string; name?: string; data?: Record<string, any> }[]) {
 if (part.type !== "data") continue;
 const name = part.name;
 const data = part.data;
 // The ask_user tool completed step is the durable "answered" marker.
 if (name === "step" && data?.callId && data.done === true) {
 resolved.add(String(data.callId));
 continue;
 }
 if (name !== "question") continue;
 if (!Array.isArray(data?.questions) || data.questions.length === 0) continue;
 const callId = String(data.callId ?? "");
 if (callId && resolved.has(callId)) continue;
 blocks.push({
 callId,
 threadId: String(data.threadId ?? ""),
 questions: data.questions as AskQuestion[],
 });
 }
 }
 const result = blocks.length === 0 ? EMPTY_BLOCKS : blocks;
 questionCache.set(messages, result);
 return result;
}

function useQuestionBlocks(): QuestionBlock[] {
 return useAuiState((s) => questionBlocksFor(s.thread.messages as readonly unknown[]));
}

export function usePendingQuestions(): QuestionBlock[] {
 const blocks = useQuestionBlocks();
 const answered = useAgUiUiStore((s) => s.answeredQuestions);
 return useMemo(
 () => blocks.filter((block) => block.callId && !answered[block.callId]),
 [blocks, answered],
 );
}

export function AgUiInterruptDock() {
 const pending = usePendingQuestions();
 const markAnswered = useAgUiUiStore((s) => s.markQuestionAnswered);
 if (!pending.length) return null;
 return (
 <div className="mb-3 flex flex-col gap-3">
 {pending.map((block) => (
 <QuestionForm key={block.callId} block={block} onAnswered={() => markAnswered(block.callId)} />
 ))}
 </div>
 );
}

function QuestionForm({ block, onAnswered }: { block: QuestionBlock; onAnswered: () => void }) {
 const questions = useMemo(
 () =>
 block.questions.map((q, i) => ({
 ...q,
 id: q.id || (block.callId ? block.callId + "-" + (i + 1) : "q" + (i + 1)),
 options: Array.isArray(q.options) ? q.options : [],
 })),
 [block],
 );
 const [answers, setAnswers] = useState<Record<string, Answer>>({});
 const [busy, setBusy] = useState(false);
 const [error, setError] = useState("");

 const patch = (id: string, next: Partial<Answer>) =>
 setAnswers((prev) => ({ ...prev, [id]: { ...(prev[id] ?? EMPTY_ANSWER), ...next } }));

 const answered = questions.filter((q) => isAnswered(answers[q.id as string] ?? EMPTY_ANSWER));
 const canSubmit = answered.length > 0;

 /** POST one answer; retry once so a tool that hasn't parked yet still lands. */
 const postAnswer = async (questionId: string, body: Answer): Promise<Response | null> => {
 if (!block.threadId) return null;
 const url = "/v1/chats/" + encodeURIComponent(block.threadId) + "/questions/" + encodeURIComponent(questionId) + "/answer";
 const init: RequestInit = {
 method: "POST",
 headers: { "Content-Type": "application/json" },
 body: JSON.stringify(body),
 };
 const first = await fetch(url, init);
 if (first.status !== 404) return first;
 await sleep(400);
 return fetch(url, init);
 };

 const submit = async () => {
 if (busy || !canSubmit) return;
 setBusy(true);
 setError("");
 try {
 const todo = questions.filter((q) => isAnswered(answers[q.id as string] ?? EMPTY_ANSWER));
 for (const q of todo) {
 const res = await postAnswer(q.id as string, answers[q.id as string] ?? EMPTY_ANSWER);
 if (res === null) {
 setError("This question is no longer attached to a chat.");
 return;
 }
 // 404 = the run already moved on (or was answered elsewhere): resolved.
 if (res.status === 404) continue;
 if (!res.ok) {
 setError("Answer was not accepted (HTTP " + res.status + "). Try again.");
 return;
 }
 }
 onAnswered();
 } catch {
 setError("Could not reach the server. Please try again.");
 } finally {
 setBusy(false);
 }
 };

 return (
 <div className="w-full rounded-2xl border border-brand/30 bg-card p-4 ">
 <div className="flex items-center gap-2 text-[13px] font-medium text-foreground">
 <span className="size-2 rounded-full bg-brand" aria-hidden />
 Help me understand what you want
 </div>

 {questions.map((q) => {
 const id = q.id as string;
 const a = answers[id] ?? EMPTY_ANSWER;
 const multi = q.multi_select !== false;
 const noOptions = (q.options ?? []).length === 0;
 const radioName = "agui-" + block.callId + "-" + id;
 const toggle = (option: string) => {
 if (!multi) {
 patch(id, { selections: a.selections.includes(option) ? [] : [option] });
 return;
 }
 patch(id, {
 selections: a.selections.includes(option)
 ? a.selections.filter((o) => o !== option)
 : [...a.selections, option],
 });
 };
 return (
 <div key={id} className="mt-3">
 <p className="text-[14px] font-medium leading-snug text-foreground">{q.text ?? id}</p>
 <div className="mt-2 grid gap-1.5">
 {noOptions ? (
 <input
 type="text"
 value={a.other}
 onChange={(e) => patch(id, { other: e.target.value })}
 placeholder="Type your answer…"
 className="w-full rounded-lg border border-border bg-background px-3 py-2 text-[13.5px] outline-none focus:border-brand/60"
 />
 ) : (
 (q.options ?? []).map((option) => (
 <label key={option} className="flex cursor-pointer items-center gap-2.5 rounded-lg border border-border/80 bg-background px-3 py-2 text-[13.5px]">
 <input
 type={multi ? "checkbox" : "radio"}
 name={radioName}
 checked={a.selections.includes(option)}
 onChange={() => toggle(option)}
 className="accent-brand"
 />
 {option}
 </label>
 ))
 )}
 {q.allow_find_all !== false ? (
 <label className="flex cursor-pointer items-center gap-2.5 rounded-lg border border-dashed border-brand/40 bg-brand/[0.04] px-3 py-2 text-[13.5px]">
 <input
 type="checkbox"
 name={radioName}
 checked={a.find_all}
 onChange={() => patch(id, { find_all: !a.find_all })}
 className="accent-brand"
 />
 Find all you can find
 </label>
 ) : null}
 {q.allow_other !== false && !noOptions ? (
 <input
 type="text"
 value={a.other}
 onChange={(e) => patch(id, { other: e.target.value })}
 placeholder="Something else…"
 className="w-full rounded-lg border border-border bg-background px-3 py-2 text-[13.5px] outline-none focus:border-brand/60"
 />
 ) : null}
 </div>
 </div>
 );
 })}

 <div className="mt-3 flex items-center gap-3">
 <button
 type="button"
 disabled={!canSubmit || busy}
 onClick={submit}
 className="rounded-lg bg-brand px-3.5 py-2 text-[13px] font-semibold text-brand-foreground transition hover:bg-brand-strong disabled:cursor-not-allowed disabled:opacity-40"
 >
 {busy ? "Sending…" : "Send my answer"}
 </button>
 {error ? <p role="alert" className="text-[12.5px] text-danger">{error}</p> : null}
 </div>
 </div>
 );
}
