/**
 * The synthesizer's answer, with two views of the one turn:
 *
 * - Visual: OpenUI's Renderer over the generated OpenUI Lang program.
 * - Text: the on-demand long-form answer. Opening it starts a second synth
 * agent that re-writes the turn's verified evidence as a detailed,
 * point-wise paper-style markdown answer (see longFormAnswer.ts). The
 * derived-from-program markdown shows immediately underneath while it
 * writes, so the tab is never blank.
 *
 * Visual is the default; nothing is generated until the Text view is opened.
 */

import { useDeferredValue, useMemo, useState } from "react";
import { createParser, Renderer } from "@openuidev/react-lang";
import { TextMessagePartProvider } from "@assistant-ui/react";
import { Loader2Icon } from "lucide-react";
import { MarkdownText } from "../components/MarkdownText";
import { cn } from "../lib/utils";
import { useAnswerChatId, useAnswerRunId, useAnswerSources } from "./answerSources";
import { library } from "./library";
import { useLongFormAnswer } from "./longFormAnswer";
import { MedRagSourcesContext } from "./source-context";
import { openuiToMarkdown, type MarkdownSource } from "./toMarkdown";

type AnswerTab = "visual" | "text";

const TABS: { key: AnswerTab; label: string }[] = [
 { key: "visual", label: "Gen UI" },
 { key: "text", label: "Text" },
];

function AnswerTabs({
 value,
 onChange,
}: {
 value: AnswerTab;
 onChange: (tab: AnswerTab) => void;
}) {
 return (
 <div
 role="tablist"
 aria-label="Answer view"
 className="anim-rise mb-2 inline-flex items-center gap-0.5 rounded-lg bg-foreground/[0.045] p-0.5"
 >
 {TABS.map((tab) => (
 <button
 key={tab.key}
 type="button"
 role="tab"
 aria-selected={value === tab.key}
 onClick={() => onChange(tab.key)}
 className={cn(
 "rounded-[6px] px-2.5 py-1 text-[12px] font-medium transition-colors",
 value === tab.key
 ? "bg-background text-foreground "
 : "text-foreground/45 hover:text-foreground/80",
 )}
 >
 {tab.label}
 </button>
 ))}
 </div>
 );
}

/** What the second synth agent is doing, so the wait is legible. */
function LongFormBanner({
 state,
 onRetry,
}: {
 state: { status: "loading" | "ready" | "error"; error?: string };
 onRetry: () => void;
}) {
 if (state.status === "ready") return null;
 if (state.status === "error") {
 return (
 <div className="mb-2 flex flex-wrap items-center gap-2 rounded-lg border border-border bg-muted/40 px-2.5 py-1.5 text-[11.5px] text-muted-foreground">
 <span>Could not generate the full text answer ({state.error}). Showing the summary.</span>
 <button
 type="button"
 onClick={onRetry}
 className="rounded-md px-1.5 py-0.5 font-medium text-foreground/70 underline underline-offset-2 hover:text-foreground"
 >
 Retry
 </button>
 </div>
 );
 }
 return (
 <div className="mb-2 flex items-center gap-2 rounded-lg border border-border bg-muted/40 px-2.5 py-1.5 text-[11.5px] text-muted-foreground">
 <Loader2Icon className="size-3.5 shrink-0 animate-spin motion-reduce:animate-none" />
 <span>Agent Synth is writing the detailed text answer - showing the summary until it lands.</span>
 </div>
 );
}

function TextAnswer({
 derived,
 chatId,
 runId,
}: {
 derived: string;
 chatId: string;
 runId: string;
}) {
 const { state, reload } = useLongFormAnswer(chatId, runId);
 const markdown = state.status === "ready" ? state.markdown : derived;
 return (
 <div className="flex flex-col">
 <LongFormBanner state={state} onRetry={reload} />
 {markdown ? (
 <div className="answer-body">
 <TextMessagePartProvider text={markdown}>
 <MarkdownText />
 </TextMessagePartProvider>
 </div>
 ) : (
 <p className="py-1 text-[15px] text-muted-foreground">
 No text version of this answer.
 </p>
 )}
 </div>
 );
}

/** The plain markdown answer, used as the Gen UI view when the backend did
 * not produce an OpenUI program (no sources, or a markdown fallback). */
function MarkdownBody({ text }: { text: string }) {
 return (
 <div className="answer-body">
 <TextMessagePartProvider text={text}>
 <MarkdownText />
 </TextMessagePartProvider>
 </div>
 );
}

export function OpenUIAnswer({
 text,
 isStreaming,
 format = "openui",
}: {
 text: string;
 isStreaming: boolean;
 format?: string;
}) {
 const sources = useAnswerSources();
 const chatId = useAnswerChatId();
 const runId = useAnswerRunId();
 const openui = format === "openui";
 const [tab, setTab] = useState<AnswerTab>(openui ? "visual" : "text");
 // The Renderer re-parses the growing program and reconciles the whole tree on
 // every prop change. Deferring the text lets React interrupt/coalesce that
 // work behind the token stream instead of blocking each token on it; the
 // final program still lands the moment the deferred render catches up.
 const deferredText = useDeferredValue(text);
 // The Text view shows the long-form answer once it lands, and this derived
 // view meanwhile: the OpenUI program rendered to markdown, or - when the
 // answer is already markdown - the answer itself.
 const derived = useMemo(() => {
 if (!openui) return text;
 const parser = createParser(library.toJSONSchema(), "Card");
 return openuiToMarkdown(parser.parse(text).root, (sources ?? []) as MarkdownSource[]);
 }, [openui, text, sources]);
 return (
 <MedRagSourcesContext.Provider value={sources}>
 <AnswerTabs value={tab} onChange={setTab} />
 {tab === "visual" ? (
 openui ? (
 <Renderer response={deferredText} library={library} isStreaming={isStreaming} />
 ) : (
 <MarkdownBody text={text} />
 )
 ) : (
 <TextAnswer derived={derived} chatId={chatId} runId={runId} />
 )}
 </MedRagSourcesContext.Provider>
 );
}
