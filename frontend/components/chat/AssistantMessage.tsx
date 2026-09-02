// ── Assistant response: pipeline, markdown readout, sources, actions ──
"use client";

import { useState } from "react";
import { Copy, PanelRight, RotateCcw, ThumbsDown, ThumbsUp } from "lucide-react";
import type { Message } from "@/lib/types";
import { copyText, fmtNum, formatDuration, formatClock } from "@/lib/utils";
import { CapsLabel, Led } from "@/components/ui/primitives";
import { Markdown } from "@/components/chat/Markdown";
import { ThinkingLayer } from "@/components/chat/ThinkingLayer";

export function AssistantMessage({
  message,
  onCite,
  onOpenSources,
  onRegenerate,
  onRetry,
  onRate,
}: {
  message: Message;
  onCite: (n: number) => void;
  onOpenSources: () => void;
  onRegenerate: () => void;
  onRetry: () => void;
  onRate: (rating: "up" | "down" | null) => void;
}) {
  const [copied, setCopied] = useState(false);
  const working = message.status === "queued" || message.status === "streaming";
  const hasContent = message.content.trim().length > 0;
  const duration =
    message.finishedAt && message.startedAt ? formatDuration(message.finishedAt - message.startedAt) : null;

  const doCopy = async () => {
    const ok = await copyText(message.content);
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    }
  };

  if (message.status === "error") {
    return (
      <section aria-label="Research error" className="mb-10">
        {/* keep the thinking log visible so the failure is traceable */}
        <ThinkingLayer message={message} />
        <div className="panel border-err/60">
          <div className="flex items-center justify-between border-b border-line px-4 py-2">
            <CapsLabel>Research interrupted</CapsLabel>
            <Led state="err" pulse />
          </div>
          <div className="px-4 py-4">
            <p className="text-[14px] font-semibold">The evidence service didn&apos;t respond.</p>
            <p className="mono mt-2 text-[11px] leading-relaxed text-ink3">{message.error ?? "Unknown error"}</p>
            <div className="mt-4 flex items-center gap-2">
              <button type="button" onClick={onRetry} className="btn btn--accent">
                <RotateCcw size={13} /> Retry
              </button>
              <button type="button" onClick={doCopy} disabled={!hasContent} className="btn">
                Copy partial
              </button>
            </div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section aria-label="MedPat response" className="mb-10">
      {/* label row */}
      <div className="mb-2 flex items-center gap-2">
        <CapsLabel>MedPat response</CapsLabel>
        {message.startedAt && <span className="mono text-[10px] text-ink3 tnum">{formatClock(message.startedAt)}</span>}
        {working && <Led state="accent" pulse className="ml-1" />}
        {duration && <span className="mono ml-auto text-[10px] text-ink3 tnum">{duration}</span>}
      </div>

      {/* thinking layer: collapsible live log of the whole research run */}
      {(working || hasContent || (message.trace && message.trace.length > 0)) && (
        <ThinkingLayer message={message} />
      )}

      {(hasContent || !working) && (
        <div className="panel corner-ticks px-4 py-3 sm:px-5 sm:py-4">
          {message.status === "stopped" && (
            <p className="mono mb-3 border border-warn/40 bg-warn/10 px-2 py-1.5 text-[10px] uppercase tracking-[0.14em] text-warn">
              Generation stopped — partial output
            </p>
          )}
          {message.content ? (
            <Markdown text={message.content} onCite={onCite} />
          ) : (
            <p className="mono text-[12px] text-ink3">… silence from the engine.</p>
          )}
          {working && <span className="cursor-blink mt-2" aria-hidden />}
        </div>
      )}

      {/* sources strip — progressive while streaming */}
      {message.sources && message.sources.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={onOpenSources}
            className="mono mr-1 inline-flex items-center gap-1.5 border border-line-strong px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-ink2 hover:border-accent hover:text-accent"
          >
            <PanelRight size={11} /> Sources · {String(message.sources.length).padStart(2, "0")}
          </button>
          {message.sources.map((s, i) => (
            <button
              key={s.id}
              type="button"
              onClick={() => onCite(i + 1)}
              disabled={working}
              title={s.title}
              className="mono border border-line px-1.5 py-1 text-[10px] text-ink3 tnum transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
            >
              {String(i + 1).padStart(2, "0")}
            </button>
          ))}
        </div>
      )}

      {/* research-memory strip — session resume + what was persisted.
          Advisory context only, never evidence; labeled as such. */}
      {message.memory &&
        (message.memory.sessionId || message.memory.committed) && (
          <div
            className="mt-3 flex flex-wrap items-center gap-1.5"
            title={`Persistent research memory — advisory context only, never evidence. Session: ${message.memory.sessionTitle || message.memory.sessionId || "—"}`}
          >
            <span className="mono mr-0.5 inline-flex items-center gap-1.5 border border-line-strong px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-ink2">
              <Led
                state={message.memory.committed ? "ok" : "accent"}
                pulse={!message.memory.committed}
                className="!h-1.5 !w-1.5"
              />
              Memory
            </span>
            {message.memory.sessionId && (
              <span className="mono border border-line px-1.5 py-1 text-[10px] text-ink3 tnum">
                {message.memory.sessionId.slice(0, 12)}
                {message.memory.sessionId.length > 12 ? "…" : ""}
              </span>
            )}
            {(message.memory.priorClaims > 0 ||
              message.memory.priorContradictions > 0 ||
              message.memory.priorGaps > 0) && (
              <span className="mono border border-line px-1.5 py-1 text-[10px] text-ink3 tnum">
                {message.memory.priorClaims} prior claim{message.memory.priorClaims === 1 ? "" : "s"} ·{" "}
                {message.memory.priorContradictions} contrad. · {message.memory.priorGaps} gap
                {message.memory.priorGaps === 1 ? "" : "s"}
              </span>
            )}
            {message.memory.committed && (
              <span className="mono border border-line px-1.5 py-1 text-[10px] text-ink3 tnum">
                recorded {message.memory.committed.claimsCommitted} claim{message.memory.committed.claimsCommitted === 1 ? "" : "s"} ·{" "}
                {message.memory.committed.contradictions} contrad. ·{" "}
                {message.memory.committed.gaps} gap{message.memory.committed.gaps === 1 ? "" : "s"}
              </span>
            )}
          </div>
        )}

      {/* footer actions */}
      {(message.status === "complete" || message.status === "stopped") && hasContent && (
        <div className="mt-3 flex items-center gap-1 border-t border-line pt-2.5">
          <button
            type="button"
            onClick={doCopy}
            className="mono inline-flex items-center gap-1.5 px-1.5 py-1 text-[10px] uppercase tracking-[0.12em] text-ink3 hover:text-ink"
            aria-label="Copy response"
          >
            <Copy size={12} /> {copied ? "Copied" : "Copy"}
          </button>
          <button
            type="button"
            onClick={() => onRate(message.rating === "up" ? null : "up")}
            aria-label="Good response"
            aria-pressed={message.rating === "up"}
            className={`icon-btn !h-7 !w-7 ${message.rating === "up" ? "!text-accent !border-accent" : ""}`}
          >
            <ThumbsUp size={13} />
          </button>
          <button
            type="button"
            onClick={() => onRate(message.rating === "down" ? null : "down")}
            aria-label="Poor response"
            aria-pressed={message.rating === "down"}
            className={`icon-btn !h-7 !w-7 ${message.rating === "down" ? "!text-accent !border-accent" : ""}`}
          >
            <ThumbsDown size={13} />
          </button>
          <button
            type="button"
            onClick={onRegenerate}
            className="mono inline-flex items-center gap-1.5 px-1.5 py-1 text-[10px] uppercase tracking-[0.12em] text-ink3 hover:text-ink"
            aria-label="Regenerate response"
          >
            <RotateCcw size={12} /> Regenerate
          </button>
          <span className="mono ml-auto text-[10px] text-ink3 tnum">{fmtNum(message.content.length)} chars</span>
        </div>
      )}
    </section>
  );
}
