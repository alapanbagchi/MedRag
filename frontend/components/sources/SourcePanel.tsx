// ── Sources: desktop drawer / mobile bottom sheet ───────────────────
"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, ExternalLink, X } from "lucide-react";
import type { Source } from "@/lib/types";
import { cn, formatScore } from "@/lib/utils";
import { CapsLabel } from "@/components/ui/primitives";

function SourceCard({
  source,
  index,
  active,
}: {
  source: Source;
  index: number;
  active: boolean;
}) {
  const [open, setOpen] = useState(false);
  const num = String(index + 1).padStart(2, "0");
  return (
    <li
      className={cn(
        "panel flex flex-col",
        active ? "border-accent" : "border-line"
      )}
    >
      <button
        type="button"
        className="flex w-full items-start gap-3 px-3 py-3 text-left"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`Toggle source ${num}`}
      >
        <span className={cn("mono mt-0.5 w-8 flex-none text-[13px] font-semibold tnum", active ? "text-accent" : "text-ink2")}>
          {num}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-[13px] font-semibold leading-snug">{source.title}</span>
          <span className="mono mt-1.5 block text-[10.5px] leading-relaxed text-ink3">
            {source.journal} · {source.year} · {source.pmcid ?? source.id}
          </span>
        </span>
        {open ? <ChevronUp size={14} className="mt-0.5 flex-none text-ink3" /> : <ChevronDown size={14} className="mt-0.5 flex-none text-ink3" />}
      </button>

      {/* relevance */}
      <div className="flex items-center gap-2 border-t border-line px-3 py-1.5">
        <span className="mono text-[9.5px] uppercase tracking-[0.12em] text-ink3">Relevance</span>
        <div aria-hidden className="h-[4px] flex-1 border border-line bg-ground2">
          <div
            className={cn("h-[3px]", active ? "bg-accent" : "bg-ink3")}
            style={{ width: `${Math.round((source.score ?? 0) * 100)}%` }}
          />
        </div>
        <span className={cn("mono text-[10px] tnum", active ? "text-accent" : "text-ink2")}>
          {formatScore(source.score)}
        </span>
      </div>

      {open && (
        <div className="anim-fade border-t border-line px-3 py-3">
          <p className="mono mb-2 text-[10px] uppercase tracking-[0.12em] text-ink3">
            {source.authors.slice(0, 3).join(", ")}{source.authors.length > 3 ? " et al." : ""}
          </p>
          {source.snippet && (
            <p className="text-[12.5px] leading-relaxed text-ink2">{source.snippet}</p>
          )}
          {source.pmid && (
            <p className="mono mt-2 text-[10px] text-ink3">PMID {source.pmid}</p>
          )}
          {source.url && (
            <a
              href={source.url}
              target="_blank"
              rel="noreferrer"
              className="mono mt-3 inline-flex items-center gap-1.5 border border-line-strong px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-ink2 hover:border-accent hover:text-accent"
            >
              Open in PMC <ExternalLink size={11} />
            </a>
          )}
        </div>
      )}
    </li>
  );
}

export function SourcePanel({
  open,
  onClose,
  sources,
  activeCitation,
  isDesktop,
  contextualLabel,
}: {
  open: boolean;
  onClose: () => void;
  sources: Source[];
  activeCitation: number | null;
  isDesktop: boolean;
  contextualLabel?: string;
}) {
  if (!open) return null;

  const header = (
    <div className="flex h-12 flex-none items-center justify-between border-b border-line px-3">
      <div className="flex flex-col">
        <CapsLabel className="flex items-center gap-2">
          Sources <span className="text-accent tnum">{String(sources.length).padStart(2, "0")}</span>
        </CapsLabel>
        {contextualLabel && (
          <span className="mono mt-0.5 text-[9px] uppercase tracking-[0.14em] text-ink3">{contextualLabel}</span>
        )}
      </div>
      <button type="button" onClick={onClose} aria-label="Close sources" className="icon-btn">
        <X size={15} />
      </button>
    </div>
  );

  const body = (
    <ul className="flex-1 space-y-2 overflow-y-auto p-3">
      {sources.map((s, i) => (
        <SourceCard key={s.id} source={s} index={i} active={activeCitation != null && activeCitation === i + 1} />
      ))}
    </ul>
  );

  if (isDesktop) {
    return (
      <div
        role="complementary"
        aria-label="Sources"
        className="anim-drawer fixed inset-y-0 right-0 z-40 flex w-[420px] max-w-[92vw] flex-col border-l border-line bg-ground shadow-2xl"
      >
        {header}
        {body}
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label="Sources">
      <button type="button" aria-label="Close sources" className="anim-fade absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="anim-sheet absolute inset-x-0 bottom-0 flex max-h-[64vh] flex-col border-t border-line bg-ground shadow-2xl">
        {header}
        {body}
      </div>
    </div>
  );
}
