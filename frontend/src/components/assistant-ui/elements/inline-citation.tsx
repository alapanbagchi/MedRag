"use client";

import { useAuiState } from "@assistant-ui/react";
import { PreviewCard } from "@base-ui/react/preview-card";
import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/types";
import { floating, mono } from "@/lib/surfaces";
import { domainOf, safeHref } from "@/lib/url";
import { partArgs, partName } from "@/lib/parts";

function pmcUrl(pmcid: string): string {
  return "https://pmc.ncbi.nlm.nih.gov/articles/" + encodeURIComponent(pmcid) + "/";
}

interface CitationProps {
  index: number;
  source: Source;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function Citation({ index, source, open, onOpenChange }: CitationProps) {
  const domain = domainOf(source.url) || (source.pmcid ? "pmc.ncbi.nlm.nih.gov" : "");
  const title = source.title || source.pmcid || source.id || "Source";
  const href = safeHref(source.url) ?? (source.pmcid ? pmcUrl(source.pmcid) : undefined);
  const initial = (domain[0] ?? title[0] ?? "S").toUpperCase();
  return (
    <PreviewCard.Root open={open} onOpenChange={onOpenChange}>
      <PreviewCard.Trigger
        delay={0}
        render={<button type="button" />}
        className={cn(
          "mx-0.5 inline-flex h-4 min-w-4 translate-y-[-2px] cursor-default items-center justify-center rounded-[5px] px-1 align-middle font-mono text-[10px] font-medium tabular-nums transition-colors",
          open
            ? "bg-foreground text-background"
            : "bg-foreground/[0.06] text-foreground/45 hover:text-foreground/90",
        )}
      >
        {index + 1}
      </PreviewCard.Trigger>
      <PreviewCard.Portal>
        <PreviewCard.Positioner side="top" sideOffset={8}>
          <PreviewCard.Popup
            className={cn(
              floating,
              "z-50 w-64 origin-(--transform-origin) rounded-2xl p-3.5 outline-none",
              "transition-[opacity,scale] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] motion-reduce:transition-none",
              "data-[starting-style]:scale-[0.97] data-[starting-style]:opacity-0",
              "data-[ending-style]:scale-[0.97] data-[ending-style]:opacity-0",
            )}
          >
            <div className="flex items-center gap-1.5">
              <span className="bg-foreground/[0.06] text-foreground/45 flex size-4 items-center justify-center rounded text-[9px] font-medium">
                {initial}
              </span>
              <span className={cn(mono, "text-foreground/40")}>
                {domain || title}
              </span>
            </div>
            {href ? (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="mt-2 block text-[13px] leading-snug font-medium hover:underline"
              >
                {title}
              </a>
            ) : (
              <p className="mt-2 text-[13px] leading-snug font-medium">
                {title}
              </p>
            )}
            {source.snippet ? (
              <p className="text-foreground/50 mt-1 text-[13px] leading-relaxed">
                {source.snippet}
              </p>
            ) : null}
          </PreviewCard.Popup>
        </PreviewCard.Positioner>
      </PreviewCard.Portal>
    </PreviewCard.Root>
  );
}

// One shared source list per content reference (all markers in a render reuse
// it instead of re-walking the whole message each).
const sourceCache = new WeakMap<object, Source[]>();

/** All sources in this message's Sources card, in order. */
function messageSources(content: unknown): Source[] {
  if (!Array.isArray(content)) return [];
  const cached = sourceCache.get(content);
  if (cached) return cached;
  const out: Source[] = [];
  for (const part of content) {
    if (partName(part) !== "sources") continue;
    const data = partArgs(part);
    const list = Array.isArray(data)
      ? data
      : (data as { sources?: unknown } | undefined)?.sources;
    if (!Array.isArray(list)) continue;
    for (const s of list) {
      if (s && typeof s === "object") out.push(s as Source);
    }
  }
  sourceCache.set(content, out);
  return out;
}

/**
 * [n] / [pid] marker in answer text -> citation card. Numeric markers index the
 * Sources card (1-based); anything else matches a verified passage id,
 * case-insensitively. Unresolvable markers render as the original text.
 */
export function InlineCiteRef({ raw, id }: { raw: string; id: string }) {
  const content = useAuiState((s) => s.message.content);
  const { source, index } = useMemo(() => {
    const sources = messageSources(content);
    if (/^[0-9]+$/.test(id)) {
      const i = Number(id) - 1;
      return { source: sources[i], index: i };
    }
    const needle = id.toLowerCase();
    const i = sources.findIndex(
      (s) =>
        Array.isArray(s.passage_ids) &&
        s.passage_ids.some((p) => typeof p === "string" && p.toLowerCase() === needle),
    );
    return { source: i >= 0 ? sources[i] : undefined, index: i };
  }, [content, id]);
  const [open, setOpen] = useState(false);
  if (!source || index < 0) return <>{raw}</>;
  return <Citation index={index} source={source} open={open} onOpenChange={setOpen} />;
}
