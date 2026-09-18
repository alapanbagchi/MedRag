"use client";

import { PreviewCard } from "@base-ui/react/preview-card";
import { createContext, useContext, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import type { CitationStyleDef, Source } from "@/lib/types";
import { floating, mono } from "@/lib/surfaces";
import { domainOf, safeHref } from "@/lib/url";
import type { CitationStyleCatalog } from "@/lib/citations";

/**
 * The verified source list for the answer currently being rendered.
 *
 * Provided once per assistant message (AgUiAssistantMessage) so the markdown
 * answer, the OpenUI visual answer and the bottom reference list all resolve
 * inline markers against the same numbered list.
 */
export const CitationSourcesContext = createContext<Source[]>([]);

export function useCitationSources(): Source[] {
  return useContext(CitationSourcesContext);
}

const EMPTY_CATALOG: CitationStyleCatalog = { styles: [], defaultStyleId: "" };

/** The style catalog emitted by the backend (the dropdown's options). */
export const CitationStylesContext = createContext<CitationStyleCatalog>(EMPTY_CATALOG);

export function useCitationStyleCatalog(): CitationStyleCatalog {
  return useContext(CitationStylesContext);
}

/** Used until the backend's catalog arrives (or when it is absent). */
export const FALLBACK_CITATION_STYLE: CitationStyleDef = {
  id: "apa",
  label: "APA",
  authorYear: true,
  separator: "; ",
  prefix: "(",
  suffix: ")",
};

/**
 * The active citation style for the message. AgUiAssistantMessage resolves it
 * (saved choice, message default, or an on-demand catalog style) and provides
 * it here, so every badge and the reference list agree without extra reads.
 */
export const CitationActiveStyleContext = createContext<CitationStyleDef>(
  FALLBACK_CITATION_STYLE,
);

export function useActiveCitationStyle(): CitationStyleDef {
  return useContext(CitationActiveStyleContext);
}

function pmcUrl(pmcid: string): string {
  return "https://pmc.ncbi.nlm.nih.gov/articles/" + encodeURIComponent(pmcid) + "/";
}

/** Scroll a badge click down to its reference entry in the same message. */
export function focusReference(index: number, scope?: Element | null): void {
  const root: ParentNode = scope ?? document;
  const target = root.querySelector<HTMLElement>("#ref-" + (index + 1));
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.classList.add("cite-target-flash");
  window.setTimeout(() => target.classList.remove("cite-target-flash"), 1600);
}

// ---------------------------------------------------------------------------
// Hover: highlight the sentence the citation supports
// ---------------------------------------------------------------------------
// The blue span under a hovered citation marks just the claim that refers to
// it, never the whole paragraph. The CSS Custom Highlight API paints an
// arbitrary text range without touching React's DOM, so nothing re-renders.

const CITE_HIGHLIGHT = "medrag-cite";

interface TextSpan {
  node: Text;
  start: number;
  end: number;
}

function textSpans(root: Element): { spans: TextSpan[]; full: string } {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const spans: TextSpan[] = [];
  let full = "";
  let current = walker.nextNode();
  while (current) {
    const text = current as Text;
    spans.push({
      node: text,
      start: full.length,
      end: full.length + text.data.length,
    });
    full += text.data;
    current = walker.nextNode();
  }
  return { spans, full };
}

/** Character offset of an element's inline content within its block. */
function offsetOf(root: Element, target: Node): number | null {
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
  );
  let offset = 0;
  let current: Node | null = walker.nextNode();
  while (current) {
    if (current === target) return offset;
    if (current.nodeType === Node.TEXT_NODE) {
      offset += (current as Text).data.length;
    }
    current = walker.nextNode();
  }
  return null;
}

/** Sentence bounds around an offset, stopping at . ! ? or a newline. */
function sentenceBounds(full: string, offset: number): [number, number] {
  const ends = (index: number): boolean => {
    const char = full[index];
    if (char === "\n") return true;
    if (char !== "." && char !== "!" && char !== "?") return false;
    const next = full[index + 1];
    return next === undefined || /\s/.test(next);
  };
  let start = offset;
  while (start > 0 && !ends(start - 1)) start -= 1;
  let end = offset;
  while (end < full.length && !ends(end)) end += 1;
  if (end < full.length) end += 1;
  return [start, end];
}

function locate(
  spans: TextSpan[],
  offset: number,
): { node: Text; offset: number } {
  for (const span of spans) {
    if (offset >= span.start && offset <= span.end) {
      return { node: span.node, offset: offset - span.start };
    }
  }
  const last = spans[spans.length - 1]!;
  return { node: last.node, offset: last.node.data.length };
}

function highlightSentence(node: HTMLElement): void {
  const registry = (
    CSS as unknown as { highlights?: Map<string, unknown> }
  ).highlights;
  const HighlightCtor = (
    globalThis as unknown as { Highlight?: new (range: Range) => unknown }
  ).Highlight;
  if (!registry || !HighlightCtor) return;
  const block =
    node.closest(
      "p, li, td, th, blockquote, figcaption, dd, dt, h1, h2, h3, h4, h5, h6",
    ) ?? node.closest("span");
  if (!block || !block.contains(node)) return;
  const { spans, full } = textSpans(block);
  if (spans.length === 0) return;
  const offset = offsetOf(block, node);
  if (offset === null) return;
  const [start, end] = sentenceBounds(full, offset);
  const from = locate(spans, start);
  const to = locate(spans, end);
  const range = document.createRange();
  range.setStart(from.node, from.offset);
  range.setEnd(to.node, to.offset);
  registry.set(CITE_HIGHLIGHT, new HighlightCtor(range));
}

function clearSentenceHighlight(): void {
  const registry = (
    CSS as unknown as { highlights?: Map<string, unknown> }
  ).highlights;
  registry?.delete(CITE_HIGHLIGHT);
}

export interface ResolvedSource {
  source: Source;
  index: number;
}

/**
 * One inline citation cluster: a clickable badge carrying the style's label
 * (e.g. "(Zhang et al., 2025)" or "(1)").
 *
 * Hover or focus opens a preview listing the cited sources with their
 * bibliography entries and tints the sentence the citation supports; click
 * scrolls down to that source's entry in the reference list.
 */
export function InlineCitationBadge({
  label,
  sources,
}: {
  label: string;
  sources: ResolvedSource[];
}) {
  const [open, setOpen] = useState(false);
  const nodeRef = useRef<HTMLElement | null>(null);
  // Hovering highlights the sentence the citation supports; clicking scrolls
  // to the full reference. Driven off PreviewCard's open state so the
  // highlight survives moving onto the popup.
  useEffect(() => {
    if (!open) return;
    const node = nodeRef.current;
    if (node) highlightSentence(node);
    return () => clearSentenceHighlight();
  }, [open]);

  const style = useActiveCitationStyle();
  const primary = sources[0];
  const domain = primary
    ? domainOf(primary.source.url) || (primary.source.pmcid ? "pmc.ncbi.nlm.nih.gov" : "")
    : "";
  const title = primary
    ? primary.source.title || primary.source.pmcid || primary.source.id || "Source"
    : label;
  const initial = (domain[0] ?? title[0] ?? "S").toUpperCase();
  const aria =
    sources.length === 1
      ? "Citation: " + title
      : "Citation: " + sources.length + " sources";
  const className = cn(
    "cite-chip cite-chip-labeled motion-reduce:transition-none",
    open && "cite-chip-open",
  );

  return (
    <PreviewCard.Root open={open} onOpenChange={setOpen}>
      <PreviewCard.Trigger
        delay={0}
        onPointerEnter={(event) => {
          nodeRef.current = event.currentTarget as HTMLElement;
        }}
        onFocus={(event) => {
          nodeRef.current = event.currentTarget as HTMLElement;
        }}
        render={
          <button
            type="button"
            aria-label={aria}
            onClick={() =>
              primary &&
              focusReference(
                primary.index,
                nodeRef.current?.closest("[data-citation-scope]"),
              )
            }
          />
        }
        className={className}
      >
        {label}
      </PreviewCard.Trigger>
      <PreviewCard.Portal>
        <PreviewCard.Positioner side="top" sideOffset={8}>
          <PreviewCard.Popup
            className={cn(
              floating,
              "z-50 w-80 origin-(--transform-origin) rounded-2xl p-3.5 outline-none",
              "transition-[opacity,scale] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] motion-reduce:transition-none",
              "data-[starting-style]:scale-[0.97] data-[starting-style]:opacity-0",
              "data-[ending-style]:scale-[0.97] data-[ending-style]:opacity-0",
            )}
          >
            <div className="flex items-center gap-1.5">
              <span className="flex size-4 items-center justify-center rounded bg-foreground/[0.06] text-[9px] font-medium text-foreground/45">
                {sources.length === 1 ? initial : sources.length}
              </span>
              <span className={cn(mono, "truncate text-foreground/40")}>
                {sources.length === 1
                  ? domain || primary?.source.sourceName || title
                  : sources.length + " sources"}
              </span>
              <span className="ml-auto shrink-0 font-mono text-[10px] text-foreground/35">
                {style.label}
              </span>
            </div>
            <ul className="mt-2 flex max-h-72 flex-col gap-2.5 overflow-y-auto">
              {sources.map(({ source, index }) => {
                const sourceHref =
                  safeHref(source.url) ??
                  (source.pmcid ? pmcUrl(source.pmcid) : undefined);
                const sourceTitle =
                  source.title || source.pmcid || source.id || "Source";
                const entry =
                  source.citations?.[style.id]?.entry || source.citation || "";
                return (
                  <li key={index} className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="flex size-4 shrink-0 items-center justify-center rounded bg-foreground/[0.06] text-[9px] font-medium text-foreground/45">
                        {index + 1}
                      </span>
                      <span className={cn(mono, "truncate text-foreground/40")}>
                        {domainOf(source.url) || source.sourceName || "Source"}
                      </span>
                    </div>
                    {sourceHref ? (
                      <a
                        href={sourceHref}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-1 block text-[13px] leading-snug font-medium hover:underline"
                      >
                        {sourceTitle}
                      </a>
                    ) : (
                      <p className="mt-1 text-[13px] leading-snug font-medium">
                        {sourceTitle}
                      </p>
                    )}
                    {entry ? (
                      <p className="mt-1 text-[12px] leading-relaxed text-foreground/55">
                        {entry}
                      </p>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          </PreviewCard.Popup>
        </PreviewCard.Positioner>
      </PreviewCard.Portal>
    </PreviewCard.Root>
  );
}
