/**
 * Citation helpers shared by the markdown answer and the OpenUI visual answer.
 *
 * Pure functions only (no React, no assistant-ui) so the OpenUI component
 * library can import them without pulling the chat runtime into the prompt
 * generator.
 */

import type { CitationStyleDef, Source } from "./types";
import { partArgs, partName } from "./parts";

export const EMPTY_SOURCES: Source[] = [];

// The source list per content array, cached by array identity: a stream that
// only appends thinking/status parts keeps returning the same array, so the
// citation provider does not churn on every token.
const sourceCache = new WeakMap<object, Source[]>();

/**
 * All sources for this message, in citation order.
 *
 * The synthesizer's 'answer_sources' part is the verified, numbered list and
 * wins when present. Otherwise the per-leg 'sources' cards are flattened in
 * order - the fallback an older markdown answer used.
 */
export function messageSources(content: unknown): Source[] {
  if (!Array.isArray(content)) return EMPTY_SOURCES;
  const key = content as unknown as object;
  const cached = sourceCache.get(key);
  if (cached) return cached;
  const ledger: Source[] = [];
  const perLeg: Source[] = [];
  for (const part of content) {
    const name = partName(part);
    if (name === "answer_sources") {
      const list = (partArgs(part) as { sources?: unknown } | undefined)?.sources;
      if (Array.isArray(list)) {
        for (const source of list) {
          if (source && typeof source === "object") ledger.push(source as Source);
        }
      }
      continue;
    }
    if (name !== "sources") continue;
    const data = partArgs(part);
    const list = Array.isArray(data)
      ? data
      : (data as { sources?: unknown } | undefined)?.sources;
    if (!Array.isArray(list)) continue;
    for (const source of list) {
      if (source && typeof source === "object") perLeg.push(source as Source);
    }
  }
  const result = ledger.length > 0 ? ledger : perLeg;
  const stable = result.length === 0 ? EMPTY_SOURCES : result;
  sourceCache.set(key, stable);
  return stable;
}


export interface CitationStyleCatalog {
  styles: CitationStyleDef[];
  defaultStyleId: string;
}

export const EMPTY_CATALOG: CitationStyleCatalog = { styles: [], defaultStyleId: "" };
const catalogCache = new WeakMap<object, CitationStyleCatalog>();

/**
 * The citation-style catalog for this message, emitted alongside the source
 * list. The backend owns the labels and the inline prefix/suffix, so the
 * dropdown and the badges cannot drift from the CSL styles.
 */
export function citationStyles(content: unknown): CitationStyleCatalog {
  if (!Array.isArray(content)) return EMPTY_CATALOG;
  const key = content as unknown as object;
  const cached = catalogCache.get(key);
  if (cached) return cached;
  let result = EMPTY_CATALOG;
  for (let i = content.length - 1; i >= 0; i -= 1) {
    const part = content[i];
    if (partName(part) !== "citation_styles") continue;
    const data = partArgs(part) as
      | { styles?: unknown; default?: unknown }
      | undefined;
    const list = Array.isArray(data?.styles)
      ? (data.styles as CitationStyleDef[]).filter(
          (s) => s && typeof s === "object" && typeof s.id === "string",
        )
      : [];
    if (list.length) {
      result = {
        styles: list,
        defaultStyleId:
          typeof data?.default === "string" ? data.default : list[0]!.id,
      };
    }
    break;
  }
  catalogCache.set(key, result);
  return result;
}

/** The active style, falling back to the message default and then the first. */
export function pickCitationStyle(
  catalog: CitationStyleCatalog,
  selected: string,
): CitationStyleDef | undefined {
  return (
    catalog.styles.find((style) => style.id === selected) ??
    catalog.styles.find((style) => style.id === catalog.defaultStyleId) ??
    catalog.styles[0]
  );
}

/** The inline label core for a source in a style (no surrounding text). */
export function inlineCore(source: Source, styleId: string): string {
  const exact = source.citations?.[styleId]?.inline;
  if (exact) return exact;
  // The requested style is still rendering: fall back to an available one
  // (APA first) so the badge never flashes a raw title.
  if (source.citations) {
    for (const citation of Object.values(source.citations)) {
      if (citation?.inline) return citation.inline;
    }
  }
  return source.title || source.ref || source.sourceName || "source";
}



/** One entry from the backend's full CSL catalog. */
export interface CatalogStyle {
  id: string;
  title: string;
  short?: string;
  /** CSL citation-format: author-date | numeric | note | author | label. */
  format?: string;
  class?: string;
}

export interface CitationRenderEntry {
  ref: string;
  index: number;
  inline: string;
  entry: string;
}

/** The backend's on-demand rendering of one style for one turn. */
export interface CitationRenderResult {
  style: string;
  label: string;
  authorYear: boolean;
  separator: string;
  prefix: string;
  suffix: string;
  entries: CitationRenderEntry[];
}

/** A descriptor for a catalog style before its CSL has been rendered. */
export function catalogDescriptor(entry: CatalogStyle): CitationStyleDef {
  const authorYear = entry.format === "author-date" || entry.format === "author";
  return {
    id: entry.id,
    label: entry.title,
    authorYear,
    separator: authorYear ? "; " : ",",
    prefix: "(",
    suffix: ")",
  };
}

/** Merge a rendered style's labels/entries into the source list for that style. */
export function applyRender(
  sources: Source[],
  render: CitationRenderResult | undefined,
): Source[] {
  if (!render || !Array.isArray(render.entries) || render.entries.length === 0) {
    return sources;
  }
  const byRef = new Map<string, CitationRenderEntry>();
  const byIndex = new Map<number, CitationRenderEntry>();
  for (const entry of render.entries) {
    if (entry.ref) byRef.set(entry.ref, entry);
    byIndex.set(entry.index, entry);
  }
  return sources.map((source, position) => {
    const hit =
      (source.ref ? byRef.get(source.ref) : undefined) ??
      byIndex.get(source.index ?? position + 1);
    if (!hit) return source;
    return {
      ...source,
      citations: {
        ...(source.citations ?? {}),
        [render.style]: { inline: hit.inline, entry: hit.entry },
      },
    };
  });
}

/** The bibliography entry for a source in a style. */
export function entryFor(source: Source, styleId: string): string {
  return (
    source.citations?.[styleId]?.entry ||
    source.citation ||
    [source.title, source.url].filter(Boolean).join(" - ") ||
    source.id ||
    source.sourceName ||
    "Source"
  );
}

export interface CitationToken {
  kind: "text" | "cite";
  /** The text to render for a text token, or the raw marker for a cite. */
  value: string;
  /** The id to resolve for a cite token ('4', 'P1', ...). */
  id?: string;
}

// '[4]', '[p8]', reversed 'p[4]', and parenthesized ref groups '(P1, P10)'.
const CITATION_SPLIT =
  /(\[[A-Za-z]*\d[\w-]*\]|\b[Pp]\[\d{1,4}\]|\([Pp]\d[\w-]*(?:\s*[,;]\s*[Pp]\d[\w-]*)*\))/g;
const BRACKETED = /^\[([A-Za-z]*\d[\w-]*)\]$/;
const REVERSED = /^[Pp]\[(\d{1,4})\]$/;
const PAREN_GROUP = /^\((.*)\)$/;
const PAREN_REFS = /^[Pp]\d[\w-]*(?:\s*[,;]\s*[Pp]\d[\w-]*)*$/;
const SINGLE_REF = /^[Pp]\d[\w-]*$/;

/**
 * Split answer text into plain text and citation markers. Every marker is
 * rendered as a badge; nothing is left as a literal '[n]'.
 */
export function tokenizeCitations(text: string): CitationToken[] {
  const out: CitationToken[] = [];
  for (const part of text.split(CITATION_SPLIT)) {
    if (!part) continue;
    const match = BRACKETED.exec(part);
    if (match) {
      out.push({ kind: "cite", value: part, id: match[1] ?? "" });
      continue;
    }
    const reversed = REVERSED.exec(part);
    if (reversed) {
      out.push({ kind: "cite", value: part, id: "p" + reversed[1] });
      continue;
    }
    const paren = PAREN_GROUP.exec(part);
    if (paren && PAREN_REFS.test(paren[1] ?? "")) {
      // Emit only the refs: the badge adds the style's own wrapper, so the
      // model's parentheses are not rendered twice.
      for (const token of (paren[1] ?? "").split(/(\s*[,;]\s*)/)) {
        if (SINGLE_REF.test(token)) out.push({ kind: "cite", value: token, id: token });
      }
      continue;
    }
    out.push({ kind: "text", value: part });
  }
  return out;
}

export interface ResolvedCitation {
  source: Source;
  index: number;
}

/**
 * Resolve a citation id against the source list. A numeric id is a 1-based
 * index; anything else matches a verified passage ref or id, case-insensitively.
 */
export function resolveCitation(
  sources: Source[],
  id: string,
): ResolvedCitation | undefined {
  if (/^[0-9]+$/.test(id)) {
    const index = Number(id) - 1;
    const source = sources[index];
    return source ? { source, index } : undefined;
  }
  const needle = id.toLowerCase();
  const index = sources.findIndex((source) =>
    Array.isArray(source.passage_ids) &&
    source.passage_ids.some((p) => typeof p === "string" && p.toLowerCase() === needle),
  );
  return index >= 0 ? { source: sources[index]!, index } : undefined;
}

/** A short label for a source, used when no formatted citation is present. */
export function sourceLabel(source: Source): string {
  return (
    source.citation ||
    [source.title, source.url].filter(Boolean).join(" - ") ||
    source.id ||
    source.sourceName ||
    "Source"
  );
}
