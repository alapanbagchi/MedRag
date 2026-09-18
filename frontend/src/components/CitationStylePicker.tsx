"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  BookMarkedIcon,
  CheckIcon,
  ChevronDownIcon,
  SearchIcon,
} from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { useCitationStyle } from "@/lib/citationStyle";
import { useCslCatalog } from "@/lib/cslCatalog";
import type { CatalogStyle } from "@/lib/citations";

/** Cap the rendered result rows; the rest are reached by typing more. */
const MAX_RESULTS = 80;

/** The style every answer defaults to when the reader has not chosen one. */
const DEFAULT_STYLE_ID = "apa";

/** The styles the backend pins for every answer, with short, friendly labels. */
const COMMON_STYLES: CatalogStyle[] = [
  { id: "apa", title: "APA", format: "author-date" },
  { id: "harvard", title: "Harvard", format: "author-date" },
  { id: "chicago", title: "Chicago", format: "author-date" },
  { id: "vancouver", title: "Vancouver", format: "numeric" },
  { id: "ama", title: "AMA", format: "numeric" },
];

function Row({
  label,
  hint,
  selected,
  onClick,
}: {
  label: string;
  hint?: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center justify-between gap-2 rounded-md px-2.5 py-1.5 text-left text-[13px] transition-colors",
        selected
          ? "bg-foreground/[0.06] text-foreground"
          : "text-foreground/75 hover:bg-foreground/[0.04] hover:text-foreground",
      )}
    >
      <span className="truncate">{label}</span>
      <span className="flex shrink-0 items-center gap-1.5">
        {hint ? (
          <span className="text-[10px] tracking-wide text-foreground/30 uppercase">
            {hint}
          </span>
        ) : null}
        {selected ? <CheckIcon className="size-3.5 text-primary" /> : null}
      </span>
    </button>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="py-1">
      <div className="px-2.5 pt-1 pb-1 text-[10px] font-semibold tracking-wider text-foreground/35 uppercase">
        {title}
      </div>
      {children}
    </div>
  );
}

/**
 * The composer's citation-style picker.
 *
 * Lives in the input bar (not per answer): the chosen style is an app-wide
 * preference, so one control restyles every answer's inline badges and
 * reference list. The pinned styles are always offered; the full CSL catalog
 * (thousands of styles) is fetched once and searched by title, and choosing
 * one outside the pinned set asks the backend to render it on demand.
 */
export function CitationStylePicker({ className }: { className?: string }) {
  const selectedId = useCitationStyle((state) => state.styleId);
  const setStyleId = useCitationStyle((state) => state.setStyleId);
  const catalog = useCslCatalog((state) => state.styles);
  const loaded = useCslCatalog((state) => state.loaded);
  const error = useCslCatalog((state) => state.error);
  const load = useCslCatalog((state) => state.load);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (open) void load();
  }, [open, load]);
  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);

  const resolvedId = selectedId || DEFAULT_STYLE_ID;
  const current = useMemo(
    () =>
      COMMON_STYLES.find((entry) => entry.id === resolvedId) ??
      catalog.find((entry) => entry.id === resolvedId),
    [catalog, resolvedId],
  );

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return [] as CatalogStyle[];
    const out: CatalogStyle[] = [];
    for (const entry of catalog) {
      if (out.length >= MAX_RESULTS) break;
      if (
        entry.id.toLowerCase().includes(needle) ||
        entry.title.toLowerCase().includes(needle) ||
        (entry.short ?? "").toLowerCase().includes(needle)
      ) {
        out.push(entry);
      }
    }
    return out;
  }, [catalog, query]);

  const choose = (id: string) => {
    setStyleId(id);
    setOpen(false);
  };
  const hasQuery = query.trim().length > 0;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label="Citation style"
          className={cn(
            "flex h-8 max-w-[170px] shrink-0 items-center gap-1.5 rounded-full border border-border/70 px-2.5 text-[12.5px] font-medium text-muted-foreground transition-colors hover:border-border hover:text-foreground",
            className,
          )}
        >
          <BookMarkedIcon className="size-3.5 shrink-0" />
          <span className="truncate">{current?.short || current?.title || resolvedId}</span>
          <ChevronDownIcon className="size-3 shrink-0 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent side="top" align="start" className="w-[360px] p-0">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <SearchIcon className="size-3.5 shrink-0 text-foreground/40" />
          <input
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={
              loaded
                ? "Search " + catalog.length.toLocaleString() + " styles"
                : "Loading styles"
            }
            className="w-full bg-transparent text-[13px] outline-none placeholder:text-foreground/35"
          />
        </div>
        <div className="max-h-80 overflow-y-auto p-1">
          {!hasQuery ? (
            <Section title="Common">
              {COMMON_STYLES.map((style) => (
                <Row
                  key={style.id}
                  label={style.title}
                  hint={style.format}
                  selected={style.id === resolvedId}
                  onClick={() => choose(style.id)}
                />
              ))}
            </Section>
          ) : results.length === 0 ? (
            <p className="px-3 py-6 text-center text-[12px] text-foreground/40">
              {loaded ? "No matching style." : "Loading styles"}
            </p>
          ) : (
            <Section
              title={
                results.length + " result" + (results.length === 1 ? "" : "s")
              }
            >
              {results.map((entry) => (
                <Row
                  key={entry.id}
                  label={entry.title}
                  hint={entry.format}
                  selected={entry.id === resolvedId}
                  onClick={() => choose(entry.id)}
                />
              ))}
            </Section>
          )}
          {error ? (
            <p className="text-danger px-3 py-2 text-[11px]">
              Could not load styles: {error}
            </p>
          ) : null}
        </div>
      </PopoverContent>
    </Popover>
  );
}
