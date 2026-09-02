// ── Sources: desktop drawer / mobile bottom sheet ───────────────────
// Two views: the source LIST (cards) and the ARTICLE view — clicking a
// reference opens the PMC article in the panel with the cited passage
// highlighted and scrolled into view.
"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp, ExternalLink, X } from "lucide-react";
import type { Source } from "@/lib/types";
import { cn, formatScore } from "@/lib/utils";
import { CapsLabel, Led } from "@/components/ui/primitives";
import { RAG_API_URL } from "@/lib/rag-client";

interface ArticleAnchor {
  paragraph: number; // flat paragraph index
  start: number;
  end: number;
  text: string;
}

interface Article {
  pmcid: string;
  title: string;
  url: string;
  total_paragraphs: number;
  anchor: ArticleAnchor | null;
  sections: { heading: string; paragraphs: string[] }[];
}

type LoadState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; article: Article; forKey: string }
  | { status: "error"; message: string };

function pmcidOf(source: Source): string | null {
  const id = source.pmcid ?? source.id;
  return id && /^PMC\d+$/.test(id) ? id : null;
}

function Highlighted({
  text,
  start,
  end,
}: {
  text: string;
  start: number;
  end: number;
}) {
  const s = Math.max(0, Math.min(start, text.length));
  const e = Math.max(s, Math.min(end, text.length));
  if (s >= e) return <>{text}</>;
  return (
    <>
      {text.slice(0, s)}
      <mark className="bg-accent/20 text-ink">{text.slice(s, e)}</mark>
      {text.slice(e)}
    </>
  );
}

function SourceCard({
  source,
  index,
  active,
  onSelect,
}: {
  source: Source;
  index: number;
  active: boolean;
  onSelect: () => void;
}) {
  const [open, setOpen] = useState(false);
  const num = String(index + 1).padStart(2, "0");
  return (
    <li className={cn("panel flex flex-col", active ? "border-accent" : "border-line")}>
      <button
        type="button"
        onClick={onSelect}
        title="Open article with the cited passage highlighted"
        className="flex w-full items-start gap-3 px-3 py-3 text-left"
      >
        <span className={cn("mono mt-0.5 w-8 flex-none text-[13px] font-semibold tnum", active ? "text-accent" : "text-ink2")}>
          {num}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-[13px] font-semibold leading-snug">{source.title}</span>
          <span className="mono mt-1.5 block text-[10.5px] leading-relaxed text-ink3">
            {source.journal} · {source.year} · {source.pmcid ?? source.id}
          </span>
          <span className="mono mt-1.5 block text-[9px] uppercase tracking-[0.14em] text-accent/80">
          {source.isWeb ? "Open website →" : "Open article →"}
        </span>
        </span>
      </button>

      {/* relevance + expand */}
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
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label={`Toggle excerpt for source ${num}`}
          className="icon-btn !h-6 !w-6"
        >
          {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>
      </div>

      {open && (
        <div className="anim-fade border-t border-line px-3 py-3">
          <p className="mono mb-2 text-[10px] uppercase tracking-[0.12em] text-ink3">
            {source.authors.slice(0, 3).join(", ")}{source.authors.length > 3 ? " et al." : ""}
          </p>
          {source.snippet && (
            <p className="text-[12.5px] leading-relaxed text-ink2">{source.snippet}</p>
          )}
          {source.pmid && <p className="mono mt-2 text-[10px] text-ink3">PMID {source.pmid}</p>}
          {source.url && (
            <a
              href={source.url}
              target="_blank"
              rel="noreferrer"
              title={source.isWeb ? `Open the website with the cited passage highlighted` : undefined}
              className="mono mt-3 inline-flex items-center gap-1.5 border border-line-strong px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-ink2 hover:border-accent hover:text-accent"
            >
              {source.isWeb ? "Open website — highlight passage" : "Open in PMC"} <ExternalLink size={11} />
            </a>
          )}
          {source.isWeb && source.highlight && (
            <p className="mt-2 border-t border-line/60 pt-2 text-[11px] leading-relaxed text-ink2">
              <span className="mono text-[9px] uppercase tracking-[0.14em] text-accent">Highlighted passage </span>
              “{source.highlight}”
            </p>
          )}
        </div>
      )}
    </li>
  );
}

function ArticleView({ article }: { article: Article }) {
  const anchorRef = useRef<HTMLParagraphElement | null>(null);
  const scrolledRef = useRef(false);
  if (!article) return null; // defensive: never render without data
  const an = article.anchor;
  const target = an?.paragraph ?? -1;

  // scroll the cited passage into view once per article
  useEffect(() => {
    if (scrolledRef.current || !an) return;
    scrolledRef.current = true;
    const t = window.setTimeout(() => {
      anchorRef.current?.scrollIntoView({ block: "center" });
    }, 60);
    return () => window.clearTimeout(t);
  }, [an]);

  let pIdx = -1;
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* article meta */}
      <div className="border-b border-line px-3 py-3">
        <p className="text-[13.5px] font-semibold leading-snug">{article.title}</p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span className="mono text-[9.5px] uppercase tracking-[0.12em] text-ink3">
            {article.pmcid} · {article.total_paragraphs} paragraphs
          </span>
          {an && (
            <button
              type="button"
              onClick={() => anchorRef.current?.scrollIntoView({ block: "center" })}
              className="mono inline-flex items-center gap-1 border border-accent/60 px-1.5 py-0.5 text-[9.5px] uppercase tracking-[0.12em] text-accent hover:bg-accent/10"
            >
              ↑ Cited passage p{an.paragraph + 1}
            </button>
          )}
          <a
            href={article.url}
            target="_blank"
            rel="noreferrer"
            className="mono inline-flex items-center gap-1 border border-line-strong px-1.5 py-0.5 text-[9.5px] uppercase tracking-[0.12em] text-ink2 hover:border-accent hover:text-accent"
          >
            Open in PMC <ExternalLink size={10} />
          </a>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {article.sections.map((sec, si) => (
          <section key={`${article.pmcid}-${si}`} className="px-3 py-2.5">
            <h3 className="mono border-b border-line pb-1 text-[9.5px] uppercase tracking-[0.18em] text-ink3">
              {sec.heading}
            </h3>
            {sec.paragraphs.map((text) => {
              pIdx += 1;
              const isAnchor = an != null && pIdx === target;
              return (
                <p
                  key={pIdx}
                  ref={isAnchor ? anchorRef : undefined}
                  className={cn(
                    "mt-2 text-[13px] leading-relaxed text-ink2",
                    isAnchor && "border-l-2 border-accent bg-accent/[0.05] py-1 pl-3"
                  )}
                >
                  {isAnchor && an ? (
                    <>
                      <span className="mono mr-2 align-middle text-[8.5px] uppercase tracking-[0.14em] text-accent">
                        ↑ cited
                      </span>
                      <Highlighted text={text} start={an.start} end={an.end} />
                    </>
                  ) : (
                    text
                  )}
                </p>
              );
            })}
          </section>
        ))}
      </div>
    </div>
  );
}

export function SourcePanel({
  open,
  onClose,
  sources,
  activeCitation,
  isDesktop,
  contextualLabel,
  onSelectCitation,
}: {
  open: boolean;
  onClose: () => void;
  sources: Source[];
  activeCitation: number | null;
  isDesktop: boolean;
  contextualLabel?: string;
  onSelectCitation?: (n: number) => void;
}) {
  const [view, setView] = useState<"article" | "list">("article");
  const [load, setLoad] = useState<LoadState>({ status: "idle" });
  const [retryTick, setRetryTick] = useState(0);

  const active = activeCitation != null ? sources[activeCitation - 1] : undefined;
  const pmcid = active ? pmcidOf(active) : null;
  const fetchKey = active && pmcid ? `${pmcid}|${active.snippet ?? ""}` : "";

  // a citation click opens the article; without one, show the list
  useEffect(() => {
    if (activeCitation != null) setView("article");
    else setView("list");
  }, [activeCitation]);

  // fetch the article for the active source (one request per pmcid+excerpt)
  useEffect(() => {
    if (view !== "article" || !active || !pmcid || !RAG_API_URL || !fetchKey) {
      setLoad({ status: "idle" });
      return;
    }
    if (load.status === "ready" && load.forKey === fetchKey) return;
    const ctrl = new AbortController();
    setLoad({ status: "loading" });
    (async () => {
      try {
        const url = `${RAG_API_URL}/v1/articles/${encodeURIComponent(pmcid)}` +
          (active.snippet ? `?anchor=${encodeURIComponent(active.snippet)}` : "");
        const res = await fetch(url, { signal: ctrl.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text().catch(() => "")).slice(0, 160)}`);
        const article = (await res.json()) as Article;
        setLoad({ status: "ready", article, forKey: fetchKey });
      } catch (err) {
        if (ctrl.signal.aborted) return;
        setLoad({ status: "error", message: err instanceof Error ? err.message : "fetch failed" });
      }
    })();
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, fetchKey, active?.id, retryTick]);

  if (!open) return null;

  const header = (
    <div className="flex h-12 flex-none items-center justify-between border-b border-line px-3">
      <div className="flex flex-col">
        <CapsLabel className="flex items-center gap-2">
          {view === "article" ? "Article" : "Sources"}{" "}
          {view === "article" && active && pmcid ? (
            <span className="mono text-accent tnum">{pmcid}</span>
          ) : (
            <span className="text-accent tnum">{String(sources.length).padStart(2, "0")}</span>
          )}
        </CapsLabel>
        {contextualLabel && (
          <span className="mono mt-0.5 text-[9px] uppercase tracking-[0.14em] text-ink3">{contextualLabel}</span>
        )}
      </div>
      <div className="flex items-center gap-1.5">
        <div className="flex border border-line">
          <button
            type="button"
            disabled={activeCitation == null}
            onClick={() => setView("article")}
            className={cn(
              "mono px-2 py-1 text-[9.5px] uppercase tracking-[0.12em]",
              view === "article" ? "bg-accent/15 text-accent" : "text-ink3 hover:text-ink",
              activeCitation == null && "cursor-not-allowed opacity-40"
            )}
          >
            Article
          </button>
          <button
            type="button"
            onClick={() => setView("list")}
            className={cn(
              "mono px-2 py-1 text-[9.5px] uppercase tracking-[0.12em]",
              view === "list" ? "bg-accent/15 text-accent" : "text-ink3 hover:text-ink"
            )}
          >
            List
          </button>
        </div>
        <button type="button" onClick={onClose} aria-label="Close sources" className="icon-btn">
          <X size={15} />
        </button>
      </div>
    </div>
  );

  const citationChips = activeCitation != null && (
    <div className="flex flex-none items-center gap-1.5 overflow-x-auto border-b border-line px-3 py-2">
      <span className="mono text-[9px] uppercase tracking-[0.16em] text-ink3">Refs</span>
      {sources.map((s, i) => {
        const n = i + 1;
        return (
          <button
            key={s.id}
            type="button"
            onClick={() => onSelectCitation?.(n)}
            title={s.title}
            className={cn(
              "mono flex-none border px-1.5 py-0.5 text-[9.5px] tnum transition-colors",
              activeCitation === n
                ? "border-accent bg-accent/15 text-accent"
                : "border-line text-ink3 hover:border-accent hover:text-accent"
            )}
          >
            {String(n).padStart(2, "0")}
          </button>
        );
      })}
    </div>
  );

  let body: React.ReactNode;
  if (view === "list") {
    body = (
      <ul className="flex-1 space-y-2 overflow-y-auto p-3">
        {sources.map((s, i) => (
          <SourceCard
            key={s.id}
            source={s}
            index={i}
            active={activeCitation === i + 1}
            onSelect={() => onSelectCitation?.(i + 1)}
          />
        ))}
      </ul>
    );
  } else if (!active || !pmcid) {
    body = active && active.isWeb ? (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-center">
        <p className="mono text-[10.5px] uppercase tracking-[0.14em] text-ink3">
          Web source · {active.journal}
        </p>
        {active.snippet && (
          <p className="mono max-w-md text-[11px] leading-relaxed text-ink2">
            “{active.snippet}”
          </p>
        )}
        {active.highlight && (
          <p className="mono max-w-md border border-accent/40 bg-accent/[0.05] px-2 py-1.5 text-[10.5px] leading-relaxed text-ink">
            Highlight: {active.highlight}
          </p>
        )}
        {active.url && (
          <a
            href={active.url}
            target="_blank"
            rel="noreferrer"
            title="Open the website with the cited passage highlighted"
            className="btn btn--accent inline-flex items-center gap-1.5"
          >
            Open website — highlight passage <ExternalLink size={13} />
          </a>
        )}
      </div>
    ) : (
      <div className="flex flex-1 items-center justify-center p-6 text-center">
        <p className="mono text-[11px] uppercase tracking-[0.14em] text-ink3">
          {active ? "No PMCID for this source" : "Select a source"}
        </p>
      </div>
    );
  } else if (load.status === "loading") {
    body = (
      <div className="flex flex-1 items-center justify-center gap-2.5 p-6">
        <Led state="accent" pulse />
        <span className="mono text-[10.5px] uppercase tracking-[0.14em] text-ink2">
          Fetching article · {pmcid}
        </span>
      </div>
    );
  } else if (load.status === "idle") {
    body = (
      <div className="flex flex-1 flex-col items-center justify-center gap-2.5 p-6 text-center">
        <p className="mono text-[10.5px] uppercase tracking-[0.14em] text-ink3">
          {RAG_API_URL ? "Loading article…" : "No backend connected — mock mode"}
        </p>
        {active?.url && (
          <a
            href={active.url}
            target="_blank"
            rel="noreferrer"
            className="mono inline-flex items-center gap-1.5 border border-line-strong px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-ink2 hover:border-accent hover:text-accent"
          >
            {active.isWeb ? "Open website — highlight passage" : "Open in PMC"} <ExternalLink size={11} />
          </a>
        )}
      </div>
    );
  } else if (load.status === "error") {
    body = (
      <div className="flex flex-1 flex-col p-3">
        <div className="panel border-err/60">
          <p className="mono text-[10px] uppercase tracking-[0.14em] text-err">Article unavailable</p>
          <p className="mono mt-2 break-all text-[10.5px] leading-relaxed text-ink3">{load.message}</p>
          {active?.snippet && (
            <p className="mt-2 border-t border-line pt-2 text-[12px] leading-relaxed text-ink2">
              <span className="mono text-[9px] uppercase tracking-[0.14em] text-ink3">Cited passage </span>
              {active.snippet}
            </p>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setRetryTick((t) => t + 1)}
              className="btn btn--accent"
            >
              Retry
            </button>
            {active.url && (
              <a
                href={active.url}
                target="_blank"
                rel="noreferrer"
                className="mono inline-flex items-center gap-1.5 border border-line-strong px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-ink2 hover:border-accent hover:text-accent"
              >
                {active.isWeb ? "Open website — highlight passage" : "Open in PMC"} <ExternalLink size={11} />
              </a>
            )}
          </div>
        </div>
      </div>
    );
  } else if (load.status === "ready") {
    body = <ArticleView article={load.article} />;
  } else {
    body = null;
  }

  if (isDesktop) {
    return (
      <div
        role="complementary"
        aria-label="Sources"
        className="anim-drawer fixed inset-y-0 right-0 z-40 flex w-[440px] max-w-[92vw] flex-col border-l border-line bg-ground shadow-2xl"
      >
        {header}
        {citationChips}
        {body}
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label="Sources">
      <button type="button" aria-label="Close sources" className="anim-fade absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="anim-sheet absolute inset-x-0 bottom-0 flex max-h-[78vh] flex-col border-t border-line bg-ground shadow-2xl">
        {header}
        {citationChips}
        {body}
      </div>
    </div>
  );
}