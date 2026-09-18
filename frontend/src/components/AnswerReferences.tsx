"use client";

import { useActiveCitationStyle, useCitationSources } from "./CitationBadge";
import { entryFor } from "@/lib/citations";
import { safeHref } from "@/lib/url";

function pmcUrl(pmcid: string): string {
  return "https://pmc.ncbi.nlm.nih.gov/articles/" + encodeURIComponent(pmcid) + "/";
}

/**
 * The answer's reference list, ordered at the bottom of the message.
 *
 * Both the Visual and Text answers share it. Each entry is the citation for
 * the currently selected style (APA/Harvard/Chicago/Vancouver/AMA), so it
 * always matches the inline badges, and the list is rendered from the
 * backend's verified source list, never from model output.
 */
export function AnswerReferences() {
  const sources = useCitationSources();
  const style = useActiveCitationStyle();
  const items = sources
    .map((source, index) => ({ source, index }))
    .filter(({ source }) =>
      Boolean(source.title || source.url || source.citation || source.pmcid || source.id),
    );
  if (items.length === 0) return null;

  return (
    <section aria-label="References" className="mt-8 border-t border-border/40 pt-4">
      <div className="mb-3 text-[12px] font-medium tracking-[0.06em] text-foreground/35">
        References
      </div>
      <ol className="flex flex-col gap-3">
        {items.map(({ source, index }) => {
          const href =
            safeHref(source.url) ??
            (source.pmcid ? pmcUrl(source.pmcid) : undefined);
          const label = entryFor(source, style.id);
          return (
            <li
              key={source.id ?? source.pmcid ?? index}
              id={"ref-" + (index + 1)}
              className="cite-reference flex scroll-mt-24 items-start gap-2.5 text-[16px] leading-6 text-foreground/60"
            >
              <span className="mt-px w-5 shrink-0 text-right text-[13px] leading-6 tabular-nums text-foreground/30">
                {index + 1}
              </span>
              {href ? (
                <a
                  href={href}
                  target="_blank"
                  rel="noreferrer"
                  className="min-w-0 transition-colors hover:text-foreground/85 hover:underline"
                >
                  {label}
                </a>
              ) : (
                <span className="min-w-0">{label}</span>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
