/** Wire types shared by the AG-UI transport and the message renderers. */

export interface Source {
  id?: string;
  pmcid?: string | null;
  pmid?: string | null;
  title?: string;
  authors?: string[];
  journal?: string;
  year?: number;
  score?: number | null;
  snippet?: string;
  url?: string | null;
  /** Verified passage ids from this document (backend ledger keys). */
  passage_ids?: string[];
  /** Journal/site/database label from the backend source list. */
  sourceName?: string;
  /** citeproc-formatted reference text for the default style. */
  citation?: string;
  /** Per-style formatted citations, keyed by style id. */
  citations?: Record<string, SourceCitation>;
  /** 1-based citation number, the same one the inline badge shows. */
  index?: number;
  /** The ledger passage ref (P1, ...) this source resolves from. */
  ref?: string;
}

/** One source's citation in one style. */
export interface SourceCitation {
  /** The inline label core, e.g. "Zhang et al., 2025" or "1". */
  inline: string;
  /** The bibliography entry. */
  entry: string;
}

/** A citation style the answer dropdown can switch to. */
export interface CitationStyleDef {
  id: string;
  label: string;
  authorYear: boolean;
  /** Joins clustered inline markers, e.g. "; " or ",". */
  separator: string;
  prefix: string;
  suffix: string;
}
