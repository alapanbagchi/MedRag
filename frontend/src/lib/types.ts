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
}
