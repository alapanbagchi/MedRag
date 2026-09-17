/**
 * Renderers for MedRAG's AG-UI CUSTOM events.
 *
 * The backend emits each custom event via `ctx.emit(...)`; the AG-UI runtime
 * turns it into a `data` message part named after the event. `step`,
 * `status`, and `thinking` are handled directly by AssistantMessage; only
 * the standalone cards below are registered here.
 */

import type { DataMessagePartComponent } from "@assistant-ui/react";
import { safeHref } from "../lib/url";

type Data = Record<string, any>;

const Shell = ({ children }: { children: React.ReactNode }) => (
  <div className="anim-rise my-1 overflow-hidden rounded-xl border border-border bg-card shadow-[0_4px_18px_rgba(13,14,26,0.05)]">
    {children}
  </div>
);

const Dot = () => (
  <span className="mt-[7px] size-1.5 shrink-0 rounded-full bg-[#4b8cf5]" aria-hidden />
);

export const SourcesCard: DataMessagePartComponent<Data> = ({ data }) => {
  const sources: Data[] = data?.sources ?? [];
  if (!sources.length) return null;
  return (
    <Shell>
      <div className="px-4 py-2.5 text-[13px] font-medium text-muted-foreground">{sources.length} sources</div>
      <ul className="space-y-0.5 px-4 pb-3">
        {sources.slice(0, 6).map((s, i) => {
          const href = safeHref(s.url) ?? (s.pmcid ? "https://pmc.ncbi.nlm.nih.gov/articles/" + encodeURIComponent(String(s.pmcid)) + "/" : null);
          const label = s.title || s.id || "Untitled source";
          return (
            <li key={String(s.id ?? s.pmcid ?? i)} className="flex items-start gap-2">
              <Dot />
              {href ? (
                <a href={href} target="_blank" rel="noreferrer" className="truncate text-[14px] leading-5 text-foreground underline-offset-2 hover:underline">
                  {label}
                </a>
              ) : (
                <span className="truncate text-[14px] leading-5 text-foreground">{label}</span>
              )}
            </li>
          );
        })}
        {sources.length > 6 ? (
          <li className="pl-3.5 text-[11px] text-muted-foreground">+{sources.length - 6} more</li>
        ) : null}
      </ul>
    </Shell>
  );
};

export const VerdictTableCard: DataMessagePartComponent<Data> = ({ data }) => {
  const rows: Data[] = data?.rows ?? [];
  if (!rows.length) return null;
  return (
    <Shell>
      <div className="px-4 py-2.5 text-[13px] font-medium text-muted-foreground">{rows.length} verdicts</div>
      <ul className="space-y-0.5 px-4 pb-3 text-[13px] text-foreground">
        {rows.slice(0, 8).map((r, i) => (
          <li key={String(r.evidence_id ?? r.id ?? i)} className="truncate">{r.evidence_id ?? r.id ?? JSON.stringify(r)}</li>
        ))}
      </ul>
    </Shell>
  );
};

export const RunStatsCard: DataMessagePartComponent<Data> = ({ data }) => {
  const usage = data?.usage ?? {};
  const tokens = (Number(usage.prompt) || 0) + (Number(usage.completion) || 0);
  if (!tokens) return null;
  return (
    <div className="px-1 py-1 text-[11px] text-muted-foreground">
      {tokens.toLocaleString()} tokens · {Number(usage.prompt) || 0} in / {Number(usage.completion) || 0} out
    </div>
  );
};

export const ErrorCard: DataMessagePartComponent<Data> = ({ data }) => (
  <div className="my-1 rounded-xl border border-red-200 bg-red-50 px-4 py-2.5 text-[13px] text-red-700">
    <span className="font-semibold">{data?.leg ? data.leg + " failed" : "Run failed"}</span>
    {data?.message ? <span className="ml-2">{String(data.message)}</span> : null}
  </div>
);

export const ToolProgressCard: DataMessagePartComponent<Data> = ({ data }) => {
  const name = String(data?.name ?? "");
  if (name.startsWith("local_search") || name.startsWith("retrieve")) return null;
  return (
    <div className="px-1 py-0.5 text-[12px] text-muted-foreground">
      {data?.detail ?? name ?? "tool progress"}
    </div>
  );
};

/** Name -> renderer map for the standalone AG-UI cards. */
export const AGUI_DATA_BY_NAME: Record<string, DataMessagePartComponent<any>> = {
  sources: SourcesCard,
  verdict_table: VerdictTableCard,
  run_stats: RunStatsCard,
  error_detail: ErrorCard,
  tool_progress: ToolProgressCard,
};
