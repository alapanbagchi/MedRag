import type { Source } from "../lib/types";

function domainOf(source: Source): string {
  try {
    if (source.url) return new URL(source.url).hostname.replace(/^www\./, "");
  } catch {
    // ignore
  }
  return (source.journal ?? "").split("·").pop()?.trim() ?? "web";
}

/**
 * Source logo badge. Local-corpus PMC rows get the local PMC mark;
 * everything else gets the publisher / website mark (initial tile).
 */
export function SourceLogo({ source, size = "md" }: { source: Source; size?: "sm" | "md" }) {
  const isPmc = !!source.pmcid;
  const box = size === "sm" ? "size-5 text-[8px]" : "size-6 text-[9px]";
  if (isPmc) {
    return (
      <span
        title={`Local corpus · ${source.pmcid}`}
        className={`flex ${box} shrink-0 items-center justify-center rounded-md font-bold tracking-tight text-white`}
        style={{ background: "linear-gradient(135deg, #1883AE 0%, #18AE95 100%)" }}
      >
        PMC
      </span>
    );
  }
  const domain = domainOf(source);
  const initial = (domain.charAt(0) || "W").toUpperCase();
  return (
    <span
      title={domain}
      className={`flex ${box} shrink-0 items-center justify-center rounded-md font-bold text-white`}
      style={{ backgroundColor: "#0D0E1A" }}
    >
      {initial}
    </span>
  );
}

/** Publisher / id line under a source title. */
export function SourceOrigin({ source }: { source: Source }) {
  const domain = domainOf(source);
  if (source.pmcid) {
    return (
      <span className="truncate font-mono text-[11px] text-muted-foreground">
        {source.pmcid} · local corpus
      </span>
    );
  }
  return <span className="truncate text-[11px] text-muted-foreground">{domain}</span>;
}
