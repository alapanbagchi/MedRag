import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { cloneElement, isValidElement, type ReactNode } from "react";
import {
  InlineCitationBadge,
  useActiveCitationStyle,
  useCitationSources,
  type ResolvedSource,
} from "./CitationBadge";
import {
  inlineCore,
  resolveCitation,
  tokenizeCitations,
  type CitationToken,
} from "@/lib/citations";
import type { CitationStyleDef, Source } from "@/lib/types";

/**
 * A run of one or more adjacent citation markers rendered as ONE badge:
 * "(Zhang et al., 2025; Smith, 2024)" for author-year styles, "(1,2)" for
 * numeric ones. Unresolved ids are shown without brackets and never linked.
 */
function CitationCluster({
  ids,
  sources,
  style,
}: {
  ids: string[];
  sources: Source[];
  style: CitationStyleDef;
}) {
  const hits: ResolvedSource[] = [];
  const labels: string[] = [];
  for (const id of ids) {
    const resolved = resolveCitation(sources, id);
    if (resolved) {
      hits.push(resolved);
      labels.push(inlineCore(resolved.source, style.id));
    } else {
      labels.push(id);
    }
  }
  if (hits.length === 0) {
    return (
      <span className="cite-chip cite-chip-missing" title="Unresolved citation">
        {labels.join(", ")}
      </span>
    );
  }
  const label = style.prefix + labels.join(style.separator) + style.suffix;
  return <InlineCitationBadge label={label} sources={hits} />;
}

/** Render a token stream, merging consecutive citation markers into clusters. */
function renderTokens(
  tokens: CitationToken[],
  sources: Source[],
  style: CitationStyleDef,
  keyBase: number,
): ReactNode[] {
  const out: ReactNode[] = [];
  let cluster: string[] = [];
  let key = 0;
  const flush = () => {
    if (!cluster.length) return;
    out.push(
      <CitationCluster
        key={keyBase + "-c" + key++}
        ids={cluster}
        sources={sources}
        style={style}
      />,
    );
    cluster = [];
  };
  for (const token of tokens) {
    if (token.kind === "cite") {
      cluster.push(token.id ?? "");
      continue;
    }
    // Drop only the whitespace between markers of one cluster.
    if (cluster.length && token.value.trim() === "") continue;
    flush();
    out.push(token.value);
  }
  flush();
  return out;
}

/** Render a plain string's citation markers as style-aware badges. */
export function CitationText({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  const sources = useCitationSources();
  const style = useActiveCitationStyle();
  return (
    <span className={className}>
      {renderTokens(tokenizeCitations(text), sources, style, 0)}
    </span>
  );
}

/** Walk a markdown-rendered child tree and badge every citation marker. */
function renderCitationNodes(
  node: ReactNode,
  sources: Source[],
  style: CitationStyleDef,
  keyBase = 0,
): ReactNode {
  if (typeof node === "string") {
    return renderTokens(tokenizeCitations(node), sources, style, keyBase);
  }
  if (Array.isArray(node)) {
    return node.map((child, index) =>
      renderCitationNodes(child, sources, style, keyBase * 31 + index),
    );
  }
  if (isValidElement(node)) {
    const props = (node.props ?? {}) as { children?: ReactNode };
    if (props.children == null) return node;
    return cloneElement(
      node,
      undefined,
      renderCitationNodes(props.children, sources, style, keyBase + 1),
    );
  }
  return node;
}

/**
 * The answer markdown hierarchy, shared by the plain markdown answer, the
 * OpenUI Text view and the OpenUI MarkDownRenderer component.
 *
 * One opacity scale carries the hierarchy: headings sit at full foreground,
 * body copy at 80%, table headers and list markers fade further back. The size
 * steps stay deliberately small so the difference reads as structure rather
 * than shouting, and tables are ruled instead of boxed.
 */
export const markdownComponents: Components = {
  h1: ({ children }) => <Heading level={1}>{children}</Heading>,
  h2: ({ children }) => <Heading level={2}>{children}</Heading>,
  h3: ({ children }) => <Heading level={3}>{children}</Heading>,
  p: ({ children }) => {
    const sources = useCitationSources();
    const style = useActiveCitationStyle();
    return (
      <p className="my-3 text-[16px] leading-7 text-foreground/80">
        {renderCitationNodes(children, sources, style)}
      </p>
    );
  },
  strong: ({ children }) => (
    <strong className="font-semibold text-foreground">{children}</strong>
  ),
  em: ({ children }) => <em className="text-foreground/90 italic">{children}</em>,
  ul: ({ children }) => {
    const sources = useCitationSources();
    const style = useActiveCitationStyle();
    return (
      <ul className="my-3 list-disc space-y-1.5 pl-5 text-[16px] leading-7 text-foreground/80">
        {renderCitationNodes(children, sources, style)}
      </ul>
    );
  },
  ol: ({ children }) => {
    const sources = useCitationSources();
    const style = useActiveCitationStyle();
    return (
      <ol className="my-3 list-decimal space-y-1.5 pl-5 text-[16px] leading-7 text-foreground/80">
        {renderCitationNodes(children, sources, style)}
      </ol>
    );
  },
  li: ({ children }) => {
    const sources = useCitationSources();
    const style = useActiveCitationStyle();
    return (
      <li className="pl-0.5 leading-7 marker:text-foreground/30">
        {renderCitationNodes(children, sources, style)}
      </li>
    );
  },
  blockquote: ({ children }) => {
    const sources = useCitationSources();
    const style = useActiveCitationStyle();
    return (
      <blockquote className="my-4 rounded-r-lg border-l-2 border-foreground/15 bg-foreground/[0.03] py-1.5 pl-3.5 pr-3 text-[15px] leading-7 text-foreground/75 [&_p]:my-1 [&_p]:text-inherit">
        {renderCitationNodes(children, sources, style)}
      </blockquote>
    );
  },
  hr: () => <hr className="my-6 border-border/60" />,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="font-medium text-primary underline decoration-primary/30 underline-offset-2 transition-colors hover:decoration-primary"
    >
      {children}
    </a>
  ),
  code: ({ children }) => (
    <code className="rounded-md bg-foreground/[0.06] px-1.5 py-0.5 font-mono text-[13px] text-foreground/90">
      {children}
    </code>
  ),
  pre: ({ children }) => (
    <pre className="my-4 overflow-x-auto rounded-xl bg-foreground/[0.04] p-3.5 font-mono text-[13px] leading-6">
      {children}
    </pre>
  ),
  table: ({ children }) => (
    <div className="my-4 overflow-x-auto rounded-xl border border-border/70">
      <table className="w-full border-collapse text-[14px]">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-foreground/[0.035]">{children}</thead>,
  th: ({ children }) => (
    <th className="px-3 py-2 text-left text-[11px] font-semibold tracking-wider text-foreground/55 uppercase">
      {children}
    </th>
  ),
  td: ({ children }) => {
    const sources = useCitationSources();
    const style = useActiveCitationStyle();
    return (
      <td className="border-t border-border/60 px-3 py-2 align-top leading-6 text-foreground/80">
        {renderCitationNodes(children, sources, style)}
      </td>
    );
  },
};

function Heading({ level, children }: { level: 1 | 2 | 3; children: ReactNode }) {
  const sources = useCitationSources();
  const style = useActiveCitationStyle();
  const content = renderCitationNodes(children, sources, style);
  if (level === 1) {
    return (
      <h1 className="mb-3 mt-1 text-[22px] leading-[1.25] font-semibold tracking-tight text-foreground">
        {content}
      </h1>
    );
  }
  if (level === 2) {
    return (
      <h2 className="mb-2.5 mt-7 text-[18px] leading-snug font-semibold tracking-tight text-foreground first:mt-0">
        {content}
      </h2>
    );
  }
  return (
    <h3 className="mb-2 mt-5 text-[15px] font-semibold text-foreground/90">
      {content}
    </h3>
  );
}

/** A block of markdown with citation-aware components (OpenUI text blocks). */
export function CitationMarkdown({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  return (
    <div className={className}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
