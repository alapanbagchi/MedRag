// ── Markdown renderer with inline citation chips ────────────────────
"use client";

import { cloneElement, isValidElement } from "react";
import type { ReactElement, ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Walk rendered children; split text nodes on [n] and turn matches into
 *  interactive citation chips. Applied at block level so nested markdown
 *  (strong, em, code) keeps working. */
function withCitations(node: ReactNode, onCite: (n: number) => void, keyPrefix = "c"): ReactNode {
  if (typeof node === "string") {
    const parts = node.split(/(\[\d+\])/g);
    if (parts.length === 1) return node;
    return parts.map((part, i) => {
      const m = /^\[(\d+)\]$/.exec(part);
      if (!m) return part;
      const n = parseInt(m[1], 10);
      return (
        <button
          type="button"
          key={`${keyPrefix}-${i}`}
          onClick={(e) => {
            e.stopPropagation();
            onCite(n);
          }}
          className="cite-ref tnum"
          aria-label={`Citation ${n}`}
        >
          {n}
        </button>
      );
    });
  }
  if (Array.isArray(node)) {
    return node.map((child, i) => withCitations(child, onCite, `${keyPrefix}-${i}`));
  }
  if (isValidElement(node)) {
    const el = node as ReactElement<{ children?: ReactNode }>;
    if (el.props && el.props.children != null) {
      return cloneElement(el, { children: withCitations(el.props.children, onCite, keyPrefix) });
    }
  }
  return node;
}

export function Markdown({
  text,
  onCite,
}: {
  text: string;
  onCite: (n: number) => void;
}) {
  const components = {
    h1: (props: { children?: ReactNode }) => (
      <h1 className="mb-3 mt-7 text-[20px] font-extrabold uppercase tracking-tight">{withCitations(props.children, onCite)}</h1>
    ),
    h2: (props: { children?: ReactNode }) => (
      <h2 className="mt-8 border-t border-line pt-3 text-[16px] font-extrabold uppercase tracking-[0.01em]">{withCitations(props.children, onCite)}</h2>
    ),
    h3: (props: { children?: ReactNode }) => (
      <h3 className="mb-1.5 mt-6 text-[14px] font-bold">{withCitations(props.children, onCite)}</h3>
    ),
    p: (props: { children?: ReactNode }) => (
      <p className="my-3 max-w-[72ch] text-[14.5px] leading-[1.72] text-ink/90">{withCitations(props.children, onCite)}</p>
    ),
    ul: (props: { children?: ReactNode }) => (
      <ul className="my-3 flex list-disc flex-col gap-1.5 pl-5 marker:text-accent">
        {props.children}
      </ul>
    ),
    ol: (props: { children?: ReactNode }) => (
      <ol className="my-3 flex list-decimal flex-col gap-1.5 pl-6 marker:font-bold marker:text-accent">
        {props.children}
      </ol>
    ),
    li: (props: { children?: ReactNode }) => (
      <li className="max-w-[72ch] pl-1 text-[14.5px] leading-[1.65] text-ink/90">
        {withCitations(props.children, onCite)}
      </li>
    ),
    strong: (props: { children?: ReactNode }) => <strong className="font-bold text-ink">{props.children}</strong>,
    em: (props: { children?: ReactNode }) => <em className="italic">{props.children}</em>,
    a: (props: { href?: string; children?: ReactNode }) => (
      <a href={props.href} target="_blank" rel="noreferrer" className="text-accent underline decoration-1 underline-offset-2 hover:brightness-110">
        {props.children}
      </a>
    ),
    blockquote: (props: { children?: ReactNode }) => (
      <blockquote className="my-3 border border-line bg-ground2/60 px-4 py-3 text-[13.5px] leading-relaxed text-ink2 [border-left:2px_solid_var(--accent)]">
        {withCitations(props.children, onCite)}
      </blockquote>
    ),
    hr: () => <hr className="my-5 border-t border-line" />,
    code: (props: { children?: ReactNode }) => (
      <code className="mono rounded-none border border-line bg-ground2 px-1 py-0.5 text-[12px] text-ink">{props.children}</code>
    ),
    pre: (props: { children?: ReactNode }) => (
      <pre className="mono my-3 overflow-x-auto border border-line bg-ground2 p-3 text-[12px] leading-relaxed text-ink">
        {props.children}
      </pre>
    ),
    table: (props: { children?: ReactNode }) => (
      <div className="my-4 overflow-x-auto border border-line">
        <table className="w-full border-collapse text-[13px]">{props.children}</table>
      </div>
    ),
    thead: (props: { children?: ReactNode }) => <thead className="border-b border-line bg-ground2">{props.children}</thead>,
    th: (props: { children?: ReactNode }) => (
      <th className="border-r border-line px-3 py-2 text-left text-[11px] font-bold uppercase tracking-[0.08em] last:border-r-0">
        {withCitations(props.children, onCite)}
      </th>
    ),
    td: (props: { children?: ReactNode }) => (
      <td className="border-r border-line px-3 py-2 align-top leading-relaxed text-ink/90 last:border-r-0">
        {withCitations(props.children, onCite)}
      </td>
    ),
  };

  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {text}
    </ReactMarkdown>
  );
}
