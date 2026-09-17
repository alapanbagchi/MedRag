import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import { isValidElement, type ReactNode } from "react";
import { InlineCiteRef } from "./assistant-ui/elements/inline-citation";

/**
 * Turns every citation marker into a source badge: `[n]` indexes the
 * Sources card (1-based), anything else (e.g. `[P1]`) matches a
 * verified passage ref or id in the Sources card (case-insensitive).
 * Parenthesized ref groups (`(P1, P10)`) split into one badge per ref.
 * Unresolvable markers render as plain text.
 */
function withCitations(node: ReactNode, keyBase = 0): ReactNode[] {
  const out: ReactNode[] = [];
  const emit = (n: ReactNode, key: number) => out.push(n);

  if (typeof node === "string") {
    // `[4]` / `[p8]`, reversed `p[4]`, and parenthesized ref groups
    // `(P1, P10)` the model sometimes emits instead of brackets.
    const parts = node.split(/(\[[A-Za-z]*\d[\w-]*\]|\b[Pp]\[\d{1,4}\]|\([Pp]\d[\w-]*(?:\s*[,;]\s*[Pp]\d[\w-]*)*\))/g);
    parts.forEach((part, i) => {
      if (!part) return;
      const match = /^\[([A-Za-z]*\d[\w-]*)\]$/.exec(part);
      const reversed = /^[Pp]\[(\d{1,4})\]$/.exec(part);
      const paren = /^\((.*)\)$/.exec(part);
      if (match) {
        emit(<InlineCiteRef key={`cite-${keyBase}-${i}`} raw={part} id={match[1]} />, i);
      } else if (reversed) {
        emit(<InlineCiteRef key={`cite-${keyBase}-${i}`} raw={part} id={`p${reversed[1]}`} />, i);
      } else if (paren && /^[Pp]\d[\w-]*(?:\s*[,;]\s*[Pp]\d[\w-]*)*$/.test(paren[1] ?? "")) {
        // One badge per ref, keeping the original parens and separators
        // so unresolvable ids render exactly as written.
        emit("(", i);
        (paren[1] ?? "").split(/(\s*[,;]\s*)/).forEach((tok, j) => {
          if (!tok) return;
          if (/^[Pp]\d[\w-]*$/.test(tok)) {
            emit(<InlineCiteRef key={`cite-${keyBase}-${i}-${j}`} raw={tok} id={tok} />, i);
          } else {
            emit(tok, i);
          }
        });
        emit(")", i);
      } else {
        emit(part, i);
      }
    });
    return out;
  }

  if (Array.isArray(node)) {
    node.forEach((child, i) => out.push(...withCitations(child, keyBase * 31 + i)));
    return out;
  }

  if (isValidElement(node)) {
    const props = (node.props ?? {}) as { children?: ReactNode };
    if (props.children != null) {
      emit(
        {
          ...node,
          props: { ...props, children: withCitations(props.children, keyBase + 1) },
        },
        keyBase,
      );
    } else {
      emit(node, keyBase);
    }
    return out;
  }

  emit(node, keyBase);
  return out;
}

const components: Components = {
  p: ({ children }) => <p className="my-3 text-[17px] leading-8">{withCitations(children)}</p>,
  h1: ({ children }) => <h1 className="mb-2.5 mt-6 text-2xl font-semibold tracking-tight">{withCitations(children)}</h1>,
  h2: ({ children }) => <h2 className="mb-2.5 mt-5 text-xl font-semibold tracking-tight">{withCitations(children)}</h2>,
  h3: ({ children }) => <h3 className="mb-2 mt-4 text-lg font-semibold">{withCitations(children)}</h3>,
  ul: ({ children }) => <ul className="my-3 ml-5 list-disc space-y-2 text-[17px]">{withCitations(children)}</ul>,
  ol: ({ children }) => <ol className="my-3 ml-5 list-decimal space-y-2 text-[17px]">{withCitations(children)}</ol>,
  li: ({ children }) => <li className="leading-8">{withCitations(children)}</li>,
  blockquote: ({ children }) => (
    <blockquote className="my-3 border-l-2 border-primary/40 pl-3 text-muted-foreground">
      {withCitations(children)}
    </blockquote>
  ),
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2 hover:brightness-110">
      {children}
    </a>
  ),
  code: ({ children }) => (
    <code className="rounded-md bg-muted px-1.5 py-0.5 font-mono text-[13px]">{children}</code>
  ),
  pre: ({ children }) => (
    <pre className="my-3 overflow-x-auto rounded-xl bg-muted p-3.5 font-mono text-[13px]">{children}</pre>
  ),
  table: ({ children }) => (
    <div className="my-3 overflow-x-auto">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-border bg-muted/50 px-2.5 py-1.5 text-left font-semibold">{children}</th>
  ),
  td: ({ children }) => <td className="border border-border px-2.5 py-1.5">{withCitations(children)}</td>,
};

/** Assistant answer text: GFM markdown + [n] citation chips + stream reveal. */
export function MarkdownText() {
  return (
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm]}
      components={components}
      className="text-foreground"
      defer
    />
  );
}