import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import { isValidElement, type ReactNode } from "react";

/**
 * Wraps every `[n]` citation marker in a numbered chip, matching the
 * "Sources" card numbering (source index + 1).
 */
function withCitations(node: ReactNode, keyBase = 0): ReactNode[] {
  const out: ReactNode[] = [];
  const emit = (n: ReactNode, key: number) => out.push(n);

  if (typeof node === "string") {
    const parts = node.split(/(\[\d+\])/g);
    parts.forEach((part, i) => {
      if (!part) return;
      const match = /^\[(\d+)\]$/.exec(part);
      if (match) {
        emit(
          <sup
            key={`cite-${keyBase}-${i}`}
            title={`Source ${match[1]}`}
            className="cite-chip"
            suppressHydrationWarning
          >
            {match[1]}
          </sup>,
          i,
        );
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
  p: ({ children }) => <p className="my-2.5 leading-7 text-[15px]">{withCitations(children)}</p>,
  h1: ({ children }) => <h1 className="mb-2 mt-5 text-xl font-semibold tracking-tight">{withCitations(children)}</h1>,
  h2: ({ children }) => <h2 className="mb-2 mt-4 text-lg font-semibold tracking-tight">{withCitations(children)}</h2>,
  h3: ({ children }) => <h3 className="mb-2 mt-3 text-base font-semibold">{withCitations(children)}</h3>,
  ul: ({ children }) => <ul className="my-2.5 ml-5 list-disc space-y-1.5">{withCitations(children)}</ul>,
  ol: ({ children }) => <ol className="my-2.5 ml-5 list-decimal space-y-1.5">{withCitations(children)}</ol>,
  li: ({ children }) => <li className="leading-7">{withCitations(children)}</li>,
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
    />
  );
}