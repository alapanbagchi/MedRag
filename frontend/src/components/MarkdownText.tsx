import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import { markdownComponents } from "./CitationMarkdown";

/**
 * Assistant answer text: GFM markdown plus numbered, clickable citation
 * badges. The source list is provided once per message (AgUiAssistantMessage),
 * so [n] / [Pn] markers resolve against the same numbered list the bottom
 * reference section renders.
 */
export function MarkdownText() {
  return (
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm]}
      components={markdownComponents}
      className="text-foreground"
      defer
    />
  );
}
