"use client";

import { defineComponent } from "@openuidev/react-lang";
import { z } from "zod";
import { CitationMarkdown } from "@/components/CitationMarkdown";

const MarkDownRendererSchema = z.object({
  text: z.string(),
});

export const MarkDownRenderer = defineComponent({
  name: "MarkDownRenderer",
  props: MarkDownRendererSchema,
  description: "Renders markdown text with GFM support and clickable [n] citation badges.",
  component: ({ props }) => {
    const text = props.text == null ? "" : String(props.text);
    return (
      <CitationMarkdown
        text={text}
        className="prose prose-neutral dark:prose-invert max-w-none text-sm"
      />
    );
  },
});
