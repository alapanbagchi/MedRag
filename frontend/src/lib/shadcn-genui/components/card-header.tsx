"use client";

import { CardDescription, CardTitle, CardHeader as ShadcnCardHeader } from "@/components/ui/card";
import { defineComponent } from "@openuidev/react-lang";
import { z } from "zod";
import { CitationText } from "@/components/CitationMarkdown";

const CardHeaderSchema = z.object({
  title: z.string(),
  description: z.string().optional(),
});

export const CardHeader = defineComponent({
  name: "CardHeader",
  props: CardHeaderSchema,
  description: "Title/description header block for a Card.",
  component: ({ props }) => (
    <ShadcnCardHeader className="p-0">
      <CardTitle>
        <CitationText text={String(props.title ?? "")} />
      </CardTitle>
      {props.description && (
        <CardDescription>
          <CitationText text={String(props.description)} />
        </CardDescription>
      )}
    </ShadcnCardHeader>
  ),
});
