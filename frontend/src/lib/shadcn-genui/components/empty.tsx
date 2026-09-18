"use client";

import {
  Empty as ShadcnEmpty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const EmptySchema = z.object({
  title: z.string(),
  description: z.string().optional(),
  content: z.array(z.any()).default([]),
});

export const Empty = defineComponent({
  name: "Empty",
  props: EmptySchema,
  description:
    'Empty state when a filter returns zero studies, e.g. title "No RCTs match this population". title: headline; description: explanation; content: optional actions.',
  component: ({ props, renderNode }) => (
    <ShadcnEmpty>
      <EmptyHeader>
        <EmptyTitle>{props.title}</EmptyTitle>
        {props.description && <EmptyDescription>{props.description}</EmptyDescription>}
      </EmptyHeader>
      {props.content && props.content.length > 0 && (
        <EmptyContent>{renderNode(props.content)}</EmptyContent>
      )}
    </ShadcnEmpty>
  ),
});
