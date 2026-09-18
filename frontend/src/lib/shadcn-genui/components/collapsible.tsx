"use client";

import {
  Collapsible as ShadcnCollapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { defineComponent } from "@openuidev/react-lang";
import { ChevronDownIcon } from "lucide-react";
import * as React from "react";
import { z } from "zod";

const CollapsibleSchema = z.object({
  trigger: z.any(),
  content: z.array(z.any()).default([]),
  defaultOpen: z.boolean().optional(),
});

export const Collapsible = defineComponent({
  name: "Collapsible",
  props: CollapsibleSchema,
  description:
    "Single expandable section for nested evidence (e.g. subgroup analyses). trigger: header node; content: revealed nodes; defaultOpen: start expanded.",
  component: ({ props, renderNode }) => (
    <ShadcnCollapsible defaultOpen={props.defaultOpen} className="rounded-md border">
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className="hover:bg-muted/50 group flex w-full items-center justify-between gap-2 p-3 text-left text-sm font-medium"
        >
          <span className="flex-1">{renderNode(props.trigger)}</span>
          <ChevronDownIcon className="size-4 shrink-0 transition-transform group-data-[state=open]:rotate-180" />
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-3 border-t p-3">
        {renderNode(props.content)}
      </CollapsibleContent>
    </ShadcnCollapsible>
  ),
});
