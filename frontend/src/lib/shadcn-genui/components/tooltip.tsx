"use client";

import {
  Tooltip as ShadcnTooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const TooltipSchema = z.object({
  trigger: z.any(),
  text: z.string(),
});

export const Tooltip = defineComponent({
  name: "Tooltip",
  props: TooltipSchema,
  description:
    "One-line definition shown on hover. trigger: the term node (e.g. TextContent 'GRADE certainty'); text: the definition. Never parenthesize definitions in prose.",
  component: ({ props, renderNode }) => (
    <ShadcnTooltip>
      <TooltipTrigger asChild>
        <span className="cursor-help underline decoration-dotted underline-offset-2">
          {renderNode(props.trigger)}
        </span>
      </TooltipTrigger>
      <TooltipContent>{props.text}</TooltipContent>
    </ShadcnTooltip>
  ),
});
