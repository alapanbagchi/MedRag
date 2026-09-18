"use client";

import {
  Popover as ShadcnPopover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const PopoverSchema = z.object({
  trigger: z.any(),
  content: z.array(z.any()).default([]),
});

export const Popover = defineComponent({
  name: "Popover",
  props: PopoverSchema,
  description:
    "Inline detail on click for a single data point. trigger: the clickable node; content: nodes shown in the floating panel. Use instead of a full Dialog for one number.",
  component: ({ props, renderNode }) => (
    <ShadcnPopover>
      <PopoverTrigger asChild>
        <span className="inline-flex cursor-pointer items-center">{renderNode(props.trigger)}</span>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80 space-y-2">
        {renderNode(props.content)}
      </PopoverContent>
    </ShadcnPopover>
  ),
});
