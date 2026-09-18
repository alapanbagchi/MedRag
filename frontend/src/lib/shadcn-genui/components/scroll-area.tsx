"use client";

import { ScrollArea as ShadcnScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const ScrollAreaSchema = z.object({
  content: z.array(z.any()).default([]),
  orientation: z.enum(["vertical", "horizontal"]).optional(),
  maxHeight: z.number().optional(),
});

export const ScrollArea = defineComponent({
  name: "ScrollArea",
  props: ScrollAreaSchema,
  description:
    "Fixed-height scroll region for long evidence lists, so the Card cannot grow unbounded. content: nodes; orientation: scroll axis; maxHeight: pixel cap (default 320).",
  component: ({ props, renderNode }) => {
    const horizontal = props.orientation === "horizontal";
    return (
      <ShadcnScrollArea
        className="rounded-md border"
        style={{ maxHeight: props.maxHeight ? props.maxHeight + "px" : "20rem" }}
      >
        <div className={horizontal ? "flex w-max gap-3 p-3" : "space-y-3 p-3"}>
          {renderNode(props.content)}
        </div>
        {horizontal && <ScrollBar orientation="horizontal" />}
      </ShadcnScrollArea>
    );
  },
});
