"use client";

import { AspectRatio as ShadcnAspectRatio } from "@/components/ui/aspect-ratio";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const AspectRatioSchema = z.object({
  ratio: z.number(),
  content: z.array(z.any()).default([]),
});

export const AspectRatio = defineComponent({
  name: "AspectRatio",
  props: AspectRatioSchema,
  description:
    "Constrain an image thumbnail or forest-plot preview to a fixed ratio. ratio: width / height (e.g. 1.7778 for 16:9); content: nodes.",
  component: ({ props, renderNode }) => (
    <ShadcnAspectRatio ratio={props.ratio} className="overflow-hidden rounded-md">
      <div className="h-full w-full">{renderNode(props.content)}</div>
    </ShadcnAspectRatio>
  ),
});
