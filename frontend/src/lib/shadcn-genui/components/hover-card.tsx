"use client";

import {
  HoverCard as ShadcnHoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const HoverCardSchema = z.object({
  trigger: z.any(),
  content: z.array(z.any()).default([]),
});

export const HoverCard = defineComponent({
  name: "HoverCard",
  props: HoverCardSchema,
  description:
    "Citation preview on hover. trigger: the visible node (e.g. a Badge marker); content: nodes revealed in the card. Never repeat the full citation inline.",
  component: ({ props, renderNode }) => (
    <ShadcnHoverCard>
      <HoverCardTrigger asChild>
        <span className="inline-flex cursor-help items-center">{renderNode(props.trigger)}</span>
      </HoverCardTrigger>
      <HoverCardContent className="w-80 space-y-2">{renderNode(props.content)}</HoverCardContent>
    </ShadcnHoverCard>
  ),
});
