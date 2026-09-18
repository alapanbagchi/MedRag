"use client";

import { Button } from "@/components/ui/button";
import {
  Sheet as ShadcnSheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const SheetSchema = z.object({
  trigger: z.string(),
  side: z.enum(["top", "right", "bottom", "left"]).optional(),
  title: z.string(),
  description: z.string().optional(),
  content: z.array(z.any()).default([]),
});

export const Sheet = defineComponent({
  name: "Sheet",
  props: SheetSchema,
  description:
    'Side panel for full-text context without covering the answer. trigger: button text; side: "top" | "right" | "bottom" | "left" (default right); title/description: header; content: panel nodes.',
  component: ({ props, renderNode }) => (
    <ShadcnSheet>
      <SheetTrigger asChild>
        <Button variant="outline">{props.trigger}</Button>
      </SheetTrigger>
      <SheetContent side={props.side ?? "right"}>
        <SheetHeader>
          <SheetTitle>{props.title}</SheetTitle>
          {props.description && <SheetDescription>{props.description}</SheetDescription>}
        </SheetHeader>
        <div className="flex-1 space-y-3 overflow-y-auto px-4 pb-4">{renderNode(props.content)}</div>
      </SheetContent>
    </ShadcnSheet>
  ),
});
