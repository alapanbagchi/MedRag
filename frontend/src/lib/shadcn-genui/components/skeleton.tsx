"use client";

import { Skeleton as ShadcnSkeleton } from "@/components/ui/skeleton";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const ROUNDED: Record<string, string> = {
  none: "rounded-none",
  sm: "rounded-sm",
  md: "rounded-md",
  lg: "rounded-lg",
  full: "rounded-full",
};

function toCssSize(value: string | number | undefined): string | undefined {
  if (value == null) return undefined;
  if (typeof value === "number") return value + "px";
  return /^[0-9]+(\.[0-9]+)?$/.test(value) ? value + "px" : value;
}

const SkeletonSchema = z.object({
  width: z.union([z.string(), z.number()]).optional(),
  height: z.union([z.string(), z.number()]).optional(),
  rounded: z.enum(["none", "sm", "md", "lg", "full"]).optional(),
});

export const Skeleton = defineComponent({
  name: "Skeleton",
  props: SkeletonSchema,
  description:
    "Streaming placeholder. Match width/height to the component that will replace it to prevent layout shift. width/height accept px numbers or CSS strings; rounded: none | sm | md | lg | full.",
  component: ({ props }) => (
    <ShadcnSkeleton
      className={ROUNDED[props.rounded ?? "md"]}
      style={{ width: toCssSize(props.width), height: toCssSize(props.height) }}
    />
  ),
});
