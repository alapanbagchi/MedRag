"use client";

import { Spinner as ShadcnSpinner } from "@/components/ui/spinner";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const SpinnerSchema = z.object({
  label: z.string().optional(),
});

export const Spinner = defineComponent({
  name: "Spinner",
  props: SpinnerSchema,
  description: "Inline loading indicator while a single passage resolves. label: optional text.",
  component: ({ props }) => (
    <div className="text-muted-foreground flex items-center gap-2 text-sm">
      <ShadcnSpinner className="size-4" />
      {props.label && <span>{props.label}</span>}
    </div>
  ),
});
