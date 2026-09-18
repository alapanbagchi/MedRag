"use client";

import {
  ToggleGroup as ShadcnToggleGroup,
  ToggleGroupItem as ShadcnToggleGroupItem,
} from "@/components/ui/toggle-group";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const ToggleGroupSchema = z.object({
  items: z.array(z.string()).default([]),
  multiple: z.boolean().optional(),
  defaultValue: z.string().optional(),
});

export const ToggleGroup = defineComponent({
  name: "ToggleGroup",
  props: ToggleGroupSchema,
  description:
    'Study-design filter as toggle buttons (e.g. RCT / Cohort / Case-control). items: labels; multiple: allow more than one on; defaultValue: initially active label. Never write "you can filter by..." in prose.',
  component: ({ props }) => {
    const items = (props.items ?? []).map((item) => String(item));
    const [single, setSingle] = React.useState<string>(props.defaultValue ?? "");
    const [multi, setMulti] = React.useState<string[]>(
      props.defaultValue ? [props.defaultValue] : [],
    );

    const groupClass = "flex flex-wrap items-center gap-1 rounded-md border p-1";

    if (props.multiple) {
      return (
        <ShadcnToggleGroup type="multiple" value={multi} onValueChange={setMulti} className={groupClass}>
          {items.map((label) => (
            <ShadcnToggleGroupItem key={label} value={label}>
              {label}
            </ShadcnToggleGroupItem>
          ))}
        </ShadcnToggleGroup>
      );
    }

    return (
      <ShadcnToggleGroup type="single" value={single} onValueChange={setSingle} className={groupClass}>
        {items.map((label) => (
          <ShadcnToggleGroupItem key={label} value={label}>
            {label}
          </ShadcnToggleGroupItem>
        ))}
      </ShadcnToggleGroup>
    );
  },
});
