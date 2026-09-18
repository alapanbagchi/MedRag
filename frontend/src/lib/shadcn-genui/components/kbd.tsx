"use client";

import { Kbd as ShadcnKbd, KbdGroup } from "@/components/ui/kbd";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const KbdSchema = z.object({
  keys: z.union([z.string(), z.array(z.string())]),
});

export const Kbd = defineComponent({
  name: "Kbd",
  props: KbdSchema,
  description:
    'Keyboard shortcut display. keys: one string ("Cmd+K") or a string array rendered as a combo.',
  component: ({ props }) => {
    const keys = Array.isArray(props.keys) ? props.keys : [props.keys];
    return (
      <KbdGroup>
        {keys.map((key, i) => (
          <ShadcnKbd key={i}>{String(key)}</ShadcnKbd>
        ))}
      </KbdGroup>
    );
  },
});
