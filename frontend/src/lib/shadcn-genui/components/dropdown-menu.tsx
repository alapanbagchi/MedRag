"use client";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu as ShadcnDropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem as ShadcnDropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { defineComponent } from "@openuidev/react-lang";
import { ChevronDownIcon } from "lucide-react";
import * as React from "react";
import { z } from "zod";

import { actionSchema } from "../action";
import { useGenuiAction } from "./use-genui-action";

const DropdownMenuItemSchema = z.object({
  label: z.string(),
  action: actionSchema,
  variant: z.enum(["default", "destructive"]).optional(),
  inset: z.boolean().optional(),
});

export const DropdownMenuItem = defineComponent({
  name: "DropdownMenuItem",
  props: DropdownMenuItemSchema,
  description:
    'One row action (e.g. "Show abstract", "Open in PubMed", "Exclude"). label: row text; action: fired on select; variant: "default" | "destructive".',
  component: () => null,
});

const DropdownMenuSchema = z.object({
  trigger: z.string(),
  items: z.array(DropdownMenuItem.ref).default([]),
  label: z.string().optional(),
});

export const DropdownMenu = defineComponent({
  name: "DropdownMenu",
  props: DropdownMenuSchema,
  description:
    "Row-level actions opened from a button. trigger: button text; items: DropdownMenuItem[]; label: optional group heading.",
  component: ({ props }) => {
    const dispatch = useGenuiAction();
    const items = (props.items ?? []).filter((item) => item?.props?.label != null);

    return (
      <ShadcnDropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm">
            {props.trigger}
            <ChevronDownIcon className="size-3.5" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {props.label && <DropdownMenuLabel>{props.label}</DropdownMenuLabel>}
          {props.label && <DropdownMenuSeparator />}
          {items.map((item, i) => {
            const label = String(item.props.label);
            return (
              <ShadcnDropdownMenuItem
                key={i}
                variant={item.props.variant ?? "default"}
                inset={item.props.inset}
                onSelect={() => dispatch(label, item.props.action)}
              >
                {label}
              </ShadcnDropdownMenuItem>
            );
          })}
        </DropdownMenuContent>
      </ShadcnDropdownMenu>
    );
  },
});
