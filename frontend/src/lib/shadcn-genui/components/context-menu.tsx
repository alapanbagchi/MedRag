"use client";

import {
  ContextMenu as ShadcnContextMenu,
  ContextMenuContent,
  ContextMenuItem as ShadcnContextMenuItem,
  ContextMenuLabel,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

import { actionSchema } from "../action";
import { useGenuiAction } from "./use-genui-action";

const ContextMenuItemSchema = z.object({
  label: z.string(),
  action: actionSchema,
  variant: z.enum(["default", "destructive"]).optional(),
  inset: z.boolean().optional(),
});

export const ContextMenuItem = defineComponent({
  name: "ContextMenuItem",
  props: ContextMenuItemSchema,
  description:
    'One right-click action (e.g. "Copy citation", "Highlight", "Compare"). label: row text; action: fired on select.',
  component: () => null,
});

const ContextMenuSchema = z.object({
  trigger: z.any(),
  items: z.array(ContextMenuItem.ref).default([]),
  label: z.string().optional(),
});

export const ContextMenu = defineComponent({
  name: "ContextMenu",
  props: ContextMenuSchema,
  description:
    "Right-click actions on a passage. trigger: the region node; items: ContextMenuItem[]; label: optional heading.",
  component: ({ props, renderNode }) => {
    const dispatch = useGenuiAction();
    const items = (props.items ?? []).filter((item) => item?.props?.label != null);

    return (
      <ShadcnContextMenu>
        <ContextMenuTrigger asChild>
          <div className="rounded-md border border-dashed p-3">{renderNode(props.trigger)}</div>
        </ContextMenuTrigger>
        <ContextMenuContent>
          {props.label && <ContextMenuLabel>{props.label}</ContextMenuLabel>}
          {props.label && <ContextMenuSeparator />}
          {items.map((item, i) => {
            const label = String(item.props.label);
            return (
              <ShadcnContextMenuItem
                key={i}
                variant={item.props.variant ?? "default"}
                inset={item.props.inset}
                onSelect={() => dispatch(label, item.props.action)}
              >
                {label}
              </ShadcnContextMenuItem>
            );
          })}
        </ContextMenuContent>
      </ShadcnContextMenu>
    );
  },
});
