"use client";

import {
  Menubar as ShadcnMenubar,
  MenubarContent,
  MenubarItem as ShadcnMenubarItem,
  MenubarMenu as ShadcnMenubarMenu,
  MenubarTrigger,
} from "@/components/ui/menubar";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

import { actionSchema } from "../action";
import { useGenuiAction } from "./use-genui-action";

const MenubarItemSchema = z.object({
  label: z.string(),
  action: actionSchema,
  variant: z.enum(["default", "destructive"]).optional(),
  inset: z.boolean().optional(),
});

export const MenubarItem = defineComponent({
  name: "MenubarItem",
  props: MenubarItemSchema,
  description: "One action inside a MenubarMenu. label: row text; action: fired on select.",
  component: () => null,
});

const MenubarMenuSchema = z.object({
  trigger: z.string(),
  items: z.array(MenubarItem.ref).default([]),
});

export const MenubarMenu = defineComponent({
  name: "MenubarMenu",
  props: MenubarMenuSchema,
  description: "One top-level menubar menu. trigger: menu label; items: MenubarItem[].",
  component: () => null,
});

const MenubarSchema = z.object({
  menus: z.array(MenubarMenu.ref).default([]),
});

export const Menubar = defineComponent({
  name: "Menubar",
  props: MenubarSchema,
  description: "Desktop-style menu bar for a full research console. menus: MenubarMenu[].",
  component: ({ props }) => {
    const dispatch = useGenuiAction();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const menus = ((props.menus ?? []) as any[]).filter((menu) => menu?.props?.trigger != null);

    return (
      <ShadcnMenubar>
        {menus.map((menu, i) => {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const items = ((menu.props.items ?? []) as any[]).filter(
            (item) => item?.props?.label != null,
          );
          return (
            <ShadcnMenubarMenu key={i}>
              <MenubarTrigger>{String(menu.props.trigger)}</MenubarTrigger>
              <MenubarContent>
                {items.map((item, j) => {
                  const label = String(item.props.label);
                  return (
                    <ShadcnMenubarItem
                      key={j}
                      variant={item.props.variant ?? "default"}
                      inset={item.props.inset}
                      onSelect={() => dispatch(label, item.props.action)}
                    >
                      {label}
                    </ShadcnMenubarItem>
                  );
                })}
              </MenubarContent>
            </ShadcnMenubarMenu>
          );
        })}
      </ShadcnMenubar>
    );
  },
});
