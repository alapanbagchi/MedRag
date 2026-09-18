"use client";

import {
  NavigationMenu as ShadcnNavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem as ShadcnNavigationMenuItem,
  NavigationMenuList,
  NavigationMenuTrigger,
} from "@/components/ui/navigation-menu";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const NavigationMenuItemSchema = z.object({
  trigger: z.string(),
  content: z.array(z.any()).default([]),
});

export const NavigationMenuItem = defineComponent({
  name: "NavigationMenuItem",
  props: NavigationMenuItemSchema,
  description: "One top-level section. trigger: section label; content: nodes revealed in the dropdown.",
  component: () => null,
});

const NavigationMenuSchema = z.object({
  items: z.array(NavigationMenuItem.ref).default([]),
});

export const NavigationMenu = defineComponent({
  name: "NavigationMenu",
  props: NavigationMenuSchema,
  description:
    'Top-level sections: "Evidence", "Subgroups", "Heterogeneity", "GRADE". items: NavigationMenuItem[].',
  component: ({ props, renderNode }) => {
    const items = (props.items ?? []).filter((item) => item?.props?.trigger != null);
    if (items.length === 0) return null;

    return (
      <ShadcnNavigationMenu>
        <NavigationMenuList>
          {items.map((item, i) => (
            <ShadcnNavigationMenuItem key={i}>
              <NavigationMenuTrigger>{String(item.props.trigger)}</NavigationMenuTrigger>
              <NavigationMenuContent>
                <div className="w-[min(90vw,28rem)] space-y-2 p-3">
                  {renderNode(item.props.content)}
                </div>
              </NavigationMenuContent>
            </ShadcnNavigationMenuItem>
          ))}
        </NavigationMenuList>
      </ShadcnNavigationMenu>
    );
  },
});
