"use client";

import { Button } from "@/components/ui/button";
import {
  Command as ShadcnCommand,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem as ShadcnCommandItem,
  CommandList,
} from "@/components/ui/command";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

import { actionSchema } from "../action";
import { useGenuiAction } from "./use-genui-action";

const CommandItemSchema = z.object({
  label: z.string(),
  description: z.string().optional(),
  keywords: z.string().optional(),
  action: actionSchema,
});

export const CommandItem = defineComponent({
  name: "CommandItem",
  props: CommandItemSchema,
  description:
    "One selectable row in a Command palette. label: primary text; description: secondary text; keywords: extra search terms; action: fired on select.",
  component: () => null,
});

const CommandSchema = z.object({
  placeholder: z.string().optional(),
  items: z.array(CommandItem.ref).default([]),
  empty: z.string().optional(),
  dialog: z.boolean().optional(),
});

export const Command = defineComponent({
  name: "Command",
  props: CommandSchema,
  description:
    'Evidence search palette (Cmd+K). placeholder: input hint; items: CommandItem[]; empty: no-results text; dialog: true renders a button that opens the palette modally.',
  component: ({ props }) => {
    const dispatch = useGenuiAction();
    const items = (props.items ?? []).filter((item) => item?.props?.label != null);
    const [open, setOpen] = React.useState(false);
    const [query, setQuery] = React.useState("");

    const rows = (
      <CommandList>
        <CommandEmpty>{props.empty ?? "No results found."}</CommandEmpty>
        <CommandGroup>
          {items.map((item, i) => {
            const label = String(item.props.label);
            const value = [label, item.props.description, item.props.keywords]
              .filter(Boolean)
              .join(" ");
            return (
              <ShadcnCommandItem
                key={i}
                value={value}
                onSelect={() => {
                  if (item.props.action) dispatch(label, item.props.action);
                }}
              >
                <div className="flex flex-col">
                  <span>{label}</span>
                  {item.props.description && (
                    <span className="text-muted-foreground text-xs">
                      {String(item.props.description)}
                    </span>
                  )}
                </div>
              </ShadcnCommandItem>
            );
          })}
        </CommandGroup>
      </CommandList>
    );

    if (props.dialog) {
      return (
        <>
          <Button
            variant="outline"
            className="w-full justify-start"
            onClick={() => setOpen(true)}
          >
            <span className="text-muted-foreground text-sm">
              {props.placeholder ?? "Search evidence..."}
            </span>
          </Button>
          <CommandDialog
            open={open}
            onOpenChange={setOpen}
            title={props.placeholder ?? "Search"}
            description="Search the verified evidence."
          >
            <CommandInput
              placeholder={props.placeholder ?? "Search evidence..."}
              value={query}
              onValueChange={setQuery}
            />
            {rows}
          </CommandDialog>
        </>
      );
    }

    return (
      <ShadcnCommand className="rounded-md border">
        <CommandInput
          placeholder={props.placeholder ?? "Search evidence..."}
          value={query}
          onValueChange={setQuery}
        />
        {rows}
      </ShadcnCommand>
    );
  },
});
