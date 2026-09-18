"use client";

import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { defineComponent } from "@openuidev/react-lang";
import { CheckIcon, ChevronsUpDownIcon } from "lucide-react";
import * as React from "react";
import { z } from "zod";

import { actionSchema } from "../action";
import { useGenuiAction } from "./use-genui-action";

const ComboboxItemSchema = z.object({
  value: z.string(),
  label: z.string(),
  description: z.string().optional(),
  action: actionSchema,
});

export const ComboboxItem = defineComponent({
  name: "ComboboxItem",
  props: ComboboxItemSchema,
  description:
    "One option for Combobox. value: unique id; label: visible text; description: secondary text; action: fired on select.",
  component: () => null,
});

const ComboboxSchema = z.object({
  trigger: z.string(),
  items: z.array(ComboboxItem.ref).default([]),
  placeholder: z.string().optional(),
  searchPlaceholder: z.string().optional(),
});

export const Combobox = defineComponent({
  name: "Combobox",
  props: ComboboxSchema,
  description:
    "Searchable select for filtering studies (design, population, intervention). trigger: button label; items: ComboboxItem[]; placeholder: idle text; searchPlaceholder: input hint.",
  component: ({ props }) => {
    const dispatch = useGenuiAction();
    const items = (props.items ?? []).filter((item) => item?.props?.value != null);
    const [open, setOpen] = React.useState(false);
    const [selected, setSelected] = React.useState<string | null>(null);

    const selectedItem = items.find((item) => String(item.props.value) === selected);

    return (
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between font-normal"
          >
            <span className={selectedItem ? "" : "text-muted-foreground"}>
              {selectedItem ? String(selectedItem.props.label) : (props.placeholder ?? props.trigger)}
            </span>
            <ChevronsUpDownIcon className="ml-2 size-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-[var(--radix-popover-trigger-width)] p-0">
          <Command>
            <CommandInput placeholder={props.searchPlaceholder ?? "Search..."} />
            <CommandList>
              <CommandEmpty>No results.</CommandEmpty>
              <CommandGroup>
                {items.map((item, i) => {
                  const value = String(item.props.value);
                  const label = String(item.props.label);
                  return (
                    <CommandItem
                      key={i}
                      value={[label, item.props.description].filter(Boolean).join(" ")}
                      onSelect={() => {
                        setSelected(value);
                        setOpen(false);
                        if (item.props.action) dispatch(label, item.props.action);
                      }}
                    >
                      <CheckIcon
                        className={selected === value ? "mr-2 size-4 opacity-100" : "mr-2 size-4 opacity-0"}
                      />
                      <div className="flex flex-col">
                        <span>{label}</span>
                        {item.props.description && (
                          <span className="text-muted-foreground text-xs">
                            {String(item.props.description)}
                          </span>
                        )}
                      </div>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    );
  },
});
