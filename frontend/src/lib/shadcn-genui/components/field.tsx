"use client";

import {
  Field as ShadcnField,
  FieldDescription,
  FieldLabel,
} from "@/components/ui/field";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const FieldSchema = z.object({
  label: z.string(),
  control: z.any(),
  description: z.string().optional(),
});

export const Field = defineComponent({
  name: "Field",
  props: FieldSchema,
  description:
    "Generic form field wrapper when FormControl is too opinionated. label: field label; control: the input node; description: helper text.",
  component: ({ props, renderNode }) => (
    <ShadcnField>
      <FieldLabel>{props.label}</FieldLabel>
      {renderNode(props.control)}
      {props.description && <FieldDescription>{props.description}</FieldDescription>}
    </ShadcnField>
  ),
});
