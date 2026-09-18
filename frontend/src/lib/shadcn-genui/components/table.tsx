"use client";

import {
  Table as ShadcnTable,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";
import { CitationText } from "@/components/CitationMarkdown";

const ColSchema = z.object({
  header: z.string(),
  type: z.enum(["string", "number", "boolean"]).optional(),
  unit: z.string().optional(),
});

export const Col = defineComponent({
  name: "Col",
  props: ColSchema,
  description:
    "Column definition for Table. header: column name; type: string | number | boolean (numbers are right-aligned); unit: measurement unit shown after the header (e.g. mg, %, RR, 95% CI).",
  component: () => null,
});

const TableSchema = z.object({
  columns: z.array(Col.ref),
  rows: z.array(z.array(z.any())),
  title: z.string().optional(),
  caption: z.string().optional(),
});

export const Table = defineComponent({
  name: "Table",
  props: TableSchema,
  description:
    "Data table. title: what the table shows (e.g. 'Baseline characteristics by arm'); columns: Col[] with header/type/unit; rows: 2D array of values; caption: source, n, or caveat. Every numeric column should carry its unit.",
  component: ({ props }) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const columns = ((props.columns ?? []) as any[]).map((c) => ({
      header: String(c?.props?.header ?? ""),
      type: c?.props?.type ?? "string",
      unit: c?.props?.unit ? String(c.props.unit) : "",
    }));
    const rows = (props.rows ?? []) as unknown[][];

    return (
      <div className="space-y-1.5">
        {props.title && <p className="text-foreground text-sm font-medium">{props.title}</p>}
        <div className="rounded-md border">
          <ShadcnTable>
            <TableHeader>
              <TableRow>
                {columns.map((col, i) => (
                  <TableHead key={i} className={col.type === "number" ? "text-right" : ""}>
                    {col.header}
                    {col.unit && (
                      <span className="text-muted-foreground font-normal"> ({col.unit})</span>
                    )}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row, ri) => (
                <TableRow key={ri}>
                  {columns.map((col, ci) => (
                    <TableCell
                      key={ci}
                      className={col.type === "number" ? "text-right tabular-nums" : ""}
                    >
                      <CitationText text={String(row[ci] ?? "")} />
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </ShadcnTable>
        </div>
        {props.caption && (
          <p className="text-muted-foreground text-xs leading-relaxed">{props.caption}</p>
        )}
      </div>
    );
  },
});
