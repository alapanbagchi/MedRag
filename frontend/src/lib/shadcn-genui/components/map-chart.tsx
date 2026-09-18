"use client";

import * as echarts from "echarts/core";
import { feature } from "topojson-client";
import worldTopologyRaw from "world-atlas/countries-110m.json?raw";

import { EChart, type EChartOption } from "@/components/ui/echart";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

import { ChartFrame } from "./chart-frame";
import { cssVar, seriesPalette } from "@/lib/theme";

const metaSchema = z
  .object({
    title: z.string().optional(),
    caption: z.string().optional(),
    unit: z.string().optional(),
    legend: z.boolean().optional(),
    roam: z.boolean().optional(),
  })
  .optional();

type Meta = { title?: string; caption?: string; unit?: string; legend?: boolean; roam?: boolean };

/** Atlas names that differ from the common English name a model will emit. */
const NAME_ALIASES: Record<string, string> = {
  "United States of America": "United States",
  "Dem. Rep. Congo": "DR Congo",
  "Central African Rep.": "Central African Republic",
  "Dominican Rep.": "Dominican Republic",
  "Eq. Guinea": "Equatorial Guinea",
  "S. Sudan": "South Sudan",
  "Bosnia and Herz.": "Bosnia and Herzegovina",
  "Solomon Is.": "Solomon Islands",
  "Dem. Rep. Korea": "North Korea",
  "Republic of Korea": "South Korea",
  "Czechia": "Czech Republic",
  "Lao PDR": "Laos",
  "Macedonia": "North Macedonia",
  "eSwatini": "Eswatini",
};

let worldRegistered = false;

function ensureWorldMap() {
  if (worldRegistered) return;
  try {
    const topology = JSON.parse(worldTopologyRaw) as {
      objects: Record<string, unknown>;
    };
    const geometry = topology.objects.countries;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const geo = feature(topology as any, geometry as any) as unknown;
    echarts.registerMap("world", geo as never);
    worldRegistered = true;
  } catch {
    worldRegistered = false;
  }
}

// Register at module load: the map must exist before any MapChart sets its
// option, and React runs a child EChart effect before the parent effect.
ensureWorldMap();

type Virtual = { props: Record<string, unknown> };

function nodesOf(value: unknown): Virtual[] {
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is Virtual => !!x && typeof x === "object" && "props" in (x as object));
}

function text(value: unknown): string {
  if (value == null) return "";
  return typeof value === "string" ? value : String(value);
}

function num(value: unknown): number | undefined {
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

const MapPointSchema = z.object({
  lat: z.number(),
  lng: z.number(),
  label: z.string().optional(),
  value: z.number().optional(),
});

export const MapPoint = defineComponent({
  name: "MapPoint",
  props: MapPointSchema,
  description:
    "One marker on a MapChart. lat/lng: coordinates in decimal degrees; label: place or study name; value: optional magnitude used for the marker size and tooltip.",
  component: () => null,
});

const MapRegionSchema = z.object({
  name: z.string(),
  value: z.number(),
  label: z.string().optional(),
});

export const MapRegion = defineComponent({
  name: "MapRegion",
  props: MapRegionSchema,
  description:
    "One shaded country on a MapChart. name: country name in common English (United States, China, Brazil); value: magnitude that drives the color scale; label: optional display name.",
  component: () => null,
});

export const MapChart = defineComponent({
  name: "MapChart",
  props: z.object({
    points: z.array(MapPoint.ref).optional(),
    regions: z.array(MapRegion.ref).optional(),
    region: z.string().optional(),
    meta: metaSchema,
  }),
  description:
    "World map for geographic evidence: shaded countries (regions) and/or located study sites (points). Use for geographic distribution or country-level rates. Any country shading must come from a cited numeric value.",
  component: ({ props }) => {
    React.useEffect(() => {
      ensureWorldMap();
    }, []);

    const points = nodesOf(props.points);
    const regions = nodesOf(props.regions);
    if (points.length === 0 && regions.length === 0) return null;

    const regionValues = regions
      .map((r) => num(r.props.value))
      .filter((v): v is number => v != null);
    const regionMin = regionValues.length ? Math.min.apply(null, regionValues) : 0;
    const regionMax = regionValues.length ? Math.max.apply(null, regionValues) : 1;

    const pointValues = points
      .map((p) => num(p.props.value))
      .filter((v): v is number => v != null);
    const pointMax = pointValues.length ? Math.max.apply(null, pointValues) : 1;

    const m = (props.meta ?? {}) as Meta;
    const unit = m.unit;
    const roam = m.roam !== false;

    const option = {
      color: seriesPalette(5),
      tooltip: {
        trigger: "item",
        formatter: (params: { name?: string; value?: unknown }) => {
          const label = params.name ?? "";
          const raw = Array.isArray(params.value) ? params.value[2] : params.value;
          const value = typeof raw === "number" ? raw : undefined;
          return value == null
            ? label
            : label + ": " + value + (unit ? " " + unit : "");
        },
      },
      legend:
        m.legend !== false
          ? { top: 0, data: points.length ? ["Study sites"] : [] }
          : { show: false },
      geo: {
        map: "world",
        roam,
        nameMap: NAME_ALIASES,
        zoom: 1.05,
        itemStyle: {
          areaColor: cssVar("--chart-surface", "#eef2f6"),
          borderColor: cssVar("--chart-line", "#c9d4d9"),
          borderWidth: 0.5,
        },
        emphasis: {
          itemStyle: { areaColor: cssVar("--chart-surface-strong", "#dbe7f3") },
          label: { show: false },
        },
        select: { disabled: true },
      },
      visualMap:
        regions.length > 0
          ? {
              min: regionMin,
              max: regionMax === regionMin ? regionMin + 1 : regionMax,
              seriesIndex: 0,
              left: 8,
              bottom: 8,
              calculable: true,
              text: unit ? ["High (" + unit + ")", "Low"] : ["High", "Low"],
              inRange: {
                color: [
                  cssVar("--chart-heat-low", "#e8f3f1"),
                  cssVar("--chart-heat-mid", "#4b8cf5"),
                  cssVar("--chart-heat-high", "#0b3f8f"),
                ],
              },
            }
          : undefined,
      series: [
        regions.length > 0
          ? {
              name: "regions",
              type: "map",
              geoIndex: 0,
              data: regions.map((r) => ({
                name: r.props.label ? text(r.props.label) : text(r.props.name),
                value: num(r.props.value) ?? 0,
              })),
            }
          : undefined,
        points.length > 0
          ? {
              name: "Study sites",
              type: "effectScatter",
              coordinateSystem: "geo",
              effect: { show: true, scale: 1.6, brushType: "stroke" },
              symbolSize: (value: unknown) => {
                const magnitude = Array.isArray(value) ? Number(value[2]) : undefined;
                if (magnitude != null && pointMax > 0) {
                  return 8 + (magnitude / pointMax) * 14;
                }
                return 9;
              },
              label: {
                show: true,
                formatter: "{b}",
                position: "right",
                fontSize: 9,
                color: cssVar("--foreground", "#0d0e1a"),
              },
              itemStyle: { color: cssVar("--series-4", "#e5484d") },
              data: points.map((p) => ({
                name: p.props.label ? text(p.props.label) : text(p.props.lat) + ", " + text(p.props.lng),
                value: [num(p.props.lng) ?? 0, num(p.props.lat) ?? 0, num(p.props.value)],
              })),
            }
          : undefined,
      ].filter(Boolean),
    } as unknown as EChartOption;

    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <EChart option={option} height={360} />
      </ChartFrame>
    );
  },
});
