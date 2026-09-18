/**
 * OpenUI Lang -> markdown.
 *
 * The Text tab shows the same answer as the Visual tab, converted from the
 * parsed program rather than a second model call: prose passes through,
 * structure becomes headings and blockquotes, tables and every chart's data
 * become GFM tables, and the verified ledger supplies the reference list.
 *
 * Coverage is deliberate: charts, maps and tables are converted from their
 * data so the reading view never silently drops a figure the visual view
 * shows. Anything unmapped falls through to a generic text scrape.
 */

import type { ElementNode } from "@openuidev/react-lang";

export interface MarkdownSource {
  title: string;
  sourceName: string;
  url?: string;
}

type Props = Record<string, unknown>;

const NL = String.fromCharCode(10);

function isNode(value: unknown): value is ElementNode {
  if (!value || typeof value !== "object") return false;
  const node = value as { type?: unknown; typeName?: unknown };
  return node.type === "element" && typeof node.typeName === "string";
}

function nodes(value: unknown): ElementNode[] {
  if (Array.isArray(value)) return value.filter(isNode);
  return isNode(value) ? [value] : [];
}

function str(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function cell(value: unknown): string | null {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  const text = str(value);
  return text || null;
}

function markdownTable(headers: string[], rows: (string | null)[][]): string {
  const body = rows.filter((row) => row.some((value) => value !== null));
  if (headers.length === 0 || body.length === 0) return "";
  const head = "| " + headers.join(" | ") + " |";
  const rule = "| " + headers.map(() => "---").join(" | ") + " |";
  const lines = body.map((row) => "| " + row.map((value) => value ?? "").join(" | ") + " |");
  return [head, rule].concat(lines).join(NL);
}

function blockquote(text: string): string {
  return text
    .split(NL)
    .map((line) => (line ? "> " + line : ">"))
    .join(NL);
}

function renderList(value: unknown): string {
  return nodes(value)
    .map(renderNode)
    .filter((part) => part.length > 0)
    .join(NL + NL);
}

/** Chart chrome (title, caption, unit) lives in the optional meta object. */
function metaOf(node: ElementNode): { title: string; caption: string; unit: string } {
  const props = (node.props ?? {}) as Props;
  const meta = (props.meta ?? {}) as Props;
  return { title: str(meta.title), caption: str(meta.caption), unit: str(meta.unit) };
}

function withMeta(node: ElementNode, body: string): string {
  const meta = metaOf(node);
  const title = meta.title ? "**" + meta.title + "**" : "";
  const caption = meta.caption ? "_" + meta.caption + "_" : "";
  return [title, body, caption].filter(Boolean).join(NL + NL);
}

function unitSuffix(node: ElementNode): string {
  const unit = metaOf(node).unit;
  return unit ? " (" + unit + ")" : "";
}

function renderColumns(columns: ElementNode[]): string {
  const headers = columns.map((column) => {
    const header = str(column.props?.header) || str(column.props?.label);
    const unit = str(column.props?.unit);
    return unit ? header + " (" + unit + ")" : header;
  });
  const data = columns.map((column) =>
    Array.isArray(column.props?.data) ? column.props.data : [],
  );
  const rowCount = data.reduce((max, values) => Math.max(max, values.length), 0);
  const rows: (string | null)[][] = [];
  for (let i = 0; i < rowCount; i += 1) {
    rows.push(data.map((values) => cell(values[i])));
  }
  return markdownTable(headers, rows);
}

function renderTable(node: ElementNode): string {
  const props = (node.props ?? {}) as Props;
  const body = renderColumns(nodes(props.columns));
  return withMeta(node, body);
}

function renderSeriesChart(node: ElementNode): string {
  const props = (node.props ?? {}) as Props;
  const meta = metaOf(node);
  const labels = Array.isArray(props.labels) ? props.labels.map((l) => str(l)) : [];
  const series = nodes(props.series).map((entry) => ({
    name: str(entry.props?.category) || "Value",
    values: Array.isArray(entry.props?.values) ? entry.props.values : [],
  }));
  if (labels.length === 0 || series.length === 0) return "";
  const suffix = meta.unit ? " (" + meta.unit + ")" : "";
  const headers = [str(props.xLabel) || "Category"].concat(series.map((s) => s.name + suffix));
  const rows: (string | null)[][] = labels.map((label, i) => {
    const row: (string | null)[] = [label];
    for (const entry of series) row.push(cell(entry.values[i]));
    return row;
  });
  const axis = [str(props.xLabel), str(props.yLabel)].filter(Boolean).join(" / ");
  const table = markdownTable(headers, rows);
  if (!table) return "";
  const body = axis ? "_Chart (" + axis + ")_" + NL + NL + table : table;
  return withMeta(node, body);
}

function renderSlices(node: ElementNode): string {
  const slices = nodes(node.props?.slices);
  const rows = slices.map((slice) => [str(slice.props?.category), cell(slice.props?.value)]);
  return withMeta(node, markdownTable(["Category", "Value" + unitSuffix(node)], rows));
}

function renderWaterfall(node: ElementNode): string {
  const items = nodes(node.props?.items);
  const rows = items.map((item) => [
    str(item.props?.category),
    cell(item.props?.value),
    item.props?.total ? "total" : "",
  ]);
  return withMeta(node, markdownTable(["Step", "Change" + unitSuffix(node), "Kind"], rows));
}

function renderHierarchy(node: ElementNode, key: string): string {
  const items = nodes(node.props?.[key]);
  const rows = items.map((item) => [
    str(item.props?.name),
    cell(item.props?.value),
    str(item.props?.parent),
  ]);
  return withMeta(
    node,
    markdownTable(["Node", "Value" + unitSuffix(node), "Parent"], rows),
  );
}

function renderFunnel(node: ElementNode): string {
  const stages = nodes(node.props?.stages);
  const rows = stages.map((stage) => [str(stage.props?.name), cell(stage.props?.value)]);
  return withMeta(node, markdownTable(["Stage", "Value" + unitSuffix(node)], rows));
}

function renderSankey(node: ElementNode): string {
  const links = nodes(node.props?.links);
  const rows = links.map((link) => [
    str(link.props?.source),
    str(link.props?.target),
    cell(link.props?.value),
  ]);
  return withMeta(
    node,
    markdownTable(["From", "To", "Value" + unitSuffix(node)], rows),
  );
}

function renderGauge(node: ElementNode): string {
  const props = (node.props ?? {}) as Props;
  const label = str(props.label) || "Value";
  return withMeta(node, markdownTable(["Metric", "Value" + unitSuffix(node)], [[label, cell(props.value)]]));
}

function renderHeatmap(node: ElementNode): string {
  const props = (node.props ?? {}) as Props;
  const cells = nodes(props.cells);
  const headers = [str(props.xLabel) || "Column", str(props.yLabel) || "Row", "Value" + unitSuffix(node)];
  const rows = cells.map((entry) => [
    str(entry.props?.x),
    str(entry.props?.y),
    cell(entry.props?.value),
  ]);
  return withMeta(node, markdownTable(headers, rows));
}

function renderMap(node: ElementNode): string {
  const props = (node.props ?? {}) as Props;
  const regions = nodes(props.regions);
  const points = nodes(props.points);
  const regionTable = markdownTable(
    ["Country", "Value" + unitSuffix(node)],
    regions.map((region) => [
      str(region.props?.label) || str(region.props?.name),
      cell(region.props?.value),
    ]),
  );
  const pointTable = markdownTable(
    ["Site", "Latitude", "Longitude", "Value"],
    points.map((point) => [
      str(point.props?.label) || "Site",
      cell(point.props?.lat),
      cell(point.props?.lng),
      cell(point.props?.value),
    ]),
  );
  return withMeta(node, [regionTable, pointTable].filter(Boolean).join(NL + NL));
}

function renderEntityList(props: Props): string {
  const rows = Array.isArray(props.rows) ? props.rows : [];
  const body = rows
    .filter((row): row is Record<string, unknown> => !!row && typeof row === "object")
    .map((row) => [str(row.left), str(row.right)]);
  const header =
    props.header && typeof props.header === "object"
      ? (props.header as Record<string, unknown>)
      : null;
  const headers = header ? [str(header.left) || "Item", str(header.right) || "Value"] : ["Item", "Value"];
  return markdownTable(headers, body);
}

const TEXT_KEYS = [
  "text",
  "textMarkdown",
  "title",
  "heading",
  "label",
  "description",
  "subtitle",
  "details",
  "summary",
  "caption",
  "alt",
  "message",
];

function renderGeneric(node: ElementNode): string {
  const props = (node.props ?? {}) as Props;
  const parts: string[] = [];
  for (const key of TEXT_KEYS) {
    const value = str(props[key]);
    if (value) parts.push(value);
  }
  for (const key of Object.keys(props)) {
    if (TEXT_KEYS.indexOf(key) >= 0) continue;
    if (key === "meta") continue;
    for (const child of nodes(props[key])) {
      const text = renderNode(child);
      if (text) parts.push(text);
    }
  }
  return parts.join(NL + NL);
}

/** Inline citation-style components: show the marker and the detail. */
function renderPaired(node: ElementNode): string {
  const props = (node.props ?? {}) as Props;
  const trigger = renderList(props.trigger);
  const detail = str(props.text) || renderList(props.content);
  return [trigger, detail].filter(Boolean).join(" - ");
}

function renderNode(node: ElementNode): string {
  const props = (node.props ?? {}) as Props;
  switch (node.typeName) {
    case "Card":
    case "Carousel":
      return renderList(props.children);
    case "CardHeader": {
      const title = str(props.title);
      const subtitle = str(props.subtitle);
      return [title ? "# " + title : "", subtitle].filter(Boolean).join(NL + NL);
    }
    case "InlineHeader": {
      const heading = str(props.heading) || str(props.title);
      const description = str(props.description);
      return [heading ? "### " + heading : "", description].filter(Boolean).join(NL + NL);
    }
    case "TextContent":
      return str(props.text);
    case "MarkDownRenderer":
      return str(props.textMarkdown);
    case "Callout":
    case "TextCallout": {
      const title = str(props.title);
      const description = str(props.description);
      const inner = [title ? "**" + title + "**" : "", description].filter(Boolean).join(NL + NL);
      return inner ? blockquote(inner) : "";
    }
    case "Table":
      return renderTable(node);
    case "BarChart":
    case "LineChart":
    case "AreaChart":
    case "HorizontalBarChart":
    case "ScatterChart":
      return renderSeriesChart(node);
    case "PieChart":
    case "RadialChart":
    case "SingleStackedBarChart":
      return renderSlices(node);
    case "WaterfallChart":
      return renderWaterfall(node);
    case "SunburstChart":
      return renderHierarchy(node, "nodes");
    case "TreemapChart":
      return renderHierarchy(node, "nodes");
    case "FunnelChart":
      return renderFunnel(node);
    case "SankeyChart":
      return renderSankey(node);
    case "GaugeChart":
      return renderGauge(node);
    case "HeatmapChart":
      return renderHeatmap(node);
    case "MapChart":
      return renderMap(node);
    case "HoverCard":
    case "Tooltip":
      return renderPaired(node);
    case "Popover": {
      const trigger = renderList(props.trigger);
      const content = renderList(props.content);
      return [trigger, content].filter(Boolean).join(NL + NL);
    }
    case "Collapsible": {
      const heading = renderList(props.trigger);
      const content = renderList(props.content);
      return [heading ? "## " + heading : "", content].filter(Boolean).join(NL + NL);
    }
    case "Sheet": {
      const title = str(props.title);
      const description = str(props.description);
      const content = renderList(props.content);
      return [title ? "### " + title : "", description, content].filter(Boolean).join(NL + NL);
    }
    case "Resizable":
      return renderList(props.panels);
    case "ResizablePanel":
    case "ScrollArea":
    case "AspectRatio":
      return renderList(props.content);
    case "ToggleGroup": {
      const items = Array.isArray(props.items) ? props.items.map((item) => str(item)).filter(Boolean) : [];
      return items.length ? "**" + items.join(" | ") + "**" : "";
    }
    case "Breadcrumb":
      return nodes(props.items)
        .map((item) => str(item.props?.label))
        .filter(Boolean)
        .join(" > ");
    case "Sonner":
      return str(props.message);
    case "Spinner":
      return str(props.label);
    case "Empty": {
      const title = str(props.title);
      const description = str(props.description);
      return [title ? "**" + title + "**" : "", description].filter(Boolean).join(NL + NL);
    }
    case "Field": {
      const label = str(props.label);
      const control = renderNode(props.control as ElementNode);
      const description = str(props.description);
      return [label ? "**" + label + "**" : "", control, description].filter(Boolean).join(NL + NL);
    }
    case "ListBlock": {
      return nodes(props.items)
        .map((item) => {
          const title = str(item.props?.title);
          const subtitle = str(item.props?.subtitle);
          const line = [title, subtitle].filter(Boolean).join(" - ");
          return line ? "- " + line : "";
        })
        .filter(Boolean)
        .join(NL);
    }
    case "FollowUpBlock": {
      const items = nodes(props.items)
        .map((item) => str(item.props?.text))
        .filter(Boolean);
      return items.length
        ? "## Suggested follow-ups" + NL + NL + items.map((t) => "- " + t).join(NL)
        : "";
    }
    case "Steps":
      return nodes(props.items)
        .map((item, index) => {
          const line = [str(item.props?.title), str(item.props?.details)]
            .filter(Boolean)
            .join(" - ");
          return line ? String(index + 1) + ". " + line : "";
        })
        .filter(Boolean)
        .join(NL);
    case "SectionBlock":
      return renderList(props.sections);
    case "SectionItem":
    case "TabItem":
    case "AccordionItem": {
      const heading = str(props.trigger) || str(props.value);
      const content = renderList(props.content);
      return [heading ? "## " + heading : "", content].filter(Boolean).join(NL + NL);
    }
    case "Tabs":
    case "Accordion":
      return renderList(props.items);
    case "TagBlock": {
      const tags = Array.isArray(props.tags)
        ? props.tags.map((tag) => (typeof tag === "string" ? tag : str((tag as ElementNode)?.props?.text))).filter(Boolean)
        : [];
      return tags.length ? "**" + tags.join(" | ") + "**" : "";
    }
    case "Tag":
      return str(props.text);
    case "EntityList":
      return renderEntityList(props);
    case "Image": {
      const src = str(props.src);
      return src ? "![" + (str(props.alt) || "image") + "](" + src + ")" : "";
    }
    case "Skeleton":
    case "Separator":
      return "";
    default:
      return renderGeneric(node);
  }
}

export function openuiToMarkdown(
  root: ElementNode | null,
  sources: MarkdownSource[] = [],
): string {
  // The reference list is rendered once, below the answer, by
  // AnswerReferences (from the verified source list). The derived markdown is
  // body-only so the Text view cannot show the sources twice.
  void sources;
  return (root ? renderNode(root) : "").trim();
}
