"use client";

import type { ComponentGroup, PromptOptions } from "@openuidev/react-lang";
import { createLibrary, defineComponent } from "@openuidev/react-lang";
import { z } from "zod";

// Content
import { Alert } from "./components/alert";
import { Avatar } from "./components/avatar";
import { ShadcnBadgeComponent } from "./components/badge";
import { CardHeader } from "./components/card-header";
import { CodeBlock } from "./components/code-block";
import { Image, ImageBlock } from "./components/image";
import { MarkDownRenderer } from "./components/markdown-renderer";
import { Progress } from "./components/progress";
import { Separator } from "./components/separator";
import { TextContent } from "./components/text-content";

// Charts
import {
  AreaChartCondensed,
  BarChartCondensed,
  LineChartCondensed,
  PieChartComponent,
  Point,
  RadarChartComponent,
  RadialChartComponent,
  ScatterChartComponent,
  ScatterSeries,
  Series,
  Slice,
} from "./components/charts";

// Forms
import { CheckBoxGroup, CheckBoxItem } from "./components/checkbox-group";
import { DatePicker } from "./components/date-picker";
import { Form } from "./components/form";
import { FormControl } from "./components/form-control";
import { Input } from "./components/input";
import { Label } from "./components/label";
import { RadioGroup, RadioItem } from "./components/radio-group";
import { Select, SelectItem } from "./components/select";
import { Slider } from "./components/slider";
import { SwitchGroup, SwitchItem } from "./components/switch-group";
import { TextArea } from "./components/textarea";

// Buttons
import { Button } from "./components/button";
import { Buttons } from "./components/buttons";

// Layout
import { Accordion, AccordionItemDef } from "./components/accordion";
import { Carousel } from "./components/carousel";
import { TabItem, Tabs } from "./components/tabs";

// Data Display
import { Col, Table } from "./components/table";
import { Tag, TagBlock } from "./components/tag";

// Chat-specific
import { FollowUpBlock, FollowUpItem } from "./components/follow-up-block";

// Advanced charts & maps
import {
  FlowLink,
  FlowNode,
  FunnelChart,
  FunnelStage,
  GaugeChart,
  HeatCell,
  HeatmapChart,
  SankeyChart,
  SunburstChart,
  SunburstNode,
  TreemapChart,
  TreemapNode,
  WaterfallChart,
  WaterfallItem,
} from "./components/advanced-charts";
import { MapChart, MapPoint, MapRegion } from "./components/map-chart";

// New components
import { AlertDialogBlock } from "./components/alert-dialog-block";
import { CalendarBlock } from "./components/calendar-block";
import { DialogBlock } from "./components/dialog-block";
import { DrawerBlock } from "./components/drawer-block";
import { PaginationBlock } from "./components/pagination-block";
import { Blockquote, Heading, InlineCode } from "./components/typography";

// Evidence search, citations, disclosure & feedback
import { AspectRatio } from "./components/aspect-ratio";
import { Breadcrumb, BreadcrumbItem } from "./components/breadcrumb";
import { ButtonGroup } from "./components/button-group";
import { Collapsible } from "./components/collapsible";
import { Combobox, ComboboxItem } from "./components/combobox";
import { Command, CommandItem } from "./components/command";
import { ContextMenu, ContextMenuItem } from "./components/context-menu";
import { DropdownMenu, DropdownMenuItem } from "./components/dropdown-menu";
import { Empty } from "./components/empty";
import { Field } from "./components/field";
import { HoverCard } from "./components/hover-card";
import { InputGroup } from "./components/input-group";
import { Kbd } from "./components/kbd";
import { Menubar, MenubarItem, MenubarMenu } from "./components/menubar";
import { NavigationMenu, NavigationMenuItem } from "./components/navigation-menu";
import { Popover } from "./components/popover";
import { Resizable, ResizablePanel } from "./components/resizable";
import { ScrollArea } from "./components/scroll-area";
import { Sheet } from "./components/sheet";
import { Skeleton } from "./components/skeleton";
import { Sonner } from "./components/sonner";
import { Spinner } from "./components/spinner";
import { ToggleGroup } from "./components/toggle-group";
import { Tooltip } from "./components/tooltip";

import { CardSourceProvider } from "@openuidev/react-ui";
import { useMedRagSources } from "@/openui/source-context";

import { shadcnPromptOptions } from "./prompt-options.mjs";
import { ChatContentChildUnion } from "./unions";

export { shadcnPromptOptions };

const ChatCardChildUnion = z.union([...ChatContentChildUnion.options, Tabs.ref, Carousel.ref]);

const ChatCard = defineComponent({
  name: "Card",
  props: z.object({
    children: z.array(ChatCardChildUnion),
  }),
  description:
    "Vertical container for all content in a chat response. Children stack top to bottom automatically.",
  // MedRAG: the root Card is the only place the verified ledger enters the
  // program. Sources come from the app, never from the model, so a
  // hallucinated reference cannot render.
  component: ({ props, renderNode }) => {
    const sources = useMedRagSources();
    // The answer renders directly on the page background: no Card frame, so
    // charts and prose are not boxed into a second container.
    return (
      <CardSourceProvider sources={sources}>
        <div className="space-y-4">{renderNode(props.children)}</div>
      </CardSourceProvider>
    );
  },
});

// ── Component Groups ──

export const shadcnComponentGroups: ComponentGroup[] = [
  {
    name: "Content",
    components: [
      "CardHeader",
      "TextContent",
      "MarkDownRenderer",
      "Alert",
      "Badge",
      "Avatar",
      "CodeBlock",
      "Image",
      "ImageBlock",
      "Progress",
      "Separator",
    ],
  },
  {
    name: "Tables",
    components: ["Table", "Col"],
  },
  {
    name: "Charts (2D)",
    components: ["BarChart", "LineChart", "AreaChart", "RadarChart", "Series"],
  },
  {
    name: "Charts (1D)",
    components: ["PieChart", "RadialChart", "Slice"],
  },
  {
    name: "Charts (Scatter)",
    components: ["ScatterChart", "ScatterSeries", "Point"],
  },
  {
    name: "Charts (Advanced)",
    components: [
      "WaterfallChart",
      "WaterfallItem",
      "SunburstChart",
      "SunburstNode",
      "TreemapChart",
      "TreemapNode",
      "FunnelChart",
      "FunnelStage",
      "SankeyChart",
      "FlowNode",
      "FlowLink",
      "GaugeChart",
      "HeatmapChart",
      "HeatCell",
    ],
    notes: [
      "Pick the chart by the data's shape, not by habit:",
      "- Change to a running total -> WaterfallChart.",
      "- Nested parts of a whole (2-4 levels) -> SunburstChart; a flat split -> PieChart.",
      "- Many categories or two hierarchy levels -> TreemapChart; pair with a Table for exact values.",
      "- Ordered attrition or screening stages -> FunnelChart.",
      "- Movement between stages or groups -> SankeyChart.",
      "- One KPI against a maximum -> GaugeChart.",
      "- A value at every row x column combination -> HeatmapChart.",
    ],
  },
  {
    name: "Maps",
    components: ["MapChart", "MapPoint", "MapRegion"],
    notes: [
      "- Use MapChart for geographic distribution: MapRegion for country-level shading, MapPoint for study sites.",
      "- Every region value and point value must come from a cited passage; never invent coordinates or rates.",
      "- Country names use common English (United States, China, Brazil); points use decimal lat/lng.",
    ],
  },
  {
    name: "Forms",
    components: [
      "Form",
      "FormControl",
      "Label",
      "Input",
      "TextArea",
      "Select",
      "SelectItem",
      "DatePicker",
      "Slider",
      "CheckBoxGroup",
      "CheckBoxItem",
      "RadioGroup",
      "RadioItem",
      "SwitchGroup",
      "SwitchItem",
    ],
    notes: [
      "- Define EACH FormControl as its own reference — do NOT inline all controls in one array.",
      "- NEVER nest Form inside Form.",
      "- Form requires explicit buttons. Always pass a Buttons(...) reference as the third Form argument.",
      "- rules is an optional object: { required: true, email: true, min: 8, maxLength: 100 }",
      "- The renderer shows error messages automatically — do NOT generate error text in the UI",
    ],
  },
  {
    name: "Buttons",
    components: ["Button", "Buttons"],
  },
  {
    name: "Follow-ups",
    components: ["FollowUpBlock", "FollowUpItem"],
    notes: [
      "- Use FollowUpBlock with FollowUpItem references at the end of a response to suggest next actions.",
      "- Clicking a FollowUpItem sends its text to the LLM as a user message.",
    ],
  },
  {
    name: "Layout",
    components: ["Tabs", "TabItem", "Accordion", "AccordionItem", "Carousel"],
    notes: [
      "- Use Tabs to present alternative views — each TabItem has a value id, trigger label, and content array.",
      "- Carousel takes an array of slides, where each slide is an array of content.",
      "- IMPORTANT: Every slide in a Carousel must have the same structure.",
    ],
  },
  {
    name: "Data Display",
    components: ["TagBlock", "Tag"],
  },
  {
    name: "Typography",
    components: ["Heading", "Blockquote", "InlineCode"],
    notes: [
      '- Heading levels: "h1" | "h2" | "h3" | "h4". Each renders with appropriate shadcn/ui typography styles.',
      "- Blockquote for styled quotes with optional cite attribution.",
      "- InlineCode for monospace code snippets within text.",
    ],
  },
  {
    name: "Calendar",
    components: ["CalendarBlock"],
    notes: [
      '- CalendarBlock renders a standalone interactive calendar. mode: "single" | "multiple" | "range".',
      "- Use numberOfMonths to show multiple months side by side.",
      "- Use defaultMonth (ISO date string) to set the initial visible month.",
    ],
  },
  {
    name: "Navigation",
    components: ["PaginationBlock"],
    notes: ["- PaginationBlock takes currentPage and totalPages."],
  },
  {
    name: "Overlays",
    components: ["DialogBlock", "AlertDialogBlock", "DrawerBlock"],
    notes: [
      "- DialogBlock renders a button that opens a modal dialog with content inside.",
      "- AlertDialogBlock renders a confirmation dialog with cancel/confirm actions.",
      "- DrawerBlock renders a bottom drawer panel triggered by a button.",
    ],
  },
  {
    name: "Evidence Search",
    components: ["Command", "CommandItem", "Combobox", "ComboboxItem", "ToggleGroup"],
    notes: [
      "- Evidence search MUST be a Command palette (Cmd+K), never a prose instruction or a bare Input.",
      '- Use ToggleGroup to filter by study design ("RCT" / "Cohort" / "Case-control"), never write "you can filter by..." in text.',
      "- Use Combobox to pick a population, design, or intervention from a searchable list.",
    ],
  },
  {
    name: "Citations & Terms",
    components: ["HoverCard", "Tooltip"],
    notes: [
      "- Cite inline as plain [n]; the application renders each as a clickable citation badge (author-year by default) and lists the references at the bottom. Never wrap a marker in HoverCard/Badge, add a citations column, or repeat the full citation inline.",
      "- Wrap technical terms (GRADE certainty, RR) in a Tooltip with a one-line definition; never parenthesize the definition in prose.",
    ],
  },
  {
    name: "Details & Disclosure",
    components: [
      "Popover",
      "Collapsible",
      "Sheet",
      "Resizable",
      "ResizablePanel",
      "ScrollArea",
    ],
    notes: [
      "- Single-point detail goes in a Popover, never a full DialogBlock.",
      "- Nested evidence such as subgroup analyses goes in a Collapsible.",
      "- Full-text context goes in a Sheet (side panel) so the answer stays visible.",
      "- Side-by-side evidence goes in Resizable; never stack two tables meant to be compared.",
      "- Long evidence lists go in ScrollArea; never let the Card grow unbounded.",
    ],
  },
  {
    name: "Feedback & Loading",
    components: ["Sonner", "Skeleton", "Spinner", "Empty"],
    notes: [
      "- Streaming placeholders MUST be Skeleton sized like the eventual table/chart; never leave a blank region.",
      "- Transient confirmations use Sonner, never a permanent Alert.",
      '- When a filter returns zero studies, use Empty (e.g. "No RCTs match this population").',
    ],
  },
  {
    name: "Research Navigation",
    components: [
      "Breadcrumb",
      "BreadcrumbItem",
      "NavigationMenu",
      "NavigationMenuItem",
      "DropdownMenu",
      "DropdownMenuItem",
      "ContextMenu",
      "ContextMenuItem",
      "Menubar",
      "MenubarMenu",
      "MenubarItem",
      "Kbd",
    ],
    notes: [
      "- Breadcrumb shows the research trail: Question > Filtered evidence (12) > Included studies (5) > Synthesis.",
      "- DropdownMenu holds per-study row actions (Show abstract, Open in PubMed, Exclude).",
      "- ContextMenu holds right-click passage actions (Copy citation, Highlight, Compare).",
      "- Kbd shows shortcuts such as Cmd+K.",
    ],
  },
  {
    name: "Utility",
    components: ["AspectRatio", "ButtonGroup", "InputGroup", "Field"],
    notes: [
      "- AspectRatio constrains image thumbnails and forest-plot previews.",
      '- ButtonGroup groups related actions, e.g. "Include / Exclude / Maybe" as one control.',
      "- InputGroup is an input with addons (leading label / trailing hint).",
      "- Field is a generic field wrapper when FormControl is too opinionated.",
    ],
  },
];


// ── Library ──

export const shadcnChatLibrary = createLibrary({
  root: "Card",
  componentGroups: shadcnComponentGroups,
  components: [
    // Root
    ChatCard,
    CardHeader,
    // Content
    TextContent,
    MarkDownRenderer,
    Alert,
    ShadcnBadgeComponent,
    Avatar,
    CodeBlock,
    Image,
    ImageBlock,
    Progress,
    Separator,
    // Tables
    Table,
    Col,
    // Charts (2D)
    BarChartCondensed,
    LineChartCondensed,
    AreaChartCondensed,
    RadarChartComponent,
    Series,
    // Charts (1D)
    PieChartComponent,
    RadialChartComponent,
    Slice,
    // Charts (Scatter)
    ScatterChartComponent,
    ScatterSeries,
    Point,
    // Charts (Advanced)
    WaterfallChart,
    WaterfallItem,
    SunburstChart,
    SunburstNode,
    TreemapChart,
    TreemapNode,
    FunnelChart,
    FunnelStage,
    SankeyChart,
    FlowNode,
    FlowLink,
    GaugeChart,
    HeatmapChart,
    HeatCell,
    // Maps
    MapChart,
    MapPoint,
    MapRegion,
    // Forms
    Form,
    FormControl,
    Label,
    Input,
    TextArea,
    Select,
    SelectItem,
    DatePicker,
    Slider,
    CheckBoxGroup,
    CheckBoxItem,
    RadioGroup,
    RadioItem,
    SwitchGroup,
    SwitchItem,
    // Buttons
    Button,
    Buttons,
    // Follow-ups
    FollowUpBlock,
    FollowUpItem,
    // Layout
    Tabs,
    TabItem,
    Accordion,
    AccordionItemDef,
    Carousel,
    // Data Display
    TagBlock,
    Tag,
    // Typography
    Heading,
    Blockquote,
    InlineCode,
    // Navigation
    PaginationBlock,
    // Overlays
    DialogBlock,
    AlertDialogBlock,
    DrawerBlock,
    // Calendar
    CalendarBlock,
    // Evidence search
    Command,
    CommandItem,
    Combobox,
    ComboboxItem,
    ToggleGroup,
    // Citations & terms
    HoverCard,
    Tooltip,
    // Details & disclosure
    Popover,
    Collapsible,
    Sheet,
    Resizable,
    ResizablePanel,
    ScrollArea,
    // Feedback & loading
    Sonner,
    Skeleton,
    Spinner,
    Empty,
    Field,
    // Navigation
    Breadcrumb,
    BreadcrumbItem,
    NavigationMenu,
    NavigationMenuItem,
    DropdownMenu,
    DropdownMenuItem,
    ContextMenu,
    ContextMenuItem,
    Menubar,
    MenubarMenu,
    MenubarItem,
    Kbd,
    // Utility
    AspectRatio,
    ButtonGroup,
    InputGroup,
  ],
});
