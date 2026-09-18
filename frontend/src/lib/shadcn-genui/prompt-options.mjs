/**
 * shadcn GenUI prompt options: the worked examples and rules the model-facing
 * system prompt is built from. Kept as plain data (.mjs) so the Node prompt
 * generator can import it alongside the renderer entry.
 *
 * Layout:
 *   examples         — core rendering (1–12) + evidence-interaction (13–17)
 *   additionalRules  — grouped: structure, variants, selection, evidence
 *                      patterns, pre-flight
 */

// ── Examples: core rendering ──

export const shadcnExamples = [
  `Example 1 — Table with follow-ups:
root = Card([title, tbl, followUps])
title = TextContent("Top Languages", "large-heavy")
tbl = Table(cols, rows)
cols = [Col("Language", "string"), Col("Users (M)", "number"), Col("Year", "number")]
rows = [["Python", 15.7, 1991], ["JavaScript", 14.2, 1995], ["Java", 12.1, 1995]]
followUps = FollowUpBlock([fu1, fu2])
fu1 = FollowUpItem("Tell me more about Python")
fu2 = FollowUpItem("Show me a JavaScript comparison")`,

  `Example 2 — Form with validation:
root = Card([title, form])
title = TextContent("Contact Us", "large-heavy")
form = Form("contact", btns, [nameField, emailField, msgField])
nameField = FormControl("Name", Input("name", "Your name", "text", { required: true, minLength: 2 }))
emailField = FormControl("Email", Input("email", "you@example.com", "email", { required: true, email: true }))
msgField = FormControl("Message", TextArea("message", "Tell us more...", 4, { required: true, minLength: 10 }))
btns = Buttons([Button("Submit", { type: "continue_conversation" }, "default")])`,

  `Example 3 — Alert variants:
root = Card([info, success, warning, danger])
info = Alert("Update available", "A new version is available for download.", "info")
success = Alert("Payment confirmed", "Your transaction was successful.", "success")
warning = Alert("Disk almost full", "You have less than 10% storage remaining.", "warning")
danger = Alert("Account suspended", "Please contact support immediately.", "destructive")`,

  `Example 4 — Bar chart with badges:
root = Card([header, badges, chart, followUps])
header = CardHeader("Monthly Revenue", "Q4 2024 performance across regions")
badges = TagBlock([Tag("Live data", "default"), Tag("USD", "secondary"), Tag("Grouped", "outline")])
chart = BarChart(["Oct", "Nov", "Dec"], [s1, s2], "grouped", "Month", "Revenue ($K)")
s1 = Series("North America", [420, 380, 510])
s2 = Series("Europe", [310, 290, 340])
followUps = FollowUpBlock([FollowUpItem("Show as line chart"), FollowUpItem("Add Asia-Pacific")])`,

  `Example 5 — Buttons with all variants:
root = Card([title, btns])
title = TextContent("Button Styles", "large-heavy")
btns = Buttons([b1, b2, b3, b4, b5, b6])
b1 = Button("Default", { type: "continue_conversation" }, "default")
b2 = Button("Secondary", { type: "continue_conversation" }, "secondary")
b3 = Button("Outline", { type: "continue_conversation" }, "outline")
b4 = Button("Ghost", { type: "continue_conversation" }, "ghost")
b5 = Button("Link", { type: "continue_conversation" }, "link")
b6 = Button("Destructive", { type: "continue_conversation" }, "destructive")`,

  `Example 6 — Tabs with charts:
root = Card([header, tabs])
header = CardHeader("Sales Dashboard", "Compare metrics across time periods")
tabs = Tabs([tab1, tab2, tab3])
tab1 = TabItem("revenue", "Revenue", [revChart])
tab2 = TabItem("users", "Users", [usersChart])
tab3 = TabItem("breakdown", "Breakdown", [pieChart])
revChart = BarChart(["Jan", "Feb", "Mar", "Apr"], [Series("Revenue", [45, 52, 61, 58])], "grouped", "Month", "USD ($K)")
usersChart = LineChart(["Jan", "Feb", "Mar", "Apr"], [Series("Active", [1200, 1350, 1500, 1420]), Series("New", [300, 420, 380, 450])], "Month", "Users")
pieChart = PieChart([Slice("Desktop", 62), Slice("Mobile", 31), Slice("Tablet", 7)])`,

  `Example 7 — Typography showcase:
root = Card([h1, h2, h3, quote, codeEx, sep, text])
h1 = Heading("Welcome to the Platform", "h1")
h2 = Heading("Getting Started", "h2")
h3 = Heading("Prerequisites", "h3")
quote = Blockquote("The best way to predict the future is to invent it.", "Alan Kay")
codeEx = InlineCode("npm install @acme/sdk")
sep = Separator()
text = TextContent("Follow the steps below to get up and running.")`,

  `Example 8 — Dialog and AlertDialog:
root = Card([title, btns])
title = TextContent("Actions Demo", "large-heavy")
btns = Buttons([viewBtn, deleteBtn])
viewBtn = DialogBlock("View Details", "Product Details", "Full specifications for Widget Pro", [detailText, detailTable], "outline")
detailText = TextContent("Here are the complete specifications:")
detailTable = Table([Col("Spec", "string"), Col("Value", "string")], [["Weight", "2.5 kg"], ["Dimensions", "30x20x10 cm"]])
deleteBtn = AlertDialogBlock("Delete Item", "Are you sure?", "This action cannot be undone. This will permanently delete the item.", "Delete", "Cancel", "destructive")`,

  `Example 9 — Pagination:
root = Card([title, table, pagination])
title = TextContent("Search Results", "large-heavy")
table = Table([Col("Name", "string"), Col("Status", "string")], [["Item 1", "Active"], ["Item 2", "Pending"], ["Item 3", "Active"]])
pagination = PaginationBlock(2, 10)`,

  `Example 10 — Drawer with content:
root = Card([title, drawerBtn])
title = TextContent("Report Summary", "large-heavy")
drawerBtn = DrawerBlock("View Full Report", "Quarterly Report Q4 2024", "Detailed breakdown of performance metrics", [chart, summary])
chart = BarChart(["Oct", "Nov", "Dec"], [Series("Revenue", [42, 38, 51])], "grouped", "Month", "Revenue ($K)")
summary = TextContent("Overall revenue increased by 12% compared to Q3.")`,

  `Example 11 — Standalone calendar:
root = Card([title, cal])
title = TextContent("Pick a Date", "large-heavy")
cal = CalendarBlock("single", "2025-01-01", 1)`,

  `Example 12 — Range calendar with two months:
root = Card([title, desc, cal])
title = TextContent("Select Travel Dates", "large-heavy")
desc = TextContent("Choose your check-in and check-out dates.", "small")
cal = CalendarBlock("range", "2025-06-01", 2)`,

  // ── Examples: evidence interaction ──

  `Example 13 — Inline citations (the app renders the badges):
root = Card([title, para, followUps])
title = TextContent("Aspirin for primary prevention", "large-heavy")
para = MarkDownRenderer("Low-dose aspirin reduced non-fatal MI [1] but increased major bleeding [2]. The application turns each [n] into a clickable badge and lists the references at the bottom.")
followUps = FollowUpBlock([FollowUpItem("Show forest plot"), FollowUpItem("Filter to RCTs only")])`,

  `Example 14 — Tooltip term and Popover data point:
root = Card([title, term, point])
title = TextContent("Heterogeneity and certainty", "large-heavy")
term = Tooltip(termTrigger, "How confident we are that the true effect lies close to the estimate.")
termTrigger = TextContent("GRADE certainty", "small-heavy")
point = Popover(pointTrigger, [pointBody])
pointTrigger = TextContent("Major bleeding RR 1.54", "small-heavy")
pointBody = TextContent("ATT Collaboration 2009 — 95% CI 1.30–1.82 — n=95,000", "small")`,

  `Example 15 — Command evidence search and study-design filter:
root = Card([title, search, designFilter])
title = TextContent("Search the evidence", "large-heavy")
search = Command("Search 128 passages...", [r1, r2], "No passages match your query.", true)
r1 = CommandItem("Aspirin and major bleeding", "ATT 2009 — RCT meta-analysis", "bleeding hemorrhage", { type: "continue_conversation" })
r2 = CommandItem("Aspirin and myocardial infarction", "ATT 2009 — RCT meta-analysis", "MI infarction", { type: "continue_conversation" })
designFilter = ToggleGroup(["RCT", "Cohort", "Case-control"], true, "RCT")`,

  `Example 16 — Resizable evidence comparison:
root = Card([title, compare])
title = TextContent("Conflicting estimates", "large-heavy")
compare = Resizable([left, right], "horizontal", 50)
left = ResizablePanel([leftTitle, leftTable])
leftTitle = TextContent("Trial A", "small-heavy")
leftTable = Table([Col("Outcome", "string"), Col("RR", "number")], [["MI", 0.88], ["Bleeding", 1.54]])
right = ResizablePanel([rightTitle, rightTable])
rightTitle = TextContent("Trial B", "small-heavy")
rightTable = Table([Col("Outcome", "string"), Col("RR", "number")], [["MI", 0.94], ["Bleeding", 1.32]])`,

  `Example 17 — Skeleton, Empty and Sonner:
root = Card([title, loading, none, notify])
title = TextContent("Loading evidence", "large-heavy")
loading = Skeleton("100%", 120, "md")
none = Empty("No RCTs match this population", "Try widening the population filter.")
notify = Sonner("Citation copied to clipboard", "success", "Copy citation")`,

  `Example 18 — Informative bar chart with title, units and legend:
root = Card([heading, chart, followUps])
heading = CardHeader("Major bleeding by aspirin dose", "Pooled estimates from the ATT Collaboration (n=95,000)")
chart = BarChart(["Low dose", "High dose"], [gi, ic], "grouped", "Aspirin dose", "Events", { title: "Major bleeding events per 1000 patient-years", caption: "ATT Collaboration 2009, RCT meta-analysis", unit: "per 1000 patient-years", legend: true, showValues: true })
gi = Series("Gastrointestinal", [3.2, 5.1])
ic = Series("Intracranial", [0.4, 0.9])
followUps = FollowUpBlock([FollowUpItem("Show the underlying table"), FollowUpItem("Compare bleeding definitions")])`,

  `Example 19 — Waterfall of additive risk:
root = Card([heading, wf])
heading = CardHeader("Where the bleeding risk comes from", "Additive decomposition of the pooled risk difference")
wf = WaterfallChart([w1, w2, w3, wTotal], "Component", "Risk difference", { title: "Contribution to absolute bleeding risk", caption: "ATT Collaboration 2009", unit: "events per 1000 patient-years", legend: true })
w1 = WaterfallItem("Baseline risk", 4.0)
w2 = WaterfallItem("Age over 65", 2.1)
w3 = WaterfallItem("Prior GI bleed", 3.4)
wTotal = WaterfallItem("Total", 0, true)`,

  `Example 20 — Geographic map of trial sites and enrolment:
root = Card([heading, map])
heading = CardHeader("Where the trials were run", "Recruitment sites and country enrolment")
map = MapChart([p1, p2], [r1, r2], "world", { title: "Trial sites and country enrolment", caption: "41 RCTs, 2020-2024", unit: "participants", legend: true, roam: true })
p1 = MapPoint(51.5, -0.12, "London", 1240)
p2 = MapPoint(40.71, -74.0, "New York", 980)
r1 = MapRegion("United Kingdom", 1240)
r2 = MapRegion("United States", 4310)`,

  `Example 21 — Screening funnel with a unit-bearing table:
root = Card([heading, funnel, table])
heading = CardHeader("From search to synthesis", "PRISMA-style selection of the evidence base")
funnel = FunnelChart([f1, f2, f3, f4], { title: "Study selection", unit: "records", legend: true })
f1 = FunnelStage("Records screened", 4820)
f2 = FunnelStage("Full texts assessed", 312)
f3 = FunnelStage("Eligible", 74)
f4 = FunnelStage("Included", 41)
table = Table([c1, c2, c3], [["Aspirin", 21, 0.88], ["Placebo", 20, 1.0]], "Included studies by arm", "Counts and pooled effects from the 41 included RCTs")
c1 = Col("Arm", "string")
c2 = Col("Studies", "number", "n")
c3 = Col("MI risk ratio", "number", "RR")`,

  `Example 22 — Sunburst of studies by outcome and design:
root = Card([heading, sunburst])
heading = CardHeader("Evidence by outcome and design", "Nested count of included studies")
sunburst = SunburstChart([s1, s2, s3, s4, s5], { title: "Studies by outcome (inner ring) and design (outer ring)", unit: "studies" })
s1 = SunburstNode("Mortality", 18)
s2 = SunburstNode("Mortality - RCT", 12, "Mortality")
s3 = SunburstNode("Mortality - Cohort", 6, "Mortality")
s4 = SunburstNode("Bleeding", 23)
s5 = SunburstNode("Bleeding - RCT", 15, "Bleeding")`,

  `Example 23 — Country ranking with a map and a ranked chart:
root = Card([heading, bar, map, table])
heading = CardHeader("Countries with the highest hypertension prevalence", "Age-standardised adult prevalence, latest comparable year")
bar = BarChart(["Nigeria", "Ghana", "Kenya", "South Africa"], [prev], "grouped", "Country", "Prevalence", { title: "Highest age-standardised prevalence", caption: "WHO 2023", unit: "% of adults", legend: false, showValues: true })
prev = Series("Prevalence", [30.4, 27.1, 24.5, 22.8])
map = MapChart([], [m1, m2, m3, m4], "world", { title: "Prevalence by country", caption: "WHO 2023", unit: "% of adults", legend: true })
m1 = MapRegion("Nigeria", 30.4)
m2 = MapRegion("Ghana", 27.1)
m3 = MapRegion("Kenya", 24.5)
m4 = MapRegion("South Africa", 22.8)
table = Table([c1, c2], [["Nigeria", 30.4], ["Ghana", 27.1], ["Kenya", 24.5], ["South Africa", 22.8]], "Age-standardised adult prevalence", "WHO 2023; citations are carried in the prose")
c1 = Col("Country", "string")
c2 = Col("Prevalence", "number", "% of adults")`,
];

// ── Rules: structure & layout ──

export const shadcnRulesStructure = [
  "Every response is a single Card(children) — children stack vertically automatically.",
  "Card is the only layout container. Do NOT use Stack. Use Tabs to switch between sections, Carousel for horizontal scroll.",
  "Use FollowUpBlock at the END of a Card to suggest what the user can do or ask next.",
  "Carousel takes an array of slides, where each slide is an array of content.",
  "Every slide in a Carousel must use the same component structure in the same order.",
  "Use CardHeader for section titles. Use TextContent for body text. Use MarkDownRenderer for rich formatted text with links, bold, lists.",
  "Use Heading for section titles with level: \"h1\" | \"h2\" | \"h3\" | \"h4\". Use Blockquote for quotes. Use InlineCode for inline code.",
  "Use CodeBlock with a language prop for code snippets. Always set the language for syntax context.",
  "Use Progress for completion/loading indicators.",
  "Use Avatar for user/profile images. Use Image/ImageBlock for content images.",
  "Use Separator to divide peer sections inside a Card.",
  "Use Breadcrumb to show how the evidence set was narrowed.",
  "Use NavigationMenu and Menubar only in a multi-panel research workspace, not for a simple answer.",
];

// ── Rules: variants & styling ──

export const shadcnRulesVariants = [
  "Button variant mapping — \"default\" (filled primary), \"secondary\" (muted), \"outline\" (bordered), \"ghost\" (transparent), \"link\" (underlined text), \"destructive\" (red/danger). Use the right variant for the context.",
  "Button size mapping — \"default\" (standard), \"xs\" (extra small), \"sm\" (small), \"lg\" (large), \"icon\" (square icon-only).",
  "Badge/Tag variants — \"default\" (filled primary), \"secondary\" (muted fill), \"destructive\" (red), \"outline\" (bordered), \"ghost\" (minimal).",
  "Alert variants — \"default\" (neutral), \"destructive\" (red error), \"info\" (blue informational), \"success\" (green confirmation), \"warning\" (amber caution). Always pick the variant that matches the message tone.",
  "Use Kbd for shortcuts such as Cmd+K.",
  "Use AspectRatio to constrain image thumbnails and forest-plot previews.",
  "Use ButtonGroup for related actions such as Include / Exclude / Maybe.",
];

// ── Rules: component selection ──

export const shadcnRulesSelection = [
  "When the user asks for a specific component (e.g. 'show me an accordion'), generate a realistic, fully-populated example of that component with sample data.",
  "Use DialogBlock to show a button that opens a modal dialog with content inside. Good for details/previews.",
  "Use AlertDialogBlock for confirmation dialogs (delete, logout, etc). Confirm action sends message to LLM.",
  "Use DrawerBlock for bottom panels with additional content. Good for details/reports. Use Sheet for full-text context; use DrawerBlock only for a bottom panel.",
  "Use PaginationBlock for paginated data. currentPage/totalPages are required.",
  "Use CalendarBlock for standalone calendar display. mode: \"single\" (pick one date), \"multiple\" (pick many), \"range\" (date range). Use numberOfMonths to show side-by-side months.",
  "Use Collapsible for nested evidence such as subgroup analyses; use Accordion when several sections are peers.",
  "Use InputGroup for a search field with add-ons.",
  "Use Field only when FormControl is too opinionated for a single field.",
  "For forms, define one FormControl reference per field so controls can stream progressively.",
  "For forms, always provide the second Form argument with Buttons(...) actions.",
  "Never nest Form inside Form.",
];

// ── Rules: evidence interaction patterns ──

export const shadcnRulesEvidence = [
  "Evidence search -> Command (Cmd+K). Never print a search box as prose.",
  "Citation marker -> plain inline [n]. The application renders each [n] as a clickable citation badge (author-year, e.g. (Zhang et al., 2025), by default) and lists the references at the bottom; never wrap a marker in a Badge/HoverCard and never repeat the full citation inline.",
  "Term definition -> Tooltip. Never parenthesize definitions in prose.",
  "Single-point detail -> Popover. Never open a full Dialog for one number.",
  "Study-design filter -> ToggleGroup. Never write filter instructions in prose.",
  "Side-by-side comparison -> Resizable. Never stack two tables that are meant to be compared.",
  "Long passage list -> ScrollArea inside the parent. Never let the Card grow unbounded.",
  "Streaming state -> Skeleton matching the eventual dimensions. Never show a blank region.",
  "Zero results -> Empty. Never leave an empty table or a blank region.",
  "Transient confirmation -> Sonner. Never add a permanent Alert for a copy action.",
];

// ── Rules: charts & tables ──

export const shadcnRulesCharts = [
  "INFORMATIVE CHARTS: every chart sets meta.title (what it shows) and meta.unit (the value unit); cartesian charts also set xLabel and yLabel. A chart without a title or unit is incomplete.",
  "LEGEND: keep the legend on whenever a chart has more than one series, slice, or marker; set meta.legend to false only for a single series.",
  "CHART SELECTION - pick by the shape of the data, not by habit:",
  "- Compare a value across categories -> BarChart.",
  "- Trend over an ordered axis or time -> LineChart; cumulative volume -> AreaChart.",
  "- Flat parts of one whole (2-7 slices) -> PieChart; nested parts of a whole -> SunburstChart.",
  "- Many categories or two hierarchy levels -> TreemapChart.",
  "- Signed changes to a running total -> WaterfallChart.",
  "- Screening or attrition stages -> FunnelChart.",
  "- Movement between stages or groups -> SankeyChart.",
  "- Two numeric variables or correlation -> ScatterChart.",
  "- The same 0-100 dimensions across entities -> RadarChart.",
  "- A ranked set on independent scales -> RadialChart.",
  "- A value at every row x column combination -> HeatmapChart.",
  "- One KPI against a maximum -> GaugeChart.",
  "- Geographic distribution -> MapChart (regions for country shading, points for sites).",
  "TABLE LABELS: every Table sets title (what it shows) and caption (source, n, caveat); every numeric Col sets its unit (mg, %, RR, 95% CI). A number without a unit is a bug.",
  "Never leave a unit only in prose when a table column or chart axis carries numbers - attach it to the header or axis.",
  "If a chart and a table show the same numbers, keep one; if both are needed, say why in the caption.",
  "GEOGRAPHIC DATA: when the answer ranks, compares, or maps a metric across countries or regions, you MUST include BOTH a MapChart (one MapRegion per country) AND a ranked BarChart sorted highest to lowest. A Table alone is INCOMPLETE.",
  "When the question asks for the highest, lowest, or top-N of anything, sort the BarChart from highest to lowest; a rank column in the Table is supplementary, never the only view.",
  "Every MapChart sets meta.title and meta.unit, and every country value must come from a cited passage.",
  "PRE-FLIGHT: country- or region-level results include a MapChart and a ranked BarChart, not only a Table.",
  "PRE-FLIGHT: every chart has meta.title, meta.unit, a legend, and labeled axes.",
  "PRE-FLIGHT: every numeric table column declares its unit.",
  "PRE-FLIGHT: the chart type matches the data shape in CHART SELECTION.",
];

// ── Rules: pre-flight checks ──

export const shadcnRulesPreflight = [
  "PRE-FLIGHT: evidence search is a Command palette, not a prose instruction.",
  "PRE-FLIGHT: citation markers are plain inline [n] - no Badge/HoverCard wrapper, no citations column, no sources list.",
  "PRE-FLIGHT: term definitions use Tooltip, not parentheticals.",
  "PRE-FLIGHT: single-point detail uses Popover, not DialogBlock.",
  "PRE-FLIGHT: side-by-side evidence uses Resizable, not sequential tables.",
  "PRE-FLIGHT: long evidence lists are wrapped in ScrollArea.",
  "PRE-FLIGHT: streaming placeholders are Skeleton with dimensions matching the eventual component.",
  "PRE-FLIGHT: transient confirmations use Sonner, not a permanent Alert.",
  "PRE-FLIGHT: zero-result states use Empty, not a blank region.",
];

// ── Aggregate export (kept flat for the existing prompt generator) ──

export const shadcnAdditionalRules = [
  ...shadcnRulesStructure,
  ...shadcnRulesVariants,
  ...shadcnRulesSelection,
  ...shadcnRulesEvidence,
  ...shadcnRulesCharts,
  ...shadcnRulesPreflight,
];

export const shadcnPromptOptions = {
  examples: shadcnExamples,
  additionalRules: shadcnAdditionalRules,
  // Optional: named groups for a generator that wants sectioned prompts.
  ruleGroups: {
    structure: shadcnRulesStructure,
    variants: shadcnRulesVariants,
    selection: shadcnRulesSelection,
    evidence: shadcnRulesEvidence,
    charts: shadcnRulesCharts,
    preflight: shadcnRulesPreflight,
  },
};