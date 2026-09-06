import { MinusIcon, PlusIcon, ScanIcon, XIcon } from "lucide-react";
import { useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  EDGE_LABEL,
  PROVENANCE_LABEL,
  isEvidenceBacked,
  type EdgeKind,
  type MemNode,
  type MemoryGraphData,
} from "../lib/memory-graph";

const NODE_W = 240;
const NODE_H = 78;
const COL_X = 300;
const ROW_Y = 122;

const COLUMNS: MemNode["kind"][][] = [["entity"], ["claim"], ["evidence"], ["gap", "contradiction"]];

const KIND_DOT: Record<MemNode["kind"], string> = {
  entity: "#1883AE",
  claim: "#18AE95",
  evidence: "#0D0E1A",
  gap: "#b07d10",
  contradiction: "#d93025",
};

const KIND_LABEL: Record<MemNode["kind"], string> = {
  entity: "entity",
  claim: "claim",
  evidence: "evidence",
  gap: "gap",
  contradiction: "conflict",
};

function edgeStyle(kind: EdgeKind): { stroke: string; dash?: string; width: number } {
  switch (kind) {
    case "supported_by":
      return { stroke: "#18AE95", width: 2 };
    case "contradicted_by":
      return { stroke: "#d93025", width: 2, dash: "6 4" };
    case "related_to":
      return { stroke: "#9fb0ba", width: 1.5, dash: "2 4" };
    default:
      return { stroke: "#1883AE", width: 1.5 };
  }
}

function wrapTitle(title: string): [string, string] {
  if (title.length <= 38) return [title, ""];
  const cut = title.lastIndexOf(" ", 38);
  const at = cut > 12 ? cut : 38;
  return [title.slice(0, at), title.slice(at + 1, at + 40)];
}

interface Pos {
  x: number;
  y: number;
}

function layout(data: MemoryGraphData): Map<string, Pos> {
  const pos = new Map<string, Pos>();
  COLUMNS.forEach((kinds, col) => {
    const inCol = data.nodes.filter((n) => kinds.includes(n.kind));
    inCol.forEach((n, i) => {
      pos.set(n.id, { x: 60 + col * COL_X, y: 40 + i * ROW_Y });
    });
  });
  return pos;
}

/** Per-chat graph memory: entities/claims/evidence as nodes, typed edges. */
export function MemoryGraph({ data }: { data: MemoryGraphData }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [t, setT] = useState({ x: 24, y: 16, k: 1 });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [fitted, setFitted] = useState(false);

  const pos = useMemo(() => layout(data), [data]);
  const byId = useMemo(() => new Map(data.nodes.map((n) => [n.id, n])), [data]);

  // Fit the whole graph on first paint.
  useLayoutEffect(() => {
    if (fitted) return;
    const w = wrapRef.current?.clientWidth ?? 1000;
    const contentW = 60 + (COLUMNS.length - 1) * COL_X + NODE_W;
    setT({ x: 24, y: 16, k: Math.min(1, Math.max(0.45, (w - 380) / contentW)) });
    setFitted(true);
  }, [fitted]);

  const selected = selectedId ? byId.get(selectedId) : undefined;
  const linked = useMemo(() => {
    if (!selectedId) return [];
    return data.edges
      .filter((e) => e.from === selectedId || e.to === selectedId)
      .map((e) => {
        const other = byId.get(e.from === selectedId ? e.to : e.from);
        return { edge: e, other, outgoing: e.from === selectedId };
      });
  }, [data.edges, byId, selectedId]);

  const onWheel = (e: React.WheelEvent) => {
    const k = Math.min(2.2, Math.max(0.4, t.k * (e.deltaY < 0 ? 1.1 : 0.9)));
    setT((p) => ({ ...p, k }));
  };

  const onBackgroundDown = (e: React.MouseEvent) => {
    if ((e.target as Element).closest("[data-node]")) return;
    const start = { x: e.clientX, y: e.clientY, tx: t.x, ty: t.y };
    const move = (m: MouseEvent) => {
      setT((p) => ({ ...p, x: start.tx + (m.clientX - start.x), y: start.ty + (m.clientY - start.y) }));
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  const zoomBy = (f: number) =>
    setT((p) => ({ ...p, k: Math.min(2.2, Math.max(0.4, p.k * f)) }));
  const reset = () => {
    setT({ x: 24, y: 16, k: 1 });
    setSelectedId(null);
  };

  const backed = data.nodes.filter((n) => isEvidenceBacked(n.provenance)).length;

  return (
    <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-border bg-[#F7FAFB]">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-border/70 bg-white px-5 py-3">
        <div className="min-w-0">
          <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
            Graph memory · this chat
          </p>
          <h2 className="truncate text-[15px] font-semibold text-[#0D0E1A]">{data.question}</h2>
        </div>
        <div className="ml-auto flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
          <span>{data.nodes.length} nodes</span>
          <span aria-hidden>·</span>
          <span>{data.edges.length} relations</span>
          <span aria-hidden>·</span>
          <span className="text-[#18AE95]">{backed} evidence-backed</span>
        </div>
      </div>

      <div ref={wrapRef} className="relative min-h-0 flex-1">
        <svg
          className="h-full w-full cursor-grab active:cursor-grabbing"
          onWheel={onWheel}
          onMouseDown={onBackgroundDown}
          role="img"
          aria-label={`Memory graph for ${data.question}`}
        >
          <defs>
            <marker id="mem-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#8a97a3" />
            </marker>
          </defs>
          <g transform={`translate(${t.x},${t.y}) scale(${t.k})`}>
            {data.edges.map((e) => {
              const a = pos.get(e.from);
              const b = pos.get(e.to);
              if (!a || !b) return null;
              const x1 = a.x + NODE_W;
              const y1 = a.y + NODE_H / 2;
              const x2 = b.x;
              const y2 = b.y + NODE_H / 2;
              const dx = Math.max(40, (x2 - x1) / 2);
              const st = edgeStyle(e.kind);
              return (
                <g key={e.id}>
                  <path
                    d={`M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`}
                    fill="none"
                    stroke={st.stroke}
                    strokeWidth={st.width}
                    strokeDasharray={st.dash}
                    markerEnd="url(#mem-arrow)"
                  />
                  <text
                    x={(x1 + x2) / 2}
                    y={(y1 + y2) / 2 - 6}
                    textAnchor="middle"
                    fontSize="10.5"
                    fill="#5b6672"
                    style={{ paintOrder: "stroke", stroke: "#F7FAFB", strokeWidth: 4 }}
                  >
                    {EDGE_LABEL[e.kind]}
                  </text>
                </g>
              );
            })}
            {data.nodes.map((n) => {
              const p = pos.get(n.id);
              if (!p) return null;
              const [l1, l2] = wrapTitle(n.title);
              const active = n.id === selectedId;
              return (
                <g
                  key={n.id}
                  data-node={n.id}
                  transform={`translate(${p.x},${p.y})`}
                  onClick={() => setSelectedId(n.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") setSelectedId(n.id);
                  }}
                  tabIndex={0}
                  role="button"
                  aria-label={`${n.kind}: ${n.title}`}
                  className="cursor-pointer outline-none"
                >
                  <rect
                    width={NODE_W}
                    height={NODE_H}
                    rx={14}
                    fill="#ffffff"
                    stroke={active ? "#1883AE" : "#e3e9ec"}
                    strokeWidth={active ? 2.5 : 1.5}
                  />
                  <rect x={14} y={12} width={4} height={NODE_H - 24} rx={2} fill={KIND_DOT[n.kind]} />
                  <text x={28} y={28} fontSize={9.5} fill="#8a97a3" fontFamily="IBM Plex Mono, monospace" letterSpacing={1.2}>
                    {KIND_LABEL[n.kind].toUpperCase()}
                  </text>
                  <text x={28} y={45} fontSize={12.5} fontWeight={600} fill="#0D0E1A">
                    {l1.length > 34 ? `${l1.slice(0, 34)}…` : l1}
                  </text>
                  {l2 ? (
                    <text x={28} y={60} fontSize={12.5} fill="#3c4450">
                      {l2.length > 34 ? `${l2.slice(0, 34)}…` : l2}
                    </text>
                  ) : n.confidence != null ? (
                    <g>
                      <rect x={28} y={54} width={120} height={5} rx={2.5} fill="#eef2f4" />
                      <rect x={28} y={54} width={120 * n.confidence} height={5} rx={2.5} fill="#18AE95" />
                      <text x={154} y={59} fontSize={10} fill="#8a97a3">
                        {Math.round(n.confidence * 100)}%
                      </text>
                    </g>
                  ) : null}
                </g>
              );
            })}
          </g>
        </svg>

        <div className="absolute bottom-3 left-3 flex items-center gap-1 rounded-full border border-border bg-white/95 px-2 py-1 shadow-sm">
          <button type="button" onClick={() => zoomBy(1.2)} aria-label="Zoom in" className="flex size-7 items-center justify-center rounded-full text-[#3c4450] hover:bg-muted">
            <PlusIcon className="size-4" />
          </button>
          <button type="button" onClick={() => zoomBy(1 / 1.2)} aria-label="Zoom out" className="flex size-7 items-center justify-center rounded-full text-[#3c4450] hover:bg-muted">
            <MinusIcon className="size-4" />
          </button>
          <button type="button" onClick={reset} aria-label="Reset view" className="flex size-7 items-center justify-center rounded-full text-[#3c4450] hover:bg-muted">
            <ScanIcon className="size-4" />
          </button>
        </div>

        <div className="absolute bottom-3 right-3 hidden items-center gap-3 rounded-full border border-border bg-white/95 px-3 py-1.5 font-mono text-[10px] text-muted-foreground shadow-sm xl:flex">
          <span><i className="mr-1 inline-block size-2 rounded-full bg-[#18AE95]" />claim</span>
          <span><i className="mr-1 inline-block size-2 rounded-full bg-[#0D0E1A]" />evidence</span>
          <span><i className="mr-1 inline-block size-2 rounded-full bg-[#1883AE]" />entity</span>
          <span className="text-[#18AE95]">— supported</span>
          <span className="text-[#d93025]">- - contradicted</span>
        </div>

        {selected ? (
          <aside className="absolute right-3 top-3 flex max-h-[calc(100%-24px)] w-[300px] flex-col overflow-hidden rounded-2xl border border-border bg-white shadow-[0_18px_50px_rgba(13,14,26,0.14)]">
            <div className="flex items-start gap-2 border-b border-border/70 px-4 py-3">
              <div className="min-w-0 flex-1">
                <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                  {selected.kind} · {selected.id}
                </p>
                <h3 className="mt-0.5 text-[14px] font-semibold leading-5 text-[#0D0E1A]">
                  {selected.title}
                </h3>
              </div>
              <button type="button" onClick={() => setSelectedId(null)} aria-label="Close details" className="flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground hover:bg-muted">
                <XIcon className="size-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
              <span
                className={`inline-flex rounded-full px-2.5 py-1 text-[11px] font-semibold ${
                  isEvidenceBacked(selected.provenance)
                    ? "bg-[#18AE95]/10 text-[#0f7a62]"
                    : "bg-[#b07d10]/10 text-[#8a5f06]"
                }`}
              >
                {PROVENANCE_LABEL[selected.provenance]}
              </span>
              {selected.status ? (
                <p className="mt-2 text-[12.5px] text-[#3c4450]">
                  Status <span className="font-semibold text-[#0D0E1A]">{selected.status}</span>
                  {selected.confidence != null ? (
                    <> · confidence {Math.round(selected.confidence * 100)}%</>
                  ) : null}
                </p>
              ) : null}
              {selected.detail ? (
                <p className="mt-2 text-[12.5px] leading-5 text-[#3c4450]">{selected.detail}</p>
              ) : null}
              {selected.citations?.length ? (
                <div className="mt-3">
                  <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                    Provenance
                  </p>
                  <ul className="mt-1.5 space-y-1">
                    {selected.citations.map((c) => (
                      <li key={c.handle} className="flex items-center gap-1.5 font-mono text-[11.5px]">
                        <span className={`size-1.5 rounded-full ${c.verified ? "bg-[#18AE95]" : "bg-[#b07d10]"}`} />
                        <span className="text-[#0D0E1A]">{c.handle}</span>
                        <span className="text-muted-foreground">{c.verified ? "verified" : "unverified"}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {selected.validFrom ? (
                <p className="mt-2.5 font-mono text-[11px] text-muted-foreground">valid since {selected.validFrom}</p>
              ) : null}
              {linked.length ? (
                <div className="mt-3">
                  <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                    Relations · {linked.length}
                  </p>
                  <ul className="mt-1.5 space-y-1">
                    {linked.map(({ edge, other, outgoing }) => (
                      <li key={edge.id}>
                        <button
                          type="button"
                          onClick={() => other && setSelectedId(other.id)}
                          className="flex w-full items-center gap-1.5 rounded-lg px-1.5 py-1 text-left text-[12.5px] transition hover:bg-muted"
                        >
                          <span className="shrink-0 font-medium text-[#1883AE]">
                            {outgoing ? EDGE_LABEL[edge.kind] : `← ${EDGE_LABEL[edge.kind]}`}
                          </span>
                          <span className="min-w-0 flex-1 truncate text-[#3c4450]">{other?.title}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          </aside>
        ) : null}
      </div>
    </div>
  );
}

