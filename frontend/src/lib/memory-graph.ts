/**
 * Graph memory types, mirroring backend/src/memory/{models,enums}.py —
 * ClaimRecord / EvidenceReferenceRecord / ResearchGapRecord /
 * ContradictionRecord as nodes; ClaimEvidenceLinkRecord (role) and
 * ClaimRelationRecord (relation) as labeled edges.
 */

export type ProvenanceClass =
  | "user_assertion"
  | "model_inference"
  | "unverified_information"
  | "verified_evidence"
  | "evidence_derived_claim";

export type ClaimStatus =
  | "supported"
  | "contradicted"
  | "mixed"
  | "superseded"
  | "invalidated"
  | "unresolved";

export type EdgeKind =
  | "supported_by"
  | "contradicted_by"
  | "refined_by"
  | "supersedes"
  | "derived_from"
  | "related_to";

export type MemNodeKind = "entity" | "claim" | "evidence" | "gap" | "contradiction";

export interface MemNode {
  id: string;
  kind: MemNodeKind;
  title: string;
  detail?: string;
  provenance: ProvenanceClass;
  status?: ClaimStatus;
  confidence?: number; // 0..1
  /** Evidence handles, e.g. "PMCID:PMC10001459" (verified) */
  citations?: Array<{ handle: string; verified: boolean }>;
  validFrom?: string;
}

export interface MemEdge {
  id: string;
  from: string;
  to: string;
  kind: EdgeKind;
  weight?: number;
}

export interface MemoryGraphData {
  question: string;
  nodes: MemNode[];
  edges: MemEdge[];
}

export const PROVENANCE_LABEL: Record<ProvenanceClass, string> = {
  user_assertion: "User assertion",
  model_inference: "Model inference",
  unverified_information: "Unverified",
  verified_evidence: "Verified evidence",
  evidence_derived_claim: "Evidence-backed claim",
};

/** Only these classes may be presented as evidence-backed (backend invariant). */
export function isEvidenceBacked(p: ProvenanceClass): boolean {
  return p === "verified_evidence" || p === "evidence_derived_claim";
}

export const EDGE_LABEL: Record<EdgeKind, string> = {
  supported_by: "supported by",
  contradicted_by: "contradicted by",
  refined_by: "refined by",
  supersedes: "supersedes",
  derived_from: "derived from",
  related_to: "related to",
};

/** Demo per-chat memory for the radiology thread. */
export const DEMO_MEMORY_GRAPH: MemoryGraphData = {
  question: "Can AI replace radiologists?",
  nodes: [
    { id: "ent-mammo", kind: "entity", title: "AI mammography", provenance: "user_assertion", detail: "Topic entity extracted from the research question." },
    { id: "ent-recall", kind: "entity", title: "Recall rate", provenance: "user_assertion", detail: "Outcome entity: screening recall decisions." },
    { id: "ent-workforce", kind: "entity", title: "Radiologist workforce", provenance: "user_assertion", detail: "Workforce entity: staffing and roles." },
    {
      id: "clm-accuracy", kind: "claim", title: "AI matches radiologists in controlled diagnostic studies",
      provenance: "evidence_derived_claim", status: "supported", confidence: 0.88,
      citations: [{ handle: "PMCID:PMC10000019", verified: true }],
      validFrom: "2024-03",
    },
    {
      id: "clm-recall", kind: "claim", title: "AI assistance reduces mammography recall rates",
      provenance: "evidence_derived_claim", status: "supported", confidence: 0.91,
      citations: [{ handle: "PMCID:PMC10001459", verified: true }],
      validFrom: "2024-06",
    },
    {
      id: "clm-shift", kind: "claim", title: "Diagnostic performance degrades on out-of-distribution scanners",
      provenance: "evidence_derived_claim", status: "supported", confidence: 0.84,
      citations: [{ handle: "PMCID:PMC10003314", verified: true }],
      validFrom: "2023-11",
    },
    {
      id: "clm-replace", kind: "claim", title: "AI will fully replace radiologists by 2030",
      provenance: "model_inference", status: "contradicted", confidence: 0.32,
      citations: [{ handle: "PMCID:PMC10000019", verified: true }],
      validFrom: "2024-01",
    },
    { id: "ev-19", kind: "evidence", title: "Deep learning for chest radiograph diagnosis", provenance: "verified_evidence", citations: [{ handle: "PMCID:PMC10000019", verified: true }], validFrom: "2024" },
    { id: "ev-59", kind: "evidence", title: "AI-assisted mammography screening cohort", provenance: "verified_evidence", citations: [{ handle: "PMCID:PMC10001459", verified: true }], validFrom: "2024" },
    { id: "ev-14", kind: "evidence", title: "Failure modes under distribution shift", provenance: "verified_evidence", citations: [{ handle: "PMCID:PMC10003314", verified: true }], validFrom: "2023" },
    { id: "gap-long", kind: "gap", title: "Long-term prospective screening outcomes", provenance: "unverified_information", detail: "Open research gap: missing long-horizon evidence." },
    { id: "con-recall", kind: "contradiction", title: "Recall-rate effect varies by site and reader mix", provenance: "unverified_information", status: "unresolved", detail: "Context-dependent conflict, preserved not averaged." },
  ],
  edges: [
    { id: "e1", from: "clm-accuracy", to: "ev-19", kind: "supported_by", weight: 0.94 },
    { id: "e2", from: "clm-recall", to: "ev-59", kind: "supported_by", weight: 0.91 },
    { id: "e3", from: "clm-shift", to: "ev-14", kind: "supported_by", weight: 0.86 },
    { id: "e4", from: "clm-replace", to: "ev-19", kind: "contradicted_by", weight: 0.88 },
    { id: "e5", from: "clm-recall", to: "clm-accuracy", kind: "derived_from" },
    { id: "e6", from: "clm-shift", to: "clm-recall", kind: "related_to" },
    { id: "e7", from: "ent-mammo", to: "clm-recall", kind: "related_to" },
    { id: "e8", from: "ent-recall", to: "clm-recall", kind: "related_to" },
    { id: "e9", from: "ent-workforce", to: "clm-replace", kind: "related_to" },
    { id: "e10", from: "gap-long", to: "clm-recall", kind: "related_to" },
    { id: "e11", from: "con-recall", to: "clm-recall", kind: "related_to" },
  ],
};
