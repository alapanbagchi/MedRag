# Example trace — surgical repair techniques vs recurrent coarctation

Question (acceptance test 1, V2.1 parts 1-41):

    Which surgical repair techniques were associated with recurrent coarctation
    and what are the percentages and p-values?

Run: `python -m medrag.retrieval_v2 "<query>" --output retrieval_runs/v2_pageindex_surgical_fix`
Artifacts: retrieval_runs/v2_pageindex_surgical_fix/{v2_result.json, v2_evidence.md, v2_trace.log}

## 1. Planner output — ONE coherent requirement (no H1..H4 fragmentation)

    Requirements: 1 | Queries: 4 | Type: numerical

    H1
        topic:               recurrent coarctation
        target:              surgical repair techniques
        focus:               comparative_numerical
        condition:           recurrent coarctation
        requested_fields:    [percentage, p-value]
        preferred_evidence_types: [table_row, table_summary, table_footnotes, results, figure]

The old /expand contract generated four requirements ("Specific surgical
repair techniques", "Reported recurrence rates", "Statistical significance
(p-values)", "Percentages of recurrence") and nine queries. The new contract
produces exactly one evidence obligation.

Retrieval variants (V0..V3, each retrieved independently):

    V0  Which surgical repair techniques were associated with recurrent
        coarctation and what are the percentages and p-values?
    V1  surgical repair techniques recurrent coarctation
    V2  surgical repair techniques coarctation
    V3  surgical repair techniques recurrent coarctation percentage p-value

## 2. Global retrieval — paper discovery (unchanged BM25 + pgvector)

    PMC11743609  rank 1 (score 0.990)  - the paper containing Table 2
    + 11 more papers in the research neighborhood (12 selected)

## 3. PageIndex paper-local navigation (official pageindex 0.2.10 trees)

PAGEINDEX_SEARCH (PMC11743609, H1):

    query: Find the Results sections, tables, and table footnotes that report
           surgical repair techniques associated with recurrent coarctation
           including percentage, p-value.
    selected_nodes:
        0060  Table T2   Results > Recurrent coarctation (re-CoA) > Table T2   rel=1.00
        0048  Table T1   Results > Table T1                                    rel=0.86
        0067  "Data are shown in Table 2..."                                   rel=0.86
        0058/0059/0062  Re-CoA paragraphs (Results)                            rel=0.49-0.55
    status: success

PAGEINDEX_NODE_RESOLVED (0060):

    chunk_ids: [PMC11743609_T2_summary, PMC11743609_T2_row_0..21,
                PMC11743609_T2_footnotes]

## 4. Paper-local hybrid retrieval (per variant, BM25 + pgvector + PageIndex)

    local_search_PMC11743609_H1:
        variants: 4 | pageindex_status: success | candidates: 20+
        top candidates now include table_row nodes WITH table context
        (the anchor-metadata defect is fixed: rows are no longer degraded to
        generic paragraphs and the WHOLE table is assembled)

## 5. Table-aware evidence context (parts 15-16, 31)

    PMC11743609_T2_row_13 "End-to-end" context_text:

        Section: Results > Recurrent coarctation (re-CoA)
        Table: Table 2: Characteristics of patients with and without early re-CoA.
        Headers: Variables | No re-CoA (n = 24) [IQR] or n (%) | re-CoA (n = 4) [IQR] or n (%) | p
        Row: End-to-end | 2 (8) | 2 (50) | 0.04
        Footnotes: Data are presented as median [IQR], n/N (%); ... re-CoA, recurrent coarctation; ...

    detected_fields: {percentage: true, p-value: true}

## 6. Final evidence set (18 slots)

    13 table_row + 5 paragraph candidates; 15 with PageIndex provenance,
    13 with table context. Examples:

        PMC11743609_T2_row_7/8       table_row   percentage+p-value (Table 2)
        PMC11997790_ivaf042-T2_row_0 table_row   "Aortic arch roof-plasty plus
                                                 coarctation repair" (another paper)
        PMC11743609_T1_row_*         table_row   percentages (baseline Table 1)
        PMC12028424_0                paragraph   percentages + p-values (prose)

## 7. Coverage

    covered: ['H1'] | uncovered: [] | fraction 1.000

Coverage is achieved by the CONTEXTUAL table evidence (row + header +
footnote as one unit) — not by a paragraph that merely mentions the topic.

## 8. Metrics / timings

    requirement_recall 1.0 | final evidence: 18 | pageindex_evidence: 15
    plan 8ms | global 20.2s | local 6.9s | medcpt 67.5s | total ~96s
