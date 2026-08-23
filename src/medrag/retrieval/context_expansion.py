"""
Hierarchical Context Expansion for MedRAG.

After precise chunk-level retrieval and reranking, this module expands
selected chunks into larger structural context for answer generation.

Architecture:
    SMALL CHUNKS → FIND THE EVIDENCE
    LARGER STRUCTURAL CONTEXT → UNDERSTAND THE EVIDENCE

The expansion is token-budget-aware and preserves citation traceability.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class ExpansionConfig:
    """Configuration for hierarchical context expansion."""
    
    # Token budget
    max_total_tokens: int = 12000
    max_expanded_tokens_per_hit: int = 2000
    
    # Paragraph window
    paragraph_window_before: int = 1
    paragraph_window_after: int = 1
    
    # Section expansion
    max_section_tokens: int = 3000
    
    # Table expansion
    max_table_rows: int = 50
    include_table_context: bool = True  # include before/after paragraphs
    
    # Figure expansion
    include_figure_context: bool = True
    
    # Fallback strategy
    fallback_window_sizes: List[int] = field(default_factory=lambda: [2, 1, 0])
    
    # Priority weights for budget allocation
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "reranker_score": 0.4,
        "coverage_count": 0.3,
        "query_coverage": 0.2,
        "unique_evidence": 0.1,
    })


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class ExpandedEvidence:
    """A deduplicated expanded evidence object for the generator."""
    
    # Identity
    document_id: str
    context_type: str  # "paragraph", "subsection", "table", "figure"
    
    # Source tracking
    source_chunk_ids: List[str]
    section_path: List[str]
    
    # Text content
    text: str
    
    # Original retrieval evidence
    retrieval_evidence: List[Dict[str, Any]]
    
    # Priority/ranking
    priority_score: float = 0.0
    token_count: int = 0
    
    # Coverage
    covered_requirements: List[int] = field(default_factory=list)
    matched_queries: List[str] = field(default_factory=list)
    
    # Budget
    truncated: bool = False
    original_token_count: int = 0


# ===========================================================================
# Token counting (simple heuristic)
# ===========================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: 1 token ≈ 4 characters)."""
    return len(text) // 4


# ===========================================================================
# Structural grouping
# ===========================================================================

@dataclass
class ChunkGroup:
    """A group of related chunks from the same structural unit."""
    
    document_id: str
    section_path: Tuple[str, ...]
    chunk_type: str  # "paragraph", "table", "figure"
    
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    chunk_positions: List[int] = field(default_factory=list)  # document_position
    
    @property
    def chunk_ids(self) -> List[str]:
        return [_get_chunk_id(c) for c in self.chunks]
    
    @property
    def min_position(self) -> int:
        return min(self.chunk_positions) if self.chunk_positions else 0
    
    @property
    def max_position(self) -> int:
        return max(self.chunk_positions) if self.chunk_positions else 0
    
    @property
    def is_contiguous(self) -> bool:
        """Check if chunks are contiguous (no gaps)."""
        if len(self.chunk_positions) <= 1:
            return True
        sorted_pos = sorted(self.chunk_positions)
        return all(sorted_pos[i+1] - sorted_pos[i] == 1 for i in range(len(sorted_pos)-1))


def group_chunks_by_structure(
    selected_chunks: List[Dict[str, Any]],
) -> Dict[Tuple[str, Tuple[str, ...], str], ChunkGroup]:
    """
    Group selected chunks by (document_id, section_path, chunk_type).
    
    This allows us to merge nearby paragraphs and collapse table rows.
    """
    groups: Dict[Tuple[str, Tuple[str, ...], str], ChunkGroup] = {}
    
    for chunk in selected_chunks:
        doc_id = chunk.get("document_id", "")
        breadcrumb = chunk.get("breadcrumb", [])
        if isinstance(breadcrumb, str):
            try:
                import ast
                breadcrumb = ast.literal_eval(breadcrumb)
            except:
                breadcrumb = [breadcrumb]
        section_path = tuple(breadcrumb) if breadcrumb else ("",)
        chunk_type = chunk.get("chunk_type", "paragraph")
        
        # Normalize table types to "table"
        if chunk_type in ("table_summary", "table_row", "table_footnotes"):
            chunk_type = "table"
        
        key = (doc_id, section_path, chunk_type)
        if key not in groups:
            groups[key] = ChunkGroup(
                document_id=doc_id,
                section_path=section_path,
                chunk_type=chunk_type,
            )
        
        group = groups[key]
        group.chunks.append(chunk)
        pos = chunk.get("document_position", 0)
        if pos is not None:
            group.chunk_positions.append(pos)
    
    return groups


# ===========================================================================
# Paragraph expansion
# ===========================================================================

def expand_paragraph_group(
    group: ChunkGroup,
    all_chunks_in_doc: List[Dict[str, Any]],
    config: ExpansionConfig,
) -> ExpandedEvidence:
    """
    Expand a group of paragraph chunks to include surrounding context.
    
    Strategy:
    1. Find the target chunks' positions
    2. Expand to window [min_pos - before, max_pos + after]
    3. If section is small enough, expand to full subsection
    """
    # Sort chunks by position
    sorted_chunks = sorted(group.chunks, key=lambda c: c.get("document_position", 0))
    
    # Find all chunks in the same section for this document
    section_chunks = [
        c for c in all_chunks_in_doc
        if c.get("document_id") == group.document_id
        and _breadcrumb_match(c.get("breadcrumb", []), group.section_path)
        and c.get("chunk_type") == "paragraph"
    ]
    section_chunks.sort(key=lambda c: c.get("document_position", 0))
    
    # Get positions of selected chunks
    selected_positions = set(c.get("document_position", 0) for c in sorted_chunks)
    
    # Strategy 1: Try full subsection if small enough
    section_tokens = sum(estimate_tokens(c.get("text", "")) for c in section_chunks)
    if section_tokens <= config.max_section_tokens:
        expanded_text = "\n\n".join(c.get("text", "") for c in section_chunks)
        return ExpandedEvidence(
            document_id=group.document_id,
            context_type="subsection",
            source_chunk_ids=[_get_chunk_id(c) for c in sorted_chunks],
            section_path=list(group.section_path),
            text=expanded_text,
            retrieval_evidence=_build_retrieval_evidence(sorted_chunks),
            token_count=section_tokens,
            covered_requirements=_merge_requirements(sorted_chunks),
            matched_queries=_merge_queries(sorted_chunks),
        )
    
    # Strategy 2: Paragraph window around selected chunks
    min_pos = min(selected_positions) if selected_positions else 0
    max_pos = max(selected_positions) if selected_positions else 0
    
    window_chunks = [
        c for c in section_chunks
        if min_pos - config.paragraph_window_before <= c.get("document_position", 0) <= max_pos + config.paragraph_window_after
    ]
    
    expanded_text = "\n\n".join(c.get("text", "") for c in window_chunks)
    token_count = estimate_tokens(expanded_text)
    
    # Check if we need to shrink
    truncated = False
    if token_count > config.max_expanded_tokens_per_hit:
        # Try smaller windows
        for window_size in config.fallback_window_sizes:
            window_chunks = [
                c for c in section_chunks
                if min_pos - window_size <= c.get("document_position", 0) <= max_pos + window_size
            ]
            expanded_text = "\n\n".join(c.get("text", "") for c in window_chunks)
            token_count = estimate_tokens(expanded_text)
            if token_count <= config.max_expanded_tokens_per_hit:
                break
        else:
            # Use only the selected chunks
            expanded_text = "\n\n".join(c.get("text", "") for c in sorted_chunks)
            token_count = estimate_tokens(expanded_text)
            truncated = True
    
    return ExpandedEvidence(
        document_id=group.document_id,
        context_type="paragraph_window",
        source_chunk_ids=[_get_chunk_id(c) for c in sorted_chunks],
        section_path=list(group.section_path),
        text=expanded_text,
        retrieval_evidence=_build_retrieval_evidence(sorted_chunks),
        token_count=token_count,
        truncated=truncated,
        original_token_count=section_tokens,
        covered_requirements=_merge_requirements(sorted_chunks),
        matched_queries=_merge_queries(sorted_chunks),
    )


# ===========================================================================
# Table expansion
# ===========================================================================

def expand_table_group(
    group: ChunkGroup,
    all_chunks_in_doc: List[Dict[str, Any]],
    config: ExpansionConfig,
) -> ExpandedEvidence:
    """
    Expand table chunks to include full table context.
    
    Strategy:
    1. Find all chunks belonging to this table
    2. Include summary, rows, footnotes
    3. Add surrounding paragraphs if within budget
    """
    # Find the table prefix
    table_prefix = _extract_table_prefix(group.chunks[0]["chunk_id"])
    
    # Find ALL chunks for this table
    all_table_chunks = [
        c for c in all_chunks_in_doc
        if c.get("document_id") == group.document_id
        and _extract_table_prefix(c.get("chunk_id", c.get("id", ""))) == table_prefix
    ]
    
    # Sort: summary first, then rows by position, then footnotes
    summary_chunks = [c for c in all_table_chunks if c.get("chunk_type") == "table_summary"]
    row_chunks = [c for c in all_table_chunks if c.get("chunk_type") == "table_row"]
    footnote_chunks = [c for c in all_table_chunks if c.get("chunk_type") == "table_footnotes"]
    
    row_chunks.sort(key=lambda c: c.get("document_position", 0))
    
    # Limit rows if too many
    if len(row_chunks) > config.max_table_rows:
        # Keep rows that were selected + nearby
        selected_ids = set(_get_chunk_id(c) for c in group.chunks)
        selected_rows = [c for c in row_chunks if _get_chunk_id(c) in selected_ids]
        other_rows = [c for c in row_chunks if _get_chunk_id(c) not in selected_ids]
        row_chunks = selected_rows + other_rows[:config.max_table_rows - len(selected_rows)]
    
    # Build table text
    parts = []
    for c in summary_chunks:
        parts.append(c.get("text", ""))
    for c in row_chunks:
        parts.append(c.get("text", ""))
    for c in footnote_chunks:
        parts.append(c.get("text", ""))
    
    expanded_text = "\n".join(parts)
    token_count = estimate_tokens(expanded_text)
    
    # Add surrounding paragraphs if within budget and configured
    surrounding_text = ""
    if config.include_table_context:
        # Find paragraphs before and after the table
        table_positions = [c.get("document_position", 0) for c in all_table_chunks if c.get("document_position")]
        if table_positions:
            table_pos = min(table_positions)
            before_chunks = [
                c for c in all_chunks_in_doc
                if c.get("document_id") == group.document_id
                and c.get("chunk_type") == "paragraph"
                and c.get("document_position", 0) == table_pos - 1
            ]
            after_chunks = [
                c for c in all_chunks_in_doc
                if c.get("document_id") == group.document_id
                and c.get("chunk_type") == "paragraph"
                and c.get("document_position", 0) == max(table_positions) + 1
            ]
            
            context_parts = []
            for c in before_chunks:
                context_parts.append(c.get("text", ""))
            context_parts.append(expanded_text)
            for c in after_chunks:
                context_parts.append(c.get("text", ""))
            
            candidate_text = "\n\n".join(context_parts)
            candidate_tokens = estimate_tokens(candidate_text)
            
            if candidate_tokens <= config.max_expanded_tokens_per_hit:
                surrounding_text = candidate_text
                expanded_text = surrounding_text
                token_count = candidate_tokens
    
    all_source_ids = list(set(
        _get_chunk_id(c) for c in all_table_chunks
        if c.get("chunk_type") in ("table_summary", "table_row", "table_footnotes")
    ))
    
    return ExpandedEvidence(
        document_id=group.document_id,
        context_type="table",
        source_chunk_ids=all_source_ids,
        section_path=list(group.section_path),
        text=expanded_text,
        retrieval_evidence=_build_retrieval_evidence(group.chunks),
        token_count=token_count,
        covered_requirements=_merge_requirements(group.chunks),
        matched_queries=_merge_queries(group.chunks),
    )


# ===========================================================================
# Figure expansion
# ===========================================================================

def expand_figure_group(
    group: ChunkGroup,
    all_chunks_in_doc: List[Dict[str, Any]],
    config: ExpansionConfig,
) -> ExpandedEvidence:
    """
    Expand figure chunks to include caption and surrounding context.
    """
    # Find surrounding paragraphs
    figure_chunks = group.chunks
    figure_positions = [c.get("document_position", 0) for c in figure_chunks if c.get("document_position")]
    
    parts = []
    source_ids = []
    
    # Add figure text (caption)
    for c in figure_chunks:
        parts.append(c.get("text", ""))
        source_ids.append(_get_chunk_id(c))
    
    # Add surrounding paragraphs if configured
    if config.include_figure_context and figure_positions:
        table_pos = min(figure_positions)
        before_chunks = [
            c for c in all_chunks_in_doc
            if c.get("document_id") == group.document_id
            and c.get("chunk_type") == "paragraph"
            and c.get("document_position", 0) == table_pos - 1
        ]
        after_chunks = [
            c for c in all_chunks_in_doc
            if c.get("document_id") == group.document_id
            and c.get("chunk_type") == "paragraph"
            and c.get("document_position", 0) == max(figure_positions) + 1
        ]
        
        context_parts = []
        for c in before_chunks:
            context_parts.append(c.get("text", ""))
        context_parts.extend(parts)
        for c in after_chunks:
            context_parts.append(c.get("text", ""))
        
        candidate_text = "\n\n".join(context_parts)
        candidate_tokens = estimate_tokens(candidate_text)
        
        if candidate_tokens <= config.max_expanded_tokens_per_hit:
            parts = context_parts
    
    expanded_text = "\n\n".join(parts)
    token_count = estimate_tokens(expanded_text)
    
    return ExpandedEvidence(
        document_id=group.document_id,
        context_type="figure",
        source_chunk_ids=source_ids,
        section_path=list(group.section_path),
        text=expanded_text,
        retrieval_evidence=_build_retrieval_evidence(group.chunks),
        token_count=token_count,
        covered_requirements=_merge_requirements(group.chunks),
        matched_queries=_merge_queries(group.chunks),
    )


# ===========================================================================
# Deduplication / merging
# ===========================================================================

def deduplicate_expansions(
    expansions: List[ExpandedEvidence],
    config: ExpansionConfig,
) -> List[ExpandedEvidence]:
    """
    Deduplicate overlapping or adjacent expansions.
    
    Merges expansions that:
    1. Are from the same document and section
    2. Have overlapping or adjacent text
    """
    if not expansions:
        return []
    
    # Group by document_id and section_path
    doc_sections: Dict[Tuple[str, Tuple], List[ExpandedEvidence]] = defaultdict(list)
    for exp in expansions:
        key = (exp.document_id, tuple(exp.section_path))
        doc_sections[key].append(exp)
    
    merged = []
    for key, exps in doc_sections.items():
        # Check if expansions overlap
        if len(exps) == 1:
            merged.append(exps[0])
            continue
        
        # Try to merge overlapping expansions
        merged_exp = _try_merge_exps(exps, config)
        merged.append(merged_exp)
    
    return merged


def _try_merge_exps(
    exps: List[ExpandedEvidence],
    config: ExpansionConfig,
) -> ExpandedEvidence:
    """Try to merge multiple expansions from the same section."""
    if len(exps) == 1:
        return exps[0]
    
    # Sort by priority score
    exps.sort(key=lambda e: e.priority_score, reverse=True)
    
    # Merge text
    all_text_parts = []
    all_source_ids = []
    all_retrieval = []
    all_requirements = set()
    all_queries = set()
    total_tokens = 0
    
    for exp in exps:
        # Check if adding this would exceed budget
        if total_tokens + exp.token_count > config.max_expanded_tokens_per_hit:
            # Truncate if needed
            remaining = config.max_expanded_tokens_per_hit - total_tokens
            if remaining > 100:  # Only add if meaningful
                truncated_text = _truncate_to_tokens(exp.text, remaining)
                all_text_parts.append(truncated_text)
                total_tokens += estimate_tokens(truncated_text)
            continue
        
        all_text_parts.append(exp.text)
        all_source_ids.extend(exp.source_chunk_ids)
        all_retrieval.extend(exp.retrieval_evidence)
        all_requirements.update(exp.covered_requirements)
        all_queries.update(exp.matched_queries)
        total_tokens += exp.token_count
    
    return ExpandedEvidence(
        document_id=exps[0].document_id,
        context_type="merged_" + exps[0].context_type,
        source_chunk_ids=list(set(all_source_ids)),
        section_path=exps[0].section_path,
        text="\n\n".join(all_text_parts),
        retrieval_evidence=all_retrieval,
        token_count=total_tokens,
        truncated=total_tokens >= config.max_expanded_tokens_per_hit,
        covered_requirements=sorted(all_requirements),
        matched_queries=sorted(all_queries),
    )


# ===========================================================================
# Budget allocation
# ===========================================================================

def allocate_budget(
    expansions: List[ExpandedEvidence],
    config: ExpansionConfig,
) -> List[ExpandedEvidence]:
    """
    Allocate token budget across expansions based on priority.
    
    Higher priority evidence gets more context budget.
    """
    if not expansions:
        return []
    
    # Calculate priority scores
    for exp in expansions:
        exp.priority_score = _calculate_priority(exp)
    
    # Sort by priority
    expansions.sort(key=lambda e: e.priority_score, reverse=True)
    
    # Allocate budget
    remaining_budget = config.max_total_tokens
    selected = []
    
    for exp in expansions:
        if exp.token_count <= remaining_budget:
            selected.append(exp)
            remaining_budget -= exp.token_count
        else:
            # Try to fit a truncated version
            if remaining_budget > 200:  # Minimum meaningful context
                truncated_text = _truncate_to_tokens(exp.text, remaining_budget)
                truncated_exp = ExpandedEvidence(
                    document_id=exp.document_id,
                    context_type=exp.context_type,
                    source_chunk_ids=exp.source_chunk_ids,
                    section_path=exp.section_path,
                    text=truncated_text,
                    retrieval_evidence=exp.retrieval_evidence,
                    token_count=estimate_tokens(truncated_text),
                    truncated=True,
                    original_token_count=exp.token_count,
                    covered_requirements=exp.covered_requirements,
                    matched_queries=exp.matched_queries,
                    priority_score=exp.priority_score,
                )
                selected.append(truncated_exp)
                remaining_budget = 0
            break
    
    return selected


def _calculate_priority(exp: ExpandedEvidence) -> float:
    """Calculate priority score for budget allocation."""
    # Base score from retrieval evidence
    scores = []
    for evidence in exp.retrieval_evidence:
        scores.append(evidence.get("reranker_score", 0.0))
    
    base_score = max(scores) if scores else 0.0
    
    # Coverage bonus
    coverage_bonus = len(exp.covered_requirements) * 0.1
    
    # Query coverage bonus
    query_bonus = len(exp.matched_queries) * 0.05
    
    return base_score + coverage_bonus + query_bonus


# ===========================================================================
# Helper functions
# ===========================================================================

def _get_chunk_id(c: Dict[str, Any]) -> str:
    """Get chunk ID from dict, handling both 'chunk_id' and 'id' keys."""
    return c.get("chunk_id", c.get("id", ""))


def _extract_table_prefix(chunk_id: str) -> Optional[str]:
    """Extract table prefix from chunk ID."""
    import re
    m = re.match(r'^(.+)_(?:summary|row_\d+|footnotes)$', chunk_id)
    if m:
        return m.group(1)
    return None


def _breadcrumb_match(breadcrumb, section_path: Tuple[str, ...]) -> bool:
    """Check if a breadcrumb matches a section path."""
    # Handle numpy arrays and other iterables
    if breadcrumb is None:
        return True
    try:
        if len(breadcrumb) == 0:
            return True
    except:
        return True
    
    if not section_path:
        return True
    
    # Convert to list if needed
    if not isinstance(breadcrumb, list):
        try:
            breadcrumb = list(breadcrumb)
        except:
            return True
    
    bc = [str(b).lower().strip() for b in breadcrumb if b]
    sp = [str(s).lower().strip() for s in section_path if s]
    return bc[:len(sp)] == sp[:len(bc)]


def _build_retrieval_evidence(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build retrieval evidence list from chunks."""
    evidence = []
    for c in chunks:
        evidence.append({
            "chunk_id": c.get("chunk_id", c.get("id", "")),
            "text": c.get("text", "")[:500],  # Truncate for metadata
            "reranker_score": c.get("reranker_score", 0.0),
            "covered_requirements": c.get("covered_requirements", []),
            "selection_rank": c.get("selection_rank", 0),
        })
    return evidence


def _merge_requirements(chunks: List[Dict[str, Any]]) -> List[int]:
    """Merge covered requirements from multiple chunks."""
    reqs = set()
    for c in chunks:
        for r in c.get("covered_requirements", []):
            reqs.add(r)
    return sorted(reqs)


def _merge_queries(chunks: List[Dict[str, Any]]) -> List[str]:
    """Merge matched queries from multiple chunks."""
    queries = set()
    for c in chunks:
        for q in c.get("matched_queries", []):
            queries.add(q)
    return sorted(queries)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    max_chars = max_tokens * 4  # Rough estimate
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ===========================================================================
# Main expansion function
# ===========================================================================

def expand_selected_context(
    selected_chunks: List[Dict[str, Any]],
    corpus_df: Any,  # pandas DataFrame
    config: Optional[ExpansionConfig] = None,
) -> Dict[str, Any]:
    """
    Main entry point for hierarchical context expansion.
    
    Args:
        selected_chunks: List of selected chunk dicts with retrieval metadata
        corpus_df: Full corpus DataFrame for looking up neighboring chunks
        config: Expansion configuration
    
    Returns:
        Dict with expanded evidence and diagnostics
    """
    if config is None:
        config = ExpansionConfig()
    
    if not selected_chunks:
        return {
            "expanded_evidence": [],
            "diagnostics": _empty_diagnostics(),
        }
    
    # Step 1: Group chunks by structure
    groups = group_chunks_by_structure(selected_chunks)
    
    # Step 2: Expand each group
    expansions = []
    for key, group in groups.items():
        doc_id, section_path, chunk_type = key
        
        # Get all chunks in this document for context lookup
        doc_chunks = corpus_df[
            corpus_df["document_id"] == doc_id
        ].to_dict("records") if corpus_df is not None else []
        
        if chunk_type == "table":
            exp = expand_table_group(group, doc_chunks, config)
        elif chunk_type == "figure":
            exp = expand_figure_group(group, doc_chunks, config)
        else:
            exp = expand_paragraph_group(group, doc_chunks, config)
        
        expansions.append(exp)
    
    # Step 3: Deduplicate overlapping expansions
    deduplicated = deduplicate_expansions(expansions, config)
    
    # Step 4: Allocate budget
    final_evidence = allocate_budget(deduplicated, config)
    
    # Step 5: Build diagnostics
    diagnostics = _build_diagnostics(
        selected_chunks, expansions, deduplicated, final_evidence, config
    )
    
    return {
        "expanded_evidence": [_evidence_to_dict(e) for e in final_evidence],
        "diagnostics": diagnostics,
    }


def _evidence_to_dict(exp: ExpandedEvidence) -> Dict[str, Any]:
    """Convert ExpandedEvidence to dict for JSON serialization."""
    return {
        "document_id": exp.document_id,
        "context_type": exp.context_type,
        "source_chunk_ids": exp.source_chunk_ids,
        "section_path": exp.section_path,
        "text": exp.text,
        "retrieval_evidence": exp.retrieval_evidence,
        "token_count": exp.token_count,
        "truncated": exp.truncated,
        "original_token_count": exp.original_token_count,
        "covered_requirements": exp.covered_requirements,
        "matched_queries": exp.matched_queries,
        "priority_score": round(exp.priority_score, 4),
    }


def _empty_diagnostics() -> Dict[str, Any]:
    """Return empty diagnostics."""
    return {
        "original_chunk_count": 0,
        "expanded_evidence_count": 0,
        "merged_count": 0,
        "original_token_count": 0,
        "expanded_token_count": 0,
        "expansion_ratio": 0.0,
        "paragraph_expansions": 0,
        "subsection_expansions": 0,
        "table_expansions": 0,
        "figure_expansions": 0,
        "truncated_count": 0,
        "requirements_preserved": 0,
    }


def _build_diagnostics(
    selected_chunks: List[Dict[str, Any]],
    expansions: List[ExpandedEvidence],
    deduplicated: List[ExpandedEvidence],
    final_evidence: List[ExpandedEvidence],
    config: ExpansionConfig,
) -> Dict[str, Any]:
    """Build comprehensive diagnostics."""
    original_tokens = sum(estimate_tokens(c.get("text", "")) for c in selected_chunks)
    expanded_tokens = sum(e.token_count for e in final_evidence)
    
    # Count by type
    type_counts = defaultdict(int)
    for e in final_evidence:
        type_counts[e.context_type] += 1
    
    # Count truncated
    truncated_count = sum(1 for e in final_evidence if e.truncated)
    
    # Requirements preserved
    all_reqs = set()
    for e in final_evidence:
        all_reqs.update(e.covered_requirements)
    
    return {
        "original_chunk_count": len(selected_chunks),
        "expanded_evidence_count": len(final_evidence),
        "merged_count": len(expansions) - len(deduplicated),
        "original_token_count": original_tokens,
        "expanded_token_count": expanded_tokens,
        "expansion_ratio": round(expanded_tokens / original_tokens, 2) if original_tokens > 0 else 0,
        "paragraph_expansions": type_counts.get("paragraph_window", 0) + type_counts.get("subsection", 0),
        "subsection_expansions": type_counts.get("subsection", 0),
        "table_expansions": type_counts.get("table", 0),
        "figure_expansions": type_counts.get("figure", 0),
        "truncated_count": truncated_count,
        "requirements_preserved": len(all_reqs),
        "source_chunks_per_expansion": [len(e.source_chunk_ids) for e in final_evidence],
    }


# ===========================================================================
# CLI integration
# ===========================================================================

def print_expansion_debug(expansions: List[ExpandedEvidence]) -> None:
    """Print detailed expansion debug output."""
    for i, exp in enumerate(expansions, 1):
        print(f"\n{'='*60}")
        print(f"EXPANDED EVIDENCE {i}")
        print(f"{'='*60}")
        print(f"Document: {exp.document_id}")
        print(f"Section: {' > '.join(exp.section_path)}")
        print(f"Context type: {exp.context_type}")
        print(f"\nTriggered by:")
        for evidence in exp.retrieval_evidence:
            print(f"  - {evidence['chunk_id']}")
            print(f"    reranker_score: {evidence.get('reranker_score', 'N/A')}")
            print(f"    requirements: {evidence.get('covered_requirements', [])}")
        print(f"\nMerged source chunks: {exp.source_chunk_ids}")
        print(f"Expanded tokens: {exp.token_count:,}")
        if exp.truncated:
            print(f"(truncated from {exp.original_token_count:,} tokens)")
        print(f"\nTEXT:")
        print(exp.text[:1000] + "..." if len(exp.text) > 1000 else exp.text)
