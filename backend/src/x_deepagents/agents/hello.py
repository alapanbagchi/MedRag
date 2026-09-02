"""Research agent builders (deepagents runtime).

make_research_agent: the per-task research agent on GEMMA with the full tool
set - hybrid retrieve, searxng web search, and UMLS lookup - so it can
freely choose how to gather evidence for its task.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from src.x_deepagents.config import build_model_for_role, build_think_model


def make_hello_agent(model: Any = None):
    """Build a minimal deep agent (no tools yet)."""
    return create_deep_agent(
        model=model or build_model_for_role("decompose"),
        system_prompt=(
            "You are a scaffold hello agent for the xdeep research system. "
            "Answer the user concisely; no tools are attached yet."
        ),
        tools=[],
    )


def make_research_agent(model: Any = None, tools: list[Any] | None = None):
    """The per-task research agent (GEMMA, long-running).

    Tools: hybrid retrieve + searxng web search + UMLS lookup. The agent
    freely chooses between corpus retrieval and web search, and may enrich
    medical concepts via UMLS before searching. The workflow ORDER around it
    is enforced by rules.py, not here.
    """
    from src.x_deepagents.tools import tools_for_agent

    return create_deep_agent(
        model=model or build_think_model(),
        system_prompt=(
            "You are a research agent for the xdeep medical literature "
            "system, working on ONE evidence task.\n\n"
            "EVIDENCE POLICY (strict):\n"
            "1. The LOCAL CORPUS is your primary evidence source. Use the "
            "retrieve tool (hybrid retrieval + reranking over the curated "
            "medical corpus) FIRST and HEAVILY - most of your evidence must "
            "come from here.\n"
            "2. Use umls_lookup to expand medical terminology (synonyms, "
            "preferred names) to improve your retrieval queries.\n"
            "3. Use searxng_search (web) ONLY WHEN YOU NEED MORE INFO the "
            "corpus does not provide - a newer guideline, a specific paper, "
            "or an evidence gap. You decide HOW to search: pick the queries "
            "most likely to surface relevant, high-quality material.\n"
            "4. WEB TRUST (hard rule): only TRUSTED medical journals and "
            "official medical websites count as evidence (NEJM, Lancet, BMJ, "
            "JAMA, PubMed/PMC, WHO, CDC, NIH, Mayo Clinic, Medscape, "
            "UpToDate...). The searxng tool already drops social media and "
            "forums; never rely on Reddit, X/Twitter, Facebook, YouTube, "
            "blogs, or forums even if surfaced.\n"
            "5. Gather candidate passages; they are CANDIDATES, not verified "
            "evidence - they will be verified separately."
        ),
        tools=list(tools) if tools is not None else tools_for_agent(),
    )
