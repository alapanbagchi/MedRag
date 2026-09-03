"""python -m src.agents "question" - research CLI.

Builds the full research graph and (with --run) executes the end-to-end
research pipeline with LIVE PROGRESS: each stage prints as it runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json


def main() -> int:
    p = argparse.ArgumentParser(prog="x_deepagents", description="xdeep research CLI")
    p.add_argument("question", nargs="?", default="Does vitamin D lower blood pressure?")
    p.add_argument("--run", action="store_true", help="execute the research pipeline")
    p.add_argument("--json", action="store_true", help="print the run state as JSON")
    args = p.parse_args()

    from src.agents.graph import _new_run, build_research_graph, progress
    from src.agents.logging import close_run_log, log_path, open_run_log

    graph = build_research_graph()
    print("x_deepagents research graph compiled OK")
    print(f"(logging to {log_path()})")

    if args.run:
        open_run_log(query=args.question)
        try:
            run = asyncio.run(_stream_run(graph, args.question))
        finally:
            close_run_log()
        print("\n--- ANSWER ---")
        print(run.answer)
        if args.json:
            payload = {
                "run_id": run.run_id,
                "question": run.question,
                "phase": run.phase.value,
                "requirements": [r.summary() for r in run.requirements],
                "verified": [i.to_dict() for i in run.verified_items()],
                "contradictions": [c.model_dump(mode="json") for c in run.contradictions],
                "gap_resolutions": [g.model_dump(mode="json") for g in run.gap_resolutions],
                "gaps": run.gaps,
                "answer": run.answer,
            }
            print("\n--- RUN JSON ---")
            print(json.dumps(payload, indent=2))
    else:
        print("(dry run - add --run to execute the pipeline)")
        try:
            print(graph.get_graph().draw_ascii())
        except Exception as exc:
            print(f"(graph diagram unavailable: {exc})")
    return 0


async def _stream_run(graph, question: str):
    """Stream the graph, printing every node/completion as it happens."""
    from src.agents.graph import _new_run, progress

    final_state = None
    async for update in graph.astream(
        {"question": question, "state": _new_run(question)},
        stream_mode="updates",
    ):
        # update = {node_name: state_patch_dict}
        for node_name, patch in update.items():
            st = (patch or {}).get("state")
            if st is not None:
                progress(f"[graph] node completed: {node_name} "
                         f"(phase -> {st.phase.value})")
                final_state = st
    if final_state is None:
        raise RuntimeError("graph produced no state update")
    return final_state


if __name__ == "__main__":
    raise SystemExit(main())
