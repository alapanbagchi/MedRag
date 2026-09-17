"""Stateful research-run ledger: the JSON the orchestrator fills and steers by.

The run state is the server-side memory of a research run. The plan, every
spawned task, every judge-verified passage, every gap verdict, and the
findings returned per task all land here — written by code at the
tool/execution boundary, never by the model free-handing JSON.

The orchestrator steers from compact progress views (counts per task and
requirement, unresolved gaps, budget state), not by re-reading full
evidence text. Sub-agent legs still receive full passage text in their own
tool results — close reading stays where synthesis happens.
"""

from src.runstate.models import (
    ChatState,
    EvidenceRecord,
    GapRecord,
    RequirementRecord,
    RunState,
    RunStatus,
    TaskRecord,
    TaskState,
)
from src.runstate.progress import (
    progress_view,
    render_receipt,
    requirement_view,
    task_detail,
    turn_view,
)
from src.runstate.store import (
    RunStore,
    get_store,
    record_evidence,
    record_gaps,
    record_requirements,
    runstate_dir,
    set_store,
    snapshot_chat,
)
from src.runstate.tools import check_gaps, run_progress

__all__ = [
    "ChatState",
    "check_gaps",
    "EvidenceRecord",
    "GapRecord",
    "RequirementRecord",
    "RunState",
    "RunStatus",
    "RunStore",
    "TaskRecord",
    "TaskState",
    "get_store",
    "progress_view",
    "record_evidence",
    "record_gaps",
    "record_requirements",
    "render_receipt",
    "requirement_view",
    "run_progress",
    "runstate_dir",
    "set_store",
    "snapshot_chat",
    "task_detail",
    "turn_view",
]
