import json, os, sys, asyncio
os.environ["AGUI_EVENTS"] = "all"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "backend")))
from src.agents.ag_ui_transport import encode_ag_ui_events

BODY = ("## Answer\n\nAdult hypertension is defined by sustained office blood pressure readings of "
        "**140/90 mmHg or higher** [1]. The WHO fact sheet confirms the same cutoffs [2].\n\n"
        "## Limitations\n\n- Thresholds differ slightly for out-of-office measurement.\n")

VALID_OPENUI = "\n".join([
  'header = CardHeader("Hypertension diagnostic thresholds", "Evidence summary")',
  'body = TextContent("Adult hypertension is defined by sustained office blood pressure readings of 140/90 mmHg or higher [1].", "default")',
  'fu1 = FollowUpItem("What about the elderly?")',
  'follow = FollowUpBlock([fu1])',
  'root = Card([header, body, follow])',
])

GOOD_SOURCES = [
  {"id": "PMC11111111", "pmcid": "PMC11111111", "title": "Hypertension guideline one", "journal": "PMC", "year": 2024, "snippet": "140/90 mmHg.", "isWeb": False},
  {"id": "PMC22222222", "pmcid": "PMC22222222", "title": "Hypertension guideline two", "journal": "PMC", "year": 2023, "snippet": "Confirmation required.", "isWeb": False},
]

async def gen(format_name, program, out):
    async def pipeline():
        yield {"type": "status", "stage": "understanding", "message": "understanding"}
        yield {"type": "plan", "items": [
            {"id": "T1", "text": "Which measurements define hypertension?", "done": True, "started": True},
        ]}
        yield {"type": "task", "id": "T1", "state": "started", "question": "Which measurements define hypertension?", "model": "gpt-4.1", "ts": "2026-09-18T12:00:01"}
        for i in range(30):
            yield {"type": "thinking", "agent": "T1", "delta": "Checking the guideline thresholds carefully. ", "done": i == 29, "origin": "", "ts": "2026-09-18T12:00:%02d" % (i % 60)}
        yield {"type": "task", "id": "T1", "state": "done", "question": "Which measurements define hypertension?", "model": "gpt-4.1", "ts": "2026-09-18T12:00:20"}
        yield {"type": "sources", "sources": GOOD_SOURCES}
        yield {"type": "answer_format", "format": format_name, "run_id": "run123", "chat_id": "chat123"}
        yield {"type": "answer_sources", "sources": GOOD_SOURCES}
        step = 10
        for i in range(0, len(program), step):
            yield {"type": "token", "content": program[i:i+step], "done": i + step >= len(program)}
        yield {"type": "done", "usage": {"prompt": 1000, "completion": 400}}
    frames = []
    async for frame in encode_ag_ui_events(pipeline(), thread_id="chat123", run_id="run123"):
        frames.append(frame)
    with open(out, "w") as f:
        json.dump(frames, f)
    print(format_name, "frames", len(frames))

async def main():
    base = os.path.dirname(os.path.abspath(__file__)) + os.sep
    await gen("markdown", BODY, base + "agui-fixture-markdown.json")
    await gen("openui", VALID_OPENUI, base + "agui-fixture-openui.json")

asyncio.run(main())
