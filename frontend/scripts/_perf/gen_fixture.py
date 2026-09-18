import json, random, sys, asyncio, os
os.environ["AGUI_EVENTS"] = "all"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "backend")))
from src.agents.ag_ui_transport import encode_ag_ui_events

random.seed(7)

def long_text(n, w=8):
    words = "hypertension systolic diastolic blood pressure guideline evidence cohort trial patients threshold risk reduction meta analysis randomized controlled outcomes measurement".split()
    return " ".join(random.choice(words) for _ in range(max(1, n // 6 * w // 8)))

openui = []
openui.append('answer = Card(children=[header, summary, table, chart, followups])')
openui.append('header = CardHeader(title="Hypertension diagnostic thresholds", subtitle="Evidence summary")')
openui.append('summary = TextContent(text="Adult hypertension is defined by sustained office blood pressure readings of 140/90 mmHg or higher [1]. The WHO fact sheet confirms the same cutoffs [2].")')
col = 'columns = [' + ", ".join('ColumnDef(label="R%d", data=[%s])' % (i, ",".join(str(random.randint(1,9)) for _ in range(6))) for i in range(1,5)) + ']'
openui.append(col)
openui.append('table = Table(columns=columns)')
openui.append('chart = LineChart(labels=["2020","2021","2022","2023","2024","2025"], xLabel="Year", yLabel="Prevalence", series=[Series(category="Adults", values=[%s])])' % ",".join(str(random.randint(20,45)) for _ in range(6)))
openui.append('followups = FollowUpBlock(items=[' + ", ".join('FollowUpItem(text="What are the guideline differences for the elderly population %d?")' % i for i in range(1,6)) + '])')
for s in range(1, 7):
    openui.append('sec%d = SectionBlock(sections=[secitem%d])' % (s, s))
    openui.append('secitem%d = SectionItem(trigger="Clinical implication %d", content=[body%d])' % (s, s, s))
    openui.append('body%d = MarkDownRenderer(textMarkdown="%s")' % (s, long_text(600)))
    openui.append('callout%d = Callout(title="Evidence note %d", description="%s")' % (s, s, long_text(160)))
openui.append('answer = Card(children=[header, summary, table, chart, followups] + [' + ", ".join('sec%d' % s for s in range(1, 7)) + '] + [' + ", ".join('callout%d' % s for s in range(1, 7)) + '])')
program = chr(10).join(openui)

def thinking_deltas(agent, n, chars=26):
    chunks = []
    for i in range(n):
        chunks.append({"type":"thinking","agent":agent,"delta":long_text(chars)[:chars],"done":i==n-1,"origin":"","ts":"2026-09-18T12:00:%02d" % (i%60)})
    return chunks

async def pipeline():
    yield {"type":"status","stage":"understanding","message":"understanding the question"}
    yield {"type":"plan","items":[
        {"id":"T1","text":"Which specific blood pressure measurements define hypertension in routine adult care?","done":False,"started":True},
        {"id":"T2","text":"What do the WHO and ESC/ESH guidelines recommend for diagnostic thresholds?","done":False,"started":False},
    ]}
    yield {"type":"task","id":"T1","state":"started","question":"Which specific blood pressure measurements define hypertension?","model":"gpt-4.1","ts":"2026-09-18T12:00:01"}
    yield {"type":"task","id":"T2","state":"started","question":"What do the WHO and ESC/ESH guidelines recommend?","model":"gpt-4.1","ts":"2026-09-18T12:00:02"}
    for i,(tid,q) in enumerate([("T1","Which specific blood pressure measurements define hypertension?"),("T2","What do the WHO and ESC/ESH guidelines recommend?")]):
        yield {"type":"tool_call","call_id":"%s:call_spawn%d"%(tid,i),"name":"spawn_subagent","args":{"task":q,"task_id":tid,"depth":"deep"},"ts":"2026-09-18T12:00:0%d"%(i+3)}
        yield {"type":"tool_result","call_id":"%s:call_spawn%d"%(tid,i),"name":"spawn_subagent","result":{"ok":True},"ok":True,"ts":"2026-09-18T12:00:0%d"%(i+3)}
    for tid in ("T1","T2"):
        for e in thinking_deltas(tid, 160):
            yield e
    passages = [{"chunk_id":"PMC%d.p%d"%(11717708+i,i),"document_id":"PMC%d"%(11717708+i),"section":"Results","chunk_type":"text","score":0.9-0.03*i,"text":long_text(400),"title":"Epidemiology and diagnosis of adult hypertension %d"%i,"journal":"BMC Fam Pract"} for i in range(10)]
    for tid in ("T1","T2"):
        cid = "%s:call_retrieve"%tid
        yield {"type":"tool_call","call_id":cid,"name":"local_search","args":{"query":"hypertension diagnostic threshold"},"ts":"2026-09-18T12:00:10"}
        yield {"type":"tool_progress","call_id":cid,"name":"local_search","stage":"retrieved","result":passages,"ts":"2026-09-18T12:00:11"}
        verdicts = "\n".join('{"passage_id":"PMC%d.p%d","intent_score":%d,"coverage":["R1"],"reason":"Directly states the threshold."}'%(11717708+i,i,random.randint(1,5)) for i in range(10))
        result = json.dumps({"passages":passages}) + "\n---VERDICTS---\n" + verdicts
        yield {"type":"tool_result","call_id":cid,"name":"local_search","result":result,"ok":True,"ts":"2026-09-18T12:00:12"}
    for e in thinking_deltas("", 160):
        yield e
    yield {"type":"task","id":"T1","state":"done","question":"Which specific blood pressure measurements define hypertension?","model":"gpt-4.1","ts":"2026-09-18T12:00:20"}
    yield {"type":"task","id":"T2","state":"done","question":"What do the WHO and ESC/ESH guidelines recommend?","model":"gpt-4.1","ts":"2026-09-18T12:00:21"}
    sources = [{"id":"PMC11717708","pmcid":"PMC11717708","title":"Epidemiology and diagnosis of adult hypertension in primary care","journal":"PMC","year":2024,"snippet":long_text(80),"isWeb":False} for _ in range(8)]
    yield {"type":"sources","sources":sources}
    yield {"type":"answer_format","format":"openui","run_id":"run123","chat_id":"chat123"}
    yield {"type":"answer_sources","sources":sources}
    step = 12
    for i in range(0, len(program), step):
        yield {"type":"token","content":program[i:i+step],"done": i + step >= len(program)}
    for e in thinking_deltas("synthesizer", 60):
        yield e
    yield {"type":"done","usage":{"prompt":12000,"completion":3000}}

async def main():
    frames = []
    async for frame in encode_ag_ui_events(pipeline(), thread_id="chat123", run_id="run123"):
        frames.append(frame)
    with open(os.path.join(HERE, "agui-fixture.json"), "w") as f:
        json.dump(frames, f)
    print("frames", len(frames), "bytes", sum(len(x) for x in frames))

asyncio.run(main())
