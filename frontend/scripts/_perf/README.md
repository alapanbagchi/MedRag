# AG-UI streaming perf harness

A backend-free way to exercise the AG-UI surface under a realistic token
stream and measure it. The fixtures are generated with the real backend
encoder, so the frames are byte-identical to what `/v1/ag-ui` emits.

## 1. Generate the fixtures

```bash
cd backend && .venv/bin/python ../frontend/scripts/_perf/gen_fixture.py    # full agent+answer run
cd backend && .venv/bin/python ../frontend/scripts/_perf/gen_variant.py    # small markdown + openui runs
```

## 2. Run the app

```bash
cd frontend && npm run dev        # or: npm run build && npm run preview -- --port 4180
```

## 3. Measure

```bash
cd frontend
APP_URL=http://localhost:5174 LABEL=after node scripts/_perf/measure.mjs
APP_URL=http://localhost:5174 LABEL=sheet OPEN_SHEET_MID=1 node scripts/_perf/measure.mjs
```

`measure.mjs` reports long-task count/total, frame percentiles (p50/p95/p99),
React commit count, and the wall-clock to drain the stream. The drain time is
backpressure-bound: the harness sends one chunk per macrotask, so a faster app
finishes sooner.

## 4. Functional check

```bash
APP_URL=http://localhost:5174 FIXTURE=agui-fixture-markdown.json \
  EXPECT="Answer|140/90 mmHg|Sources" node scripts/_perf/verify_variant.mjs
APP_URL=http://localhost:5174 FIXTURE=agui-fixture-openui.json \
  EXPECT="Hypertension diagnostic thresholds" node scripts/_perf/verify_variant.mjs
```

## Reference numbers (1440x900 headless Chromium, 1369 chunks)

| run | drain | long-task total | p99 frame |
| --- | --- | --- | --- |
| dev, before | 64.5s | 57.1s | 283ms |
| dev, after | 10.0s | 0.56s | 33ms |
| dev, after, agent sheet open | 12.3s | 0.70s | 33ms |
| production, after | 4.3s | 0.52s | 17ms |
