// Perf harness: feeds a synthetic AG-UI SSE stream through the real app and
// records long tasks, rAF jank, React commits, and how long the UI keeps
// working after the stream ends. No backend / no LLM needed.
import puppeteer from "puppeteer-core";
import { readFileSync } from "node:fs";

const FRAMES = JSON.parse(
  readFileSync(new URL("./agui-fixture.json", import.meta.url), "utf8"),
);
const APP_URL = process.env.APP_URL || "http://localhost:5174";
const STEP_MS = Number(process.env.STEP_MS || 3);
const OPEN_SHEET = process.env.OPEN_SHEET === "1";
const LABEL = process.env.LABEL || "run";

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium",
  headless: true,
  args: [
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-features=CalculateNativeWinOcclusion",
  ],
  defaultViewport: { width: 1440, height: 900 },
});
const page = await browser.newPage();
const errs = [];
page.on("console", (m) => { if (m.type() === "error") errs.push(m.text().slice(0, 200)); });
page.on("pageerror", (e) => errs.push("pageerror: " + e.message));

await page.evaluateOnNewDocument(
  (frames, stepMs, openSheet) => {
    const P = {
      longTasks: [],
      frameDeltas: [],
      commits: 0,
      textLen: [],
      start: 0,
      streamEnd: 0,
      sent: 0,
      total: frames.length,
    };
    window.__PERF = P;

    // React commit counter via the DevTools global hook.
    const hook = (window.__REACT_DEVTOOLS_GLOBAL_HOOK__ =
      window.__REACT_DEVTOOLS_GLOBAL_HOOK__ || {});
    hook.supportsFiber = true;
    hook.renderers = new Map();
    hook.getFiberRoots = function () { return new Set(); };
    hook.emit = function () {};
    hook.sub = function (fn) { return fn; };
    hook.inject = function (renderer) {
      const id = hook.renderers.size + 1;
      hook.renderers.set(id, renderer);
      return id;
    };
    hook.checkDCE = function () {};
    hook.onCommitFiberRoot = function () { P.commits += 1; };
    hook.onCommitFiberUnmount = function () {};
    hook.onPostCommitFiberRoot = function () {};

    try {
      new PerformanceObserver((list) => {
        for (const e of list.getEntries()) P.longTasks.push(e.duration);
      }).observe({ entryTypes: ["longtask"] });
    } catch {}

    let last = performance.now();
    function tick(t) {
      P.frameDeltas.push(t - last);
      last = t;
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    // Sample visible text length to watch the UI fall behind/ahead.
    setInterval(() => {
      try { P.textLen.push([Math.round(performance.now()), document.body.innerText.length]); } catch {}
    }, 100);

    const realFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
      const url = typeof input === "string" ? input : (input && input.url) || "";
      if (url.includes("/v1/ag-ui")) {
        P.fetchCalled = true;
        P.visibility = document.visibilityState;
        const enc = new TextEncoder();
        let i = 0;
        P.start = performance.now();
        // MessageChannel macrotasks are not subject to headless timer
        // throttling; one frame per task lets React commit between chunks.
        const channel = new MessageChannel();
        const nextTask = () =>
          new Promise((resolve) => {
            channel.port1.onmessage = () => resolve();
            channel.port2.postMessage(null);
          });
        const stream = new ReadableStream({
          async pull(controller) {
            P.pulls = (P.pulls || 0) + 1;
            await nextTask();
            if (i < frames.length) {
              try { controller.enqueue(enc.encode(frames[i])); } catch {}
              i += 1;
              P.sent = i;
            }
            if (i >= frames.length) {
              try { controller.close(); } catch {}
              P.streamEnd = performance.now();
            }
          },
        });
        return Promise.resolve(
          new Response(stream, {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
          }),
        );
      }
      return realFetch(input, init);
    };
  },
  FRAMES,
  STEP_MS,
  OPEN_SHEET,
);

await page.goto(APP_URL, { waitUntil: "load", timeout: 30000 });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(1200);
await page.waitForSelector("textarea", { timeout: 10000 });
await page.type("textarea", "Which blood pressure measurements define hypertension?", { delay: 2 });
await page.keyboard.press("Enter");
console.log("sent");

// Optionally open the agent sheet as soon as its trigger appears, so the
// sheet-open path is exercised while tokens are still streaming.
if (process.env.OPEN_SHEET_MID === "1") {
  const t0 = Date.now();
  while (Date.now() - t0 < 40000) {
    const ok = await page.evaluate(() => {
      const btn = [...document.querySelectorAll("button")].find(
        (b) => (b.textContent || "").includes("agentic flow") || (b.textContent || "").includes("Agent flow"),
      );
      if (btn) { btn.click(); return true; }
      return false;
    });
    if (ok) break;
    await sleep(150);
  }
}

// Wait for stream end (or timeout), then let the UI settle.
const deadline = Date.now() + 240000;
let ended = false;
while (Date.now() < deadline) {
  const st = await page.evaluate(() => ({ end: (window.__PERF || {}).streamEnd || 0, sent: (window.__PERF || {}).sent || 0, total: (window.__PERF || {}).total || 0, pulls: (window.__PERF || {}).pulls || 0 }));
  if (!ended) console.error("progress sent=" + st.sent + "/" + st.total + " pulls=" + st.pulls + " errs=" + errs.length + (errs.length ? " :: " + errs[errs.length - 1] : ""));
  if (st.end) { ended = true; break; }
  await sleep(1000);
}
console.log("stream ended:", ended);

if (OPEN_SHEET) {
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll("button")].find((b) => (b.textContent || "").includes("agentic flow"));
    btn?.click();
  });
  await sleep(300);
}

// Settle: snapshot long tasks every 300ms until 600ms pass with no new ones.
let lastLong = 0;
let idleSince = Date.now();
while (Date.now() - idleSince < 800) {
  await sleep(300);
  const n = await page.evaluate(() => (window.__PERF || {}).longTasks.length);
  if (n !== lastLong) { lastLong = n; idleSince = Date.now(); }
}

const perf = await page.evaluate(() => {
  const P = window.__PERF;
  const lt = P.longTasks;
  const fd = P.frameDeltas;
  const sorted = fd.slice().sort((a, b) => a - b);
  const pct = (p) => (sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] : 0);
  const visibleEnd = P.textLen.length ? P.textLen[P.textLen.length - 1][1] : 0;
  return {
    longTaskCount: lt.length,
    longTaskTotal: Math.round(lt.reduce((a, b) => a + b, 0)),
    longTaskMax: Math.round(lt.length ? Math.max(...lt) : 0),
    longTaskOver50: lt.filter((d) => d > 50).length,
    frameCount: fd.length,
    frameP50: +pct(0.5).toFixed(1),
    frameP95: +pct(0.95).toFixed(1),
    frameP99: +pct(0.99).toFixed(1),
    framesOver33: fd.filter((d) => d > 33).length,
    framesOver50: fd.filter((d) => d > 50).length,
    commits: P.commits,
    streamMs: Math.round(P.streamEnd - P.start),
    sent: P.sent,
    total: P.total,
    visibleEnd,
    textSamples: P.textLen.length,
  };
});
// Measure processing tail: after streamEnd, how long until long tasks stop.
const tail = await page.evaluate(() => {
  const P = window.__PERF;
  const lastTaskTime = P.longTaskTimes ? P.longTaskTimes[P.longTaskTimes.length - 1] : 0;
  return { lastTaskTime, streamEnd: P.streamEnd };
});
console.log(JSON.stringify({ label: LABEL, ...perf, consoleErrors: errs.slice(0, 3) }, null, 2));
await page.screenshot({ path: new URL("./" + LABEL + ".png", import.meta.url).pathname });
await browser.close();
