// Verify one fixture renders an expected answer contract.
import puppeteer from "puppeteer-core";
import { readFileSync } from "node:fs";

const file = process.env.FIXTURE || "agui-fixture-markdown.json";
const FRAMES = JSON.parse(readFileSync(new URL("./" + file, import.meta.url), "utf8"));
const EXPECT = (process.env.EXPECT || "").split("|").filter(Boolean);
const APP_URL = process.env.APP_URL || "http://localhost:5174";

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium", headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--disable-background-timer-throttling", "--disable-renderer-backgrounding"],
  defaultViewport: { width: 1440, height: 900 },
});
const page = await browser.newPage();
const msgs = [];
page.on("console", (m) => { if (m.type() === "error") msgs.push(m.text()); });
page.on("pageerror", (e) => msgs.push("pageerror: " + e.message));
await page.evaluateOnNewDocument((frames) => {
  window.__PERF = { sent: 0, streamEnd: 0 };
  const realFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (url.includes("/v1/ag-ui")) {
      const enc = new TextEncoder(); let i = 0;
      const channel = new MessageChannel();
      const nextTask = () => new Promise((res) => { channel.port1.onmessage = () => res(); channel.port2.postMessage(null); });
      const stream = new ReadableStream({ async pull(c) { await nextTask(); if (i < frames.length) { try { c.enqueue(enc.encode(frames[i])); } catch {} i += 1; window.__PERF.sent = i; } if (i >= frames.length) { try { c.close(); } catch {} window.__PERF.streamEnd = performance.now(); } } });
      return Promise.resolve(new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
    }
    return realFetch(input, init);
  };
}, FRAMES);
await page.goto(APP_URL, { waitUntil: "load", timeout: 30000 });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(1000);
await page.waitForSelector("textarea", { timeout: 10000 });
await page.type("textarea", "Which blood pressure measurements define hypertension?", { delay: 1 });
await page.keyboard.press("Enter");
const deadline = Date.now() + 30000;
while (Date.now() < deadline) { if (await page.evaluate(() => (window.__PERF || {}).streamEnd || 0)) break; await sleep(200); }
await sleep(1200);
const result = await page.evaluate(() => document.body.innerText.replace(/\s+/g, " "));
const checks = {};
for (const e of EXPECT) checks[e] = result.includes(e);
console.log(JSON.stringify({ fixture: file, checks, text: result.slice(0, 700), errors: msgs.slice(0, 3) }, null, 2));
await browser.close();
