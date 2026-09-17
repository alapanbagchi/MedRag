import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";

const OUT = "/tmp/ui-verify/";
mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const QUESTION = process.env.Q ?? "How does diet affect hypertension?";

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium",
  headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
  defaultViewport: { width: 1600, height: 980, deviceScaleFactor: 2 },
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("pageerror:", e.message));
await page.goto("http://localhost:5200", { waitUntil: "load", timeout: 30000 });
await sleep(1200);
await page.waitForSelector("textarea", { timeout: 10000 });
await page.type("textarea", QUESTION, { delay: 3 });
await page.keyboard.press("Enter");
console.log("sent:", QUESTION);

const bodyText = () => page.evaluate(() => document.body.innerText.replace(/\s+/g, " "));
const hasComposer = () => page.evaluate(() => !!document.querySelector("textarea"));

// --- ask_user capture ---
let sawQuestion = false;
for (let i = 0; i < 60 && !sawQuestion; i++) {
  const t = await bodyText();
  if (t.includes("Help me understand what you want") || t.includes("Send my answer")) {
    sawQuestion = true;
    break;
  }
  await sleep(2000);
}
if (sawQuestion) {
  console.log("QUESTION FORM shown; composer present:", await hasComposer());
  await page.screenshot({ path: OUT + "ask-open.png" });
  // answer: select first option of each question, then submit
  await page.evaluate(() => {
    const labels = Array.from(document.querySelectorAll("label"));
    const seen = new Set();
    for (const l of labels) {
      const input = l.querySelector("input[type=checkbox],input[type=radio]");
      if (!input) continue;
      const name = input.getAttribute("name") || "";
      if (seen.has(name)) continue;
      seen.add(name);
      input.click();
    }
  });
  await sleep(400);
  await page.evaluate(() => {
    const btn = Array.from(document.querySelectorAll("button")).find((b) =>
      (b.innerText || "").includes("Send my answer"),
    );
    btn?.click();
  });
  await sleep(700);
  await page.screenshot({ path: OUT + "ask-answered.png" });
  console.log("answered; composer back:", await hasComposer());
} else {
  console.log("no question form shown");
}

// --- web search capture ---
try {
  await page.waitForFunction(() => document.body.innerText.includes("Plan of action"), { timeout: 240000, polling: 1500 });
  console.log("plan published");
} catch { console.log("plan never appeared"); }
await sleep(8000);
try {
  const btn = await page.waitForSelector("::-p-text(Click to see the agentic flow)", { timeout: 20000 });
  await btn.click();
  console.log("sheet opened");
} catch { console.log("trigger not found / already open"); }

const selectPane = (i) =>
  page.evaluate((idx) => {
    const btns = Array.from(document.querySelectorAll('[data-slot="agent-panel"] button')).filter((b) =>
      b.className.includes("anim-agent-in"),
    );
    btns[idx]?.click();
  }, i);
const paneText = () =>
  page.evaluate(() => {
    const p = document.querySelector('[data-slot="agent-panel"]');
    return p ? p.innerText.replace(/\s+/g, " ").trim() : "";
  });
const paneCount = () =>
  page.evaluate(() =>
    Array.from(document.querySelectorAll('[data-slot="agent-panel"] button')).filter((b) =>
      b.className.includes("anim-agent-in"),
    ).length,
  );

let sawWeb = false;
for (let tick = 0; tick < 90; tick++) {
  const n = await paneCount();
  for (let i = 0; i < n; i++) {
    await selectPane(i);
    await sleep(160);
    const txt = await paneText();
    if (txt.includes("Can you search the web for")) {
      console.log(`WEB pane ${i}:`, txt.slice(0, 700));
      await page.screenshot({ path: OUT + "web-search.png" });
      sawWeb = true;
      break;
    }
  }
  if (sawWeb) break;
  await sleep(2000);
}
if (!sawWeb) console.log("no web-search copy seen");
await browser.close();
console.log("done");
