import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";
const OUT = "/tmp/ui-verify/"; mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium", headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
  defaultViewport: { width: 1600, height: 980, deviceScaleFactor: 2 },
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("pageerror:", e.message));
await page.goto("http://localhost:5200", { waitUntil: "load", timeout: 30000 });
await sleep(1200);
await page.waitForSelector("textarea", { timeout: 10000 });
await page.type("textarea", "Is proenkephalin (PENK) predictive of acute kidney injury after cardiac surgery?", { delay: 3 });
await page.keyboard.press("Enter");
console.log("sent");
try {
  await page.waitForFunction(() => document.body.innerText.includes("Plan of action"), { timeout: 240000, polling: 1500 });
  console.log("plan published");
} catch { console.log("plan never appeared"); }
await sleep(8000);
try { const b = await page.waitForSelector("::-p-text(Click to see the agentic flow)", { timeout: 20000 }); await b.click(); console.log("sheet opened"); }
catch { console.log("trigger not found"); }
const sel = (i) => page.evaluate((idx) => {
  const btns = Array.from(document.querySelectorAll('[data-slot="agent-panel"] button')).filter((b) => b.className.includes("anim-agent-in"));
  btns[idx]?.click();
}, i);
const txt = () => page.evaluate(() => { const p = document.querySelector('[data-slot="agent-panel"]'); return p ? p.innerText.replace(/\s+/g, " ").trim() : ""; });
const cnt = () => page.evaluate(() => Array.from(document.querySelectorAll('[data-slot="agent-panel"] button')).filter((b) => b.className.includes("anim-agent-in")).length);
let seen = false;
for (let t = 0; t < 120 && !seen; t++) {
  const n = await cnt();
  for (let i = 0; i < n; i++) {
    await sel(i); await sleep(150);
    const s = await txt();
    if (s.includes("Can you search the web for") && s.includes("I found some sources online")) {
      console.log(`WEB pane ${i}:`, s.slice(0, 900));
      await page.screenshot({ path: OUT + "web-search.png" });
      seen = true; break;
    }
  }
  if (!seen) await sleep(2000);
}
if (!seen) console.log("web flow not seen");
await browser.close(); console.log("done");
