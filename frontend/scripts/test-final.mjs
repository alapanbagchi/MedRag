// Definitive E2E: one xdeep run, browser stays open until the stream ends.
import puppeteer from "puppeteer-core";

const QUESTION =
  "Which specific blood pressure measurements define hypertension in routine adult care?";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium",
  headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
  defaultViewport: { width: 1440, height: 900 },
});
const page = await browser.newPage();
const errs = [];
page.on("console", (m) => {
  if (m.type() === "error") errs.push(m.text().slice(0, 300));
});
page.on("pageerror", (e) => errs.push(`pageerror: ${e.message}`));

await page.goto("http://localhost:5174", { waitUntil: "load", timeout: 30000 });
await sleep(900);
await page.type("textarea", QUESTION, { delay: 3 });
await page.keyboard.press("Enter");

try {
  await page.waitForSelector("::-p-text(Thinking)", { timeout: 25000 });
  console.log("live thinking panel ✓");
} catch {
  console.log("thinking panel missing!");
}
await sleep(4000);
await page.screenshot({ path: new URL("../.ui-shots/10-streaming.png", import.meta.url).pathname });

// wait for the final answer (token text) — the terminal signal is an answer
// beyond the plan + a collapsed panel, or the sources card.
const t0 = Date.now();
console.log("waiting for answer…");
try {
  await page.waitForFunction(
    () => {
      const text = document.body.innerText ?? "";
      return (text.includes("Sources") || /Threshold|mmHg|diagnos/i.test(text.split("Thinking")[0] ?? "")) && !text.includes("Working…");
    },
    { timeout: 900000, polling: 3000 },
  );
  console.log(`answer/sources seen after ${Math.round((Date.now() - t0) / 1000)}s`);
} catch {
  console.log("answer wait timed out");
}
await sleep(2000);
const finalText = await page.evaluate(() => document.body.innerText);
console.log("sources card:", finalText.includes("Sources"));
console.log("panel collapsed:", finalText.includes("View thoughts"));
console.log("answer length:", Math.max(...[0, finalText.indexOf("Sources") === -1 ? 800 : finalText.indexOf("Sources")]));
await page.screenshot({ path: new URL("../.ui-shots/11-complete.png", import.meta.url).pathname });
console.log("console errors:", errs.length ? errs.slice(0, 5) : "none");
await browser.close();
console.log("done");