// Long-run completion test: waits (up to 9 min) for the xdeep research run
// to finish end-to-end, then checks answer + sources + collapsed panel.
import puppeteer from "puppeteer-core";

const BASE = process.env.UI_URL ?? "http://localhost:5174";
const QUESTION =
  "What is the diagnostic threshold that defines hypertension in adult clinical practice?";

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

await page.goto(BASE, { waitUntil: "load", timeout: 30000 });
await sleep(800);
await page.type("textarea", QUESTION, { delay: 3 });
await page.keyboard.press("Enter");

const sawThinking = await (async () => {
  try {
    await page.waitForSelector("::-p-text(Thinking)", { timeout: 20000 });
    return true;
  } catch {
    return false;
  }
})();
console.log("live thinking panel:", sawThinking);
console.log("waiting for completion (up to 540s)…");
const t0 = Date.now();
try {
  await page.waitForFunction(
    () => {
      const text = document.body.innerText ?? "";
      return text.includes("Sources") && !text.includes("Working…");
    },
    { timeout: 540000, polling: 3000 },
  );
  console.log(`run completed ✓ after ${Math.round((Date.now() - t0) / 1000)}s`);
} catch {
  console.log("completion wait timed out");
}
await sleep(1200);
const finalText = await page.evaluate(() => document.body.innerText);
console.log("has Sources card:", finalText.includes("Sources"));
console.log("has answer text:", /\S{120,}/.test(finalText.split("Sources")[0] ?? ""));
console.log("panel collapsed:", finalText.includes("View thoughts"));
console.log("console errors:", errs.length ? errs.slice(0, 5) : "none");
await page.screenshot({ path: new URL("../.ui-shots/08-complete-xdeep.png", import.meta.url).pathname });
await browser.close();
console.log("done");