// Headless UI smoke test for the MedRAG frontend (assistant-ui based).
// Drives the real xdeep backend through the Vite dev proxy and captures screenshots.
import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";

const BASE = process.env.UI_URL ?? "http://localhost:5174";
const OUT = new URL("../.ui-shots/", import.meta.url).pathname;
mkdirSync(OUT, { recursive: true });

const QUESTION =
  "Is proenkephalin (PENK) predictive of acute kidney injury after cardiac surgery?";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium",
  headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
  defaultViewport: { width: 1440, height: 900 },
});

const page = await browser.newPage();
const consoleErrors = [];
page.on("console", (msg) => {
  if (msg.type() === "error") consoleErrors.push(msg.text().slice(0, 300));
});
page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));

const shot = (name) => page.screenshot({ path: `${OUT}${name}.png` });
const has = async (selector, timeout = 4000) => {
  try {
    await page.waitForSelector(selector, { timeout });
    return true;
  } catch {
    return false;
  }
};

// 1. Empty state
await page.goto(BASE, { waitUntil: "load", timeout: 30000 });
await sleep(900);
console.log("empty state ✓", (await page.evaluate(() => document.body.innerText)).slice(0, 80).replace(/\s+/g, " "));
await shot("01-empty");

// 2. Send a question
await page.type("textarea", QUESTION, { delay: 4 });
await sleep(150);
await shot("02-composed");
await page.keyboard.press("Enter");

// 3. Streaming: thinking panel opens live with tool cards
const sawThinking = await has("::-p-text(Thinking)", 15000);
console.log("live thinking panel:", sawThinking);
await sleep(5000);
await shot("03-streaming");

// 4. Wait for the run to finish (answer + sources + collapse)
console.log("waiting for completion…");
try {
  await page.waitForFunction(
    () => {
      const text = document.body.innerText ?? "";
      return text.includes("Sources") && !text.includes("Working…");
    },
    { timeout: 300000, polling: 2000 },
  );
  console.log("run completed ✓");
} catch {
  console.log("completion wait timed out");
}
await sleep(1500);
await shot("04-complete");

// panel should collapse to "View thoughts" once idle
const collapsed = await has("::-p-text(View thoughts)", 5000);
console.log("panel collapsed after run:", collapsed);
await shot("05-collapsed");

// sidebar: thread auto-titled, xdeep badge
const sidebar = await page.evaluate(() => {
  const t = document.body.innerText;
  return {
    titled: /proenkephalin/i.test(t),
    xdeep: t.includes("xdeep") && t.includes("deep research"),
  };
});
console.log("sidebar titled:", sidebar.titled, "| xdeep badge:", sidebar.xdeep);
await shot("06-sidebar");

// new chat
const newChat = await page.$( "::-p-text(New chat)");
if (newChat) await newChat.click();
await sleep(700);
await shot("07-new-chat");

console.log("console errors:", consoleErrors.length ? consoleErrors.slice(0, 5) : "none");
await browser.close();
console.log("done — screenshots in", OUT);