import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";

const OUT = "/tmp/ui-verify/";
mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium",
  headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
  defaultViewport: { width: 1500, height: 950, deviceScaleFactor: 2 },
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("pageerror:", e.message));
await page.goto("http://localhost:5200", { waitUntil: "load", timeout: 30000 });
await sleep(1200);
await page.waitForSelector("textarea", { timeout: 10000 });
await page.type("textarea", "How does diet affect hypertension?", { delay: 3 });
await page.keyboard.press("Enter");
console.log("sent");

const hasComposer = () => page.evaluate(() => !!document.querySelector('[data-slot="thread-footer"] textarea'));
const bodyText = () => page.evaluate(() => document.body.innerText.replace(/\s+/g, " "));

let seen = false;
for (let i = 0; i < 70 && !seen; i++) {
  const t = await bodyText();
  if (t.includes("Send my answer") || t.includes("Help me understand")) {
    seen = true;
    break;
  }
  await sleep(2000);
}
console.log("question form:", seen, "| composer visible:", await hasComposer());
if (seen) {
  await page.screenshot({ path: OUT + "ask-open.png" });
  await page.evaluate(() => {
    const labels = Array.from(document.querySelectorAll("label"));
    const seenNames = new Set();
    for (const l of labels) {
      const input = l.querySelector("input[type=checkbox],input[type=radio]");
      if (!input) continue;
      const name = input.getAttribute("name") || "";
      if (seenNames.has(name)) continue;
      seenNames.add(name);
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
  await sleep(1200);
  console.log("after answer | composer visible:", await hasComposer());
  await page.screenshot({ path: OUT + "ask-answered.png" });
}
await browser.close();
console.log("done");
