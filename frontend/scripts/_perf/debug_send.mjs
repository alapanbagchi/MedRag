// Drive the real app + real backend and report what happens after Send.
import puppeteer from "puppeteer-core";

const APP_URL = process.env.APP_URL || "http://localhost:5173";
const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium", headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--disable-background-timer-throttling", "--disable-renderer-backgrounding"],
  defaultViewport: { width: 1440, height: 900 },
});
const page = await browser.newPage();
const logs = [];
const reqs = [];
page.on("console", (m) => logs.push("[" + m.type() + "] " + m.text().slice(0, 300)));
page.on("pageerror", (e) => logs.push("[pageerror] " + e.message.slice(0, 400)));
page.on("request", (r) => { if (r.url().includes("/v1/")) reqs.push("REQ " + r.method() + " " + r.url()); });
page.on("response", (r) => { if (r.url().includes("/v1/")) reqs.push("RES " + r.status() + " " + r.url()); });
page.on("requestfailed", (r) => { if (r.url().includes("/v1/")) reqs.push("FAIL " + r.url() + " " + (r.failure()?.errorText || "")); });

await page.goto(APP_URL, { waitUntil: "load", timeout: 30000 });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(1200);
await page.waitForSelector("textarea", { timeout: 10000 });
const before = await page.evaluate(() => {
  const ta = document.querySelector("textarea");
  return {
    placeholder: ta ? ta.getAttribute("placeholder") : null,
    disabled: ta ? ta.disabled : null,
    sendButtons: [...document.querySelectorAll("button")].filter((b) => (b.getAttribute("aria-label") || "") === "Send").length,
    cancelButtons: [...document.querySelectorAll("button")].filter((b) => (b.getAttribute("aria-label") || "") === "Stop").length,
  };
});
await page.type("textarea", "What is hypertension?", { delay: 3 });
await page.keyboard.press("Enter");
console.log("pressed enter");
await sleep(9000);
const after = await page.evaluate(() => {
  const text = document.body.innerText.replace(/\s+/g, " ");
  return {
    text: text.slice(0, 600),
    hasUserMsg: text.includes("What is hypertension?"),
    textareas: document.querySelectorAll("textarea").length,
    sendButtons: [...document.querySelectorAll("button")].filter((b) => (b.getAttribute("aria-label") || "") === "Send").length,
    cancelButtons: [...document.querySelectorAll("button")].filter((b) => (b.getAttribute("aria-label") || "") === "Stop").length,
    agentPanel: !!document.querySelector("[data-slot='agent-panel']"),
    hasErrorBanner: text.includes("Run failed"),
  };
});
console.log(JSON.stringify({ before, after, reqs: reqs.slice(0, 12), logs: logs.slice(0, 12) }, null, 2));
await browser.close();
