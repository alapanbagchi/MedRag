import puppeteer from "puppeteer-core";

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium",
  headless: true,
  args: ["--no-sandbox", "--disable-gpu"],
});
const page = await browser.newPage();
const errs = [];
page.on("console", (m) => {
  if (m.type() === "error") errs.push(m.text().slice(0, 300));
});
page.on("pageerror", (e) => errs.push(`pageerror: ${e.message}`));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

await page.goto("http://localhost:5174", { waitUntil: "load", timeout: 30000 });
await sleep(1000);
await page.type(
  "textarea",
  "What blood pressure threshold defines hypertension in adult clinical practice?",
  { delay: 2 },
);
await page.keyboard.press("Enter");
await sleep(14000);

const dump = await page.evaluate(() => {
  const main = document.querySelector("main");
  return {
    text: (main?.innerText ?? "").slice(0, 2600),
    cards: [...(main?.querySelectorAll("button") ?? [])]
      .map((b) => (b.textContent ?? "").trim())
      .filter((t) => t.length > 0 && t.length < 60)
      .slice(0, 15),
  };
});
console.log("=== TEXT ===\n", dump.text);
console.log("=== BUTTON LABELS ===\n", dump.cards.join(" | "));
console.log("ERRORS:", errs.slice(0, 4).join("\n"));
await browser.close();