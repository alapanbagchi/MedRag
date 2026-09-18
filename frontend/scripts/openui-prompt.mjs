#!/usr/bin/env node
/**
 * Generate the OpenUI artifacts from src/openui/library.tsx.
 *
 * Per the OpenUI docs the serialized library spec is the artifact, and the
 * model-facing prompt is rendered from it with generateSystemPrompt:
 *
 *   1. the OpenUI CLI bundles library.tsx and emits the serialized spec;
 *   2. generateSystemPrompt({ library: spec, promptOptions }) renders the prompt.
 *
 *   node scripts/openui-prompt.mjs          write the artifacts
 *   node scripts/openui-prompt.mjs --check  fail if the committed files are stale
 *
 * The prompt lands in backend/src/prompts/openui_answer.txt (loaded by the
 * Python synthesizer); the spec lands in src/openui/library-spec.json. The
 * renderer library, the spec and the prompt are one contract, so regenerating
 * is required whenever the library or prompt-options.mjs changes —
 * `openui:check` is the guard against drift.
 */

import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { generateSystemPrompt } from "@openuidev/lang-core";
import { promptOptions } from "../src/openui/prompt-options.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const frontend = path.resolve(here, "..");
const repo = path.resolve(frontend, "..");

const ENTRY = "src/openui/library.tsx";
const CLI = path.join(frontend, "node_modules", ".bin", "openui");
const PROMPT_OUT = path.join(repo, "backend/src/prompts/openui_answer.txt");
const SPEC_OUT = path.join(frontend, "src/openui/library-spec.json");

const check = process.argv.includes("--check");

const tmp = mkdtempSync(path.join(tmpdir(), "openui-prompt-"));
// The CLI writes the PROMPT to --out and the spec JSON alongside it with a
// .spec.json extension. We only keep the spec; the prompt is regenerated from
// it below with generateSystemPrompt.
const tmpPrompt = path.join(tmp, "prompt.txt");
const tmpSpec = path.join(tmp, "prompt.spec.json");

try {
  // 1. The CLI resolves the library through esbuild and serializes the spec
  //    to the path given by --out. That spec is the artifact.
  execFileSync(CLI, ["generate", ENTRY, "--out", tmpPrompt], {
    cwd: frontend,
    stdio: ["ignore", "ignore", "inherit"],
    env: { ...process.env, DO_NOT_TRACK: "1" },
  });

  const spec = readFileSync(tmpSpec, "utf8");

  // 2. The prompt is derived from the spec, not read from CLI output. One spec
  //    in, one prompt out — the documented backend integration.
  const prompt =
    generateSystemPrompt({
      library: JSON.parse(spec),
      promptOptions,
    }) + "\n";

  if (check) {
    const stale = [];
    for (const [label, file, next] of [
      ["prompt", PROMPT_OUT, prompt],
      ["spec", SPEC_OUT, spec],
    ]) {
      if (!existsSync(file)) {
        stale.push(`${path.relative(repo, file)} (missing)`);
        continue;
      }
      if (readFileSync(file, "utf8") !== next) {
        stale.push(path.relative(repo, file));
      }
    }
    if (stale.length > 0) {
      console.error(
        "OpenUI artifacts are stale:\n  " +
          stale.join("\n  ") +
          "\n" +
          "Run: npm --prefix frontend run openui:prompt",
      );
      process.exit(1);
    }
    console.log("OpenUI artifacts are up to date.");
  } else {
    writeFileSync(PROMPT_OUT, prompt);
    writeFileSync(SPEC_OUT, spec);
    console.log("wrote " + path.relative(repo, PROMPT_OUT));
    console.log("wrote " + path.relative(repo, SPEC_OUT));
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
