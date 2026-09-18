#!/usr/bin/env bash
set -uo pipefail
cd /home/alapanbagchi/PycharmProjects/MedRag/backend
set -a; . ./.env; set +a
for m in omen-alpha union-alpha qwen3.7-plus; do
  echo "== $m =="
  code=$(curl -s -m 40 -o /tmp/llm_test.json -w "%{http_code}" "${LLM_BASE_URL%/}/chat/completions" \
    -H "Authorization: Bearer ${LLM_API_KEY}" -H "Content-Type: application/json" \
    -H "x-opencode-session: perf-check" \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":5}")
  echo "http $code"
  head -c 300 /tmp/llm_test.json; echo
done