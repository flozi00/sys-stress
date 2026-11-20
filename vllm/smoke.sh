#!/usr/bin/env bash
# One real request against the served model: latency, tokens, sample output.
#
#   export VLLM_HOST=user@host
#   ./smoke.sh
#   ./smoke.sh "Schreibe ein Haiku über Grace Blackwell."
set -euo pipefail

: "${VLLM_HOST:?set VLLM_HOST=user@host (never commit it)}"
PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
PROMPT="${1:-Explain in two sentences why Grace-Blackwell can offload experts to CPU memory.}"

ssh "${VLLM_HOST}" bash -s <<REMOTE
set -euo pipefail
curl -s -m 600 "http://localhost:${PORT}/v1/chat/completions" \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "messages": [{"role": "user", "content": $(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],
    "max_tokens": 256,
    "temperature": 0.2
  }' | python3 -c '
import json, sys
d = json.load(sys.stdin)
if "error" in d:
    print("ERROR:", d["error"]); sys.exit(1)
u = d.get("usage", {})
print("tokens:", u.get("prompt_tokens"), "->", u.get("completion_tokens"))
print("---")
print(d["choices"][0]["message"]["content"][:1200])
'
REMOTE
