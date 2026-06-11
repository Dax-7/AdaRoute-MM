#!/usr/bin/env sh
set -eu

BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
MAX_ATTEMPTS="${OLLAMA_WAIT_ATTEMPTS:-120}"
SLEEP_SECONDS="${OLLAMA_WAIT_SLEEP_SECONDS:-2}"

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  if curl -fsS "$BASE_URL/api/tags" >/dev/null 2>&1; then
    echo "Ollama is available at $BASE_URL"
    exit 0
  fi
  echo "Waiting for Ollama at $BASE_URL ($attempt/$MAX_ATTEMPTS)..."
  attempt=$((attempt + 1))
  sleep "$SLEEP_SECONDS"
done

echo "Timed out waiting for Ollama at $BASE_URL" >&2
exit 1
