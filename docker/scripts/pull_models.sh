#!/usr/bin/env sh
set -eu

export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

MODELS="${ADAROUTE_ROUTER_MODEL:-qwen2.5:1.5b} ${ADAROUTE_SMALL_MODEL:-qwen2.5:1.5b} ${ADAROUTE_MEDIUM_MODEL:-phi3:latest} ${ADAROUTE_LARGE_MODEL:-sam860/gemma3n:e2b-Q3_K_XL} ${ADAROUTE_VLM_MODEL:-moondream:latest}"

attempt=1
while [ "$attempt" -le 120 ]; do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  echo "Waiting for local Ollama service ($attempt/120)..."
  attempt=$((attempt + 1))
  sleep 2
done

seen=" "
for model in $MODELS; do
  case "$seen" in
    *" $model "*) continue ;;
  esac
  seen="$seen$model "
  echo "Pulling Ollama model: $model"
  ollama pull "$model"
done

echo "Configured Ollama models are present."
