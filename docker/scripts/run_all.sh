#!/usr/bin/env sh
set -eu

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
export PYTHONPATH="${PYTHONPATH:-/app}"

DATASET="${ADAROUTE_DATASET:-data/datasets/v3_1_text_fusion/fusion_v3_1_1000_200-200-400-100-100.jsonl}"
RESULTS_DIR="${ADAROUTE_RESULTS_DIR:-results/docker_8gb}"
RUN_ID="${ADAROUTE_RUN_ID:-docker_8gb_$(date +%Y%m%d-%H%M%S)}"
SAMPLE_SIZE="${ADAROUTE_SAMPLE_SIZE:-100}"
CONFIG="${ADAROUTE_CONFIG:-configs/default.yaml}"
OVERRIDE_CONFIG="${ADAROUTE_OVERRIDE_CONFIG:-configs/v3_2_text.yaml}"
PROMPTS="${ADAROUTE_PROMPTS:-configs/prompts_v3_2.yaml}"
MODES="${ADAROUTE_MODES:-always_small,always_middle,always_gemma,difficulty_routing,random_routing,adaroute_mm_full}"

/app/docker/scripts/check_ollama.sh
python scripts/check_ollama_models.py --config "$CONFIG" --override-config "$OVERRIDE_CONFIG"

NO_RESUME_FLAG=""
if [ "${ADAROUTE_NO_RESUME:-0}" = "1" ]; then
  NO_RESUME_FLAG="--no-resume"
fi

python /app/docker/scripts/run_docker_8gb_suite.py \
  --dataset "$DATASET" \
  --results-dir "$RESULTS_DIR" \
  --run-id "$RUN_ID" \
  --sample-size "$SAMPLE_SIZE" \
  --config "$CONFIG" \
  --override-config "$OVERRIDE_CONFIG" \
  --prompts "$PROMPTS" \
  --modes "$MODES" \
  $NO_RESUME_FLAG
