#!/usr/bin/env bash
set -euo pipefail

STATE_FILE="data/live/injector_state.json"
INPUT_FILE="data/live/input_frames.csv"
MODEL_FILE="data/models/grid8_readout.ngm"
RESULTS_FILE="data/live/grid_model_results.csv"
EVENTS_FILE="data/live/grid_events.csv"
EMBEDDINGS_FILE="data/live/grid_embeddings.csv"
EXTRA_ARGS=()

if [[ $# -gt 0 && "$1" != --* ]]; then
  STATE_FILE="${1:-$STATE_FILE}"
  INPUT_FILE="${2:-$INPUT_FILE}"
  MODEL_FILE="${3:-$MODEL_FILE}"
  RESULTS_FILE="${4:-$RESULTS_FILE}"
  EVENTS_FILE="${5:-$EVENTS_FILE}"
  EMBEDDINGS_FILE="${6:-$EMBEDDINGS_FILE}"
  EXTRA_ARGS=("${@:7}")
else
  EXTRA_ARGS=("$@")
fi
RUN_ID="${RUN_ID:-grid_live_headless_smoke}"

python3 tools/live/inject_chunks.py \
  --state "$STATE_FILE" \
  --out "$INPUT_FILE" \
  --events data/live/input_events.csv

cargo run --bin grid_live_headless -- \
  --input "$INPUT_FILE" \
  --model "$MODEL_FILE" \
  --out-results "$RESULTS_FILE" \
  --out-events "$EVENTS_FILE" \
  --out-embeddings "$EMBEDDINGS_FILE" \
  --gate-threshold 0.0 \
  --run-id "$RUN_ID" \
  "${EXTRA_ARGS[@]}"

printf 'Grid input:      %s\n' "$INPUT_FILE"
printf 'Grid results:    %s\n' "$RESULTS_FILE"
printf 'Grid events:     %s\n' "$EVENTS_FILE"
printf 'Grid embeddings: %s\n' "$EMBEDDINGS_FILE"
