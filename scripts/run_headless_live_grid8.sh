#!/usr/bin/env bash
set -euo pipefail

STATE_FILE="${1:-data/live/injector_state.json}"
INPUT_FILE="${2:-data/live/input_frames.csv}"
MODEL_FILE="${3:-data/models/grid8_readout.ngm}"
RESULTS_FILE="${4:-data/live/grid_model_results.csv}"
EVENTS_FILE="${5:-data/live/grid_events.csv}"
EMBEDDINGS_FILE="${6:-data/live/grid_embeddings.csv}"
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
  --run-id "$RUN_ID"

printf 'Grid input:      %s\n' "$INPUT_FILE"
printf 'Grid results:    %s\n' "$RESULTS_FILE"
printf 'Grid events:     %s\n' "$EVENTS_FILE"
printf 'Grid embeddings: %s\n' "$EMBEDDINGS_FILE"
