#!/usr/bin/env bash
set -euo pipefail

STATE_FILE="${1:-data/live/injector_state.json}"
INPUT_FILE="${2:-data/live/input_frames.csv}"
MODEL_FILE="${3:-data/models/peak_pair_readout.npm}"
RESULTS_FILE="${4:-data/live/model_results.csv}"
EVENTS_FILE="${5:-data/live/events.csv}"
RUN_ID="${RUN_ID:-live_headless_smoke}"

python3 tools/live/inject_chunks.py \
  --state "$STATE_FILE" \
  --out "$INPUT_FILE" \
  --events data/live/input_events.csv

cargo run --bin live_headless -- \
  --input "$INPUT_FILE" \
  --model "$MODEL_FILE" \
  --out-results "$RESULTS_FILE" \
  --out-events "$EVENTS_FILE" \
  --gate-threshold 0.0 \
  --run-id "$RUN_ID"

printf 'Live input:   %s\n' "$INPUT_FILE"
printf 'Live results: %s\n' "$RESULTS_FILE"
printf 'Live events:  %s\n' "$EVENTS_FILE"
