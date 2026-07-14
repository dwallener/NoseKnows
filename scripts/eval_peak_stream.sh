#!/usr/bin/env bash
set -euo pipefail

STREAM_FILE="${1:-data/streams/smoke_stream.csv}"
MODEL_FILE="${2:-data/models/peak_pair_readout.npm}"
RUN_DIR="${3:-data/runs/peak_stream}"
RUN_ID="${4:-peak_stream_smoke}"
RESULTS_FILE="$RUN_DIR/results.csv"

cargo run --bin peak_stream_eval -- \
  --stream "$STREAM_FILE" \
  --model "$MODEL_FILE" \
  --gate-threshold 0.0 \
  --out-results "$RESULTS_FILE" \
  --run-id "$RUN_ID"

printf 'Peak stream results: %s\n' "$RESULTS_FILE"
