#!/usr/bin/env bash
set -euo pipefail

STREAM_FILE="${1:-data/streams/smoke_stream.csv}"
MODEL_FILE="${2:-data/models/snn_stream_smoke.nsm}"
OUT_FILE="${3:-data/streams/stream_preview.svg}"
START_ROW="${4:-0}"
ROWS="${5:-3000}"

cargo run --bin stream_viz -- \
  --stream "$STREAM_FILE" \
  --model "$MODEL_FILE" \
  --out "$OUT_FILE" \
  --start-row "$START_ROW" \
  --rows "$ROWS" \
  --columns 900
