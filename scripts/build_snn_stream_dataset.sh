#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-data/training/snn_comprehensive}"
OUT_FILE="${2:-data/streams/snn_comprehensive_stream.csv}"
NO_SCENT_RATIO="${3:-0.5}"
LIMIT="${4:-}"

if [[ -n "$LIMIT" ]]; then
  cargo run --bin stitch_stream -- \
    --input "$INPUT_DIR" \
    --out "$OUT_FILE" \
    --no-scent-ratio "$NO_SCENT_RATIO" \
    --limit "$LIMIT"
else
  cargo run --bin stitch_stream -- \
    --input "$INPUT_DIR" \
    --out "$OUT_FILE" \
    --no-scent-ratio "$NO_SCENT_RATIO"
fi
