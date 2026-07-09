#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-data/training/snn_comprehensive}"
OUT_FILE="${2:-data/streams/snn_comprehensive_stream.csv}"
INITIAL_NO_SCENT="${3:-3}"
MAX_GAP_NO_SCENT="${4:-3}"
LIMIT="${5:-}"

if [[ -n "$LIMIT" ]]; then
  cargo run --bin stitch_stream -- \
    --input "$INPUT_DIR" \
    --out "$OUT_FILE" \
    --initial-no-scent-captures "$INITIAL_NO_SCENT" \
    --max-gap-no-scent-captures "$MAX_GAP_NO_SCENT" \
    --limit "$LIMIT"
else
  cargo run --bin stitch_stream -- \
    --input "$INPUT_DIR" \
    --out "$OUT_FILE" \
    --initial-no-scent-captures "$INITIAL_NO_SCENT" \
    --max-gap-no-scent-captures "$MAX_GAP_NO_SCENT"
fi
