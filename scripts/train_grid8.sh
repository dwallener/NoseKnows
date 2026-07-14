#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/views/peak_single_note}"
OUT_FILE="${2:-data/models/grid8_readout.ngm}"
EPOCHS="${3:-250}"
LOOKBACK_SECS="${4:-8}"

cargo run --bin grid_train -- \
  --data "$DATA_DIR" \
  --out "$OUT_FILE" \
  --epochs "$EPOCHS" \
  --lookback-secs "$LOOKBACK_SECS"
