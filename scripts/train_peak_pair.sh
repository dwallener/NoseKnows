#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/training/snn_comprehensive}"
OUT_FILE="${2:-data/models/peak_pair_readout.npm}"
EPOCHS="${3:-250}"
HOLD_SECS="${4:-8}"

cargo run --bin peak_train -- \
  --data "$DATA_DIR" \
  --out "$OUT_FILE" \
  --epochs "$EPOCHS" \
  --validation 0.2 \
  --hold-secs "$HOLD_SECS"
