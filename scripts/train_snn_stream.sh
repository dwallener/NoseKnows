#!/usr/bin/env bash
set -euo pipefail

STREAM_FILE="${1:-data/streams/snn_comprehensive_stream.csv}"
OUT_MODEL="${2:-data/models/snn_stream_readout.nsm}"
EPOCHS="${3:-50}"

cargo run --bin snn_stream_train -- \
  --stream "$STREAM_FILE" \
  --out "$OUT_MODEL" \
  --epochs "$EPOCHS" \
  --validation 0.2 \
  --window 30 \
  --stride 1 \
  --silence-weight 3.0
