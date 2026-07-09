#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="${1:-data/training/snn_comprehensive}"
STREAM_FILE="${2:-data/streams/smoke_stream.csv}"
MODEL_FILE="${3:-data/models/snn_stream_smoke.nsm}"
PREVIEW_FILE="${4:-data/streams/stream_preview.svg}"
PER_BUCKET_LIMIT="${5:-8}"
EPOCHS="${6:-3}"

scripts/build_snn_training_dataset.sh "$TRAINING_DIR" 4 100
scripts/build_snn_stream_dataset.sh "$TRAINING_DIR" "$STREAM_FILE" 3 3 "$PER_BUCKET_LIMIT"

cargo run --bin snn_stream_train -- \
  --stream "$STREAM_FILE" \
  --out "$MODEL_FILE" \
  --epochs "$EPOCHS" \
  --validation 0.2 \
  --window 30 \
  --stride 1 \
  --silence-weight 3.0

scripts/viz_snn_stream.sh "$STREAM_FILE" "$MODEL_FILE" "$PREVIEW_FILE" 0 15000

printf 'Smoke stream preview: %s\n' "$PREVIEW_FILE"
