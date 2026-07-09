#!/usr/bin/env bash
set -euo pipefail

STREAM_FILE="${1:-data/streams/smoke_stream.csv}"
MODEL_FILE="${2:-data/models/peak_pair_readout.npm}"

cargo run --bin peak_stream_eval -- \
  --stream "$STREAM_FILE" \
  --model "$MODEL_FILE" \
  --gate-threshold 0.0
