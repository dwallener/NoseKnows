#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/training/snn_comprehensive}"
VARIANTS="${2:-4}"
NO_SCENT_SAMPLES="${3:-100}"

cargo run --bin synthesize -- \
  --probe all \
  --out "$OUT_DIR" \
  --variants "$VARIANTS" \
  --no-scent-samples "$NO_SCENT_SAMPLES"

find "$OUT_DIR" -maxdepth 1 -name '*.csv' -type f | wc -l
