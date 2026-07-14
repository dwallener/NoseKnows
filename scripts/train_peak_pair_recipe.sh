#!/usr/bin/env bash
set -euo pipefail

RECIPE="${1:-recipes/peak_single_note.toml}"
OUT_FILE="${2:-data/models/peak_pair_readout.npm}"
EPOCHS="${3:-250}"
HOLD_SECS="${4:-8}"

MATERIALIZE_ARGS=()
if [[ "${NOSEKNOWS_ALLOW_STDLIB_DATASET:-}" == "1" ]]; then
  MATERIALIZE_ARGS+=(--allow-stdlib-fallback)
fi

DATA_DIR="$(python3 tools/dataset/materialize.py "$RECIPE" --print-output-dir "${MATERIALIZE_ARGS[@]}")"

cargo run --bin peak_train -- \
  --data "$DATA_DIR" \
  --out "$OUT_FILE" \
  --epochs "$EPOCHS" \
  --validation 0.2 \
  --hold-secs "$HOLD_SECS"
