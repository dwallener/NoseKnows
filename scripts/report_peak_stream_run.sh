#!/usr/bin/env bash
set -euo pipefail

RESULTS_FILE="${1:-data/runs/peak_stream/results.csv}"
MANIFEST_FILE="${2:-data/manifest/captures.csv}"
OUT_DIR="${3:-data/runs/peak_stream/report}"

ARGS=()
if [[ "${NOSEKNOWS_ALLOW_STDLIB_DATASET:-}" == "1" ]]; then
  ARGS+=(--allow-stdlib-fallback)
fi

python3 tools/dataset/analyze_results.py "$RESULTS_FILE" \
  --manifest "$MANIFEST_FILE" \
  --out-dir "$OUT_DIR" \
  "${ARGS[@]}"
