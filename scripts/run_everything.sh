#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s --smoke [--work-harder|--work-hardest] [TRAINING_DIR STREAM_FILE MODEL_FILE PREVIEW_FILE PER_BUCKET_LIMIT EPOCHS]\n' "$0"
  printf '\n'
  printf 'Modes:\n'
  printf '  --smoke         Rebuild probe data, stitch a balanced smoke stream, train, and visualize rows 0..15000.\n'
  printf '\n'
  printf 'Effort presets:\n'
  printf '  default         PER_BUCKET_LIMIT=8   EPOCHS=3\n'
  printf '%s\n' '  --work-harder  PER_BUCKET_LIMIT=32  EPOCHS=10'
  printf '%s\n' '  --work-hardest PER_BUCKET_LIMIT=128 EPOCHS=25'
}

MODE=""
PER_BUCKET_LIMIT_DEFAULT=8
EPOCHS_DEFAULT=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      MODE="smoke"
      shift
      ;;
    --work-harder)
      PER_BUCKET_LIMIT_DEFAULT=32
      EPOCHS_DEFAULT=10
      shift
      ;;
    --work-hardest)
      PER_BUCKET_LIMIT_DEFAULT=128
      EPOCHS_DEFAULT=25
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ "$MODE" != "smoke" ]]; then
  printf 'Missing mode. Use --smoke.\n\n' >&2
  usage >&2
  exit 2
fi

TRAINING_DIR="${1:-data/training/snn_comprehensive}"
STREAM_FILE="${2:-data/streams/smoke_stream.csv}"
MODEL_FILE="${3:-data/models/snn_stream_smoke.nsm}"
PREVIEW_FILE="${4:-data/streams/stream_preview.svg}"
PER_BUCKET_LIMIT="${5:-$PER_BUCKET_LIMIT_DEFAULT}"
EPOCHS="${6:-$EPOCHS_DEFAULT}"

printf 'Run mode=smoke per_bucket_limit=%s epochs=%s preview_rows=0..15000\n' "$PER_BUCKET_LIMIT" "$EPOCHS"

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
