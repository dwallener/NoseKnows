#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s --smoke [--work-harder|--work-hardest] [TRAINING_DIR STREAM_FILE MODEL_FILE PREVIEW_FILE PER_BUCKET_LIMIT EPOCHS]\n' "$0"
  printf '\n'
  printf 'Modes:\n'
  printf '  --smoke         Rebuild probe data, stitch a balanced smoke stream, train, and render static + interactive views.\n'
  printf '\n'
  printf 'Effort presets:\n'
  printf '  default         PER_BUCKET_LIMIT=8    EPOCHS=3   VIEWER_ROWS=20000\n'
  printf '%s\n' '  --work-harder  PER_BUCKET_LIMIT=32   EPOCHS=100   VIEWER_ROWS=200000'
  printf '%s\n' '  --work-hardest PER_BUCKET_LIMIT=300  EPOCHS=1000  VIEWER_ROWS=2000000'
  printf '\n'
  printf 'PER_BUCKET_LIMIT is the number of scent captures stitched from each balanced bucket: 1-note, 2-note, and 3-note.\n'
}

MODE=""
PER_BUCKET_LIMIT_DEFAULT=8
EPOCHS_DEFAULT=3
VIEWER_ROWS_DEFAULT=20000
MAX_BUCKETS_DEFAULT=12000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      MODE="smoke"
      shift
      ;;
    --work-harder)
      PER_BUCKET_LIMIT_DEFAULT=32
      EPOCHS_DEFAULT=100
      VIEWER_ROWS_DEFAULT=200000
      MAX_BUCKETS_DEFAULT=20000
      shift
      ;;
    --work-hardest)
      PER_BUCKET_LIMIT_DEFAULT=300
      EPOCHS_DEFAULT=1000
      VIEWER_ROWS_DEFAULT=2000000
      MAX_BUCKETS_DEFAULT=30000
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
VIEWER_FILE="data/streams/stream_viewer.html"
VIEWER_ROWS="$VIEWER_ROWS_DEFAULT"
MAX_BUCKETS="$MAX_BUCKETS_DEFAULT"

printf 'Run mode=smoke per_bucket_limit=%s epochs=%s preview_rows=0..15000 viewer_rows=0..%s max_buckets=%s\n' "$PER_BUCKET_LIMIT" "$EPOCHS" "$VIEWER_ROWS" "$MAX_BUCKETS"

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

cargo run --bin stream_viewer -- \
  --stream "$STREAM_FILE" \
  --model "$MODEL_FILE" \
  --out "$VIEWER_FILE" \
  --start-row 0 \
  --rows "$VIEWER_ROWS" \
  --columns 900 \
  --max-buckets "$MAX_BUCKETS"

printf 'Smoke stream preview: %s\n' "$PREVIEW_FILE"
printf 'Smoke stream viewer: %s\n' "$VIEWER_FILE"
