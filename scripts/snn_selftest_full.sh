#!/usr/bin/env bash
set -uo pipefail

MODEL="${1:-data/models/snn_accordion_single_note_probe.nsm}"
BASE_DIR="${2:-data/selftest/generated}"
status=0

cargo run --bin synthesize -- --probe no-scent --out "$BASE_DIR/no_scent" || exit 1
cargo run --bin synthesize -- --probe single --out "$BASE_DIR/single_note" || exit 1
cargo run --bin synthesize -- --probe two --out "$BASE_DIR/two_note" || exit 1
cargo run --bin synthesize -- --probe three --out "$BASE_DIR/three_note" || exit 1

run_selftest() {
  local rubric="$1"
  local data="$2"

  if ! cargo run --bin snn_selftest -- \
    --rubric "$rubric" \
    --data "$data" \
    --model "$MODEL"; then
    status=1
  fi
}

run_selftest display-no-scent "$BASE_DIR/no_scent"
run_selftest display-single "$BASE_DIR/single_note"
run_selftest display-two "$BASE_DIR/two_note"
run_selftest display-three "$BASE_DIR/three_note"

exit "$status"
