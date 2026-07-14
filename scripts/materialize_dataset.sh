#!/usr/bin/env bash
set -euo pipefail

RECIPE="${1:-recipes/peak_single_note.toml}"
shift || true

python3 tools/dataset/materialize.py "$RECIPE" "$@"
