#!/usr/bin/env bash
set -euo pipefail

cargo run --bin live_ui -- "$@"
