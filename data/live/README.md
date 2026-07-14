# Live Runtime Artifacts

This directory is reserved for generated live-injection artifacts. Runtime CSV/JSON files are ignored by git.

The headless live path is additive and does not replace the existing stream evaluators:

```sh
scripts/run_headless_live.sh
```

The default run:

1. Reads or creates `data/live/injector_state.json`.
2. Uses the Python/Daft-side input orchestrator to materialize `data/live/input_frames.csv`.
3. Runs the Rust peak-pair streaming model one frame at a time.
4. Writes `data/live/model_results.csv` and `data/live/events.csv`.

The model remains Rust-only. The Python side owns input orchestration and writes plain frame CSVs.

Default generated artifacts:

```text
data/live/injector_state.json
data/live/input_frames.csv
data/live/input_events.csv
data/live/model_results.csv
data/live/events.csv
```
