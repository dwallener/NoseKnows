# Live Runtime Artifacts

This directory is reserved for generated live-injection artifacts. Runtime CSV/JSON files are ignored by git.

The headless live path is additive and does not replace the existing stream evaluators:

```sh
scripts/run_headless_live.sh
```

The thin local UX layer is also additive:

```sh
scripts/live_ui.sh
```

It serves a small browser UI for editing injector state, materializing input frames, running the headless Rust model, and viewing the generated result timeline. The UI does not contain model math; it is a consumer/controller around the same files used by the headless path.

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
