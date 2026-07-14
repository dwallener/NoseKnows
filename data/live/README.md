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

The UI can run either current live readout:

- `Peak-pair`: the earlier sample-and-hold peak/pair model.
- `Grid8 rolling`: the `8 active sensors x 8 one-second lookback buckets` model.

Use the model selector in the header before clicking `Run Headless`. `Train Grid8` retrains the rolling-grid readout from the current no-scent/single-note materialized view.

The timeline includes a `Dominant` row above the 14 fragrance-label rows. This is a lightweight display readout over the model output, not a separate model: it uses a short trailing vote across recent non-silent top-3 predictions and emits one dominant label only when the winner has enough persistence and margin over the runner-up.

The default run:

1. Reads or creates `data/live/injector_state.json`.
2. Uses the Python/Daft-side input orchestrator to materialize `data/live/input_frames.csv`.
3. Runs the Rust peak-pair streaming model one frame at a time.
4. Writes `data/live/model_results.csv`, `data/live/events.csv`, and `data/live/embeddings.csv`.

The model remains Rust-only. The Python side owns input orchestration and writes plain frame CSVs.

## Scent Embeddings

The headless live runners also emit `scent_embedding_v1`, a 1024-dimensional vector intended for downstream NoseLLM-style consumers. This embedding is not the human-facing display rule; it preserves current output, recent history, pairwise label coactivation, and model-specific feature prefixes.

See `EMBEDDING.md` for the ontology and dimension map.

## Grid8 Experiment

The first rolling-grid experiment represents live state as:

```text
8 active sensors x 8 one-second lookback buckets
```

Train the no-scent/single-note grid readout with:

```sh
scripts/train_grid8.sh
```

Run the same live input sequence through the grid model with:

```sh
scripts/run_headless_live_grid8.sh
```

This path is intended to test sparse rolling readout behavior. Most one-second grid windows should remain silent; labels should appear only once enough recent sensor history has accumulated. Because the rolling grid keeps eight seconds of memory, post-segment carryover is expected unless the evaluator adds an explicit transition/grace policy.

Default generated artifacts:

```text
data/live/injector_state.json
data/live/input_frames.csv
data/live/input_events.csv
data/live/model_results.csv
data/live/events.csv
data/live/embeddings.csv
data/live/grid_model_results.csv
data/live/grid_events.csv
data/live/grid_embeddings.csv
```
