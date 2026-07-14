# PLAN-001: Live Injection and Streaming Model Boundary

## Goal

Move NoseKnows toward live/realtime sensing while keeping the model boundary clean.

The next system should let a user inject synthetic scent notes interactively, emit continuous no-scent frames when no note is active, process ADC-like frames one quantum at a time, and show a continuously scrolling view of model internals and output.

## Non-Negotiable Boundary

Daft is input orchestration and data management.

Rust is model execution.

Do not mix these responsibilities.

```text
Daft / Python
  owns: scenario tables, selected notes, synthetic input chunks, manifests, replay logs, result joins, reports

Rust
  owns: model.step(frame), rolling state, quantization, feature expansion, readout, gating, model files
```

Rust should never import Daft or depend on Python runtime state. Daft/Python should never contain model math that must later run in firmware or a standalone Rust process.

## Proposed Live Loop

The live injector should be chunked, not per-row Daft calls.

```text
1. User selects active note(s), intensity, and optional duration.
2. If no note is active, the injector state resolves to No Scent.
3. Daft/Python materializes the next small chunk of ADC-like frames.
4. Rust consumes those frames one at a time through the streaming model API.
5. Rust appends model internals and output rows.
6. Viewer tails the output rows and scrolls continuously.
```

The chunk size should be small enough to feel live and large enough to keep Daft out of the hot path. A starting point is 1-5 seconds of frames per materialized chunk.

## Data Flow

```text
injector state
    |
    v
Daft/Python input orchestrator
    |
    v
ADC-like frame chunk
    |
    v
Rust streaming runner
    |
    v
model state/output rows
    |
    v
viewer + Daft/Python reports
```

Suggested generated artifacts:

```text
data/live/injector_state.json
data/live/input_frames.csv
data/live/model_results.csv
data/live/events.csv
```

These are runtime artifacts and should remain git-ignored.

## Injector Semantics

The injector should support:

- no active note, which continuously emits no-scent frames
- one active note
- two active notes
- three active notes
- optional intensity per note
- optional duration per injection
- deterministic replay seed for demos and debugging

The first implementation can be synthetic only. Hardware serial input should later use the same downstream `FrameSource` interface.

## Headless Mode

The live system must support a headless run mode.

`scripts/run_headless_live.sh` executes the same injector/materializer/model loop without opening or serving the scrolling viewer. It still writes all normal artifacts:

```text
data/live/input_frames.csv
data/live/model_results.csv
data/live/events.csv
optional report artifacts
```

Headless mode is required for:

- reproducible smoke tests
- long unattended synthetic runs
- CI-style regression checks
- demo preparation without a browser
- later hardware capture runs where visualization is optional

The viewer should be a consumer of result rows, not a requirement for producing them.

## Rust Runtime Shape

Refactor the current peak-pair stream evaluator into an incremental model object:

```text
model.step(frame) -> StepOutput
```

`StepOutput` should include enough state for the live viewer and result ledger:

```text
timestamp / row index
adc values
held peak values
quantized peak bins
selected pairwise features or feature summary
label scores
gated labels
truth labels, when known
segment/injection id, when known
```

The same stepping API should support:

```text
SyntheticInjectorSource
CsvReplaySource
SerialEsp32Source
```

This keeps synthetic live injection, offline stream replay, and real hardware capture aligned.

## Training Implication

Training should match the stepping semantics.

The model should be trained and evaluated on row-wise streams where labels are sparse:

- no-scent rows should produce no label output
- active scent rows should only produce label output after enough evidence accumulates
- brief carryover from sample-and-hold is acceptable and should be measured, not hidden

Daft/Python can construct those stream datasets and result ledgers. Rust should consume the resulting plain stream CSVs and write plain result CSVs.

## Viewer Direction

The current giant SVG and scrubber are useful prototypes. The live viewer should become a bounded rolling timeline fed from `model_results.csv` or an equivalent local stream.

Panels to keep:

- truth/injection strip
- ADC traces
- held peak bins
- accordion/pairwise feature activity summary
- label evidence
- gated readout

The viewer should make it easy to see:

- what was injected
- what the model currently believes
- whether the model is silent during no-scent
- how long evidence persists after an injection ends

## Why Daft Still Matters

Using Daft for input orchestration is useful because it gives us one consistent layer for:

- scenario construction
- reproducible replay
- dataset materialization
- truth/model-output joins
- post-run summaries
- demoable provenance

The live injector can therefore build on Daft without compromising the model boundary.

## Near-Term Implementation Steps

1. Add a `data/live/` contract and ignore generated live artifacts. Done.
2. Define an injector-state schema for active notes, intensity, duration, and seed. Done for the first JSON sequence schema.
3. Add a Daft/Python chunk materializer that appends ADC-like frame chunks from injector state. Done as `tools/live/inject_chunks.py`.
4. Refactor peak-pair replay into a reusable Rust `step(frame)` runtime. Done as `noseknows::peak::PeakRuntime`.
5. Add a Rust live runner that consumes frame chunks and appends result rows. Done as `cargo run --bin live_headless`.
6. Add `run-headless` so the live runner can execute without the viewer. Done as `scripts/run_headless_live.sh`.
7. Add a compact scrolling viewer over the live result rows.
8. Keep `CsvReplaySource` and `SyntheticInjectorSource` working before adding `SerialEsp32Source`.

## Open Questions

- What chunk size feels live while keeping Daft comfortably outside the hot path?
- Should intensity be linear, logarithmic, or recipe-defined per note?
- Should injected multi-note samples combine by raw addition, compressed addition, max-with-saturation, or a more physical chamber model?
- How should the readout score no-scent during transition windows?
- Which fields are required in `model_results.csv` for useful post-run Daft reports?
