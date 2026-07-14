# NoseKnows Daft Dataset Layer

Daft is an optional offline dataset-construction layer for NoseKnows. It does not sit inside the Rust model or training runtime.

The intended boundary is:

```text
dataset recipe
    |
    v
Daft/Python materialization
    |
    v
plain artifacts: manifest CSV, capture CSV directory, stream CSV, feature table
    |
    v
Rust trainer / evaluator / viewer
```

Rust code should consume files and directories. It should not import Daft, call Daft, or depend on Python runtime state.

## Why This Exists

NoseKnows now has several ways to describe and derive training data:

- fragrance-response recipes
- raw and synthetic capture CSVs
- single-note, multi-note, no-scent, and stream probes
- derived encodings such as absolute peaks, quantized bins, slopes, saturation flags, and rolling features

Without a dataset layer, every trainer ends up reimplementing ad hoc directory walking and filter logic. The Daft layer gives us one consistent place to ask:

- train on only no-scent plus single-note captures
- build a balanced one/two/three-note view
- exclude designer-style complex captures
- materialize a smoke-sized subset
- preserve enough metadata to know exactly what was trained

## Responsibilities

Daft/Python owns:

- manifest construction
- filtering and balancing
- split assignment
- joining metadata
- feature/materialized dataset construction
- copying or writing artifacts for downstream use
- inference-result ledger analysis and reporting

Rust owns:

- model math
- training loops
- inference
- embedded/exportable representations
- evaluator/viewer logic
- writing plain inference result artifacts

## Current Experiment

The first implementation is intentionally narrow:

```sh
scripts/materialize_dataset.sh recipes/peak_single_note.toml
```

This reads capture CSVs, builds a manifest, filters the manifest according to the recipe, and copies selected captures into a materialized view directory. Existing Rust trainers can use that directory unchanged.

For the current peak-pair model:

```sh
scripts/train_peak_pair_recipe.sh recipes/peak_single_note.toml
```

That wrapper performs:

```text
1. materialize dataset recipe
2. cargo run --bin peak_train -- --data <recipe output_dir> ...
```

## Daft Dependency

The materializer is Daft-first. In the active Codex environment Daft may not be installed, so the tool has an explicit `--allow-stdlib-fallback` mode for local smoke testing. That fallback exists only to keep the workflow testable without installing dependencies; the intended evaluation path is to install/use Daft and run the same recipes.

## Manifest Shape

The generated manifest is one row per capture:

```text
sample_id
sample_name
source_path
source_kind
label_1
label_2
label_3
label_count
row_count
duration_ms
adc0_peak..adc8_peak
saturation_count
```

This gives us enough metadata to select datasets without reopening every capture repeatedly, while still preserving links back to raw material.

## Test Result Ledger

The same boundary applies to testing:

```text
materialized test dataset
    |
    v
Rust evaluator
    |
    v
plain results CSV
    |
    v
Daft/Python report step
```

Rust evaluators should write result rows with identifiers, labels, predictions, scores, pass/fail flags, and provenance. Daft/Python then treats those result rows as a ledger: join them to the manifest, compute grouped metrics, and write stable report artifacts.

For the peak stream evaluator:

```sh
scripts/eval_peak_stream.sh
```

This runs `peak_stream_eval` and writes:

```text
data/runs/peak_stream/results.csv
```

Then summarize the run with:

```sh
scripts/report_peak_stream_run.sh
```

The report step writes:

```text
data/runs/peak_stream/report/report.md
data/runs/peak_stream/report/summary_by_label_count.csv
data/runs/peak_stream/report/summary_by_label.csv
data/runs/peak_stream/report/failure_reasons.csv
```

The current analyzer is also Daft-first, with the same explicit fallback for environments where Daft is not installed:

```sh
NOSEKNOWS_ALLOW_STDLIB_DATASET=1 scripts/report_peak_stream_run.sh
```

The intent is that future evaluators follow the same pattern:

```text
Rust evaluator --out-results <run>/results.csv
Daft/Python analyzer <run>/results.csv --manifest data/manifest/captures.csv
```

## Rule

If a future trainer needs a different training set, add or edit a dataset recipe first. Do not add another bespoke directory-walk filter inside the trainer unless the filter is truly model-specific.

If a future evaluator needs deeper reporting, add fields to the plain result artifact first, then extend the Daft/Python analyzer. Do not make the Rust evaluator depend on Daft.
