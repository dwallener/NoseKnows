# SNN Training Data

This directory is reserved for generated SNN training datasets. CSV files are ignored by git.

Build the current comprehensive synthetic SNN training set with:

```sh
scripts/build_snn_training_dataset.sh
```

Defaults:

```text
output directory: data/training/snn_comprehensive
variants per fragrance combination: 4
no-scent samples: 100
```

The default dataset contains:

```text
no_scent=100
single_note=56
two_note=364
three_note=1456
total=1976 CSV files
```

The generated CSVs are intentionally not tracked. Regenerate them whenever the synthetic matrix, probe variant count, or no-scent count changes.

Dataset recipes can materialize selected views from this generated corpus without changing the Rust trainers. For example, the current peak-pair single-note recipe selects no-scent plus single-note captures:

```sh
scripts/materialize_dataset.sh recipes/peak_single_note.toml --allow-stdlib-fallback
```

That writes selected capture copies under `data/views/` and manifests under `data/manifest/`. Those generated view/manifest artifacts are ignored by git.

The full set is a training/stress dataset for the SNN path, not a benchmark result by itself. The current accordion trainer can consume it directly:

```sh
cargo run --bin snn_train -- --data data/training/snn_comprehensive --out data/models/snn_accordion_comprehensive.nsm --epochs 250 --validation 0.2 --accordion
```

The current trainer still needs blend-aware and no-scent-aware objective tuning before this full dataset should be treated as a default production model recipe.
