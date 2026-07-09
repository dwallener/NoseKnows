# SNN Self-Test Probe Data

This directory is reserved for generated SNN self-test probe CSVs. The CSV files are ignored by git because the exhaustive probe set is generated data.

Run the full self-test workflow with:

```sh
scripts/snn_selftest_full.sh
```

By default this regenerates and tests under:

```text
data/selftest/generated/no_scent
data/selftest/generated/single_note
data/selftest/generated/two_note
data/selftest/generated/three_note
```

You can also regenerate the default manual probe sets individually with:

```sh
cargo run --bin synthesize -- --probe no-scent
cargo run --bin synthesize -- --probe single
cargo run --bin synthesize -- --probe two
cargo run --bin synthesize -- --probe three
```

Default output directories:

```text
data/selftest/no_scent    50 clean-air captures
data/selftest/single_note 14 one-label captures
data/selftest/two_note    91 two-label combinations
data/selftest/three_note  364 three-label combinations
```
