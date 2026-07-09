# SNN Self-Test Probe Data

This directory is reserved for generated SNN self-test probe CSVs. The CSV files are ignored by git because the exhaustive probe set is generated data.

Regenerate the segregated probe sets with:

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
