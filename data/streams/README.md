# SNN Stream Data

This directory is reserved for generated long-stream SNN datasets. CSV files are ignored by git.

Build the default stream from the comprehensive capture dataset with:

```sh
scripts/build_snn_stream_dataset.sh
```

Defaults:

```text
input captures: data/training/snn_comprehensive
output stream:  data/streams/snn_comprehensive_stream.csv
balanced scent buckets: single-note == two-note == three-note
initial no-scent captures: 3
random no-scent captures after each scent: 0..3
```

For a small smoke stream:

```sh
scripts/build_snn_stream_dataset.sh data/training/snn_comprehensive data/streams/smoke_stream.csv 3 3 8
```

The smoke command writes 8 single-note, 8 two-note, and 8 three-note scent segments. The stream stitcher starts with a no-scent prelude, balances the three scent buckets, shuffles fragrance captures, writes each capture as one scent segment, then inserts a random number of whole no-scent captures. This keeps the long-stream data tied to the same synthetic source as the capture training and self-test datasets while preserving clean-air row continuity for delta features.

Train the separate stream readout model with:

```sh
scripts/train_snn_stream.sh
```

The stream model is intentionally separate from the capture-level accordion model, but it uses the same seeded accordion motifs. It trains a rolling, baseline-relative `16 -> 64 -> 14` readout over a labeled timeline where no-scent rows have all-false 14-label targets.

Render a compact stream preview with:

```sh
scripts/viz_snn_stream.sh
```

Defaults:

```text
stream: data/streams/smoke_stream.csv
model:  data/models/snn_stream_smoke.nsm
out:    data/streams/stream_preview.svg
rows:   3000
```

The stream preview is a horizontally compact timeline, not the capture-level spike poster. It renders ground truth, ADC traces, rolling input features, the 64-motif accordion responses used by the stream readout, label evidence, and gated readout lanes over the selected row window.
