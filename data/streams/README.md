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
no-scent ratio: 0.5
```

For a small smoke stream:

```sh
scripts/build_snn_stream_dataset.sh data/training/snn_comprehensive data/streams/smoke_stream.csv 0.5 8
```

The stream stitcher shuffles fragrance captures, writes each capture as one scent segment, then inserts no-scent rows derived from generated no-scent captures. This keeps the long-stream data tied to the same synthetic source as the capture training and self-test datasets.

Train the separate stream readout model with:

```sh
scripts/train_snn_stream.sh
```

The initial stream model is intentionally separate from the capture-level accordion model. It trains a rolling, baseline-relative spike-feature readout over a labeled timeline where no-scent rows have all-false 14-label targets.
