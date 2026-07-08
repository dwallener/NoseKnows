# NoseKnows TODO

## Immediate Scope

- Finish the hardware sensor package.
- Collect repeated real captures for a small set of known fragrance samples.
- Keep the learning problem constrained to one fragrance exposure per capture.
- Compare real captures against synthetic captures before changing the model architecture.

## Feature Extraction

- Add analysis for baseline-relative sensor values.
- Add first-difference and rolling-slope features.
- Add per-channel summaries: peak, time-to-peak, decay rate, tail level, range, and area under curve.
- Compare absolute, normalized, derivative, and combined feature sets on real repeated captures.

## Continuous Classification Research

- Defer overlapping/realtime scent classification until the single-fragrance capture path is stable.
- Consider wrapping the feature extractor in a lightweight temporal model:
  - small RNN
  - small LSTM
  - small Temporal Convolutional Network (TCN)
- Evaluate whether temporal models should consume raw sensor windows, derived feature windows, or both.
- Preserve ESP32-class deployment constraints when selecting model size and operations.

## Synthetic Data

- Keep numeric-matrix and designer-phase generators as pipeline test fixtures.
- Avoid treating synthetic accuracy as real-world fragrance accuracy.
- Add new synthetic recipes only when they clarify a specific training or evaluation question.
