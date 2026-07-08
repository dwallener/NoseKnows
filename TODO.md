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
- Keep ESP32-class deployment in mind, but do not let size constraints drive the current research path before the representation is working.

## Spiking Network Research

- Keep SNN encoder/training/export code separate from the transformer/NoseLLM path while reusing the same CSV datasets.
- Explore whether the sensor time series can be represented as spike trains.
- Use `cargo run --bin spikes` to inspect rate, latency, mixed input encodings, and final 14-label output spike trains on existing CSV captures.
- Improve `cargo run --bin snn_train` beyond the current linear-initialized fixed-point LIF fine-tuning scaffold.
- Treat label emission as accumulated evidence over a capture/window, not as a required decision at every spike subslot.
- Add null/clean-air captures later and require the model to support a no-output or low-confidence state instead of always forcing a fragrance label.
- Compare simple rate coding, threshold-crossing events, and delta-based event coding.
- Explore whether a small hidden LIF layer, per-neuron decay, or richer temporal readout improves the current direct 16-input-to-14-output LIF bank.
- Re-evaluate the SNN classifier once real repeated captures exist.
- Evaluate SNN fit for ESP32-class inference: memory footprint, integer arithmetic, event sparsity, latency, and implementation complexity.
- Treat SNN work as exploratory until it beats or meaningfully simplifies the tiny transformer/baseline path.

## Synthetic Data

- Keep numeric-matrix and designer-phase generators as pipeline test fixtures.
- Avoid treating synthetic accuracy as real-world fragrance accuracy.
- Add new synthetic recipes only when they clarify a specific training or evaluation question.
