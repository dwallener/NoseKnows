# NoseKnows Roadmap

This file tracks larger work streams. `TODO.md` is intentionally short and should only contain the next active tasks.

## Hardware and Real Data

- Finish the hardware sensor package.
- Collect repeated real captures for a small set of known fragrance samples.
- Compare real captures against synthetic captures before changing model architecture.
- Add real capture analysis for baseline-relative values, first differences, rolling slopes, peak, time-to-peak, decay rate, tail level, range, and area under curve.
- Re-evaluate all synthetic assumptions after real repeated captures exist.

## Dataset and Result Ledger

- Keep using recipe materialization for new training/evaluation views before adding bespoke trainer-side filters.
- Add ratio-based dataset selection: specify total sample count plus per-label-count ratios instead of only absolute `limits_per_label_count` caps.
- Add recipe support for stream materialization so the same dataset layer can construct capture views and stitched stream inputs.
- Extend run reports to compare multiple model/run IDs side by side when tuning thresholds, masks, or encoders in parallel.
- Let Daft/Python catalog and report over embedding artifacts, while Rust remains responsible for computing embeddings.

## Live Runtime

- Keep the live model boundary from PLAN-001:
  - Python/Daft owns input orchestration.
  - Rust owns model execution.
- Improve the Grid8 rolling readout and no-scent transition policy.
- Decide how live truth should treat carryover windows after an injection ends: strict no-scent immediately, or a transition/grace interval while the rolling grid still contains scent evidence.
- Add hardware serial input behind the same downstream `FrameSource` concept used by synthetic live injection and CSV replay.
- Keep headless replay working for every UI-visible model path.

## Gain / Focus Bridge

- Follow PLAN-002 for manual gain/attenuation validation.
- Validate the zero-input invariant whenever the gain stage changes.
- Add golden vectors for labels.
- Compare no-focus and focus runs using dominant duration, label-frame counts, clip counts, and embedding similarity.
- Add UI controls only after the headless focus workflow is coherent.
- Later explore learned or golden-vector-derived masks.

## Embedding and NoseLLM Bridge

- Treat `scent_embedding_v1` as the explicit 1024-dimensional bridge for downstream systems.
- Preserve raw 14-label output and model results next to embeddings for debuggability.
- Compare explicit ontology-backed embeddings with a learned projection over the same live state.
- Decide whether embeddings should be emitted every frame, every stable readout, or both.
- Build a small retrieval/evaluation harness around embedding similarity to golden labels and captured samples.

## Continuous Classification

- Defer true overlapping/realtime scent separation until single-exposure capture and live replay are stable.
- Keep the first continuous-training stream constrained to one fragrance exposure at a time with explicit no-scent gaps.
- Evaluate whether temporal models should consume raw sensor windows, derived feature windows, or both.
- Consider lightweight temporal models only after the representation is stable:
  - small RNN
  - small LSTM
  - small Temporal Convolutional Network

## SNN Research

- Keep SNN work exploratory until it beats or meaningfully simplifies the tiny transformer/baseline paths.
- Improve `cargo run --bin snn_train` beyond the current linear-initialized fixed-point LIF fine-tuning scaffold.
- Use accordion contribution diagnostics to explain gated-label mistakes before changing motifs or thresholds.
- Compare simple rate coding, threshold-crossing events, and delta-based event coding.
- Explore hidden LIF layers, per-neuron decay, and richer temporal readout.
- Evaluate SNN fit for ESP32-class inference: memory footprint, integer arithmetic, event sparsity, latency, and implementation complexity.

## ESP32-Class Deployment

- Keep model sizes small enough for ESP32-class hardware, but do not let deployment constraints drive research before the representation is working.
- Later export the simplest successful readout path to integer/fixed-point form.
- Keep firmware and host-side data collection aligned on ADC channel ordering and metadata.
