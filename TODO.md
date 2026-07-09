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
- Keep the first continuous-training stream constrained to one fragrance exposure at a time with explicit no-scent gaps.
- Use `scripts/build_snn_stream_dataset.sh` to stitch generated capture datasets into long labeled streams with about 50% no-scent rows.
- Use `scripts/train_snn_stream.sh` for the separate stream readout model; do not replace the capture-level accordion model with it yet.
- Current stream scaffold walks one row at a time and trains rolling baseline-relative rate/delta spike features against sparse per-row labels.
- Next stream-model step: decide whether to move the stream trainer from the 16-feature rolling readout to the full 64-pattern accordion state.
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
- Use the rolling-window gated readout preview to compare raw label spikes against thresholded live-style "report/no-report" label decisions.
- Keep no-scent/clean-air captures in the dataset permanently; a correct no-scent readout is silence, not a forced fragrance label.
- Use `cargo run --bin snn_selftest` as the end-to-end SNN regression harness with separate rubrics:
  - No-scent should produce silent gated readout.
  - Single-note should produce a longer-lived dominant signal on the correct label.
  - Two-note should make both expected labels visible, while allowing one of them to be weaker if the other is solid.
  - Three-note should make at least two of the three expected labels visible.
  - Small transient spillover on one or two other labels is acceptable when it does not dominate strongly.
  - Use `--rubric display-no-scent`, `--rubric display-single`, `--rubric display-two`, and `--rubric display-three` as separate product-facing wheel acceptance checks.
  - Use `--rubric display` / `--rubric display-all` only as an aggregate display check.
  - Keep the default strict rubric as a diagnostic for top-1/spillover failure analysis.
  - Use `scripts/snn_selftest_full.sh` to regenerate stored self-test probes and run all four rubrics end to end.
  - Current display-rubric checkpoint: 92/100 overall, 50/50 no-scent silent, 42/50 single-note pass.
  - Segregated self-test fixtures live under `data/selftest/`: `no_scent`, `single_note`, `two_note`, and `three_note`.
  - Current segregated exhaustive-probe checkpoint: no-scent 50/50, single 9/14, two 10/91, three 53/364.
  - Self-test detail now reports all four cases:
    - no-scent false-positive labels.
    - single-note visible/weak/missing expected-label coverage and dominant-label confusion.
    - two-note and three-note visible/weak/missing expected-label coverage, coverage histograms, wrong dominants, and dominant-over-missing-label replacement patterns.
  - Current blend diagnostic read: two-note has 69/182 expected slots visible, 4 weak, 109 missing; three-note has 283/1092 visible, 11 weak, 798 missing.
  - Current dominant blend failure attractor is `Dry Woods`, especially replacing `Amber`, `Fruity`, `Green`, `Soft Amber`, `Woody Amber`, and `Water`.
  - Current checkpoint: no-scent passes, but several single-note classes are still silent or confused in the accordion LIF readout.
  - Current largest dominant-label confusions: Fruity -> Woods, Green -> Floral, Water -> Floral, and several Amber/Woods-family labels -> Silent.
  - Current failure split after targeted tuning: raw_silent=3, gate_silent=3, wrong_dominant=9, spillover=6, no_scent_fp=0.
  - Next SNN fixes should distinguish readout tuning from label-mapping fixes:
    - gate_silent heavy labels may need slower/per-label readout integration.
    - raw_silent labels need stronger pattern/label mapping before the gate can help.
    - wrong_dominant labels need label-side contrast or better discriminating motifs.
    - spillover-dominant labels need inhibition/calibration, not more sensitivity.
- Current tuning plan:
  - Add a base-note gate policy for `Floral Amber`, `Amber`, `Woody Amber`, and `Dry Woods`: longer readout window and lower minimum gated count.
  - Add targeted final-label lateral inhibition where `Green` and `Water` suppress the generic `Floral` accumulator.
  - Keep `Citrus` untouched while it remains clean.
  - Do not prune `Woods` weights for `Fruity -> Woods` until contribution diagnostics identify the responsible motifs.
  - Base-note gate tuning helped materially; Green/Water still lose to Floral, so the next step should be better label-side contrast or motif discrimination rather than blindly increasing inhibition.
- Use accordion contribution diagnostics to explain gated-label mistakes:
  - For each stored label and top gated label, print top firing pattern neurons.
  - Include pattern firing count, pattern-to-label weight, contribution score, and pattern name.
  - Use this to answer "why this wrong label?" and "why not the correct label?" before changing motifs or thresholds.
- Compare simple rate coding, threshold-crossing events, and delta-based event coding.
- Explore whether a small hidden LIF layer, per-neuron decay, or richer temporal readout improves the current direct 16-input-to-14-output LIF bank.
- Iterate on the first differentiation-layer scaffold between spike generation and fragrance-label generation:
  - Input: 16-channel rate/latency spike bundle.
  - Current Layer 1: 64 seeded emergent-pattern LIF neurons with winner-take-few lateral inhibition.
- Current local circuit rule: adc4-linked motifs use intrinsic threshold adaptation to attenuate sustained sticky-sensor tails.
  - Current encoder guardrail: clean-air jitter below the absolute range/delta floor should not create spike activity.
  - Role: generate spike trains/signals from patterns that are actually differentiable between sensor clusters.
  - Current Layer 2: sparse supervised fixed-point mapping from emergent-pattern activity to the 14 fragrance-wheel labels.
  - Next: replace or refine the seeded mini-patterns with an unsupervised/self-organizing update rule.
- Compare adapted accordion output against the previous Soft Amber/Woods false-positive case, then decide whether label-side contrast/corroboration is still needed.
- Re-evaluate the SNN classifier once real repeated captures exist.
- Evaluate SNN fit for ESP32-class inference: memory footprint, integer arithmetic, event sparsity, latency, and implementation complexity.
- Treat SNN work as exploratory until it beats or meaningfully simplifies the tiny transformer/baseline path.

## Synthetic Data

- Keep numeric-matrix and designer-phase generators as pipeline test fixtures.
- Avoid treating synthetic accuracy as real-world fragrance accuracy.
- Comprehensive SNN synthetic training data is generated by `scripts/build_snn_training_dataset.sh`.
- Current comprehensive SNN training dataset target: `data/training/snn_comprehensive`, with no_scent=100, single_note=56, two_note=364, three_note=1456.
- Initial accordion smoke test on `data/training/snn_comprehensive` loaded 1976 captures and showed the current objective is not yet blend/no-scent balanced: fragrance any@3 is high, but no-scent false positives remain at 100%.
- Next comprehensive-training fix: make the accordion/LIF objective explicitly optimize silent no-scent behavior while preserving expected-label coverage for one-, two-, and three-note captures.
- Matrix synthetic generation now supports exact-ratio no-scent samples and optional single-note samples.
- SNN training now treats no-scent samples as first-class all-false targets where silent output is success.
- SNN LIF readouts now carry a learned/exported per-label bias so no-scent silence survives fixed-point training and visualization.
- Use no-scent samples to calibrate gated readout false-positive behavior.
- Use single-note samples to debug whether individual wheel labels can be represented before multi-label blends.
- Add new synthetic recipes only when they clarify a specific training or evaluation question.
