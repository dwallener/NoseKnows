# NoseKnows

NoseKnows is an early prototype for a 9-channel gas-sensor scent classifier.

The repository currently has five pieces:

- A Rust desktop/web demo that renders the fragrance wheel and lights the top 3 categories.
- ESP32-S3 firmware that streams 9 ADC readings over USB serial.
- A Rust host collector that writes labeled serial captures to CSV.
- A synthetic capture generator for fake end-to-end testing.
- A small Rust training/inference scaffold for the 14-label fragrance wheel.

## Prototype Scope

NoseKnows currently models one fragrance exposure per capture. A capture is expected to contain one named sample, one controlled exposure/recovery timeline, and one set of three fragrance-wheel labels.

The current system does not attempt overlapping scent separation, continuous room-state tracking, source attribution, or real-time classification of multiple simultaneous fragrances. Designer-style synthetic captures may include top, heart, and base note phases, but those phases are treated as the evolution of one fragrance sample, not separate overlapping scents.

## Training Architecture

All training experiments use the same captured CSV dataset shape under `data/raw/`. The transformer/NoseLLM path and the spiking neural network path should remain separate implementations with separate model artifacts.

The transformer path consumes downsampled analog time series and writes model files under `data/models/`. The SNN path should consume the same CSV captures through the spike encoder, then train/export separate SNN parameters for fixed-point inference. Shared data, separate training code.

## Wheel Demo

Run the default random demo:

```sh
cargo run -- --random
```

Run the slower recording-oriented simulation:

```sh
cargo run -- --simulation
```

The app prints a local URL such as:

```text
NoseKnows demo running at http://127.0.0.1:7878
```

If the default port is busy, it falls forward through nearby ports.

### Demo Modes

`--random` emits random top-3 category activations every 3 seconds.

`--simulation` emits sample-style status text such as:

```text
processing sample 0032 - 00:04 / 00:11
```

Simulation mode randomizes the interval between samples from 7 to 13 seconds.

## ESP32-S3 Firmware

Firmware lives in `firmware/` and uses PlatformIO with the Arduino framework.

Build from the repository root:

```sh
PLATFORMIO_CORE_DIR=/Users/damir00/Sandbox/NoseKnows/.platformio pio run -d firmware
```

Upload from the repository root:

```sh
PLATFORMIO_CORE_DIR=/Users/damir00/Sandbox/NoseKnows/.platformio pio run -d firmware --target upload
```

Monitor serial:

```sh
PLATFORMIO_CORE_DIR=/Users/damir00/Sandbox/NoseKnows/.platformio pio device monitor --port /dev/cu.usbmodem21401 --baud 115200
```

The firmware uses GPIO1 through GPIO9 as ADC inputs:

```text
adc0 GPIO1
adc1 GPIO2
adc2 GPIO3
adc3 GPIO4
adc4 GPIO5
adc5 GPIO6
adc6 GPIO7
adc7 GPIO8
adc8 GPIO9
```

Serial frames are CSV-like:

```text
NK_ADC,seq,ms,adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7,adc8
```

Example:

```text
NK_ADC,2399,240510,2687,745,730,536,565,671,647,670,148
```

## Hardware Notes

The initial GPIO1 test used 3.3V across a 12k/22k resistor divider. Expected ADC input:

```text
3.3V * 22k / (12k + 22k) = 2.14V
```

At 12-bit ADC scale, that predicts roughly:

```text
4095 * 2.14 / 3.3 ~= 2655
```

Observed GPIO1 readings were around `2687-2699`, which is consistent with the divider and confirms `adc0`/GPIO1 is wired and sampled correctly.

Keep external analog voltages within the ESP32-S3 ADC-safe range. Do not drive GPIO inputs from powered external circuitry while the ESP board itself is unpowered.

## Host Data Collection

Run the console collector from the repository root:

```sh
cargo run --bin collect
```

The collector prompts for:

- sample name
- top 3 fragrance-wheel labels
- collection duration in seconds
- serial port, defaulting to `/dev/cu.usbmodem21401`
- baud rate, defaulting to `115200`

Labels are validated against the 14-slice Michael Edwards wheel:

```text
Floral, Soft Floral, Floral Amber, Amber, Soft Amber, Woody Amber, Woods,
Mossy Woods, Dry Woods, Aromatic, Citrus, Water, Green, Fruity
```

It listens for `NK_ADC` frames from the ESP32-S3 firmware and writes timestamped CSV files to `data/raw/`.

CSV columns:

```text
sample_id,sample_name,label_1,label_2,label_3,host_elapsed_ms,host_unix_ms,device_seq,device_ms,adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7,adc8
```

Close the PlatformIO serial monitor before running the collector; only one process can own the serial port at a time.

## Tiny Transformer Training

For fake end-to-end testing, generate collector-shaped synthetic captures:

```sh
cargo run --bin synthesize -- --out data/raw --samples 100
```

These files are deliberately artificial. They exist to exercise the data loader, training loop, model save path, and inference path before the real sensor dataset is large enough.

The synthetic generator uses the numeric 14-subfamily matrix. Matrix values are treated as target 12-bit ADC peaks, inactive `0` entries become clean-air baseline values between `150` and `300`, and decay is modeled from the listed `T90` targets. Synthetic captures are 900 rows at 100 ms, or about 90 seconds, so the longer tails are visible.

Matrix generation now includes no-scent examples by default:

```text
--no-scent-ratio 0.25
```

No-scent captures are written with:

```text
label_1,label_2,label_3 = No Scent,No Scent,No Scent
```

Training treats `No Scent` as a pseudo-label outside the 14 fragrance-wheel labels, producing an all-false 14-label target. This gives the gated readout explicit examples where it should stay silent.

Single-note synthetic captures can also be generated when needed:

```sh
cargo run --bin synthesize -- --out data/raw --samples 100 --single-note-ratio 0.20
```

Single-note captures use one real fragrance label followed by two `No Scent` pseudo-labels, for example:

```text
Citrus,No Scent,No Scent
```

For SNN self-test fixtures, keep probe types segregated under `data/selftest/`:

```sh
scripts/snn_selftest_full.sh
```

The script regenerates stored probe datasets under `data/selftest/generated/`, runs all four display rubrics, and continues through every rubric even when a failing case returns nonzero. To generate the manual default probe directories one at a time:

```sh
cargo run --bin synthesize -- --probe no-scent
cargo run --bin synthesize -- --probe single
cargo run --bin synthesize -- --probe two
cargo run --bin synthesize -- --probe three
```

These commands generate:

```text
data/selftest/no_scent    50 clean-air captures
data/selftest/single_note 14 one-label captures, one per wheel label
data/selftest/two_note    91 two-label combinations
data/selftest/three_note  364 three-label combinations
```

The CSVs are generated data and are ignored by git. The directory contract is tracked in `data/selftest/README.md`.

Build the current comprehensive SNN training dataset with:

```sh
scripts/build_snn_training_dataset.sh
```

By default this writes `data/training/snn_comprehensive` with 100 no-scent captures and 4 randomized variants for every single-, two-, and three-note combination:

```text
no_scent=100 single_note=56 two_note=364 three_note=1456 total=1976
```

This dataset is intended to train and stress the SNN path across clean air, single labels, and simple blends without mixing in designer-phase captures. An initial accordion training smoke test over the full set loaded 1,976 captures and reached high any-in-top-3 recovery on fragrance samples, but still produced false-positive label output for no-scent cases. Treat the comprehensive set as ready for training experiments, not as proof that the current SNN objective/readout is finished.

Train against the comprehensive SNN set with:

```sh
cargo run --bin snn_train -- --data data/training/snn_comprehensive --out data/models/snn_accordion_comprehensive.nsm --epochs 250 --validation 0.2 --accordion
```

For continuous-training experiments, build a long labeled stream from the same capture dataset:

```sh
scripts/build_snn_stream_dataset.sh
```

By default this writes `data/streams/snn_comprehensive_stream.csv`, shuffling fragrance captures and inserting no-scent gaps so about 50% of stream rows are clean air. The stitcher derives both scent and no-scent rows from generated capture CSVs, so gaps found in stream training can still be fixed at the common synthetic source.

Train the separate stream readout model with:

```sh
scripts/train_snn_stream.sh
```

The first stream model is deliberately separate from the capture-level accordion model. It walks the stitched CSV one row at a time, builds rolling baseline-relative rate/delta spike features, and trains with no-scent rows as all-false targets. A small smoke stream can be built and trained with:

```sh
scripts/build_snn_stream_dataset.sh data/training/snn_comprehensive data/streams/smoke_stream.csv 0.5 8
cargo run --bin snn_stream_train -- --stream data/streams/smoke_stream.csv --out data/models/snn_stream_smoke.nsm --epochs 3 --validation 0.2 --window 30 --stride 1
```

The initial smoke run produced a 14,400-row stream with exactly 50% no-scent rows and reached roughly 97% no-scent silence after three epochs. Treat this as a stream-training scaffold, not the final live classifier.

Render a compact rolling timeline preview for the stream model with:

```sh
scripts/viz_snn_stream.sh
```

The preview writes `data/streams/stream_preview.svg` by default. It shows a bounded row window with a ground-truth strip, compact ADC traces, rolling rate/delta feature lanes, label evidence heatmap, and gated top-3 readout lanes. Use the optional script arguments to inspect a different stream/model/window:

```sh
scripts/viz_snn_stream.sh data/streams/snn_comprehensive_stream.csv data/models/snn_stream_readout.nsm data/streams/stream_preview.svg 0 3000
```

The matrix maps into the existing 9-column CSV shape as:

```text
adc0 IO1  MQ-2
adc1 IO2  MQ-3
adc2 IO16 MQ-5
adc3 IO17 MQ-6
adc4 IO18 MQ-7
adc5 IO21 MQ-8
adc6 IO22 MQ-9
adc7 IO23 MQ-135
adc8 MQ-4 placeholder, held constant for now
```

Generate phase-layered designer-fragrance captures:

```sh
cargo run --bin synthesize -- --out data/raw --samples 100 --designer
```

Designer mode uses top/heart/base note timing instead of activating all three labels at once. It currently has five recipe families:

```text
Sauvage Type       Aromatic, Citrus, Woody Amber
Santal 33 Type     Dry Woods, Woods, Soft Floral
Black Orchid Type  Amber, Woody Amber, Floral Amber
Acqua di Gio Type  Water, Citrus, Floral
Flowerbomb Type    Floral, Amber, Green
```

Each generated variant jitters phase starts, peak ADC values, decay targets, residual offsets, and rise timing so the training loop sees repeated but non-identical versions of the same fragrance translation.

Train the first small sequence classifier against captured CSV files:

```sh
cargo run --bin train -- --data data/raw --out data/models/tiny_transformer.ntm --epochs 100
```

The trainer reads each CSV as one labeled fragrance capture, downsamples the 9 ADC channels to a fixed 32-step sequence, and trains against the three stored fragrance labels. It assumes the CSV represents one sample event, not a stream containing multiple overlapping scents. The model is intentionally small for early experiments: one single-head self-attention block, a small feed-forward block, mean pooling, and a 14-label output head.

By default, the trainer keeps the tiny transformer encoder fixed and trains the output head with plain Rust gradient updates. For an end-to-end training smoke test over all model parameters, use full-model mode:

```sh
cargo run --bin train -- --data data/raw --out data/models/full_transformer.ntm --full-model --epochs 200
```

Full-model mode still trains the output head with stable gradients, then applies a small SPSA update over all roughly 3k parameters. It is slower and noisier than the output-head path, but it removes the previous limitation where only the classifier head was learned.

Generated model parameter files under `data/models/` are ignored by git.

Run inference against one captured CSV:

```sh
cargo run --bin train -- --model data/models/tiny_transformer.ntm --predict data/raw/synthetic_0000.csv
```

Run the training-quality sanity check:

```sh
cargo run --bin quality -- --data data/raw --epochs 250 --validation 0.2
```

The quality runner trains a plain logistic baseline on summary features from each 9-channel capture, then reports train/validation loss, primary-label top-1 accuracy, and any-label top-3 accuracy. This is a guardrail: if the baseline cannot learn the synthetic data, the data generator or labels are suspect; if the baseline can learn but the tiny transformer scaffold cannot, the transformer/training loop is the weak link.

With 100 numeric-matrix captures and 100 designer-phase captures, the quality baseline currently reaches about:

```text
validation primary-label top-1: 92.5%
validation any-label top-3:     100%
```

## Spike Train Preview

Generate a self-contained SVG preview of spike encodings for one captured or synthetic CSV:

```sh
cargo run --bin spikes -- --input data/raw/synthetic_0000.csv --out data/spikes.svg --model data/models/snn_lif.nsm --bins 180 --subslots 5 --rate-budget 5 --latency-budget 5
```

The preview renders input spike-train panels over the same 9 ADC channels, then runs the mixed input stream through the saved SNN model and renders downstream SNN spike trains. Direct LIF models render five panels:

```text
pure latency  positive dV/dt maps to quantized sub-sample latency slots
pure rate     log-scaled amplitude emits up to the rate budget per sample
mixed         the union of rate events and latency events overlaid in one panel
final layer   14 SNN output spike trains, one per fragrance-wheel label
gated readout 14 label rows showing only decisions that clear the evidence gate
```

Accordion models add a sixth panel between `mixed` and `final layer`:

```text
accordion     64 emergent-pattern spike trains from the differentiation layer
```

The gated readout is a comparison output; it does not replace the raw final-layer spike panel. By default, it emits top-3 label decisions only when the accumulated readout satisfies:

```text
top label spike count in rolling window >= 3
top label spike count - fourth label spike count in rolling window >= 1
upstream activity in rolling window >= 12
rolling window = 6 binned samples
```

Those thresholds can be tuned with:

```sh
--gate-min-top 3 --gate-margin 1 --gate-min-activity 12 --gate-window 6
```

The visualizer now uses an event-list model instead of a boolean raster. With the default `5` subslots, latency is quantized into 20% buckets within each binned sample period. With the default `rate-budget=5` and `latency-budget=5`, each active encoded channel can emit up to `10` events per binned sample, or `80` total events across the 8 active SNN sensor channels.

Only the first 8 ADC channels are active SNN inputs right now. `adc8` remains the MQ-4 placeholder and is shown for continuity with the collector CSV shape, but it is not encoded into the 16-stream SNN input bundle. The active SNN input streams are:

```text
inputs 0..7    rate streams for adc0..adc7
inputs 8..15   latency streams for adc0..adc7
```

The final output panel is raw LIF output activity. Individual spike subslots update membrane state; they are not treated as mandatory per-subslot fragrance decisions. The gated readout panel is rolling-window evidence over recent output spikes and upstream activity, so labels can appear and disappear as evidence enters or leaves the live-style window.

To keep no-scent captures from turning clean-air jitter into artificial spikes, the SNN encoder applies small absolute floors before per-capture normalization:

```text
minimum active channel range: 80 ADC counts
minimum positive delta:       25 ADC counts
```

Channels below those floors stay silent for rate and latency encoding.

## SNN Training

Train the exploratory fixed-point LIF SNN against the simple synthetic captures:

```sh
cargo run --bin snn_train -- --data data/raw --out data/models/snn_lif.nsm --epochs 250 --validation 0.2
```

Train the first accordion SNN scaffold:

```sh
cargo run --bin snn_train -- --data data/raw --out data/models/snn_accordion.nsm --epochs 250 --validation 0.2 --accordion
```

By default, `snn_train` excludes `designer_*` phase-layered captures so the first SNN task stays focused on non-complex single-subfamily combinations. Use `--include-designer` later when we intentionally want phased top/heart/base training examples in this path.

The direct SNN scaffold has two training stages:

1. A multilabel linear model learns from 16 spike-count features.
2. A fixed-point LIF bank is initialized from that linear model, including per-label bias, and fine-tuned against top-3 label recovery plus no-scent silence.

The accordion scaffold inserts a differentiation layer:

```text
16 rate/latency spike streams
  -> 64 seeded emergent-pattern LIF neurons with winner-take-few lateral inhibition
  -> 14 fragrance-wheel label neurons
```

The 64 seeded mini-patterns are intentionally interpretable first guesses: single rate/latency channels, small co-activated sensor pairs, onset-plus-tail relationships, and broader cluster-history motifs. The first layer is not label-aware; the supervised training happens in the sparse `64 -> 14` label mapping.

The accordion layer also applies a local sensory-adaptation rule to motifs involving `adc4` / IO18 / MQ-7. When one of those motifs fires, its effective threshold is temporarily raised and then decays each subslot:

```text
effective_threshold = base_threshold + adaptation[pattern]
on adc4-linked pattern spike: adaptation += 450
each subslot: adaptation *= 224 / 256
adaptation cap: 2400
```

This preserves real onset evidence from the sticky sensor while attenuating sustained tail/DC-offset firing before it can dominate `Soft Amber` and `Woods` readout.

The exported `.nsm` file is a separate SNN artifact from the transformer `.ntm` model files. Both paths share the same `data/raw/*.csv` captures, but the training code and inference assumptions remain separate.

Current smoke-test behavior on the simple synthetic set is intentionally modest but correctly shaped: primary top-1 accuracy is still weak, while any-label top-3 recovery is high. That is enough for now to inspect spike encodings, LIF accumulation, and output spike trains before real repeated hardware captures exist.

When `No Scent` samples are present, SNN training reports additional metrics:

```text
silence  percent of no-scent samples with no active label output
fp       false-positive rate on no-scent samples
```

For no-scent rows, a silent output counts as success for the combined `p@1` and `any@3` scores. During LIF fine-tuning, no-scent samples use a separate suppression update: only labels that actually activate are pushed down, and with a smaller update than normal fragrance-label correction.

Run the end-to-end SNN self-test:

```sh
cargo run --bin snn_selftest -- \
  --rubric display \
  --data data/raw_single_note_probe \
  --model data/models/snn_accordion_single_note_probe.nsm
```

The self-test reuses the same spike encoder, SNN model format, and rolling gated readout semantics as the SVG visualizer. Rubrics are intentionally split by capture type so no-scent, single-note, two-note, and three-note behavior can be evaluated independently:

```text
strict             diagnostic no-scent + single-note check
display-no-scent   no-scent only; gated readout must stay silent
display-single     single-note only; correct note must be display-visible
display-two        two-note only; both notes should be visible, with one allowed weak
display-three      three-note only; at least two expected notes must be visible
display-all        aggregate display rubric over no-scent plus 1/2/3-note captures
```

The legacy `--rubric display` alias maps to `display-all`. On the current `data/raw_single_note_probe` set that means no-scent plus single-note, because no two- or three-note probe captures are present yet.

The default `strict` single-note rule is intentionally pragmatic rather than antiseptic:

```text
correct label decisions >= 3
correct label has the highest decision count
at most 2 other labels may appear
each spillover label may appear at most 3 times
```

Those bounds can be adjusted with:

```sh
--min-correct 3 --max-spillover-labels 2 --max-spillover 3
```

For product-facing wheel behavior, use the display rubric:

```sh
cargo run --bin snn_selftest -- \
  --rubric display-single \
  --data data/raw_single_note_probe \
  --model data/models/snn_accordion_single_note_probe.nsm
```

The display rubric treats the output like a live nose display rather than a mass spectrometer:

```text
Single note  correct label must be visible in the top 3
              correct label must have at least 3 gated decisions
              a wrong dominant label is tolerated only if it is close
Two note     both expected labels should be top-3 visible
              one expected label may be weak if the other is solid
              an unexpected dominant label is tolerated only if it is close
Three note   at least two of three expected labels must be top-3 visible
              an unexpected dominant label is tolerated only if it is close
```

The wrong-dominance tolerance is controlled with:

```sh
--display-max-dominant-gap 8
```

Current display-rubric checkpoint:

```text
Self-test checked=100 passed=92 failed=8 skipped=0
No Scent: 50/50 silent | Single note: 42/50 pass | Two note: 0/0 pass | Three note: 0/0 pass
Failure kinds: raw_silent=3 gate_silent=3 wrong_dominant=2 spillover=0 no_scent_fp=0
```

Separated checks:

```text
display-no-scent: 50/50 pass
display-single:   42/50 pass on data/raw_single_note_probe
```

Current segregated exhaustive-probe checkpoint:

```text
display-no-scent on data/selftest/no_scent:    50/50 pass
display-single   on data/selftest/single_note: 9/14 pass
display-two      on data/selftest/two_note:    10/91 pass
display-three    on data/selftest/three_note:  53/364 pass
```

The self-test prints detail for all four probe types:

```text
No-scent detail:
  silent count
  false-positive sample count
  false-positive labels, if any

Single-note detail:
  expected-label coverage: visible / weak / missing
  visible-label coverage histogram
  wrong/silent dominant labels
  single-note dominant-label confusion table

Two-note / three-note detail:
  expected-label coverage by label
  visible-label coverage histogram
  wrong/silent dominant labels
  common missing-or-weak expected labels grouped by the dominant replacement
```

For the display rubrics, `visible` means the expected label is in the gated top 3 with at least `--min-correct` decisions. `weak` means the expected label has gated evidence but does not clear the visible threshold. `missing` means the expected label has no gated evidence.

Current detailed exhaustive-probe snapshot:

```text
no-scent:
  silent=50 false_positive_samples=0

single-note:
  expected_slots=14 visible=11 weak=0 missing=3

two-note:
  expected_slots=182 visible=69 weak=4 missing=109
  visible-label coverage histogram: 0=32 1=49 2=10 3=0
  most common wrong dominant: Dry Woods=35
  most common replacement pattern: Dry Woods over Amber/Fruity/Green/Soft Amber/Woody Amber/Water

three-note:
  expected_slots=1092 visible=283 weak=11 missing=798
  visible-label coverage histogram: 0=143 1=163 2=54 3=4
  most common wrong dominant: Dry Woods=120
  most common replacement pattern: Dry Woods over Amber/Floral/Fruity/Woody Amber/Green/Soft Amber/Water
```

This is the more relevant robustness benchmark. The strict rubric remains useful as a diagnostic because it reveals wrong-dominant and spillover patterns, but it should not be treated as the product acceptance criterion.

As of this checkpoint, no-scent passes cleanly, while the stricter all-single-note check still exposes known class confusions and silent labels in the current accordion LIF readout. Treat this as a regression/self-test harness, not proof that the SNN classifier is finished.

The self-test also prints a single-note confusion table. Current dominant-label failures are concentrated in a few useful buckets:

```text
Fruity -> Woods
Green  -> Floral
Water  -> Floral
Amber / Floral Amber / Woody Amber / Dry Woods -> Silent
```

Some labels, such as `Floral`, `Soft Floral`, `Aromatic`, and `Mossy Woods`, often have the correct dominant label but still fail the stricter self-test because spillover persists too long. That points to label-side calibration/contrast rather than a broken spike encoder.

The self-test also splits failures by where they first appear:

```text
raw_silent      raw final SNN output has no label spikes
gate_silent     raw output exists, but rolling gate does not emit enough evidence
wrong_dominant  gated output is dominated by the wrong label
spillover       correct label dominates, but secondary labels persist too long
```

Current checkpoint:

```text
before targeted tuning: raw_silent=3 gate_silent=7 wrong_dominant=9 spillover=11 no_scent_fp=0
after targeted tuning:  raw_silent=3 gate_silent=3 wrong_dominant=9 spillover=6  no_scent_fp=0
```

That means the base-note silence hypothesis is only partly a rolling-window issue: several heavy labels produce weak raw spikes that fail the gate, but some are already silent at the raw output layer. Fruity/Green/Water confusions are raw-label mapping problems rather than gated-readout artifacts.

The first tuning pass is intentionally targeted:

```text
base-note gate policy:
  Floral Amber, Amber, Woody Amber, Dry Woods
  lower minimum gated count
  longer rolling readout window

top-note contrast:
  Green and Water inhibit the generic Floral accumulator
  inhibition is currently one LIF threshold per competing spike
  Citrus is left alone while it remains clean

mapping-prune deferred:
  Fruity -> Woods needs contribution diagnostics before pruning Woods weights
```

The base-note gate policy improved the self-test materially. The Green/Water-vs-Floral collision improved in raw counts but still does not pass, so further improvement probably needs better label-side contrast/training or more discriminating motifs rather than simply increasing inhibition again.

The SNN model export includes the learned readout bias:

```text
bias.<label>=...
label_bias.<label>=...
```

Older `.nsm` files without bias still load with zero bias. New no-scent-trained models preserve the learned silent/no-report behavior in both training and spike preview.

Current status:

- Real collector path is working and writes training-shaped CSV files.
- The current learning problem is constrained to one fragrance exposure per capture.
- Transformer and SNN experiments share datasets but stay separate in code and model artifacts.
- Synthetic captures let us test the full train/infer loop without claiming real-world scent accuracy.
- The transformer model consumes one `32 x 9` downsampled time series per CSV and predicts the top 3 of the 14 fragrance-wheel labels.
- The direct SNN model consumes mixed rate/latency spike events and accumulates activity in a 14-output fixed-point LIF bank.
- The accordion SNN model expands those events through a 64-neuron differentiation layer before label mapping.
- The present trainers are intentionally simple and CPU-only; they are scaffolding steps before more serious training infrastructure.
- Training quality is now measured with a validation split and a simple baseline before we invest in end-to-end transformer backpropagation.
- Spike-train previews are available for rate, latency, mixed encodings, accordion pattern activity, raw final SNN output activity, and gated readout decisions.
- SNN training now has a separate first-pass fixed-point LIF scaffold over the same CSV dataset format.
