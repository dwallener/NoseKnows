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

The preview renders input spike-train panels over the same 9 ADC channels, then runs the mixed input stream through the saved SNN LIF model and renders the final 14 fragrance output spike trains:

```text
pure latency  positive dV/dt maps to quantized sub-sample latency slots
pure rate     log-scaled amplitude emits up to the rate budget per sample
mixed         the union of rate events and latency events overlaid in one panel
final layer   14 SNN output spike trains, one per fragrance-wheel label
```

The visualizer now uses an event-list model instead of a boolean raster. With the default `5` subslots, latency is quantized into 20% buckets within each binned sample period. With the default `rate-budget=5` and `latency-budget=5`, each channel can emit up to `10` events per binned sample, or `90` total events across the 9-channel sensor package.

This is an exploratory visualization for possible spiking-network work. It is not yet a classifier and does not change the current single-fragrance capture training path.

## SNN Training

Train the first separate fixed-point LIF SNN scaffold:

```sh
cargo run --bin snn_train -- --data data/raw --out data/models/snn_lif.nsm --epochs 250
```

The SNN trainer uses the same CSV files as the transformer path, but it does not share training code or model artifacts. It spike-encodes the active sensor channels into 16 input streams:

```text
inputs 0..7   rate streams for active sensors adc0..adc7
inputs 8..15  latency streams for active sensors adc0..adc7
```

Those 16 streams drive a direct 14-output fixed-point leaky integrate-and-fire bank, one output neuron per fragrance-wheel label. The saved `.nsm` file records the integer weight matrix and encoder constants for later firmware/export work.

Training is deliberately separate from the transformer path. The current SNN scaffold trains in two stages:

```text
1. train a multilabel linear model on spike-count features
2. initialize and fine-tune the integer LIF bank, keeping the best validation checkpoint
```

By default, SNN training excludes `designer_*` complex phased captures and uses the simpler matrix-style samples. To include designer captures later:

```sh
cargo run --bin snn_train -- --data data/raw --out data/models/snn_lif.nsm --epochs 250 --include-designer
```

This is currently a scaffold, not the final SNN training method. Its purpose is to separate spike-encoding quality from LIF-bank behavior while preserving an ESP32-friendly fixed-point inference target.

Current status:

- Real collector path is working and writes training-shaped CSV files.
- The current learning problem is constrained to one fragrance exposure per capture.
- Transformer and SNN experiments share datasets but stay separate in code and model artifacts.
- Synthetic captures let us test the full train/infer loop without claiming real-world scent accuracy.
- The current model consumes one `32 x 9` downsampled time series per CSV and predicts the top 3 of the 14 fragrance-wheel labels.
- The present trainer is intentionally simple and CPU-only; it is a scaffolding step before a real autograd-backed Rust implementation.
- Training quality is now measured with a validation split and a simple baseline before we invest in end-to-end transformer backpropagation.
- Spike-train previews are available for rate, latency, and mixed encodings as an exploratory SNN input view.
- SNN training now has a separate first-pass fixed-point LIF scaffold over the same CSV dataset format.
