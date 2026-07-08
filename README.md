# NoseKnows

NoseKnows is an early prototype for a 9-channel gas-sensor scent classifier.

The repository currently has five pieces:

- A Rust desktop/web demo that renders the fragrance wheel and lights the top 3 categories.
- ESP32-S3 firmware that streams 9 ADC readings over USB serial.
- A Rust host collector that writes labeled serial captures to CSV.
- A synthetic capture generator for fake end-to-end testing.
- A small Rust training/inference scaffold for the 14-label fragrance wheel.

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

Train the first small sequence classifier against captured CSV files:

```sh
cargo run --bin train -- --data data/raw --out data/models/tiny_transformer.ntm --epochs 100
```

The trainer reads each CSV as one labeled scent sample, downsamples the 9 ADC channels to a fixed 32-step sequence, and trains against the three stored fragrance labels. The model is intentionally small for early experiments: one single-head self-attention block, a small feed-forward block, mean pooling, and a 14-label output head.

This first trainer keeps the tiny transformer encoder fixed and trains the output head with plain Rust gradient updates. It is meant to validate the data path, label contract, model save path, and inference path before moving to a faster autograd backend for end-to-end training.

Generated model parameter files under `data/models/` are ignored by git.

Run inference against one captured CSV:

```sh
cargo run --bin train -- --model data/models/tiny_transformer.ntm --predict data/raw/synthetic_0000.csv
```

Current status:

- Real collector path is working and writes training-shaped CSV files.
- Synthetic captures let us test the full train/infer loop without claiming real-world scent accuracy.
- The current model consumes one `32 x 9` downsampled time series per CSV and predicts the top 3 of the 14 fragrance-wheel labels.
- The present trainer is intentionally simple and CPU-only; it is a scaffolding step before a real autograd-backed Rust implementation.
