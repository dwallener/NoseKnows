# NoseKnows

NoseKnows is an early prototype for a 9-channel gas-sensor scent classifier.

The repository currently has two pieces:

- A Rust desktop/web demo that renders the fragrance wheel and lights the top 3 categories.
- ESP32-S3 firmware that streams 9 ADC readings over USB serial.

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
