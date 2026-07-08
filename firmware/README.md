# NoseKnows ESP32-S3 Firmware

Initial hardware-test firmware for a generic ESP32-S3 board using 9 analog gas sensor inputs.

## Default ADC Pins

The first pass uses ADC1-capable GPIOs:

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

If the board labels pins differently, update `SENSOR_PINS` in `src/main.cpp`.

## Build And Upload

From the repository root:

```sh
PLATFORMIO_CORE_DIR=/Users/damir00/Sandbox/NoseKnows/.platformio pio run -d firmware
PLATFORMIO_CORE_DIR=/Users/damir00/Sandbox/NoseKnows/.platformio pio run -d firmware --target upload
PLATFORMIO_CORE_DIR=/Users/damir00/Sandbox/NoseKnows/.platformio pio device monitor --port /dev/cu.usbmodem21401 --baud 115200
```

From this `firmware/` directory:

```sh
PLATFORMIO_CORE_DIR=../.platformio pio run
PLATFORMIO_CORE_DIR=../.platformio pio run --target upload
PLATFORMIO_CORE_DIR=../.platformio pio device monitor --port /dev/cu.usbmodem21401 --baud 115200
```

This repo uses a project-local PlatformIO package cache at `../.platformio` to avoid
machine-global PlatformIO permission problems.

Current detected USB serial device:

```text
/dev/cu.usbmodem21401
```

If PlatformIO does not auto-detect it, add `--upload-port /dev/cu.usbmodem21401`
or `--port /dev/cu.usbmodem21401` for monitor.

If PlatformIO fails with `No module named pip` while installing `tool-esptoolpy`,
repair the PlatformIO pipx environment once:

```sh
/Users/damir00/.local/pipx/venvs/platformio/bin/python -m ensurepip --upgrade
```

Then rerun the `PLATFORMIO_CORE_DIR=../.platformio pio run` command.

Serial output is line-oriented:

```text
NK_ADC,seq,ms,adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7,adc8
```

The default sample period is 100 ms. Change it in `platformio.ini` with `NOSEKNOWS_SAMPLE_PERIOD_MS`.
