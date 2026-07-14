#!/usr/bin/env python3
"""Materialize live synthetic ADC frame chunks from injector state.

This is the Daft/Python input-orchestration side of PLAN-001. It writes plain
CSV frames for the Rust live runner; it does not contain model logic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

NO_SCENT = "No Scent"
LABELS = [
    "Floral",
    "Soft Floral",
    "Floral Amber",
    "Amber",
    "Soft Amber",
    "Woody Amber",
    "Woods",
    "Mossy Woods",
    "Dry Woods",
    "Aromatic",
    "Citrus",
    "Water",
    "Green",
    "Fruity",
]

CHANNELS = 9
SAMPLE_PERIOD_MS = 100
BASELINE = [220.0, 225.0, 210.0, 215.0, 220.0, 205.0, 215.0, 225.0, 180.0]

# adc0 MQ-2, adc1 MQ-3, adc2 MQ-5, adc3 MQ-6, adc4 MQ-7,
# adc5 MQ-8, adc6 MQ-9, adc7 MQ-135, adc8 MQ-4 placeholder.
PROFILES: dict[str, list[float]] = {
    "Citrus": [3800, 2000, 3500, 3700, 0, 0, 0, 2200, 0],
    "Water": [0, 800, 0, 0, 0, 0, 0, 1800, 0],
    "Green": [0, 2500, 0, 600, 0, 0, 0, 3000, 0],
    "Fruity": [1200, 3800, 0, 0, 0, 0, 0, 2200, 0],
    "Floral": [0, 4095, 0, 0, 0, 0, 0, 1500, 0],
    "Soft Floral": [0, 2200, 0, 0, 2100, 0, 0, 2200, 0],
    "Floral Amber": [1100, 4095, 0, 0, 3400, 0, 3000, 3800, 0],
    "Soft Amber": [900, 1200, 0, 0, 3700, 0, 0, 3500, 0],
    "Amber": [2200, 2500, 0, 0, 3900, 0, 3800, 4095, 0],
    "Woody Amber": [3500, 1800, 0, 0, 3000, 0, 3400, 4000, 0],
    "Woods": [3100, 900, 0, 0, 0, 0, 0, 3200, 0],
    "Mossy Woods": [2800, 2900, 0, 0, 3100, 0, 0, 3000, 0],
    "Dry Woods": [2000, 1100, 0, 0, 3900, 0, 3600, 3800, 0],
    "Aromatic": [2400, 3600, 0, 2900, 2900, 0, 0, 3400, 0],
}


def main() -> None:
    args = parse_args()
    state = load_or_create_state(args.state)
    rows = materialize_rows(state)
    write_rows(args.out, rows)
    write_events(args.events, rows)
    print(f"Materialized live input frames: {args.out}")
    print(f"frames={len(rows)} segments={len(set(row['stream_segment'] for row in rows))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=Path("data/live/injector_state.json"))
    parser.add_argument("--out", type=Path, default=Path("data/live/input_frames.csv"))
    parser.add_argument("--events", type=Path, default=Path("data/live/input_events.csv"))
    return parser.parse_args()


def load_or_create_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())

    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "sample_period_ms": SAMPLE_PERIOD_MS,
        "seed": 20260714,
        "sequence": [
            {"notes": [], "duration_secs": 4},
            {"notes": ["Citrus"], "duration_secs": 10, "intensity": 1.0},
            {"notes": [], "duration_secs": 4},
            {"notes": ["Water", "Citrus"], "duration_secs": 10, "intensity": 0.85},
            {"notes": [], "duration_secs": 4},
            {"notes": ["Amber"], "duration_secs": 10, "intensity": 0.8},
            {"notes": [], "duration_secs": 6},
        ],
    }
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def materialize_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    period_ms = int(state.get("sample_period_ms", SAMPLE_PERIOD_MS))
    sequence = state.get("sequence")
    if not sequence:
        sequence = [
            {
                "notes": state.get("active_notes", []),
                "duration_secs": state.get("duration_secs", state.get("chunk_secs", 10)),
                "intensity": state.get("intensity", 1.0),
            }
        ]

    rows: list[dict[str, Any]] = []
    elapsed_ms = 0
    device_seq = 0
    for segment_index, segment in enumerate(sequence):
        notes = normalize_notes(segment.get("notes", []))
        duration_secs = float(segment.get("duration_secs", state.get("chunk_secs", 10)))
        intensity = float(segment.get("intensity", 1.0))
        row_count = max(1, round(duration_secs * 1000 / period_ms))
        labels = labels_for_notes(notes)
        segment_id = f"live_{segment_index:04}_{slug(notes)}"

        for offset in range(row_count):
            t = offset * period_ms / 1000.0
            adc = synthesize_adc(notes, t, duration_secs, intensity)
            rows.append(
                {
                    "stream_segment": segment_id,
                    "source_sample_id": segment_id,
                    "sample_id": segment_id,
                    "sample_name": segment_name(notes),
                    "label_1": labels[0],
                    "label_2": labels[1],
                    "label_3": labels[2],
                    "host_elapsed_ms": elapsed_ms,
                    "host_unix_ms": 0,
                    "device_seq": device_seq,
                    "device_ms": elapsed_ms,
                    **{f"adc{index}": round(value) for index, value in enumerate(adc)},
                }
            )
            elapsed_ms += period_ms
            device_seq += 1

    return rows


def normalize_notes(notes: Any) -> list[str]:
    if notes in (None, "", NO_SCENT):
        return []
    if isinstance(notes, str):
        notes = [notes]
    normalized = []
    for note in notes:
        candidate = str(note).strip()
        if not candidate or candidate == NO_SCENT:
            continue
        match = next((label for label in LABELS if label.lower() == candidate.lower()), None)
        if match is None:
            raise ValueError(f"unknown note {candidate!r}")
        if match not in normalized:
            normalized.append(match)
    return normalized[:3]


def labels_for_notes(notes: list[str]) -> list[str]:
    labels = notes[:3]
    while len(labels) < 3:
        labels.append(NO_SCENT)
    return labels


def segment_name(notes: list[str]) -> str:
    return "No Scent" if not notes else " + ".join(notes)


def slug(notes: list[str]) -> str:
    if not notes:
        return "no_scent"
    return "_".join(note.lower().replace(" ", "_") for note in notes)


def synthesize_adc(notes: list[str], t: float, duration_secs: float, intensity: float) -> list[float]:
    if not notes:
        return BASELINE[:]

    rise = 1.0 - math.exp(-t / 1.2)
    release_start = max(0.0, duration_secs - 2.0)
    release = 1.0 if t < release_start else math.exp(-(t - release_start) / 2.5)
    envelope = min(1.0, rise) * release

    adc = BASELINE[:]
    for note in notes:
        profile = PROFILES[note]
        for channel, peak in enumerate(profile):
            if peak <= 0:
                continue
            target = BASELINE[channel] + (peak - BASELINE[channel]) * intensity * envelope
            # Blends compress into a 12-bit ADC rather than adding linearly forever.
            adc[channel] = max(adc[channel], target)

    return [min(4095.0, max(0.0, value)) for value in adc]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stream_segment",
        "source_sample_id",
        "sample_id",
        "sample_name",
        "label_1",
        "label_2",
        "label_3",
        "host_elapsed_ms",
        "host_unix_ms",
        "device_seq",
        "device_ms",
        *[f"adc{index}" for index in range(CHANNELS)],
    ]
    with path.open("w") as handle:
        handle.write(",".join(fields) + "\n")
        for row in rows:
            handle.write(",".join(csv_escape(str(row[field])) for field in fields) + "\n")


def write_events(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("stream_segment,row_start,row_end,labels\n")
        index = 0
        while index < len(rows):
            segment = rows[index]["stream_segment"]
            start = index
            while index < len(rows) and rows[index]["stream_segment"] == segment:
                index += 1
            labels = "|".join(
                label
                for label in [
                    rows[start]["label_1"],
                    rows[start]["label_2"],
                    rows[start]["label_3"],
                ]
                if label != NO_SCENT
            ) or NO_SCENT
            handle.write(f"{csv_escape(segment)},{start},{index},{csv_escape(labels)}\n")


def csv_escape(value: str) -> str:
    if any(ch in value for ch in [",", '"', "\n", "\r"]):
        return '"' + value.replace('"', '""') + '"'
    return value


if __name__ == "__main__":
    main()
