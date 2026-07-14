#!/usr/bin/env python3
"""Recipe-driven dataset materialization for NoseKnows.

This tool is intentionally outside the Rust training path. It builds plain
artifacts that existing Rust binaries can consume without knowing about Daft.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

LABEL_NO_SCENT = "No Scent"
ADC_COLUMNS = [f"adc{i}" for i in range(9)]


def main() -> int:
    args = parse_args()
    recipe = load_recipe(args.recipe)
    output_dir = Path(recipe["output_dir"])

    if args.print_output_dir:
        materialize(recipe, require_daft=not args.allow_stdlib_fallback, quiet=True)
        print(output_dir)
        return 0

    result = materialize(recipe, require_daft=not args.allow_stdlib_fallback, quiet=False)
    print(
        "Materialized "
        f"{result['selected']} / {result['total']} capture(s) "
        f"to {output_dir}"
    )
    print(f"Manifest: {recipe['manifest_path']}")
    print(f"View manifest: {recipe['view_manifest_path']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument(
        "--allow-stdlib-fallback",
        action="store_true",
        help="Use Python stdlib filtering if Daft is not installed.",
    )
    parser.add_argument(
        "--print-output-dir",
        action="store_true",
        help="Materialize quietly, then print only the recipe output_dir.",
    )
    return parser.parse_args()


def load_recipe(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        recipe = tomllib.load(handle)

    required = [
        "inputs",
        "output_dir",
        "manifest_path",
        "view_manifest_path",
        "include_label_counts",
    ]
    for key in required:
        if key not in recipe:
            raise SystemExit(f"{path} missing required key: {key}")
    return recipe


def materialize(
    recipe: dict[str, Any], *, require_daft: bool, quiet: bool
) -> dict[str, int]:
    rows = build_manifest_rows([Path(value) for value in recipe["inputs"]])
    selected = select_rows(rows, recipe, require_daft=require_daft)

    if recipe.get("shuffle", False):
        rng = random.Random(int(recipe.get("seed", 0)))
        rng.shuffle(selected)

    selected = apply_limits(selected, recipe.get("limits_per_label_count", {}))

    manifest_path = Path(recipe["manifest_path"])
    view_manifest_path = Path(recipe["view_manifest_path"])
    output_dir = Path(recipe["output_dir"])
    write_manifest(manifest_path, rows)
    write_manifest(view_manifest_path, selected)

    if recipe.get("copy_csv", True):
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in selected:
            source = Path(row["source_path"])
            shutil.copy2(source, output_dir / source.name)

    if not quiet:
        print_selection_summary(rows, selected)

    return {"total": len(rows), "selected": len(selected)}


def build_manifest_rows(input_dirs: list[Path]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for input_dir in input_dirs:
        paths.extend(sorted(input_dir.glob("*.csv")))

    rows = []
    for path in paths:
        row = capture_manifest_row(path)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: row["sample_id"])
    return rows


def capture_manifest_row(path: Path) -> dict[str, Any] | None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader, None)
        if first is None:
            return None

        row_count = 1
        max_elapsed = int_or_zero(first.get("host_elapsed_ms"))
        peaks = {column: int_or_zero(first.get(column)) for column in ADC_COLUMNS}
        for record in reader:
            row_count += 1
            max_elapsed = max(max_elapsed, int_or_zero(record.get("host_elapsed_ms")))
            for column in ADC_COLUMNS:
                peaks[column] = max(peaks[column], int_or_zero(record.get(column)))

    labels = [first.get("label_1", ""), first.get("label_2", ""), first.get("label_3", "")]
    label_count = sum(1 for label in labels if label and label != LABEL_NO_SCENT)
    sample_id = first.get("sample_id", path.stem)
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "sample_name": first.get("sample_name", ""),
        "source_path": str(path),
        "source_kind": source_kind(sample_id, label_count),
        "label_1": labels[0],
        "label_2": labels[1],
        "label_3": labels[2],
        "label_count": label_count,
        "row_count": row_count,
        "duration_ms": max_elapsed,
        "saturation_count": sum(1 for value in peaks.values() if value >= 4095),
    }
    for column in ADC_COLUMNS:
        result[f"{column}_peak"] = peaks[column]
    return result


def source_kind(sample_id: str, label_count: int) -> str:
    if sample_id.startswith("designer_"):
        return "designer"
    if sample_id.startswith("no_scent_") or label_count == 0:
        return "no_scent"
    if sample_id.startswith("single_"):
        return "single"
    if sample_id.startswith("two_"):
        return "two"
    if sample_id.startswith("three_"):
        return "three"
    if sample_id.startswith("synthetic_"):
        return "synthetic"
    return "real"


def select_rows(
    rows: list[dict[str, Any]], recipe: dict[str, Any], *, require_daft: bool
) -> list[dict[str, Any]]:
    try:
        import daft  # type: ignore
        from daft import col  # type: ignore
    except ModuleNotFoundError:
        if require_daft:
            raise SystemExit(
                "Daft is not installed. Install Daft for the intended path, "
                "or pass --allow-stdlib-fallback for local smoke testing."
            )
        return select_rows_stdlib(rows, recipe)

    include_counts = [int(value) for value in recipe["include_label_counts"]]
    exclude_prefixes = list(recipe.get("exclude_prefixes", []))

    df = daft.from_pylist(rows)
    df = df.where(col("label_count").is_in(include_counts))
    for prefix in exclude_prefixes:
        df = df.where(~col("sample_id").str.startswith(prefix))
    selected = df.collect().to_pylist()
    return [dict(row) for row in selected]


def select_rows_stdlib(rows: list[dict[str, Any]], recipe: dict[str, Any]) -> list[dict[str, Any]]:
    include_counts = {int(value) for value in recipe["include_label_counts"]}
    exclude_prefixes = tuple(recipe.get("exclude_prefixes", []))
    return [
        row
        for row in rows
        if int(row["label_count"]) in include_counts
        and not str(row["sample_id"]).startswith(exclude_prefixes)
    ]


def apply_limits(rows: list[dict[str, Any]], limits: dict[str, Any]) -> list[dict[str, Any]]:
    if not limits:
        return rows

    counts: dict[int, int] = {}
    selected = []
    normalized = {int(key): int(value) for key, value in limits.items()}
    for row in rows:
        label_count = int(row["label_count"])
        limit = normalized.get(label_count)
        if limit is None:
            selected.append(row)
            continue
        current = counts.get(label_count, 0)
        if current < limit:
            selected.append(row)
            counts[label_count] = current + 1
    return selected


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "sample_name",
        "source_path",
        "source_kind",
        "label_1",
        "label_2",
        "label_3",
        "label_count",
        "row_count",
        "duration_ms",
        *[f"{column}_peak" for column in ADC_COLUMNS],
        "saturation_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_selection_summary(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    print("Dataset manifest summary:")
    for title, data in [("all", rows), ("selected", selected)]:
        counts: dict[int, int] = {}
        for row in data:
            label_count = int(row["label_count"])
            counts[label_count] = counts.get(label_count, 0) + 1
        parts = ", ".join(f"{key}-note={counts[key]}" for key in sorted(counts))
        print(f"  {title}: {len(data)} capture(s) {parts}")


def int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    sys.exit(main())
