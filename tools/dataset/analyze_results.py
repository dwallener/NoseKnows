#!/usr/bin/env python3
"""Analyze NoseKnows Rust inference result artifacts.

Rust writes plain result CSVs. This tool treats those results as a test ledger:
it joins optional manifest metadata, summarizes pass/fail behavior, and writes
small CSV/Markdown artifacts for review.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

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


def main() -> int:
    args = parse_args()
    rows = load_result_rows(args.results, args.manifest, require_daft=not args.allow_stdlib_fallback)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bucket_rows = bucket_summary(rows)
    label_rows = label_summary(rows)
    failure_rows = failure_summary(rows)

    write_csv(args.out_dir / "summary_by_label_count.csv", bucket_rows)
    write_csv(args.out_dir / "summary_by_label.csv", label_rows)
    write_csv(args.out_dir / "failure_reasons.csv", failure_rows)
    write_report(args.out_dir / "report.md", rows, bucket_rows, label_rows, failure_rows, args)

    print(f"Analyzed {len(rows)} result row(s)")
    print(f"Report: {args.out_dir / 'report.md'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/captures.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/runs/peak_stream_report"))
    parser.add_argument(
        "--allow-stdlib-fallback",
        action="store_true",
        help="Use Python stdlib analysis if Daft is not installed.",
    )
    return parser.parse_args()


def load_result_rows(results: Path, manifest: Path, *, require_daft: bool) -> list[dict[str, Any]]:
    try:
        import daft  # type: ignore
        from daft import col  # type: ignore
    except ModuleNotFoundError:
        if require_daft:
            raise SystemExit(
                "Daft is not installed. Install Daft for the intended path, "
                "or pass --allow-stdlib-fallback for local smoke testing."
            )
        return load_result_rows_stdlib(results, manifest)

    df = daft.read_csv(str(results))
    if manifest.exists():
        manifest_df = daft.read_csv(str(manifest)).select(
            col("sample_id").alias("source_sample_id"),
            col("source_kind").alias("manifest_source_kind"),
            col("saturation_count").alias("manifest_saturation_count"),
            col("duration_ms").alias("manifest_duration_ms"),
        )
        df = df.join(manifest_df, on="source_sample_id", how="left")
    return [normalize_row(dict(row)) for row in df.collect().to_pylist()]


def load_result_rows_stdlib(results: Path, manifest: Path) -> list[dict[str, Any]]:
    manifest_by_sample = {}
    if manifest.exists():
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                manifest_by_sample[row.get("sample_id", "")] = row

    rows = []
    with results.open(newline="") as handle:
        for row in csv.DictReader(handle):
            manifest_row = manifest_by_sample.get(row.get("source_sample_id", ""))
            if manifest_row:
                row["manifest_source_kind"] = manifest_row.get("source_kind", "")
                row["manifest_saturation_count"] = manifest_row.get("saturation_count", "")
                row["manifest_duration_ms"] = manifest_row.get("duration_ms", "")
            rows.append(normalize_row(row))
    return rows


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in [
        "label_count",
        "covered_labels",
        "target_labels",
        "row_start",
        "row_end",
        "settled_skip_rows",
    ]:
        row[key] = int_or_zero(row.get(key))
    for key in ["gate_threshold", "score_1", "score_2", "score_3"]:
        row[key] = float_or_zero(row.get(key))
    for key in ["silent", "p_at_1", "any_at_3", "passed"]:
        row[key] = bool_value(row.get(key))
    return row


def bucket_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["label_count"], []).append(row)

    output = []
    for label_count in sorted(groups):
        group = groups[label_count]
        total_targets = sum(row["target_labels"] for row in group)
        covered = sum(row["covered_labels"] for row in group)
        output.append(
            {
                "label_count": label_count,
                "segments": len(group),
                "passed": count_true(group, "passed"),
                "pass_rate": pct(count_true(group, "passed"), len(group)),
                "silent": count_true(group, "silent"),
                "false_positive": sum(1 for row in group if label_count == 0 and not row["silent"]),
                "p_at_1": pct(count_true(group, "p_at_1"), len(group)),
                "any_at_3": pct(count_true(group, "any_at_3"), len(group)),
                "coverage": pct(covered, total_targets),
                "covered_labels": covered,
                "target_labels": total_targets,
            }
        )
    return output


def label_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for label in LABELS:
        support = 0
        predicted = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for row in rows:
            expected = {row.get("label_1"), row.get("label_2"), row.get("label_3")} - {"No Scent", ""}
            preds = {row.get("pred_1"), row.get("pred_2"), row.get("pred_3")} if not row["silent"] else set()
            if label in expected:
                support += 1
            if label in preds:
                predicted += 1
            if label in expected and label in preds:
                true_positive += 1
            elif label in expected:
                false_negative += 1
            elif label in preds:
                false_positive += 1
        output.append(
            {
                "label": label,
                "support": support,
                "predicted": predicted,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": pct(true_positive, predicted),
                "recall": pct(true_positive, support),
            }
        )
    return output


def failure_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[int, str], int] = {}
    for row in rows:
        if row["passed"]:
            continue
        key = (row["label_count"], row.get("failure_reason", "failed"))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"label_count": label_count, "failure_reason": reason, "count": count}
        for (label_count, reason), count in sorted(counts.items())
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    run_id = rows[0].get("run_id", "") if rows else ""
    lines = [
        "# NoseKnows Inference Run Report",
        "",
        f"- run_id: `{run_id}`",
        f"- results: `{args.results}`",
        f"- manifest: `{args.manifest}`",
        f"- rows: `{len(rows)}`",
        "",
        "## By Label Count",
        "",
        "| label_count | segments | pass_rate | any_at_3 | coverage | false_positive |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bucket_rows:
        lines.append(
            f"| {row['label_count']} | {row['segments']} | {row['pass_rate']:.2f}% | "
            f"{row['any_at_3']:.2f}% | {row['coverage']:.2f}% | {row['false_positive']} |"
        )

    lines.extend(["", "## Lowest Recall Labels", ""])
    for row in sorted(label_rows, key=lambda item: (item["recall"], -item["support"]))[:8]:
        if row["support"] == 0:
            continue
        lines.append(
            f"- {row['label']}: recall {row['recall']:.2f}% "
            f"({row['true_positive']}/{row['support']}), fp={row['false_positive']}"
        )

    lines.extend(["", "## Failure Reasons", ""])
    if failure_rows:
        for row in failure_rows:
            lines.append(
                f"- label_count={row['label_count']} {row['failure_reason']}: {row['count']}"
            )
    else:
        lines.append("- none")

    path.write_text("\n".join(lines) + "\n")


def count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row[key])


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator * 100.0 / denominator


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    sys.exit(main())
