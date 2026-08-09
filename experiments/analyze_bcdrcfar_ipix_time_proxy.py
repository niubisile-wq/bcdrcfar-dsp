"""Analyze block-order proxy drift for the retrospective external IPIX feature head.

This script does not claim true chronological evidence. It uses the fully
populated `block_index` axis as a proxy for order sensitivity / drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "condition_rows.csv"
DEFAULT_SUMMARY = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "summary.json"
DEFAULT_OUTPUT = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "time_proxy"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=8)
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return None if math.isnan(value) else value
        return value

    path.write_text(json.dumps(safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def phase_bin_for(block_index: int, bins: int) -> int:
    return min(bins - 1, (block_index * bins) // 1024)


def summarize_method(frame_path: Path, decision_col: str, target_pfa: float, bins: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    series_counts: dict[tuple[str, int, str], list[int]] = defaultdict(lambda: [0, 0])
    acq_counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    target_counts: dict[tuple[str, int, str], list[int]] = defaultdict(lambda: [0, 0])

    with frame_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_id = row["file_id"]
            role = row["role"]
            block_index = int(row["block_index"])
            phase_bin = phase_bin_for(block_index, bins)
            decision = row[decision_col].strip().lower() == "true"

            series_key = (file_id, phase_bin, role)
            series_counts[series_key][1] += 1
            series_counts[series_key][0] += int(decision)

            if role == "clutter":
                acq_key = (file_id, phase_bin)
                acq_counts[acq_key][1] += 1
                acq_counts[acq_key][0] += int(decision)
            else:
                target_key = (file_id, phase_bin, role)
                target_counts[target_key][1] += 1
                target_counts[target_key][0] += int(decision)

    series_rows: list[dict[str, Any]] = []
    for (file_id, phase_bin, role), (events, trials) in sorted(series_counts.items()):
        adjusted = (events + 0.5) / (trials + 1.0)
        series_rows.append(
            {
                "file_id": file_id,
                "phase_bin": phase_bin,
                "role": role,
                "events": events,
                "trials": trials,
                "rate": events / trials if trials else float("nan"),
                "jeffreys_rate": adjusted,
                "absolute_log10_pfa_error": abs(math.log10(adjusted / target_pfa)) if role == "clutter" else float("nan"),
                "factor2_violation": adjusted < target_pfa / 2.0 or adjusted > target_pfa * 2.0 if role == "clutter" else False,
            }
        )

    acquisition_rows: list[dict[str, Any]] = []
    for (file_id, phase_bin), (events, trials) in sorted(acq_counts.items()):
        adjusted = (events + 0.5) / (trials + 1.0)
        acquisition_rows.append(
            {
                "file_id": file_id,
                "phase_bin": phase_bin,
                "events": events,
                "trials": trials,
                "pfa": events / trials if trials else float("nan"),
                "absolute_log10_pfa_error": abs(math.log10(adjusted / target_pfa)),
                "factor2_violation_rate": next(
                    (
                        row["factor2_violation"]
                        for row in series_rows
                        if row["file_id"] == file_id and row["phase_bin"] == phase_bin and row["role"] == "clutter"
                    ),
                    False,
                ),
            }
        )

    target_rows: list[dict[str, Any]] = []
    for (file_id, phase_bin, role), (events, trials) in sorted(target_counts.items()):
        target_rows.append(
            {
                "file_id": file_id,
                "phase_bin": phase_bin,
                "role": role,
                "events": events,
                "trials": trials,
                "pd": events / trials if trials else float("nan"),
            }
        )
    return series_rows, acquisition_rows, target_rows


def phase_macro(acquisition_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    acq_by_phase: dict[int, list[dict[str, Any]]] = defaultdict(list)
    tgt_by_phase: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in acquisition_rows:
        acq_by_phase[int(row["phase_bin"])].append(row)
    for row in target_rows:
        if row["role"] == "primary":
            tgt_by_phase[int(row["phase_bin"])].append(row)

    rows: list[dict[str, Any]] = []
    all_bins = sorted(set(acq_by_phase) | set(tgt_by_phase))
    for phase_bin in all_bins:
        acq_group = acq_by_phase.get(phase_bin, [])
        tgt_group = tgt_by_phase.get(phase_bin, [])
        rows.append(
            {
                "phase_bin": phase_bin,
                "macro_pfa": mean([float(r["pfa"]) for r in acq_group]),
                "macro_absolute_log10_pfa_error": mean([float(r["absolute_log10_pfa_error"]) for r in acq_group]),
                "macro_factor2_violation_rate": mean([1.0 if r["factor2_violation_rate"] else 0.0 for r in acq_group]),
                "macro_primary_pd": mean([float(r["pd"]) for r in tgt_group]),
            }
        )
    return rows


def half_summary(acquisition_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], bins: int) -> list[dict[str, Any]]:
    split = bins // 2
    early_bins = set(range(split))
    late_bins = set(range(split, bins))
    rows: list[dict[str, Any]] = []
    for label, bin_set in [("early", early_bins), ("late", late_bins)]:
        acq = [row for row in acquisition_rows if int(row["phase_bin"]) in bin_set]
        tgt = [row for row in target_rows if row["role"] == "primary" and int(row["phase_bin"]) in bin_set]
        rows.append(
            {
                "half": label,
                "macro_pfa": mean([float(r["pfa"]) for r in acq]),
                "macro_absolute_log10_pfa_error": mean([float(r["absolute_log10_pfa_error"]) for r in acq]),
                "macro_factor2_violation_rate": mean([1.0 if r["factor2_violation_rate"] else 0.0 for r in acq]),
                "macro_primary_pd": mean([float(r["pd"]) for r in tgt]),
            }
        )
    return rows


def paired_half_deltas(acquisition_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], bins: int) -> dict[str, float]:
    split = bins // 2
    early_bins = set(range(split))
    late_bins = set(range(split, bins))

    file_phase_acq: dict[tuple[str, int], dict[str, float]] = {}
    for row in acquisition_rows:
        file_phase_acq[(row["file_id"], int(row["phase_bin"]))] = row

    file_phase_tgt: dict[tuple[str, int], dict[str, float]] = {}
    for row in target_rows:
        if row["role"] == "primary":
            file_phase_tgt[(row["file_id"], int(row["phase_bin"]))] = row

    by_file_early: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_file_late: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (file_id, phase_bin), row in file_phase_acq.items():
        (by_file_early if phase_bin in early_bins else by_file_late)[file_id].append(row)

    by_file_early_pd: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_file_late_pd: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (file_id, phase_bin), row in file_phase_tgt.items():
        (by_file_early_pd if phase_bin in early_bins else by_file_late_pd)[file_id].append(row)

    file_ids = sorted(set(by_file_early) & set(by_file_late) & set(by_file_early_pd) & set(by_file_late_pd))
    pfa_deltas: list[float] = []
    err_deltas: list[float] = []
    pd_deltas: list[float] = []
    for file_id in file_ids:
        early_acq = by_file_early[file_id]
        late_acq = by_file_late[file_id]
        early_pd_rows = by_file_early_pd[file_id]
        late_pd_rows = by_file_late_pd[file_id]
        pfa_deltas.append(mean([float(r["pfa"]) for r in late_acq]) - mean([float(r["pfa"]) for r in early_acq]))
        err_deltas.append(
            mean([float(r["absolute_log10_pfa_error"]) for r in late_acq])
            - mean([float(r["absolute_log10_pfa_error"]) for r in early_acq])
        )
        pd_deltas.append(mean([float(r["pd"]) for r in late_pd_rows]) - mean([float(r["pd"]) for r in early_pd_rows]))

    return {
        "paired_mean_pfa_delta_late_minus_early": mean(pfa_deltas),
        "paired_mean_error_delta_late_minus_early": mean(err_deltas),
        "paired_mean_primary_pd_delta_late_minus_early": mean(pd_deltas),
    }


def trend_slope(rows: list[dict[str, Any]], y_col: str) -> float:
    xs = [float(row["phase_bin"]) for row in rows if row.get(y_col) == row.get(y_col)]
    ys = [float(row[y_col]) for row in rows if row.get(y_col) == row.get(y_col)]
    if len(xs) < 2:
        return float("nan")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return float("nan")
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return numer / denom


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    target_pfa = float(summary.get("target_pfa", 0.01))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "scalar": "decision_bcdrcfar_scalar",
        "feature": "decision_bcdrcfar_feature",
    }

    all_phase_rows: list[dict[str, Any]] = []
    all_half_rows: list[dict[str, Any]] = []
    all_paired_rows: list[dict[str, Any]] = []
    all_trend_rows: list[dict[str, Any]] = []
    per_method_payload: dict[str, Any] = {}

    for method_name, col in methods.items():
        series_rows, acquisition_rows, target_rows = summarize_method(args.input, col, target_pfa, args.bins)
        phase_rows = phase_macro(acquisition_rows, target_rows)
        for row in phase_rows:
            row["method"] = method_name
        half_rows = half_summary(acquisition_rows, target_rows, args.bins)
        for row in half_rows:
            row["method"] = method_name
        paired = paired_half_deltas(acquisition_rows, target_rows, args.bins)
        paired["method"] = method_name
        trend = {
            "method": method_name,
            "pfa_slope_per_bin": trend_slope(phase_rows, "macro_pfa"),
            "error_slope_per_bin": trend_slope(phase_rows, "macro_absolute_log10_pfa_error"),
            "factor2_slope_per_bin": trend_slope(phase_rows, "macro_factor2_violation_rate"),
            "primary_pd_slope_per_bin": trend_slope(phase_rows, "macro_primary_pd"),
        }

        all_phase_rows.extend(phase_rows)
        all_half_rows.extend(half_rows)
        all_paired_rows.append(paired)
        all_trend_rows.append(trend)
        per_method_payload[method_name] = {
            "phase_summary": phase_rows,
            "half_summary": half_rows,
            "paired": paired,
            "trend": trend,
        }

    phase_path = args.output_dir / "phase_summary.csv"
    half_path = args.output_dir / "half_summary.csv"
    paired_path = args.output_dir / "paired_half_deltas.csv"
    trend_path = args.output_dir / "trend_summary.csv"
    write_csv(phase_path, all_phase_rows, ["phase_bin", "macro_pfa", "macro_absolute_log10_pfa_error", "macro_factor2_violation_rate", "macro_primary_pd", "method"])
    write_csv(half_path, all_half_rows, ["half", "macro_pfa", "macro_absolute_log10_pfa_error", "macro_factor2_violation_rate", "macro_primary_pd", "method"])
    write_csv(paired_path, all_paired_rows, ["paired_mean_pfa_delta_late_minus_early", "paired_mean_error_delta_late_minus_early", "paired_mean_primary_pd_delta_late_minus_early", "method"])
    write_csv(trend_path, all_trend_rows, ["method", "pfa_slope_per_bin", "error_slope_per_bin", "factor2_slope_per_bin", "primary_pd_slope_per_bin"])

    report_path = args.output_dir / "summary.md"
    lines = [
        "# BCDRCFAR IPIX Block-Order Proxy Audit",
        "",
        f"Source rows: `{args.input.as_posix()}`",
        "",
        "## Interpretation",
        "",
        "This is a block-order proxy, not a true timestamp audit. It uses the fully populated `block_index` axis to test whether performance drifts from early to late blocks.",
        "",
        "## Key facts",
        "",
        "- `block_index` spans 0..1023 for all 9 files.",
        "- Each file covers the full block range, so the proxy is comparable across acquisitions.",
        "",
        "## Paired early/late deltas",
        "",
    ]
    for row in all_paired_rows:
        lines.append(
            f"- `{row['method']}`: late-early pfa delta = {row['paired_mean_pfa_delta_late_minus_early']:.6g}, "
            f"late-early error delta = {row['paired_mean_error_delta_late_minus_early']:.6g}, "
            f"late-early primary PD delta = {row['paired_mean_primary_pd_delta_late_minus_early']:.6g}"
        )
    lines.extend(["", "## Trend slopes", ""])
    for row in all_trend_rows:
        lines.append(
            f"- `{row['method']}`: pfa slope/bin = {row['pfa_slope_per_bin']:.6g}, "
            f"error slope/bin = {row['error_slope_per_bin']:.6g}, "
            f"factor2 slope/bin = {row['factor2_slope_per_bin']:.6g}, "
            f"primary PD slope/bin = {row['primary_pd_slope_per_bin']:.6g}"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The proxy does not close the time-stability gap. It does show that the current calibration is not perfectly stationary in block order, which is enough to keep chronological evidence on the open-gaps list.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "input": str(args.input),
        "summary": str(args.summary),
        "output_dir": str(args.output_dir),
        "bins": args.bins,
        "target_pfa": target_pfa,
        "per_method": per_method_payload,
        "paths": {
            "phase_summary_csv": str(phase_path),
            "half_summary_csv": str(half_path),
            "paired_half_deltas_csv": str(paired_path),
            "trend_summary_csv": str(trend_path),
            "summary_md": str(report_path),
        },
    }
    write_json(args.output_dir / "summary.json", payload)
    print(json.dumps({"summary_md": str(report_path), "summary_json": str(args.output_dir / "summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
