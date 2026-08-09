"""Diagnose acquisition-dependent multiplier spread for IPIX BC-DRCFAR development."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "block_ratios.csv"
SUMMARY = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "summary.json"
OUTPUT = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "multiplier_spread"


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    values = sorted(values)
    idx = int((len(values) - 1) * q)
    return float(values[idx])


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def load_rows() -> list[dict[str, str]]:
    with INPUT.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = [row for row in load_rows() if row["role"] == "clutter"]
    global_vals = [float(row["ratio_bcdrcfar"]) for row in rows]
    global_stats = {
        "q50": quantile(global_vals, 0.50),
        "q90": quantile(global_vals, 0.90),
        "q99": quantile(global_vals, 0.99),
        "mean": mean(global_vals),
    }
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    current_global = float(summary["multipliers"]["bcdrcfar"])

    per_file: list[dict[str, float | str]] = []
    for file_id in sorted({row["file_id"] for row in rows}):
        vals = [float(row["ratio_bcdrcfar"]) for row in rows if row["file_id"] == file_id]
        per_file.append(
            {
                "file_id": file_id,
                "blocks": len(vals),
                "q50": quantile(vals, 0.50),
                "q90": quantile(vals, 0.90),
                "q99": quantile(vals, 0.99),
                "mean": mean(vals),
                "rel_to_global": quantile(vals, 0.99) / current_global,
            }
        )

    per_polarization: list[dict[str, float | str]] = []
    for file_id in sorted({row["file_id"] for row in rows}):
        for polarization in sorted({row["polarization"] for row in rows if row["file_id"] == file_id}):
            vals = [
                float(row["ratio_bcdrcfar"])
                for row in rows
                if row["file_id"] == file_id and row["polarization"] == polarization
            ]
            per_polarization.append(
                {
                    "file_id": file_id,
                    "polarization": polarization,
                    "q50": quantile(vals, 0.50),
                    "q90": quantile(vals, 0.90),
                    "q99": quantile(vals, 0.99),
                    "mean": mean(vals),
                }
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "global_stats.json").write_text(json.dumps(global_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUTPUT / "per_file.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_id", "blocks", "q50", "q90", "q99", "mean", "rel_to_global"])
        writer.writeheader()
        writer.writerows(per_file)
    with (OUTPUT / "per_file_polarization.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_id", "polarization", "q50", "q90", "q99", "mean"])
        writer.writeheader()
        writer.writerows(per_polarization)

    print(json.dumps({
        "current_global_multiplier": current_global,
        "global_stats": global_stats,
        "per_file": per_file,
        "per_polarization": per_polarization,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
