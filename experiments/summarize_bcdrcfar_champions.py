from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "results" / "bcdrcfar_development_cv"
SWEEP_DIR = ROOT / "results" / "bcdrcfar_multiplier_sweep"
OUT_DIR = ROOT / "results" / "bcdrcfar_champion_tracking"


def _read_overall(path: Path, label: str) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "label": label,
                "source": str(path.relative_to(ROOT)),
                "method": row["method"],
                "cells": int(row.get("cells", 0)),
                "pooled_pfa": float(row["pooled_pfa"]),
                "factor2_violation_rate": float(row["factor2_violation_rate"]),
                "median_absolute_log10_pfa_error": float(row["median_absolute_log10_pfa_error"]),
                "pd_at_0db": float(row.get("pd_at_0db", row.get("pd", float("nan")))),
            }
        )
    return rows


def _latest_sweep_summary() -> Path | None:
    paths = sorted(SWEEP_DIR.glob("*_summary.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    candidates = [
        ("k24_single_n1024", CV_DIR / "v2_scaleeq_3a1aa9f4_seed2026080711_K24_n1024_pfa0.01_overall.csv"),
        ("k24_single_n4096", CV_DIR / "v2_scaleeq_3a1aa9f4_seed2026080711_K24_n4096_pfa0.01_overall.csv"),
        (
            "k24_single_2seed_n1024",
            CV_DIR / "v2_scaleeq_3a1aa9f4_seeds2026080711-2026080712_K24_n1024_pfa0.01_overall.csv",
        ),
        (
            "k24_single_3seed_n1024",
            CV_DIR / "v2_scaleeq_3a1aa9f4_seeds2026080711-2026080712-2026080713_K24_n1024_pfa0.01_overall.csv",
        ),
        ("ensemble3_m1_n1024", CV_DIR / "median_ensemble3_v2_K24_n1024_pfa0.01_m1.000_overall.csv"),
        ("ensemble3_m099_n1024", CV_DIR / "median_ensemble3_v2_K24_n1024_pfa0.01_m0.990_overall.csv"),
    ]
    for label, path in candidates:
        if path.exists():
            rows.extend(_read_overall(path, label))

    sweep_path = _latest_sweep_summary()
    if sweep_path is not None:
        rows.extend(_read_overall(sweep_path, f"fold0_sweep_{sweep_path.stem}"))

    frame = pd.DataFrame(rows)
    summary_path = OUT_DIR / "development_champion_candidates.csv"
    frame.to_csv(summary_path, index=False)

    bcd = frame[frame["method"].astype(str).str.startswith("bcdrcfar")].copy()
    feasible = bcd[bcd["factor2_violation_rate"] <= 0.35].copy()
    if feasible.empty:
        champion = {}
    else:
        champion_row = feasible.sort_values(
            ["factor2_violation_rate", "median_absolute_log10_pfa_error", "pd_at_0db"],
            ascending=[True, True, False],
        ).iloc[0]
        champion = champion_row.to_dict()

    payload = {
        "status": "BCDRCFAR_DEVELOPMENT_CHAMPION_TRACKING_COMPLETE",
        "rows": int(len(frame)),
        "feasible_bcd_rows_factor2_le_0p35": int(len(feasible)),
        "current_champion": champion,
        "summary": str(summary_path),
        "calibration_opened": False,
        "locked_opened": False,
    }
    manifest_path = OUT_DIR / "development_champion_tracking_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
