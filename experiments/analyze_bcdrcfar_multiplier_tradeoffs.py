from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "results" / "bcdrcfar_development_evaluation"
OUT_DIR = ROOT / "results" / "bcdrcfar_multiplier_tradeoff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="median_ensemble3_*_fold*_n1024_pfa0.01_m*.csv")
    parser.add_argument("--baseline-method", default="bcdrcfar")
    parser.add_argument("--candidate-method", default="bcdrcfar")
    parser.add_argument("--baseline-multiplier", default="1.000000")
    parser.add_argument("--candidate-multiplier", default="0.990000")
    parser.add_argument("--max-factor2", type=float, default=0.35)
    return parser.parse_args()


def _load_by_multiplier(multiplier: str) -> pd.DataFrame:
    pattern = f"median_ensemble3_*_fold*_n1024_pfa0.01_m{multiplier}_conditions.csv"
    frames = []
    for path in sorted(EVALUATION.glob(pattern)):
        frame = pd.read_csv(path)
        frame["source"] = str(path.relative_to(ROOT))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summarize(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    pfa = frame[frame["endpoint"] == "pfa"].copy()
    pd_rows = frame[frame["endpoint"] == "pd"].copy()
    rows = []
    for keys, group in pfa.groupby(["scenario", "severity", "method"], sort=True):
        scenario, severity, method = keys
        det = pd_rows[
            (pd_rows["scenario"] == scenario)
            & (pd_rows["severity"] == severity)
            & (pd_rows["method"] == method)
        ]
        rows.append(
            {
                "label": label,
                "scenario": scenario,
                "severity": severity,
                "method": method,
                "cells": int(group["cell_id"].nunique()),
                "pooled_pfa": float(group["events"].sum() / group["trials"].sum()),
                "factor2_violation_rate": float(group["factor2_violation"].mean()),
                "median_absolute_log10_pfa_error": float(group["absolute_log10_pfa_error"].median()),
                "pd_at_0db": float(det["rate"].mean()) if not det.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _summarize(_load_by_multiplier(args.baseline_multiplier), f"m{args.baseline_multiplier}")
    candidate = _summarize(_load_by_multiplier(args.candidate_multiplier), f"m{args.candidate_multiplier}")
    if baseline.empty or candidate.empty:
        payload = {
            "status": "WAITING_FOR_MULTIPLIER_ARTIFACTS",
            "baseline_rows": int(len(baseline)),
            "candidate_rows": int(len(candidate)),
            "calibration_opened": False,
            "locked_opened": False,
        }
        (OUT_DIR / "tradeoff_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
        return

    key_cols = ["scenario", "severity", "method"]
    merged = candidate.merge(
        baseline,
        on=key_cols,
        suffixes=("_candidate", "_baseline"),
    )
    merged["factor2_delta"] = merged["factor2_violation_rate_candidate"] - merged["factor2_violation_rate_baseline"]
    merged["pd_delta"] = merged["pd_at_0db_candidate"] - merged["pd_at_0db_baseline"]
    merged["pfa_delta"] = merged["pooled_pfa_candidate"] - merged["pooled_pfa_baseline"]
    merged["risk_gated_candidate"] = (
        (merged["factor2_violation_rate_candidate"] <= args.max_factor2)
        & (merged["factor2_delta"] <= 0.0)
        & (merged["pd_delta"] > 0.0)
    )
    out_path = OUT_DIR / "stratified_multiplier_tradeoff.csv"
    merged.to_csv(out_path, index=False)
    payload = {
        "status": "BCDRCFAR_MULTIPLIER_TRADEOFF_ANALYSIS_COMPLETE",
        "candidate_multiplier": args.candidate_multiplier,
        "baseline_multiplier": args.baseline_multiplier,
        "strata": int(len(merged)),
        "risk_gated_candidate_strata": int(merged["risk_gated_candidate"].sum()),
        "summary": str(out_path),
        "calibration_opened": False,
        "locked_opened": False,
    }
    (OUT_DIR / "tradeoff_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(merged.sort_values(["risk_gated_candidate", "pd_delta"], ascending=[False, False]).to_string(index=False))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
