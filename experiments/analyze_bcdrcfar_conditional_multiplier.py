#!/usr/bin/env python3
"""Analyze whether a lower threshold multiplier is safe only in selected strata.

This is a development-only diagnostic.  It does not open locked data and does not
select a final claim by itself.  The intended use is to compare m=1.0 against a
candidate multiplier such as m=0.99 at the condition level, then identify
scenario/severity strata where the candidate improves Pd without worsening PFA
control beyond configured tolerances.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", default="results/bcdrcfar_development_evaluation")
    parser.add_argument("--output-dir", default="results/bcdrcfar_conditional_multiplier")
    parser.add_argument("--prefix", default="median_ensemble3")
    parser.add_argument("--target-pfa", default="0.01")
    parser.add_argument("--decisions", type=int, default=1024)
    parser.add_argument("--baseline-multiplier", type=float, default=1.0)
    parser.add_argument("--candidate-multiplier", type=float, default=0.99)
    parser.add_argument("--method", default="bcdrcfar")
    parser.add_argument("--max-factor2-delta", type=float, default=0.0)
    parser.add_argument("--max-median-error-delta", type=float, default=0.02)
    parser.add_argument("--min-pd-delta", type=float, default=0.0)
    return parser.parse_args()


def load_conditions(evaluation_dir: Path, prefix: str, decisions: int, target_pfa: str, multiplier: float) -> pd.DataFrame:
    pattern = f"{prefix}_*_fold*_n{decisions}_pfa{target_pfa}_m{multiplier:.6f}_conditions.csv"
    frames: list[pd.DataFrame] = []
    for path in sorted(evaluation_dir.glob(pattern)):
        fold = int(path.name.split("_fold", 1)[1].split("_", 1)[0])
        frame = pd.read_csv(path)
        frame["development_fold"] = fold
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No condition files matched {pattern!r} in {evaluation_dir}")
    return pd.concat(frames, ignore_index=True)


def summarize_by_stratum(df: pd.DataFrame, method: str) -> pd.DataFrame:
    d = df[df["method"].eq(method)].copy()
    group_cols = ["scenario", "severity", "endpoint"]
    agg = (
        d.groupby(group_cols, dropna=False)
        .agg(
            cells=("cell_id", "nunique"),
            rows=("cell_id", "size"),
            events=("events", "sum"),
            trials=("trials", "sum"),
            factor2_violation_rate=("factor2_violation", "mean"),
            median_absolute_log10_pfa_error=("absolute_log10_pfa_error", "median"),
            mean_rate=("rate", "mean"),
        )
        .reset_index()
    )
    agg["pooled_rate"] = agg["events"] / agg["trials"]
    return agg


def summarize_overall(df: pd.DataFrame, method: str, label: str) -> dict[str, float | str | int]:
    d = df[df["method"].eq(method)].copy()
    pfa = d[d["endpoint"].eq("pfa")]
    pd_endpoint = d[d["endpoint"].eq("pd")]
    return {
        "label": label,
        "cells": int(pfa["cell_id"].nunique()),
        "rows": int(len(d)),
        "pfa_events": int(pfa["events"].sum()),
        "pfa_trials": int(pfa["trials"].sum()),
        "pooled_pfa": float(pfa["events"].sum() / pfa["trials"].sum()),
        "factor2_violation_rate": float(pfa["factor2_violation"].mean()),
        "median_absolute_log10_pfa_error": float(pfa["absolute_log10_pfa_error"].median()),
        "pd": float(pd_endpoint["events"].sum() / pd_endpoint["trials"].sum()),
    }


def main() -> int:
    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_raw = load_conditions(evaluation_dir, args.prefix, args.decisions, args.target_pfa, args.baseline_multiplier)
    candidate_raw = load_conditions(evaluation_dir, args.prefix, args.decisions, args.target_pfa, args.candidate_multiplier)
    common_folds = sorted(set(baseline_raw["development_fold"]).intersection(candidate_raw["development_fold"]))
    if not common_folds:
        raise RuntimeError("No common development folds between baseline and candidate multiplier files.")
    baseline = summarize_by_stratum(baseline_raw[baseline_raw["development_fold"].isin(common_folds)], args.method)
    candidate = summarize_by_stratum(candidate_raw[candidate_raw["development_fold"].isin(common_folds)], args.method)

    keys = ["scenario", "severity", "endpoint"]
    merged = candidate.merge(baseline, on=keys, suffixes=("_candidate", "_baseline"))
    for col in ["factor2_violation_rate", "median_absolute_log10_pfa_error", "pooled_rate", "mean_rate"]:
        merged[f"delta_{col}"] = merged[f"{col}_candidate"] - merged[f"{col}_baseline"]

    pfa = merged[merged["endpoint"].eq("pfa")].copy()
    pd_endpoint = merged[merged["endpoint"].eq("pd")].copy()
    pd_delta = pd_endpoint[["scenario", "severity", "delta_pooled_rate", "pooled_rate_candidate", "pooled_rate_baseline"]].rename(
        columns={
            "delta_pooled_rate": "pd_delta",
            "pooled_rate_candidate": "pd_candidate",
            "pooled_rate_baseline": "pd_baseline",
        }
    )
    risk = pfa.merge(pd_delta, on=["scenario", "severity"], how="left")
    risk["safe_candidate"] = (
        risk["delta_factor2_violation_rate"].le(args.max_factor2_delta)
        & risk["delta_median_absolute_log10_pfa_error"].le(args.max_median_error_delta)
        & risk["pd_delta"].fillna(-1).ge(args.min_pd_delta)
    )
    risk = risk.sort_values(
        ["safe_candidate", "pd_delta", "delta_factor2_violation_rate", "delta_median_absolute_log10_pfa_error"],
        ascending=[False, False, True, True],
    )

    merged_path = output_dir / f"{args.prefix}_m{args.candidate_multiplier:.3f}_vs_m{args.baseline_multiplier:.3f}_strata_all.csv"
    risk_path = output_dir / f"{args.prefix}_m{args.candidate_multiplier:.3f}_vs_m{args.baseline_multiplier:.3f}_risk_gate.csv"
    overall_path = output_dir / f"{args.prefix}_m{args.candidate_multiplier:.3f}_vs_m{args.baseline_multiplier:.3f}_gated_overall.csv"
    merged.to_csv(merged_path, index=False)
    risk.to_csv(risk_path, index=False)

    safe_keys = set(
        map(tuple, risk.loc[risk["safe_candidate"], ["scenario", "severity"]].itertuples(index=False, name=None))
    )
    key_cols = ["development_fold", "cell_id", "scenario", "severity", "method", "endpoint", "scr_db"]
    baseline_common = baseline_raw[baseline_raw["development_fold"].isin(common_folds)].copy()
    candidate_common = candidate_raw[candidate_raw["development_fold"].isin(common_folds)].copy()
    candidate_common["_use_candidate"] = list(zip(candidate_common["scenario"], candidate_common["severity"]))
    candidate_common = candidate_common[candidate_common["_use_candidate"].isin(safe_keys)].drop(columns=["_use_candidate"])
    baseline_common["_use_baseline"] = list(zip(baseline_common["scenario"], baseline_common["severity"]))
    baseline_common = baseline_common[~baseline_common["_use_baseline"].isin(safe_keys)].drop(columns=["_use_baseline"])
    gated = pd.concat([candidate_common, baseline_common], ignore_index=True)

    overall = pd.DataFrame(
        [
            summarize_overall(baseline_raw[baseline_raw["development_fold"].isin(common_folds)], args.method, "baseline"),
            summarize_overall(candidate_raw[candidate_raw["development_fold"].isin(common_folds)], args.method, "candidate"),
            summarize_overall(gated, args.method, "gated_candidate_where_safe"),
        ]
    )
    overall["common_development_folds"] = ",".join(map(str, common_folds))
    overall["safe_strata"] = len(safe_keys)
    overall.to_csv(overall_path, index=False)

    print(f"wrote {merged_path}")
    print(f"wrote {risk_path}")
    print(f"wrote {overall_path}")
    print(f"common_development_folds={common_folds}")
    print(overall.to_string(index=False))
    print(risk[[
        "scenario",
        "severity",
        "safe_candidate",
        "factor2_violation_rate_baseline",
        "factor2_violation_rate_candidate",
        "delta_factor2_violation_rate",
        "median_absolute_log10_pfa_error_baseline",
        "median_absolute_log10_pfa_error_candidate",
        "delta_median_absolute_log10_pfa_error",
        "pd_baseline",
        "pd_candidate",
        "pd_delta",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
