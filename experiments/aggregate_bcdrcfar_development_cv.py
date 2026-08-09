from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "results" / "bcdrcfar_development_evaluation"
OUTPUT = ROOT / "results" / "bcdrcfar_development_cv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-tag", default="864f7e3e")
    parser.add_argument("--seed", type=int, default=2026080711)
    parser.add_argument("--decisions", type=int, default=1024)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--reference-cells", type=int, default=8)
    parser.add_argument("--worst-decile-weight", type=float, default=0.5)
    parser.add_argument("--feature-schema", choices=("v1", "v2"), default="v1")
    parser.add_argument("--evaluation-prefix", default="", help="Optional evaluated artifact prefix, e.g. median_ensemble3.")
    parser.add_argument("--model-threshold-multiplier", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_prefix = "scaleeq" if args.feature_schema == "v1" else "taildep_v2_scaleeq"
    loss_tag = "" if args.feature_schema == "v1" else "smooth_l1_"
    if args.evaluation_prefix:
        pattern = (
            f"{args.evaluation_prefix}_*_fold*_n{args.decisions}_"
            f"pfa{args.target_pfa:g}_m{args.model_threshold_multiplier:.6f}_conditions.csv"
        )
    else:
        pattern = (
            f"{training_prefix}_{args.oracle_tag}_seed{args.seed}_fold*_L128_K{args.reference_cells}_pfa{args.target_pfa:g}_"
            f"{loss_tag}worst{args.worst_decile_weight:g}_*_fold*_n{args.decisions}_"
            f"pfa{args.target_pfa:g}_m{args.model_threshold_multiplier:.6f}_conditions.csv"
        )
    paths = sorted(EVALUATION.glob(pattern))
    fold_paths: dict[int, Path] = {}
    for path in paths:
        match = re.search(rf"seed{args.seed}_fold(\d+)_L", path.name)
        if args.evaluation_prefix and not match:
            match = re.search(r"_fold(\d+)_n", path.name)
        if match:
            fold_paths[int(match.group(1))] = path
    if set(fold_paths) != set(range(6)):
        raise RuntimeError(f"expected exactly folds 0..5, found {sorted(fold_paths)}")

    frames = []
    for fold, path in sorted(fold_paths.items()):
        frame = pd.read_csv(path)
        if frame["cell_id"].nunique() != 84 or set(frame["endpoint"]) != {"pfa", "pd"}:
            raise RuntimeError(f"incomplete fold artifact: {path}")
        frame["development_fold"] = fold
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    pfa = all_rows[all_rows["endpoint"] == "pfa"].copy()
    pd_rows = all_rows[all_rows["endpoint"] == "pd"].copy()

    fold_rows = []
    for (fold, method), group in pfa.groupby(["development_fold", "method"]):
        detection = pd_rows[
            (pd_rows["development_fold"] == fold) & (pd_rows["method"] == method)
        ]
        fold_rows.append(
            {
                "development_fold": int(fold),
                "method": method,
                "cells": int(group["cell_id"].nunique()),
                "pooled_pfa": float(group["events"].sum() / group["trials"].sum()),
                "factor2_violation_rate": float(group["factor2_violation"].mean()),
                "median_absolute_log10_pfa_error": float(group["absolute_log10_pfa_error"].median()),
                "pd_at_0db": float(detection["rate"].mean()),
            }
        )
    fold_frame = pd.DataFrame(fold_rows)
    overall = pfa.groupby("method").agg(
        cells=("cell_id", "nunique"),
        pooled_events=("events", "sum"),
        pooled_trials=("trials", "sum"),
        factor2_violation_rate=("factor2_violation", "mean"),
        median_absolute_log10_pfa_error=("absolute_log10_pfa_error", "median"),
    )
    overall["pooled_pfa"] = overall["pooled_events"] / overall["pooled_trials"]
    overall["pd_at_0db"] = pd_rows.groupby("method")["rate"].mean()
    overall = overall.reset_index()

    classical = overall[overall["method"] != "bcdrcfar"].sort_values(
        ["factor2_violation_rate", "median_absolute_log10_pfa_error"]
    )
    strongest = str(classical.iloc[0]["method"])
    wide_violation = pfa.pivot(index="cell_id", columns="method", values="factor2_violation").astype(float)
    wide_error = pfa.pivot(index="cell_id", columns="method", values="absolute_log10_pfa_error")
    violation_difference = wide_violation["bcdrcfar"] - wide_violation[strongest]
    error_difference = wide_error["bcdrcfar"] - wide_error[strongest]
    rng = np.random.default_rng(int(args.seed) + 991)
    n = len(violation_difference)
    indices = rng.integers(0, n, size=(int(args.bootstrap_replicates), n))
    violation_bootstrap = violation_difference.to_numpy()[indices].mean(axis=1)
    error_bootstrap = error_difference.to_numpy()[indices].mean(axis=1)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag_prefix = (
        f"{args.feature_schema}_scaleeq_{args.oracle_tag}_seed{args.seed}_K{args.reference_cells}"
        if not args.evaluation_prefix
        else f"{args.evaluation_prefix}_{args.feature_schema}_K{args.reference_cells}"
    )
    tag = f"{tag_prefix}_n{args.decisions}_pfa{args.target_pfa:g}_m{args.model_threshold_multiplier:.3f}"
    fold_path = OUTPUT / f"{tag}_fold_metrics.csv"
    overall_path = OUTPUT / f"{tag}_overall.csv"
    fold_frame.to_csv(fold_path, index=False)
    overall.to_csv(overall_path, index=False)
    bcd_folds = fold_frame[fold_frame["method"] == "bcdrcfar"]
    payload = {
        "status": "BCDRCFAR_DEVELOPMENT_SIX_FOLD_SCREENING_COMPLETE",
        "claim_status": (
            "DEVELOPMENT_ONLY_ENGINEERING_OOF"
            if int(args.decisions) >= 4096
            else "DEVELOPMENT_ONLY_LOW_MONTE_CARLO_SCREENING"
        ),
        "folds": 6,
        "cells": int(pfa[pfa["method"] == "bcdrcfar"]["cell_id"].nunique()),
        "decisions_per_endpoint_per_cell": int(args.decisions),
        "evaluation_prefix": args.evaluation_prefix or None,
        "model_threshold_multiplier": float(args.model_threshold_multiplier),
        "bcdrcfar_median_fold_factor2_violation_rate": float(
            bcd_folds["factor2_violation_rate"].median()
        ),
        "bcdrcfar_fold_factor2_violation_rate_range": [
            float(bcd_folds["factor2_violation_rate"].min()),
            float(bcd_folds["factor2_violation_rate"].max()),
        ],
        "strongest_classical_baseline": strongest,
        "paired_factor2_violation_rate_difference_vs_strongest": float(violation_difference.mean()),
        "paired_factor2_difference_95ci": [
            float(np.quantile(violation_bootstrap, 0.025)),
            float(np.quantile(violation_bootstrap, 0.975)),
        ],
        "paired_mean_absolute_log10_error_difference_vs_strongest": float(error_difference.mean()),
        "paired_error_difference_95ci": [
            float(np.quantile(error_bootstrap, 0.025)),
            float(np.quantile(error_bootstrap, 0.975)),
        ],
        "stage_a_relative_gate_passed": bool(
            violation_difference.mean() <= -0.10 and error_difference.mean() < 0.0
        ),
        "absolute_reliability_stop_triggered": bool(
            bcd_folds["factor2_violation_rate"].median() > 0.50
        ),
        "fold_metrics_sha256": sha256(fold_path),
        "overall_sha256": sha256(overall_path),
        "calibration_opened": False,
        "locked_opened": False,
    }
    if args.feature_schema == "v2" and args.reference_cells == 8:
        v1_pattern = (
            f"scaleeq_{args.oracle_tag}_seed{args.seed}_fold*_L128_K8_pfa{args.target_pfa:g}_"
            f"worst{args.worst_decile_weight:g}_*_fold*_n{args.decisions}_"
            f"pfa{args.target_pfa:g}_m1.000000_conditions.csv"
        )
        v1_frames = []
        for path in sorted(EVALUATION.glob(v1_pattern)):
            frame = pd.read_csv(path)
            v1_frames.append(frame[(frame["method"] == "bcdrcfar") & (frame["endpoint"] == "pfa")])
        v1 = pd.concat(v1_frames, ignore_index=True).drop_duplicates("cell_id").set_index("cell_id") if v1_frames else pd.DataFrame()
        v2 = pfa[pfa["method"] == "bcdrcfar"].set_index("cell_id")
        if len(v1) == 504 and len(v2) == 504 and set(v1.index) == set(v2.index):
            v1 = v1.loc[v2.index]
            delta_violation = v2["factor2_violation"].astype(float) - v1["factor2_violation"].astype(float)
            delta_error = v2["absolute_log10_pfa_error"] - v1["absolute_log10_pfa_error"]
            delta_violation_bootstrap = delta_violation.to_numpy()[indices].mean(axis=1)
            delta_error_bootstrap = delta_error.to_numpy()[indices].mean(axis=1)
            payload.update({
                "paired_factor2_violation_rate_difference_v2_vs_v1": float(delta_violation.mean()),
                "paired_factor2_difference_v2_vs_v1_95ci": [
                    float(np.quantile(delta_violation_bootstrap, 0.025)),
                    float(np.quantile(delta_violation_bootstrap, 0.975)),
                ],
                "paired_mean_absolute_log10_error_difference_v2_vs_v1": float(delta_error.mean()),
                "paired_error_difference_v2_vs_v1_95ci": [
                    float(np.quantile(delta_error_bootstrap, 0.025)),
                    float(np.quantile(delta_error_bootstrap, 0.975)),
                ],
            })
        else:
            payload["v2_vs_v1_comparison_status"] = "UNAVAILABLE_AT_THIS_DECISION_BUDGET"
    manifest_path = OUTPUT / f"{tag}_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(overall.to_string(index=False))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
