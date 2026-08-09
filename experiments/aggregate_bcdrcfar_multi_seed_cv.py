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
    parser.add_argument("--oracle-tag", default="3a1aa9f4")
    parser.add_argument("--seeds", default="2026080711,2026080712,2026080713")
    parser.add_argument("--decisions", type=int, default=1024)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--reference-cells", type=int, default=24)
    parser.add_argument("--worst-decile-weight", type=float, default=0.5)
    parser.add_argument("--feature-schema", choices=("v2",), default="v2")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(piece.strip()) for piece in args.seeds.split(",") if piece.strip()]
    loss_tag = "smooth_l1_"
    pattern = (
        f"taildep_v2_scaleeq_{args.oracle_tag}_seed*_fold*_L128_K{args.reference_cells}_"
        f"pfa{args.target_pfa:g}_{loss_tag}worst{args.worst_decile_weight:g}_*_fold*_"
        f"n{args.decisions}_pfa{args.target_pfa:g}_m1.000000_conditions.csv"
    )
    seed_fold_paths: dict[tuple[int, int], Path] = {}
    for path in sorted(EVALUATION.glob(pattern)):
        match = re.search(r"seed(\d+)_fold(\d+)_L", path.name)
        if not match:
            continue
        seed = int(match.group(1))
        fold = int(match.group(2))
        if seed in seeds:
            seed_fold_paths[(seed, fold)] = path
    expected = {(seed, fold) for seed in seeds for fold in range(6)}
    if set(seed_fold_paths) != expected:
        missing = sorted(expected - set(seed_fold_paths))
        raise RuntimeError(f"missing seed/fold artifacts: {missing}")

    frames = []
    for (seed, fold), path in sorted(seed_fold_paths.items()):
        frame = pd.read_csv(path)
        if frame["cell_id"].nunique() != 84 or set(frame["endpoint"]) != {"pfa", "pd"}:
            raise RuntimeError(f"incomplete fold artifact: {path}")
        frame["training_seed"] = seed
        frame["development_fold"] = fold
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    pfa = all_rows[all_rows["endpoint"] == "pfa"].copy()
    pd_rows = all_rows[all_rows["endpoint"] == "pd"].copy()

    fold_rows = []
    for (seed, fold, method), group in pfa.groupby(["training_seed", "development_fold", "method"], sort=True):
        detection = pd_rows[
            (pd_rows["training_seed"] == seed)
            & (pd_rows["development_fold"] == fold)
            & (pd_rows["method"] == method)
        ]
        fold_rows.append(
            {
                "training_seed": int(seed),
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
        seed_fold_cells=("cell_id", "size"),
        pooled_events=("events", "sum"),
        pooled_trials=("trials", "sum"),
        factor2_violation_rate=("factor2_violation", "mean"),
        median_absolute_log10_pfa_error=("absolute_log10_pfa_error", "median"),
    )
    overall["pooled_pfa"] = overall["pooled_events"] / overall["pooled_trials"]
    overall["pd_at_0db"] = pd_rows.groupby("method")["rate"].mean()
    overall = overall.reset_index()

    bcd_folds = fold_frame[fold_frame["method"] == "bcdrcfar"]
    classical = overall[overall["method"] != "bcdrcfar"].sort_values(
        ["factor2_violation_rate", "median_absolute_log10_pfa_error"]
    )
    strongest = str(classical.iloc[0]["method"])
    pfa_index = ["training_seed", "development_fold", "cell_id"]
    wide_violation = pfa.pivot_table(index=pfa_index, columns="method", values="factor2_violation", aggfunc="mean")
    wide_error = pfa.pivot_table(index=pfa_index, columns="method", values="absolute_log10_pfa_error", aggfunc="mean")
    violation_difference = wide_violation["bcdrcfar"].astype(float) - wide_violation[strongest].astype(float)
    error_difference = wide_error["bcdrcfar"].astype(float) - wide_error[strongest].astype(float)
    rng = np.random.default_rng(min(seeds) + 1991)
    n = len(violation_difference)
    indices = rng.integers(0, n, size=(int(args.bootstrap_replicates), n))
    violation_bootstrap = violation_difference.to_numpy()[indices].mean(axis=1)
    error_bootstrap = error_difference.to_numpy()[indices].mean(axis=1)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = (
        f"v2_scaleeq_{args.oracle_tag}_seeds{'-'.join(str(seed) for seed in seeds)}_"
        f"K{args.reference_cells}_n{args.decisions}_pfa{args.target_pfa:g}"
    )
    fold_path = OUTPUT / f"{tag}_fold_metrics.csv"
    overall_path = OUTPUT / f"{tag}_overall.csv"
    manifest_path = OUTPUT / f"{tag}_manifest.json"
    fold_frame.to_csv(fold_path, index=False)
    overall.to_csv(overall_path, index=False)
    payload = {
        "status": "BCDRCFAR_DEVELOPMENT_MULTI_SEED_CV_COMPLETE",
        "claim_status": "DEVELOPMENT_ONLY_ENGINEERING_OOF",
        "training_seeds": seeds,
        "folds_per_seed": 6,
        "cells_per_fold": 84,
        "decisions_per_endpoint_per_cell": int(args.decisions),
        "reference_cells": int(args.reference_cells),
        "target_pfa": float(args.target_pfa),
        "bcdrcfar_median_seed_fold_factor2_violation_rate": float(
            bcd_folds["factor2_violation_rate"].median()
        ),
        "bcdrcfar_seed_fold_factor2_violation_rate_range": [
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
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(overall.to_string(index=False))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
