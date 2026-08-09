"""Ablate feature groups for the deployable IPIX background-feature head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEV_FEATURES = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "block_ratios.csv"
RET_FEATURES = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "condition_rows.csv"
BEST_MODEL = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "feature_calibration" / "best_model.json"
LABELS = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "lowrank_calibration" / "best_multipliers.json"
OUT_DIR = ROOT / "results" / "bcdrcfar_ipix" / "feature_ablation"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-features", type=Path, default=DEV_FEATURES)
    parser.add_argument("--ret-features", type=Path, default=RET_FEATURES)
    parser.add_argument("--best-model", type=Path, default=BEST_MODEL)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    return parser.parse_args(argv)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    eye = np.eye(x_aug.shape[1], dtype=float)
    eye[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + alpha * eye, x_aug.T @ y)


def apply_beta(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    return x_aug @ beta


def summarize(frame: pd.DataFrame, log_multiplier: np.ndarray, target_pfa: float) -> dict[str, float]:
    data = frame.copy()
    data["pred_multiplier"] = np.exp(log_multiplier)
    data["decision"] = data["ratio_bcdrcfar"] >= data["pred_multiplier"]
    clutter = data[data["role"] == "clutter"]
    target = data[data["role"] == "primary"]
    acquisition = clutter.groupby("file_id", sort=True).agg(
        pfa=("decision", "mean"),
        events=("decision", "sum"),
        trials=("decision", "size"),
    )
    primary = target.groupby("file_id", sort=True).agg(pd=("decision", "mean"))
    return {
        "macro_pfa": float(acquisition["pfa"].mean()),
        "macro_factor2": float(((acquisition["pfa"] < 0.5 * target_pfa) | (acquisition["pfa"] > 2.0 * target_pfa)).mean()),
        "macro_pd": float(primary["pd"].mean()),
    }


def build_design(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    design = pd.DataFrame(index=frame.index)
    for column in features:
        if column == "pol_hh":
            design[column] = (frame["polarization"].astype(str) == "hh").astype(float)
        elif column == "pol_hv":
            design[column] = (frame["polarization"].astype(str) == "hv").astype(float)
        elif column == "pol_vh":
            design[column] = (frame["polarization"].astype(str) == "vh").astype(float)
        elif column == "pol_vv":
            design[column] = (frame["polarization"].astype(str) == "vv").astype(float)
        else:
            design[column] = frame[column].to_numpy(dtype=float)
    return design


def load_group_labels(path: Path) -> dict[tuple[str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str], float] = {}
    for key, value in payload["multipliers"].items():
        if key == "__global__" or "|" not in key:
            continue
        file_id, pol = key.split("|", 1)
        result[(str(file_id), str(pol))] = float(value)
    return result


def evaluate_family(train: pd.DataFrame, test: pd.DataFrame, features: list[str], ridge: float, target_pfa: float) -> tuple[dict[str, float], dict[str, float]]:
    design_train = build_design(train, features)
    design_test = build_design(test, features)
    x_train = design_train.to_numpy(dtype=float)
    y_train = np.log(train["label_multiplier"].to_numpy(dtype=float))
    beta = ridge_fit(x_train, y_train, ridge)
    train_summary = summarize(train, apply_beta(x_train, beta), target_pfa)
    test_summary = summarize(test, apply_beta(design_test.to_numpy(dtype=float), beta), target_pfa)
    return train_summary, test_summary


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.best_model.read_text(encoding="utf-8"))
    ridge = float(payload["ridge"])
    all_features = list(payload["features"])
    if all_features and all_features[0] != "log_scale":
        raise RuntimeError("unexpected feature order")

    dev = pd.read_csv(args.dev_features)
    ret = pd.read_csv(args.ret_features)
    usable_dev = dev[dev["role"].isin(["clutter", "primary"])].copy()
    usable_ret = ret[ret["role"].isin(["clutter", "primary"])].copy()
    labels = load_group_labels(args.labels)
    usable_dev["label_multiplier"] = [
        labels.get((str(file_id), str(pol)))
        for file_id, pol in zip(usable_dev["file_id"].astype(str), usable_dev["polarization"].astype(str), strict=True)
    ]
    usable_dev = usable_dev.dropna(subset=["label_multiplier"]).reset_index(drop=True)

    group_specs = {
        "all": all_features,
        "no_anchor": [f for f in all_features if f not in {"anchor_entropy", "anchor_max", "anchor_gap"}],
        "no_uncertainty": [f for f in all_features if f != "uncertainty"],
        "no_series_shift": [f for f in all_features if f != "series_threshold_shift"],
        "no_polarization": [f for f in all_features if not f.startswith("pol_")],
        "log_scale_only": ["log_scale"],
        "background_core": [f for f in all_features if f in {"log_scale", "anchor_entropy", "anchor_max", "anchor_gap", "uncertainty", "series_threshold_shift"}],
    }

    rows: list[dict[str, Any]] = []
    for name, features in group_specs.items():
        if not features:
            continue
        cv_summaries: list[dict[str, float]] = []
        for held_out in sorted(usable_dev["file_id"].astype(str).unique().tolist()):
            train_mask = usable_dev["file_id"].astype(str) != held_out
            test_mask = ~train_mask
            _, fold_summary = evaluate_family(
                usable_dev[train_mask].copy(),
                usable_dev[test_mask].copy(),
                features,
                ridge,
                float(args.target_pfa),
            )
            cv_summaries.append(fold_summary)
        _, ret_summary = evaluate_family(usable_dev, usable_ret, features, ridge, float(args.target_pfa))
        rows.append(
            {
                "family": name,
                "feature_count": len(features),
                "dev_cv_macro_pfa": float(np.mean([item["macro_pfa"] for item in cv_summaries])),
                "dev_cv_macro_factor2": float(np.mean([item["macro_factor2"] for item in cv_summaries])),
                "dev_cv_macro_pd": float(np.mean([item["macro_pd"] for item in cv_summaries])),
                "ret_macro_pfa": ret_summary["macro_pfa"],
                "ret_macro_factor2": ret_summary["macro_factor2"],
                "ret_macro_pd": ret_summary["macro_pd"],
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["ret_macro_pfa", "ret_macro_factor2", "ret_macro_pd"], ascending=[True, True, False]).to_csv(output_dir / "ablation.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
