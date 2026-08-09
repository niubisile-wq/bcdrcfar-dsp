"""Compare feature-head behavior under different teacher label families."""

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
DEFAULT_LABELS = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "lowrank_calibration" / "best_multipliers.json"
OUT_DIR = ROOT / "results" / "bcdrcfar_ipix" / "teacher_dependence"
FEATURE_COLUMNS = ["log_scale", "anchor_entropy", "anchor_max", "anchor_gap", "uncertainty", "series_threshold_shift", "pol_hh", "pol_hv", "pol_vh", "pol_vv"]
BASE_COLUMNS = ["log_scale", "anchor_entropy", "anchor_max", "anchor_gap", "uncertainty", "series_threshold_shift"]
DEV_USECOLS = ["file_id", "polarization", "role", "ratio_bcdrcfar", *BASE_COLUMNS]
RET_USECOLS = ["file_id", "polarization", "role", "ratio_bcdrcfar", *BASE_COLUMNS]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-features", type=Path, default=DEV_FEATURES)
    parser.add_argument("--ret-features", type=Path, default=RET_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=1.0)
    return parser.parse_args(argv)


def load_group_labels(path: Path) -> dict[tuple[str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    multipliers = payload.get("multipliers", payload)
    result: dict[tuple[str, str], float] = {}

    def ingest(prefix: str | None, obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "__global__":
                    continue
                if prefix == "file_id":
                    result[(str(key), "__global__")] = float(value)
                elif prefix == "polarization":
                    result[("__global__", str(key))] = float(value)
                elif prefix == "file_id_polarization":
                    parsed = json.loads(str(key))
                    result[(str(parsed[0]), str(parsed[1]))] = float(value)
                else:
                    ingest(str(key), value)
        else:
            return

    for key, value in multipliers.items():
        if isinstance(value, dict) and key in {"file_id", "polarization", "file_id_polarization"}:
            ingest(key, value)
            continue
        if key == "__global__":
            continue
        if "|" in key:
            file_id, pol = key.split("|", 1)
            result[(str(file_id), str(pol))] = float(value)
        elif key.startswith("["):
            parsed = json.loads(key)
            result[(str(parsed[0]), str(parsed[1]))] = float(value)
        elif isinstance(value, (int, float)):
            result[(str(key), "__global__")] = float(value)
    return result


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


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    if alpha <= 0:
        beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        return beta
    penalty = np.eye(x_aug.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    x_stack = np.vstack([x_aug, np.sqrt(alpha) * penalty])
    y_stack = np.concatenate([y, np.zeros(penalty.shape[0], dtype=float)])
    beta, *_ = np.linalg.lstsq(x_stack, y_stack, rcond=None)
    return beta


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


def fit_and_eval(train: pd.DataFrame, test: pd.DataFrame, features: list[str], ridge: float, target_pfa: float) -> dict[str, float]:
    x_train = build_design(train, features).to_numpy(dtype=float)
    y_train = np.log(train["label_multiplier"].to_numpy(dtype=float))
    beta = ridge_fit(x_train, y_train, ridge)
    test_summary = summarize(test, apply_beta(build_design(test, features).to_numpy(dtype=float), beta), target_pfa)
    return test_summary


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    labels = load_group_labels(args.labels)

    dev = pd.read_csv(args.dev_features, usecols=DEV_USECOLS)
    ret = pd.read_csv(args.ret_features, usecols=RET_USECOLS)
    dev = dev[dev["role"].isin(["clutter", "primary"])].copy()
    ret = ret[ret["role"].isin(["clutter", "primary"])].copy()
    dev["label_multiplier"] = [
        labels.get((str(file_id), str(pol)))
        or labels.get((str(file_id), "__global__"))
        or labels.get(("__global__", str(pol)))
        or labels.get(("__global__", "__global__"))
        for file_id, pol in zip(dev["file_id"].astype(str), dev["polarization"].astype(str), strict=True)
    ]
    dev = dev.dropna(subset=["label_multiplier"]).reset_index(drop=True)

    features = FEATURE_COLUMNS

    cv_summaries: list[dict[str, float]] = []
    for held_out in sorted(dev["file_id"].astype(str).unique().tolist()):
        train_mask = dev["file_id"].astype(str) != held_out
        test_mask = ~train_mask
        fold_summary = fit_and_eval(dev[train_mask].copy(), dev[test_mask].copy(), features, float(args.ridge), float(args.target_pfa))
        cv_summaries.append(fold_summary)

    dev_summary = fit_and_eval(dev, dev, features, float(args.ridge), float(args.target_pfa))
    ret_summary = fit_and_eval(dev, ret, features, float(args.ridge), float(args.target_pfa))

    rows = {
        "labels": str(args.labels),
        "ridge": float(args.ridge),
        "dev_cv_macro_pfa": float(np.mean([item["macro_pfa"] for item in cv_summaries])),
        "dev_cv_macro_factor2": float(np.mean([item["macro_factor2"] for item in cv_summaries])),
        "dev_cv_macro_pd": float(np.mean([item["macro_pd"] for item in cv_summaries])),
        "dev_overall_macro_pfa": dev_summary["macro_pfa"],
        "dev_overall_macro_factor2": dev_summary["macro_factor2"],
        "dev_overall_macro_pd": dev_summary["macro_pd"],
        "ret_macro_pfa": ret_summary["macro_pfa"],
        "ret_macro_factor2": ret_summary["macro_factor2"],
        "ret_macro_pd": ret_summary["macro_pd"],
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
