"""Fit a deployable background-feature calibration head from IPIX blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "results" / "bcdrcfar_ipix" / "development_quick64" / "block_ratios.csv"
DEFAULT_LABELS = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "lowrank_calibration" / "best_multipliers.json"
DEFAULT_OUTPUT = ROOT / "results" / "bcdrcfar_ipix" / "development_quick64" / "feature_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    return parser.parse_args()


def build_design(frame: pd.DataFrame) -> pd.DataFrame:
    pol = pd.get_dummies(frame["polarization"].astype(str), prefix="pol", drop_first=False)
    base = frame[[
        "log_scale",
        "anchor_entropy",
        "anchor_max",
        "anchor_gap",
        "uncertainty",
        "series_threshold_shift",
    ]].copy()
    return pd.concat([base, pol], axis=1)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    eye = np.eye(x_aug.shape[1], dtype=float)
    eye[0, 0] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + alpha * eye, x_aug.T @ y)
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


def load_group_labels(path: Path) -> dict[tuple[str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    multipliers = payload["multipliers"]
    result: dict[tuple[str, str], float] = {}
    for key, value in multipliers.items():
        if key == "__global__":
            continue
        if "|" in key:
            file_id, pol = key.split("|", 1)
        else:
            continue
        result[(str(file_id), str(pol))] = float(value)
    return result


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.features)
    labels = load_group_labels(args.labels)
    usable = frame[frame["role"].isin(["clutter", "primary"])].copy()
    usable["label_multiplier"] = [
        labels.get((str(file_id), str(pol)))
        for file_id, pol in zip(usable["file_id"].astype(str), usable["polarization"].astype(str), strict=True)
    ]
    usable = usable.dropna(subset=["label_multiplier"]).reset_index(drop=True)
    design = build_design(usable)
    x = design.to_numpy(dtype=float)
    y = np.log(usable["label_multiplier"].to_numpy(dtype=float))
    rows: list[dict[str, float]] = []
    best: dict[str, object] | None = None
    for alpha in map(float, args.ridge):
        fold_summaries: list[dict[str, float]] = []
        for held_out in sorted(usable["file_id"].astype(str).unique().tolist()):
            train_mask = usable["file_id"].astype(str) != held_out
            test_mask = ~train_mask
            beta = ridge_fit(x[train_mask.to_numpy()], y[train_mask.to_numpy()], alpha)
            test_pred = apply_beta(x[test_mask.to_numpy()], beta)
            test_summary = summarize(usable[test_mask].copy(), test_pred, float(args.target_pfa))
            fold_summaries.append(test_summary)
        beta = ridge_fit(x, y, alpha)
        overall_pred = apply_beta(x, beta)
        overall_summary = summarize(usable, overall_pred, float(args.target_pfa))
        row = {
            "ridge": alpha,
            "cv_macro_pfa": float(np.mean([item["macro_pfa"] for item in fold_summaries])),
            "cv_macro_factor2": float(np.mean([item["macro_factor2"] for item in fold_summaries])),
            "cv_macro_pd": float(np.mean([item["macro_pd"] for item in fold_summaries])),
            "overall_macro_pfa": overall_summary["macro_pfa"],
            "overall_macro_factor2": overall_summary["macro_factor2"],
            "overall_macro_pd": overall_summary["macro_pd"],
        }
        rows.append(row)
        score = row["cv_macro_pfa"] + 2.0 * row["cv_macro_factor2"] - 0.5 * row["cv_macro_pd"]
        if best is None or score < float(best["score"]):
            best = {
                "score": float(score),
                "alpha": alpha,
                "beta": beta,
                "cv": {
                    "macro_pfa": row["cv_macro_pfa"],
                    "macro_factor2": row["cv_macro_factor2"],
                    "macro_pd": row["cv_macro_pd"],
                },
                "overall": overall_summary,
                "features": list(design.columns),
            }

    assert best is not None
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "ridge_grid.csv", index=False)
    (output_dir / "best_model.json").write_text(
        json.dumps(
            {
                "ridge": best["alpha"],
                "features": best["features"],
                "cv": best["cv"],
                "overall": best["overall"],
                "beta": best["beta"].tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps({"best_ridge": best["alpha"], "cv": best["cv"], "overall": best["overall"]}, indent=2))


if __name__ == "__main__":
    main()
