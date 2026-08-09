"""Fit acquisition/polarization grouped tail multipliers on IPIX development blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "block_ratios.csv"
DEFAULT_OUTPUT = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "grouped_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    return parser.parse_args()


def q(values: pd.Series, target_pfa: float) -> float:
    return float(values.quantile(1.0 - float(target_pfa), interpolation="higher"))


def fit_multipliers(frame: pd.DataFrame, key_cols: list[str], target_pfa: float) -> dict[str, float]:
    clutter = frame[frame["role"] == "clutter"]
    if not key_cols:
        return {"__global__": q(clutter["ratio_bcdrcfar"], target_pfa)}
    grouped = clutter.groupby(key_cols, sort=True)["ratio_bcdrcfar"].apply(lambda s: q(s, target_pfa))
    if len(key_cols) == 1:
        return {str(key): float(value) for key, value in grouped.items()}
    return {json.dumps(list(key), ensure_ascii=False): float(value) for key, value in grouped.items()}


def apply_multiplier(row: pd.Series, schemes: dict[str, dict[str, float]], target_pfa: float) -> pd.Series:
    ratio = float(row["ratio_bcdrcfar"])
    file_id = str(row["file_id"])
    polarization = str(row["polarization"])
    outputs: dict[str, float] = {}
    for scheme, table in schemes.items():
        if scheme == "global":
            multiplier = table["__global__"]
        elif scheme == "file_id":
            multiplier = table.get(file_id, table["__global__"])
        elif scheme == "polarization":
            multiplier = table.get(polarization, table["__global__"])
        elif scheme == "file_id_polarization":
            multiplier = table.get(json.dumps([file_id, polarization], ensure_ascii=False), table["__global__"])
        else:
            raise KeyError(scheme)
        outputs[f"decision__{scheme}"] = float(ratio >= multiplier)
        outputs[f"rate__{scheme}"] = float(ratio >= multiplier)
        outputs[f"multiplier__{scheme}"] = float(multiplier)
    return pd.Series(outputs)


def summarize(frame: pd.DataFrame, scheme: str) -> dict[str, float]:
    clutter = frame[frame["role"] == "clutter"]
    target = frame[frame["role"].isin(["primary", "secondary"])]
    series = frame.copy()
    if scheme == "global":
        group_cols = []
    elif scheme == "file_id":
        group_cols = ["file_id"]
    elif scheme == "polarization":
        group_cols = ["polarization"]
    elif scheme == "file_id_polarization":
        group_cols = ["file_id", "polarization"]
    else:
        raise KeyError(scheme)

    if group_cols:
        clutter_multipliers = clutter.groupby(group_cols)["ratio_bcdrcfar"].transform(lambda s: q(s, 0.01))
        target_multipliers = target.groupby(group_cols)["ratio_bcdrcfar"].transform(lambda s: q(s, 0.01))
    else:
        multiplier = q(clutter["ratio_bcdrcfar"], 0.01)
        clutter_multipliers = pd.Series(multiplier, index=clutter.index)
        target_multipliers = pd.Series(multiplier, index=target.index)

    clutter_decision = clutter["ratio_bcdrcfar"].to_numpy(dtype=float) >= clutter_multipliers.to_numpy(dtype=float)
    target_decision = target["ratio_bcdrcfar"].to_numpy(dtype=float) >= target_multipliers.to_numpy(dtype=float)
    file_level = clutter.assign(
        decision=clutter_decision,
        threshold=clutter_multipliers.to_numpy(dtype=float),
    ).groupby("file_id", sort=True).agg(
        pfa=("decision", "mean"),
        threshold=("threshold", "mean"),
    )
    target_level = target.assign(decision=target_decision).groupby("file_id", sort=True).agg(pd=("decision", "mean"))
    return {
        "scheme": scheme,
        "macro_pfa": float(file_level["pfa"].mean()),
        "macro_factor2": float(((file_level["pfa"] < 0.005) | (file_level["pfa"] > 0.02)).mean()),
        "macro_pd": float(target_level["pd"].mean()),
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clutter = frame[frame["role"] == "clutter"].copy()
    schemes = {
        "global": fit_multipliers(frame, [], args.target_pfa),
        "file_id": fit_multipliers(frame, ["file_id"], args.target_pfa),
        "polarization": fit_multipliers(frame, ["polarization"], args.target_pfa),
        "file_id_polarization": fit_multipliers(frame, ["file_id", "polarization"], args.target_pfa),
    }
    summary = pd.DataFrame([summarize(frame, scheme) for scheme in schemes])
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "multipliers.json").write_text(
        json.dumps(schemes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps({"input": str(args.input), "target_pfa": float(args.target_pfa)}, indent=2))


if __name__ == "__main__":
    main()
