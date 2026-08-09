"""Evaluate a deployable background-feature head on acquisition-disjoint IPIX data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_bcdrcfar_ipix import load_models, reference_bins_for_cut, sha256, write_json  # noqa: E402
from src.bcdrcfar.baselines import BASELINE_NAMES, classical_cfar_outputs  # noqa: E402
from src.real_data import (  # noqa: E402
    apply_ipix_reference_transform,
    fit_ipix_reference_transform,
    load_ipix_series,
    nonoverlapping_windows,
)


CONFIG = ROOT / "configs" / "bcdrcfar_ipix_protocol.json"
DATA = ROOT / "data" / "raw" / "ipix"
OUTPUT = ROOT / "results" / "bcdrcfar_ipix"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("development", "retrospective_external"), required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--scalar-calibration", type=Path, required=True)
    parser.add_argument("--feature-calibration", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def feature_multiplier(feature_payload: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    beta = np.asarray(feature_payload["beta"], dtype=float)
    columns = list(feature_payload["features"])
    design = pd.DataFrame(index=rows.index)
    for column in columns:
        if column == "pol_hh":
            design[column] = (rows["polarization"].astype(str) == "hh").astype(float)
        elif column == "pol_hv":
            design[column] = (rows["polarization"].astype(str) == "hv").astype(float)
        elif column == "pol_vh":
            design[column] = (rows["polarization"].astype(str) == "vh").astype(float)
        elif column == "pol_vv":
            design[column] = (rows["polarization"].astype(str) == "vv").astype(float)
        elif column in rows.columns:
            design[column] = rows[column].to_numpy(dtype=float)
        else:
            raise KeyError(f"unknown feature column: {column}")
    x = np.column_stack([np.ones(len(design), dtype=float), design.to_numpy(dtype=float)])
    if x.shape[1] != len(beta):
        raise ValueError("feature beta does not match design matrix")
    return np.exp(x @ beta)


def summarize(
    frame: pd.DataFrame, target_pfa: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["file_id", "polarization", "range_bin", "role"]
    series_rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(keys, sort=True):
        for method in ["bcdrcfar_scalar", "bcdrcfar_feature", *BASELINE_NAMES]:
            decisions = group[f"decision_{method}"]
            events = int(decisions.sum())
            trials = int(len(group))
            rate = float(events / trials) if trials else float("nan")
            adjusted = (events + 0.5) / (trials + 1.0) if trials else float("nan")
            series_rows.append(
                {
                    **dict(zip(keys, key)),
                    "method": method,
                    "events": events,
                    "trials": trials,
                    "rate": rate,
                    "jeffreys_rate": adjusted,
                    "absolute_log10_pfa_error": (
                        abs(float(np.log10(adjusted / target_pfa))) if key[-1] == "clutter" else np.nan
                    ),
                    "factor2_violation": (
                        bool(adjusted < target_pfa / 2.0 or adjusted > target_pfa * 2.0)
                        if key[-1] == "clutter"
                        else False
                    ),
                }
            )

    series = pd.DataFrame(series_rows)
    clutter = series[series["role"] == "clutter"]
    acquisition_rows: list[dict[str, Any]] = []
    for (file_id, method), group in clutter.groupby(["file_id", "method"], sort=True):
        events = int(group["events"].sum())
        trials = int(group["trials"].sum())
        adjusted = (events + 0.5) / (trials + 1.0)
        acquisition_rows.append(
            {
                "file_id": file_id,
                "method": method,
                "events": events,
                "trials": trials,
                "pfa": events / trials,
                "absolute_log10_pfa_error": abs(float(np.log10(adjusted / target_pfa))),
                "series_factor2_violation_rate": float(group["factor2_violation"].mean()),
            }
        )

    acquisition = pd.DataFrame(acquisition_rows)
    target = (
        series[series["role"].isin(["primary", "secondary"])]
        .groupby(["file_id", "role", "method"], sort=True)
        .agg(events=("events", "sum"), trials=("trials", "sum"))
        .reset_index()
    )
    target["pd"] = target["events"] / target["trials"]
    return series, acquisition, target


def bootstrap_summary(
    acquisition: pd.DataFrame,
    target: pd.DataFrame,
    strongest: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    error = acquisition.pivot(index="file_id", columns="method", values="absolute_log10_pfa_error")
    pd_primary = target[target["role"] == "primary"].pivot(index="file_id", columns="method", values="pd")
    common = sorted(set(error.index) & set(pd_primary.index))
    error_diff = (error.loc[common, "bcdrcfar_feature"] - error.loc[common, strongest]).to_numpy()
    pd_diff = (pd_primary.loc[common, "bcdrcfar_feature"] - pd_primary.loc[common, strongest]).to_numpy()
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(common), size=(int(replicates), len(common)))
    error_boot = error_diff[indices].mean(axis=1)
    pd_boot = pd_diff[indices].mean(axis=1)
    return {
        "strongest_development_selected_baseline": strongest,
        "paired_mean_acquisition_error_difference": float(error_diff.mean()),
        "paired_error_difference_95ci": np.quantile(error_boot, [0.025, 0.975]).tolist(),
        "paired_mean_primary_pd_difference": float(pd_diff.mean()),
        "paired_primary_pd_difference_95ci": np.quantile(pd_boot, [0.025, 0.975]).tolist(),
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not str(protocol["status"]).startswith("frozen_"):
        raise RuntimeError("IPIX interface protocol must be frozen before scoring")

    model_paths = [path.resolve() for path in args.models]
    model_hashes = {str(path): sha256(path) for path in model_paths}
    if len(model_paths) != 3:
        raise RuntimeError("the frozen IPIX policy requires exactly three all504 seed models")
    if len(set(model_hashes.values())) != 3:
        raise RuntimeError("the frozen IPIX policy requires three distinct model artifacts")

    scalar_payload = json.loads(args.scalar_calibration.read_text(encoding="utf-8"))
    if scalar_payload["protocol_sha256"] != sha256(CONFIG):
        raise RuntimeError("scalar calibration protocol hash mismatch")
    if scalar_payload["model_sha256"] != model_hashes:
        raise RuntimeError("scalar calibration model hashes do not match supplied models")
    scalar_multipliers = {key: float(value) for key, value in scalar_payload["multipliers"].items()}
    strongest_baseline = str(scalar_payload["strongest_development_selected_baseline"])

    feature_payload = json.loads(args.feature_calibration.read_text(encoding="utf-8"))
    if feature_payload.get("features") is None or feature_payload.get("beta") is None:
        raise RuntimeError("feature calibration is malformed")

    device = torch.device(args.device)
    models = load_models(model_paths, device)
    files = protocol[f"{args.cohort}_files"]

    output_dir = args.output_dir.resolve() / f"{args.cohort}_featurehead"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    target_pfa = float(protocol["target_pfa"])
    for file_id, spec in files.items():
        path = args.data_dir.resolve() / spec["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)

        primary = int(spec["primary"])
        target_zone = {int(value) for value in spec["secondary"]} | {primary}
        probe = load_ipix_series(path, protocol["polarizations"][0], primary, preprocess="raw")
        nrange = int(probe.metadata["nrange"])
        clutter_bins = [value for value in range(1, nrange + 1) if value not in target_zone]

        for polarization in protocol["polarizations"]:
            raw = {
                range_bin: load_ipix_series(path, polarization, range_bin, preprocess="raw").samples
                for range_bin in range(1, nrange + 1)
            }
            transform = fit_ipix_reference_transform([raw[value] for value in clutter_bins])
            blocks = {
                value: nonoverlapping_windows(
                    apply_ipix_reference_transform(raw[value], transform),
                    int(protocol["block_length"]),
                    max_windows=args.max_blocks,
                )
                for value in range(1, nrange + 1)
            }
            counts = {len(value) for value in blocks.values()}
            if len(counts) != 1 or next(iter(counts)) == 0:
                raise RuntimeError(f"inconsistent block count in {file_id}/{polarization}")

            nblocks = next(iter(counts))
            for cut_bin in range(1, nrange + 1):
                role = "primary" if cut_bin == primary else ("secondary" if cut_bin in target_zone else "clutter")
                reference_bins = reference_bins_for_cut(cut_bin, clutter_bins, int(protocol["maximum_reference_cells"]))
                cut = blocks[cut_bin]
                reference = np.stack([blocks[value] for value in reference_bins], axis=1)

                for start in range(0, len(cut), int(args.batch_size)):
                    stop = min(len(cut), start + int(args.batch_size))
                    cut_part = cut[start:stop]
                    ref_part = reference[start:stop]

                    cut_tensor = torch.as_tensor(
                        np.stack([cut_part.real, cut_part.imag], axis=-1), dtype=torch.float32, device=device
                    )
                    ref_tensor = torch.as_tensor(
                        np.stack([ref_part.real, ref_part.imag], axis=-1), dtype=torch.float32, device=device
                    )
                    alpha = torch.full((len(cut_part),), target_pfa, dtype=torch.float32, device=device)

                    thresholds: list[torch.Tensor] = []
                    feature_output: dict[str, torch.Tensor] | None = None
                    with torch.no_grad():
                        for model in models:
                            model_output = model(cut_tensor, ref_tensor, alpha)
                            thresholds.append(model_output["absolute_threshold"])
                            if feature_output is None:
                                feature_output = model_output
                    if feature_output is None:
                        raise RuntimeError("model did not produce any outputs")

                    threshold = torch.median(torch.stack(thresholds), dim=0).values.cpu().numpy()
                    score = np.sqrt(np.mean(np.abs(cut_part) ** 2, axis=1))
                    bcd_ratio = score / np.maximum(threshold, 1e-12)

                    scale_np = feature_output["scale"].detach().cpu().numpy()
                    anchor_weights = feature_output["anchor_weights"].detach().cpu().numpy()
                    sorted_weights = np.sort(anchor_weights, axis=1)[:, ::-1]
                    feature_rows = pd.DataFrame(
                        {
                            "log_scale": np.log(scale_np),
                            "anchor_entropy": -np.sum(anchor_weights * np.log(np.maximum(anchor_weights, 1e-8)), axis=1),
                            "anchor_max": sorted_weights[:, 0],
                            "anchor_gap": sorted_weights[:, 0] - sorted_weights[:, 1],
                            "uncertainty": feature_output["uncertainty"].detach().cpu().numpy(),
                            "series_threshold_shift": feature_output.get(
                                "series_threshold_shift", torch.zeros_like(feature_output["scale"])
                            ).detach().cpu().numpy(),
                        }
                    )
                    feature_rows["pol_hh"] = float(polarization == "hh")
                    feature_rows["pol_hv"] = float(polarization == "hv")
                    feature_rows["pol_vh"] = float(polarization == "vh")
                    feature_rows["pol_vv"] = float(polarization == "vv")
                    feature_rows["polarization"] = polarization
                    feature_multiplier_values = feature_multiplier(feature_payload, feature_rows)

                    baseline_outputs = classical_cfar_outputs(cut_part, ref_part, np.full(len(cut_part), target_pfa, dtype=float))
                    for block_index in range(len(cut_part)):
                        row: dict[str, Any] = {
                            "file_id": file_id,
                            "polarization": polarization,
                            "range_bin": cut_bin,
                            "role": role,
                            "block_index": start + block_index,
                            "reference_cells": len(reference_bins),
                            "ratio_bcdrcfar": float(bcd_ratio[block_index]),
                            "ratio_bcdrcfar_scalar": float(bcd_ratio[block_index]),
                            "ratio_bcdrcfar_feature": float(bcd_ratio[block_index]),
                            "feature_multiplier": float(feature_multiplier_values[block_index]),
                            "log_scale": float(feature_rows.iloc[block_index]["log_scale"]),
                            "anchor_entropy": float(feature_rows.iloc[block_index]["anchor_entropy"]),
                            "anchor_max": float(feature_rows.iloc[block_index]["anchor_max"]),
                            "anchor_gap": float(feature_rows.iloc[block_index]["anchor_gap"]),
                            "uncertainty": float(feature_rows.iloc[block_index]["uncertainty"]),
                            "series_threshold_shift": float(feature_rows.iloc[block_index]["series_threshold_shift"]),
                        }
                        for method in BASELINE_NAMES:
                            row[f"ratio_{method}"] = float(
                                baseline_outputs[method]["score"][block_index]
                                / max(baseline_outputs[method]["threshold"][block_index], 1e-12)
                            )
                        all_rows.append(row)
                print(f"[{file_id}] {polarization}: {nblocks} blocks x {nrange} CUT bins", flush=True)

    frame = pd.DataFrame(all_rows)
    frame["decision_bcdrcfar_scalar"] = frame["ratio_bcdrcfar_scalar"] >= scalar_multipliers["bcdrcfar"]
    frame["decision_bcdrcfar_feature"] = frame["ratio_bcdrcfar_feature"] >= frame["feature_multiplier"]
    for method in BASELINE_NAMES:
        frame[f"decision_{method}"] = frame[f"ratio_{method}"] >= scalar_multipliers[method]

    series, acquisition, target = summarize(frame, target_pfa)
    strongest = strongest_baseline
    bootstrap = bootstrap_summary(
        acquisition,
        target,
        strongest,
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    series_path = output_dir / "series_metrics.csv"
    acquisition_path = output_dir / "acquisition_metrics.csv"
    target_path = output_dir / "target_metrics.csv"
    rows_path = output_dir / "condition_rows.csv"
    frame.to_csv(rows_path, index=False)
    series.to_csv(series_path, index=False)
    acquisition.to_csv(acquisition_path, index=False)
    target.to_csv(target_path, index=False)
    summary = acquisition.copy()
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    payload = {
        "status": "BCDRCFAR_IPIX_FEATURE_HEAD_EVALUATION_COMPLETE",
        "cohort": args.cohort,
        "feature_calibration_sha256": sha256(args.feature_calibration),
        "scalar_calibration_sha256": sha256(args.scalar_calibration),
        "model_sha256": model_hashes,
        "rows_sha256": sha256(rows_path),
        "series_metrics_sha256": sha256(series_path),
        "acquisition_metrics_sha256": sha256(acquisition_path),
        "target_metrics_sha256": sha256(target_path),
        "summary_sha256": sha256(summary_path),
        "summary_json_sha256": None,
        "target_pfa": target_pfa,
        "strongest_development_selected_baseline": strongest_baseline,
        "feature_heads": feature_payload["features"],
        "feature_head_macro_pfa": float(acquisition[acquisition["method"] == "bcdrcfar_feature"]["pfa"].mean()),
        "feature_head_macro_absolute_log10_pfa_error": float(
            acquisition[acquisition["method"] == "bcdrcfar_feature"]["absolute_log10_pfa_error"].mean()
        ),
        "feature_head_macro_series_factor2_violation_rate": float(
            acquisition[acquisition["method"] == "bcdrcfar_feature"]["series_factor2_violation_rate"].mean()
        ),
        "feature_head_macro_primary_pd": float(
            target[(target["method"] == "bcdrcfar_feature") & (target["role"] == "primary")]["pd"].mean()
        ),
        **bootstrap,
    }
    summary_json_path = output_dir / "summary.json"
    summary_payload = {key: value for key, value in payload.items() if key != "summary_json_sha256"}
    write_json(summary_json_path, summary_payload)
    payload["summary_json_sha256"] = sha256(summary_json_path)
    write_json(output_dir / "manifest.json", payload)
    print(acquisition.to_string(index=False))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
