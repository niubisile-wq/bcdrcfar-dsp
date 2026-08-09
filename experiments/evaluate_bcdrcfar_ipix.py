"""Evaluate frozen BC-DRCFAR thresholds on acquisition-disjoint IPIX data.

The five development acquisitions estimate one scalar domain-calibration
factor per method.  The nine retrospective external acquisitions may only be
scored with that frozen calibration artifact and the exact same model hashes.
"""

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

from src.bcdrcfar.baselines import BASELINE_NAMES, classical_cfar_outputs  # noqa: E402
from src.bcdrcfar.model import BCDRCFAR  # noqa: E402
from src.bcdrcfar.protocol import load_protocol as load_bcd_protocol  # noqa: E402
from src.real_data import (  # noqa: E402
    apply_ipix_reference_transform,
    fit_ipix_reference_transform,
    load_ipix_series,
    nonoverlapping_windows,
)


CONFIG = ROOT / "configs" / "bcdrcfar_ipix_protocol.json"
MODEL_CONFIG = ROOT / "configs" / "bcdrcfar_protocol.json"
DATA = ROOT / "data" / "raw" / "ipix"
OUTPUT = ROOT / "results" / "bcdrcfar_ipix"
METHODS = ("bcdrcfar", *BASELINE_NAMES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    path.write_text(
        json.dumps(safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("development", "retrospective_external"), required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def reference_bins_for_cut(
    cut_bin: int, clutter_bins: list[int], maximum: int
) -> list[int]:
    """Return deterministic nearest-first references without duplicating cells."""

    candidates = [value for value in clutter_bins if value != int(cut_bin)]
    candidates.sort(key=lambda value: (abs(value - int(cut_bin)), value))
    result = candidates[: int(maximum)]
    if len(result) < 4 or len(result) != len(set(result)):
        raise ValueError("each IPIX CUT requires at least four unique clutter references")
    return result


def load_models(paths: list[Path], device: torch.device) -> list[BCDRCFAR]:
    protocol = load_bcd_protocol(MODEL_CONFIG)
    models: list[BCDRCFAR] = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("fit_scope") != "all504":
            raise RuntimeError(f"external IPIX evaluation requires an all504 model: {path}")
        model = BCDRCFAR(
            hidden_channels=int(protocol["method"]["hidden_channels"]),
            distribution_pool_bins=int(protocol["method"]["distribution_pool_bins"]),
            dilations=tuple(int(value) for value in protocol["method"]["dilations"]),
            maximum_score_multiplier=float(protocol["method"]["maximum_score_multiplier"]),
            maximum_threshold_multiplier=float(protocol["method"]["maximum_threshold_multiplier"]),
            learn_score=bool(checkpoint.get("learn_score", True)),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.eval().to(device)
        models.append(model)
    return models


def bcd_ratios(
    cut: np.ndarray,
    reference: np.ndarray,
    *,
    models: list[BCDRCFAR],
    target_pfa: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ratios: list[np.ndarray] = []
    _features: dict[str, list[np.ndarray]] = {
        "log_scale": [],
        "anchor_entropy": [],
        "anchor_max": [],
        "anchor_gap": [],
        "uncertainty": [],
        "series_threshold_shift": [],
    }
    for start in range(0, len(cut), int(batch_size)):
        stop = min(len(cut), start + int(batch_size))
        cut_part = cut[start:stop]
        ref_part = reference[start:stop]
        cut_tensor = torch.as_tensor(
            np.stack([cut_part.real, cut_part.imag], axis=-1), dtype=torch.float32, device=device
        )
        ref_tensor = torch.as_tensor(
            np.stack([ref_part.real, ref_part.imag], axis=-1), dtype=torch.float32, device=device
        )
        alpha = torch.full((len(cut_part),), float(target_pfa), dtype=torch.float32, device=device)
        thresholds = []
        output0 = None
        with torch.no_grad():
            for model in models:
                output = model(cut_tensor, ref_tensor, alpha)
                thresholds.append(output["absolute_threshold"])
                if output0 is None:
                    output0 = output
        threshold = torch.median(torch.stack(thresholds), dim=0).values.cpu().numpy()
        score = np.sqrt(np.mean(np.abs(cut_part) ** 2, axis=1))
        ratios.append(score / np.maximum(threshold, 1e-12))
        if output0 is None:
            raise RuntimeError("model did not produce any outputs")
        anchor_weights = output0["anchor_weights"].detach().cpu().numpy()
        sorted_weights = np.sort(anchor_weights, axis=1)[:, ::-1]
        entropy = -np.sum(anchor_weights * np.log(np.maximum(anchor_weights, 1e-8)), axis=1)
        _features["log_scale"].append(np.log(output0["scale"].detach().cpu().numpy()))
        _features["anchor_entropy"].append(entropy)
        _features["anchor_max"].append(sorted_weights[:, 0])
        _features["anchor_gap"].append(sorted_weights[:, 0] - sorted_weights[:, 1])
        _features["uncertainty"].append(output0["uncertainty"].detach().cpu().numpy())
        _features["series_threshold_shift"].append(
            output0.get("series_threshold_shift", torch.zeros_like(output0["scale"])).detach().cpu().numpy()
        )
    features = {key: np.concatenate(value) for key, value in _features.items()}
    return np.concatenate(ratios), features


def score_series(
    cut: np.ndarray,
    reference: np.ndarray,
    *,
    models: list[BCDRCFAR],
    target_pfa: float,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    bcd_ratio, features = bcd_ratios(
        cut,
        reference,
        models=models,
        target_pfa=target_pfa,
        batch_size=batch_size,
        device=device,
    )
    output = {"bcdrcfar": bcd_ratio}
    baseline = classical_cfar_outputs(
        cut, reference, np.full(len(cut), float(target_pfa), dtype=float)
    )
    for method in BASELINE_NAMES:
        output[method] = baseline[method]["score"] / np.maximum(
            baseline[method]["threshold"], 1e-12
        )
    return output, features


def acquisition_rows(
    file_id: str,
    spec: dict[str, Any],
    path: Path,
    *,
    protocol: dict[str, Any],
    models: list[BCDRCFAR],
    max_blocks: int | None,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    primary = int(spec["primary"])
    target_zone = {int(value) for value in spec["secondary"]} | {primary}
    probe = load_ipix_series(path, protocol["polarizations"][0], primary, preprocess="raw")
    nrange = int(probe.metadata["nrange"])
    clutter_bins = [value for value in range(1, nrange + 1) if value not in target_zone]
    rows: list[dict[str, Any]] = []
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
                max_windows=max_blocks,
            )
            for value in range(1, nrange + 1)
        }
        counts = {len(value) for value in blocks.values()}
        if len(counts) != 1 or next(iter(counts)) == 0:
            raise RuntimeError(f"inconsistent block count in {file_id}/{polarization}")
        for cut_bin in range(1, nrange + 1):
            role = "primary" if cut_bin == primary else ("secondary" if cut_bin in target_zone else "clutter")
            reference_bins = reference_bins_for_cut(
                cut_bin, clutter_bins, int(protocol["maximum_reference_cells"])
            )
            cut = blocks[cut_bin]
            reference = np.stack([blocks[value] for value in reference_bins], axis=1)
            ratios, features = score_series(
                cut,
                reference,
                models=models,
                target_pfa=float(protocol["target_pfa"]),
                batch_size=batch_size,
                device=device,
            )
            for block_index in range(len(cut)):
                row: dict[str, Any] = {
                    "file_id": file_id,
                    "polarization": polarization,
                    "range_bin": cut_bin,
                    "role": role,
                    "block_index": block_index,
                    "reference_cells": len(reference_bins),
                }
                for method in METHODS:
                    row[f"ratio_{method}"] = float(ratios[method][block_index])
                for key, values in features.items():
                    row[key] = float(values[block_index])
                rows.append(row)
        print(f"[{file_id}] {polarization}: {next(iter(counts))} blocks x {nrange} CUT bins", flush=True)
    return rows


def fit_calibration(frame: pd.DataFrame, target_pfa: float) -> dict[str, float]:
    clutter = frame[frame["role"] == "clutter"]
    return {
        method: float(
            np.quantile(
                clutter[f"ratio_{method}"].to_numpy(dtype=float),
                1.0 - float(target_pfa),
                method="higher",
            )
        )
        for method in METHODS
    }


def summarize(
    frame: pd.DataFrame, multipliers: dict[str, float], target_pfa: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["file_id", "polarization", "range_bin", "role"]
    series_rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(keys, sort=True):
        for method in METHODS:
            events = int((group[f"ratio_{method}"] >= multipliers[method]).sum())
            trials = len(group)
            rate = events / trials
            adjusted = (events + 0.5) / (trials + 1.0)
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
    acquisition_rows_out = []
    for (file_id, method), group in clutter.groupby(["file_id", "method"], sort=True):
        events = int(group["events"].sum())
        trials = int(group["trials"].sum())
        adjusted = (events + 0.5) / (trials + 1.0)
        acquisition_rows_out.append(
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
    acquisition = pd.DataFrame(acquisition_rows_out)
    target = (
        series[series["role"].isin(["primary", "secondary"])]
        .groupby(["file_id", "role", "method"], sort=True)
        .agg(events=("events", "sum"), trials=("trials", "sum"))
        .reset_index()
    )
    target["pd"] = target["events"] / target["trials"]
    return series, acquisition, target


def choose_strongest_baseline(acquisition: pd.DataFrame) -> str:
    baseline = (
        acquisition[acquisition["method"].isin(BASELINE_NAMES)]
        .groupby("method")["absolute_log10_pfa_error"]
        .mean()
        .sort_values()
    )
    return str(baseline.index[0])


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
    error_diff = (error.loc[common, "bcdrcfar"] - error.loc[common, strongest]).to_numpy()
    pd_diff = (pd_primary.loc[common, "bcdrcfar"] - pd_primary.loc[common, strongest]).to_numpy()
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
    device = torch.device(args.device)
    models = load_models(model_paths, device)
    cohort_key = f"{args.cohort}_files"
    files = protocol[cohort_key]
    if args.cohort == "retrospective_external" and args.max_blocks is not None:
        raise RuntimeError("external evaluation cannot use a quick block subset")
    calibration_payload = None
    if args.cohort == "retrospective_external":
        if args.calibration is None:
            raise RuntimeError("external evaluation requires the frozen development calibration")
        calibration_payload = json.loads(args.calibration.read_text(encoding="utf-8"))
        if calibration_payload["protocol_sha256"] != sha256(CONFIG):
            raise RuntimeError("calibration protocol hash mismatch")
        if calibration_payload["model_sha256"] != model_hashes:
            raise RuntimeError("calibration model hashes do not match supplied models")
    tag = args.cohort + (f"_quick{args.max_blocks}" if args.max_blocks is not None else "_full")
    output = args.output_dir.resolve() / tag
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for file_id, spec in files.items():
        path = args.data_dir.resolve() / spec["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        all_rows.extend(
            acquisition_rows(
                str(file_id),
                spec,
                path,
                protocol=protocol,
                models=models,
                max_blocks=args.max_blocks,
                batch_size=int(args.batch_size),
                device=device,
            )
        )
    frame = pd.DataFrame(all_rows)
    ratios_path = output / "block_ratios.csv"
    frame.to_csv(ratios_path, index=False)
    if calibration_payload is None:
        multipliers = fit_calibration(frame, float(protocol["target_pfa"]))
    else:
        multipliers = {key: float(value) for key, value in calibration_payload["multipliers"].items()}
    series, acquisition, target = summarize(frame, multipliers, float(protocol["target_pfa"]))
    strongest = (
        choose_strongest_baseline(acquisition)
        if calibration_payload is None
        else str(calibration_payload["strongest_development_selected_baseline"])
    )
    bootstrap = bootstrap_summary(
        acquisition,
        target,
        strongest,
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    series_path = output / "series_metrics.csv"
    acquisition_path = output / "acquisition_metrics.csv"
    target_path = output / "target_metrics.csv"
    series.to_csv(series_path, index=False)
    acquisition.to_csv(acquisition_path, index=False)
    target.to_csv(target_path, index=False)
    bcd = acquisition[acquisition["method"] == "bcdrcfar"]
    primary = target[(target["method"] == "bcdrcfar") & (target["role"] == "primary")]
    summary = {
        "status": "BCDRCFAR_IPIX_EVALUATION_COMPLETE",
        "cohort": args.cohort,
        "evidence_level": (
            "REAL_DATA_INTERFACE_DEVELOPMENT"
            if args.cohort == "development"
            else "METHOD_PREFROZEN_RETROSPECTIVE_EXTERNAL_NOT_BLIND"
        ),
        "acquisitions": len(files),
        "max_blocks": args.max_blocks,
        "target_pfa": float(protocol["target_pfa"]),
        "multipliers": multipliers,
        "bcdrcfar_macro_pfa": float(bcd["pfa"].mean()),
        "bcdrcfar_macro_absolute_log10_pfa_error": float(bcd["absolute_log10_pfa_error"].mean()),
        "bcdrcfar_macro_series_factor2_violation_rate": float(bcd["series_factor2_violation_rate"].mean()),
        "bcdrcfar_macro_primary_pd": float(primary["pd"].mean()),
        **bootstrap,
        "protocol_sha256": sha256(CONFIG),
        "model_sha256": model_hashes,
        "block_ratios_sha256": sha256(ratios_path),
        "series_metrics_sha256": sha256(series_path),
        "acquisition_metrics_sha256": sha256(acquisition_path),
        "target_metrics_sha256": sha256(target_path),
        "w1d_calibration_opened": False,
        "w1d_locked_opened": False,
    }
    write_json(output / "summary.json", summary)
    if args.cohort == "development" and args.max_blocks is None:
        write_json(
            output / "frozen_domain_calibration.json",
            {
                "status": "FROZEN_FROM_FIVE_IPIX_DEVELOPMENT_ACQUISITIONS",
                "protocol_sha256": sha256(CONFIG),
                "model_sha256": model_hashes,
                "multipliers": multipliers,
                "strongest_development_selected_baseline": strongest,
                "source_summary_sha256": sha256(output / "summary.json"),
            },
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
