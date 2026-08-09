from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bcdrcfar.evaluation import condition_rows, evaluate_synthetic_condition  # noqa: E402
from src.bcdrcfar.model import BCDRCFAR  # noqa: E402
from src.bcdrcfar.protocol import load_protocol  # noqa: E402


CONFIG = ROOT / "configs" / "bcdrcfar_protocol.json"
MANIFEST = ROOT / "data" / "manifests" / "bcdrcfar_w1d_cells.csv"
DEFAULT_MODEL = ROOT / "results" / "bcdrcfar_development" / "quick_model.pt"
OUTPUT = ROOT / "results" / "bcdrcfar_development_evaluation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


class MedianThresholdEnsemble(nn.Module):
    """Median aggregation of fixed-score Stage-A threshold networks."""

    def __init__(self, models: list[BCDRCFAR]) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError("an ensemble requires at least two models")
        self.models = nn.ModuleList(models)

    def forward(self, cut_iq, reference_iq, target_pfa, reference_mask=None):
        outputs = [model(cut_iq, reference_iq, target_pfa, reference_mask) for model in self.models]
        threshold = torch.median(
            torch.stack([output["normalized_threshold"] for output in outputs], dim=0), dim=0
        ).values
        result = dict(outputs[0])
        result["normalized_threshold"] = threshold
        result["absolute_threshold"] = threshold * outputs[0]["scale"]
        result["decision"] = result["normalized_score"] >= threshold
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ensemble-models", type=Path, nargs="+")
    parser.add_argument("--validation-fold", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=84)
    parser.add_argument("--decisions", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--reference-cells", type=int, default=8)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--scr-db", type=float, default=0.0)
    parser.add_argument("--model-threshold-multiplier", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(CONFIG)
    manifest = pd.read_csv(MANIFEST)
    cells = manifest[
        (manifest["split"] == "development")
        & (manifest["development_fold"] == int(args.validation_fold))
    ].sort_values("cell_id").head(int(args.max_cells))
    if cells.empty or set(cells["split"]) != {"development"}:
        raise RuntimeError("evaluation may only use held-out development cells")
    model_paths = [Path(path) for path in args.ensemble_models] if args.ensemble_models else [args.model]
    models = []
    for model_path in model_paths:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        member = BCDRCFAR(
            hidden_channels=int(protocol["method"]["hidden_channels"]),
            distribution_pool_bins=int(protocol["method"]["distribution_pool_bins"]),
            dilations=tuple(int(value) for value in protocol["method"]["dilations"]),
            maximum_score_multiplier=float(protocol["method"]["maximum_score_multiplier"]),
            maximum_threshold_multiplier=float(protocol["method"]["maximum_threshold_multiplier"]),
            learn_score=bool(checkpoint.get("learn_score", True)),
        )
        member.load_state_dict(checkpoint["state_dict"], strict=False)
        models.append(member)
    model = models[0] if len(models) == 1 else MedianThresholdEnsemble(models)
    model.to(args.device)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for _, cell in cells.iterrows():
        common = dict(
            model=model,
            scenario=str(cell["scenario"]),
            severity=str(cell["severity"]),
            parameter_seed=int(cell["parameter_seed"]),
            slow_time_length=int(args.length),
            reference_cells=int(args.reference_cells),
            target_pfa=float(args.target_pfa),
            decisions=int(args.decisions),
            batch_size=int(args.batch_size),
            model_threshold_multiplier=float(args.model_threshold_multiplier),
            device=args.device,
        )
        pfa = evaluate_synthetic_condition(sequence_seed=int(cell["sequence_seed"]), **common)
        rows.extend(
            condition_rows(
                pfa,
                cell_id=str(cell["cell_id"]),
                scenario=str(cell["scenario"]),
                severity=str(cell["severity"]),
                target_pfa=float(args.target_pfa),
                endpoint="pfa",
            )
        )
        pd_counters = evaluate_synthetic_condition(
            sequence_seed=int(cell["target_seed"]),
            scr_db=float(args.scr_db),
            swerling="I",
            **common,
        )
        rows.extend(
            condition_rows(
                pd_counters,
                cell_id=str(cell["cell_id"]),
                scenario=str(cell["scenario"]),
                severity=str(cell["severity"]),
                target_pfa=float(args.target_pfa),
                endpoint="pd",
                scr_db=float(args.scr_db),
            )
        )
    frame = pd.DataFrame(rows)
    pfa_frame = frame[frame["endpoint"] == "pfa"]
    pd_frame = frame[frame["endpoint"] == "pd"]
    pfa_summary = pfa_frame.groupby("method").agg(
        pooled_events=("events", "sum"),
        pooled_trials=("trials", "sum"),
        median_absolute_log10_pfa_error=("absolute_log10_pfa_error", "median"),
        factor2_violation_rate=("factor2_violation", "mean"),
    )
    pfa_summary["pooled_pfa"] = pfa_summary["pooled_events"] / pfa_summary["pooled_trials"]
    pd_summary = pd_frame.groupby("method").agg(pd=("rate", "mean"), cells=("cell_id", "nunique"))
    summary = pfa_summary.join(pd_summary).reset_index()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_digest = combined_sha256(model_paths)
    model_tag = (
        model_paths[0].stem.replace("_model", "")
        if len(model_paths) == 1
        else f"median_ensemble{len(model_paths)}"
    )
    model_hash_tag = model_digest[:8]
    tag = (
        f"{model_tag}_{model_hash_tag}_fold{args.validation_fold}_n{args.decisions}_"
        f"pfa{args.target_pfa:g}_m{args.model_threshold_multiplier:.6f}"
    )
    raw_path = OUTPUT / f"{tag}_conditions.csv"
    summary_path = OUTPUT / f"{tag}_summary.csv"
    frame.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    payload = {
        "status": "BCDRCFAR_DEVELOPMENT_ENGINEERING_EVALUATION_COMPLETE",
        "claim_status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "cells": int(cells["cell_id"].nunique()),
        "decisions_per_endpoint_per_cell": int(args.decisions),
        "target_pfa": float(args.target_pfa),
        "scr_db": float(args.scr_db),
        "model_threshold_multiplier": float(args.model_threshold_multiplier),
        "device": str(args.device),
        "elapsed_seconds": time.perf_counter() - started,
        "model_sha256": model_digest,
        "ensemble_size": len(model_paths),
        "model_paths": [str(path) for path in model_paths],
        "conditions_sha256": sha256(raw_path),
        "summary_sha256": sha256(summary_path),
        "calibration_opened": False,
        "locked_opened": False,
    }
    (OUTPUT / f"{tag}_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
