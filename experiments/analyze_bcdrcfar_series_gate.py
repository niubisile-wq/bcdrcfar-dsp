"""Inspect the learned series threshold gate on synthetic development cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_bcdrcfar_stage_b_score import base_normalized_score  # noqa: E402
from src.bcdrcfar.model import BCDRCFAR  # noqa: E402
from src.bcdrcfar.protocol import load_protocol  # noqa: E402
from src.bcdrcfar.synthetic_stream import make_synthetic_batch  # noqa: E402


CONFIG = ROOT / "configs" / "bcdrcfar_protocol.json"
MANIFEST = ROOT / "data" / "manifests" / "bcdrcfar_w1d_cells.csv"
OUTPUT = ROOT / "results" / "bcdrcfar_stage_b" / "series_gate_diagnostics"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=5)
    parser.add_argument("--decisions", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--reference-cells", type=int, default=24)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--scr-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026080801)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(path: Path, protocol: dict, device: torch.device) -> BCDRCFAR:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
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
    return model


def main() -> None:
    args = parse_args()
    protocol = load_protocol(CONFIG)
    manifest = pd.read_csv(MANIFEST)
    development = manifest[manifest["split"] == "development"].copy()
    cells = development[development["development_fold"] == int(args.validation_fold)].sort_values("cell_id")
    if cells.empty:
        raise RuntimeError("no held-out development cells for the selected fold")
    model = load_model(args.model, protocol, torch.device(args.device))
    rows: list[dict[str, object]] = []
    for _, cell in cells.iterrows():
        cell_frame = cell.to_frame().T.reset_index(drop=True)
        for endpoint, target_probability, scr_db, seed_offset in (
            ("h0", 0.0, 0.0, 0),
            ("h1", 1.0, float(args.scr_db), 1_000_003),
        ):
            generated = 0
            shift_means: list[float] = []
            shift_stds: list[float] = []
            multiplier_means: list[float] = []
            score_ratio_means: list[float] = []
            pfa_events = 0
            pfa_trials = 0
            while generated < int(args.decisions):
                size = min(int(args.batch_size), int(args.decisions) - generated)
                batch = make_synthetic_batch(
                    manifest=cell_frame,
                    batch_size=size,
                    slow_time_length=int(args.length),
                    reference_cells=int(args.reference_cells),
                    target_pfa_values=[float(args.target_pfa)],
                    selected_indices=np.zeros(size, dtype=int),
                    scr_db_values=[scr_db],
                    target_probability=target_probability,
                    seed=int(args.seed) + seed_offset + generated,
                    device=args.device,
                )
                with torch.no_grad():
                    out = model(batch.cut_iq, batch.reference_iq, batch.target_pfa, batch.reference_mask)
                    anchored_threshold = (out["anchor_weights"] * out["anchors"]).sum(dim=1)
                    multiplier = out["normalized_threshold"] / anchored_threshold.clamp_min(1e-8)
                    shift = out["series_threshold_shift"]
                    base = base_normalized_score(batch.cut_iq, out["scale"])
                    score_ratio = out["normalized_score"] / base.clamp_min(1e-8)
                    decision = (out["normalized_score"] >= out["normalized_threshold"]).cpu().numpy()
                shift_means.append(float(shift.mean().cpu()))
                shift_stds.append(float(shift.std(unbiased=False).cpu()))
                multiplier_means.append(float(multiplier.mean().cpu()))
                score_ratio_means.append(float(score_ratio.mean().cpu()))
                pfa_events += int(np.count_nonzero(decision))
                pfa_trials += int(decision.size)
                generated += size
            rows.append(
                {
                    "cell_id": str(cell["cell_id"]),
                    "scenario": str(cell["scenario"]),
                    "severity": str(cell["severity"]),
                    "endpoint": endpoint,
                    "series_shift_mean": float(np.mean(shift_means)),
                    "series_shift_std": float(np.mean(shift_stds)),
                    "threshold_multiplier_mean": float(np.mean(multiplier_means)),
                    "score_ratio_mean": float(np.mean(score_ratio_means)),
                    "pooled_rate": float(pfa_events / max(pfa_trials, 1)),
                }
            )
    frame = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model.stem}_{sha256(args.model)[:8]}_fold{args.validation_fold}_n{args.decisions}"
    csv_path = OUTPUT / f"{stem}.csv"
    json_path = OUTPUT / f"{stem}.json"
    frame.to_csv(csv_path, index=False)
    payload = {
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "csv_sha256": sha256(csv_path),
        "cells": int(frame["cell_id"].nunique()),
        "validation_fold": int(args.validation_fold),
        "decisions": int(args.decisions),
        "reference_cells": int(args.reference_cells),
        "target_pfa": float(args.target_pfa),
        "scr_db": float(args.scr_db),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(frame.groupby("endpoint").agg(
        pooled_rate=("pooled_rate", "mean"),
        series_shift_mean=("series_shift_mean", "mean"),
        threshold_multiplier_mean=("threshold_multiplier_mean", "mean"),
        score_ratio_mean=("score_ratio_mean", "mean"),
    ).to_string())
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
