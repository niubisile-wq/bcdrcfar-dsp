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

from experiments.evaluate_bcdrcfar_development import MedianThresholdEnsemble  # noqa: E402
from src.bcdrcfar.baselines import BASELINE_NAMES, classical_cfar_outputs  # noqa: E402
from src.bcdrcfar.evaluation import DecisionCounter, condition_rows  # noqa: E402
from src.bcdrcfar.model import BCDRCFAR  # noqa: E402
from src.bcdrcfar.protocol import load_protocol  # noqa: E402
from src.bcdrcfar.simulation import inject_swerling_target, simulate_complex_clutter  # noqa: E402


CONFIG = ROOT / "configs" / "bcdrcfar_protocol.json"
MANIFEST = ROOT / "data" / "manifests" / "bcdrcfar_w1d_cells.csv"
OUTPUT = ROOT / "results" / "bcdrcfar_multiplier_sweep"


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


def parse_multipliers(raw: str) -> list[float]:
    values = [float(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if not values:
        raise ValueError("at least one multiplier is required")
    return sorted(set(values))


def load_model(model_paths: list[Path], protocol: dict) -> nn.Module:
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
    return models[0] if len(models) == 1 else MedianThresholdEnsemble(models)


def _iq(values: np.ndarray, device: torch.device) -> torch.Tensor:
    channels = np.stack([values.real, values.imag], axis=-1).astype(np.float32)
    return torch.as_tensor(channels, device=device)


def evaluate_cell(
    *,
    model: nn.Module,
    scenario: str,
    severity: str,
    sequence_seed: int,
    parameter_seed: int,
    slow_time_length: int,
    reference_cells: int,
    target_pfa: float,
    decisions: int,
    batch_size: int,
    multipliers: list[float],
    scr_db: float | None,
    device: str,
) -> dict[str, DecisionCounter]:
    torch_device = torch.device(device)
    counters = {f"bcdrcfar_m{multiplier:.3f}": DecisionCounter() for multiplier in multipliers}
    counters.update({name: DecisionCounter() for name in BASELINE_NAMES})
    model.eval()
    generated = 0
    with torch.no_grad():
        while generated < decisions:
            size = min(batch_size, decisions - generated)
            clutter = simulate_complex_clutter(
                scenario,
                severity,
                (size, reference_cells + 1, slow_time_length),
                seed=int(sequence_seed) + 104729 * (generated // batch_size),
                parameter_seed=int(parameter_seed),
            )
            cut = clutter[:, 0]
            reference = clutter[:, 1:]
            if scr_db is not None:
                cut = inject_swerling_target(
                    cut,
                    reference,
                    float(scr_db),
                    swerling="I",
                    seed=int(sequence_seed) + 1_000_003 + generated,
                )
            alpha = np.full(size, float(target_pfa), dtype=np.float32)
            output = model(
                _iq(cut, torch_device),
                _iq(reference, torch_device),
                torch.as_tensor(alpha, device=torch_device),
                torch.ones((size, reference_cells), dtype=torch.bool, device=torch_device),
            )
            score = output["normalized_score"]
            threshold = output["normalized_threshold"]
            for multiplier in multipliers:
                decision = score >= threshold * float(multiplier)
                counters[f"bcdrcfar_m{multiplier:.3f}"].update(decision.cpu().numpy())
            for name, result in classical_cfar_outputs(cut, reference, alpha).items():
                counters[name].update(result["decision"])
            generated += size
    return counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--ensemble-models", type=Path, nargs="+")
    parser.add_argument("--validation-fold", type=int, required=True)
    parser.add_argument("--max-cells", type=int, default=84)
    parser.add_argument("--decisions", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--reference-cells", type=int, default=24)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--scr-db", type=float, default=0.0)
    parser.add_argument("--multipliers", default="0.70,0.80,0.90,1.00")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_paths = [Path(path) for path in args.ensemble_models] if args.ensemble_models else [Path(args.model)]
    multipliers = parse_multipliers(args.multipliers)
    protocol = load_protocol(CONFIG)
    manifest = pd.read_csv(MANIFEST)
    cells = manifest[
        (manifest["split"] == "development")
        & (manifest["development_fold"] == int(args.validation_fold))
    ].sort_values("cell_id").head(int(args.max_cells))
    if cells.empty:
        raise RuntimeError("no held-out development cells selected")
    model = load_model(model_paths, protocol).to(args.device)

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
            multipliers=multipliers,
            device=args.device,
        )
        pfa = evaluate_cell(sequence_seed=int(cell["sequence_seed"]), scr_db=None, **common)
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
        pd_counts = evaluate_cell(sequence_seed=int(cell["target_seed"]), scr_db=float(args.scr_db), **common)
        rows.extend(
            condition_rows(
                pd_counts,
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
    summary = pfa_frame.groupby("method").agg(
        pooled_events=("events", "sum"),
        pooled_trials=("trials", "sum"),
        median_absolute_log10_pfa_error=("absolute_log10_pfa_error", "median"),
        factor2_violation_rate=("factor2_violation", "mean"),
    )
    summary["pooled_pfa"] = summary["pooled_events"] / summary["pooled_trials"]
    summary = summary.join(pd_frame.groupby("method").agg(pd=("rate", "mean"), cells=("cell_id", "nunique"))).reset_index()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    digest = combined_sha256(model_paths)
    tag = (
        f"fold{args.validation_fold}_ensemble{len(model_paths)}_{digest[:8]}_K{args.reference_cells}_"
        f"n{args.decisions}_pfa{args.target_pfa:g}_mgrid"
    )
    raw_path = OUTPUT / f"{tag}_conditions.csv"
    summary_path = OUTPUT / f"{tag}_summary.csv"
    manifest_path = OUTPUT / f"{tag}_manifest.json"
    frame.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    payload = {
        "status": "BCDRCFAR_MULTIPLIER_SWEEP_COMPLETE",
        "claim_status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "validation_fold": int(args.validation_fold),
        "cells": int(cells["cell_id"].nunique()),
        "decisions_per_endpoint_per_cell": int(args.decisions),
        "reference_cells": int(args.reference_cells),
        "target_pfa": float(args.target_pfa),
        "multipliers": multipliers,
        "device": str(args.device),
        "elapsed_seconds": time.perf_counter() - started,
        "model_sha256": digest,
        "ensemble_size": len(model_paths),
        "model_paths": [str(path) for path in model_paths],
        "conditions_sha256": sha256(raw_path),
        "summary_sha256": sha256(summary_path),
        "calibration_opened": False,
        "locked_opened": False,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
