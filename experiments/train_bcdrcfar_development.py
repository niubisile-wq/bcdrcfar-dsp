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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bcdrcfar.model import BCDRCFAR, count_trainable_parameters  # noqa: E402
from src.bcdrcfar.protocol import load_protocol  # noqa: E402
from src.bcdrcfar.synthetic_stream import make_synthetic_batch  # noqa: E402
from src.bcdrcfar.training import bcdrcfar_loss  # noqa: E402


CONFIG = ROOT / "configs" / "bcdrcfar_protocol.json"
MANIFEST = ROOT / "data" / "manifests" / "bcdrcfar_w1d_cells.csv"
OUTPUT = ROOT / "results" / "bcdrcfar_development"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fold", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026080711)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--condition-batched", action="store_true")
    parser.add_argument("--single-target-pfa", type=float)
    parser.add_argument("--threshold-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(CONFIG)
    manifest = pd.read_csv(MANIFEST)
    development = manifest[manifest["split"] == "development"].copy()
    training = development[development["development_fold"] != int(args.validation_fold)].reset_index(drop=True)
    validation = development[development["development_fold"] == int(args.validation_fold)].reset_index(drop=True)
    if len(training) != 420 or len(validation) != 84:
        raise RuntimeError("development-fold isolation failed")
    steps = 8 if args.quick else int(args.steps)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    device = torch.device(args.device)
    model = BCDRCFAR(
        hidden_channels=int(protocol["method"]["hidden_channels"]),
        distribution_pool_bins=int(protocol["method"]["distribution_pool_bins"]),
        dilations=tuple(int(value) for value in protocol["method"]["dilations"]),
        maximum_score_multiplier=float(protocol["method"]["maximum_score_multiplier"]),
        maximum_threshold_multiplier=float(protocol["method"]["maximum_threshold_multiplier"]),
        learn_score=not bool(args.threshold_only),
    ).to(device)
    if args.threshold_only:
        for module in (model.cut_encoder, model.cut_condition, model.score_residual, model.uncertainty_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=1e-4,
    )
    history = []
    started = time.perf_counter()
    model.train()
    lengths = list(protocol["w1d"]["slow_time_lengths"])
    reference_counts = list(protocol["w1d"]["reference_cell_counts"])
    for step in range(steps):
        length = int(lengths[step % len(lengths)])
        reference_count = int(reference_counts[(step // len(lengths)) % len(reference_counts)])
        source_manifest = training.iloc[[step % len(training)]] if args.condition_batched else training
        pfa_values = [float(args.single_target_pfa)] if args.single_target_pfa is not None else protocol["w1d"]["target_pfa"]
        batch = make_synthetic_batch(
            source_manifest,
            batch_size=int(args.batch_size),
            slow_time_length=length,
            reference_cells=reference_count,
            target_pfa_values=pfa_values,
            scr_db_values=protocol["w1d"]["scr_db"],
            target_probability=0.25 if args.condition_batched else 0.5,
            seed=int(args.seed) + 1009 * step,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.cut_iq, batch.reference_iq, batch.target_pfa, batch.reference_mask)
        loss, parts = bcdrcfar_loss(
            output,
            batch.label,
            batch.target_pfa,
            batch.scenarios,
            threshold_only=bool(args.threshold_only),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append({"step": step, "length": length, "reference_cells": reference_count, **{key: float(value.cpu()) for key, value in parts.items()}})

    model.eval()
    validation_rows = []
    with torch.no_grad():
        for batch_index, length in enumerate(lengths):
            batch = make_synthetic_batch(
                validation,
                batch_size=max(32, int(args.batch_size)),
                slow_time_length=int(length),
                reference_cells=16,
                target_pfa_values=(
                    [float(args.single_target_pfa)]
                    if args.single_target_pfa is not None
                    else protocol["w1d"]["target_pfa"]
                ),
                scr_db_values=protocol["w1d"]["scr_db"],
                target_probability=0.5,
                seed=int(args.seed) + 900001 + batch_index,
                device=device,
            )
            output = model(batch.cut_iq, batch.reference_iq, batch.target_pfa, batch.reference_mask)
            for index in range(len(batch.label)):
                validation_rows.append(
                    {
                        "cell_id": batch.cell_ids[index],
                        "scenario": batch.scenarios[index],
                        "length": int(length),
                        "target_pfa": float(batch.target_pfa[index].cpu()),
                        "label": int(batch.label[index].cpu()),
                        "scr_db": float(batch.scr_db[index].cpu()),
                        "score": float(output["normalized_score"][index].cpu()),
                        "threshold": float(output["normalized_threshold"][index].cpu()),
                        "decision": int(output["decision"][index].cpu()),
                    }
                )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mode_suffix = ""
    if args.condition_batched:
        mode_suffix += "_conditionbatched"
    if args.single_target_pfa is not None:
        mode_suffix += f"_pfa{args.single_target_pfa:g}"
    if args.threshold_only:
        mode_suffix += "_thresholdonly"
    tag = "quick" if args.quick else f"seed{args.seed}_fold{args.validation_fold}{mode_suffix}"
    model_path = OUTPUT / f"{tag}_model.pt"
    history_path = OUTPUT / f"{tag}_history.csv"
    predictions_path = OUTPUT / f"{tag}_validation_predictions.csv"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "protocol_version": protocol["protocol_version"],
            "learn_score": model.learn_score,
        },
        model_path,
    )
    pd.DataFrame(history).to_csv(history_path, index=False)
    pd.DataFrame(validation_rows).to_csv(predictions_path, index=False)
    payload = {
        "status": "BCDRCFAR_QUICK_ENGINEERING_RUN_COMPLETE" if args.quick else "BCDRCFAR_DEVELOPMENT_FOLD_RUN_COMPLETE",
        "claim_status": "ENGINEERING_ONLY" if args.quick else "DEVELOPMENT_ONLY",
        "steps": steps,
        "device": str(device),
        "condition_batched": bool(args.condition_batched),
        "single_target_pfa": args.single_target_pfa,
        "threshold_only": bool(args.threshold_only),
        "elapsed_seconds": time.perf_counter() - started,
        "trainable_parameters": count_trainable_parameters(model),
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "history_sha256": sha256(history_path),
        "validation_predictions_sha256": sha256(predictions_path),
        "calibration_opened": False,
        "locked_opened": False,
    }
    (OUTPUT / f"{tag}_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
