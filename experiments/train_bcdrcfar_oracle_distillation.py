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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bcdrcfar.model import BCDRCFAR, count_trainable_parameters  # noqa: E402
from src.bcdrcfar.protocol import load_protocol  # noqa: E402
from src.bcdrcfar.synthetic_stream import make_synthetic_batch  # noqa: E402


CONFIG = ROOT / "configs" / "bcdrcfar_protocol.json"
MANIFEST = ROOT / "data" / "manifests" / "bcdrcfar_w1d_cells.csv"
DEFAULT_ORACLE = ROOT / "results" / "bcdrcfar_development_oracle" / "L128_K8_pfa0.01_n8192_scaleeq.csv"
OUTPUT = ROOT / "results" / "bcdrcfar_oracle_distillation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-fold", type=int, default=5)
    parser.add_argument(
        "--train-all-development",
        action="store_true",
        help=(
            "Fit the frozen final candidate on all 504 development cells. "
            "The reported post-fit audit is not a validation estimate."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--worst-decile-weight", type=float, default=0.5)
    parser.add_argument("--point-loss", choices=("smooth_l1", "mse"), default="smooth_l1")
    parser.add_argument("--repeats-per-cell", type=int, default=1)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026080711)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(CONFIG)
    manifest = pd.read_csv(MANIFEST)
    oracle = pd.read_csv(args.oracle)
    expected_columns = {
        "cell_id",
        "oracle_normalized_threshold",
        "slow_time_length",
        "reference_cells",
        "target_pfa",
    }
    if not expected_columns.issubset(oracle.columns) or oracle["cell_id"].nunique() != 504:
        raise RuntimeError("oracle table is incomplete or malformed")
    oracle_by_cell = oracle.set_index("cell_id")["oracle_normalized_threshold"].to_dict()
    development = manifest[manifest["split"] == "development"].copy()
    if args.train_all_development:
        training = development.reset_index(drop=True)
        validation = development.reset_index(drop=True)
        fit_scope = "all504"
        audit_kind = "post_fit_development_audit_not_validation"
        if len(training) != 504:
            raise RuntimeError("all-development fit requires exactly 504 development cells")
    else:
        training = development[development["development_fold"] != int(args.validation_fold)].reset_index(drop=True)
        validation = development[development["development_fold"] == int(args.validation_fold)].reset_index(drop=True)
        fit_scope = f"fold{args.validation_fold}"
        audit_kind = "held_out_development_fold_validation"
        if len(training) != 420 or len(validation) != 84:
            raise RuntimeError("development fold isolation failed")
    length = int(oracle["slow_time_length"].iloc[0])
    reference_cells = int(oracle["reference_cells"].iloc[0])
    target_pfa = float(oracle["target_pfa"].iloc[0])
    if oracle[["slow_time_length", "reference_cells", "target_pfa"]].drop_duplicates().shape[0] != 1:
        raise RuntimeError("one distillation run requires one L/K/Pfa oracle grid")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    device = torch.device(args.device)
    model = BCDRCFAR(
        hidden_channels=int(protocol["method"]["hidden_channels"]),
        distribution_pool_bins=int(protocol["method"]["distribution_pool_bins"]),
        dilations=tuple(int(value) for value in protocol["method"]["dilations"]),
        maximum_score_multiplier=float(protocol["method"]["maximum_score_multiplier"]),
        maximum_threshold_multiplier=float(protocol["method"]["maximum_threshold_multiplier"]),
        learn_score=False,
    ).to(device)
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
    for step in range(int(args.steps)):
        repeats_per_cell = int(args.repeats_per_cell)
        if repeats_per_cell < 1 or int(args.batch_size) % repeats_per_cell:
            raise ValueError("batch-size must be divisible by repeats-per-cell")
        selection_rng = np.random.default_rng(int(args.seed) + 65537 * step)
        selected_indices = np.repeat(
            selection_rng.integers(0, len(training), size=int(args.batch_size) // repeats_per_cell),
            repeats_per_cell,
        )
        batch = make_synthetic_batch(
            training,
            batch_size=int(args.batch_size),
            slow_time_length=length,
            reference_cells=reference_cells,
            target_pfa_values=[target_pfa],
            scr_db_values=[0.0],
            target_probability=0.0,
            seed=int(args.seed) + 1009 * step,
            device=device,
            selected_indices=selected_indices,
        )
        target = torch.as_tensor(
            [oracle_by_cell[cell_id] for cell_id in batch.cell_ids],
            dtype=torch.float32,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.cut_iq, batch.reference_iq, batch.target_pfa, batch.reference_mask)
        log_error = torch.log(output["normalized_threshold"].clamp_min(1e-8)) - torch.log(target)
        if args.point_loss == "mse":
            point_loss = 0.5 * log_error.square()
        else:
            point_loss = F.smooth_l1_loss(log_error, torch.zeros_like(log_error), reduction="none")
        worst_count = max(1, int(round(0.1 * len(point_loss))))
        loss = point_loss.mean() + float(args.worst_decile_weight) * torch.topk(
            point_loss, worst_count
        ).values.mean()
        if repeats_per_cell > 1 and float(args.consistency_weight) > 0.0:
            grouped_log_threshold = torch.log(output["normalized_threshold"].clamp_min(1e-8)).reshape(
                -1, repeats_per_cell
            )
            consistency_loss = grouped_log_threshold.var(dim=1, unbiased=False).mean()
            loss = loss + float(args.consistency_weight) * consistency_loss
        else:
            consistency_loss = torch.zeros((), device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "mean_absolute_log_error": float(log_error.detach().abs().mean().cpu()),
                "worst_decile_loss": float(torch.topk(point_loss.detach(), worst_count).values.mean().cpu()),
                "consistency_loss": float(consistency_loss.detach().cpu()),
            }
        )

    model.eval()
    validation_rows = []
    with torch.no_grad():
        for repeat in range(8):
            batch = make_synthetic_batch(
                validation,
                batch_size=len(validation),
                slow_time_length=length,
                reference_cells=reference_cells,
                target_pfa_values=[target_pfa],
                scr_db_values=[0.0],
                target_probability=0.0,
                seed=int(args.seed) + 9_000_001 + repeat,
                device=device,
            )
            output = model(batch.cut_iq, batch.reference_iq, batch.target_pfa, batch.reference_mask)
            for index, cell_id in enumerate(batch.cell_ids):
                predicted = float(output["normalized_threshold"][index].cpu())
                target = float(oracle_by_cell[cell_id])
                validation_rows.append(
                    {
                        "repeat": repeat,
                        "cell_id": cell_id,
                        "scenario": batch.scenarios[index],
                        "oracle_threshold": target,
                        "predicted_threshold": predicted,
                        "absolute_log_error": abs(float(np.log(predicted / target))),
                        "relative_error": predicted / target - 1.0,
                    }
                )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    oracle_hash = sha256(args.oracle)
    teacher_kind = "scaleeq" if "oracle_normalized_threshold" in oracle.columns else "absolute"
    tag = (
        f"{model.feature_schema}_{teacher_kind}_{oracle_hash[:8]}_seed{args.seed}_{fit_scope}_"
        f"L{length}_K{reference_cells}_pfa{target_pfa:g}_{args.point_loss}_"
        f"worst{args.worst_decile_weight:g}_rep{args.repeats_per_cell}_cons{args.consistency_weight:g}"
    )
    model_path = OUTPUT / f"{tag}_model.pt"
    history_path = OUTPUT / f"{tag}_history.csv"
    validation_path = OUTPUT / f"{tag}_validation.csv"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "protocol_version": protocol["protocol_version"],
            "learn_score": False,
            "training_mode": "development_oracle_distillation",
            "fit_scope": fit_scope,
            "audit_kind": audit_kind,
            "seed": int(args.seed),
            "feature_schema": model.feature_schema,
            "oracle_sha256": oracle_hash,
        },
        model_path,
    )
    pd.DataFrame(history).to_csv(history_path, index=False)
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(validation_path, index=False)
    payload = {
        "status": "BCDRCFAR_DEVELOPMENT_ORACLE_DISTILLATION_COMPLETE",
        "claim_status": "DEVELOPMENT_ONLY",
        "fit_scope": fit_scope,
        "audit_kind": audit_kind,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "worst_decile_weight": float(args.worst_decile_weight),
        "point_loss": str(args.point_loss),
        "repeats_per_cell": int(args.repeats_per_cell),
        "consistency_weight": float(args.consistency_weight),
        "training_cells": len(training),
        "validation_cells": len(validation),
        "validation_repeats": 8,
        "median_validation_absolute_log_error": float(validation_frame["absolute_log_error"].median()),
        "worst_decile_validation_absolute_log_error": float(
            validation_frame["absolute_log_error"].quantile(0.90)
        ),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "trainable_parameters": count_trainable_parameters(model),
        "model_sha256": sha256(model_path),
        "oracle_sha256": oracle_hash,
        "history_sha256": sha256(history_path),
        "validation_sha256": sha256(validation_path),
        "calibration_opened": False,
        "locked_opened": False,
    }
    (OUTPUT / f"{tag}_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
