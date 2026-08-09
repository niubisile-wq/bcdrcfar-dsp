"""Train the Stage-B target-sensitive score with the Stage-A threshold frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
OUTPUT = ROOT / "results" / "bcdrcfar_stage_b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-model", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--h0-identity-weight", type=float, default=5.0)
    parser.add_argument("--h0-upper-tail-weight", type=float, default=10.0)
    parser.add_argument(
        "--h0-scenario-spread-weight",
        type=float,
        default=0.0,
        help="Optional dispersion penalty on H0 log-multiplier means across synthetic scenarios.",
    )
    parser.add_argument(
        "--h0-consistency-weight",
        type=float,
        default=0.0,
        help="Optional penalty that keeps two H0 draws for the same selected cells aligned.",
    )
    parser.add_argument(
        "--h0-worst-group-weight",
        type=float,
        default=0.0,
        help="Optional worst-group H0 penalty over scenario/severity strata using factor-2 exceedance.",
    )
    parser.add_argument(
        "--h0-group-temperature",
        type=float,
        default=0.2,
        help="Temperature for the soft worst-group aggregation on H0 strata.",
    )
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument(
        "--scr-db-values",
        type=float,
        nargs="+",
        default=[-10.0, -8.0, -6.0, -4.0, -2.0, 0.0],
        help="H1 curriculum SCR values used during Stage-B development training.",
    )
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--calibration-decisions", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=2026080721)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def base_normalized_score(cut_iq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    magnitude = torch.sqrt(cut_iq[..., 0].square() + cut_iq[..., 1].square()).clamp_min(1e-12)
    return torch.sqrt(torch.mean(magnitude.square(), dim=1)) / scale.clamp_min(1e-12)


def worst_group_exceedance_loss(
    log_multiplier: torch.Tensor,
    scenarios: tuple[str, ...],
    severities: tuple[str, ...],
    *,
    exceedance_threshold: float,
    temperature: float,
) -> torch.Tensor:
    group_scores: list[torch.Tensor] = []
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(zip(scenarios, severities, strict=True)):
        grouped.setdefault((str(key[0]), str(key[1])), []).append(index)
    for indices in grouped.values():
        local = torch.as_tensor(indices, device=log_multiplier.device)
        excess = F.relu(log_multiplier[local] - float(exceedance_threshold))
        if int(excess.numel()) > 0:
            top_count = max(1, int(round(0.25 * int(excess.numel()))))
            group_scores.append(torch.topk(excess, top_count).values.mean())
    if not group_scores:
        return log_multiplier.sum() * 0.0
    stacked = torch.stack(group_scores)
    temp = max(float(temperature), 1e-3)
    worst_count = max(1, int(round(0.2 * len(group_scores))))
    return torch.topk(stacked, worst_count).values.mean()


def main() -> None:
    args = parse_args()
    if int(args.batch_size) < 8:
        raise ValueError("batch-size must be at least eight")
    protocol = load_protocol(CONFIG)
    manifest = pd.read_csv(MANIFEST)
    development = manifest[manifest["split"] == "development"].copy()
    training = development[
        development["development_fold"] != int(args.validation_fold)
    ].reset_index(drop=True)
    validation = development[
        development["development_fold"] == int(args.validation_fold)
    ].reset_index(drop=True)
    if len(training) != 420 or len(validation) != 84:
        raise RuntimeError("Stage-B fold isolation failed")
    checkpoint = torch.load(args.stage_a_model, map_location="cpu", weights_only=True)
    if bool(checkpoint.get("learn_score", True)):
        raise RuntimeError("Stage B must start from a fixed-score Stage-A checkpoint")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    model = BCDRCFAR(
        hidden_channels=int(protocol["method"]["hidden_channels"]),
        distribution_pool_bins=int(protocol["method"]["distribution_pool_bins"]),
        dilations=tuple(int(value) for value in protocol["method"]["dilations"]),
        maximum_score_multiplier=float(protocol["method"]["maximum_score_multiplier"]),
        maximum_threshold_multiplier=float(protocol["method"]["maximum_threshold_multiplier"]),
        learn_score=True,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.cut_encoder,
        model.cut_condition,
        model.score_residual,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=1e-4,
    )
    target_pfa = 0.01
    length = 128
    reference_cells = 24
    scr_values = [float(value) for value in args.scr_db_values]
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()
    for step in range(int(args.steps)):
        selection_rng = np.random.default_rng(int(args.seed) + 65537 * step)
        selected_indices = selection_rng.integers(0, len(training), size=int(args.batch_size))
        selected_rows = training.iloc[selected_indices].reset_index(drop=True)
        common = dict(
            manifest=training,
            batch_size=int(args.batch_size),
            slow_time_length=length,
            reference_cells=reference_cells,
            target_pfa_values=[target_pfa],
            selected_indices=selected_indices,
            device=device,
        )
        h0 = make_synthetic_batch(
            **common,
            scr_db_values=[0.0],
            target_probability=0.0,
            seed=int(args.seed) + 1009 * step,
        )
        h0_consistency = None
        if float(args.h0_consistency_weight) > 0.0:
            h0_consistency = make_synthetic_batch(
                **common,
                scr_db_values=[0.0],
                target_probability=0.0,
                seed=int(args.seed) + 1009 * step + 777777,
            )
        h1 = make_synthetic_batch(
            **common,
            scr_db_values=scr_values,
            target_probability=1.0,
            seed=int(args.seed) + 1009 * step,
        )
        optimizer.zero_grad(set_to_none=True)
        out0 = model(h0.cut_iq, h0.reference_iq, h0.target_pfa, h0.reference_mask)
        out1 = model(h1.cut_iq, h1.reference_iq, h1.target_pfa, h1.reference_mask)
        base0 = base_normalized_score(h0.cut_iq, out0["scale"])
        base1 = base_normalized_score(h1.cut_iq, out1["scale"])
        log_multiplier0 = torch.log(out0["normalized_score"].clamp_min(1e-8)) - torch.log(
            base0.clamp_min(1e-8)
        )
        log_multiplier1 = torch.log(out1["normalized_score"].clamp_min(1e-8)) - torch.log(
            base1.clamp_min(1e-8)
        )
        margin1 = torch.log(out1["normalized_score"].clamp_min(1e-8)) - torch.log(
            out1["normalized_threshold"].clamp_min(1e-8)
        )
        temperature = float(args.temperature)
        detection_loss = F.softplus(-margin1 / temperature).mean()
        ranking_loss = F.softplus(-(log_multiplier1 - log_multiplier0) / temperature).mean()
        identity_loss = F.smooth_l1_loss(log_multiplier0, torch.zeros_like(log_multiplier0))
        tail_count = max(1, int(round(0.05 * len(log_multiplier0))))
        upper_tail_loss = torch.topk(F.relu(log_multiplier0).square(), tail_count).values.mean()
        scenario_spread_loss = torch.zeros((), device=device)
        if float(args.h0_scenario_spread_weight) > 0.0:
            scenario_means = []
            for scenario in sorted(set(h0.scenarios)):
                mask = torch.tensor([name == scenario for name in h0.scenarios], device=device)
                if int(mask.sum()) > 0:
                    scenario_means.append(log_multiplier0[mask].mean())
            if len(scenario_means) > 1:
                scenario_means_t = torch.stack(scenario_means)
                scenario_spread_loss = torch.mean((scenario_means_t - scenario_means_t.mean()).square())
        worst_group_loss = torch.zeros((), device=device)
        if float(args.h0_worst_group_weight) > 0.0:
            worst_group_loss = worst_group_exceedance_loss(
                log_multiplier0,
                h0.scenarios,
                tuple(str(value) for value in selected_rows["severity"]),
                exceedance_threshold=math.log(2.0),
                temperature=float(args.h0_group_temperature),
            )
        consistency_loss = torch.zeros((), device=device)
        if h0_consistency is not None:
            out0b = model(
                h0_consistency.cut_iq,
                h0_consistency.reference_iq,
                h0_consistency.target_pfa,
                h0_consistency.reference_mask,
            )
            base0b = base_normalized_score(h0_consistency.cut_iq, out0b["scale"])
            log_multiplier0b = torch.log(out0b["normalized_score"].clamp_min(1e-8)) - torch.log(
                base0b.clamp_min(1e-8)
            )
            consistency_loss = F.mse_loss(log_multiplier0, log_multiplier0b)
        loss = (
            detection_loss
            + float(args.ranking_weight) * ranking_loss
            + float(args.h0_identity_weight) * identity_loss
            + float(args.h0_upper_tail_weight) * upper_tail_loss
            + float(args.h0_scenario_spread_weight) * scenario_spread_loss
            + float(args.h0_worst_group_weight) * worst_group_loss
            + float(args.h0_consistency_weight) * consistency_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "detection_loss": float(detection_loss.detach().cpu()),
                "ranking_loss": float(ranking_loss.detach().cpu()),
                "h0_identity_loss": float(identity_loss.detach().cpu()),
                "h0_upper_tail_loss": float(upper_tail_loss.detach().cpu()),
                "h0_scenario_spread_loss": float(scenario_spread_loss.detach().cpu()),
                "h0_worst_group_loss": float(worst_group_loss.detach().cpu()),
                "h0_consistency_loss": float(consistency_loss.detach().cpu()),
                "mean_h0_log_multiplier": float(log_multiplier0.detach().mean().cpu()),
                "mean_h1_log_multiplier": float(log_multiplier1.detach().mean().cpu()),
            }
        )

    # Estimate one engineering multiplier using training-fold H0 only.  This is
    # not the frozen W1d calibration split and never reads validation outcomes.
    model.eval()
    ratios: list[np.ndarray] = []
    generated = 0
    with torch.no_grad():
        while generated < int(args.calibration_decisions):
            size = min(int(args.batch_size), int(args.calibration_decisions) - generated)
            batch = make_synthetic_batch(
                training,
                batch_size=size,
                slow_time_length=length,
                reference_cells=reference_cells,
                target_pfa_values=[target_pfa],
                scr_db_values=[0.0],
                target_probability=0.0,
                seed=int(args.seed) + 50_000_003 + generated,
                device=device,
            )
            output = model(batch.cut_iq, batch.reference_iq, batch.target_pfa, batch.reference_mask)
            ratios.append(
                (
                    output["normalized_score"] / output["normalized_threshold"].clamp_min(1e-8)
                ).cpu().numpy()
            )
            generated += size
    ratio = np.concatenate(ratios)
    development_multiplier = float(np.quantile(ratio, 1.0 - target_pfa, method="higher"))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stage_a_hash = sha256(args.stage_a_model)
    tag = (
        f"stageb_spectralgate_v1_seed{args.seed}_fold{args.validation_fold}_from{stage_a_hash[:8]}_"
        f"steps{args.steps}_h0{args.h0_identity_weight:g}_tail{args.h0_upper_tail_weight:g}"
        f"_wg{float(args.h0_worst_group_weight):g}"
    )
    model_path = OUTPUT / f"{tag}_model.pt"
    history_path = OUTPUT / f"{tag}_history.csv"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "protocol_version": protocol["protocol_version"],
            "learn_score": True,
            "training_mode": "stage_b_target_score_frozen_stage_a_threshold",
            "score_feature_schema": "conv_plus_reference_standardized_spectral_crest_phase_coherence_v1",
            "validation_fold": int(args.validation_fold),
            "seed": int(args.seed),
            "stage_a_model_sha256": stage_a_hash,
            "development_threshold_multiplier": development_multiplier,
        },
        model_path,
    )
    pd.DataFrame(history).to_csv(history_path, index=False)
    payload = {
        "status": "BCDRCFAR_STAGE_B_DEVELOPMENT_TRAINING_COMPLETE",
        "claim_status": "DEVELOPMENT_ONLY_NOT_CALIBRATION_OR_LOCKED",
        "score_feature_schema": "conv_plus_reference_standardized_spectral_crest_phase_coherence_v1",
        "validation_fold": int(args.validation_fold),
        "training_cells": len(training),
        "held_out_development_cells_not_read_for_training_or_multiplier": len(validation),
        "steps": int(args.steps),
        "scr_db_values": scr_values,
        "seed": int(args.seed),
        "trainable_parameters": count_trainable_parameters(model),
        "development_threshold_multiplier": development_multiplier,
        "multiplier_source_h0_decisions": int(args.calibration_decisions),
        "model_sha256": sha256(model_path),
        "stage_a_model_sha256": stage_a_hash,
        "history_sha256": sha256(history_path),
        "elapsed_seconds": time.perf_counter() - started,
        "w1d_calibration_opened": False,
        "w1d_locked_opened": False,
    }
    (OUTPUT / f"{tag}_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
