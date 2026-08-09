"""Fit a low-rank acquisition/polarization calibration head on IPIX development blocks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "block_ratios.csv"
DEFAULT_OUTPUT = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "lowrank_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--positive-weight", type=float, default=0.2)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026080807)
    return parser.parse_args()


@dataclass(frozen=True)
class Encoders:
    file_to_index: dict[str, int]
    pol_to_index: dict[str, int]
    index_to_file: list[str]
    index_to_pol: list[str]


def build_encoders(frame: pd.DataFrame) -> Encoders:
    files = sorted(frame["file_id"].astype(str).unique().tolist())
    pols = sorted(frame["polarization"].astype(str).unique().tolist())
    return Encoders(
        file_to_index={value: index for index, value in enumerate(files)},
        pol_to_index={value: index for index, value in enumerate(pols)},
        index_to_file=files,
        index_to_pol=pols,
    )


def summarize(frame: pd.DataFrame, multipliers: dict[tuple[str, str], float], target_pfa: float) -> dict[str, float]:
    data = frame.copy()
    data["multiplier"] = [
        multipliers[(str(file_id), str(pol))]
        for file_id, pol in zip(data["file_id"].astype(str), data["polarization"].astype(str), strict=True)
    ]
    data["decision"] = data["ratio_bcdrcfar"] >= data["multiplier"]
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


def fit_model(frame: pd.DataFrame, encoders: Encoders, *, args: argparse.Namespace) -> dict[tuple[str, str], float]:
    target_pfa = float(args.target_pfa)
    temperature = float(args.temperature)
    clutter = frame[frame["role"] == "clutter"].copy().reset_index(drop=True)
    positive = frame[frame["role"] == "primary"].copy().reset_index(drop=True)
    file_idx_clutter = torch.tensor([encoders.file_to_index[str(v)] for v in clutter["file_id"]], dtype=torch.long)
    pol_idx_clutter = torch.tensor([encoders.pol_to_index[str(v)] for v in clutter["polarization"]], dtype=torch.long)
    ratio_clutter = torch.tensor(np.log(clutter["ratio_bcdrcfar"].to_numpy(dtype=float)), dtype=torch.float32)
    file_idx_positive = torch.tensor([encoders.file_to_index[str(v)] for v in positive["file_id"]], dtype=torch.long)
    pol_idx_positive = torch.tensor([encoders.pol_to_index[str(v)] for v in positive["polarization"]], dtype=torch.long)
    ratio_positive = torch.tensor(np.log(positive["ratio_bcdrcfar"].to_numpy(dtype=float)), dtype=torch.float32)
    clutter_groups: dict[tuple[str, str], np.ndarray] = {}
    for (file_id, pol), group in clutter.groupby(["file_id", "polarization"], sort=True):
        clutter_groups[(str(file_id), str(pol))] = group.index.to_numpy(dtype=int)
    positive_groups: dict[tuple[str, str], np.ndarray] = {}
    for (file_id, pol), group in positive.groupby(["file_id", "polarization"], sort=True):
        positive_groups[(str(file_id), str(pol))] = group.index.to_numpy(dtype=int)

    n_files = len(encoders.index_to_file)
    n_pols = len(encoders.index_to_pol)
    rank = int(args.rank)
    global_log_multiplier = torch.nn.Parameter(torch.tensor(0.0))
    file_bias = torch.nn.Parameter(torch.zeros(n_files))
    pol_bias = torch.nn.Parameter(torch.zeros(n_pols))
    file_latent = torch.nn.Parameter(torch.zeros(n_files, rank))
    pol_latent = torch.nn.Parameter(torch.zeros(n_pols, rank))
    torch.nn.init.normal_(file_latent, mean=0.0, std=0.01)
    torch.nn.init.normal_(pol_latent, mean=0.0, std=0.01)
    optimizer = torch.optim.Adam(
        [global_log_multiplier, file_bias, pol_bias, file_latent, pol_latent],
        lr=float(args.learning_rate),
    )

    def log_multiplier(file_idx: torch.Tensor, pol_idx: torch.Tensor) -> torch.Tensor:
        interaction = (file_latent[file_idx] * pol_latent[pol_idx]).sum(dim=1)
        return global_log_multiplier + file_bias[file_idx] + pol_bias[pol_idx] + interaction

    groups = sorted(clutter_groups)

    for _ in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        clutter_loss = torch.zeros((), dtype=torch.float32)
        for file_id, pol in groups:
            idx = torch.as_tensor(clutter_groups[(file_id, pol)], dtype=torch.long)
            lm = log_multiplier(file_idx_clutter[idx], pol_idx_clutter[idx])
            pfa_hat = torch.sigmoid((ratio_clutter[idx] - lm) / temperature).mean().clamp_min(1e-6)
            clutter_loss = clutter_loss + (pfa_hat - torch.tensor(target_pfa)).square()
        clutter_loss = clutter_loss / max(1, len(groups))

        positive_loss = torch.zeros((), dtype=torch.float32)
        for file_id, pol in positive_groups:
            idx = torch.as_tensor(positive_groups[(file_id, pol)], dtype=torch.long)
            lm = log_multiplier(file_idx_positive[idx], pol_idx_positive[idx])
            miss_hat = torch.sigmoid((lm - ratio_positive[idx]) / temperature).mean()
            positive_loss = positive_loss + miss_hat
        positive_loss = positive_loss / max(1, len(positive_groups))

        reg = (
            file_bias.square().mean()
            + pol_bias.square().mean()
            + file_latent.square().mean()
            + pol_latent.square().mean()
            + global_log_multiplier.square()
        )
        loss = clutter_loss + float(args.positive_weight) * positive_loss + float(args.regularization) * reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        multipliers: dict[tuple[str, str], float] = {}
        for file_id in encoders.index_to_file:
            for pol in encoders.index_to_pol:
                fi = torch.tensor([encoders.file_to_index[file_id]])
                pi = torch.tensor([encoders.pol_to_index[pol]])
                multiplier = torch.exp(log_multiplier(fi, pi)).item()
                multipliers[(file_id, pol)] = float(multiplier)
        return multipliers


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    encoders = build_encoders(frame)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))

    rank = int(args.rank)
    candidate_positive_weights = [0.01, 0.02, 0.03, 0.05, 0.1]
    results: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for positive_weight in candidate_positive_weights:
        local_args = argparse.Namespace(**vars(args))
        local_args.positive_weight = positive_weight
        multipliers = fit_model(frame, encoders, args=local_args)
        summary = summarize(frame, multipliers, float(args.target_pfa))
        summary["positive_weight"] = float(positive_weight)
        summary["rank"] = rank
        results.append(summary)
        score = summary["macro_pfa"] + 2.0 * summary["macro_factor2"] - 0.5 * summary["macro_pd"]
        if best is None or score < float(best["score"]):
            best = {"score": float(score), "summary": summary, "multipliers": multipliers}

    assert best is not None
    result_frame = pd.DataFrame(results).sort_values(["macro_pfa", "macro_factor2", "macro_pd"], ascending=[True, True, False])
    result_frame.to_csv(output_dir / "grid_summary.csv", index=False)
    serializable = {
        f"{file_id}|{pol}": float(multiplier)
        for (file_id, pol), multiplier in best["multipliers"].items()
    }
    (output_dir / "best_multipliers.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "target_pfa": float(args.target_pfa),
                "positive_weight": float(best["summary"]["positive_weight"]),
                "summary": best["summary"],
                "multipliers": serializable,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(result_frame.to_string(index=False))
    print(json.dumps(best["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
