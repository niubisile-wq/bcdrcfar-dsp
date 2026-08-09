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

from src.bcdrcfar.model import BCDRCFAR  # noqa: E402
from src.bcdrcfar.protocol import load_protocol  # noqa: E402
from src.bcdrcfar.simulation import simulate_complex_clutter  # noqa: E402


def _iq(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack([values.real, values.imag], axis=-1).astype(np.float32), device=device)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=5)
    parser.add_argument("--decisions-per-cell", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--reference-cells", type=int, default=8)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = ROOT / "configs" / "bcdrcfar_protocol.json"
    manifest_path = ROOT / "data" / "manifests" / "bcdrcfar_w1d_cells.csv"
    protocol = load_protocol(protocol_path)
    manifest = pd.read_csv(manifest_path)
    cells = manifest[
        (manifest["split"] == "development")
        & (manifest["development_fold"] != int(args.validation_fold))
    ].sort_values("cell_id")
    if len(cells) != 420 or set(cells["split"]) != {"development"}:
        raise RuntimeError("multiplier fitting must use exactly five development folds")
    device = torch.device(args.device)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = BCDRCFAR(
        hidden_channels=int(protocol["method"]["hidden_channels"]),
        distribution_pool_bins=int(protocol["method"]["distribution_pool_bins"]),
        dilations=tuple(int(value) for value in protocol["method"]["dilations"]),
        maximum_score_multiplier=float(protocol["method"]["maximum_score_multiplier"]),
        maximum_threshold_multiplier=float(protocol["method"]["maximum_threshold_multiplier"]),
        learn_score=bool(checkpoint.get("learn_score", True)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    ratios: list[np.ndarray] = []
    with torch.no_grad():
        for _, cell in cells.iterrows():
            remaining = int(args.decisions_per_cell)
            offset = 0
            while remaining:
                size = min(int(args.batch_size), remaining)
                acquisition = simulate_complex_clutter(
                    str(cell["scenario"]),
                    str(cell["severity"]),
                    (size, int(args.reference_cells) + 1, int(args.length)),
                    seed=int(cell["sequence_seed"]) + 104729 * offset,
                    parameter_seed=int(cell["parameter_seed"]),
                )
                cut, reference = acquisition[:, 0], acquisition[:, 1:]
                output = model(
                    _iq(cut, device),
                    _iq(reference, device),
                    torch.full((size,), float(args.target_pfa), device=device),
                    torch.ones((size, int(args.reference_cells)), dtype=torch.bool, device=device),
                )
                ratio = output["normalized_score"] / output["normalized_threshold"].clamp_min(1e-8)
                ratios.append(ratio.cpu().numpy())
                remaining -= size
                offset += 1
    values = np.concatenate(ratios)
    multiplier = float(np.quantile(values, 1.0 - float(args.target_pfa), method="higher"))
    output_dir = ROOT / "results" / "bcdrcfar_development_multiplier"
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.stem.replace('_model', '')}_fold{args.validation_fold}_pfa{args.target_pfa:g}"
    payload = {
        "status": "BCDRCFAR_DEVELOPMENT_MULTIPLIER_FITTED",
        "claim_status": "DEVELOPMENT_ONLY",
        "fit_cells": int(cells["cell_id"].nunique()),
        "fit_decisions": int(values.size),
        "held_out_development_fold": int(args.validation_fold),
        "target_pfa": float(args.target_pfa),
        "threshold_multiplier": multiplier,
        "empirical_fit_pfa": float(np.mean(values >= multiplier)),
        "model_sha256": sha256(args.model),
        "manifest_sha256": sha256(manifest_path),
        "calibration_opened": False,
        "locked_opened": False,
    }
    path = output_dir / f"{tag}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
