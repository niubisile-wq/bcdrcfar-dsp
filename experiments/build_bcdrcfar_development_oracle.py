from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bcdrcfar.oracle import estimate_rms_oracle_threshold  # noqa: E402


MANIFEST = ROOT / "data" / "manifests" / "bcdrcfar_w1d_cells.csv"
OUTPUT = ROOT / "results" / "bcdrcfar_development_oracle"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--reference-cells", type=int, default=8)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--max-cells", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(MANIFEST)
    cells = manifest[manifest["split"] == "development"].sort_values("cell_id")
    if args.max_cells is not None:
        cells = cells.head(int(args.max_cells))
    if cells.empty or set(cells["split"]) != {"development"}:
        raise RuntimeError("oracle construction may only use development cells")
    rows = []
    started = time.perf_counter()
    for _, cell in cells.iterrows():
        result = estimate_rms_oracle_threshold(
            scenario=str(cell["scenario"]),
            severity=str(cell["severity"]),
            parameter_seed=int(cell["parameter_seed"]),
            sequence_seed=int(cell["sequence_seed"]) + 7_000_001,
            slow_time_length=int(args.length),
            reference_cells=int(args.reference_cells),
            target_pfa=float(args.target_pfa),
            decisions=int(args.decisions),
            batch_size=int(args.batch_size),
        )
        rows.append(
            {
                "cell_id": str(cell["cell_id"]),
                "scenario": str(cell["scenario"]),
                "severity": str(cell["severity"]),
                "development_fold": int(cell["development_fold"]),
                "parameter_seed": int(cell["parameter_seed"]),
                "oracle_sequence_seed": int(cell["sequence_seed"]) + 7_000_001,
                "slow_time_length": int(args.length),
                "reference_cells": int(args.reference_cells),
                "target_pfa": float(args.target_pfa),
                **result,
            }
        )
    frame = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = f"L{args.length}_K{args.reference_cells}_pfa{args.target_pfa:g}_n{args.decisions}_scaleeq"
    table_path = OUTPUT / f"{tag}.csv"
    frame.to_csv(table_path, index=False)
    payload = {
        "status": "BCDRCFAR_DEVELOPMENT_ORACLE_BUILT",
        "claim_status": "DEVELOPMENT_ONLY_TEACHER_LABELS",
        "cells": int(frame["cell_id"].nunique()),
        "decisions_per_cell": int(args.decisions),
        "total_decisions": int(len(frame) * int(args.decisions)),
        "slow_time_length": int(args.length),
        "reference_cells": int(args.reference_cells),
        "target_pfa": float(args.target_pfa),
        "teacher_statistic": "CUT_RMS_over_reference_median_magnitude",
        "elapsed_seconds": time.perf_counter() - started,
        "manifest_sha256": sha256(MANIFEST),
        "table_sha256": sha256(table_path),
        "calibration_opened": False,
        "locked_opened": False,
    }
    (OUTPUT / f"{tag}_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
