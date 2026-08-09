"""Frozen protocol validation and leakage-resistant W1d manifest construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REQUIRED_SCENARIOS = {
    "candidate_family",
    "gamma_shape_shift",
    "g0_inverse_gamma",
    "correlated",
    "contaminated",
    "mixture",
    "state_switching",
}


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != "DEVELOPMENT_ONLY_BEFORE_CALIBRATION_OR_LOCKED":
        raise ValueError("BC-DRCFAR protocol must begin development-only")
    if protocol.get("claim_boundary", {}).get("llm_in_scope") is not False:
        raise ValueError("LLM experiments are outside the BC-DRCFAR paper")
    w1d = protocol["w1d"]
    if set(w1d["scenarios"]) != REQUIRED_SCENARIOS:
        raise ValueError("W1d scenario set is not frozen as required")
    per_stratum = int(w1d["cells_per_scenario_severity"])
    split_total = sum(int(value) for value in w1d["split_counts_per_stratum"].values())
    if per_stratum != split_total:
        raise ValueError("W1d split counts must sum to cells_per_scenario_severity")
    if sorted(w1d["slow_time_lengths"]) != [128, 256, 512]:
        raise ValueError("W1d slow-time lengths must remain 128/256/512")
    pfas = sorted(float(value) for value in w1d["target_pfa"])
    if pfas != [0.001, 0.003, 0.01] or float(w1d["primary_target_pfa"]) != 0.001:
        raise ValueError("target-Pfa grid or primary endpoint changed")
    method = protocol["method"]
    if not method.get("scale_equivariant") or not method.get("pfa_monotone_by_construction"):
        raise ValueError("scale equivariance and Pfa monotonicity are required")
    if int(method["maximum_parameters"]) > 500000:
        raise ValueError("parameter budget exceeds the frozen deployment limit")
    if int(protocol["success_gates"]["minimum_external_sessions"]) < 8:
        raise ValueError("external confirmation requires at least eight sessions")


def _derive_seed(master_seed: int, *parts: object) -> int:
    payload = "|".join([str(master_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def build_w1d_manifest(protocol: Mapping[str, Any]) -> pd.DataFrame:
    validate_protocol(protocol)
    w1d = protocol["w1d"]
    master_seed = int(protocol["master_seed"])
    rows: list[dict[str, Any]] = []
    split_template = [
        split
        for split, count in w1d["split_counts_per_stratum"].items()
        for _ in range(int(count))
    ]
    for scenario in w1d["scenarios"]:
        for severity_index, severity in enumerate(w1d["severity_levels"]):
            rng = np.random.default_rng(_derive_seed(master_seed, "split", scenario, severity))
            splits = np.asarray(split_template, dtype="U16")[rng.permutation(len(split_template))]
            for cell_index, split in enumerate(splits):
                cell_id = f"w1d-{scenario}-s{severity_index}-{cell_index:02d}"
                parameter_seed = _derive_seed(master_seed, "parameters", scenario, severity, cell_index)
                sequence_seed = _derive_seed(master_seed, "sequence", scenario, severity, cell_index)
                target_seed = _derive_seed(master_seed, "target", scenario, severity, cell_index)
                cell_spec = {
                    "cell_id": cell_id,
                    "scenario": scenario,
                    "severity": severity,
                    "severity_index": severity_index,
                    "split": str(split),
                    "parameter_seed": parameter_seed,
                    "sequence_seed": sequence_seed,
                    "target_seed": target_seed,
                }
                rows.append({**cell_spec, "cell_spec_json": json.dumps(cell_spec, sort_keys=True, separators=(",", ":"))})
    frame = pd.DataFrame(rows).sort_values(["scenario", "severity_index", "cell_id"]).reset_index(drop=True)
    frame["development_fold"] = -1
    development = frame[frame["split"] == "development"]
    for (scenario, severity), group in development.groupby(["scenario", "severity"], sort=True):
        ordered = group.sort_values("cell_id")
        fold_rng = np.random.default_rng(_derive_seed(master_seed, "development-fold", scenario, severity))
        folds = np.resize(np.arange(6, dtype=int), len(ordered))[fold_rng.permutation(len(ordered))]
        frame.loc[ordered.index, "development_fold"] = folds
    expected = len(w1d["scenarios"]) * len(w1d["severity_levels"]) * int(w1d["cells_per_scenario_severity"])
    if len(frame) != expected or frame["cell_id"].nunique() != expected:
        raise RuntimeError("W1d manifest construction failed its size/uniqueness invariant")
    counts = frame.groupby(["scenario", "severity", "split"]).size()
    for scenario in w1d["scenarios"]:
        for severity in w1d["severity_levels"]:
            for split, count in w1d["split_counts_per_stratum"].items():
                if int(counts.loc[(scenario, severity, split)]) != int(count):
                    raise RuntimeError(f"unbalanced W1d stratum: {scenario}/{severity}/{split}")
            local_folds = frame[(frame["scenario"] == scenario) & (frame["severity"] == severity) & (frame["split"] == "development")]["development_fold"].value_counts()
            if set(local_folds.index) != set(range(6)) or not (local_folds == 3).all():
                raise RuntimeError(f"unbalanced development folds: {scenario}/{severity}")
    return frame


__all__ = ["build_w1d_manifest", "load_protocol", "validate_protocol"]
