"""Constant-memory condition-level evaluation for BC-DRCFAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .baselines import BASELINE_NAMES, classical_cfar_outputs
from .simulation import inject_swerling_target, simulate_complex_clutter


@dataclass
class DecisionCounter:
    events: int = 0
    trials: int = 0

    def update(self, decisions: np.ndarray) -> None:
        values = np.asarray(decisions, dtype=bool)
        self.events += int(np.count_nonzero(values))
        self.trials += int(values.size)

    @property
    def rate(self) -> float:
        return float(self.events / self.trials) if self.trials else float("nan")


def _iq(values: np.ndarray, device: torch.device) -> torch.Tensor:
    channels = np.stack([values.real, values.imag], axis=-1).astype(np.float32)
    return torch.as_tensor(channels, device=device)


def evaluate_synthetic_condition(
    model: nn.Module,
    *,
    scenario: str,
    severity: str,
    sequence_seed: int,
    parameter_seed: int | None = None,
    slow_time_length: int,
    reference_cells: int,
    target_pfa: float,
    decisions: int,
    batch_size: int = 64,
    scr_db: float | None = None,
    swerling: str = "I",
    model_threshold_multiplier: float = 1.0,
    device: str | torch.device = "cpu",
) -> dict[str, DecisionCounter]:
    """Evaluate one frozen parameter cell without retaining Monte Carlo IQ."""

    if decisions < 1 or batch_size < 1:
        raise ValueError("decisions and batch_size must be positive")
    torch_device = torch.device(device)
    counters = {name: DecisionCounter() for name in ("bcdrcfar", *BASELINE_NAMES)}
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
                parameter_seed=parameter_seed,
            )
            cut = clutter[:, 0]
            reference = clutter[:, 1:]
            if scr_db is not None:
                cut = inject_swerling_target(
                    cut,
                    reference,
                    float(scr_db),
                    swerling=swerling,
                    seed=int(sequence_seed) + 1_000_003 + generated,
                )
            alpha = np.full(size, float(target_pfa), dtype=np.float32)
            output = model(
                _iq(cut, torch_device),
                _iq(reference, torch_device),
                torch.as_tensor(alpha, device=torch_device),
                torch.ones((size, reference_cells), dtype=torch.bool, device=torch_device),
            )
            model_decision = output["normalized_score"] >= (
                output["normalized_threshold"] * float(model_threshold_multiplier)
            )
            counters["bcdrcfar"].update(model_decision.cpu().numpy())
            for name, result in classical_cfar_outputs(cut, reference, alpha).items():
                counters[name].update(result["decision"])
            generated += size
    if any(counter.trials != decisions for counter in counters.values()):
        raise RuntimeError("streaming evaluation did not complete every decision")
    return counters


def condition_rows(
    counters: dict[str, DecisionCounter],
    *,
    cell_id: str,
    scenario: str,
    severity: str,
    target_pfa: float,
    endpoint: str,
    scr_db: float | None = None,
) -> Iterable[dict[str, object]]:
    for method, counter in counters.items():
        row: dict[str, object] = {
            "cell_id": cell_id,
            "scenario": scenario,
            "severity": severity,
            "method": method,
            "endpoint": endpoint,
            "target_pfa": float(target_pfa),
            "events": counter.events,
            "trials": counter.trials,
            "rate": counter.rate,
        }
        if endpoint == "pfa":
            smoothed = (counter.events + 0.5) / (counter.trials + 1.0)
            row["absolute_log10_pfa_error"] = abs(float(np.log10(smoothed / target_pfa)))
            row["factor2_violation"] = bool(smoothed < 0.5 * target_pfa or smoothed > 2.0 * target_pfa)
        if scr_db is not None:
            row["scr_db"] = float(scr_db)
        yield row


__all__ = ["DecisionCounter", "condition_rows", "evaluate_synthetic_condition"]
