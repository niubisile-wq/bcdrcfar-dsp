"""Development-only Monte Carlo oracle targets for threshold distillation."""

from __future__ import annotations

import numpy as np

from .simulation import simulate_complex_clutter


def estimate_rms_oracle_threshold(
    *,
    scenario: str,
    severity: str,
    parameter_seed: int,
    sequence_seed: int,
    slow_time_length: int,
    reference_cells: int,
    target_pfa: float,
    decisions: int,
    batch_size: int = 512,
) -> dict[str, float | int]:
    """Estimate the unconditional CUT-RMS threshold for one parameter cell.

    Reference cells are generated in every acquisition because acquisition-level
    normalization and nonstationary texture make the CUT marginal depend on the
    full local scene. Only scalar RMS scores are retained.
    """

    if decisions < 100 or batch_size < 1:
        raise ValueError("oracle estimation requires at least 100 decisions")
    if not 0.0 < target_pfa < 0.1:
        raise ValueError("target_pfa must lie in (0, 0.1)")
    scores = np.empty(int(decisions), dtype=np.float32)
    normalized_scores = np.empty(int(decisions), dtype=np.float32)
    generated = 0
    chunk = 0
    while generated < decisions:
        size = min(int(batch_size), decisions - generated)
        acquisition = simulate_complex_clutter(
            scenario,
            severity,
            (size, int(reference_cells) + 1, int(slow_time_length)),
            seed=int(sequence_seed) + 104729 * chunk,
            parameter_seed=int(parameter_seed),
        )
        local_scores = np.sqrt(np.mean(np.abs(acquisition[:, 0]) ** 2, axis=1))
        reference_scale = np.median(np.abs(acquisition[:, 1:]), axis=(1, 2))
        scores[generated : generated + size] = local_scores
        normalized_scores[generated : generated + size] = local_scores / np.maximum(reference_scale, 1e-8)
        generated += size
        chunk += 1
    probability = 1.0 - float(target_pfa)
    threshold = float(np.quantile(scores, probability, method="higher"))
    normalized_threshold = float(np.quantile(normalized_scores, probability, method="higher"))
    empirical_pfa = float(np.mean(scores >= threshold))
    return {
        "oracle_threshold": threshold,
        "oracle_normalized_threshold": normalized_threshold,
        "oracle_empirical_pfa": empirical_pfa,
        "oracle_decisions": int(decisions),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_max": float(np.max(scores)),
        "normalized_score_mean": float(np.mean(normalized_scores)),
        "normalized_score_std": float(np.std(normalized_scores)),
        "normalized_score_max": float(np.max(normalized_scores)),
    }


__all__ = ["estimate_rms_oracle_threshold"]
