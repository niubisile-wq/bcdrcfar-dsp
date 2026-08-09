"""Fair scale-equivariant classical baselines for slow-time integrated CFAR."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.stats import f, gamma


BASELINE_NAMES = ("ca_cfar", "go_cfar", "os_cfar", "trimmed_mean_cfar")


def _validate(cut_iq: np.ndarray, reference_iq: np.ndarray, target_pfa: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cut = np.asarray(cut_iq)
    reference = np.asarray(reference_iq)
    alpha = np.asarray(target_pfa, dtype=float).reshape(-1)
    if cut.ndim != 2 or reference.ndim != 3:
        raise ValueError("cut_iq and reference_iq must be complex BxL and BxKxL arrays")
    if cut.shape[0] != reference.shape[0] or cut.shape[1] != reference.shape[2] or len(alpha) != len(cut):
        raise ValueError("CUT, reference, and target-Pfa dimensions do not match")
    if reference.shape[1] < 4 or np.any((alpha <= 0.0) | (alpha >= 0.1)):
        raise ValueError("at least four reference cells and target Pfa in (0,0.1) are required")
    return cut, reference, alpha


def classical_cfar_outputs(
    cut_iq: np.ndarray,
    reference_iq: np.ndarray,
    target_pfa: np.ndarray,
) -> Mapping[str, dict[str, np.ndarray]]:
    """Apply four baselines to a common slow-time RMS detection statistic."""

    cut, reference, alpha = _validate(cut_iq, reference_iq, target_pfa)
    length = cut.shape[1]
    cells = reference.shape[1]
    score_power = np.mean(np.abs(cut) ** 2, axis=1)
    cell_power = np.mean(np.abs(reference) ** 2, axis=2)

    ca_scale = np.mean(cell_power, axis=1)
    ca_ratio = f.isf(alpha, 2 * length, 2 * length * cells)

    midpoint = cells // 2
    go_scale = np.maximum(np.mean(cell_power[:, :midpoint], axis=1), np.mean(cell_power[:, midpoint:], axis=1))
    go_ratio = f.isf(alpha, 2 * length, 2 * length * max(2, midpoint))

    gamma_median = gamma.ppf(0.5, a=length, scale=1.0 / length)
    os_scale = np.median(cell_power, axis=1) / max(float(gamma_median), 1e-12)
    os_ratio = gamma.isf(alpha, a=length, scale=1.0 / length)

    ordered = np.sort(cell_power, axis=1)
    trim = max(1, int(np.floor(0.1 * cells)))
    retained = ordered[:, trim : cells - trim]
    tm_scale = np.mean(retained, axis=1)
    tm_ratio = f.isf(alpha, 2 * length, 2 * length * retained.shape[1])

    thresholds = {
        "ca_cfar": ca_scale * ca_ratio,
        "go_cfar": go_scale * go_ratio,
        "os_cfar": os_scale * os_ratio,
        "trimmed_mean_cfar": tm_scale * tm_ratio,
    }
    return {
        name: {
            "score": np.sqrt(score_power),
            "threshold": np.sqrt(np.maximum(threshold, 0.0)),
            "decision": score_power >= threshold,
        }
        for name, threshold in thresholds.items()
    }


__all__ = ["BASELINE_NAMES", "classical_cfar_outputs"]
