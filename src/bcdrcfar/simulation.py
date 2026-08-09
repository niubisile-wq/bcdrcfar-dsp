"""Complex sea-clutter and small-target simulation for the W1d panel."""

from __future__ import annotations

from typing import Sequence

import numpy as np


SCENARIOS = (
    "candidate_family",
    "gamma_shape_shift",
    "g0_inverse_gamma",
    "correlated",
    "contaminated",
    "mixture",
    "state_switching",
)
SEVERITY_INDEX = {"weak": 0, "moderate": 1, "strong": 2, "extreme": 3}


def _complex_gaussian(rng: np.random.Generator, shape: Sequence[int]) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def _unit_power(values: np.ndarray) -> np.ndarray:
    # Normalize acquisition-level power only.  Per-cell normalization would
    # erase the range-dependent texture and impulsive tails that CFAR must
    # actually handle.
    power = np.mean(np.abs(values) ** 2, axis=(-2, -1), keepdims=True)
    return values / np.sqrt(np.maximum(power, 1e-12))


def _piecewise_texture(
    rng: np.random.Generator,
    shape: tuple[int, int, int],
    sampler,
    *,
    block_length: int = 16,
) -> np.ndarray:
    blocks = int(np.ceil(shape[-1] / block_length))
    coarse = sampler(shape[:2] + (blocks,))
    return np.repeat(coarse, block_length, axis=-1)[..., : shape[-1]]


def simulate_complex_clutter(
    scenario: str,
    severity: str,
    shape: tuple[int, int, int],
    *,
    seed: int,
    parameter_seed: int | None = None,
) -> np.ndarray:
    """Return B x K x L complex clutter with unit acquisition-level power."""

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown clutter scenario: {scenario}")
    if severity not in SEVERITY_INDEX:
        raise ValueError(f"unknown severity: {severity}")
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("shape must be positive B x K x L")
    rng = np.random.default_rng(seed)
    parameter_rng = np.random.default_rng(seed if parameter_seed is None else parameter_seed)
    scale_jitter = float(parameter_rng.uniform(0.82, 1.18))
    probability_jitter = float(parameter_rng.uniform(0.80, 1.20))
    correlation_jitter = float(parameter_rng.uniform(-0.06, 0.06))
    level = SEVERITY_INDEX[severity]
    values = _complex_gaussian(rng, shape)

    if scenario == "candidate_family":
        family = level % 4
        if family == 1:
            amplitude = rng.weibull(1.8, shape)
            values = amplitude * np.exp(1j * np.angle(values))
        elif family == 2:
            texture = _piecewise_texture(rng, shape, lambda size: rng.lognormal(0.0, 0.35 * scale_jitter, size))
            values = values * np.sqrt(texture)
        elif family == 3:
            candidate_shape = 1.5 * scale_jitter
            texture = _piecewise_texture(rng, shape, lambda size: rng.gamma(candidate_shape, 1.0 / candidate_shape, size))
            values = values * np.sqrt(texture)
    elif scenario == "gamma_shape_shift":
        gamma_shape = [4.0, 2.0, 1.0, 0.55][level] * scale_jitter
        texture = _piecewise_texture(
            rng, shape, lambda size: rng.gamma(gamma_shape, 1.0 / gamma_shape, size)
        )
        values = values * np.sqrt(texture)
    elif scenario == "g0_inverse_gamma":
        inverse_shape = max(1.25, [8.0, 4.0, 2.5, 1.7][level] * scale_jitter)
        texture = _piecewise_texture(
            rng, shape, lambda size: 1.0 / rng.gamma(inverse_shape, 1.0 / inverse_shape, size)
        )
        values = values * np.sqrt(texture)
    elif scenario == "correlated":
        rho = float(np.clip([0.15, 0.35, 0.60, 0.82][level] + correlation_jitter, 0.02, 0.95))
        innovation = _complex_gaussian(rng, shape)
        values[..., 0] = innovation[..., 0]
        for index in range(1, shape[-1]):
            values[..., index] = rho * values[..., index - 1] + np.sqrt(1.0 - rho**2) * innovation[..., index]
    elif scenario == "contaminated":
        probability = min(0.08, [0.002, 0.005, 0.012, 0.025][level] * probability_jitter)
        multiplier = [5.0, 8.0, 12.0, 18.0][level] * scale_jitter
        spikes = rng.random(shape) < probability
        values = values * np.where(spikes, multiplier, 1.0)
    elif scenario == "mixture":
        probability = min(0.8, [0.10, 0.20, 0.35, 0.50][level] * probability_jitter)
        blocks = int(np.ceil(shape[-1] / 16))
        component = rng.random(shape[:2] + (blocks,)) < probability
        heavy = rng.lognormal(0.0, [0.45, 0.65, 0.85, 1.05][level] * scale_jitter, shape[:2] + (blocks,))
        component = np.repeat(component, 16, axis=-1)[..., : shape[-1]]
        heavy = np.repeat(heavy, 16, axis=-1)[..., : shape[-1]]
        values = values * np.sqrt(np.where(component, heavy, 1.0))
    elif scenario == "state_switching":
        ratio = [1.25, 1.6, 2.2, 3.0][level] * scale_jitter
        switch = rng.integers(max(1, shape[-1] // 4), max(2, 3 * shape[-1] // 4), size=shape[:2])
        time = np.arange(shape[-1])[None, None, :]
        high = time >= switch[..., None]
        values = values * np.where(high, ratio, 1.0)

    return _unit_power(values).astype(np.complex64)


def inject_swerling_target(
    clutter_cut: np.ndarray,
    reference: np.ndarray,
    scr_db: float | np.ndarray,
    *,
    swerling: str,
    seed: int,
) -> np.ndarray:
    """Inject a coherent moving target into B x L complex CUT windows."""

    cut = np.asarray(clutter_cut, dtype=np.complex64)
    ref = np.asarray(reference, dtype=np.complex64)
    if cut.ndim != 2 or ref.ndim != 3 or cut.shape[0] != ref.shape[0] or cut.shape[1] != ref.shape[2]:
        raise ValueError("CUT must be B x L and reference must be B x K x L")
    if swerling not in {"I", "III"}:
        raise ValueError("swerling must be I or III")
    rng = np.random.default_rng(seed)
    batch, length = cut.shape
    scr = np.broadcast_to(np.asarray(scr_db, dtype=float), (batch,))
    clutter_power = np.mean(np.abs(ref) ** 2, axis=(1, 2))
    target_power = clutter_power * 10.0 ** (scr / 10.0)
    if swerling == "I":
        fluctuation = rng.exponential(1.0, size=batch)
    else:
        fluctuation = rng.gamma(2.0, 0.5, size=batch)
    phase = rng.uniform(-np.pi, np.pi, size=batch)
    doppler = rng.uniform(-0.18, 0.18, size=batch)
    time = np.arange(length, dtype=float)[None, :]
    signal = np.sqrt(target_power * fluctuation)[:, None] * np.exp(1j * (phase[:, None] + 2.0 * np.pi * doppler[:, None] * time))
    return (cut + signal).astype(np.complex64)


__all__ = ["SCENARIOS", "inject_swerling_target", "simulate_complex_clutter"]
