"""Deterministic on-the-fly development batches for BC-DRCFAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .simulation import inject_swerling_target, simulate_complex_clutter


@dataclass(frozen=True)
class SyntheticBatch:
    cut_iq: Tensor
    reference_iq: Tensor
    reference_mask: Tensor
    target_pfa: Tensor
    label: Tensor
    scr_db: Tensor
    cell_ids: tuple[str, ...]
    scenarios: tuple[str, ...]


def _iq(values: np.ndarray) -> np.ndarray:
    return np.stack([values.real, values.imag], axis=-1).astype(np.float32)


def make_synthetic_batch(
    manifest: pd.DataFrame,
    *,
    batch_size: int,
    slow_time_length: int,
    reference_cells: int,
    target_pfa_values: Sequence[float],
    scr_db_values: Sequence[float],
    target_probability: float,
    seed: int,
    device: str | torch.device = "cpu",
    selected_indices: Sequence[int] | None = None,
) -> SyntheticBatch:
    if manifest.empty or set(manifest["split"]) != {"development"}:
        raise ValueError("synthetic training batches may only use development cells")
    if slow_time_length not in {128, 256, 512} or reference_cells not in {8, 16, 24}:
        raise ValueError("batch dimensions are outside the frozen W1d grid")
    rng = np.random.default_rng(seed)
    if selected_indices is None:
        indices = rng.integers(0, len(manifest), size=batch_size)
    else:
        indices = np.asarray(selected_indices, dtype=int)
        if indices.shape != (batch_size,) or np.any((indices < 0) | (indices >= len(manifest))):
            raise ValueError("selected_indices must contain one valid manifest index per batch item")
    selected = manifest.iloc[indices]
    labels = rng.random(batch_size) < float(target_probability)
    pfas = rng.choice(np.asarray(target_pfa_values, dtype=np.float32), size=batch_size)
    scrs = rng.choice(np.asarray(scr_db_values, dtype=np.float32), size=batch_size)
    references = []
    cuts = []
    for local_index, (_, row) in enumerate(selected.iterrows()):
        local_seed = int(row["sequence_seed"]) ^ int(seed) ^ (local_index * 104729)
        # Draw CUT and references from one acquisition so they share the same
        # scene scale while retaining range-dependent texture.
        acquisition = simulate_complex_clutter(
            str(row["scenario"]),
            str(row["severity"]),
            (1, reference_cells + 1, slow_time_length),
            seed=local_seed,
            parameter_seed=int(row["parameter_seed"]) if "parameter_seed" in row else None,
        )[0]
        cut = acquisition[0]
        reference = acquisition[1:]
        if labels[local_index]:
            cut = inject_swerling_target(
                cut[None, :],
                reference[None, :, :],
                float(scrs[local_index]),
                swerling="I" if (local_seed % 2 == 0) else "III",
                seed=local_seed + 2,
            )[0]
        references.append(reference)
        cuts.append(cut)
    return SyntheticBatch(
        cut_iq=torch.as_tensor(_iq(np.stack(cuts)), device=device),
        reference_iq=torch.as_tensor(_iq(np.stack(references)), device=device),
        reference_mask=torch.ones((batch_size, reference_cells), dtype=torch.bool, device=device),
        target_pfa=torch.as_tensor(pfas, dtype=torch.float32, device=device),
        label=torch.as_tensor(labels.astype(np.float32), device=device),
        scr_db=torch.as_tensor(scrs, dtype=torch.float32, device=device),
        cell_ids=tuple(str(value) for value in selected["cell_id"]),
        scenarios=tuple(str(value) for value in selected["scenario"]),
    )


__all__ = ["SyntheticBatch", "make_synthetic_batch"]
