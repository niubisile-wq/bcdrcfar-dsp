"""Losses and deterministic training utilities for BC-DRCFAR development."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


def bcdrcfar_loss(
    output: Mapping[str, Tensor],
    label: Tensor,
    target_pfa: Tensor,
    scenarios: Sequence[str],
    *,
    temperature: float = 100.0,
    threshold_only: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    score = output["normalized_score"].clamp_min(1e-8)
    threshold = output["normalized_threshold"].clamp_min(1e-8)
    log_ratio = torch.log(score) - torch.log(threshold)
    logits = float(temperature) * log_ratio
    detection = F.binary_cross_entropy_with_logits(logits, label)
    negative = label < 0.5
    positive = ~negative
    if negative.any():
        soft_false_alarm = torch.sigmoid(logits[negative])
        alpha = target_pfa[negative]
        far = torch.mean((torch.log(soft_false_alarm.clamp_min(1e-7)) - torch.log(alpha)) ** 2)
    else:
        far = logits.sum() * 0.0
    if positive.any():
        pd = F.softplus(-logits[positive]).mean()
    else:
        pd = logits.sum() * 0.0

    scenario_indices: dict[str, list[int]] = defaultdict(list)
    for index, scenario in enumerate(scenarios):
        if bool(negative[index]):
            scenario_indices[str(scenario)].append(index)
    group_losses = []
    quantile_losses = []
    for indices in scenario_indices.values():
        local = torch.as_tensor(indices, device=logits.device)
        local_probability = torch.sigmoid(logits[local]).mean().clamp_min(1e-7)
        local_alpha = torch.exp(torch.log(target_pfa[local]).mean())
        group_losses.append((torch.log(local_probability) - torch.log(local_alpha)) ** 2)
        quantile = torch.quantile(log_ratio[local], float((1.0 - local_alpha).detach().cpu()))
        quantile_losses.append(quantile.square())
    if group_losses:
        stacked = torch.stack(group_losses)
        worst_count = max(1, int(round(0.2 * len(stacked))))
        worst_group = torch.topk(stacked, worst_count).values.mean()
    else:
        worst_group = logits.sum() * 0.0
    if quantile_losses:
        quantile_alignment = torch.stack(quantile_losses).mean()
    else:
        quantile_alignment = logits.sum() * 0.0
    uncertainty = output["uncertainty"]
    uncertainty_target = torch.abs(label - torch.sigmoid(logits)).detach()
    uncertainty_loss = F.smooth_l1_loss(uncertainty, uncertainty_target)
    if threshold_only:
        total = quantile_alignment
    else:
        total = detection + 0.25 * far + 0.25 * worst_group + 2.0 * quantile_alignment + pd + 0.05 * uncertainty_loss
    return total, {
        "total": total.detach(),
        "detection": detection.detach(),
        "far": far.detach(),
        "worst_group": worst_group.detach(),
        "quantile_alignment": quantile_alignment.detach(),
        "pd": pd.detach(),
        "uncertainty": uncertainty_loss.detach(),
    }


__all__ = ["bcdrcfar_loss"]
