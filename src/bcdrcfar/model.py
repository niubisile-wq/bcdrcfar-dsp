"""Lightweight scale-equivariant BC-DRCFAR network."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class _ConvEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(input_channels, hidden_channels, kernel_size=1), nn.SiLU()]
        for dilation in dilations:
            layers.extend(
                [
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=hidden_channels,
                    ),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                    nn.SiLU(),
                ]
            )
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        encoded = self.network(values)
        return torch.cat([encoded.mean(dim=-1), encoded.amax(dim=-1)], dim=-1)


class BCDRCFAR(nn.Module):
    """Background-conditioned detector with monotone analytic Pfa anchors."""

    feature_schema = "taildep_v2"
    anchor_names = (
        "ca_power",
        "trimmed_power",
        "rayleigh_median",
        "weibull_log_moment",
        "upper_order_statistic",
        "robust_high_quantile",
    )
    tail_feature_width = 29

    def __init__(
        self,
        *,
        hidden_channels: int = 32,
        distribution_pool_bins: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        maximum_score_multiplier: float = 4.0,
        maximum_threshold_multiplier: float = 4.0,
        learn_score: bool = True,
    ) -> None:
        super().__init__()
        self.distribution_pool_bins = int(distribution_pool_bins)
        self.maximum_score_log = math.log(float(maximum_score_multiplier))
        self.maximum_threshold_log = math.log(float(maximum_threshold_multiplier))
        self.learn_score = bool(learn_score)
        self.distribution_encoder = _ConvEncoder(1, hidden_channels, (1, 2))
        self.reference_series_encoder = _ConvEncoder(2, hidden_channels, (1, 2))
        self.background_temporal_encoder = _ConvEncoder(3, hidden_channels, dilations)
        self.cut_encoder = _ConvEncoder(5, hidden_channels, dilations)
        background_width = hidden_channels * 4 + self.tail_feature_width
        cut_width = hidden_channels * 2
        self.background_fusion = nn.Sequential(
            nn.Linear(background_width, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.SiLU(),
        )
        self.cut_condition = nn.Sequential(
            nn.Linear(cut_width + hidden_channels * 2, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
        )
        self.score_residual = nn.Linear(hidden_channels, 1)
        self.anchor_logits = nn.Linear(hidden_channels * 2, len(self.anchor_names))
        self.threshold_residual = nn.Linear(hidden_channels * 2, 1)
        self.uncertainty_head = nn.Linear(hidden_channels, 1)
        self.reference_series_threshold_gate = nn.Linear(hidden_channels * 2, hidden_channels * 2)
        self._initialize_cfar_heads()

    def _initialize_cfar_heads(self) -> None:
        nn.init.zeros_(self.score_residual.weight)
        nn.init.zeros_(self.score_residual.bias)
        nn.init.zeros_(self.threshold_residual.weight)
        nn.init.zeros_(self.threshold_residual.bias)
        nn.init.zeros_(self.reference_series_threshold_gate.weight)
        nn.init.zeros_(self.reference_series_threshold_gate.bias)
        nn.init.zeros_(self.anchor_logits.weight)
        with torch.no_grad():
            self.anchor_logits.bias.copy_(torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0, -1.0]))

    @staticmethod
    def _validate_inputs(cut_iq: Tensor, reference_iq: Tensor, target_pfa: Tensor, reference_mask: Tensor | None) -> Tensor:
        if cut_iq.ndim != 3 or cut_iq.shape[-1] != 2:
            raise ValueError("cut_iq must have shape B x L x 2")
        if reference_iq.ndim != 4 or reference_iq.shape[-1] != 2:
            raise ValueError("reference_iq must have shape B x K x L x 2")
        if reference_iq.shape[0] != cut_iq.shape[0] or reference_iq.shape[2] != cut_iq.shape[1]:
            raise ValueError("CUT and reference dimensions do not match")
        if target_pfa.numel() != cut_iq.shape[0] or torch.any((target_pfa <= 0) | (target_pfa >= 1)):
            raise ValueError("target_pfa must have one value in (0,1) per batch item")
        if not torch.isfinite(cut_iq).all() or not torch.isfinite(reference_iq).all():
            raise ValueError("I/Q inputs must be finite")
        if reference_mask is None:
            reference_mask = torch.ones(reference_iq.shape[:2], dtype=torch.bool, device=reference_iq.device)
        if reference_mask.shape != reference_iq.shape[:2] or not reference_mask.any(dim=1).all():
            raise ValueError("reference_mask must be B x K with at least one true cell per item")
        return reference_mask.bool()

    @staticmethod
    def _magnitude(iq: Tensor) -> Tensor:
        return torch.sqrt(torch.clamp(iq.square().sum(dim=-1), min=1e-12))

    @staticmethod
    def _tail_features_from_rows(rows: Tensor, cell_rms: Tensor) -> Tensor:
        """Explicit scale-free tail descriptors that preserve rare extremes.

        Adaptive pooling is intentionally retained for the distribution encoder,
        while these statistics prevent the upper 0.1--1% of reference samples
        from being averaged away.  All values are normalized by the reference
        median before this function is called.
        """

        sorted_rows = torch.sort(rows, dim=1).values
        row_indices = torch.tensor(
            [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0],
            dtype=rows.dtype,
            device=rows.device,
        ).mul(rows.shape[1] - 1).round().long()
        log_quantiles = torch.log(sorted_rows.index_select(1, row_indices).clamp_min(1e-6))
        hill_features = []
        for fraction in (0.01, 0.02, 0.05):
            count = max(2, int(round(rows.shape[1] * fraction)))
            tail = sorted_rows[:, -count:]
            hill_features.append(torch.log(tail / tail[:, :1].clamp_min(1e-6)).mean(dim=1))
        tail_shape = torch.stack(hill_features, dim=1)
        extreme_ratios = torch.stack(
            [log_quantiles[:, 6] - log_quantiles[:, 4], log_quantiles[:, 4] - log_quantiles[:, 3]],
            dim=1,
        )
        log_rows = torch.log(rows.clamp_min(1e-6))
        log_mean = log_rows.mean(dim=1)
        centered = log_rows - log_mean[:, None]
        log_std = torch.sqrt(centered.square().mean(dim=1).clamp_min(1e-8))
        standardized = centered / log_std[:, None]
        moments = torch.stack(
            [log_mean, log_std, standardized.pow(3).mean(dim=1), standardized.pow(4).mean(dim=1)],
            dim=1,
        )
        exceedance = torch.stack([(rows > level).to(rows.dtype).mean(dim=1) for level in (1.5, 2.0, 3.0, 4.0)], dim=1)
        sorted_cell_rms = torch.sort(cell_rms, dim=1).values
        cell_indices = torch.tensor([0.50, 0.75, 0.90, 1.0], dtype=rows.dtype, device=rows.device).mul(
            cell_rms.shape[1] - 1
        ).round().long()
        log_cell_quantiles = torch.log(sorted_cell_rms.index_select(1, cell_indices).clamp_min(1e-6))
        cell_log_std = torch.log(cell_rms.clamp_min(1e-6)).std(dim=1, unbiased=False, keepdim=True)
        return torch.cat(
            [log_quantiles, moments, exceedance, log_cell_quantiles, cell_log_std, tail_shape, extreme_ratios],
            dim=1,
        )

    def _dependence_features(self, normalized_iq: Tensor, mask: Tensor) -> Tensor:
        magnitude = self._magnitude(normalized_iq)
        log_magnitude = torch.log(magnitude.clamp_min(1e-6))
        weight = mask.to(normalized_iq.dtype)
        denominator = weight.sum(dim=1).clamp_min(1.0)
        correlations = []
        for lag in (1, 4):
            left = log_magnitude[..., lag:]
            right = log_magnitude[..., :-lag]
            left = left - left.mean(dim=-1, keepdim=True)
            right = right - right.mean(dim=-1, keepdim=True)
            correlation = (left * right).mean(dim=-1) / torch.sqrt(
                left.square().mean(dim=-1).clamp_min(1e-8)
                * right.square().mean(dim=-1).clamp_min(1e-8)
            )
            correlations.append((correlation * weight).sum(dim=1) / denominator)
        current = normalized_iq[..., 1:, :]
        previous = normalized_iq[..., :-1, :]
        cross_real = (current * previous).sum(dim=-1).sum(dim=-1)
        cross_imag = (current[..., 1] * previous[..., 0] - current[..., 0] * previous[..., 1]).sum(dim=-1)
        coherence = torch.sqrt(cross_real.square() + cross_imag.square()) / torch.sqrt(
            current.square().sum(dim=(-1, -2)).clamp_min(1e-8)
            * previous.square().sum(dim=(-1, -2)).clamp_min(1e-8)
        )
        mean_coherence = (coherence * weight).sum(dim=1) / denominator
        return torch.stack([*correlations, mean_coherence], dim=1)

    def _scale_and_background(self, reference_iq: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        magnitude = self._magnitude(reference_iq)
        valid = mask.unsqueeze(-1).expand_as(magnitude)
        if bool(mask.all()):
            rows = magnitude.flatten(1)
            scale_tensor = torch.quantile(rows, 0.5, dim=1).clamp_min(1e-6)
            padded = torch.sort(torch.log(rows / scale_tensor[:, None]).clamp(-12.0, 12.0), dim=1).values
            normalized_rows = rows / scale_tensor[:, None]
            cell_rms = torch.sqrt(magnitude.square().mean(dim=2).clamp_min(1e-12)) / scale_tensor[:, None]
            tail_features = self._tail_features_from_rows(normalized_rows, cell_rms)
        else:
            scales = []
            sorted_rows = []
            feature_rows = []
            cell_rms_rows = []
            for index in range(len(magnitude)):
                row = magnitude[index][valid[index]]
                scale = torch.quantile(row, 0.5).clamp_min(1e-6)
                scales.append(scale)
                sorted_rows.append(torch.sort(torch.log(row / scale).clamp(-12.0, 12.0)).values)
                normalized_row = (row / scale).unsqueeze(0)
                local_cell_rms = torch.sqrt(magnitude[index][mask[index]].square().mean(dim=1).clamp_min(1e-12)).div(scale)
                cell_rms_row = torch.zeros(magnitude.shape[1], device=magnitude.device, dtype=magnitude.dtype)
                cell_rms_row[mask[index]] = local_cell_rms
                cell_rms_rows.append(cell_rms_row)
                local_cell_rms = local_cell_rms.unsqueeze(0)
                feature_rows.append(self._tail_features_from_rows(normalized_row, local_cell_rms).squeeze(0))
            scale_tensor = torch.stack(scales)
            maximum = max(row.numel() for row in sorted_rows)
            padded = torch.stack([F.pad(row, (0, maximum - row.numel()), value=float(row[-1])) for row in sorted_rows])
            tail_features = torch.stack(feature_rows)
            cell_rms = torch.stack(cell_rms_rows)
        if bool(mask.all()):
            cell_rms = cell_rms
        distribution = F.adaptive_avg_pool1d(padded.unsqueeze(1), self.distribution_pool_bins)

        normalized = reference_iq / scale_tensor[:, None, None, None]
        dependence_features = self._dependence_features(normalized, mask)
        normalized_magnitude = self._magnitude(normalized)
        phase = torch.atan2(normalized[..., 1], normalized[..., 0])
        weight = mask[:, :, None].to(normalized.dtype)
        denominator = weight.sum(dim=1).clamp_min(1.0)
        log_amplitude = (torch.log(normalized_magnitude.clamp_min(1e-6)) * weight).sum(dim=1) / denominator
        mean_cos = (torch.cos(phase) * weight).sum(dim=1) / denominator
        mean_sin = (torch.sin(phase) * weight).sum(dim=1) / denominator
        temporal = torch.stack([log_amplitude, mean_cos, mean_sin], dim=1)
        series_input = torch.stack([torch.log(cell_rms.clamp_min(1e-6)), mask.to(cell_rms.dtype)], dim=1)
        series_features = self.reference_series_encoder(series_input)
        tail_features = torch.cat([tail_features, dependence_features], dim=1)
        return scale_tensor, distribution, temporal, tail_features, series_features

    def _cut_channels(self, cut_iq: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
        normalized = cut_iq / scale[:, None, None]
        magnitude = self._magnitude(normalized)
        phase = torch.atan2(normalized[..., 1], normalized[..., 0])
        delta = torch.cat([torch.zeros_like(phase[:, :1]), phase[:, 1:] - phase[:, :-1]], dim=1)
        channels = torch.stack(
            [
                normalized[..., 0],
                normalized[..., 1],
                torch.log(magnitude.clamp_min(1e-6)),
                torch.cos(delta),
                torch.sin(delta),
            ],
            dim=1,
        )
        base_score = torch.sqrt(torch.mean(magnitude.square(), dim=1).clamp_min(1e-12))
        return channels, base_score

    @staticmethod
    def _cut_detection_evidence(cut_iq: Tensor, reference_iq: Tensor, mask: Tensor) -> Tensor:
        """Background-standardized spectral and phase-coherence evidence.

        A coherent moving target produces a concentrated Doppler line and
        stable inter-pulse phase increments.  Standardizing both descriptors
        against the same-time reference cells keeps them scale free and avoids
        using an absolute, scene-specific cutoff.
        """

        cut = torch.complex(cut_iq[..., 0], cut_iq[..., 1])
        reference = torch.complex(reference_iq[..., 0], reference_iq[..., 1])

        def spectral_crest(values: Tensor) -> Tensor:
            power = torch.fft.fft(values, dim=-1).abs().square()
            return power.amax(dim=-1) / power.sum(dim=-1).clamp_min(1e-8)

        def phase_coherence(values: Tensor) -> Tensor:
            increment = values[..., 1:] * values[..., :-1].conj()
            unit = increment / increment.abs().clamp_min(1e-8)
            return unit.mean(dim=-1).abs()

        weight = mask.to(cut_iq.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        evidence = []
        for feature_index, (cut_feature, reference_feature) in enumerate((
            (spectral_crest(cut), spectral_crest(reference)),
            (phase_coherence(cut), phase_coherence(reference)),
        )):
            mean = (reference_feature * weight).sum(dim=1) / count
            variance = ((reference_feature - mean[:, None]).square() * weight).sum(dim=1) / count
            z_score = (cut_feature - mean) / torch.sqrt(variance.clamp_min(1e-6))
            # Development H0 screening places the 99th percentiles near
            # z=3--4, whereas a 0 dB coherent target has a much larger crest.
            # A smooth upper-tail gate prevents ordinary H0 variation from
            # becoming a global score multiplier that calibration cancels.
            location = 4.0 if feature_index == 0 else 3.5
            evidence.append(torch.sigmoid((z_score - location) / 0.75))
        return torch.stack(evidence, dim=1)

    def analytic_anchors(self, reference_iq: Tensor, target_pfa: Tensor, mask: Tensor, scale: Tensor) -> Tensor:
        magnitude = self._magnitude(reference_iq) / scale[:, None, None]
        cell_power = magnitude.square().mean(dim=2)
        length = magnitude.shape[2]
        alpha = target_pfa.reshape(-1)
        normal = torch.distributions.Normal(
            torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype),
            torch.tensor(1.0, device=alpha.device, dtype=alpha.dtype),
        )
        z = normal.icdf(1.0 - alpha)
        z90 = normal.icdf(torch.tensor(0.90, device=alpha.device, dtype=alpha.dtype))
        if bool(mask.all()):
            values = cell_power.clamp_min(1e-8)
            count = values.shape[1]
            mean_power = values.mean(dim=1)
            sorted_power = torch.sort(values, dim=1).values
            trim = max(1, int(0.1 * count))
            trimmed_power = sorted_power[:, trim : max(trim + 1, count - trim)].mean(dim=1)
            ratio = (1.0 + z * math.sqrt(1.0 / length + 1.0 / (length * count))).clamp_min(1.0)
            ca = torch.sqrt(mean_power * ratio)
            trimmed = torch.sqrt(trimmed_power * ratio)
            rms = torch.sqrt(values)
            median = torch.quantile(rms, 0.5, dim=1)
            rayleigh = median * torch.sqrt((1.0 + z / math.sqrt(length)).clamp_min(1.0))
            log_std = torch.log(rms).std(dim=1, unbiased=False).clamp(0.02, 1.0)
            weibull = median * torch.exp(log_std * z)
            q90 = torch.quantile(rms, 0.90, dim=1)
            order = q90 * torch.exp(log_std * (z - z90).clamp_min(0.0))
            robust = q90 * torch.exp(0.5 * log_std * (z - z90).clamp_min(0.0))
            return torch.stack([ca, trimmed, rayleigh, weibull, order, robust], dim=1).clamp_min(1e-6)
        rows = []
        for index in range(len(cell_power)):
            values = cell_power[index][mask[index]].clamp_min(1e-8)
            count = values.numel()
            mean_power = values.mean()
            sorted_power = torch.sort(values).values
            trim = max(1, int(0.1 * count))
            trimmed_power = sorted_power[trim : max(trim + 1, count - trim)].mean()
            ratio = (1.0 + z[index] * math.sqrt(1.0 / length + 1.0 / (length * count))).clamp_min(1.0)
            ca = torch.sqrt(mean_power * ratio)
            trimmed = torch.sqrt(trimmed_power * ratio)
            rms = torch.sqrt(values)
            median = torch.quantile(rms, 0.5)
            rayleigh = median * torch.sqrt((1.0 + z[index] / math.sqrt(length)).clamp_min(1.0))
            log_std = torch.log(rms).std(unbiased=False).clamp(0.02, 1.0)
            weibull = median * torch.exp(log_std * z[index])
            q90 = torch.quantile(rms, 0.90)
            order = q90 * torch.exp(log_std * (z[index] - z90).clamp_min(0.0))
            robust = q90 * torch.exp(0.5 * log_std * (z[index] - z90).clamp_min(0.0))
            rows.append(torch.stack([ca, trimmed, rayleigh, weibull, order, robust]))
        return torch.stack(rows).clamp_min(1e-6)

    def forward(
        self,
        cut_iq: Tensor,
        reference_iq: Tensor,
        target_pfa: Tensor,
        reference_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        mask = self._validate_inputs(cut_iq, reference_iq, target_pfa, reference_mask)
        scale, distribution, temporal, tail_features, series_features = self._scale_and_background(reference_iq, mask)
        distribution_embedding = self.distribution_encoder(distribution)
        temporal_embedding = self.background_temporal_encoder(temporal)
        background = self.background_fusion(
            torch.cat([distribution_embedding, temporal_embedding, tail_features], dim=1)
        )
        cut_channels, base_score = self._cut_channels(cut_iq, scale)
        if self.learn_score:
            cut_embedding = self.cut_encoder(cut_channels)
            detection_evidence = self._cut_detection_evidence(cut_iq, reference_iq, mask)
            cut_embedding = cut_embedding.clone()
            cut_embedding[:, : detection_evidence.shape[1]] = detection_evidence
            conditioned = self.cut_condition(torch.cat([cut_embedding, background], dim=1))
            score_multiplier = torch.exp(torch.tanh(self.score_residual(conditioned).squeeze(1)) * self.maximum_score_log)
            uncertainty = F.softplus(self.uncertainty_head(conditioned).squeeze(1))
        else:
            score_multiplier = torch.ones_like(base_score)
            uncertainty = torch.zeros_like(base_score)
        normalized_score = base_score * score_multiplier

        anchors = self.analytic_anchors(reference_iq, target_pfa.reshape(-1), mask, scale)
        series_threshold_shift = 0.05 * torch.tanh(self.reference_series_threshold_gate(series_features)).mean(dim=1)
        anchor_weights = torch.softmax(self.anchor_logits(background), dim=1)
        anchored_threshold = torch.sum(anchor_weights * anchors, dim=1)
        threshold_multiplier = torch.exp(
            torch.tanh(self.threshold_residual(background).squeeze(1)) * self.maximum_threshold_log
        )
        normalized_threshold = anchored_threshold * threshold_multiplier
        return {
            "normalized_score": normalized_score,
            "normalized_threshold": normalized_threshold,
            "absolute_threshold": normalized_threshold * scale,
            "scale": scale,
            "uncertainty": uncertainty,
            "decision": normalized_score >= normalized_threshold,
            "anchor_weights": anchor_weights,
            "anchors": anchors,
            "series_threshold_shift": series_threshold_shift,
        }


__all__ = ["BCDRCFAR", "count_trainable_parameters"]
