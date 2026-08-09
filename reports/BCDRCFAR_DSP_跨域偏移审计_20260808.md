# BCDRCFAR Cross-Domain Shift Audit

## Bottom line

The score is not domain-stable across the full radar stack. IPIX stays inside the accepted domain family, but St Andrews is only boundary evidence, the confirmatory transfer remains NO_GO, and the semantic scan-domain / NEXRAD checks stay rejected or abstained.

## Domain support and risk map

| domain | support outside 1-99% | mean outside features | median robust distance | q95 robust distance | risk AUROC | risk Spearman | multihead Spearman | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| IPIX | 0.436 | 0.498 | 0.801 | 0.939 | 0.597 | 0.107 | 0.766 | ACCEPT |
| St_Andrews_24GHz | 0.826 | 1.776 | 1.727 | 3.011 | 0.350 | -0.248 | 0.076 | ABSTAIN |
| St_Andrews_94GHz | 0.906 | 2.039 | 2.159 | 3.215 | 0.180 | -0.491 | 0.387 | ABSTAIN |
| IPIX_269_high_sea | 0.340 | 0.383 | 0.801 | 0.885 | 0.475 | -0.015 | 0.583 | ACCEPT |
| IPIX_287_low_sea | 0.262 | 0.324 | 0.841 | 0.956 | 0.798 | 0.222 | 0.086 | ACCEPT |

## Feature direction flips

- top classifier features: `log_soft_threshold, head_contamination, log_hard_soft_ratio, head_distribution, head_global`
- top feature event-direction flip count: `3`

## Gate map

- p4 domain reliability: `CLOSED`
- p4 real confirmatory: `NO_GO`
- p4 scan domain: `NO_GO`
- St Andrews holdout: `CLOSED`
- NEXRAD negative control: `ABSTAIN`

## Interpretation

This is the stronger cross-domain statement now available: the method has a bounded IPIX-centric acceptance region, but the same scoring semantics do not transfer as a universal domain-validity rule.
The St Andrews rows remain boundary-only evidence, and the NEXRAD row remains a deliberate ABSTAIN control.

## Raw audit numbers

- `IPIX`: `records = 188416`, `risk_any_event_auroc = 0.5965013677030264`, `risk_spearman_with_false_alarm_count = 0.10657574548676281`, `risk_multihead_spearman = 0.7655601493145722`
- `St_Andrews_24GHz`: `records = 13193`, `risk_any_event_auroc = 0.3500298271831648`, `risk_spearman_with_false_alarm_count = -0.24796210563991422`, `risk_multihead_spearman = 0.07646811411791093`
- `St_Andrews_94GHz`: `records = 76960`, `risk_any_event_auroc = 0.1803293711238031`, `risk_spearman_with_false_alarm_count = -0.49078455515514835`, `risk_multihead_spearman = 0.3871887841967545`
- `IPIX_269_high_sea`: `records = 512`, `risk_any_event_auroc = 0.47523613963039013`, `risk_spearman_with_false_alarm_count = -0.015133649148216764`, `risk_multihead_spearman = 0.5832306043361829`
- `IPIX_287_low_sea`: `records = 512`, `risk_any_event_auroc = 0.7982751540041069`, `risk_spearman_with_false_alarm_count = 0.22189092914539843`, `risk_multihead_spearman = 0.08617353529371373`

## Feature orientation

- top classifier features: `log_soft_threshold, head_contamination, log_hard_soft_ratio, head_distribution, head_global`
- top feature event-direction flip count: `3`

These numbers strengthen the claim that the risk orientation is domain-dependent and that the cross-domain boundary is real, not just an artifact of one summary statistic.

## St Andrews pilot addendum

The separate `st_andrews_pilot` stress test makes the boundary even sharper:

- 24GHz -> 94GHz stays far below the target PFA, but only `12%` of the pooled methods land inside factor-2.
- 94GHz -> 24GHz is asymmetric and more fragile; only the `soft` method reaches `within_factor2 = True` on pooled transfer, while the pooled factor-2 fraction is `0.0`.
- family selection is not fixed: `weibull` and `lomax` dominate many 24GHz settings, while the per-bin selection flips with direction and local range bin.

This pilot does not upgrade the claim to confirmatory status. It does strengthen the interpretation that transfer is direction-dependent and family-specific, not a universal cross-domain rule.
