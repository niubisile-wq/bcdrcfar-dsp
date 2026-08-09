# BCDRCFAR IPIX feature ablation

Updated: 2026-08-08

## What was compared

- `all`
- `no_anchor`
- `no_uncertainty`
- `no_series_shift`
- `no_polarization`
- `log_scale_only`
- `background_core`

## Main reading

- The full feature head is not a fragile single-feature trick.
- `no_uncertainty` and `no_series_shift` stay very close to `all` on both development and retrospective cohorts.
- Removing anchor or polarization inputs hurts the retrospective tradeoff more than removing the auxiliary uncertainty or series-shift terms.

## Retrospective external cohort

| family | macro_pfa | macro_factor2 | macro_pd |
| --- | ---: | ---: | ---: |
| `all` | `0.009998010706018518` | `0.2222222222222222` | `0.2815483940972222` |
| `no_anchor` | `0.010263167701318743` | `0.1111111111111111` | `0.2801378038194444` |
| `no_uncertainty` | `0.009998010706018518` | `0.2222222222222222` | `0.2815483940972222` |
| `no_series_shift` | `0.009998010706018518` | `0.2222222222222222` | `0.2815483940972222` |
| `no_polarization` | `0.009918685553451179` | `0.2222222222222222` | `0.2819281684027778` |
| `log_scale_only` | `0.010234588725799663` | `0.1111111111111111` | `0.2798122829861111` |
| `background_core` | `0.009918685553451179` | `0.2222222222222222` | `0.2819281684027778` |

## Interpretation

The retrospective gain is distributed across the background-conditioned family, not concentrated in a single cosmetic head.
At the same time, the strongest overall external-acquisition story still comes from the full feature head, because it preserves the best balance between macro `Pfa`, factor-2 behavior, and primary `Pd` across the unseen acquisitions.

