# BCDRCFAR IPIX long-horizon holdout

Updated: 2026-08-08

## Holdout date

- `1993-11-18`

## Key result

This audit combines the frozen development-featurehead and retrospective-external featurehead rows across the comparable IPIX A-polarization sea-clutter acquisitions and then splits them by true acquisition date.

- Scalar route:
  - early macro `Pd`: `0.4662543402777778`
  - late macro `Pd`: `0.2625`
  - delta macro absolute log10 `Pfa` error: `+0.1031561295977163`
- Feature-head route:
  - early macro `Pd`: `0.4742024739583333`
  - late macro `Pd`: `0.27822265625`
  - delta macro absolute log10 `Pfa` error: `+0.02963164055104725`

## Interpretation

The later acquisition date still degrades primary detection probability for both routes.
The feature head softens the calibration-error drift relative to the scalar route, but it does not eliminate temporal degradation.
This supports the manuscript boundary that the method improves acquisition-level reliability inside the observed regime, while time generalization remains bounded.

