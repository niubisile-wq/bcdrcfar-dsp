# BCDRCFAR IPIX calibration tradeoffs

## What this page shows

This comparison keeps the two cohorts separate:

- `development_full`: used to compare scalar, grouped, and low-rank calibration families.
- `retrospective_external`: used to show what the feature head buys on unseen acquisitions.

## Development comparison

| scheme                       |   macro_pfa |   macro_factor2 |   macro_primary_pd |   macro_absolute_log10_pfa_error | note                                              |
|:-----------------------------|------------:|----------------:|-------------------:|---------------------------------:|:--------------------------------------------------|
| global scalar                |  0.00979226 |        0.845455 |         0.374756   |                         0.278332 | baseline scalar calibration                       |
| grouped:global               |  0.00979226 |        0.4      |         0.246623   |                       nan        | grouped tail multiplier calibration               |
| grouped:file_id              |  0.0100098  |        0        |         0.0100098  |                       nan        | grouped tail multiplier calibration               |
| grouped:polarization         |  0.00982999 |        0.2      |         0.00968424 |                       nan        | grouped tail multiplier calibration               |
| grouped:file_id_polarization |  0.0100533  |        0        |         0.010026   |                       nan        | grouped tail multiplier calibration               |
| lowrank rank=2 w=0.03        |  0.011479   |        0.2      |         0.39292    |                       nan        | best low-rank calibration on the development grid |

## Retrospective comparison

| scheme       |   macro_pfa |   macro_factor2 |   macro_primary_pd |   macro_absolute_log10_pfa_error | note                                          |
|:-------------|------------:|----------------:|-------------------:|---------------------------------:|:----------------------------------------------|
| scalar       |  0.00833248 |        0.829826 |           0.264865 |                         0.213198 | same retrospective cohort, scalar calibration |
| feature head |  0.00999801 |        0.845988 |           0.281548 |                         0.199494 | external acquisition-disjoint feature head    |

## Reading

- Grouped calibration buys factor-2 control, but the PD collapse shows it is too restrictive as a main story.
- Low-rank calibration keeps the probability-of-detection side much healthier while preserving acceptable Pfa on development.
- The feature head is the best external-acquisition reliability story we currently have, but the series-level failure rate remains high, so it should be presented as a calibrator, not a solved CFAR system.
