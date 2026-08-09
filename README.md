# BC-DRCFAR for Reliable Radar Target Detection in Nonstationary Sea Clutter

This repository contains code, lightweight result summaries, figure-generation
materials, and manuscript source files for the paper:

**Background-Conditioned CFAR for Reliable Radar Target Detection in
Nonstationary Sea Clutter**

BC-DRCFAR treats same-time reference clutter as calibration context. The method
uses background descriptors to condition both target-evidence scoring and the
threshold used at the declared false-alarm operating point.

## Repository contents

- `src/bcdrcfar/`: core BC-DRCFAR model, classical CFAR baselines, synthetic
  stream generation, training helpers, and evaluation utilities.
- `src/real_data.py`, `src/mat_radar.py`, `src/birmingham_626.py`: public radar
  data loading and audit helpers used by the paper workflow.
- `experiments/`: script entry points for synthetic calibration, IPIX
  retrospective evaluation, feature-head calibration, ablation, and audit runs.
- `results/`: lightweight CSV summaries used by the manuscript figures and
  tables.
- `reports/`: selected audit records supporting the final manuscript claims.
- `manuscript/`: Elsevier LaTeX source, compiled PDF, figure PDFs, bibliography,
  and figure-generation script.

Large raw radar files, intermediate block-level predictions, fitted model
artifacts, virtual environments, downloaded literature PDFs, and temporary
working files are intentionally not included.

## Environment

Tested with Python 3.12 on Windows. A minimal environment can be created with:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS users can use the equivalent activation command:

```bash
source .venv/bin/activate
```

## Manuscript build

The manuscript source is under `manuscript/`. Compile with:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode main.tex
```

The compiled manuscript PDF is also provided as `manuscript/main.pdf`.

## Data

The manuscript uses public radar data sources and project-generated synthetic
benchmarks. Public raw data are not mirrored here. See `DATA_AVAILABILITY.md`
for source locations and access notes.

The included CSV files are lightweight processed summaries sufficient to audit
the reported figure/table values without redistributing large third-party radar
files.

## Reuse

Code is released under the MIT License. Third-party public radar datasets remain
under their original providers' terms.
