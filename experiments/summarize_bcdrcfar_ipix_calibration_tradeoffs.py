"""Summarize grouped, low-rank, and feature-head BC-DRCFAR tradeoffs on IPIX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEV_SUMMARY = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "summary.json"
GROUPED_SUMMARY = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "grouped_calibration" / "summary.csv"
LOWRANK_BEST = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "lowrank_calibration" / "best_multipliers.json"
LOWRANK_GRID = ROOT / "results" / "bcdrcfar_ipix" / "development_full" / "lowrank_calibration" / "grid_summary.csv"
RETRO_FEATURE = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "summary.json"
RETRO_FEATURE_CSV = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "summary.csv"
RETRO_TARGET_CSV = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "target_metrics.csv"
OUT_DIR = ROOT / "reports"
OUT_MD = OUT_DIR / "BCDRCFAR_DSP_三路线比较_20260808.md"
OUT_CSV = OUT_DIR / "BCDRCFAR_DSP_三路线比较_20260808.csv"
OUT_JSON = OUT_DIR / "BCDRCFAR_DSP_三路线比较_20260808.json"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    return parser.parse_args(argv)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_lowrank_best(grid: pd.DataFrame, best: dict[str, Any]) -> pd.Series:
    if not best:
        raise RuntimeError("low-rank best multipliers are missing")
    rank = int(best["summary"]["rank"])
    weight = float(best["summary"]["positive_weight"])
    match = grid[(grid["rank"] == rank) & (grid["positive_weight"] == weight)]
    if match.empty:
        raise RuntimeError("low-rank best configuration not found in grid summary")
    return match.iloc[0]


def format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_empty_"
    return frame[columns].to_markdown(index=False)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    out_md = args.out_md.resolve()
    out_csv = args.out_csv.resolve()
    out_json = args.out_json.resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    dev = read_json(DEV_SUMMARY)
    grouped = pd.read_csv(GROUPED_SUMMARY)
    lowrank_best = read_json(LOWRANK_BEST)
    lowrank_grid = pd.read_csv(LOWRANK_GRID)
    retro_summary = pd.read_csv(RETRO_FEATURE_CSV)
    retro_target = pd.read_csv(RETRO_TARGET_CSV)
    retro_summary["method"] = retro_summary["method"].astype(str)
    retro_target["method"] = retro_target["method"].astype(str)
    retro_target["role"] = retro_target["role"].astype(str)

    retro_rows = []
    for method, note in [("bcdrcfar_scalar", "same retrospective cohort, scalar calibration"), ("bcdrcfar_feature", "external acquisition-disjoint feature head")]:
        subset = retro_summary[retro_summary["method"] == method]
        target_subset = retro_target[(retro_target["method"] == method) & (retro_target["role"] == "primary")]
        if subset.empty:
            raise RuntimeError(f"missing retrospective rows for {method}")
        if target_subset.empty:
            raise RuntimeError(f"missing retrospective target rows for {method}")
        retro_rows.append(
            {
                "family": "retrospective_feature_head" if method == "bcdrcfar_feature" else "retrospective_scalar",
                "cohort": "retrospective_external",
                "scheme": "feature head" if method == "bcdrcfar_feature" else "scalar",
                "macro_pfa": float(subset["pfa"].mean()),
                "macro_factor2": float(subset["series_factor2_violation_rate"].mean()),
                "macro_primary_pd": float(target_subset["pd"].mean()),
                "macro_absolute_log10_pfa_error": float(subset["absolute_log10_pfa_error"].mean()),
                "note": note,
            }
        )
    retro = read_json(RETRO_FEATURE)

    dev_rows = [
        {
            "family": "development_scalar_global",
            "cohort": "development_full",
            "scheme": "global scalar",
            "macro_pfa": float(dev["bcdrcfar_macro_pfa"]),
            "macro_factor2": float(dev["bcdrcfar_macro_series_factor2_violation_rate"]),
            "macro_primary_pd": float(dev["bcdrcfar_macro_primary_pd"]),
            "macro_absolute_log10_pfa_error": float(dev["bcdrcfar_macro_absolute_log10_pfa_error"]),
            "note": "baseline scalar calibration",
        }
    ]
    for _, row in grouped.iterrows():
        dev_rows.append(
            {
                "family": "development_grouped",
                "cohort": "development_full",
                "scheme": f"grouped:{row['scheme']}",
                "macro_pfa": float(row["macro_pfa"]),
                "macro_factor2": float(row["macro_factor2"]),
                "macro_primary_pd": float(row["macro_pd"]),
                "macro_absolute_log10_pfa_error": float("nan"),
                "note": "grouped tail multiplier calibration",
            }
        )

    best_lowrank_row = pick_lowrank_best(lowrank_grid, lowrank_best)
    dev_rows.append(
        {
            "family": "development_lowrank_best",
            "cohort": "development_full",
            "scheme": f"lowrank rank={int(best_lowrank_row['rank'])} w={float(best_lowrank_row['positive_weight']):g}",
            "macro_pfa": float(best_lowrank_row["macro_pfa"]),
            "macro_factor2": float(best_lowrank_row["macro_factor2"]),
            "macro_primary_pd": float(best_lowrank_row["macro_pd"]),
            "macro_absolute_log10_pfa_error": float("nan"),
            "note": "best low-rank calibration on the development grid",
        }
    )

    retro_rows[0]["note"] = "same retrospective cohort, scalar calibration"
    retro_rows[1]["note"] = "external acquisition-disjoint feature head"

    dev_table = pd.DataFrame(dev_rows)
    retro_table = pd.DataFrame(retro_rows)
    combined = pd.concat([dev_table, retro_table], ignore_index=True)
    combined.to_csv(out_csv, index=False)

    report_lines = [
        "# BCDRCFAR IPIX calibration tradeoffs",
        "",
        "## What this page shows",
        "",
        "This comparison keeps the two cohorts separate:",
        "",
        "- `development_full`: used to compare scalar, grouped, and low-rank calibration families.",
        "- `retrospective_external`: used to show what the feature head buys on unseen acquisitions.",
        "",
        "## Development comparison",
        "",
        format_table(
            dev_table,
            [
                "scheme",
                "macro_pfa",
                "macro_factor2",
                "macro_primary_pd",
                "macro_absolute_log10_pfa_error",
                "note",
            ],
        ),
        "",
        "## Retrospective comparison",
        "",
        format_table(
            retro_table,
            [
                "scheme",
                "macro_pfa",
                "macro_factor2",
                "macro_primary_pd",
                "macro_absolute_log10_pfa_error",
                "note",
            ],
        ),
        "",
        "## Reading",
        "",
        "- Grouped calibration buys factor-2 control, but the PD collapse shows it is too restrictive as a main story.",
        "- Low-rank calibration keeps the probability-of-detection side much healthier while preserving acceptable Pfa on development.",
        "- The feature head is the best external-acquisition reliability story we currently have, but the series-level failure rate remains high, so it should be presented as a calibrator, not a solved CFAR system.",
    ]
    out_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    payload = {
        "status": "BCDRCFAR_IPIX_CALIBRATION_TRADEOFF_SUMMARY_COMPLETE",
        "development_summary_sha256": sha256(DEV_SUMMARY),
        "grouped_summary_rows": int(len(grouped)),
        "lowrank_grid_rows": int(len(lowrank_grid)),
        "lowrank_best": lowrank_best,
        "development_table": dev_table.to_dict(orient="records"),
        "retrospective_table": retro_table.to_dict(orient="records"),
        "output_files": {
            "markdown": str(out_md),
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_md)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
