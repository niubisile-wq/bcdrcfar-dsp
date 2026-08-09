"""Analyze a longer-horizon chronological holdout across all comparable IPIX files.

This combines the frozen development-featurehead and retrospective-external
featurehead condition rows for the comparable A-polarization sea-clutter
acquisitions, then splits by true acquisition date.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_data import load_ipix_series


DEV_ROWS = ROOT / "results" / "bcdrcfar_ipix" / "development_featurehead" / "condition_rows.csv"
RET_ROWS = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "condition_rows.csv"
DATA_DIR = ROOT / "data" / "raw" / "ipix"
OUT_DIR = ROOT / "results" / "bcdrcfar_ipix" / "long_horizon_holdout"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-rows", type=Path, default=DEV_ROWS)
    parser.add_argument("--ret-rows", type=Path, default=RET_ROWS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--bootstrap", type=int, default=3000)
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
            return None if math.isnan(value) else value
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, float):
            return None if math.isnan(value) else value
        return value

    path.write_text(json.dumps(safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(frame: pd.DataFrame, decision_col: str, target_pfa: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["decision"] = work[decision_col].astype(bool)

    acquisition_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for file_id, group in work.groupby("file_id", sort=True):
        clutter = group[group["role"] == "clutter"]
        primary = group[group["role"] == "primary"]
        events = int(clutter["decision"].sum())
        trials = int(len(clutter))
        adjusted = (events + 0.5) / (trials + 1.0)
        acquisition_rows.append(
            {
                "file_id": file_id,
                "events": events,
                "trials": trials,
                "pfa": float(events / trials) if trials else float("nan"),
                "absolute_log10_pfa_error": abs(float(np.log10(adjusted / target_pfa))),
            }
        )
        if not primary.empty:
            target_rows.append(
                {
                    "file_id": file_id,
                    "role": "primary",
                    "events": int(primary["decision"].sum()),
                    "trials": int(len(primary)),
                    "pd": float(primary["decision"].mean()),
                }
            )

    return pd.DataFrame(acquisition_rows), pd.DataFrame(target_rows)


def cohort_metrics(acq: pd.DataFrame, tgt: pd.DataFrame) -> dict[str, float]:
    primary = tgt[tgt["role"] == "primary"]
    return {
        "macro_pfa": float(acq["pfa"].mean()) if not acq.empty else float("nan"),
        "macro_absolute_log10_pfa_error": float(acq["absolute_log10_pfa_error"].mean()) if not acq.empty else float("nan"),
        "macro_primary_pd": float(primary["pd"].mean()) if not primary.empty else float("nan"),
    }


def bootstrap_delta(early: pd.DataFrame, late: pd.DataFrame, metric: str, replicates: int, seed: int) -> tuple[float, tuple[float, float]]:
    if early.empty or late.empty:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    e = early[metric].to_numpy(dtype=float)
    l = late[metric].to_numpy(dtype=float)
    observed = float(l.mean() - e.mean())
    deltas = []
    for _ in range(replicates):
        deltas.append(float(rng.choice(l, size=len(l), replace=True).mean() - rng.choice(e, size=len(e), replace=True).mean()))
    return observed, (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def linear_slope(summary: pd.DataFrame, metric: str) -> float:
    if summary.empty or len(summary) < 2:
        return float("nan")
    x = pd.to_datetime(summary["date"]).map(datetime.toordinal).to_numpy(dtype=float)
    y = summary[metric].to_numpy(dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0:
        return float("nan")
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def load_dates(file_ids: list[str], data_dir: Path) -> dict[str, pd.Timestamp]:
    dates: dict[str, pd.Timestamp] = {}
    for fid in file_ids:
        s = load_ipix_series(data_dir / f"ipix_{fid}.cdf", "hh", 1, preprocess="raw")
        dates[fid] = pd.to_datetime(str(s.metadata["date"]), format="%Y/%m/%d %H:%M:%S")
    return dates


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dev = pd.read_csv(args.dev_rows)
    ret = pd.read_csv(args.ret_rows)
    frame = pd.concat([dev, ret], ignore_index=True)
    frame["file_id"] = frame["file_id"].astype(str).str.zfill(3)
    # Only the comparable sea-clutter A acquisitions are used in this audit.
    file_ids = sorted(frame["file_id"].drop_duplicates().tolist())
    dates = load_dates(file_ids, args.data_dir)
    frame["date"] = frame["file_id"].map(dates)
    frame["date_day"] = pd.to_datetime(frame["date"]).dt.normalize()

    # Use the retrospective holdout date as the late cohort; this keeps the
    # earlier development acquisitions as the longer-horizon early cohort.
    holdout_date = pd.Timestamp("1993-11-18")

    outputs: dict[str, Any] = {}
    date_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    for method_name, col in [("scalar", "decision_bcdrcfar_scalar"), ("feature", "decision_bcdrcfar_feature")]:
        acq, tgt = summarize(frame[["file_id", "role", "date", "date_day", col]].rename(columns={col: "decision"}), "decision", 0.01)
        acq = acq.merge(frame[["file_id", "date", "date_day"]].drop_duplicates(), on="file_id", how="left")
        tgt = tgt.merge(frame[["file_id", "date", "date_day"]].drop_duplicates(), on="file_id", how="left")

        per_date = []
        for d, g in acq.groupby("date_day", sort=True):
            t = tgt[tgt["date_day"] == d]
            per_date.append({"method": method_name, "date": pd.Timestamp(d).strftime("%Y-%m-%d"), "files": int(g["file_id"].nunique()), **cohort_metrics(g, t)})
        per_date_df = pd.DataFrame(per_date).sort_values("date")
        date_rows.extend(per_date_df.to_dict(orient="records"))

        early = acq[acq["date_day"] < holdout_date]
        late = acq[acq["date_day"] == holdout_date]
        early_t = tgt[tgt["date_day"] < holdout_date]
        late_t = tgt[tgt["date_day"] == holdout_date]
        early_metrics = cohort_metrics(early, early_t)
        late_metrics = cohort_metrics(late, late_t)
        delta_pfa = bootstrap_delta(early, late, "pfa", args.bootstrap, args.seed)
        delta_err = bootstrap_delta(early, late, "absolute_log10_pfa_error", args.bootstrap, args.seed + 1)
        delta_pd = bootstrap_delta(early_t, late_t, "pd", args.bootstrap, args.seed + 2)

        cohort_rows.extend([
            {"method": method_name, "cohort": "early", "date_range": f"{pd.Timestamp(per_date_df['date'].min()).strftime('%Y-%m-%d')}..{(holdout_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}", **early_metrics},
            {"method": method_name, "cohort": "late", "date_range": holdout_date.strftime("%Y-%m-%d"), **late_metrics},
        ])
        delta_rows.append({
            "method": method_name,
            "delta_macro_pfa": delta_pfa[0],
            "delta_macro_pfa_ci_low": delta_pfa[1][0],
            "delta_macro_pfa_ci_high": delta_pfa[1][1],
            "delta_macro_absolute_log10_pfa_error": delta_err[0],
            "delta_macro_absolute_log10_pfa_error_ci_low": delta_err[1][0],
            "delta_macro_absolute_log10_pfa_error_ci_high": delta_err[1][1],
            "delta_macro_primary_pd": delta_pd[0],
            "delta_macro_primary_pd_ci_low": delta_pd[1][0],
            "delta_macro_primary_pd_ci_high": delta_pd[1][1],
        })
        outputs[method_name] = {
            "early": early_metrics,
            "late": late_metrics,
            "delta": delta_rows[-1],
            "date_slope_pfa": linear_slope(per_date_df, "macro_pfa"),
            "date_slope_error": linear_slope(per_date_df, "macro_absolute_log10_pfa_error"),
            "date_slope_pd": linear_slope(per_date_df, "macro_primary_pd"),
        }

    file_date_rows = [
        {"file_id": fid, "date": dates[fid].strftime("%Y-%m-%d %H:%M:%S")}
        for fid in file_ids
    ]
    pd.DataFrame(file_date_rows).to_csv(args.output_dir / "file_dates.csv", index=False)
    pd.DataFrame(date_rows).to_csv(args.output_dir / "date_summary.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(args.output_dir / "cohort_summary.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(args.output_dir / "cohort_deltas.csv", index=False)
    write_json(args.output_dir / "summary.json", {
        "holdout_date": holdout_date.strftime("%Y-%m-%d"),
        "file_dates": file_date_rows,
        "per_method": outputs,
    })

    lines = [
        "# BCDRCFAR Long-Horizon Chronological Holdout",
        "",
        f"Holdout date: `{holdout_date.strftime('%Y-%m-%d')}`",
        "",
    ]
    for row in delta_rows:
        lines.append(
            f"- `{row['method']}`: delta pfa = {row['delta_macro_pfa']:.6g}, delta error = {row['delta_macro_absolute_log10_pfa_error']:.6g}, delta primary PD = {row['delta_macro_primary_pd']:.6g}"
        )
    lines.extend(["", "## Coverage", ""])
    for row in file_date_rows:
        lines.append(f"- `{row['file_id']}` -> `{row['date']}`")
    lines.extend(["", "## Interpretation", "", "This audit extends the true time window by combining the frozen development and retrospective IPIX acquisitions into one chronological sequence. It strengthens the time-horizon evidence, but it still remains IPIX-only rather than a multi-site time generalization test."])
    report = args.output_dir / "summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary_md": str(report), "summary_json": str(args.output_dir / 'summary.json')}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
