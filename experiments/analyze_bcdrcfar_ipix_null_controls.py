"""Analyze null controls for the retrospective external IPIX feature head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "condition_rows.csv"
DEFAULT_SUMMARY = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "summary.json"
DEFAULT_OUTPUT = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead" / "null_controls"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
            return None if np.isnan(value) else value
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, float):
            return None if np.isnan(value) else value
        return value

    path.write_text(
        json.dumps(safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args(argv)


def series_summary(frame: pd.DataFrame, decision_col: str, target_pfa: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["decision"] = work[decision_col].astype(bool)
    keys = ["file_id", "polarization", "range_bin", "role"]

    series_rows: list[dict[str, Any]] = []
    for key, group in work.groupby(keys, sort=True):
        events = int(group["decision"].sum())
        trials = int(len(group))
        adjusted = (events + 0.5) / (trials + 1.0)
        role = str(key[-1])
        series_rows.append(
            {
                **dict(zip(keys, key)),
                "events": events,
                "trials": trials,
                "rate": float(events / trials) if trials else float("nan"),
                "jeffreys_rate": float(adjusted),
                "absolute_log10_pfa_error": (
                    abs(float(np.log10(adjusted / target_pfa))) if role == "clutter" else np.nan
                ),
                "factor2_violation": bool(
                    adjusted < target_pfa / 2.0 or adjusted > target_pfa * 2.0
                )
                if role == "clutter"
                else False,
            }
        )

    series = pd.DataFrame(series_rows)
    series_columns = [
        "file_id",
        "polarization",
        "range_bin",
        "role",
        "events",
        "trials",
        "rate",
        "jeffreys_rate",
        "absolute_log10_pfa_error",
        "factor2_violation",
    ]
    if series.empty:
        series = pd.DataFrame(columns=series_columns)
    else:
        series = series[series_columns]

    clutter = series[series["role"] == "clutter"].copy()
    acquisition_rows: list[dict[str, Any]] = []
    for file_id, group in clutter.groupby("file_id", sort=True):
        events = int(group["events"].sum())
        trials = int(group["trials"].sum())
        adjusted = (events + 0.5) / (trials + 1.0)
        acquisition_rows.append(
            {
                "file_id": file_id,
                "events": events,
                "trials": trials,
                "pfa": float(events / trials) if trials else float("nan"),
                "absolute_log10_pfa_error": abs(float(np.log10(adjusted / target_pfa))),
                "series_factor2_violation_rate": float(group["factor2_violation"].mean()),
            }
        )
    acquisition_columns = [
        "file_id",
        "events",
        "trials",
        "pfa",
        "absolute_log10_pfa_error",
        "series_factor2_violation_rate",
    ]
    acquisition = pd.DataFrame(acquisition_rows, columns=acquisition_columns)

    target = (
        series[series["role"].isin(["primary", "secondary"])]
        .groupby(["file_id", "role"], sort=True)
        .agg(events=("events", "sum"), trials=("trials", "sum"))
        .reset_index()
    )
    if target.empty:
        target = pd.DataFrame(columns=["file_id", "role", "events", "trials", "pd"])
    else:
        target = target[["file_id", "role", "events", "trials"]]
        target["pd"] = target["events"] / target["trials"]
    return series, acquisition, target


def macro_metrics(acquisition: pd.DataFrame, target: pd.DataFrame) -> dict[str, float]:
    primary = target[target["role"] == "primary"] if not target.empty else pd.DataFrame(columns=["pd"])
    return {
        "macro_pfa": float(acquisition["pfa"].mean()) if not acquisition.empty else float("nan"),
        "macro_absolute_log10_pfa_error": float(acquisition["absolute_log10_pfa_error"].mean())
        if not acquisition.empty
        else float("nan"),
        "macro_series_factor2_violation_rate": float(acquisition["series_factor2_violation_rate"].mean())
        if not acquisition.empty
        else float("nan"),
        "macro_primary_pd": float(primary["pd"].mean()) if not primary.empty else float("nan"),
    }


def paired_deltas(
    control_acquisition: pd.DataFrame,
    control_target: pd.DataFrame,
    baseline_acquisition: pd.DataFrame,
    baseline_target: pd.DataFrame,
) -> dict[str, float]:
    control_primary = control_target[control_target["role"] == "primary"][["file_id", "pd"]] if not control_target.empty else pd.DataFrame(columns=["file_id", "pd"])
    baseline_primary = baseline_target[baseline_target["role"] == "primary"][["file_id", "pd"]] if not baseline_target.empty else pd.DataFrame(columns=["file_id", "pd"])
    merged_error = control_acquisition[["file_id", "absolute_log10_pfa_error"]].merge(
        baseline_acquisition[["file_id", "absolute_log10_pfa_error"]],
        on="file_id",
        suffixes=("_control", "_baseline"),
        how="inner",
    )
    merged_pd = control_primary.merge(
        baseline_primary,
        on="file_id",
        suffixes=("_control", "_baseline"),
        how="inner",
    )
    if merged_error.empty or merged_pd.empty:
        return {
            "paired_mean_acquisition_error_difference": float("nan"),
            "paired_mean_primary_pd_difference": float("nan"),
        }
    return {
        "paired_mean_acquisition_error_difference": float(
            (merged_error["absolute_log10_pfa_error_control"] - merged_error["absolute_log10_pfa_error_baseline"]).mean()
        ),
        "paired_mean_primary_pd_difference": float(
            (merged_pd["pd_control"] - merged_pd["pd_baseline"]).mean()
        ),
    }


def bootstrap_paired_deltas(
    control_acquisition: pd.DataFrame,
    control_target: pd.DataFrame,
    baseline_acquisition: pd.DataFrame,
    baseline_target: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    control_primary = control_target[control_target["role"] == "primary"][["file_id", "pd"]] if not control_target.empty else pd.DataFrame(columns=["file_id", "pd"])
    baseline_primary = baseline_target[baseline_target["role"] == "primary"][["file_id", "pd"]] if not baseline_target.empty else pd.DataFrame(columns=["file_id", "pd"])
    merged_error = control_acquisition[["file_id", "absolute_log10_pfa_error"]].merge(
        baseline_acquisition[["file_id", "absolute_log10_pfa_error"]],
        on="file_id",
        suffixes=("_control", "_baseline"),
        how="inner",
    )
    merged_pd = control_primary.merge(
        baseline_primary,
        on="file_id",
        suffixes=("_control", "_baseline"),
        how="inner",
    )
    error_diff = (
        merged_error["absolute_log10_pfa_error_control"] - merged_error["absolute_log10_pfa_error_baseline"]
    ).to_numpy()
    pd_diff = (merged_pd["pd_control"] - merged_pd["pd_baseline"]).to_numpy()
    rng = np.random.default_rng(int(seed))
    if len(error_diff) == 0 or len(pd_diff) == 0:
        raise RuntimeError("paired bootstrap has no common acquisitions")
    indices = rng.integers(0, len(error_diff), size=(int(replicates), len(error_diff)))
    error_boot = error_diff[indices].mean(axis=1)
    pd_boot = pd_diff[indices].mean(axis=1)
    return {
        "paired_mean_acquisition_error_difference": float(error_diff.mean()),
        "paired_error_difference_95ci": np.quantile(error_boot, [0.025, 0.975]).tolist(),
        "paired_mean_primary_pd_difference": float(pd_diff.mean()),
        "paired_primary_pd_difference_95ci": np.quantile(pd_boot, [0.025, 0.975]).tolist(),
    }


def permute_values(values: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    permuted = values.copy()
    for idx in groups:
        if len(idx) <= 1:
            continue
        permuted[idx] = values[idx][rng.permutation(len(idx))]
    return permuted


def group_indices(frame: pd.DataFrame, columns: list[str]) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for _, group in frame.groupby(columns, sort=False):
        groups.append(group.index.to_numpy())
    return groups


def stratified_delta_table(frame: pd.DataFrame, group_cols: list[str], target_pfa: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, subset in frame.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        _, feature_acq, feature_target = series_summary(subset, "decision_bcdrcfar_feature", target_pfa)
        _, scalar_acq, scalar_target = series_summary(subset, "decision_bcdrcfar_scalar", target_pfa)
        feature_primary = feature_target[feature_target["role"] == "primary"]["pd"]
        scalar_primary = scalar_target[scalar_target["role"] == "primary"]["pd"]
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "feature_macro_absolute_log10_pfa_error": float(feature_acq["absolute_log10_pfa_error"].mean()),
                "scalar_macro_absolute_log10_pfa_error": float(scalar_acq["absolute_log10_pfa_error"].mean()),
                "feature_macro_primary_pd": float(feature_primary.mean()),
                "scalar_macro_primary_pd": float(scalar_primary.mean()),
                "error_delta": float(
                    feature_acq["absolute_log10_pfa_error"].mean() - scalar_acq["absolute_log10_pfa_error"].mean()
                ),
                "pd_delta": float(feature_primary.mean() - scalar_primary.mean()),
            }
        )
    return pd.DataFrame(rows)


def series_localization_table(
    feature_series: pd.DataFrame,
    scalar_series: pd.DataFrame,
) -> pd.DataFrame:
    merged = feature_series.merge(
        scalar_series,
        on=["file_id", "polarization", "range_bin", "role"],
        suffixes=("_feature", "_scalar"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "file_id",
                "polarization",
                "range_bin",
                "role",
                "feature_rate",
                "scalar_rate",
                "rate_delta",
                "feature_jeffreys_rate",
                "scalar_jeffreys_rate",
                "jeffreys_delta",
                "feature_factor2_violation",
                "scalar_factor2_violation",
                "factor2_delta",
                "feature_absolute_log10_pfa_error",
                "scalar_absolute_log10_pfa_error",
                "absolute_log10_pfa_error_delta",
            ]
        )
    merged["rate_delta"] = merged["rate_feature"] - merged["rate_scalar"]
    merged["jeffreys_delta"] = merged["jeffreys_rate_feature"] - merged["jeffreys_rate_scalar"]
    merged["factor2_delta"] = merged["factor2_violation_feature"].astype(int) - merged["factor2_violation_scalar"].astype(int)
    merged["absolute_log10_pfa_error_delta"] = (
        merged["absolute_log10_pfa_error_feature"] - merged["absolute_log10_pfa_error_scalar"]
    )
    return merged[
        [
            "file_id",
            "polarization",
            "range_bin",
            "role",
            "rate_feature",
            "rate_scalar",
            "rate_delta",
            "jeffreys_rate_feature",
            "jeffreys_rate_scalar",
            "jeffreys_delta",
            "factor2_violation_feature",
            "factor2_violation_scalar",
            "factor2_delta",
            "absolute_log10_pfa_error_feature",
            "absolute_log10_pfa_error_scalar",
            "absolute_log10_pfa_error_delta",
        ]
    ].rename(
        columns={
            "rate_feature": "feature_rate",
            "rate_scalar": "scalar_rate",
            "jeffreys_rate_feature": "feature_jeffreys_rate",
            "jeffreys_rate_scalar": "scalar_jeffreys_rate",
            "factor2_violation_feature": "feature_factor2_violation",
            "factor2_violation_scalar": "scalar_factor2_violation",
            "absolute_log10_pfa_error_feature": "feature_absolute_log10_pfa_error",
            "absolute_log10_pfa_error_scalar": "scalar_absolute_log10_pfa_error",
        }
    )


def operating_point_table(frame: pd.DataFrame, target_pfa: float, scales: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ratio = frame["ratio_bcdrcfar_feature"].to_numpy(dtype=float)
    multiplier = frame["feature_multiplier"].to_numpy(dtype=float)
    for scale in scales:
        decision = ratio >= (multiplier * float(scale))
        work = frame.copy()
        work["decision_scaled"] = decision
        _, acquisition, target = series_summary(work, "decision_scaled", target_pfa)
        rows.append(
            {
                "scale_factor": float(scale),
                "macro_pfa": float(acquisition["pfa"].mean()) if not acquisition.empty else float("nan"),
                "macro_absolute_log10_pfa_error": float(acquisition["absolute_log10_pfa_error"].mean())
                if not acquisition.empty
                else float("nan"),
                "macro_series_factor2_violation_rate": float(acquisition["series_factor2_violation_rate"].mean())
                if not acquisition.empty
                else float("nan"),
                "macro_primary_pd": float(target[target["role"] == "primary"]["pd"].mean())
                if not target.empty
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def null_control_samples(
    frame: pd.DataFrame,
    *,
    target_pfa: float,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = frame.reset_index(drop=True).copy()
    base["file_id"] = base["file_id"].astype(str)
    scalar_series, scalar_acq, scalar_target = series_summary(base, "decision_bcdrcfar_scalar", target_pfa)
    observed_series, observed_acq, observed_target = series_summary(base, "decision_bcdrcfar_feature", target_pfa)
    observed = {
        "scalar": {
            "series": scalar_series,
            "acquisition": scalar_acq,
            "target": scalar_target,
        },
        "feature": {
            "series": observed_series,
            "acquisition": observed_acq,
            "target": observed_target,
        },
    }

    ratio = base["ratio_bcdrcfar_feature"].to_numpy(dtype=float)
    multiplier = base["feature_multiplier"].to_numpy(dtype=float)
    samples: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))

    modes = {
        "feature_permutation": ["file_id", "polarization", "range_bin"],
        "acquisition_shuffle": ["polarization", "range_bin", "block_index"],
        "polarization_shuffle": ["file_id", "range_bin", "block_index"],
    }
    grouped_indices = {name: group_indices(base, cols) for name, cols in modes.items()}

    for name, groups in grouped_indices.items():
        for replicate in range(int(replicates)):
            permuted = permute_values(multiplier, groups, rng)
            control = base.copy()
            control["decision_feature_null"] = ratio >= permuted
            _, acquisition, target = series_summary(control, "decision_feature_null", target_pfa)
            metrics = macro_metrics(acquisition, target)
            delta = paired_deltas(acquisition, target, scalar_acq, scalar_target)
            samples.append(
                {
                    "mode": name,
                    "replicate": replicate,
                    **metrics,
                    "paired_error_delta_vs_scalar": delta["paired_mean_acquisition_error_difference"],
                    "paired_pd_delta_vs_scalar": delta["paired_mean_primary_pd_difference"],
                }
            )

    for offset in range(8):
        bootstrap = bootstrap_paired_deltas(
            observed["feature"]["acquisition"],
            observed["feature"]["target"],
            observed["scalar"]["acquisition"],
            observed["scalar"]["target"],
            replicates=replicates,
            seed=seed + offset,
        )
        seed_rows.append(
            {
                "seed": int(seed + offset),
                "bootstrap_mean_acquisition_error_difference": bootstrap["paired_mean_acquisition_error_difference"],
                "bootstrap_error_ci_low": float(bootstrap["paired_error_difference_95ci"][0]),
                "bootstrap_error_ci_high": float(bootstrap["paired_error_difference_95ci"][1]),
                "bootstrap_mean_primary_pd_difference": bootstrap["paired_mean_primary_pd_difference"],
                "bootstrap_pd_ci_low": float(bootstrap["paired_primary_pd_difference_95ci"][0]),
                "bootstrap_pd_ci_high": float(bootstrap["paired_primary_pd_difference_95ci"][1]),
            }
        )

    return pd.DataFrame(samples), {
        "observed_scalar_series": observed["scalar"]["series"],
        "observed_scalar_acquisition": observed["scalar"]["acquisition"],
        "observed_scalar_target": observed["scalar"]["target"],
        "observed_feature_series": observed["feature"]["series"],
        "observed_feature_acquisition": observed["feature"]["acquisition"],
        "observed_feature_target": observed["feature"]["target"],
        "seed_sensitivity": pd.DataFrame(seed_rows),
    }


def format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_empty_"
    return frame[columns].to_markdown(index=False)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.input)
    frame["file_id"] = frame["file_id"].astype(str)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    target_pfa = float(summary["target_pfa"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, aux = null_control_samples(
        frame,
        target_pfa=target_pfa,
        replicates=int(args.replicates),
        seed=int(args.seed),
    )

    observed_scalar_acq = aux["observed_scalar_acquisition"]
    observed_scalar_tgt = aux["observed_scalar_target"]
    observed_scalar_series = aux["observed_scalar_series"]
    observed_feature_acq = aux["observed_feature_acquisition"]
    observed_feature_tgt = aux["observed_feature_target"]
    observed_feature_series = aux["observed_feature_series"]
    seed_sensitivity = aux["seed_sensitivity"]

    observed_metrics = {
        "scalar": {
            **macro_metrics(observed_scalar_acq, observed_scalar_tgt),
        },
        "feature": {
            **macro_metrics(observed_feature_acq, observed_feature_tgt),
        },
    }
    observed_deltas = paired_deltas(
        observed_feature_acq, observed_feature_tgt, observed_scalar_acq, observed_scalar_tgt
    )
    bootstrap = bootstrap_paired_deltas(
        observed_feature_acq,
        observed_feature_tgt,
        observed_scalar_acq,
        observed_scalar_tgt,
        replicates=int(args.replicates),
        seed=int(args.seed),
    )
    operating_point = operating_point_table(frame, target_pfa, [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    series_localization = series_localization_table(observed_feature_series, observed_scalar_series)
    mode_summary = (
        samples.groupby("mode", sort=True)
        .agg(
            macro_absolute_log10_pfa_error_mean=("macro_absolute_log10_pfa_error", "mean"),
            macro_absolute_log10_pfa_error_low=("macro_absolute_log10_pfa_error", lambda s: float(s.quantile(0.025))),
            macro_absolute_log10_pfa_error_high=("macro_absolute_log10_pfa_error", lambda s: float(s.quantile(0.975))),
            macro_primary_pd_mean=("macro_primary_pd", "mean"),
            macro_primary_pd_low=("macro_primary_pd", lambda s: float(s.quantile(0.025))),
            macro_primary_pd_high=("macro_primary_pd", lambda s: float(s.quantile(0.975))),
            paired_error_delta_vs_scalar_mean=("paired_error_delta_vs_scalar", "mean"),
            paired_error_delta_vs_scalar_low=("paired_error_delta_vs_scalar", lambda s: float(s.quantile(0.025))),
            paired_error_delta_vs_scalar_high=("paired_error_delta_vs_scalar", lambda s: float(s.quantile(0.975))),
            paired_pd_delta_vs_scalar_mean=("paired_pd_delta_vs_scalar", "mean"),
            paired_pd_delta_vs_scalar_low=("paired_pd_delta_vs_scalar", lambda s: float(s.quantile(0.025))),
            paired_pd_delta_vs_scalar_high=("paired_pd_delta_vs_scalar", lambda s: float(s.quantile(0.975))),
        )
        .reset_index()
    )
    mode_summary["p_error_better_than_observed"] = [
        float(
            (samples.loc[samples["mode"] == mode, "paired_error_delta_vs_scalar"] <= observed_deltas["paired_mean_acquisition_error_difference"]).mean()
        )
        for mode in mode_summary["mode"]
    ]
    mode_summary["p_pd_better_than_observed"] = [
        float(
            (samples.loc[samples["mode"] == mode, "paired_pd_delta_vs_scalar"] >= observed_deltas["paired_mean_primary_pd_difference"]).mean()
        )
        for mode in mode_summary["mode"]
    ]

    samples_path = output_dir / "null_control_samples.csv"
    series_metrics_path = output_dir / "series_metrics.csv"
    series_localization_path = output_dir / "series_failure_localization.csv"
    localization_path = output_dir / "failure_localization.csv"
    seed_sensitivity_path = output_dir / "seed_sensitivity.csv"
    observed_path = output_dir / "observed_metrics.csv"
    operating_point_path = output_dir / "operating_point_summary.csv"
    samples.to_csv(samples_path, index=False)
    observed_scalar_series.to_csv(series_metrics_path.with_name("series_metrics_scalar.csv"), index=False)
    observed_feature_series.to_csv(series_metrics_path.with_name("series_metrics_feature.csv"), index=False)
    series_localization.to_csv(series_localization_path, index=False)
    seed_sensitivity.to_csv(seed_sensitivity_path, index=False)
    operating_point.to_csv(operating_point_path, index=False)
    localization = pd.concat(
        [
            stratified_delta_table(frame, ["file_id"], target_pfa).assign(scope="file_id"),
            stratified_delta_table(frame, ["polarization"], target_pfa).assign(scope="polarization"),
            stratified_delta_table(frame, ["range_bin"], target_pfa).assign(scope="range_bin"),
            operating_point.assign(scope="operating_point"),
        ],
        ignore_index=True,
    )
    localization.to_csv(localization_path, index=False)
    pd.DataFrame(
        [
            {"method": "scalar", **observed_metrics["scalar"]},
            {"method": "feature", **observed_metrics["feature"]},
            {"method": "feature_minus_scalar", **observed_deltas},
            {
                "method": "bootstrap_feature_minus_scalar",
                **bootstrap,
            },
        ]
    ).to_csv(observed_path, index=False)
    mode_summary_path = output_dir / "null_mode_summary.csv"
    mode_summary.to_csv(mode_summary_path, index=False)

    report_lines = [
        "# IPIX null controls",
        "",
        f"- input: `{args.input}`",
        f"- summary: `{args.summary}`",
        f"- target pfa: `{target_pfa}`",
        f"- replicates: `{int(args.replicates)}`",
        f"- seed: `{int(args.seed)}`",
        "",
        "## Observed metrics",
        "",
        format_table(
            pd.DataFrame(
                [
                    {"method": "scalar", **observed_metrics["scalar"]},
                    {"method": "feature", **observed_metrics["feature"]},
                    {"method": "feature_minus_scalar", **observed_deltas},
                    {"method": "bootstrap_feature_minus_scalar", **bootstrap},
                ]
            ),
            [
                "method",
                "macro_pfa",
                "macro_absolute_log10_pfa_error",
                "macro_series_factor2_violation_rate",
                "macro_primary_pd",
                "paired_mean_acquisition_error_difference",
                "paired_error_difference_95ci",
                "paired_mean_primary_pd_difference",
                "paired_primary_pd_difference_95ci",
            ],
        ),
        "",
        "## Null modes",
        "",
        format_table(
            mode_summary,
            [
                "mode",
                "macro_absolute_log10_pfa_error_mean",
                "macro_absolute_log10_pfa_error_low",
                "macro_absolute_log10_pfa_error_high",
                "macro_primary_pd_mean",
                "macro_primary_pd_low",
                "macro_primary_pd_high",
                "paired_error_delta_vs_scalar_mean",
                "paired_error_delta_vs_scalar_low",
                "paired_error_delta_vs_scalar_high",
                "paired_pd_delta_vs_scalar_mean",
                "paired_pd_delta_vs_scalar_low",
                "paired_pd_delta_vs_scalar_high",
                "p_error_better_than_observed",
                "p_pd_better_than_observed",
            ],
        ),
        "",
        "## Failure localization",
        "",
        "### Stratified",
        "",
        format_table(
            localization,
            [
                "scope",
                "file_id",
                "polarization",
                "range_bin",
                "error_delta",
                "pd_delta",
            ],
        ),
        "",
        "## Series-level failure localization",
        "",
        format_table(
            series_localization.sort_values(
                ["role", "absolute_log10_pfa_error_delta", "rate_delta"],
                ascending=[True, False, False],
            ),
            [
                "file_id",
                "polarization",
                "range_bin",
                "role",
                "feature_rate",
                "scalar_rate",
                "rate_delta",
                "feature_jeffreys_rate",
                "scalar_jeffreys_rate",
                "jeffreys_delta",
                "feature_factor2_violation",
                "scalar_factor2_violation",
                "factor2_delta",
                "feature_absolute_log10_pfa_error",
                "scalar_absolute_log10_pfa_error",
                "absolute_log10_pfa_error_delta",
            ],
        ),
        "",
        "### Operating point",
        "",
        format_table(
            operating_point,
            [
                "scale_factor",
                "macro_pfa",
                "macro_absolute_log10_pfa_error",
                "macro_series_factor2_violation_rate",
                "macro_primary_pd",
            ],
        ),
    ]
    report_path = output_dir / "summary.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    payload = {
        "status": "BCDRCFAR_IPIX_NULL_CONTROLS_COMPLETE",
        "input_sha256": sha256(args.input),
        "summary_sha256": sha256(args.summary),
        "target_pfa": target_pfa,
        "replicates": int(args.replicates),
        "seed": int(args.seed),
        "observed": observed_metrics,
        "observed_deltas": observed_deltas,
        "bootstrap": bootstrap,
        "output_files": {
            "samples_csv": str(samples_path),
            "failure_localization_csv": str(localization_path),
            "seed_sensitivity_csv": str(seed_sensitivity_path),
            "observed_metrics_csv": str(observed_path),
            "series_metrics_scalar_csv": str(series_metrics_path.with_name("series_metrics_scalar.csv")),
            "series_metrics_feature_csv": str(series_metrics_path.with_name("series_metrics_feature.csv")),
            "series_failure_localization_csv": str(series_localization_path),
            "operating_point_summary_csv": str(operating_point_path),
            "null_mode_summary_csv": str(mode_summary_path),
            "summary_md": str(report_path),
        },
        "null_mode_summary": mode_summary.to_dict(orient="records"),
        "operating_point_summary": operating_point.to_dict(orient="records"),
        "series_failure_localization": series_localization.head(100).to_dict(orient="records"),
    }
    write_json(output_dir / "summary.json", payload)
    print(report_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
