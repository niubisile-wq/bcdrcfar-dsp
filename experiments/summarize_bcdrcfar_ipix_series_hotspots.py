"""Summarize the strongest series-level failure hotspots for retrospective IPIX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "bcdrcfar_ipix"
    / "retrospective_external_featurehead"
    / "null_controls"
    / "series_failure_localization.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "bcdrcfar_ipix"
    / "retrospective_external_featurehead"
    / "null_controls"
    / "series_hotspots"
)


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
        if isinstance(value, float):
            return None if pd.isna(value) else value
        if isinstance(value, (pd.Series, pd.Index)):
            return [safe(item) for item in value.tolist()]
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        return value

    path.write_text(
        json.dumps(safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args(argv)


def weighted_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        weights = group["absolute_log10_pfa_error_delta"].abs().fillna(0.0)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "rows": int(len(group)),
                "clutter_rows": int((group["role"] == "clutter").sum()),
                "secondary_rows": int((group["role"] == "secondary").sum()),
                "primary_rows": int((group["role"] == "primary").sum()),
                "mean_abs_delta": float(weights.mean()) if len(weights) else float("nan"),
                "median_abs_delta": float(weights.median()) if len(weights) else float("nan"),
                "max_abs_delta": float(weights.max()) if len(weights) else float("nan"),
                "mean_rate_delta": float(group["rate_delta"].mean()) if not group.empty else float("nan"),
                "mean_jeffreys_delta": float(group["jeffreys_delta"].mean()) if not group.empty else float("nan"),
                "factor2_delta_sum": float(group["factor2_delta"].sum()) if not group.empty else float("nan"),
                "feature_factor2_rate": float(group["feature_factor2_violation"].mean()) if not group.empty else float("nan"),
                "scalar_factor2_rate": float(group["scalar_factor2_violation"].mean()) if not group.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def top_rows(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    hot = frame.copy()
    hot["abs_delta"] = hot["absolute_log10_pfa_error_delta"].abs()
    cols = [
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
        "abs_delta",
    ]
    return hot.sort_values(["abs_delta", "rate_delta"], ascending=[False, False]).head(int(top_n))[cols]


def format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_empty_"
    return frame[columns].to_markdown(index=False)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    frame["file_id"] = frame["file_id"].astype(str)
    frame["polarization"] = frame["polarization"].astype(str)
    frame["role"] = frame["role"].astype(str)

    clutter = frame[frame["role"] == "clutter"].copy()
    hotspots = weighted_summary(clutter, ["file_id", "polarization", "range_bin"]).sort_values(
        ["mean_abs_delta", "max_abs_delta"], ascending=[False, False]
    )
    by_file = weighted_summary(clutter, ["file_id"]).sort_values("mean_abs_delta", ascending=False)
    by_pol = weighted_summary(clutter, ["polarization"]).sort_values("mean_abs_delta", ascending=False)
    by_range = weighted_summary(clutter, ["range_bin"]).sort_values("mean_abs_delta", ascending=False)
    top = top_rows(clutter, int(args.top_n))

    hotspots_path = output_dir / "series_hotspots.csv"
    hotspots.to_csv(hotspots_path, index=False)
    by_file.to_csv(output_dir / "by_file.csv", index=False)
    by_pol.to_csv(output_dir / "by_polarization.csv", index=False)
    by_range.to_csv(output_dir / "by_range_bin.csv", index=False)
    top.to_csv(output_dir / "top_rows.csv", index=False)

    report_lines = [
        "# Series-level hotspots",
        "",
        f"- input: `{input_path}`",
        f"- top_n: `{int(args.top_n)}`",
        f"- clutter rows: `{len(clutter)}`",
        "",
        "## Top rows",
        "",
        format_table(
            top,
            [
                "file_id",
                "polarization",
                "range_bin",
                "feature_absolute_log10_pfa_error",
                "scalar_absolute_log10_pfa_error",
                "absolute_log10_pfa_error_delta",
                "feature_factor2_violation",
                "scalar_factor2_violation",
            ],
        ),
        "",
        "## By file",
        "",
        format_table(
            by_file,
            [
                "file_id",
                "rows",
                "mean_abs_delta",
                "median_abs_delta",
                "max_abs_delta",
                "feature_factor2_rate",
                "scalar_factor2_rate",
            ],
        ),
        "",
        "## By polarization",
        "",
        format_table(
            by_pol,
            [
                "polarization",
                "rows",
                "mean_abs_delta",
                "median_abs_delta",
                "max_abs_delta",
                "feature_factor2_rate",
                "scalar_factor2_rate",
            ],
        ),
        "",
        "## By range bin",
        "",
        format_table(
            by_range,
            [
                "range_bin",
                "rows",
                "mean_abs_delta",
                "median_abs_delta",
                "max_abs_delta",
                "feature_factor2_rate",
                "scalar_factor2_rate",
            ],
        ),
    ]
    report_path = output_dir / "summary.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    payload = {
        "status": "BCDRCFAR_IPIX_SERIES_HOTSPOTS_SUMMARY_COMPLETE",
        "input_sha256": sha256(input_path),
        "input_rows": int(len(frame)),
        "clutter_rows": int(len(clutter)),
        "top_n": int(args.top_n),
        "output_files": {
            "series_hotspots_csv": str(hotspots_path),
            "by_file_csv": str(output_dir / "by_file.csv"),
            "by_polarization_csv": str(output_dir / "by_polarization.csv"),
            "by_range_bin_csv": str(output_dir / "by_range_bin.csv"),
            "top_rows_csv": str(output_dir / "top_rows.csv"),
            "summary_md": str(report_path),
        },
        "top_rows": top.head(int(args.top_n)).to_dict(orient="records"),
        "by_file": by_file.head(10).to_dict(orient="records"),
        "by_polarization": by_pol.to_dict(orient="records"),
        "by_range_bin": by_range.to_dict(orient="records"),
    }
    write_json(output_dir / "summary.json", payload)
    print(report_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
