"""Summarize cross-domain / holdout / negative-control pressure evidence for BC-DRCFAR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
OUT_MD = OUT_DIR / "BCDRCFAR_DSP_压力矩阵_20260808.md"
OUT_CSV = OUT_DIR / "BCDRCFAR_DSP_压力矩阵_20260808.csv"
OUT_JSON = OUT_DIR / "BCDRCFAR_DSP_压力矩阵_20260808.json"

INPUTS = {
    "p4_domain_reliability": ROOT / "results" / "p4_domain_reliability" / "summary.json",
    "p4_real_confirmatory": ROOT / "results" / "p4_real_confirmatory" / "decision.json",
    "p4_ipix_scan_domain": ROOT / "results" / "p4_ipix_scan_domain" / "failure_audit.json",
    "p4_st_andrews_holdout": ROOT / "results" / "p4_st_andrews_holdout" / "summary.json",
    "p5_nexrad_negative_control": ROOT / "results" / "p5_nexrad_negative_control" / "summary.json",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    return parser.parse_args(argv)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def make_rows() -> list[dict[str, Any]]:
    domain = read_json(INPUTS["p4_domain_reliability"])
    real = read_json(INPUTS["p4_real_confirmatory"])
    scan = read_json(INPUTS["p4_ipix_scan_domain"])
    st = read_json(INPUTS["p4_st_andrews_holdout"])
    nexrad = read_json(INPUTS["p5_nexrad_negative_control"])

    rows: list[dict[str, Any]] = []
    for entry in domain["domain_metrics"]:
        rows.append(
            {
                "axis": "domain_reliability",
                "artifact": entry["domain"],
                "gate": entry["domain_gate"],
                "pass_signal": "ACCEPT" if entry["domain_gate"] == "ACCEPT" else "ABSTAIN",
                "evidence": f"coverage={entry['reliable_coverage']:.3f}, auroc={entry['reliable_risk_auroc']:.3f}",
                "boundary": domain["claim_boundary"],
                "next_step": domain["next_requirement"],
            }
        )

    rows.append(
        {
            "axis": "confirmatory_real",
            "artifact": "P4_real_confirmatory",
            "gate": real["p4_real_gate"],
            "pass_signal": "NO_GO" if real["p4_real_gate"] == "NO_GO" else "PASS",
            "evidence": (
                f"gain_over_ood={real['macro_auroc_gain_over_multihead_ood']:.3f}, "
                f"gain_ci_low={real['macro_auroc_gain_ci'][0]:.3f}, "
                f"positive_acq={real['positive_auroc_acquisitions']}, "
                f"rate_ratio={real['geometric_mean_within_series_rate_ratio']:.3f}"
            ),
            "boundary": "partial external transfer, not a full P4 pass",
            "next_step": "Treat as partial external transfer only; do not retune on the confirmatory set.",
        }
    )

    rows.append(
        {
            "axis": "scan_domain",
            "artifact": "IPIX scan domain",
            "gate": scan["frozen_decision"]["domain_gate_confirmation"],
            "pass_signal": "NO_GO" if scan["frozen_decision"]["domain_gate_confirmation"] == "NO_GO" else "PASS",
            "evidence": (
                f"observed_action={scan['frozen_decision']['observed_action']}, "
                f"expected_action={scan['frozen_decision']['expected_action']}, "
                f"coverage={scan['frozen_decision']['reliable_coverage']:.3f}"
            ),
            "boundary": scan["frozen_decision"]["claim_boundary"],
            "next_step": scan["next_method_requirement"],
        }
    )

    rows.append(
        {
            "axis": "st_andrews_holdout",
            "artifact": "St Andrews holdout",
            "gate": st["llm_gate"],
            "pass_signal": "CLOSED",
            "evidence": (
                f"24GHz direct_risk_auroc={st['frequency_metrics'][0]['direct_risk_block_any_event_auroc']:.3f}, "
                f"94GHz direct_risk_auroc={st['frequency_metrics'][1]['direct_risk_block_any_event_auroc']:.3f}, "
                f"ratio_ci_24={st['risk_rate_ratio_ci']['24GHz']}, ratio_ci_94={st['risk_rate_ratio_ci']['94GHz']}"
            ),
            "boundary": st["claim_boundary"],
            "next_step": "Treat as exploratory untouched-prefix holdout only; it does not open the gate.",
        }
    )

    rows.append(
        {
            "axis": "negative_control",
            "artifact": "NEXRAD negative control",
            "gate": nexrad["contract_result"]["contract_action"],
            "pass_signal": "ABSTAIN",
            "evidence": (
                f"failed_criteria={','.join(nexrad['contract_result']['failed_criteria'])}, "
                f"observed_action={nexrad['integrity_and_gate_checks']['observed_action_equals_frozen_expectation']}"
            ),
            "boundary": nexrad["claim_boundary"],
            "next_step": "Use as contract-proof that unsupported radar products must still ABSTAIN.",
        }
    )
    return rows


def table_md(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_empty_"
    cols = ["axis", "artifact", "gate", "pass_signal", "evidence", "boundary", "next_step"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    out_md = args.out_md.resolve()
    out_csv = args.out_csv.resolve()
    out_json = args.out_json.resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    rows = make_rows()
    out_csv.write_text(
        "axis,artifact,gate,pass_signal,evidence,boundary,next_step\n"
        + "\n".join(
            ",".join(
                '"' + str(row.get(col, "")).replace('"', '""') + '"'
                for col in ["axis", "artifact", "gate", "pass_signal", "evidence", "boundary", "next_step"]
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    report = [
        "# BCDRCFAR pressure matrix",
        "",
        "This matrix separates true pass/fail evidence from boundary-only evidence.",
        "",
        table_md(rows),
        "",
        "## Reading",
        "",
        "- `ACCEPT` here means the domain-reliability gate accepted the domain, not that CFAR is solved.",
        "- `NO_GO` and `ABSTAIN` are preserved as boundaries, not rewritten as weaker positives.",
        "- The St Andrews and NEXRAD rows are pressure tests: useful for delimiting scope, not for opening the confirmatory gate.",
    ]
    out_md.write_text("\n".join(report) + "\n", encoding="utf-8")

    payload = {
        "status": "BCDRCFAR_PRESSURE_MATRIX_SUMMARY_COMPLETE",
        "rows": rows,
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
