"""Summarize cross-domain shift evidence for BC-DRCFAR.

This report consolidates the existing p4/p5 evidence into a single boundary
map that separates support-domain evidence from gate evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

DOMAIN_SHIFT = ROOT / "results" / "p4_domain_shift_audit" / "summary.json"
DOMAIN_RELIABILITY = ROOT / "results" / "p4_domain_reliability" / "summary.json"
REAL_CONFIRMATORY = ROOT / "results" / "p4_real_confirmatory" / "decision.json"
SCAN_DOMAIN = ROOT / "results" / "p4_ipix_scan_domain" / "decision.json"
ST_ANDREWS = ROOT / "results" / "p4_st_andrews_holdout" / "summary.json"
NEXRAD = ROOT / "results" / "p5_nexrad_negative_control" / "summary.json"

OUT_MD = REPORTS / "BCDRCFAR_DSP_跨域偏移审计_20260808.md"
OUT_CSV = REPORTS / "BCDRCFAR_DSP_跨域偏移审计_20260808.csv"
OUT_JSON = REPORTS / "BCDRCFAR_DSP_跨域偏移审计_20260808.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Iterable[str] | None = None) -> int:
    domain_shift = read_json(DOMAIN_SHIFT)
    domain_rel = read_json(DOMAIN_RELIABILITY)
    real_confirmatory = read_json(REAL_CONFIRMATORY)
    scan_domain = read_json(SCAN_DOMAIN)
    st_andrews = read_json(ST_ANDREWS)
    nexrad = read_json(NEXRAD)

    domain_gate_map = {
        item["domain"]: item["domain_gate"] for item in domain_rel.get("domain_metrics", [])
    }

    rows: list[dict[str, Any]] = []
    for item in domain_shift.get("domain_metrics", []):
        domain = item["domain"]
        support = domain_shift.get("support", {}).get(domain, {})
        rows.append(
            {
                "domain": domain,
                "records": item.get("records"),
                "support_fraction_outside_any_synthetic_1_99pct_bound": support.get(
                    "fraction_outside_any_synthetic_1_99pct_bound"
                ),
                "support_mean_outside_feature_count": support.get("mean_outside_feature_count"),
                "support_median_robust_feature_distance": support.get("median_robust_feature_distance"),
                "support_q95_robust_feature_distance": support.get("q95_robust_feature_distance"),
                "risk_any_event_auroc": item.get("risk_any_event_auroc"),
                "risk_spearman_with_false_alarm_count": item.get("risk_spearman_with_false_alarm_count"),
                "risk_multihead_spearman": item.get("risk_multihead_spearman"),
                "domain_gate": domain_gate_map.get(domain, "UNKNOWN"),
            }
        )

    write_csv(
        OUT_CSV,
        rows,
        [
            "domain",
            "records",
            "support_fraction_outside_any_synthetic_1_99pct_bound",
            "support_mean_outside_feature_count",
            "support_median_robust_feature_distance",
            "support_q95_robust_feature_distance",
            "risk_any_event_auroc",
            "risk_spearman_with_false_alarm_count",
            "risk_multihead_spearman",
            "domain_gate",
        ],
    )

    payload = {
        "domain_shift_adjudication": domain_shift.get("adjudication"),
        "domain_shift_top_five_classifier_features": domain_shift.get("top_five_classifier_features", []),
        "top_feature_event_direction_flip_count": domain_shift.get("top_feature_event_direction_flip_count"),
        "domain_reliability_gate": domain_rel.get("llm_gate"),
        "accepted_domains": domain_rel.get("accepted_domains", []),
        "abstained_domains": domain_rel.get("abstained_domains", []),
        "real_confirmatory_gate": real_confirmatory.get("p4_real_gate"),
        "scan_domain_gate": scan_domain.get("domain_gate_confirmation"),
        "st_andrews_gate": st_andrews.get("llm_gate"),
        "nexrad_gate": nexrad.get("contract_result", {}).get("contract_action"),
        "p4_real_gain_over_multihead_ood": real_confirmatory.get("macro_auroc_gain_over_multihead_ood"),
        "p4_real_gate_ci": real_confirmatory.get("macro_auroc_gain_ci"),
        "st_andrews_direct_risk_auroc": {
            "24GHz": st_andrews.get("frequency_metrics", [{}])[0].get("direct_risk_block_any_event_auroc"),
            "94GHz": st_andrews.get("frequency_metrics", [{}, {}])[1].get("direct_risk_block_any_event_auroc"),
        },
        "nexrad_failed_criteria": nexrad.get("contract_result", {}).get("failed_criteria", []),
        "domain_table": rows,
    }
    write_json(OUT_JSON, payload)

    lines = [
        "# BCDRCFAR Cross-Domain Shift Audit",
        "",
        "## Bottom line",
        "",
        "The score is not domain-stable across the full radar stack. IPIX stays inside the accepted domain family, but St Andrews is only boundary evidence, the confirmatory transfer remains NO_GO, and the semantic scan-domain / NEXRAD checks stay rejected or abstained.",
        "",
        "## Domain support and risk map",
        "",
        "| domain | support outside 1-99% | mean outside features | median robust distance | q95 robust distance | risk AUROC | risk Spearman | multihead Spearman | gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['domain']} | {row['support_fraction_outside_any_synthetic_1_99pct_bound']:.3f} | "
            f"{row['support_mean_outside_feature_count']:.3f} | {row['support_median_robust_feature_distance']:.3f} | "
            f"{row['support_q95_robust_feature_distance']:.3f} | {row['risk_any_event_auroc']:.3f} | "
            f"{row['risk_spearman_with_false_alarm_count']:.3f} | {row['risk_multihead_spearman']:.3f} | {row['domain_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Feature direction flips",
            "",
            f"- top classifier features: `{', '.join(payload['domain_shift_top_five_classifier_features'])}`",
            f"- top feature event-direction flip count: `{payload['top_feature_event_direction_flip_count']}`",
            "",
            "## Gate map",
            "",
            f"- p4 domain reliability: `{payload['domain_reliability_gate']}`",
            f"- p4 real confirmatory: `{payload['real_confirmatory_gate']}`",
            f"- p4 scan domain: `{payload['scan_domain_gate']}`",
            f"- St Andrews holdout: `{payload['st_andrews_gate']}`",
            f"- NEXRAD negative control: `{payload['nexrad_gate']}`",
            "",
            "## Interpretation",
            "",
            "This is the stronger cross-domain statement now available: the method has a bounded IPIX-centric acceptance region, but the same scoring semantics do not transfer as a universal domain-validity rule.",
            "The St Andrews rows remain boundary-only evidence, and the NEXRAD row remains a deliberate ABSTAIN control.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"md": str(OUT_MD), "csv": str(OUT_CSV), "json": str(OUT_JSON)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
