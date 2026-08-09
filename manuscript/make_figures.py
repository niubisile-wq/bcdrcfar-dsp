from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

TARGET_PFA = 0.01

COLORS = {
    "go": "#6f7c85",
    "scalar": "#a57955",
    "feature": "#2f6f73",
    "bc": "#2f6f73",
    "accent": "#c9503f",
    "light": "#e8f0ef",
    "dark": "#263238",
    "line": "#9aa5a8",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.dpi": 160,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout(pad=0.8)
    for ext in ["pdf", "svg", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(OUT / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def rounded_box(ax, xy, w, h, text, fc="#f7f8f8", ec="#6f7c85", lw=1.0):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        fc=fc,
        ec=ec,
        lw=lw,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center")
    return box


def arrow(ax, start, end, color="#5d686d", lw=1.1):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def fig1_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.50,
        0.965,
        "BC-DRCFAR: same-time background conditions both score and threshold",
        ha="center",
        va="top",
        fontsize=9,
        weight="bold",
        color=COLORS["dark"],
    )
    ax.text(0.15, 0.86, "Inputs", ha="center", weight="bold", color=COLORS["dark"])
    ax.text(0.51, 0.86, "BC-DRCFAR", ha="center", weight="bold", color=COLORS["dark"])
    ax.text(0.91, 0.86, "Decision", ha="center", weight="bold", color=COLORS["dark"])

    rounded_box(ax, (0.04, 0.66), 0.22, 0.12, "Cell under test\n$x$", fc="#fff7ee", ec=COLORS["accent"])
    rounded_box(
        ax,
        (0.04, 0.40),
        0.22,
        0.16,
        "Same-time\nreference cells\n$R=\\{r_i\\}_{i=1}^{K}$",
        fc="#eef4f3",
        ec=COLORS["bc"],
    )
    rounded_box(ax, (0.04, 0.20), 0.22, 0.12, "Declared false-alarm\nrate $\\alpha$", fc="#f4f4f4", ec=COLORS["dark"])

    rounded_box(
        ax,
        (0.34, 0.34),
        0.24,
        0.26,
        "Background\nrepresentation\n$\\phi(R)$\n\nlocal distribution\nclutter dynamics\ntail shape\nreference series",
        fc="#eef4f3",
        ec=COLORS["bc"],
    )
    rounded_box(ax, (0.64, 0.61), 0.18, 0.14, "Score branch\n$s(x,R)$", fc="#fff7ee", ec=COLORS["accent"])
    rounded_box(ax, (0.64, 0.28), 0.18, 0.18, "Threshold branch\n$\\tau(R;\\alpha)$", fc="#eef4f3", ec=COLORS["bc"])
    rounded_box(
        ax,
        (0.86, 0.40),
        0.12,
        0.22,
        "CFAR\n$s(x,R)>\\tau(R;\\alpha)$\n\ntarget / clutter",
        fc="#f4f4f4",
        ec=COLORS["dark"],
        lw=1.2,
    )

    arrow(ax, (0.26, 0.72), (0.64, 0.68), COLORS["accent"])
    arrow(ax, (0.26, 0.48), (0.34, 0.48), COLORS["bc"])
    arrow(ax, (0.58, 0.52), (0.64, 0.68), COLORS["bc"])
    arrow(ax, (0.58, 0.42), (0.64, 0.37), COLORS["bc"])
    arrow(ax, (0.26, 0.26), (0.64, 0.33), COLORS["dark"])
    arrow(ax, (0.82, 0.68), (0.86, 0.54), COLORS["accent"])
    arrow(ax, (0.82, 0.37), (0.86, 0.48), COLORS["bc"])

    ax.annotate(
        "innovation:\nbackground-conditioned\nthreshold",
        xy=(0.73, 0.30),
        xytext=(0.73, 0.16),
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["bc"],
        arrowprops=dict(arrowstyle="-|>", color=COLORS["bc"], lw=1.1, shrinkA=3, shrinkB=5),
    )
    ax.text(
        0.50,
        0.07,
        "Key idea: the reference window is learned as context, not reduced to a single noise estimate.",
        ha="center",
        color=COLORS["dark"],
    )
    save(fig, "fig1_method_overview")


def fig2_synthetic_calibration() -> None:
    metrics = [
        ("Factor-2 violations", [83.3, 18.8], "%"),
        ("Mean absolute\nlog$_{10}$-$P_{fa}$ error", [1.070, 0.174], ""),
        ("$P_d$ at 0 dB", [0.740, 0.835], ""),
    ]
    methods = ["GO-CFAR", "BC-DRCFAR"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (title, vals, unit), lab in zip(axes, metrics, ["a", "b", "c"]):
        bars = ax.bar([0, 1], vals, color=[COLORS["go"], COLORS["bc"]], width=0.58)
        ax.set_xticks([0, 1], methods, rotation=20, ha="right")
        ax.set_title(title, pad=5)
        ymax = max(vals) * 1.35
        ax.set_ylim(0, ymax)
        for b, v in zip(bars, vals):
            suffix = unit
            label = f"{v:.1f}{suffix}" if unit == "%" else f"{v:.3f}"
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.035, label, ha="center", va="bottom", fontsize=7)
        panel_label(ax, lab)
    fig.suptitle("Synthetic benchmark: calibration reliability improves without sacrificing detection", y=1.02, fontsize=8.5)
    save(fig, "fig2_synthetic_calibration")


def fig3_ipix_acquisition() -> None:
    base = ROOT / "results" / "bcdrcfar_ipix" / "retrospective_external_featurehead"
    acq = pd.read_csv(base / "acquisition_metrics.csv")
    target = pd.read_csv(base / "target_metrics.csv")
    keep = ["go_cfar", "bcdrcfar_scalar", "bcdrcfar_feature"]
    labels = {"go_cfar": "GO-CFAR", "bcdrcfar_scalar": "Scalar", "bcdrcfar_feature": "Feature head"}
    colors = {"go_cfar": COLORS["go"], "bcdrcfar_scalar": COLORS["scalar"], "bcdrcfar_feature": COLORS["feature"]}
    file_ids = sorted(acq["file_id"].astype(str).unique(), key=lambda x: int(x))
    x = np.arange(len(file_ids))

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    ax = axes[0]
    for method in keep:
        sub = acq[acq["method"] == method].copy()
        sub["file_id"] = sub["file_id"].astype(str)
        vals = [sub.loc[sub["file_id"] == f, "absolute_log10_pfa_error"].iloc[0] for f in file_ids]
        ax.plot(x, vals, marker="o", ms=3.2, lw=1.1, color=colors[method])
        ax.text(x[-1] + 0.22, vals[-1], labels[method], color=colors[method], va="center", fontsize=6)
    ax.set_xlim(-0.4, len(file_ids) - 0.15)
    ax.set_xticks(x, file_ids, rotation=0)
    ax.set_xlabel("IPIX acquisition")
    ax.set_ylabel("Abs. log$_{10}$-$P_{fa}$ error")
    ax.set_title("Acquisition-level calibration")
    panel_label(ax, "a")

    ax = axes[1]
    width = 0.25
    for j, method in enumerate(keep):
        sub = acq[acq["method"] == method].copy()
        sub["file_id"] = sub["file_id"].astype(str)
        vals = [sub.loc[sub["file_id"] == f, "pfa"].iloc[0] for f in file_ids]
        ax.scatter(x + (j - 1) * width, vals, color=colors[method], s=16, label=labels[method], zorder=3)
    ax.axhline(TARGET_PFA, color=COLORS["dark"], lw=0.8, ls="--")
    ax.axhspan(TARGET_PFA / 2, TARGET_PFA * 2, color=COLORS["light"], zorder=0)
    ax.set_yscale("log")
    ax.set_xticks(x, file_ids)
    ax.set_title("Declared $P_{fa}$ band")
    ax.set_ylabel("Empirical $P_{fa}$")
    panel_label(ax, "b")

    ax = axes[2]
    primary = target[(target["role"] == "primary") & (target["method"].isin(keep))]
    means = primary.groupby("method")["pd"].mean().reindex(keep)
    bars = ax.bar(np.arange(len(keep)), means.values, color=[colors[m] for m in keep], width=0.62)
    ax.set_xticks(np.arange(len(keep)), [labels[m] for m in keep], rotation=20, ha="right")
    ax.set_ylabel("$P_d$")
    ax.set_title("Detection trade-off")
    ax.set_ylim(0, max(means.values) * 1.35)
    for b, v in zip(bars, means.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    panel_label(ax, "c")
    save(fig, "fig3_ipix_acquisition")


def fig4_mechanism_audit() -> None:
    ab = pd.read_csv(ROOT / "results" / "bcdrcfar_ipix" / "feature_ablation" / "ablation.csv")
    lh = pd.read_csv(ROOT / "results" / "bcdrcfar_ipix" / "long_horizon_holdout" / "cohort_summary.csv")
    dom = pd.read_csv(ROOT / "reports" / "BCDRCFAR_DSP_跨域偏移审计_20260808.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1))
    ax = axes[0, 0]
    order = ["log_scale_only", "no_anchor", "no_polarization", "background_core", "all"]
    labels = ["log only", "no anchor", "no pol.", "core", "all"]
    sub = ab.set_index("family").loc[order]
    ax.bar(np.arange(len(order)), sub["ret_macro_pfa"], color=[COLORS["go"], COLORS["accent"], COLORS["scalar"], "#7fa7a9", COLORS["feature"]])
    ax.axhline(TARGET_PFA, color=COLORS["dark"], lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(order)), labels, rotation=25, ha="right")
    ax.set_ylabel("Retrospective macro $P_{fa}$")
    ax.set_title("Background-feature ablation")
    panel_label(ax, "a")

    ax = axes[0, 1]
    cohorts = ["early", "late"]
    for method, color, label in [("scalar", COLORS["scalar"], "Scalar"), ("feature", COLORS["feature"], "Feature head")]:
        vals = [lh[(lh["method"] == method) & (lh["cohort"] == c)]["macro_absolute_log10_pfa_error"].iloc[0] for c in cohorts]
        ax.plot(cohorts, vals, marker="o", lw=1.2, color=color, label=label)
    ax.set_ylabel("Abs. log$_{10}$-$P_{fa}$ error")
    ax.set_title("Long-horizon drift audit")
    ax.legend(fontsize=6)
    panel_label(ax, "b")

    ax = axes[1, 0]
    dom_order = ["IPIX", "IPIX_269_high_sea", "IPIX_287_low_sea", "St_Andrews_24GHz", "St_Andrews_94GHz"]
    dsub = dom.set_index("domain").loc[dom_order]
    colors = [COLORS["feature"] if g == "ACCEPT" else COLORS["go"] for g in dsub["domain_gate"]]
    ax.bar(np.arange(len(dom_order)), dsub["support_fraction_outside_any_synthetic_1_99pct_bound"], color=colors)
    ax.set_xticks(np.arange(len(dom_order)), ["IPIX", "269 high", "287 low", "StA 24", "StA 94"], rotation=25, ha="right")
    ax.set_ylabel("Outside synthetic\n1--99% support")
    ax.set_title("Domain-gate traceability")
    for i, gate in enumerate(dsub["domain_gate"]):
        ax.text(i, dsub.iloc[i]["support_fraction_outside_any_synthetic_1_99pct_bound"] + 0.025, gate, ha="center", fontsize=6)
    panel_label(ax, "c")

    ax = axes[1, 1]
    vals = [171, 148]
    bars = ax.bar([0, 1], vals, color=[COLORS["go"], COLORS["feature"]], width=0.55)
    ax.set_xticks([0, 1], ["Indexed\nchunks", "Decoded\npayloads"])
    ax.set_ylabel("Chunk count")
    ax.set_title("Raw-data reproducibility chain")
    ax.set_ylim(0, 190)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", va="bottom")
    ax.text(0.5, 0.12, "Birmingham 626 prefix audit", transform=ax.transAxes, ha="center", color=COLORS["line"], fontsize=6)
    panel_label(ax, "d")
    save(fig, "fig4_mechanism_audit")


def fig5_method_position() -> None:
    rows = [
        "Classical CFAR",
        "Statistical CFAR",
        "Feature detectors",
        "Deep detectors",
        "Copula CFAR",
        "BC-DRCFAR",
    ]
    cols = [
        "$P_{fa}$\ninterface",
        "Background\nuse",
        "Evidence\nscore",
        "Threshold\ncalibration",
        "Acquisition\nreliability",
    ]
    cells = [
        ["explicit", "local", "limited", "fixed", "indirect"],
        ["explicit", "model", "limited", "model-based", "partial"],
        ["indirect", "features", "strong", "separate", "limited"],
        ["indirect", "learned", "strong", "separate", "limited"],
        ["explicit", "dependence", "moderate", "probability", "partial"],
        ["explicit", "descriptors", "strong", "conditioned", "explicit"],
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.98, "Method position: BC-DRCFAR adds background-conditioned calibration", ha="center", va="top", fontsize=8.5)

    left = 0.19
    top = 0.18
    bottom = 0.08
    right = 0.98
    n_rows = len(rows)
    n_cols = len(cols)
    cw = (right - left) / n_cols
    ch = (1 - top - bottom) / n_rows

    for j, col in enumerate(cols):
        x = left + j * cw + cw / 2
        ax.text(x, 1 - top + 0.015, col, ha="center", va="bottom", fontsize=6.8, fontweight="bold")

    for i, row in enumerate(rows):
        y = 1 - top - (i + 1) * ch
        is_bc = row == "BC-DRCFAR"
        ax.text(left - 0.02, y + ch / 2, row, ha="right", va="center", fontsize=6.8, fontweight="bold" if is_bc else "normal")
        for j in range(n_cols):
            x = left + j * cw
            if is_bc:
                fc = COLORS["feature"]
                tc = "white"
                weight = "bold"
            else:
                fc = "#eef4f3" if j in (1, 3) else "#f7f8f8"
                tc = COLORS["dark"]
                weight = "normal"
            ax.add_patch(plt.Rectangle((x, y), cw - 0.004, ch - 0.006, fc=fc, ec="white", lw=1.0))
            ax.text(x + cw / 2, y + ch / 2, cells[i][j], ha="center", va="center", fontsize=6.4, color=tc, fontweight=weight)

    ax.text(
        left,
        0.025,
        "BC-DRCFAR keeps the declared false-alarm interface and moves learned background representation into the threshold-calibration step.",
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=COLORS["feature"],
    )
    save(fig, "fig5_method_position")


def main() -> None:
    # Fig. 1 is maintained from the author-approved PowerPoint schematic.
    # Do not regenerate it here, otherwise the manuscript will revert to the draft schematic.
    fig2_synthetic_calibration()
    fig3_ipix_acquisition()
    fig4_mechanism_audit()
    fig5_method_position()


if __name__ == "__main__":
    main()
