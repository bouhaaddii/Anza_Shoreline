#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the manuscript rate summary and Figure 7 from saved transect rates.

This script aggregates the archived ``rates_<method>.csv`` files only. It does
not rerun shoreline extraction, CNN inference, or the transect calculations.
The manuscript compares methods descriptively because transect observations
are paired across methods and may be spatially autocorrelated.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
ANA = BASE / "outputs" / "change" / "analysis"
TABLES = BASE / "outputs" / "change" / "tables"
FIG_DIR = BASE / "outputs" / "extraction" / "figures"
MODELS = ["classical", "unet", "segnet", "fcn"]
LABELS = {
    "classical": "Classical (MNDWI)",
    "unet": "U-Net",
    "segnet": "SegNet",
    "fcn": "FCN",
}
COLORS = {
    "classical": "#465362",
    "unet": "#2878A5",
    "segnet": "#E69F00",
    "fcn": "#8E6C9F",
}
FIG_NAME = "Fig13_article_rate_distributions"


def load_saved_rates():
    rates = {}
    for model in MODELS:
        path = ANA / f"rates_{model}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing archived rates: {path}")
        frame = pd.read_csv(path).sort_values("transect_id").reset_index(drop=True)
        required = {"transect_id", "n_obs", "nsm", "epr", "lrr", "theil_sen",
                    "volume_m3", "mass_t"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
        if frame["transect_id"].duplicated().any():
            raise ValueError(f"Duplicate transect IDs in {path.name}")
        rates[model] = frame

    expected = rates[MODELS[0]]["transect_id"].tolist()
    for model in MODELS[1:]:
        if rates[model]["transect_id"].tolist() != expected:
            raise ValueError("The four saved files do not contain the same transect IDs")
    if len(expected) != 40:
        raise ValueError(f"Expected the manuscript's 40 common transects, found {len(expected)}")
    if any(not (frame["n_obs"] == 32).all() for frame in rates.values()):
        raise ValueError("Expected 32 seasonal shoreline observations per transect")
    return rates


def summary_rows(rates):
    rows = []
    for model in MODELS:
        d = rates[model]
        epr = d["epr"].dropna()
        rows.append({
            "model": LABELS[model],
            "n_transects": int(epr.size),
            "epr_mean": epr.mean(),
            "epr_std": epr.std(ddof=1),
            "epr_median": epr.median(),
            "nsm_mean": d["nsm"].mean(),
            "lrr_mean": d["lrr"].mean(),
            "theil_sen_mean": d["theil_sen"].mean(),
            "pct_erosion": (epr < 0).mean() * 100,
            "pct_accretion": (epr > 0).mean() * 100,
            "pct_epr_zero": (epr == 0).mean() * 100,
            "net_volume_m3": d["volume_m3"].sum(),
            "erosion_m3": d.loc[d["volume_m3"] < 0, "volume_m3"].sum(),
            "accretion_m3": d.loc[d["volume_m3"] > 0, "volume_m3"].sum(),
            "net_mass_t": d["mass_t"].sum(),
        })
    return pd.DataFrame(rows)


def descriptive_rows(rates):
    rows = []
    for model in MODELS:
        epr = rates[model]["epr"].dropna()
        rows.append({
            "model": LABELS[model],
            "epr_count": int(epr.size),
            "epr_mean": epr.mean(),
            "epr_std": epr.std(ddof=1),
            "epr_median": epr.median(),
            "epr_min": epr.min(),
            "epr_max": epr.max(),
            "epr_q25": epr.quantile(0.25),
            "epr_q75": epr.quantile(0.75),
            "epr_skew": epr.skew(),
            "epr_kurtosis": epr.kurt(),
            "pct_erosion": (epr < 0).mean() * 100,
            "pct_accretion": (epr > 0).mean() * 100,
            "pct_epr_zero": (epr == 0).mean() * 100,
        })
    return pd.DataFrame(rows)


def make_figure(rates):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#68727D",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25), sharex=True,
                             constrained_layout=True)
    positions = np.arange(1, len(MODELS) + 1)
    names = [f"{LABELS[m]}\n(n = 40)" for m in MODELS]
    for ax, field, panel, title in (
        (axes[0], "epr", "a", "End Point Rate"),
        (axes[1], "lrr", "b", "Linear Regression Rate"),
    ):
        values = [rates[m][field].dropna().to_numpy() for m in MODELS]
        violins = ax.violinplot(values, positions=positions, widths=0.76,
                                showmeans=False, showmedians=False,
                                showextrema=False, bw_method="scott")
        for body, model in zip(violins["bodies"], MODELS):
            body.set_facecolor(COLORS[model])
            body.set_edgecolor("#303840")
            body.set_linewidth(0.8)
            body.set_alpha(0.72)
        box = ax.boxplot(values, positions=positions, widths=0.18,
                         patch_artist=True, showfliers=False,
                         medianprops={"color": "white", "linewidth": 1.5},
                         whiskerprops={"color": "#303840", "linewidth": 1.0},
                         capprops={"color": "#303840", "linewidth": 1.0},
                         boxprops={"edgecolor": "#303840", "linewidth": 0.8})
        for patch in box["boxes"]:
            patch.set_facecolor("#FFFFFF")
            patch.set_alpha(0.78)
        ax.axhline(0, color="#303840", linewidth=0.8, linestyle=(0, (2, 2)))
        ax.set_xticks(positions, names)
        ax.set_ylabel(f"{title} (m yr$^{{-1}}$)")
        ax.set_title(f"({panel}) {title}", loc="left")
        ax.grid(axis="y", color="#D9E0E5", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.08)
    fig.suptitle("Change-rate distributions across extraction methods",
                 fontsize=11, fontweight="bold")
    png = FIG_DIR / f"{FIG_NAME}.png"
    pdf = FIG_DIR / f"{FIG_NAME}.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def update_manifest(png):
    path = FIG_DIR / "figures_manifest.json"
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    entries = [entry for entry in entries if entry.get("figure") != FIG_NAME]
    entries.append({
        "figure": FIG_NAME,
        "file": png.name,
        "caption": (
            "Per-transect (a) End Point Rate and (b) Linear Regression Rate "
            "for four shoreline-extraction methods on the same 40 transects. "
            "Violins show distributions; box plots show the median and "
            "interquartile range. Values are descriptive; no between-method "
            "significance test is reported because outputs are paired and "
            "transects may be spatially autocorrelated."
        ),
    })
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rates = load_saved_rates()
    summary_rows(rates).round(4).to_csv(
        TABLES / "rates_summary_by_model.csv", index=False)
    descriptive_rows(rates).round(4).to_csv(
        TABLES / "advanced_statistics.csv", index=False)
    png, pdf = make_figure(rates)
    update_manifest(png)
    print("Summaries and figure created from the saved 40-transect rate files.")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
