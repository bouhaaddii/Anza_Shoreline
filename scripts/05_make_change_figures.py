#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STUDY 2 - STEP 02: ADVANCED PUBLICATION FIGURES

Draws the figure set from the archived analysis products in
outputs/change/analysis/. No shoreline extraction or change analysis is run.

12 figures (320-dpi PNG + vector PDF) + captions in figures_manifest.json.

Run from the project folder:  python scripts/05_make_change_figures.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.stats import linregress, kendalltau, kruskal, t as t_dist
from visual_style import (ACCRETION, CHANGE_CMAP, EROSION, MODEL_COLORS,
                          STABLE, STYLE)

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    HAS_SCALEBAR = True
except ImportError:
    HAS_SCALEBAR = False

# ---------------------------------------------------------------------------
# PATHS AND STYLE
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs" / "change"
ANA = OUT / "analysis"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

MODELS = ["classical", "unet", "segnet", "fcn"]
MODEL_LABEL = {"classical": "Classical (MNDWI)", "unet": "U-Net",
               "segnet": "SegNet", "fcn": "FCN"}
MODEL_COLOR = MODEL_COLORS
SEASONS = ["winter", "spring", "summer", "autumn"]
ERO, ACC, STA = EROSION, ACCRETION, STABLE

MANIFEST = []

plt.rcParams.update(STYLE)


def save(fig, name, caption):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    MANIFEST.append({"figure": name, "file": f"{name}.png", "caption": caption})
    print(f"  [OK] {name}")


def panel_letter(ax, letter, x=-0.10, y=1.06):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")


def north_arrow(ax, x=0.95, y=0.92):
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.09),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="k"))


def add_scalebar(ax):
    if HAS_SCALEBAR:
        ax.add_artist(ScaleBar(1, units="m", location="lower left",
                               box_alpha=0.7, font_properties={"size": 7}))


def km_axes(ax, nbins=4):
    fmt = FuncFormatter(lambda v, _: f"{v/1000:.1f}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=nbins))
        axis.set_major_formatter(fmt)
    ax.set_xlabel("Easting UTM 29N (km)")
    ax.set_ylabel("Northing (km)")


# ---------------------------------------------------------------------------
# DATA LOADING (archived analysis products)
# ---------------------------------------------------------------------------
def load_products():
    required = [ANA / "transects.shp", ANA / "baseline.shp",
                ANA / "mean_shoreline.shp"]
    required += [ANA / f"{stem}_{m}.csv"
                 for m in MODELS
                 for stem in ("distances", "rates", "annual_volumes", "annual_epr")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing archived analysis results in outputs/change/analysis: "
                         + ", ".join(missing))
    transects = gpd.read_file(ANA / "transects.shp").rename(
        columns={"transect_i": "transect_id", "distance_a": "distance_along"})
    baseline = gpd.read_file(ANA / "baseline.shp").geometry.iloc[0]
    mean_line = gpd.read_file(ANA / "mean_shoreline.shp").geometry.iloc[0]
    dists, rates, ann_v, ann_e = {}, {}, {}, {}
    for m in MODELS:
        d = pd.read_csv(ANA / f"distances_{m}.csv")
        d["date"] = pd.to_datetime(d["date"])
        dists[m] = d
        rates[m] = pd.read_csv(ANA / f"rates_{m}.csv")
        ann_v[m] = pd.read_csv(ANA / f"annual_volumes_{m}.csv")
        ann_e[m] = pd.read_csv(ANA / f"annual_epr_{m}.csv")
    # Use the archived vectors that correspond to the saved change analysis.
    archive = DATA / "coastlines"
    gdfs = {}
    cands = {"classical": archive / "all_coastlines_utm.shp"}
    for m in ["unet", "segnet", "fcn"]:
        cands[m] = archive / f"all_coastlines_{m}_utm.shp"
    missing = [str(p) for p in cands.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing archived shoreline results in data/coastlines: "
                         + ", ".join(missing))
    for m, path in cands.items():
        g = gpd.read_file(path)
        g["date"] = pd.to_datetime(g["date"])
        g["year"] = g["date"].dt.year
        g["season"] = g["season"].str.lower()
        gdfs[m] = g.sort_values("date").reset_index(drop=True)
    return transects, baseline, mean_line, dists, rates, ann_v, ann_e, gdfs


# ---------------------------------------------------------------------------
# FIGURES 1-8 (corrected versions)
# ---------------------------------------------------------------------------
def fig01_framework(gdfs, transects, baseline, mean_line, rates):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)
    for _, row in gdfs["unet"].iterrows():
        ax1.plot(*row.geometry.xy, color="#0072B2", lw=0.5, alpha=0.35)
    ax1.plot(*baseline.xy, color="k", lw=1.5, label="Baseline (200 m landward)")
    ax1.plot(*mean_line.xy, color="#D55E00", lw=1.0, ls="--",
             label="Time-mean shoreline")
    for tr in transects.itertuples():
        ax1.plot(*tr.geometry.xy, color="grey", lw=0.5, alpha=0.6)
        if tr.transect_id % 5 == 0 or tr.transect_id == 1:
            x, y = tr.geometry.coords[-1]
            ax1.annotate(str(tr.transect_id), (x, y), fontsize=6, color="dimgrey")
    ax1.set_aspect("equal")
    ax1.set_title("Transect framework (U-Net shorelines)")
    km_axes(ax1)
    ax1.legend(loc="upper left", fontsize=7)
    add_scalebar(ax1); north_arrow(ax1)
    panel_letter(ax1, "a", x=0.0, y=1.10)

    gt = transects.merge(rates["unet"][["transect_id", "epr"]], on="transect_id")
    vmax = np.nanpercentile(np.abs(gt["epr"]), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    gt.plot(ax=ax2, column="epr", cmap=CHANGE_CMAP, norm=norm, linewidth=2.4)
    last = gdfs["unet"].iloc[-1]
    ax2.plot(*last.geometry.xy, color="k", lw=0.9, ls="--",
             label="Shoreline autumn 2025")
    sm = ScalarMappable(norm=norm, cmap=CHANGE_CMAP)
    cbar = fig.colorbar(sm, ax=ax2, shrink=0.75, pad=0.02)
    cbar.set_label("EPR (m yr$^{-1}$)")
    ax2.set_aspect("equal")
    ax2.set_title("End Point Rate per transect (U-Net)")
    km_axes(ax2)
    ax2.legend(loc="upper left", fontsize=7)
    add_scalebar(ax2); north_arrow(ax2)
    panel_letter(ax2, "b", x=0.0, y=1.10)
    save(fig, "Fig01_transect_framework_and_EPR_map",
         "Geometric framework of the change analysis. (a) Baseline built by "
         "offsetting the smoothed time-mean shoreline (amber dashed) 200 m "
         "landward (black), 50-m-spaced shore-normal transects (grey; IDs "
         "south -> north) and the U-Net shorelines (teal). (b) Transects "
         "coloured by End Point Rate; rust = erosion, blue = accretion; black "
         "dashed = most recent shoreline (autumn 2025). UTM 29N coordinates.")


def fig02_spacetime(dists):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, m, letter in zip(axes, ["classical", "unet"], "ab"):
        piv = dists[m].pivot_table(index="transect_id", columns="date",
                                   values="distance", aggfunc="mean")
        anom = piv.sub(piv.mean(axis=1), axis=0)
        vmax = np.nanpercentile(np.abs(anom.values), 98)
        im = ax.imshow(anom.values, aspect="auto", cmap=CHANGE_CMAP,
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(MODEL_LABEL[m])
        ax.set_xlabel("Season (2018 - 2025)")
        ax.set_ylabel("Transect ID (S -> N)")
        cols = list(piv.columns)
        ticks = [i for i, c in enumerate(cols) if c.month == 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([c.year for c in cols if c.month == 1], fontsize=7)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Shoreline anomaly (m)")
        panel_letter(ax, letter)
    save(fig, "Fig02_spacetime_anomaly_heatmap",
         "Space-time diagram of the shoreline position anomaly (deviation "
         "from the per-transect 2018-2025 mean). (a) Classical MNDWI; "
         "(b) U-Net. Rust = landward of the mean (erosion), blue = seaward "
         "(accretion); transect 1 = southern (harbour-side) end. The U-Net "
         "field is markedly less noisy than the classical one.")


def fig03_rates_by_transect(rates):
    r = rates["unet"].sort_values("transect_id")
    metrics = [("epr", "EPR (m yr$^{-1}$)"), ("nsm", "NSM (m)"),
               ("lrr", "LRR (m yr$^{-1}$)")]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.6), sharex=True,
                             constrained_layout=True)
    for ax, (k, lab), letter in zip(axes, metrics, "abc"):
        colors = [ERO if v < 0 else ACC for v in r[k]]
        ax.bar(r["transect_id"], r[k], color=colors, width=0.82)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel(lab)
        mu = r[k].mean()
        ax.axhline(mu, color="grey", ls="--", lw=0.9)
        ax.annotate(f"mean = {mu:.2f}", xy=(0.995, mu),
                    xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7, color="grey")
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        panel_letter(ax, letter, x=-0.07)
    axes[-1].set_xlabel("Transect ID (south -> north)")
    save(fig, "Fig03_rates_by_transect",
         "Per-transect change metrics from the U-Net shorelines: (a) End "
         "Point Rate, (b) Net Shoreline Movement, (c) Linear Regression "
         "Rate. Rust bars = erosion, blue = accretion; dashed line = "
         "embayment mean.")


def fig04_distributions(rates):
    import seaborn as sns
    frames = []
    for m in MODELS:
        d = rates[m][["epr", "lrr"]].copy()
        d["model"] = MODEL_LABEL[m]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    groups = [rates[m]["epr"].dropna().values for m in MODELS]
    H, p_kw = kruskal(*groups)

    order = [MODEL_LABEL[m] for m in MODELS]
    pal = {MODEL_LABEL[m]: MODEL_COLOR[m] for m in MODELS}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, metric, letter in ((ax1, "epr", "a"), (ax2, "lrr", "b")):
        sns.violinplot(data=df, x="model", y=metric, order=order,
                       hue="model", palette=pal, dodge=False,
                       inner=None, cut=0, linewidth=0.7, alpha=0.35, ax=ax)
        sns.boxplot(data=df, x="model", y=metric, order=order,
                    hue="model", palette=pal, dodge=False,
                    width=0.18, showfliers=False, ax=ax,
                    boxprops=dict(alpha=0.9),
                    medianprops=dict(color="white", lw=1.3))
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_xlabel("")
        ax.set_ylabel(f"{metric.upper()} (m yr$^{{-1}}$)")
        ax.tick_params(axis="x", rotation=15)
        panel_letter(ax, letter)
    ax1.set_title(f"Kruskal-Wallis H = {H:.1f}, p = {p_kw:.1e}")
    for i, m in enumerate(MODELS):
        n = rates[m]["epr"].notna().sum()
        ax1.annotate(f"n = {n}", (i, ax1.get_ylim()[0]),
                     ha="center", va="bottom", fontsize=7)
    save(fig, "Fig04_rate_distributions_by_model",
         "Distribution of the per-transect (a) EPR and (b) LRR for the four "
         "extraction pipelines (violin: kernel density; inner box: "
         "interquartile range, white line = median). The classical pipeline "
         "shows the widest dispersion; the Kruskal-Wallis test assesses "
         "whether the four EPR distributions share the same median.")


def fig05_volume_budget(rates, ann_v):
    r = rates["unet"].sort_values("transect_id")
    a = ann_v["unet"]
    fig = plt.figure(figsize=(12, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    colors = [ERO if v < 0 else ACC for v in r["volume_m3"]]
    ax1.bar(r["transect_id"], r["volume_m3"], color=colors, width=0.82)
    ax1.axhline(0, color="k", lw=0.7)
    ax1.set_xlabel("Transect ID (south -> north)")
    ax1.set_ylabel("Volume change (m$^3$)")
    ax1.grid(axis="y", alpha=0.25, lw=0.4)
    panel_letter(ax1, "a", x=-0.045)

    yearly = a.groupby("year_start")["dvol"].sum()
    labels = [f"{y}-{y+1}" for y in yearly.index]
    ax2.bar(labels, yearly.values,
            color=[ERO if v < 0 else ACC for v in yearly.values])
    ax2b = ax2.twinx()
    ax2b.plot(labels, yearly.cumsum().values, "o-", color="k", lw=1.3, ms=4)
    ax2b.set_ylabel("Cumulative (m$^3$)")
    ax2b.spines["right"].set_visible(True)
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_ylabel("Annual volume change (m$^3$)")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    panel_letter(ax2, "b")

    ero = [rates[m].loc[rates[m]["volume_m3"] < 0, "volume_m3"].sum() for m in MODELS]
    acc = [rates[m].loc[rates[m]["volume_m3"] > 0, "volume_m3"].sum() for m in MODELS]
    net = [rates[m]["volume_m3"].sum() for m in MODELS]
    y = np.arange(len(MODELS))
    ax3.barh(y + 0.22, ero, height=0.2, color=ERO, label="Erosion")
    ax3.barh(y, acc, height=0.2, color=ACC, label="Accretion")
    ax3.barh(y - 0.22, net, height=0.2, color="#555555", label="Net")
    ax3.axvline(0, color="k", lw=0.7)
    ax3.set_yticks(y)
    ax3.set_yticklabels([MODEL_LABEL[m] for m in MODELS])
    ax3.set_xlabel("Volume 2018-2025 (m$^3$)")
    ax3.legend(loc="lower right", fontsize=7)
    panel_letter(ax3, "c")
    save(fig, "Fig05_sediment_volume_budget",
         "Sediment volume budget (active envelope 5 m, transect cell 50 m). "
         "(a) Per-transect volume change 2018-2025 (U-Net); (b) annual and "
         "cumulative embayment-integrated budget (U-Net); (c) erosion / "
         "accretion / net decomposition for the four pipelines.")


def fig06_seasonal(dists):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    d = dists["unet"].copy()
    d["anom"] = d["distance"] - d.groupby("transect_id")["distance"].transform("mean")
    data = [d[d["season"] == s]["anom"].values for s in SEASONS]
    bp = axes[0].boxplot(data, labels=[s.capitalize() for s in SEASONS],
                         patch_artist=True, widths=0.55, showfliers=False,
                         medianprops=dict(color="black", lw=1.2))
    cols = ["#4477AA", "#66CCEE", "#CCBB44", "#EE6677"]
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    axes[0].axhline(0, color="k", lw=0.7, ls=":")
    axes[0].set_ylabel("Shoreline anomaly (m)")
    panel_letter(axes[0], "a")

    for m in MODELS:
        dd = dists[m].copy()
        dd["anom"] = dd["distance"] - dd.groupby("transect_id")["distance"].transform("mean")
        means = dd.groupby("season")["anom"].mean().reindex(SEASONS)
        ses = dd.groupby("season")["anom"].sem().reindex(SEASONS)
        axes[1].errorbar([s.capitalize() for s in SEASONS], means.values,
                         yerr=1.96 * ses.values, fmt="o-", capsize=3, lw=1.2,
                         ms=4, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
    axes[1].axhline(0, color="k", lw=0.7, ls=":")
    axes[1].set_ylabel("Mean shoreline anomaly (m)")
    axes[1].legend(fontsize=7)
    panel_letter(axes[1], "b")
    save(fig, "Fig06_seasonal_cycle",
         "Seasonality of the shoreline position (anomaly relative to each "
         "transect's 2018-2025 mean). (a) U-Net distribution by season; "
         "(b) seasonal means with 95% confidence intervals for the four "
         "pipelines.")


def fig07_model_intercomparison(rates):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                   constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.9, 1]})
    for m in MODELS:
        r = rates[m].sort_values("transect_id")
        ax1.plot(r["transect_id"], r["epr"], "o-", ms=2.6, lw=1.0,
                 color=MODEL_COLOR[m], label=MODEL_LABEL[m], alpha=0.85)
    ax1.axhline(0, color="k", lw=0.7, ls=":")
    ax1.set_xlabel("Transect ID (south -> north)")
    ax1.set_ylabel("EPR (m yr$^{-1}$)")
    ax1.legend(ncol=2, fontsize=7)
    ax1.grid(alpha=0.25, lw=0.4)
    panel_letter(ax1, "a", x=-0.07)

    piv = pd.DataFrame({m: rates[m].set_index("transect_id")["epr"] for m in MODELS})
    corr = piv.corr(method="pearson")
    im = ax2.imshow(corr.values, cmap="cividis", vmin=0, vmax=1)
    ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
    ax2.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=30,
                        ha="right", fontsize=7)
    ax2.set_yticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=7)
    for i in range(4):
        for j in range(4):
            v = corr.values[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax2, shrink=0.85, label="Pearson r")
    panel_letter(ax2, "b")
    save(fig, "Fig07_model_intercomparison",
         "Sensitivity of the change signal to the extraction pipeline. "
         "(a) Per-transect EPR profiles of the four pipelines on the common "
         "transect grid; (b) Pearson correlation matrix of the EPR profiles.")
    corr.round(3).to_csv(TAB / "epr_profile_correlations.csv")


def fig08_trend(dists):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True,
                             constrained_layout=True)
    for ax, m, letter in zip(axes.flat, MODELS, "abcd"):
        d = dists[m]
        ts = d.groupby("date")["distance"].mean()
        se = d.groupby("date")["distance"].sem()
        x = np.array([(dt - ts.index[0]).days / 365.25 for dt in ts.index])
        slope, intercept, r, p, stderr = linregress(x, ts.values)
        tau, p_mk = kendalltau(np.arange(len(ts)), ts.values)
        n = len(x)
        tval = t_dist.ppf(0.975, n - 2)
        yfit = intercept + slope * x
        resid_se = np.sqrt(np.sum((ts.values - yfit) ** 2) / (n - 2))
        ci = tval * resid_se * np.sqrt(1 / n + (x - x.mean()) ** 2 /
                                       np.sum((x - x.mean()) ** 2))
        ax.errorbar(ts.index, ts.values, yerr=1.96 * se.values, fmt="o",
                    ms=3, capsize=2, color=MODEL_COLOR[m], alpha=0.8)
        ax.plot(ts.index, yfit, "-", color=MODEL_COLOR[m], lw=1.5)
        ax.fill_between(ts.index, yfit - ci, yfit + ci,
                        color=MODEL_COLOR[m], alpha=0.15)
        ax.set_title(f"{MODEL_LABEL[m]}   LRR = {slope:.2f} m yr$^{{-1}}$   "
                     f"MK $\\tau$ = {tau:.2f} (p = {p_mk:.2f})", fontsize=8.5)
        ax.set_ylabel("Mean distance to baseline (m)")
        ax.grid(alpha=0.25, lw=0.4)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        panel_letter(ax, letter)
    save(fig, "Fig08_embayment_trend_analysis",
         "Embayment-mean shoreline position versus time for the four "
         "pipelines. Points: seasonal means +/- 95% CI across transects; "
         "line: least-squares fit with 95% confidence band; panel titles "
         "give the embayment-scale LRR and the Mann-Kendall test.")


# ---------------------------------------------------------------------------
# NEW FIGURES 9-12
# ---------------------------------------------------------------------------
def fig09_classification_map(transects, rates):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2),
                                   constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    gt = transects.merge(rates["unet"][["transect_id", "epr"]], on="transect_id")
    gt["cls"] = np.where(gt["epr"] < 0, "Erosion",
                         np.where(gt["epr"] > 0, "Accretion", "Stable"))
    for cls, color in (("Erosion", ERO), ("Accretion", ACC), ("Stable", STA)):
        sel = gt[gt["cls"] == cls]
        if not sel.empty:
            sel.plot(ax=ax1, color=color, linewidth=2.4, label=cls)
    ax1.set_aspect("equal")
    ax1.set_title("Transect classification (U-Net)")
    km_axes(ax1)
    ax1.legend(loc="upper left", fontsize=7)
    add_scalebar(ax1); north_arrow(ax1)
    panel_letter(ax1, "a", x=0.0, y=1.10)

    fracs = []
    for m in MODELS:
        e = rates[m]["epr"]
        fracs.append([(e < 0).mean() * 100, (e == 0).mean() * 100,
                      (e > 0).mean() * 100])
    fracs = np.array(fracs)
    y = np.arange(len(MODELS))
    left = np.zeros(len(MODELS))
    for i, (lab, color) in enumerate((("Erosion", ERO), ("Stable", STA),
                                      ("Accretion", ACC))):
        ax2.barh(y, fracs[:, i], left=left, color=color, label=lab, height=0.6)
        for j in range(len(MODELS)):
            if fracs[j, i] > 6:
                ax2.text(left[j] + fracs[j, i] / 2, y[j], f"{fracs[j, i]:.0f}%",
                         ha="center", va="center", fontsize=7,
                         color="white" if lab != "Stable" else "black")
        left += fracs[:, i]
    ax2.set_yticks(y)
    ax2.set_yticklabels([MODEL_LABEL[m] for m in MODELS])
    ax2.set_xlabel("Share of transects (%)")
    ax2.set_xlim(0, 100)
    ax2.legend(ncol=3, fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, 1.12))
    panel_letter(ax2, "b", x=-0.18, y=1.10)
    save(fig, "Fig09_erosion_accretion_classification",
         "Erosion / stable / accretion classification (sign of the EPR). "
         "(a) Map of the U-Net classification on the common transect grid; "
         "(b) share of transects per class for the four pipelines.")


def fig10_epr_vs_lrr(rates):
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    lims = [-3, 3]
    for m in MODELS:
        r = rates[m]
        ax.scatter(r["epr"], r["lrr"], s=16, color=MODEL_COLOR[m],
                   alpha=0.75, label=MODEL_LABEL[m], edgecolor="none")
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("EPR (m yr$^{-1}$)")
    ax.set_ylabel("LRR (m yr$^{-1}$)")
    rr = pd.concat([rates[m][["epr", "lrr"]] for m in MODELS])
    r_all = rr.corr().iloc[0, 1]
    ax.set_title(f"EPR vs LRR per transect (all pipelines, r = {r_all:.2f})")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    save(fig, "Fig10_EPR_vs_LRR_consistency",
         "Consistency between the two rate estimators at the transect scale. "
         "EPR (end-point) against LRR (least-squares) for the four "
         "pipelines; dashed line = 1:1. Points off the diagonal flag "
         "transects whose evolution is non-monotonic, where the regression "
         "rate is the more robust descriptor.")


def fig11_annual_epr_heatmap(ann_e):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, m, letter in zip(axes, ["classical", "unet"], "ab"):
        piv = ann_e[m].pivot_table(index="transect_id", columns="period",
                                   values="epr_annual")
        vmax = np.nanpercentile(np.abs(piv.values), 98)
        im = ax.imshow(piv.values, aspect="auto", cmap=CHANGE_CMAP,
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(MODEL_LABEL[m])
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=45, fontsize=6.5)
        ax.set_ylabel("Transect ID (S -> N)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Annual EPR (m yr$^{-1}$)")
        panel_letter(ax, letter)
    save(fig, "Fig11_annual_EPR_heatmap",
         "Inter-annual variability of the change rate: year-to-year EPR per "
         "transect for (a) the classical MNDWI and (b) the U-Net pipeline. "
         "Rust = erosional year, blue = accretional year. Alternating "
         "erosional (2020-2021, 2024-2025) and recovery (2019-2020, "
         "2023-2024) years dominate the record.")


def fig12_uncertainty(rates):
    # Confidence intervals are archived study outputs; this figure only plots them.
    ci_path = TAB / "bootstrap_epr_ci.csv"
    if not ci_path.exists():
        raise SystemExit(f"Missing archived bootstrap confidence intervals: {ci_path}")
    ci_table = pd.read_csv(ci_path).set_index("model")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                   constrained_layout=True)
    for i, m in enumerate(MODELS):
        label = MODEL_LABEL[m]
        if label not in ci_table.index:
            raise SystemExit(f"No archived bootstrap interval for {label} in {ci_path}")
        row = ci_table.loc[label]
        mean, lo, hi = row["epr_mean"], row["ci_low"], row["ci_high"]
        ax1.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]],
                     fmt="o", ms=6, capsize=5, lw=1.5, color=MODEL_COLOR[m])
    ax1.axhline(0, color="k", lw=0.7, ls=":")
    ax1.set_xticks(range(len(MODELS)))
    ax1.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=15)
    ax1.set_ylabel("Mean EPR (m yr$^{-1}$)")
    ax1.set_title("Archived bootstrap 95% CI")
    panel_letter(ax1, "a")

    x = np.arange(len(MODELS)); w = 0.38
    lrr = [rates[m]["lrr"].mean() for m in MODELS]
    sen = [rates[m]["theil_sen"].mean() for m in MODELS]
    ax2.bar(x - w / 2, lrr, w, label="LRR (OLS)", color="#888888")
    ax2.bar(x + w / 2, sen, w, label="Theil-Sen", color="#44AA99")
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=15)
    ax2.set_ylabel("Mean regression rate (m yr$^{-1}$)")
    ax2.legend(fontsize=7)
    panel_letter(ax2, "b")
    save(fig, "Fig12_rate_uncertainty",
         "Robustness of the embayment-mean rate. (a) Mean EPR per pipeline "
         "with the archived bootstrap 95% confidence intervals "
         "transect set): the classical and FCN intervals include zero, "
         "whereas U-Net and SegNet resolve a weakly positive (accretional) "
         "mean EPR. (b) Mean OLS regression rate versus the outlier-robust "
         "Theil-Sen estimator; the sign disagreement with the EPR reflects "
         "the non-monotonic character of the record.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("STUDY 2 - generating figures from archived change results")
    transects, baseline, mean_line, dists, rates, ann_v, ann_e, gdfs = load_products()

    fig01_framework(gdfs, transects, baseline, mean_line, rates)
    fig02_spacetime(dists)
    fig03_rates_by_transect(rates)
    fig04_distributions(rates)
    fig05_volume_budget(rates, ann_v)
    fig06_seasonal(dists)
    fig07_model_intercomparison(rates)
    fig08_trend(dists)
    fig09_classification_map(transects, rates)
    fig10_epr_vs_lrr(rates)
    fig11_annual_epr_heatmap(ann_e)
    fig12_uncertainty(rates)

    (OUT / "figures_manifest.json").write_text(json.dumps(MANIFEST, indent=2))
    print(f"\nDone. {len(MANIFEST)} figures in {FIG}")


if __name__ == "__main__":
    main()
