#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STUDY 1 - STEP 03: ADVANCED PUBLICATION FIGURES
Shoreline extraction at Anza Beach: classical MNDWI vs U-Net / SegNet / FCN.

Reads everything from  <study_root>/data/  and writes all outputs to
<study_root>/outputs/  (figures/ , tables/ , figures_manifest.json).

Run from the project folder:
    python scripts/03_make_extraction_figures.py

Figures use the shorelines produced by the extraction steps; Figure 4 compares
those generated vectors without needing a second model-inference pass.
"""

import os
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from visual_style import MODEL_COLORS, MNDWI_CMAP, STYLE
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import Point

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    HAS_SCALEBAR = True
except ImportError:
    HAS_SCALEBAR = False

# ----------------------------------------------------------------------------
# PATHS AND CONSTANTS
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]          # standalone study root
DATA = BASE / "data"
OUT = BASE / "outputs" / "extraction"
FIG = OUT / "figures"
TAB = OUT / "tables"

for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

MODELS = ["classical", "unet", "segnet", "fcn"]
MODEL_LABEL = {"classical": "Classical (MNDWI)", "unet": "U-Net",
               "segnet": "SegNet", "fcn": "FCN"}
MODEL_COLOR = MODEL_COLORS
SEASONS = ["winter", "spring", "summer", "autumn"]
SEASON_LABEL = {"winter": "Winter", "spring": "Spring",
                "summer": "Summer", "autumn": "Autumn"}

MANIFEST = []

# ----------------------------------------------------------------------------
# GLOBAL STYLE (journal standard)
# ----------------------------------------------------------------------------
plt.rcParams.update(STYLE)


def save(fig, name, caption):
    """Save figure as 300-dpi PNG + vector PDF and register in the manifest."""
    png = FIG / f"{name}.png"
    fig.savefig(png)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    MANIFEST.append({"figure": name, "file": str(png.name), "caption": caption})
    print(f"  [OK] {name}")


def panel_letter(ax, letter, x=-0.08, y=1.06):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")


def north_arrow(ax, x=0.95, y=0.92):
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.09),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="k"))


def add_scalebar(ax, dx=1.0):
    if HAS_SCALEBAR:
        ax.add_artist(ScaleBar(dx, units="m", location="lower left",
                               box_alpha=0.7, font_properties={"size": 7}))


def km_axes(ax, nbins=4):
    """Kilometre tick formatting for UTM map axes."""
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v/1000:.1f}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=nbins))
        axis.set_major_formatter(fmt)
    ax.set_xlabel("Easting UTM 29N (km)", fontsize=7)
    ax.set_ylabel("Northing (km)", fontsize=7)


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
def load_coastlines():
    """Load the archived shoreline vectors supplied with the study."""
    gdfs = {}
    archive = DATA / "coastlines"
    paths = {"classical": archive / "all_coastlines_utm.shp"}
    for m in ["unet", "segnet", "fcn"]:
        paths[m] = archive / f"all_coastlines_{m}_utm.shp"
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing archived shoreline results in data/coastlines: "
                         + ", ".join(missing))
    for m, path in paths.items():
        gdfs[m] = gpd.read_file(path)
    for m, g in gdfs.items():
        g["date"] = pd.to_datetime(g["date"])
        g["year"] = g["date"].dt.year
        g["season"] = g["season"].str.lower()
        gdfs[m] = g.sort_values("date").reset_index(drop=True)
    return gdfs


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read()
        return arr, src.transform, src.res, src.bounds


# ----------------------------------------------------------------------------
# FIGURE 1 - STUDY AREA AND SEASONAL DATA
# ----------------------------------------------------------------------------
def fig01_study_area(year=2024):
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), constrained_layout=True)
    m_per_deg = 111_320 * np.cos(np.deg2rad(30.45))
    im = None
    for j, season in enumerate(SEASONS):
        s2, tr, res, _ = read_raster(DATA / f"S2_{year}_{season}.tif")
        # bands stored as [B2, B3, B4, B8, B11] -> RGB = B4,B3,B2 = idx 2,1,0
        rgb = np.stack([s2[2], s2[1], s2[0]], axis=-1)
        p2, p98 = np.nanpercentile(rgb, [2, 98])
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-9), 0, 1)
        ax = axes[0, j]
        ax.imshow(rgb)
        ax.set_title(f"{SEASON_LABEL[season]} {year}")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Sentinel-2 RGB", fontsize=9)
            add_scalebar(ax, dx=res[0] * m_per_deg)
            north_arrow(ax)

        mndwi, _, _, _ = read_raster(DATA / f"MNDWI_{year}_{season}.tif")
        ax = axes[1, j]
        im = ax.imshow(mndwi[0], cmap=MNDWI_CMAP, vmin=-0.8, vmax=0.8)
        ax.contour(mndwi[0], levels=[0.0], colors="k", linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("MNDWI", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[1, :], shrink=0.8, pad=0.01)
    cbar.set_label("MNDWI (-)")
    panel_letter(axes[0, 0], "a"); panel_letter(axes[1, 0], "b")
    save(fig, "Fig01_study_area_seasonal_data",
         f"Seasonal Sentinel-2 true-colour composites (a) and MNDWI rasters (b) "
         f"over the Anza embayment for {year}. Black contour: MNDWI = 0 "
         f"(land-water interface used by the classical pipeline).")


# ----------------------------------------------------------------------------
# FIGURE 2 - PIXEL-LEVEL TEST METRICS
# ----------------------------------------------------------------------------
def fig02_pixel_metrics():
    tm = json.load(open(DATA / "test_metrics.json"))
    metrics = ["iou", "f1", "accuracy", "precision", "recall"]
    mlabels = ["IoU", "F1", "Accuracy", "Precision", "Recall"]
    dl = ["unet", "segnet", "fcn"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    x = np.arange(len(metrics)); w = 0.26
    for i, m in enumerate(dl):
        vals = [tm[m][k] for k in metrics]
        ax1.bar(x + (i - 1) * w, vals, w, color=MODEL_COLOR[m],
                label=MODEL_LABEL[m], edgecolor="white", linewidth=0.4)
    ax1.set_xticks(x); ax1.set_xticklabels(mlabels)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("Score (-)")
    ax1.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax1.legend(loc="lower right")
    panel_letter(ax1, "a")

    for i, m in enumerate(dl):
        vals = [tm[m][k] for k in metrics]
        ax2.plot(mlabels, vals, "o-", color=MODEL_COLOR[m], lw=1.4,
                 ms=5, label=MODEL_LABEL[m])
        for xl, v in zip(mlabels, vals):
            ax2.annotate(f"{v:.4f}", (xl, v), textcoords="offset points",
                         xytext=(0, 5 if m == "unet" else (-11 if m == "fcn" else 5)),
                         fontsize=6.5, ha="center", color=MODEL_COLOR[m])
    ax2.set_ylim(0.9965, 1.0002)
    ax2.set_ylabel("Score (-)  [zoom]")
    ax2.axhline(1.0, color="grey", lw=0.6, ls=":")
    panel_letter(ax2, "b")

    save(fig, "Fig02_pixel_level_metrics",
         "Pixel-level test-set performance of the three CNNs against the "
         "threshold-derived reference labels. (a) Full-scale scores; "
         "(b) zoom on the 0.9965-1.0 range revealing the consistent ranking "
         "U-Net > SegNet > FCN. Note that these scores quantify agreement "
         "with the MNDWI reference, not independent ground truth.")

    rows = [{"model": MODEL_LABEL[m], **{k: round(tm[m][k], 5) for k in metrics}} for m in dl]
    pd.DataFrame(rows).to_csv(TAB / "pixel_metrics.csv", index=False)


# ----------------------------------------------------------------------------
# FIGURE 3 - COASTLINE STACKS PER METHOD
# ----------------------------------------------------------------------------
def fig03_coastline_stacks(gdfs):
    # A single, four-panel row follows the wide map layout used in the article.
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 3.6), constrained_layout=True)
    norm = Normalize(vmin=2018, vmax=2025)
    cmap = plt.get_cmap("viridis")
    xmin, ymin, xmax, ymax = gdfs["classical"].total_bounds
    pad = 60
    for ax, m, letter in zip(axes, MODELS, "abcd"):
        g = gdfs[m]
        for _, row in g.iterrows():
            frac = row["year"] + (SEASONS.index(row["season"]) / 4.0)
            ax.plot(*row.geometry.xy, color=cmap(norm(frac)), lw=0.8, alpha=0.85)
        ax.set_xlim(xmin - pad, xmax + pad); ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal")
        ax.set_title(MODEL_LABEL[m], color=MODEL_COLOR[m], pad=4)
        km_axes(ax)
        if letter != "a":
            ax.set_ylabel("")
        ax.tick_params(labelsize=6.5)
        ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          boxstyle="round,pad=0.12"))
        if letter == "a":
            add_scalebar(ax); north_arrow(ax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("Year")
    save(fig, "Fig03_coastline_stacks_by_method",
         "Seasonal shorelines produced by each pipeline from 2018 to 2025, "
         "coloured by acquisition year. Local differences in shoreline "
         "position and shape are shown without smoothing. Coordinates in "
         "UTM zone 29N (EPSG:32629).")


# ----------------------------------------------------------------------------
# FIGURE 4 - QUALITATIVE SHORELINE OUTPUTS
# ----------------------------------------------------------------------------
def fig04_qualitative(gdfs):
    """Compare generated shoreline vectors over the same MNDWI scenes."""
    scenes = [("2018", "winter"), ("2022", "summer"), ("2025", "autumn")]
    methods = ["classical", "unet", "segnet", "fcn"]
    fig, axes = plt.subplots(len(scenes), 5, figsize=(12.2, 6.8),
                             constrained_layout=True, squeeze=False)
    titles = ["MNDWI input"] + [MODEL_LABEL[m] for m in methods]
    for c, title in enumerate(titles):
        color = MODEL_COLOR[methods[c - 1]] if c else "#26323D"
        axes[0, c].set_title(title, color=color)
    for r, (yr, sea) in enumerate(scenes):
        mndwi, _, _, bounds = read_raster(DATA / f"MNDWI_{yr}_{sea}.tif")
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        for c, m in enumerate([None] + methods):
            ax = axes[r, c]
            ax.imshow(mndwi[0], cmap=MNDWI_CMAP, vmin=-0.8, vmax=0.8,
                      extent=extent, alpha=0.82)
            if m is not None:
                match = gdfs[m][(gdfs[m]["year"] == int(yr)) &
                                (gdfs[m]["season"] == sea)]
                if match.empty:
                    ax.text(0.5, 0.08, "No shoreline output", transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=7,
                            color="#465362",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                                      boxstyle="round,pad=0.25"))
                else:
                    geom = gpd.GeoSeries([match.iloc[0].geometry], crs=32629).to_crs(4326).iloc[0]
                    ax.plot(*geom.xy, color=MODEL_COLOR[m], lw=1.35,
                            solid_capstyle="round")
            ax.set_xlim(bounds.left, bounds.right)
            ax.set_ylim(bounds.bottom, bounds.top)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"{SEASON_LABEL[sea]} {yr}", fontsize=8)
    save(fig, "Fig04_qualitative_segmentation",
         "MNDWI input scenes and corresponding shoreline vectors from the "
         "classical, U-Net, SegNet and FCN pipelines. The three dates show an "
         "early, middle and late example. Empty panels explicitly indicate "
         "that a pipeline produced no shoreline for that scene; irregular "
         "segments are shown without post hoc smoothing.")


# ----------------------------------------------------------------------------
# FIGURE 5 - SHORELINE LENGTH ANALYSIS
# ----------------------------------------------------------------------------
def fig05_length_analysis(gdfs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8),
                                   constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.9, 1]})
    stats_rows = []
    for m in MODELS:
        g = gdfs[m].copy()
        g["len"] = g.geometry.length
        ax1.plot(g["date"], g["len"], "o-", ms=3, lw=1.0,
                 color=MODEL_COLOR[m], label=MODEL_LABEL[m], alpha=0.9)
        stats_rows.append({"model": MODEL_LABEL[m], "n": len(g),
                           "mean_m": g["len"].mean(), "std_m": g["len"].std(),
                           "min_m": g["len"].min(), "max_m": g["len"].max()})
    ax1.set_ylabel("Shoreline length (m)")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(ncol=2)
    ax1.grid(alpha=0.25, lw=0.4)
    panel_letter(ax1, "a")

    data = [gdfs[m].geometry.length.values for m in MODELS]
    bp = ax2.boxplot(data, labels=[MODEL_LABEL[m] for m in MODELS],
                     patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", lw=1.2))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(MODEL_COLOR[m]); patch.set_alpha(0.45)
    ax2.set_ylabel("Shoreline length (m)")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", alpha=0.25, lw=0.4)
    panel_letter(ax2, "b")

    save(fig, "Fig05_shoreline_length_analysis",
         "Reconstructed shoreline length. (a) Seasonal time series per "
         "pipeline (2018-2025); (b) length distributions. The classical "
         "pipeline is systematically ~1% shorter than the CNNs, reflecting "
         "its stronger simplification of the boundary.")
    pd.DataFrame(stats_rows).round(1).to_csv(TAB / "length_summary.csv", index=False)


# ----------------------------------------------------------------------------
# FIGURE 6 - POSITION AGREEMENT DL vs CLASSICAL
# ----------------------------------------------------------------------------
def fig06_agreement(gdfs, step=10.0):
    ref_all = gdfs["classical"]
    records = []
    along_profiles = {m: [] for m in ["unet", "segnet", "fcn"]}
    for _, ref_row in ref_all.iterrows():
        ref = ref_row.geometry
        key = (ref_row["year"], ref_row["season"])
        chain = np.arange(0, ref.length, step)
        pts = [ref.interpolate(d) for d in chain]
        for m in ["unet", "segnet", "fcn"]:
            g = gdfs[m]
            match = g[(g["year"] == key[0]) & (g["season"] == key[1])]
            if match.empty:
                continue
            line = match.iloc[0].geometry
            d = np.array([p.distance(line) for p in pts])
            records.append({"model": m, "year": key[0], "season": key[1],
                            "mean_offset_m": d.mean(), "p95_offset_m": np.percentile(d, 95)})
            along_profiles[m].append(np.interp(
                np.linspace(0, 1, 200), chain / ref.length, d))

    df = pd.DataFrame(records)
    df.to_csv(TAB / "agreement_stats.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8),
                                   constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1, 1.7]})
    data = [df[df["model"] == m]["mean_offset_m"].values for m in ["unet", "segnet", "fcn"]]
    bp = ax1.boxplot(data, labels=[MODEL_LABEL[m] for m in ["unet", "segnet", "fcn"]],
                     patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", lw=1.2))
    for patch, m in zip(bp["boxes"], ["unet", "segnet", "fcn"]):
        patch.set_facecolor(MODEL_COLOR[m]); patch.set_alpha(0.5)
    ax1.axhline(10, color="grey", ls="--", lw=0.8)
    ax1.annotate("1 pixel (10 m)", xy=(0.98, 10), xycoords=("axes fraction", "data"),
                 ha="right", va="bottom", fontsize=7, color="grey")
    ax1.set_ylabel("Mean offset to classical shoreline (m)")
    panel_letter(ax1, "a")

    xs = np.linspace(0, 100, 200)
    for m in ["unet", "segnet", "fcn"]:
        prof = np.vstack(along_profiles[m])
        mean = prof.mean(axis=0); sd = prof.std(axis=0)
        ax2.plot(xs, mean, color=MODEL_COLOR[m], lw=1.3, label=MODEL_LABEL[m])
        ax2.fill_between(xs, mean - sd, mean + sd, color=MODEL_COLOR[m], alpha=0.15)
    ax2.axhline(10, color="grey", ls="--", lw=0.8)
    ax2.set_xlabel("Alongshore position (% of shoreline, S to N)")
    ax2.set_ylabel("Offset to classical shoreline (m)")
    ax2.legend()
    ax2.grid(alpha=0.25, lw=0.4)
    panel_letter(ax2, "b")

    save(fig, "Fig06_position_agreement",
         "Planimetric agreement between each CNN shoreline and the classical "
         "MNDWI shoreline of the same season (32 seasons). (a) Distribution "
         "of the season-mean absolute offset; (b) alongshore profile of the "
         "mean offset (shading: +/-1 sigma across seasons). Mean offsets are "
         "generally modest, but localized excursions exceed the 10-m pixel "
         "scale, especially for SegNet.")


# ----------------------------------------------------------------------------
# FIGURE 9 - SEASONAL AGREEMENT HEATMAP
# ----------------------------------------------------------------------------
def fig09_agreement_heatmap():
    """Mean DL-vs-classical offset per (year, season) - reads the CSV written
    by fig06_agreement, so call it after fig06."""
    df = pd.read_csv(TAB / "agreement_stats.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    vmax = np.nanpercentile(df["mean_offset_m"], 98)
    im = None
    for ax, m, letter in zip(axes, ["unet", "segnet", "fcn"], "abc"):
        piv = (df[df["model"] == m]
               .pivot_table(index="season", columns="year",
                            values="mean_offset_m", aggfunc="mean")
               .reindex(SEASONS))
        im = ax.imshow(piv.values, cmap="cividis", vmin=0, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, fontsize=7, rotation=45)
        ax.set_yticks(range(4))
        ax.set_yticklabels([SEASON_LABEL[s] for s in SEASONS], fontsize=7)
        ax.set_title(MODEL_LABEL[m])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if v < vmax * 0.6 else "black")
        panel_letter(ax, letter, x=-0.05, y=1.12)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cbar.set_label("Mean offset to classical (m)")
    save(fig, "Fig09_seasonal_agreement_heatmap",
         "Season-by-season planimetric agreement between each CNN shoreline "
         "and the classical MNDWI shoreline (mean absolute offset, m). "
         "The alongshore mean offset and its spatial spread compare each "
         "CNN with the classical shoreline. Larger offsets and uncertainty "
         "bands show where the extracted boundaries diverge locally.")


# ----------------------------------------------------------------------------
# FIGURE 10 - FOUR-METHOD OVERLAY ON THE MNDWI RASTER
# ----------------------------------------------------------------------------
def fig10_method_overlay(gdfs, scenes=(("2019", "summer"), ("2024", "winter"))):
    """All four shorelines of the same scene overlaid on the MNDWI raster,
    reprojected on the fly (raster in EPSG:4326, lines in UTM 29N)."""
    fig, axes = plt.subplots(1, len(scenes), figsize=(11, 5.4),
                             constrained_layout=True)
    m_per_deg = 111_320 * np.cos(np.deg2rad(30.45))
    for ax, (yr, sea), letter in zip(np.atleast_1d(axes), scenes, "ab"):
        mndwi, tr, res, bounds = read_raster(DATA / f"MNDWI_{yr}_{sea}.tif")
        ax.imshow(mndwi[0], cmap=MNDWI_CMAP, vmin=-1, vmax=1, alpha=0.75,
                  extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
        for m in MODELS:
            g = gdfs[m]
            match = g[(g["year"] == int(yr)) & (g["season"] == sea)]
            if match.empty:
                continue
            geom = gpd.GeoSeries([match.iloc[0].geometry],
                                 crs=32629).to_crs(4326).iloc[0]
            ax.plot(*geom.xy, color=MODEL_COLOR[m], lw=1.3,
                    label=MODEL_LABEL[m],
                    ls="--" if m == "classical" else "-")
        ax.set_title(f"{SEASON_LABEL[sea]} {yr}")
        ax.set_xticks([]); ax.set_yticks([])
        # axes are in degrees here (imshow extent) -> dx = metres per degree
        add_scalebar(ax, dx=m_per_deg)
        if letter == "a":
            ax.legend(loc="upper left", fontsize=7)
            north_arrow(ax)
        panel_letter(ax, letter, x=0.0, y=1.06)
    save(fig, "Fig10_four_method_overlay",
         "Shorelines extracted by the four pipelines for two contrasting "
         "scenes, overlaid on the corresponding MNDWI raster. The "
         "overlays show local agreement and disagreement along the harbour "
         "revetment and northern headland; "
         "all vector outputs are shown without smoothing.")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("STUDY 1 - generating figures from archived shoreline results")
    print(f"  data:    {DATA}")
    print(f"  outputs: {OUT}\n")
    gdfs = load_coastlines()

    fig01_study_area()
    fig02_pixel_metrics()
    fig03_coastline_stacks(gdfs)
    fig04_qualitative(gdfs)
    fig05_length_analysis(gdfs)
    fig06_agreement(gdfs)
    fig09_agreement_heatmap()
    fig10_method_overlay(gdfs)

    mpath = OUT / "figures_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else []
    keep = [m for m in manifest
            if m["figure"] not in {e["figure"] for e in MANIFEST}]
    mpath.write_text(json.dumps(keep + MANIFEST, indent=2))
    print(f"\nDone. {len(MANIFEST)} figures in {FIG}")
    print(f"Tables in {TAB}")


if __name__ == "__main__":
    main()
