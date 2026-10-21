#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STUDY 2 - STEP 01: DETAILED TRANSECT-BASED CHANGE ANALYSIS

Runs the full, corrected DSAS-style computation for the four shoreline
pipelines (Classical MNDWI, U-Net, SegNet, FCN) on a COMMON transect grid
and writes every intermediate and final product to outputs/analysis/ and
outputs/tables/. See analysis_core.py for the documented methodology.

Outputs
  outputs/analysis/baseline.shp                  baseline (landward anchors)
  outputs/analysis/mean_shoreline.shp            smoothed time-mean shoreline
  outputs/analysis/transects.shp                 common transect grid
  outputs/analysis/distances_<model>.csv         raw position time series
  outputs/analysis/rates_<model>.csv             NSM/EPR/LRR/Theil-Sen/volume
  outputs/analysis/annual_volumes_<model>.csv    year-to-year volumes
  outputs/analysis/annual_epr_<model>.csv        year-to-year EPR
  outputs/analysis/rates_<model>.shp             transects joined with EPR
  outputs/tables/rates_summary_by_model.csv      headline statistics
  outputs/tables/trend_tests_by_model.csv        OLS / Theil-Sen / Mann-Kendall
  outputs/tables/advanced_statistics.csv         full distribution statistics

Run:  python scripts/01_transect_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import skew, kurtosis, shapiro, t as t_dist

import analysis_core as core

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs" / "change"
ANA = OUT / "analysis"
TAB = OUT / "tables"
for d in (ANA, TAB):
    d.mkdir(parents=True, exist_ok=True)

MODELS = ["classical", "unet", "segnet", "fcn"]
MODEL_LABEL = {"classical": "Classical (MNDWI)", "unet": "U-Net",
               "segnet": "SegNet", "fcn": "FCN"}


def load_coastlines():
    """Load only shorelines produced by this project's extraction steps.

    Previously exported vectors under data/coastlines/ are retained as source
    material but are never loaded as automatic fallbacks.
    """
    extraction = BASE / "outputs" / "extraction"
    paths = {"classical": extraction / "classical_shorelines" /
             "all_coastlines_classical_utm.shp"}
    for m in ["unet", "segnet", "fcn"]:
        paths[m] = extraction / "dl_shorelines" / f"all_coastlines_{m}_utm.shp"
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing shoreline outputs. Run extraction steps 01/02 first: "
                         + ", ".join(missing))
    gdfs = {}
    for m, src in paths.items():
        g = gpd.read_file(src)
        g["date"] = pd.to_datetime(g["date"])
        g["year"] = g["date"].dt.year
        g["season"] = g["season"].str.lower()
        gdfs[m] = g.sort_values("date").reset_index(drop=True)
        print(f"  {MODEL_LABEL[m]:<18} {len(g)} shorelines  ({src.parent.name})")
    return gdfs


def advanced_stats(rates):
    """Distribution statistics of the EPR field (as in the original study)."""
    epr = rates["epr"].dropna()
    n = len(epr)
    se = epr.std() / np.sqrt(n)
    ci = t_dist.interval(0.95, df=n - 1, loc=epr.mean(), scale=se)
    sh_w, sh_p = shapiro(epr) if 3 <= n <= 5000 else (np.nan, np.nan)
    return {"epr_count": n, "epr_mean": epr.mean(), "epr_std": epr.std(),
            "epr_median": epr.median(), "epr_min": epr.min(),
            "epr_max": epr.max(), "epr_q25": epr.quantile(0.25),
            "epr_q75": epr.quantile(0.75), "epr_skew": skew(epr),
            "epr_kurtosis": kurtosis(epr), "epr_ci95_low": ci[0],
            "epr_ci95_high": ci[1], "shapiro_W": sh_w, "shapiro_p": sh_p,
            "pct_erosion": (epr < 0).mean() * 100,
            "pct_accretion": (epr > 0).mean() * 100,
            "pct_stable": (epr == 0).mean() * 100}


def main():
    print("STEP 01 - transect-based change analysis (common DSAS grid)")
    gdfs = load_coastlines()

    # ---- common geometry (built once, from the classical shorelines) -------
    transects, baseline, mean_line = core.build_transects_and_baseline(
        gdfs["classical"], gdfs["classical"].crs)
    print(f"  baseline {baseline.length:.0f} m -> {len(transects)} transects")
    crs = gdfs["classical"].crs
    transects.to_file(ANA / "transects.shp")
    gpd.GeoDataFrame(geometry=[baseline], crs=crs).to_file(ANA / "baseline.shp")
    gpd.GeoDataFrame(geometry=[mean_line], crs=crs).to_file(
        ANA / "mean_shoreline.shp")

    # ---- per-model computation --------------------------------------------
    summary, trends, advanced = [], [], []
    for m in MODELS:
        dist = core.transect_distances(gdfs[m], transects)
        rates = core.rates_from_distances(dist)
        ann_v = core.annual_volumes(dist)
        ann_e = core.annual_epr(dist)

        dist.to_csv(ANA / f"distances_{m}.csv", index=False)
        rates.to_csv(ANA / f"rates_{m}.csv", index=False)
        ann_v.to_csv(ANA / f"annual_volumes_{m}.csv", index=False)
        ann_e.to_csv(ANA / f"annual_epr_{m}.csv", index=False)
        transects.merge(rates, on="transect_id").to_file(ANA / f"rates_{m}.shp")

        summary.append({
            "model": MODEL_LABEL[m], "n_transects": len(rates),
            "epr_mean": rates["epr"].mean(), "epr_std": rates["epr"].std(),
            "epr_median": rates["epr"].median(),
            "nsm_mean": rates["nsm"].mean(), "lrr_mean": rates["lrr"].mean(),
            "theil_sen_mean": rates["theil_sen"].mean(),
            "pct_erosion": (rates["epr"] < 0).mean() * 100,
            "pct_accretion": (rates["epr"] > 0).mean() * 100,
            "pct_stable": (rates["epr"] == 0).mean() * 100,
            "net_volume_m3": rates["volume_m3"].sum(),
            "erosion_m3": rates.loc[rates["volume_m3"] < 0, "volume_m3"].sum(),
            "accretion_m3": rates.loc[rates["volume_m3"] > 0, "volume_m3"].sum(),
            "net_mass_t": rates["mass_t"].sum()})
        trends.append({"model": MODEL_LABEL[m], **core.embayment_trend(dist)})
        advanced.append({"model": MODEL_LABEL[m], **advanced_stats(rates)})
        print(f"  {MODEL_LABEL[m]:<18} EPR = {rates['epr'].mean():+.3f} "
              f"m/yr   net volume = {rates['volume_m3'].sum():+,.0f} m3")

    pd.DataFrame(summary).round(3).to_csv(TAB / "rates_summary_by_model.csv",
                                          index=False)
    pd.DataFrame(trends).round(4).to_csv(TAB / "trend_tests_by_model.csv",
                                         index=False)
    pd.DataFrame(advanced).round(4).to_csv(TAB / "advanced_statistics.csv",
                                           index=False)
    print(f"\n  analysis products -> {ANA}")
    print(f"  summary tables    -> {TAB}")


if __name__ == "__main__":
    main()
