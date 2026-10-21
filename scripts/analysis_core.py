#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared computation core for Study 2 (coastal-change calculations).

Implements the corrected DSAS-style geometry and every change metric, fully
documented so the methodology is transparent and citable:

  Baseline    : smoothed time-mean shoreline offset 200 m landward.
  Transects   : one every 50 m, shore-normal, one-sided (500 m total:
                200 m landward anchor -> 300 m seaward), numbered S -> N,
                non-crossing by construction (strong low-pass smoothing of
                the mean shoreline keeps the curvature radius > offset).
  Distances   : landward-anchor -> shoreline intersection, so LARGER
                distance = seaward position; EPR > 0 = accretion,
                EPR < 0 = erosion.
  Metrics     : NSM, EPR, LRR (+ p, R2), Theil-Sen slope, Mann-Kendall.
  Volumes     : dV = dS x active height (5 m) x cell width (50 m);
                mass = V x density (1600 kg/m3) x (1 - porosity 0.4).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from scipy.stats import linregress, kendalltau, theilslopes

# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------
TRANSECT_SPACING = 50.0     # m along the baseline
TRANSECT_LENGTH = 500.0     # m total (landward + seaward)
BASELINE_OFFSET = 200.0     # m landward of the time-mean shoreline
ACTIVE_HEIGHT = 5.0         # m, active beach-face envelope
DENSITY = 1600.0            # kg m-3, sediment density
POROSITY = 0.4              # -, sediment porosity
SEASONS = ["winter", "spring", "summer", "autumn"]


# ---------------------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------------------
def mean_shoreline(gdf, n_pts=300, smooth_win=61, simplify_m=5):
    """Time-averaged shoreline: every coastline is resampled at n_pts fixed
    fractional positions (orientation normalised south -> north) and the
    coordinates are averaged, then strongly low-pass filtered (moving
    average over `smooth_win` samples) so that the local curvature radius
    stays well above BASELINE_OFFSET (prevents transect crossing)."""
    stack = []
    for geom in gdf.geometry:
        if geom.geom_type != "LineString":
            continue
        coords = np.asarray(geom.coords)
        if coords[0, 1] > coords[-1, 1]:
            coords = coords[::-1]
        line = LineString(coords)
        fr = np.linspace(0, 1, n_pts)
        stack.append(np.array([[line.interpolate(f, normalized=True).x,
                                line.interpolate(f, normalized=True).y]
                               for f in fr]))
    mean_xy = np.mean(np.stack(stack), axis=0)
    kernel = np.ones(smooth_win) / smooth_win
    pad = smooth_win // 2
    for c in (0, 1):
        col = np.pad(mean_xy[:, c], pad, mode="edge")
        mean_xy[:, c] = np.convolve(col, kernel, mode="valid")
    return LineString(mean_xy).simplify(simplify_m)


def build_transects_and_baseline(gdf, crs):
    """Stations every TRANSECT_SPACING m on the mean shoreline; shore-normal
    direction from a +/- 25-m window; landward side identified with a fixed
    ENE unit vector (land lies east-northeast of the shoreline at Anza).
    Returns (transects GeoDataFrame, baseline LineString, mean shoreline)."""
    mean_line = mean_shoreline(gdf)
    half_win = TRANSECT_SPACING / 2.0
    u_land = np.array([1.0, 0.7]) / np.linalg.norm([1.0, 0.7])
    rows, anchors = [], []
    for k, d in enumerate(np.arange(0, mean_line.length + 1, TRANSECT_SPACING)):
        d = min(d, mean_line.length)
        p = mean_line.interpolate(d)
        a = mean_line.interpolate(max(d - half_win, 0))
        b = mean_line.interpolate(min(d + half_win, mean_line.length))
        v = np.array([b.x - a.x, b.y - a.y])
        n = np.array([-v[1], v[0]])
        n = n / (np.linalg.norm(n) + 1e-12)
        if np.dot(n, u_land) < 0:
            n = -n
        p_land = (p.x + n[0] * BASELINE_OFFSET, p.y + n[1] * BASELINE_OFFSET)
        sea_len = TRANSECT_LENGTH - BASELINE_OFFSET
        p_sea = (p.x - n[0] * sea_len, p.y - n[1] * sea_len)
        anchors.append(p_land)
        rows.append({"transect_id": k + 1, "distance_along": d,
                     "geometry": LineString([p_land, p_sea])})
    return gpd.GeoDataFrame(rows, crs=crs), LineString(anchors), mean_line


# ---------------------------------------------------------------------------
# DISTANCES AND RATES
# ---------------------------------------------------------------------------
def transect_distances(gdf, transects):
    """Intersection of every shoreline with every transect; the position is
    the distance from the landward anchor (transect start point). MultiPoint
    intersections keep the most landward point; line overlaps use their
    midpoint."""
    base_pts = {r.transect_id: Point(r.geometry.coords[0])
                for r in transects.itertuples()}
    recs = []
    for row in gdf.itertuples():
        for tr in transects.itertuples():
            inter = tr.geometry.intersection(row.geometry)
            if inter.is_empty:
                continue
            if inter.geom_type == "MultiPoint":
                inter = min(inter.geoms,
                            key=lambda p: p.distance(base_pts[tr.transect_id]))
            elif inter.geom_type == "LineString":
                inter = inter.interpolate(0.5, normalized=True)
            elif inter.geom_type != "Point":
                continue
            recs.append({"transect_id": tr.transect_id, "date": row.date,
                         "year": row.year, "season": row.season,
                         "distance": inter.distance(base_pts[tr.transect_id])})
    return pd.DataFrame(recs)


def rates_from_distances(dist):
    """Per-transect change metrics from the distance time series:
    NSM  (m)      last minus first shoreline position;
    EPR  (m/yr)   NSM / elapsed years;
    LRR  (m/yr)   OLS slope of position vs time (+ p-value, R2);
    TSS  (m/yr)   Theil-Sen robust slope;
    volume / mass through the active-envelope model."""
    out = []
    for tid, g in dist.groupby("transect_id"):
        g = g.sort_values("date")
        if len(g) < 2:
            continue
        nsm = g.iloc[-1]["distance"] - g.iloc[0]["distance"]
        yrs = (g.iloc[-1]["date"] - g.iloc[0]["date"]).days / 365.25
        x = ((g["date"] - g["date"].min()).dt.days / 365.25).values
        y = g["distance"].values
        slope, _, r, p, _ = linregress(x, y)
        ts_slope = theilslopes(y, x)[0] if len(g) > 3 else np.nan
        out.append({"transect_id": tid, "n_obs": len(g), "nsm": nsm,
                    "epr": nsm / yrs if yrs else np.nan,
                    "lrr": slope, "lrr_r2": r ** 2, "lrr_p": p,
                    "theil_sen": ts_slope})
    df = pd.DataFrame(out)
    df["volume_m3"] = df["nsm"] * ACTIVE_HEIGHT * TRANSECT_SPACING
    df["mass_t"] = df["volume_m3"] * DENSITY * (1 - POROSITY) / 1000.0
    return df


def annual_volumes(dist):
    """Year-to-year volume change per transect (annual-mean positions)."""
    recs = []
    for tid, g in dist.groupby("transect_id"):
        ym = g.groupby("year")["distance"].mean()
        yrs = sorted(ym.index)
        for y1, y2 in zip(yrs[:-1], yrs[1:]):
            recs.append({"transect_id": tid, "period": f"{y1}-{y2}",
                         "year_start": y1,
                         "dvol": (ym[y2] - ym[y1]) * ACTIVE_HEIGHT * TRANSECT_SPACING})
    return pd.DataFrame(recs)


def annual_epr(dist):
    """Year-to-year EPR per transect (annual-mean positions)."""
    recs = []
    for tid, g in dist.groupby("transect_id"):
        ym = g.groupby("year")["distance"].mean()
        yrs = sorted(ym.index)
        for y1, y2 in zip(yrs[:-1], yrs[1:]):
            recs.append({"transect_id": tid, "year_start": y1,
                         "period": f"{y1}-{y2}",
                         "epr_annual": (ym[y2] - ym[y1]) / (y2 - y1)})
    return pd.DataFrame(recs)


def embayment_trend(dist):
    """Embayment-mean position trend: OLS + Theil-Sen + Mann-Kendall."""
    ts = dist.groupby("date")["distance"].mean()
    x = np.array([(d - ts.index[0]).days / 365.25 for d in ts.index])
    slope, intercept, r, p, _ = linregress(x, ts.values)
    tau, p_mk = kendalltau(np.arange(len(ts)), ts.values)
    sen = theilslopes(ts.values, x)[0]
    return {"lrr_m_per_yr": slope, "lrr_p": p, "lrr_r2": r ** 2,
            "theil_sen_m_per_yr": sen, "mk_tau": tau, "mk_p": p_mk}
