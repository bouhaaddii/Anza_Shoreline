#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared core for Study 1 shoreline extraction (classical and deep learning).

Every function is documented step by step so the methodology is fully
transparent and citable. Both `01_classical_extraction.py` and
`02_dl_extraction.py` import from this module, which guarantees that the
vectorisation chain is strictly identical for the two approaches (the only
difference between pipelines is how the binary water mask is produced).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import LineString
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label as ndlabel
from skimage import measure
from pyproj import Transformer

# ----------------------------------------------------------------------------
# PARAMETERS (identical for both pipelines)
# ----------------------------------------------------------------------------
MNDWI_THRESHOLD = 0.0     # water if MNDWI > 0 (Xu, 2006)
SMOOTH_SIGMA = 1.5        # Gaussian pre-smoothing of the raster (pixels)
MIN_AREA_RATIO = 0.10     # keep components >= 10% of the largest one
BORDER_MARGIN = 5         # drop contour points within 5 px of the image edge
MIN_SEGMENT_PTS = 50      # keep the longest inland segment above this size
LINE_SMOOTH_SIGMA = 1.0   # Gaussian smoothing of the vector line (points)
SIMPLIFY_TOL_M = 0.5      # Douglas-Peucker tolerance in metres (UTM)
UTM_EPSG = 32629          # UTM zone 29N

SEASONS = ["winter", "spring", "summer", "autumn"]
SEASON_MONTH = {"winter": "01", "spring": "04", "summer": "07", "autumn": "10"}


# ----------------------------------------------------------------------------
# RASTER I/O
# ----------------------------------------------------------------------------
def read_mndwi(path):
    """Read a single-band MNDWI GeoTIFF; returns (array, transform, crs)."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.transform, src.crs


def parse_name(path):
    """MNDWI_YYYY_season.tif -> (year:int, season:str, date:pd.Timestamp)."""
    stem = Path(path).stem                      # MNDWI_2018_winter
    _, year, season = stem.split("_")
    date = pd.Timestamp(f"{year}-{SEASON_MONTH[season]}-01")
    return int(year), season, date


# ----------------------------------------------------------------------------
# STEP 1 - BINARY WATER MASK (classical definition; DL replaces this step)
# ----------------------------------------------------------------------------
def water_mask_classical(mndwi):
    """Gaussian smoothing followed by zero-thresholding of the MNDWI."""
    smoothed = gaussian_filter(mndwi, sigma=SMOOTH_SIGMA)
    return (smoothed > MNDWI_THRESHOLD).astype(np.uint8)


# ----------------------------------------------------------------------------
# STEP 2 - MORPHOLOGICAL CLEANING
# ----------------------------------------------------------------------------
def clean_mask(mask):
    """Connected-component filtering: keep the components whose area is at
    least MIN_AREA_RATIO of the largest water body (removes wave foam,
    boats, small ponds...)."""
    lab, n = ndlabel(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = np.where(sizes >= sizes.max() * MIN_AREA_RATIO)[0]
    return np.isin(lab, keep).astype(np.uint8)


# ----------------------------------------------------------------------------
# STEP 3 - LAND-WATER CONTOUR
# ----------------------------------------------------------------------------
def extract_contour(mask, valid=None):
    """Marching-squares contour of the cleaned mask at level 0.5. The longest
    contour is retained as the candidate land-water interface. Points closer
    than BORDER_MARGIN pixels to the raster edge OR to a NoData pixel are
    removed (image-edge and NoData-collar artefacts), and the longest
    remaining continuous segment with at least MIN_SEGMENT_PTS points becomes
    the shoreline (row/col coordinates)."""
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)                       # longest
    H, W = mask.shape
    inside = ((contour[:, 0] > BORDER_MARGIN) & (contour[:, 0] < H - BORDER_MARGIN) &
              (contour[:, 1] > BORDER_MARGIN) & (contour[:, 1] < W - BORDER_MARGIN))
    if valid is not None:
        # treat NoData areas exactly like the image border: erode the valid
        # footprint by BORDER_MARGIN and reject contour points outside it
        from scipy.ndimage import binary_erosion
        valid_er = binary_erosion(valid, iterations=BORDER_MARGIN)
        rr = np.clip(np.round(contour[:, 0]).astype(int), 0, H - 1)
        cc = np.clip(np.round(contour[:, 1]).astype(int), 0, W - 1)
        inside &= valid_er[rr, cc]
    # orientation filter: the shoreline of interest runs roughly north-south,
    # whereas scene-edge and swath artefacts form long, almost horizontal
    # boundary segments. Keep points whose local (smoothed) tangent is not
    # quasi-horizontal: |d(row)| > 0.35 |d(col)|.
    from scipy.ndimage import uniform_filter1d
    d_row = uniform_filter1d(np.gradient(contour[:, 0]), 9)
    d_col = uniform_filter1d(np.gradient(contour[:, 1]), 9)
    not_horizontal = np.abs(d_row) > 0.35 * np.abs(d_col)
    not_horizontal = uniform_filter1d(not_horizontal.astype(float), 9) > 0.5
    inside &= not_horizontal
    # split into continuous runs of inside points, keep the longest
    best, cur = [], []
    for pt, ok in zip(contour, inside):
        if ok:
            cur.append(pt)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    if len(best) < MIN_SEGMENT_PTS:
        return None
    return np.array(best)


# ----------------------------------------------------------------------------
# STEP 4 - VECTORISATION AND REPROJECTION
# ----------------------------------------------------------------------------
def contour_to_utm_line(contour_rc, transform, src_crs):
    """(row, col) contour -> smoothed LineString in UTM 29N.
    (i)   light Gaussian smoothing of the pixel coordinates,
    (ii)  pixel -> geographic coordinates through the affine transform,
    (iii) reprojection to UTM zone 29N,
    (iv)  Douglas-Peucker simplification at SIMPLIFY_TOL_M metres."""
    rr = gaussian_filter1d(contour_rc[:, 0], LINE_SMOOTH_SIGMA)
    cc = gaussian_filter1d(contour_rc[:, 1], LINE_SMOOTH_SIGMA)
    xs, ys = rasterio.transform.xy(transform, rr, cc)
    tf = Transformer.from_crs(src_crs, UTM_EPSG, always_xy=True)
    xu, yu = tf.transform(xs, ys)
    line = LineString(np.column_stack([xu, yu])).simplify(SIMPLIFY_TOL_M)
    return line


def extract_shoreline(mask, transform, src_crs, valid=None):
    """Full step 2-4 chain from a binary water mask to a UTM shoreline.
    `valid` is the finite-data footprint of the source raster (True where the
    MNDWI is defined); NoData collars are excluded like image borders."""
    cleaned = clean_mask(mask)
    contour = extract_contour(cleaned, valid)
    if contour is None:
        return None, cleaned
    return contour_to_utm_line(contour, transform, src_crs), cleaned


# ----------------------------------------------------------------------------
# SHAPEFILE EXPORT
# ----------------------------------------------------------------------------
def save_shorelines(records, out_dir, tag):
    """records: list of dicts with keys geometry/year/season/date/length_m.
    Writes the combined shapefile + a summary CSV; returns the GeoDataFrame."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(records, crs=UTM_EPSG)
    gdf["date"] = gdf["date"].astype(str)
    gdf.to_file(out_dir / f"all_coastlines_{tag}_utm.shp")
    gdf.drop(columns="geometry").to_csv(out_dir / f"coastlines_summary_{tag}.csv",
                                        index=False)
    return gdf
