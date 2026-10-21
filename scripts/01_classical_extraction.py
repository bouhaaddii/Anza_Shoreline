#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STUDY 1 - STEP 01: CLASSICAL SHORELINE EXTRACTION (MNDWI THRESHOLDING)

Detailed, fully documented implementation of the traditional pipeline:

    MNDWI raster
      -> Gaussian smoothing (sigma = 1.5 px)
      -> zero threshold (water: MNDWI > 0)              [Xu, 2006]
      -> connected-component cleaning (>= 10% of the largest water body)
      -> marching-squares land-water contour
      -> border-artefact removal (5-px margin) + longest inland segment
      -> vector smoothing, reprojection to UTM 29N, 0.5-m simplification

Inputs :  data/MNDWI_YYYY_season.tif           (32 rasters, 2018-2025)
Outputs:  outputs/classical_shorelines/        shapefile + summary CSV
          outputs/classical_shorelines/masks/  cleaned water masks (GeoTIFF)
          outputs/figures/Fig07_classical_processing_steps.(png|pdf)
          outputs/figures_manifest.json        (updated)

Run:  python scripts/01_classical_extraction.py
"""

import json
import glob
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import extraction_core as core

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs" / "extraction"
SHORE_DIR = OUT / "classical_shorelines"
MASK_DIR = SHORE_DIR / "masks"
FIG = OUT / "figures"
for d in (SHORE_DIR, MASK_DIR, FIG):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "savefig.dpi": 300, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})

STEP_FIGURE_SCENE = ("2024", "winter")     # scene used for the diagram


def update_manifest(entry):
    mpath = OUT / "figures_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else []
    manifest = [m for m in manifest if m["figure"] != entry["figure"]]
    manifest.append(entry)
    mpath.write_text(json.dumps(manifest, indent=2))


def processing_steps_figure(mndwi, raw_mask, cleaned, contour, line_utm):
    """Five-panel diagram of the classical chain (used as a methods figure)."""
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.1), constrained_layout=True)
    axes[0].imshow(mndwi, cmap="RdBu", vmin=-0.8, vmax=0.8)
    axes[0].set_title("1. MNDWI raster")
    axes[1].imshow(raw_mask, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("2. Threshold > 0")
    axes[2].imshow(cleaned, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("3. Component cleaning")
    axes[3].imshow(cleaned, cmap="Greys_r", vmin=0, vmax=1, alpha=0.35)
    if contour is not None:
        axes[3].plot(contour[:, 1], contour[:, 0], color="#D55E00", lw=1.2)
    axes[3].set_title("4. Coastal contour")
    axes[4].imshow(mndwi, cmap="Greys", vmin=-1, vmax=1, alpha=0.6)
    if contour is not None:
        axes[4].plot(contour[:, 1], contour[:, 0], color="#0072B2", lw=1.4)
    axes[4].set_title("5. Final shoreline")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    for i, ax in enumerate(axes):
        ax.text(0.02, 0.98, "abcde"[i], transform=ax.transAxes, weight="bold",
                fontsize=11, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
    name = "Fig07_classical_processing_steps"
    fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    update_manifest({"figure": name, "file": f"{name}.png", "caption":
        "Processing chain of the classical MNDWI shoreline extraction, "
        f"illustrated on the {STEP_FIGURE_SCENE[1]} {STEP_FIGURE_SCENE[0]} "
        "scene: (a) MNDWI raster; (b) binary water mask after Gaussian "
        "smoothing and zero-thresholding; (c) connected-component cleaning; "
        "(d) marching-squares land-water contour after border-artefact "
        "removal; (e) final smoothed shoreline."})
    print(f"  [OK] {name}")


def main():
    print("STEP 01 - classical MNDWI extraction")
    records = []
    files = sorted(glob.glob(str(DATA / "MNDWI_*.tif")))
    print(f"  {len(files)} MNDWI rasters found")

    for path in files:
        year, season, date = core.parse_name(path)
        mndwi, transform, crs = core.read_mndwi(path)

        valid = np.isfinite(mndwi)
        raw_mask = core.water_mask_classical(mndwi)             # step 1
        line, cleaned = core.extract_shoreline(raw_mask, transform, crs,
                                               valid=valid)     # steps 2-4

        # save the cleaned mask (GeoTIFF) - reused as DL training label
        mask_path = MASK_DIR / f"mask_{year}_{season}.tif"
        if not mask_path.exists():
            with rasterio.open(
                    mask_path, "w", driver="GTiff",
                    height=cleaned.shape[0], width=cleaned.shape[1], count=1,
                    dtype="uint8", crs=crs, transform=transform) as dst:
                dst.write(cleaned[None].astype(np.uint8))

        if line is None:
            print(f"  [WARN] no shoreline for {year} {season}")
            continue
        records.append({"geometry": line, "year": year, "season": season,
                        "date": date, "length_m": round(line.length, 1)})
        print(f"  {year} {season:<7} length = {line.length:7.1f} m")

        if (str(year), season) == STEP_FIGURE_SCENE:
            contour = core.extract_contour(cleaned, valid)
            processing_steps_figure(mndwi, raw_mask, cleaned, contour, line)

    gdf = core.save_shorelines(records, SHORE_DIR, "classical")
    print(f"\n  {len(gdf)} shorelines -> {SHORE_DIR}")
    print(f"  mean length = {gdf['length_m'].mean():.1f} m")


if __name__ == "__main__":
    main()
