# Shoreline Extraction and Coastal Change Analysis — Anza Beach, Morocco (2018–2025)

This English-language package contains the code, input data, saved outputs, and publication figures used for the shoreline study. It covers Sentinel-2 data download, shoreline extraction with MNDWI and CNN methods, transect-based change calculations, and figure generation. It does not contain bathymetry or SDB (satellite-derived bathymetry) material.

## Contents

- `scripts/00_download_seasonal_composites.js`: Google Earth Engine script that creates and exports seasonal Sentinel-2 composites and MNDWI rasters to Google Drive.
- `scripts/01_classical_extraction.py`: classical MNDWI thresholding and vector shoreline extraction.
- `scripts/02_dl_extraction.py`: U-Net, SegNet, and FCN definitions, optional training, checkpoint loading, evaluation, and shoreline inference.
- `scripts/04_calculate_coastal_change.py`: common-transect shoreline-change calculations.
- `scripts/03_make_extraction_figures.py` and `scripts/05_make_change_figures.py`: publication figures from the archived data; shared styling is in `scripts/visual_style.py`.
- `scripts/06_make_architecture_figure.py`: code-derived U-Net, SegNet, and FCN architecture schematic used as Figure 3.
- `scripts/07_make_workflow_figure.py`: experimental-design flowchart used as Figure 2.
- `scripts/08_make_submission_rate_outputs.py`: builds descriptive rate summaries and the manuscript's EPR/LRR distribution figure from the already saved per-transect rate CSVs. It does not rerun shoreline extraction, CNN inference, or transect calculations.
- `data/`: the study's 32 seasonal multispectral composites, matching MNDWI rasters, label masks, saved shoreline vectors, and pixel metrics.
- `models/checkpoints/`: the saved U-Net, SegNet, and FCN checkpoints used by the inference script.
- `outputs/extraction/tables/`: archived extraction metrics and geometry summaries.
- `outputs/change/analysis/` and `outputs/change/tables/`: archived transects, shoreline distances, EPR/LRR products, summaries, and confidence intervals used for the study figures.
- `outputs/extraction/figures/` and `outputs/change/figures/`: saved figures and figure manifests.
- `requirements.txt`: Python dependencies.
- `run_pipeline.ps1`: runs the two figure-generation scripts using the archived study outputs.

## Run the figures from saved results

Install the Python dependencies from this folder:

```powershell
python -m pip install -r requirements.txt
```

Then run:

```powershell
./run_pipeline.ps1
python scripts/08_make_submission_rate_outputs.py
```

This default command regenerates the result figures from the archived vectors, metrics, rasters, and change-analysis tables. It does not rerun CNN inference, shoreline extraction, or coastal-change calculations. The methods schematics in Figures 2 and 3 are generated separately with scripts 07 and 06.

To run the visualization steps separately:

```powershell
python scripts/03_make_extraction_figures.py
python scripts/05_make_change_figures.py
python scripts/06_make_architecture_figure.py
python scripts/07_make_workflow_figure.py
```

## Download and processing workflow

1. Open `scripts/00_download_seasonal_composites.js` in the Google Earth Engine Code Editor. Sign in with an Earth Engine account, review the study-area and export-folder settings, run the script, then start the generated tasks from the Tasks tab. The source collection is `COPERNICUS/S2_SR_HARMONIZED`; exports are written to Google Drive.
2. Place downloaded `S2_YYYY_season.tif` and `MNDWI_YYYY_season.tif` files in `data/` using the existing filenames.
3. Run `python scripts/01_classical_extraction.py` for the MNDWI reference shorelines. Run `python scripts/02_dl_extraction.py` to load the included checkpoints, evaluate the networks, and generate CNN shoreline outputs. The CNN script can be configured to train models; training is not required to inspect the archived study outputs.
4. Run `python scripts/04_calculate_coastal_change.py` to recalculate the transect-based analysis from the available shoreline vectors.
5. Run `./run_pipeline.ps1` to make figures from the saved analysis products.

Extraction and change scripts write new outputs and may replace files with matching names. For the exact saved results used in the manuscript, use the archived files in this package and run only the default visualization pipeline.

## Interpretation and data notes

The CNN masks were trained using labels derived from the classical MNDWI pipeline. The archived pixel metrics therefore measure agreement with threshold-derived labels, not independent validation against field-survey shorelines. The 128 × 128 patches have a 64-pixel stride, so neighboring patches overlap; the random train/validation/test split is at patch level and is not held out by scene, year, or spatial block. Test-set scores may therefore be optimistic and must not be interpreted as independent shoreline accuracy.

For the manuscript's change-rate comparison, the primary observations are the four saved `outputs/change/analysis/rates_<method>.csv` files. Each contains the same 40 transects and 32 seasonal shoreline observations per transect. `scripts/08_make_submission_rate_outputs.py` regenerates the descriptive summaries directly from these files. Erosion and accretion are the negative and positive EPR signs; the reported zero-rate share means exactly EPR = 0 and is not a tolerance-based stability class. Rate distributions are descriptive; no between-method significance test is reported because the methods are paired on the same transects and neighboring transects may be spatially dependent.

The change analysis does not apply tidal or runup corrections. Rates describe shoreline-position changes in the supplied seasonal composites and may include water-level influence. Volume and mass estimates in the archived analysis are scenario calculations based on a 5 m active height, 50 m transect spacing, sediment density of 1,600 kg/m³, and porosity of 0.4; they are not direct seabed-volume measurements.

Sentinel-2 data are from the Copernicus programme and were accessed through Google Earth Engine. The code, inputs, saved outputs, and visualization scripts are publicly available at https://github.com/bouhaaddii/Anza_Shoreline. The repository currently has no DOI. The manuscript links to this public record; an archived release with a persistent DOI may be added if the authors choose to create one.
