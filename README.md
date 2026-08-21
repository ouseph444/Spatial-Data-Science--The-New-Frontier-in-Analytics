# Spatial Data Science: The New Frontier in Analytics

### A practical, project-based Python GIS course — GeoPandas · Shapely · Rasterio

A complete, self-contained course built around **the Vallmara Basin**, an
entirely fictional coastal region. Nothing is downloaded; the whole 16 MB
dataset is generated on your machine from a fixed random seed, so every number
in the course is reproducible.

---

## What is in this folder

| File | What it is |
|---|---|
| `Spatial_Data_Science_Course.ipynb` | **The course.** 189 cells (127 markdown, 62 code) — lessons, exercises, solutions and the capstone |
| `Modules/` | Individual module folders that break down Spatial_Data_Science_Course.ipynb. Each module will be updated as it is completed. |
| `Spatial_Data_Science_Course.pdf` | The same course as a Jupyter-styled printable study guide (201 pages, A4) |
| `generate_data.py` | Builds the entire fictional dataset (~15 s, ~16 MB) |
| `requirements.txt` | pip dependencies |
| `environment.yml` | conda/mamba dependencies |
| `course_src/` | The source the notebook and PDF are generated from |
| `data/` | Created by `generate_data.py` (see `data/README_DATA.md`) |


---

## Quick start

```bash
# 1. create an environment (Python 3.11 or 3.12 recommended)
python3.12 -m venv gis-env
source gis-env/bin/activate            # Windows: gis-env\Scripts\activate
pip install -r requirements.txt

# 2. generate the fictional dataset  (~15 seconds, ~16 MB)
python generate_data.py

# 3. open the course
jupyter lab Spatial_Data_Science_Course.ipynb
```

Prefer conda? `conda env create -f environment.yml && conda activate gis-course`.

Then **run the notebook from top to bottom.** Every one of the 62 code cells
runs clean. Expect about **4 minutes** on a warm dataset, or **6 minutes** on a
first run including data generation; the slowest cells are the
spatially-validated machine-learning lessons (A11, A12) and the capstone.

---

## The course at a glance

| Module | Level | Lessons | Focus |
|---|---|---|---|
| **0** | Setup | 5 | Environment, folder layout, the dataset and its ground truth |
| **1** | Beginner | 14 | GeoDataFrames, geometry, CRS, plotting, spatial joins, opening rasters |
| **2** | Intermediate | 16 | Buffers, overlays, joins in depth, raster masking/resampling/zonal statistics, data cleaning |
| **3** | Advanced | 14 | MCDA, accessibility, autocorrelation, hotspots, clustering, spatial ML with honest validation |
| **4** | Capstone | 12 stages | A complete climate-resilience assessment, end to end |
| **+** | Exercises | 23 problems | Beginner → Challenge, with full worked solutions |

Every lesson follows the same four-cell rhythm: **what and why → code →
line-by-line explanation → what you should see**. Expected outputs quote the
real numbers, so you can tell success from silent failure.

---

## The dataset

**The Vallmara Basin**, Republic of Kestria — a fictional 48 × 36 km coastal
region with ~640 000 residents, six rivers, a mountain ridge and a flood problem.

* **Vector** — 12 GeoPackage layers (24 districts, 460 census blocks, 5 200
  buildings, 441 land-use polygons, 181 flood-hazard polygons, 83 facilities,
  transit, rivers, coastline) plus GeoJSON roads and protected areas, plus a
  legacy shapefile.
* **Raster** — 7 GeoTIFFs: 25 m DEM, land cover, rainfall, NDVI, population
  density, a 4-band multispectral image, and land-surface temperature.
* **Tabular** — 5 CSVs: sensor stations and readings, flood incidents,
  district socio-economics, a land-cover legend.
* **Three CRS on purpose** — EPSG:32633 (UTM 33N), EPSG:4326 and EPSG:3857.
* **Twelve deliberate data-quality defects** — invalid geometries, empty
  geometries, sentinel values, impossible dates, swapped coordinates, duplicate
  and orphan keys, inconsistent categories. All documented in
  `data/README_DATA.md`, all fixed during the course.

**The generating process is documented.** Rainfall, land-surface temperature,
PM2.5, population density and flood-incident reporting all follow stated
equations, so at every step you can ask the question that matters:
**did my analysis recover the truth?**

---

## Recommended folder structure

```
spatial-data-science/
├── Spatial_Data_Science_Course.ipynb
├── Spatial_Data_Science_Course.pdf
├── generate_data.py
├── requirements.txt
├── environment.yml
├── course_src/                     <- regenerate the notebook and PDF
└── data/                           <- created by generate_data.py
    ├── README_DATA.md
    ├── vector/   raster/   tabular/   outputs/
```

Everything you create goes in `data/outputs/`. The shipped dataset is **16 MB**;
a full notebook run adds about **55 MB** of derived rasters and results there.
Delete `data/outputs/` any time — the notebook recreates it.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: libgdal.so…` / `DLL load failed` | pip wheels for `rasterio` and `geopandas` bundled different GDAL builds | Install everything from **one** channel: all-pip or all-conda-forge. Never mix. |
| `CRSError: Invalid projection` | Stale PROJ data directory | `pip install --force-reinstall pyproj`; unset any stray `PROJ_LIB` |
| `ModuleNotFoundError: statsmodels` | Optional in some stacks | `pip install statsmodels` (needed for Lesson A9) |
| `ImportError: Missing optional dependency 'pyarrow'` | GeoParquet cell | `pip install pyarrow`, or skip — the cell degrades gracefully |
| `mapclassify … required for the 'scheme' keyword` | Missing choropleth classifier | `pip install mapclassify` |
| `contextily` connection error | No internet | Basemaps are optional throughout; skip those lines |
| Shapefile attributes are gibberish | `.dbf` encoding | `gpd.read_file(path, encoding="latin-1")` |
| `TopologyException: found non-noded intersection` | Invalid input geometry | Run `make_valid()` first — Lesson I2 |
| Everything plots in a tiny corner | Layers in different CRS | Reproject **all** layers before plotting — Lesson B5 |
| A notebook cell is very slow | A11/A12 fit models with spatial CV | Expected; ~60–90 s each |

---

## Regenerating the notebook and PDF

The `.ipynb` and the `.pdf` are both generated from `course_src/`, which is why
they can never disagree.

```bash
pip install markdown pygments weasyprint      # PDF toolchain only
python course_src/build_notebook.py           # -> Spatial_Data_Science_Course.ipynb
python course_src/build_pdf.py                # -> Spatial_Data_Science_Course.pdf
```

---

## A note on the fiction

Every place, person, agency, measurement and currency in this course is
invented. The Vallmara Basin does not exist. That is deliberate: with real open
data you never know the true generating process, so you cannot tell a correct
analysis from a plausible-looking wrong one. Here you can — which is what makes
the validation exercises throughout the course possible.
