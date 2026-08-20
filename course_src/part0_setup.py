# -*- coding: utf-8 -*-
"""Module 0 - Orientation, environment and the dataset."""
from _cells import md, code

CELLS = [

md(r'''
# Spatial Data Science: The New Frontier in Analytics

### A practical, project-based course in Python GIS — GeoPandas · Shapely · Rasterio

---

**Who this course is for.** You already know Python, statistics and data
analysis. You are comfortable with `pandas`, you know what a train/test split
is, and you can read a regression table. What you have *not* done much of is
**spatial** analysis — and that turns out to be a genuinely different discipline,
not just `pandas` with an `x` and a `y` column.

**The one-sentence pitch for spatial data science.** Ordinary data science
assumes rows are exchangeable and independent; spatial data science starts from
the opposite assumption — *Tobler's First Law*: **"everything is related to
everything else, but near things are more related than distant things."**
Once observations carry a location, three things change at once:

| | Ordinary tabular analysis | Spatial analysis |
|---|---|---|
| **Rows** | Independent samples | Auto-correlated; nearby rows leak information into each other |
| **Joins** | On a key (`id == id`) | On a *relation* (`intersects`, `within`, `nearest`) |
| **Features** | Given in the table | *Engineered from geometry*: distance to, density of, elevation at, area of |
| **Validation** | Random k-fold | Random k-fold **overfits**; you need spatially blocked folds |
| **Units** | Whatever the column says | Depend on the coordinate reference system — degrees are not metres |

Everything in this course exists to make those five rows second nature.

---

## What you will build

You are the newly hired spatial data scientist for the regional government of the
**Vallmara Basin**, a completely fictional 48 km × 36 km coastal region in the
imaginary Republic of Kestria. Over 4 modules you will progress from "how do I
open a shapefile" to delivering a **climate-resilience assessment**: a
statistically defensible, map-backed estimate of which neighbourhoods are most
exposed to flooding, least served by emergency infrastructure, and most
vulnerable in socio-economic terms.

## Course structure

| Module | Level | Lessons | Focus |
|---|---|---|---|
| **0** | Setup | 5 | Environment, folder layout, generating the dataset |
| **1** | Beginner | 14 | GeoDataFrames, geometry, CRS, plotting, basic joins, opening rasters |
| **2** | Intermediate | 16 | Buffers, overlays, spatial joins in depth, raster masking / resampling / zonal statistics, data cleaning |
| **3** | Advanced | 14 | MCDA, accessibility, spatial autocorrelation, hotspots, clustering, spatial ML with proper validation |
| **4** | Capstone | 12 stages | A complete end-to-end spatial data science project |
| **+** | Exercises | 4 sets | Beginner → Challenge, with a full **Solutions** section at the end |

**Lesson format.** Every lesson follows the same four-cell rhythm:

1. **Markdown** — what we are about to learn, why it matters, the concept, and the expected outcome.
2. **Code** — the runnable cell.
3. **Explanation** — line-by-line for anything unfamiliar.
4. **Expected output** — what you should see on screen, so you can tell success from silent failure.

> **How to use this document.** Run every cell. When a lesson says "notice that
> the number is wrong", stop and make sure you understand *why* it is wrong
> before moving on. Roughly half of practical GIS competence is knowing which
> results to distrust.
'''),

md(r'''
## 0.1 — Environment and installation

### Recommended Python version

**Python 3.11 or 3.12.** These have full binary-wheel coverage for the whole
geospatial stack. Python 3.13 mostly works; 3.14 is still ahead of some wheels
at the time of writing. If in doubt, use **3.12**.

### The packages, and why each one is here

| Package | Role in this course | Why you need it |
|---|---|---|
| **geopandas** ≥ 1.0 | The spine of the course | A `DataFrame` with a geometry column: spatial joins, overlays, CRS handling, plotting |
| **shapely** ≥ 2.0 | Geometry engine | The objects *inside* the geometry column; buffers, intersections, predicates. Shapely 2.x is vectorised — an order of magnitude faster than 1.8 |
| **rasterio** ≥ 1.3 | Raster I/O and processing | Reads/writes GeoTIFF, gives you NumPy arrays plus the affine transform that ties them to the ground |
| **numpy** | Raster maths | Rasters *are* NumPy arrays; every raster operation is array algebra |
| **pandas** | Attribute tables | GeoDataFrame is a subclass; all your `groupby`/`merge` skills transfer directly |
| **matplotlib** | All static maps | GeoPandas `.plot()` returns matplotlib axes; you compose maps as you would any figure |
| **seaborn** | Statistical graphics | Distribution and relationship plots for the non-spatial half of the analysis |
| **scipy** | Distances, KDE, filters | `cKDTree` for nearest-neighbour work, `ndimage` for raster filters/focal statistics, `stats` for tests |
| **scikit-learn** | Machine learning | Clustering (DBSCAN/KMeans) and the predictive models in Module 3 |
| **statsmodels** | Regression *inference* | scikit-learn gives you predictions; statsmodels gives you standard errors, t-statistics, p-values and AIC — which is what Lesson A9 is about |
| **pyarrow** | GeoParquet I/O | The fastest way to store intermediate spatial results; optional but recommended |
| **pyogrio** | Fast vector I/O | GeoPandas 1.x uses it as the default engine; 5–20× faster than Fiona for large files |
| **fiona** | Alternative vector I/O | Still the fallback engine; useful when you need per-feature streaming |
| **contextily** | Web basemap tiles | Optional context under your maps (needs internet; the course works without it) |
| **mapclassify** | Choropleth classification | Quantiles, natural breaks (Fisher–Jenks), std-mean — needed for honest thematic maps |

### `pip` installation (virtual environment — recommended)

```bash
python3.12 -m venv gis-env
source gis-env/bin/activate          # Windows: gis-env\Scripts\activate
python -m pip install --upgrade pip

pip install "numpy>=1.26" "pandas>=2.1" "geopandas>=1.0" "shapely>=2.0" \
            "rasterio>=1.3" "pyogrio>=0.8" "fiona>=1.9" \
            "matplotlib>=3.8" "seaborn>=0.13" "scipy>=1.11" \
            "scikit-learn>=1.4" "statsmodels>=0.14" "pyarrow>=15.0" \
            "mapclassify>=2.6" "contextily>=1.5" "jupyterlab>=4.0"
```

Or, from the `requirements.txt` shipped with this course:

```bash
pip install -r requirements.txt
```

### Conda / mamba installation (recommended if pip gives you GDAL trouble)

Conda ships pre-built GDAL/GEOS/PROJ binaries, which removes the single most
common source of installation pain.

```bash
conda create -n gis-env -c conda-forge python=3.12 \
    geopandas shapely rasterio pyogrio fiona \
    numpy pandas matplotlib seaborn scipy scikit-learn statsmodels \
    pyarrow mapclassify contextily jupyterlab
conda activate gis-env
```

(Substitute `mamba` for `conda` if you have it — same commands, much faster solve.)

### Launching Jupyter

```bash
cd path/to/this/course
jupyter lab                 # modern interface, recommended
# or
jupyter notebook            # classic interface
```

Then open `Spatial_Data_Science_Course.ipynb`.

### Recommended folder structure

```
spatial-data-science/
├── Spatial_Data_Science_Course.ipynb   <- this notebook
├── Spatial_Data_Science_Course.pdf     <- the printable study guide
├── generate_data.py                    <- builds the fictional dataset
├── requirements.txt
├── environment.yml
└── data/                               <- created by generate_data.py
    ├── README_DATA.md
    ├── vector/
    │   ├── vallmara.gpkg               (12 layers, EPSG:32633)
    │   ├── roads.geojson               (EPSG:4326)
    │   ├── protected_areas.geojson     (EPSG:4326)
    │   └── schools_shp/schools.shp     (EPSG:32633)
    ├── raster/
    │   ├── dem_25m.tif
    │   ├── landcover_25m.tif
    │   ├── rainfall_annual_250m.tif
    │   ├── ndvi_50m.tif
    │   ├── popdens_100m.tif
    │   ├── multispectral_50m.tif
    │   └── lst_summer_100m_3857.tif    (EPSG:3857 - on purpose)
    ├── tabular/
    │   ├── sensor_stations.csv
    │   ├── sensor_readings.csv
    │   ├── flood_incidents.csv
    │   ├── district_socioeconomic.csv
    │   └── landcover_legend.csv
    └── outputs/                        <- everything you create goes here
```

### Generating the dataset

From the course folder, with your environment active:

```bash
python generate_data.py
```

It takes 10–20 seconds and writes about 16 MB into `./data/`. It is driven by a
fixed seed (42), so your files will be identical to the ones described here.
Re-running it simply overwrites everything — safe at any time.

### Troubleshooting the usual suspects

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: libgdal.so...` / `DLL load failed` | pip wheels for `rasterio` and `geopandas` bundled *different* GDAL builds | Install both from the **same** channel: either all-pip or all-conda-forge. Never mix. |
| `CRSError: Invalid projection` | Stale or missing PROJ data directory | `pip install --force-reinstall pyproj`, or unset a stray `PROJ_LIB` environment variable |
| `.shp` opens but attributes are gibberish | Shapefile `.dbf` encoding | `gpd.read_file(path, encoding="latin-1")`, or move to GeoPackage |
| `fiona.errors.DriverError: ... not recognized` | Missing sidecar files | A shapefile is **4+ files** (`.shp .shx .dbf .prj`). Copy the whole folder. |
| Reading a big file takes minutes | Fiona engine on a huge layer | `gpd.read_file(..., engine="pyogrio")` (default in GeoPandas 1.x) |
| `contextily` raises a connection error | No internet | Basemaps are optional throughout; skip those two lines |
| Everything plots in a tiny corner of the map | Layers in different CRS | Reproject **all** layers to one CRS before plotting (Lesson B5) |
| `TopologyException: found non-noded intersection` | Invalid input geometry | Run `make_valid()` first (Lesson I2) |
| Memory error on a raster | Reading a whole large raster at once | Read a window, or use `rasterio` block iteration (Lesson I13) |
'''),

md(r'''
## 0.2 — Check your environment

**What we are about to do.** Import every library the course uses and print its
version. **Why it matters:** 90% of "the code doesn't work" reports in GIS are
environment problems, not logic problems. Catching a broken GDAL binding *now*
costs you two minutes; catching it in Module 3 costs you an afternoon.

**Concept — the GIS software stack.** Python geospatial libraries are thin,
pleasant wrappers over three C/C++ libraries that do the real work:

* **GEOS** — planar geometry predicates and operations (used by Shapely).
* **GDAL/OGR** — reading and writing 200+ raster and vector formats (used by Rasterio, Pyogrio, Fiona).
* **PROJ** — coordinate reference systems and datum transformations (used by pyproj, and hence by everything).

When a geospatial install breaks, it is almost always because two Python
packages were compiled against *different versions* of one of those three.

**Expected outcome.** A version table, and a confirmation line that GEOS, GDAL
and PROJ are all reachable.

**What the next cell does:** imports the stack, prints versions, and prints the
underlying C-library versions so you can confirm a coherent install.
'''),

code(r'''
import sys, platform

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn

print(f"Python        {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
print("-" * 58)
for name, mod in [("numpy", np), ("pandas", pd), ("geopandas", gpd),
                  ("shapely", shapely), ("rasterio", rasterio),
                  ("matplotlib", matplotlib), ("seaborn", sns),
                  ("scipy", scipy), ("scikit-learn", sklearn)]:
    print(f"{name:<14} {mod.__version__}")

print("-" * 58)
print(f"GEOS (via shapely)   {shapely.geos_version_string}")
print(f"GDAL (via rasterio)  {rasterio.__gdal_version__}")
import pyproj
print(f"PROJ (via pyproj)    {pyproj.proj_version_str}")
print(f"vector I/O engine    {gpd.options.io_engine or 'pyogrio (default)'}")
'''),

md(r'''
**Explanation.**

* `shapely.geos_version_string` — the GEOS build Shapely is linked against.
  Shapely **2.0+** is required for this course: it rewrote the geometry layer on
  top of NumPy arrays, which is why `gdf.geometry.buffer(500)` on 100 000 rows
  now takes milliseconds rather than seconds.
* `rasterio.__gdal_version__` — GDAL ≥ 3.4 is what you want. GDAL 3 changed the
  **axis-order convention**: `EPSG:4326` is officially *(latitude, longitude)*.
  Rasterio and GeoPandas always hand you *(x, y)* = *(longitude, latitude)*
  regardless, but you will meet other tools that do not — this is a classic
  source of "my points are in the Indian Ocean" bugs.
* `pyproj.proj_version_str` — PROJ ≥ 8 supports datum-shift grids downloaded on
  demand, which matters for sub-metre accuracy work.
* `gpd.options.io_engine` — GeoPandas 1.x reads vector files through **pyogrio**
  by default (vectorised, fast). `engine="fiona"` is the streaming alternative.

**Expected output.** Nine version lines, then three C-library lines. You need at
minimum: geopandas ≥ 1.0, shapely ≥ 2.0, rasterio ≥ 1.3. If any import raises,
fix it now using the troubleshooting table above.
'''),

md(r'''
## 0.3 — Generate the dataset

**What we are about to do.** Run the dataset generator and confirm every file
landed where the course expects it.

**Why it matters.** Real GIS projects fail on *data logistics* far more often
than on analysis. Establishing a single, explicit `DATA` root — and never
hard-coding a path again — is the cheapest reproducibility win available.

**Concept — why fictional data is the right teaching data.** Real open GIS data
carries licences, downloads, breaking API changes and hidden regional quirks.
More importantly, with real data you never know the *true* generating process,
so you cannot tell a correct analysis from a plausible-looking wrong one. Here
the data-generating process is known and documented, so at every step you can
ask: **did my analysis recover the truth?** That is the same logic as a
simulation study in statistics.

**Expected outcome.** The generator prints a progress log and a summary of row
counts; then a directory listing confirms the file tree.

**What the next cell does:** runs `generate_data.py` if `data/` is missing (so
re-running the notebook is cheap), then lists every generated file with its size.
'''),

code(r'''
import subprocess
from pathlib import Path

# --- Single source of truth for every path in this notebook -----------------
ROOT = Path.cwd()                 # the folder containing generate_data.py
DATA = ROOT / "data"
VEC  = DATA / "vector"
RAS  = DATA / "raster"
TAB  = DATA / "tabular"
OUT  = DATA / "outputs"           # everything we create goes here

GPKG = VEC / "vallmara.gpkg"      # the main multi-layer GeoPackage

if not GPKG.exists():
    print("data/ not found - running generate_data.py ...\n")
    res = subprocess.run([sys.executable, str(ROOT / "generate_data.py")],
                         capture_output=True, text=True)
    print(res.stdout[-2000:] or res.stderr[-2000:])
else:
    print("Dataset already present - skipping generation.\n")

OUT.mkdir(parents=True, exist_ok=True)

total = 0
for p in sorted(DATA.rglob("*")):
    if p.is_file() and not p.name.startswith("."):
        size = p.stat().st_size
        total += size
        print(f"  {str(p.relative_to(ROOT)):<52} {size/1e6:8.2f} MB")
print(f"\nTotal: {total/1e6:.1f} MB")
'''),

md(r'''
**Explanation.**

* `Path.cwd()` — the notebook's working directory. If you moved the notebook,
  set `ROOT` explicitly instead, e.g. `ROOT = Path("/Users/you/spatial-data-science")`.
* `sys.executable` rather than `"python"` — this guarantees the generator runs
  in **the same interpreter as the notebook**. Using `"python"` is a classic way
  to accidentally generate data with a different environment that lacks rasterio.
* `capture_output=True` keeps the generator's log out of the way unless you want it.
* `DATA.rglob("*")` — recursive glob; `p.is_file()` filters out directories.
* `OUT.mkdir(parents=True, exist_ok=True)` — the idempotent way to ensure an
  output folder exists. Never wrap this in an `if not exists` check; `exist_ok`
  already does it race-free.

**Expected output.** About **21 files, ~16 MB total**, laid out as:

| Folder | Files |
|---|---|
| `data/raster/` | 7 GeoTIFFs (`dem_25m.tif` is the largest, ~7 MB) |
| `data/vector/` | `vallmara.gpkg` (~2.8 MB), 2 GeoJSONs, a 5-file shapefile |
| `data/tabular/` | 5 CSVs |
| `data/` | `README_DATA.md` |

If the generator ran, you will additionally see its 14-step progress log ending
with row counts (24 districts, 460 census blocks, 791 road segments,
5 200 buildings, 83 facilities, 181 flood-zone polygons, 393 flood incidents).
'''),

md(r'''
## 0.4 — The Vallmara Basin: scenario and data dictionary

### The story

The **Vallmara Basin** occupies 1 395 km² of the west coast of the fictional
Republic of Kestria. The Kestrian Sea lies to the west; the land climbs eastward
across a flat coastal plain, through farmland and forest, to the **Corran Ridge**
at roughly 900 m. Six rivers — the **Vallmara**, **Kestrel Brook**, **Fyr**,
**Corran Water**, **Tarn Beck** and **Halvorn Rill** — drain the uplands and
regularly spill onto the plain.

About **600 000 people** live in the basin, most of them in **Vallmara City** and
two secondary centres. The regional government has three questions for you:

1. **Which places flood, and what is at stake there?**
2. **Who cannot reach a hospital, a clinic or a fire station quickly enough?**
3. **Where should the next protected area / solar farm / clinic go?**

The whole course is scaffolding for answering those three questions properly.

### Coordinate reference systems in this dataset (read this twice)

| CRS | Used by | Units | Purpose |
|---|---|---|---|
| **EPSG:32633** — WGS 84 / UTM zone 33N | `vallmara.gpkg`, six of the seven rasters, `schools.shp` | **metres** | The **analysis CRS**. Distances, areas and buffers are only meaningful here. |
| **EPSG:4326** — WGS 84 geographic | `roads.geojson`, `protected_areas.geojson`, the `lon`/`lat` columns in the CSVs | **degrees** | The exchange CRS. Degrees are an *angle*, not a length. |
| **EPSG:3857** — Web Mercator | `lst_summer_100m_3857.tif` | metres (but lying) | The web-map CRS. Areas are inflated by up to 1/cos²(latitude). Never compute statistics in it. |

The mixture is **deliberate**. Real projects always arrive this way.

### Vector layers — `data/vector/vallmara.gpkg` (EPSG:32633)

| Layer | Geometry | Rows | Key attributes |
|---|---|---|---|
| `districts` | Polygon | 24 | `district_id`, `name`, `district_type` (urban_core / suburban / rural / upland_rural), `area_km2`, `dist_core_km`, `mean_elev_m`, `coastal`, `population`, `households`, `pop_density_km2` |
| `census_blocks` | Polygon | 460 | `block_id`, `district_id`, `area_km2`, `population`, `households`, `pop_density_km2` |
| `landuse` | Polygon | 441 | `lu_id`, `class_code` (1–8), `landuse_class`, `area_ha` |
| `rivers` | LineString | 6 | `river_id`, `name`, `strahler_order`, `mean_discharge_m3s`, `perennial`, `length_km` |
| `flood_zones` | Polygon | 181 | `zone_id`, `hazard_class`, `return_period_yr` (100 or 500), `area_km2` |
| `buildings` | Polygon | 5 200 | `building_id`, `use_type`, `floors`, `footprint_m2`, `year_built`, `construction`, `has_basement`, `ground_elev_m`, `value_kvs` |
| `facilities` | Point | 83 | `facility_id`, `facility_type`, `capacity`, `staff`, `opening_year`, `is_24h` |
| `transit_routes` | LineString | 6 | `route_id`, `headway_min`, `daily_riders`, `length_km` |
| `bus_stops` | Point | 130 | `stop_id`, `route_id`, `shelter`, `boardings_daily` |
| `sea` / `land_boundary` / `coastline` | Polygon / Polygon / LineString | 1 each | study-area masks |

*Money is in **VS**, the fictional Vallmara Shilling; `value_kvs` = thousands of VS.*

### Raster layers — `data/raster/`

| File | Resolution | CRS | dtype | NoData | Variable |
|---|---|---|---|---|---|
| `dem_25m.tif` | 25 m | 32633 | float32 | −9999 | Elevation, metres above sea level. Sea = NoData |
| `landcover_25m.tif` | 25 m | 32633 | uint8 | 0 | Class 1–8 (see `landcover_legend.csv`) |
| `rainfall_annual_250m.tif` | 250 m | 32633 | float32 | −9999 | Mean annual rainfall, mm |
| `ndvi_50m.tif` | 50 m | 32633 | float32 | −9999 | NDVI, with two circular cloud gaps |
| `popdens_100m.tif` | 100 m | 32633 | float32 | −9999 | Persons per km² |
| `multispectral_50m.tif` | 50 m | 32633 | uint16 | 0 | 4 bands: Blue, Green, Red, NIR (reflectance × 10 000) |
| `lst_summer_100m_3857.tif` | ~134 m* | **3857** | float32 | −9999 | Summer land-surface temperature, °C |

\* *Generated on a 100 m UTM grid, then reprojected to Web Mercator — which
resamples onto a new grid, so the stored resolution is ~134 units. That is the
Web-Mercator scale factor at 41.7° N, visible in the metadata.*

### Tabular data — `data/tabular/`

| File | Rows | Contents |
|---|---|---|
| `sensor_stations.csv` | 24 | Station metadata **plus `lon`/`lat` and `x_utm`/`y_utm`** |
| `sensor_readings.csv` | 866 | Monthly 2022-01 → 2024-12: `pm25_ugm3`, `rainfall_mm`, `temp_c` |
| `flood_incidents.csv` | 393 | `date`, `lon`, `lat`, `depth_cm`, `damage_kvs`, `cause`, `injuries` |
| `district_socioeconomic.csv` | 27 | `median_income_vs`, `unemployment_rate`, `pct_over65`, `pct_tertiary_edu`, `hospital_beds_per_1000`, `vehicles_per_household` |
| `landcover_legend.csv` | 8 | class code → label |

### The deliberate data-quality problems

These are teaching material, not bugs. You will fix all of them in Module 2.

1. `landuse.landuse_class` — 45 rows with inconsistent case or stray whitespace.
2. `landuse` — 3 self-intersecting "bow-tie" polygons and 2 duplicate rows.
3. `census_blocks` — one row with an **empty** geometry.
4. `districts.population` — 2 missing values.
5. `buildings.year_built` — 240 NaN, 12 rows dated **1066**, 6 dated **2199**.
6. `buildings.value_kvs` — 95 NaN.
7. `facilities.capacity` — 11 rows using the sentinel **−999**; 3 duplicated points.
8. `roads.speed_limit_kmh` — NaN on every residential street.
9. `sensor_readings` — 45 NaN PM2.5, 45 rainfall values of **−999**, 2 duplicate rows.
10. `flood_incidents.csv` — 6 rows with **lon/lat swapped**, 4 at **(0, 0)**,
    3 in another country, 25 NaN depths, 8 negative damage values.
11. `district_socioeconomic.csv` — 2 duplicated `district_id` keys, 1 orphan key
    `D99` matching no polygon, 2 NaN incomes.
12. Three different CRS across the files.

### The ground truth baked into the simulation

Because you know the generating process, you can *grade your own analysis*:

* Rainfall = `470 + 0.62 × elevation + a south→north gradient` (+ noise).
* Land-surface temperature = `31.5 − 0.0062 × elevation + 6.4 × urban intensity`
  — a textbook **urban heat island**.
* Urban intensity `u` is a smooth field peaking at the city core, and
  population density is `9 500·u^2.1 + 22` persons/km², so `u` is recoverable
  from the density raster as `u = ((density − 22)/9 500)^(1/2.1)`.
* PM2.5 = `7.5 + 16·u + 9·exp(−d_motorway / 1 800 m) + seasonal + noise`.
  Because `u` and `d_motorway` are correlated, fitting the decay **without** `u`
  gives a wildly biased e-folding distance — a built-in lesson in confounding.
* Flood incidents were drawn **72% from the 100-year zone, 28% from the 500-year zone**;
  depth increases near rivers and decreases with elevation.
* Population density decays from the city core. **Districts are built by dissolving
  census blocks**, so each district polygon is exactly the union of its blocks and
  its population is exactly their sum — a built-in consistency check (the only
  exception being the two districts whose population was deliberately blanked).
* Building value decays exponentially with distance from the core.
* `multispectral_50m.tif` is built so that `(NIR − RED)/(NIR + RED)` **reproduces**
  `ndvi_50m.tif` to within 0.003 — you can verify your own band maths.
'''),

md(r'''
## 0.5 — Global setup cell

**What we are about to do.** Establish the notebook-wide conventions: imports,
plotting defaults, display options, the analysis CRS, and a couple of small
helper functions used throughout.

**Why it matters.** In spatial work, a *single declared analysis CRS* is a
discipline, not a convenience. Almost every wrong number in applied GIS comes
from silently mixing coordinate systems. We define `CRS_UTM` once and reproject
everything into it at load time.

**Concept — the analysis CRS.** Choose one projected CRS whose units are metres
and whose distortion is small over your study area, do *all* measurement and
modelling in it, and reproject only at the very end for display or delivery.
For the Vallmara Basin that is **EPSG:32633 (UTM zone 33N)**: the region spans
about 0.6° of longitude, comfortably inside the 6°-wide zone, where UTM scale
error stays under 1 part in 2 500 (i.e. < 0.4 m per km).

**Expected outcome.** No visible output beyond a confirmation line — but every
later cell depends on this one.

**What the next cell does:** imports everything, sets pandas/matplotlib
defaults, defines `CRS_UTM` and two helpers (`fresh_ax` for consistently styled
map axes, and `describe_gdf` for a quick spatial summary of any layer).
'''),

code(r'''
# ---------------------------------------------------------------- imports ---
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import rasterio
from rasterio.plot import show as rshow
import shapely
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box

# ------------------------------------------------------------ conventions ---
CRS_UTM     = "EPSG:32633"   # WGS 84 / UTM 33N  - metres - THE ANALYSIS CRS
CRS_WGS84   = "EPSG:4326"    # WGS 84 geographic - degrees - storage/exchange
CRS_WEBMERC = "EPSG:3857"    # Web Mercator      - metres  - web maps only

# ------------------------------------------------------------- display -----
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 130)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

sns.set_theme(style="whitegrid", context="notebook")
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "figure.facecolor": "white",
    "axes.grid": False,          # grids are noise on maps
    "font.size": 9,
})
warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")

# ------------------------------------------------------------- helpers -----
def fresh_ax(figsize=(9, 7), title=None):
    """A map-styled matplotlib axes: equal aspect, no ticks, optional title."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, weight="bold", loc="left")
    return fig, ax


def describe_gdf(gdf, name="layer"):
    """One-line spatial summary of any GeoDataFrame - use it constantly."""
    xmin, ymin, xmax, ymax = gdf.total_bounds
    geoms = gdf.geom_type.value_counts().to_dict()
    print(f"{name:<16} rows={len(gdf):>6}  cols={gdf.shape[1]:>3}  "
          f"crs={str(gdf.crs.to_string() if gdf.crs else 'NONE'):<12} "
          f"geom={geoms}")
    print(f"{'':<16} bounds=({xmin:,.0f}, {ymin:,.0f}) -> ({xmax:,.0f}, {ymax:,.0f})  "
          f"empty={int(gdf.geometry.is_empty.sum())}  "
          f"invalid={int((~gdf.geometry.is_valid).sum())}  "
          f"null={int(gdf.geometry.isna().sum())}")


print("Setup complete. Analysis CRS =", CRS_UTM)
'''),

md(r'''
**Explanation.**

* **`CRS_UTM` as a named constant.** Every reprojection in the notebook refers to
  this name. If you later port the analysis to another region, you change one line.
* `pd.set_option("display.float_format", ...)` — thousands separators and three
  decimals. Spatial tables are full of coordinates in the millions and areas in
  the millionths; unformatted output is unreadable.
* `sns.set_theme(...)` then `axes.grid: False` — Seaborn's grid is excellent for
  statistical plots and terrible for maps, so we turn it off globally and let
  Seaborn's own plotting functions re-enable it per-figure.
* `fresh_ax()` — `set_aspect("equal")` is **not cosmetic**. In a projected CRS,
  one unit of x must be drawn the same size as one unit of y, otherwise the map is
  sheared and every visual judgement you make about shape and distance is wrong.
* `describe_gdf()` — prints the five things that actually matter about a layer:
  row count, CRS, geometry types, bounding box, and the count of
  empty/invalid/null geometries. Get in the habit of running it on every layer
  you load; it catches missing CRS and broken geometry before they poison an
  analysis.

**Expected output.** A single line: `Setup complete. Analysis CRS = EPSG:32633`.
'''),

]
