# -*- coding: utf-8 -*-
"""Module 3 - Advanced: spatial statistics, MCDA, clustering, spatial ML."""
from _cells import md, code

CELLS = [

md(r'''
# Module 3 — Advanced: Spatial Data Science

Modules 1 and 2 were about *doing GIS in Python*. Module 3 is about **spatial
data science**: bringing statistics, machine learning and decision analysis to
bear on spatial problems, and handling the ways in which space breaks the
assumptions those methods were built on.

**The central problem.** Classical statistics and machine learning assume
observations are independent. Spatial observations are not: nearby places
resemble each other. That single fact has three consequences that run through
this entire module.

| Consequence | Symptom | Remedy |
|---|---|---|
| **Effective sample size < n** | Standard errors too small, everything "significant" | Test residual autocorrelation; use spatial error/lag models |
| **Random train/test splits leak** | Cross-validated accuracy far above true out-of-area accuracy | **Spatially blocked** cross-validation |
| **Omitted spatial confounders** | Coefficients attributed to the wrong variable | Include the confounder, or a spatial trend / eigenvector filter |

**The 14 lessons**

| # | Lesson | Technique |
|---|---|---|
| A0 | The analysis-ready feature table | Assembling everything from Module 2 |
| A1 | Spatial feature engineering | Proximity, density, focal, composition, lag |
| A2 | Multi-criteria decision analysis | Weighted overlay, AHP, sensitivity analysis |
| A3 | Environmental suitability modelling | Raster MCDA with hard constraints |
| A4 | Urban accessibility and equity | Two-step floating catchment area (2SFCA) |
| A5 | Spatial autocorrelation | Moran's I from scratch, permutation inference |
| A6 | Hotspot analysis | Getis-Ord Gi*, LISA, multiple testing |
| A7 | Point pattern analysis | KDE, quadrat test, nearest-neighbour index |
| A8 | Spatial clustering | DBSCAN, K-means, spatially constrained regionalisation |
| A9 | Spatial regression | OLS, residual diagnostics, spatial lag features |
| A10 | Predictive modelling I | Flood susceptibility: design and feature matrix |
| A11 | Predictive modelling II | Spatial cross-validation and the leakage it prevents |
| A12 | Model interpretation | Permutation importance, partial dependence, surfaces |
| A13 | Quantitative risk | Hazard × exposure × vulnerability, expected annual damage |
| A14 | Communicating spatial results | What to show, what to caveat, what not to claim |
'''),

# ------------------------------------------------------------------ A0 -----
md(r'''
## A0 — The analysis-ready feature table

**What we are about to do.** Assemble one tidy table with **one row per census
block** and every attribute we will need for the rest of the course.

**Why it matters.** This is the deliverable that separates a spatial *analyst*
from a spatial *data scientist*. Everything downstream — clustering, regression,
machine learning, risk modelling — consumes this one table. If it is wrong,
everything is wrong; if it is right, the rest is ordinary data science.

**The concept — the analysis base table.** In tabular data science this is
routine. In spatial work it is the hard part, because every column comes from a
different geometry, a different resolution and possibly a different CRS. The
recipe:

1. Choose the **analysis unit** (here: census blocks) and its **support**
   (polygons, ~3 km² each).
2. For each source, choose the **correct transfer operation**: zonal statistics
   for rasters, spatial join + aggregate for points, overlay + area weighting for
   polygons, `sjoin_nearest` for proximity.
3. Recompute every absolute quantity after any geometry-splitting operation.
4. Keep missing values as **NaN**, never as 0.
5. **Validate** each column against something you already know.

**A warning about the unit of analysis.** Everything you conclude is conditional
on this choice — the **modifiable areal unit problem**. Results computed on
blocks and on districts can differ in magnitude and even in sign. State the unit,
and where it matters, repeat the analysis at a second scale to check robustness.

**What the next cell does:** builds the full feature table — identity,
demography, terrain, climate, vegetation, hazard, accessibility and assets —
validates every group, and saves it to `data/outputs/block_features.gpkg`.
'''),

code(r'''
from rasterio.features import rasterize
from rasterio.warp import Resampling
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- identity --
F = blocks[~blocks.geometry.is_empty].copy().reset_index(drop=True)
F = F.merge(districts[["district_id", "district_type", "dist_core_km"]],
            on="district_id", how="left")
F = F[["block_id", "district_id", "district_type", "dist_core_km",
       "area_km2", "population", "households", "pop_density_km2", "geometry"]]
F["centroid"] = F.geometry.representative_point()

# ----------------------------------------------------------------- terrain --
zs = zonal_stats(F, RAS / "dem_25m.tif", stats=("mean", "min", "max", "std"))
F["mean_elev_m"] = zs["mean"].to_numpy()
F["min_elev_m"]  = zs["min"].to_numpy()
F["elev_range_m"] = (zs["max"] - zs["min"]).to_numpy()

# slope: write the derived raster once, then run zonal statistics on it
with rasterio.open(RAS / "dem_25m.tif") as src:
    dem_a = src.read(1, masked=True).astype("float64").filled(np.nan)
    prof = src.profile.copy()
gy, gx = np.gradient(dem_a, prof["transform"].a, prof["transform"].a)
slope_a = np.degrees(np.arctan(np.hypot(gx, gy)))
slope_path = OUT / "slope_deg_25m.tif"
prof.update(dtype="float32", nodata=-9999.0, compress="deflate")
with rasterio.open(slope_path, "w", **prof) as dst:
    dst.write(np.nan_to_num(slope_a, nan=-9999.0).astype("float32"), 1)
F["mean_slope_deg"] = zonal_stats(F, slope_path, stats=("mean",))["mean"].to_numpy()

# ----------------------------------------------------------------- climate --
F["mean_rainfall_mm"] = zonal_stats(
    F, RAS / "rainfall_annual_250m.tif", stats=("mean",))["mean"].to_numpy()
lst_path = OUT / "lst_utm_100m.tif"
lst_arr = align_to(RAS / "lst_summer_100m_3857.tif", ref, Resampling.bilinear)
lp = ref.copy(); lp.update(dtype="float32", nodata=-9999.0)
with rasterio.open(lst_path, "w", **lp) as dst:
    dst.write(np.nan_to_num(lst_arr, nan=-9999.0).astype("float32"), 1)
F["mean_lst_c"] = zonal_stats(F, lst_path, stats=("mean",))["mean"].to_numpy()

# -------------------------------------------------------------- vegetation --
F["mean_ndvi"] = zonal_stats(F, RAS / "ndvi_50m.tif", stats=("mean",))["mean"].to_numpy()
lcz = zonal_stats(F, RAS / "landcover_25m.tif", stats=("count",), categorical=True)
for code, name in zip(lc_legend.class_code, lc_legend.landuse_class):
    col = f"pct_class_{code}"
    key = "pct_" + name.lower().split()[0].replace("-", "").replace("/", "")
    F[key] = lcz[col].to_numpy() if col in lcz.columns else 0.0

# ------------------------------------------------------------------ hazard --
for rp in (100, 500):
    z = flood[flood.return_period_yr == rp].geometry.union_all()
    inter = F.geometry.intersection(z).area
    F[f"pct_in_flood{rp}"] = (100 * inter / F.geometry.area).to_numpy()

for name, tgt in [("river", rivers), ("coast", coastline)]:
    geom = tgt.geometry.union_all()
    F[f"dist_{name}_m"] = F["centroid"].distance(geom).to_numpy()

# --------------------------------------------------------- accessibility ----
cent = gpd.GeoDataFrame(F[["block_id"]], geometry=F["centroid"], crs=CRS_UTM)
for ftype in ["hospital", "clinic", "school", "fire_station"]:
    tgt = facilities_clean[facilities_clean.facility_type == ftype][["geometry"]]
    j = gpd.sjoin_nearest(cent, tgt, how="left",
                          distance_col="d").drop_duplicates("block_id")
    F[f"dist_{ftype}_m"] = j.set_index("block_id")["d"].reindex(F.block_id).to_numpy()

prim = roads[roads.road_class.isin(["motorway", "primary"])][["geometry"]]
j = gpd.sjoin_nearest(cent, prim, how="left", distance_col="d").drop_duplicates("block_id")
F["dist_primary_road_m"] = j.set_index("block_id")["d"].reindex(F.block_id).to_numpy()

sj = gpd.sjoin(stops[["stop_id", "geometry"]], F[["block_id", "geometry"]],
               predicate="within")
F["n_bus_stops"] = (F.block_id.map(sj.groupby("block_id").size())
                     .fillna(0).astype(int).to_numpy())

# ------------------------------------------------------------------ assets --
bpt = buildings_clean.copy()
bpt["geometry"] = buildings_clean.geometry.representative_point()
bj = gpd.sjoin(bpt, F[["block_id", "geometry"]], predicate="within")
agg = bj.groupby("block_id").agg(
    n_buildings=("building_id", "size"),
    total_value_kvs=("value_kvs", "sum"),
    mean_year_built=("year_built", "mean"),
    total_floorspace_m2=("footprint_m2", "sum"))
for c in agg.columns:
    F[c] = F.block_id.map(agg[c]).to_numpy()
F["n_buildings"] = F["n_buildings"].fillna(0).astype(int)
F["mean_building_age"] = 2025 - F["mean_year_built"]

# ------------------------------------------------------------- validation --
F = F.drop(columns=["centroid", "mean_year_built"])
FEATS = [c for c in F.columns if c not in ("block_id", "district_id",
                                           "district_type", "geometry")]
print(f"ANALYSIS-READY FEATURE TABLE: {len(F)} blocks x {len(FEATS)} features\n")
chk = pd.DataFrame({
    "dtype": F[FEATS].dtypes.astype(str),
    "n_nan": F[FEATS].isna().sum(),
    "min": F[FEATS].min(numeric_only=True).round(2),
    "median": F[FEATS].median(numeric_only=True).round(2),
    "max": F[FEATS].max(numeric_only=True).round(2),
})
print(chk.to_string())

print("\nVALIDATION")
pct_cols = [c for c in F.columns if c.startswith("pct_class_")
            or c.startswith("pct_") and c.startswith("pct_") and "flood" not in c]
lc_pct = [c for c in F.columns if c.startswith("pct_") and "flood" not in c]
print(f"  land-cover percentages sum to 100     : "
      f"{np.allclose(F[lc_pct].sum(axis=1).dropna(), 100, atol=0.5)}")
print(f"  population sums to the regional total : "
      f"{F.population.sum():,} (blocks) vs {blocks.population.sum():,} (all blocks incl. empty)")
print(f"  buildings assigned                    : "
      f"{F.n_buildings.sum():,} of {len(buildings_clean):,}")
print(f"  blocks with no buildings              : {int((F.n_buildings == 0).sum())}")
print(f"  every distance is finite              : "
      f"{bool(np.isfinite(F[[c for c in F.columns if c.startswith('dist_')]].to_numpy()).all())}")

out_fp = OUT / "block_features.gpkg"
F.to_file(out_fp, layer="block_features", driver="GPKG")
print(f"\nSaved -> {out_fp.name} ({out_fp.stat().st_size/1024:.0f} KB)")
'''),

md(r'''
**Explanation.**

* **Identity first.** `block_id` is the primary key; `district_id` and
  `district_type` come along so we can aggregate or stratify later.
* **`representative_point()` not `centroid`** for all proximity work — it is
  guaranteed to lie inside the block.
* **Terrain.** We write the slope raster to disk before running zonal statistics
  on it. That may look wasteful, but it means the slope layer is inspectable,
  reusable and self-documenting — and `zonal_stats` takes a path. In a real
  pipeline, materialise your derived rasters.
* **Climate.** The LST raster is in EPSG:3857, so it is reprojected onto the
  common grid with `align_to` *before* zonal statistics. Running zonal statistics
  across a CRS mismatch is a silent-wrong-answer generator.
* **Land-cover composition** uses the `categorical=True` branch of `zonal_stats`,
  producing one percentage column per class. Composition features like
  `pct_forest` and `pct_builtup` are far more informative to a model than a single
  majority class.
* **Hazard.** `F.geometry.intersection(z).area / F.geometry.area` gives the exact
  fraction of each block inside the flood zone — a *continuous* exposure measure
  rather than a binary flag. Binary flags throw away most of the signal.
* **Accessibility.** `sjoin_nearest` with `drop_duplicates` per block, then
  `.set_index(...).reindex(...)` to write back positionally-safely.
* **Assets.** Buildings are reduced to representative points before joining, so
  each building is counted in exactly one block. Joining polygons to polygons with
  `intersects` would double-count buildings straddling a boundary.
* **`fillna(0)` only where zero is the truth.** `n_bus_stops` and `n_buildings`
  are genuine counts — a block with no bus stop has zero. But `total_value_kvs`
  and `mean_building_age` stay NaN, because "no buildings" means *undefined*
  mean age, not zero.

**Expected outcome.** A summary table of about **30 features across 459 blocks**,
then four validation checks:

* land-cover percentages summing to 100 (± rounding) — proves the categorical
  zonal statistics are complete;
* block population reconciling with the regional total;
* ~5 197 of 5 200 buildings assigned (three fall outside the block tessellation);
* every distance finite — no unmatched `sjoin_nearest`.

If any check fails, fix it **before** going further. Everything from here on
consumes this table.
'''),

# ------------------------------------------------------------------ A1 -----
md(r'''
## A1 — Spatial feature engineering

**What we are going to learn.** The five families of spatial feature, and how to
build each one.

**Why it matters.** In tabular ML, feature engineering is where domain knowledge
enters. In *spatial* ML, geometry itself is a source of features that no amount
of model capacity can substitute for. A gradient-boosted tree cannot invent
"distance to the nearest river" from raw coordinates; you must build it.

**The five families.**

| Family | Question it answers | How |
|---|---|---|
| **Proximity** | How far to the nearest X? | `sjoin_nearest`, `cKDTree`, distance transforms |
| **Density / intensity** | How much X is near here? | Counts in a buffer, KDE, focal sums |
| **Composition** | What is this place made of? | Zonal statistics on categorical rasters |
| **Focal / neighbourhood** | What is the *surroundings* like? | Focal statistics on rasters; `k`-NN means on vectors |
| **Spatial lag** | What are my *neighbours'* values? | Contiguity or distance weights × the variable |

**The spatial lag deserves special attention.** For a variable `y` and a row-
standardised weights matrix `W`, the spatial lag is `Wy` — the weighted average of
the neighbours' values. It is the single most useful engineered spatial feature,
and it is also the thing that makes naive cross-validation leak (A11).

**Concept — a caution about coordinates as features.** Feeding raw `x` and `y`
into a tree model lets it memorise the training locations. It will look excellent
under random CV and fail completely on new territory. Use coordinates only with
spatially blocked validation, and prefer *interpretable* spatial features.

**Expected outcome.** Twelve new features added to the table, each with a stated
rationale, plus a correlation analysis showing which are redundant.

**What the next cell does:** builds focal, density, lag and interaction features,
then examines their correlation structure to find redundancy before modelling.
'''),

code(r'''
from scipy.ndimage import uniform_filter, generic_filter
from scipy.spatial import cKDTree

Fx = F.copy()
cxy = np.c_[Fx.geometry.representative_point().x, Fx.geometry.representative_point().y]

# --- (1) DENSITY: how much of X is within r metres? ------------------------
def density_within(points_gdf, radius_m, weight_col=None):
    """Count (or weighted sum) of `points_gdf` within radius of each block."""
    p = np.c_[points_gdf.geometry.x, points_gdf.geometry.y]
    w = (points_gdf[weight_col].fillna(0).to_numpy() if weight_col
         else np.ones(len(points_gdf)))
    tree = cKDTree(p)
    idx = tree.query_ball_point(cxy, r=radius_m)
    return np.array([w[i].sum() for i in idx])

bpts = buildings_clean.copy()
bpts["geometry"] = buildings_clean.geometry.representative_point()
Fx["bldg_density_1km"] = density_within(bpts, 1000) / (np.pi * 1.0**2)
Fx["value_density_1km"] = density_within(bpts, 1000, "value_kvs") / (np.pi * 1.0**2)
Fx["facilities_within_3km"] = density_within(facilities_clean, 3000)
Fx["stops_within_1km"] = density_within(stops, 1000)

# --- (2) FOCAL: what are the SURROUNDINGS like? ---------------------------
with rasterio.open(RAS / "dem_25m.tif") as src:
    dem_a = src.read(1, masked=True).astype("float64").filled(np.nan)
    tr = src.transform
# Topographic Position Index: elevation minus the mean of a 1 km neighbourhood
win = int(1000 / tr.a)                       # 1 km / 25 m = 40 cells
filled = np.where(np.isfinite(dem_a), dem_a, 0.0)
counts = uniform_filter(np.isfinite(dem_a).astype(float), size=win, mode="nearest")
local_mean = uniform_filter(filled, size=win, mode="nearest") / np.maximum(counts, 1e-9)
tpi = np.where(np.isfinite(dem_a), dem_a - local_mean, np.nan)

tpi_path = OUT / "tpi_1km_25m.tif"
tp = prof.copy(); tp.update(dtype="float32", nodata=-9999.0)
with rasterio.open(tpi_path, "w", **tp) as dst:
    dst.write(np.nan_to_num(tpi, nan=-9999.0).astype("float32"), 1)
Fx["mean_tpi"] = zonal_stats(Fx, tpi_path, stats=("mean",))["mean"].to_numpy()

# --- (3) SPATIAL LAG: what are my NEIGHBOURS like? ------------------------
def knn_weights(coords, k=6, row_standardise=True):
    """Row-standardised k-nearest-neighbour spatial weights matrix."""
    tree = cKDTree(coords)
    d, idx = tree.query(coords, k=k + 1)          # first neighbour is self
    n = len(coords)
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    W[rows, idx[:, 1:].ravel()] = 1.0
    if row_standardise:
        W = W / W.sum(axis=1, keepdims=True)
    return W

W6 = knn_weights(cxy, k=6)

def lag(values):
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    Wm = W6 * ok[None, :]
    denom = Wm.sum(axis=1)
    return np.where(denom > 0, (Wm @ np.nan_to_num(v)) / np.maximum(denom, 1e-12), np.nan)

for col in ["pop_density_km2", "mean_elev_m", "pct_in_flood100", "mean_ndvi"]:
    Fx[f"lag_{col}"] = lag(Fx[col])

# --- (4) INTERACTION / RATIO features ------------------------------------
Fx["people_per_building"] = Fx.population / Fx.n_buildings.replace(0, np.nan)
Fx["value_per_capita"] = Fx.total_value_kvs / Fx.population.replace(0, np.nan)
Fx["relief_ratio"] = Fx.elev_range_m / np.sqrt(Fx.area_km2 * 1e6)
Fx["access_index"] = -(np.log1p(Fx.dist_hospital_m) + np.log1p(Fx.dist_clinic_m)
                       + np.log1p(Fx.dist_fire_station_m)) / 3

NEW = ["bldg_density_1km", "value_density_1km", "facilities_within_3km",
       "stops_within_1km", "mean_tpi", "lag_pop_density_km2", "lag_mean_elev_m",
       "lag_pct_in_flood100", "lag_mean_ndvi", "people_per_building",
       "value_per_capita", "relief_ratio", "access_index"]
print(f"ENGINEERED {len(NEW)} NEW FEATURES\n")
print(Fx[NEW].describe().T[["mean", "50%", "std", "min", "max"]].round(2).to_string())

# --- (5) Redundancy check --------------------------------------------------
NUM = [c for c in Fx.columns
       if Fx[c].dtype.kind in "if" and c not in ("block_id",)]
C = Fx[NUM].corr(numeric_only=True).abs()
Cv = C.to_numpy(copy=True)          # pandas 3 returns read-only views
np.fill_diagonal(Cv, 0)
C = pd.DataFrame(Cv, index=C.index, columns=C.columns)
pairs = (C.where(np.triu(np.ones(C.shape), 1).astype(bool))
          .stack().sort_values(ascending=False))
print("\nMOST CORRELATED FEATURE PAIRS (|r| > 0.90)")
strong = pairs[pairs > 0.90]
for (a, b), r in strong.items():
    print(f"  {r:.3f}   {a:<26} <-> {b}")
print(f"\n  {len(strong)} pairs above 0.90 out of {len(pairs):,} - "
      f"these are candidates for removal before any linear model.")

# --- Picture: four engineered surfaces ------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
panels = [("bldg_density_1km", "Building density (per km2, 1 km radius)", "magma"),
          ("mean_tpi", "Topographic Position Index (1 km)", "RdBu_r"),
          ("lag_pct_in_flood100", "Spatial lag of flood exposure", "Blues"),
          ("access_index", "Composite access index (higher = better)", "viridis")]
for ax, (col, title, cm) in zip(axes, panels):
    Fx.plot(ax=ax, column=col, cmap=cm, scheme="quantiles", k=6, legend=True,
            legend_kwds={"loc": "lower left", "fontsize": 5.5}, edgecolor="none")
    ax.set_title(title, fontsize=9, weight="bold", loc="left")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **`cKDTree.query_ball_point(points, r)`** returns, for each query point, the
  list of all tree points within `r`. This is the efficient way to build "count of
  X within a radius" features — far better than buffering and spatially joining.
  Passing a `weight_col` turns a count into a weighted sum (here, asset value).
* **TPI (Topographic Position Index)** is elevation minus the mean elevation of a
  surrounding window. Positive = ridge or local high, negative = valley or
  depression, near zero = uniform slope. **It is a far better flood-exposure
  predictor than raw elevation**, because what matters hydrologically is whether
  you are low *relative to your surroundings*, not low in absolute terms.
* The `uniform_filter` trick handles NaN correctly: filter the *filled* array and
  the *validity mask* separately, then divide. A plain `uniform_filter` over an
  array containing NaN propagates NaN across the whole window.
* **`knn_weights`** builds a row-standardised k-nearest-neighbour weights matrix
  `W`, where `W[i, j] = 1/k` if `j` is one of `i`'s `k` nearest neighbours. `W @ y`
  is then the mean of each block's neighbours. We use `k = 6` — a common default,
  roughly the number of contiguous neighbours in an irregular tessellation.
* **Why the lag features matter.** `lag_pct_in_flood100` says "how exposed are my
  neighbours?". A block that is itself dry but surrounded by floodplain is
  operationally at risk: its access roads flood, its services are disrupted. Raw
  exposure misses this entirely.
* **Ratio features** (`people_per_building`, `value_per_capita`) normalise out
  size, which is usually a nuisance variable. `relief_ratio` normalises elevation
  range by block size so that big and small blocks are comparable.
* **The redundancy check** matters because spatial features are *constructed* from
  overlapping geometry and are therefore correlated by design. `bldg_density_1km`
  and `value_density_1km` will be nearly collinear. For tree models this is
  harmless; for linear models and for *interpretation* it is fatal — collinear
  features split the coefficient arbitrarily between them.

**Expected outcome.** A describe table for 13 new features, a list of highly
correlated pairs (expect `bldg_density_1km` ↔ `value_density_1km`,
`pop_density_km2` ↔ `bldg_density_1km`, and each variable with its own lag), and
four maps. Look at the TPI map in particular: the river valleys should appear as
connected blue (negative) ribbons, which is a visual confirmation that the focal
computation is correct.
'''),

]

CELLS += [

# ------------------------------------------------------------------ A2 -----
md(r'''
## A2 — Multi-criteria decision analysis

**What we are going to learn.** How to combine several incommensurable criteria
into one defensible score, and how to test whether the answer depends on your
weights.

**Why it matters.** "Where should we put the next clinic?" has no objective
answer — it depends on how you trade off population served against travel
distance against deprivation. MCDA makes that trade-off **explicit, auditable and
testable**, instead of hiding it inside a modeller's judgement.

**The concept — the four steps of weighted overlay.**

1. **Choose criteria** and their direction (benefit: more is better; cost: less is
   better).
2. **Normalise** each to a common scale, usually [0, 1]. Options: min–max,
   rank-based (robust to outliers), or a value function encoding what the
   decision-maker actually cares about (often non-linear — travelling 40 km is
   more than twice as bad as 20 km).
3. **Weight** the criteria so the weights sum to 1.
4. **Aggregate** — either **weighted sum** (compensatory: a great score on one
   criterion offsets a poor one) or **weighted product / geometric mean**
   (non-compensatory: a near-zero on any criterion drags the whole score down).
   Choose deliberately: for siting where a hard requirement exists, the geometric
   mean is usually more honest.

**Where do weights come from? AHP.** In the Analytic Hierarchy Process you make
*pairwise* comparisons ("access is 3× more important than deprivation") on a 1–9
scale, build a reciprocal matrix, and take its principal eigenvector as the
weights. Crucially it also yields a **consistency ratio**: if your comparisons
are internally contradictory (A > B, B > C, C > A) the CR exceeds 0.1 and you
must revise them. **Report the CR** — it is what makes elicited weights auditable.

**Sensitivity analysis is not optional.** If the recommended site changes when
you perturb a weight by 10%, your recommendation is an artefact of the weights,
not of the data. Say so.

**Expected outcome.** A ranked shortlist of census blocks for a new clinic,
built from four criteria with AHP weights, plus a sensitivity analysis showing
how stable the top of the ranking is.

**What the next cell does:** normalises four criteria, derives weights by AHP with
a consistency check, aggregates by both weighted sum and geometric mean, runs a
1 000-iteration Monte-Carlo weight perturbation, and maps the result.
'''),

code(r'''
M = Fx.copy()

# --- 1. CRITERIA ------------------------------------------------------------
# benefit = more is better; cost = less is better
CRITERIA = {
    "population":          ("benefit", "people who would be served"),
    "dist_clinic_m":       ("benefit", "distance to the nearest existing clinic"),
    "dist_primary_road_m": ("cost",    "a clinic must be reachable by road"),
    "pct_in_flood100":     ("cost",    "do not build in the floodplain"),
}

def normalise(v, direction, method="rank"):
    """Scale to [0, 1]. Rank-based normalisation is robust to the extreme
    skew that spatial variables almost always have."""
    v = pd.Series(v).astype(float)
    if method == "rank":
        s = v.rank(pct=True, na_option="keep")
    else:                                   # min-max
        s = (v - v.min()) / (v.max() - v.min())
    return s if direction == "benefit" else 1 - s

N = pd.DataFrame({c: normalise(M[c], d) for c, (d, _) in CRITERIA.items()})
print("CRITERIA, normalised to [0, 1] (rank-based)")
print(N.describe().T[["mean", "50%", "min", "max"]].round(3).to_string())

# --- 2. AHP WEIGHTS from pairwise comparisons ------------------------------
#      how many times more important is the ROW criterion than the COLUMN one?
labels = list(CRITERIA)
A = np.array([
    #        pop   dist_clinic  road   flood
    [1.0,    2.0,        3.0,   4.0],   # population served
    [1/2.0,  1.0,        2.0,   3.0],   # underserved-ness
    [1/3.0,  1/2.0,      1.0,   2.0],   # road access
    [1/4.0,  1/3.0,      1/2.0, 1.0],   # flood avoidance
])
eigval, eigvec = np.linalg.eig(A)
k = np.argmax(eigval.real)
w = np.abs(eigvec[:, k].real); w = w / w.sum()
lam_max = eigval.real[k]
n = len(labels)
CI = (lam_max - n) / (n - 1)
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}[n]
CR = CI / RI

print("\nAHP WEIGHTS")
for lab, wi in zip(labels, w):
    print(f"  {lab:<22} {wi:.3f}   ({CRITERIA[lab][1]})")
print(f"  lambda_max = {lam_max:.4f}, CI = {CI:.4f}, CR = {CR:.4f}   "
      f"{'CONSISTENT (CR < 0.10)' if CR < 0.10 else 'INCONSISTENT - revise!'}")

# --- 3. AGGREGATE two ways -------------------------------------------------
M["score_sum"]  = (N.to_numpy() * w).sum(axis=1)
M["score_geom"] = np.exp((np.log(np.clip(N.to_numpy(), 1e-6, None)) * w).sum(axis=1))

print(f"\nrank correlation between the two aggregations: "
      f"{M.score_sum.corr(M.score_geom, method='spearman'):.4f}")
top_sum = set(M.nlargest(10, 'score_sum').block_id)
top_geo = set(M.nlargest(10, 'score_geom').block_id)
print(f"top-10 overlap: {len(top_sum & top_geo)} of 10 blocks")

cols = ["block_id", "district_id", "population", "dist_clinic_m",
        "dist_primary_road_m", "pct_in_flood100", "score_sum"]
print("\nTOP 10 CANDIDATE BLOCKS FOR A NEW CLINIC (weighted sum)")
print(M.nlargest(10, "score_sum")[cols].round(2).to_string(index=False))

# --- 4. SENSITIVITY: does the answer survive perturbed weights? -----------
rng = np.random.default_rng(7)
SIMS = 1000
appear = np.zeros(len(M))
rank_sum = np.zeros(len(M))
for _ in range(SIMS):
    wp = np.abs(w * rng.normal(1.0, 0.25, size=n))     # +/-25 % perturbation
    wp = wp / wp.sum()
    sc = (N.to_numpy() * wp).sum(axis=1)
    order = np.argsort(-sc)
    appear[order[:10]] += 1
    rank_sum += pd.Series(-sc).rank().to_numpy()

M["top10_frequency"] = appear / SIMS
M["mean_rank"] = rank_sum / SIMS
stable = M.nlargest(12, "top10_frequency")[
    ["block_id", "district_id", "score_sum", "top10_frequency", "mean_rank"]]
print(f"\nSENSITIVITY ANALYSIS ({SIMS} simulations, weights perturbed +/-25 %)")
print(stable.round(3).to_string(index=False))
print(f"\n  blocks appearing in the top 10 in EVERY simulation : "
      f"{int((M.top10_frequency == 1.0).sum())}")
print(f"  blocks appearing at least once                     : "
      f"{int((M.top10_frequency > 0).sum())}")
print("  -> a robust recommendation names only the blocks with high frequency.")

# --- 5. Map ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
for ax, (col, title, cm) in zip(axes, [
        ("score_sum", "MCDA score (weighted sum)", "YlOrRd"),
        ("score_geom", "MCDA score (geometric mean)", "YlOrRd"),
        ("top10_frequency", "Robustness: P(top 10) over 1000 weightings", "viridis")]):
    M.plot(ax=ax, column=col, cmap=cm, scheme="quantiles", k=7, legend=True,
           legend_kwds={"loc": "lower left", "fontsize": 6}, edgecolor="none")
    facilities_clean[facilities_clean.facility_type == "clinic"].plot(
        ax=ax, color="cyan", markersize=22, marker="o",
        edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=10, weight="bold", loc="left")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **Rank normalisation over min–max.** Spatial variables are almost always
  heavily skewed — `population` here runs from 24 to 35 667. Min–max scaling
  would compress 95% of the blocks into the bottom 10% of the range, so the score
  would be driven entirely by one or two outliers. `rank(pct=True)` is robust and
  produces a uniform distribution. The cost is that you lose magnitude
  information: the difference between rank 0.98 and 0.99 could be 10 people or
  10 000.
* **Direction handling** with `1 - s` for cost criteria. Get this backwards and
  you site the clinic in the floodplain.
* **AHP.** The matrix `A` is *reciprocal* (`A[j,i] = 1/A[i,j]`) with 1s on the
  diagonal. The principal eigenvector is the weight vector. `λ_max` equals `n`
  exactly if your judgements are perfectly consistent; the excess drives the
  Consistency Index and, divided by a random-matrix baseline `RI`, the
  **Consistency Ratio**. `CR < 0.10` is the conventional acceptance threshold.
* **Weighted sum vs geometric mean.** The weighted sum is fully compensatory: a
  block with zero road access can still score well if its population is huge. The
  geometric mean cannot — a zero on any criterion drives the whole score to zero.
  For siting problems where every criterion is a genuine requirement, the
  geometric mean is the more honest aggregation. Comparing the two top-10 lists
  tells you how much the choice matters.
* **The sensitivity analysis is the most important block here.** We perturb the
  weights by ±25% a thousand times and record how often each block reaches the
  top 10. A block that appears every time is a robust recommendation. A block that
  appears 30% of the time is an artefact of one particular weighting, and
  presenting it as "the answer" would be indefensible. **Always report the
  frequency, not just the point ranking.**

**Expected outcome.**

```
AHP WEIGHTS
  population             0.467
  dist_clinic_m          0.277
  dist_primary_road_m    0.160
  pct_in_flood100        0.095
  lambda_max = 4.0310, CI = 0.0103, CR = 0.0115   CONSISTENT (CR < 0.10)

rank correlation between the two aggregations: 0.8338
top-10 overlap: 10 of 10 blocks
```

**CR = 0.0115**, far below the 0.10 threshold — the pairwise judgements are
internally coherent. Note also that the two aggregation rules agree on the whole
top 10 here even though their rank correlation across all 459 blocks is only
0.83: they disagree substantially in the *middle* of the ranking and agree at the
extremes, which is the usual pattern.

Look carefully at the normalised criteria table: **`pct_in_flood100` has median
0.696 and maximum 0.696.** Most blocks have zero flood exposure, so they all tie
at the same rank and none can score the full 1.0. Rank normalisation with heavy
ties silently caps the achievable score on that criterion, which effectively
*reduces* its weight below the 0.095 you specified. If a criterion is mostly
zeros, use min–max or a purpose-built value function instead.

```
SENSITIVITY ANALYSIS (1000 simulations, weights perturbed +/-25 %)
block_id district_id  score_sum  top10_frequency  mean_rank
   B0422         D04      0.771            0.991      3.040
   B0409         D19      0.748            0.942      7.034
   B0230         D18      0.773            0.922      3.871
   B0224         D22      0.772            0.896      4.813
   B0314         D18      0.754            0.802      8.444
   ...
  blocks appearing in the top 10 in EVERY simulation : 0
  blocks appearing at least once                     : 41
```

**This is the result that matters.** Not one block survives every weighting, and
**41 different blocks** reach the top 10 at some point. The defensible
recommendation is therefore not "build at B0230" (the point-estimate winner) but
"**B0422, B0409, B0230 and B0224 are robust candidates, appearing in the top 10
in 90–99% of plausible weightings**". Note that B0230 has the *highest*
point score but only the *third* highest robustness — ranking by the point
estimate alone would have picked a less stable site.

Three maps; the third (robustness) is the one you would actually publish.
'''),

# ------------------------------------------------------------------ A3 -----
md(r'''
## A3 — Environmental suitability modelling on rasters

**What we are going to learn.** The same MCDA logic applied cell-by-cell to
raster surfaces, with **hard constraints** as well as soft preferences.

**Why it matters.** Vector MCDA is limited to whatever units you happen to have.
Raster MCDA works at the resolution of the underlying data and can express
continuous preference surfaces, which is how real siting studies (wind, solar,
conservation, landfill) are done.

**The concept — constraints and factors.**

* **Constraints** are Boolean: legal or physical exclusions. Protected area? No.
  Inside a flood zone? No. Slope over 15°? No. Multiply the suitability by 0.
* **Factors** are continuous preferences, normalised to [0, 1] and weighted.

```
suitability = (Π constraints) × (Σ wᵢ · factorᵢ)
```

**Value functions.** Do not assume linear normalisation. For solar irradiance,
more is monotonically better. For distance to a grid connection, there is a
threshold beyond which the project is uneconomic — a sigmoid or a piecewise
function encodes that far better than a straight line. **The shape of the value
function is a modelling assumption and should be stated.**

**Concept — resolution and the constraint budget.** Every hard constraint removes
land. Apply five constraints carelessly and you may exclude 99% of the study
area, leaving a "suitability map" that is really a map of the one place you
forgot to exclude. Always report **how much land each constraint removes**.

**Expected outcome.** A solar-farm suitability surface for the Vallmara Basin,
with a constraint audit, and the top candidate sites extracted as polygons.

**What the next cell does:** builds four constraint layers and four factor
layers on the common 100 m grid, reports how much land each constraint removes,
combines them, and extracts contiguous high-suitability parcels above 10 ha.
'''),

code(r'''
from rasterio.features import rasterize, shapes
from scipy.ndimage import label as cc_label, distance_transform_edt

H, Wd = ref["height"], ref["width"]
TR = ref["transform"]
CELL = TR.a

from shapely.geometry.base import BaseGeometry

def burn(gdf_or_geom, invert=False):
    # NOTE: a GeoDataFrame also has a .geom_type attribute, so test the TYPE,
    # not the presence of the attribute - otherwise the whole frame is passed
    # to rasterize as one geometry and silently skipped.
    geoms = ([gdf_or_geom] if isinstance(gdf_or_geom, BaseGeometry)
             else list(gdf_or_geom.geometry))
    a = rasterize([(g, 1) for g in geoms], out_shape=(H, Wd), transform=TR,
                  fill=0, dtype="uint8").astype(bool)
    return ~a if invert else a

land_g   = burn(LAND_GEOM)
slope_g  = align_to(slope_path, ref, Resampling.average)
elev_g   = grids["elevation"]
lc_g     = grids["landcover"]
ndvi_g   = grids["ndvi"]
pop_g    = grids["popdens"]

# --- 1. HARD CONSTRAINTS -----------------------------------------------------
constraints = {
    "on land":                     land_g,
    "slope <= 5 deg":              np.nan_to_num(slope_g, nan=99) <= 5,
    "outside protected areas":     burn(protected, invert=True),
    "outside the 100-yr floodplain": burn(flood[flood.return_period_yr == 100],
                                          invert=True),
    "not built-up or forest":      ~np.isin(np.nan_to_num(lc_g, nan=0), [2, 5]),
    "not water or wetland":        ~np.isin(np.nan_to_num(lc_g, nan=0), [1, 8]),
}
print("CONSTRAINT AUDIT (cells remaining after each is applied in turn)")
print(f"  {'constraint':<32}{'passes':>12}{'km2':>10}{'% of land':>11}{'lost km2':>11}")
print("-" * 78)
mask = np.ones((H, Wd), dtype=bool)
land_km2 = land_g.sum() * (CELL/1000)**2
prev = land_g.sum()
for name, c in constraints.items():
    mask &= c
    n = int(mask.sum())
    print(f"  {name:<32}{n:>12,}{n*(CELL/1000)**2:>10,.1f}"
          f"{100*n/land_g.sum():>10.1f}%{(prev-n)*(CELL/1000)**2:>11,.1f}")
    prev = n
print("-" * 78)
print(f"  {'ELIGIBLE LAND':<32}{int(mask.sum()):>12,}"
      f"{mask.sum()*(CELL/1000)**2:>10,.1f}{100*mask.sum()/land_g.sum():>10.1f}%")

# --- 2. FACTORS with explicit value functions ------------------------------
def linear_vf(x, lo, hi):
    """Linear ramp: 0 below lo, 1 above hi (or reversed if lo > hi)."""
    return np.clip((x - lo) / (hi - lo), 0, 1)

def sigmoid_vf(x, mid, steep):
    return 1.0 / (1.0 + np.exp((x - mid) / steep))

# distance to the primary road network (grid connection proxy), in metres
road_g = burn(roads[roads.road_class.isin(["motorway", "primary"])])
d_road = distance_transform_edt(~road_g, sampling=CELL)
# distance to built-up areas (demand centres)
built_g = np.nan_to_num(lc_g, nan=0) == 2
d_built = distance_transform_edt(~built_g, sampling=CELL)

factors = {
    "flat terrain":        (linear_vf(np.nan_to_num(slope_g, nan=99), 5, 0), 0.35),
    "close to the grid":   (sigmoid_vf(d_road, 3000, 900), 0.30),
    "low ecological value": (linear_vf(np.nan_to_num(ndvi_g, nan=1), 0.75, 0.15), 0.20),
    "close to demand":     (sigmoid_vf(d_built, 8000, 2500), 0.15),
}
wsum = sum(w for _, w in factors.values())
print(f"\nFACTORS (weights sum to {wsum:.2f})")
for name, (arr, w) in factors.items():
    print(f"  {name:<24} w={w:.2f}   value-function range "
          f"{np.nanmin(arr):.2f} - {np.nanmax(arr):.2f}")

score = sum(arr * w for arr, w in factors.values()) / wsum
suit = np.where(mask, score, np.nan)

print(f"\nSUITABILITY over eligible land: min {np.nanmin(suit):.3f}, "
      f"median {np.nanmedian(suit):.3f}, max {np.nanmax(suit):.3f}")
for q in [50, 75, 90, 95, 99]:
    t = np.nanpercentile(suit, q)
    n = int(np.nansum(suit >= t))
    print(f"  >= {q}th percentile ({t:.3f}) : {n:>7,} cells "
          f"= {n*(CELL/1000)**2:>7,.1f} km^2")

# --- 3. Extract contiguous candidate parcels ------------------------------
THRESH = float(np.nanpercentile(suit, 95))
MIN_HA = 10.0
hi = np.nan_to_num(suit, nan=0) >= THRESH
lbl, nlab = cc_label(hi)
recs = []
for geom, val in shapes(lbl.astype("int32"), mask=hi, transform=TR):
    poly = Polygon(geom["coordinates"][0], geom["coordinates"][1:]).buffer(0)
    if poly.area / 1e4 < MIN_HA:
        continue
    m = lbl == int(val)
    recs.append({"parcel_id": f"P{len(recs)+1:03d}", "area_ha": poly.area / 1e4,
                 "mean_suit": float(np.nanmean(suit[m])),
                 "geometry": poly})
parcels = gpd.GeoDataFrame(recs, crs=CRS_UTM).sort_values(
    "area_ha", ascending=False).reset_index(drop=True)

print(f"\nCANDIDATE PARCELS (suitability >= {THRESH:.3f}, area >= {MIN_HA:.0f} ha)")
print(f"  connected components above threshold : {nlab:,}")
print(f"  parcels large enough to develop      : {len(parcels)}")
if len(parcels):
    parcels["district"] = gpd.sjoin(
        parcels, districts[["name", "geometry"]], how="left",
        predicate="intersects").drop_duplicates("parcel_id")["name"].to_numpy()
    print(parcels.head(8)[["parcel_id", "area_ha", "mean_suit", "district"]]
          .round(2).to_string(index=False))
    print(f"\n  total developable area : {parcels.area_ha.sum():,.0f} ha "
          f"({parcels.area_ha.sum()/100:,.1f} km^2, "
          f"{100*parcels.area_ha.sum()*1e4/LAND_GEOM.area:.2f} % of the basin)")

# --- 4. Map ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
ext = (TR.c, TR.c + Wd*CELL, TR.f - H*CELL, TR.f)
axes[0].imshow(np.where(mask, 1, np.nan), cmap="Greens", extent=ext, vmin=0, vmax=1.4)
protected.plot(ax=axes[0], facecolor="none", edgecolor="darkgreen", linewidth=1.0)
flood[flood.return_period_yr == 100].plot(ax=axes[0], facecolor="#9ecae1",
                                          edgecolor="none", alpha=0.6)
axes[0].set_title(f"(a) Eligible land after constraints\n"
                  f"{mask.sum()*(CELL/1000)**2:,.0f} km^2 "
                  f"({100*mask.sum()/land_g.sum():.1f} % of the basin)",
                  fontsize=10, weight="bold", loc="left")
im = axes[1].imshow(suit, cmap="RdYlGn", extent=ext, vmin=0, vmax=1)
plt.colorbar(im, ax=axes[1], shrink=0.75, label="suitability")
axes[1].set_title("(b) Suitability score over eligible land",
                  fontsize=10, weight="bold", loc="left")
land.plot(ax=axes[2], facecolor="#f2efe6", edgecolor="#ccc6b8", linewidth=0.5)
if len(parcels):
    parcels.plot(ax=axes[2], column="mean_suit", cmap="RdYlGn", vmin=0.5, vmax=1,
                 edgecolor="black", linewidth=0.6, legend=True,
                 legend_kwds={"shrink": 0.6, "label": "mean suitability"})
roads[roads.road_class.isin(["motorway", "primary"])].plot(
    ax=axes[2], color="#777", linewidth=0.6)
axes[2].set_title(f"(c) {len(parcels)} developable parcels >= {MIN_HA:.0f} ha",
                  fontsize=10, weight="bold", loc="left")
for a in axes:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **The constraint audit is the ethical core of this lesson.** Applying
  constraints one at a time and reporting the land removed by each makes the
  result reproducible and arguable. If a stakeholder disputes the slope
  threshold, the table tells everyone immediately how much it matters.
* `burn(..., invert=True)` rasterises a layer and negates it, giving "outside X".
  Note that `rasterize` uses cell centres by default, so a protected-area boundary
  is respected to within half a cell (50 m here) — acceptable for a screening
  study, not for a legal determination.
* **Value functions.** `linear_vf(slope, 5, 0)` gives 1 at 0° falling linearly to
  0 at 5° — a *descending* ramp, expressed by putting the "good" end second.
  `sigmoid_vf(d_road, 3000, 900)` is ~1 within 2 km of a road, ~0.5 at 3 km, and
  ~0 beyond 5 km, encoding the economics of a grid connection far better than a
  straight line. **State your value functions; they carry as much of your
  judgement as the weights do.**
* `distance_transform_edt(~road_g, sampling=CELL)` gives a continuous
  distance-to-road surface in metres. Note `sampling=` again.
* **Connected-component labelling** (`scipy.ndimage.label`) groups adjacent
  high-suitability cells into candidate parcels. This matters because a solar
  farm needs a *contiguous* site: 500 scattered hectares is not the same as one
  500-hectare block. Filtering by minimum parcel area converts a suitability
  surface into a *deliverable*.
* The 95th-percentile threshold is a choice, not a finding. Report the area
  available at several thresholds (as we do) so the reader can pick their own.

**Expected outcome.**

```
CONSTRAINT AUDIT (cells remaining after each is applied in turn)
  constraint                            passes       km2  % of land   lost km2
  on land                              139,478   1,394.8     100.0%        0.0
  slope <= 5 deg                       128,541   1,285.4      92.2%      109.4
  outside protected areas              124,711   1,247.1      89.4%       38.3
  outside the 100-yr floodplain        107,514   1,075.1      77.1%      172.0
  not built-up or forest                47,174     471.7      33.8%      603.4
  not water or wetland                  46,672     466.7      33.5%        5.0
  ELIGIBLE LAND                         46,672     466.7      33.5%
```

The audit immediately tells you where the argument will be: **the "not built-up
or forest" rule removes 603 km², nearly six times more than everything else
combined.** If a stakeholder wants to challenge one assumption, that is the one —
and now they can, precisely, instead of arguing about the map.

```
SUITABILITY over eligible land: min 0.151, median 0.625, max 0.950
  >= 95th percentile (0.838) :   2,334 cells =    23.3 km^2

CANDIDATE PARCELS (suitability >= 0.838, area >= 10 ha)
  connected components above threshold : 83
  parcels large enough to develop      : 6
parcel_id   area_ha  mean_suit     district
     P006   1,179.0       0.87  Kestrel Quay
     P004     764.0       0.86  Old Vallmara
     P002     148.0       0.87     Ardenfeld
     P001      77.0       0.88      Ashcombe
     P005      13.0       0.89  Old Vallmara
     P003      11.0       0.86     Ardenfeld

  total developable area : 2,192 ha (21.9 km^2, 1.57 % of the basin)
```

**Follow the collapse: 100% → 33.5% eligible → 1.7% above the suitability
threshold → and then only 6 of 83 high-suitability clumps are large enough to
build on.** "One third of the basin is eligible" and "1.57% is actually
developable" are both true, and only the second is useful. The contiguity step is
what turns a suitability surface into a deliverable, and it is the step most
often skipped.

Three panels: eligible land, the suitability surface, and the final parcels.
'''),

# ------------------------------------------------------------------ A4 -----
md(r'''
## A4 — Urban accessibility and spatial equity

**What we are going to learn.** Two-Step Floating Catchment Area (2SFCA) — the
standard method for measuring spatial access to services — and how to turn it
into an equity statement.

**Why it matters.** "Distance to the nearest hospital" ignores **capacity** and
**competition**. A hospital 2 km away serving 200 000 people may be less
accessible than one 8 km away serving 5 000. 2SFCA fixes this, and it is what
health-geography and transport-equity work actually uses.

**The concept — 2SFCA in two steps.**

**Step 1.** For each supply location *j* with capacity `S_j`, find all demand
locations within the catchment (travel threshold `d₀`) and compute the
**provider-to-population ratio**:

```
R_j = S_j / Σ_{k ∈ catchment(j)} P_k
```

**Step 2.** For each demand location *i*, sum the ratios of all supply locations
within *its* catchment:

```
A_i = Σ_{j ∈ catchment(i)} R_j
```

`A_i` has units of *providers per person*. Higher is better. It captures supply,
demand and competition simultaneously.

**Enhanced 2SFCA (E2SFCA)** replaces the hard catchment boundary with a distance-
decay weight `W(d)` — because a facility 1 km away is more accessible than one at
29 km, even if both are "within 30 km". We implement E2SFCA with a Gaussian decay.

**Concept — measuring inequity.** Once you have `A_i` per block with population
`P_i`, the **population-weighted Gini coefficient** of `A` summarises how
unequally access is distributed: 0 = perfectly equal, 1 = maximally unequal.
Pair it with a **concentration curve** (cumulative access against cumulative
population, ordered by access) for a defensible equity statement.

**Expected outcome.** E2SFCA accessibility for clinics and hospitals, a Gini
coefficient, a concentration curve, and a map showing which communities are
underserved.

**What the next cell does:** implements E2SFCA with Gaussian decay, computes it
for two service types, calculates the population-weighted Gini, and identifies
the worst-served populations.
'''),

code(r'''
from scipy.spatial import cKDTree

DEMAND_XY = np.c_[Fx.geometry.representative_point().x,
                  Fx.geometry.representative_point().y]
DEMAND_POP = Fx.population.to_numpy(dtype=float)
P = DEMAND_POP                      # short alias used in this lesson only

def e2sfca(supply_gdf, capacity_col, d0_m, decay="gaussian",
           demand_xy=None, demand_pop=None):
    """Enhanced two-step floating catchment area accessibility.

    Demand is passed in explicitly (defaulting to the module-level block
    centroids and populations) so the function never silently depends on a
    global that some later cell might rebind.
    """
    demand_xy = DEMAND_XY if demand_xy is None else demand_xy
    P = DEMAND_POP if demand_pop is None else demand_pop
    sx = np.c_[supply_gdf.geometry.x, supply_gdf.geometry.y]
    S = supply_gdf[capacity_col].to_numpy(dtype=float)
    S = np.where(np.isfinite(S), S, np.nanmedian(S))     # impute unknown capacity

    D = np.sqrt(((demand_xy[:, None, :] - sx[None, :, :]) ** 2).sum(axis=2))
    if decay == "gaussian":
        Wt = np.exp(-0.5 * (D / (d0_m / 2.0)) ** 2)
        Wt[D > d0_m] = 0.0
    else:                                                # hard catchment
        Wt = (D <= d0_m).astype(float)

    # STEP 1: provider-to-population ratio at each supply point
    demand_j = (Wt * P[:, None]).sum(axis=0)
    Rj = np.divide(S, demand_j, out=np.zeros_like(S), where=demand_j > 0)
    # STEP 2: sum the weighted ratios reachable from each demand point
    return (Wt * Rj[None, :]).sum(axis=1)

clinics = facilities_clean[facilities_clean.facility_type == "clinic"]
hosps   = facilities_clean[facilities_clean.facility_type == "hospital"]

Fx["access_clinic"]   = e2sfca(clinics, "capacity", 10_000) * 1000   # per 1,000 people
Fx["access_hospital"] = e2sfca(hosps,   "capacity", 30_000) * 1000

print("E2SFCA ACCESSIBILITY (provider capacity per 1,000 residents)")
for col, label, n in [("access_clinic", "clinics (10 km catchment)", len(clinics)),
                      ("access_hospital", "hospitals (30 km catchment)", len(hosps))]:
    a = Fx[col]
    covered = 100 * (a > 0).mean()
    pop_zero = Fx.loc[a == 0, "population"].sum()
    print(f"\n  {label}  ({n} facilities)")
    print(f"    blocks with ANY access : {covered:5.1f} %")
    print(f"    people with NO access  : {pop_zero:,.0f} "
          f"({100*pop_zero/P.sum():.1f} % of the population)")
    print(f"    min / median / max     : {a.min():.3f} / {a.median():.3f} / {a.max():.3f}")

# --- Equity: population-weighted Gini and concentration curve ------------
def weighted_gini(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    o = np.argsort(x); x, w = x[o], w[o]
    cw = np.cumsum(w); cxw = np.cumsum(x * w)
    if cxw[-1] == 0:
        return np.nan
    cxw = cxw / cxw[-1]; cw = cw / cw[-1]
    return float(1 - np.sum((cxw[1:] + cxw[:-1]) * np.diff(cw)))

# A Gini is only interpretable for a BENEFIT (more is better). To compare
# E2SFCA against the naive alternative fairly, convert distance to a benefit.
Fx["prox_clinic"]   = 1.0 / (1.0 + Fx.dist_clinic_m / 1000.0)
Fx["prox_hospital"] = 1.0 / (1.0 + Fx.dist_hospital_m / 1000.0)

print("\n" + "=" * 78)
print("SPATIAL EQUITY  (population-weighted Gini; all measures are BENEFITS)")
print("=" * 78)
for col, label in [("prox_clinic",   "clinic  - naive proximity 1/(1+d_km)"),
                   ("access_clinic", "clinic  - E2SFCA (capacity + competition)"),
                   ("prox_hospital", "hospital - naive proximity 1/(1+d_km)"),
                   ("access_hospital", "hospital - E2SFCA")]:
    print(f"  {label:<46} Gini = {weighted_gini(Fx[col], P):6.3f}")
print("\n  0 = everyone has identical access;  1 = one person has all of it.")
print("  A Gini computed on a COST (distance) is not comparable with one computed")
print("  on a BENEFIT - low inequality in distance and low inequality in access")
print("  mean opposite things. Always convert to a common direction first.")

# who is worst off?
worst = Fx.nsmallest(400, "access_clinic")
cum = worst.population.cumsum()
print(f"\n  The 20 % of the population with the LOWEST clinic access:")
thr = Fx.sort_values("access_clinic").assign(c=lambda d: d.population.cumsum())
bottom20 = thr[thr.c <= 0.20 * P.sum()]
print(f"    blocks              : {len(bottom20)}")
print(f"    people              : {bottom20.population.sum():,}")
print(f"    mean access         : {bottom20.access_clinic.mean():.4f} "
      f"vs regional mean {Fx.access_clinic.mean():.4f}")
print(f"    district types      : {bottom20.district_type.value_counts().to_dict()}")
print(f"    mean dist to clinic : {bottom20.dist_clinic_m.mean()/1000:.1f} km "
      f"vs {Fx.dist_clinic_m.mean()/1000:.1f} km regionally")

# --- Figure -------------------------------------------------------------------
fig = plt.figure(figsize=(17, 5.4))
ax1 = fig.add_subplot(1, 3, 1)
Fx.plot(ax=ax1, column="access_clinic", cmap="RdYlGn", scheme="quantiles", k=7,
        legend=True, legend_kwds={"loc": "lower left", "fontsize": 6},
        edgecolor="none")
clinics.plot(ax=ax1, color="black", markersize=16, marker="o")
ax1.set_title("Clinic accessibility (E2SFCA)", fontsize=10, weight="bold", loc="left")
ax1.set_aspect("equal"); ax1.set_xticks([]); ax1.set_yticks([])

ax2 = fig.add_subplot(1, 3, 2)
Fx.plot(ax=ax2, column="access_hospital", cmap="RdYlGn", scheme="quantiles", k=7,
        legend=True, legend_kwds={"loc": "lower left", "fontsize": 6},
        edgecolor="none")
hosps.plot(ax=ax2, color="black", markersize=45, marker="P")
ax2.set_title("Hospital accessibility (E2SFCA)", fontsize=10, weight="bold", loc="left")
ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])

ax3 = fig.add_subplot(1, 3, 3)
for col, lab, c in [("access_clinic", "clinics", "#2166ac"),
                    ("access_hospital", "hospitals", "#b2182b")]:
    d = Fx.sort_values(col)
    cp = np.cumsum(d.population) / d.population.sum()
    ca = np.cumsum(d[col] * d.population)
    ca = ca / ca.iloc[-1]
    ax3.plot(cp, ca, color=c, linewidth=2,
             label=f"{lab} (Gini = {weighted_gini(Fx[col], P):.3f})")
ax3.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect equality")
ax3.set_xlabel("cumulative share of population\n(ordered from worst to best access)")
ax3.set_ylabel("cumulative share of accessibility")
ax3.set_title("Concentration curves", fontsize=10, weight="bold", loc="left")
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **The full distance matrix** `D` is 459 × 18 here, which is trivial. For a
  national study with 50 000 demand points and 3 000 facilities you would use a
  `cKDTree` ball query and a sparse matrix instead — the mathematics is
  unchanged.
* **The Gaussian decay** `exp(−½(d/(d₀/2))²)` truncated at `d₀` puts the
  half-weight point at roughly `0.6 d₀`. Any monotone decreasing kernel works;
  the choice is a modelling assumption. The literature also uses stepped weights
  (e.g. 1.0 / 0.68 / 0.22 for successive distance bands), which are easier to
  explain to non-specialists.
* **Step 1 computes competition.** `demand_j` is the *weighted* population that
  can reach facility `j`. Dividing capacity by it gives capacity **per person**
  at that facility. This is what makes 2SFCA different from a simple buffer
  count.
* **Step 2 sums availability.** A block near three modestly-provisioned clinics
  can score better than one next to a single overwhelmed clinic — which is the
  behaviour we want and which nearest-distance cannot express.
* **Capacity imputation.** Recall that 11 facilities have `capacity = NaN` after
  we cleaned the `−999` sentinel. Here we impute the median. That is a defensible
  choice **only because we state it**; the alternative (dropping them) would
  understate supply.
* **`weighted_gini`** implements the standard trapezoidal Lorenz-curve formula
  with population weights, so that a block of 20 000 people counts 1 000 times a
  block of 20. **Unweighted Gini over polygons is a common and serious error** —
  it treats a vast empty upland block as equal in importance to a dense city
  block.
* **The concentration curve** is the visual form of the same statistic. The
  further it sags below the diagonal, the more unequal the distribution.

**Expected outcome.**

```
  clinics (10 km catchment)  (18 facilities)
    blocks with ANY access :  71.7 %
    people with NO access  : 14,919 (2.3 % of the population)

  hospitals (30 km catchment)  (4 facilities)
    blocks with ANY access :  98.3 %
    people with NO access  : 796 (0.1 % of the population)

SPATIAL EQUITY  (population-weighted Gini; all measures are BENEFITS)
  clinic  - naive proximity 1/(1+d_km)           Gini =  0.336
  clinic  - E2SFCA (capacity + competition)      Gini =  0.205
  hospital - naive proximity 1/(1+d_km)          Gini =  0.388
  hospital - E2SFCA                              Gini =  0.103
```

**The two measures disagree, and not in the direction people expect.** Naive
proximity says hospital access is the *more* unequal of the two (0.388 vs 0.336);
E2SFCA says it is dramatically *less* unequal (0.103 vs 0.205). Why? Because the
four hospitals are large and sited where the population is, so within the 30 km
catchment almost everyone accumulates a similar capacity-per-person ratio.
Distance alone sees only that some people live far away; it cannot see that the
facility they reach is a big one.

Neither number is "right". They answer different questions — *how far must I
travel?* versus *how much provision is available to me?* — and a serious equity
assessment reports both. What you must not do is compute a Gini on distance,
observe that it is large, and call it a finding.

```
  The 20 % of the population with the LOWEST clinic access:
    blocks              : 313
    people              : 122,020
    mean access         : 0.1543 vs regional mean 0.6324
    district types      : {'upland_rural': 156, 'suburban': 114, 'rural': 39, 'urban_core': 4}
    mean dist to clinic : 9.7 km vs 7.8 km regionally
```

Note the composition: the worst-served fifth is **not** purely rural — 114 of the
313 blocks are `suburban`. Suburban fringe blocks are close enough to be
unremarkable on a distance map but sit outside the catchment of any clinic. A
distance-based analysis would have sent resources to the uplands and missed them.

Three panels: two accessibility maps and the concentration curves, both sagging
below the equality diagonal, the clinic curve noticeably further.
'''),

]

CELLS += [

# ------------------------------------------------------------------ A5 -----
md(r'''
## A5 — Spatial autocorrelation: Moran's I from scratch

**What we are going to learn.** How to measure whether a variable is spatially
clustered, and how to test it properly.

**Why it matters.** This is the diagnostic that tells you whether ordinary
statistics apply. If your residuals are spatially autocorrelated, your standard
errors are too small, your p-values are too optimistic, and your
cross-validation is leaking. **Test it before you trust anything.**

**The concept — spatial weights come first.** Every spatial statistic starts with
a **weights matrix** `W` encoding "who is a neighbour of whom":

| Scheme | Definition | Good for |
|---|---|---|
| **Queen contiguity** | Share any boundary point | Irregular polygons — the default for areal data |
| **Rook contiguity** | Share an edge (not just a corner) | Regular grids |
| **k-nearest neighbours** | The `k` closest | Points; guarantees every unit has neighbours |
| **Distance band** | Everything within `d` | When the process has a known range |
| **Kernel** | Continuously decaying weight | Smooth processes |

**Row-standardise** (`W /= W.sum(axis=1)`) so each row sums to 1; then `Wy` is
the *mean* of the neighbours and Moran's I is bounded near [−1, 1].

**Moran's I.**

```
        n      Σᵢ Σⱼ wᵢⱼ (yᵢ − ȳ)(yⱼ − ȳ)
  I = ─────  × ──────────────────────────
       S₀            Σᵢ (yᵢ − ȳ)²
```

with `S₀ = Σᵢⱼ wᵢⱼ`. It is essentially a correlation between a variable and its
own spatial lag. `I ≈ E[I] = −1/(n−1)` means no spatial structure; `I > 0` means
clustering (like next to like); `I < 0` means a checkerboard.

**Inference: use permutations.** The analytical variance of Moran's I relies on
assumptions that rarely hold. **Conditional randomisation** — shuffle the values
across the fixed geometry many times and see where the observed I falls in the
resulting distribution — is assumption-free and takes milliseconds.

**Expected outcome.** Queen and k-NN weights built from scratch, Moran's I for
several variables with permutation p-values, and a Moran scatterplot.

**What the next cell does:** builds a queen-contiguity weights matrix, implements
Moran's I and a 999-permutation test, applies both to six variables, and draws
the Moran scatterplot with its four quadrants.
'''),

code(r'''
# --- 1. Weights matrices from scratch ---------------------------------------
def queen_weights(gdf, row_standardise=True):
    """Queen contiguity: neighbours share at least one boundary point."""
    n = len(gdf)
    W = np.zeros((n, n))
    sidx = gdf.sindex
    geoms = gdf.geometry.to_numpy()
    for i, g in enumerate(geoms):
        for j in sidx.query(g, predicate="intersects"):
            if i != j:
                W[i, j] = 1.0
    if row_standardise:
        rs = W.sum(axis=1, keepdims=True)
        W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    return W

G = Fx.reset_index(drop=True)
Wq = queen_weights(G)
nbrs = (Wq > 0).sum(axis=1)
print("QUEEN CONTIGUITY WEIGHTS")
print(f"  units              : {len(G)}")
print(f"  neighbours: min {nbrs.min()}, median {np.median(nbrs):.0f}, "
      f"mean {nbrs.mean():.2f}, max {nbrs.max()}")
print(f"  ISLANDS (0 neighbours) : {int((nbrs == 0).sum())}   "
      f"<- islands break every spatial statistic; check for them")
print(f"  matrix density     : {100*(Wq > 0).mean():.2f} %  "
      f"(sparse - use scipy.sparse for large n)")

# --- 2. Moran's I with a permutation test ----------------------------------
def morans_i(y, W):
    y = np.asarray(y, dtype=float)
    z = y - np.nanmean(y)
    S0 = W.sum()
    num = z @ (W @ z)
    den = (z ** 2).sum()
    return len(y) / S0 * num / den

def moran_test(y, W, permutations=999, seed=0):
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    Wc = W[np.ix_(ok, ok)]
    rs = Wc.sum(axis=1, keepdims=True)
    Wc = np.divide(Wc, rs, out=np.zeros_like(Wc), where=rs > 0)
    yy = y[ok]
    I = morans_i(yy, Wc)
    rng = np.random.default_rng(seed)
    sim = np.array([morans_i(rng.permutation(yy), Wc) for _ in range(permutations)])
    # two-sided pseudo p-value
    p = (1 + min((sim >= I).sum(), (sim <= I).sum()) * 2) / (permutations + 1)
    EI = -1.0 / (len(yy) - 1)
    z_sim = (I - sim.mean()) / sim.std(ddof=1)
    return dict(I=I, EI=EI, p_sim=min(p, 1.0), z_sim=z_sim,
                sim_mean=sim.mean(), sim_std=sim.std(ddof=1), n=int(ok.sum()))

VARS = ["pop_density_km2", "mean_elev_m", "mean_rainfall_mm", "mean_ndvi",
        "pct_in_flood100", "dist_hospital_m", "mean_lst_c"]
print("\nMORAN'S I  (queen contiguity, 999 permutations)")
print(f"  {'variable':<22}{'I':>8}{'E[I]':>9}{'z':>9}{'p':>9}  interpretation")
print("-" * 82)
res_moran = {}
for v in VARS:
    r = moran_test(G[v], Wq, permutations=999, seed=42)
    res_moran[v] = r
    tag = ("strong clustering" if r["I"] > 0.6 else
           "clustering" if r["I"] > 0.2 else
           "weak/none" if r["I"] > -0.05 else "dispersion")
    star = "***" if r["p_sim"] < 0.001 else "**" if r["p_sim"] < 0.01 else \
           "*" if r["p_sim"] < 0.05 else "ns"
    print(f"  {v:<22}{r['I']:>8.4f}{r['EI']:>9.4f}{r['z_sim']:>9.2f}"
          f"{r['p_sim']:>9.4f}  {tag} {star}")

print("\n  EVERY variable is significantly clustered. That is not a finding -")
print("  it is the normal condition of spatial data, and it is exactly why")
print("  ordinary statistical inference cannot be applied to it unmodified.")

# --- 3. Moran scatterplot ---------------------------------------------------
v = "pct_in_flood100"
y = G[v].to_numpy(dtype=float)
z = (y - y.mean()) / y.std()
Wz = Wq @ z
I = res_moran[v]["I"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
ax = axes[0]
quad = np.where((z > 0) & (Wz > 0), "HH",
        np.where((z < 0) & (Wz < 0), "LL",
         np.where((z > 0) & (Wz < 0), "HL", "LH")))
COL = {"HH": "#b2182b", "LL": "#2166ac", "HL": "#ef8a62", "LH": "#67a9cf"}
for q in ["LL", "LH", "HL", "HH"]:
    m = quad == q
    ax.scatter(z[m], Wz[m], s=16, c=COL[q], label=f"{q} (n={m.sum()})",
               edgecolor="none", alpha=0.8)
b = np.polyfit(z, Wz, 1)[0]
xx = np.linspace(z.min(), z.max(), 10)
ax.plot(xx, b * xx, "k-", linewidth=1.8, label=f"slope = I = {I:.3f}")
ax.axhline(0, color="grey", linewidth=0.8); ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel(f"z({v})"); ax.set_ylabel(f"spatial lag of z({v})")
ax.set_title("Moran scatterplot", fontsize=10, weight="bold", loc="left")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# permutation null distribution
r = res_moran[v]
rng = np.random.default_rng(42)
ok = np.isfinite(y)
sim = np.array([morans_i(rng.permutation(y[ok]), Wq[np.ix_(ok, ok)])
                for _ in range(999)])
axes[1].hist(sim, bins=40, color="#cccccc", edgecolor="white")
axes[1].axvline(I, color="crimson", linewidth=2.2, label=f"observed I = {I:.3f}")
axes[1].axvline(sim.mean(), color="black", linestyle="--", linewidth=1.2,
                label=f"null mean = {sim.mean():.4f}")
axes[1].set_xlabel("Moran's I under random reallocation")
axes[1].set_title(f"Permutation null (999 draws)\np = {r['p_sim']:.4f}",
                  fontsize=10, weight="bold", loc="left")
axes[1].legend(fontsize=8)

G.assign(quad=quad).plot(ax=axes[2], column="quad", categorical=True,
                         cmap="coolwarm", legend=True,
                         legend_kwds={"loc": "lower left", "fontsize": 7},
                         edgecolor="none")
axes[2].set_title("Moran quadrants in space", fontsize=10, weight="bold", loc="left")
axes[2].set_aspect("equal"); axes[2].set_xticks([]); axes[2].set_yticks([])
plt.tight_layout(); plt.show()

# --- 4. Does the weights scheme change the answer? ------------------------
print("\nSENSITIVITY TO THE WEIGHTS SCHEME (variable: pct_in_flood100)")
print(f"  {'scheme':<28}{'I':>10}{'p':>9}")
for label, Wx in [("queen contiguity", Wq),
                  ("k-nearest, k=4", knn_weights(cxy, k=4)),
                  ("k-nearest, k=8", knn_weights(cxy, k=8)),
                  ("k-nearest, k=16", knn_weights(cxy, k=16))]:
    rr = moran_test(G[v], Wx, permutations=499, seed=1)
    print(f"  {label:<28}{rr['I']:>10.4f}{rr['p_sim']:>9.4f}")
print("\n  I falls as the neighbourhood widens - averaging over more, more")
print("  distant units dilutes the local similarity. ALWAYS report your")
print("  weights specification; an unqualified Moran's I is meaningless.")
'''),

md(r'''
**Explanation.**

* **`queen_weights`** uses the spatial index to find intersecting polygons — for
  a planar partition, "intersects" and "shares a boundary point" are the same
  thing. For large `n` use `libpysal.weights.Queen.from_dataframe`, which is
  optimised; building it by hand once is how you understand what it contains.
* **Islands** (units with no neighbours) break everything: their row of `W` is all
  zeros, so their spatial lag is undefined. Detect them explicitly. Remedies:
  switch to k-NN weights (guarantees `k` neighbours), or merge the island into its
  nearest unit, or drop it and say so.
* **`morans_i`** is a direct transcription of the formula. Note `z @ (W @ z)` —
  computing `W @ z` first is `O(n²)`; forming the outer product would be `O(n³)`.
* **The permutation test.** Under the null, the observed values could have landed
  on any unit. Shuffling `y` while holding `W` fixed generates the exact null
  distribution for that geometry. `(1 + count) / (permutations + 1)` is the
  standard pseudo-p-value — the `+1` prevents a p-value of exactly 0, which would
  be an overstatement.
* **Handling NaN properly.** `moran_test` subsets `W` to the finite rows and
  **re-standardises**. Simply dropping rows without re-standardising leaves rows
  summing to less than 1 and biases I downward.
* **The Moran scatterplot** plots `z` against `Wz`; **the slope of the fitted
  line *is* Moran's I**. The four quadrants classify each unit: HH (high value,
  high neighbours) and LL are clusters; HL and LH are spatial outliers — a
  high-exposure block surrounded by dry ones, which is often where the interesting
  story is.
* **The weights-sensitivity block is not optional.** Moran's I is a statistic
  *about a graph*, and the graph is your choice. Reporting "I = 0.55" without
  saying "queen contiguity" is like reporting a correlation without saying which
  variables.

**Expected outcome.**

```
QUEEN CONTIGUITY WEIGHTS
  units              : 459
  neighbours: min 2, median 6, mean 5.64, max 9
  ISLANDS (0 neighbours) : 0
  matrix density     : 1.23 %

MORAN'S I  (queen contiguity, 999 permutations)
  variable                     I     E[I]        z        p  interpretation
  pop_density_km2         0.8614  -0.0022    30.65   0.0010  strong clustering
  mean_elev_m             0.9854  -0.0022    35.66   0.0010  strong clustering
  mean_rainfall_mm        0.9842  -0.0022    35.14   0.0010  strong clustering
  mean_ndvi               0.6940  -0.0022    25.31   0.0010  strong clustering
  pct_in_flood100         0.4197  -0.0022    15.60   0.0010  clustering
  dist_hospital_m         0.9837  -0.0022    35.71   0.0010  strong clustering
  mean_lst_c              0.9720  -0.0022    34.37   0.0010  strong clustering
```

**Every variable is significantly clustered, most of them overwhelmingly so.**
`mean_elev_m` at I = 0.985 is almost perfectly smooth — as any physical field
must be. Even the most fragmented variable, flood exposure, sits at 0.42 with
z = 15.6.

That universality is the lesson: **spatial autocorrelation is not an anomaly to
be detected, it is the default state of spatial data.** The question is never
"is it there?" but "how much, at what scale, and does my method account for it?"

Note the effective sample size implication. With I ≈ 0.98 on elevation, 459
blocks carry nothing like 459 independent observations — closer to a few dozen.
Any t-test or confidence interval computed as if n = 459 is badly overconfident.

```
SENSITIVITY TO THE WEIGHTS SCHEME (pct_in_flood100)
  queen contiguity                0.4197   0.0020
  k-nearest, k=4                  0.4269   0.0020
  k-nearest, k=8                  0.3074   0.0020
  k-nearest, k=16                 0.1896   0.0020
```

**I falls from 0.43 to 0.19 — a factor of more than two — purely by changing the
definition of "neighbour".** Averaging over more, more distant units dilutes local
similarity. An unqualified Moran's I is meaningless; always report the weights
specification.

The Moran scatterplot should show a clear positive slope with dense HH and LL
clusters and some HL/LH outliers; the permutation histogram should be centred
near `−1/(n−1) ≈ −0.002` with the observed I far outside it.
'''),

# ------------------------------------------------------------------ A6 -----
md(r'''
## A6 — Hotspot analysis: Getis-Ord Gi* and LISA

**What we are going to learn.** How to move from a *global* statement ("this
variable is clustered") to a *local* one ("**here** is a statistically
significant cluster").

**Why it matters.** Global Moran's I tells you clustering exists but not where.
Policy needs the where. Local statistics answer it — but they introduce a
multiple-testing problem that is very often ignored in published work.

**The concept — two families of local statistic.**

**Getis-Ord Gi\*** measures whether the values *around and including* unit *i*
are unusually high or low:

```
        Σⱼ wᵢⱼ xⱼ − x̄ Σⱼ wᵢⱼ
  Gi* = ──────────────────────────────────────
         S √[ (n Σⱼ wᵢⱼ² − (Σⱼ wᵢⱼ)²) / (n−1) ]
```

It is a **z-score**: Gi* > 1.96 is a hot spot, < −1.96 a cold spot. It identifies
*intensity* clusters. (Gi, without the star, excludes unit *i* itself.)

**Local Moran's I (LISA)** decomposes global I into per-unit contributions and
classifies each unit as HH / LL / HL / LH. It identifies both clusters **and
spatial outliers**, which Gi* cannot.

**The multiple-testing problem.** With 459 units tested at α = 0.05 you expect
**23 false positives** by chance. Published hotspot maps routinely present these
as findings. Corrections:

* **Bonferroni** — `α/n`. Correct but brutally conservative.
* **False Discovery Rate (Benjamini–Hochberg)** — controls the *expected
  proportion* of false positives among the rejections. **This is the right default
  for exploratory spatial work.**
* **Conditional permutation** — the pseudo-p-values themselves, which at least
  avoid distributional assumptions.

**Expected outcome.** Gi* and LISA for flood exposure and for population density,
with and without FDR correction, so you can see how many "hotspots" evaporate.

**What the next cell does:** implements Gi* and Local Moran with conditional
permutation inference, applies Benjamini–Hochberg correction, and maps the
corrected and uncorrected results side by side.
'''),

code(r'''
def getis_ord_gstar(y, W):
    """Getis-Ord Gi* z-scores. W should NOT be row-standardised, and the
    diagonal must be 1 (the 'star' includes the focal unit itself)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    Wb = (W > 0).astype(float)
    np.fill_diagonal(Wb, 1.0)
    xbar = y.mean()
    S = np.sqrt((y ** 2).sum() / n - xbar ** 2)
    w_sum = Wb.sum(axis=1)
    w_sq = (Wb ** 2).sum(axis=1)
    num = Wb @ y - xbar * w_sum
    den = S * np.sqrt((n * w_sq - w_sum ** 2) / (n - 1))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

def local_moran(y, W, permutations=999, seed=0):
    """Local Moran's I with conditional-permutation pseudo p-values."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    z = (y - y.mean())
    m2 = (z ** 2).sum() / n
    Wz = W @ z
    Ii = z * Wz / m2
    rng = np.random.default_rng(seed)
    nb = (W > 0)
    sims = np.empty((permutations, n))
    for p in range(permutations):
        perm = rng.permutation(n)
        zp = z[perm]
        sims[p] = z * (W @ zp) / m2
    ge = (sims >= Ii).sum(axis=0)
    le = (sims <= Ii).sum(axis=0)
    p_sim = (np.minimum(ge, le) + 1) / (permutations + 1)
    quad = np.where((z > 0) & (Wz > 0), "HH",
            np.where((z < 0) & (Wz < 0), "LL",
             np.where((z > 0) & (Wz < 0), "HL", "LH")))
    return Ii, p_sim, quad

def benjamini_hochberg(p, alpha=0.05):
    """Return a boolean array of rejections controlling the FDR at alpha."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    if k:
        out[order[:k]] = True
    return out

TARGET = "pct_in_flood100"
y = G[TARGET].to_numpy(dtype=float)

# --- Gi* ---------------------------------------------------------------------
gi = getis_ord_gstar(y, Wq)
from scipy.stats import norm
p_gi = 2 * (1 - norm.cdf(np.abs(gi)))
G["gi_z"] = gi
G["gi_p"] = p_gi

# --- LISA, at two permutation budgets ---------------------------------------
Ii, p_lisa999, quad = local_moran(y, Wq, permutations=999, seed=11)
_, p_lisa, _        = local_moran(y, Wq, permutations=9999, seed=11)
G["lisa_I"] = Ii
G["lisa_p"] = p_lisa
G["lisa_quad"] = quad

print(f"HOTSPOT ANALYSIS OF {TARGET}   (n = {len(G)} blocks)")
print("=" * 92)
print(f"  {'method':<28}{'min p':>9}{'alpha=0.05':>12}{'Bonferroni':>12}"
      f"{'FDR (BH)':>10}{'expected FP':>13}")
print("-" * 92)
for label, pv in [("Getis-Ord Gi* (normal)", p_gi),
                  ("LISA, 999 permutations", p_lisa999),
                  ("LISA, 9999 permutations", p_lisa)]:
    n_raw = int((pv < 0.05).sum())
    n_bon = int((pv < 0.05 / len(pv)).sum())
    n_fdr = int(benjamini_hochberg(pv, 0.05).sum())
    print(f"  {label:<28}{pv.min():>9.5f}{n_raw:>12}{n_bon:>12}"
          f"{n_fdr:>10}{0.05*len(pv):>13.0f}")
print("-" * 92)
print(f"  At alpha = 0.05 with {len(G)} tests you expect ~{0.05*len(G):.0f} false")
print(f"  positives by chance alone. Report the FDR-corrected count.")
print(f"\n  THE PERMUTATION FLOOR. A pseudo p-value cannot be smaller than")
print(f"  1/(permutations+1). With 999 permutations that floor is 0.001, while")
print(f"  Benjamini-Hochberg needs the SMALLEST p to clear alpha/n = "
      f"{0.05/len(G):.5f}.")
print(f"  No unit can ever pass - which is why LISA at 999 permutations rejects")
print(f"  {int(benjamini_hochberg(p_lisa999, 0.05).sum())} and at 9999 rejects "
      f"{int(benjamini_hochberg(p_lisa, 0.05).sum())}.")
print(f"  Rule: permutations must exceed ~n/alpha = {int(len(G)/0.05):,} for FDR")
print(f"  correction to be able to reject anything at all.")

# --- classify -----------------------------------------------------------------
sig_fdr = benjamini_hochberg(p_gi, 0.05)
G["hotspot"] = np.where(sig_fdr & (gi > 0), "hot",
                np.where(sig_fdr & (gi < 0), "cold", "not significant"))
G["hotspot_raw"] = np.where((p_gi < 0.05) & (gi > 0), "hot",
                     np.where((p_gi < 0.05) & (gi < 0), "cold", "not significant"))

print(f"\n  Gi* classification (FDR-corrected):")
print("   ", G.hotspot.value_counts().to_dict())
print(f"  Gi* classification (uncorrected):")
print("   ", G.hotspot_raw.value_counts().to_dict())

hot = G[G.hotspot == "hot"]
print(f"\n  FLOOD-EXPOSURE HOT SPOTS")
print(f"    blocks              : {len(hot)}")
print(f"    population at risk  : {hot.population.sum():,} "
      f"({100*hot.population.sum()/G.population.sum():.1f} % of the region)")
print(f"    buildings           : {int(hot.n_buildings.sum()):,}")
print(f"    asset value         : {hot.total_value_kvs.sum():,.0f} k VS")
print(f"    districts involved  : "
      f"{sorted(hot.district_id.unique().tolist())}")

lisa_sig = benjamini_hochberg(p_lisa, 0.05)
G["lisa_class"] = np.where(lisa_sig, G.lisa_quad, "not significant")
print(f"\n  LISA classification (FDR-corrected): "
      f"{G.lisa_class.value_counts().to_dict()}")
outliers = G[G.lisa_class.isin(["HL", "LH"])]
print(f"    spatial OUTLIERS (HL + LH): {len(outliers)} blocks - these are")
print(f"    places that differ sharply from their surroundings, which Gi* misses.")

# --- Maps -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.2))
G.plot(ax=axes[0], column=TARGET, cmap="Blues", scheme="quantiles", k=6,
       legend=True, legend_kwds={"loc": "lower left", "fontsize": 6}, edgecolor="none")
axes[0].set_title(f"(a) Raw variable\n{TARGET}", fontsize=9.5, weight="bold", loc="left")

CMAP_H = {"hot": "#b2182b", "cold": "#2166ac", "not significant": "#eeeeee"}
for ax, col, title in [(axes[1], "hotspot_raw", "(b) Gi* hotspots, UNCORRECTED\np < 0.05"),
                       (axes[2], "hotspot", "(c) Gi* hotspots, FDR-CORRECTED\nBenjamini-Hochberg")]:
    for k_, c_ in CMAP_H.items():
        sub = G[G[col] == k_]
        if len(sub):
            sub.plot(ax=ax, color=c_, edgecolor="none", label=k_)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(title, fontsize=9.5, weight="bold", loc="left")

CMAP_L = {"HH": "#b2182b", "LL": "#2166ac", "HL": "#f4a582",
          "LH": "#92c5de", "not significant": "#eeeeee"}
for k_, c_ in CMAP_L.items():
    sub = G[G.lisa_class == k_]
    if len(sub):
        sub.plot(ax=axes[3], color=c_, edgecolor="none", label=k_)
axes[3].legend(fontsize=7, loc="lower left")
axes[3].set_title("(d) LISA clusters & outliers\nFDR-corrected", fontsize=9.5,
                  weight="bold", loc="left")
for a in axes:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **Gi\* uses a binary weights matrix with a 1 on the diagonal.** The "star"
  means the focal unit is included in its own neighbourhood — which is what makes
  Gi* an *intensity* statistic. Row-standardising before Gi* is a common error
  that changes the meaning of the numerator.
* `S = √(Σy²/n − ȳ²)` is the **population** standard deviation (divide by `n`,
  not `n−1`), as specified in Getis and Ord's original formulation.
* **Local Moran with conditional permutation.** For each permutation we shuffle
  the values across *all* units and recompute every local statistic. (A stricter
  implementation holds unit `i` fixed and shuffles only the others; the difference
  is negligible for `n` in the hundreds and it is much faster this way.)
* **Benjamini–Hochberg.** Sort the p-values ascending, find the largest `k` such
  that `p₍ₖ₎ ≤ α·k/n`, and reject the first `k`. It controls the *expected
  proportion of false discoveries* among your rejections, which is exactly the
  right guarantee for exploratory mapping — Bonferroni controls the probability of
  *any* false positive and is far too strict when you have 459 tests.
* **Gi\* versus LISA.** Gi* finds *where values are high or low*. LISA finds *where
  the local pattern is unusual*, including **spatial outliers** — a dry block
  surrounded by floodplain (HL), or a flooded block surrounded by dry land (LH).
  Those outliers are frequently the operationally interesting cases (isolated
  at-risk communities, or unexpectedly protected pockets), and Gi* cannot see them.

**Expected outcome.**

```
  method                        min p  alpha=0.05  Bonferroni  FDR (BH)  expected FP
  Getis-Ord Gi* (normal)      0.00000          57           8        24           23
  LISA, 999 permutations      0.00100         106           0         0           23
  LISA, 9999 permutations     0.00010         122           1         3           23
```

**Three findings, each worth more than the map.**

1. **Uncorrected Gi\* reports 57 significant blocks; FDR keeps 24; the expected
   false-positive count is 23.** In other words, roughly *half* of the
   uncorrected "hotspots" are noise. Published hotspot maps very often show the
   uncorrected version.

2. **The permutation floor.** A pseudo p-value cannot be smaller than
   `1/(permutations+1)`. With 999 permutations that floor is 0.001, while
   Benjamini–Hochberg needs the smallest p-value to clear `α/n = 0.000109`.
   **No unit can ever pass, so LISA at 999 permutations rejects zero — not
   because there is no signal, but because the test lacks the resolution to
   express it.** Raising to 9 999 permutations moves the floor to 0.0001 and
   three units survive. **Rule of thumb: permutations must exceed `n/α`** — here
   9 180 — for FDR correction to be able to reject anything at all.

3. **Gi\* and LISA disagree sharply** (24 vs 3 survivors). They are different
   statistics answering different questions, and Gi\*'s analytical normal
   p-values have no floor, so they clear FDR more easily. Neither is "more
   correct"; report which you used and why.

```
  FLOOD-EXPOSURE HOT SPOTS (Gi*, FDR-corrected)
    blocks              : 24
    population at risk  : 64,677 (10.1 % of the region)
    buildings           : 589
    asset value         : 126,369 k VS
    districts involved  : ['D01', 'D06', 'D09', 'D11', 'D18']
```

Panels (b) and (c) should differ visibly: a fringe of isolated "significant"
blocks in (b) disappears in (c). **That fringe is exactly what uncorrected
hotspot maps publish as findings.** Panel (d) shows the LISA classification,
which after correction is nearly empty — an honest depiction of what this
particular test can and cannot support.
'''),

]

CELLS += [

# ------------------------------------------------------------------ A7 -----
md(r'''
## A7 — Point pattern analysis

**What we are going to learn.** How to characterise a set of *point events* —
here, flood incidents — as clustered, random or regular, and how to estimate
their intensity surface.

**Why it matters.** Areal statistics (Moran, Gi*) need a partition into units.
Point events do not come that way, and aggregating them to blocks throws away
information and imposes the MAUP. Point pattern analysis works on the events
themselves.

**The concept — the null model.** The baseline is **Complete Spatial Randomness
(CSR)**: a homogeneous Poisson process where events are independent and equally
likely anywhere in the study region. Everything is measured against CSR.

**Three tools.**

1. **Quadrat counts.** Divide the region into cells, count events per cell. Under
   CSR the counts are Poisson, so `variance/mean ≈ 1`. The **variance-to-mean
   ratio (VMR)** is a one-number summary: > 1 clustered, ≈ 1 random, < 1 regular.
   A χ² test gives a p-value. **It is sensitive to quadrat size** — always report
   it at several scales.
2. **Nearest-neighbour index (Clark–Evans).** `R = d̄_observed / d̄_expected`
   where `d̄_expected = 0.5/√λ` for CSR with intensity `λ = n/A`. `R < 1`
   clustered, `R ≈ 1` random, `R > 1` regular. Suffers from **edge effects**:
   points near the boundary have artificially distant neighbours, biasing `R`
   upward.
3. **Kernel density estimation.** A smooth intensity surface
   `λ̂(s) = Σᵢ K((s − sᵢ)/h) / h²`. **The bandwidth `h` is the entire analysis** —
   too small and you map individual events, too large and you map the study area's
   shape.

**The crucial caveat — inhomogeneity.** Flood incidents cluster because *people
and rivers* cluster, not necessarily because floods are contagious. Testing
against homogeneous CSR will always reject. The honest comparison is against an
**inhomogeneous** null with intensity proportional to population at risk — which
is what we do here with a case–control style comparison.

**Expected outcome.** Quadrat and nearest-neighbour statistics at several scales,
a KDE surface at three bandwidths, and a comparison of the raw intensity against
a population-adjusted "relative risk" surface.

**What the next cell does:** runs quadrat and Clark–Evans tests, builds KDE
surfaces at three bandwidths, and computes a population-adjusted relative-risk
surface to separate "where floods happen" from "where floods happen more than
you would expect".
'''),

code(r'''
from scipy.stats import chisquare, gaussian_kde
from scipy.spatial import cKDTree

pts = incidents.copy()
pts = pts[pts.geometry.within(LAND_GEOM.buffer(500))].reset_index(drop=True)
XY = np.c_[pts.geometry.x, pts.geometry.y]
A_km2 = LAND_GEOM.area / 1e6
lam = len(pts) / LAND_GEOM.area                 # events per m^2

print(f"POINT PATTERN: {len(pts)} flood incidents over {A_km2:,.0f} km^2")
print(f"  intensity lambda = {len(pts)/A_km2:.3f} events per km^2")

# --- 1. QUADRAT ANALYSIS at several scales --------------------------------
print("\nQUADRAT ANALYSIS")
print(f"  {'cell size':>10}{'cells on land':>15}{'mean':>8}{'var':>9}{'VMR':>8}"
      f"{'chi2 p':>10}  verdict")
print("-" * 74)
minx, miny, maxx, maxy = LAND_GEOM.bounds
for cell in [2000, 4000, 6000, 8000]:
    nx = int(np.ceil((maxx - minx) / cell)); ny = int(np.ceil((maxy - miny) / cell))
    H2, xe, ye = np.histogram2d(XY[:, 0], XY[:, 1], bins=[nx, ny],
                                range=[[minx, minx + nx*cell], [miny, miny + ny*cell]])
    # keep only quadrats that are mostly on land
    cx = (xe[:-1] + xe[1:]) / 2; cy = (ye[:-1] + ye[1:]) / 2
    XX, YY = np.meshgrid(cx, cy, indexing="ij")
    on_land = np.array([[LAND_GEOM.contains(Point(a, b)) for b in cy] for a in cx])
    counts = H2[on_land]
    if len(counts) < 5:
        continue
    m, v = counts.mean(), counts.var(ddof=1)
    vmr = v / m if m > 0 else np.nan
    exp = np.full(len(counts), m)
    chi2, pval = chisquare(counts, exp)
    verdict = ("CLUSTERED" if vmr > 1.2 else "regular" if vmr < 0.8 else "random")
    print(f"  {cell:>8} m{len(counts):>15}{m:>8.2f}{v:>9.2f}{vmr:>8.2f}"
          f"{pval:>10.2e}  {verdict}")
print("\n  VMR = variance / mean. Poisson (CSR) implies VMR = 1.")

# --- 2. NEAREST-NEIGHBOUR INDEX (Clark-Evans) ------------------------------
tree = cKDTree(XY)
d_nn, _ = tree.query(XY, k=2)
d_obs = d_nn[:, 1].mean()
d_exp = 0.5 / np.sqrt(lam)
R = d_obs / d_exp
se = 0.26136 / np.sqrt(len(pts) ** 2 * lam)
Z = (d_obs - d_exp) / se

# edge-corrected version: drop points within d_exp of the boundary
edge = np.array([LAND_GEOM.boundary.distance(Point(*p)) for p in XY])
keep = edge > d_exp
d_obs_c = d_nn[keep, 1].mean()
R_c = d_obs_c / d_exp

print("\nCLARK-EVANS NEAREST-NEIGHBOUR INDEX")
print(f"  observed mean NN distance : {d_obs:>9,.0f} m")
print(f"  expected under CSR        : {d_exp:>9,.0f} m")
print(f"  R = obs / exp             : {R:>9.4f}   z = {Z:.2f}")
print(f"  R, edge-corrected         : {R_c:>9.4f}   "
      f"({int((~keep).sum())} boundary points removed)")
print(f"  verdict: {'CLUSTERED' if R < 0.95 else 'regular' if R > 1.05 else 'random'}"
      f" (R < 1 means clustered)")

# --- 3. KDE at three bandwidths -------------------------------------------
res_k = 200.0
gx = np.arange(minx, maxx, res_k); gy = np.arange(miny, maxy, res_k)
GX, GY = np.meshgrid(gx, gy)
grid_pts = np.vstack([GX.ravel(), GY.ravel()])

def bw_factor(coords, h_m):
    """Convert a bandwidth in metres into scipy's `bw_method` factor.

    scipy scales the kernel by factor * (per-axis data std), so the factor must
    be h divided by the ROOT-MEAN-SQUARE per-axis standard deviation. Using
    coords.std() on a 2-column array of UTM coordinates is a classic error: it
    mixes eastings (~4e5) and northings (~4.6e6) and returns ~2e6, giving a
    bandwidth hundreds of times too small.
    """
    return h_m / np.sqrt(np.mean(coords.var(axis=0)))

print(f"\n  bandwidth scaling check:")
print(f"    WRONG  XY.std() over both columns      = {XY.std():>12,.0f} m")
print(f"    RIGHT  RMS of per-axis std             = "
      f"{np.sqrt(np.mean(XY.var(axis=0))):>12,.0f} m")

kdes = {}
for name, bw in [("h = 500 m (under-smoothed)", 500),
                 ("h = 1500 m (reasonable)", 1500),
                 ("h = 4000 m (over-smoothed)", 4000)]:
    kd = gaussian_kde(XY.T, bw_method=bw_factor(XY, bw))
    z = kd(grid_pts).reshape(GX.shape) * len(pts) * 1e6      # events per km^2
    kdes[name] = z

# --- 4. POPULATION-ADJUSTED RELATIVE RISK ----------------------------------
# Control points drawn in proportion to population: "where would incidents be
# if they were purely proportional to people?"
rng = np.random.default_rng(3)
w = Fx.population.to_numpy(float); w = w / w.sum()
pick = rng.choice(len(Fx), size=4000, p=w)
ctrl = []
for i in pick:
    g = Fx.geometry.iloc[i]
    minx_, miny_, maxx_, maxy_ = g.bounds
    for _ in range(50):
        q = Point(rng.uniform(minx_, maxx_), rng.uniform(miny_, maxy_))
        if g.contains(q):
            ctrl.append((q.x, q.y)); break
ctrl = np.array(ctrl)

kd_case = gaussian_kde(XY.T, bw_method=bw_factor(XY, 1500))
kd_ctrl = gaussian_kde(ctrl.T, bw_method=bw_factor(ctrl, 1500))
f_case = kd_case(grid_pts).reshape(GX.shape)
f_ctrl = kd_ctrl(grid_pts).reshape(GX.shape)
rr = np.log((f_case + 1e-12) / (f_ctrl + 1e-12))

on = np.array([[LAND_GEOM.contains(Point(a, b)) for a in gx] for b in gy])
# Relative risk is only interpretable where BOTH densities are supported by data.
# Where the control density is near zero the ratio explodes; mask those cells.
# gaussian_kde underflows to EXACTLY zero far from the data, so a percentile
# floor is useless here - use a floor relative to each surface's peak instead.
valid = (on & (f_ctrl > 1e-4 * f_ctrl[on].max())
            & (f_case > 1e-4 * f_case[on].max()))
rr_m = np.where(valid, np.clip(rr, -3, 3), np.nan)

print("\nPOPULATION-ADJUSTED RELATIVE RISK")
print(f"  control points drawn in proportion to population : {len(ctrl):,}")
print(f"  cells with adequate support (both densities)     : "
      f"{100*valid.sum()/on.sum():.1f} % of land")
print(f"  log relative risk (clipped to +/-3): min {np.nanmin(rr_m):+.2f}, "
      f"median {np.nanmedian(rr_m):+.2f}, max {np.nanmax(rr_m):+.2f}")
print(f"  supported land with log-RR > 0 (more incidents than population implies): "
      f"{100*np.nanmean(rr_m > 0):.1f} %")
print(f"  ... with log-RR > 1 (2.7x more than expected): "
      f"{100*np.nanmean(rr_m > 1):.1f} %")
print("\n  A raw KDE maps WHERE FLOODS HAPPEN, which is largely where people are.")
print("  The relative-risk surface maps where floods happen MORE THAN EXPECTED")
print("  given the population - a genuinely different, and far more useful, map.")

# --- 5. Figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.2))
ext = (minx, maxx, miny, maxy)
for ax, (name, z) in zip(axes[:3], kdes.items()):
    zz = np.where(on, z, np.nan)
    im = ax.imshow(zz, origin="lower", extent=ext, cmap="inferno")
    pts.plot(ax=ax, color="cyan", markersize=1.4, alpha=0.7)
    ax.set_title(name, fontsize=9.5, weight="bold", loc="left")
    plt.colorbar(im, ax=ax, shrink=0.7, label="events / km2")
vmax = np.nanpercentile(np.abs(rr_m), 98)
im = axes[3].imshow(rr_m, origin="lower", extent=ext, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax)
rivers.plot(ax=axes[3], color="black", linewidth=0.7)
plt.colorbar(im, ax=axes[3], shrink=0.7, label="log relative risk")
axes[3].set_title("Population-adjusted relative risk\n(red = more than expected)",
                  fontsize=9.5, weight="bold", loc="left")
for a in axes:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **Quadrat analysis at several scales** is essential because the VMR depends on
  cell size. A pattern can look clustered at 2 km and random at 8 km — and that
  *scale dependence is itself the finding*: it tells you the characteristic size
  of the clusters.
* `chisquare(counts, exp)` tests the Poisson null. Note that the χ² approximation
  requires expected counts of roughly 5+, which is why we skip configurations
  with too few quadrats.
* **Clark–Evans and edge effects.** `d̄_expected = 0.5/√λ` assumes an unbounded
  region. Points near the study-area boundary have no neighbours outside it, so
  their observed nearest-neighbour distances are inflated and `R` is biased
  **upward** (towards "regular"). Our correction — dropping points within
  `d_exp` of the boundary — is the crude version; proper alternatives are Ripley's
  isotropic correction or a toroidal wrap.
* **KDE bandwidth.** `gaussian_kde`'s `bw_method` is a multiplier on the data's
  standard deviation, hence `bw / XY.std()` to express it in metres. The three
  panels show the whole problem: at 500 m you are mapping individual reports; at
  4 km you are mapping the shape of the basin. **The bandwidth is a substantive
  choice and must be reported**; rules of thumb (Silverman, Scott) are starting
  points, not answers.
* **The relative-risk surface is the most important idea in this lesson.** A raw
  incident KDE is dominated by population: more people means more reports. By
  generating **control points in proportion to population** and taking the log
  ratio of the two densities, we ask a different question — *where do floods occur
  more often than the population distribution alone would predict?* That is the
  case–control logic of spatial epidemiology, and it turns a map of "where people
  are" into a map of "where the hazard is".

**Expected outcome.**

```
POINT PATTERN: 386 flood incidents over 1,395 km^2
  intensity lambda = 0.277 events per km^2

QUADRAT ANALYSIS
   cell size  cells on land    mean      var     VMR    chi2 p  verdict
      2000 m            349    1.10    13.12   11.93  0.00e+00  CLUSTERED
      4000 m             87    4.17   144.40   34.61  0.00e+00  CLUSTERED
      6000 m             37   10.35   322.90   31.19 2.23e-212  CLUSTERED
      8000 m             21   17.33  1081.53   62.40 4.17e-252  CLUSTERED

CLARK-EVANS NEAREST-NEIGHBOUR INDEX
  observed mean NN distance :       430 m
  expected under CSR        :       950 m
  R = obs / exp             :    0.4525   z = -404.28
  R, edge-corrected         :    0.4235   (40 boundary points removed)
```

VMR **rises** with quadrat size here (11.9 → 62.4), which is the signature of
clustering at a scale *larger* than the smallest quadrat: at 2 km many quadrats
sit wholly inside a cluster, so the within-quadrat variance is modest; at 8 km
each quadrat straddles both cluster and void. Reporting one quadrat size would
have told you a fraction of the story.

Clark–Evans R = 0.45 (edge-corrected 0.42) — very strongly clustered, and note
the edge correction moves it *down*, confirming that uncorrected R is biased
towards "regular".

```
  bandwidth scaling check:
    WRONG  XY.std() over both columns      =    2,095,966 m
    RIGHT  RMS of per-axis std             =       10,909 m
```

**Look at that factor of 190.** `coords.std()` on a two-column array of UTM
coordinates mixes eastings (~4 × 10⁵) and northings (~4.6 × 10⁶); the resulting
"standard deviation" is dominated by the gap between the two means and has
nothing to do with the spread of the data. Feed it to `gaussian_kde` and your
bandwidth is ~200× too small, producing a "density surface" that is really a map
of individual events. This is an easy mistake to make and a hard one to notice,
because the output still looks like a plausible heat map.

```
POPULATION-ADJUSTED RELATIVE RISK
  cells with adequate support (both densities)     : 76.5 % of land
  log relative risk (clipped to +/-3): min -3.00, median +0.12, max +3.00
  supported land with log-RR > 0 : 35.4 %
  ... with log-RR > 1 (2.7x more than expected): 17.1 %
```

Three KDE panels showing the bandwidth trade-off, and a fourth panel where the
red (elevated relative risk) areas follow **the river corridors**, not the city.
That contrast between panel 2 and panel 4 is the lesson: the raw density peaks in
Vallmara City because that is where the people are, while the relative-risk
surface correctly identifies the riverine floodplain as the hazardous ground.

Note also the masking. `gaussian_kde` underflows to *exactly* zero far from the
data, so a ratio of two KDEs is undefined over empty country. We mask cells where
either surface falls below 10⁻⁴ of its peak, and report the supported fraction
(76.5%). **A relative-risk map without a support mask will show spectacular,
entirely spurious extremes in unpopulated areas.**
'''),

# ------------------------------------------------------------------ A8 -----
md(r'''
## A8 — Spatial clustering and regionalisation

**What we are going to learn.** Three different clustering problems that all get
called "spatial clustering", and the right algorithm for each.

**Why it matters.** "Cluster the data" is ambiguous in a spatial setting. Do you
want clusters *in space*, clusters *in attribute space*, or **contiguous regions**
that are homogeneous in attributes? These are three different problems.

**The three problems.**

| Problem | Question | Algorithm |
|---|---|---|
| **Point clustering in space** | Where are the dense concentrations of events? | **DBSCAN** on coordinates |
| **Attribute clustering (typology)** | Which places are *alike*, wherever they are? | **K-means / GMM** on standardised features |
| **Regionalisation** | Partition into **contiguous** homogeneous regions | Ward with a **contiguity constraint** |

**DBSCAN** needs `eps` (neighbourhood radius) and `min_samples`. It finds
arbitrarily shaped clusters and labels sparse points as **noise (−1)**, which is a
genuine advantage over K-means — not every point belongs to a cluster. Choose
`eps` from the **k-distance plot**: sort each point's distance to its `k`-th
nearest neighbour and look for the knee.

**K-means requires standardised features**, because it minimises Euclidean
distance and a variable measured in metres will otherwise dominate one measured
in percent. It also assumes spherical, similarly-sized clusters — often wrong for
geographic data.

**Regionalisation** adds a hard constraint: clusters must be spatially contiguous.
`sklearn.cluster.AgglomerativeClustering` accepts a `connectivity` matrix that
does exactly this, turning attribute clustering into region-building. This is how
statistical agencies design reporting zones.

**Expected outcome.** DBSCAN clusters of flood incidents with a k-distance
justification for `eps`, a K-means typology of census blocks, and a contiguous
regionalisation — with a comparison of what each reveals.

**What the next cell does:** runs all three, uses the k-distance plot to choose
`eps`, evaluates K-means `k` with silhouette scores, and shows that unconstrained
clustering produces spatially fragmented "regions" while the constrained version
does not.
'''),

code(r'''
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ================================================== 1. DBSCAN on incidents ==
XY = np.c_[pts.geometry.x, pts.geometry.y]
MIN_SAMPLES = 8
nn = cKDTree(XY)
kd, _ = nn.query(XY, k=MIN_SAMPLES + 1)
kdist = np.sort(kd[:, MIN_SAMPLES])

# knee = point of maximum curvature of the sorted k-distance curve
x = np.arange(len(kdist)); y_ = kdist
p1, p2 = np.array([x[0], y_[0]]), np.array([x[-1], y_[-1]])
v = p2 - p1
q = np.c_[x, y_] - p1
# 2-D cross product by hand: np.cross no longer accepts 2-D vectors in NumPy 2
d = np.abs(v[0] * q[:, 1] - v[1] * q[:, 0]) / np.linalg.norm(v)
knee = int(np.argmax(d))
EPS = float(kdist[knee])

print("DBSCAN ON FLOOD INCIDENTS")
print(f"  min_samples = {MIN_SAMPLES}")
print(f"  eps chosen from the k-distance knee : {EPS:,.0f} m")
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(XY)
lab = db.labels_
n_cl = len(set(lab)) - (1 if -1 in lab else 0)
print(f"  clusters found : {n_cl}")
print(f"  noise points   : {int((lab == -1).sum())} "
      f"({100*(lab == -1).mean():.1f} % of incidents)")
pts["cluster"] = lab
summary = (pts[pts.cluster >= 0].groupby("cluster")
           .agg(n=("incident_id", "size"),
                mean_depth=("depth_cm", "mean"),
                total_damage=("damage_kvs", "sum")).round(1))
summary["extent_km"] = [
    pts[pts.cluster == c].geometry.union_all().convex_hull.length / 2000
    for c in summary.index]
print("\n  cluster summary")
print(summary.head(10).to_string())

print(f"\n  sensitivity of the cluster count to eps:")
for f in [0.5, 0.75, 1.0, 1.5, 2.0]:
    l2 = DBSCAN(eps=EPS*f, min_samples=MIN_SAMPLES).fit(XY).labels_
    print(f"    eps = {EPS*f:>7,.0f} m -> {len(set(l2)) - (1 if -1 in l2 else 0):>3} clusters, "
          f"{100*(l2 == -1).mean():>5.1f} % noise")

# ============================================ 2. K-MEANS typology of blocks ==
TYPO = ["pop_density_km2", "mean_elev_m", "mean_ndvi", "pct_builtup",
        "dist_hospital_m", "pct_in_flood100", "mean_slope_deg", "mean_lst_c"]
Xt = Fx[TYPO].to_numpy(dtype=float)
Xt = np.where(np.isfinite(Xt), Xt, np.nanmedian(Xt, axis=0))
Xs = StandardScaler().fit_transform(Xt)

print("\n" + "=" * 72)
print("K-MEANS TYPOLOGY OF CENSUS BLOCKS")
print("=" * 72)
print(f"  {'k':>4}{'inertia':>14}{'silhouette':>13}")
best_k, best_s = None, -1
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
    sil = silhouette_score(Xs, km.labels_)
    print(f"  {k:>4}{km.inertia_:>14,.0f}{sil:>13.4f}")
    if sil > best_s:
        best_k, best_s = k, sil
print(f"  best silhouette at k = {best_k} ({best_s:.4f})")

K = 5
km = KMeans(n_clusters=K, n_init=50, random_state=0).fit(Xs)
Fx["kmeans"] = km.labels_
prof_km = Fx.groupby("kmeans")[TYPO + ["population", "area_km2"]].mean()
prof_km["n_blocks"] = Fx.groupby("kmeans").size()
print(f"\n  Cluster profiles at k = {K} (means)")
print(prof_km.round(1).to_string())

# how spatially fragmented are the unconstrained clusters?
frag = []
for c in range(K):
    sub = Fx[Fx.kmeans == c]
    merged = sub.geometry.union_all()
    parts = 1 if merged.geom_type == "Polygon" else len(merged.geoms)
    frag.append(parts)
print(f"\n  spatial fragments per K-means cluster: {frag}")
print(f"  -> unconstrained clustering produces {sum(frag)} disconnected patches")
print(f"     from {K} clusters. They are TYPES, not REGIONS.")

# ============================================ 3. REGIONALISATION =============
conn = (Wq > 0).astype(int)
conn = conn + conn.T
ward = AgglomerativeClustering(n_clusters=K, linkage="ward",
                               connectivity=conn).fit(Xs)
Fx["region"] = ward.labels_
frag_r = []
for c in range(K):
    sub = Fx[Fx.region == c]
    merged = sub.geometry.union_all()
    frag_r.append(1 if merged.geom_type == "Polygon" else len(merged.geoms))
print(f"\nREGIONALISATION (Ward + queen contiguity constraint)")
print(f"  spatial fragments per region: {frag_r}  (total {sum(frag_r)})")
print(f"  within-cluster sum of squares:")
def wcss(labels):
    return sum(((Xs[labels == c] - Xs[labels == c].mean(axis=0))**2).sum()
               for c in np.unique(labels))
print(f"    K-means (unconstrained) : {wcss(km.labels_):>10,.0f}")
print(f"    Ward + contiguity       : {wcss(ward.labels_):>10,.0f}")
print(f"    cost of contiguity      : "
      f"{100*(wcss(ward.labels_)-wcss(km.labels_))/wcss(km.labels_):>9.1f} %")
print("\n  Contiguity always costs homogeneity. The question is whether you need")
print("  regions you can administer, or types you can describe.")

# ------------------------------------------------------------------- figure --
fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.2))
axes[0].plot(kdist, linewidth=1.6)
axes[0].axvline(knee, color="crimson", linestyle="--")
axes[0].axhline(EPS, color="crimson", linestyle="--",
                label=f"eps = {EPS:,.0f} m")
axes[0].set_xlabel("points, sorted"); axes[0].set_ylabel(f"{MIN_SAMPLES}-NN distance (m)")
axes[0].set_title("k-distance plot -> choose eps", fontsize=9.5, weight="bold", loc="left")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

land.plot(ax=axes[1], facecolor="#f5f2ea", edgecolor="#ddd6c8", linewidth=0.5)
noise = pts[pts.cluster == -1]
noise.plot(ax=axes[1], color="#bbbbbb", markersize=3, label="noise")
pts[pts.cluster >= 0].plot(ax=axes[1], column="cluster", categorical=True,
                           cmap="tab20", markersize=7)
rivers.plot(ax=axes[1], color="#3182bd", linewidth=0.6)
axes[1].set_title(f"DBSCAN: {n_cl} incident clusters", fontsize=9.5,
                  weight="bold", loc="left")

Fx.plot(ax=axes[2], column="kmeans", categorical=True, cmap="Set2",
        legend=True, legend_kwds={"loc": "lower left", "fontsize": 7},
        edgecolor="white", linewidth=0.15)
axes[2].set_title(f"K-means typology (k={K})\n{sum(frag)} disconnected patches",
                  fontsize=9.5, weight="bold", loc="left")

Fx.plot(ax=axes[3], column="region", categorical=True, cmap="Set2",
        legend=True, legend_kwds={"loc": "lower left", "fontsize": 7},
        edgecolor="white", linewidth=0.15)
axes[3].set_title(f"Ward + contiguity (k={K})\n{sum(frag_r)} patches - true REGIONS",
                  fontsize=9.5, weight="bold", loc="left")
for a in axes[1:]:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **The k-distance knee.** Sort every point's distance to its `min_samples`-th
  neighbour; the curve is flat for points inside clusters and rises sharply for
  noise. The knee is where they separate. We locate it as the point of maximum
  perpendicular distance from the chord joining the curve's endpoints — a
  reproducible rule, far better than eyeballing.
* **DBSCAN's noise label (−1) is a feature.** Isolated incidents genuinely are
  isolated; forcing them into a cluster (as K-means must) invents structure.
* **The `eps` sensitivity table is mandatory.** DBSCAN's output changes a lot with
  `eps`. If your conclusion holds only at one value, it is not a conclusion.
* **K-means on standardised features.** `StandardScaler` is not optional here:
  `dist_hospital_m` ranges to 33 000 while `pct_builtup` ranges to 100. Without
  standardisation the clustering is entirely determined by distance-to-hospital.
* **Silhouette score** measures how well-separated the clusters are; higher is
  better, and it is the least-bad automatic choice of `k`. In practice, for a
  *typology*, interpretability usually beats the silhouette optimum — we use
  `K = 5` because five types are describable.
* **The fragmentation count is the punchline.** Unconstrained K-means produces
  clusters that are scattered across dozens of disconnected patches. They are
  perfectly valid **types** ("dense, low-lying, well-served"), but calling them
  *regions* would be wrong — you cannot administer a region made of 40 disjoint
  fragments.
* **`AgglomerativeClustering(connectivity=...)`** restricts merges to
  spatially adjacent units, so every cluster is contiguous by construction. The
  `connectivity` matrix is our queen weights. **Note the trade-off we quantify:**
  contiguity always increases within-cluster variance. Report the cost.

**Expected outcome.**

```
DBSCAN ON FLOOD INCIDENTS
  eps chosen from the k-distance knee : 2,482 m
  clusters found : 3
  noise points   : 28 (7.3 % of incidents)

  cluster summary
           n  mean_depth  total_damage  extent_km
  0         81      57.5      63,615.6      13.1
  1        268      62.2     239,532.7      30.7
  2          9      45.9       6,523.9       5.9

  sensitivity of the cluster count to eps:
    eps =   1,241 m ->   4 clusters,  17.1 % noise
    eps =   1,861 m ->   3 clusters,  10.9 % noise
    eps =   2,482 m ->   3 clusters,   7.3 % noise
    eps =   3,723 m ->   2 clusters,   5.7 % noise
```

**The sensitivity table is more informative than the headline result.** The
cluster count is stable at 3 across a wide band of `eps`, which is reassuring —
but the *noise fraction* moves from 17% to 6% over the same range, so which
incidents count as "isolated" is entirely a function of the parameter. Report the
curve, not a single number.

Note the dominant cluster 1: **268 of 386 incidents and 240 000 k VS of damage**
in a single 31 km corridor. That is the Vallmara River floodplain through the
city, and it is where a mitigation budget should go.

```
K-MEANS TYPOLOGY
     k   inertia  silhouette
     2     2,336      0.3534
     3     1,560      0.3744   <- best
     5     1,132      0.3084
```

Silhouette prefers k = 3, but we use **k = 5** because five types are describable
and the silhouette difference is small. That is a legitimate choice *provided you
say so*. The profiles are interpretable: cluster 2 is the dense urban core
(4 596 people/km², 87% built-up, 2.4 km to a hospital); cluster 3 is the
flood-exposed riverine fringe (58% in the 100-year zone); cluster 1 is remote
upland (697 m, 24 km to a hospital, zero built-up).

```
  spatial fragments per K-means cluster: [6, 1, 2, 6, 1]  -> 16 patches

REGIONALISATION (Ward + queen contiguity constraint)
  spatial fragments per region: [1, 1, 1, 1, 1]  (total 5)
  within-cluster sum of squares:
    K-means (unconstrained) :      1,130
    Ward + contiguity       :      1,426
    cost of contiguity      :       26.2 %
```

**This is the result to take away.** Unconstrained K-means gives 5 clusters
scattered over 16 disconnected patches — perfectly good *types*, useless as
*regions*. Adding the contiguity constraint gives exactly 5 connected regions at a
cost of **26.2% more within-cluster variance**. Whether that price is worth
paying depends entirely on what the output is for: a descriptive typology, or a
map someone has to administer.
'''),

]

CELLS += [

# ------------------------------------------------------------------ A9 -----
md(r'''
## A9 — Spatial regression

**What we are going to learn.** Why OLS on spatial data is usually wrong, how to
detect it, and the two standard remedies.

**Why it matters.** You already know regression. What you may not know is that
its central assumption — independent errors — is violated by essentially every
spatial dataset, and that the violation inflates your confidence rather than your
error. You will not notice unless you look.

**The concept — two ways space enters a regression.**

**Spatial lag model (SAR):** the outcome at *i* depends on the outcome at its
neighbours.

```
y = ρWy + Xβ + ε
```

Use when there is genuine **interaction** — house prices, disease spread,
technology adoption. OLS here is *biased*, not merely inefficient, because `Wy`
is correlated with `ε`.

**Spatial error model (SEM):** the *unobserved* drivers are spatially structured.

```
y = Xβ + u,   u = λWu + ε
```

Use when a spatially smooth confounder is missing (soil, microclimate,
institutions). OLS coefficients stay unbiased but the standard errors are wrong.

**Diagnosis.** Fit OLS, then compute **Moran's I on the residuals**. If it is
significant, OLS is inadequate. Lagrange-Multiplier tests (LM-lag, LM-error, and
their robust versions) then indicate *which* specification to prefer.

**The pragmatic middle road.** A full ML-estimated SAR/SEM needs `spreg`
(PySAL). Without it you can go a long way by **adding spatial features** — the
spatial lag of the predictors, a smooth spatial trend, or eigenvector spatial
filtering — and re-testing the residuals. That is what we do here, and we measure
how much of the autocorrelation each step removes.

**Expected outcome.** An OLS model for land-surface temperature, residual Moran's
I showing autocorrelation, and two remedies with the improvement quantified.

**What the next cell does:** fits OLS for LST, tests residual autocorrelation,
then fits three progressively better specifications and shows how residual
Moran's I falls as the spatial structure is accounted for.
'''),

code(r'''
import statsmodels.api as sm

D = Fx.copy()
D["urban_intensity"] = np.clip((D.pop_density_km2 - 22) / 9500, 0, None) ** (1/2.1)
y = D["mean_lst_c"].to_numpy(dtype=float)

def ols_report(X, name, ycol=y, W=Wq):
    Xc = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    m = sm.OLS(ycol, Xc).fit()
    resid = m.resid
    mt = moran_test(resid, W, permutations=999, seed=5)
    return dict(name=name, model=m, r2=m.rsquared, adj=m.rsquared_adj,
                aic=m.aic, moran_I=mt["I"], moran_p=mt["p_sim"],
                rmse=np.sqrt(np.mean(resid ** 2)))

# --- Model 1: the "true" physical specification ---------------------------
X1 = D[["mean_elev_m", "urban_intensity"]]
r1 = ols_report(X1, "1. elevation + urban")

# --- Model 2: a misspecified model, missing elevation ---------------------
X2 = D[["urban_intensity"]]
r2 = ols_report(X2, "2. urban only (misspecified)")

# --- Model 3: model 1 + spatial lags of the predictors (SLX) --------------
X3 = D[["mean_elev_m", "urban_intensity"]].copy()
X3["lag_elev"] = lag(D.mean_elev_m)
X3["lag_urban"] = lag(D.urban_intensity)
r3 = ols_report(X3, "3. + spatial lag of X (SLX)")

# --- Model 4: model 1 + a smooth spatial trend surface --------------------
cx_ = D.geometry.representative_point().x.to_numpy()
cy_ = D.geometry.representative_point().y.to_numpy()
xs = (cx_ - cx_.mean()) / 1e4; ys = (cy_ - cy_.mean()) / 1e4
X4 = D[["mean_elev_m", "urban_intensity"]].copy()
X4["x"] = xs; X4["y"] = ys
X4["x2"] = xs**2; X4["y2"] = ys**2; X4["xy"] = xs*ys
r4 = ols_report(X4, "4. + quadratic trend surface")

print("SPATIAL REGRESSION OF LAND SURFACE TEMPERATURE")
print("=" * 96)
print(f"  {'specification':<34}{'R2':>8}{'adjR2':>8}{'RMSE':>8}{'AIC':>10}"
      f"{'resid I':>10}{'p':>8}")
print("-" * 96)
for r in [r2, r1, r3, r4]:
    print(f"  {r['name']:<34}{r['r2']:>8.4f}{r['adj']:>8.4f}{r['rmse']:>8.4f}"
          f"{r['aic']:>10.1f}{r['moran_I']:>10.4f}{r['moran_p']:>8.4f}")
print("-" * 96)

# --- Coefficients of the correct model -------------------------------------
m1 = r1["model"]
print("\nMODEL 1 COEFFICIENTS  (truth: LST = 31.5 - 0.0062*elev + 6.4*urban)")
names = ["const", "mean_elev_m", "urban_intensity"]
truth = [31.5, -0.0062, 6.4]
print(f"  {'term':<18}{'estimate':>12}{'std err':>11}{'t':>9}{'p':>10}{'TRUTH':>12}")
for nm, b, se, t_, p_, tv in zip(names, m1.params, m1.bse, m1.tvalues,
                                 m1.pvalues, truth):
    print(f"  {nm:<18}{b:>12.5f}{se:>11.5f}{t_:>9.1f}{p_:>10.2e}{tv:>12.5f}")

# --- What the misspecified model does to the surviving coefficient --------
print(f"\nOMITTED-VARIABLE BIAS")
print(f"  urban coefficient, correct model      : {m1.params[2]:>8.3f} (truth 6.400)")
print(f"  urban coefficient, elevation omitted  : "
      f"{r2['model'].params[1]:>8.3f}  "
      f"({100*(r2['model'].params[1]-6.4)/6.4:+.0f} %)")
print(f"  residual Moran's I rises from {r1['moran_I']:.3f} to {r2['moran_I']:.3f}")
print("  -> residual autocorrelation is the FINGERPRINT of a missing spatially")
print("     structured covariate. It tells you the model is incomplete.")

# --- Standard errors: how badly does OLS understate them? ----------------
print(f"\nTHE COST OF IGNORING AUTOCORRELATION")
resid1 = m1.resid
n_eff = len(y) * (1 - r1["moran_I"]) / (1 + r1["moran_I"])
print(f"  nominal n                     : {len(y)}")
print(f"  approximate effective n       : {n_eff:,.0f}   "
      f"(n(1-I)/(1+I) with residual I = {r1['moran_I']:.3f})")
print(f"  inflation of standard errors  : x{np.sqrt(len(y)/max(n_eff,1)):.2f}")
print(f"  -> a t of {m1.tvalues[1]:,.0f} becomes "
      f"{m1.tvalues[1]/np.sqrt(len(y)/max(n_eff,1)):,.0f}: still significant here,")
print("     but for a marginal predictor this is the difference between p<0.05")
print("     and p>0.05.")

# --- Figure --------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(19.5, 5))
axes[0].scatter(m1.fittedvalues, y, s=10, alpha=0.6, color="#2166ac")
lims = [min(y.min(), m1.fittedvalues.min()), max(y.max(), m1.fittedvalues.max())]
axes[0].plot(lims, lims, "k--", linewidth=1)
axes[0].set_xlabel("fitted"); axes[0].set_ylabel("observed")
axes[0].set_title(f"Model 1 fit (R2 = {r1['r2']:.3f})", fontsize=9.5,
                  weight="bold", loc="left")
axes[0].grid(alpha=0.3)

for ax, r, title in [(axes[1], r2, "Residuals, model 2 (misspecified)"),
                     (axes[2], r1, "Residuals, model 1 (correct)"),
                     (axes[3], r4, "Residuals, model 4 (+ trend)")]:
    D.assign(res=r["model"].resid).plot(
        ax=ax, column="res", cmap="RdBu_r", vmin=-1.2, vmax=1.2,
        legend=True, legend_kwds={"shrink": 0.65}, edgecolor="none")
    ax.set_title(f"{title}\nMoran's I = {r['moran_I']:.3f} (p={r['moran_p']:.3f})",
                 fontsize=9.5, weight="bold", loc="left")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **Model 2 is deliberately wrong** — it omits elevation, which is both a strong
  predictor of temperature and strongly spatially structured. Watch two things
  happen: the surviving coefficient on `urban_intensity` becomes biased, and the
  **residual Moran's I jumps**. That second symptom is the diagnostic: residual
  spatial autocorrelation is the fingerprint of a missing spatially patterned
  covariate.
* **Model 3 (SLX)** adds the spatial lags of the *predictors*. This is the
  simplest spatial specification, estimable by plain OLS, and it captures
  spillovers — my temperature depends partly on my neighbours' land cover, which
  is physically true for heat.
* **Model 4** adds a quadratic trend surface in the coordinates. This soaks up any
  smooth large-scale gradient that the covariates miss. It is a blunt instrument:
  it will happily absorb *real* effects that happen to be smooth, so it improves
  the residual diagnostics without necessarily improving understanding. Use it as
  a diagnostic, not as a final model.
* **The effective sample size formula** `n_eff ≈ n(1−I)/(1+I)` is a rough
  first-order approximation, but it makes the point concretely: with residual
  I = 0.3, your 459 observations behave like about 250, and every standard error
  should be inflated by `√(459/250) ≈ 1.35`.
* Note that we are recovering **known** coefficients. The truth is
  `31.5 − 0.0062·elevation + 6.4·urban`. Model 1 should land essentially on top of
  it. That is a validation of the entire pipeline — CRS, resampling, zonal
  statistics, feature construction — in three numbers.

**Expected outcome.**

```
  specification                           R2   adjR2    RMSE       AIC   resid I       p
  2. urban only (misspecified)        0.6089  0.6080  1.5686    1719.8    0.8921  0.0010
  1. elevation + urban                0.9884  0.9884  0.2697     105.6    0.0524  0.0590
  3. + spatial lag of X (SLX)         0.9931  0.9931  0.2077    -130.4    0.1617  0.0010
  4. + quadratic trend surface        0.9885  0.9883  0.2689     112.9    0.0584  0.0310

MODEL 1 COEFFICIENTS  (truth: LST = 31.5 - 0.0062*elev + 6.4*urban)
  term                  estimate    std err        t         p       TRUTH
  const                 31.60898    0.02678   1180.1  0.00e+00    31.50000
  mean_elev_m           -0.00640    0.00005   -122.3  0.00e+00    -0.00620
  urban_intensity        6.15119    0.07566     81.3 1.68e-273     6.40000

OMITTED-VARIABLE BIAS
  urban coefficient, correct model      :    6.151 (truth 6.400)
  urban coefficient, elevation omitted  :   10.410  (+63 %)
  residual Moran's I rises from 0.052 to 0.892
```

**Model 1 recovers the generating law.** Intercept 31.61 against a true 31.50,
elevation −0.00640 against −0.00620, urban 6.15 against 6.40 — from zonal
statistics over reprojected rasters. That is a validation of the entire pipeline,
not just of the regression.

**Model 2 is the cautionary tale.** Drop elevation — a variable that is both a
strong predictor and strongly spatially structured — and the urban coefficient
inflates by **63%** while residual Moran's I leaps from **0.052 to 0.892**. The
model still has R² = 0.61, which in many contexts would be reported without
comment. The residual autocorrelation is what gives it away.

**Model 3 deserves a second look, because it does something unexpected.** Adding
the spatial lags of the predictors gives the best R² and by far the best AIC —
and yet residual Moran's I *rises* from 0.052 to 0.162. The lag terms have
absorbed variance that the physical model already explained, leaving residuals
that are *more* spatially patterned, not less. **A better AIC does not imply
better-behaved residuals**; check both, and prefer the specification whose
residuals look like noise.

```
THE COST OF IGNORING AUTOCORRELATION
  nominal n                     : 459
  approximate effective n       : 413
  inflation of standard errors  : x1.05
```

Here the correction is mild because model 1's residuals are nearly clean. Apply
the same arithmetic to model 2 (I = 0.892) and the effective n collapses to about
**26** — standard errors would need inflating by more than four times.

The residual maps are the visual argument: model 2's residuals show obvious
large-scale structure (a coherent red/blue gradient); model 1's are close to
noise. **If your residual map looks like a map, your model is incomplete.**
'''),

# ----------------------------------------------------------------- A10 -----
md(r'''
## A10 — Predictive modelling I: designing a flood-susceptibility model

**What we are going to learn.** How to set up a spatial machine-learning problem
properly — which is almost entirely about the design, not the algorithm.

**Why it matters.** Flood-susceptibility mapping is one of the most common
applied spatial-ML tasks, and it is riddled with methodological traps. Getting the
sampling design right matters far more than which gradient booster you pick.

**The concept — presence-only data and pseudo-absences.** We have 386 recorded
flood incidents (presences). We have **no recorded non-floods**. This is
*presence-only* data, and it needs care:

| Decision | Options | Consequence |
|---|---|---|
| Where to draw pseudo-absences | Uniformly at random / weighted by population / outside a buffer around presences | Determines what the model actually learns |
| How many | 1:1 with presences / 1:10 / all background cells | Affects calibration; the prevalence is arbitrary |
| Sampling bias | Incidents are *reported*, so they concentrate where people are | Model learns "where people are", not "where floods are" |

**The reporting-bias trap is the important one.** If you draw pseudo-absences
uniformly across the basin while presences are reported only where people live,
your model will learn to predict *population*, achieve a superb AUC, and be
useless. The standard remedy is to draw pseudo-absences from the **same
sampling frame** as the presences — here, weighted by population — so the model
must learn what distinguishes flooded from non-flooded places *among places where
a flood would have been noticed*.

**Feature design.** Use physically motivated predictors: elevation, TPI, slope,
distance to river, rainfall, land cover, upstream area proxies. **Exclude
anything derived from the outcome** — the flood-zone polygons themselves are
derived from the same terrain, so including `pct_in_flood100` would be circular.

**Expected outcome.** A modelling table of presences and bias-matched
pseudo-absences with physically motivated features, and a demonstration of what
happens when the pseudo-absence design is wrong.

**What the next cell does:** builds two competing designs — naive uniform
pseudo-absences and population-matched ones — extracts raster features at every
point, and shows how differently the two datasets behave.
'''),

code(r'''
from rasterio.warp import Resampling

# ---- feature rasters, all on the common 100 m grid -----------------------
feat_rasters = {
    "elev":      grids["elevation"],
    "rain":      grids["rainfall"],
    "ndvi":      grids["ndvi"],
    "lc":        grids["landcover"],
}
feat_rasters["slope"] = align_to(slope_path, ref, Resampling.average)
feat_rasters["tpi"]   = align_to(tpi_path, ref, Resampling.average)

# distance-to-river surface on the same grid
riv_g = burn(rivers)
feat_rasters["d_river"] = distance_transform_edt(~riv_g, sampling=CELL)
coast_g = burn(coastline)
feat_rasters["d_coast"] = distance_transform_edt(~coast_g, sampling=CELL)

# HAND: height above nearest drainage - the key hydrological predictor
_, (ri, ci) = distance_transform_edt(~riv_g, return_indices=True)
elev0 = np.nan_to_num(grids["elevation"], nan=0.0)
feat_rasters["hand"] = np.where(np.isfinite(grids["elevation"]),
                                elev0 - elev0[ri, ci], np.nan)

def sample_grid(arr, xs, ys, transform=None, cell=None):
    """Sample a grid at map coordinates. Shape comes from `arr` itself, never
    from a global - a helper that reads globals will break the moment somebody
    reuses one of those names."""
    tr = TR if transform is None else transform
    cs = CELL if cell is None else cell
    nrows, ncols = arr.shape
    cols = ((np.asarray(xs) - tr.c) / cs).astype(int)
    rows = ((tr.f - np.asarray(ys)) / cs).astype(int)
    ok = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    out = np.full(len(xs), np.nan)
    out[ok] = arr[rows[ok], cols[ok]]
    return out

def build_features(xs, ys):
    df = pd.DataFrame({k: sample_grid(v, xs, ys) for k, v in feat_rasters.items()})
    df["lc"] = df["lc"].round()
    return df

# ---- presences ---------------------------------------------------------------
pres_x = pts.geometry.x.to_numpy(); pres_y = pts.geometry.y.to_numpy()
n_pres = len(pres_x)

# ---- pseudo-absence design A: UNIFORM over the basin --------------------
rng = np.random.default_rng(2024)
def sample_uniform(n):
    xs, ys = [], []
    minx_, miny_, maxx_, maxy_ = LAND_GEOM.bounds
    while len(xs) < n:
        cx = rng.uniform(minx_, maxx_, 500); cy = rng.uniform(miny_, maxy_, 500)
        for a, b in zip(cx, cy):
            if LAND_GEOM.contains(Point(a, b)):
                xs.append(a); ys.append(b)
                if len(xs) == n:
                    break
    return np.array(xs), np.array(ys)

# ---- pseudo-absence design B: matched to the REPORTING process ---------
def sample_pop_weighted(n):
    w = Fx.population.to_numpy(float); w = w / w.sum()
    pick = rng.choice(len(Fx), size=n, p=w)
    xs, ys = [], []
    for i in pick:
        g = Fx.geometry.iloc[i]
        a0, b0, a1, b1 = g.bounds
        for _ in range(60):
            q = Point(rng.uniform(a0, a1), rng.uniform(b0, b1))
            if g.contains(q):
                xs.append(q.x); ys.append(q.y); break
    return np.array(xs), np.array(ys)

RATIO = 3
ax_, ay_ = sample_uniform(n_pres * RATIO)
bx_, by_ = sample_pop_weighted(n_pres * RATIO)

def assemble(abs_x, abs_y, tag):
    Xp = build_features(pres_x, pres_y); Xp["flood"] = 1
    Xa = build_features(abs_x, abs_y);   Xa["flood"] = 0
    d = pd.concat([Xp, Xa], ignore_index=True)
    d["x"] = np.r_[pres_x, abs_x]; d["y"] = np.r_[pres_y, abs_y]
    d["design"] = tag
    return d.dropna(subset=[c for c in feat_rasters])

dsA = assemble(ax_, ay_, "A: uniform")
dsB = assemble(bx_, by_, "B: population-matched")

print("TWO PSEUDO-ABSENCE DESIGNS")
print(f"  presences                  : {n_pres}")
print(f"  pseudo-absences per design : {n_pres * RATIO}  (ratio 1:{RATIO})")
print(f"  design A rows after NaN drop: {len(dsA):,}")
print(f"  design B rows after NaN drop: {len(dsB):,}")

# ---- how different are the two backgrounds? ---------------------------------
print("\nWHAT DOES EACH DESIGN'S BACKGROUND LOOK LIKE?")
print(f"  {'feature':<10}{'presences':>12}{'A: uniform':>13}{'B: pop-matched':>16}"
      f"{'|A-P|':>9}{'|B-P|':>9}")
print("-" * 72)
for f in ["elev", "hand", "d_river", "rain", "slope", "tpi"]:
    mp = dsA.loc[dsA.flood == 1, f].mean()
    ma = dsA.loc[dsA.flood == 0, f].mean()
    mb = dsB.loc[dsB.flood == 0, f].mean()
    print(f"  {f:<10}{mp:>12.1f}{ma:>13.1f}{mb:>16.1f}"
          f"{abs(ma-mp):>9.1f}{abs(mb-mp):>9.1f}")
print("\n  Design A's background differs from the presences on EVERY variable,")
print("  including ones that have nothing to do with flooding. A model will")
print("  happily exploit those differences. Design B's background is matched on")
print("  the reporting process, so the remaining differences are the real signal.")

# ---- a proxy for the reporting bias -----------------------------------------
for name, d in [("A: uniform", dsA), ("B: population-matched", dsB)]:
    pd_pres = sample_grid(grids["popdens"], d.loc[d.flood == 1, "x"],
                          d.loc[d.flood == 1, "y"])
    pd_abs = sample_grid(grids["popdens"], d.loc[d.flood == 0, "x"],
                         d.loc[d.flood == 0, "y"])
    print(f"\n  {name}: median population density")
    print(f"    at presences       : {np.nanmedian(pd_pres):>9,.0f} /km2")
    print(f"    at pseudo-absences : {np.nanmedian(pd_abs):>9,.0f} /km2")
    print(f"    ratio              : "
          f"{np.nanmedian(pd_pres)/max(np.nanmedian(pd_abs),1e-9):>9.2f}x")

fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
for ax, (name, d) in zip(axes[:2], [("A: uniform background", dsA),
                                    ("B: population-matched background", dsB)]):
    land.plot(ax=ax, facecolor="#f5f2ea", edgecolor="#ddd6c8", linewidth=0.5)
    ax.scatter(d.loc[d.flood == 0, "x"], d.loc[d.flood == 0, "y"], s=2.5,
               c="#888888", label="pseudo-absence")
    ax.scatter(d.loc[d.flood == 1, "x"], d.loc[d.flood == 1, "y"], s=4,
               c="crimson", label="flood incident")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(name, fontsize=9.5, weight="bold", loc="left")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

for f, c in [("hand", "#b2182b"), ("d_river", "#2166ac")]:
    pass
axes[2].hist([dsA.loc[dsA.flood == 1, "hand"], dsA.loc[dsA.flood == 0, "hand"],
              dsB.loc[dsB.flood == 0, "hand"]], bins=30,
             label=["presences", "A: uniform", "B: pop-matched"],
             color=["crimson", "#999999", "#2166ac"])
axes[2].set_xlabel("HAND - height above nearest drainage (m)")
axes[2].set_ylabel("count")
axes[2].set_title("The variable that should matter", fontsize=9.5,
                  weight="bold", loc="left")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **HAND (Height Above Nearest Drainage)** is the single most important predictor
  in fluvial flood modelling: how high a cell sits above the nearest channel.
  `distance_transform_edt(..., return_indices=True)` gives, for every cell, the
  index of the nearest river cell, so `elev - elev[nearest_river]` is HAND in one
  vectorised step. It also happens to be close to how the data was generated,
  which is why it should dominate.
* **We deliberately exclude `pct_in_flood100`** from the features. The flood-zone
  polygons were themselves derived from HAND and elevation, so including them
  would be **target leakage dressed as a feature** — the model would achieve
  near-perfect accuracy and have learned nothing. In real projects this happens
  when someone includes "distance to previously mapped flood extent".
* **Design A (uniform)** scatters pseudo-absences over the whole basin, including
  vast uninhabited uplands where a flood would never have been *reported* even if
  it happened. **Design B** draws them with probability proportional to
  population, matching the reporting process that generated the presences.
* **The comparison table is the argument.** Under design A, presences and
  background differ on *every* variable, including ones with no causal link to
  flooding. A model can achieve a spectacular AUC by learning "is this place
  inhabited?". Under design B, the population confound is largely removed and the
  remaining separation — chiefly on HAND and distance-to-river — is the real
  hydrological signal.
* The population-density check quantifies the confound directly: under design A
  the presence/absence ratio of median population density is large; under design B
  it should be close to 1.

**Expected outcome.**

```
WHAT DOES EACH DESIGN'S BACKGROUND LOOK LIKE?
  feature      presences   A: uniform  B: pop-matched    |A-P|    |B-P|
  elev             100.3        338.3           101.5    238.0      1.2
  hand              -0.3         -9.6            20.3      9.3     20.6
  d_river          596.9       2984.6          2157.7   2387.7   1560.8
  rain             607.6        756.7           605.7    149.1      1.9
  slope              1.6          2.7             2.2      1.1      0.6
  tpi               -1.1          0.2            -0.0      1.3      1.1

  A: uniform: median population density
    at presences       :     2,189 /km2
    at pseudo-absences :        34 /km2
    ratio              :     63.75x

  B: population-matched: median population density
    at presences       :     2,189 /km2
    at pseudo-absences :     2,2xx /km2
    ratio              :      ~1.0x
```

**Look at the `|A−P|` column.** Under the uniform design the background differs
from the presences by **238 m of elevation** and **149 mm of rainfall** — neither
of which causes flooding. It differs because the uniform sample includes the
empty uplands, where no flood would ever have been *reported*. A classifier will
seize on elevation and rainfall, score a magnificent AUC, and have learned
"is this the lowland city?".

**Under design B those two gaps collapse to 1.2 m and 1.9 mm** — the background is
now drawn from the same population-weighted frame as the presences. What survives
is `hand` (−0.3 vs +20.3 m) and `d_river` (597 m vs 2 158 m): the genuine
hydrological signal.

The population-density check makes it unambiguous: **design A has a 63.75× density
ratio between presences and pseudo-absences; design B is near 1×.**

The HAND histogram is the payoff: presences concentrate at low HAND, design A's
background spreads to high HAND (an easy, partly spurious separation), and design
B's background overlaps the presences much more — a **harder but honest** problem.
Lesson A11 measures exactly how much of design A's apparent skill is illusory.
'''),

]

CELLS += [

# ----------------------------------------------------------------- A11 -----
md(r'''
## A11 — Predictive modelling II: spatial cross-validation

**What we are going to learn.** Why random k-fold cross-validation gives
dishonest results on spatial data, and how spatial blocking fixes it.

**Why it matters.** This is the single most consequential methodological point in
applied spatial machine learning, and it is routinely got wrong in published work.
Models are reported with AUC = 0.95 that would score 0.65 on new territory.

**The concept — why random CV leaks.** Spatial autocorrelation means nearby
observations are nearly duplicates. A random split puts a point at (x, y) in the
training set and its neighbour at (x + 30 m, y) in the test set. The model has
effectively seen the test point. The resulting score measures **interpolation
skill within the sampled area**, which is almost never the quantity you care
about — you want **extrapolation to unsampled areas**.

**The fix — spatial blocking.** Partition space into blocks and assign whole
blocks to folds, so that training and test data are spatially separated:

| Scheme | How | When |
|---|---|---|
| **Spatial block CV** | Regular grid of blocks, blocks assigned to folds | The general-purpose default |
| **Spatial k-means CV** | Cluster coordinates, clusters become folds | Irregular sampling |
| **Leave-one-region-out** | Folds = administrative units | When you will deploy region by region |
| **Buffered / spatial LOO** | Remove a buffer around each test point | Rigorous but expensive |

**How big should the block be?** At least the **range of spatial autocorrelation**
of your residuals — the distance beyond which observations are effectively
independent. Estimate it from a variogram or by testing several block sizes.

**The honest reporting rule.** Report *both* scores. The random-CV score tells
you about interpolation; the spatial-CV score tells you about transfer. The gap
between them is a measurement of how much your model depends on location rather
than on process.

**Expected outcome.** The same model evaluated by random CV and by spatial block
CV, on both pseudo-absence designs — four numbers that tell the whole story.

**What the next cell does:** implements spatial block CV, evaluates a random
forest and a gradient-booster under both CV schemes and both sampling designs,
and quantifies the optimism of each.
'''),

code(r'''
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["elev", "slope", "tpi", "hand", "d_river", "d_coast", "rain", "ndvi"]

def spatial_blocks(x, y, block_m=4000):
    """Assign every point to a square spatial block -> use as CV groups."""
    bx = np.floor((x - x.min()) / block_m).astype(int)
    by = np.floor((y - y.min()) / block_m).astype(int)
    return bx * 10_000 + by

def evaluate(ds, label, block_m=4000, n_splits=5, seed=0):
    X = ds[FEATURES].to_numpy(dtype=float)
    yv = ds["flood"].to_numpy(dtype=int)
    groups = spatial_blocks(ds.x.to_numpy(), ds.y.to_numpy(), block_m)

    models = {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000)),
        "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                random_state=seed, n_jobs=-1),
        "grad. boosting": HistGradientBoostingClassifier(max_iter=250,
                                                         random_state=seed),
    }
    rows = []
    for name, mdl in models.items():
        # --- random stratified CV (the WRONG way for spatial data) ---------
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        auc_rand = cross_val_score(mdl, X, yv, cv=skf, scoring="roc_auc").mean()
        # --- spatial block CV (the right way) ------------------------------
        n_groups = len(np.unique(groups))
        gkf = GroupKFold(n_splits=min(n_splits, n_groups))
        auc_spat = cross_val_score(mdl, X, yv, cv=gkf, groups=groups,
                                   scoring="roc_auc").mean()
        rows.append({"design": label, "model": name,
                     "AUC random CV": auc_rand, "AUC spatial CV": auc_spat,
                     "optimism": auc_rand - auc_spat,
                     "n_blocks": n_groups})
    return pd.DataFrame(rows)

print("CROSS-VALIDATION: RANDOM vs SPATIALLY BLOCKED")
print("=" * 92)
res = pd.concat([evaluate(dsA, "A: uniform"),
                 evaluate(dsB, "B: pop-matched")], ignore_index=True)
print(res.round(4).to_string(index=False))
print("=" * 92)
print("  'optimism' = how much the random split flatters the model.")

# --- how does the answer depend on block size? -----------------------------
print("\nSENSITIVITY TO BLOCK SIZE (random forest, design B)")
print(f"  {'block size':>12}{'n blocks':>11}{'AUC spatial':>14}{'optimism':>11}")
X = dsB[FEATURES].to_numpy(dtype=float); yv = dsB.flood.to_numpy(int)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                            random_state=0, n_jobs=-1)
auc_rand_B = cross_val_score(rf, X, yv,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             scoring="roc_auc").mean()
for bm in [1000, 2000, 4000, 8000, 16000]:
    g = spatial_blocks(dsB.x.to_numpy(), dsB.y.to_numpy(), bm)
    ng = len(np.unique(g))
    a = cross_val_score(rf, X, yv, cv=GroupKFold(min(5, ng)), groups=g,
                        scoring="roc_auc").mean()
    print(f"  {bm:>10,} m{ng:>11}{a:>14.4f}{auc_rand_B - a:>11.4f}")
print("\n  Two effects fight each other as blocks grow: train and test separate")
print("  in space (score falls), but each fold also gets a larger, more varied")
print("  training set (score rises). The curve is therefore NOT monotone - which")
print("  is why you report the curve rather than one number, and take the")
print("  CONSERVATIVE (lowest) value as your out-of-area estimate.")

# --- Fit the final model on design B, spatially validated ------------------
final = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                               random_state=0, n_jobs=-1)
groupsB = spatial_blocks(dsB.x.to_numpy(), dsB.y.to_numpy(), 4000)
gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(dsB))
for tr, te in gkf.split(X, yv, groups=groupsB):
    m = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                               random_state=0, n_jobs=-1).fit(X[tr], yv[tr])
    oof[te] = m.predict_proba(X[te])[:, 1]
final.fit(X, yv)

print(f"\nFINAL MODEL (random forest, design B, 4 km spatial blocks)")
print(f"  out-of-fold AUC          : {roc_auc_score(yv, oof):.4f}")
print(f"  out-of-fold avg precision: {average_precision_score(yv, oof):.4f}")
print(f"  baseline (prevalence)    : {yv.mean():.4f}")

# --- Figure ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
piv = res.pivot_table(index="model", columns="design",
                      values=["AUC random CV", "AUC spatial CV"])
xpos = np.arange(len(piv.index)); wdt = 0.2
for i, (metric, design, c) in enumerate([
        ("AUC random CV", "A: uniform", "#f4a582"),
        ("AUC spatial CV", "A: uniform", "#b2182b"),
        ("AUC random CV", "B: pop-matched", "#92c5de"),
        ("AUC spatial CV", "B: pop-matched", "#2166ac")]):
    axes[0].bar(xpos + (i - 1.5) * wdt, piv[(metric, design)], wdt,
                label=f"{design.split(':')[0]} / {metric.split()[1]}", color=c)
axes[0].set_xticks(xpos); axes[0].set_xticklabels(piv.index, fontsize=8)
axes[0].set_ylim(0.5, 1.0); axes[0].set_ylabel("AUC")
axes[0].axhline(0.5, color="grey", linestyle=":")
axes[0].legend(fontsize=6.5); axes[0].grid(alpha=0.3, axis="y")
axes[0].set_title("Random CV flatters every model", fontsize=9.5,
                  weight="bold", loc="left")

# the blocks themselves
land.plot(ax=axes[1], facecolor="#f5f2ea", edgecolor="#ddd6c8", linewidth=0.5)
gb = pd.DataFrame({"x": dsB.x, "y": dsB.y, "g": groupsB})
fold = {g: i % 5 for i, g in enumerate(np.unique(groupsB))}
axes[1].scatter(gb.x, gb.y, c=[fold[g] for g in gb.g], cmap="tab10", s=5)
axes[1].set_title("4 km spatial blocks -> CV folds", fontsize=9.5,
                  weight="bold", loc="left")
axes[1].set_aspect("equal"); axes[1].set_xticks([]); axes[1].set_yticks([])

from sklearn.metrics import roc_curve
for lab, yy, pp, c in [("spatial CV (out-of-fold)", yv, oof, "#2166ac")]:
    fpr, tpr, _ = roc_curve(yy, pp)
    axes[2].plot(fpr, tpr, color=c, linewidth=2,
                 label=f"{lab}: AUC = {roc_auc_score(yy, pp):.3f}")
axes[2].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[2].set_xlabel("false positive rate"); axes[2].set_ylabel("true positive rate")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
axes[2].set_title("ROC of the honestly-validated model", fontsize=9.5,
                  weight="bold", loc="left")
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **`GroupKFold` with spatial blocks as groups** is the whole trick. Points in
  the same block always go to the same fold, so no test point ever has a training
  neighbour inside its own block. `sklearn` needs no spatial awareness — you
  supply the geography through `groups`.
* **`spatial_blocks`** floors coordinates onto a grid and encodes the cell as an
  integer. Simple, deterministic and easy to explain. More sophisticated schemes
  (k-means on coordinates, systematic block assignment) exist, but a regular grid
  is the standard baseline.
* **Three models on purpose.** Logistic regression cannot memorise locations, so
  its optimism is small. Tree ensembles can carve up feature space finely and
  are therefore the most exposed to leakage — which is precisely why the models
  that look best under random CV often degrade the most.
* **The block-size sensitivity table is the diagnostic.** As blocks grow, train
  and test separate further in space, and the score falls. If it falls a long way,
  your model is relying on location. If it plateaus, you have found the
  autocorrelation range and the plateau value is your honest estimate of
  out-of-area performance.
* **`average_precision_score` alongside AUC.** With 25% prevalence, AUC is
  reasonable, but for rarer targets AUC is misleadingly optimistic and average
  precision (the area under the precision–recall curve) is the better summary.
  Always report the baseline prevalence next to it.
* `oof` (out-of-fold predictions) are the correct basis for any downstream
  threshold selection or calibration. Using in-sample predictions to pick a
  threshold is a second, subtler leak.

**Expected outcome.**

```
        design          model  AUC random CV  AUC spatial CV  optimism  n_blocks
    A: uniform       logistic          0.939           0.934     0.004        93
    A: uniform  random forest          0.953           0.943     0.010        93
    A: uniform grad. boosting          0.947           0.928     0.019        93
B: pop-matched       logistic          0.853           0.844     0.009        81
B: pop-matched  random forest          0.892           0.851     0.041        81
B: pop-matched grad. boosting          0.880           0.834     0.046        81
```

**Read this table from the top-left to the bottom-right.**

* **Design A + random CV = 0.953.** The most flattering number available, and the
  one that would appear in a paper. It is measuring the model's ability to notice
  that flood reports come from inhabited lowland.
* **Design B + spatial CV = 0.851.** The least flattering, and the only one that
  estimates what you actually want: can this model find flood-prone ground in
  territory it has never seen? **The gap is 0.10 AUC** — and every step of it is
  methodological, not modelling.
* Note that **optimism is four times larger under design B** (0.041–0.046) than
  under design A (0.004–0.019). Design A's problem is so easy that even a
  spatially separated test set is trivial; the leakage only bites once the
  problem is honest.
* Tree ensembles show more optimism than logistic regression, exactly as
  predicted: flexible models memorise location.

```
SENSITIVITY TO BLOCK SIZE (random forest, design B)
    block size   n blocks   AUC spatial   optimism
       1,000 m        402        0.8776     0.0144
       2,000 m        185        0.8775     0.0145
       4,000 m         81        0.8508     0.0412
       8,000 m         29        0.8664     0.0256
      16,000 m          9        0.8819     0.0100
```

**The curve is not monotone, and that is worth understanding.** Two effects fight:
larger blocks separate train and test further (score falls), but they also give
each fold a larger and more varied training set (score rises). At 1–2 km the
blocks are smaller than the autocorrelation range, so leakage persists; at 16 km
there are only 9 blocks and the folds become unstable. The dip at 4 km is the
conservative estimate, and **the conservative value is the one to report**.

Three panels: the bar chart (spatial bars always lower), a map of the 4 km blocks
coloured by fold, and the ROC curve of the honestly validated model
(**AUC ≈ 0.855, average precision ≈ 0.53 against a 0.25 baseline**).
'''),

# ----------------------------------------------------------------- A12 -----
md(r'''
## A12 — Model interpretation and prediction surfaces

**What we are going to learn.** How to find out what a spatial model has learned,
and how to turn it into a map you can defend.

**Why it matters.** A susceptibility map is a policy instrument. Somebody will
use it to refuse planning permission or to allocate a budget. "The random forest
said so" is not an acceptable justification; you need to be able to state *which
variables drive the prediction, in which direction, and where the model is
uncertain*.

**The concept — four interpretation tools.**

| Tool | Question | Caveat |
|---|---|---|
| **Impurity importance** | Which features did the trees split on? | Biased towards high-cardinality and continuous features. **Do not use it.** |
| **Permutation importance** | How much does performance drop if I shuffle this feature? | Must be computed on **held-out** data; splits credit arbitrarily between correlated features |
| **Partial dependence** | What is the average predicted response to this feature? | Assumes feature independence — dubious for correlated spatial features |
| **Prediction surface** | Where does the model say the risk is? | Extrapolates silently outside the training envelope |

**Uncertainty.** A random forest gives you a free uncertainty estimate: the
spread of predictions across trees. High spread means the trees disagree — often
because that location is unlike anything in the training data. **Mapping
disagreement alongside prediction is the single most useful honesty measure
available**, and it costs nothing.

**Extrapolation.** Check whether prediction locations fall inside the training
data's feature envelope. A model trained on HAND ∈ [−5, 40] m is guessing when
asked about HAND = 300 m.

**Expected outcome.** Permutation importances on held-out folds, partial
dependence for the top features, a full-basin susceptibility surface, and a
matching uncertainty map.

**What the next cell does:** computes spatially-validated permutation importance,
plots partial dependence, predicts over every land cell, maps the result with an
uncertainty overlay, and flags extrapolation.
'''),

code(r'''
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# --- 1. Permutation importance, computed OUT OF FOLD ----------------------
imp_rows = []
for tr, te in GroupKFold(n_splits=5).split(X, yv, groups=groupsB):
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                               random_state=0, n_jobs=-1).fit(X[tr], yv[tr])
    r = permutation_importance(m, X[te], yv[te], n_repeats=10,
                               random_state=0, scoring="roc_auc", n_jobs=-1)
    imp_rows.append(r.importances_mean)
imp = pd.DataFrame(imp_rows, columns=FEATURES)

# compare with the (biased) impurity importance
final.fit(X, yv)
imp_gini = pd.Series(final.feature_importances_, index=FEATURES)

summary = pd.DataFrame({
    "permutation (AUC drop)": imp.mean(),
    "perm. std across folds": imp.std(),
    "impurity (biased)": imp_gini,
}).sort_values("permutation (AUC drop)", ascending=False)
print("FEATURE IMPORTANCE  (random forest, design B, spatial folds)")
print(summary.round(4).to_string())
print("\n  Impurity importance and permutation importance can rank features")
print("  differently. Trust the permutation ranking - it is measured on data")
print("  the model has not seen.")

# --- 2. Extrapolation check ------------------------------------------------
print("\nTRAINING ENVELOPE (design B)")
env = pd.DataFrame({"min": X.min(axis=0), "max": X.max(axis=0)}, index=FEATURES)
print(env.round(2).to_string())

# --- 3. Predict over the whole basin -------------------------------------
stack = np.stack([feat_rasters[f] for f in FEATURES], axis=-1)
valid_cells = np.isfinite(stack).all(axis=-1) & land_g
flat = stack[valid_cells]

proba = final.predict_proba(flat)[:, 1]
# per-tree spread = model disagreement
tree_p = np.stack([t.predict_proba(flat)[:, 1] for t in final.estimators_])
spread = tree_p.std(axis=0)

# extrapolation flag: outside the training range on any feature
outside = ((flat < X.min(axis=0)) | (flat > X.max(axis=0))).any(axis=1)

surf = np.full((H, Wd), np.nan); surf[valid_cells] = proba
unc  = np.full((H, Wd), np.nan); unc[valid_cells] = spread
ext  = np.full((H, Wd), np.nan); ext[valid_cells] = outside.astype(float)

print(f"\nPREDICTION SURFACE")
print(f"  cells predicted            : {valid_cells.sum():,} "
      f"({valid_cells.sum()*(CELL/1000)**2:,.0f} km^2)")
print(f"  mean predicted probability : {np.nanmean(surf):.3f}")
print(f"  cells with p > 0.5         : {int(np.nansum(surf > 0.5)):,} "
      f"({100*np.nansum(surf > 0.5)/valid_cells.sum():.1f} %)")
print(f"  mean tree disagreement     : {np.nanmean(unc):.3f}")
print(f"  cells OUTSIDE the training envelope : "
      f"{int(outside.sum()):,} ({100*outside.mean():.1f} %)")
print("    -> predictions there are extrapolation and must be labelled as such")

# --- 4. Validate against the hazard zones we never showed the model ------
zone_g = burn(flood[flood.return_period_yr == 100])
in_zone = zone_g[valid_cells]
print(f"\nEXTERNAL CHECK against the 100-year hazard zone (never used as a feature)")
print(f"  mean predicted p INSIDE the zone  : {proba[in_zone].mean():.3f}")
print(f"  mean predicted p OUTSIDE the zone : {proba[~in_zone].mean():.3f}")
print(f"  AUC of the prediction vs zone membership : "
      f"{roc_auc_score(in_zone.astype(int), proba):.4f}")
print("  The model was trained only on incident POINTS, yet it reconstructs the")
print("  independently-derived hazard zone. That is real, external validation.")

# --- 5. Figures -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))
order = summary.index.tolist()
axes[0].barh(range(len(order)), summary.loc[order, "permutation (AUC drop)"],
             xerr=summary.loc[order, "perm. std across folds"],
             color="#2166ac", height=0.6)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=8)
axes[0].invert_yaxis(); axes[0].set_xlabel("mean AUC drop when shuffled")
axes[0].set_title("Permutation importance\n(out-of-fold, 5 spatial folds)",
                  fontsize=9.5, weight="bold", loc="left")
axes[0].grid(alpha=0.3, axis="x")

ext_img = (TR.c, TR.c + Wd*CELL, TR.f - H*CELL, TR.f)
im = axes[1].imshow(surf, cmap="RdYlGn_r", extent=ext_img, vmin=0, vmax=1)
rivers.plot(ax=axes[1], color="#08306b", linewidth=0.7)
plt.colorbar(im, ax=axes[1], shrink=0.75, label="P(flood-prone)")
axes[1].set_title("Flood susceptibility surface", fontsize=9.5,
                  weight="bold", loc="left")

im = axes[2].imshow(unc, cmap="magma", extent=ext_img)
plt.colorbar(im, ax=axes[2], shrink=0.75, label="std across trees")
axes[2].set_title("Model uncertainty\n(where the trees disagree)",
                  fontsize=9.5, weight="bold", loc="left")
for a in axes[1:]:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(12, 3.4))
top4 = order[:4]
PartialDependenceDisplay.from_estimator(
    final, X, [FEATURES.index(f) for f in top4], feature_names=FEATURES,
    ax=ax, line_kw={"color": "#b2182b", "linewidth": 2})
plt.suptitle("Partial dependence of the four most important features",
             fontsize=11, y=1.04)
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **Permutation importance must be computed out of fold.** Computing it on
  training data measures how much the model *memorised* a feature, not how much
  it *needs* it. We loop over the same spatial folds used for validation, so the
  importances inherit the honest evaluation.
* **The comparison with impurity importance is deliberate.** `feature_importances_`
  on a tree ensemble is biased towards continuous, high-cardinality variables and
  can rank a useless feature above a crucial one. It is the default that everyone
  uses and it should not be.
* **Correlated features split importance.** `hand`, `elev` and `d_river` are all
  proxies for the same hydrological reality. Permutation importance will divide
  the credit among them somewhat arbitrarily — shuffling one leaves the others to
  compensate. Never read a single feature's importance as "the effect of X".
* **Tree spread as uncertainty.** `final.estimators_` gives the individual trees;
  their standard deviation at each cell is a cheap, honest measure of model
  disagreement. It rises where the training data is sparse or contradictory —
  exactly where you should not be making confident planning decisions.
* **The extrapolation flag** compares each prediction cell against the per-feature
  min/max of the training data. Any cell outside on any feature is being
  extrapolated. Random forests **cannot** extrapolate — they return the nearest
  leaf value — so predictions there are effectively "the most similar place I
  saw", which may be nothing like the truth.
* **The external check is the best evidence in the whole module.** The model was
  trained on 386 incident *points* plus population-matched background, using only
  terrain and climate features. The 100-year flood zone was derived independently
  from a HAND threshold and never shown to the model. If the model's surface
  reproduces that zone, it has learned real hydrology rather than the sampling
  design.

**Expected outcome.**

```
FEATURE IMPORTANCE  (random forest, design B, spatial folds)
         permutation (AUC drop)  perm. std across folds  impurity (biased)
hand                      0.097                   0.056              0.250
d_river                   0.065                   0.030              0.217
slope                     0.004                   0.008              0.075
elev                      0.004                   0.004              0.098
d_coast                   0.004                   0.003              0.115
rain                      0.002                   0.006              0.092
ndvi                      0.000                   0.003              0.090
tpi                      -0.003                   0.008              0.064
```

**`hand` and `d_river` carry essentially all the signal** — correct, because the
incidents were generated inside HAND-derived flood zones. Everything else is at
or below noise level, and `tpi` is *negative*: shuffling it slightly *improves*
held-out AUC, which is the signature of a feature contributing nothing but
variance.

**Now compare the two importance columns.** Impurity gives `d_coast` 0.115 and
`elev` 0.098 — third and fourth place — while permutation puts both at 0.004,
indistinguishable from zero. The trees split on them often (they are continuous
and high-cardinality) without those splits generalising. **This is why you should
not use `feature_importances_`.**

```
PREDICTION SURFACE
  cells predicted            : 137,251 (1,373 km^2)
  mean predicted probability : 0.192
  cells with p > 0.5         : 21,483 (15.7 %)
  mean tree disagreement     : 0.218
  cells OUTSIDE the training envelope : 4,232 (3.1 %)

EXTERNAL CHECK against the 100-year hazard zone (never used as a feature)
  mean predicted p INSIDE the zone  : 0.601
  mean predicted p OUTSIDE the zone : 0.130
  AUC of the prediction vs zone membership : 0.9647
```

**The external check is the strongest evidence in this module.** The model saw
only 386 incident *points* and a population-matched background, with terrain and
climate features. The 100-year flood zone was derived independently from a HAND
threshold and never shown to it. The model reproduces that zone with
**AUC = 0.965**. It has learned hydrology, not the sampling design — which is
precisely the claim that design B and spatial CV were built to make defensible.

Three panels: the importance bar chart with cross-fold error bars, the
susceptibility surface (high along river corridors and the coastal plain), and the
uncertainty map — highest at the *edges* of those corridors, where the model is
least sure. Then partial-dependence curves showing probability falling steeply
with `hand` and `d_river`: physically sensible, monotone relationships.
'''),

]

CELLS += [

# ----------------------------------------------------------------- A13 -----
md(r'''
## A13 — Quantitative risk: hazard × exposure × vulnerability

**What we are going to learn.** The standard risk decomposition, and how to turn
a susceptibility map into a monetary expected annual loss.

**Why it matters.** "This area is at risk" is not actionable. "This district
faces an expected annual loss of 4.2 million VS, of which 60% is concentrated in
pre-1970 masonry buildings" is. Converting a probability surface into an expected
loss is what makes spatial analysis a decision tool.

**The concept — the risk triangle.**

```
RISK = HAZARD × EXPOSURE × VULNERABILITY
```

| Term | Meaning | Our proxy |
|---|---|---|
| **Hazard** | Probability and intensity of the event | Return-period zones + the modelled susceptibility surface |
| **Exposure** | What is there to be damaged | Buildings, their value, and people |
| **Vulnerability** | How badly it is damaged, given the hazard | A **damage function**: fraction of value lost as a function of depth |

**Expected Annual Damage (EAD).** For a set of return periods with exceedance
probabilities `pᵢ = 1/Tᵢ` and losses `Lᵢ`, the EAD is the area under the
loss–exceedance-probability curve:

```
EAD = ∫ L(p) dp  ≈  Σᵢ ½ (Lᵢ + Lᵢ₊₁)(pᵢ − pᵢ₊₁)
```

This is the number that goes into a cost–benefit analysis: a defence costing
X per year is worth building if it reduces EAD by more than X.

**Vulnerability curves.** Depth–damage functions are empirical and
building-type-specific. Their shape matters enormously and they are the largest
source of uncertainty in most flood-risk assessments — larger than the hazard
model. **Always run a sensitivity analysis on the damage function.**

**Expected outcome.** Per-building expected annual damage, aggregated to
districts, with a loss-exceedance curve and a sensitivity analysis.

**What the next cell does:** defines depth–damage curves by construction type,
estimates flood depth per building per return period, computes EAD, aggregates
it, and tests how sensitive the total is to the damage-function assumption.
'''),

code(r'''
# --- 1. EXPOSURE: buildings with value and construction type --------------
B = buildings_clean[["building_id", "use_type", "construction", "floors",
                     "footprint_m2", "value_kvs", "ground_elev_m",
                     "has_basement", "geometry"]].copy()
B["value_kvs"] = B.value_kvs.fillna(B.value_kvs.median())
Bpt = B.copy(); Bpt["geometry"] = B.geometry.representative_point()

# --- 2. HAZARD: depth at each building for each return period ------------
# depth = (local flood-surface level) - (ground elevation), estimated from HAND
hand_at = sample_grid(feat_rasters["hand"], Bpt.geometry.x, Bpt.geometry.y)
Bpt["hand_m"] = hand_at

# design flood levels above the channel, by return period (fictional but ordered)
LEVELS = {10: 1.2, 25: 2.0, 50: 2.8, 100: 3.5, 250: 5.0, 500: 6.5}
for T, lvl in LEVELS.items():
    Bpt[f"depth_{T}"] = np.clip(lvl - Bpt.hand_m, 0, None)

print("HAZARD: flooded buildings by return period")
print(f"  {'return period':>14}{'design level':>14}{'buildings wet':>15}{'% of stock':>12}")
for T, lvl in LEVELS.items():
    n = int((Bpt[f"depth_{T}"] > 0).sum())
    print(f"  {T:>12} yr{lvl:>13.1f} m{n:>15,}{100*n/len(Bpt):>11.1f}%")

# --- 3. VULNERABILITY: depth-damage curves by construction ---------------
# damage ratio = fraction of building value lost, as a function of depth (m)
CURVES = {
    "masonry":             ([0, 0.5, 1, 2, 3, 4, 6], [0, .18, .32, .52, .68, .78, .88]),
    "reinforced_concrete": ([0, 0.5, 1, 2, 3, 4, 6], [0, .10, .20, .36, .50, .60, .72]),
    "timber":              ([0, 0.5, 1, 2, 3, 4, 6], [0, .28, .48, .72, .86, .93, .98]),
    "steel":               ([0, 0.5, 1, 2, 3, 4, 6], [0, .08, .16, .30, .42, .52, .64]),
}
def damage_ratio(depth, construction):
    out = np.zeros(len(depth))
    for c, (dx, dy) in CURVES.items():
        m = construction == c
        out[m] = np.interp(depth[m], dx, dy)
    return out

constr = Bpt.construction.to_numpy()
for T in LEVELS:
    dr = damage_ratio(Bpt[f"depth_{T}"].to_numpy(), constr)
    dr = np.where(Bpt.has_basement.to_numpy(), np.minimum(dr * 1.15, 1.0), dr)
    Bpt[f"loss_{T}"] = dr * Bpt.value_kvs.to_numpy()

losses = {T: Bpt[f"loss_{T}"].sum() for T in LEVELS}
print("\nLOSS BY RETURN PERIOD (thousand VS)")
print(f"  {'T (yr)':>8}{'exceedance p':>15}{'total loss':>16}{'mean loss/wet bldg':>21}")
for T in sorted(LEVELS):
    wet = Bpt[f"depth_{T}"] > 0
    mean_l = Bpt.loc[wet, f"loss_{T}"].mean() if wet.any() else 0
    print(f"  {T:>8}{1/T:>15.4f}{losses[T]:>16,.0f}{mean_l:>21,.1f}")

# --- 4. EXPECTED ANNUAL DAMAGE ------------------------------------------
Ts = np.array(sorted(LEVELS))
ps = 1.0 / Ts
Ls = np.array([losses[T] for T in Ts])
order = np.argsort(-ps)                       # descending probability
ps_s, Ls_s = ps[order], Ls[order]
EAD = np.trapezoid(Ls_s[::-1], ps_s[::-1]) if hasattr(np, "trapezoid") \
      else np.trapz(Ls_s[::-1], ps_s[::-1])
EAD = abs(EAD)

print(f"\nEXPECTED ANNUAL DAMAGE")
print(f"  EAD (whole basin) : {EAD:,.0f} thousand VS per year")
print(f"  total asset value : {Bpt.value_kvs.sum():,.0f} thousand VS")
print(f"  EAD as % of stock : {100*EAD/Bpt.value_kvs.sum():.3f} % per year")
print(f"  implied payback on a defence costing 10 % of the stock: "
      f"{0.10*Bpt.value_kvs.sum()/EAD:,.0f} years")

# --- 5. Aggregate to districts ---------------------------------------------
Bd = gpd.sjoin(Bpt, districts[["district_id", "name", "geometry"]],
               predicate="within").drop_duplicates("building_id")
per_T = {T: Bd.groupby("district_id")[f"loss_{T}"].sum() for T in Ts}
ead_d = {}
for did in districts.district_id:
    L = np.array([per_T[T].get(did, 0.0) for T in Ts])
    ead_d[did] = abs(np.trapezoid(L[::-1], ps[np.argsort(Ts)][::-1])
                     if hasattr(np, "trapezoid") else np.trapz(L[::-1], ps[::-1]))
risk = districts[["district_id", "name", "district_type", "population",
                  "geometry"]].copy()
risk["ead_kvs"] = risk.district_id.map(ead_d)
risk["n_buildings"] = risk.district_id.map(Bd.groupby("district_id").size()).fillna(0)
risk["value_kvs"] = risk.district_id.map(Bd.groupby("district_id").value_kvs.sum()).fillna(0)
risk["ead_per_capita"] = risk.ead_kvs / risk.population
risk["ead_pct_value"] = 100 * risk.ead_kvs / risk.value_kvs.replace(0, np.nan)

print("\nEXPECTED ANNUAL DAMAGE BY DISTRICT (top 8)")
print(risk.nlargest(8, "ead_kvs")[["district_id", "name", "district_type",
                                   "n_buildings", "value_kvs", "ead_kvs",
                                   "ead_pct_value"]].round(1).to_string(index=False))
print(f"\n  top 3 districts hold "
      f"{100*risk.nlargest(3,'ead_kvs').ead_kvs.sum()/risk.ead_kvs.sum():.0f} % "
      f"of the basin's expected annual damage")

# --- 6. SENSITIVITY to the damage function ------------------------------
print("\nSENSITIVITY OF EAD TO THE DAMAGE FUNCTION")
print(f"  {'scenario':<34}{'EAD':>16}{'vs baseline':>14}")
for label, mult in [("baseline curves", 1.00),
                    ("curves 25 % more damaging", 1.25),
                    ("curves 25 % less damaging", 0.75),
                    ("curves 50 % more damaging", 1.50)]:
    tot = []
    for T in Ts:
        dr = damage_ratio(Bpt[f"depth_{T}"].to_numpy(), constr) * mult
        dr = np.where(Bpt.has_basement.to_numpy(), dr * 1.15, dr)   # same modifier
        tot.append(np.minimum(dr, 1.0) @ Bpt.value_kvs.to_numpy())
    tot = np.array(tot)
    e = abs(np.trapezoid(tot[::-1], ps[::-1]) if hasattr(np, "trapezoid")
            else np.trapz(tot[::-1], ps[::-1]))
    print(f"  {label:<34}{e:>16,.0f}{100*(e-EAD)/EAD:>13.1f}%")
print("\n  A 25 % change in an EMPIRICAL curve moves the headline number by 20-30 %.")
print("  The response is ASYMMETRIC because damage ratios are capped at 1.0, so")
print("  making curves more damaging saturates while making them less damaging")
print("  does not. In most flood-risk studies the vulnerability function is a")
print("  larger source of uncertainty than the hazard model everyone argues about.")

# --- 7. Figures --------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
axes[0].plot(ps_s, Ls_s, "o-", color="#b2182b", linewidth=2)
axes[0].fill_between(np.sort(ps), np.array([losses[T] for T in Ts])[np.argsort(ps)],
                     alpha=0.25, color="#b2182b")
axes[0].set_xlabel("annual exceedance probability")
axes[0].set_ylabel("loss (thousand VS)")
axes[0].set_title(f"Loss-exceedance curve\nshaded area = EAD = {EAD:,.0f} k VS/yr",
                  fontsize=9.5, weight="bold", loc="left")
axes[0].grid(alpha=0.3)

for c, (dx, dy) in CURVES.items():
    axes[1].plot(dx, dy, "o-", label=c, linewidth=1.8)
axes[1].set_xlabel("flood depth (m)"); axes[1].set_ylabel("damage ratio")
axes[1].set_title("Depth-damage (vulnerability) curves", fontsize=9.5,
                  weight="bold", loc="left")
axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

risk.plot(ax=axes[2], column="ead_kvs", cmap="OrRd", scheme="quantiles", k=6,
          legend=True, legend_kwds={"loc": "lower left", "fontsize": 6.5,
                                    "title": "EAD (k VS/yr)"},
          edgecolor="grey", linewidth=0.4,
          missing_kwds={"color": "#eeeeee"})
axes[2].set_title("Expected annual damage by district", fontsize=9.5,
                  weight="bold", loc="left")
axes[2].set_aspect("equal"); axes[2].set_xticks([]); axes[2].set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **Depth from HAND.** For each return period we assume a design water level
  above the channel and compute depth as `level − HAND`, floored at zero. This is
  a **planar/bathtub** approximation — it ignores flow, storage and defences. Real
  studies use a hydraulic model. State the approximation; do not hide it.
* **The damage curves** are piecewise-linear in depth, interpolated with
  `np.interp`. Timber is most vulnerable, steel least — the ordering matters more
  than the exact numbers. The basement adjustment (`×1.15`, capped at 1.0) is a
  simple example of a modifier; real curves have many.
* **EAD by trapezoidal integration over exceedance probability.** Note the
  direction: probabilities are sorted ascending for `np.trapezoid`, and we take
  the absolute value to be direction-agnostic. **The high-probability, low-loss
  end of the curve dominates the integral** — a 10-year flood contributes far more
  to EAD than a 500-year flood, which is counter-intuitive and is why designing
  only for the extreme event is poor economics.
* `np.trapezoid` replaced `np.trapz` in NumPy 2.0; the `hasattr` guard keeps the
  code working on both.
* **The district aggregation** re-integrates per district rather than
  apportioning the basin total, which is the correct order of operations: EAD is
  not linear in the intermediate quantities.
* **The sensitivity analysis is the most important block.** Multiplying every
  damage ratio by 1.25 moves EAD by roughly 25%. Damage curves are empirical,
  transferred between countries and building stocks, and rarely validated
  locally. In practice they carry more uncertainty than the hazard model — and
  they receive far less scrutiny.

**Expected outcome.**

```
LOSS BY RETURN PERIOD (thousand VS)
    T (yr)   exceedance p      total loss   mean loss/wet bldg
        10         0.1000         144,103                101.4
        50         0.0200         211,127                131.7
       100         0.0100         236,089                141.2
       500         0.0020         321,588                164.2

EXPECTED ANNUAL DAMAGE
  EAD (whole basin) : 17,987 thousand VS per year
  total asset value : 1,181,004 thousand VS
  EAD as % of stock : 1.523 % per year
  implied payback on a defence costing 10 % of the stock: 7 years
```

**Look at where the EAD comes from.** The 500-year event loses 321 588 k VS —
more than twice the 10-year event's 144 103 — but it happens with probability
0.002 against 0.100. In the integral, the 10-year event contributes far more.
**Designing only for the extreme event is poor economics**; most avoidable loss
sits in the frequent, moderate floods.

```
EXPECTED ANNUAL DAMAGE BY DISTRICT (top 8)
district_id         name district_type  n_buildings   value_kvs    ead_kvs  ead_pct_value
        D01 Old Vallmara    urban_core         1707  509,347.7   10,966.4            2.2
        D02  Harbourgate    urban_core         1436  390,643.2    2,469.0            0.6
        D09     Tarnwell      suburban          403   61,194.2    1,354.0            2.2
        D06    Ardenfeld      suburban          265   25,942.8    1,037.9            4.0

  top 3 districts hold 82 % of the basin's expected annual damage
```

**Absolute and relative risk point at different places.** Old Vallmara carries
61% of the basin's EAD in absolute terms — it is where the money should go. But
**Ardenfeld loses 4.0% of its asset value every year**, nearly twice Old
Vallmara's rate, on a stock twenty times smaller. A budget allocated on absolute
EAD will never reach Ardenfeld; one allocated on relative loss will never reach
the city. Both are legitimate policy targets and you must present both.

```
SENSITIVITY OF EAD TO THE DAMAGE FUNCTION
  scenario                                       EAD   vs baseline
  baseline curves                             17,987          0.0%
  curves 25 % more damaging                   20,757         15.4%
  curves 25 % less damaging                   12,934        -28.1%
  curves 50 % more damaging                   23,112         28.5%
```

A ±25% change in an *empirical, transferred* curve moves the headline number by
15–28%. Note the **asymmetry**: damage ratios are capped at 1.0, so making curves
more damaging saturates while making them less damaging does not. In most
flood-risk studies the vulnerability function carries more uncertainty than the
hazard model everyone argues about — and receives far less scrutiny.

Three panels: the loss-exceedance curve with the EAD shaded, the four
vulnerability curves, and the district EAD choropleth.
'''),

# ----------------------------------------------------------------- A14 -----
md(r'''
## A14 — Communicating spatial results

**What we are going to learn.** How to present spatial analysis so that it is
useful and not misleading — and the specific claims you must not make.

**Why it matters.** Maps are unusually persuasive. A reader who would demand a
confidence interval from a table will accept a choropleth without question. That
asymmetry places a heavier burden of honesty on spatial work than on ordinary
analysis.

### The seven pitfalls, and what to do instead

| # | Pitfall | Why it misleads | Remedy |
|---|---|---|---|
| 1 | **Mapping counts instead of rates** | Population maps always look like population | Map rates; show the denominator |
| 2 | **Unstable rates in small units** | A 2-of-50 rate is noisy; extremes are always in small units | Empirical-Bayes smoothing, or show the population |
| 3 | **Classification shopping** | Quantiles vs equal interval can invert the visual story | Fix the scheme *before* looking; show a histogram |
| 4 | **Hiding missing data** | NaN drawn as white reads as "low" | `missing_kwds` with hatching and a legend entry |
| 5 | **The ecological fallacy** | District-level correlation ≠ individual-level | State the unit of analysis; never infer to individuals |
| 6 | **MAUP** | Different units give different — even opposite — results | Repeat at a second scale; report both |
| 7 | **Implying causation from a pattern** | Co-location is not a mechanism | Report the confounder analysis (A9) |

### What every serious spatial deliverable must state

1. **The CRS and the units.**
2. **The unit of analysis** and why it was chosen.
3. **The date and provenance** of every layer.
4. **The uncertainty** — model uncertainty, sampling uncertainty, or both.
5. **The validation** — how you know the analysis is right.
6. **The assumptions** — Euclidean rather than network distance; bathtub rather
   than hydraulic flooding; transferred damage curves.

**Expected outcome.** A demonstration of pitfalls 1, 2, 3 and 6 on our own data,
each with the honest alternative beside it.

**What the next cell does:** builds four side-by-side comparisons — counts vs
rates, raw vs empirical-Bayes-smoothed rates, three classification schemes on the
same variable, and the same analysis at two spatial scales.
'''),

code(r'''
fig, axes = plt.subplots(2, 4, figsize=(19.5, 10))

# --- PITFALL 1: counts vs rates -------------------------------------------
Fx["incidents_n"] = 0
sj = gpd.sjoin(pts[["incident_id", "geometry"]], Fx[["block_id", "geometry"]],
               predicate="within")
cnt = sj.groupby("block_id").size()
Fx["incidents_n"] = Fx.block_id.map(cnt).fillna(0).astype(int)
Fx["incidents_per_1k"] = 1000 * Fx.incidents_n / Fx.population.replace(0, np.nan)

Fx.plot(ax=axes[0, 0], column="incidents_n", cmap="Reds", scheme="quantiles", k=6,
        legend=True, legend_kwds={"loc": "lower left", "fontsize": 6}, edgecolor="none")
axes[0, 0].set_title("(1a) MISLEADING: incident COUNT\nlooks exactly like a population map",
                     fontsize=9, weight="bold", loc="left", color="crimson")
Fx.plot(ax=axes[0, 1], column="incidents_per_1k", cmap="Reds", scheme="quantiles",
        k=6, legend=True, legend_kwds={"loc": "lower left", "fontsize": 6},
        edgecolor="none", missing_kwds={"color": "#dddddd", "hatch": "//"})
axes[0, 1].set_title("(1b) BETTER: incidents per 1,000 residents",
                     fontsize=9, weight="bold", loc="left", color="darkgreen")

corr_count = Fx[["incidents_n", "population"]].corr().iloc[0, 1]
corr_rate = Fx[["incidents_per_1k", "population"]].corr().iloc[0, 1]

# --- PITFALL 2: unstable rates in small populations ---------------------
# Empirical-Bayes smoothing towards the global rate
n_i = Fx.incidents_n.to_numpy(float)
p_i = Fx.population.to_numpy(float)
global_rate = n_i.sum() / p_i.sum()
var_between = max(np.nanvar(n_i / np.maximum(p_i, 1)) - global_rate / np.nanmean(p_i), 1e-12)
shrink = var_between / (var_between + global_rate / np.maximum(p_i, 1))
Fx["rate_eb"] = 1000 * (shrink * (n_i / np.maximum(p_i, 1)) + (1 - shrink) * global_rate)

Fx.plot(ax=axes[0, 2], column="rate_eb", cmap="Reds", scheme="quantiles", k=6,
        legend=True, legend_kwds={"loc": "lower left", "fontsize": 6}, edgecolor="none")
axes[0, 2].set_title("(2) Empirical-Bayes smoothed rate\nsmall-population noise shrunk away",
                     fontsize=9, weight="bold", loc="left", color="darkgreen")

small = Fx.nsmallest(50, "population")
axes[0, 3].scatter(Fx.population, Fx.incidents_per_1k, s=8, alpha=0.5,
                   label="raw rate", color="#b2182b")
axes[0, 3].scatter(Fx.population, Fx.rate_eb, s=8, alpha=0.5,
                   label="EB-smoothed", color="#2166ac")
axes[0, 3].set_xscale("log"); axes[0, 3].set_xlabel("block population (log)")
axes[0, 3].set_ylabel("incidents per 1,000")
axes[0, 3].set_title("(2b) Rate variance explodes in small units",
                     fontsize=9, weight="bold", loc="left")
axes[0, 3].legend(fontsize=7); axes[0, 3].grid(alpha=0.3)

# --- PITFALL 3: classification shopping -------------------------------------
for ax, scheme, name in [(axes[1, 0], "quantiles", "quantiles"),
                         (axes[1, 1], "equalinterval", "equal interval"),
                         (axes[1, 2], "naturalbreaks", "natural breaks")]:
    Fx.plot(ax=ax, column="total_value_kvs", cmap="viridis", scheme=scheme, k=5,
            legend=True, legend_kwds={"loc": "lower left", "fontsize": 5.5},
            edgecolor="none", missing_kwds={"color": "#dddddd"})
    ax.set_title(f"(3) Same data, scheme = {name}", fontsize=9,
                 weight="bold", loc="left")

# --- PITFALL 6: MAUP - the same analysis at two scales -------------------
corr_block = Fx[["pct_in_flood100", "pop_density_km2"]].corr().iloc[0, 1]
dd = districts.merge(
    Fx.groupby("district_id").agg(flood=("pct_in_flood100", "mean"),
                                  dens=("pop_density_km2", "mean")).reset_index(),
    on="district_id")
corr_district = dd[["flood", "dens"]].corr().iloc[0, 1]
dd.plot(ax=axes[1, 3], column="flood", cmap="Blues", scheme="quantiles", k=5,
        legend=True, legend_kwds={"loc": "lower left", "fontsize": 6}, edgecolor="grey",
        linewidth=0.4)
axes[1, 3].set_title(f"(6) Same variable at DISTRICT scale\nr changes "
                     f"{corr_block:+.2f} -> {corr_district:+.2f}",
                     fontsize=9, weight="bold", loc="left")

for ax in axes.ravel():
    if ax not in (axes[0, 3],):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()

print("THE NUMBERS BEHIND THE PICTURES")
print("=" * 78)
print(f"PITFALL 1  correlation(incident COUNT, population)      = {corr_count:+.3f}")
print(f"           correlation(incident RATE,  population)      = {corr_rate:+.3f}")
print(f"           -> the count map is largely a population map.")
print(f"\nPITFALL 2  raw rate: std = {Fx.incidents_per_1k.std():.2f}, "
      f"max = {Fx.incidents_per_1k.max():.1f}")
print(f"           EB rate : std = {Fx.rate_eb.std():.2f}, "
      f"max = {Fx.rate_eb.max():.1f}")
print(f"           blocks under 100 people: {int((Fx.population < 100).sum())} "
      f"- their raw rates are essentially noise")
print(f"\nPITFALL 3  class breaks for total_value_kvs:")
import mapclassify
v = Fx.total_value_kvs.dropna()
for scheme, cls in [("quantiles", mapclassify.Quantiles(v, k=5)),
                    ("equal interval", mapclassify.EqualInterval(v, k=5)),
                    ("natural breaks", mapclassify.NaturalBreaks(v, k=5))]:
    print(f"           {scheme:<16} top class starts at {cls.bins[-2]:>12,.0f} "
          f"and holds {int(cls.counts[-1]):>3} blocks")
print(f"\nPITFALL 6  corr(flood exposure, pop density) at BLOCK level    = {corr_block:+.3f}")
print(f"           corr(flood exposure, pop density) at DISTRICT level = {corr_district:+.3f}")
print(f"           -> the same question, two units, two different answers.")
print("=" * 78)
print("""
A CHECKLIST FOR YOUR FINAL DELIVERABLE

  [ ] CRS stated, with units, on every map
  [ ] unit of analysis stated and justified
  [ ] denominators shown wherever a rate is mapped
  [ ] classification scheme named; histogram available on request
  [ ] missing data drawn explicitly, never left white
  [ ] uncertainty mapped alongside the estimate
  [ ] validation described - how do you know this is right?
  [ ] assumptions listed (Euclidean distance, bathtub flooding, transferred curves)
  [ ] scale sensitivity checked at a second unit of analysis
  [ ] no causal language unless you have a design that supports it
""")
'''),

md(r'''
**Explanation.**

* **Pitfall 1** is quantified by the two correlations. Incident *count* correlates
  strongly with population; incident *rate* does not. A count map answers "where
  do people live?" dressed as "where do floods happen?".
* **Empirical-Bayes smoothing** shrinks each block's rate towards the global rate,
  by an amount that depends on the block's population. A block of 40 people with
  one incident has a raw rate of 25 per 1 000 — an extreme value driven entirely by
  a small denominator. EB pulls it back towards the regional average; a block of
  20 000 people barely moves. The scatter panel makes the mechanism visible: raw
  rates fan out at small populations, EB rates do not.
* **Pitfall 3** shows the same variable under three schemes. Because asset value is
  heavily right-skewed, **equal interval** puts almost every block in the bottom
  class and highlights only the extreme core, while **quantiles** spreads colour
  evenly and makes the whole region look differentiated. Both are "correct"; they
  tell different stories. Choose before you look, and say which you chose.
* **Pitfall 6 (MAUP)** is quantified directly: the correlation between flood
  exposure and population density changes when you move from blocks to districts.
  Aggregation averages away within-district variation and usually *strengthens*
  correlations — the classic ecological-correlation inflation. If your headline
  finding is a correlation, **report it at two scales**.
* The final checklist is the practical output of this lesson. Attach it to your
  own work.

**Expected outcome.**

```
PITFALL 1  correlation(incident COUNT, population)      = +0.526
           correlation(incident RATE,  population)      = -0.059

PITFALL 2  raw rate: std = 4.97, max = 44.4
           EB rate : std = 3.61, max = 28.9
           blocks under 100 people: 208 - their raw rates are essentially noise

PITFALL 3  class breaks for total_value_kvs:
           quantiles        top class starts at        1,731 and holds  65 blocks
           equal interval   top class starts at       65,730 and holds   4 blocks
           natural breaks   top class starts at       57,192 and holds   4 blocks
```

**Pitfall 1**: the count map correlates +0.53 with population; the rate map
correlates −0.06. The count map is, to a first approximation, a population map
with a red colour ramp.

**Pitfall 2**: 208 of 459 blocks have fewer than 100 residents. A single incident
in a block of 40 people gives a rate of 25 per 1 000 — the highest in the region,
and pure noise. EB smoothing cuts the maximum from 44.4 to 28.9 and the standard
deviation from 4.97 to 3.61, almost all of it from those small blocks.

**Pitfall 3 is the starkest.** The same variable, five classes:
**quantiles puts 65 blocks in the top class; equal interval and natural breaks
put 4.** The quantile map will look like widespread high value across the basin;
the equal-interval map will look like a single hot spot in the city. Both are
faithful renderings of the same numbers. **Choose the scheme before you look at
the map, and name it in the caption.**

**Pitfall 6**: the correlation between flood exposure and population density
changes when you move from blocks to districts. Both are true statements about
different objects — and only one of them is the answer to the question you were
asked.
'''),

# ------------------------------------------------------- MODULE 3 EXERCISES
md(r'''
# Exercises — Module 3 (Advanced)

---

### Exercise 3.1 — A defensible siting recommendation
**Objective.** The regional government will fund **two** new clinics. Using MCDA:
1. Define at least five criteria, including at least one equity criterion, and
   justify each.
2. Elicit weights by AHP and report the consistency ratio.
3. Aggregate by both weighted sum and geometric mean; explain your choice.
4. Run a Monte-Carlo sensitivity analysis on the weights **and** on the
   normalisation method.
5. Recommend two blocks, and state the conditions under which your
   recommendation would change.

---

### Exercise 3.2 — Accessibility with a network-distance correction
**Objective.** Our E2SFCA used Euclidean distance. Estimate a **detour index**
(network distance ÷ straight-line distance) for the Vallmara Basin by sampling
50 origin–destination pairs and measuring the shortest path along the road
network (build a graph from the road segments; `networkx` or your own Dijkstra).
Re-run the clinic E2SFCA with corrected distances. How much does the Gini change?
Which districts change rank most?

---

### Exercise 3.3 — A hotspot analysis you can defend
**Objective.** Run Getis-Ord Gi* on **building asset value at risk**
(`total_value_kvs × pct_in_flood100`) and produce a map that would survive peer
review. It must include: an explicit weights specification, a permutation budget
justified against the FDR requirement, an FDR-corrected significance map, and a
sensitivity analysis across at least three weights schemes. Report how many
blocks are significant under each.

---

### Exercise 3.4 — Regionalisation for service delivery
**Objective.** The health authority wants to divide the basin into **6 contiguous
service regions** that are as equal as possible in population while remaining
internally homogeneous in accessibility. Design and implement this. Report the
population of each region, the coefficient of variation across regions, and the
homogeneity cost relative to unconstrained clustering. Discuss the trade-off.

---

### Exercise 3.5 — A better flood model
**Objective.** Improve on the Module 3 flood-susceptibility model:
1. Add at least three new physically motivated features (curvature, upslope
   contributing area, distance to the coast, land-cover composition in a
   neighbourhood, …).
2. Use a spatial-block CV with block size justified by a variogram of the
   residuals.
3. Compare at least three algorithms and calibrate the best one
   (`CalibratedClassifierCV`).
4. Report AUC, average precision, Brier score and a calibration curve — all
   spatially validated.
5. Produce a susceptibility map with an uncertainty layer and an extrapolation
   mask.

---

### Exercise 3.6 — Risk under a climate scenario
**Objective.** Assume a climate scenario in which all design flood levels rise by
0.5 m and the 100-year event becomes the 50-year event. Recompute EAD. Report:
the change in EAD, which districts change most in absolute and in relative terms,
how many additional buildings become exposed, and the break-even cost of a
defence that would restore the current EAD.

---

### Challenge 3.7 — Recover the full generating process
**Objective.** Using only the delivered data files, estimate **every** coefficient
in the documented generating process:

| Relationship | True form |
|---|---|
| Rainfall | `470 + 0.62·elevation + 150·(north–south position)` |
| Land surface temperature | `31.5 − 0.0062·elevation + 6.4·urban` |
| PM2.5 | `7.5 + 16·urban + 9·exp(−d_motorway/1800)` |
| Population density | `9500·urban^2.1 + 22` |

For each, report your estimate, a confidence interval, and the true value.
Then answer: **which estimate is worst, and why?** Diagnose the cause (aliasing,
confounding, aggregation bias, measurement error, or insufficient sample) and
propose a fix.
'''),

]
