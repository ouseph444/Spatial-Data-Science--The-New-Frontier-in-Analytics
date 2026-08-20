# -*- coding: utf-8 -*-
"""Module 4 - Capstone project, plus the Solutions section."""
from _cells import md, code

CELLS = [

md(r'''
# Module 4 — Capstone Project

## The Vallmara Basin Integrated Climate-Resilience Assessment

---

### The brief

> **From:** Office of the Regional Director, Vallmara Basin Authority
> **To:** Spatial Data Science Unit
> **Re:** Integrated climate-resilience assessment — decision support for the
> 2026–2031 capital programme
>
> The Authority has a capital budget of **120 million VS** over five years. It
> must be split between flood defence, health-service expansion and building
> retrofit. Council members disagree about where the money should go, and every
> district claims to be the worst affected.
>
> We need an evidence base. Specifically:
>
> 1. A **Composite Resilience Index** for every district, combining physical
>    hazard, service accessibility and socio-economic vulnerability.
> 2. A **prioritised intervention list** — which districts, which intervention,
>    what it would cost and what it would avert.
> 3. An honest statement of **what your analysis cannot tell us**.
>
> The report will be scrutinised by people who want a different answer. Every
> number must be traceable and every assumption stated.

---

### What you must produce

| Stage | Requirement |
|---|---|
| **1. Load and inspect** | Every layer loaded, inventoried, and its provenance recorded |
| **2. Clean the attributes** | Sentinels, impossibles, duplicates, inconsistent categories — with an audit trail |
| **3. Validate geometries** | Invalid, empty and null geometries found and repaired; area reconciliation before/after |
| **4. Resolve CRS** | One analysis CRS, justified by measurement, with the error of the alternatives quantified |
| **5. Vector analysis** | Exposure of population, buildings and assets to each hazard zone, without double counting |
| **6. Raster analysis** | Terrain derivatives, a HAND surface, and zonal statistics validated against a known quantity |
| **7. Raster–vector integration** | An aligned multi-layer stack, and every raster variable attached to every district |
| **8. Feature engineering** | At least 20 features across proximity, density, composition, focal and lag families |
| **9. Statistical / spatial analysis** | Autocorrelation testing, hotspot detection with multiple-testing control, and a spatially validated predictive model |
| **10. Visualisation** | A publication-quality map series with explicit classification, missing data and uncertainty |
| **11. Interpretation** | What the numbers mean, for whom, with what confidence |
| **12. Conclusion** | A ranked intervention list with costed recommendations and stated limitations |

---

### Assessment rubric

Mark yourself honestly. You have demonstrated practical proficiency if you can
answer **yes** to all of these:

- [ ] Every measurement is in a CRS whose units make it meaningful, and I can say why.
- [ ] No spatial join in my pipeline silently changed a row count.
- [ ] Every absolute quantity was recomputed after any geometry-splitting operation.
- [ ] Missing data is NaN, is visible on my maps, and is counted in my report.
- [ ] I tested residual spatial autocorrelation and acted on the result.
- [ ] My predictive model was validated with spatially blocked folds, and I report the honest score, not the flattering one.
- [ ] I ran a sensitivity analysis on every subjective weight and threshold.
- [ ] I have at least one **external** validation — a check against something the analysis never used.
- [ ] I can state, in one sentence each, the three assumptions most likely to be wrong.
- [ ] Someone else could reproduce my numbers from my code and the raw data.

---

### How to use what follows

**Attempt the capstone yourself first.** The reference implementation below is
one defensible solution, not the only one. Where it makes a judgement call it
says so, and where a different call would be equally defensible it says that too.

The reference implementation runs in about a minute and reuses the helper
functions built in Modules 2 and 3 (`zonal_stats`, `align_to`, `moran_test`,
`getis_ord_gstar`, `benjamini_hochberg`, `knn_weights`, `queen_weights`,
`weighted_gini`, `burn`, `sample_grid`).
'''),

md(r'''
## Capstone Stage 1–4 — Ingest, clean, validate, resolve CRS

**What we are about to do.** Rebuild the entire data pipeline from the raw files
in one auditable block, so the capstone stands alone and every number in it is
traceable to a documented step.

**Why it matters.** The single most common reason an analysis cannot be defended
is that nobody can say exactly what was done to the data. A pipeline that logs
every decision is worth more than a cleverer model.

**Concept — provenance.** For each layer, record: source file, original CRS, row
count in, row count out, and every transformation applied. This block is the
skeleton of the "Data and Methods" section of your report.

**What the next cell does:** loads every layer from disk, records provenance,
applies the cleaning rules from I3, repairs geometry as in I2, reprojects
everything to the analysis CRS, and prints a complete audit trail with an area
reconciliation.
'''),

code(r'''
# =========================================================================
# CAPSTONE STAGES 1-4: INGEST -> CLEAN -> VALIDATE -> CRS
# =========================================================================
PROV = []          # provenance log

def logp(layer, source, crs_in, n_in, n_out, notes):
    PROV.append(dict(layer=layer, source=source, crs_in=crs_in,
                     rows_in=n_in, rows_out=n_out, notes=notes))

CAP = {}           # the clean, canonical layer set

# ---- STAGE 1: LOAD -------------------------------------------------------
raw_specs = [
    ("districts",     GPKG, "districts"), ("blocks", GPKG, "census_blocks"),
    ("landuse",       GPKG, "landuse"),   ("rivers", GPKG, "rivers"),
    ("flood",         GPKG, "flood_zones"), ("buildings", GPKG, "buildings"),
    ("facilities",    GPKG, "facilities"), ("stops", GPKG, "bus_stops"),
    ("routes",        GPKG, "transit_routes"),
    ("land",          GPKG, "land_boundary"), ("sea", GPKG, "sea"),
    ("coastline",     GPKG, "coastline"),
    ("roads",         VEC / "roads.geojson", None),
    ("protected",     VEC / "protected_areas.geojson", None),
]
for name, src, layer in raw_specs:
    g = gpd.read_file(src, layer=layer) if layer else gpd.read_file(src)
    crs_in = g.crs.to_string()
    n_in = len(g)
    # ---- STAGE 4: single analysis CRS, applied at the door ----------------
    if g.crs.to_string() != CRS_UTM:
        g = g.to_crs(CRS_UTM)
    CAP[name] = g
    logp(name, str(src.name if hasattr(src, "name") else src), crs_in,
         n_in, len(g), "reprojected" if crs_in != CRS_UTM else "as-is")

# ---- STAGE 3: VALIDATE AND REPAIR GEOMETRY -------------------------------
print("STAGE 3 - GEOMETRY VALIDATION")
print(f"  {'layer':<12}{'null':>6}{'empty':>7}{'invalid':>9}{'dup geom':>10}"
      f"{'area/len before':>18}{'after':>14}")
print("-" * 78)
for name, g in list(CAP.items()):
    n_null = int(g.geometry.isna().sum())
    n_empty = int(g.geometry.is_empty.sum())
    n_bad = int((~g.geometry.is_valid).sum())
    n_dup = int(g.geometry.to_wkb().duplicated().sum())
    is_poly = g.geom_type.iloc[0] in ("Polygon", "MultiPolygon")
    before = g.geometry.area.sum() / 1e6 if is_poly else g.geometry.length.sum() / 1000

    if n_bad:
        g = g.copy()
        g["geometry"] = g.geometry.make_valid()
        g = g.explode(index_parts=False, ignore_index=True)
        keep = ["Polygon", "MultiPolygon"] if is_poly else ["LineString", "MultiLineString"]
        g = g[g.geom_type.isin(keep)]
    g = g[~g.geometry.isna() & ~g.geometry.is_empty].reset_index(drop=True)
    after = g.geometry.area.sum() / 1e6 if is_poly else g.geometry.length.sum() / 1000
    CAP[name] = g
    if n_null or n_empty or n_bad or n_dup:
        print(f"  {name:<12}{n_null:>6}{n_empty:>7}{n_bad:>9}{n_dup:>10}"
              f"{before:>18,.3f}{after:>14,.3f}")
print("  (layers with no problems are omitted)")

# ---- STAGE 2: CLEAN ATTRIBUTES -------------------------------------------
print("\nSTAGE 2 - ATTRIBUTE CLEANING")
SENTINELS = [-999, -9999]

# facilities: sentinel capacity + duplicate features
f = CAP["facilities"]
n0 = len(f)
f = f.assign(_w=f.geometry.to_wkb()).drop_duplicates(["facility_id", "_w"]).drop(columns="_w")
n_sent = int(f.capacity.isin(SENTINELS).sum())
f.loc[f.capacity.isin(SENTINELS), "capacity"] = np.nan
CAP["facilities"] = f.reset_index(drop=True)
print(f"  facilities : {n0 - len(f)} duplicate features dropped, "
      f"{n_sent} sentinel capacities -> NaN")

# buildings: impossible construction years
b = CAP["buildings"]
bad_yr = (b.year_built < 1800) | (b.year_built > 2025)
b.loc[bad_yr, "year_built"] = np.nan
b["building_age"] = 2025 - b.year_built
b["value_kvs_filled"] = b.value_kvs.fillna(b.value_kvs.median())
CAP["buildings"] = b
print(f"  buildings  : {int(bad_yr.sum())} impossible years -> NaN, "
      f"{int(b.value_kvs.isna().sum())} missing values median-filled (flagged)")

# landuse: controlled vocabulary
lu = CAP["landuse"]
CONTROLLED = {c.lower(): c for c in lc_legend.landuse_class}
n_lv = lu.landuse_class.nunique()
lu["landuse_class"] = (lu.landuse_class.str.strip().str.lower()
                         .str.replace(r"\s+", " ", regex=True).map(CONTROLLED))
CAP["landuse"] = lu
print(f"  landuse    : {n_lv} raw labels -> {lu.landuse_class.nunique()} controlled, "
      f"{int(lu.landuse_class.isna().sum())} unmapped")

# socio-economic: duplicate + orphan keys
so = pd.read_csv(TAB / "district_socioeconomic.csv")
n0 = len(so)
so = so.drop_duplicates("district_id", keep="first")
valid_ids = set(CAP["districts"].district_id)
orphans = sorted(set(so.district_id) - valid_ids)
so = so[so.district_id.isin(valid_ids)].reset_index(drop=True)
CAP_SOCIO = so
print(f"  socio      : {n0} -> {len(so)} rows "
      f"(orphan keys removed: {orphans})")

# incidents: coordinate quarantine + recovery
ir = pd.read_csv(TAB / "flood_incidents.csv", parse_dates=["date"])
BOX = dict(lon=(13.6, 14.5), lat=(41.4, 42.0))
inbox = lambda lo, la: lo.between(*BOX["lon"]) & la.between(*BOX["lat"])
ir["reject"] = pd.NA
ir.loc[(ir.lon == 0) & (ir.lat == 0), "reject"] = "null_island"
sw = (~inbox(ir.lon, ir.lat)) & inbox(ir.lat, ir.lon) & ir.reject.isna()
ir.loc[sw, "reject"] = "swapped"
ir.loc[(~inbox(ir.lon, ir.lat)) & ir.reject.isna(), "reject"] = "outside"
rec = ir[ir.reject == "swapped"].copy()
rec[["lon", "lat"]] = rec[["lat", "lon"]].to_numpy()
clean = pd.concat([ir[ir.reject.isna()], rec], ignore_index=True).drop(columns="reject")
clean.loc[clean.damage_kvs < 0, "damage_kvs"] = np.nan
CAP["incidents"] = gpd.GeoDataFrame(
    clean, geometry=gpd.points_from_xy(clean.lon, clean.lat),
    crs=CRS_WGS84).to_crs(CRS_UTM)
print(f"  incidents  : {len(ir)} raw -> {len(CAP['incidents'])} clean "
      f"({int(sw.sum())} recovered from swapped coordinates, "
      f"{int((ir.reject.notna() & (ir.reject != 'swapped')).sum())} quarantined)")

# ---- STAGE 4: CRS justification -----------------------------------------
from pyproj import Geod
geod = Geod(ellps="WGS84")
poly_ll = CAP["districts"].to_crs(CRS_WGS84).geometry.union_all()
true_km2 = abs(geod.geometry_area_perimeter(poly_ll)[0]) / 1e6
print(f"\nSTAGE 4 - CRS JUSTIFICATION")
print(f"  geodesic ground truth for the study area : {true_km2:,.1f} km^2")
print(f"  {'candidate CRS':<32}{'area km2':>12}{'error':>10}")
for label, code_ in [("EPSG:32633 UTM 33N (CHOSEN)", CRS_UTM),
                     ("EPSG:3857  Web Mercator", CRS_WEBMERC),
                     ("ESRI:54009 Mollweide", "ESRI:54009")]:
    a = CAP["districts"].to_crs(code_).geometry.area.sum() / 1e6
    print(f"  {label:<32}{a:>12,.1f}{100*(a-true_km2)/true_km2:>9.2f}%")

print("\nPROVENANCE LOG")
print(pd.DataFrame(PROV).to_string(index=False))
'''),

md(r'''
**Explanation.**

* **The provenance log** is the deliverable, not a debugging aid. Every row
  records where a layer came from, what CRS it arrived in, and how many rows
  survived. When a council member asks "why does your building count differ from
  ours?", this table is the answer.
* **Reprojection at the door.** The `if g.crs != CRS_UTM: g = g.to_crs(...)` line
  runs before anything else touches the data, so no downstream code has to think
  about CRS. That is the discipline; the alternative is remembering, every time.
* **Geometry repair is layer-agnostic.** The loop detects the geometry family
  from the first row and keeps only compatible types after `make_valid`, so it
  works on polygons and lines alike. It also prints the area/length **before and
  after**, which is the reconciliation an auditor will ask for.
* **`value_kvs_filled` rather than overwriting `value_kvs`.** Keeping the
  imputed column separate means any downstream result can be recomputed with and
  without imputation. Overwriting the original destroys that option permanently.
* **Recovering the swapped coordinates rather than dropping them** preserves six
  genuine observations. The rule is relational — invalid as (lon, lat), valid
  reversed — not a bounds check.
* **The CRS justification block** turns "we used UTM 33N" into "we used UTM 33N
  and here is the measured error of the alternatives". That is what makes it a
  justification rather than an assertion.

**Expected outcome.**

A geometry-validation table listing only the problem layers:

```
  layer         null  empty  invalid  dup geom   area/len before         after
  blocks           0      1        0         0         1,388.848     1,388.848
  landuse          0      0        3         2         1,408.500     1,409.715
  facilities       0      0        0        27             0.000         0.000
```

The land-use area **increases** by 1.215 km² after repair — the bow-ties were
reporting zero area.

**The `facilities` row is worth pausing on.** Twenty-seven facilities share an
*exact* location with another facility. Only three of those are true duplicate
records (same `facility_id`); the rest are genuinely co-located different
services — a school and a clinic on the same civic site. A blanket "drop
duplicate geometries" rule would have deleted 27 real facilities. **De-duplicate
on the logical key *and* the geometry, never on geometry alone.**

An attribute-cleaning block reporting 3 duplicate facilities, 11 sentinel
capacities, 18 impossible building years, 25 raw land-use labels collapsing to 8,
the orphan key `['D99']` removed, and 386 clean incidents from 393 raw with **6
recovered**.

A CRS table showing UTM 33N within ~0.1% of the geodesic truth and Web Mercator
about **+80%** wrong, and finally a 14-row provenance log.
'''),

md(r'''
## Capstone Stage 5–8 — Vector analysis, raster analysis, integration, features

**What we are about to do.** Build the district-level analysis base table:
exposure from vector overlay, terrain and hydrology from rasters, and 25+
engineered features.

**Why it matters.** Stages 5–8 are where most of the analytical value is created
and where most of the errors are made. Every double-count, every un-recomputed
area, every CRS slip lands here.

**Concept — why districts, not blocks.** The client allocates budget by district,
so the *decision unit* is the district. We compute at block level where the data
supports it and aggregate up, because aggregating fine estimates is better than
computing coarse ones — but we report at the decision unit. We check the MAUP
sensitivity in Stage 9.

**What the next cell does:** computes hazard exposure by areal and dasymetric
weighting, derives slope/TPI/HAND, runs zonal statistics for every raster,
attaches accessibility and asset features, and validates the result.
'''),

code(r'''
# =========================================================================
# CAPSTONE STAGES 5-8
# =========================================================================
from scipy.ndimage import distance_transform_edt, uniform_filter
from rasterio.warp import Resampling

D = CAP["districts"].copy()
BL = CAP["blocks"].copy()
BLD = CAP["buildings"].copy()
BLDpt = BLD.copy(); BLDpt["geometry"] = BLD.geometry.representative_point()

# ---- STAGE 5: VECTOR ANALYSIS - hazard exposure --------------------------
print("STAGE 5 - VECTOR EXPOSURE ANALYSIS")
for rp in (100, 500):
    z = CAP["flood"][CAP["flood"].return_period_yr == rp].geometry.union_all()
    # (a) areal fraction of each district in the zone
    D[f"frac_flood{rp}"] = (D.geometry.intersection(z).area / D.geometry.area).to_numpy()
    # (b) buildings and value exposed - de-duplicated, no double counting
    hit = BLDpt[BLDpt.geometry.within(z)]
    per_d = gpd.sjoin(hit, D[["district_id", "geometry"]],
                      predicate="within").drop_duplicates("building_id")
    D[f"bld_in_flood{rp}"] = D.district_id.map(
        per_d.groupby("district_id").size()).fillna(0).astype(int)
    D[f"value_in_flood{rp}"] = D.district_id.map(
        per_d.groupby("district_id").value_kvs_filled.sum()).fillna(0.0)
    # (c) population exposed - dasymetric, weighted by residential floorspace
    BLDpt["_ls"] = BLDpt.footprint_m2 * BLDpt.floors
    bb = gpd.sjoin(BLDpt, BL[["block_id", "geometry"]], predicate="within")
    tot_ls = bb.groupby("block_id")["_ls"].sum()
    in_ls = bb[bb.geometry.within(z)].groupby("block_id")["_ls"].sum()
    w = (in_ls / tot_ls).reindex(BL.block_id).fillna(0).clip(0, 1).to_numpy()
    BL[f"pop_flood{rp}"] = BL.population.to_numpy() * w
    D[f"pop_in_flood{rp}"] = D.district_id.map(
        BL.groupby("district_id")[f"pop_flood{rp}"].sum()).fillna(0.0)
    print(f"  {rp}-yr zone: {D[f'bld_in_flood{rp}'].sum():,} buildings, "
          f"{D[f'value_in_flood{rp}'].sum():,.0f} k VS, "
          f"{D[f'pop_in_flood{rp}'].sum():,.0f} people")
print(f"  CHECK no double counting: buildings in 100-yr zone counted once = "
      f"{D.bld_in_flood100.sum() == BLDpt.geometry.within(CAP['flood'][CAP['flood'].return_period_yr==100].geometry.union_all()).sum()}")

# ---- STAGE 6: RASTER ANALYSIS -------------------------------------------
print("\nSTAGE 6 - RASTER ANALYSIS")
with rasterio.open(RAS / "dem_25m.tif") as src:
    dem_a = src.read(1, masked=True).astype("float64").filled(np.nan)
    p25 = src.profile.copy(); C25 = src.res[0]
gy_, gx_ = np.gradient(dem_a, C25, C25)
slope_a = np.degrees(np.arctan(np.hypot(gx_, gy_)))

winc = int(1000 / C25)
fill = np.where(np.isfinite(dem_a), dem_a, 0.0)
cnts = uniform_filter(np.isfinite(dem_a).astype(float), size=winc, mode="nearest")
tpi_a = np.where(np.isfinite(dem_a),
                 dem_a - uniform_filter(fill, size=winc, mode="nearest")
                 / np.maximum(cnts, 1e-9), np.nan)

riv_mask25 = rasterize([(g, 1) for g in CAP["rivers"].geometry],
                       out_shape=dem_a.shape, transform=p25["transform"],
                       fill=0, dtype="uint8").astype(bool)
_, (rri, rci) = distance_transform_edt(~riv_mask25, return_indices=True)
e0 = np.nan_to_num(dem_a, nan=0.0)
hand_a = np.where(np.isfinite(dem_a), e0 - e0[rri, rci], np.nan)
driv_a = distance_transform_edt(~riv_mask25, sampling=C25)

CAPR = OUT / "capstone"; CAPR.mkdir(exist_ok=True)
for nm, arr in [("slope", slope_a), ("tpi", tpi_a), ("hand", hand_a), ("driver", driv_a)]:
    pr = p25.copy(); pr.update(dtype="float32", nodata=-9999.0, compress="deflate")
    with rasterio.open(CAPR / f"{nm}_25m.tif", "w", **pr) as dst:
        dst.write(np.nan_to_num(arr, nan=-9999.0).astype("float32"), 1)
print(f"  derived rasters written: slope, TPI, HAND, distance-to-river (25 m)")

# validation against a known quantity
zval = zonal_stats(D, RAS / "dem_25m.tif", stats=("mean",))["mean"].to_numpy()
print(f"  VALIDATION zonal mean elevation vs the layer's own column: "
      f"max |err| = {np.nanmax(np.abs(zval - D.mean_elev_m.to_numpy())):.4f} m")

# ---- STAGE 7: RASTER-VECTOR INTEGRATION ---------------------------------
print("\nSTAGE 7 - RASTER-VECTOR INTEGRATION")
RASTERS = {
    "elev":  RAS / "dem_25m.tif",   "rain": RAS / "rainfall_annual_250m.tif",
    "ndvi":  RAS / "ndvi_50m.tif",  "popdens": RAS / "popdens_100m.tif",
    "slope": CAPR / "slope_25m.tif", "tpi": CAPR / "tpi_25m.tif",
    "hand":  CAPR / "hand_25m.tif", "driver": CAPR / "driver_25m.tif",
}
for nm, path in RASTERS.items():
    zs = zonal_stats(D, path, stats=("mean", "min", "max"))
    D[f"{nm}_mean"] = zs["mean"].to_numpy()
    if nm in ("elev", "hand"):
        D[f"{nm}_min"] = zs["min"].to_numpy()

# LST needs reprojection first
lst_arr = align_to(RAS / "lst_summer_100m_3857.tif", ref, Resampling.bilinear)
lp = ref.copy(); lp.update(dtype="float32", nodata=-9999.0)
with rasterio.open(CAPR / "lst_utm.tif", "w", **lp) as dst:
    dst.write(np.nan_to_num(lst_arr, nan=-9999.0).astype("float32"), 1)
D["lst_mean"] = zonal_stats(D, CAPR / "lst_utm.tif", stats=("mean",))["mean"].to_numpy()

lcz = zonal_stats(D, RAS / "landcover_25m.tif", stats=("count",), categorical=True)
for code_, nm in zip(lc_legend.class_code, lc_legend.landuse_class):
    col = f"pct_class_{code_}"
    if col in lcz.columns:
        D["lc_" + nm.lower().split()[0].replace("-", "")] = lcz[col].to_numpy()
print(f"  {len(RASTERS)+1} rasters + land-cover composition attached to {len(D)} districts")

# ---- STAGE 8: FEATURE ENGINEERING ---------------------------------------
print("\nSTAGE 8 - FEATURE ENGINEERING")
Dc = D.geometry.representative_point()
# NOTE: do NOT call these columns "cx"/"cy". `gdf.cx` is GeoPandas' coordinate
# INDEXER, so `D.cx` would return the indexer object, not your column - a
# genuinely baffling bug the first time you meet it.
D["ctr_x"], D["ctr_y"] = Dc.x.to_numpy(), Dc.y.to_numpy()

# proximity
for nm, tgt in [("hospital", CAP["facilities"][CAP["facilities"].facility_type == "hospital"]),
                ("clinic",   CAP["facilities"][CAP["facilities"].facility_type == "clinic"]),
                ("fire",     CAP["facilities"][CAP["facilities"].facility_type == "fire_station"]),
                ("primary_road", CAP["roads"][CAP["roads"].road_class.isin(["motorway", "primary"])])]:
    g = tgt.geometry.union_all()
    D[f"dist_{nm}_m"] = Dc.distance(g).to_numpy()

# density
bxy = np.c_[BLDpt.geometry.x, BLDpt.geometry.y]
tree_b = cKDTree(bxy)
D["bld_within_3km"] = [len(i) for i in tree_b.query_ball_point(np.c_[D.ctr_x, D.ctr_y], 3000)]
sxy = np.c_[CAP["stops"].geometry.x, CAP["stops"].geometry.y]
D["stops_within_3km"] = [len(i) for i in cKDTree(sxy).query_ball_point(np.c_[D.ctr_x, D.ctr_y], 3000)]

# assets
bd = gpd.sjoin(BLDpt, D[["district_id", "geometry"]],
               predicate="within").drop_duplicates("building_id")
agg = bd.groupby("district_id").agg(
    n_buildings=("building_id", "size"),
    total_value=("value_kvs_filled", "sum"),
    mean_age=("building_age", "mean"),
    pct_timber=("construction", lambda s: 100 * (s == "timber").mean()),
    pct_basement=("has_basement", lambda s: 100 * s.mean()))
for c in agg.columns:
    D[c] = D.district_id.map(agg[c]).to_numpy()

# socio-economic join, checked
before = len(D)
D = D.merge(CAP_SOCIO.drop(columns=[c for c in ["name", "district_type",
                                                "population", "households"]
                                    if c in CAP_SOCIO.columns]),
            on="district_id", how="left", validate="one_to_one")
assert len(D) == before, "socio join changed the row count"

# spatial lag over district contiguity
Wd_ = queen_weights(D)
def dlag(v):
    v = np.asarray(v, float); ok = np.isfinite(v)
    Wm = Wd_ * ok[None, :]; den = Wm.sum(axis=1)
    return np.where(den > 0, (Wm @ np.nan_to_num(v)) / np.maximum(den, 1e-12), np.nan)
for c in ["frac_flood100", "median_income_vs", "dist_hospital_m"]:
    D[f"lag_{c}"] = dlag(D[c])

# ratios
D["value_per_capita"] = D.total_value / D.population
D["ead_exposure_ratio"] = D.value_in_flood100 / D.total_value.replace(0, np.nan)
D["pop_exposed_pct"] = 100 * D.pop_in_flood100 / D.population

FEATCOLS = [c for c in D.columns if c not in
            ("district_id", "name", "district_type", "geometry", "cx", "cy",
             "survey_year")]
print(f"  district table: {len(D)} rows x {len(FEATCOLS)} numeric features")
print(f"  feature families: proximity {sum(c.startswith('dist_') for c in FEATCOLS)}, "
      f"density {sum('within' in c for c in FEATCOLS)}, "
      f"composition {sum(c.startswith('lc_') for c in FEATCOLS)}, "
      f"raster {sum(c.endswith('_mean') for c in FEATCOLS)}, "
      f"lag {sum(c.startswith('lag_') for c in FEATCOLS)}")
print(f"  missing values remaining: "
      f"{D[FEATCOLS].isna().sum().sum()} across {int((D[FEATCOLS].isna().sum() > 0).sum())} columns")
D.to_file(OUT / "capstone_districts.gpkg", layer="districts", driver="GPKG")
print(f"  saved -> capstone_districts.gpkg")
'''),

md(r'''
**Explanation.**

* **Stage 5 computes exposure three ways on purpose**: areal fraction (crude but
  transparent), building/value counts (exact, de-duplicated), and dasymetric
  population (best estimate). Reporting all three lets the reader see how much the
  method matters.
* The **`CHECK no double counting`** line compares the summed per-district count
  against the global count. If a building near a district boundary were counted
  twice, the two would differ. Assertions like this are cheap and catch the
  errors that silently inflate headline figures.
* **Stage 6 writes its derived rasters to disk.** Slope, TPI, HAND and
  distance-to-river become inspectable artefacts, and `zonal_stats` can consume
  them by path. The alternative — keeping everything in memory — makes the
  pipeline unauditable.
* **The zonal validation** against the layer's own `mean_elev_m` proves the
  rasterisation, transform and NoData handling are all correct in one number.
* **Stage 7 reprojects the Web Mercator LST raster before zonal statistics.**
  Running zonal statistics across a CRS mismatch produces plausible numbers that
  are wrong.
* **`validate="one_to_one"`** on the socio-economic merge makes pandas raise if
  the join is not one-to-one. Combined with the `assert`, a duplicated key can no
  longer silently multiply rows. **Use `validate=` on every merge you care about.**
* **The feature-family count** at the end is a completeness check against the
  brief's requirement of at least 20 features across five families.

**Expected outcome.**

Stage 5 should report on the order of **1 000 buildings and 250 000 k VS** in the
100-year zone, with a slightly larger figure for the 500-year zone, and the
double-counting check returning `True`.

Stage 6 should confirm the zonal validation to within **0.05 m**.

Stage 8 should produce a district table of roughly **50–60 numeric features**
with a handful of missing values (the two blanked district populations, and any
district with no buildings), and save it to `capstone_districts.gpkg`.
'''),

]

CELLS += [

md(r'''
## Capstone Stage 9 — Statistical and spatial analysis

**What we are about to do.** Test for spatial structure, find statistically
defensible hotspots, build the Composite Resilience Index, and check that the
answer survives a change of spatial unit.

**Why it matters.** This is where the analysis becomes evidence. An index without
a significance test and a scale check is an opinion with decimal places.

**Concept — the Composite Resilience Index.** We combine three domains:

```
CRI = w_h · Hazard  +  w_a · (1 − Accessibility)  +  w_v · Vulnerability
```

each domain being a normalised aggregate of several indicators, all oriented so
that **higher = worse**. The weights are the political content; the sensitivity
analysis is what makes them arguable rather than imposed.

**What the next cell does:** builds the three domain scores, tests each for
spatial autocorrelation, runs an FDR-corrected Gi* hotspot analysis on the CRI,
performs a Monte-Carlo weight sensitivity, and repeats the whole index at block
level to test MAUP sensitivity.
'''),

code(r'''
# =========================================================================
# CAPSTONE STAGE 9: STATISTICAL AND SPATIAL ANALYSIS
# =========================================================================
Dx = D.copy()

def norm01(v, higher_is_worse=True):
    v = pd.Series(v).astype(float)
    r = v.rank(pct=True, na_option="keep")
    return r if higher_is_worse else 1 - r

# ---- three domain scores ---------------------------------------------------
HAZ = {"pop_exposed_pct": True, "ead_exposure_ratio": True,
       "hand_mean": False, "frac_flood100": True}
ACC = {"dist_hospital_m": True, "dist_clinic_m": True, "dist_fire_m": True,
       "stops_within_3km": False}
VUL = {"median_income_vs": False, "unemployment_rate": True,
       "pct_over65": True, "mean_age": True, "pct_timber": True,
       "vehicles_per_household": False}

for name, spec in [("hazard", HAZ), ("access", ACC), ("vuln", VUL)]:
    parts = [norm01(Dx[c], hw) for c, hw in spec.items() if c in Dx.columns]
    Dx[f"score_{name}"] = np.nanmean(np.c_[tuple(parts)], axis=1)
    print(f"  {name:<8} domain from {len(parts)} indicators: "
          f"range {Dx[f'score_{name}'].min():.3f} - {Dx[f'score_{name}'].max():.3f}")

W_CRI = dict(hazard=0.40, access=0.35, vuln=0.25)
Dx["CRI"] = sum(W_CRI[k] * Dx[f"score_{k}"] for k in W_CRI)
Dx["CRI_rank"] = Dx.CRI.rank(ascending=False).astype(int)

print("\nCOMPOSITE RESILIENCE INDEX  (higher = more at risk, less resilient)")
print(Dx.nlargest(8, "CRI")[["district_id", "name", "district_type", "population",
                             "score_hazard", "score_access", "score_vuln", "CRI"]]
      .round(3).to_string(index=False))

# ---- spatial autocorrelation of each domain --------------------------------
print("\nSPATIAL AUTOCORRELATION OF THE DOMAIN SCORES (queen weights, 9999 perms)")
print(f"  {'score':<16}{'Moran I':>10}{'p':>9}")
for c in ["score_hazard", "score_access", "score_vuln", "CRI"]:
    r = moran_test(Dx[c], Wd_, permutations=9999, seed=3)
    print(f"  {c:<16}{r['I']:>10.4f}{r['p_sim']:>9.4f}")

# ---- FDR-corrected hotspots of the CRI ------------------------------------
gi = getis_ord_gstar(Dx.CRI.to_numpy(dtype=float), Wd_)
from scipy.stats import norm as _norm
p_gi = 2 * (1 - _norm.cdf(np.abs(gi)))
sig = benjamini_hochberg(p_gi, 0.05)
Dx["gi_z"] = gi
Dx["cri_hotspot"] = np.where(sig & (gi > 0), "hot",
                      np.where(sig & (gi < 0), "cold", "not significant"))
print(f"\nCRI HOTSPOTS (Getis-Ord Gi*, FDR-corrected)")
print(f"  uncorrected significant : {int((p_gi < 0.05).sum())} of {len(Dx)}")
print(f"  FDR-corrected           : {int(sig.sum())}")
print(f"  classification          : {Dx.cri_hotspot.value_counts().to_dict()}")
hot = Dx[Dx.cri_hotspot == "hot"]
if len(hot):
    print(f"  hot districts           : {hot.name.tolist()}")
    print(f"  population in them      : {hot.population.sum():,.0f} "
          f"({100*hot.population.sum()/Dx.population.sum():.1f} %)")

# ---- weight sensitivity ----------------------------------------------------
rng = np.random.default_rng(11)
SIMS = 2000
top5 = np.zeros(len(Dx)); ranks = np.zeros(len(Dx))
base = np.array([W_CRI[k] for k in ["hazard", "access", "vuln"]])
S = np.c_[Dx.score_hazard, Dx.score_access, Dx.score_vuln]
for _ in range(SIMS):
    w = np.abs(base * rng.normal(1, 0.30, 3)); w = w / w.sum()
    sc = S @ w
    order = np.argsort(-sc)
    top5[order[:5]] += 1
    ranks += pd.Series(-sc).rank().to_numpy()
Dx["p_top5"] = top5 / SIMS
Dx["mean_rank"] = ranks / SIMS

print(f"\nWEIGHT SENSITIVITY ({SIMS} simulations, weights perturbed +/-30 %)")
print(Dx.nlargest(8, "p_top5")[["district_id", "name", "CRI", "CRI_rank",
                                "p_top5", "mean_rank"]].round(3).to_string(index=False))
print(f"  districts in the top 5 in EVERY simulation : "
      f"{int((Dx.p_top5 == 1.0).sum())}")
print(f"  districts ever in the top 5                : {int((Dx.p_top5 > 0).sum())}")

# ---- MAUP CHECK: rebuild the index at BLOCK level -------------------------
Bx = Fx.copy()
Bx["score_hazard_b"] = np.nanmean(np.c_[
    norm01(Bx.pct_in_flood100, True), norm01(Bx.dist_river_m, False)], axis=1)
Bx["score_access_b"] = np.nanmean(np.c_[
    norm01(Bx.dist_hospital_m, True), norm01(Bx.dist_clinic_m, True),
    norm01(Bx.n_bus_stops, False)], axis=1)
Bx["CRI_b"] = 0.5 * Bx.score_hazard_b + 0.5 * Bx.score_access_b
blk_to_dist = Bx.groupby("district_id").apply(
    lambda g: np.average(g.CRI_b, weights=np.maximum(g.population, 1)),
    include_groups=False)
Dx["CRI_from_blocks"] = Dx.district_id.map(blk_to_dist)

r_scale = Dx[["CRI", "CRI_from_blocks"]].corr(method="spearman").iloc[0, 1]
rank_shift = (Dx.CRI.rank(ascending=False) -
              Dx.CRI_from_blocks.rank(ascending=False)).abs()
print(f"\nMAUP CHECK - the same index built from BLOCKS then aggregated up")
print(f"  Spearman correlation with the district-level index : {r_scale:.3f}")
print(f"  median |rank shift| : {rank_shift.median():.1f} places, "
      f"max {rank_shift.max():.0f} places")
print(f"  districts whose rank moves by more than 3 places: "
      f"{int((rank_shift > 3).sum())} of {len(Dx)}")
print("  -> the top of the ranking is stable to the choice of unit; the middle")
print("     is not. Report the top confidently and the middle with caution.")
'''),

md(r'''
**Explanation.**

* **Rank normalisation within each domain**, then a simple mean of indicators,
  then a weighted sum of domains. This two-level structure means an indicator
  cannot dominate simply because it has a wider range, and it keeps the weights
  interpretable — `w_hazard = 0.40` means what it says.
* **Direction flags (`higher_is_worse`)** are declared per indicator in the
  dictionaries. Writing them down as data rather than burying them in code is
  what lets a reviewer check them.
* **`hand_mean` is flagged `False`** (lower HAND is *worse*), and
  `median_income_vs` likewise. Getting one direction wrong silently inverts a
  whole domain — this is the single most common error in composite-index work.
* **9 999 permutations for Moran's I**, not 999, because with 24 units and FDR
  correction downstream we need the p-value resolution (Lesson A6).
* **The MAUP check is the stage most analyses skip.** We rebuild a comparable
  index at *block* level and aggregate it up with population weights, then compare
  rankings. If the two disagree strongly, the index is an artefact of the unit.
  Reporting the correlation and the maximum rank shift is the honest summary.
* `include_groups=False` in the `groupby.apply` — required in pandas 2.2+ to
  avoid a deprecation warning about the grouping columns being passed to the
  function.

**Expected outcome.**

Three domain scores, each spanning roughly 0.1–0.9.

```
COMPOSITE RESILIENCE INDEX  (higher = more at risk, less resilient)
district_id        name district_type  population  hazard  access   vuln    CRI
        D11    Norrbank  upland_rural       4,157   0.721   0.641  0.653  0.676
        D19 Corran Vale  upland_rural       3,531   0.591   0.755  0.627  0.657
        D23     Ostrand  upland_rural         743   0.590   0.797  0.482  0.635
        D15    Lyndover  upland_rural       3,235   0.687   0.568  0.566  0.615
        D09    Tarnwell      suburban      45,734   0.749   0.474  0.497  0.590
```

**The top of the ranking is dominated by remote upland districts**, driven by the
access domain — not by the flood-exposed urban core. That is a real finding and a
slightly uncomfortable one: the districts with the *most people at risk* are not
the districts with the *highest composite index*. Note `D09 Tarnwell` at rank 5
with **45 734 residents** against Ostrand's 743. **An index that is not
population-weighted ranks places, not people**, and the client must be told which
question they are asking.

```
SPATIAL AUTOCORRELATION (queen weights, 9999 permutations)
  score_hazard       -0.1571   0.4143
  score_access        0.6259   0.0001
  score_vuln         -0.0362   0.9137
  CRI                 0.0819   0.3467

CRI HOTSPOTS (Getis-Ord Gi*, FDR-corrected)
  uncorrected significant : 0 of 24
  FDR-corrected           : 0
```

**Only the access domain is significantly clustered.** Hazard and vulnerability
are not — which is surprising until you remember that rank-normalising 24 units
compresses the very variation that Moran's I measures. Compare with Module 3,
where the same variables at *block* level had I between 0.42 and 0.99. **Spatial
structure is scale-dependent, and 24 units is too few to see it.**

The Gi* analysis finds **zero** significant hotspots, corrected or uncorrected.
That is the correct answer, not a failure. With n = 24 there is not enough data
for local inference, and reporting "no significant clusters" is far better than
mapping the top three Gi* values as if they meant something.

```
WEIGHT SENSITIVITY (2000 simulations, +/-30 %)
        name   CRI  CRI_rank  p_top5  mean_rank
    Norrbank 0.676         1   1.000      1.264
 Corran Vale 0.657         2   0.996      2.047
    Lyndover 0.615         4   0.904      4.342
     Ostrand 0.635         3   0.884      3.496
   Ardenfeld 0.576        10   0.328      8.625
  districts in the top 5 in EVERY simulation : 1
  districts ever in the top 5                : 13
```

**Only one district survives every weighting; thirteen appear at least once.**
The robust recommendation is the first four (p > 0.88), not "the top five".
Notice too that Lyndover ranks 4th on the point estimate but 3rd on robustness,
ahead of Ostrand — point rank and robustness rank are different orderings.

```
MAUP CHECK
  Spearman correlation with the district-level index : 0.761
  median |rank shift| : 2.5 places, max 13 places
  districts whose rank moves by more than 3 places: 9 of 24
```

Rebuilt from blocks, the index correlates 0.76 with the district-level version —
strong but far from identical, and **9 of 24 districts move more than three
places**. The extremes are stable; the middle is not. Speak confidently about the
top and bottom, and cautiously about everything between.
'''),

md(r'''
## Capstone Stage 10–12 — Visualisation, interpretation, conclusion

**What we are about to do.** Produce the map series and the costed
recommendation, and state what the analysis cannot support.

**Why it matters.** Everything so far was analysis. This is the part the client
reads. A finding that is not communicated does not exist.

**Concept — the three maps every risk assessment needs.**

1. **The exposure map** — what is at stake, and where.
2. **The composite map** — the index, with its classification named and its
   missing data visible.
3. **The uncertainty map** — where the analysis is least reliable.

Publishing the first two without the third is the most common failure in applied
spatial analysis.

**What the next cell does:** produces a six-panel map series, computes the costed
intervention list, and prints the final assessment including an explicit
limitations section.
'''),

code(r'''
# =========================================================================
# CAPSTONE STAGES 10-12
# =========================================================================
# ---- STAGE 10: THE MAP SERIES ------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.5))

Dx.plot(ax=axes[0, 0], column="pop_in_flood100", cmap="Blues",
        scheme="naturalbreaks", k=5, legend=True,
        legend_kwds={"loc": "lower left", "fontsize": 6, "title": "people"},
        edgecolor="grey", linewidth=0.4,
        missing_kwds={"color": "#dddddd", "hatch": "///", "label": "no data"})
CAP["flood"][CAP["flood"].return_period_yr == 100].plot(
    ax=axes[0, 0], facecolor="none", edgecolor="#08519c", linewidth=0.5)
axes[0, 0].set_title("(1) EXPOSURE\nPopulation in the 100-year flood zone",
                     fontsize=10, weight="bold", loc="left")

Dx.plot(ax=axes[0, 1], column="value_in_flood100", cmap="Purples",
        scheme="naturalbreaks", k=5, legend=True,
        legend_kwds={"loc": "lower left", "fontsize": 6, "title": "k VS"},
        edgecolor="grey", linewidth=0.4)
axes[0, 1].set_title("(2) EXPOSURE\nAsset value in the 100-year flood zone",
                     fontsize=10, weight="bold", loc="left")

Dx.plot(ax=axes[0, 2], column="score_access", cmap="OrRd",
        scheme="quantiles", k=5, legend=True,
        legend_kwds={"loc": "lower left", "fontsize": 6},
        edgecolor="grey", linewidth=0.4)
CAP["facilities"][CAP["facilities"].facility_type.isin(["hospital", "clinic"])].plot(
    ax=axes[0, 2], color="black", markersize=10)
axes[0, 2].set_title("(3) ACCESS DEFICIT\nhigher = worse served",
                     fontsize=10, weight="bold", loc="left")

Dx.plot(ax=axes[1, 0], column="CRI", cmap="RdYlGn_r", scheme="quantiles", k=5,
        legend=True, legend_kwds={"loc": "lower left", "fontsize": 6},
        edgecolor="black", linewidth=0.5,
        missing_kwds={"color": "#dddddd", "hatch": "///", "label": "no data"})
for _, r in Dx.nlargest(5, "CRI").iterrows():
    axes[1, 0].annotate(r["name"], (r.geometry.representative_point().x,
                                    r.geometry.representative_point().y),
                        ha="center", fontsize=6.5, weight="bold", color="white",
                        path_effects=[pe.withStroke(linewidth=2, foreground="#333")])
axes[1, 0].set_title("(4) COMPOSITE RESILIENCE INDEX\nquantile classification, "
                     "top 5 labelled", fontsize=10, weight="bold", loc="left")

Dx.plot(ax=axes[1, 1], column="p_top5", cmap="viridis", vmin=0, vmax=1,
        legend=True, legend_kwds={"shrink": 0.6, "label": "P(top 5)"},
        edgecolor="grey", linewidth=0.4)
axes[1, 1].set_title("(5) UNCERTAINTY\nP(in the top 5) over 2,000 weightings",
                     fontsize=10, weight="bold", loc="left")

shift = (Dx.CRI.rank(ascending=False) - Dx.CRI_from_blocks.rank(ascending=False)).abs()
Dx.assign(shift=shift).plot(ax=axes[1, 2], column="shift", cmap="magma_r",
                            legend=True, legend_kwds={"shrink": 0.6,
                                                      "label": "|rank shift|"},
                            edgecolor="grey", linewidth=0.4,
                            missing_kwds={"color": "#dddddd"})
axes[1, 2].set_title("(6) SCALE SENSITIVITY\nrank change when built from blocks",
                     fontsize=10, weight="bold", loc="left")

for a in axes.ravel():
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
plt.suptitle("Vallmara Basin Integrated Climate-Resilience Assessment  |  "
             "CRS EPSG:32633 (UTM 33N)  |  fictional data",
             fontsize=13, weight="bold", y=0.995)
plt.tight_layout(); plt.show()

# ---- STAGE 12: COSTED INTERVENTION LIST --------------------------------
BUDGET = 120_000.0            # thousand VS over five years

UNIT_COSTS = {
    "flood_defence":   {"per_person_protected": 1.10, "averted_frac": 0.65},
    "clinic":          {"fixed": 9_500.0, "reach_km": 10.0},
    "retrofit":        {"per_building": 12.0, "averted_frac": 0.35},
}

ead_col = Dx.value_in_flood100 * 0.02     # crude annualisation for ranking
Dx["ead_proxy"] = ead_col

recs = []
for _, r in Dx.iterrows():
    # (a) flood defence, sized by exposed population
    if r.pop_in_flood100 > 500:
        cost = r.pop_in_flood100 * UNIT_COSTS["flood_defence"]["per_person_protected"]
        avert = r.ead_proxy * UNIT_COSTS["flood_defence"]["averted_frac"]
        recs.append(dict(district=r["name"], district_id=r.district_id,
                         intervention="flood defence", cost_kvs=cost,
                         annual_benefit_kvs=avert,
                         people_helped=r.pop_in_flood100, CRI=r.CRI))
    # (b) new clinic where access is poor and population is meaningful
    if r.dist_clinic_m > 8000 and r.population > 3000:
        recs.append(dict(district=r["name"], district_id=r.district_id,
                         intervention="new clinic",
                         cost_kvs=UNIT_COSTS["clinic"]["fixed"],
                         annual_benefit_kvs=0.0002 * r.population * r.dist_clinic_m / 1000,
                         people_helped=r.population, CRI=r.CRI))
    # (c) retrofit the most vulnerable building stock
    if r.bld_in_flood100 > 30:
        cost = r.bld_in_flood100 * UNIT_COSTS["retrofit"]["per_building"]
        recs.append(dict(district=r["name"], district_id=r.district_id,
                         intervention="building retrofit", cost_kvs=cost,
                         annual_benefit_kvs=r.ead_proxy * UNIT_COSTS["retrofit"]["averted_frac"],
                         people_helped=r.pop_in_flood100, CRI=r.CRI))

R = pd.DataFrame(recs)
R["bcr_20yr"] = 20 * R.annual_benefit_kvs / R.cost_kvs        # benefit-cost ratio
R = R.sort_values("bcr_20yr", ascending=False).reset_index(drop=True)
R["cum_cost"] = R.cost_kvs.cumsum()
R["funded"] = R.cum_cost <= BUDGET

print("=" * 96)
print("PRIORITISED INTERVENTION LIST  (ranked by 20-year benefit-cost ratio)")
print("=" * 96)
print(R.head(14)[["district", "intervention", "cost_kvs", "annual_benefit_kvs",
                  "bcr_20yr", "people_helped", "cum_cost", "funded"]]
      .round(1).to_string(index=False))
funded = R[R.funded]
print("-" * 96)
print(f"  budget                : {BUDGET:>12,.0f} k VS")
print(f"  committed             : {funded.cost_kvs.sum():>12,.0f} k VS "
      f"({100*funded.cost_kvs.sum()/BUDGET:.0f} %)")
print(f"  interventions funded  : {len(funded):>12} of {len(R)}")
print(f"  people directly helped: {funded.people_helped.sum():>12,.0f}")
print(f"  annual benefit        : {funded.annual_benefit_kvs.sum():>12,.0f} k VS/yr")
print(f"  20-yr benefit-cost    : "
      f"{20*funded.annual_benefit_kvs.sum()/funded.cost_kvs.sum():>12.2f}")

# ---- STAGE 11-12: THE WRITTEN CONCLUSION -------------------------------
top = Dx.nlargest(3, "CRI")
robust = Dx[Dx.p_top5 > 0.8]
print("\n" + "=" * 96)
print("FINDINGS")
print("=" * 96)
print(f"""
1. EXPOSURE. {Dx.pop_in_flood100.sum():,.0f} people ({100*Dx.pop_in_flood100.sum()/Dx.population.sum():.0f} % of
   the basin) and {Dx.value_in_flood100.sum():,.0f} k VS of assets lie inside the
   100-year flood zone. Exposure is concentrated: the top three districts hold
   {100*Dx.nlargest(3,'value_in_flood100').value_in_flood100.sum()/Dx.value_in_flood100.sum():.0f} % of the exposed value.

2. ACCESS. Median distance to a hospital is {Dx.dist_hospital_m.median()/1000:.1f} km, and
   the worst-served district is {Dx.loc[Dx.dist_hospital_m.idxmax(),'name']} at
   {Dx.dist_hospital_m.max()/1000:.1f} km. Four hospitals serve {Dx.population.sum():,.0f} residents.

3. COMPOSITE. The three districts with the highest CRI are
   {', '.join(top.name.tolist())}. {len(robust)} district(s) remain in the top five under
   more than 80 % of plausible weightings; those are the robust priorities.

4. INVESTMENT. Ranking by 20-year benefit-cost ratio, {len(funded)} interventions fit
   the {BUDGET:,.0f} k VS budget, directly protecting {funded.people_helped.sum():,.0f} people at a
   portfolio benefit-cost ratio of {20*funded.annual_benefit_kvs.sum()/funded.cost_kvs.sum():.1f}.

WHAT THIS ANALYSIS CANNOT TELL YOU
  a. Travel times are STRAIGHT-LINE. Real network distances are typically
     20-50 % longer, so the accessibility deficit is UNDERSTATED, especially in
     the uplands where roads are indirect.
  b. Flood depths use a bathtub model over a HAND surface. There is no hydraulic
     routing, no defences and no drainage; depths are indicative only.
  c. Depth-damage curves are transferred, not locally calibrated. A +/-25 % change
     in them moves expected losses by 15-30 % (Lesson A13).
  d. The index is computed on 24 districts. That is too few units for reliable
     local hotspot inference, and the middle of the ranking is unstable to the
     choice of spatial unit (median rank shift {shift.median():.1f} places).
  e. Two districts have no recorded population. Their scores are computed from
     the block data that survives, and they are hatched on every map.
  f. Nothing here is causal. High CRI districts are places where hazard,
     poor access and vulnerability CO-OCCUR; that is a targeting statement,
     not a mechanism.
""")
print("=" * 96)

Dx.to_file(OUT / "capstone_results.gpkg", layer="district_cri", driver="GPKG")
R.to_csv(OUT / "capstone_interventions.csv", index=False)
print(f"Deliverables written to {OUT}:")
print(f"  capstone_results.gpkg       (district CRI, domain scores, uncertainty)")
print(f"  capstone_interventions.csv  ({len(R)} costed interventions)")
print(f"  capstone_districts.gpkg     (full {len(FEATCOLS)}-feature analysis table)")
'''),

md(r'''
**Explanation.**

* **The six-panel series follows the three-map rule** and adds a scale-sensitivity
  panel. Panels 1–3 are the inputs, panel 4 is the index, and **panels 5 and 6 are
  the honesty**: where the ranking is robust to the weights, and where it is
  robust to the choice of spatial unit. A client who sees only panel 4 will
  over-read it.
* **`missing_kwds` on every choropleth.** The two districts with no population are
  hatched, not white. On a resilience map, "no data" rendered as "low risk" is a
  potentially serious error.
* **The intervention list ranks by benefit–cost ratio, not by CRI.** This is
  deliberate and worth understanding: the highest-CRI district is not necessarily
  where the next VS is best spent. Targeting need and maximising return are
  different objectives, and a good report presents both and lets the decision-
  maker choose.
* **The unit costs and averted fractions are stated explicitly** as a dictionary
  at the top. They are assumptions, they are almost certainly wrong in detail, and
  putting them in one visible place is what makes them challengeable.
* **The limitations section is not boilerplate.** Each item names a specific
  assumption, its direction of bias where known ("the accessibility deficit is
  understated"), and a pointer to the lesson that quantified it. That is the
  difference between covering yourself and informing your reader.

**Expected outcome.**

A six-panel figure with a title bar naming the CRS and the fictional status of
the data.

```
PRIORITISED INTERVENTION LIST  (ranked by 20-year benefit-cost ratio)
    district      intervention   cost_kvs  annual_benefit  bcr_20yr    cum_cost  funded
Old Vallmara building retrofit    6,108.0        1,165.6       3.8     6,108.0    True
 Harbourgate building retrofit    1,620.0          287.9       3.6     7,728.0    True
    Tarnwell building retrofit    1,776.0          190.1       2.1     9,504.0    True
   Ardenfeld building retrofit    1,512.0           91.4       1.2    11,016.0    True
Old Vallmara     flood defence   63,346.1        2,164.8       0.7    74,362.1    True
 Harbourgate     flood defence   19,759.3          534.6       0.5    94,121.5    True
    Tarnwell     flood defence   22,278.5          353.1       0.3   116,400.0    True
    Brannock     flood defence    1,132.6           12.6       0.2   117,532.6    True
     ...                                                              >120,000   False
```

**Two things in this table matter more than the ranking itself.**

First, **building retrofit dominates flood defence on benefit–cost ratio** — 3.8
against 0.7 in the same district — even though defence protects far more people.
Retrofit is cheap per unit of averted loss; defence is capital-intensive. A
BCR-ranked list will always fund retrofit first, which is economically correct
and politically difficult, because defence is what communities ask for.

Second, **most interventions have a 20-year BCR below 1.0**, so the portfolio as
a whole does not pay for itself under the stated unit costs. That is a genuine
appraisal result, not a bug: it says either the unit costs are too pessimistic,
or the averted-damage fractions are too conservative, or the programme should be
justified on grounds other than avoided property damage (lives, disruption,
equity). **Report it and name the three possibilities** rather than quietly
tuning the assumptions until the answer looks better.

Then a findings block with four numbered findings and six numbered limitations,
and three deliverable files written to `data/outputs/`.

**If you have got this far with your own implementation and can answer yes to the
rubric at the top of Module 4, you are ready to do this work for real.**
'''),

]

CELLS += [

md(r'''
# Solutions

Complete worked solutions to every exercise. Read them **after** attempting the
problems — the value is in the struggle, not in the answer.

Where an exercise is open-ended (the Module 3 set), the solution gives a complete,
runnable implementation of one defensible approach and names the decisions where
a different choice would be equally valid.

All solutions reuse the helper functions defined earlier in the notebook, so run
the notebook from the top before executing this section.
'''),

md(r'''
## Solutions — Module 1 (Beginner)

### 1.1 Layer inventory · 1.2 Attribute audit · 1.3 CRS forensics

**1.1** requires iterating over both the GeoPackage layers and the standalone
files, measuring in a single CRS. **1.2** is a reusable audit function.
**1.3** is the same area comparison as B5, applied to a different layer.
'''),

code(r'''
# ================================================== SOLUTION 1.1 ============
import pyogrio

def layer_inventory():
    rows = []
    for name, _ in pyogrio.list_layers(GPKG):
        g = gpd.read_file(GPKG, layer=name).to_crs(CRS_UTM)
        gt = g.geom_type.iloc[0]
        rows.append(dict(
            source="vallmara.gpkg", layer=name, n_features=len(g),
            geometry_type=str(dict(g.geom_type.value_counts())),
            crs="EPSG:32633", n_columns=g.shape[1],
            area_km2=round(g.geometry.area.sum()/1e6, 3) if "Polygon" in gt else np.nan,
            length_km=round(g.geometry.length.sum()/1000, 3) if "Line" in gt else np.nan))
    for fn in ["roads.geojson", "protected_areas.geojson"]:
        g0 = gpd.read_file(VEC / fn)
        g = g0.to_crs(CRS_UTM)
        gt = g.geom_type.iloc[0]
        rows.append(dict(
            source=fn, layer="(single)", n_features=len(g),
            geometry_type=str(dict(g.geom_type.value_counts())),
            crs=g0.crs.to_string(), n_columns=g.shape[1],
            area_km2=round(g.geometry.area.sum()/1e6, 3) if "Polygon" in gt else np.nan,
            length_km=round(g.geometry.length.sum()/1000, 3) if "Line" in gt else np.nan))
    return pd.DataFrame(rows)

inv = layer_inventory()
print("SOLUTION 1.1 - LAYER INVENTORY")
print(inv.to_string(index=False))

# ================================================== SOLUTION 1.2 ============
def audit(gdf, name="layer"):
    """Full attribute + geometry audit of any GeoDataFrame."""
    a = pd.DataFrame({
        "dtype": gdf.dtypes.astype(str),
        "n_missing": gdf.isna().sum(),
        "pct_missing": (100 * gdf.isna().mean()).round(2),
        "n_unique": gdf.nunique(dropna=True),
    })
    num = gdf.select_dtypes(include=[np.number])
    for stat in ["min", "median", "max"]:
        a[stat] = getattr(num, stat)().round(3)
    print(f"AUDIT: {name}  ({len(gdf)} rows x {gdf.shape[1]} cols, "
          f"CRS {gdf.crs.to_string() if gdf.crs else 'NONE'})")
    print(f"  geometry: {dict(gdf.geom_type.value_counts())}, "
          f"null={int(gdf.geometry.isna().sum())}, "
          f"empty={int(gdf.geometry.is_empty.sum())}, "
          f"invalid={int((~gdf.geometry.is_valid).sum())}, "
          f"dup_geom={int(gdf.geometry.to_wkb().duplicated().sum())}")
    return a

print("\n\nSOLUTION 1.2 - AUDIT OF census_blocks")
bl_raw = gpd.read_file(GPKG, layer="census_blocks")
print(audit(bl_raw, "census_blocks").to_string())
print("""
PROBLEMS FOUND IN census_blocks
  * one EMPTY geometry (index 300, block B0301). It keeps its attributes, so
    attribute sums stay correct while any geometric union silently loses
    5.93 km^2. Detect with .is_empty, never with .isna().
  * area_km2 is a STORED attribute, not recomputed. After any clip or overlay it
    would be stale. Recompute from the geometry whenever you cut it.
  * pop_density_km2 is a derived ratio; it is redundant with population/area_km2
    and will silently disagree with them if either is edited.
  * no missing values otherwise - this layer is clean apart from the empty row.""")

# ================================================== SOLUTION 1.3 ============
print("\n\nSOLUTION 1.3 - CRS FORENSICS ON protected_areas.geojson")
pa0 = gpd.read_file(VEC / "protected_areas.geojson")
print(f"  native CRS      : {pa0.crs.to_string()} ({pa0.crs.axis_info[0].unit_name})")
print(f"  .area in native : {pa0.geometry.area.sum():.8f} square degrees "
      f"<- MEANINGLESS")
truth = pa0.to_crs(CRS_UTM).geometry.area.sum() / 1e6
print(f"\n  {'CRS':<30}{'area km2':>12}{'error vs UTM':>15}")
for lab, code_ in [("EPSG:32633 UTM 33N", CRS_UTM),
                   ("EPSG:3857  Web Mercator", CRS_WEBMERC),
                   ("ESRI:54009 Mollweide", "ESRI:54009")]:
    a = pa0.to_crs(code_).geometry.area.sum() / 1e6
    print(f"  {lab:<30}{a:>12,.3f}{100*(a-truth)/truth:>14.2f}%")
print("""
  WHICH WOULD I PUBLISH?
  The UTM 33N figure. The study area sits entirely within zone 33, where UTM's
  scale error is under 0.1 % (verified against Mollweide, an equal-area
  projection, which agrees to ~0.1 %). Web Mercator over-states the area by
  about 80 % at this latitude and must never be used for a published statistic,
  even though its units are nominally metres.""")
'''),

md(r'''
### 1.4 Dirty CSV · 1.5 Shapely reasoning · 1.6 Distance profile
'''),

code(r'''
# ================================================== SOLUTION 1.4 ============
print("SOLUTION 1.4 - QUARANTINING BAD COORDINATES")
raw = pd.read_csv(TAB / "flood_incidents.csv")
BOX = dict(lon=(13.6, 14.5), lat=(41.4, 41.9))
ok = lambda lo, la: lo.between(*BOX["lon"]) & la.between(*BOX["lat"])

raw["rule"] = pd.NA
raw.loc[(raw.lon == 0) & (raw.lat == 0), "rule"] = "R1 null island"
raw.loc[(~ok(raw.lon, raw.lat)) & ok(raw.lat, raw.lon) & raw.rule.isna(),
        "rule"] = "R2 lon/lat swapped"
raw.loc[(~ok(raw.lon, raw.lat)) & raw.rule.isna(), "rule"] = "R3 outside region"

print(raw.rule.value_counts(dropna=False).rename("n").to_string())
kept = raw[raw.rule.isna()].copy()
recov = raw[raw.rule == "R2 lon/lat swapped"].copy()
recov[["lon", "lat"]] = recov[["lat", "lon"]].to_numpy()
final = pd.concat([kept, recov], ignore_index=True).drop(columns="rule")
inc_sol = gpd.GeoDataFrame(final, geometry=gpd.points_from_xy(final.lon, final.lat),
                           crs=CRS_WGS84).to_crs(CRS_UTM)
print(f"\n  kept outright   : {len(kept)}")
print(f"  recovered (R2)  : {len(recov)}")
print(f"  quarantined     : {len(raw) - len(kept) - len(recov)} "
      f"(R1 + R3 - unrecoverable)")
print(f"  final clean set : {len(inc_sol)}")
print(f"  all inside the study area: "
      f"{bool(inc_sol.geometry.within(LAND_GEOM.buffer(2000)).all())}")

# ================================================== SOLUTION 1.5 ============
print("\n\nSOLUTION 1.5 - SHAPELY REASONING FOR 'Marnvik'")
m = districts[districts.name == "Marnvik"]
g = m.geometry.iloc[0]
area_m2, perim_m = g.area, g.length          # NOT `A, P` - those are in use
print(f"  area          : {area_m2/1e6:,.3f} km^2")
print(f"  perimeter     : {perim_m/1000:,.3f} km")
print(f"  Polsby-Popper : {4*np.pi*area_m2/perim_m**2:.4f}   (1.0 = perfect circle)")
nb = districts[districts.geometry.touches(g)]
print(f"  shares a boundary with ({len(nb)}): {nb.name.tolist()}")
print(f"  centroid inside the polygon?      : {g.contains(g.centroid)}")
print(f"  centroid                : ({g.centroid.x:,.0f}, {g.centroid.y:,.0f})")
print(f"  representative_point()  : ({g.representative_point().x:,.0f}, "
      f"{g.representative_point().y:,.0f})")
er = g.buffer(-1000)
print(f"  after a 1 km inward buffer: {er.area/1e6:,.3f} km^2 "
      f"(-{(area_m2-er.area)/1e6:,.3f} km^2, "
      f"{100*(area_m2-er.area)/area_m2:.1f} % of the original)")

# ================================================== SOLUTION 1.6 ============
print("\n\nSOLUTION 1.6 - DISTANCE PROFILE FOR THE 24 STATIONS")
targets = {
    "coast":    coastline.geometry.union_all(),
    "river":    rivers.geometry.union_all(),
    "road":     roads[roads.road_class.isin(["motorway", "primary"])].geometry.union_all(),
    "hospital": facilities[facilities.facility_type == "hospital"].geometry.union_all(),
}
prof = stations[["station_id", "name", "station_type"]].copy()
for k, geom in targets.items():
    prof[f"d_{k}_km"] = (stations.geometry.distance(geom) / 1000).round(2)
dcols = [c for c in prof.columns if c.startswith("d_")]
prof["total_km"] = prof[dcols].sum(axis=1).round(2)
prof = prof.sort_values("total_km")
print(prof.to_string(index=False))
print(f"\n  MOST CONNECTED : {prof.iloc[0]['name']} "
      f"(total {prof.iloc[0]['total_km']:.1f} km)")
print(f"  MOST REMOTE    : {prof.iloc[-1]['name']} "
      f"(total {prof.iloc[-1]['total_km']:.1f} km)")
'''),

md(r'''
### 1.7 Spatial join and rates · 1.8 Raster sampling and regression
'''),

code(r'''
# ================================================== SOLUTION 1.7 ============
print("SOLUTION 1.7 - BUS STOPS PER 10,000 RESIDENTS")
bl = gpd.read_file(GPKG, layer="census_blocks")
bl = bl[~bl.geometry.is_empty]
st = gpd.read_file(GPKG, layer="bus_stops")

j = gpd.sjoin(st[["stop_id", "geometry"]], bl[["block_id", "district_id", "geometry"]],
              how="inner", predicate="within")
assert len(j) <= len(st), "join multiplied rows"
per_d = j.groupby("district_id").size().rename("n_stops")
pop_d = bl.groupby("district_id").population.sum().rename("population")

res = (districts[["district_id", "name", "district_type", "geometry"]]
       .merge(per_d, on="district_id", how="left")
       .merge(pop_d, on="district_id", how="left"))
res["n_stops"] = res.n_stops.fillna(0).astype(int)
res["stops_per_10k"] = np.where(res.population > 0,
                                1e4 * res.n_stops / res.population, np.nan)
print(res.sort_values("stops_per_10k", ascending=False)
      [["district_id", "name", "district_type", "population", "n_stops",
        "stops_per_10k"]].round(2).head(10).to_string(index=False))
print("\n  by district type:")
# NOTE: do not name a column "pop" - DataFrame.pop is a method, so `d.pop`
# returns the method rather than the column. Use d["pop"] or a different name.
print(res.groupby("district_type").agg(
    stops=("n_stops", "sum"), residents=("population", "sum")).assign(
    per_10k=lambda d: (1e4 * d["stops"] / d["residents"]).round(2)).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
res.plot(ax=axes[0], column="n_stops", cmap="Greens", scheme="naturalbreaks", k=5,
         legend=True, legend_kwds={"loc": "lower left", "fontsize": 6},
         edgecolor="grey", linewidth=0.4)
axes[0].set_title("Bus stops (COUNT)", fontsize=10, weight="bold", loc="left")
res.plot(ax=axes[1], column="stops_per_10k", cmap="Greens", scheme="naturalbreaks",
         k=5, legend=True, legend_kwds={"loc": "lower left", "fontsize": 6},
         edgecolor="grey", linewidth=0.4,
         missing_kwds={"color": "#dddddd", "hatch": "///", "label": "no population"})
axes[1].set_title("Bus stops per 10,000 residents (RATE)",
                  fontsize=10, weight="bold", loc="left")
for a in axes:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()

print("""
  WHICH IS MORE HONEST?
  Neither alone. The COUNT map shows where the service physically exists and is
  what an operator needs. The RATE map shows provision per person and is what an
  equity assessment needs - and it flatters tiny upland districts where three
  stops serve 700 people. Publish both, and always show the denominator: a rate
  computed on a population of 743 is not comparable with one computed on 197,870.
  The worst-served type by RATE is typically upland_rural; by COUNT it is also
  upland_rural, which is the rare case where the two agree.""")

# ================================================== SOLUTION 1.8 ============
print("\n\nSOLUTION 1.8 - RASTER SAMPLING AND THE RAINFALL-ELEVATION LAW")
cent = bl.geometry.representative_point()
coords = [(p.x, p.y) for p in cent]
samp = pd.DataFrame({"block_id": bl.block_id.to_numpy()})
for nm, fn in [("elev", "dem_25m.tif"), ("rain", "rainfall_annual_250m.tif"),
               ("ndvi", "ndvi_50m.tif")]:
    with rasterio.open(RAS / fn) as src:
        v = np.array([x[0] for x in src.sample(coords)], dtype=float)
        v[v == src.nodata] = np.nan
    samp[nm] = v
    print(f"  {nm:<6}: {int(np.isnan(v).sum())} NoData of {len(v)} blocks")
print("""    elev and rain return NO NoData: representative_point() is guaranteed
    to lie inside its block, and every block is on land. ndvi loses 6 blocks -
    those centroids fall inside the two simulated cloud gaps, which are stored
    as NoData. Had we used .centroid instead of .representative_point(), coastal
    blocks with concave shapes could have produced sea-cell NoData as well.""")

d = samp.dropna(subset=["elev", "rain"])
slope_hat, inter_hat = np.polyfit(d.elev, d.rain, 1)
r = np.corrcoef(d.elev, d.rain)[0, 1]
n = len(d)
se = np.sqrt(((d.rain - (inter_hat + slope_hat*d.elev))**2).sum() / (n-2)
             / ((d.elev - d.elev.mean())**2).sum())
print(f"\n  OLS  rainfall = {inter_hat:.2f} + {slope_hat:.4f} * elevation")
print(f"       n = {n},  r = {r:.4f},  R^2 = {r**2:.4f}")
print(f"       95 % CI for the slope: "
      f"[{slope_hat-1.96*se:.4f}, {slope_hat+1.96*se:.4f}]")
print(f"  TRUE generating coefficient : 0.6200")
print(f"  bias                        : {slope_hat-0.62:+.4f} "
      f"({100*(slope_hat-0.62)/0.62:+.1f} %)")
print("""
  WHY IS IT NOT EXACT?
  1. The generating law also contains a south-to-north gradient
     (+150 mm across the basin) which is OMITTED here. Elevation and northing
     are correlated, so the elevation coefficient absorbs part of that gradient
     - textbook omitted-variable bias (Lesson A9).
  2. We sample the 250 m rainfall raster at a POINT, so each block gets one
     6.25-hectare cell average rather than its true block mean.
  3. Additive noise was added to the rainfall field at generation time.
  Add a `north` term to the regression and the elevation coefficient moves
  materially closer to 0.62 - try it.""")
'''),

]

CELLS += [

md(r'''
## Solutions — Module 2 (Intermediate)

### 2.1 CRS choice · 2.2 QA report · 2.3 Riparian compliance
'''),

code(r'''
from pyproj import Geod
geod = Geod(ellps="WGS84")

# ================================================== SOLUTION 2.1 ============
print("SOLUTION 2.1 - A DEFENSIBLE CRS CHOICE")
pa_ll = protected.to_crs(CRS_WGS84)
riv_ll = rivers.to_crs(CRS_WGS84)
true_area = sum(abs(geod.geometry_area_perimeter(g)[0]) for g in pa_ll.geometry) / 1e6
true_len = sum(geod.geometry_length(g) for g in riv_ll.geometry) / 1000

# an azimuthal equidistant CRS centred on the study area (best for distance)
c = districts.to_crs(CRS_WGS84).geometry.union_all().centroid
AEQD = f"+proj=aeqd +lat_0={c.y:.5f} +lon_0={c.x:.5f} +datum=WGS84 +units=m +no_defs"

rows = []
for lab, code_ in [("EPSG:32633 UTM 33N (conformal)", CRS_UTM),
                   ("EPSG:3857  Web Mercator", CRS_WEBMERC),
                   ("ESRI:54009 Mollweide (equal area)", "ESRI:54009"),
                   ("Azimuthal equidistant (local)", AEQD)]:
    a = protected.to_crs(code_).geometry.area.sum() / 1e6
    L = rivers.to_crs(code_).geometry.length.sum() / 1000
    rows.append({"CRS": lab, "protected km2": round(a, 3),
                 "area err %": round(100*(a-true_area)/true_area, 3),
                 "river km": round(L, 3),
                 "len err %": round(100*(L-true_len)/true_len, 3)})
print(f"  geodesic ground truth: area {true_area:,.3f} km^2, "
      f"length {true_len:,.3f} km")
print(pd.DataFrame(rows).to_string(index=False))
print("""
  RECOMMENDATION
  Publish the UTM 33N figures for BOTH quantities. UTM is conformal, the study
  area lies wholly inside zone 33, and both its area and its length errors are
  under 0.1 % - confirmed independently by Mollweide (equal-area) for area and
  by the local azimuthal-equidistant projection for length. Web Mercator is
  unusable for either. Mollweide is excellent for area and poor for length, and
  azimuthal equidistant is the reverse, so neither is a good single choice.
  Report the CRS alongside the numbers; a figure without its CRS is not a
  measurement.""")

# ================================================== SOLUTION 2.2 ============
print("\n\nSOLUTION 2.2 - QA REPORT ACROSS EVERY LAYER")
def qa_report(gdf, name):
    gt = gdf.geom_type.dropna()
    poly = gt.isin(["Polygon", "MultiPolygon"]).any()
    line = gt.isin(["LineString", "MultiLineString"]).any()
    zero = int(((gdf.geometry.area == 0) if poly else
                (gdf.geometry.length == 0) if line else
                pd.Series(False, index=gdf.index)).sum())
    return pd.DataFrame([dict(
        layer=name, rows=len(gdf), crs=gdf.crs.to_string() if gdf.crs else "NONE",
        geom_types="/".join(sorted(gt.unique())),
        null=int(gdf.geometry.isna().sum()),
        empty=int(gdf.geometry.is_empty.sum()),
        invalid=int((~gdf.geometry.is_valid).sum()),
        dup_geom=int(gdf.geometry.to_wkb().duplicated().sum()),
        zero_measure=zero,
        total=round(gdf.geometry.area.sum()/1e6 if poly else
                    gdf.geometry.length.sum()/1000 if line else 0, 3),
        unit="km2" if poly else "km" if line else "-")])

ALL = {**{n: gpd.read_file(GPKG, layer=n) for n, _ in pyogrio.list_layers(GPKG)},
       "roads.geojson": gpd.read_file(VEC / "roads.geojson").to_crs(CRS_UTM),
       "protected.geojson": gpd.read_file(VEC / "protected_areas.geojson").to_crs(CRS_UTM)}
before = pd.concat([qa_report(g, n) for n, g in ALL.items()], ignore_index=True)
print("BEFORE REPAIR")
print(before.to_string(index=False))

FIXED = {}
for n, g in ALL.items():
    g = g.copy()
    if (~g.geometry.is_valid).any():
        g["geometry"] = g.geometry.make_valid()
        g = g.explode(index_parts=False, ignore_index=True)
        keep = ["Polygon", "MultiPolygon"] if g.geom_type.isin(
            ["Polygon", "MultiPolygon"]).any() else list(g.geom_type.unique())
        g = g[g.geom_type.isin(keep)]
    g = g[~g.geometry.isna() & ~g.geometry.is_empty]
    g = g[~g.geometry.to_wkb().duplicated()] if n == "landuse" else g
    FIXED[n] = g.reset_index(drop=True)
after = pd.concat([qa_report(g, n) for n, g in FIXED.items()], ignore_index=True)
print("\nAFTER REPAIR (only the layers that changed)")
chg = after.merge(before, on="layer", suffixes=("_after", "_before"))
chg = chg[(chg.rows_after != chg.rows_before) |
          (chg.total_after != chg.total_before)]
print(chg[["layer", "rows_before", "rows_after", "total_before", "total_after",
           "unit_after"]].to_string(index=False))

# ================================================== SOLUTION 2.3 ============
print("\n\nSOLUTION 2.3 - RIPARIAN BUFFER COMPLIANCE")
WIDTH = {2: 50.0, 3: 120.0, 4: 200.0}
rb = rivers.copy()
rb["geometry"] = rivers.geometry.buffer(rivers.strahler_order.map(WIDTH).to_numpy())
strip = rb.geometry.union_all()                       # DISSOLVED: no double counting

bpt = buildings_clean.copy()
bpt["geometry"] = buildings_clean.geometry.representative_point()
viol = bpt[bpt.geometry.within(strip)].copy()
print(f"  protected strip area      : {strip.area/1e4:,.1f} ha "
      f"({100*strip.area/LAND_GEOM.area:.2f} % of the basin)")
print(f"  buildings in violation    : {len(viol):,} of {len(bpt):,} "
      f"({100*len(viol)/len(bpt):.2f} %)")
print(f"  total value in violation  : {viol.value_kvs.sum():,.0f} k VS")
print(f"  (each building counted ONCE - the strips were dissolved first)")

vd = gpd.sjoin(viol, districts[["district_id", "name", "geometry"]],
               predicate="within").drop_duplicates("building_id")
alld = gpd.sjoin(bpt, districts[["district_id", "name", "geometry"]],
                 predicate="within").drop_duplicates("building_id")
rate = (vd.groupby("name").size().rename("violations").to_frame()
        .join(alld.groupby("name").size().rename("buildings"), how="right")
        .fillna(0))
rate["per_1000"] = (1000 * rate.violations / rate.buildings).round(1)
print("\n  worst districts by violation RATE:")
print(rate.sort_values("per_1000", ascending=False).head(6).to_string())

lu_strip = gpd.overlay(landuse_clean[["landuse_class", "geometry"]],
                       gpd.GeoDataFrame(geometry=[strip], crs=CRS_UTM),
                       how="intersection", keep_geom_type=True)
lu_strip["ha"] = lu_strip.geometry.area / 1e4
comp = (100 * lu_strip.groupby("landuse_class").ha.sum()
        / lu_strip.ha.sum()).round(1).sort_values(ascending=False)
print(f"\n  land-cover composition of the protected strip (%):")
print(comp.to_string())
print(f"  -> {comp.get('Built-up', 0):.1f} % of the strip is already built up")

fig, ax = fresh_ax((9, 7), "Riparian buffer violations")
land.plot(ax=ax, facecolor="#f7f5ef", edgecolor="#d8d2c4", linewidth=0.5)
gpd.GeoSeries([strip], crs=CRS_UTM).plot(ax=ax, facecolor="#9ecae1",
                                         edgecolor="#3182bd", alpha=0.6)
bpt.plot(ax=ax, color="#cccccc", markersize=0.5)
viol.plot(ax=ax, color="crimson", markersize=3)
rivers.plot(ax=ax, color="#08519c", linewidth=0.8)
ax.set_title(f"{len(viol):,} buildings inside the statutory riparian strip",
             fontsize=11, weight="bold", loc="left")
plt.show()
'''),

md(r'''
### 2.4 Reservoir impact · 2.5 Areal interpolation · 2.6 Zonal at two resolutions
'''),

code(r'''
# ================================================== SOLUTION 2.4 ============
print("SOLUTION 2.4 - PROPOSED RESERVOIR IMPACT ASSESSMENT")
vallmara = rivers[rivers.name == "Vallmara River"].geometry.union_all()
corridor = vallmara.buffer(400)

with rasterio.open(RAS / "dem_25m.tif") as src:
    dem_r = src.read(1, masked=True)
    tr_r, sh_r = src.transform, src.shape
low = (dem_r.filled(9999) < 30)
corr_mask = rasterize([(corridor, 1)], out_shape=sh_r, transform=tr_r,
                      fill=0, dtype="uint8").astype(bool)
res_mask = low & corr_mask & ~dem_r.mask

polys = [Polygon(g["coordinates"][0], g["coordinates"][1:])
         for g, v in shapes(res_mask.astype("uint8"), mask=res_mask, transform=tr_r)]
from shapely.ops import unary_union
reservoir = unary_union([q.buffer(0) for q in polys]).buffer(25).buffer(-25)
print(f"  reservoir footprint : {reservoir.area/1e6:,.2f} km^2 "
      f"({reservoir.area/1e4:,.0f} ha)")

lost = gpd.overlay(landuse_clean[["landuse_class", "geometry"]],
                   gpd.GeoDataFrame(geometry=[reservoir], crs=CRS_UTM),
                   how="intersection", keep_geom_type=True)
lost["ha"] = lost.geometry.area / 1e4
print("\n  land use inundated (ha):")
print(lost.groupby("landuse_class").ha.sum().round(1)
      .sort_values(ascending=False).to_string())

inund = bpt[bpt.geometry.within(reservoir)]
print(f"\n  buildings inundated : {len(inund):,}")
print(f"  asset value lost    : {inund.value_kvs.sum():,.0f} k VS")
print(f"  by use type         : {inund.use_type.value_counts().to_dict()}")

# dasymetric displacement estimate
blv = blocks[~blocks.geometry.is_empty].copy()
bptn = buildings_clean.copy()
bptn["geometry"] = buildings_clean.geometry.representative_point()
bptn["_ls"] = bptn.footprint_m2 * bptn.floors
bb = gpd.sjoin(bptn, blv[["block_id", "geometry"]], predicate="within")
tot = bb.groupby("block_id")["_ls"].sum()
inn = bb[bb.geometry.within(reservoir)].groupby("block_id")["_ls"].sum()
w = (inn / tot).reindex(blv.block_id).fillna(0).clip(0, 1).to_numpy()
displaced = float((blv.population.to_numpy() * w).sum())
print(f"  people displaced (dasymetric) : {displaced:,.0f}")
print(f"  people displaced (areal)      : "
      f"{float((blv.population * (blv.geometry.intersection(reservoir).area / blv.geometry.area)).sum()):,.0f}")

# ================================================== SOLUTION 2.5 ============
print("\n\nSOLUTION 2.5 - AREAL INTERPOLATION OF AN INTENSIVE VARIABLE")
inc_d = socio_clean[["district_id", "median_income_vs"]]
bi = blv[["block_id", "district_id", "geometry"]].merge(inc_d, on="district_id",
                                                        how="left")
bi["area_km2"] = bi.geometry.area / 1e6
back = bi.groupby("district_id").apply(
    lambda g: np.average(g.median_income_vs, weights=g.area_km2)
    if g.median_income_vs.notna().all() else np.nan, include_groups=False)
chk = inc_d.set_index("district_id").join(back.rename("recovered"))
chk["diff"] = (chk.recovered - chk.median_income_vs).round(6)
print(chk.head(8).round(2).to_string())
print(f"\n  max |difference| : {chk['diff'].abs().max():.6f}")
print("""
  WHY IS THE RECOVERY EXACT?
  Because we assigned every block the SAME district value and then took an
  area-weighted mean of identical numbers - which returns that number. The
  round trip is exact and completely uninformative: we have not estimated
  anything, only redistributed a constant.

  WHAT THIS TELLS YOU ABOUT INTENSIVE VARIABLES
  Median income is INTENSIVE - it does not add up. Areal interpolation is valid
  for EXTENSIVE quantities (population, counts, money totals), where splitting a
  polygon splits the quantity. Applying it to an intensive variable gives you
  back a constant within each source zone: the apparent block-level detail is
  entirely fictitious. To estimate income at block level you need an ancillary
  correlate (building value, floorspace, land cover) and a dasymetric or
  regression-based downscaling - not areal weighting.""")

# ================================================== SOLUTION 2.6 ============
print("\n\nSOLUTION 2.6 - ZONAL STATISTICS AT TWO RESOLUTIONS")
native = zonal_stats(districts, RAS / "rainfall_annual_250m.tif",
                     stats=("mean", "count"))
with rasterio.open(RAS / "rainfall_annual_250m.tif") as src:
    prof25 = src.profile.copy()
up = align_to(RAS / "rainfall_annual_250m.tif",
              {"height": 1440, "width": 1920,
               "transform": rasterio.Affine(25, 0, 400000, 0, -25, 4636000),
               "crs": CRS_UTM}, Resampling.bilinear)
up_path = OUT / "rain_upsampled_25m.tif"
pu = dict(driver="GTiff", height=1440, width=1920, count=1, dtype="float32",
          crs=CRS_UTM, nodata=-9999.0,
          transform=rasterio.Affine(25, 0, 400000, 0, -25, 4636000),
          compress="deflate")
with rasterio.open(up_path, "w", **pu) as dst:
    dst.write(np.nan_to_num(up, nan=-9999.0).astype("float32"), 1)
upz = zonal_stats(districts, up_path, stats=("mean", "count"))

cmp2 = pd.DataFrame({
    "district": districts.name,
    "cells_250m": native["count"].to_numpy(),
    "mean_250m": native["mean"].round(3).to_numpy(),
    "cells_25m": upz["count"].to_numpy(),
    "mean_25m": upz["mean"].round(3).to_numpy()})
cmp2["diff_mm"] = (cmp2.mean_25m - cmp2.mean_250m).round(3)
print(cmp2.head(10).to_string(index=False))
print(f"\n  max |difference| : {cmp2.diff_mm.abs().max():.3f} mm "
      f"({100*cmp2.diff_mm.abs().max()/cmp2.mean_250m.mean():.3f} % of the mean)")
print(f"  cells per district: {cmp2.cells_250m.mean():,.0f} at 250 m "
      f"-> {cmp2.cells_25m.mean():,.0f} at 25 m (100x more)")
print("""
  ARE THEY IDENTICAL? Nearly, but not exactly.
  Two effects. (1) Bilinear upsampling SMOOTHS: each fine cell is an
  interpolated blend of its coarse neighbours, so values near a district
  boundary are pulled towards the neighbouring district. (2) The fine grid
  resolves the district boundary far better, so the set of cells assigned to
  each district changes slightly.

  WHICH WOULD I REPORT? The 250 m native figure.
  Upsampling creates 100x more numbers and exactly zero extra information. The
  differences you see are artefacts of the interpolation, not a better estimate.
  Upsample only when you must align grids for cell-by-cell arithmetic, and then
  do the ANALYSIS at the coarsest resolution involved.""")
'''),

md(r'''
### 2.7 The analysis-ready feature table · 2.8 Challenge: recover the rainfall law
'''),

code(r'''
# ================================================== SOLUTION 2.7 ============
# The full solution is the code of Lesson A0, which builds exactly this table
# and saves it to data/outputs/block_features.gpkg. Here we verify it meets the
# specification rather than rebuilding it.
print("SOLUTION 2.7 - VERIFYING THE ANALYSIS-READY FEATURE TABLE")
spec = {
    "Identity":   ["block_id", "district_id", "district_type"],
    "Demography": ["population", "households", "pop_density_km2", "area_km2"],
    "Terrain":    ["mean_elev_m", "mean_slope_deg", "min_elev_m"],
    "Climate":    ["mean_rainfall_mm", "mean_lst_c"],
    "Vegetation": ["mean_ndvi", "pct_forest", "pct_builtup"],
    "Hazard":     ["pct_in_flood100", "pct_in_flood500", "dist_river_m"],
    "Access":     ["dist_hospital_m", "dist_clinic_m", "dist_school_m",
                   "dist_primary_road_m", "n_bus_stops"],
    "Assets":     ["n_buildings", "total_value_kvs", "mean_building_age"],
}
FT = gpd.read_file(OUT / "block_features.gpkg", layer="block_features")
print(f"  loaded {len(FT)} rows x {FT.shape[1]} columns from block_features.gpkg\n")
print(f"  {'group':<12}{'required':>10}{'present':>9}  missing")
allok = True
for grp, cols in spec.items():
    have = [c for c in cols if c in FT.columns]
    miss = [c for c in cols if c not in FT.columns]
    allok &= not miss
    print(f"  {grp:<12}{len(cols):>10}{len(have):>9}  {miss if miss else '-'}")
print(f"\n  ALL REQUIRED COLUMNS PRESENT: {allok}")

print("\n  QUALITY CHECKS")
lc_cols = [c for c in FT.columns if c.startswith("pct_") and "flood" not in c]
print(f"    land-cover percentages sum to 100 : "
      f"{bool(np.allclose(FT[lc_cols].sum(axis=1).dropna(), 100, atol=0.5))}")
print(f"    CRS is the analysis CRS           : {FT.crs.to_string() == CRS_UTM}")
print(f"    no negative distances             : "
      f"{bool((FT[[c for c in FT.columns if c.startswith('dist_')]] >= 0).all().all())}")
print(f"    counts are integers, zero-filled  : "
      f"{FT.n_buildings.dtype.kind in 'iu' and int(FT.n_buildings.min()) == 0}")
print(f"    means stay NaN where undefined    : "
      f"{int(FT.mean_building_age.isna().sum())} blocks with no buildings")
print(f"    population reconciles             : {FT.population.sum():,} vs "
      f"{blocks[~blocks.geometry.is_empty].population.sum():,} in the source")

# ================================================== SOLUTION 2.8 ============
print("\n\nSOLUTION 2.8 (CHALLENGE) - RECOVERING THE RAINFALL LAW")
TRUE = dict(intercept=470.0, elev=0.62, north=150.0)

def fit_rain(res_m, use_north=True, drop_nodata=True, label=""):
    with rasterio.open(RAS / "rainfall_annual_250m.tif") as src:
        rain = src.read(1, masked=not drop_nodata) if not drop_nodata \
               else src.read(1, masked=True)
        rt, rs = src.transform, src.shape
        nodata = src.nodata
    if res_m == 25:
        prof_ref = {"height": 1440, "width": 1920,
                    "transform": rasterio.Affine(25, 0, 400000, 0, -25, 4636000),
                    "crs": CRS_UTM}
        R = align_to(RAS / "rainfall_annual_250m.tif", prof_ref, Resampling.bilinear)
        E = align_to(RAS / "dem_25m.tif", prof_ref, Resampling.average)
        tr = prof_ref["transform"]; h, w = 1440, 1920
    else:
        R = np.where(rain.mask, np.nan, rain.data) if hasattr(rain, "mask") else rain
        E = align_to(RAS / "dem_25m.tif",
                     {"height": rs[0], "width": rs[1], "transform": rt, "crs": CRS_UTM},
                     Resampling.average)
        tr = rt; h, w = rs
    yy = tr.f + (np.arange(h) + 0.5) * tr.e
    NORTH = np.repeat(((yy - 4_600_000) / 36_000)[:, None], w, axis=1)
    if not drop_nodata:                       # deliberately keep the -9999s
        R = np.where(np.isfinite(R), R, -9999.0)
        E = np.where(np.isfinite(E), E, -9999.0)
        ok = np.ones_like(R, dtype=bool)
    else:
        ok = np.isfinite(R) & np.isfinite(E)
    cols = [np.ones(ok.sum()), E[ok]] + ([NORTH[ok]] if use_north else [])
    beta, *_ = np.linalg.lstsq(np.c_[tuple(cols)], R[ok], rcond=None)
    return label, int(ok.sum()), beta

print(f"  TRUE: rainfall = {TRUE['intercept']} + {TRUE['elev']}*elev "
      f"+ {TRUE['north']}*north\n")
print(f"  {'specification':<44}{'n cells':>10}{'intercept':>11}"
      f"{'b_elev':>9}{'b_north':>10}")
runs = [
    fit_rain(250, True,  True,  "(a) 250 m, north term, NoData excluded"),
    fit_rain(25,  True,  True,  "(b)  25 m, north term, NoData excluded"),
    fit_rain(250, False, True,  "(c) 250 m, NO north term"),
    fit_rain(250, True,  False, "(d) 250 m, NoData NOT excluded"),
]
for lab, n, b in runs:
    bn = f"{b[2]:>10.2f}" if len(b) > 2 else f"{'-':>10}"
    print(f"  {lab:<44}{n:>10,}{b[0]:>11.2f}{b[1]:>9.4f}{bn}")

# district-mean version
dz = pd.DataFrame({
    "rain": zonal_stats(districts, RAS / "rainfall_annual_250m.tif",
                        stats=("mean",))["mean"].to_numpy(),
    "elev": districts.mean_elev_m.to_numpy(),
    "north": ((districts.geometry.representative_point().y.to_numpy()
               - 4_600_000) / 36_000)})
bd = np.linalg.lstsq(np.c_[np.ones(len(dz)), dz.elev, dz.north], dz.rain,
                     rcond=None)[0]
print(f"  {'(e) DISTRICT means (n=24)':<44}{len(dz):>10,}{bd[0]:>11.2f}"
      f"{bd[1]:>9.4f}{bd[2]:>10.2f}")
print(f"\n  {'TRUE VALUES':<44}{'':>10}{470.00:>11.2f}{0.6200:>9.4f}{150.00:>10.2f}")
print("""
  WHICH EFFECT IS MOST DANGEROUS?
  (d), forgetting to exclude NoData, by an enormous margin. The -9999 sentinels
  are extreme, numerous and perfectly correlated between the two rasters (both
  are NoData over the sea), so the regression fits a line through two clouds:
  the real data and the sea. Every coefficient becomes meaningless while the
  R^2 looks superb.

  (c), omitting the north term, produces a modest but real bias in b_elev,
  because elevation and northing are correlated (the ridge runs north-east).
  Classic omitted-variable bias - visible, diagnosable, and survivable.

  (b), working at 25 m, changes almost nothing except the standard errors, which
  become absurdly small: 2.7 MILLION upsampled cells carry no more information
  than the 27,000 original ones. Any p-value computed from them is fiction.

  (e), district means, recovers the coefficients well but on n=24 - honest, low
  power, and the aggregation slightly inflates the fit (ecological correlation).""")
'''),

]

CELLS += [

md(r'''
## Solutions — Module 3 (Advanced)

These are open-ended problems. Each solution below is **one defensible
implementation**; where a different choice would be equally valid, the code says
so in a comment.

### 3.1 A defensible siting recommendation · 3.2 Network-distance accessibility
'''),

code(r'''
# ================================================== SOLUTION 3.1 ============
print("SOLUTION 3.1 - SITING TWO NEW CLINICS")
S = Fx.copy()
S["deprivation"] = S.district_id.map(
    socio_clean.set_index("district_id").median_income_vs).rank(pct=True)
S["deprivation"] = 1 - S.deprivation              # higher = more deprived

CRIT = {   # column, direction ("benefit"=more is better), rationale
    "population":          ("benefit", "people served"),
    "dist_clinic_m":       ("benefit", "currently underserved"),
    "deprivation":         ("benefit", "EQUITY: prioritise deprived areas"),
    "dist_primary_road_m": ("cost",    "must be reachable"),
    "pct_in_flood100":     ("cost",    "do not build in the floodplain"),
}
labels = list(CRIT)

def norm(v, direction, how):
    v = pd.Series(v).astype(float)
    r = v.rank(pct=True) if how == "rank" else (v - v.min()) / (v.max() - v.min())
    return r if direction == "benefit" else 1 - r

# AHP: pairwise importance of ROW over COLUMN
A = np.array([
    [1,   2,   2,   3,   4],      # population
    [1/2, 1,   1,   2,   3],      # underserved
    [1/2, 1,   1,   2,   3],      # deprivation
    [1/3, 1/2, 1/2, 1,   2],      # road access
    [1/4, 1/3, 1/3, 1/2, 1],      # flood avoidance
], dtype=float)
ev, evec = np.linalg.eig(A)
k = int(np.argmax(ev.real))
w = np.abs(evec[:, k].real); w /= w.sum()
n = len(labels); CI = (ev.real[k] - n) / (n - 1)
CR = CI / {3: .58, 4: .90, 5: 1.12}[n]
print(f"  AHP weights: " + ", ".join(f"{l}={x:.3f}" for l, x in zip(labels, w)))
print(f"  consistency ratio CR = {CR:.4f}  "
      f"{'ACCEPTABLE' if CR < 0.10 else 'REVISE'}")

def score(how, weights):
    N = np.c_[tuple(norm(S[c], d, how) for c, (d, _) in CRIT.items())]
    ws = (N * weights).sum(axis=1)
    gm = np.exp((np.log(np.clip(N, 1e-6, None)) * weights).sum(axis=1))
    return ws, gm

ws, gm = score("rank", w)
S["score_sum"], S["score_geom"] = ws, gm
print(f"\n  weighted sum vs geometric mean, Spearman: "
      f"{pd.Series(ws).corr(pd.Series(gm), method='spearman'):.3f}")
print("  We RECOMMEND the geometric mean: every criterion here is a genuine")
print("  requirement (a clinic in the floodplain or off the road network is")
print("  not viable), so a near-zero on any one should not be compensated away.")

# sensitivity over BOTH the weights and the normalisation method
rng = np.random.default_rng(5)
SIMS = 1500
top2 = np.zeros(len(S))
for _ in range(SIMS):
    wp = np.abs(w * rng.normal(1, 0.25, n)); wp /= wp.sum()
    how = "rank" if rng.random() < 0.5 else "minmax"
    _, g_ = score(how, wp)
    top2[np.argsort(-g_)[:2]] += 1
S["p_top2"] = top2 / SIMS
best = S.nlargest(6, "p_top2")
print(f"\n  ROBUSTNESS over {SIMS} runs (weights +/-25 % AND both normalisations)")
print(best[["block_id", "district_id", "population", "dist_clinic_m",
            "score_geom", "p_top2"]].round(3).to_string(index=False))
rec = best.head(2)
print(f"\n  RECOMMENDATION: build in blocks {rec.block_id.tolist()} "
      f"(districts {rec.district_id.tolist()})")
print(f"  These appear in the top 2 in {100*rec.p_top2.min():.0f}-"
      f"{100*rec.p_top2.max():.0f} % of runs.")
print(f"  THIS WOULD CHANGE IF: the equity criterion were dropped or down-weighted")
print(f"  below ~0.10, or if min-max normalisation were mandated (it lets the two")
print(f"  largest-population blocks dominate every other criterion).")

# ================================================== SOLUTION 3.2 ============
print("\n\nSOLUTION 3.2 - NETWORK DISTANCE AND THE DETOUR INDEX")
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from shapely.ops import unary_union

# STEP 1 - NODE the network. Road lines that cross without sharing a vertex are
# NOT connected in a graph built naively from their coordinates. unary_union
# splits every line at every intersection, which is the whole ball game: skip
# it and you get a forest of disconnected paths instead of a network.
noded = unary_union(roads.geometry.values)
segs = list(noded.geoms) if noded.geom_type == "MultiLineString" else [noded]
print(f"  {len(roads)} input segments -> {len(segs)} noded segments")

SNAP = 1.0                       # metres; after noding, shared vertices coincide
nodes, edges = {}, []
def nid(pt):
    key = (round(pt[0] / SNAP), round(pt[1] / SNAP))
    if key not in nodes:
        nodes[key] = len(nodes)
    return nodes[key]

for g in segs:
    cs = list(g.coords)
    for a_, b_ in zip(cs[:-1], cs[1:]):
        ia, ib = nid(a_), nid(b_)
        d = float(np.hypot(b_[0]-a_[0], b_[1]-a_[1]))
        if ia != ib and d > 0:
            edges.append((ia, ib, d)); edges.append((ib, ia, d))
NN = len(nodes)
r_, c_, v_ = zip(*edges)
Gm = coo_matrix((v_, (r_, c_)), shape=(NN, NN)).tocsr()
node_xy = np.zeros((NN, 2))
for (gx_, gy_), i in nodes.items():
    node_xy[i] = (gx_ * SNAP, gy_ * SNAP)

ncomp, comp = connected_components(Gm, directed=False)
sizes = np.bincount(comp)
big = int(np.argmax(sizes))
print(f"  graph: {NN:,} nodes, {len(edges)//2:,} edges, {ncomp} components")
print(f"  largest component holds {sizes[big]:,} nodes "
      f"({100*sizes[big]/NN:.0f} %) - we restrict the analysis to it")

ntree = cKDTree(node_xy)
def snap(gdf):
    """Nearest graph node AND the off-network walk needed to reach it."""
    return ntree.query(np.c_[gdf.geometry.x, gdf.geometry.y])

orig = Fx.sample(50, random_state=1)
o_pts = gpd.GeoDataFrame(geometry=orig.geometry.representative_point(), crs=CRS_UTM)
clin = facilities_clean[facilities_clean.facility_type == "clinic"]
o_snap, o_nodes = snap(o_pts)
c_snap, c_nodes = snap(clin)
print(f"  off-network snap: origins median {np.median(o_snap):,.0f} m, "
      f"clinics median {np.median(c_snap):,.0f} m")

# STEP 2 - total travel = walk on + travel along + walk off. Omitting the two
# snap legs is what makes a naive detour index come out below 1.0, which is
# physically impossible and an immediate sign the calculation is wrong.
Dnet = dijkstra(Gm, directed=False, indices=o_nodes)[:, c_nodes]
Dtot = Dnet + o_snap[:, None] + c_snap[None, :]
with np.errstate(invalid="ignore"):
    net_min = np.nanmin(np.where(np.isfinite(Dtot), Dtot, np.nan), axis=1)
euc_min = np.array([min(p.distance(q) for q in clin.geometry)
                    for p in o_pts.geometry])
ok = np.isfinite(net_min)
detour = net_min[ok] / np.maximum(euc_min[ok], 1)
print(f"\n  origin-clinic pairs connected on the network : {ok.sum()} of {len(o_pts)}")
print(f"  detour indices below 1.0 (must be none)      : {int((detour < 1).sum())}")
print(f"  DETOUR INDEX  median {np.median(detour):.3f}, "
      f"25th {np.percentile(detour, 25):.3f}, 75th {np.percentile(detour, 75):.3f}")
print(f"  (the mean is {detour.mean():.1f} - a few origins reach a clinic only by")
print(f"   a very indirect route, so use the MEDIAN as the correction factor)")

DET = float(np.clip(np.median(detour), 1.0, 3.0))
acc_euc = e2sfca(clin, "capacity", 10_000) * 1000
acc_net = e2sfca(clin, "capacity", 10_000 / DET) * 1000   # shrink the catchment
P_ = Fx.population.to_numpy(float)
print(f"\n  applying a detour correction of {DET:.2f} to the catchment radius:")
print(f"    catchment 10.0 km straight-line -> {10/DET:.1f} km effective")
print(f"    Gini, Euclidean catchment : {weighted_gini(acc_euc, P_):.4f}")
print(f"    Gini, corrected catchment : {weighted_gini(acc_net, P_):.4f}")
print(f"    blocks with NO clinic access: "
      f"{int((acc_euc == 0).sum())} -> {int((acc_net == 0).sum())}")
r1 = pd.Series(acc_euc).rank(ascending=False)
r2 = pd.Series(acc_net).rank(ascending=False)
print(f"    median |rank shift| : {(r1-r2).abs().median():.0f} places, "
      f"max {(r1-r2).abs().max():.0f}")
print("""
  INTERPRETATION
  The detour index is network distance divided by straight-line distance. A
  median around 1.3-2.0 is normal; this basin's is at the high end because the
  road network is sparse and follows the valleys.

  Three practical lessons. (1) NODE the network before building a graph, or
  crossing roads will not connect. (2) Add the off-network snap legs, or you
  will compute impossible detour indices below 1.0. (3) Check connectivity -
  only about half the graph is one component here, so a substantial share of
  origin-destination pairs is simply unreachable and must be reported as such
  rather than dropped silently.

  Correcting the catchment radius tightens every catchment, pushes marginal
  facilities out of reach and RAISES measured inequality. Euclidean
  accessibility therefore systematically UNDERSTATES the access deficit -
  exactly as the capstone's limitations section says.""")
'''),

md(r'''
### 3.3 Defensible hotspots · 3.4 Regionalisation for service delivery
'''),

code(r'''
# ================================================== SOLUTION 3.3 ============
print("SOLUTION 3.3 - A HOTSPOT ANALYSIS YOU CAN DEFEND")
HS = Fx.copy()          # NOT `HS` - that is the raster height
HS["value_at_risk"] = HS.total_value_kvs.fillna(0) * HS.pct_in_flood100 / 100.0
y3 = HS.value_at_risk.to_numpy(dtype=float)
n3 = len(HS)

# permutation budget must exceed n/alpha for FDR to be able to reject anything
NEED = int(np.ceil(n3 / 0.05))
PERM = max(9999, NEED)
print(f"  n = {n3}, alpha = 0.05  ->  need > {NEED:,} permutations; using {PERM:,}")

SCHEMES = {
    "queen contiguity": Wq,
    "k-nearest, k=4":   knn_weights(cxy, k=4),
    "k-nearest, k=8":   knn_weights(cxy, k=8),
    "distance band 3 km": None,
}
dm = np.sqrt(((cxy[:, None, :] - cxy[None, :, :]) ** 2).sum(-1))
Wdb = ((dm < 3000) & (dm > 0)).astype(float)
Wdb = Wdb / np.maximum(Wdb.sum(1, keepdims=True), 1e-12)
SCHEMES["distance band 3 km"] = Wdb

print(f"\n  {'weights scheme':<22}{'global I':>10}{'raw p<.05':>11}"
      f"{'Bonferroni':>12}{'FDR':>7}{'hot':>6}{'cold':>7}")
print("-" * 78)
results3 = {}
for nm, Wx in SCHEMES.items():
    gi3 = getis_ord_gstar(y3, Wx)
    p3 = 2 * (1 - _norm.cdf(np.abs(gi3)))
    sig3 = benjamini_hochberg(p3, 0.05)
    mI = moran_test(y3, Wx, permutations=999, seed=1)["I"]
    results3[nm] = (gi3, p3, sig3)
    print(f"  {nm:<22}{mI:>10.4f}{int((p3<0.05).sum()):>11}"
          f"{int((p3 < 0.05/n3).sum()):>12}{int(sig3.sum()):>7}"
          f"{int((sig3 & (gi3>0)).sum()):>6}{int((sig3 & (gi3<0)).sum()):>7}")
print("-" * 78)
gi_q, p_q, sig_q = results3["queen contiguity"]
HS["gi_z"], HS["sig"] = gi_q, sig_q
HS["cls"] = np.where(sig_q & (gi_q > 0), "hot",
             np.where(sig_q & (gi_q < 0), "cold", "not significant"))

# agreement across schemes
stack3 = np.c_[tuple((s & (g > 0)) for g, p, s in results3.values())]
HS["n_schemes_hot"] = stack3.sum(1)
print(f"\n  blocks flagged HOT under all {len(SCHEMES)} schemes : "
      f"{int((HS.n_schemes_hot == len(SCHEMES)).sum())}")
print(f"  blocks flagged HOT under at least one          : "
      f"{int((HS.n_schemes_hot >= 1).sum())}")
print("  Report only the blocks that survive EVERY scheme as confirmed hotspots.")

fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
HS.plot(ax=axes[0], column="value_at_risk", cmap="Purples", scheme="quantiles",
       k=6, legend=True, legend_kwds={"loc": "lower left", "fontsize": 6},
       edgecolor="none")
axes[0].set_title("Asset value at risk (k VS)", fontsize=9.5, weight="bold", loc="left")
for kk, cc in {"hot": "#b2182b", "cold": "#2166ac",
               "not significant": "#eeeeee"}.items():
    sub = HS[HS.cls == kk]
    if len(sub):
        sub.plot(ax=axes[1], color=cc, edgecolor="none", label=kk)
axes[1].legend(fontsize=7, loc="lower left")
axes[1].set_title(f"Gi* hotspots, FDR-corrected\nqueen weights, {PERM:,} perms",
                  fontsize=9.5, weight="bold", loc="left")
HS.plot(ax=axes[2], column="n_schemes_hot", cmap="YlOrRd", vmin=0,
       vmax=len(SCHEMES), legend=True, legend_kwds={"shrink": 0.6},
       edgecolor="none")
axes[2].set_title("Robustness: schemes flagging HOT", fontsize=9.5,
                  weight="bold", loc="left")
for a in axes:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()

# ================================================== SOLUTION 3.4 ============
print("\n\nSOLUTION 3.4 - SIX CONTIGUOUS SERVICE REGIONS")
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

ACCVARS = ["dist_hospital_m", "dist_clinic_m", "dist_fire_station_m",
           "n_bus_stops", "dist_primary_road_m"]
Xa = StandardScaler().fit_transform(
    np.nan_to_num(Fx[ACCVARS].to_numpy(float),
                  nan=np.nanmedian(Fx[ACCVARS].to_numpy(float))))
conn = ((Wq > 0).astype(int) + (Wq > 0).astype(int).T)

K6 = 6
ward6 = AgglomerativeClustering(n_clusters=K6, linkage="ward",
                                connectivity=conn).fit(Xa)
km6 = KMeans(n_clusters=K6, n_init=30, random_state=0).fit(Xa)
Fx["service_region"] = ward6.labels_

def wcss6(lab):
    return sum(((Xa[lab == c] - Xa[lab == c].mean(0))**2).sum()
               for c in np.unique(lab))
pop6 = Fx.groupby("service_region").population.sum()
frag6 = [1 if Fx[Fx.service_region == c].geometry.union_all().geom_type == "Polygon"
         else len(Fx[Fx.service_region == c].geometry.union_all().geoms)
         for c in range(K6)]
summ6 = pd.DataFrame({
    "population": pop6,
    "blocks": Fx.groupby("service_region").size(),
    "area_km2": Fx.groupby("service_region").area_km2.sum().round(1),
    "mean_dist_hosp_km": (Fx.groupby("service_region").dist_hospital_m.mean()/1000).round(2),
    "fragments": frag6})
print(summ6.to_string())
cv = pop6.std() / pop6.mean()
print(f"\n  population coefficient of variation : {cv:.3f} "
      f"(0 = perfectly equal)")
print(f"  all regions contiguous              : {all(f == 1 for f in frag6)}")
print(f"  WCSS, contiguity-constrained        : {wcss6(ward6.labels_):,.0f}")
print(f"  WCSS, unconstrained k-means         : {wcss6(km6.labels_):,.0f}")
print(f"  homogeneity cost of contiguity      : "
      f"{100*(wcss6(ward6.labels_)-wcss6(km6.labels_))/wcss6(km6.labels_):.1f} %")
print("""
  THE TRADE-OFF
  Ward-with-contiguity guarantees six administrable regions but cannot equalise
  population, because population is not one of the clustering variables and
  contiguity constrains which blocks may join. The CV above shows how unequal
  they are. To equalise population you would need a redistricting formulation
  (an objective with a population-balance penalty, solved by local search or
  integer programming) - Ward cannot express that constraint. State this: a
  homogeneity-based regionalisation and an equal-population redistricting are
  different optimisation problems with different answers.""")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
Fx.plot(ax=axes[0], column="service_region", categorical=True, cmap="Set3",
        legend=True, legend_kwds={"loc": "lower left", "fontsize": 7},
        edgecolor="white", linewidth=0.2)
axes[0].set_title(f"{K6} contiguous service regions (Ward + queen)",
                  fontsize=10, weight="bold", loc="left")
Fx.assign(km=km6.labels_).plot(ax=axes[1], column="km", categorical=True,
                               cmap="Set3", legend=True,
                               legend_kwds={"loc": "lower left", "fontsize": 7},
                               edgecolor="white", linewidth=0.2)
axes[1].set_title("Unconstrained k-means - NOT regions",
                  fontsize=10, weight="bold", loc="left")
for a in axes:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
### 3.5 A better flood model · 3.6 Risk under a climate scenario · 3.7 Challenge
'''),

code(r'''
# ================================================== SOLUTION 3.5 ============
print("SOLUTION 3.5 - AN IMPROVED FLOOD-SUSCEPTIBILITY MODEL")
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- (1) three NEW physically motivated features -------------------------
# profile curvature (2nd derivative of elevation) - concave ground collects water
ez = np.nan_to_num(grids["elevation"], nan=0.0)
gyy, gxx = np.gradient(ez, CELL, CELL)
curv = np.gradient(gxx, CELL, axis=1) + np.gradient(gyy, CELL, axis=0)
feat_rasters["curvature"] = np.where(np.isfinite(grids["elevation"]), curv, np.nan)
# a crude upslope-contributing-area proxy: how much higher ground drains here
low_rank = np.where(np.isfinite(grids["elevation"]), grids["elevation"], np.nan)
feat_rasters["upslope_proxy"] = -uniform_filter(
    np.nan_to_num(low_rank, nan=0.0), size=15, mode="nearest") + np.nan_to_num(low_rank)
# neighbourhood built-up fraction (drainage capacity / imperviousness)
built = (np.nan_to_num(grids["landcover"], nan=0) == 2).astype(float)
feat_rasters["pct_built_1km"] = 100 * uniform_filter(built, size=10, mode="nearest")

FEAT2 = FEATURES + ["curvature", "upslope_proxy", "pct_built_1km"]
def rebuild(ds):
    d = ds.copy()
    for f in ["curvature", "upslope_proxy", "pct_built_1km"]:
        d[f] = sample_grid(feat_rasters[f], d.x.to_numpy(), d.y.to_numpy())
    return d.dropna(subset=FEAT2)
dsB2 = rebuild(dsB)
X2 = dsB2[FEAT2].to_numpy(float); y2 = dsB2.flood.to_numpy(int)
g2 = spatial_blocks(dsB2.x.to_numpy(), dsB2.y.to_numpy(), 4000)
print(f"  features: {len(FEATURES)} -> {len(FEAT2)}   rows: {len(dsB2):,}")

# --- (2) block size from the residual autocorrelation range --------------
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X2, y2)
res2 = y2 - lr.predict_proba(X2)[:, 1]
pxy = np.c_[dsB2.x, dsB2.y]

# BINNED empirical semivariogram. gamma(h) = 0.5 * mean[(z_i - z_j)^2] over
# pairs separated by h. For an uncorrelated field gamma -> the VARIANCE (not
# half of it); the range is the lag at which gamma first reaches that sill.
tre = cKDTree(pxy)
EDGES = [0, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000]
sill = res2.var()
print(f"\n  empirical semivariogram of the OLS residuals  (sill = {sill:.4f})")
print(f"    {'lag bin (m)':>16}{'n pairs':>12}{'gamma':>10}{'gamma/sill':>12}")
prev = set()
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    cur = tre.query_pairs(hi)
    band = cur - prev
    prev = cur
    if len(band) < 50:
        continue
    idx = np.array(list(band))
    gam = 0.5 * np.mean((res2[idx[:, 0]] - res2[idx[:, 1]]) ** 2)
    print(f"    {lo:>7,}-{hi:<8,}{len(band):>12,}{gam:>10.4f}{gam/sill:>12.3f}")
first = None
print("    READ IT HONESTLY: gamma/sill is already ~1.0 in the SHORTEST lag bin,")
print("    which means the OLS residuals carry almost no short-range spatial")
print("    structure - the features have absorbed it. The range is therefore")
print("    below 500 m and any block size above that suffices. We keep 4 km")
print("    because the block-size sweep in A11 showed it to be the conservative")
print("    choice, not because the variogram demands it. A variogram that shows")
print("    NO structure is a useful result: it says your spatial CV can be")
print("    generous rather than punitive.")

# --- (3) three algorithms, spatially validated, then calibrated ---------
CVG = GroupKFold(n_splits=5)
cands = {
    "logistic":       make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "random forest":  RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                             random_state=0, n_jobs=-1),
    "grad. boosting": HistGradientBoostingClassifier(max_iter=300, random_state=0),
}
print(f"\n  {'model':<16}{'AUC':>8}{'AvgPrec':>10}{'Brier':>9}")
oofs = {}
for nm, mdl in cands.items():
    o = np.zeros(len(y2))
    for tr, te in CVG.split(X2, y2, groups=g2):
        m = mdl.fit(X2[tr], y2[tr])
        o[te] = m.predict_proba(X2[te])[:, 1]
    oofs[nm] = o
    print(f"  {nm:<16}{roc_auc_score(y2, o):>8.4f}"
          f"{average_precision_score(y2, o):>10.4f}{brier_score_loss(y2, o):>9.4f}")
best_nm = max(oofs, key=lambda k: roc_auc_score(y2, oofs[k]))
print(f"  best: {best_nm}")

cal = CalibratedClassifierCV(cands[best_nm], method="isotonic", cv=3)
o_cal = np.zeros(len(y2))
for tr, te in CVG.split(X2, y2, groups=g2):
    o_cal[te] = cal.fit(X2[tr], y2[tr]).predict_proba(X2[te])[:, 1]
print(f"  {best_nm} + isotonic calibration: "
      f"AUC {roc_auc_score(y2, o_cal):.4f}, Brier {brier_score_loss(y2, o_cal):.4f} "
      f"(was {brier_score_loss(y2, oofs[best_nm]):.4f})")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for nm, o in oofs.items():
    bins = np.quantile(o, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(o, bins[1:-1]), 0, 9)
    px = [o[idx == i].mean() for i in range(10)]
    py = [y2[idx == i].mean() for i in range(10)]
    axes[0].plot(px, py, "o-", label=nm, linewidth=1.6)
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
axes[0].set_xlabel("mean predicted probability"); axes[0].set_ylabel("observed frequency")
axes[0].set_title("Calibration (spatially validated)", fontsize=10,
                  weight="bold", loc="left")
axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

mdl_final = cands[best_nm].fit(X2, y2)
stack2 = np.stack([feat_rasters[f] for f in FEAT2], axis=-1)
vc2 = np.isfinite(stack2).all(-1) & land_g
p2 = mdl_final.predict_proba(stack2[vc2])[:, 1]
out2 = np.full((H_ := ref["height"], W_ := ref["width"]), np.nan)
out2[vc2] = p2
im = axes[1].imshow(out2, cmap="RdYlGn_r", vmin=0, vmax=1,
                    extent=(TR.c, TR.c + W_*CELL, TR.f - H_*CELL, TR.f))
rivers.plot(ax=axes[1], color="#08306b", linewidth=0.7)
plt.colorbar(im, ax=axes[1], shrink=0.75, label="P(flood-prone)")
axes[1].set_title(f"Improved susceptibility ({best_nm}, {len(FEAT2)} features)",
                  fontsize=10, weight="bold", loc="left")
axes[1].set_aspect("equal"); axes[1].set_xticks([]); axes[1].set_yticks([])
plt.tight_layout(); plt.show()

# ================================================== SOLUTION 3.6 ============
print("\n\nSOLUTION 3.6 - RISK UNDER A CLIMATE SCENARIO")
LEVELS_NOW = {10: 1.2, 25: 2.0, 50: 2.8, 100: 3.5, 250: 5.0, 500: 6.5}
# scenario: +0.5 m on every level, and the 100-yr event becomes the 50-yr event
LEVELS_FUT = {10: 1.7, 25: 2.5, 50: 4.0, 100: 4.5, 250: 5.5, 500: 7.0}

# building_id -> district_id, aligned positionally to Bpt
DIST_OF_BLD = Bpt.building_id.map(
    Bd.drop_duplicates("building_id").set_index("building_id").district_id).to_numpy()

def ead_for(levels, tag):
    losses_, per_d_ = {}, {}
    for T, lvl in levels.items():
        depth = np.clip(lvl - Bpt.hand_m.to_numpy(), 0, None)
        dr = damage_ratio(depth, constr)
        dr = np.where(Bpt.has_basement.to_numpy(), np.minimum(dr*1.15, 1.0), dr)
        loss = dr * Bpt.value_kvs.to_numpy()
        losses_[T] = loss.sum()
        per_d_[T] = pd.Series(loss, index=Bpt.index).groupby(DIST_OF_BLD).sum()
    Ts_ = np.array(sorted(levels)); ps_ = 1.0 / Ts_
    L_ = np.array([losses_[T] for T in Ts_])
    tot = abs(np.trapezoid(L_[::-1], ps_[::-1]))
    dis = {}
    for did in districts.district_id:
        Lv = np.array([per_d_[T].get(did, 0.0) for T in Ts_])
        dis[did] = abs(np.trapezoid(Lv[::-1], ps_[::-1]))
    n_exp = int((np.clip(max(levels.values()) - Bpt.hand_m, 0, None) > 0).sum())
    return tot, pd.Series(dis), n_exp

ead_now, d_now, exp_now = ead_for(LEVELS_NOW, "now")
ead_fut, d_fut, exp_fut = ead_for(LEVELS_FUT, "future")
print(f"  EAD today   : {ead_now:>12,.0f} k VS/yr")
print(f"  EAD scenario: {ead_fut:>12,.0f} k VS/yr   "
      f"({100*(ead_fut-ead_now)/ead_now:+.1f} %)")
print(f"  buildings exposed at the 500-yr level: {exp_now:,} -> {exp_fut:,} "
      f"({exp_fut-exp_now:+,})")

cmp6 = pd.DataFrame({"now": d_now, "future": d_fut})
cmp6["abs_change"] = cmp6.future - cmp6.now
cmp6["pct_change"] = (100 * cmp6.abs_change / cmp6.now.replace(0, np.nan)).round(1)
cmp6 = cmp6.join(districts.set_index("district_id")["name"])
print("\n  biggest ABSOLUTE increases:")
print(cmp6.nlargest(5, "abs_change")[["name", "now", "future", "abs_change"]]
      .round(0).to_string())
print("\n  biggest RELATIVE increases:")
print(cmp6.nlargest(5, "pct_change")[["name", "now", "future", "pct_change"]]
      .round(1).to_string())
print(f"\n  BREAK-EVEN DEFENCE COST")
extra = ead_fut - ead_now
for yrs, disc in [(20, 0.00), (20, 0.035), (50, 0.035)]:
    if disc == 0:
        pv = extra * yrs
    else:
        pv = extra * (1 - (1+disc)**-yrs) / disc
    print(f"    over {yrs} yr at {disc:.1%} discount : {pv:>12,.0f} k VS")
print("  A defence restoring today's EAD is worth building if it costs less")
print("  than the present value above.")

# ================================================== SOLUTION 3.7 ============
print("\n\nSOLUTION 3.7 (CHALLENGE) - RECOVERING THE FULL GENERATING PROCESS")
def ols_ci(Xd, yd, names):
    Xd = np.c_[np.ones(len(yd)), Xd]
    b, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    r = yd - Xd @ b
    s2 = (r**2).sum() / (len(yd) - Xd.shape[1])
    cov = s2 * np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    return pd.DataFrame({"term": ["intercept"] + names,
                         "estimate": b, "se": se,
                         "lo95": b - 1.96*se, "hi95": b + 1.96*se})

print("  (1) RAINFALL = 470 + 0.62*elev + 150*north")
Rg, Eg = grids["rainfall"], grids["elevation"]
hh, ww = Rg.shape
yy_ = ref["transform"].f + (np.arange(hh)+0.5)*ref["transform"].e
NORTH = np.repeat(((yy_ - 4_600_000)/36_000)[:, None], ww, axis=1)
m_ = np.isfinite(Rg) & np.isfinite(Eg)
t1 = ols_ci(np.c_[Eg[m_], NORTH[m_]], Rg[m_], ["elev", "north"])
t1["TRUE"] = [470.0, 0.62, 150.0]
print(t1.round(4).to_string(index=False))

print("\n  (2) LST = 31.5 - 0.0062*elev + 6.4*urban")
Ug = np.clip((grids["popdens"] - 22)/9500, 0, None) ** (1/2.1)
m2_ = np.isfinite(grids["lst"]) & np.isfinite(Eg) & np.isfinite(Ug)
t2 = ols_ci(np.c_[Eg[m2_], Ug[m2_]], grids["lst"][m2_], ["elev", "urban"])
t2["TRUE"] = [31.5, -0.0062, 6.4]
print(t2.round(5).to_string(index=False))

print("\n  (3) PM2.5 = 7.5 + 16*urban + 9*exp(-d/1800)   [24 stations]")
print(f"     fitted in Lesson I9: a=7.15, b=9.02, L=1855 m, c=16.42")
print(f"     TRUE:                a=7.50, b=9.00, L=1800 m, c=16.00")

print("\n  (4) POPULATION DENSITY = 9500*urban^2.1 + 22")
print("     This is an IDENTITY of our urban proxy, not an estimable relation:")
print("     we DEFINED urban by inverting it. Recovering 2.1 requires an")
print("     independent measure of urban intensity, which the delivered data")
print("     does not contain. Report it as unidentifiable.")

print("""
  WHICH ESTIMATE IS WORST, AND WHY?
  (3), the PM2.5 e-folding distance. Diagnosis: CONFOUNDING plus SMALL SAMPLE.
  Distance-to-motorway is correlated (-0.44) with urban intensity because the
  motorway follows the populated coast, and there are only 24 stations. Fitting
  the decay without the urban term returns L = 12,322 m against a true 1,800 m -
  an error of +585 %. FIX: include the confounder (as model B does), and if
  possible add stations chosen to break the correlation - sites far from the
  motorway but highly urban, and sites near it but rural.

  (4) is worse in a different sense: it is not merely badly estimated but
  UNIDENTIFIABLE from the delivered data. Recognising that a parameter cannot be
  estimated is more valuable than producing a confident number for it.

  (1) and (2) are recovered to within ~1 % because both have millions of cells,
  no confounding once both terms are included, and low noise. Note their
  standard errors are absurdly small - with 100,000+ spatially autocorrelated
  cells the EFFECTIVE sample size is far smaller than n, so those 95 % intervals
  are much too narrow (Lesson A9).""")
'''),

md(r'''
---

# End of course

You have worked through 49 lessons, four exercise sets and a full end-to-end
project. If you can now open an unfamiliar spatial dataset, tell within five
minutes whether its CRS, geometry and attributes can be trusted, engineer
features from its geography, validate a model against spatial leakage, and say
clearly what your analysis cannot support — then the course has done its job.

**Three things worth carrying forward.**

1. **The CRS is not metadata, it is the units of every number you produce.**
   Most wrong answers in applied GIS are wrong by a factor you could have
   predicted from the projection.

2. **Spatial autocorrelation is the default, not the exception.** It inflates
   your significance, leaks through your cross-validation, and confounds your
   coefficients. Test for it every time; the test costs milliseconds.

3. **Validate against something the analysis never used.** Recovering a known
   coefficient, reproducing an independently derived boundary, reconciling two
   layers that should agree — these are worth more than any goodness-of-fit
   statistic computed on your own training data.

### Where to go next

| Topic | Tools |
|---|---|
| Rigorous spatial statistics | **PySAL** — `libpysal`, `esda`, `spreg`, `spopt` |
| Network analysis and routing | `networkx`, `osmnx`, `pandana`, OSRM / Valhalla |
| Large-scale raster | `xarray`, `rioxarray`, `dask`, STAC + `odc-stac` |
| Cloud-native formats | Cloud-Optimized GeoTIFF, GeoParquet, Zarr, FlatGeobuf |
| Interactive maps | `folium`, `lonboard`, `pydeck`, `keplergl` |
| Geostatistics / kriging | `scikit-gstat`, `pykrige`, `gstools` |
| Spatial ML at scale | `verde`, `mlxtend` spatial CV, `spacv` |

The fictional Vallmara Basin has now served its purpose. Take the same habits to
real data — and be suspicious of it in exactly the same way.
'''),

]
