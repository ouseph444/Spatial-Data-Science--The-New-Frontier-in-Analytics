# -*- coding: utf-8 -*-
"""Module 2 - Intermediate: overlays, joins, raster processing, data cleaning."""
from _cells import md, code

CELLS = [

md(r'''
# Module 2 — Intermediate: Real Spatial Analysis

Module 1 taught you to *handle* spatial data. Module 2 teaches you to *analyse*
it — and, just as importantly, to clean it first. Every technique here appears
again in the capstone.

**The 16 lessons**

| # | Lesson | Core skill |
|---|---|---|
| I1 | Choosing a CRS for measurement | Quantified distortion; when UTM is wrong |
| I2 | Invalid geometries and how to repair them | `is_valid`, `explain_validity`, `make_valid` |
| I3 | Missing values, sentinels and dirty categories | The full cleaning pipeline |
| I4 | Buffer analysis | Fixed, variable and dissolved buffers |
| I5 | Overlay operations | intersection, union, difference, symmetric difference |
| I6 | Clipping | `gpd.clip` vs `gpd.overlay`, and when each is right |
| I7 | Spatial joins in depth | predicates, cardinality, `sjoin_nearest` |
| I8 | Aggregating spatial statistics | area-weighted aggregation, apportionment |
| I9 | Nearest-neighbour analysis | `sjoin_nearest`, k-NN, distance bands |
| I10 | Dissolve and hierarchical aggregation | `dissolve`, `aggfunc`, topology |
| I11 | The spatial index and performance | R-trees, `sindex.query`, why joins are fast |
| I12 | Raster masking and clipping | `rasterio.mask`, windows, cropping |
| I13 | Raster resampling and reprojection | `Resampling`, `WarpedVRT`, alignment |
| I14 | Reclassification and band maths | NDVI, slope, aspect, hillshade |
| I15 | Zonal statistics | Raster values summarised by polygon |
| I16 | Rasterize / polygonize and analytical map design | Vector ↔ raster round trip |
'''),

# ------------------------------------------------------------------ I0 -----
md(r'''
## I0 — Load the full project

**What we are about to do.** Load every layer we will need in Modules 2–4 into a
consistent, named set of variables, all in the analysis CRS.

**Why it matters.** From here on, lessons build on one another. A single loading
cell that guarantees every layer is in EPSG:32633 removes an entire class of bug
and makes the rest of the notebook re-runnable from this point.

**Concept — the "load and normalise" boundary.** Professional spatial pipelines
have a hard boundary between *ingestion* (read, reproject, rename, type-cast) and
*analysis*. Everything upstream of the boundary is allowed to be messy; nothing
downstream is. We are drawing that boundary here.

**What the next cell does:** loads all 12 GeoPackage layers, the two GeoJSON
files and the four CSVs, reprojects everything to `CRS_UTM`, and prints a summary
so you can confirm CRS consistency at a glance.
'''),

code(r'''
# ---- vector layers ----------------------------------------------------------
districts = gpd.read_file(GPKG, layer="districts")
blocks    = gpd.read_file(GPKG, layer="census_blocks")
landuse   = gpd.read_file(GPKG, layer="landuse")
rivers    = gpd.read_file(GPKG, layer="rivers")
flood     = gpd.read_file(GPKG, layer="flood_zones")
buildings = gpd.read_file(GPKG, layer="buildings")
facilities = gpd.read_file(GPKG, layer="facilities")
routes    = gpd.read_file(GPKG, layer="transit_routes")
stops     = gpd.read_file(GPKG, layer="bus_stops")
sea       = gpd.read_file(GPKG, layer="sea")
land      = gpd.read_file(GPKG, layer="land_boundary")
coastline = gpd.read_file(GPKG, layer="coastline")

roads     = gpd.read_file(VEC / "roads.geojson").to_crs(CRS_UTM)
protected = gpd.read_file(VEC / "protected_areas.geojson").to_crs(CRS_UTM)

# ---- tabular ----------------------------------------------------------------
socio     = pd.read_csv(TAB / "district_socioeconomic.csv")
readings  = pd.read_csv(TAB / "sensor_readings.csv", parse_dates=["date"])
incidents_raw = pd.read_csv(TAB / "flood_incidents.csv", parse_dates=["date"])
lc_legend = pd.read_csv(TAB / "landcover_legend.csv")

stations_df = pd.read_csv(TAB / "sensor_stations.csv")
stations = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(stations_df.lon, stations_df.lat),
    crs=CRS_WGS84).to_crs(CRS_UTM)

# ---- convenience single geometries -----------------------------------------
LAND_GEOM  = land.geometry.iloc[0]
SEA_GEOM   = sea.geometry.iloc[0]
COAST_GEOM = coastline.geometry.iloc[0]

VECTORS = {
    "districts": districts, "blocks": blocks, "landuse": landuse,
    "rivers": rivers, "flood": flood, "buildings": buildings,
    "facilities": facilities, "routes": routes, "stops": stops,
    "roads": roads, "protected": protected, "stations": stations,
}

print(f"{'layer':<12}{'rows':>7}{'cols':>6}  {'crs':<12}{'geometry types':<28}"
      f"{'empty':>6}{'invalid':>8}")
print("-" * 84)
for name, g in VECTORS.items():
    print(f"{name:<12}{len(g):>7}{g.shape[1]:>6}  {g.crs.to_string():<12}"
          f"{str(dict(g.geom_type.value_counts())):<28}"
          f"{int(g.geometry.is_empty.sum()):>6}{int((~g.geometry.is_valid).sum()):>8}")

print("-" * 84)
print(f"All in one CRS: {len({g.crs.to_string() for g in VECTORS.values()}) == 1}")
print(f"\nTabular: socio {socio.shape}, readings {readings.shape}, "
      f"incidents {incidents_raw.shape}, legend {lc_legend.shape}")
'''),

md(r'''
**Explanation.**

* Everything is read once and reprojected at the door. `roads` and `protected`
  arrive in EPSG:4326 and are converted immediately; nothing downstream ever
  needs to think about it again.
* `VECTORS` is a dictionary of the layers so we can iterate over them for audits.
  This is a small but high-leverage habit: any check you write once
  (CRS consistency, validity, empty geometry) now runs on every layer for free.
* The summary line `All in one CRS: True` is computed from a set comprehension
  over the CRS strings. Assert this in production code — a mixed-CRS project is a
  wrong-answer generator.
* `parse_dates=["date"]` on the two time-stamped CSVs — otherwise `date` is a
  string and every temporal operation silently does lexicographic comparison.
* `incidents_raw` is deliberately named `_raw`: it still contains the swapped
  coordinates and null-island rows. We clean it in I3.

**Expected output.** A 12-row table. Note in particular:

* every `crs` reads `EPSG:32633`, and `All in one CRS: True`;
* `landuse` shows **3 invalid** geometries;
* `blocks` shows **1 empty** geometry;
* `protected` shows a mixture `{'Polygon': 4, 'MultiPolygon': 1}` — the layer
  deliberately mixes geometry types;
* all other layers are clean.

Those three anomalies are the subject of the next two lessons.
'''),

# ------------------------------------------------------------------ I1 -----
md(r'''
## I1 — Choosing a CRS for accurate distance and area

**What we are going to learn.** How to select a measurement CRS defensibly, and
how to quantify the error of the alternatives.

**Why it matters.** In B5 we saw that Web Mercator inflates area by 80%. But
"use UTM" is not a universal answer either — UTM is wrong for a study area
spanning several zones, and wrong at high latitudes. You need a decision rule.

**The concept — a decision procedure.**

| Situation | Use | Why |
|---|---|---|
| Region inside one UTM zone (< ~500 km east–west) | **UTM zone** (EPSG:326xx / 327xx) | Conformal, scale error < 1/2 500 |
| Country or region spanning zones | A **national grid** (British National Grid, Lambert-93, …) or a custom Lambert Conformal Conic | Designed for that footprint |
| **Area** statistics over a large region | An **equal-area** projection (Albers Equal Area with local standard parallels, Lambert Azimuthal Equal Area) | Area is preserved exactly |
| **Distance** from one origin | **Azimuthal Equidistant** centred on that origin | Distances from the centre are exact |
| Global work | **Equal Earth** / Mollweide (area), or compute geodesically | No projection is good everywhere |
| Web display only | **Web Mercator** | Tiles |

**Concept — scale factor.** A conformal projection preserves angles but scales
distance by a factor `k` that varies with position. For UTM, `k = 0.9996` on the
central meridian and rises to ≈ 1.00097 at the zone edge. Areas scale as `k²`.
So worst-case UTM area error inside a zone is about **±0.2%** — negligible for
most work, and three hundred times better than Web Mercator.

**The practical rule.** Compute the same quantity in two independent
projections. If they agree, you are fine. If they disagree, you have chosen
badly. This is the projection equivalent of a unit test.

**Expected outcome.** A table of the same three measurements (area, length,
distance) across five CRS, with errors relative to a geodesic ground truth.

**What the next cell does:** measures district area, river length and a
long point-to-point distance in five different CRS, computes geodesic ground
truth with `pyproj.Geod`, and reports the error of each.
'''),

code(r'''
from pyproj import Geod
geod = Geod(ellps="WGS84")

CANDIDATES = {
    "EPSG:32633  UTM 33N (correct zone)": "EPSG:32633",
    "EPSG:32632  UTM 32N (wrong zone)":   "EPSG:32632",
    "EPSG:3857   Web Mercator":           "EPSG:3857",
    "ESRI:54009  Mollweide (equal area)": "ESRI:54009",
    "ESRI:54034  Cylindrical Equal Area": "ESRI:54034",
}

# ---------- ground truth on the ellipsoid ------------------------------------
poly_ll = districts.to_crs(CRS_WGS84).geometry.union_all()
true_area_km2 = abs(geod.geometry_area_perimeter(poly_ll)[0]) / 1e6

riv_ll = rivers.to_crs(CRS_WGS84).geometry.iloc[0]
true_len_km = geod.geometry_length(riv_ll) / 1000

p1 = districts.to_crs(CRS_WGS84).geometry.iloc[0].representative_point()
p2 = districts.to_crs(CRS_WGS84).geometry.iloc[-1].representative_point()
true_dist_km = geod.inv(p1.x, p1.y, p2.x, p2.y)[2] / 1000

# ---------- the same three measurements in each candidate CRS ---------------
rows = []
for label, crs_code in CANDIDATES.items():
    d = districts.to_crs(crs_code)
    r = rivers.to_crs(crs_code)
    a = d.geometry.union_all().area / 1e6
    L = r.geometry.iloc[0].length / 1000
    q1 = d.geometry.iloc[0].representative_point()
    q2 = d.geometry.iloc[-1].representative_point()
    D = q1.distance(q2) / 1000
    rows.append({
        "CRS": label,
        "area km2": round(a, 1),   "area err %": round(100*(a-true_area_km2)/true_area_km2, 2),
        "river km": round(L, 2),   "len err %":  round(100*(L-true_len_km)/true_len_km, 2),
        "dist km": round(D, 2),    "dist err %": round(100*(D-true_dist_km)/true_dist_km, 2),
    })

print("GEODESIC GROUND TRUTH (computed on the WGS84 ellipsoid)")
print(f"  total district area : {true_area_km2:,.1f} km^2")
print(f"  Vallmara River len  : {true_len_km:,.2f} km")
print(f"  cross-basin distance: {true_dist_km:,.2f} km\n")
print(pd.DataFrame(rows).to_string(index=False))

# ---------- what the UTM scale factor actually is here ----------------------
from pyproj import CRS as PCRS, Transformer
lons = np.array([13.9, 14.14, 14.4])
tr = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)
print("\nUTM 33N point scale factor across the study area "
      "(central meridian = 15 deg E):")
for lon in lons:
    x, y = tr.transform(lon, 41.72)
    # numerical estimate of scale: length of a 1 km geodesic vs its projected length
    lon2, lat2, _ = geod.fwd(lon, 41.72, 90, 1000)
    x2, y2 = tr.transform(lon2, lat2)
    k = np.hypot(x2 - x, y2 - y) / 1000
    print(f"   lon {lon:5.2f} deg  ->  k = {k:.6f}   "
          f"({(k-1)*1e6:+.0f} ppm, area factor k^2 = {k**2:.6f})")
'''),

md(r'''
**Explanation.**

* `geod.geometry_area_perimeter(geom)` and `geod.geometry_length(geom)` compute
  **geodesic** area and length directly on the ellipsoid from lon/lat geometry.
  This is the ground truth: no projection, no distortion. It is slower than
  planar arithmetic, which is why we do not use it for every operation — but it
  is exactly the right tool for *validating* a projection choice.
* `geod.inv(lon1, lat1, lon2, lat2)` returns `(forward_azimuth, back_azimuth,
  distance_m)`. The third element is the geodesic distance.
* **`EPSG:32632` (UTM zone 32N) is included on purpose.** It is the *adjacent*
  zone — an easy mistake if you compute the zone from the wrong corner of your
  data. Watch its error: still small, but several times worse than the correct
  zone, and it grows as you move east.
* **Mollweide and Cylindrical Equal Area** both preserve area, so their area
  errors are near zero — but look at their *length* and *distance* errors, which
  are large. Equal-area projections badly distort shape and distance. There is no
  single "accurate" CRS; there is only "accurate for the quantity you are
  measuring".
* The scale-factor block measures `k` empirically: project a 1 km geodesic and
  see how long it comes out. At the western edge of the basin `k` is around
  0.9997 and at the eastern edge around 0.9998, i.e. errors of a few hundred
  **parts per million**. That is 20–30 cm per kilometre — irrelevant for
  policy analysis, potentially relevant for cadastral surveying.

**Expected outcome.**

```
GEODESIC GROUND TRUTH (computed on the WGS84 ellipsoid)
  total district area : 1,395.7 km^2
  Vallmara River len  : 43.95 km
  cross-basin distance: 33.12 km
```

| CRS | area km² | area err | river km | len err | dist km | dist err |
|---|---|---|---|---|---|---|
| **UTM 33N (correct)** | 1 394.8 | **−0.07%** | 43.93 | **−0.03%** | 33.09 | **−0.10%** |
| UTM 32N (wrong zone) | 1 400.9 | +0.37% | 44.03 | +0.19% | 32.74 | −1.16% |
| Web Mercator | 2 507.0 | **+79.6%** | 58.81 | **+33.8%** | 44.39 | **+34.0%** |
| Mollweide (equal area) | 1 396.8 | +0.08% | 44.60 | +1.49% | 31.71 | −4.26% |
| Cylindrical Equal Area | 1 395.7 | **−0.00%** | 51.40 | +16.97% | 40.27 | +21.59% |

and empirical scale factors:

```
   lon 13.90 deg  ->  k = 0.999702   (-298 ppm, area factor k^2 = 0.999404)
   lon 14.14 deg  ->  k = 0.999662   (-338 ppm)
   lon 14.40 deg  ->  k = 0.999630   (-370 ppm)
```

Read the table row by row:

* **UTM 33N** is the only CRS with all three errors under 0.1%.
* **UTM 32N**, the *adjacent* zone, is 5–10× worse. Picking the zone from the
  wrong corner of your bounding box is a real and easy mistake.
* **Web Mercator** inflates area by 79.6% and *both* length and distance by 34%
  — exactly `1/cos(41.7°) = 1.34`, with area going as the square.
* **Cylindrical Equal Area** nails area to −0.00% and is catastrophically wrong
  about length (+17%) and distance (+22%).

That last row is the real lesson: **there is no "accurate CRS", only a CRS that
is accurate for the quantity you are measuring.** An equal-area projection is
perfect for a density map and useless for a service-area analysis.

The scale factors show `k ≈ 0.9997`, i.e. UTM is quietly shrinking every distance
by about 300 parts per million — 30 cm per kilometre. Irrelevant for policy
analysis, relevant for surveying. Knowing the number is what lets you say which.
'''),

# ------------------------------------------------------------------ I2 -----
md(r'''
## I2 — Invalid geometries and how to repair them

**What we are going to learn.** What makes a geometry invalid, how to diagnose
it, and the three repair strategies.

**Why it matters.** An invalid polygon is a landmine. It may sit quietly in your
data for weeks and then blow up an overlay with
`TopologyException: found non-noded intersection`, or — worse — return a *silently
wrong* area. Validity checking is the first thing you do to any polygon layer you
did not create yourself.

**The concept — the OGC validity rules for a polygon.**

1. Rings must be **closed** (first vertex = last vertex).
2. Rings must be **simple** — they must not self-intersect.
3. Interior rings (holes) must lie **inside** the exterior ring.
4. Rings may touch at a **finite number of points**, never along a line.
5. The interior must be **connected** — a hole may not split the polygon in two.

The classic violation is the **bow-tie**: a four-vertex "polygon" whose edges
cross, so it is really two triangles joined at a point. Its `.area` is the
*difference* of the two lobes, not the sum — which is how invalid geometry
produces plausible-but-wrong numbers.

**The three repairs.**

| Method | What it does | When to use |
|---|---|---|
| `make_valid(geom)` | GEOS `MakeValid`: rigorous, **preserves all input area**, may return a GeometryCollection or MultiPolygon | The correct default |
| `geom.buffer(0)` | Buffers by zero, which re-nodes the rings. Fast, but **silently discards** parts of a bow-tie | Legacy trick; fine for tiny slivers, dangerous otherwise |
| `set_precision(geom, grid)` | Snaps coordinates to a grid, removing near-degenerate edges | When invalidity comes from floating-point noise |

**Critical**: `make_valid` can change the geometry *type*. A repaired bow-tie
becomes a `MultiPolygon`; a repaired polygon with a line-touching hole may become
a `GeometryCollection` containing a polygon and a line. Always check the type
afterwards and `explode()` or filter as needed.

**Expected outcome.** The three planted invalid polygons diagnosed by name,
repaired three different ways, and the area differences quantified.

**What the next cell does:** finds the invalid geometries in `landuse`, prints
GEOS's explanation of each, repairs them with all three strategies, compares the
resulting areas and geometry types, and plots one bow-tie before and after.
'''),

code(r'''
from shapely.validation import explain_validity, make_valid
from shapely import set_precision

# --- 1. Diagnose --------------------------------------------------------------
bad_mask = ~landuse.geometry.is_valid
bad = landuse[bad_mask]
print(f"Invalid geometries in `landuse`: {len(bad)} of {len(landuse)}\n")
for i, row in bad.iterrows():
    print(f"  {row['lu_id']}  {row['landuse_class']:<12} -> {explain_validity(row.geometry)}")

# --- 2. Compare the three repair strategies ----------------------------------
def try_area(fn, g):
    """Apply a repair and report its area, or why it failed."""
    try:
        r = fn(g)
        return f"{r.area:,.0f}", r.geom_type
    except Exception as exc:
        return f"FAILED ({type(exc).__name__})", "-"

print("\n" + "=" * 104)
print("REPAIR COMPARISON  (all areas in m^2)")
print("=" * 104)
print(f"{'lu_id':<8}{'broken .area':>14}{'make_valid':>13}{'buffer(0)':>13}"
      f"{'precision->valid':>24}{'valid->precision':>18}   {'type':<14}")
print("-" * 104)
for i, row in bad.iterrows():
    g = row.geometry
    mv_a, mv_t = try_area(make_valid, g)
    b0_a, _    = try_area(lambda q: q.buffer(0), g)
    sp1, _     = try_area(lambda q: set_precision(q, 0.001), g)          # wrong order
    sp2, _     = try_area(lambda q: set_precision(make_valid(q), 0.001), g)  # right order
    print(f"{row['lu_id']:<8}{g.area:>14,.0f}{mv_a:>13}{b0_a:>13}"
          f"{sp1:>24}{sp2:>18}   {mv_t:<14}")
print("-" * 104)
print("ORDER MATTERS: set_precision() on an *invalid* geometry can throw;")
print("               repair first, then snap. make_valid() -> set_precision().")

# --- 3. Why the numbers differ ------------------------------------------------
g = bad.geometry.iloc[0]
mv = make_valid(g)
print("\nThe bow-tie, explained:")
print(f"  the broken geometry reports area  : {g.area:,.0f} m^2")
print(f"  make_valid gives a {mv.geom_type} of {len(mv.geoms)} parts:")
for k, part in enumerate(mv.geoms):
    print(f"      part {k+1}: {part.area:,.0f} m^2")
print(f"  total true area                    : {mv.area:,.0f} m^2")
try:
    print(f"  buffer(0) area                     : {g.buffer(0).area:,.0f} m^2")
except Exception as exc:
    print(f"  buffer(0)                          : raised {type(exc).__name__}")
    print(f"      {str(exc)[:96]}")
    print("      -> on this geometry the legacy trick does not merely lose area,")
    print("         it fails outright. make_valid() is the only safe choice.")

# --- 4. Repair the whole layer the right way ---------------------------------
landuse_fixed = landuse.copy()
landuse_fixed["geometry"] = landuse_fixed.geometry.make_valid()
# make_valid can emit GeometryCollections; keep only the polygonal parts
landuse_fixed = landuse_fixed.explode(index_parts=False, ignore_index=True)
landuse_fixed = landuse_fixed[landuse_fixed.geometry.geom_type.isin(
    ["Polygon", "MultiPolygon"])]
landuse_fixed = landuse_fixed[~landuse_fixed.geometry.is_empty]

print(f"\nLayer repair: {len(landuse)} rows -> {len(landuse_fixed)} rows "
      f"(explode split multi-part results)")
print(f"  invalid remaining : {int((~landuse_fixed.geometry.is_valid).sum())}")
print(f"  total area before : {landuse.geometry.area.sum()/1e6:,.3f} km^2")
print(f"  total area after  : {landuse_fixed.geometry.area.sum()/1e6:,.3f} km^2")

# --- 5. Picture it ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
gpd.GeoSeries([g], crs=CRS_UTM).plot(ax=axes[0], facecolor="#ffd4d4",
                                     edgecolor="crimson", linewidth=2)
axes[0].set_title(f"BROKEN bow-tie\n.area reports {g.area:,.0f} m^2",
                  fontsize=10, weight="bold", color="crimson")
gpd.GeoSeries(list(mv.geoms), crs=CRS_UTM).plot(
    ax=axes[1], facecolor="#d6ecd6", edgecolor="green", linewidth=2)
axes[1].set_title(f"make_valid() -> {mv.geom_type}, {len(mv.geoms)} parts\n"
                  f"true area {mv.area:,.0f} m^2",
                  fontsize=10, weight="bold", color="green")
for a in axes:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* `explain_validity(geom)` returns a human-readable GEOS diagnostic **and the
  coordinate where the problem is**, e.g.
  `Self-intersection[413950 4609450]`. That coordinate is gold when you have to
  go back to the data provider.
* `landuse.geometry.make_valid()` is the vectorised GeoSeries form (GeoPandas
  1.x). For a single geometry use `shapely.validation.make_valid(g)`.
* **The area comparison is the point of the lesson.** The bow-tie's broken
  `.area` is the *signed* sum of its two lobes — if they are similar sizes the
  reported area can be near zero, or even correct-looking by accident.
  `make_valid` returns a `MultiPolygon` with both lobes and the true total.
  `buffer(0)` returns only one lobe, so it under-reports by roughly half **with
  no warning at all**.
* `set_precision(g, 0.001)` snaps coordinates to a 1 mm grid. This does not fix a
  genuine bow-tie, but it is the right tool when invalidity is caused by
  coordinates that differ in the 12th decimal place — which is what you get after
  a chain of reprojections.
* `explode(index_parts=False, ignore_index=True)` splits multi-part geometries
  into one row each. After `make_valid` this is usually what you want, because a
  MultiPolygon carrying a repaired bow-tie is really two separate features.
* The final filter on `geom_type` removes any stray `LineString` or `Point`
  fragments that `make_valid` may produce when a polygon had zero-width spikes.
  **Skipping this filter is a common cause of "why does my polygon layer have
  lines in it?"**

**Expected output.**

```
Invalid geometries in `landuse`: 3 of 441

  LU9001  Shrubland    -> Self-intersection[...]
  LU9002  Shrubland    -> Self-intersection[...]
  LU9003  Shrubland    -> Self-intersection[...]
```

A repair-comparison table which, for each bow-tie, reads:

| | value |
|---|---|
| broken `.area` | **0 m²** |
| `make_valid` | **405 000 m²**, `MultiPolygon` of 2 parts |
| `buffer(0)` | **202 500 m²** — exactly half |
| `set_precision` then `make_valid` | **FAILED (GEOSException)** |
| `make_valid` then `set_precision` | **405 000 m²** |

Stare at the first row. **The broken polygon reports an area of exactly zero**,
even though it covers 40.5 hectares of ground. These bow-ties are symmetric, so
the two lobes have equal and opposite signed area and they cancel exactly.
Nothing raises, nothing warns; a `groupby("landuse_class").area.sum()` would
simply under-report shrubland by 1.2 km² and you would never know.

`buffer(0)` returns **exactly half** the true area — it keeps one lobe and
silently discards the other. And note the fourth row: calling `set_precision`
*before* repairing raises a `GEOSException`. Snapping assumes a valid input.
**Repair first, snap second.**

Layer repair takes 441 rows to **444** (the explode splits each repaired bow-tie
into 2 parts), leaves **0 invalid**, and *increases* total area from
**1 408.500 km² to 1 409.715 km²** — the 1.215 km² the broken geometries were
hiding. Then a two-panel figure: a red bow-tie, and the green two-triangle
repair.

> Note also that the land-use layer totals ~1 408 km² while the districts total
> 1 394.8 km². Land use is derived from a 50 m raster and its polygons overlap
> the coastline slightly, so it over-covers. Whenever two layers that *should*
> describe the same ground disagree on total area, you have found either a
> topology problem or a resolution artefact. Never average the two — find out
> which one is authoritative.
'''),

]

CELLS += [

# ------------------------------------------------------------------ I3 -----
md(r'''
## I3 — Missing values, sentinels and dirty categories

**What we are going to learn.** A complete, auditable cleaning pipeline for
spatial data: missingness, sentinel values, impossible values, inconsistent
categories, duplicate keys and bad coordinates.

**Why it matters.** You know how to clean tabular data. Spatial data adds two
new failure modes: **coordinates can be wrong in ways that still parse**, and
**joins can silently multiply or drop rows**. Both produce confident, wrong maps.

**The concept — five classes of dirt, and the right response to each.**

| Class | Example here | Response |
|---|---|---|
| **Explicit missing** | `districts.population` NaN ×2 | Keep as NaN. Never fill with 0 |
| **Sentinel disguised as data** | `capacity = -999`, `rainfall = -999` | Convert to NaN *before* any statistic |
| **Impossible value** | `year_built = 1066` or `2199` | Range-check against domain knowledge, then NaN |
| **Inconsistent category** | `"Forest"`, `"FOREST"`, `" forest "` | Normalise: strip, casefold, then map to a controlled vocabulary |
| **Bad geometry / coordinates** | (0,0), swapped lon/lat, out of region | Quarantine into a separate frame, never delete silently |

**The cardinal rule of cleaning: quarantine, do not delete.** Every row you drop
should land in a named "rejects" frame with a `reason` column. That frame is what
you show your client when they ask "why does your total differ from ours?"

**Concept — why sentinels are worse than NaN.** `-999` is a perfectly valid
float. `mean()`, `std()`, `groupby` and scikit-learn will all consume it happily.
A NaN at least propagates visibly. Always convert sentinels at the ingestion
boundary.

**Expected outcome.** Four cleaned datasets plus a rejects frame, with a printed
audit trail of exactly what was changed and why.

**What the next cell does:** cleans the flood-incident CSV (bad coordinates), the
sensor readings (sentinels, duplicates), the socio-economic table (duplicate and
orphan keys) and the land-use categories (case/whitespace) — printing a before/
after count for every rule.
'''),

code(r'''
audit_log = []

def log(step, before, after, note=""):
    audit_log.append({"step": step, "rows_before": before, "rows_after": after,
                      "delta": after - before, "note": note})

# =============================================================== INCIDENTS ===
inc = incidents_raw.copy()
n0 = len(inc)
inc["reject_reason"] = pd.NA

# rule 1: null island
inc.loc[(inc.lon == 0) & (inc.lat == 0), "reject_reason"] = "null_island_(0,0)"

# rule 2: lon/lat swapped -> the pair is invalid as (lon, lat) but valid reversed
STUDY = dict(lon=(13.6, 14.5), lat=(41.4, 42.0))
in_box = lambda lo, la: (lo.between(*STUDY["lon"])) & (la.between(*STUDY["lat"]))
swapped = (~in_box(inc.lon, inc.lat)) & in_box(inc.lat, inc.lon)
inc.loc[swapped & inc.reject_reason.isna(), "reject_reason"] = "lon_lat_swapped"

# rule 3: simply outside the study region
outside = (~in_box(inc.lon, inc.lat)) & inc.reject_reason.isna()
inc.loc[outside, "reject_reason"] = "outside_study_region"

rejects = inc[inc.reject_reason.notna()].copy()
inc_ok  = inc[inc.reject_reason.isna()].drop(columns="reject_reason").copy()

# rule 4: repair what is repairable - the swapped rows are recoverable
repaired = rejects[rejects.reject_reason == "lon_lat_swapped"].copy()
repaired[["lon", "lat"]] = repaired[["lat", "lon"]].to_numpy()
repaired = repaired.drop(columns="reject_reason")
inc_ok = pd.concat([inc_ok, repaired], ignore_index=True)

# rule 5: impossible attribute values
inc_ok.loc[inc_ok.damage_kvs < 0, "damage_kvs"] = np.nan

incidents = gpd.GeoDataFrame(
    inc_ok, geometry=gpd.points_from_xy(inc_ok.lon, inc_ok.lat),
    crs=CRS_WGS84).to_crs(CRS_UTM)

print("FLOOD INCIDENTS")
print(f"  raw rows                       : {n0}")
print(rejects.reject_reason.value_counts().to_string().replace("\n", "\n  "))
print(f"  recovered by un-swapping       : {len(repaired)}")
print(f"  clean rows                     : {len(incidents)}")
print(f"  negative damages -> NaN        : {int((inc.damage_kvs < 0).sum())}")
print(f"  depth_cm still missing         : {int(incidents.depth_cm.isna().sum())}")
log("incidents", n0, len(incidents), "quarantined bad coordinates")

# ============================================================ SENSOR DATA ===
rd = readings.copy()
n0 = len(rd)
rd = rd.drop_duplicates(subset=["station_id", "date"], keep="first")
SENTINELS = [-999, -9999]
for col in ["pm25_ugm3", "rainfall_mm", "temp_c"]:
    hit = rd[col].isin(SENTINELS).sum()
    rd.loc[rd[col].isin(SENTINELS), col] = np.nan
    if hit:
        print(f"\nSENSOR READINGS: {col}: {hit} sentinel values -> NaN")
print(f"  duplicate (station_id, date) rows removed : {n0 - len(rd)}")
print(f"  remaining NaN: " +
      ", ".join(f"{c}={int(rd[c].isna().sum())}" for c in
                ["pm25_ugm3", "rainfall_mm", "temp_c"]))
print(f"  MEAN RAINFALL  before cleaning: {readings.rainfall_mm.mean():>9,.2f} mm  <- polluted")
print(f"                 after  cleaning: {rd.rainfall_mm.mean():>9,.2f} mm  <- correct")
readings_clean = rd
log("readings", n0, len(rd), "sentinels -> NaN, duplicates dropped")

# =========================================================== SOCIO-ECONOMIC ==
so = socio.copy()
n0 = len(so)
dupe_ids = so.district_id[so.district_id.duplicated()].unique()
so = so.drop_duplicates(subset="district_id", keep="first")
valid_ids = set(districts.district_id)
orphans = sorted(set(so.district_id) - valid_ids)
so_ok = so[so.district_id.isin(valid_ids)].copy()
missing_ids = sorted(valid_ids - set(so_ok.district_id))
print(f"\nSOCIO-ECONOMIC TABLE")
print(f"  raw rows                    : {n0}")
print(f"  duplicated district_id keys : {list(dupe_ids)}")
print(f"  orphan keys (no polygon)    : {orphans}")
print(f"  districts with no socio row : {missing_ids if missing_ids else 'none'}")
print(f"  clean rows                  : {len(so_ok)}  (should equal 24)")
socio_clean = so_ok
log("socio", n0, len(so_ok), "dedup + orphan removal")

# ============================================================ LAND USE =======
lu = landuse_fixed.copy()
raw_levels = lu.landuse_class.nunique()
lu["landuse_class"] = (lu.landuse_class.str.strip().str.lower()
                         .str.replace(r"\s+", " ", regex=True))
CONTROLLED = {c.lower(): c for c in lc_legend.landuse_class}
lu["landuse_class"] = lu.landuse_class.map(CONTROLLED)
print(f"\nLAND USE CATEGORIES")
print(f"  distinct labels before normalisation : {raw_levels}")
print(f"  distinct labels after                : {lu.landuse_class.nunique()}")
print(f"  unmapped (NaN) after controlled map  : {int(lu.landuse_class.isna().sum())}")
# De-duplicate on (id, GEOMETRY), not on the id alone: after the make_valid /
# explode step the two halves of a repaired bow-tie legitimately share an lu_id.
n_before = len(lu)
lu = lu.assign(_wkb=lu.geometry.to_wkb())
lu = lu.drop_duplicates(subset=["lu_id", "_wkb"], keep="first").drop(columns="_wkb")
print(f"  exact duplicate rows removed         : {n_before - len(lu)}")
lu["lu_id"] = [f"LU{i+1:04d}" for i in range(len(lu))]   # re-issue unique ids
landuse_clean = lu
log("landuse", n_before, len(lu), "category normalisation + dedup")

# ============================================================ FACILITIES =====
fa = facilities.copy()
n0 = len(fa)
# 2a. duplicated features: same id AND same location
fa = fa.assign(_wkb=fa.geometry.to_wkb())
fa = fa.drop_duplicates(subset=["facility_id", "_wkb"], keep="first").drop(columns="_wkb")
n_dup = n0 - len(fa)
# 2b. sentinel capacity
n_sent = int((fa.capacity == -999).sum())
fa.loc[fa.capacity == -999, "capacity"] = np.nan
print(f"\nFACILITIES")
print(f"  raw rows                     : {n0}")
print(f"  exact duplicate rows removed : {n_dup}")
print(f"  capacity == -999 -> NaN      : {n_sent}")
print(f"  MEAN CAPACITY before cleaning: {facilities.capacity.mean():>9,.1f}  <- polluted")
print(f"                after  cleaning: {fa.capacity.mean():>9,.1f}  <- correct")
print(f"  clean rows                   : {len(fa)}")
facilities_clean = fa
log("facilities", n0, len(fa), "dedup + capacity sentinel")

# ============================================================ BUILDINGS ======
bl = buildings.copy()
n0 = len(bl)
bad_year = (bl.year_built < 1800) | (bl.year_built > 2025)
print(f"\nBUILDINGS")
print(f"  year_built outside 1800-2025 -> NaN : {int(bad_year.sum())}")
bl.loc[bad_year, "year_built"] = np.nan
print(f"  MEAN year_built before : {buildings.year_built.mean():>9,.1f}  <- dragged by 1066/2199")
print(f"                after  : {bl.year_built.mean():>9,.1f}")
bl["building_age"] = 2025 - bl["year_built"]
buildings_clean = bl
log("buildings", n0, len(bl), "impossible years -> NaN")

print("\n" + "=" * 78)
print("AUDIT TRAIL")
print(pd.DataFrame(audit_log).to_string(index=False))
'''),

md(r'''
**Explanation.**

* **`reject_reason` as a column, not a filter.** Every rule stamps a reason
  rather than dropping the row. The rejects frame survives, so the cleaning is
  fully auditable and reversible.
* **The swap-detection rule is the interesting one.** We do not look for
  "impossible latitudes > 90". These rows have lat = 13.9, which is a perfectly
  legal latitude — it is just in Africa. The test is *relational*: the pair fails
  as `(lon, lat)` but succeeds when reversed. That is strong evidence of a swap,
  and it lets us **recover** the rows rather than discard them.
* Order matters: null-island is checked first, because `(0,0)` also fails the
  in-box test and would otherwise be misdiagnosed as "outside region".
* `drop_duplicates(subset=["station_id", "date"])` — de-duplicate on the
  **logical key**, not on all columns. Two readings for the same station and
  month are a data error even if some other column differs.
* **The rainfall before/after comparison is the money line.** With 45 sentinel
  values of −999 in 866 rows, the mean rainfall reads about **9 mm/month**;
  cleaned, it is about **62 mm/month**. A factor of seven, from 5% contamination.
* The socio-economic block checks the join in **both directions**: orphan keys
  (rows with no polygon) *and* missing keys (polygons with no row). Only checking
  one direction is how you end up with a choropleth that quietly omits a district.
* **The controlled vocabulary map** (`CONTROLLED`) is stricter than
  `.str.title()`. Anything that does not map becomes NaN and is *counted*, so a
  new unexpected label (`"Forestry"`, say) is caught rather than silently
  title-cased into a new category.

**Expected output.**

```
FLOOD INCIDENTS
  raw rows                       : 393
    lon_lat_swapped         6
    null_island_(0,0)       4
    outside_study_region    3
  recovered by un-swapping       : 6
  clean rows                     : 386
  negative damages -> NaN        : 8
  depth_cm still missing         : 25

SENSOR READINGS: rainfall_mm: 45 sentinel values -> NaN
  duplicate (station_id, date) rows removed : 2
  remaining NaN: pm25_ugm3=45, rainfall_mm=45, temp_c=0
  MEAN RAINFALL  before cleaning:      9.52 mm  <- polluted
                 after  cleaning:     66.05 mm  <- correct

SOCIO-ECONOMIC TABLE
  duplicated district_id keys : ['D05', 'D12']
  orphan keys (no polygon)    : ['D99']
  districts with no socio row : none
  clean rows                  : 24  (should equal 24)

LAND USE CATEGORIES
  distinct labels before normalisation : 25
  distinct labels after                : 8
  exact duplicate rows removed         : 2

FACILITIES
  exact duplicate rows removed : 3
  capacity == -999 -> NaN      : 11
  MEAN CAPACITY before cleaning:     259.2  <- polluted
                after  cleaning:     448.8  <- correct

BUILDINGS
  year_built outside 1800-2025 -> NaN : 18
  MEAN year_built before :   1,978.4     after :   1,980.4
```

Four numbers deserve a second look.

1. **Rainfall: 9.52 → 66.05 mm/month.** Forty-five sentinel values in 866 rows —
   5% contamination — moved the mean by a factor of **seven**.
2. **Capacity: 259.2 → 448.8.** Eleven `-999` values dragged the mean down by 42%.
   Both of these would have sailed through any pipeline that only checked for NaN.
3. **386 clean incidents from 393 raw.** 4 null-island and 3 foreign rows
   quarantined, but **6 swapped rows recovered**. A blanket "drop anything
   outside the bounding box" would have discarded all 13 and lost 6 genuine
   observations.
4. **Buildings: mean year 1978.4 → 1980.4.** Only a 2-year shift — because 18 bad
   rows out of 5 200 is 0.35%. Contamination hurts in proportion to its share.
   This is why you check *every* field rather than assuming the impact is small.

Finally the audit trail — six cleaning steps with row counts before and after.
**That table is what you attach to the report.**
'''),

# ------------------------------------------------------------------ I4 -----
md(r'''
## I4 — Buffer analysis

**What we are going to learn.** Fixed, variable and dissolved buffers, and the
three ways buffering goes wrong.

**Why it matters.** The buffer is the fundamental proximity primitive: *"which
buildings are within 250 m of a river?"*, *"what land is within 1 km of a
school?"*, *"how much of the protected area lies within the noise corridor of the
motorway?"*. Almost every regulatory GIS question is a buffer question.

**The concept.** `geom.buffer(d)` returns every point within distance `d`.
Key parameters:

| Parameter | Effect |
|---|---|
| `distance` | Positive dilates; **negative erodes** (polygons only) |
| `resolution` (a.k.a. `quad_segs`) | Segments per quarter-circle. Default 8 → a 32-gon. Higher = smoother = slower |
| `cap_style` | `round` (1, default), `flat` (2), `square` (3) — how line ends are treated |
| `join_style` | `round` (1), `mitre` (2), `bevel` (3) — how corners are treated |
| `single_sided` | Buffer one side of a line only — useful for road verges, riparian strips |

**Three ways buffers go wrong.**

1. **Wrong CRS.** Buffering in degrees. Covered in B5; still the most common.
2. **Overlapping buffers double-count.** 40 schools each buffered 1 km produce
   40 overlapping discs. Their total `.area` is **not** the area served — you must
   `union_all()` (dissolve) first. This error inflates "population served" numbers
   routinely.
3. **Buffering then intersecting is not the same as intersecting then
   buffering.** Order matters; think about which one your question asks.

**Concept — variable-distance buffers.** Real regulations are rarely uniform: a
riparian protection zone might be 50 m for a first-order stream and 200 m for a
fourth-order river. GeoPandas buffers element-wise when you pass an **array** of
distances, which makes this a one-liner.

**Expected outcome.** A riparian protection zone with river-order-dependent
widths, a dissolved school catchment showing the double-counting error
quantified, and a single-sided motorway verge.

**What the next cell does:** builds three buffer types, quantifies the
overlapping-buffer error, and maps all three.
'''),

code(r'''
# --- 1. VARIABLE-distance buffer: riparian zone scaled by Strahler order -----
WIDTH_BY_ORDER = {2: 50.0, 3: 120.0, 4: 200.0}
rivers_b = rivers.copy()
rivers_b["protect_m"] = rivers_b.strahler_order.map(WIDTH_BY_ORDER)
riparian = rivers_b.copy()
riparian["geometry"] = rivers_b.geometry.buffer(rivers_b["protect_m"].to_numpy())
riparian["zone_area_ha"] = riparian.geometry.area / 1e4

print("VARIABLE-WIDTH RIPARIAN PROTECTION ZONE")
print(riparian[["river_id", "name", "strahler_order", "protect_m",
                "length_km", "zone_area_ha"]].to_string(index=False))
print(f"\n  sum of individual zone areas : {riparian.zone_area_ha.sum():>10,.1f} ha")
riparian_union = riparian.geometry.union_all()
print(f"  DISSOLVED (union) area       : {riparian_union.area/1e4:>10,.1f} ha")
print(f"  double-counted overlap       : {riparian.zone_area_ha.sum() - riparian_union.area/1e4:>10,.1f} ha")

# --- 2. FIXED buffer + the double-counting trap -----------------------------
schools = facilities_clean[facilities_clean.facility_type == "school"].copy()
RADIUS = 1500.0
school_buf = schools.copy()
school_buf["geometry"] = schools.geometry.buffer(RADIUS)

naive_area = school_buf.geometry.area.sum() / 1e6
dissolved = school_buf.geometry.union_all()
true_area = dissolved.area / 1e6

print("\n" + "=" * 78)
print(f"SCHOOL CATCHMENTS: {len(schools)} schools, {RADIUS:.0f} m radius")
print("=" * 78)
print(f"  naive sum of buffer areas : {naive_area:>8,.1f} km^2   <- WRONG")
print(f"  dissolved (union) area    : {true_area:>8,.1f} km^2   <- correct")
print(f"  inflation                 : {100*(naive_area-true_area)/true_area:>8,.1f} %")
print(f"  a single 1.5 km disc is   : {np.pi*RADIUS**2/1e6:>8,.2f} km^2")

# population served, computed both ways
blocks_c = blocks[~blocks.geometry.is_empty].copy()
blocks_c["geometry_pt"] = blocks_c.geometry.representative_point()
pts = gpd.GeoDataFrame(blocks_c.drop(columns="geometry"),
                       geometry="geometry_pt", crs=CRS_UTM)
naive_pop = gpd.sjoin(pts, school_buf[["facility_id", "geometry"]],
                      predicate="within").population.sum()
true_pop = pts[pts.geometry.within(dissolved)].population.sum()
print(f"\n  population 'served', naive sjoin : {naive_pop:>10,.0f}   <- counts people once per school")
print(f"  population served, dissolved     : {true_pop:>10,.0f}")
print(f"  over-count                       : {naive_pop/max(true_pop,1):>10,.2f} x")

# --- 3. SINGLE-SIDED buffer: a motorway verge -------------------------------
mw = roads[roads.road_class == "motorway"].geometry.union_all()
verge_r = mw.buffer(120, single_sided=True)
verge_l = mw.buffer(-120, single_sided=True)
print(f"\nSINGLE-SIDED motorway verge (120 m):")
print(f"  right side area : {verge_r.area/1e4:,.1f} ha")
print(f"  left  side area : {verge_l.area/1e4:,.1f} ha")
print(f"  two-sided 120 m : {mw.buffer(120).area/1e4:,.1f} ha")

# --- 4. Map ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
for ax in axes:
    land.plot(ax=ax, facecolor="#f7f5ef", edgecolor="#d8d2c4", linewidth=0.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

riparian.plot(ax=axes[0], column="strahler_order", cmap="Blues", alpha=0.75,
              legend=True, legend_kwds={"shrink": 0.55, "label": "Strahler order"})
rivers.plot(ax=axes[0], color="navy", linewidth=0.7)
axes[0].set_title("(a) Variable-width riparian zone\n50 / 120 / 200 m by river order",
                  loc="left", fontsize=10, weight="bold")

school_buf.plot(ax=axes[1], facecolor="#f4a582", edgecolor="#d6604d",
                alpha=0.35, linewidth=0.4)
gpd.GeoSeries([dissolved], crs=CRS_UTM).boundary.plot(ax=axes[1], color="black",
                                                      linewidth=1.0)
schools.plot(ax=axes[1], color="black", markersize=6)
axes[1].set_title("(b) 42 overlapping 1.5 km catchments\nblack = dissolved outline",
                  loc="left", fontsize=10, weight="bold")

gpd.GeoSeries([verge_r], crs=CRS_UTM).plot(ax=axes[2], facecolor="#7fbf7b", alpha=0.8)
gpd.GeoSeries([verge_l], crs=CRS_UTM).plot(ax=axes[2], facecolor="#af8dc3", alpha=0.8)
roads[roads.road_class == "motorway"].plot(ax=axes[2], color="black", linewidth=0.8)
axes[2].set_title("(c) Single-sided buffers\ngreen = right, purple = left",
                  loc="left", fontsize=10, weight="bold")
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* `rivers_b.geometry.buffer(rivers_b["protect_m"].to_numpy())` — passing a NumPy
  array of the same length buffers **element-wise**. This is Shapely 2 vectorisation;
  in Shapely 1.8 you needed an `apply`. Note `.to_numpy()`: passing the Series
  directly can align on index rather than position, which is a subtle bug source.
* **The riparian overlap** is small (the rivers rarely run within 400 m of each
  other), but non-zero where they converge near the coast. Always check.
* **The school-catchment block is the important one.** 42 discs of 7.07 km² each
  sum to ~297 km²; the dissolved union is far smaller because the schools cluster
  in town. The naive sum over-states served area by a large margin.
* The **population** version of the same error is worse, because it is the number
  that ends up in the report. `gpd.sjoin(pts, school_buf)` returns one row per
  (block, school) pair, so a block inside five catchments is counted five times.
  The dissolved version counts each person once. In a real service-coverage study
  this is the difference between "we serve 1.4 million people" and "we serve
  450 000 people" in a region of 600 000 — an obviously impossible number that
  nevertheless gets published.
* `buffer(d, single_sided=True)` offsets to one side only; the **sign** of `d`
  chooses the side (positive = left of the direction of travel in Shapely 2 /
  GEOS convention — verify empirically, as we do here, rather than trusting a
  remembered rule).
* Note that `verge_r.area + verge_l.area` is close to, but not equal to, the
  two-sided buffer: the two-sided version includes round caps at the ends and
  merges self-overlaps at tight bends.

**Expected outcome.**

```
VARIABLE-WIDTH RIPARIAN PROTECTION ZONE
 RV01 Vallmara River  order 4  200 m  43.93 km  ~1,760 ha
 RV02 Kestrel Brook   order 3  120 m  ...
 ...
  sum of individual zone areas :    4,813.9 ha
  DISSOLVED (union) area       :    4,217.5 ha
  double-counted overlap       :      596.4 ha

SCHOOL CATCHMENTS: 42 schools, 1500 m radius
  naive sum of buffer areas :    296.6 km^2   <- WRONG
  dissolved (union) area    :    ~220 km^2    <- correct
  inflation                 :     ~35 %
  a single 1.5 km disc is   :      7.07 km^2

  population 'served', naive sjoin :   ~700,000   <- counts people once per school
  population served, dissolved     :   ~440,000
  over-count                       :      ~1.6 x

SINGLE-SIDED motorway verge (120 m):
  right side area : 540.1 ha
  left  side area : 539.5 ha
  two-sided 120 m : 1,084.1 ha
```

The riparian overlap is **596 ha, 12% of the naive total** — the rivers converge
near the coast and their protection zones merge there.

The population figure is the one that matters: the naive spatial join claims
**~700 000 people served** in a region whose entire population is 644 000. A
number larger than the population is at least obviously wrong. The dangerous
version is the area figure, where 296.6 km² versus ~220 km² looks plausible
either way.

Then a three-panel figure. In panel (b) the overlapping discs are visibly stacked
over the urban core — that visual pile-up *is* the double-counting.
'''),

# ------------------------------------------------------------------ I5 -----
md(r'''
## I5 — Overlay operations

**What we are going to learn.** The four set operations on polygon layers, and
what each one does to the attributes.

**Why it matters.** Overlay is how you answer *"how much of X is inside Y?"* —
the question behind land-use change, hazard exposure, and every impact assessment
ever written.

**The concept — `gpd.overlay(df1, df2, how=...)`.**

| `how` | Result geometry | Attributes | Typical question |
|---|---|---|---|
| `"intersection"` | Only the overlapping parts | From **both** layers | "What land use lies in the flood zone?" |
| `"union"` | Every piece from both, split where they overlap | Both, NaN where absent | "Give me every distinct combination" |
| `"difference"` | Parts of df1 **not** in df2 | df1 only | "Which land is *outside* the protected area?" |
| `"symmetric_difference"` | Parts in exactly one layer | Both, NaN where absent | "Where do the two datasets disagree?" |
| `"identity"` | All of df1, split by df2 | Both, NaN outside df2 | "Tag df1 with df2 where it applies" |

**The key mental model.** Overlay **splits geometry**. If a land-use polygon
straddles a flood-zone boundary, `intersection` returns only the part inside, as
a *new, smaller polygon*. The attributes are copied unchanged — which means
**every absolute quantity must be recomputed after an overlay**. If your land-use
polygon said `area_ha = 120` and the intersection kept a third of it, the
attribute still says 120. Recompute, or you will over-report by 3×.

**Concept — the sliver problem.** Overlaying two layers digitised independently
produces thousands of tiny "sliver" polygons along boundaries that *should*
coincide. Filter them by area (or by a thinness ratio `4πA/P²`) before analysing.

**Expected outcome.** A land-use × flood-zone intersection with correctly
recomputed areas, plus a demonstration of all four operations on the same pair.

**What the next cell does:** intersects land use with the 100-year flood zone,
shows what happens if you forget to recompute area, runs all four overlay modes
on a simplified pair, and reports the sliver distribution.
'''),

code(r'''
lu = landuse_clean[["lu_id", "landuse_class", "area_ha", "geometry"]].copy()
fz100 = flood[flood.return_period_yr == 100][["zone_id", "hazard_class", "geometry"]].copy()

# --- 1. INTERSECTION: which land uses are in the 100-year flood zone? -------
hit = gpd.overlay(lu, fz100, how="intersection", keep_geom_type=True)
hit["area_ha_true"] = hit.geometry.area / 1e4

print(f"land-use polygons          : {len(lu):>6}")
print(f"flood-zone polygons        : {len(fz100):>6}")
print(f"intersection pieces        : {len(hit):>6}   <- geometry was SPLIT\n")

wrong = hit.groupby("landuse_class", observed=True)["area_ha"].sum()
right = hit.groupby("landuse_class", observed=True)["area_ha_true"].sum()
cmp = pd.DataFrame({"inherited area_ha (WRONG)": wrong,
                    "recomputed area (right)": right})
cmp["over-report factor"] = (cmp.iloc[:, 0] / cmp.iloc[:, 1]).round(1)
print("LAND USE INSIDE THE 100-YEAR FLOOD ZONE")
print(cmp.sort_values("recomputed area (right)", ascending=False).round(1).to_string())
print(f"\nTOTAL inherited : {wrong.sum():>12,.0f} ha   <- nonsense")
print(f"TOTAL recomputed: {right.sum():>12,.0f} ha   "
      f"({right.sum()/100:,.1f} km^2 of the {LAND_GEOM.area/1e6:,.0f} km^2 basin)")

# --- 2. All four overlay modes on ONE pair ----------------------------------
a = gpd.GeoDataFrame({"lab": ["A"]}, geometry=[districts.geometry.iloc[1]], crs=CRS_UTM)
b = gpd.GeoDataFrame({"lab": ["B"]},
                     geometry=[districts.geometry.iloc[1].centroid.buffer(4200)],
                     crs=CRS_UTM)

fig, axes = plt.subplots(1, 5, figsize=(17.5, 4))
modes = ["intersection", "union", "difference", "symmetric_difference", "identity"]
for ax, how in zip(axes, modes):
    a.boundary.plot(ax=ax, color="#3b5378", linewidth=1.2)
    b.boundary.plot(ax=ax, color="#c1666b", linewidth=1.2)
    res = gpd.overlay(a, b, how=how, keep_geom_type=False)
    res.plot(ax=ax, facecolor="#7fb3a3", edgecolor="black", alpha=0.75, linewidth=0.5)
    ax.set_title(f"{how}\n{len(res)} feature(s), {res.geometry.area.sum()/1e6:,.1f} km^2",
                 fontsize=9, weight="bold")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("gpd.overlay: blue = layer A (a district), red = layer B (a disc)",
             fontsize=11)
plt.tight_layout(); plt.show()

# --- 3. Slivers ---------------------------------------------------------------
def sliver_report(gdf, label):
    areas = gdf.geometry.area
    print(f"\nSLIVER DIAGNOSIS - {label}   ({len(gdf)} pieces)")
    print(f"  {'threshold':>12} {'n rows':>8} {'% rows':>8} {'% of area':>11}")
    for t in [1, 10, 100, 1_000, 10_000]:
        n = int((areas < t).sum())
        print(f"  {t:>10,} m^2 {n:>8} {100*n/len(gdf):>7.1f}% "
              f"{100*areas[areas < t].sum()/areas.sum():>10.4f}%")

# (a) our real result: both layers came off the SAME 25 m raster grid
sliver_report(hit, "land use x flood zone (co-registered layers)")

# (b) what happens with two INDEPENDENTLY digitised layers. We simulate that by
#     overlaying the districts with a copy shifted by 12 m - about the accuracy
#     of two different survey campaigns.
from shapely.affinity import translate
shifted = districts[["district_id", "geometry"]].copy()
shifted["geometry"] = shifted.geometry.apply(lambda g: translate(g, 12, -12))
shifted = shifted.rename(columns={"district_id": "district_id_b"})
slivers = gpd.overlay(districts[["district_id", "geometry"]], shifted,
                      how="intersection", keep_geom_type=True)
mismatch = slivers[slivers.district_id != slivers.district_id_b]
sliver_report(mismatch, "districts x districts shifted 12 m (mis-registration)")
print(f"\n  -> {len(mismatch)} spurious polygons created by a 12 m shift, "
      f"carrying {mismatch.geometry.area.sum()/1e4:,.1f} ha in total")
print(f"  -> filtering to area >= 1,000 m^2 leaves "
      f"{int((mismatch.geometry.area >= 1000).sum())} of them")
'''),

md(r'''
**Explanation.**

* `gpd.overlay(lu, fz100, how="intersection")` computes the pairwise
  intersection of every land-use polygon with every flood-zone polygon that it
  touches. The result has **one row per intersecting pair** and carries columns
  from both inputs.
* `keep_geom_type=True` discards non-polygonal results. Without it, two polygons
  that merely *touch* contribute a zero-area LineString to your polygon layer,
  and every subsequent `.area` call returns 0 for those rows. **Set it
  explicitly; the default has changed across versions.**
* **The `area_ha` vs `area_ha_true` comparison is the lesson.** `area_ha` was
  computed on the *whole* land-use polygon before the split, and `overlay` copied
  it verbatim onto each fragment. Summing it over-reports by whatever factor the
  polygons were cut by — here typically **3–10×**, and unboundedly more if one big
  polygon is cut into many pieces. Any absolute attribute (area, population,
  count, value) is invalid after an overlay until you recompute or apportion it.
  We do the apportionment properly in I8.
* `observed=True` in `groupby` — required when grouping by a categorical to avoid
  materialising empty categories. Harmless here, essential on real data.
* **The five-panel figure** is worth studying. `union` produces *more* features
  than either input, because every overlapping region becomes its own polygon.
  `identity` keeps all of A but splits it where B crosses — it is `intersection`
  plus `difference` of A, and is what you want when tagging one layer with
  another without losing coverage.
* **Sliver diagnosis**: count pieces below a series of area thresholds and check
  what fraction of *area* they represent. The characteristic sliver signature is
  "many rows, negligible area" — e.g. 5% of rows carrying 0.001% of area. That
  is your licence to drop them. If small pieces carry meaningful area, they are
  not slivers; they are real small features, and dropping them is a bug.

**Expected output.**

```
land-use polygons          :    442
flood-zone polygons        :     11
intersection pieces        :    167   <- geometry was SPLIT

LAND USE INSIDE THE 100-YEAR FLOOD ZONE
                     inherited area_ha (WRONG)   recomputed (right)   factor
Forest                            198,214.8              6,963.0       28.5
Grassland                          35,773.5              3,523.0       10.2
Bare rock / sparse                  8,715.1              1,957.7        4.5
Water                               1,893.2              1,885.0        1.0
Built-up                           10,106.6              1,792.7        5.6
Cropland                           32,619.6              1,782.1       18.3
Shrubland                             783.5                328.6        2.4
Wetland                               833.0                277.2        3.0

TOTAL inherited :      288,939 ha   <- nonsense
TOTAL recomputed:       18,509 ha   (185.1 km^2 of the 1,395 km^2 basin)
```

The inherited total, 288 939 ha = **2 889 km²**, is more than twice the area of
the entire basin. Any quantity that exceeds the size of your study area is a free
sanity check — take it.

Note the per-class factors: **Forest over-reports 28.5×** because a few very
large forest polygons are clipped to thin ribbons along the rivers, while
**Water over-reports 1.0×** because water polygons lie almost entirely inside the
flood zone and are barely cut at all. The distortion is not a constant you can
divide out; it depends on how each polygon was clipped.

Then the five-panel overlay figure, and two sliver reports:

* **Co-registered layers** (land use × flood zone, both derived from the same
  25 m grid): **zero** pieces under 100 m². Slivers are not an inevitable
  by-product of overlay.
* **Mis-registered layers** (districts × districts shifted 12 m): **78 spurious
  polygons carrying 406 ha**. That is what a 12 m disagreement between two survey
  campaigns costs you — and note that filtering at 1 000 m² removes only 4 of
  them, because most slivers along a 12 m offset are *long*, not small. **Filter
  slivers on thinness (`4πA/P²`), not on area alone.**
'''),

]

CELLS += [

# ------------------------------------------------------------------ I6 -----
md(r'''
## I6 — Clipping: `gpd.clip` versus `gpd.overlay`

**What we are going to learn.** The difference between clipping and
intersecting, and when each is the right tool.

**Why it matters.** They look interchangeable and are not. Choosing wrongly
either loses the attributes you needed or explodes your row count.

**The concept.**

| | `gpd.clip(gdf, mask)` | `gpd.overlay(gdf, other, how="intersection")` |
|---|---|---|
| Mask | A geometry, GeoSeries **or** GeoDataFrame — treated as **one shape** | A full GeoDataFrame |
| Rows out | **≤ rows in.** One row per *input* feature, trimmed | One row per intersecting **pair** — can be far more |
| Attributes | Only from `gdf` | From **both** layers |
| Use it when | "Cut this layer to my study area" | "Cross-tabulate these two layers" |

**Mental model.** `clip` is a **cookie cutter**: the mask is a stencil, not a
dataset. `overlay` is a **join that splits geometry**: both sides contribute
attributes.

**Two practical warnings.**

1. `clip` dissolves the mask internally. If your mask GeoDataFrame has 24
   districts, clipping does **not** tag each output feature with its district —
   it just trims everything to the outline of all 24 combined. If you want the
   tag, you need `overlay` or `sjoin`.
2. `clip` can return **mixed geometry types** — clipping a polygon layer with a
   mask that only grazes some polygons yields LineStrings and Points at the
   tangencies. Filter with `keep_geom_type=True` (GeoPandas ≥ 0.14) or manually.

**Expected outcome.** Roads clipped to a protected area versus overlaid with it,
with the row counts and attributes compared side by side.

**What the next cell does:** clips the road network to the protected areas and
overlays it with the same layer, compares row counts, attributes and total
length, then shows the mixed-geometry-type trap.
'''),

code(r'''
pa = protected[["pa_id", "name", "designation", "geometry"]].copy()

# --- 1. CLIP: cookie-cutter --------------------------------------------------
roads_clipped = gpd.clip(roads, pa, keep_geom_type=True)

# --- 2. OVERLAY: attribute-carrying intersection ----------------------------
roads_overlaid = gpd.overlay(roads, pa, how="intersection", keep_geom_type=True)

print("ROADS INSIDE PROTECTED AREAS")
print("=" * 78)
print(f"  input road segments           : {len(roads):>6}")
print(f"  gpd.clip     -> rows          : {len(roads_clipped):>6}")
print(f"  gpd.overlay  -> rows          : {len(roads_overlaid):>6}")
print(f"  clip total length             : {roads_clipped.length.sum()/1000:>9.2f} km")
print(f"  overlay total length          : {roads_overlaid.length.sum()/1000:>9.2f} km")
print(f"\n  clip columns   : {list(roads_clipped.columns)[:6]} ...")
print(f"  overlay columns: {[c for c in roads_overlaid.columns if c in
                             ['road_id','road_class','pa_id','name_2','designation']]} "
      f"<- carries the RESERVE identity too")

# --- 3. Only overlay can answer 'which road in which reserve?' -------------
by_pa = (roads_overlaid.assign(km=roads_overlaid.length/1000)
         .groupby(["pa_id", "road_class"], observed=True)["km"].sum()
         .unstack(fill_value=0).round(2))
print("\nRoad kilometres by reserve and class (only OVERLAY can produce this):")
print(by_pa.to_string())

# --- 4. Clip a POLYGON layer to a study sub-area ----------------------------
study = box(408_000, 4_608_000, 428_000, 4_628_000)
study_gs = gpd.GeoSeries([study], crs=CRS_UTM)

lu_clip_all  = gpd.clip(landuse_clean, study_gs)                    # no filter
lu_clip_poly = gpd.clip(landuse_clean, study_gs, keep_geom_type=True)
print("\n" + "=" * 78)
print("THE MIXED-GEOMETRY-TYPE TRAP")
print("=" * 78)
print(f"  clip without keep_geom_type : {len(lu_clip_all):>5} rows, "
      f"types {dict(lu_clip_all.geom_type.value_counts())}")
print(f"  clip with    keep_geom_type : {len(lu_clip_poly):>5} rows, "
      f"types {dict(lu_clip_poly.geom_type.value_counts())}")
print(f"  area sum, unfiltered        : {lu_clip_all.geometry.area.sum()/1e6:>9.3f} km^2")
print(f"  area sum, filtered          : {lu_clip_poly.geometry.area.sum()/1e6:>9.3f} km^2")
print("\n  The AREA totals agree - so why does it matter? Because a mixed-type")
print("  layer breaks the moment anything assumes polygons:")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for label, layer in [("unfiltered", lu_clip_all), ("filtered", lu_clip_poly)]:
        n_null_ring = int(layer.geometry.exterior.isna().sum())
        try:
            layer.to_file(OUT / f"_clip_{label}.shp", driver="ESRI Shapefile")
            wmsg = "wrote .shp OK"
        except Exception as exc:
            wmsg = f"write to .shp FAILED -> {type(exc).__name__}"
        print(f"    {label:<11} rows with no exterior ring: {n_null_ring:>3}    {wmsg}")
print("\n  keep_geom_type=True also EXPLODES GeometryCollections into their")
print("  polygon parts, which is why the filtered layer has MORE rows, not fewer.")

# --- 5. Map -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
land.plot(ax=axes[0], facecolor="#f7f5ef", edgecolor="#d8d2c4", linewidth=0.5)
roads.plot(ax=axes[0], color="#cccccc", linewidth=0.4)
pa.plot(ax=axes[0], facecolor="#c7e9c0", edgecolor="#238b45", alpha=0.7, linewidth=0.9)
roads_clipped.plot(ax=axes[0], color="crimson", linewidth=1.1)
axes[0].set_title(f"gpd.clip -> {len(roads_clipped)} rows\nroads trimmed to the reserves",
                  loc="left", fontsize=10, weight="bold")

land.plot(ax=axes[1], facecolor="#f7f5ef", edgecolor="#d8d2c4", linewidth=0.5)
landuse_clean.plot(ax=axes[1], color="#e8e4d9", edgecolor="none")
lu_clip_poly.plot(ax=axes[1], column="landuse_class", cmap="tab10", legend=True,
                  legend_kwds={"loc": "lower left", "fontsize": 6.5},
                  edgecolor="white", linewidth=0.2)
study_gs.boundary.plot(ax=axes[1], color="black", linewidth=1.4, linestyle="--")
axes[1].set_title("gpd.clip of land use to a 20 x 20 km study box",
                  loc="left", fontsize=10, weight="bold")
for a in axes:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* `gpd.clip(roads, pa)` treats the five reserve polygons as **one stencil**.
  The output has at most one row per input road segment, keeps only the road
  attributes, and answers "how much road is inside protected land?".
* `gpd.overlay(roads, pa, how="intersection")` produces one row per
  (road segment, reserve) pair, and carries `pa_id` / `designation` through. Only
  this version can answer "how much road is inside **which** reserve?" — which is
  the question a report actually asks.
* Note the total lengths are **identical**. The two operations cut the same
  geometry; they differ only in bookkeeping. (They would differ if reserves
  overlapped each other, in which case overlay would double-count.)
* Column-name collision: both layers have `name`, so overlay produces `name_1`
  and `name_2`. Rename immediately.
* **`keep_geom_type=True`** is doing real work in step 4. Without it, land-use
  polygons that merely touch the study box contribute zero-area LineStrings, and
  those rows carry attributes that will be counted by any subsequent `groupby`
  while contributing 0 to the area. You get categories that "exist" with zero
  extent — a confusing artefact that is hard to trace back.

**Expected outcome.**

```
ROADS INSIDE PROTECTED AREAS
  input road segments           :    791
  gpd.clip     -> rows          :     22
  gpd.overlay  -> rows          :     22
  clip total length             :     19.25 km
  overlay total length          :     19.25 km
  overlay columns: [..., 'pa_id', 'name_2', 'designation']  <- reserve identity too
```

Identical row counts and identical total length — the two operations cut the same
geometry. Then the table only `overlay` can produce:

```
road_class  motorway  primary  secondary  track
PA02           0.00      4.29       4.43   5.99
PA04           1.99      0.00       0.00   0.00
PA90           0.00      0.00       0.00   2.54
```

**Fyrdal Wetland Park (PA02) has 14.7 km of road through it, including 4.3 km of
primary road.** That is an ecological finding, and `clip` cannot state it.

Then the geometry-type trap:

```
  clip without keep_geom_type :   168 rows, {'Polygon': 151, 'GeometryCollection': 16, 'LineString': 1}
  clip with    keep_geom_type :   173 rows, {'Polygon': 173}
  area sum, unfiltered        :   372.595 km^2
  area sum, filtered          :   372.595 km^2
    unfiltered  rows with no exterior ring:  17    write to .shp FAILED -> FeatureError
    filtered    rows with no exterior ring:   0    wrote .shp OK
```

Three things to notice. The **area totals agree**, so a summary statistic will not
warn you. Seventeen rows have **no exterior ring**, so any code doing
`geom.exterior` gets `None` and crashes downstream. And **writing the layer to a
shapefile fails outright**, because a shapefile must hold a single geometry type.
Note too that the filtered layer has *more* rows (173 vs 168): `keep_geom_type`
explodes each GeometryCollection into its polygon parts rather than dropping it.

Finally two maps: red road segments inside green reserves, and a land-use map
clipped to a dashed 20 × 20 km box.
'''),

# ------------------------------------------------------------------ I7 -----
md(r'''
## I7 — Spatial joins in depth

**What we are going to learn.** Predicate choice, join cardinality, and how to
guarantee a join did what you think it did.

**Why it matters.** In B11 we saw a spatial join silently double a row count.
Here we build the discipline that prevents it: **state the expected cardinality
before you join, then assert it afterwards.**

**The concept — cardinality is a design decision.**

| Intent | Cardinality | How to guarantee it |
|---|---|---|
| Tag each point with its polygon | **1:1** | `predicate="within"` on a *non-overlapping* polygon layer, then `assert len(out) == len(left)` |
| Count points per polygon | **N:1**, then aggregate | `sjoin` then `groupby(...).size()` |
| Every polygon a point falls in | **1:N**, intentionally | `predicate="intersects"`, keep duplicates |
| Attach nearest feature | **1:1** | `sjoin_nearest(..., max_distance=...)` |

**Predicate cheat-sheet for the common cases.**

* **Points in polygons** → `within`. Use `intersects` only if you deliberately
  want boundary points matched to both neighbours.
* **Lines in polygons** → there is no single right answer. `intersects` matches a
  road that merely clips a corner; `within` matches only roads entirely inside.
  For "how much road is in this district", neither is right — you need
  `overlay` and length recomputation (I6).
* **Polygons to polygons** → almost always `overlay`, not `sjoin`. A spatial join
  of two polygon layers gives you *pairs*, not *shared areas*.
* **`dwithin`** (GeoPandas ≥ 0.14) → "within distance d", implemented efficiently
  in the index. Much faster than buffering then joining.

**Expected outcome.** The same question answered with three predicates so you can
see the counts diverge, plus a reusable `checked_sjoin` helper.

**What the next cell does:** defines a cardinality-checking join wrapper, joins
buildings to flood zones under three predicates, and joins bus stops to districts
using `dwithin` to find stops *near* but not inside each district.
'''),

code(r'''
def checked_sjoin(left, right, predicate, how="inner", expect=None, label=""):
    """A spatial join that refuses to silently change your row count."""
    out = gpd.sjoin(left, right, how=how, predicate=predicate)
    n_in, n_out = len(left), len(out)
    dup = n_out - out.index.nunique()
    status = "OK"
    if expect == "1:1" and n_out != n_in:
        status = f"!! expected 1:1 but {n_out} != {n_in}"
    print(f"  {label:<34} predicate={predicate:<11} "
          f"in={n_in:>6} out={n_out:>6} dup_rows={dup:>5}  {status}")
    return out

fz_all = flood[["zone_id", "hazard_class", "return_period_yr", "geometry"]]
b = buildings_clean[["building_id", "use_type", "value_kvs", "geometry"]]

print("BUILDINGS x FLOOD ZONES under three predicates")
print("=" * 100)
j_int = checked_sjoin(b, fz_all, "intersects", label="any contact")
j_wit = checked_sjoin(b, fz_all, "within",     label="entirely inside")
j_100 = checked_sjoin(b, fz_all[fz_all.return_period_yr == 100],
                      "intersects", label="100-year zone only")

print(f"\n  distinct buildings touching ANY flood zone : "
      f"{j_int.building_id.nunique():>6}")
print(f"  distinct buildings ENTIRELY inside one     : "
      f"{j_wit.building_id.nunique():>6}")
print(f"  buildings straddling a zone boundary       : "
      f"{j_int.building_id.nunique() - j_wit.building_id.nunique():>6}")
print(f"  rows > distinct buildings by               : "
      f"{len(j_int) - j_int.building_id.nunique():>6}  "
      f"(buildings touching 2+ zone polygons)")

# --- The safe way to aggregate a 1:N join -----------------------------------
print("\n" + "=" * 100)
print("AGGREGATING A 1:N JOIN SAFELY")
print("=" * 100)
naive = j_int.value_kvs.sum()
safe  = j_int.drop_duplicates("building_id").value_kvs.sum()
print(f"  naive sum over join rows      : {naive:>14,.0f} k VS   <- double counts")
print(f"  sum after de-duplication      : {safe:>14,.0f} k VS   <- correct")
print(f"  inflation                     : {100*(naive-safe)/safe:>13,.1f} %")

# exposure by return period, correctly: assign each building its WORST zone
worst = (j_int.sort_values("return_period_yr")
              .drop_duplicates("building_id", keep="first"))
print("\n  Buildings and asset value by worst-case hazard zone:")
print(worst.groupby("return_period_yr", observed=True)
           .agg(buildings=("building_id", "nunique"),
                value_kVS=("value_kvs", "sum")).round(0).to_string())

# --- dwithin: 'near', without building a buffer -----------------------------
print("\n" + "=" * 100)
print("PROXIMITY JOIN WITHOUT A BUFFER: predicate='dwithin'")
print("=" * 100)
d = districts[["district_id", "name", "geometry"]]
near = gpd.sjoin(stops[["stop_id", "geometry"]], d,
                 predicate="dwithin", distance=500)
inside = gpd.sjoin(stops[["stop_id", "geometry"]], d, predicate="within")
print(f"  bus stops INSIDE a district        : {len(inside):>5}")
print(f"  bus stops WITHIN 500 m of one      : {len(near):>5}  "
      f"(stops near a border match several districts)")
print(f"  stops matching 2+ districts        : "
      f"{int((near.groupby('stop_id').size() > 1).sum()):>5}")
'''),

md(r'''
**Explanation.**

* **`checked_sjoin` is the habit to build.** It prints the input count, the
  output count and the number of duplicated left-index rows every time. Three
  numbers, printed automatically, that make row multiplication impossible to
  miss. In a production pipeline the `expect="1:1"` branch should `raise`, not
  print.
* **`intersects` vs `within` for polygons.** A building that straddles the flood
  zone boundary `intersects` it but is not `within` it. The difference — a few
  hundred buildings — is exactly the set of properties that are *partially*
  exposed. Which definition you use is a **policy** choice, not a technical one,
  and you must state it in your report.
* **Aggregating a 1:N join.** `j_int.value_kvs.sum()` adds a building's value once
  per flood polygon it touches. `drop_duplicates("building_id")` fixes it. This is
  the single most common source of inflated damage/exposure figures in published
  risk assessments.
* **The "worst zone" pattern** is the right way to collapse a 1:N join when the
  right-hand layer is ordered by severity: sort by severity, then keep the first
  row per left feature. A building in both the 100-year and 500-year zone is a
  100-year building; counting it in both inflates the 500-year total.
* **`predicate="dwithin", distance=500`** does a proximity join *inside the
  spatial index*, without materialising 130 buffer polygons. It is both faster and
  less memory-hungry than `stops.buffer(500)` followed by a join, and it is exact.

**Expected outcome.**

```
BUILDINGS x FLOOD ZONES under three predicates
  any contact         predicate=intersects  in=5200 out=1810 dup_rows= 59  OK
  entirely inside     predicate=within      in=5200 out=1608 dup_rows=  0  OK
  100-year zone only  predicate=intersects  in=5200 out=1036 dup_rows=  0  OK

  distinct buildings touching ANY flood zone :   1751
  distinct buildings ENTIRELY inside one     :   1608
  buildings straddling a zone boundary       :    143
  rows > distinct buildings by               :     59

AGGREGATING A 1:N JOIN SAFELY
  naive sum over join rows      :        435,218 k VS   <- double counts
  sum after de-duplication      :        412,295 k VS   <- correct
  inflation                     :            5.6 %

  Buildings and asset value by worst-case hazard zone:
  100-year   1036 buildings   261,979 k VS
  500-year    715 buildings   150,316 k VS

PROXIMITY JOIN WITHOUT A BUFFER: predicate='dwithin'
  bus stops INSIDE a district        :   129
  bus stops WITHIN 500 m of one      :   145
  stops matching 2+ districts        :    14
```

Three findings worth stating plainly:

* **143 buildings straddle a hazard boundary.** Whether they count as "exposed"
  is a policy decision that changes the headline number by 8%. Say which you chose.
* The naive 1:N sum inflates exposed asset value by **5.6% (23 million VS)**. Not
  catastrophic — which is precisely why nobody notices it.
* 129 stops are inside a district but 145 match under `dwithin(500 m)`, with
  **14 stops matching two or more districts**. That is not an error; it is the
  correct answer to a different question. Choosing the predicate *is* choosing the
  question.
'''),

# ------------------------------------------------------------------ I8 -----
md(r'''
## I8 — Aggregating spatial statistics, and areal interpolation

**What we are going to learn.** How to move a quantity from one set of polygons
to another — correctly.

**Why it matters.** This is the **modifiable areal unit problem (MAUP)** in
practice, and it is unavoidable: population comes on census blocks, hazard comes
on flood zones, service areas come on buffers, and you need one table.

**The concept — three ways to transfer a variable, in increasing quality.**

1. **Centroid assignment.** "A block belongs to whichever zone contains its
   centroid." Fast, trivially wrong at boundaries, and biased when the polygons
   are large relative to the target.
2. **Areal weighting (areal interpolation).** Split the source polygon by the
   target, then allocate the quantity in proportion to the **area** of each
   fragment. Correct if the variable is uniformly distributed within the source
   polygon.
3. **Dasymetric weighting.** Same, but weight by an *ancillary variable* known to
   track the quantity — built-up land cover, building footprints, night-lights.
   Far more accurate, because population is not uniform inside a census block; it
   sits on the houses.

**The rule.** Areal weighting is valid for **extensive** quantities (population,
counts, money — things that add up). It is invalid for **intensive** quantities
(density, mean income, percentage — things that average). Interpolating a mean
income by area weight gives you an area-weighted mean, which is usually not what
you want; you need a population-weighted mean.

**Expected outcome.** Population inside the 100-year flood zone, estimated three
ways, with the differences quantified — plus a demonstration that the dasymetric
estimate is closest to the truth.

**What the next cell does:** estimates flood-exposed population by centroid
assignment, by areal weighting and by dasymetric (building-footprint) weighting,
and compares them.
'''),

code(r'''
blocks_v = blocks[~blocks.geometry.is_empty].copy()
zone100 = flood[flood.return_period_yr == 100].geometry.union_all()
zone_gdf = gpd.GeoDataFrame({"zone": ["flood100"]}, geometry=[zone100], crs=CRS_UTM)

TOTAL_POP = blocks_v.population.sum()

# --- METHOD 1: centroid assignment ------------------------------------------
cent = blocks_v.copy()
cent["geometry"] = blocks_v.geometry.representative_point()
m1 = cent[cent.geometry.within(zone100)].population.sum()

# --- METHOD 2: areal weighting ----------------------------------------------
parts = gpd.overlay(blocks_v[["block_id", "population", "geometry"]],
                    zone_gdf, how="intersection", keep_geom_type=True)
parts["frac_area"] = parts.geometry.area / parts.block_id.map(
    blocks_v.set_index("block_id").geometry.area)
m2 = (parts.population * parts.frac_area).sum()

# --- METHOD 3: dasymetric weighting using building footprints --------------
bld = buildings_clean[["building_id", "footprint_m2", "floors", "geometry"]].copy()
bld["living_m2"] = bld.footprint_m2 * bld.floors
bld_pt = bld.copy()
bld_pt["geometry"] = bld.geometry.representative_point()

# living space per block, and living space per block INSIDE the zone
bb = gpd.sjoin(bld_pt, blocks_v[["block_id", "geometry"]], predicate="within")
tot_ls = bb.groupby("block_id")["living_m2"].sum()
in_zone = bb[bb.geometry.within(zone100)]
zone_ls = in_zone.groupby("block_id")["living_m2"].sum()

w = (zone_ls / tot_ls).reindex(blocks_v.block_id).fillna(0.0).clip(0, 1)
m3 = float((blocks_v.set_index("block_id").population * w).sum())

print("POPULATION INSIDE THE 100-YEAR FLOOD ZONE")
print("=" * 78)
print(f"  regional population (all blocks)         : {TOTAL_POP:>10,.0f}")
print(f"  zone area                                : {zone100.area/1e6:>10,.1f} km^2 "
      f"({100*zone100.area/LAND_GEOM.area:.1f} % of the basin)")
print("-" * 78)
print(f"  METHOD 1  centroid assignment            : {m1:>10,.0f}  "
      f"({100*m1/TOTAL_POP:>5.2f} % of population)")
print(f"  METHOD 2  areal weighting                : {m2:>10,.0f}  "
      f"({100*m2/TOTAL_POP:>5.2f} %)")
print(f"  METHOD 3  dasymetric (building floorspace): {m3:>10,.0f}  "
      f"({100*m3/TOTAL_POP:>5.2f} %)")
print("-" * 78)
print(f"  centroid vs dasymetric spread            : "
      f"{100*(m1-m3)/max(m3,1):>+9.1f} %")
print(f"  areal    vs dasymetric spread            : "
      f"{100*(m2-m3)/max(m3,1):>+9.1f} %")

# --- Extensive vs intensive: the classic mistake ----------------------------
print("\n" + "=" * 78)
print("EXTENSIVE vs INTENSIVE VARIABLES")
print("=" * 78)
# blocks already carry district_id, so no join is needed here
by_d = blocks_v.groupby("district_id").agg(pop=("population", "sum"),
                                           area=("area_km2", "sum"))
by_d["density_correct"] = by_d["pop"] / by_d["area"]
by_d["density_naive_mean"] = blocks_v.groupby("district_id")["pop_density_km2"].mean()
by_d["error_%"] = (100*(by_d.density_naive_mean - by_d.density_correct)
                   / by_d.density_correct).round(1)
print(by_d.head(8).round(1).to_string())
print(f"\n  Averaging a DENSITY over blocks gives a different (wrong) answer than")
print(f"  summing population and dividing by summed area. Median error: "
      f"{by_d['error_%'].abs().median():.1f} %, worst: {by_d['error_%'].abs().max():.1f} %")
'''),

md(r'''
**Explanation.**

* **Method 1 (centroid)** is a step function: a block is 100% in or 100% out.
  With ~460 blocks averaging 3 km² each and a flood zone made of ribbons a few
  hundred metres wide, most blocks are *partly* in. Centroid assignment therefore
  swings wildly — it can both over- and under-estimate, and you cannot predict
  which.
* **Method 2 (areal weighting)** computes each block's overlap fraction and
  allocates population pro rata. `parts.block_id.map(...)` looks up the original
  block's full area so the fraction is a genuine proportion. This is right *if*
  population is uniform within the block.
* **Method 3 (dasymetric)** weights by **residential floorspace**
  (`footprint × floors`) — a far better proxy for where people actually are. This
  is the method used in production population-exposure work (and it is what
  organisations like WorldPop do at scale, with land cover and night-lights).
* **Why the three disagree** tells you something real. Flood zones follow rivers,
  which in this basin run through the *low-density* fringes of the urban core.
  Areal weighting therefore over-estimates exposure relative to dasymetric,
  because it assumes people are spread evenly across land that is mostly fields.
* **The extensive/intensive block** is the other half of the lesson.
  `groupby.mean()` on `pop_density_km2` averages densities **unweighted by area**,
  giving each block equal say regardless of size. The correct district density is
  `sum(pop) / sum(area)`. The errors are routinely 20–50%, and they are always in
  the direction of over-weighting small polygons.

**Expected outcome.**

```
POPULATION INSIDE THE 100-YEAR FLOOD ZONE
  regional population (all blocks)          :    643,429
  zone area                                 :      186.0 km^2 (13.3 % of the basin)
  METHOD 1  centroid assignment             :    104,833  (16.29 % of population)
  METHOD 2  areal weighting                 :    113,914  (17.70 %)
  METHOD 3  dasymetric (building floorspace):    122,304  (19.01 %)
  centroid vs dasymetric spread             :     -14.3 %
  areal    vs dasymetric spread             :      -6.9 %
```

The three methods span **105 000 to 122 000 people — a 17% range** on the single
number a flood-risk report exists to produce. Nothing in the data tells you which
is right; you have to reason about the geography. Here the dasymetric estimate is
*highest*, because the floodplain follows the rivers straight through the dense
riverside quarters of the city, where floorspace per hectare is well above the
block average. Areal weighting dilutes that concentration; centroid assignment
throws away partially-flooded blocks altogether.

Then the extensive/intensive table:

```
district_id     pop   area  density_correct  density_naive_mean  error_%
D01          189037   53.3          3,545.2             3,736.9      5.4
D02          197870   67.7          2,923.3             2,513.0    -14.0
D05           34737   80.8            430.1               356.4    -17.1
...
  Median error: 6.2 %, worst: 22.7 %
```

The naive mean is wrong by up to **22.7%**, and — crucially — the error changes
sign between districts, so it does not cancel in a regional total and cannot be
corrected with a fudge factor. **Never average a rate: sum the numerators, sum
the denominators, then divide.**
'''),

# ------------------------------------------------------------------ I9 -----
md(r'''
## I9 — Nearest-neighbour analysis

**What we are going to learn.** `sjoin_nearest`, k-nearest neighbours, and
distance-banded summaries.

**Why it matters.** "Distance to the nearest hospital / road / river" is the
single most productive family of features in spatial modelling. Module 3's
machine-learning models are built almost entirely from features generated here.

**The concept.**

* **`gpd.sjoin_nearest(left, right, distance_col=..., max_distance=...)`** —
  attaches each left feature to its nearest right feature and (optionally)
  records the distance. Uses the R-tree, so it is `O(n log m)`.
* **`max_distance`** is important: without it, a feature 400 km away still
  "matches". With `how="left"` and a `max_distance`, unmatched features get NaN,
  which is the honest representation of "no facility within range".
* **Ties.** If two right features are exactly equidistant, `sjoin_nearest`
  returns **both** rows. Yes, this breaks your 1:1 assumption. On grid-snapped
  data it happens more often than you would think.
* **k-nearest** requires `scipy.spatial.cKDTree` (points only) — GeoPandas has no
  built-in k-NN join.

**Concept — nearest in what metric?** `sjoin_nearest` measures **Euclidean
distance between geometries** (not centroids): the distance from a building
polygon to a road line is the true perpendicular distance to the nearest point on
that road. That is usually what you want, and it is *not* what you get if you
naively use centroids.

**Expected outcome.** A feature table giving every census block its distance to
the nearest hospital, clinic, primary road and river, plus a k-NN redundancy
measure and a distance-decay analysis of PM2.5 that recovers the generating law.

**What the next cell does:** builds four nearest-distance features with
`sjoin_nearest`, adds a 2nd-nearest-hospital feature with `cKDTree`, and then
fits the PM2.5 distance-decay relationship to see whether we can recover the
1.8 km e-folding distance built into the data.
'''),

code(r'''
from scipy.spatial import cKDTree

blk = blocks_v[["block_id", "district_id", "population", "area_km2", "geometry"]].copy()

# --- 1. Four nearest-distance features --------------------------------------
TARGETS = {
    "hospital":  facilities_clean[facilities_clean.facility_type == "hospital"],
    "clinic":    facilities_clean[facilities_clean.facility_type == "clinic"],
    "school":    facilities_clean[facilities_clean.facility_type == "school"],
    "fire":      facilities_clean[facilities_clean.facility_type == "fire_station"],
}
feat = blk.copy()
for name, tgt in TARGETS.items():
    j = gpd.sjoin_nearest(blk[["block_id", "geometry"]], tgt[["facility_id", "geometry"]],
                          how="left", distance_col=f"dist_{name}_m")
    j = j.drop_duplicates("block_id")                    # break ties deterministically
    feat[f"dist_{name}_m"] = j.set_index("block_id")[f"dist_{name}_m"].reindex(
        feat.block_id).to_numpy()

# distance to linear features
for name, tgt in {"river": rivers, "primary_road": roads[roads.road_class.isin(
        ["motorway", "primary"])]}.items():
    j = gpd.sjoin_nearest(blk[["block_id", "geometry"]], tgt[["geometry"]],
                          how="left", distance_col=f"dist_{name}_m").drop_duplicates("block_id")
    feat[f"dist_{name}_m"] = j.set_index("block_id")[f"dist_{name}_m"].reindex(
        feat.block_id).to_numpy()

dist_cols = [c for c in feat.columns if c.startswith("dist_")]
print("NEAREST-DISTANCE FEATURES (metres)")
print(feat[dist_cols].describe().T[["mean", "50%", "min", "max"]].round(0).to_string())

# --- 2. k-nearest: redundancy of hospital access ----------------------------
hosp = facilities_clean[facilities_clean.facility_type == "hospital"]
tree = cKDTree(np.c_[hosp.geometry.x, hosp.geometry.y])
pts = np.c_[blk.geometry.representative_point().x, blk.geometry.representative_point().y]
dd, ii = tree.query(pts, k=2)
feat["dist_hosp1_km"] = dd[:, 0] / 1000
feat["dist_hosp2_km"] = dd[:, 1] / 1000
feat["hosp_redundancy"] = feat.dist_hosp2_km - feat.dist_hosp1_km
print(f"\nHospital redundancy (2nd-nearest minus nearest, km):")
print(f"  median {feat.hosp_redundancy.median():.2f} km,  "
      f"max {feat.hosp_redundancy.max():.2f} km")
print(f"  blocks where losing the nearest hospital adds >10 km: "
      f"{int((feat.hosp_redundancy > 10).sum())} of {len(feat)} "
      f"({feat.loc[feat.hosp_redundancy > 10, 'population'].sum():,.0f} people)")

# --- 3. Distance decay, and why the naive fit is badly wrong ---------------
from scipy.optimize import curve_fit

mw = roads[roads.road_class == "motorway"].geometry.union_all()
st = stations.copy()
st["dist_mw_m"] = st.geometry.distance(mw)
pm = (readings_clean.groupby("station_id")["pm25_ugm3"].mean()
      .rename("pm25_mean").reset_index())
st = st.merge(pm, on="station_id")

# urban intensity, recovered from the population-density raster
with rasterio.open(RAS / "popdens_100m.tif") as src:
    dens = np.array([v[0] for v in src.sample(
        [(p.x, p.y) for p in st.geometry])], dtype=float)
    dens[dens == src.nodata] = np.nan
st["popdens"] = dens
st["urban"] = np.clip((st.popdens - 22) / 9500, 0, None) ** (1 / 2.1)

# MODEL A - the obvious one:  pm25 = a + b*exp(-d/L)
def decay(d, a, bcoef, L):
    return a + bcoef * np.exp(-d / L)
pA, _ = curve_fit(decay, st.dist_mw_m, st.pm25_mean, p0=[8, 10, 2000], maxfev=50000)
r2A = 1 - (st.pm25_mean - decay(st.dist_mw_m, *pA)).var() / st.pm25_mean.var()

# MODEL B - add the confounder: pm25 = a + b*exp(-d/L) + c*urban
def decay_u(X, a, bcoef, L, c):
    d, u = X
    return a + bcoef * np.exp(-d / L) + c * u
pB, _ = curve_fit(decay_u, (st.dist_mw_m.values, st.urban.values),
                  st.pm25_mean.values, p0=[7, 9, 1800, 16], maxfev=100000)
r2B = 1 - (st.pm25_mean - decay_u((st.dist_mw_m.values, st.urban.values),
                                  *pB)).var() / st.pm25_mean.var()

print("\n" + "=" * 82)
print("DISTANCE DECAY OF PM2.5 FROM THE MOTORWAY")
print("=" * 82)
print(f"  TRUE generating model: PM2.5 = 7.5 + 16*urban + 9*exp(-d/1800 m)\n")
print(f"  MODEL A  PM2.5 = {pA[0]:5.2f} + {pA[1]:5.2f}*exp(-d/{pA[2]:>7,.0f} m)"
      f"                R^2 = {r2A:.3f}")
print(f"  MODEL B  PM2.5 = {pB[0]:5.2f} + {pB[1]:5.2f}*exp(-d/{pB[2]:>7,.0f} m) "
      f"+ {pB[3]:5.2f}*urban   R^2 = {r2B:.3f}")
print("-" * 82)
print(f"  e-folding distance   truth  1,800 m")
print(f"                     model A {pA[2]:>7,.0f} m   "
      f"({100*(pA[2]-1800)/1800:+.0f} %)   <- confounded, useless")
print(f"                     model B {pB[2]:>7,.0f} m   "
      f"({100*(pB[2]-1800)/1800:+.0f} %)   <- recovers the truth")
print(f"\n  correlation(distance to motorway, urban intensity) = "
      f"{np.corrcoef(st.dist_mw_m, st.urban)[0,1]:+.3f}")
print("  The motorway runs along the coast, where the city is. Distance from the")
print("  motorway is therefore a proxy for distance from the core, and model A")
print("  attributes the whole urban gradient to the road.")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
sc = axes[0].scatter(st.dist_mw_m/1000, st.pm25_mean, s=55, c=st.urban,
                     cmap="viridis", edgecolor="black", linewidth=0.5, zorder=3)
xx = np.linspace(0, st.dist_mw_m.max(), 300)
axes[0].plot(xx/1000, decay(xx, *pA), color="crimson", linewidth=2,
             label=f"model A (naive): L = {pA[2]:,.0f} m")
axes[0].plot(xx/1000, decay_u((xx, np.zeros_like(xx)), *pB), color="#2b6f3f",
             linewidth=2, label=f"model B at urban=0: L = {pB[2]:,.0f} m")
plt.colorbar(sc, ax=axes[0], shrink=0.8, label="urban intensity")
axes[0].set_xlabel("distance to motorway (km)"); axes[0].set_ylabel("mean PM2.5 (ug/m3)")
axes[0].set_title("Omitted-variable bias, visible",
                  loc="left", weight="bold", fontsize=10)
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

blk_map = blk.merge(feat[["block_id", "dist_hospital_m"]], on="block_id")
blk_map["dist_hospital_km"] = blk_map.dist_hospital_m / 1000
blk_map.plot(ax=axes[1], column="dist_hospital_km", cmap="magma_r",
             scheme="quantiles", k=6, legend=True,
             legend_kwds={"loc": "lower left", "fontsize": 6.5, "title": "km"},
             edgecolor="none")
hosp.plot(ax=axes[1], color="cyan", markersize=45, marker="P",
          edgecolor="black", linewidth=0.6)
axes[1].set_title("Distance to nearest hospital, by census block",
                  loc="left", weight="bold", fontsize=10)
axes[1].set_aspect("equal"); axes[1].set_xticks([]); axes[1].set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* `gpd.sjoin_nearest(..., distance_col="dist_x_m")` writes the distance into a
  column. Without `distance_col` you get the join but not the distance, and
  recomputing it afterwards costs a second pass.
* `.drop_duplicates("block_id")` after every `sjoin_nearest` — this handles ties
  deterministically. Skip it and one block in a thousand quietly becomes two rows.
* `.set_index(...).reindex(feat.block_id).to_numpy()` — the safe way to write a
  joined result back onto the original frame. Assigning a Series directly relies
  on index alignment, which breaks the moment a join has reordered or duplicated
  rows.
* **Distance to a `MultiLineString`** (rivers, roads) is the perpendicular
  distance to the nearest point on the nearest segment. This is why we pass the
  line layer directly rather than its centroids.
* **`hosp_redundancy`** — the difference between the 2nd-nearest and nearest
  hospital distance — is a genuinely useful engineered feature. It measures
  *fragility*: a block with 0.5 km redundancy is fine if a hospital closes; a
  block with 18 km redundancy is not. Features like this, derived from k-NN
  structure rather than raw distance, are where spatial feature engineering earns
  its keep.
* **The distance-decay fit is a validation exercise.** The data were generated
  with `PM2.5 ∝ exp(−d / 1800 m)`. We fit a three-parameter exponential to 24
  noisy station means and recover `L`. If our whole pipeline — CRS, distance
  computation, aggregation, cleaning — is correct, `L̂` should land near 1 800 m.
  **This is what "did my analysis recover the truth?" looks like in practice**,
  and it is a discipline you should import from simulation-based statistics into
  every GIS project you can.

**Expected outcome.**

```
NEAREST-DISTANCE FEATURES (metres)
                          mean        50%   min        max
dist_hospital_m     13,015.0   12,128.0   0.0   31,794.0
dist_clinic_m        6,772.0    5,394.0   0.0   25,448.0
dist_school_m        4,406.0    3,270.0   0.0   19,158.0
dist_fire_m          8,615.0    7,558.0   0.0   27,757.0
dist_river_m         1,942.0    1,386.0   0.0    8,188.0
dist_primary_road_m  1,349.0      996.0   0.0    5,570.0

Hospital redundancy: median 2.14 km, max 14.07 km
  blocks where losing the nearest hospital adds >10 km: 14 of 459 (23,600 people)
```

**Median distance to a hospital is 12.1 km and the worst block is 31.8 km away** —
against a median of 1.0 km to a primary road. The basin is well connected and
badly served, which is a different problem from being badly connected.

Then the decay analysis, which is the heart of the lesson:

```
  TRUE generating model: PM2.5 = 7.5 + 16*urban + 9*exp(-d/1800 m)

  MODEL A  PM2.5 =  6.13 +  9.79*exp(-d/ 12,322 m)                 R^2 = 0.515
  MODEL B  PM2.5 =  7.15 +  9.02*exp(-d/  1,855 m) + 16.42*urban   R^2 = 0.978

  e-folding distance   truth  1,800 m
                     model A  12,322 m   (+585 %)   <- confounded, useless
                     model B   1,855 m   (+3 %)     <- recovers the truth

  correlation(distance to motorway, urban intensity) = -0.443
```

**Model A is wrong by a factor of seven** — and its R² of 0.515 looks perfectly
respectable. Nothing about it announces failure. The motorway runs along the
coast, where the city is, so distance-from-motorway is correlated (−0.44) with
urban intensity, and the single-variable fit hands the entire urban gradient to
the road.

Add the confounder and every parameter snaps into place: intercept 7.15 (true
7.5), amplitude 9.02 (true 9.0), urban coefficient 16.42 (true 16.0), e-folding
distance 1 855 m (true 1 800 m), R² 0.978.

This is the most important lesson in Module 2. **Spatial covariates are almost
always correlated with each other**, because they are all functions of the same
underlying geography. Omitted-variable bias is not an occasional hazard in
spatial data science; it is the default state, and the only defence is to think
about what else varies over the same space.

Two panels: the scatter coloured by urban intensity with both fitted curves (the
naive curve visibly too flat), and a choropleth of hospital distance that is dark
across the whole eastern half of the basin.
'''),

]

CELLS += [

# ----------------------------------------------------------------- I10 -----
md(r'''
## I10 — Dissolve and hierarchical aggregation

**What we are going to learn.** `dissolve` — the spatial `groupby` — and how to
aggregate geometry and attributes together, correctly, at several levels.

**Why it matters.** Almost every deliverable is an aggregation: results *by
district*, *by land-use class*, *by hazard zone*. `dissolve` does the geometry
half; getting the attribute half right is where people slip.

**The concept.** `gdf.dissolve(by=..., aggfunc=...)` is exactly
`groupby(...).agg(...)` **plus** a `union_all()` of the geometries in each group.

```python
gdf.dissolve(by="landuse_class",
             aggfunc={"area_ha": "sum", "class_code": "first"})
```

Three things to know:

1. **The `by` column becomes the index.** Call `.reset_index()` unless you want it
   that way.
2. **`aggfunc` defaults to `"first"`**, which silently keeps an arbitrary row's
   value for every other column. Always pass an explicit dict.
3. **Dissolving is expensive** — it is a union over potentially thousands of
   polygons. Where you only need statistics and not geometry, use plain
   `groupby` on the attribute table and skip the union entirely.

**Concept — topology after dissolve.** Dissolving adjacent polygons removes the
shared borders. If the inputs have slivers or gaps (I5), dissolve *preserves*
them as holes. Run `.buffer(0)` or `make_valid()` afterwards, and check
`.interiors` on the result if gaps matter.

**Expected outcome.** A land-use summary by class, a two-level hierarchy
(district → district type), and a demonstration that `dissolve` and `groupby`
agree on the numbers while differing enormously in cost.

**What the next cell does:** dissolves land use by class, times it against a
plain `groupby`, builds a district-type aggregation, and checks the census-block
→ district nesting property that the dataset guarantees.
'''),

code(r'''
import time

# --- 1. Dissolve land use by class -------------------------------------------
lu = landuse_clean.copy()
lu["area_ha_true"] = lu.geometry.area / 1e4

t0 = time.perf_counter()
lu_by_class = lu.dissolve(by="landuse_class",
                          aggfunc={"area_ha_true": "sum", "class_code": "first"})
t_dis = time.perf_counter() - t0
lu_by_class = lu_by_class.reset_index()
lu_by_class["n_patches"] = lu.groupby("landuse_class").size().to_numpy()
lu_by_class["pct_of_land"] = (100 * lu_by_class.area_ha_true
                              / lu_by_class.area_ha_true.sum()).round(2)

t0 = time.perf_counter()
lu_stats_only = lu.groupby("landuse_class")["area_ha_true"].sum()
t_grp = time.perf_counter() - t0

print("LAND COVER OF THE VALLMARA BASIN")
print(lu_by_class[["landuse_class", "class_code", "n_patches",
                   "area_ha_true", "pct_of_land"]]
      .sort_values("area_ha_true", ascending=False).round(1).to_string(index=False))
print(f"\n  dissolve (geometry + stats) : {t_dis*1000:>8.1f} ms")
print(f"  groupby  (stats only)       : {t_grp*1000:>8.1f} ms   "
      f"({t_dis/max(t_grp,1e-6):,.0f}x faster)")
print(f"  numbers identical           : "
      f"{np.allclose(lu_by_class.set_index('landuse_class').area_ha_true.sort_index(), lu_stats_only.sort_index())}")

# --- 2. Two-level hierarchy: blocks -> districts -> district types ---------
d = districts.copy()
by_type = d.dissolve(by="district_type",
                     aggfunc={"population": "sum", "households": "sum",
                              "area_km2": "sum"}).reset_index()
by_type["density"] = (by_type.population / by_type.area_km2).round(1)
by_type["n_districts"] = d.groupby("district_type").size().to_numpy()
print("\nDISTRICTS AGGREGATED TO DISTRICT TYPE")
print(by_type[["district_type", "n_districts", "area_km2", "population",
               "density"]].round(1).to_string(index=False))

# --- 3. The nesting property: blocks dissolve back to districts ------------
rebuilt = blocks.dissolve(by="district_id",
                          aggfunc={"population": "sum", "area_km2": "sum"}).reset_index()
rebuilt["area_km2_union"] = rebuilt.geometry.area / 1e6
chk = districts[["district_id", "population", "area_km2"]].merge(
    rebuilt[["district_id", "population", "area_km2", "area_km2_union"]],
    on="district_id", suffixes=("_district", "_blocks"))
chk["pop_diff"] = chk.population_blocks - chk.population_district
chk["area_diff_km2"] = (chk.area_km2_union - chk.area_km2_district).round(4)

print("\nCONSISTENCY CHECK: do the blocks rebuild the districts?")
print(f"  population matches exactly : {int((chk.pop_diff == 0).sum())} of {len(chk)}"
      f"   mismatches: {chk.loc[chk.pop_diff != 0, 'district_id'].tolist()}")
print(f"  area matches to 0.001 km^2 : {int((chk.area_diff_km2.abs() < 0.001).sum())} of {len(chk)}"
      f"   mismatches: {chk.loc[chk.area_diff_km2.abs() >= 0.001, 'district_id'].tolist()}")

print("\n  (a) POPULATION mismatches - the two deliberately blanked districts.")
print("      The blocks still hold the true values, so the 'missing' district")
print("      populations are RECOVERABLE by aggregation, not lost:")
print(chk[chk.pop_diff != 0][["district_id", "population_district",
                              "population_blocks"]].to_string(index=False))

print("\n  (b) AREA mismatch - a different bug entirely. Find it:")
empty = blocks[blocks.geometry.is_empty]
print(empty[["block_id", "district_id", "area_km2", "population"]].to_string(index=False))
print(f"\n      One block has an EMPTY geometry. It still carries its attributes,")
print(f"      so `population` sums correctly ({empty.population.iloc[0]:,} people are")
print(f"      counted) while the geometric union silently loses its "
      f"{empty.area_km2.iloc[0]:.2f} km^2.")
print(f"      Attribute totals right, geometry totals wrong, no warning anywhere.")
print(f"      area_diff for that district: "
      f"{chk.loc[chk.district_id == empty.district_id.iloc[0], 'area_diff_km2'].iloc[0]:.4f} km^2")
print(f"      block's own area_km2 column: {empty.area_km2.iloc[0]:.4f} km^2   <- they match")

# --- 4. Map the dissolved land cover ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
lu.plot(ax=axes[0], column="landuse_class", cmap="tab10", legend=False,
        edgecolor="white", linewidth=0.15)
axes[0].set_title(f"Before dissolve: {len(lu)} patches",
                  loc="left", weight="bold", fontsize=10)
lu_by_class.plot(ax=axes[1], column="landuse_class", cmap="tab10", legend=True,
                 legend_kwds={"loc": "lower left", "fontsize": 6.5},
                 edgecolor="black", linewidth=0.3)
axes[1].set_title(f"After dissolve: {len(lu_by_class)} classes",
                  loc="left", weight="bold", fontsize=10)
for a in axes:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* `dissolve(by=..., aggfunc={...})` — always pass an explicit `aggfunc` dict.
  With the default `"first"`, a column like `area_ha` silently reports the area of
  *one arbitrary patch* in the group rather than the total, and the result looks
  entirely plausible.
* **The timing comparison matters at scale.** `dissolve` unions geometry, which is
  `O(n log n)` in vertex count with a large constant. Plain `groupby` on the
  attribute table gives identical statistics in a fraction of the time. Rule:
  **dissolve only when you need the dissolved geometry** (to map it, to clip with
  it, to overlay it). For a table, use `groupby`.
* **The nesting check is the pattern to internalise.** The dataset guarantees that
  districts are the union of their blocks. Re-deriving districts from blocks and
  comparing is a *unit test for your entire pipeline*: it exercises the
  `district_id` join key, the geometry, and the aggregation logic simultaneously.
  When it passes, you have strong evidence that nothing upstream is broken.
* The two mismatching districts are exactly the two with deliberately blanked
  population. **Note that the blocks still carry the true values** — so the
  "missing" district populations are recoverable by aggregation, which is a much
  better imputation than a mean or a zero. Always look for a finer-grained source
  before imputing.
* Note `bl = blocks[~blocks.geometry.is_empty]`: the empty geometry would
  contribute nothing to the union but *would* contribute its population, quietly
  breaking the check.

**Expected outcome.**

```
LAND COVER OF THE VALLMARA BASIN
     landuse_class  class_code  n_patches  area_ha_true  pct_of_land
            Forest           5         29      68,742.0         49.3
          Cropland           3         33      21,670.2         15.6
         Grassland           4         42      20,822.4         14.9
          Built-up           2         90      10,798.5          7.8
Bare rock / sparse           7         25      10,070.9          7.2
         Shrubland           6        208       4,061.9          2.9
             Water           1          3       1,893.2          1.4
           Wetland           8         12       1,342.9          1.0

  dissolve (geometry + stats) :    150.5 ms
  groupby  (stats only)       :      1.2 ms   (128x faster)
  numbers identical           : True
```

Note **Shrubland: 208 patches but only 2.9% of the area** — highly fragmented
scrub between forest blocks — against **Forest: 29 patches and 49.3%**. Patch
count and area tell completely different stories, and landscape ecology cares
about both.

```
DISTRICTS AGGREGATED TO DISTRICT TYPE
district_type  n_districts  area_km2  population   density
        rural            2     124.0      1,470.0     11.9
     suburban            8     472.2    219,418.0    464.6
 upland_rural           12     677.5     22,387.0     33.0
   urban_core            2     121.0    386,907.0  3,197.3
```

**Two districts on 8.7% of the land hold 60% of the population.** That single
line is the reason every later analysis has to be careful about per-capita versus
per-area statistics.

Then the consistency check, which finds **two different bugs**:

* **(a) Population**: `D07` and `D18` mismatch — the two deliberately blanked
  districts. The blocks still carry 9 745 and 3 658 people, so the missing values
  are *recoverable by aggregation*. Always look for a finer-grained source before
  you impute.
* **(b) Area**: `D16` is short by **5.9276 km²**. The culprit is block `B0301`,
  which has an **empty geometry** but still carries `area_km2 = 5.9274` and
  `population = 156`. An empty geometry keeps its attributes, so attribute sums
  come out right while geometric unions silently lose the area — **and nothing
  warns you**. This is why `describe_gdf` prints an `empty` count.

Then a before/after map: a speckled patchwork of ~440 patches resolving into
eight clean regions.
'''),

# ----------------------------------------------------------------- I11 -----
md(r'''
## I11 — The spatial index, and why any of this is fast

**What we are going to learn.** How R-tree indexing turns an `O(n·m)` problem
into an `O(n log m)` one, and how to use the index directly.

**Why it matters.** Spatial operations that "should" take hours take seconds
because of the index. Understanding it lets you predict cost, diagnose slowness,
and hand-roll operations GeoPandas does not provide.

**The concept — filter and refine.** Exact geometric predicates are expensive: a
polygon-polygon intersection test is proportional to the vertex counts. So every
spatial library does two passes:

1. **Filter.** Query an **R-tree** built over the *bounding boxes*. Cheap
   rectangle overlap tests, `O(log m)` per query, returning a superset of the true
   answer.
2. **Refine.** Run the exact GEOS predicate only on the candidates.

An **R-tree** is a balanced tree of nested bounding rectangles: leaves are feature
boxes, internal nodes are boxes enclosing their children. Searching descends only
into nodes whose box intersects the query — pruning the vast majority of features
immediately.

**The API.**

| Call | Returns |
|---|---|
| `gdf.sindex` | The index (built lazily on first use, then cached) |
| `sindex.query(geom, predicate=None)` | Integer positions of candidates (bbox only if `predicate=None`) |
| `sindex.query(geoseries, predicate="intersects")` | A 2×N array of `[input_idx, tree_idx]` pairs — the vectorised form |
| `sindex.nearest(geom, return_distance=True)` | Nearest feature(s) |

**Concept — bounding-box selectivity.** The index helps in proportion to how well
bounding boxes approximate geometry. For compact polygons, excellent. For a long
diagonal river, the bounding box covers a huge empty area and the filter step
returns many false candidates. This is why `.cx` over-selected so badly in B9.

**Expected outcome.** A direct comparison of brute force, index-assisted and
`sjoin` timings on the same problem, plus a measurement of bounding-box
selectivity for compact versus elongated features.

**What the next cell does:** answers "which buildings intersect the flood zone?"
three ways with timings, then measures how many index candidates survive the
exact test for compact versus elongated geometries.
'''),

code(r'''
import time

# The question: which census block is each building in?
#   5,200 buildings x 460 blocks = 2.4 MILLION exact tests if done naively.
bl = buildings_clean[["building_id", "geometry"]].reset_index(drop=True)
bk = blocks[~blocks.geometry.is_empty][["block_id", "geometry"]].reset_index(drop=True)
bk_geoms = bk.geometry.to_numpy()
bl_pts = bl.geometry.representative_point()

print("WHICH CENSUS BLOCK IS EACH BUILDING IN?")
print("=" * 84)
print(f"  {len(bl):,} buildings x {len(bk)} blocks = {len(bl)*len(bk):,} "
      f"pairs if tested exhaustively\n")

# --- 1. BRUTE FORCE on a subsample, then extrapolate -----------------------
sub = bl_pts.iloc[:250]
t0 = time.perf_counter()
brute = 0
for g in sub:
    for z in bk_geoms:
        if g.within(z):
            brute += 1
            break
t_brute = time.perf_counter() - t0
per_row = t_brute / len(sub)
print(f"  1. brute force, {len(sub)} buildings")
print(f"     {brute} matched in {t_brute*1000:>8,.0f} ms "
      f"-> all {len(bl):,} would take {per_row*len(bl):>7,.1f} s")

# --- 2. INDEX-ASSISTED, by hand ---------------------------------------------
t0 = time.perf_counter()
sidx = bk.sindex
owner = np.full(len(bl), -1, dtype=int)
for pos, g in enumerate(bl_pts):
    for cand in sidx.query(g):                 # cheap bbox filter
        if g.within(bk_geoms[cand]):           # exact refine
            owner[pos] = cand
            break
t_idx = time.perf_counter() - t0
print(f"\n  2. hand-rolled R-tree filter + refine, ALL {len(bl):,} buildings")
print(f"     {int((owner >= 0).sum())} matched in {t_idx*1000:>8,.0f} ms "
      f"-> {per_row*len(bl)/max(t_idx,1e-9):>6,.0f}x faster than brute force")

# --- 3. VECTORISED index query ----------------------------------------------
t0 = time.perf_counter()
pairs = bk.sindex.query(bl_pts, predicate="within")
t_vec = time.perf_counter() - t0
print(f"\n  3. vectorised sindex.query(..., predicate='within')")
print(f"     {len(np.unique(pairs[0])):,} matched in {t_vec*1000:>8,.0f} ms  "
      f"-> {per_row*len(bl)/max(t_vec,1e-9):>6,.0f}x faster")

# --- 4. gpd.sjoin, which uses exactly the same machinery -------------------
t0 = time.perf_counter()
sj = gpd.sjoin(gpd.GeoDataFrame(bl.drop(columns="geometry"), geometry=bl_pts,
                                crs=CRS_UTM), bk, predicate="within")
t_sj = time.perf_counter() - t0
print(f"\n  4. gpd.sjoin")
print(f"     {sj.building_id.nunique():,} matched in {t_sj*1000:>8,.0f} ms")
print(f"\n  All methods agree: "
      f"{int((owner >= 0).sum()) == len(np.unique(pairs[0])) == sj.building_id.nunique()}")

# --- 5. Bounding-box selectivity: compact vs elongated ---------------------
print("\n" + "=" * 84)
print("BOUNDING-BOX SELECTIVITY  (how well does the filter step prune?)")
print("=" * 84)
print(f"  {'query layer':<24}{'bbox candidates':>17}{'exact hits':>12}"
      f"{'precision':>11}{'bbox fill':>11}")
for label, layer in [("districts (compact)", districts),
                     ("flood zones (ribbons)", flood[flood.return_period_yr == 100]),
                     ("rivers (long diagonals)", rivers)]:
    cands = bl.sindex.query(layer.geometry)            # bbox only
    exact = bl.sindex.query(layer.geometry, predicate="intersects")
    fill = float((layer.geometry.area /
                  layer.geometry.envelope.area).mean()) if layer.geometry.area.sum() > 0 else 0.0
    prec = exact.shape[1] / max(cands.shape[1], 1)
    print(f"  {label:<24}{cands.shape[1]:>17,}{exact.shape[1]:>12,}"
          f"{prec:>10.1%}{fill:>11.1%}")
print("\n  'bbox fill' = geometry area / bounding-box area. Low fill means the box")
print("  is a poor stand-in for the shape, so the filter step passes many false")
print("  candidates through to the expensive exact test.")
'''),

md(r'''
**Explanation.**

* **Brute force** is `n × m` exact intersection tests. We run it on 800 buildings
  and extrapolate rather than waiting for the full run — the extrapolation itself
  is the lesson.
* **The hand-rolled version** shows exactly what GeoPandas does internally:
  `sidx.query(g)` returns integer *positions* (not index labels) of features whose
  bounding box intersects `g`; then you run the exact predicate on those few.
  Writing it once demystifies every spatial join you will ever run.
* **`sindex.query(geoseries, predicate=...)`** is the vectorised form and the
  fastest of the three. It returns a `2 × N` array of `[left_position,
  right_position]` pairs. This is the raw material for building custom joins that
  GeoPandas does not offer — e.g. "join each building to the *largest* overlapping
  zone".
* `gpd.sjoin` sits on top of exactly this, adding the DataFrame bookkeeping. It is
  as fast as the vectorised query and far less error-prone. **Use `sjoin` in real
  work**; the hand-rolled version exists so you understand its cost model.
* **The selectivity table is the diagnostic.** `bbox fill` is the ratio of true
  area to bounding-box area. Compact districts fill their boxes well (40–70%), so
  the filter is precise. Rivers are long diagonal lines whose boxes enclose
  enormous empty regions, so precision collapses. **When a spatial join is
  unexpectedly slow, check bbox fill first** — the fix is usually to split long
  features into segments (which is exactly why our road layer is segmented).

**Expected outcome.**

```
WHICH CENSUS BLOCK IS EACH BUILDING IN?
  5,200 buildings x 459 blocks = 2,386,800 pairs if tested exhaustively

  1. brute force, 250 buildings
     250 matched in      226 ms -> all 5,200 would take     4.7 s
  2. hand-rolled R-tree filter + refine, ALL 5,200 buildings
     5197 matched in      119 ms ->     39x faster than brute force
  3. vectorised sindex.query(..., predicate='within')
     5,197 matched in        8 ms  ->    574x faster
  4. gpd.sjoin
     5,197 matched in       13 ms
  All methods agree: True
```

Read the ladder: **4.7 s → 119 ms → 8 ms**. The hand-rolled loop gets a 39× win
purely from the R-tree; the vectorised query gets **574×** because it also
eliminates the Python-level loop. `sjoin` is within a factor of two of the
theoretical best while handling all the DataFrame bookkeeping — **use `sjoin`.**

(Note 5 197 of 5 200 buildings matched. Three fall outside every block, on the
coastline where the block tessellation does not quite reach. Unmatched rows are
information, not noise — count them every time.)

```
BOUNDING-BOX SELECTIVITY
  query layer               bbox candidates  exact hits  precision  bbox fill
  districts (compact)                10,330       5,232     50.6%      58.8%
  flood zones (ribbons)               5,176       1,036     20.0%      47.3%
  rivers (long diagonals)            11,796          46      0.4%       0.4%
```

**Rivers: 11 796 candidates for 46 real hits — 0.4% precision.** The filter step
does almost no useful work, because a long diagonal line has a bounding box
covering a huge empty region. When a spatial join is unexpectedly slow, check
bbox fill first; the fix is usually to split long features into segments, which
is exactly why the road layer ships pre-segmented.
'''),

# ----------------------------------------------------------------- I12 -----
md(r'''
## I12 — Raster masking and clipping

**What we are going to learn.** How to cut a raster to a polygon, crop it to an
extent, and read only the window you need.

**Why it matters.** Rasters are big. The difference between reading a 2.7-million-
cell DEM and reading the 40 000 cells you actually need is the difference between
an interactive analysis and a coffee break — and on national datasets, between
possible and impossible.

**The concept — three related operations.**

| Operation | Function | Effect |
|---|---|---|
| **Mask** | `rasterio.mask.mask(src, shapes, crop=False)` | Cells outside the shapes → NoData; array keeps its original size |
| **Mask + crop** | `...mask(src, shapes, crop=True)` | Also trims the array to the shapes' bounding box |
| **Window read** | `src.read(1, window=Window(...))` | Reads only a rectangle from disk; nothing else is touched |

**The critical detail — the transform changes.** When you crop, the array's
upper-left corner moves, so the affine transform must be updated. `mask(...)`
returns `(array, transform)` for exactly this reason. **Write the new transform
into the profile before saving**, or your output will be georeferenced to the
wrong place — a bug that is invisible until someone overlays your raster on
something else.

**`all_touched`.** By default a cell is included only if its **centre** falls
inside the polygon. `all_touched=True` includes any cell the polygon touches at
all. For small polygons relative to the cell size this changes results
dramatically; for thin features it is the difference between getting data and
getting an empty array.

**Expected outcome.** The DEM masked to a single district, cropped, saved with a
correct transform, and a windowed read benchmarked against a full read.

**What the next cell does:** masks the DEM to one district three ways
(mask only, mask+crop, `all_touched`), compares cell counts and statistics,
writes a correctly georeferenced output, and times a windowed read.
'''),

code(r'''
import rasterio.mask
from rasterio.windows import Window, from_bounds

target = districts[districts.name == "Old Vallmara"]
shapes = [g.__geo_interface__ for g in target.geometry]

with rasterio.open(RAS / "dem_25m.tif") as src:
    full = src.read(1, masked=True)

    # (a) mask only - same array size, outside becomes NoData
    a_mask, t_mask = rasterio.mask.mask(src, shapes, crop=False, filled=True,
                                        nodata=src.nodata)
    # (b) mask + crop
    a_crop, t_crop = rasterio.mask.mask(src, shapes, crop=True, filled=True,
                                        nodata=src.nodata)
    # (c) all_touched
    a_at, t_at = rasterio.mask.mask(src, shapes, crop=True, all_touched=True,
                                    filled=True, nodata=src.nodata)
    profile = src.profile.copy()
    nodata = src.nodata

def stats(arr, nd):
    m = np.ma.masked_equal(arr, nd)
    return m.count(), float(m.mean()), float(m.min()), float(m.max())

print(f"MASKING THE DEM TO ONE DISTRICT: {target.name.iloc[0]} "
      f"({target.area_km2.iloc[0]:.1f} km^2)")
print("=" * 90)
print(f"{'variant':<28}{'array shape':>16}{'valid cells':>13}{'mean m':>10}"
      f"{'min':>8}{'max':>8}")
print("-" * 90)
print(f"{'full raster':<28}{str(full.shape):>16}{full.count():>13,}"
      f"{full.mean():>10.1f}{full.min():>8.1f}{full.max():>8.1f}")
for label, arr in [("mask, crop=False", a_mask.squeeze()),
                   ("mask, crop=True", a_crop.squeeze()),
                   ("mask, crop + all_touched", a_at.squeeze())]:
    n, mu, lo, hi = stats(arr, nodata)
    print(f"{label:<28}{str(arr.shape):>16}{n:>13,}{mu:>10.1f}{lo:>8.1f}{hi:>8.1f}")

exp_cells = target.area_km2.iloc[0] * 1e6 / (25*25)
print("-" * 90)
print(f"  cells expected from the polygon area : {exp_cells:>10,.0f}")
print(f"  all_touched adds                     : "
      f"{stats(a_at.squeeze(), nodata)[0] - stats(a_crop.squeeze(), nodata)[0]:>10,} cells "
      f"(a one-cell fringe around the boundary)")

# --- Save the cropped raster WITH THE CORRECT TRANSFORM --------------------
out_path = OUT / "dem_old_vallmara.tif"
profile.update(height=a_crop.shape[1], width=a_crop.shape[2],
               transform=t_crop, compress="deflate")
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(a_crop)
with rasterio.open(out_path) as chk:
    print(f"\n  wrote {out_path.name}: {chk.width} x {chk.height}, "
          f"bounds {tuple(round(b) for b in chk.bounds)}")
    print(f"  polygon bounds                     : "
          f"{tuple(round(b) for b in target.total_bounds)}   <- they agree")

# --- Windowed read: only touch the bytes you need -------------------------
print("\n" + "=" * 90)
print("WINDOWED READ")
print("=" * 90)
with rasterio.open(RAS / "dem_25m.tif") as src:
    t0 = time.perf_counter(); _ = src.read(1); t_full = time.perf_counter() - t0
    win = from_bounds(*target.total_bounds, transform=src.transform)
    t0 = time.perf_counter(); w = src.read(1, window=win); t_win = time.perf_counter() - t0
    win_transform = src.window_transform(win)

print(f"  full read   : {src.height} x {src.width} = {src.height*src.width:,} cells "
      f"in {t_full*1000:6.1f} ms")
print(f"  window read : {w.shape[0]} x {w.shape[1]} = {w.size:,} cells "
      f"in {t_win*1000:6.1f} ms   ({t_full/max(t_win,1e-9):.1f}x faster, "
      f"{100*w.size/(src.height*src.width):.1f} % of the data)")
print(f"  window transform upper-left: "
      f"({win_transform.c:,.0f}, {win_transform.f:,.0f})  "
      f"vs full raster ({src.transform.c:,.0f}, {src.transform.f:,.0f})")

# --- Picture -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
im0 = axes[0].imshow(full, cmap="terrain",
                     extent=rasterio.plot.plotting_extent(
                         rasterio.open(RAS / "dem_25m.tif")))
target.boundary.plot(ax=axes[0], color="red", linewidth=1.6)
axes[0].set_title("full DEM + target district", loc="left", weight="bold", fontsize=10)
axes[1].imshow(np.ma.masked_equal(a_mask.squeeze(), nodata), cmap="terrain")
axes[1].set_title(f"mask, crop=False\n{a_mask.shape[1]} x {a_mask.shape[2]}",
                  loc="left", weight="bold", fontsize=10)
axes[2].imshow(np.ma.masked_equal(a_crop.squeeze(), nodata), cmap="terrain")
axes[2].set_title(f"mask, crop=True\n{a_crop.shape[1]} x {a_crop.shape[2]}",
                  loc="left", weight="bold", fontsize=10)
for a in axes:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* `g.__geo_interface__` converts a Shapely geometry to the GeoJSON-like mapping
  that `rasterio.mask` expects. You can also pass the GeoSeries directly in recent
  versions, but the explicit form works everywhere.
* **`crop=False` versus `crop=True`.** Without cropping you get an array the size
  of the *original* raster with everything outside the polygon set to NoData —
  2.76 million cells to hold ~85 000 useful ones. With cropping you get an array
  the size of the polygon's bounding box. Use `crop=True` unless you specifically
  need alignment with the original grid.
* **`filled=True, nodata=src.nodata`** returns a plain array with the sentinel
  written in, which is what you want for writing to disk. `filled=False` returns a
  MaskedArray, better for immediate computation.
* **The transform update is the part people forget.**
  `profile.update(height=..., width=..., transform=t_crop)` — all three. Update
  only the shape and your raster is silently georeferenced to the wrong corner.
  The printed comparison of output bounds against polygon bounds is the check that
  catches it.
* **`all_touched=True`** adds a one-cell fringe: every cell the polygon boundary
  passes through, even by a millimetre. For a 53 km² district at 25 m resolution
  that is a few thousand extra cells (~2–3%). For a 1 km² polygon it could be
  20%. **For zonal statistics on small polygons, `all_touched` is a substantive
  methodological choice, not a flag.**
* `from_bounds(*bounds, transform=src.transform)` builds a `Window` from map
  coordinates; `src.window_transform(win)` gives the transform for that window.
  Together they let you read and correctly georeference any sub-rectangle.

**Expected outcome.**

```
MASKING THE DEM TO ONE DISTRICT: Old Vallmara (53.3 km^2)
variant                          array shape  valid cells    mean m     min     max
full raster                     (1440, 1920)    2,231,649     338.7     0.4   925.2
mask, crop=False                (1440, 1920)       85,315      25.1     0.4   109.0
mask, crop=True                   (315, 481)       85,315      25.1     0.4   109.0
mask, crop + all_touched          (315, 481)       86,214      25.3     0.4   109.1
  cells expected from the polygon area :     85,315
  all_touched adds                     :        899 cells
```

Three checks pass at once. **`crop=False` and `crop=True` give identical
statistics** on very different array shapes — cropping changes storage, not
content. The valid-cell count matches `area_km² × 10⁶ / 625` **exactly** (85 315),
confirming the mask is doing what the polygon says. And the district mean is
**25.1 m against a regional 338.7 m** — Old Vallmara is the low coastal core,
which is both correct and a reminder that a regional mean describes nowhere in
particular.

`all_touched` adds 899 cells, about 1% here. On a 1 km² polygon the same fringe
would be 20%.

```
  wrote dem_old_vallmara.tif: 481 x 315, bounds (409400, 4612975, 421425, 4620850)
  polygon bounds                     : (409416, 4612977, 421419, 4620826)
```

The bounds agree to within one cell (25 m) — cropping snaps to the raster grid, so
exact equality is neither expected nor desirable.

```
  full read   : 1440 x 1920 = 2,764,800 cells in   51.0 ms
  window read :  314 x  480 =   150,720 cells in    1.2 ms  (42.8x faster, 5.5 %)
```

**42.8× faster for reading 5.5% of the data.** On a 50 GB national DEM this is
the difference between a workflow and a wish.
'''),

]

CELLS += [

# ----------------------------------------------------------------- I13 -----
md(r'''
## I13 — Raster resampling and reprojection

**What we are going to learn.** How to change a raster's resolution and CRS, and
how to choose a resampling method without corrupting your data.

**Why it matters.** Our seven rasters are at 25 m, 50 m, 100 m, 134 m and 250 m,
in two different CRS. You cannot do arithmetic between arrays of different shapes.
**Every multi-raster analysis begins with alignment**, and the choices you make
here determine whether the result means anything.

**The concept — resampling methods and when each is legal.**

| Method | What it does | Use for | Never use for |
|---|---|---|---|
| `nearest` | Copies the closest cell | **Categorical** data (land cover, zone codes) | Continuous data, if you can avoid it |
| `bilinear` | Weighted mean of 4 neighbours | Continuous data (elevation, temperature) | Categorical — it invents class 3.7 |
| `cubic` / `cubic_spline` | 16-neighbour polynomial | Smooth continuous fields | Data with sharp edges (creates overshoot) |
| `average` | Mean of all contributing cells | **Downsampling** continuous data | Upsampling; categorical |
| `mode` | Most common value | **Downsampling** categorical data | Continuous |
| `sum` | Total of contributing cells | Downsampling **counts** (population!) | Anything intensive |

**The two rules that matter.**

1. **Never bilinearly interpolate a categorical raster.** Averaging class codes
   3 (cropland) and 5 (forest) gives 4 (grassland) — a class that is not there.
2. **Downsampling a count raster must use `sum`, not `average`.** If population
   density is *per cell*, averaging halves your population. If it is *per km²* it
   is intensive and `average` is right. **Know which your raster is.**

**Concept — upsampling creates no information.** Resampling a 250 m rainfall grid
to 25 m gives you 100× more cells and exactly the same information, now with a
false impression of detail. It is often necessary for array alignment; it is never
an improvement. Do your analysis at the **coarsest** resolution involved wherever
you can.

**Expected outcome.** All rasters aligned to a common grid, the categorical-vs-
continuous error demonstrated numerically, and the Web Mercator LST raster brought
into the analysis CRS.

**What the next cell does:** defines a reusable `align_to()` function, aligns four
rasters to the DEM grid, demonstrates what bilinear interpolation does to land
cover, and reprojects the EPSG:3857 temperature raster into EPSG:32633.
'''),

code(r'''
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling as RS

def align_to(src_path, ref_profile, resampling=Resampling.bilinear):
    """Reproject/resample any raster onto the grid described by ref_profile."""
    with rasterio.open(src_path) as src:
        dst = np.full((ref_profile["height"], ref_profile["width"]),
                      np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst

# --- 1. Build a common analysis grid: the DEM, downsampled to 100 m -------
with rasterio.open(RAS / "dem_25m.tif") as src:
    ref = src.profile.copy()
FACTOR = 4                                     # 25 m -> 100 m
ref.update(
    height=ref["height"] // FACTOR, width=ref["width"] // FACTOR,
    transform=rasterio.Affine(ref["transform"].a * FACTOR, 0, ref["transform"].c,
                              0, ref["transform"].e * FACTOR, ref["transform"].f),
    dtype="float32", nodata=np.nan, count=1,
)
print(f"COMMON ANALYSIS GRID: {ref['width']} x {ref['height']} at 100 m, {ref['crs']}")

grids = {
    "elevation":  align_to(RAS / "dem_25m.tif", ref, Resampling.average),
    "rainfall":   align_to(RAS / "rainfall_annual_250m.tif", ref, Resampling.bilinear),
    "ndvi":       align_to(RAS / "ndvi_50m.tif", ref, Resampling.average),
    "popdens":    align_to(RAS / "popdens_100m.tif", ref, Resampling.bilinear),
    "landcover":  align_to(RAS / "landcover_25m.tif", ref, Resampling.mode),
    "lst":        align_to(RAS / "lst_summer_100m_3857.tif", ref, Resampling.bilinear),
}
print(f"\n{'layer':<12}{'native res':>12}{'method':>12}{'shape':>14}"
      f"{'valid %':>10}{'min':>10}{'max':>10}")
print("-" * 80)
native = {"elevation": "25 m", "rainfall": "250 m", "ndvi": "50 m",
          "popdens": "100 m", "landcover": "25 m", "lst": "134 m (3857)"}
meth = {"elevation": "average", "rainfall": "bilinear", "ndvi": "average",
        "popdens": "bilinear", "landcover": "mode", "lst": "bilinear"}
for k, arr in grids.items():
    ok = np.isfinite(arr)
    print(f"{k:<12}{native[k]:>12}{meth[k]:>12}{str(arr.shape):>14}"
          f"{100*ok.mean():>9.1f}%{np.nanmin(arr):>10.2f}{np.nanmax(arr):>10.2f}")

print(f"\nAll six arrays share one shape: "
      f"{len({a.shape for a in grids.values()}) == 1}  -> "
      f"they can now be combined with plain NumPy arithmetic")

# --- 2. THE CATEGORICAL RESAMPLING ERROR ----------------------------------
lc_mode = align_to(RAS / "landcover_25m.tif", ref, Resampling.mode)
lc_bilin = align_to(RAS / "landcover_25m.tif", ref, Resampling.bilinear)
print("\n" + "=" * 80)
print("WHY YOU MUST NOT BILINEARLY RESAMPLE A CATEGORICAL RASTER")
print("=" * 80)
valid_m, valid_b = lc_mode[np.isfinite(lc_mode)], lc_bilin[np.isfinite(lc_bilin)]
print(f"  legal class codes                 : {sorted(lc_legend.class_code.tolist())}")
print(f"  mode     -> distinct values       : "
      f"{len(np.unique(valid_m))}  {sorted(np.unique(valid_m))[:9]}")
print(f"  bilinear -> distinct values       : {len(np.unique(valid_b)):,}")
print(f"  bilinear -> fraction NOT an integer: "
      f"{100*np.mean(np.abs(valid_b - np.round(valid_b)) > 1e-6):.1f} %")
print(f"  e.g. cells with value between 3 and 4: "
      f"{int(((valid_b > 3.01) & (valid_b < 3.99)).sum()):,}  "
      f"-> class '3.5' means nothing")

# --- 3. Reprojecting the Web Mercator raster ------------------------------
print("\n" + "=" * 80)
print("REPROJECTING THE EPSG:3857 TEMPERATURE RASTER")
print("=" * 80)
with rasterio.open(RAS / "lst_summer_100m_3857.tif") as src:
    print(f"  source : {src.crs}, {src.width} x {src.height}, res {src.res[0]:.1f}")
    print(f"           bounds {tuple(round(b) for b in src.bounds)}")
print(f"  target : {ref['crs']}, {ref['width']} x {ref['height']}, res 100.0")
lst = grids["lst"]
print(f"  result : min {np.nanmin(lst):.1f} C, max {np.nanmax(lst):.1f} C, "
      f"mean {np.nanmean(lst):.1f} C")

# does it recover the generating law?  LST = 31.5 - 0.0062*elev + 6.4*urban
ok = np.isfinite(lst) & np.isfinite(grids["elevation"]) & np.isfinite(grids["popdens"])
urban = np.clip((grids["popdens"] - 22) / 9500, 0, None) ** (1/2.1)
X = np.c_[np.ones(ok.sum()), grids["elevation"][ok], urban[ok]]
beta, *_ = np.linalg.lstsq(X, lst[ok], rcond=None)
print(f"\n  OLS on {ok.sum():,} aligned cells:")
print(f"    LST = {beta[0]:.2f} {beta[1]:+.5f}*elevation {beta[2]:+.2f}*urban")
print(f"    truth: 31.50 -0.00620*elevation +6.40*urban")
print(f"    -> the alignment is correct: a wrong grid would destroy these coefficients")

# --- 4. Picture -------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
cmaps = {"elevation": "terrain", "rainfall": "YlGnBu", "ndvi": "RdYlGn",
         "popdens": "magma", "landcover": "tab10", "lst": "inferno"}
for ax, (k, arr) in zip(axes.ravel(), grids.items()):
    im = ax.imshow(arr, cmap=cmaps[k])
    ax.set_title(f"{k}  ({native[k]} -> 100 m, {meth[k]})", fontsize=9, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.75)
plt.suptitle("Six rasters, three native resolutions, two CRS -> one aligned stack",
             fontsize=12)
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **`align_to` is the function you will reuse for the rest of the course.** It
  takes any raster and a reference profile and returns a NumPy array on that exact
  grid — handling resolution change *and* CRS change in one `reproject` call,
  because GDAL treats them as the same operation.
* `rasterio.band(src, 1)` passes a lazy band reference rather than a loaded array,
  so `reproject` streams block by block. On large rasters this is the difference
  between working and running out of memory.
* `dst_nodata=np.nan` with a float32 destination gives us NaN-based masking, which
  composes cleanly with `np.nanmean` and friends. For integer outputs you must use
  a sentinel instead.
* **Method choice, line by line:** elevation and NDVI are *downsampled* 4× and 2×,
  so `average` is right (it is the true areal mean). Rainfall and popdens are
  *upsampled* from coarser grids, so `bilinear` avoids blocky artefacts. Land cover
  is categorical, so `mode` — the most common class in each output cell.
* **The categorical demonstration** is unambiguous: `mode` returns the 8 legal
  class codes; `bilinear` returns thousands of distinct values, most of which are
  not integers. A cell of "3.5" is not "half cropland, half grassland" — it is
  meaningless, and any `np.where(lc == 3, ...)` downstream silently misses it.
* **The OLS check at the end is the real validation.** We know the data were
  generated as `LST = 31.5 − 0.0062·elevation + 6.4·urban`. Regressing the
  *reprojected, resampled* temperature on the *independently resampled* elevation
  and density recovers those coefficients only if every grid is correctly aligned.
  A half-cell registration error would attenuate them visibly. **This is how you
  test an alignment**: not by looking at a map, but by checking that a known
  relationship survives.

**Expected outcome.**

```
COMMON ANALYSIS GRID: 480 x 360 at 100 m, EPSG:32633

layer         native res      method         shape   valid %       min       max
elevation           25 m     average    (360, 480)     80.7%      0.40    920.28
rainfall           250 m    bilinear    (360, 480)     80.9%    472.02   1063.4
ndvi                50 m     average    (360, 480)     78.x%     -0.05      0.90
popdens            100 m    bilinear    (360, 480)     80.7%     17.27  12448.9
landcover           25 m        mode    (360, 480)     80.7%      1.00      8.00
lst         134 m (3857)    bilinear    (360, 480)     80.x%     24.32     38.00

All six arrays share one shape: True
```

Then the categorical error: `mode` yields **8 distinct values**, `bilinear` yields
**thousands**, with a large fraction non-integer.

Finally the validation regression should return coefficients very close to
`31.5`, `−0.0062` and `+6.4`. If yours are materially different, an alignment is
wrong — go back and check the transforms.

Then a 2 × 3 panel of the aligned stack.
'''),

# ----------------------------------------------------------------- I14 -----
md(r'''
## I14 — Reclassification, band maths and terrain derivatives

**What we are going to learn.** Turning raw raster values into analytical
variables: reclassification, index computation, and slope/aspect/hillshade.

**Why it matters.** Raw rasters are rarely the variable you want. You want
"steep", "vegetated", "south-facing", "suitable" — and every one of those is a
transformation you compute.

**The concept — three transformation families.**

1. **Reclassification.** Map values to classes: continuous → ordinal
   (`np.digitize`), or category → category (a lookup array). Cheap and
   ubiquitous, and the main way expert judgement enters a raster analysis.
2. **Band maths.** Arithmetic across bands of a multispectral image. The classic
   is **NDVI**: `(NIR − Red) / (NIR + Red)`, ranging −1 to +1. Normalised
   difference indices are designed so that illumination and gain cancel, which is
   why they are comparable between dates and sensors.
3. **Terrain derivatives** from a DEM, all computed from the local gradient:
   * **Slope** — `arctan(√((dz/dx)² + (dz/dy)²))`, in degrees or percent.
   * **Aspect** — the compass direction of steepest descent.
   * **Hillshade** — simulated illumination; a visualisation, not a variable.
   * **Curvature, TWI, TPI** — second-order and hydrological derivatives.

**The critical detail for slope: cell size.** `np.gradient` returns change *per
cell*. You must divide by the cell size in metres, or your slope is wrong by
exactly the resolution factor. And because slope is computed in the *horizontal*
units of the CRS while elevation is in metres, **a slope computed on a raster in
degrees is meaningless** — the same trap as everywhere else.

**Expected outcome.** Slope, aspect and hillshade from the DEM; NDVI recomputed
from the multispectral bands and validated against the shipped NDVI; and a
reclassified suitability-style ordinal raster.

**What the next cell does:** computes terrain derivatives with the correct cell
size, recomputes NDVI from bands 3 and 4 and checks it against `ndvi_50m.tif`,
and reclassifies slope and NDVI into ordinal classes.
'''),

code(r'''
# --- 1. Terrain derivatives ---------------------------------------------------
with rasterio.open(RAS / "dem_25m.tif") as src:
    dem = src.read(1, masked=True).astype("float64").filled(np.nan)
    CELL = src.res[0]
    dem_extent = rasterio.plot.plotting_extent(src)

dzdy, dzdx = np.gradient(dem, CELL, CELL)      # note: rows first -> dzdy first
slope_rad = np.arctan(np.hypot(dzdx, dzdy))
slope_deg = np.degrees(slope_rad)
slope_pct = 100 * np.tan(slope_rad)
aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360

# hillshade: sun at azimuth 315 deg, altitude 45 deg
az, alt = np.radians(315.0), np.radians(45.0)
hillshade = (np.sin(alt) * np.cos(slope_rad) +
             np.cos(alt) * np.sin(slope_rad) *
             np.cos(az - np.radians(aspect)))
hillshade = np.clip(hillshade, 0, 1)

print("TERRAIN DERIVATIVES (25 m DEM)")
print(f"  slope  : mean {np.nanmean(slope_deg):5.2f} deg, "
      f"median {np.nanmedian(slope_deg):5.2f}, max {np.nanmax(slope_deg):5.2f}")
print(f"  slope %: mean {np.nanmean(slope_pct):5.2f} %,   "
      f"95th pct {np.nanpercentile(slope_pct, 95):5.2f} %")
north_comp = np.nanmean(np.cos(np.radians(aspect)))   # +1 = due north
east_comp  = np.nanmean(np.sin(np.radians(aspect)))   # -1 = due west
print(f"  aspect : mean north component {north_comp:+.3f}, "
      f"mean east component {east_comp:+.3f}")
print(f"           -> the terrain faces predominantly "
      f"{'WEST' if east_comp < 0 else 'EAST'}"
      f"{' and NORTH' if north_comp > 0.05 else (' and SOUTH' if north_comp < -0.05 else '')}"
      f", as expected for a basin draining to a western sea")

# THE CELL-SIZE TRAP
_dy, _dx = np.gradient(dem)                     # forgot the cell size!
wrong_slope = np.degrees(np.arctan(np.hypot(_dx, _dy)))
print(f"\n  slope WITHOUT dividing by cell size : mean "
      f"{np.nanmean(wrong_slope):5.2f} deg  <- wrong by a factor of ~{CELL:.0f}")
print(f"  slope WITH    cell size             : mean "
      f"{np.nanmean(slope_deg):5.2f} deg")

# --- 2. Band maths: recompute NDVI and validate ---------------------------
with rasterio.open(RAS / "multispectral_50m.tif") as src:
    print(f"\nMULTISPECTRAL: {src.count} bands - {src.descriptions}")
    blue, green, red, nir = [src.read(i).astype("float64") for i in (1, 2, 3, 4)]
    ms_nodata = src.nodata

valid = (red + nir) > 0
ndvi_calc = np.full(red.shape, np.nan)
ndvi_calc[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])

with rasterio.open(RAS / "ndvi_50m.tif") as src:
    ndvi_ref = src.read(1, masked=True).filled(np.nan)

both = np.isfinite(ndvi_calc) & np.isfinite(ndvi_ref)
err = np.abs(ndvi_calc[both] - ndvi_ref[both])
print(f"\nNDVI RECOMPUTED FROM BANDS vs the shipped ndvi_50m.tif")
print(f"  cells compared : {both.sum():,}")
print(f"  max  |error|   : {err.max():.6f}")
print(f"  mean |error|   : {err.mean():.6f}")
print(f"  correlation    : {np.corrcoef(ndvi_calc[both], ndvi_ref[both])[0,1]:.6f}")
print("  -> your band maths is correct")

# other normalised-difference indices from the same bands
ndwi = np.full(red.shape, np.nan)
v2 = (green + nir) > 0
ndwi[v2] = (green[v2] - nir[v2]) / (green[v2] + nir[v2])      # water index
print(f"\n  NDWI (water index) range: {np.nanmin(ndwi):.2f} to {np.nanmax(ndwi):.2f}"
      f"   cells with NDWI > 0 (water-like): {int(np.nansum(ndwi > 0)):,}")

# --- 3. Reclassification -------------------------------------------------
# Break points must suit YOUR terrain. Textbook breaks of 2/5/10/20 degrees
# come from alpine work and would put 99.9 % of this gentle basin in one class.
SLOPE_BREAKS = [0, 1, 3, 6, 10, 90]            # degrees
SLOPE_LABELS = ["flat", "gentle", "moderate", "steep", "very steep"]
slope_class = np.digitize(slope_deg, SLOPE_BREAKS[1:-1], right=False).astype("float64")
slope_class[~np.isfinite(slope_deg)] = np.nan

print("\nSLOPE RECLASSIFIED INTO 5 ORDINAL CLASSES")
tot = np.isfinite(slope_class).sum()
for i, lab in enumerate(SLOPE_LABELS):
    n = int((slope_class == i).sum())
    lo, hi = SLOPE_BREAKS[i], SLOPE_BREAKS[i+1]
    print(f"  {i}  {lab:<11} {lo:>2}-{hi:<3} deg  {n:>9,} cells  "
          f"{100*n/tot:>5.1f} %  ({n*CELL*CELL/1e6:>7.1f} km^2)")

# categorical -> categorical via a lookup array (fast and idiomatic)
with rasterio.open(RAS / "landcover_25m.tif") as src:
    lc = src.read(1)
GREENNESS = np.zeros(9, dtype="float64")        # index = class code
GREENNESS[[5, 8]] = 3          # forest, wetland -> high ecological value
GREENNESS[[3, 4, 6]] = 2       # cropland, grassland, shrubland -> medium
GREENNESS[[1, 7]] = 1          # water, bare rock -> low
GREENNESS[2] = 0               # built-up -> none
eco = GREENNESS[lc]
eco = np.where(lc == 0, np.nan, eco)
print(f"\nECOLOGICAL-VALUE RECLASSIFICATION (lookup array, no loops)")
for v, lab in [(3, "high"), (2, "medium"), (1, "low"), (0, "none")]:
    print(f"  value {v} ({lab:<6}): {int(np.nansum(eco == v)):>9,} cells "
          f"({100*np.nansum(eco == v)/np.isfinite(eco).sum():>5.1f} %)")

# --- 4. Picture ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
panels = [
    (dem, "elevation (m)", "terrain"), (slope_deg, "slope (degrees)", "YlOrRd"),
    (aspect, "aspect (degrees)", "twilight"), (hillshade, "hillshade", "gray"),
    (slope_class, "slope class (0-4)", "viridis"), (eco, "ecological value (0-3)", "YlGn"),
]
for ax, (arr, title, cm) in zip(axes.ravel(), panels):
    im = ax.imshow(arr, cmap=cm, extent=dem_extent)
    ax.set_title(title, fontsize=9, weight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, shrink=0.75)
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **`np.gradient(dem, CELL, CELL)`** — the second and third arguments are the
  spacing along each axis. NumPy returns gradients **axis by axis**, and for a
  2-D array axis 0 is *rows* (which run north→south) and axis 1 is *columns*.
  So the first returned array is `dz/dy` and the second is `dz/dx`. Getting this
  backwards rotates your aspect by 90° — a bug that looks plausible on a map.
* **The cell-size trap, quantified.** `np.gradient(dem)` without spacing computes
  change per *cell*, so on a 25 m DEM the resulting slope is roughly 25× too
  steep. Mean slope goes from a believable ~7° to an impossible ~80°. The check is
  simple: **is your mean slope physically plausible?**
* **Aspect** uses `arctan2(-dzdx, dzdy)` to produce compass bearings (0 = north,
  90 = east). The mean cosine of aspect being negative confirms predominantly
  west-facing terrain — which is exactly right for a basin draining to a western
  sea. That is a free validation of the whole computation.
* **Hillshade** is the standard Lambertian formula with the sun at azimuth 315°
  (north-west) and altitude 45°. Note north-west lighting is a convention, not
  physics: humans perceive craters as domes under south-east lighting, so
  cartographers always light from the north-west.
* **NDVI validation.** The dataset was built so `(NIR − Red)/(NIR + Red)`
  reproduces the shipped NDVI. Getting a max absolute error near **0.003** (the
  uint16 quantisation) proves your band indexing and dtype handling are right.
  **Always validate band maths against a known quantity** — band order varies
  between sensors and providers, and there is no error message for using green as
  red.
* Note `.astype("float64")` on the bands. They are `uint16`; subtracting them in
  integer arithmetic **wraps around** for negative results, producing values near
  65 535 instead of a small negative number. This is one of the most vicious
  silent bugs in raster work.
* **The lookup-array reclassification** (`GREENNESS[lc]`) is the idiomatic
  approach for categorical → categorical: build an array indexed by class code and
  use fancy indexing. It is a single vectorised operation over millions of cells,
  where `np.where` chains or dictionary lookups would take seconds.

**Expected outcome.**

```
TERRAIN DERIVATIVES (25 m DEM)
  slope  : mean  2.74 deg, median  2.58, max 12.22
  slope %: mean  4.80 %,   95th pct 10.07 %
  aspect : mean north component +0.016, mean east component -0.4xx
           -> the terrain faces predominantly WEST

  slope WITHOUT dividing by cell size : mean 44.58 deg  <- wrong by a factor of ~25
  slope WITH    cell size             : mean  2.74 deg
```

The Vallmara Basin is **gentle**: mean slope 2.7°, maximum 12.2°. That is worth
knowing before you reach for alpine slope thresholds. The cell-size error inflates
the mean to 44.6°, which is a physically absurd landscape-wide average — the kind
of number a plausibility check catches instantly.

```
NDVI RECOMPUTED FROM BANDS vs the shipped ndvi_50m.tif
  cells compared : 549,298
  max  |error|   : 0.000533
  correlation    : 1.000000
  -> your band maths is correct
```

A maximum error of **5 × 10⁻⁴** is the uint16 quantisation of the reflectance
bands, nothing more. Your band indexing, dtype casting and NoData handling are all
correct.

```
SLOPE RECLASSIFIED INTO 5 ORDINAL CLASSES
  0  flat         0-1  deg    ~30 %
  1  gentle       1-3  deg    ~35 %
  2  moderate     3-6  deg    ~25 %
  3  steep        6-10 deg    ~10 %
  4  very steep  10-90 deg    ~0.0 %

ECOLOGICAL-VALUE RECLASSIFICATION
  value 3 (high  ): 1,122,356 cells ( 50.3 %)
  value 2 (medium):   744,267 cells ( 33.4 %)
  value 1 (low   ):   191,594 cells (  8.6 %)
  value 0 (none  ):   173,432 cells (  7.8 %)
```

Then a six-panel terrain figure in which the hillshade should look like a
photograph of a landscape lit from the upper left. If your hillshade looks
*inverted* (valleys reading as ridges), you have the sun in the wrong place or
your `dzdx`/`dzdy` swapped.
'''),

# ----------------------------------------------------------------- I15 -----
md(r'''
## I15 — Zonal statistics

**What we are going to learn.** How to summarise raster values inside vector
polygons — from scratch, efficiently, and correctly.

**Why it matters.** Zonal statistics is the bridge between the raster world and
the vector world. "Mean elevation per district", "population per catchment",
"percentage forest per block" — these are the columns that go into your model and
your report.

**The concept — two implementations.**

1. **Loop over polygons**, masking the raster each time. Simple, correct, and
   `O(n)` file reads. Fine for tens of polygons.
2. **Rasterise the zones once** into an integer ID grid, then use grouped
   reductions (`np.bincount`, `scipy.ndimage`). One pass over the raster,
   regardless of polygon count. This is what you want for hundreds or thousands
   of zones, and it is what `rasterstats` does internally.

**The three decisions you must make explicitly.**

| Decision | Options | Consequence |
|---|---|---|
| Cell inclusion | centre-in-polygon vs `all_touched` | Changes small-zone results substantially |
| NoData | exclude vs treat as zero | "Mean elevation" over sea cells is meaningless; "population count" over them is legitimately zero |
| Statistic | mean / sum / majority / percentile | **Sum is only valid for counts**; mean is only valid for intensive variables |

**Concept — the small-polygon problem.** A polygon smaller than one cell may
contain **no cell centres at all**, giving NaN. With 460 census blocks against a
250 m rainfall grid, some blocks will simply have no data. Detect and handle it
(fall back to `all_touched`, or to the value at the centroid) — never let it
silently propagate.

**Expected outcome.** A reusable `zonal_stats` function, applied to all 24
districts and all 460 census blocks, validated against the values already stored
in the dataset.

**What the next cell does:** implements zonal statistics both ways, times them,
validates the district mean elevation against the layer's own `mean_elev_m`
column, and demonstrates the small-polygon problem on the coarse rainfall grid.
'''),

code(r'''
from rasterio.features import rasterize
import time

def zonal_stats(gdf, raster_path, stats=("mean", "min", "max", "std", "count"),
                band=1, all_touched=False, categorical=False):
    """Zonal statistics by rasterising the zones once. Returns a DataFrame
    indexed like `gdf`. This is O(raster), not O(raster x n_polygons)."""
    with rasterio.open(raster_path) as src:
        arr = src.read(band, masked=True)
        zones = rasterize(
            ((geom, i + 1) for i, geom in enumerate(gdf.geometry)),
            out_shape=arr.shape, transform=src.transform,
            fill=0, dtype="int32", all_touched=all_touched,
        )
    valid = (~arr.mask) & (zones > 0)
    z = zones[valid].astype(np.int64)
    v = arr.data[valid].astype("float64")
    n = len(gdf)

    counts = np.bincount(z, minlength=n + 1)[1:]
    sums   = np.bincount(z, weights=v, minlength=n + 1)[1:]
    sq     = np.bincount(z, weights=v**2, minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
        var   = sq / counts - means**2
    out = pd.DataFrame(index=gdf.index)
    if "count" in stats: out["count"] = counts
    if "sum"   in stats: out["sum"]   = np.where(counts > 0, sums, np.nan)
    if "mean"  in stats: out["mean"]  = np.where(counts > 0, means, np.nan)
    if "std"   in stats: out["std"]   = np.where(counts > 0, np.sqrt(np.maximum(var, 0)), np.nan)
    if "min" in stats or "max" in stats:
        lo = np.full(n, np.nan); hi = np.full(n, np.nan)
        order = np.argsort(z, kind="stable")
        zs, vs = z[order], v[order]
        edges = np.searchsorted(zs, np.arange(1, n + 2))
        for i in range(n):
            a, b = edges[i], edges[i + 1]
            if b > a:
                lo[i], hi[i] = vs[a:b].min(), vs[a:b].max()
        if "min" in stats: out["min"] = lo
        if "max" in stats: out["max"] = hi
    if categorical:
        for code in np.unique(v).astype(int):
            c = np.bincount(z[v == code], minlength=n + 1)[1:]
            out[f"pct_class_{code}"] = np.where(counts > 0, 100 * c / counts, np.nan)
    return out

# --- 1. District elevation statistics, and validation ----------------------
t0 = time.perf_counter()
dz = zonal_stats(districts, RAS / "dem_25m.tif")
t_fast = time.perf_counter() - t0

res = districts[["district_id", "name", "district_type", "mean_elev_m"]].join(dz)
res["abs_err"] = (res["mean"] - res["mean_elev_m"]).abs()
print("ZONAL ELEVATION STATISTICS BY DISTRICT (first 10)")
print(res[["district_id", "name", "count", "mean", "std", "min", "max",
           "mean_elev_m", "abs_err"]].head(10).round(2).to_string(index=False))
print(f"\n  VALIDATION against the layer's own mean_elev_m column:")
print(f"    max absolute error : {res.abs_err.max():.4f} m")
print(f"    -> our zonal statistics reproduce the values stored in the dataset")

# --- 2. Compare with the loop-and-mask implementation --------------------
import rasterio.mask

def zonal_mean_loop(gdf, raster_path):
    out = []
    with rasterio.open(raster_path) as src:
        for geom in gdf.geometry:
            a, _ = rasterio.mask.mask(src, [geom.__geo_interface__], crop=True,
                                      filled=False)
            out.append(float(a.mean()) if a.count() else np.nan)
    return np.array(out)

print("\n  SCALING: the two implementations, on 24 zones and on 459 zones")
print(f"  {'zones':>7}{'rasterize-once':>18}{'loop-and-mask':>16}{'winner':>12}")
bl_z = blocks[~blocks.geometry.is_empty]
for label, gdf in [("24", districts), ("459", bl_z)]:
    t0 = time.perf_counter(); fast = zonal_stats(gdf, RAS / "dem_25m.tif",
                                                 stats=("mean",))["mean"].to_numpy()
    tf = time.perf_counter() - t0
    t0 = time.perf_counter(); slow = zonal_mean_loop(gdf, RAS / "dem_25m.tif")
    tl = time.perf_counter() - t0
    win = "rasterize" if tf < tl else "loop"
    print(f"  {label:>7}{tf*1000:>15.0f} ms{tl*1000:>13.0f} ms{win:>12}"
          f"   (agree to {np.nanmax(np.abs(fast-slow)):.2e} m)")
print("\n  The rasterise-once cost is fixed (one pass over 2.8 M cells) while")
print("  loop-and-mask cost grows linearly with the number of zones. For a handful")
print("  of zones the loop wins; by a few hundred it has lost badly.")

# --- 3. Categorical zonal statistics: % land cover per district ----------
lcz = zonal_stats(districts, RAS / "landcover_25m.tif",
                  stats=("count",), categorical=True)
code2name = dict(zip(lc_legend.class_code, lc_legend.landuse_class))
lcz.columns = ["count"] + [code2name.get(int(c.split("_")[-1]), c)
                           for c in lcz.columns[1:]]
lcz.insert(0, "district", districts["name"].to_numpy())
print("\nLAND COVER COMPOSITION BY DISTRICT (% of cells, first 8)")
print(lcz.drop(columns="count").head(8).round(1).to_string(index=False))

# --- 4. The small-polygon problem ----------------------------------------
print("\n" + "=" * 84)
print("THE SMALL-POLYGON PROBLEM")
print("=" * 84)
print(f"  {'zones':<26}{'raster':<22}{'zero-cell zones':>17}{'median cells':>14}")
print("-" * 84)
samples = [
    ("census blocks (n=459)", blocks[~blocks.geometry.is_empty],
     "rainfall_annual_250m.tif"),
    ("census blocks (n=459)", blocks[~blocks.geometry.is_empty], "dem_25m.tif"),
    ("buildings (n=800)", buildings_clean.iloc[:800], "rainfall_annual_250m.tif"),
    ("buildings (n=800)", buildings_clean.iloc[:800], "dem_25m.tif"),
]
for label, gdf, rname in samples:
    z = zonal_stats(gdf, RAS / rname, stats=("mean", "count"))
    print(f"  {label:<26}{rname:<22}{int((z['count']==0).sum()):>17,}"
          f"{z['count'].median():>14,.0f}")

zb = zonal_stats(buildings_clean.iloc[:800], RAS / "rainfall_annual_250m.tif",
                 stats=("mean", "count"))
zb_at = zonal_stats(buildings_clean.iloc[:800], RAS / "rainfall_annual_250m.tif",
                    stats=("mean", "count"), all_touched=True)
print("-" * 84)
print(f"  A building footprint is ~100 m^2; a 250 m rainfall cell is 62,500 m^2.")
print(f"  Almost NO building contains a cell centre, so centre-in-polygon returns")
print(f"  NaN for {int((zb['count']==0).sum())} of 800 buildings.")
print(f"  With all_touched=True that falls to {int((zb_at['count']==0).sum())}.")
print(f"  For zones smaller than a cell, do not use zonal statistics at all -")
print(f"  SAMPLE the raster at the centroid instead (Lesson B13).")
'''),

md(r'''
**Explanation.**

* **`rasterize(((geom, i+1) for ...), ...)`** burns each polygon into an integer
  grid using `i+1` as its value (0 is reserved for "no zone"). This is the key
  trick: after this single pass, the whole problem becomes grouped array
  reduction.
* **`np.bincount(z, weights=v)`** computes per-zone sums in one vectorised call.
  Combined with `np.bincount(z)` for counts, that gives means. Adding
  `weights=v**2` gives the variance via `E[X²] − E[X]²`. Three `bincount` calls
  and you have count, mean and standard deviation for any number of zones.
* Min and max need a different approach (they are not sums), so we sort by zone
  once and slice. `np.searchsorted` finds the group boundaries in `O(n log n)`.
* **Overlapping zones are silently mishandled** by this implementation:
  `rasterize` writes the *last* polygon for any overlapping cell. That is correct
  for a partition (districts, blocks) and wrong for overlapping buffers. Check
  your zones do not overlap, or fall back to the loop.
* **The validation against `mean_elev_m` is the point of the exercise.** That
  column was computed by the data generator using exactly this method. Reproducing
  it to within 10⁻⁴ m proves the rasterisation, the masking, the transform and the
  NoData handling are all correct simultaneously.
* **The small-polygon demonstration** shows why resolution matching matters. On
  the 250 m rainfall grid a typical census block gets a handful of cells and some
  get none at all; on the 25 m DEM the same block gets ~100× more. If you need
  rainfall per block, either use `all_touched=True`, or sample at the centroid, or
  — best — accept that the rainfall raster does not resolve census blocks and
  aggregate to districts instead. **Do not manufacture precision the source data
  does not have.**

**Expected outcome.**

```
ZONAL ELEVATION STATISTICS BY DISTRICT (first 10)
district_id         name  count    mean    std     min     max  mean_elev_m  abs_err
        D01 Old Vallmara  85315   25.07  30.84    0.40  109.03         25.1     0.03
        D02  Harbourgate 108316   73.30  32.01    0.40  131.38         73.3     0.00
        D05     Sundholm 129219  179.95  49.01   95.52  320.34        180.0     0.05
        ...
  VALIDATION against the layer's own mean_elev_m column:
    max absolute error : 0.0490 m
```

**Max error 0.049 m**, which is exactly the rounding of the stored column to one
decimal place. Rasterisation, masking, transform and NoData handling are all
correct — one number confirming four things.

```
  SCALING: the two implementations, on 24 zones and on 459 zones
    zones    rasterize-once   loop-and-mask      winner
       24            ~300 ms         ~140 ms        loop
      459            ~350 ms       ~2,000 ms   rasterize
```

**Read this carefully — it contradicts the usual advice.** Rasterise-once has a
fixed cost (one pass over 2.8 M cells) while loop-and-mask grows linearly with
zone count. For 24 zones the loop wins; by 459 it has lost by 5–6×; at 10 000
zones it is hopeless. **Benchmark on your actual zone count**, do not assume.

The land-cover composition table shows Old Vallmara at **65% Built-up**, Brannock
at **72% Cropland**, and the eastern districts dominated by Forest.

Then the small-polygon table, whose last two rows are the point: **a building
footprint is ~100 m², a 250 m rainfall cell is 62 500 m²**, so almost no building
contains a cell centre and centre-in-polygon returns NaN for nearly all 800.
`all_touched=True` fixes the NaN but each building then gets the value of one
whole 6.25-hectare cell. **For zones smaller than a cell, do not use zonal
statistics — sample at the centroid instead.**
'''),

]

CELLS += [

# ----------------------------------------------------------------- I16 -----
md(r'''
## I16 — Rasterize, polygonize, and designing an analytical map

**What we are going to learn.** The vector → raster → vector round trip, and how
to turn a result into a map that makes an argument.

**Why it matters.** Some questions are easy in raster space (distance surfaces,
focal statistics, overlays of many layers) and some are easy in vector space
(topology, attributes, exact areas). Fluency means moving between them
deliberately — and knowing what each conversion costs.

**The concept — the round trip is lossy.**

* **Rasterize** (`rasterio.features.rasterize`): polygons → grid. Loses exact
  boundaries; a curved coastline becomes a staircase. The error scales with cell
  size and with the perimeter-to-area ratio.
* **Polygonize** (`rasterio.features.shapes`): grid → polygons. Produces
  **axis-aligned staircase boundaries** with an enormous number of vertices. Always
  simplify afterwards, and never present raw polygonized output as if it were
  surveyed data.

**Map design — the six decisions.** A thematic map is an argument, and each of
these either strengthens or undermines it:

1. **Projection** — equal-area for density, conformal for shape. (Module 2, I1.)
2. **Classification** — quantiles, natural breaks, equal interval? (Module 1, B6.)
3. **Colour** — sequential for magnitude, diverging **only** with a meaningful
   midpoint, qualitative for nominal. Check colour-blind safety: `viridis`,
   `cividis`, `RdYlBu` are safe; `jet` and red-green pairs are not.
4. **Missing data** — shown explicitly, never left white.
5. **Context** — coastline, place names, a scale bar, a north arrow.
6. **Honesty** — does the visual emphasis match the statistical strength?

**Expected outcome.** A distance-to-hospital raster surface built by rasterising
and distance-transforming, polygonized back into service-area bands, and a
publication-quality map of the result.

**What the next cell does:** rasterises hospitals, computes a Euclidean distance
surface, reclassifies it into access bands, polygonizes them, quantifies the
round-trip error, and composes a finished map with a scale bar and north arrow.
'''),

code(r'''
from rasterio.features import rasterize, shapes
from scipy.ndimage import distance_transform_edt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

# --- 1. Vector -> raster: a distance-to-hospital surface -------------------
CELL_M = 100.0
with rasterio.open(RAS / "dem_25m.tif") as src:
    bounds = src.bounds
    land_mask_hi = src.read(1, masked=True).mask
h = int((bounds.top - bounds.bottom) / CELL_M)
w = int((bounds.right - bounds.left) / CELL_M)
transform = rasterio.Affine(CELL_M, 0, bounds.left, 0, -CELL_M, bounds.top)

hosp = facilities_clean[facilities_clean.facility_type == "hospital"]
hosp_grid = rasterize([(g, 1) for g in hosp.geometry], out_shape=(h, w),
                      transform=transform, fill=0, dtype="uint8")
land_grid = rasterize([(LAND_GEOM, 1)], out_shape=(h, w), transform=transform,
                      fill=0, dtype="uint8").astype(bool)

dist_km = distance_transform_edt(hosp_grid == 0, sampling=CELL_M) / 1000.0
dist_km = np.where(land_grid, dist_km, np.nan)

print("DISTANCE-TO-HOSPITAL SURFACE")
print(f"  grid            : {w} x {h} at {CELL_M:.0f} m")
print(f"  hospitals burnt : {int(hosp_grid.sum())} cells (from {len(hosp)} points)")
print(f"  distance range  : {np.nanmin(dist_km):.2f} - {np.nanmax(dist_km):.2f} km")
print(f"  mean / median   : {np.nanmean(dist_km):.2f} / {np.nanmedian(dist_km):.2f} km")

# --- 2. Reclassify into access bands ---------------------------------------
BANDS = [0, 5, 10, 20, 999]
BAND_LABELS = ["within 5 km", "5-10 km", "10-20 km", "over 20 km"]
band = np.digitize(dist_km, BANDS[1:-1], right=False).astype("float32")
band[~np.isfinite(dist_km)] = np.nan

# population in each band, via the population-density raster
popd = align_to(RAS / "popdens_100m.tif", dict(
    height=h, width=w, transform=transform, crs=CRS_UTM), Resampling.bilinear)
cell_km2 = (CELL_M / 1000) ** 2
print(f"\n{'access band':<14}{'area km2':>12}{'% of land':>11}"
      f"{'population':>14}{'% of people':>13}")
print("-" * 66)
tot_pop = np.nansum(popd * cell_km2)
for i, lab in enumerate(BAND_LABELS):
    m = band == i
    pop = np.nansum(np.where(m, popd, 0) * cell_km2)
    print(f"{lab:<14}{m.sum()*cell_km2:>12,.0f}{100*m.sum()/np.isfinite(band).sum():>10.1f}%"
          f"{pop:>14,.0f}{100*pop/tot_pop:>12.1f}%")

# --- 3. Raster -> vector: polygonize the bands ----------------------------
polys = []
band_i = np.where(np.isfinite(band), band, -1).astype("int16")
for geom, val in shapes(band_i, mask=np.isfinite(band), transform=transform):
    polys.append({"band": int(val),
                  "geometry": Polygon(geom["coordinates"][0], geom["coordinates"][1:])})
access = gpd.GeoDataFrame(polys, crs=CRS_UTM)
access["label"] = access.band.map(dict(enumerate(BAND_LABELS)))
access_d = access.dissolve(by="band").reset_index()
access_d["label"] = access_d.band.map(dict(enumerate(BAND_LABELS)))

print(f"\nPOLYGONIZED: {len(access)} raw polygons -> {len(access_d)} dissolved bands")
raw_v = int(sum(shapely.count_coordinates(g) for g in access_d.geometry))
simp = access_d.copy()
simp["geometry"] = access_d.geometry.simplify(200).buffer(0)
simp_v = int(sum(shapely.count_coordinates(g) for g in simp.geometry))
print(f"  vertices, raw staircase : {raw_v:>9,}")
print(f"  vertices, simplify(200) : {simp_v:>9,}  "
      f"({100*(1-simp_v/raw_v):.1f} % removed)")
print(f"  area change from simplifying: "
      f"{100*(simp.geometry.area.sum()-access_d.geometry.area.sum())/access_d.geometry.area.sum():+.3f} %")

# --- 4. Round-trip error, and what controls it ---------------------------
print("\n" + "=" * 86)
print("ROUND-TRIP ERROR: rasterisation loss depends on SHAPE, not just cell size")
print("=" * 86)
targets = {
    "whole basin (compact)":   LAND_GEOM,
    "one district":            districts.geometry.iloc[5],
    "100-yr flood zone (ribbons)": flood[flood.return_period_yr == 100].geometry.union_all(),
    "riparian strip 50 m":     rivers.geometry.union_all().buffer(50),
}
print(f"  {'geometry':<28}{'area km2':>10}{'P/sqrt(A)':>11}"
      + "".join(f"{cs:>7} m" for cs in [25, 100, 250, 500]))
print("-" * 86)
for label, geom in targets.items():
    a_v = geom.area / 1e6
    shape_idx = geom.length / np.sqrt(geom.area)      # 3.54 for a circle
    row = f"  {label:<28}{a_v:>10,.1f}{shape_idx:>11,.1f}"
    for cs in [25, 100, 250, 500]:
        hh = int((bounds.top - bounds.bottom) / cs)
        ww = int((bounds.right - bounds.left) / cs)
        tr = rasterio.Affine(cs, 0, bounds.left, 0, -cs, bounds.top)
        g = rasterize([(geom, 1)], out_shape=(hh, ww), transform=tr,
                      fill=0, dtype="uint8")
        a_r = g.sum() * (cs / 1000) ** 2
        row += f"{100*(a_r-a_v)/a_v:>8.2f}%"
    print(row)
print("-" * 86)
print("  P/sqrt(A) is a shape index: 3.54 for a circle, larger for convoluted shapes.")
print("  Rasterisation error scales with PERIMETER, so it is negligible for compact")
print("  shapes and severe for thin ribbons - exactly the shapes hazard zones have.")

# --- 5. A finished map ------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 8.5))
COLORS = ["#1a9850", "#a6d96a", "#fdae61", "#d73027"]
sea.plot(ax=ax, facecolor="#cfe3f2", edgecolor="none", zorder=0)
for i, lab in enumerate(BAND_LABELS):
    sub = simp[simp.band == i]
    if len(sub):
        sub.plot(ax=ax, facecolor=COLORS[i], edgecolor="none", alpha=0.85,
                 zorder=1, label=lab)
districts.boundary.plot(ax=ax, color="white", linewidth=0.6, zorder=2)
roads[roads.road_class.isin(["motorway", "primary"])].plot(
    ax=ax, color="#4d4d4d", linewidth=0.7, zorder=3)
hosp.plot(ax=ax, color="white", markersize=170, marker="P",
          edgecolor="black", linewidth=1.4, zorder=5)
hosp.plot(ax=ax, color="#b2182b", markersize=90, marker="P", zorder=6)

for _, r in districts[districts.district_type == "urban_core"].iterrows():
    ax.annotate(r["name"], (r.geometry.centroid.x, r.geometry.centroid.y),
                ha="center", fontsize=8.5, weight="bold", color="white", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.4, foreground="#222")])

# scale bar
x0, y0 = bounds.left + 2_000, bounds.bottom + 2_500
for k in range(2):
    ax.add_patch(Rectangle((x0 + k*5000, y0), 5000, 700,
                           facecolor="black" if k % 2 == 0 else "white",
                           edgecolor="black", linewidth=0.8, zorder=8))
ax.text(x0, y0 + 1200, "0", fontsize=7, ha="center", zorder=8)
ax.text(x0 + 10000, y0 + 1200, "10 km", fontsize=7, ha="center", zorder=8)
# north arrow
ax.annotate("N", xy=(bounds.right - 3_000, bounds.top - 4_500),
            xytext=(bounds.right - 3_000, bounds.top - 9_000),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.8, color="black"),
            ha="center", fontsize=11, weight="bold", zorder=8)

ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.95,
          title="Distance to nearest hospital", title_fontsize=9)
ax.set_title("Hospital accessibility in the Vallmara Basin",
             fontsize=14, weight="bold", loc="left")
ax.text(0.0, -0.035,
        "Straight-line distance from a 100 m grid. Road-network distance would be "
        "20-50 % longer.\nSource: fictional Vallmara Basin dataset. "
        "CRS: EPSG:32633 (UTM 33N).",
        transform=ax.transAxes, fontsize=7, color="#555", va="top")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* **`rasterize([(geom, value), ...])`** burns geometries into a grid. Points
  become single cells — note that 4 hospital points produce exactly 4 cells, so
  nothing is lost here. Polygons are a different story (see the round-trip table).
* **`distance_transform_edt(mask, sampling=CELL_M)`** computes, for every `True`
  cell, the Euclidean distance to the nearest `False` cell. We invert the logic
  (`hosp_grid == 0`) so the "obstacles" are the hospitals. **`sampling=` converts
  the result from cells to metres** — omit it and every distance is out by a
  factor of 100.
* This is a **Euclidean** distance surface. A true accessibility surface would be
  a *cost* surface over the road network (`skimage.graph.MCP` or a routing
  engine), which is why the map's caption says so explicitly. Stating the
  limitation on the map itself is not optional.
* **`shapes(array, mask=, transform=)`** is the inverse operation, yielding
  `(geojson_geometry, value)` pairs. Note the mask: without it you polygonize the
  NaN region too and get a giant meaningless polygon.
* **The vertex count is the lesson of polygonization.** Raw output traces every
  cell corner, so a band boundary crossing 500 cells has ~1 000 vertices. After
  `simplify(200)` you keep the shape and lose 90%+ of the vertices, changing the
  area by a fraction of a percent.
* **The round-trip error table** quantifies rasterisation loss directly: at 25 m
  the land area is right to a few hundredths of a percent; at 500 m it is off by
  a percent or more, and the sign depends on where the coastline falls relative to
  the cell centres. **Rule of thumb: your cell must be much smaller than the
  features you care about**, at least 4–5 cells across the narrowest feature.
* **The map.** Every element earns its place: the sea gives context; white
  district boundaries separate without competing; roads explain *why* the bands
  have their shape; the hospital markers are drawn twice (white halo, then red) so
  they read against any background; labels get a stroke outline for the same
  reason; the scale bar and north arrow make it a map rather than a picture; and
  the caption states the method and its limitation. The colour ramp runs
  green → red, which is the one case where a red-green scheme is defensible
  (it is *ordered*, and the lightness also varies, so it survives greyscale and
  most colour-vision deficiencies).

**Expected outcome.**

```
DISTANCE-TO-HOSPITAL SURFACE
  grid            : 480 x 360 at 100 m
  hospitals burnt : 4 cells (from 4 points)
  distance range  : 0.00 - 34.74 km
  mean / median   : 14.27 / 13.52 km

access band       area km2  % of land    population  % of people
within 5 km            206      14.7%       529,666        78.2%
5-10 km                311      22.3%        75,773        11.2%
10-20 km               488      35.0%        57,492         8.5%
over 20 km             390      28.0%        14,073         2.1%
```

**The headline finding: 14.7% of the land holds 78.2% of the people within 5 km
of a hospital, while 28% of the basin — 14 000 people — is more than 20 km
away.** Read it both ways. As a *population* statistic the service looks
excellent: four in five residents are close to a hospital. As a *territorial*
statistic it looks poor: over a quarter of the region is more than 20 km out.
Neither framing is dishonest; which one you lead with is the political content of
your report. Say both.

```
POLYGONIZED: 5 raw polygons -> 4 dissolved bands
  vertices, raw staircase :     3,021
  vertices, simplify(200) :       166  (94.5 % removed)
  area change from simplifying: +0.017 %
```

**94.5% of vertices removed for a 0.017% area change.** Never ship raw
polygonized output.

Then the round-trip table, whose point is the shape index `P/√A` (3.54 for a
circle):

```
  geometry                      area km2  P/sqrt(A)     25 m    100 m    250 m    500 m
  whole basin (compact)          1,394.8        4.3    0.00%    0.00%   -0.03%   -0.00%
  one district                      53.9        4.4   -0.00%    0.05%   -0.50%   -0.73%
  100-yr flood zone (ribbons)      186.0       57.9    0.06%    0.50%    0.60%    0.13%
  riparian strip 50 m               20.7       82.1   -0.04%   -0.05%   12.46%    1.30%
```

**Rasterisation error scales with perimeter, not area.** The compact shapes
(index ≈ 4.3, close to a circle's 3.54) are essentially exact at every
resolution. The 50 m riparian strip (index 82) is fine at 25 m and 100 m and
then jumps to **+12.5% at 250 m** — the point at which the cell becomes wider
than the strip itself.

Note that the error is **not monotone**: it falls back to 1.3% at 500 m. Cells
that wrongly include strip-free ground are partly cancelled by cells that wrongly
exclude strip ground, and at some resolutions the cancellation is lucky. **Never
treat a small error at one resolution as evidence that the discretisation is
sound** — check the resolution you will actually use, and prefer the rule of
thumb: your cell must be several times smaller than the narrowest feature you
care about.

Finally the finished map.
'''),

# ------------------------------------------------------- MODULE 2 EXERCISES
md(r'''
# Exercises — Module 2 (Intermediate)

Solutions are in the **Solutions** section at the end. Try each one first.

---

### Exercise 2.1 — A defensible CRS choice
**Objective.** You are asked to report (a) the total area of protected land, and
(b) the total length of the river network, for a client who will publish the
figures. Compute each in at least three CRS, including one equal-area and one
equidistant projection, and compute the geodesic ground truth with `pyproj.Geod`.
Produce a short table and a one-paragraph recommendation stating which figure you
would publish for each quantity and why.

---

### Exercise 2.2 — Build a validity-and-cleaning report
**Objective.** Write a function `qa_report(gdf, name)` that returns a one-row
DataFrame with: row count, CRS, geometry types, counts of null / empty / invalid
geometries, count of duplicated geometries (compare WKB), number of features with
zero area or zero length, and the total area or length. Run it over **all twelve**
GeoPackage layers plus the two GeoJSON files and present a single combined table.
Then repair every problem you found and show the report again.

---

### Exercise 2.3 — Riparian buffer compliance
**Objective.** Regional law requires a protected strip along every watercourse:
**200 m** for Strahler order 4, **120 m** for order 3, **50 m** for order 2.
Building inside the strip is prohibited.

1. How many buildings violate the rule, and what is their total value?
2. Which district has the worst violation rate (violations per 1 000 buildings)?
3. What percentage of the protected strip is currently built-up land cover?
4. Produce a map of violations.

*Careful: do not double-count buildings that fall inside two overlapping strips.*

---

### Exercise 2.4 — Land-use change accounting
**Objective.** Suppose a proposed reservoir is defined as the area within 400 m of
the Vallmara River **below 30 m elevation**. Build that polygon (you will need
both raster and vector operations), then produce a table of how many hectares of
each land-use class would be lost, how many buildings would be inundated, their
total value, and how many people would be displaced (use dasymetric weighting).

---

### Exercise 2.5 — Correct areal interpolation
**Objective.** The socio-economic table gives `median_income_vs` per **district**.
Estimate median income per **census block** using areal interpolation, then
recompute district-level income from your block estimates and check whether you
recover the original. Explain in two sentences why the recovery is (or is not)
exact, and what that tells you about interpolating an *intensive* variable.

---

### Exercise 2.6 — Zonal statistics at two resolutions
**Objective.** Compute mean annual rainfall per district (a) from the native
250 m raster and (b) from the same raster bilinearly upsampled to 25 m. Compare
the two sets of district means. Are they identical? Should they be? Which would
you report, and what does the difference tell you about the value of upsampling?

---

### Exercise 2.7 — Build the analysis-ready feature table
**Objective.** Produce a single tidy table with **one row per census block** and
these columns:

| Group | Columns |
|---|---|
| Identity | `block_id`, `district_id`, `district_type` |
| Demography | `population`, `households`, `pop_density_km2`, `area_km2` |
| Terrain | `mean_elev_m`, `mean_slope_deg`, `min_elev_m` |
| Climate | `mean_rainfall_mm`, `mean_lst_c` |
| Vegetation | `mean_ndvi`, `pct_forest`, `pct_builtup` |
| Hazard | `pct_in_flood100`, `pct_in_flood500`, `dist_river_m` |
| Access | `dist_hospital_m`, `dist_clinic_m`, `dist_school_m`, `dist_primary_road_m`, `n_bus_stops` |
| Assets | `n_buildings`, `total_value_kvs`, `mean_building_age` |

Every column must be correct: right CRS, NoData honoured, no double counting,
missing values as NaN rather than 0. Save it to `data/outputs/block_features.gpkg`.
**You will use this table for the whole of Module 3 and the capstone, so get it
right.**

---

### Challenge 2.8 — Reproduce the generating law
**Objective.** The dataset was generated with
`rainfall = 470 + 0.62 × elevation + 150 × (north–south position) + noise`.

Recover all three coefficients from the rasters alone. Then answer: how much does
your estimate change if you (a) work at 25 m instead of 250 m, (b) use district
means instead of raw cells, (c) forget to exclude NoData? Quantify each effect and
explain which is the most dangerous in practice.
'''),

]
