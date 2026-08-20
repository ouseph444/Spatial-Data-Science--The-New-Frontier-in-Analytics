# -*- coding: utf-8 -*-
"""Module 1 - Beginner: vector fundamentals, CRS, plotting, first rasters."""
from _cells import md, code

CELLS = [

md(r'''
# Module 1 — Beginner: The Foundations

By the end of this module you will be able to open any vector or raster file,
understand exactly what is inside it, know which coordinate system it is in and
why that matters, make a competent map, and perform your first spatial join.

**The 14 lessons**

| # | Lesson | Core skill |
|---|---|---|
| B1 | Loading vector data | `gpd.read_file`, layers, formats |
| B2 | The attribute table | dtypes, `describe`, `value_counts`, missingness |
| B3 | The geometry column | Shapely objects, `geom_type`, coordinates, WKT |
| B4 | CRS I — what a coordinate system *is* | geographic vs projected, `.crs` |
| B5 | CRS II — reprojection and the degrees trap | `to_crs`, measured error |
| B6 | Plotting spatial data | choropleths, layered maps, classification |
| B7 | Points from a CSV | `points_from_xy`, `set_crs` vs `to_crs` |
| B8 | Shapely operations | buffer, centroid, area, length, predicates |
| B9 | Filtering and selecting features | attribute, bbox (`.cx`), spatial |
| B10 | Calculating distances | point–point, point–line, matrices |
| B11 | Basic spatial joins | `sjoin`, predicates, counting points in polygons |
| B12 | Opening a raster | profile, transform, NoData, masked arrays |
| B13 | Raster ↔ array ↔ ground | indexing, sampling at points, correct plotting |
| B14 | Writing data out | GeoPackage, GeoJSON, GeoTIFF, CSV |
'''),

# ------------------------------------------------------------------ B1 -----
md(r'''
## B1 — Loading vector data

**What we are going to learn.** How to discover what is inside a spatial file
and load it into Python.

**Why it matters.** Unlike a CSV, a spatial file can hold *several* datasets
("layers"), each with its own schema, geometry type and coordinate system.
Opening one blindly and getting the wrong layer — or silently getting only the
first of twelve — is a very common first mistake.

**The concept — vector data.** Vector data represents the world as discrete
objects with exact coordinates:

* **Point** — a single (x, y): a school, a sensor, a flood report.
* **LineString** — an ordered sequence of points: a road segment, a river.
* **Polygon** — a closed ring (plus optional holes): a district, a lake.
* Each has a **Multi-** variant (`MultiPoint`, `MultiLineString`, `MultiPolygon`)
  for objects made of several disjoint pieces — an archipelago, a divided estate.

Every feature pairs **one geometry** with **one row of attributes**. That pairing
is precisely what a `GeoDataFrame` is: a `pandas.DataFrame` where one column
holds Shapely geometry objects and the frame remembers a CRS.

**Formats you will meet.**

| Format | Extension | Verdict |
|---|---|---|
| **GeoPackage** | `.gpkg` | Best default. One SQLite file, many layers, no field-name limits, fast |
| **GeoJSON** | `.geojson` | Great for web/exchange, always EPSG:4326 by spec, verbose and slow for big data |
| **Shapefile** | `.shp` + `.shx` + `.dbf` + `.prj` (+ more) | Legacy. 10-character field names, 2 GB limit, no NULLs, multi-file. Still everywhere |
| **FlatGeobuf / Parquet** | `.fgb` / `.parquet` | Modern, very fast; `geopandas.read_parquet` is excellent for intermediate results |

**Expected outcome.** A list of the 12 layers in the GeoPackage, and three
loaded layers whose shapes and CRS we print.

**What the next cell does:** lists the layers in `vallmara.gpkg` with `pyogrio`,
then loads three of them and summarises each with our `describe_gdf` helper.
'''),

code(r'''
import pyogrio

# --- 1. What is inside the GeoPackage? --------------------------------------
layer_info = pyogrio.list_layers(GPKG)          # -> array of [name, geom_type]
print(f"{GPKG.name} contains {len(layer_info)} layers:\n")
for name, geom_type in layer_info:
    info = pyogrio.read_info(GPKG, layer=name)
    print(f"  {name:<16} {str(geom_type):<12} "
          f"{info['features']:>6} features   {len(info['fields'])} fields")

# --- 2. Load three layers ----------------------------------------------------
districts = gpd.read_file(GPKG, layer="districts")
rivers    = gpd.read_file(GPKG, layer="rivers")
facilities = gpd.read_file(GPKG, layer="facilities")

print("\n" + "=" * 78)
describe_gdf(districts, "districts")
describe_gdf(rivers, "rivers")
describe_gdf(facilities, "facilities")

# --- 3. A GeoDataFrame IS a DataFrame ---------------------------------------
print("\nType hierarchy:")
print("  districts           ", type(districts).__mro__[:3])
print("  districts.geometry  ", type(districts.geometry))
print("  one geometry        ", type(districts.geometry.iloc[0]))
'''),

md(r'''
**Explanation.**

* `pyogrio.list_layers(path)` returns an *N × 2* array of `[layer_name, geometry_type]`.
  Always run this before `read_file` on an unfamiliar `.gpkg` — it costs
  milliseconds and prevents loading the wrong thing.
* `pyogrio.read_info(path, layer=...)` reads only the **header**: feature count,
  field names and types, CRS, bounds. It never touches the geometries, so it is
  instant even on a 10 GB file. This is how you inspect data too big to open.
* `gpd.read_file(GPKG, layer="districts")` — the `layer=` argument is mandatory
  for multi-layer sources. Omit it and you silently get layer 0.
* `type(districts).__mro__[:3]` — the *method resolution order* shows
  `GeoDataFrame → DataFrame → NDFrame`. This is the single most useful fact in
  the course: **every pandas method you know works unchanged**. `groupby`,
  `merge`, `query`, `pivot_table`, `.loc` — all of it.
* `districts.geometry.iloc[0]` is a `shapely.geometry.Polygon`. The geometry
  column is a `GeoSeries`; the individual values are Shapely objects.

**Expected output.**

A 12-row layer inventory:

```
districts        Polygon           24 features  10 fields
census_blocks    Polygon          460 features   6 fields
landuse          Polygon          441 features   4 fields
rivers           LineString         6 features   6 fields
flood_zones      Polygon          181 features   4 fields
buildings        Polygon         5200 features   9 fields
facilities       Point             83 features   7 fields
transit_routes   LineString         6 features   6 fields
bus_stops        Point            130 features   4 fields
sea / land_boundary / coastline    1 feature each
```

Then three `describe_gdf` blocks. Check three things every time:

1. **`crs=EPSG:32633`** — all GeoPackage layers share the analysis CRS.
2. **`bounds=(403,983, 4,600,000) -> (448,000, 4,636,000)`** — coordinates in the
   hundreds of thousands and millions are the signature of a *projected* CRS in
   metres. (The western edge is 403 983 rather than 400 000 because the districts
   stop at the coastline, not at the raster's bounding box.) If you ever see
   bounds like `(-180, -90) -> (180, 90)`, you are in degrees.
3. **`empty=0 invalid=0 null=0`** for these three layers — they are clean.
   (`landuse` and `census_blocks` are not; we will meet them in Module 2.)
'''),

# ------------------------------------------------------------------ B2 -----
md(r'''
## B2 — Exploring the attribute table

**What we are going to learn.** How to interrogate the non-spatial half of a
spatial dataset.

**Why it matters.** A GIS analysis is only as good as its attributes. Before any
spatial operation, you should know the dtypes, the categorical levels, the
missingness pattern and the plausible ranges — exactly as you would for any
tabular dataset. Spatial analysts who skip this step produce beautiful maps of
nonsense.

**The concept — attributes are just a DataFrame.** `gdf.drop(columns="geometry")`
gives you a plain `DataFrame`. Everything you know applies. The *only* new habit
is to keep asking "does this attribute make sense **given where the feature is**?"
— e.g. a district with 200 000 residents and `district_type == "upland_rural"`
would be a red flag.

**Expected outcome.** dtypes, summary statistics, category counts and a
missingness report for the `districts` and `buildings` layers, plus the first
evidence of the deliberate data-quality problems.

**What the next cell does:** prints the schema and head of `districts`,
summarises its numeric and categorical columns, then loads `buildings` and runs
a missingness + implausible-value audit on it.
'''),

code(r'''
# --- 1. Schema ---------------------------------------------------------------
print("DISTRICTS dtypes")
print(districts.dtypes.to_string(), "\n")
print(districts.head(4).drop(columns="geometry").to_string(), "\n")

# --- 2. Numeric summary ------------------------------------------------------
num_cols = ["area_km2", "mean_elev_m", "dist_core_km", "population", "pop_density_km2"]
print("Numeric summary")
print(districts[num_cols].describe().T.to_string(), "\n")

# --- 3. Categorical summary --------------------------------------------------
print("district_type value counts")
print(districts["district_type"].value_counts().to_string())
print("\ncoastal:", districts["coastal"].sum(), "of", len(districts), "districts\n")

# --- 4. Missingness ----------------------------------------------------------
miss = districts.isna().sum()
print("Missing values per column")
print(miss[miss > 0].to_string() if miss.any() else "  (none)")

# --- 5. A messier layer ------------------------------------------------------
buildings = gpd.read_file(GPKG, layer="buildings")
print("\n" + "=" * 78)
print(f"BUILDINGS: {len(buildings):,} rows")
audit = pd.DataFrame({
    "dtype":     buildings.dtypes.astype(str),
    "n_missing": buildings.isna().sum(),
    "pct_missing": (100 * buildings.isna().mean()).round(2),
    "n_unique":  buildings.nunique(),
})
print(audit.to_string(), "\n")

print("year_built - implausible values:")
yb = buildings["year_built"]
print(f"  min={yb.min():.0f}  max={yb.max():.0f}  NaN={yb.isna().sum()}")
print(f"  before 1800: {(yb < 1800).sum()} rows      after 2025: {(yb > 2025).sum()} rows")
print("\nuse_type:", buildings["use_type"].value_counts().to_dict())
'''),

md(r'''
**Explanation.**

* `districts.dtypes` — note that `geometry` has dtype `geometry`. That is a real
  pandas ExtensionDtype provided by GeoPandas, not `object`. It is what lets
  GeoPandas vectorise geometric operations through Shapely 2.
* `.describe().T` — transposing puts variables in rows, which is far easier to
  read when you have many columns.
* `districts[num_cols].describe()` shows `population` with **count = 22**, not 24
  — the two deliberately missing values. `describe()` silently drops NaN; this is
  exactly how missing data slips into an analysis unnoticed. **Always compare the
  `count` row against `len(gdf)`.**
* The `audit` DataFrame is a pattern worth memorising: dtype, count and percent
  missing, and cardinality, in one table. `nunique()` instantly reveals ID
  columns (unique = n) and low-cardinality categoricals.
* `(yb < 1800).sum()` — a **range check**. Missingness is easy to spot; *wrong*
  values that are syntactically valid are not. 12 buildings dated 1066 and 6
  dated 2199 will happily flow into a "building age" feature and silently
  destroy a model unless you look.

**Expected output.**

* `districts` dtypes ending in `geometry  geometry`.
* Numeric summary where **`population` and `pop_density_km2` have count 22.000**
  while every other column has 24 — the two deliberately blanked districts.
* Population ranges from **743** (Ostrand, upland) to **197 870** (Harbourgate,
  urban core); density from **26.9** to **3 545** people/km². That three-order-of-
  magnitude spread is why the choropleth in B6 needs a thoughtful classification.
* `district_type`: **12 `upland_rural`, 8 `suburban`, 2 `urban_core`, 2 `rural`**.
* `coastal: 5 of 24 districts`.
* Missing values: `population 2`, `pop_density_km2 2`.
* For `buildings`: 5 200 rows, `year_built` **240 missing (4.62%)**, `value_kvs`
  **95 missing (1.83%)**, `min=1066`, `max=2199`, **12 rows before 1800** and
  **6 after 2025**.
* `use_type` = `{residential: 3 850, commercial: 830, industrial: 330, public: 190}`.
'''),

# ------------------------------------------------------------------ B3 -----
md(r'''
## B3 — The geometry column: points, lines and polygons

**What we are going to learn.** What actually lives in the geometry column, and
how to interrogate a single geometry.

**Why it matters.** Every spatial operation you will ever run is a method on
these objects. Understanding their structure — rings, coordinates, validity —
is what separates "I ran `intersection` and got an empty result" from "I know
why that returned empty."

**The concept — the Simple Features model.** GeoPandas geometries follow the
OGC **Simple Features** standard, implemented by Shapely on top of GEOS:

```
Geometry
├── Point                 (x, y)
├── LineString            [(x,y), (x,y), ...]           - ordered, may self-cross
├── LinearRing            a closed, simple LineString   - used for polygon rings
├── Polygon               exterior ring + list of interior rings (holes)
└── GeometryCollection
    ├── MultiPoint
    ├── MultiLineString
    └── MultiPolygon
```

Key properties, all lazily computed by GEOS:

* `.area`, `.length` — **in the units of the CRS**. In EPSG:4326 that means
  "square degrees", which is meaningless.
* `.bounds` — `(minx, miny, maxx, maxy)`.
* `.is_valid` — a polygon is valid if its rings do not self-intersect and holes
  lie inside the exterior. Invalid polygons make overlay operations throw.
* `.is_simple` — a line is simple if it does not cross itself.
* `.wkt` / `.wkb` — the text and binary serialisations. WKT is how you eyeball a
  geometry; WKB is how databases store it.

**Expected outcome.** A dissection of one polygon, one line and one point, plus
vectorised geometry properties for a whole layer.

**What the next cell does:** takes a single district polygon and prints its
type, ring structure, coordinate count, area, perimeter and truncated WKT; does
the same for a river LineString and a facility Point; then shows the *vectorised*
form of the same operations across an entire GeoSeries.
'''),

code(r'''
# --- 1. Dissect one POLYGON --------------------------------------------------
poly = districts.geometry.iloc[0]
print("POLYGON —", districts.loc[0, "name"])
print(f"  geom_type      : {poly.geom_type}")
print(f"  exterior ring  : {len(poly.exterior.coords)} vertices")
print(f"  interior rings : {len(poly.interiors)} (holes)")
print(f"  area           : {poly.area:,.0f} m^2  =  {poly.area/1e6:,.2f} km^2")
print(f"  perimeter      : {poly.length:,.0f} m")
print(f"  bounds         : {tuple(round(b) for b in poly.bounds)}")
print(f"  centroid       : ({poly.centroid.x:,.0f}, {poly.centroid.y:,.0f})")
print(f"  is_valid       : {poly.is_valid}")
print(f"  first 3 coords : {[tuple(round(c) for c in xy) for xy in list(poly.exterior.coords)[:3]]}")
print(f"  WKT (truncated): {poly.wkt[:90]} ...\n")

# --- 2. Dissect one LINESTRING ----------------------------------------------
line = rivers.geometry.iloc[0]
print("LINESTRING —", rivers.loc[0, "name"])
print(f"  geom_type      : {line.geom_type}")
print(f"  vertices       : {len(line.coords)}")
print(f"  length         : {line.length:,.0f} m  =  {line.length/1000:.2f} km")
print(f"  is_simple      : {line.is_simple}   (does it avoid crossing itself?)")
print(f"  start -> end   : {tuple(round(c) for c in line.coords[0])} -> "
      f"{tuple(round(c) for c in line.coords[-1])}")
print(f"  midpoint       : {tuple(round(c) for c in line.interpolate(0.5, normalized=True).coords[0])}\n")

# --- 3. Dissect one POINT ----------------------------------------------------
pt = facilities.geometry.iloc[0]
print("POINT —", facilities.loc[0, "name"])
print(f"  geom_type      : {pt.geom_type}   coords: ({pt.x:,.1f}, {pt.y:,.1f})")
print(f"  area           : {pt.area}      length: {pt.length}   (points have neither)\n")

# --- 4. The VECTORISED form — this is how you actually work ------------------
print("Vectorised geometry properties over the whole layer:")
geom_summary = pd.DataFrame({
    "name":        districts["name"],
    "geom_type":   districts.geometry.geom_type,
    "n_vertices":  districts.geometry.count_coordinates(),
    "area_km2":    districts.geometry.area / 1e6,
    "perim_km":    districts.geometry.length / 1000,
    "is_valid":    districts.geometry.is_valid,
})
print(geom_summary.head(6).to_string(index=False))
print(f"\nTotal land area of all districts: {districts.geometry.area.sum()/1e6:,.1f} km^2")
'''),

md(r'''
**Explanation.**

* `poly.exterior.coords` — the outer ring as a coordinate sequence. It is
  **closed**: the last vertex equals the first, so a triangle reports 4 vertices.
* `poly.interiors` — the list of holes. Vallmara districts have none, but a
  district containing a lake excluded from its area would.
* `poly.area` returns **square metres** *because the CRS is UTM (metres)*.
  Shapely itself knows nothing about CRS; it does flat Cartesian arithmetic on
  whatever numbers you give it. **The units come entirely from the CRS you chose.**
  This is the deepest trap in the whole field.
* `poly.centroid` — the area-weighted centre of mass. Note it can fall *outside*
  a concave polygon (a horseshoe-shaped district). When you need a point
  guaranteed to be inside, use `.representative_point()` instead.
* `line.interpolate(0.5, normalized=True)` — the point halfway along the line by
  distance. With `normalized=False` the argument is in CRS units (metres here).
  This is the workhorse for placing labels, sampling along transects, and
  generating stops along a route.
* `.count_coordinates()` — a GeoSeries method (Shapely 2 / GeoPandas 1) giving
  vertex counts without a Python loop. Vertex count is your first proxy for
  geometric complexity and hence for how slow an overlay will be.
* Note the pattern: **singular Shapely properties** (`poly.area`) versus
  **vectorised GeoSeries properties** (`districts.geometry.area`). The second
  returns a `pandas.Series` and is what you use in real work.

**Expected output.**

```
POLYGON — Old Vallmara
  geom_type      : Polygon
  exterior ring  : ~80 vertices
  interior rings : 0 (holes)
  area           : 53,322,xxx m^2  =  53.32 km^2
  perimeter      : ~40,000 m
  centroid       : (~417,000, ~4,619,000)
  is_valid       : True
```

Vertex counts of 35–90 are the signature of districts that were built by
**dissolving census blocks** — their boundaries follow block edges, so they are
irregular and non-convex, exactly like real administrative units.

The river prints a length of **43.93 km** with `is_simple: True` and 34 vertices.
The point prints `area 0.0  length 0.0` — points are zero-dimensional.

The final table shows six districts and
**total land area = 1 394.8 km²**, which matches the study-area figure in the
scenario description. That agreement is your first sanity check.
'''),

# ------------------------------------------------------------------ B4 -----
md(r'''
## B4 — CRS I: what a coordinate reference system actually is

**What we are going to learn.** The anatomy of a CRS, how to read one from a
layer, and the difference between geographic and projected systems.

**Why it matters.** This is the concept that most often silently breaks GIS
analyses done by strong data scientists. Distances, areas, buffers, nearest
neighbours, densities, cluster radii — **all of them are wrong** if the CRS is
wrong, and none of them will raise an error. You get plausible numbers that are
simply not true.

**The concept.** A CRS answers: *what do the numbers in the geometry column
mean?* It has three parts:

1. **Datum** — a model of the Earth's shape and its position: an ellipsoid
   (e.g. WGS 84: semi-major axis 6 378 137 m, flattening 1/298.257223563) plus a
   realisation tying it to physical reference points. Two datums can put "the
   same" latitude/longitude hundreds of metres apart.
2. **Coordinate system** — the axes and their units: (longitude, latitude) in
   degrees, or (easting, northing) in metres.
3. **Projection** (only for projected CRS) — the mathematical function flattening
   the curved surface onto a plane.

**Geographic vs projected.**

| | Geographic (e.g. EPSG:4326) | Projected (e.g. EPSG:32633) |
|---|---|---|
| Coordinates | longitude, latitude in **degrees** | easting, northing in **metres** |
| Is 1 unit a fixed length? | **No.** 1° latitude ≈ 111 km always; 1° longitude ≈ 111 km × cos(lat) — 111 km at the equator, 0 km at the pole | **Yes**, to within the projection's scale error |
| Valid for `.area`, `.length`, `.buffer`, `.distance`? | **No** | **Yes** |
| Valid for storage and exchange? | Yes — it is the universal interchange | Yes, if you record which one |

**The fundamental theorem of map projections.** No flat map can preserve area,
angle and distance simultaneously. Every projection sacrifices at least one:

* **UTM / Transverse Mercator** — *conformal* (preserves local angles/shape).
  Divides the world into 60 zones 6° wide; distortion inside a zone is tiny
  (scale error < 1/2 500). **This is what you want for regional analysis.**
* **Web Mercator (EPSG:3857)** — conformal, spans the globe, but inflates area by
  1/cos²(latitude): Greenland looks the size of Africa. Fine for tiles,
  catastrophic for statistics.
* **Equal-area projections** (Albers, Lambert Azimuthal, Mollweide) — preserve
  area, distort shape. Use for continental/global area or density work.

**The EPSG code.** Every well-known CRS has an integer identifier from the EPSG
registry. `EPSG:32633` decodes as WGS 84 / UTM zone **33** **N**orth. The zone
number is derived from longitude: `zone = floor((lon + 180) / 6) + 1`.

**Expected outcome.** Full CRS metadata for our layers, and a demonstration that
the same file can hold radically different-looking numbers for the same place.

**What the next cell does:** prints the structured CRS metadata for the UTM
districts layer and the WGS 84 roads layer, computes the correct UTM zone from
the study area's longitude, and shows the same district centroid expressed in
three CRS.
'''),

code(r'''
# --- 1. Interrogate a projected CRS -----------------------------------------
crs = districts.crs
print("=" * 78)
print("DISTRICTS CRS")
print("=" * 78)
print(f"  name          : {crs.name}")
print(f"  EPSG code     : {crs.to_epsg()}")
print(f"  is_projected  : {crs.is_projected}")
print(f"  is_geographic : {crs.is_geographic}")
print(f"  unit          : {crs.axis_info[0].unit_name}")
print(f"  axes          : {[f'{a.name} ({a.abbrev}, {a.direction})' for a in crs.axis_info]}")
print(f"  datum         : {crs.datum.name}")
print(f"  ellipsoid     : {crs.ellipsoid.name}  "
      f"(a = {crs.ellipsoid.semi_major_metre:,.0f} m)")
print(f"  area of use   : {crs.area_of_use.name}")
print(f"  bounds of use : {tuple(round(v, 2) for v in crs.area_of_use.bounds)}")

# --- 2. Interrogate a geographic CRS ----------------------------------------
roads_raw = gpd.read_file(VEC / "roads.geojson")
print("\n" + "=" * 78)
print("ROADS CRS (a different file, a different CRS - on purpose)")
print("=" * 78)
print(f"  name          : {roads_raw.crs.name}")
print(f"  EPSG code     : {roads_raw.crs.to_epsg()}")
print(f"  is_projected  : {roads_raw.crs.is_projected}")
print(f"  unit          : {roads_raw.crs.axis_info[0].unit_name}")
print(f"  bounds        : {tuple(round(v, 4) for v in roads_raw.total_bounds)}")

# --- 3. Which UTM zone SHOULD this region be in? -----------------------------
centre_lonlat = districts.to_crs(CRS_WGS84).geometry.union_all().centroid
lon, lat = centre_lonlat.x, centre_lonlat.y
zone = int((lon + 180) // 6) + 1
epsg = (32600 if lat >= 0 else 32700) + zone
print(f"\nStudy-area centre  : lon {lon:.4f}, lat {lat:.4f}")
print(f"Computed UTM zone  : {zone}{'N' if lat >= 0 else 'S'}  ->  EPSG:{epsg}")
print(f"Layer actually uses: EPSG:{districts.crs.to_epsg()}   "
      f"{'MATCH' if epsg == districts.crs.to_epsg() else 'MISMATCH!'}")

# --- 4. The same place, three coordinate systems -----------------------------
p_utm = districts.geometry.iloc[1].centroid
rows = []
for label, target in [("EPSG:32633 (UTM 33N, m)", CRS_UTM),
                      ("EPSG:4326  (WGS84, deg)", CRS_WGS84),
                      ("EPSG:3857  (WebMerc, m)", CRS_WEBMERC)]:
    g = gpd.GeoSeries([p_utm], crs=CRS_UTM).to_crs(target).iloc[0]
    rows.append({"CRS": label, "x / lon": round(g.x, 5), "y / lat": round(g.y, 5)})
print("\nThe centroid of", districts.loc[1, "name"], "expressed three ways:")
print(pd.DataFrame(rows).to_string(index=False))
'''),

md(r'''
**Explanation.**

* `gdf.crs` is a **`pyproj.CRS`** object, not a string. It exposes the full
  WKT2 definition programmatically — `datum`, `ellipsoid`, `axis_info`,
  `area_of_use`. Print it directly (`print(crs)`) to see the raw WKT.
* `crs.axis_info[0].unit_name` — the authoritative answer to "what are my units?".
  Never assume; ask. For EPSG:32633 it is `metre`; for EPSG:4326 it is `degree`.
* `crs.area_of_use.bounds` — the longitude/latitude box within which the CRS is
  *defined to be accurate*. For UTM 33N that is roughly `(12, 0, 18, 84)`. Using
  a UTM zone far outside its band inflates distortion quickly; using it 30° away
  is simply wrong.
* `crs.to_epsg()` can return `None` for a custom CRS that is not in the registry.
  Code defensively when you did not create the file.
* **The zone formula.** `zone = floor((lon + 180) / 6) + 1`, then EPSG code
  `32600 + zone` for the northern hemisphere and `32700 + zone` for the southern.
  Memorise this: it is how you pick a correct analysis CRS anywhere on Earth in
  five seconds.
* `districts.to_crs(CRS_WGS84).geometry.union_all().centroid` — dissolve all
  polygons into one and take its centroid. `union_all()` is the GeoPandas 1.x
  name; older code says `unary_union` (still works, deprecated).

**Expected output.**

```
DISTRICTS CRS
  name          : WGS 84 / UTM zone 33N
  EPSG code     : 32633
  is_projected  : True
  unit          : metre
  axes          : ['Easting (E, east)', 'Northing (N, north)']
  datum         : World Geodetic System 1984
  ellipsoid     : WGS 84  (a = 6,378,137 m)
```

Roads report `EPSG:4326`, `is_projected: False`, `unit: degree`, and bounds like
`(13.68, 41.53, 14.27, 41.86)` — **three-digit numbers instead of six-digit ones.
That contrast is the fastest way to tell degrees from metres at a glance.**

The zone computation prints `lon ≈ 13.95, lat ≈ 41.70 → zone 33N → EPSG:32633
MATCH`, confirming the dataset's analysis CRS is the right choice.

The three-way table shows the same point as roughly:

| CRS | x / lon | y / lat |
|---|---|---|
| UTM 33N (m) | 413 000 | 4 621 000 |
| WGS 84 (deg) | 13.85 | 41.73 |
| Web Mercator (m) | 1 541 000 | **5 122 000** |

Note that the Web Mercator *northing* (5 122 km) is much larger than the UTM
northing (4 621 km) for the same physical location. Web Mercator stretches the
y-axis increasingly towards the poles — the visible symptom of its area
distortion.
'''),

# ------------------------------------------------------------------ B5 -----
md(r'''
## B5 — CRS II: reprojection, and measuring the cost of getting it wrong

**What we are going to learn.** How to reproject with `to_crs`, the crucial
difference between `set_crs` and `to_crs`, and — by direct measurement — how
badly wrong your numbers are if you compute in the wrong CRS.

**Why it matters.** This lesson contains the single most important number in the
course. We will compute the area of the same region in three CRS and compare.

**The concept.**

* **`set_crs(crs)`** — *labels* the data. It changes the metadata and **not one
  coordinate**. Use it only when the file arrived with no CRS and you know from
  documentation what it is. Using it wrongly silently corrupts everything
  downstream.
* **`to_crs(crs)`** — *transforms* the data. Every coordinate is pushed through a
  PROJ pipeline (inverse projection → datum shift → forward projection). The
  metadata changes **and so do all the numbers**.

Mnemonic: **`set_crs` changes the label on the tin; `to_crs` changes what is in
the tin.**

**The workflow rule.** Load → immediately reproject everything to your analysis
CRS → do all measurement → reproject only for output. Layers in different CRS
cannot be joined, overlaid or plotted together; GeoPandas will either raise or
(worse, in older versions) give nonsense.

**Expected outcome.** A table quantifying the area error from working in
EPSG:4326 and EPSG:3857 rather than UTM, and a demonstration of a 500 m buffer
built in degrees versus metres.

**What the next cell does:** reprojects the district layer into three CRS,
computes total area in each, expresses the error in percent; then shows what a
"0.005 unit" buffer means in each system; and finally reprojects the roads layer
into the analysis CRS for the rest of the course.
'''),

code(r'''
# --- 1. Area of the SAME polygons, computed in three CRS --------------------
truth_km2 = districts.to_crs(CRS_UTM).geometry.area.sum() / 1e6

rows = []
for label, crs_code, unit in [
        ("EPSG:32633  UTM 33N", CRS_UTM, "m^2"),
        ("EPSG:4326   WGS 84 geographic", CRS_WGS84, "deg^2"),
        ("EPSG:3857   Web Mercator", CRS_WEBMERC, "m^2"),
        ("ESRI:54009  Mollweide (equal area)", "ESRI:54009", "m^2")]:
    g = districts.to_crs(crs_code)
    raw = g.geometry.area.sum()
    km2 = raw / 1e6 if unit == "m^2" else np.nan
    rows.append({
        "CRS": label,
        "raw .area sum": f"{raw:,.6g}",
        "units": unit,
        "implied km^2": f"{km2:,.1f}" if np.isfinite(km2) else "meaningless",
        "error vs UTM": (f"{100*(km2-truth_km2)/truth_km2:+.1f} %"
                         if np.isfinite(km2) else "-"),
    })
print("TOTAL AREA OF THE 24 DISTRICTS, computed in four coordinate systems")
print("=" * 92)
print(pd.DataFrame(rows).to_string(index=False))
print("=" * 92)
print(f"\nGround truth (UTM): {truth_km2:,.1f} km^2")

# --- 2. GeoPandas can also do it properly on the ellipsoid ------------------
geod_area = districts.to_crs(CRS_WGS84).geometry.to_crs(CRS_UTM).area.sum() / 1e6
print(f"Round-trip 32633 -> 4326 -> 32633: {geod_area:,.1f} km^2  "
      f"(loss from the round trip: {abs(geod_area-truth_km2):.4f} km^2)")

# --- 3. The "buffer in degrees" hack, measured honestly ---------------------
pt_utm = gpd.GeoSeries([Point(418_000, 4_618_000)], crs=CRS_UTM)

# (a) the correct way: 500 metres in a metric CRS
buf_m = pt_utm.buffer(500)
b = buf_m.total_bounds
print("\nA 500 m buffer around Vallmara City centre")
print(f"  buffer(500) in EPSG:32633 -> width {b[2]-b[0]:,.0f} m, "
      f"height {b[3]-b[1]:,.0f} m, area {buf_m.area.iloc[0]/1e6:.4f} km^2")

# (b) the common hack: "0.0045 degrees is about 500 m"
buf_deg = pt_utm.to_crs(CRS_WGS84).buffer(0.0045).to_crs(CRS_UTM)
b = buf_deg.total_bounds
print(f"  buffer(0.0045) in EPSG:4326 -> width {b[2]-b[0]:,.0f} m, "
      f"height {b[3]-b[1]:,.0f} m, area {buf_deg.area.iloc[0]/1e6:.4f} km^2")
print(f"  the 'circle' is really an ELLIPSE: "
      f"east-west radius {(b[2]-b[0])/2:,.0f} m vs north-south radius {(b[3]-b[1])/2:,.0f} m")
print(f"  1 degree of latitude  = 111,320 m everywhere")
print(f"  1 degree of longitude = 111,320 * cos(41.7 deg) = "
      f"{111_320*np.cos(np.radians(41.7)):,.0f} m HERE, and 0 m at the pole")

# --- 4. Put the roads layer into the analysis CRS ---------------------------
print("\nBefore:", roads_raw.crs, "| bounds", np.round(roads_raw.total_bounds, 3))
roads = roads_raw.to_crs(CRS_UTM)
print("After :", roads.crs, "| bounds", np.round(roads.total_bounds, 0))
print(f"\nA motorway segment measured in degrees: "
      f"{roads_raw.loc[roads_raw.road_class=='motorway'].geometry.length.iloc[0]:.6f} (degrees - useless)")
print(f"The same segment measured in metres  : "
      f"{roads.loc[roads.road_class=='motorway'].geometry.length.iloc[0]:,.0f} m")
'''),

md(r'''
**Explanation.**

* The whole point of the first block is that **`.area` never complains.** In
  EPSG:4326 it returns a number in *square degrees* — a quantity with no physical
  meaning, because a degree of longitude shrinks towards the poles. GeoPandas
  emits a warning in recent versions, but the number is still returned and will
  still flow into your report.
* **Web Mercator is the dangerous one**, because its units *are* metres, so
  nothing looks wrong. At the latitude of the Vallmara Basin (≈ 41.7° N) areas are
  inflated by `1/cos²(41.7°) ≈ 1.80`, i.e. **about +80%**. A "density per km²"
  computed in EPSG:3857 is off by nearly a factor of two — and every web map you
  have ever taken a screenshot of is in EPSG:3857.
* **Mollweide (`ESRI:54009`)** is an equal-area projection; its total agrees with
  UTM to a fraction of a percent, confirming the UTM figure is right. When two
  independent projections agree, you can trust the number.
* The round-trip test (32633 → 4326 → 32633) loses only a tiny fraction of a
  km²: PROJ transformations are essentially lossless within a datum. The danger
  is *computing* in the wrong CRS, not *converting* between them.
* `buffer(500)` in EPSG:4326 means "500 **degrees**", which wraps the entire
  planet many times over — the resulting "radius" is astronomically large. In
  practice you will more often see someone write `buffer(0.005)` intending
  "about 500 m", which is a rough approximation at one particular latitude and
  wrong everywhere else. **Never buffer in a geographic CRS.**
* The final block establishes `roads` (UTM) as the canonical roads layer for the
  rest of the notebook. `roads_raw` stays around only as a cautionary example.

**Expected output.**

```
TOTAL AREA OF THE 24 DISTRICTS, computed in four coordinate systems
 CRS                                 raw .area sum   units  implied km^2  error vs UTM
 EPSG:32633  UTM 33N                  1.39478e+09     m^2       1,394.8       +0.0 %
 EPSG:4326   WGS 84 geographic            0.15101     deg^2  meaningless        -
 EPSG:3857   Web Mercator              2.50704e+09    m^2       2,507.0      +79.7 %
 ESRI:54009  Mollweide (equal area)     1.3968e+09    m^2       1,396.8       +0.1 %
```

**Read that Web Mercator row again: +79.7%.** Then:

```
A 500 m buffer around Vallmara City centre
  buffer(500)    in EPSG:32633 -> width 1,000 m, height 1,000 m, area 0.7841 km^2
  buffer(0.0045) in EPSG:4326  -> width   749 m, height   999 m, area 0.5867 km^2
  the 'circle' is really an ELLIPSE: east-west radius 374 m vs north-south radius 500 m
  1 degree of longitude = 111,320 * cos(41.7 deg) = 83,116 m HERE, and 0 m at the pole
```

The degree buffer is not merely mis-scaled, it is the **wrong shape**: 25% too
small east–west and correct north–south. Any "nearest within 500 m" analysis
built on it silently misses features to the east and west.

Finally the roads bounds change from `(13.884, 41.548, 14.363, 41.871)` to
`(407,289, 4,600,000, 446,946, 4,636,000)`, with a motorway segment measuring
`0.011198` in degrees versus **1 185 m** in metres.

You will also see three `UserWarning: Geometry is in a geographic CRS. Results
from 'area'/'buffer'/'length' are likely incorrect.` messages. **Those warnings
are the correct behaviour** — GeoPandas is telling you exactly what this lesson
is about. Never silence them globally.
'''),

# ------------------------------------------------------------------ B6 -----
md(r'''
## B6 — Plotting spatial data

**What we are going to learn.** How to draw layers, build a layered map, and
make an honest choropleth.

**Why it matters.** A map is an argument. The classification scheme you choose
determines which places look "high" and which look "low" — the same data can
support opposite-looking maps. Choosing a scheme is an analytical decision, not
a styling one.

**The concept — `.plot()` is matplotlib.** `gdf.plot()` returns a matplotlib
`Axes`. Layering is simply passing `ax=` to subsequent calls. The important
arguments:

| Argument | Meaning |
|---|---|
| `column=` | Attribute to colour by (makes it a thematic map) |
| `cmap=` | Colormap. **Sequential** (`viridis`, `Blues`) for magnitude; **diverging** (`RdBu`) only when there is a meaningful midpoint; **qualitative** (`tab10`, `Set2`) for categories |
| `scheme=` | Classification method (needs `mapclassify`): `quantiles`, `equalinterval`, `naturalbreaks`, `stdmean`, `fisherjenks` |
| `k=` | Number of classes (5–7 is the readable maximum) |
| `legend=` / `legend_kwds=` | Legend and its formatting |
| `edgecolor=`, `linewidth=`, `alpha=`, `markersize=` | Cosmetics |
| `missing_kwds=` | **How to draw NaN.** Without this, missing values are silently invisible |

**Classification schemes and the argument they make.**

* **Equal interval** — cuts the range into equal-width bins. Honest about
  magnitude; useless when the distribution is skewed (one huge city swamps
  everything).
* **Quantiles** — equal *counts* per bin. Always produces a "colourful" map even
  when the data are uniform; can exaggerate trivial differences.
* **Natural breaks (Fisher–Jenks)** — minimises within-class variance. Usually
  the best compromise for exploratory work.
* **Standard deviation** — shows departures from the mean; requires roughly
  symmetric data.

**Expected outcome.** A 2 × 2 figure: a plain geometry plot, a categorical map,
a choropleth with an explicit classification, and a fully layered reference map.

**What the next cell does:** builds four subplots demonstrating each style,
including explicit handling of the two districts with missing population.
'''),

code(r'''
land = gpd.read_file(GPKG, layer="land_boundary")
sea  = gpd.read_file(GPKG, layer="sea")

fig, axes = plt.subplots(2, 2, figsize=(13.5, 11))

# --- (a) plain geometry ------------------------------------------------------
ax = axes[0, 0]
districts.plot(ax=ax, facecolor="#e8eef7", edgecolor="#3b5378", linewidth=0.7)
ax.set_title("(a) Plain geometry\ndistricts.plot()", loc="left", fontsize=10, weight="bold")

# --- (b) categorical ---------------------------------------------------------
ax = axes[0, 1]
districts.plot(ax=ax, column="district_type", categorical=True, cmap="Set2",
               legend=True, edgecolor="white", linewidth=0.6,
               legend_kwds={"loc": "lower left", "fontsize": 7, "frameon": True})
ax.set_title("(b) Categorical map\ncolumn='district_type'", loc="left",
             fontsize=10, weight="bold")

# --- (c) choropleth with an explicit scheme + explicit NaN handling ---------
ax = axes[1, 0]
districts.plot(ax=ax, column="pop_density_km2", cmap="YlOrRd",
               scheme="naturalbreaks", k=5, legend=True,
               edgecolor="grey", linewidth=0.4,
               legend_kwds={"loc": "lower left", "fontsize": 7,
                            "title": "people / km^2", "title_fontsize": 8},
               missing_kwds={"color": "#d9d9d9", "edgecolor": "red",
                             "hatch": "///", "label": "no data"})
ax.set_title("(c) Choropleth, natural breaks\nNaN shown explicitly (red hatching)",
             loc="left", fontsize=10, weight="bold")

# --- (d) a layered reference map --------------------------------------------
ax = axes[1, 1]
sea.plot(ax=ax, facecolor="#cfe3f2", edgecolor="none", zorder=0)
land.plot(ax=ax, facecolor="#f6f3ec", edgecolor="none", zorder=1)
districts.boundary.plot(ax=ax, color="#9aa6b8", linewidth=0.5, zorder=2)
roads[roads.road_class.isin(["motorway", "primary"])].plot(
    ax=ax, color="#7a5c3e", linewidth=0.8, zorder=3)
rivers.plot(ax=ax, color="#2f7fbf", linewidth=1.1, zorder=4)
facilities[facilities.facility_type == "hospital"].plot(
    ax=ax, color="crimson", markersize=45, marker="P",
    edgecolor="white", linewidth=0.6, zorder=5, label="hospital")
ax.legend(loc="lower left", fontsize=7, frameon=True)
ax.set_title("(d) Layered reference map\nzorder controls what covers what",
             loc="left", fontsize=10, weight="bold")

for a in axes.ravel():
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
plt.tight_layout()
plt.show()
'''),

md(r'''
**Explanation.**

* **`ax=` is the layering mechanism.** Each `.plot(ax=ax, ...)` draws onto the
  same axes. Because they are all in EPSG:32633 they line up. Plot a layer in a
  different CRS onto the same axes and it will appear as a dot in a corner (or
  not at all) — the classic "my map is empty" bug.
* **`zorder`** controls draw order explicitly. Without it, matplotlib uses call
  order, which is fine until you insert a layer and everything vanishes behind
  the polygon you drew last.
* **`missing_kwds`** is the ethical bit. By default, features with NaN in the
  `column` are drawn *in no colour at all*, i.e. they disappear into the
  background and the reader assumes they are "low". Explicitly hatching them in
  grey/red states "we do not know" — which is what the data actually says. The
  two hatched districts are Cliffmoor and Highfen, our deliberate missing values.
* **`scheme="naturalbreaks", k=5`** invokes `mapclassify.NaturalBreaks`, which
  runs Fisher–Jenks optimisation to minimise within-class variance. Compare it
  with `scheme="quantiles"`: quantiles will always fill all five colour classes
  evenly, which here would make several near-empty rural districts look
  moderately populated.
* `districts.boundary` — a `GeoSeries` of the polygon *outlines* (LineStrings).
  Plotting boundaries rather than filled polygons is how you draw an overlay grid
  without hiding what is underneath.
* `sea` and `land` give the map a ground: without them a district map floats in
  white space and the coastline is invisible.

**Expected output.** A 2 × 2 grid of maps:

* **(a)** 24 pale-blue polygons tessellating a wedge-shaped region whose western
  edge is a wavy coastline.
* **(b)** the same polygons coloured in four categories, with `upland_rural`
  dominating the east and the three `urban_core` districts clustered in the
  centre-west.
* **(c)** a yellow→red choropleth in which the urban core is deep red
  (≈ 3 000 people/km²) and the eastern districts are pale (< 100 people/km²);
  **two districts are grey with red hatching**.
* **(d)** a proper reference map: blue sea, cream land, grey district outlines,
  brown roads, blue rivers, four red hospital crosses concentrated near the core.

If panel (d) looks empty except for one tiny cluster, you have a CRS mismatch —
re-run cell B5.
'''),

# ------------------------------------------------------------------ B7 -----
md(r'''
## B7 — Building a GeoDataFrame from a CSV

**What we are going to learn.** How to turn a plain table with coordinate
columns into a spatial layer — and the one-line mistake that silently ruins it.

**Why it matters.** Most data reaches you as a CSV with `lat`/`lon` columns:
sensor readings, incident reports, GPS traces, geocoded addresses. Converting
correctly is the entry point to every subsequent spatial operation.

**The concept — three steps, in this order.**

1. **Build the geometry** — `gpd.points_from_xy(df.lon, df.lat)`.
   Note the order: **x first, then y**; i.e. **longitude first, then latitude**.
   Writing `points_from_xy(df.lat, df.lon)` is the most common error in applied
   GIS and it puts your data in the wrong hemisphere without any error message.
2. **Declare the CRS** — `crs="EPSG:4326"` in the constructor, or `.set_crs(...)`.
   This is `set_crs` semantics: you are *labelling* coordinates that already
   exist. If the file's documentation says the coordinates are lon/lat on WGS 84,
   the label is `EPSG:4326`.
3. **Reproject for analysis** — `.to_crs(CRS_UTM)` before measuring anything.

**Concept — why lon/lat order confusion exists.** Humans say "latitude,
longitude" (49.5° N, 6.1° E). Mathematics and every plotting library say
"(x, y)". Longitude *is* x. GeoPandas, Shapely and Rasterio are consistently
(x, y). The EPSG registry, meanwhile, formally defines EPSG:4326 axis order as
(latitude, longitude), which is why some tools (and some WMS servers) disagree.
**Rule: in Python, always (lon, lat).**

**Expected outcome.** The 24 sensor stations as a proper GeoDataFrame in UTM,
validated against the `x_utm`/`y_utm` columns already in the file.

**What the next cell does:** loads the stations CSV, builds points from lon/lat,
sets EPSG:4326, reprojects to UTM, and then *verifies* the result by comparing
against the independently stored UTM columns — a check you should perform every
time coordinates are involved.
'''),

code(r'''
# --- 1. Read the plain table -------------------------------------------------
stations_df = pd.read_csv(TAB / "sensor_stations.csv")
print("Raw CSV — this is NOT spatial yet:", type(stations_df).__name__)
print(stations_df.head(3).to_string(index=False), "\n")

# --- 2. lon/lat -> geometry, declare CRS, reproject --------------------------
stations = gpd.GeoDataFrame(
    stations_df.copy(),
    geometry=gpd.points_from_xy(stations_df["lon"], stations_df["lat"]),  # x, y !
    crs=CRS_WGS84,                       # LABEL the existing coordinates
).to_crs(CRS_UTM)                        # TRANSFORM into the analysis CRS

describe_gdf(stations, "stations")

# --- 3. VERIFY. Never skip this. --------------------------------------------
dx = stations.geometry.x - stations["x_utm"]
dy = stations.geometry.y - stations["y_utm"]
err = np.hypot(dx, dy)
print(f"\nCheck against the file's own UTM columns:")
print(f"  max positional error = {err.max():.4f} m   (should be < 0.2 m)")
print(f"  mean error           = {err.mean():.4f} m")
print("  (the CSV stores lon/lat rounded to 6 decimal places ~ 0.1 m, so a few")
print("   centimetres of disagreement is the rounding, not a CRS error)")

# --- 4. What the WRONG order would have done --------------------------------
wrong = gpd.GeoDataFrame(
    stations_df.copy(),
    geometry=gpd.points_from_xy(stations_df["lat"], stations_df["lon"]),  # swapped!
    crs=CRS_WGS84,
)
print(f"\nIf you swap lat/lon:")
print(f"  correct centroid : ({stations.to_crs(CRS_WGS84).geometry.x.mean():.3f} E, "
      f"{stations.to_crs(CRS_WGS84).geometry.y.mean():.3f} N)  -> Kestria")
print(f"  swapped centroid : ({wrong.geometry.x.mean():.3f} E, "
      f"{wrong.geometry.y.mean():.3f} N)  -> somewhere else entirely")
print(f"  displacement     : ~{wrong.to_crs(CRS_UTM).geometry.iloc[0].distance(stations.geometry.iloc[0])/1000:,.0f} km")

# --- 5. Plot to confirm ------------------------------------------------------
fig, ax = fresh_ax((7.5, 6), "24 environmental monitoring stations")
land.plot(ax=ax, facecolor="#f6f3ec", edgecolor="#c9c2b4")
districts.boundary.plot(ax=ax, color="#cdd4de", linewidth=0.5)
stations.plot(ax=ax, column="station_type", categorical=True, cmap="Dark2",
              markersize=55, edgecolor="black", linewidth=0.5, legend=True,
              legend_kwds={"loc": "lower left", "fontsize": 7})
for _, r in stations.iterrows():
    ax.annotate(r["station_id"], (r.geometry.x, r.geometry.y),
                xytext=(3, 3), textcoords="offset points", fontsize=5.5, color="#333")
plt.show()
'''),

md(r'''
**Explanation.**

* `gpd.points_from_xy(x, y)` — a vectorised constructor returning a
  `GeometryArray`. It is far faster than `df.apply(lambda r: Point(r.lon, r.lat), axis=1)`
  and it is the idiom you should use.
* `crs=CRS_WGS84` **in the constructor** is equivalent to `.set_crs(CRS_WGS84)`.
  It attaches a label. It does not move anything.
* `.to_crs(CRS_UTM)` then physically transforms all 24 points.
* **The verification block is the real lesson.** Because this dataset ships both
  `lon`/`lat` *and* `x_utm`/`y_utm`, we can prove the transformation is correct to
  sub-millimetre precision. In production you rarely get that luxury, so verify
  by other means: plot the points over a known boundary, check the bounding box
  against expectation, or confirm a handful of known locations.
* The "wrong order" demonstration shows the failure mode: with lat/lon swapped,
  points land at roughly (41.7° E, 13.9° N) — in the Gulf of Aden rather than
  Kestria, thousands of kilometres away. Note that **no exception is raised**.
  A swapped-coordinate dataset will happily complete every downstream operation
  and return confident, wrong answers.
* `ax.annotate(..., textcoords="offset points")` offsets the label a few pixels
  from the marker regardless of zoom level — the correct way to label map points.

**Expected output.**

* A 3-row preview of the CSV with `station_id`, `name`, `station_type`,
  `install_year`, `elevation_m`, `x_utm`, `y_utm`, `lon`, `lat`.
* `describe_gdf`: **24 rows, EPSG:32633, `geom={'Point': 24}`**, bounds inside the
  study area.
* `max positional error ≈ 0.067 m`, `mean ≈ 0.037 m` — a few centimetres, which is
  exactly the rounding of `lon`/`lat` to six decimal places. If you see hundreds
  of metres, your CRS label is wrong; if you see thousands of kilometres, your
  lon/lat are swapped.
* The swap demonstration reports `correct centroid (14.145 E, 41.714 N)`,
  `swapped centroid (41.714 E, 14.145 N)` and a displacement of **≈ 4 205 km**.
* A map of 24 labelled points spread across the basin, coloured by
  `station_type` (air_quality / rain_gauge / combined).
'''),

]

CELLS += [

# ------------------------------------------------------------------ B8 -----
md(r'''
## B8 — Basic Shapely operations

**What we are going to learn.** The core geometric constructors and predicates —
the vocabulary in which every spatial analysis is written.

**Why it matters.** GeoPandas is a thin, convenient wrapper; the actual geometry
is Shapely/GEOS. Knowing what each operation returns (and what it costs) lets you
compose analyses instead of searching for a function that does exactly your task.

**The concept — three families of operation.**

**1. Constructive operations** return *new geometry*:

| Operation | Meaning |
|---|---|
| `buffer(d)` | All points within distance `d`. Positive grows, **negative shrinks** (erosion) |
| `centroid` | Area-weighted centre of mass — may fall outside a concave shape |
| `representative_point()` | A point guaranteed to be *inside* the geometry |
| `convex_hull` | Smallest convex polygon containing the geometry |
| `envelope` | Bounding box as a polygon |
| `simplify(tol)` | Douglas–Peucker vertex reduction |
| `boundary` | Dimension−1 edge: polygon → ring, line → endpoints |
| `intersection / union / difference / symmetric_difference` | Boolean set operations |

**2. Predicates** return `True`/`False` (the DE-9IM relations):

| Predicate | True when |
|---|---|
| `intersects` | They share **any** point. The catch-all; opposite of `disjoint` |
| `contains` / `within` | One is entirely inside the other (boundaries may touch) |
| `covers` / `covered_by` | Like contains/within but tolerant of boundary-only contact |
| `touches` | They share a boundary but **no interior** |
| `crosses` | Interiors intersect but neither contains the other (line × polygon) |
| `overlaps` | Same dimension, partial overlap, neither contains the other |

**3. Measures** return numbers: `area`, `length`, `distance`, `hausdorff_distance`.

**The crucial subtlety — `contains` vs `intersects`.** A polygon that shares only
an edge with another `intersects` it but does not `contain` it. In a spatial join
this is the difference between counting a boundary feature once, twice, or never.

**Expected outcome.** A visual and numeric tour of buffers, hulls,
simplification and the predicate matrix.

**What the next cell does:** builds and plots five constructive operations on a
single district polygon, prints how simplification trades vertices for accuracy,
and evaluates the full predicate set between selected features.
'''),

code(r'''
d = districts.loc[districts["name"] == "Harbourgate"].geometry.iloc[0]

# --- 1. Constructive operations ---------------------------------------------
ops = {
    "original":               d,
    "buffer(+1500 m)":        d.buffer(1500),
    "buffer(-1000 m)":        d.buffer(-1000),
    "convex_hull":            d.convex_hull,
    "envelope":               d.envelope,
    "simplify(500 m)":        d.simplify(500),
}
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, (label, g) in zip(axes.ravel(), ops.items()):
    gpd.GeoSeries([d], crs=CRS_UTM).plot(ax=ax, facecolor="#dfe7f3",
                                         edgecolor="#4a6fa5", linewidth=0.8)
    gpd.GeoSeries([g], crs=CRS_UTM).plot(ax=ax, facecolor="none",
                                         edgecolor="crimson", linewidth=1.6)
    ax.set_title(f"{label}\narea = {g.area/1e6:,.1f} km^2", fontsize=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Constructive Shapely operations (red = result)", fontsize=12, weight="bold")
plt.tight_layout(); plt.show()

# --- 2. Simplification: the accuracy/size trade-off -------------------------
print("Douglas-Peucker simplification of one district polygon")
print(f"{'tolerance (m)':>14} {'vertices':>9} {'area (km^2)':>12} {'area err %':>11}")
base_n, base_a = shapely.count_coordinates(d), d.area
for tol in [0, 25, 100, 250, 500, 1000, 2500]:
    s = d.simplify(tol)
    print(f"{tol:>14} {shapely.count_coordinates(s):>9} {s.area/1e6:>12,.3f} "
          f"{100*(s.area-base_a)/base_a:>10.3f}%")

# --- 3. Predicates -----------------------------------------------------------
# Pick the pair PROGRAMMATICALLY rather than by name: A, and the first district
# that genuinely shares a boundary with it. Hard-coding names is exactly how
# demonstrations like this silently stop demonstrating anything.
A_name = "Harbourgate"
A_row = districts.loc[districts["name"] == A_name].iloc[0]
a = A_row.geometry

neighbours = districts[districts.geometry.touches(a)]
B_row = neighbours.iloc[0]
b = B_row.geometry
print(f"\nA = {A_name}   B = {B_row['name']}  "
      f"({len(neighbours)} districts share a boundary with A)")

# a river that actually crosses A, and a facility that is actually inside A
riv = rivers.loc[rivers.geometry.intersects(a)].geometry.iloc[0]
inside = facilities[facilities.geometry.within(a)]
hosp = inside.geometry.iloc[0]
outside = facilities[~facilities.geometry.within(a)].geometry.iloc[0]

print("\nPredicate matrix   (A = district, B = neighbouring district)")
print(f"{'relation':<14} {'A ~ B':>10} {'A ~ river':>11} {'A ~ inside pt':>14} {'A ~ outside pt':>15}")
for name in ["intersects", "contains", "within", "touches", "crosses",
             "overlaps", "disjoint", "covers"]:
    print(f"{name:<14} {str(getattr(a, name)(b)):>10} {str(getattr(a, name)(riv)):>11} "
          f"{str(getattr(a, name)(hosp)):>14} {str(getattr(a, name)(outside)):>15}")

print(f"\nArea of A                      : {a.area/1e6:,.2f} km^2")
print(f"Shared boundary length A & B   : {a.intersection(b).length:,.0f} m "
      f"(dimension: {a.intersection(b).geom_type})")
print(f"Area of that shared boundary   : {a.intersection(b).area:,.1f} m^2 "
      f"(zero - it is a LINE, not an overlap)")
print(f"Distance A -> the outside point: {a.distance(outside):,.0f} m")
print(f"Distance A -> the inside point : {a.distance(hosp):,.0f} m  "
      f"(zero, because it is inside)")
'''),

md(r'''
**Explanation.**

* `d.buffer(1500)` — a **positive** buffer dilates the polygon by 1 500 m in every
  direction. `d.buffer(-1000)` **erodes** it; erode a thin polygon by more than
  half its width and you get an **empty geometry**, which is a silent way to lose
  features. Always check `.is_empty` after a negative buffer.
* Buffers are approximated by polygons: `buffer(d, quad_segs=8)` uses 8 segments
  per quarter circle by default (32-gon). Raise it for smooth cartography, lower
  it for speed on millions of features.
* `simplify(tol)` uses **Douglas–Peucker**: it drops any vertex lying within
  `tol` of the line joining its neighbours. Note in the table that 250 m
  tolerance typically removes ~90% of vertices while changing the area by well
  under 0.5%. That is the trade you make before publishing web maps or before an
  expensive overlay. Use `simplify(tol, preserve_topology=True)` (the default) to
  avoid creating self-intersections; `preserve_topology=False` is faster but can
  produce invalid output.
* **Simplification does not preserve shared borders.** Simplifying two adjacent
  districts independently opens slivers and gaps between them. For an entire
  administrative layer use `topojson`-style tools, not per-feature `simplify`.
* The predicate matrix shows the key relationships: two adjacent districts
  `intersects` **and** `touches` (they share an edge but no interior), and
  therefore `overlaps` is `False`. A hospital inside a district gives
  `contains=True`, `intersects=True`, `touches=False`.
* `a.intersection(b).length` — for two polygons that only touch, the
  intersection is a LineString (their shared border) with zero area but positive
  length. Recognising the *dimension* of an intersection result is essential:
  polygon ∩ polygon can return a Polygon, a LineString, a Point, a
  GeometryCollection, **or** an empty geometry.

**Expected output.**

* Six panels. Harbourgate is 67.69 km²; the +1 500 m buffer grows it to ~113 km²,
  the −1 000 m buffer shrinks it to ~40 km², the **convex hull is 92.26 km²** —
  36% larger than the district, which is the quantitative statement that the
  boundary is genuinely concave. The envelope is a rectangle; `simplify(500)`
  looks almost identical but with visibly straighter edges.
* A simplification table close to:

| tolerance (m) | vertices | area km² | area err |
|---|---|---|---|
| 0 | 48 | 67.687 | 0.000% |
| 25 | 47 | 67.687 | −0.001% |
| 100 | 40 | 67.628 | −0.087% |
| 250 | 33 | 67.660 | −0.041% |
| 500 | 27 | 68.220 | **+0.787%** |
| 1000 | 13 | 69.653 | +2.903% |
| 2500 | 5 | 64.810 | −4.251% |

  **Note the error is not monotone and changes sign.** Douglas–Peucker removes
  vertices, not area: dropping a vertex on a concave stretch *adds* area while
  dropping one on a convex stretch removes it. Never assume "a bit of
  simplification just shrinks things slightly".

* `A = Harbourgate   B = Old Vallmara  (5 districts share a boundary with A)` and
  a predicate matrix:

```
relation            A ~ B   A ~ river  A ~ inside pt  A ~ outside pt
intersects           True        True           True           False
contains            False       False           True           False
within              False       False          False           False
touches              True       False          False           False
crosses             False        True          False           False
overlaps            False       False          False           False
disjoint            False       False          False            True
covers              False       False           True           False
```

  Read it row by row. Two adjacent districts `intersect` **and** `touch` but do
  not `overlap` — they share a boundary and no interior. The river `crosses` the
  district (a line passing through a polygon). A point inside is `contained`
  **and** `covered`. Everything about the outside point is `False` except
  `disjoint`.

* Finally: `Shared boundary length A & B : 16,581 m (dimension: MultiLineString)`
  with **area 0.0 m²**. Polygon ∩ polygon returned a *line*, and it came back as a
  **Multi**LineString because the two districts meet along several separate
  stretches. Blindly calling `.area` on an intersection result — assuming it must
  be a polygon — is a very common bug.
'''),

# ------------------------------------------------------------------ B9 -----
md(r'''
## B9 — Filtering and selecting spatial features

**What we are going to learn.** The four ways to select a subset of features, and
when each is appropriate.

**Why it matters.** Selection is where performance is won or lost. A bounding-box
filter that runs in microseconds can replace an exact geometric test that takes
minutes — and on 5 200 buildings against 181 flood polygons the difference is
already noticeable.

**The concept — four selection mechanisms, cheapest first.**

1. **Attribute filter** — pure pandas: `gdf[gdf.road_class == "motorway"]`.
   No geometry touched. Always do this first to shrink the problem.
2. **Coordinate/bounding-box filter** — `gdf.cx[xmin:xmax, ymin:ymax]`. The `.cx`
   indexer selects features whose *bounding box* intersects the given box. Very
   fast (uses the spatial index) but **approximate** — a diagonal river's bounding
   box covers a huge area it never enters.
3. **Spatial predicate filter** — `gdf[gdf.intersects(some_geom)]`. Exact, and
   the most common form. Internally still uses the spatial index for the
   bounding-box pre-filter, then does exact tests on the survivors.
4. **Read-time filter** — `gpd.read_file(path, bbox=..., where="...")`. The
   cheapest of all: filtering happens in GDAL before the data ever enters Python.
   Essential for files larger than memory.

**Concept — the spatial index.** GeoPandas maintains an **R-tree** (`gdf.sindex`)
over the bounding boxes. Every query is two-phase: (a) *filter* — the R-tree
returns candidate bounding boxes in `O(log n)`; (b) *refine* — exact GEOS
predicates run only on the candidates. Understanding this two-phase pattern
explains why cheap attribute filtering first makes exact tests fast.

**Expected outcome.** The same "buildings in the flood zone" question answered
four ways, with timings, plus a visual comparison of bbox vs exact selection.

**What the next cell does:** demonstrates attribute selection, `.cx` box
selection, exact predicate selection and read-time filtering, times each, and
plots the difference between the approximate and exact answers.
'''),

code(r'''
import time

flood = gpd.read_file(GPKG, layer="flood_zones")
flood100 = flood[flood["return_period_yr"] == 100]
zone_100 = flood100.geometry.union_all()      # one big MultiPolygon

# --- 1. ATTRIBUTE filter ------------------------------------------------------
motorways = roads[roads["road_class"] == "motorway"]
old_res   = buildings[(buildings["year_built"] < 1950) & (buildings["floors"] >= 3)]
print(f"1. Attribute filter")
print(f"   motorway segments            : {len(motorways):>6}")
print(f"   pre-1950 buildings, 3+ floors: {len(old_res):>6}")

# --- 2. BOUNDING-BOX filter with .cx -----------------------------------------
xmin, ymin, xmax, ymax = 410_000, 4_612_000, 424_000, 4_624_000
t0 = time.perf_counter()
in_box = buildings.cx[xmin:xmax, ymin:ymax]
t_box = time.perf_counter() - t0
print(f"\n2. Bounding-box filter (.cx)   : {len(in_box):>6} buildings "
      f"in {t_box*1000:.1f} ms")

# --- 3. EXACT spatial predicate ----------------------------------------------
t0 = time.perf_counter()
flooded_bbox = buildings[buildings.geometry.intersects(zone_100.envelope)]
t_env = time.perf_counter() - t0

t0 = time.perf_counter()
flooded_exact = buildings[buildings.geometry.intersects(zone_100)]
t_exact = time.perf_counter() - t0

print(f"\n3. Exact predicate filter")
print(f"   intersects(bounding box of zone): {len(flooded_bbox):>6} "
      f"in {t_env*1000:6.1f} ms   <- APPROXIMATE, over-counts")
print(f"   intersects(actual zone)         : {len(flooded_exact):>6} "
      f"in {t_exact*1000:6.1f} ms   <- CORRECT")
print(f"   over-count from using the bbox  : "
      f"{len(flooded_bbox) - len(flooded_exact)} buildings "
      f"({100*(len(flooded_bbox)-len(flooded_exact))/len(flooded_exact):.0f}% too many)")

# --- 4. READ-TIME filter (never loads the rest into memory) -----------------
t0 = time.perf_counter()
sub = gpd.read_file(GPKG, layer="buildings",
                    bbox=(xmin, ymin, xmax, ymax))
t_read = time.perf_counter() - t0
print(f"\n4. Read-time bbox filter        : {len(sub):>6} rows loaded "
      f"in {t_read*1000:.1f} ms (never touched the other "
      f"{len(buildings)-len(sub):,} rows)")

# --- 5. Visual comparison -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, sel, title in [
        (axes[0], flooded_bbox, f"intersects(ENVELOPE)  n={len(flooded_bbox):,}"),
        (axes[1], flooded_exact, f"intersects(ZONE)      n={len(flooded_exact):,}")]:
    land.plot(ax=ax, facecolor="#f7f5ef", edgecolor="#d8d2c4", linewidth=0.5)
    flood100.plot(ax=ax, facecolor="#9ecae1", edgecolor="none", alpha=0.75)
    buildings.plot(ax=ax, color="#cccccc", markersize=0.4, linewidth=0)
    sel.plot(ax=ax, color="crimson", markersize=1.2, linewidth=0)
    ax.set_title(title, fontsize=10, weight="bold", loc="left")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Bounding-box selection over-counts; exact predicates do not",
             fontsize=11)
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* `roads[roads["road_class"] == "motorway"]` — an ordinary boolean mask. The
  result is still a GeoDataFrame with its CRS intact. Every pandas filtering
  idiom (`.query()`, `.isin()`, `.between()`) works.
* `buildings.cx[xmin:xmax, ymin:ymax]` — the **coordinate indexer**. Slice syntax,
  `xmin:xmax` first then `ymin:ymax`. An open end means unbounded:
  `gdf.cx[:, 4_620_000:]` selects everything north of that line. Remember it tests
  **bounding boxes**, not geometries.
* `zone_100.envelope` versus `zone_100` — the whole point of the third block. The
  100-year flood zone is a set of long, thin, branching ribbons following the
  rivers; its bounding box covers most of the basin. Selecting by envelope returns
  roughly **4–5× too many buildings**. This is exactly what `.cx` does, so `.cx`
  is a *pre-filter*, never a final answer.
* `gpd.read_file(..., bbox=...)` pushes the filter down into GDAL, which uses the
  GeoPackage's own R-tree index. On a 5 GB national buildings layer this is the
  difference between a working script and an out-of-memory crash. You can also
  pass `where="road_class = 'motorway'"` — an SQL attribute filter evaluated by
  the driver.
* Note the timings: the exact predicate is only slightly slower than the envelope
  test, because both are dominated by the same R-tree lookup. **Correctness here
  is nearly free** — there is no excuse for using the approximate answer.

**Expected output.**

```
1. Attribute filter
   motorway segments            :     38
   pre-1950 buildings, 3+ floors:    359

2. Bounding-box filter (.cx)   :   3880 buildings in ~3 ms

3. Exact predicate filter
   intersects(bounding box of zone):   5198 in    ~3 ms   <- APPROXIMATE, over-counts
   intersects(actual zone)         :   1036 in  ~540 ms   <- CORRECT
   over-count from using the bbox  : 4162 buildings (402% too many)

4. Read-time bbox filter        :   3880 rows loaded in ~90 ms
```

The envelope of the 100-year flood zone contains **5 198 of the 5 200 buildings**
— it is, for practical purposes, the whole basin. Only **1 036** buildings
actually touch the floodplain. Note also that the exact test costs ~540 ms versus
~3 ms: correctness is not always free, but half a second to avoid a 400% error is
the easiest trade you will ever make.

The figure shows the same map twice: on the left almost every building is red;
on the right only the buildings actually lying on the blue floodplain ribbons
are red. **The left panel is what you get if you trust a bounding box.**
'''),

# ----------------------------------------------------------------- B10 -----
md(r'''
## B10 — Calculating distances

**What we are going to learn.** Point-to-point, point-to-line and
one-to-many distance computation, and the difference between planar and
geodesic distance.

**Why it matters.** "Distance to the nearest X" is *the* workhorse feature of
spatial data science. Distance to the coast, to a road, to a hospital, to the
nearest flood zone — these are the columns that make spatial ML models work.
Getting them right (right CRS, right geometry, right definition) is most of the
battle.

**The concept — what "distance" means.**

* **Euclidean / planar distance** — straight-line distance in the projected
  plane. `geom_a.distance(geom_b)` returns the **minimum** distance between the
  two geometries (0 if they touch or overlap). This is what GeoPandas computes,
  and it is correct *in a suitable projected CRS over a regional extent*.
* **Geodesic distance** — the true distance over the ellipsoid. Necessary for
  continental or intercontinental distances, or when working in EPSG:4326.
  Use `pyproj.Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)`.
* **Network distance** — distance *along a road network*. Almost always the
  honest answer for accessibility, and always larger than Euclidean. The ratio
  (network / Euclidean), the **detour index**, is typically 1.2–1.5 in a city.
  We use Euclidean throughout for tractability and flag it as an assumption.

**Concept — the three distance shapes.**

| Question | Tool | Cost |
|---|---|---|
| A to B | `a.distance(b)` | O(1) |
| Every A to one B | `gdf.distance(b)` (vectorised) | O(n) |
| Every A to its nearest B | `gpd.sjoin_nearest` or `scipy.spatial.cKDTree` | O(n log m) |
| Every A to every B | `cKDTree` / broadcasting | O(n·m) — beware |

**Expected outcome.** Distance-to-hospital for every district centroid,
distance-to-motorway for every sensor, and a demonstration that planar and
geodesic distances agree to ~0.1% at this scale.

**What the next cell does:** computes three kinds of distance, compares planar
UTM distance against true geodesic distance, and shows why a full pairwise
matrix is only viable for small n.
'''),

code(r'''
from scipy.spatial import cKDTree
from pyproj import Geod

hospitals = facilities[facilities["facility_type"] == "hospital"].reset_index(drop=True)
centroids = districts.copy()
centroids["geometry"] = districts.geometry.representative_point()

# --- 1. One-to-one -----------------------------------------------------------
a = centroids.geometry.iloc[0]
b = hospitals.geometry.iloc[0]
print(f"1. One-to-one: {districts.loc[0,'name']} centroid -> {hospitals.loc[0,'name']}")
print(f"   planar (UTM)  : {a.distance(b)/1000:,.3f} km")

geod = Geod(ellps="WGS84")
a_ll = gpd.GeoSeries([a], crs=CRS_UTM).to_crs(CRS_WGS84).iloc[0]
b_ll = gpd.GeoSeries([b], crs=CRS_UTM).to_crs(CRS_WGS84).iloc[0]
_, _, geo_m = geod.inv(a_ll.x, a_ll.y, b_ll.x, b_ll.y)
print(f"   geodesic      : {geo_m/1000:,.3f} km")
print(f"   difference    : {abs(geo_m - a.distance(b)):,.1f} m "
      f"({100*abs(geo_m-a.distance(b))/geo_m:.4f} %)  <- UTM is fine at this scale")

# --- 2. One-to-many: vectorised distance to a single geometry ---------------
motorway_geom = roads[roads.road_class == "motorway"].geometry.union_all()
stations["dist_motorway_m"] = stations.geometry.distance(motorway_geom)
print(f"\n2. One-to-many: every station -> the motorway (a MultiLineString)")
print(stations[["station_id", "station_type", "dist_motorway_m"]]
      .sort_values("dist_motorway_m").head(5).to_string(index=False))

# --- 3. Many-to-nearest with a KD-tree --------------------------------------
tree = cKDTree(np.c_[hospitals.geometry.x, hospitals.geometry.y])
dist, idx = tree.query(np.c_[centroids.geometry.x, centroids.geometry.y], k=1)
centroids["nearest_hospital"] = hospitals.loc[idx, "name"].to_numpy()
centroids["hosp_dist_km"] = dist / 1000
print(f"\n3. Many-to-nearest (KD-tree over {len(hospitals)} hospitals)")
print(centroids[["name", "district_type", "nearest_hospital", "hosp_dist_km"]]
      .sort_values("hosp_dist_km", ascending=False).head(6).to_string(index=False))

# --- 4. Full pairwise matrix (only because n is tiny) -----------------------
D = np.hypot(
    centroids.geometry.x.values[:, None] - hospitals.geometry.x.values[None, :],
    centroids.geometry.y.values[:, None] - hospitals.geometry.y.values[None, :],
) / 1000
print(f"\n4. Full pairwise matrix: {D.shape[0]} districts x {D.shape[1]} hospitals "
      f"= {D.size} distances")
print(pd.DataFrame(D, index=districts["name"], columns=hospitals["facility_id"])
      .round(1).head(6).to_string())
print(f"\n   Memory for this matrix        : {D.nbytes/1024:.1f} KB")
print(f"   Same approach for 5,200 buildings x 83 facilities: "
      f"{5200*83*8/1e6:.1f} MB   (fine)")
print(f"   Same approach for 1M x 100k points               : "
      f"{1e6*1e5*8/1e9:,.0f} GB  (use a KD-tree instead)")
'''),

md(r'''
**Explanation.**

* `districts.geometry.representative_point()` rather than `.centroid` — for a
  concave district the centroid can lie outside the polygon (in the sea, or in a
  neighbouring district). `representative_point()` guarantees a point inside.
  For distance-to-service analysis, a **population-weighted** centroid would be
  better still; we build one in Module 3.
* `a.distance(b)` is the minimum separation. For a Point and a MultiLineString it
  is the perpendicular distance to the closest segment — exactly what
  "distance to the road network" should mean.
* `stations.geometry.distance(motorway_geom)` — a `GeoSeries` against a **single**
  Shapely geometry broadcasts, giving a Series of distances. If you pass two
  GeoSeries of equal length instead, GeoPandas pairs them **row by row**
  (element-wise), which is a completely different operation. Know which you want.
* `union_all()` before measuring is important: without it you would have to take
  a min over 38 separate motorway segments.
* **`cKDTree`** builds a k-d tree in `O(m log m)` and answers each nearest-neighbour
  query in `O(log m)`. It works on **planar coordinates only**, which is another
  reason to be in a projected CRS. Note `k=1` returns `(distances, indices)`;
  ask for `k=3` to get the three nearest, which is how you build "distance to 2nd
  nearest hospital" features (a useful redundancy measure).
* The pairwise matrix block is a warning about scale. A full matrix is `n × m × 8`
  bytes. It is the right tool for 24 × 4; it is catastrophic for 10⁶ × 10⁵.
* The planar/geodesic comparison shows an error of order **0.01–0.1%** over
  20 km in UTM. That is the empirical justification for using a projected CRS and
  ordinary Euclidean geometry at regional scale.

**Expected output.**

* Planar `2.241 km` versus geodesic `2.241 km` — a difference of **0.7 m, or
  0.03%**. That is the empirical licence to use flat Euclidean geometry in UTM at
  this scale.
* The five stations closest to the motorway, from **438 m** (ST018) to ~2.5 km.
* A ranked table of districts by distance to the nearest hospital: **Ostrand
  31.0 km, Stonebeck 26.4 km, Halvorn 26.3 km, Willowmere 26.1 km** — all
  `upland_rural` — versus about 2 km for the urban core. Note too that *every*
  remote district's nearest hospital is the same one (Harbourgate General):
  four hospitals serve 600 000 people from a single cluster. **This is the
  accessibility inequality that Module 3 quantifies properly.**
* A 24 × 4 distance matrix (0.8 KB), and the scaling warning: the same approach
  on 1M × 100k points would need **800 GB**.
'''),

# ----------------------------------------------------------------- B11 -----
md(r'''
## B11 — Your first spatial join

**What we are going to learn.** `gpd.sjoin` — joining two layers on a *spatial
relationship* rather than a key.

**Why it matters.** This is the operation that makes spatial data science
different from ordinary data science. Instead of `df.merge(other, on="id")`, you
ask "which polygon is this point in?" and let geometry supply the key. It is how
you attach context (district, land use, hazard zone) to observations.

**The concept.**

```python
gpd.sjoin(left_gdf, right_gdf, how="inner", predicate="intersects")
```

* The result has **one row per matching pair**. If a point falls in two
  overlapping polygons you get **two rows** — silent row multiplication is the
  number-one spatial-join bug.
* `how="left"` keeps every left row (unmatched ones get NaN attributes);
  `how="inner"` keeps only matches; `how="right"` mirrors left.
* `predicate=` accepts any DE-9IM relation: `intersects` (default), `within`,
  `contains`, `touches`, `crosses`, `overlaps`, `covers`, `covered_by`, `dwithin`.
* The right layer's index arrives as `index_right`.
* **Both layers must be in the same CRS** — GeoPandas raises if they are not.

**Which predicate for points in polygons?** Use `within` (or its inverse
`contains`). `intersects` also matches a point lying exactly *on* a shared border,
which will match **both** neighbouring polygons and duplicate the row. For
floating-point coordinates this is rare but not impossible — and with data
snapped to a grid it is common.

**Expected outcome.** Every facility labelled with its district, a
count-of-points-per-polygon table, and a demonstration of the row-multiplication
trap.

**What the next cell does:** joins facilities to districts, aggregates counts per
district, joins the counts back onto the polygon layer for mapping, and then
deliberately triggers the duplication problem using overlapping flood zones.
'''),

code(r'''
# --- 1. Points-in-polygons ---------------------------------------------------
fac_d = gpd.sjoin(
    facilities,                                     # left  = points
    districts[["district_id", "name", "district_type", "geometry"]],
    how="left", predicate="within",
).rename(columns={"name_right": "district_name", "name_left": "facility_name"})

print(f"facilities in : {len(facilities)} rows")
print(f"after sjoin   : {len(fac_d)} rows   "
      f"({'no duplication' if len(fac_d)==len(facilities) else 'DUPLICATED!'})")
print(f"unmatched     : {fac_d['district_id'].isna().sum()} facilities fell outside every district\n")
print(fac_d[["facility_id", "facility_type", "district_name", "district_type"]]
      .head(6).to_string(index=False))

# --- 2. Aggregate: how many of each facility type per district? -------------
counts = (fac_d.groupby(["district_id", "facility_type"])
                .size().unstack(fill_value=0))
counts["total"] = counts.sum(axis=1)
print("\nFacility counts per district (first 8)")
print(counts.head(8).to_string())

# --- 3. Join the summary BACK onto the polygons so we can map it ------------
dist_stats = districts.merge(
    counts.reset_index(), on="district_id", how="left").fillna({"total": 0})
dist_stats["pop_per_facility"] = np.where(
    dist_stats["total"] > 0, dist_stats["population"] / dist_stats["total"], np.nan)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
dist_stats.plot(ax=axes[0], column="total", cmap="Greens", scheme="naturalbreaks",
                k=5, legend=True, edgecolor="grey", linewidth=0.4,
                legend_kwds={"loc": "lower left", "fontsize": 7, "title": "facilities"})
facilities.plot(ax=axes[0], color="black", markersize=4)
axes[0].set_title("Facility count per district", loc="left", weight="bold", fontsize=10)

dist_stats.plot(ax=axes[1], column="pop_per_facility", cmap="OrRd",
                scheme="quantiles", k=5, legend=True, edgecolor="grey", linewidth=0.4,
                legend_kwds={"loc": "lower left", "fontsize": 7, "title": "people / facility"},
                missing_kwds={"color": "#dddddd", "hatch": "//", "label": "no facility / no pop"})
axes[1].set_title("Residents per facility (higher = worse served)",
                  loc="left", weight="bold", fontsize=10)
for a in axes:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()

# --- 4. THE TRAP: joining to OVERLAPPING polygons --------------------------
print("=" * 78)
print("THE ROW-MULTIPLICATION TRAP")
print("=" * 78)
overlapping = pd.concat([flood100, flood100.assign(zone_id=flood100.zone_id + "_dup")])
overlapping = gpd.GeoDataFrame(overlapping, geometry="geometry", crs=CRS_UTM)

j = gpd.sjoin(buildings[["building_id", "geometry"]], overlapping[["zone_id", "geometry"]],
              how="inner", predicate="intersects")
print(f"buildings joined to 2 identical zone layers : {len(j):,} rows")
print(f"distinct buildings involved                  : {j['building_id'].nunique():,}")
print(f"inflation factor                             : {len(j)/j['building_id'].nunique():.2f}x")
print("\nIf you now sum building values you will DOUBLE-COUNT every exposed asset.")
print("Fix: aggregate first, or de-duplicate:")
print(f"  j.drop_duplicates('building_id')  ->  {len(j.drop_duplicates('building_id')):,} rows")
'''),

md(r'''
**Explanation.**

* `gpd.sjoin(facilities, districts, how="left", predicate="within")` — for each
  facility, find the district polygon that **contains** it. `how="left"` keeps all
  83 facilities so we can *count* the unmatched ones instead of losing them
  silently. Here 0 are unmatched because the districts tile the land exactly.
* Column-name collisions: both layers have a `name` column, so GeoPandas suffixes
  them `name_left` / `name_right`. Renaming immediately is good hygiene —
  `name_right` becomes meaningless three cells later.
* `.groupby([...]).size().unstack(fill_value=0)` — the standard pandas
  cross-tabulation. `fill_value=0` matters: a district with no hospital should
  show **0**, not NaN, otherwise later arithmetic propagates NaN.
* **Joining the summary back**: `districts.merge(counts, on="district_id")`.
  Note this is an *attribute* merge, not a spatial one — once the spatial join has
  produced a key, ordinary pandas takes over. Keep the polygon layer on the left
  so the result stays a GeoDataFrame with geometry.
* `np.where(total > 0, pop/total, np.nan)` — guarding against division by zero.
  Districts with no facility get NaN and are hatched on the map rather than
  appearing as `inf`.
* **The trap block** builds a layer with duplicate overlapping polygons. Every
  building now matches twice, so `len(j)` is double `nunique()`. In the real world
  this happens whenever your right-hand layer has overlaps: overlapping hazard
  zones, administrative boundaries with slivers, or buffers around multiple
  facilities. **Always check `len(result)` against `len(left)` after a join.**

**Expected output.**

* `facilities in : 83 rows`, `after sjoin : 83 rows (no duplication)`,
  `unmatched : 0`.
* A facility-count matrix with columns `clinic, fire_station, hospital,
  police_station, school, total`. **D01 has 24 facilities and D02 has 23, while
  D03 has 1 and several districts do not appear at all** — because
  `groupby` only creates rows for districts that matched. Districts with zero
  facilities are *missing from the index*, not zero in it. That is why step 3
  merges back onto the full polygon layer and fills with 0.
* Two maps: the left shows facility counts concentrated in the populous
  central districts; the right shows *residents per facility*, where the
  urban districts look **worse** (thousands of residents per facility) than
  the empty upland ones — a genuine analytical result and a warning that
  raw counts and per-capita rates tell opposite stories.
* The trap block prints `2,072 rows / 1,036 distinct buildings / inflation 2.00x`.
'''),

]

CELLS += [

# ----------------------------------------------------------------- B12 -----
md(r'''
## B12 — Opening a raster with Rasterio

**What we are going to learn.** The anatomy of a raster dataset: profile,
transform, CRS, NoData, bands, dtype and windows.

**Why it matters.** Rasters carry the *continuous* variables — elevation,
rainfall, temperature, reflectance, density. Half of environmental data science
is raster work, and every raster bug traces back to one of four things: the
affine transform, the NoData value, the dtype, or the CRS.

**The concept — a raster is an array plus an affine transform.**

A raster is a regular grid of cells. To place it on the Earth you need six
numbers, the **affine transform**:

```
| x |   | a  b  c | | col |          x = a*col + b*row + c
| y | = | d  e  f | | row |          y = d*col + e*row + f
| 1 |   | 0  0  1 | |  1  |
```

For a standard north-up raster: `a` = cell width, `e` = **negative** cell height
(because rows increase downward while y increases upward), `b = d = 0`, and
`(c, f)` = the coordinates of the **upper-left corner of the upper-left cell**.

**The rasterio object model.**

| Attribute | Meaning |
|---|---|
| `src.width`, `src.height` | Columns, rows |
| `src.count` | Number of bands |
| `src.dtypes` | Per-band NumPy dtype |
| `src.crs` | Coordinate reference system |
| `src.transform` | The affine transform above |
| `src.bounds` | `(left, bottom, right, top)` in CRS units |
| `src.res` | `(x_size, y_size)` per cell |
| `src.nodata` | The sentinel value meaning "no measurement" |
| `src.profile` | A dict of all of the above — pass it to `rasterio.open(..., **profile)` to write a matching file |

**NoData is the number-one raster trap.** `src.read(1)` returns the raw array
*including* the NoData sentinel (here **−9999**). Compute a mean on that and your
average elevation becomes hugely negative. Two correct approaches:

* `src.read(1, masked=True)` → a `numpy.ma.MaskedArray` that excludes NoData from
  every reduction automatically.
* Read raw, then `arr = np.where(arr == src.nodata, np.nan, arr)` and use
  `np.nanmean` etc. (requires a float dtype).

**Expected outcome.** Full metadata for all seven rasters and a demonstration of
the NoData trap with real numbers.

**What the next cell does:** prints a metadata table for every GeoTIFF in the
dataset, then opens the DEM and computes statistics the wrong way and the right
way so you can see the size of the error.
'''),

code(r'''
import rasterio
from rasterio.plot import show as rshow

# --- 1. Metadata for every raster we ship ------------------------------------
rows = []
for tif in sorted(RAS.glob("*.tif")):
    with rasterio.open(tif) as src:
        rows.append({
            "file": tif.name,
            "size": f"{src.width} x {src.height}",
            "bands": src.count,
            "dtype": src.dtypes[0],
            "res (m)": f"{src.res[0]:.0f}",
            "crs": src.crs.to_string(),
            "nodata": src.nodata,
            "MB": round(tif.stat().st_size / 1e6, 2),
        })
print("RASTER INVENTORY")
print(pd.DataFrame(rows).to_string(index=False))

# --- 2. Dissect the DEM ------------------------------------------------------
dem_path = RAS / "dem_25m.tif"
with rasterio.open(dem_path) as src:
    print("\n" + "=" * 78)
    print("dem_25m.tif")
    print("=" * 78)
    print(f"  shape (rows, cols) : {src.height} x {src.width} = {src.height*src.width:,} cells")
    print(f"  bands              : {src.count}   descriptions: {src.descriptions}")
    print(f"  dtype              : {src.dtypes[0]}")
    print(f"  crs                : {src.crs}")
    print(f"  resolution         : {src.res} (metres per cell)")
    print(f"  bounds             : {tuple(round(b) for b in src.bounds)}")
    print(f"  nodata             : {src.nodata}")
    print(f"\n  affine transform:\n{src.transform}")
    print(f"\n  upper-left corner  : ({src.transform.c:,.0f}, {src.transform.f:,.0f})")
    print(f"  pixel width  (a)   : {src.transform.a}")
    print(f"  pixel height (e)   : {src.transform.e}   <- NEGATIVE: rows go DOWN, y goes UP")

    raw     = src.read(1)                  # raw array, NoData included
    masked  = src.read(1, masked=True)     # MaskedArray, NoData excluded
    nodata  = src.nodata

# --- 3. The NoData trap, quantified -----------------------------------------
print("\n" + "=" * 78)
print("THE NODATA TRAP")
print("=" * 78)
n_nodata = int((raw == nodata).sum())
print(f"  cells total            : {raw.size:,}")
print(f"  cells that are NoData  : {n_nodata:,}  ({100*n_nodata/raw.size:.1f} % - the sea)")
print()
print(f"  WRONG  raw.mean()          = {raw.mean():>12,.2f} m   <- nonsense")
print(f"  WRONG  raw.min()           = {raw.min():>12,.2f} m   <- the sentinel itself")
print(f"  RIGHT  masked.mean()       = {masked.mean():>12,.2f} m")
print(f"  RIGHT  masked.min()/max()  = {masked.min():>8,.2f} / {masked.max():,.2f} m")

nan_arr = np.where(raw == nodata, np.nan, raw)
print(f"  RIGHT  np.nanmean(...)     = {np.nanmean(nan_arr):>12,.2f} m   (the NaN idiom)")
print(f"\n  Error from ignoring NoData : {raw.mean() - masked.mean():,.0f} m")
'''),

md(r'''
**Explanation.**

* `rasterio.open(path)` returns a lazy **dataset reader**. Opening reads only the
  header; nothing is loaded until you call `.read()`. Always use it as a context
  manager (`with ... as src:`) so the file handle closes — GDAL keeps a cache per
  open dataset and leaking handles is a real problem in loops.
* `src.profile` is the complete creation recipe. The write idiom is:
  `profile = src.profile; profile.update(dtype="float32", count=1)` then
  `rasterio.open(out, "w", **profile)`. You will use this in B14.
* `src.transform.e` is **negative** (−25.0). This is not a quirk: image row
  indices increase downwards, map y-coordinates increase upwards. A positive `e`
  means a south-up raster, which will render upside down.
* `src.descriptions` returns the per-band names we set when generating the data
  — a much better practice than remembering that "band 3 is red".
* `masked=True` returns `numpy.ma.MaskedArray`. All reductions (`mean`, `std`,
  `min`) skip masked cells. The mask itself is `masked.mask` (True = NoData) and
  the raw values are `masked.data`.
* **The numbers are the lesson.** 19% of the DEM is sea, stored as −9999. The
  naive mean is about **−1 650 m**; the correct mean is about **+340 m**. The
  error is roughly 2 000 m, and nothing warns you. Everything downstream — slope,
  hillshade, zonal statistics, a regression on elevation — inherits it.
* `np.where(raw == nodata, np.nan, raw)` requires a float dtype. For integer
  rasters (like `landcover_25m.tif`, uint8) NaN is impossible, so use masked
  arrays or keep an explicit boolean mask.

**Expected output.**

A 7-row raster inventory:

| file | size | bands | dtype | res | crs | nodata |
|---|---|---|---|---|---|---|
| dem_25m.tif | 1920 × 1440 | 1 | float32 | 25 | EPSG:32633 | −9999 |
| landcover_25m.tif | 1920 × 1440 | 1 | uint8 | 25 | EPSG:32633 | 0 |
| lst_summer_100m_3857.tif | 483 × 366 | 1 | float32 | **134** | **EPSG:3857** | −9999 |
| multispectral_50m.tif | 960 × 720 | **4** | uint16 | 50 | EPSG:32633 | 0 |
| ndvi_50m.tif | 960 × 720 | 1 | float32 | 50 | EPSG:32633 | −9999 |
| popdens_100m.tif | 480 × 360 | 1 | float32 | 100 | EPSG:32633 | −9999 |
| rainfall_annual_250m.tif | 192 × 144 | 1 | float32 | 250 | EPSG:32633 | −9999 |

Note the LST raster: its resolution reads **134 m, not 100 m**. It was generated
on a 100 m UTM grid and then reprojected to Web Mercator, and reprojection
resamples onto a *new* grid whose spacing is set by the target CRS. This is
another face of the Web-Mercator distortion — at 41.7° N, one Web-Mercator metre
is only about 0.75 real metres, so a 100 m ground cell becomes a ~134 unit cell.

Then the DEM dissection, `pixel height (e) = -25.0`, and the trap:

```
  cells total            : 2,764,800
  cells that are NoData  : 533,151  (19.3 % - the sea)

  WRONG  raw.mean()          =    -1,654.81 m   <- nonsense
  WRONG  raw.min()           =    -9,999.00 m   <- the sentinel itself
  RIGHT  masked.mean()       =       338.66 m
  RIGHT  masked.min()/max()  =       0.40 / 925.20 m
  RIGHT  np.nanmean(...)     =       338.66 m

  Error from ignoring NoData : -1,993 m
```
'''),

# ----------------------------------------------------------------- B13 -----
md(r'''
## B13 — Raster ↔ array ↔ ground: indexing, sampling and plotting

**What we are going to learn.** How to move between three coordinate spaces —
array indices `(row, col)`, map coordinates `(x, y)`, and geographic
`(lon, lat)` — and how to sample raster values at vector locations.

**Why it matters.** "Attach the elevation / rainfall / land-cover class at each
observation point" is the most common raster-to-vector operation in data science.
It is also where the affine transform stops being an abstraction.

**The concept — three conversions.**

| Direction | Rasterio API |
|---|---|
| index → coordinate | `src.xy(row, col)` (returns the **cell centre**) |
| coordinate → index | `src.index(x, y)` |
| sample values at points | `src.sample([(x, y), ...])` |
| whole-array coordinates | `rasterio.transform.xy(transform, rows, cols)` |

**The plotting trap.** `plt.imshow(array)` labels the axes with row/column
indices, so your raster will not line up with vector layers. Two fixes:

* `rasterio.plot.show(src, ax=ax)` — reads the transform and does it correctly.
* `plt.imshow(arr, extent=rasterio.plot.plotting_extent(src))` — pass the extent
  `(left, right, bottom, top)` explicitly.

**Sampling caveat — the raster value is a cell average.** Sampling at a point
returns the value of the cell that contains it. For a 250 m rainfall grid, that
is the average over 6.25 hectares, not a point measurement. When you need a
smoother estimate, resample bilinearly (Lesson I13) or take a focal mean.

**Expected outcome.** A verified round-trip between index and coordinate space,
elevation and rainfall attached to all 24 sensor stations, and a correctly
georeferenced plot with vector layers on top.

**What the next cell does:** demonstrates index↔coordinate conversion, samples
four rasters at the station points in one pass, validates the sampled elevation
against the elevation column already in the CSV, and produces a properly aligned
DEM map.
'''),

code(r'''
from rasterio.plot import plotting_extent

with rasterio.open(RAS / "dem_25m.tif") as src:
    dem = src.read(1, masked=True)
    dem_transform, dem_crs, dem_extent = src.transform, src.crs, plotting_extent(src)

    # --- 1. index <-> coordinate round trip ---------------------------------
    row, col = 700, 900
    x, y = src.xy(row, col)                 # cell CENTRE in map coordinates
    r2, c2 = src.index(x, y)                # and back again
    print("INDEX <-> COORDINATE")
    print(f"  array index (row={row}, col={col})")
    print(f"    -> map coords  ({x:,.1f}, {y:,.1f})  [EPSG:32633, metres]")
    print(f"    -> back to idx (row={r2}, col={c2})   "
          f"{'round trip OK' if (r2, c2) == (row, col) else 'MISMATCH'}")
    print(f"    -> value       {dem[row, col]:,.1f} m")

    # manual arithmetic, to prove there is no magic
    a, e, c0, f0 = src.transform.a, src.transform.e, src.transform.c, src.transform.f
    print(f"\n  by hand: x = c + (col+0.5)*a = {c0:,.0f} + {col+0.5}*{a} = {c0+(col+0.5)*a:,.1f}")
    print(f"           y = f + (row+0.5)*e = {f0:,.0f} + {row+0.5}*({e}) = {f0+(row+0.5)*e:,.1f}")

# --- 2. Sample several rasters at the station points ------------------------
coords = [(p.x, p.y) for p in stations.geometry]

samples = {}
for name, fn, band in [("elev_m", "dem_25m.tif", 1),
                       ("rain_mm", "rainfall_annual_250m.tif", 1),
                       ("ndvi", "ndvi_50m.tif", 1),
                       ("landcover", "landcover_25m.tif", 1),
                       ("popdens", "popdens_100m.tif", 1)]:
    with rasterio.open(RAS / fn) as src:
        vals = np.array([v[band - 1] for v in src.sample(coords)], dtype="float64")
        vals[vals == src.nodata] = np.nan          # honour NoData!
        samples[name] = vals

sampled = stations[["station_id", "station_type", "elevation_m"]].copy()
for k, v in samples.items():
    sampled[k] = v
legend = pd.read_csv(TAB / "landcover_legend.csv").set_index("class_code")["landuse_class"]
sampled["landcover_label"] = sampled["landcover"].map(legend)

print("\nRASTER VALUES SAMPLED AT THE 24 STATIONS (first 8)")
print(sampled.head(8).to_string(index=False))

# --- 3. Validate against the elevation already stored in the CSV -----------
err = (sampled["elev_m"] - sampled["elevation_m"]).abs()
print(f"\nValidation: |sampled elevation - CSV elevation|  "
      f"max = {err.max():.2f} m, mean = {err.mean():.2f} m")
print("  (small differences are expected: the CSV stored the value at generation time)")

# --- 4. A CORRECTLY georeferenced plot --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

axes[0].imshow(dem, cmap="terrain")
axes[0].set_title("WRONG: plt.imshow(array)\naxes are row/col indices",
                  loc="left", fontsize=10, weight="bold", color="crimson")
axes[0].set_xlabel("column"); axes[0].set_ylabel("row")

im = axes[1].imshow(dem, cmap="terrain", extent=dem_extent)
districts.boundary.plot(ax=axes[1], color="black", linewidth=0.35)
rivers.plot(ax=axes[1], color="#1f6fb4", linewidth=1.0)
stations.plot(ax=axes[1], color="red", markersize=22, edgecolor="white", linewidth=0.5)
axes[1].set_title("RIGHT: extent=plotting_extent(src)\nvectors align perfectly",
                  loc="left", fontsize=10, weight="bold", color="darkgreen")
axes[1].set_aspect("equal"); axes[1].set_xticks([]); axes[1].set_yticks([])
plt.colorbar(im, ax=axes[1], shrink=0.8, label="elevation (m)")
plt.tight_layout(); plt.show()
'''),

md(r'''
**Explanation.**

* `src.xy(row, col)` returns the **centre** of the cell. If you need a corner, use
  `src.xy(row, col, offset="ul")`. Forgetting the half-cell offset introduces a
  systematic half-pixel shift — 12.5 m on this DEM — which matters when you are
  comparing two grids of different resolution.
* `src.index(x, y)` floors to the containing cell. Coordinates outside the raster
  return out-of-range indices **without raising**, so validate before indexing.
* The "by hand" block reproduces `src.xy` from the six affine numbers. Do this
  once and the transform stops being mysterious.
* `src.sample(coords)` is a **generator** yielding one array per point (length =
  band count). It reads only the blocks it needs, so it is efficient even on huge
  rasters. Because it returns tuples, we take `v[band-1]`.
* **`vals[vals == src.nodata] = np.nan`** — `sample()` does *not* apply the mask.
  Points over the sea return −9999 and will quietly become "the lowest elevation
  in your dataset". This one line is the difference between a working feature and
  a poisoned one.
* Mapping `landcover` codes to labels via the legend CSV turns an opaque integer
  into an interpretable category — do this immediately, not at report time.
* The two panels show the plotting trap. The left axes run 0–1920 and 0–1440
  (pixels); the right axes are in metres and the districts, rivers and stations
  overlay exactly. If your vectors ever appear as a tiny cluster in one corner of
  a raster, you forgot `extent=`.

**Expected output.**

```
INDEX <-> COORDINATE
  array index (row=700, col=900)
    -> map coords  (422,512.5, 4,618,487.5)  [EPSG:32633, metres]
    -> back to idx (row=700, col=900)   round trip OK
    -> value       119.7 m

  by hand: x = c + (col+0.5)*a = 400,000 + 900.5*25.0 = 422,512.5
           y = f + (row+0.5)*e = 4,636,000 + 700.5*(-25.0) = 4,618,487.5
```

A sample table beginning with `ST001 ... elev_m 169.2, rain_mm 634.7, ndvi 0.581,
landcover 3.0 -> Cropland, popdens 1793`. Coastal stations show `elev_m` of a few
metres with `Built-up`/`Wetland` cover; upland stations show 400–750 m with
`Forest`/`Shrubland`. Rainfall ranges roughly **480–1 050 mm**, tracking
elevation — the orographic effect built into the data.

Validation error should be **< 0.1 m**. Then two panels: pixel-indexed (wrong)
and metre-indexed with perfectly aligned vectors (right).
'''),

# ----------------------------------------------------------------- B14 -----
md(r'''
## B14 — Writing data out

**What we are going to learn.** How to save vector layers, rasters and tables in
the formats you will actually be asked for.

**Why it matters.** An analysis nobody can open is not an analysis. Format
choice also has real consequences: a shapefile will silently truncate your
carefully named columns to 10 characters and convert your booleans to strings.

**The concept — matching format to purpose.**

| Purpose | Format | Call |
|---|---|---|
| Your own intermediate results | GeoPackage or GeoParquet | `gdf.to_file(p, layer=..., driver="GPKG")` / `gdf.to_parquet(p)` |
| Sharing with a GIS user | GeoPackage | same |
| Web / API | GeoJSON (EPSG:4326!) | `gdf.to_crs(4326).to_file(p, driver="GeoJSON")` |
| A colleague who insists | Shapefile | `gdf.to_file(p, driver="ESRI Shapefile")` — and check the field names |
| A derived raster | GeoTIFF | `rasterio.open(p, "w", **profile)` |
| Non-spatial summary | CSV | `df.to_csv(p, index=False)` |

**Writing a raster — the profile pattern.** Copy the source profile, update what
changed, write. This guarantees the output is georeferenced identically:

```python
with rasterio.open(src_path) as src:
    profile = src.profile
    profile.update(dtype="float32", count=1, nodata=-9999,
                   compress="deflate", tiled=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(new_array.astype("float32"), 1)
```

**Compression is free money.** `compress="deflate"` with `predictor=2` (integers)
or `predictor=3` (floats) routinely halves GeoTIFF size at negligible CPU cost.
`tiled=True` makes windowed reads fast.

**Expected outcome.** Four output files written to `data/outputs/`, plus a
demonstration of what a shapefile does to your column names.

**What the next cell does:** writes a derived GeoPackage layer, a GeoJSON in
EPSG:4326, a CSV summary and a derived GeoTIFF (elevation above the district
mean); then round-trips a layer through a shapefile to show the field-name
truncation.
'''),

code(r'''
# --- 1. GeoPackage (best default) -------------------------------------------
out_gpkg = OUT / "module1_results.gpkg"
dist_stats_out = dist_stats[["district_id", "name", "district_type", "area_km2",
                             "population", "total", "pop_per_facility", "geometry"]]
dist_stats_out = dist_stats_out.rename(columns={"total": "n_facilities"})
dist_stats_out.to_file(out_gpkg, layer="district_facility_stats", driver="GPKG")
print(f"GeoPackage : {out_gpkg.name}  ({out_gpkg.stat().st_size/1024:.0f} KB)")

# --- 2. GeoJSON for the web - ALWAYS reproject to 4326 ----------------------
out_json = OUT / "stations_wgs84.geojson"
stations.to_crs(CRS_WGS84).to_file(out_json, driver="GeoJSON")
print(f"GeoJSON    : {out_json.name}  ({out_json.stat().st_size/1024:.0f} KB)")

# --- 3. Plain CSV summary ----------------------------------------------------
out_csv = OUT / "district_summary.csv"
dist_stats_out.drop(columns="geometry").to_csv(out_csv, index=False)
print(f"CSV        : {out_csv.name}  ({out_csv.stat().st_size/1024:.0f} KB)")

# --- 4. A derived GeoTIFF: elevation relative to the regional mean ---------
out_tif = OUT / "dem_anomaly_25m.tif"
with rasterio.open(RAS / "dem_25m.tif") as src:
    arr = src.read(1, masked=True)
    anomaly = (arr - arr.mean()).astype("float32")
    profile = src.profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999.0,
                   compress="deflate", predictor=3, tiled=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(anomaly.filled(-9999.0), 1)
        dst.set_band_description(1, "Elevation anomaly vs regional mean (m)")
print(f"GeoTIFF    : {out_tif.name}  ({out_tif.stat().st_size/1e6:.2f} MB)")

# verify it reads back correctly
with rasterio.open(out_tif) as chk:
    back = chk.read(1, masked=True)
    print(f"             read back: shape {back.shape}, "
          f"mean {back.mean():.3f} (should be ~0), crs {chk.crs}")

# --- 5. What a SHAPEFILE does to your schema --------------------------------
print("\n" + "=" * 78)
print("SHAPEFILE FIELD-NAME TRUNCATION")
print("=" * 78)
demo = dist_stats_out.rename(columns={
    "district_type": "district_classification_type",
    "pop_per_facility": "population_per_facility_ratio",
    "n_facilities": "number_of_public_facilities",
})
shp_path = OUT / "shapefile_demo" / "districts.shp"
shp_path.parent.mkdir(exist_ok=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    demo.to_file(shp_path, driver="ESRI Shapefile")

back_shp = gpd.read_file(shp_path)
print(pd.DataFrame({"written": demo.columns[:-1],
                    "read back from .shp": list(back_shp.columns[:-1])}).to_string(index=False))
print(f"\nFiles created for ONE shapefile: "
      f"{sorted(p.suffix for p in shp_path.parent.glob('districts.*'))}")

# --- 6. And the modern alternative ------------------------------------------
try:
    out_pq = OUT / "districts.parquet"
    dist_stats_out.to_parquet(out_pq)
    print(f"\nGeoParquet : {out_pq.name}  ({out_pq.stat().st_size/1024:.0f} KB) "
          f"- long column names preserved, ~10x faster to read than GeoJSON")
except ImportError:
    print("\nGeoParquet : skipped (pip install pyarrow to enable it)")
'''),

md(r'''
**Explanation.**

* `to_file(path, layer=..., driver="GPKG")` — writing to an existing GeoPackage
  **adds** a layer; writing the same layer name again replaces it. This makes a
  GeoPackage an excellent project database: one file, many results, versioned by
  layer name.
* `stations.to_crs(CRS_WGS84).to_file(..., driver="GeoJSON")` — the GeoJSON
  specification (RFC 7946) mandates WGS 84 lon/lat. GeoPandas will happily write
  projected coordinates into a GeoJSON, producing a file that every web map will
  misplace. **Always reproject first.**
* The raster block shows the profile pattern. Note `anomaly.filled(-9999.0)`:
  a MaskedArray must be converted back to a plain array *with the sentinel
  re-inserted* before writing, otherwise the mask is lost and NoData cells are
  written as whatever garbage was underneath.
* `predictor=3` is the floating-point predictor; use `predictor=2` for integers
  and omit it for already-compressed data. Combined with `compress="deflate"` it
  typically halves the file.
* `set_band_description` — self-documenting rasters. Anyone opening your GeoTIFF
  in QGIS sees "Elevation anomaly vs regional mean (m)" instead of "Band 1".
* **The shapefile demo is the punchline.** `district_classification_type` becomes
  `district_c`, `population_per_facility_ratio` becomes `populatio`, and
  `number_of_public_facilities` becomes `number_of`. Truncation can even produce
  *collisions*, in which case GDAL appends digits. A shapefile also cannot store
  a true NULL in a numeric field, cannot exceed 2 GB, and needs 4–6 sidecar files
  that must travel together.
* GeoParquet is the modern intermediate format: columnar, compressed, preserves
  dtypes and long names, and reads roughly an order of magnitude faster than
  GeoJSON.

**Expected output.**

```
GeoPackage : module1_results.gpkg  (120 KB)
GeoJSON    : stations_wgs84.geojson  (9 KB)
CSV        : district_summary.csv  (1 KB)
GeoTIFF    : dem_anomaly_25m.tif  (5.20 MB)
             read back: shape (1440, 1920), mean 0.000 (should be ~0), crs EPSG:32633
GeoParquet : districts.parquet  (33 KB)
```

Then the shapefile schema comparison:

| written | read back from `.shp` |
|---|---|
| `district_id` | `district_i` |
| `name` | `name` |
| `district_classification_type` | `district_c` |
| `area_km2` | `area_km2` |
| `population` | `population` |
| `number_of_public_facilities` | `number_of_` |
| `population_per_facility_ratio` | **`populati_1`** |

Look at the last row. `population_per_facility_ratio` truncates to `populati...`,
which **collides** with an earlier field, so GDAL renamed it `populati_1`. Your
column is now called something you never chose, and a script that reads the file
by name breaks. This is not a hypothetical: it is what happens every time
somebody emails you a shapefile.

Finally, the five sidecar files `['.cpg', '.dbf', '.prj', '.shp', '.shx']` — lose
the `.prj` and the CRS is gone; lose the `.dbf` and the attributes are gone.
'''),

# ------------------------------------------------------- MODULE 1 EXERCISES
md(r'''
# Exercises — Module 1 (Beginner)

Attempt these before looking at the **Solutions** section at the end of the
notebook. Each one is answerable with material from B1–B14 alone.

---

### Exercise 1.1 — Layer inventory
**Objective.** Produce a summary table of *every* vector layer available in the
project (all 12 GeoPackage layers plus the two GeoJSON files), with columns:
`source`, `layer`, `n_features`, `geometry_type`, `crs`, `n_columns`, and the
area (km²) or length (km) as appropriate for the geometry type.
All measurements must be in EPSG:32633.

---

### Exercise 1.2 — Attribute audit
**Objective.** Write a reusable function `audit(gdf, name)` that reports, for any
layer: row count, column count, per-column dtype, count and percentage of missing
values, number of unique values, and — for numeric columns — min, median, max.
Run it on `census_blocks` and identify **every** column with a data-quality
problem. State what you think each problem is.

---

### Exercise 1.3 — CRS forensics
**Objective.** For the `protected_areas.geojson` layer:
1. Report its CRS and total area computed *in its native CRS*.
2. Compute the total area in EPSG:32633, EPSG:3857 and ESRI:54009.
3. Report the percentage error of each versus the UTM value.
4. Explain, in two sentences, which value you would publish and why.

---

### Exercise 1.4 — Building geometries from a dirty CSV
**Objective.** Load `flood_incidents.csv` and build a GeoDataFrame in EPSG:32633.
The file contains coordinate errors on purpose. Without hard-coding row numbers,
identify and quarantine:
* rows at (0, 0),
* rows whose lon/lat are swapped,
* rows outside the study region.

Report how many rows you kept, how many you quarantined, and which rule caught
each one. *(Hint: the study area's WGS 84 bounds are approximately
lon 13.6–14.3, lat 41.5–41.9.)*

---

### Exercise 1.5 — Shapely reasoning
**Objective.** For the district named `Marnvik`:
1. What is its area, perimeter and "compactness" (Polsby–Popper score,
   `4πA / P²` — 1.0 is a perfect circle)?
2. Which districts share a boundary with it? (Use `touches`.)
3. Does its centroid lie inside the polygon? Compare `.centroid` with
   `.representative_point()`.
4. By how many square kilometres does a 1 km inward buffer shrink it, and what
   percentage of the original is that?

---

### Exercise 1.6 — Distance profile
**Objective.** For every one of the 24 sensor stations compute the distance to:
the coastline, the nearest river, the nearest primary-or-better road, and the
nearest hospital. Produce a tidy DataFrame and answer: which station is the most
*remote* by the sum of all four distances, and which is the most connected?

---

### Exercise 1.7 — Spatial join and rate calculation
**Objective.** Join `bus_stops` to `census_blocks` and compute, per district:
number of bus stops, total population, and **bus stops per 10 000 residents**.
Map the result with an explicit classification scheme and explicit NaN handling.
Which district type is worst served, and is the raw count or the rate the more
honest statistic to publish?

---

### Exercise 1.8 — Raster sampling
**Objective.** Sample `dem_25m.tif`, `rainfall_annual_250m.tif` and
`ndvi_50m.tif` at the centroid of every census block. Then:
1. Report how many blocks returned NoData for each raster, and why.
2. Fit and report the simple linear regression `rainfall ~ elevation`.
3. Compare the fitted slope with the true generating coefficient (0.62 mm/m,
   stated in the scenario). How close did you get, and why is it not exact?
'''),

]
