#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 generate_data.py
 Synthetic dataset builder for the course:
   "Spatial Data Science: The New Frontier in Analytics"
=============================================================================

Creates a COMPLETELY FICTIONAL but internally consistent spatial database for
the *Vallmara Basin*, Republic of Kestria -- an imaginary coastal region.

Nothing is downloaded. Everything is synthesised from fixed random seeds, so
the dataset is byte-for-byte reproducible on any machine.

Run:
    python generate_data.py            # writes ./data/...
    python generate_data.py --out DIR  # writes DIR/...

Requires: numpy, pandas, geopandas, shapely, rasterio, scipy, pyogrio
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import gaussian_filter, distance_transform_edt
from shapely.geometry import (
    Point, LineString, MultiLineString, Polygon, MultiPolygon, box, mapping,
)
from shapely.ops import unary_union, linemerge, nearest_points
from shapely import voronoi_polygons, set_precision

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION
# -----------------------------------------------------------------------------
SEED = 42
RNG = np.random.default_rng(SEED)

# Projected working CRS: WGS 84 / UTM zone 33N (metres). Chosen because the
# fictional region sits near 13.8 deg E, 41.6 deg N, well inside zone 33.
CRS_UTM = "EPSG:32633"
CRS_WGS84 = "EPSG:4326"       # geographic lon/lat degrees
CRS_WEBMERC = "EPSG:3857"     # web mercator, used for one raster on purpose

# Region envelope in UTM 33N metres (48 km east-west x 36 km north-south)
XMIN, YMIN = 400_000.0, 4_600_000.0
XMAX, YMAX = 448_000.0, 4_636_000.0
WIDTH_M = XMAX - XMIN
HEIGHT_M = YMAX - YMIN

# Raster grids (all share the same envelope, different resolutions on purpose)
RES_DEM = 25.0        # elevation + land cover
RES_NDVI = 50.0       # vegetation index + multispectral
RES_POP = 100.0       # population density
RES_LST = 100.0       # land surface temperature (written in EPSG:3857)
RES_RAIN = 250.0      # annual rainfall (deliberately coarse -> resampling lesson)

NODATA_F = -9999.0
NODATA_U8 = 0

# Urban core of the fictional capital, "Vallmara City"
CORE = (418_000.0, 4_618_000.0)
SUBCORES = [(409_500.0, 4_628_500.0), (431_000.0, 4_610_500.0)]

DISTRICT_NAMES = [
    "Old Vallmara", "Harbourgate", "Kestrel Quay", "Marnvik", "Sundholm",
    "Ardenfeld", "Cliffmoor", "Brannock", "Tarnwell", "Eldrige Park",
    "Norrbank", "Vestmark", "Grenholt", "Ashcombe", "Lyndover",
    "Pellmyra", "Drakestad", "Highfen", "Corran Vale", "Stonebeck",
    "Willowmere", "Fyrdal", "Ostrand", "Halvorn",
]

LANDCOVER_CLASSES = {
    1: "Water",
    2: "Built-up",
    3: "Cropland",
    4: "Grassland",
    5: "Forest",
    6: "Shrubland",
    7: "Bare rock / sparse",
    8: "Wetland",
}


# -----------------------------------------------------------------------------
# 1. SMALL HELPERS
# -----------------------------------------------------------------------------
def grid_shape(res: float) -> tuple[int, int]:
    """Rows, cols for a raster of the given resolution covering the envelope."""
    return int(round(HEIGHT_M / res)), int(round(WIDTH_M / res))


def grid_transform(res: float):
    """Affine transform (north-up) for a raster of the given resolution."""
    return from_origin(XMIN, YMAX, res, res)


def grid_coords(res: float):
    """Return 2-D arrays of cell-centre X and Y coordinates in UTM metres."""
    nrows, ncols = grid_shape(res)
    xs = XMIN + (np.arange(ncols) + 0.5) * res
    ys = YMAX - (np.arange(nrows) + 0.5) * res
    return np.meshgrid(xs, ys)


def smooth_noise(shape, sigma, rng, seed_scale=1.0):
    """Spatially correlated Gaussian noise, normalised to roughly [-1, 1]."""
    raw = rng.standard_normal(shape)
    out = gaussian_filter(raw, sigma=sigma, mode="reflect")
    out = out / (np.abs(out).max() + 1e-12)
    return out * seed_scale


def fbm(shape, rng, octaves=5, base_sigma=48.0, persistence=0.55):
    """Fractional Brownian motion style multi-octave noise -> natural terrain."""
    total = np.zeros(shape, dtype=np.float64)
    amp, sigma = 1.0, base_sigma
    for _ in range(octaves):
        total += amp * smooth_noise(shape, sigma, rng)
        amp *= persistence
        sigma = max(1.0, sigma * 0.45)
    return total / (np.abs(total).max() + 1e-12)


def write_raster(path, array, res, crs, nodata, dtype, descriptions=None,
                 compress="deflate"):
    """Write a single- or multi-band GeoTIFF on the standard region grid."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    count, nrows, ncols = arr.shape
    profile = dict(
        driver="GTiff", height=nrows, width=ncols, count=count,
        dtype=dtype, crs=crs, transform=grid_transform(res), nodata=nodata,
        compress=compress, tiled=True, blockxsize=256, blockysize=256,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype))
        if descriptions:
            for i, d in enumerate(descriptions, start=1):
                dst.set_band_description(i, d)
    return path


def sample_raster_at(array, res, xs, ys, default=np.nan):
    """Nearest-cell sampling of a region-grid array at UTM coordinates."""
    cols = ((np.asarray(xs) - XMIN) / res).astype(int)
    rows = ((YMAX - np.asarray(ys)) / res).astype(int)
    nrows, ncols = array.shape
    ok = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    out = np.full(np.shape(xs), default, dtype=float)
    out[ok] = array[rows[ok], cols[ok]]
    return out


def jitter_line(p0, p1, rng, n=6, amp=350.0):
    """Turn a straight segment into a gently meandering polyline."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, n + 2)[:, None]
    pts = p0 + (p1 - p0) * t
    d = p1 - p0
    L = np.hypot(*d)
    if L < 1e-6:
        return LineString([tuple(p0), tuple(p1)])
    nvec = np.array([-d[1], d[0]]) / L
    offs = rng.normal(0, amp, size=n + 2)
    offs[0] = offs[-1] = 0.0
    offs = gaussian_filter(offs, 1.0, mode="nearest")
    pts = pts + nvec[None, :] * offs[:, None]
    return LineString([tuple(p) for p in pts])

# -----------------------------------------------------------------------------
# 2. PHYSICAL GEOGRAPHY: COASTLINE, TERRAIN, HYDROLOGY, CLIMATE
# -----------------------------------------------------------------------------
def build_coast_and_land(rng):
    """Coastline runs NNW->SSE; the sea lies to the west/south-west of it."""
    ys = np.linspace(YMAX, YMIN, 220)
    t = (YMAX - ys) / HEIGHT_M
    x_coast = (
        403_500.0
        + 11_000.0 * t                       # coast trends east going south
        + 2_100.0 * np.sin(2 * np.pi * t * 2.4)
        + 900.0 * np.sin(2 * np.pi * t * 6.1 + 1.3)
        + gaussian_filter(rng.normal(0, 260, ys.size), 3.0, mode="nearest")
    )
    coast_line = LineString(np.column_stack([x_coast, ys]))

    sea_ring = (
        list(zip(x_coast, ys))
        + [(XMIN - 5_000, YMIN - 5_000), (XMIN - 5_000, YMAX + 5_000)]
    )
    sea = Polygon(sea_ring).buffer(0)
    region = box(XMIN, YMIN, XMAX, YMAX)
    sea = sea.intersection(region)
    land = region.difference(sea)
    if isinstance(land, MultiPolygon):
        land = max(land.geoms, key=lambda g: g.area)
    return coast_line, sea, land


def build_dem(land, rng):
    """Elevation: rises inland from the coast, with two ridges and fBm relief."""
    res = RES_DEM
    nrows, ncols = grid_shape(res)
    X, Y = grid_coords(res)

    # Distance from the coastline, in kilometres (negative offshore)
    land_mask = _rasterise_mask(land, res)
    d_in = distance_transform_edt(land_mask) * res / 1000.0     # inland
    d_out = distance_transform_edt(~land_mask) * res / 1000.0   # offshore
    d_coast = d_in - d_out

    base = 7.0 * np.clip(d_coast, 0, None) ** 1.22

    # Two mountain ridges in the east / north-east
    def ridge(x0, y0, x1, y1, amp, width):
        px, py = X - x0, Y - y0
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        tt = np.clip((px * dx + py * dy) / L2, 0, 1)
        dist = np.hypot(px - tt * dx, py - tt * dy)
        return amp * np.exp(-0.5 * (dist / width) ** 2)

    base += ridge(436_000, 4_634_000, 446_000, 4_612_000, 380.0, 4_200.0)
    base += ridge(427_000, 4_602_000, 444_000, 4_601_000, 240.0, 3_400.0)

    relief = fbm((nrows, ncols), rng, octaves=6, base_sigma=60.0) * 95.0
    relief += smooth_noise((nrows, ncols), 6.0, rng) * 18.0

    dem = base + relief * np.clip(d_coast / 3.0, 0.05, 1.0)
    dem = np.clip(dem, 0.4, None)

    # A shallow coastal plain plus one inland basin (future flood-prone area)
    basin = np.exp(-0.5 * (((X - 413_500) / 5200.0) ** 2 + ((Y - 4_621_000) / 4300.0) ** 2))
    dem -= 42.0 * basin
    dem = np.clip(dem, 0.4, None)

    dem = np.where(land_mask, dem, NODATA_F)
    return dem.astype("float32"), land_mask


def _rasterise_mask(geom, res):
    """Boolean raster mask (True inside geom) on the region grid."""
    from rasterio.features import rasterize
    nrows, ncols = grid_shape(res)
    arr = rasterize(
        [(geom, 1)], out_shape=(nrows, ncols), transform=grid_transform(res),
        fill=0, dtype="uint8", all_touched=False,
    )
    return arr.astype(bool)


def trace_rivers(dem, land_mask, rng, n_main=5):
    """Trace river centrelines as monotone westward flow paths.

    A naive steepest-descent walk on a synthetic DEM gets trapped in flat
    coastal pits and produces self-intersecting spaghetti. Instead each channel
    marches one cell west per step and only chooses *which* row to move to,
    guided by the (smoothed) terrain. The result is guaranteed to be a simple
    polyline that always terminates at the coast, while local relief still
    controls the meanders.
    """
    res = RES_DEM
    f = 8                                   # walk on a 200 m grid
    d = np.where(land_mask, dem, 0.0)
    small = gaussian_filter(d[::f, ::f], 2.0, mode="nearest")
    lmc = land_mask[::f, ::f]               # coarse land mask, unsmoothed
    sm_res = res * f
    nr, nc = small.shape

    # Sources: the highest cell in each of n_main latitude bands, taken from
    # the eastern third of the basin.
    east = small[:, int(nc * 0.62):]
    bands = np.array_split(np.arange(nr), n_main + 1)
    starts = []
    for bidx in bands[:-1] + [bands[-1]]:
        sub = east[bidx, :]
        rr, cc = np.unravel_index(np.argmax(sub), sub.shape)
        starts.append((int(bidx[0] + rr), int(cc + int(nc * 0.62))))
    starts = starts[:n_main + 1]

    lines = []
    for r0, c0 in starts:
        r, c, pts = r0, c0, []
        while c > 0:
            x = XMIN + (c + 0.5) * sm_res
            y = YMAX - (r + 0.5) * sm_res
            pts.append((x, y))
            if not lmc[r, c]:               # reached the sea
                break
            best, bestv = r, np.inf
            for dr in (-2, -1, 0, 1, 2):
                rr = r + dr
                if not (0 <= rr < nr):
                    continue
                v = small[rr, c - 1] + 0.35 * abs(dr) + rng.normal(0, 1.1)
                if v < bestv:
                    best, bestv = rr, v
            r, c = best, c - 1
        if len(pts) < 12:
            continue
        arr = np.asarray(pts)
        arr[:, 1] = gaussian_filter(arr[:, 1], 2.0, mode="nearest")
        line = LineString(arr).simplify(90.0)
        if line.is_simple and line.length > 4_000:
            lines.append(line)
    return lines


def build_landcover(dem, land_mask, rivers_union, rng):
    """Categorical land cover derived from elevation, urban pull and noise."""
    res = RES_DEM
    nrows, ncols = grid_shape(res)
    X, Y = grid_coords(res)

    # "Urban pull": decays away from the core and the two sub-centres
    def gauss(cx, cy, s):
        return np.exp(-0.5 * (((X - cx) / s) ** 2 + ((Y - cy) / s) ** 2))

    urban = 1.00 * gauss(*CORE, 4_300.0)
    urban += 0.55 * gauss(*SUBCORES[0], 2_500.0)
    urban += 0.50 * gauss(*SUBCORES[1], 2_300.0)
    urban += 0.30 * smooth_noise((nrows, ncols), 10.0, rng)

    # A noisier copy is used only for the *categorical* built-up mask, so that
    # the urban footprint is ragged rather than a perfect circle. The smooth
    # `urban` field is kept for population density.
    urban_lc = (urban + 0.40 * smooth_noise((nrows, ncols), 5.0, rng)
                + 0.22 * fbm((nrows, ncols), rng, octaves=4, base_sigma=20.0))

    elev = np.where(land_mask, dem, 0.0)
    n1 = smooth_noise((nrows, ncols), 8.0, rng) + 0.5 * smooth_noise((nrows, ncols), 22.0, rng)
    n2 = smooth_noise((nrows, ncols), 26.0, rng)

    lc = np.full((nrows, ncols), 4, dtype="uint8")            # default grassland
    lc[(elev > 35) & (n2 > -0.10)] = 3                        # cropland belt
    lc[(elev > 180) & (n1 > -0.35)] = 5                       # forest
    lc[(elev > 170) & (n1 <= -0.35)] = 6                      # shrubland
    lc[(elev > 520) & (n2 > 0.15)] = 7                        # bare rock
    lc[(elev < 30) & (n2 < -0.22)] = 8                        # wetland
    lc[urban_lc > 0.46] = 2                                   # built-up

    river_mask = _rasterise_mask(rivers_union.buffer(45.0), res)
    lc[river_mask] = 1
    lc[~land_mask] = NODATA_U8                                # sea -> nodata
    return lc, urban


def build_rainfall(dem, land_mask, rng):
    """Annual rainfall (mm): orographic gain with elevation plus a N-S gradient."""
    res = RES_RAIN
    nrows, ncols = grid_shape(res)
    X, Y = grid_coords(res)
    dem_c = _resample_block_mean(dem, RES_DEM, RES_RAIN, land_mask)
    north = (Y - YMIN) / HEIGHT_M
    rain = 470.0 + 0.62 * np.nan_to_num(dem_c) + 150.0 * north
    rain += 55.0 * smooth_noise((nrows, ncols), 4.0, rng)
    lm = _rasterise_mask_bool(land_mask, RES_DEM, RES_RAIN)
    rain = np.where(lm, rain, NODATA_F)
    return rain.astype("float32")


def _resample_block_mean(arr, res_in, res_out, valid_mask=None):
    """Aggregate a fine region grid to a coarser one by block mean."""
    f = int(round(res_out / res_in))
    a = np.where(valid_mask, arr, np.nan) if valid_mask is not None else arr.astype(float)
    nrows, ncols = a.shape
    a = a[: (nrows // f) * f, : (ncols // f) * f]
    a = a.reshape(a.shape[0] // f, f, a.shape[1] // f, f)
    return np.nanmean(a, axis=(1, 3))


def _rasterise_mask_bool(mask, res_in, res_out):
    f = int(round(res_out / res_in))
    m = mask[: (mask.shape[0] // f) * f, : (mask.shape[1] // f) * f]
    m = m.reshape(m.shape[0] // f, f, m.shape[1] // f, f)
    return m.mean(axis=(1, 3)) > 0.5


def build_ndvi_and_bands(lc, land_mask, rng):
    """NDVI at 50 m plus a 4-band 'satellite' image that is consistent with it."""
    res = RES_NDVI
    nrows, ncols = grid_shape(res)
    lc_c = lc[::2, ::2][:nrows, :ncols]
    lm = land_mask[::2, ::2][:nrows, :ncols]

    means = {1: 0.02, 2: 0.18, 3: 0.56, 4: 0.45, 5: 0.79, 6: 0.38, 7: 0.11, 8: 0.52}
    ndvi = np.zeros((nrows, ncols))
    for k, v in means.items():
        ndvi[lc_c == k] = v
    ndvi += 0.06 * smooth_noise((nrows, ncols), 3.0, rng)
    ndvi += rng.normal(0, 0.015, (nrows, ncols))
    ndvi = np.clip(ndvi, -0.2, 0.95)

    # Two "cloud" patches become NoData -> a realistic gap-filling problem
    X, Y = grid_coords(res)
    cloud = (np.hypot(X - 421_500, Y - 4_630_500) < 2_600) | \
            (np.hypot(X - 407_800, Y - 4_607_000) < 1_900)

    ndvi_out = np.where(lm & ~cloud, ndvi, NODATA_F).astype("float32")

    # Reflectance bands chosen so that (NIR - RED) / (NIR + RED) == ndvi
    red = np.clip(0.16 - 0.10 * ndvi + rng.normal(0, 0.006, (nrows, ncols)), 0.01, 0.5)
    nir = red * (1 + ndvi) / np.clip(1 - ndvi, 0.05, None)
    green = np.clip(red * 1.12 + 0.02 * ndvi, 0.01, 0.6)
    blue = np.clip(red * 0.95 - 0.005, 0.005, 0.6)
    stack = np.stack([blue, green, red, nir])
    stack = np.clip(stack, 0, 1.2) * 10_000.0
    stack = np.where(lm[None, ...], stack, 0)
    return ndvi_out, stack.astype("uint16")


def build_popdens(urban, land_mask, rng):
    """People per square kilometre at 100 m resolution."""
    res = RES_POP
    u = _resample_block_mean(urban, RES_DEM, RES_POP)
    lm = _rasterise_mask_bool(land_mask, RES_DEM, RES_POP)
    dens = 9_500.0 * np.clip(u, 0, None) ** 2.1 + 22.0
    dens *= 1 + 0.35 * smooth_noise(dens.shape, 3.0, rng)
    dens = np.clip(dens, 0, None)
    return np.where(lm, dens, NODATA_F).astype("float32")


def build_lst_webmercator(dem, urban, land_mask, rng, out_path):
    """Summer land-surface temperature, written in EPSG:3857 on purpose."""
    res = RES_LST
    dem_c = np.nan_to_num(_resample_block_mean(dem, RES_DEM, RES_LST, land_mask))
    u = _resample_block_mean(urban, RES_DEM, RES_LST)
    lm = _rasterise_mask_bool(land_mask, RES_DEM, RES_LST)
    lst = 31.5 - 0.0062 * dem_c + 6.4 * np.clip(u, 0, None)     # urban heat island
    lst += 0.9 * smooth_noise(lst.shape, 3.0, rng)
    src = np.where(lm, lst, NODATA_F).astype("float32")

    src_transform = grid_transform(RES_LST)
    dst_transform, dw, dh = calculate_default_transform(
        CRS_UTM, CRS_WEBMERC, src.shape[1], src.shape[0],
        left=XMIN, bottom=YMIN, right=XMAX, top=YMAX,
    )
    dst = np.full((dh, dw), NODATA_F, dtype="float32")
    reproject(
        source=src, destination=dst,
        src_transform=src_transform, src_crs=CRS_UTM, src_nodata=NODATA_F,
        dst_transform=dst_transform, dst_crs=CRS_WEBMERC, dst_nodata=NODATA_F,
        resampling=Resampling.bilinear,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=dh, width=dw, count=1,
        dtype="float32", crs=CRS_WEBMERC, transform=dst_transform,
        nodata=NODATA_F, compress="deflate", tiled=True,
    ) as ds:
        ds.write(dst, 1)
        ds.set_band_description(1, "Land surface temperature (deg C)")
    return out_path


# -----------------------------------------------------------------------------
# 3. VECTOR LAYERS
# -----------------------------------------------------------------------------
def poisson_points(poly, n, min_dist, rng, max_tries=40000):
    """Rejection-sample n points inside poly, no two closer than min_dist."""
    minx, miny, maxx, maxy = poly.bounds
    pts, tries = [], 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if not poly.contains(p):
            continue
        if pts:
            arr = np.array(pts)
            if np.min(np.hypot(arr[:, 0] - x, arr[:, 1] - y)) < min_dist:
                continue
        pts.append((x, y))
    return np.array(pts)


def voronoi_cells(points_xy, clip_poly):
    """Voronoi tessellation of points, clipped to clip_poly, in input order."""
    from shapely.geometry import MultiPoint
    mp = MultiPoint([Point(*p) for p in points_xy])
    env = clip_poly.buffer(6_000.0).envelope
    cells = list(voronoi_polygons(mp, extend_to=env).geoms)
    tree_cells = gpd.GeoSeries(cells)
    out = []
    for x, y in points_xy:
        p = Point(x, y)
        hit = tree_cells[tree_cells.covers(p)]
        g = hit.iloc[0] if len(hit) else tree_cells.iloc[
            int(np.argmin([c.distance(p) for c in cells]))]
        out.append(g.intersection(clip_poly).buffer(0))
    return out


def build_blocks(land, rng, popdens):
    """460 census blocks: a fine Voronoi tessellation of the land area."""
    xy = poisson_points(land.buffer(-150), 460, 900.0, rng)
    cells = voronoi_cells(xy, land)
    blocks = gpd.GeoDataFrame(
        {"block_id": [f"B{i+1:04d}" for i in range(len(cells))]},
        geometry=cells, crs=CRS_UTM,
    )
    blocks["area_km2"] = blocks.geometry.area / 1e6
    cent = blocks.geometry.representative_point()
    blocks["cx"], blocks["cy"] = cent.x.values, cent.y.values

    dens = sample_raster_at(np.where(popdens > NODATA_F / 2, popdens, np.nan),
                            RES_POP, blocks["cx"], blocks["cy"], default=25.0)
    dens = np.nan_to_num(dens, nan=25.0)
    pop = dens * blocks["area_km2"].values
    pop = np.maximum(0, pop * (1 + rng.normal(0, 0.12, len(pop))))
    blocks["population"] = np.round(pop).astype(int)
    blocks["households"] = np.round(
        blocks["population"] / rng.uniform(2.0, 3.2, len(blocks))).astype(int)
    blocks["pop_density_km2"] = (blocks["population"] / blocks["area_km2"]).round(1)
    return blocks


def build_districts(blocks, land, coast_line, rng, dem, land_mask, popdens):
    """Districts are built by AGGREGATING census blocks.

    Building the coarse units from the fine ones (rather than tessellating
    independently) gives two properties that real administrative data has and
    that make the dataset far more useful for teaching:

      * district boundaries are irregular and non-convex, because they follow
        block edges;
      * every district's population is *exactly* the sum of its blocks, so the
        learner has a built-in consistency check.
    """
    from scipy.spatial import cKDTree

    seeds = poisson_points(land.buffer(-400), len(DISTRICT_NAMES), 4_200.0, rng)
    seeds = seeds[: len(DISTRICT_NAMES)]
    # Name districts from the centre outwards, so the historic core really is
    # called "Old Vallmara" and the far uplands get the outlying names.
    order = np.argsort(np.hypot(seeds[:, 0] - CORE[0], seeds[:, 1] - CORE[1]))
    seeds = seeds[order]

    tree = cKDTree(seeds)
    _, owner = tree.query(np.c_[blocks["cx"], blocks["cy"]], k=1)
    blocks["district_id"] = [f"D{i+1:02d}" for i in owner]

    gdf = blocks.dissolve(
        by="district_id",
        aggfunc={"population": "sum", "households": "sum"},
    ).reset_index()
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf["name"] = [DISTRICT_NAMES[int(d[1:]) - 1] for d in gdf["district_id"]]
    gdf["area_km2"] = (gdf.geometry.area / 1e6).round(3)

    cx = gdf.geometry.representative_point().x.to_numpy()
    cy = gdf.geometry.representative_point().y.to_numpy()
    gdf["dist_core_km"] = np.round(np.hypot(cx - CORE[0], cy - CORE[1]) / 1000.0, 2)
    gdf["mean_elev_m"] = np.round(
        zonal_mean(np.where(land_mask, dem, np.nan), RES_DEM, gdf), 1)
    gdf["coastal"] = gdf.geometry.intersects(coast_line.buffer(60.0))
    gdf["pop_density_km2"] = (gdf["population"] / gdf["area_km2"]).round(1)

    mean_dens = gdf["pop_density_km2"].to_numpy(float)
    gdf["district_type"] = np.where(
        mean_dens > 700, "urban_core",
        np.where(mean_dens > 180, "suburban",
                 np.where(gdf["mean_elev_m"] > 260, "upland_rural", "rural")))

    cols = ["district_id", "name", "district_type", "area_km2", "dist_core_km",
            "mean_elev_m", "coastal", "population", "households",
            "pop_density_km2", "geometry"]
    return gdf[cols], seeds


def zonal_mean(array, res, gdf):
    """Mean of `array` inside every polygon of `gdf` (NaN-aware)."""
    from rasterio.features import rasterize
    nrows, ncols = grid_shape(res)
    idx = rasterize(
        ((geom, i + 1) for i, geom in enumerate(gdf.geometry)),
        out_shape=(nrows, ncols), transform=grid_transform(res),
        fill=0, dtype="int32",
    )
    a = np.asarray(array, dtype=float)
    out = np.full(len(gdf), np.nan)
    for i in range(len(gdf)):
        m = (idx == i + 1) & np.isfinite(a)
        if m.any():
            out[i] = a[m].mean()
    return out


def build_roads(districts, land, coast_line, rng):
    """Motorway along the coast, primary MST between districts, local grids."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import squareform, pdist

    recs = []
    rid = 0

    def add(geom, cls, name, lanes, speed, surface, aadt):
        nonlocal rid
        rid += 1
        recs.append(dict(
            road_id=f"R{rid:05d}", name=name, road_class=cls, lanes=lanes,
            speed_limit_kmh=speed, surface=surface, aadt=aadt, geometry=geom))

    # --- Motorway: coastline shifted 2.5 km inland -------------------------
    cands = []
    for off in (2_500.0, -2_500.0):
        c = coast_line.offset_curve(off, join_style=2)
        c = linemerge(c) if isinstance(c, MultiLineString) else c
        if isinstance(c, MultiLineString):
            c = max(c.geoms, key=lambda g: g.length)
        if isinstance(c, LineString):
            cands.append((c.intersection(land).length, c))
    mw = max(cands, key=lambda t: t[0])[1]
    mw = mw.simplify(150.0).intersection(land)
    parts = [mw] if isinstance(mw, LineString) else [g for g in mw.geoms
                                                     if isinstance(g, LineString)]
    for i, g in enumerate(parts):
        if g.length < 800:
            continue
        add(g, "motorway", "M1 Kestria Coastal Motorway", 4, 120.0, "asphalt",
            int(rng.integers(28_000, 52_000)))

    # --- Primary network: minimum spanning tree of district centroids -------
    cent = np.column_stack([districts.geometry.centroid.x, districts.geometry.centroid.y])
    D = squareform(pdist(cent))
    mst = minimum_spanning_tree(D).toarray()
    prim_names = ["A2 Vallmara Road", "A4 Northern Highway", "A7 Fyrdal Road",
                  "A9 Corran Vale Road", "A11 Ostrand Link"]
    for i in range(len(cent)):
        for j in range(len(cent)):
            if mst[i, j] > 0:
                ln = jitter_line(cent[i], cent[j], rng, n=5, amp=420)
                ln = ln.intersection(land)
                if isinstance(ln, MultiLineString):
                    ln = max(ln.geoms, key=lambda g: g.length)
                if not isinstance(ln, LineString) or ln.length < 300:
                    continue
                add(ln, "primary", prim_names[(i + j) % len(prim_names)], 2, 90.0,
                    "asphalt", int(rng.integers(6_000, 20_000)))

    # --- Secondary: extra chords between reasonably close districts ---------
    for i in range(len(cent)):
        order = np.argsort(D[i])[1:4]
        for j in order:
            if j <= i or D[i, j] > 9_000:
                continue
            if rng.random() > 0.55:
                continue
            ln = jitter_line(cent[i], cent[j], rng, n=4, amp=300).intersection(land)
            if isinstance(ln, MultiLineString):
                ln = max(ln.geoms, key=lambda g: g.length)
            if not isinstance(ln, LineString) or ln.length < 300:
                continue
            add(ln, "secondary", f"B{100 + i}{j} Road", 2, 70.0,
                rng.choice(["asphalt", "asphalt", "gravel"]),
                int(rng.integers(1_500, 8_000)))

    # --- Residential grid inside the urban core and sub-centres -------------
    for (cx, cy), half, step in [(CORE, 3_600, 600), (SUBCORES[0], 1_600, 550),
                                 (SUBCORES[1], 1_500, 550)]:
        for gx in np.arange(cx - half, cx + half + 1, step):
            ln = LineString([(gx, cy - half), (gx, cy + half)]).intersection(land)
            if isinstance(ln, LineString) and ln.length > 200:
                add(ln, "residential", f"Street {int(gx)%997}", 1, np.nan,
                    "asphalt", int(rng.integers(200, 2_500)))
        for gy in np.arange(cy - half, cy + half + 1, step):
            ln = LineString([(cx - half, gy), (cx + half, gy)]).intersection(land)
            if isinstance(ln, LineString) and ln.length > 200:
                add(ln, "residential", f"Avenue {int(gy)%997}", 1, np.nan,
                    "asphalt", int(rng.integers(200, 2_500)))

    # --- Unpaved tracks in the uplands --------------------------------------
    up = poisson_points(land, 40, 2_500.0, rng)
    from scipy.spatial import cKDTree
    tree = cKDTree(up)
    pairs = set()
    for k, pt in enumerate(up):
        for j in tree.query(pt, k=3)[1][1:]:
            pairs.add(tuple(sorted((k, int(j)))))
    for k, j in sorted(pairs):
        ln = jitter_line(up[k], up[j], rng, n=6, amp=420).intersection(land)
        if isinstance(ln, MultiLineString):
            ln = max(ln.geoms, key=lambda g: g.length)
        if isinstance(ln, LineString) and ln.length > 500:
            add(ln, "track", f"Track {k}-{j}", 1, 30.0, "dirt",
                int(rng.integers(20, 400)))

    roads = gpd.GeoDataFrame(recs, geometry="geometry", crs=CRS_UTM)
    roads = _segment_roads(roads, max_len=1_200.0)
    roads["length_m"] = roads.geometry.length.round(1)
    roads["oneway"] = np.where(roads["road_class"].eq("motorway"), True,
                               RNG.random(len(roads)) < 0.08)
    return roads


def _segment_roads(roads, max_len=1_200.0):
    """Chop every polyline into segments no longer than max_len metres.

    Real road layers are segment-based (one row per stretch between junctions),
    which is what makes per-segment attributes and spatial joins meaningful.
    """
    rows = []
    for _, r in roads.iterrows():
        g = r.geometry
        n = max(1, int(np.ceil(g.length / max_len)))
        step = g.length / n
        for k in range(n):
            d0, d1 = k * step, (k + 1) * step
            pts = [g.interpolate(d0)]
            inner = [c for c in g.coords
                     if d0 < g.project(Point(c)) < d1]
            pts += [Point(c) for c in inner]
            pts.append(g.interpolate(d1))
            seg = LineString([(p.x, p.y) for p in pts])
            if seg.length < 25:
                continue
            d = r.to_dict()
            d["geometry"] = seg
            d["segment_no"] = k + 1
            rows.append(d)
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=roads.crs)
    out["road_id"] = [f"R{i+1:05d}" for i in range(len(out))]
    return out


def build_rivers(river_lines, land, rng):
    names = ["Vallmara River", "Kestrel Brook", "Fyr River", "Corran Water",
             "Tarn Beck", "Halvorn Rill"]
    keep = []
    for ln in river_lines:
        g = ln.intersection(land)
        if isinstance(g, MultiLineString):
            g = max(g.geoms, key=lambda q: q.length)
        if isinstance(g, LineString) and g.length >= 3_000:
            keep.append(g)
    keep.sort(key=lambda q: -q.length)      # longest channel = main stem
    recs = []
    for i, g in enumerate(keep):
        order = 4 if i == 0 else (3 if i < 3 else 2)
        recs.append(dict(
            river_id=f"RV{len(recs)+1:02d}",
            name=names[len(recs) % len(names)],
            strahler_order=order,
            mean_discharge_m3s=round(float(rng.uniform(1.2, 4.5) * order ** 1.7), 2),
            perennial=bool(order >= 3 or rng.random() < 0.5),
            geometry=g,
        ))
    riv = gpd.GeoDataFrame(recs, geometry="geometry", crs=CRS_UTM)
    riv["length_km"] = (riv.geometry.length / 1000).round(2)
    return riv


def build_flood_zones(rivers, dem, land_mask, land, coast_line):
    """Flood hazard from HAND (height above nearest drainage) + coastal surge.

    HAND is the standard cheap proxy for fluvial flood exposure: for every cell,
    how high it sits above the nearest river cell. Low HAND + short distance to
    a channel = floodable.
    """
    from rasterio.features import shapes
    res = RES_DEM
    elev = np.where(land_mask, dem, 0.0).astype(float)

    rmask = _rasterise_mask(rivers.geometry.union_all().buffer(30), res)
    dist_cells, (ri, ci) = distance_transform_edt(~rmask, return_indices=True)
    dist_river = dist_cells * res
    hand = elev - elev[ri, ci]

    cmask = _rasterise_mask(coast_line.buffer(40), res)
    dist_coast = distance_transform_edt(~cmask) * res

    z100 = ((hand < 3.5) & (dist_river < 800)) | ((elev < 2.5) & (dist_coast < 900))
    z500 = ((hand < 9.0) & (dist_river < 1_900)) | ((elev < 6.0) & (dist_coast < 2_000))
    z100 &= land_mask
    z500 = z500 & land_mask & ~z100

    out = []
    for arr, label, rp in [(z100, "100-year flood zone", 100),
                           (z500, "500-year flood zone", 500)]:
        a8 = arr.astype("uint8")
        k = 0
        for geom, val in shapes(a8, mask=arr, transform=grid_transform(res)):
            poly = Polygon(geom["coordinates"][0], geom["coordinates"][1:]).buffer(0)
            if poly.is_empty or poly.area < 30_000:
                continue
            poly = poly.simplify(30.0).intersection(land).buffer(0)
            if poly.is_empty or not isinstance(poly, (Polygon, MultiPolygon)):
                continue
            for q in ([poly] if isinstance(poly, Polygon) else list(poly.geoms)):
                if q.area < 30_000:
                    continue
                k += 1
                out.append(dict(zone_id=f"FZ{rp}_{k:03d}", hazard_class=label,
                                return_period_yr=rp, geometry=q))
    fz = gpd.GeoDataFrame(out, geometry="geometry", crs=CRS_UTM)
    fz["area_km2"] = (fz.geometry.area / 1e6).round(4)
    return fz.sort_values(["return_period_yr", "zone_id"]).reset_index(drop=True)


def build_landuse(lc, land, rng):
    """Vectorise land cover into tidy polygons, then inject dirty records."""
    from rasterio.features import shapes
    res = RES_DEM
    f = 2
    small = lc[::f, ::f]
    tr = from_origin(XMIN, YMAX, res * f, res * f)
    recs = []
    for geom, val in shapes(small, mask=small > 0, transform=tr):
        v = int(val)
        poly = Polygon(geom["coordinates"][0], geom["coordinates"][1:]).buffer(0)
        if poly.is_empty or poly.area < 22_000:
            continue
        recs.append(dict(class_code=v, landuse_class=LANDCOVER_CLASSES[v],
                         geometry=poly.simplify(30.0)))
    lu = gpd.GeoDataFrame(recs, geometry="geometry", crs=CRS_UTM)
    lu = lu[lu.geometry.notna() & ~lu.geometry.is_empty].reset_index(drop=True)
    lu["lu_id"] = [f"LU{i+1:04d}" for i in range(len(lu))]
    lu["area_ha"] = (lu.geometry.area / 10_000).round(3)

    # ---- deliberate data-quality problems (documented in the README) -------
    # 1. inconsistent category spelling / whitespace / casing
    idx = RNG.choice(lu.index, size=45, replace=False)
    lu.loc[idx[:15], "landuse_class"] = lu.loc[idx[:15], "landuse_class"].str.upper()
    lu.loc[idx[15:30], "landuse_class"] = " " + lu.loc[idx[15:30], "landuse_class"] + " "
    lu.loc[idx[30:], "landuse_class"] = lu.loc[idx[30:], "landuse_class"].str.lower()

    # 2. three self-intersecting "bow-tie" polygons (invalid geometries)
    bad = []
    for k in range(3):
        cx = XMIN + 8_000 + k * 9_000
        cy = YMIN + 9_000 + k * 6_500
        bad.append(dict(
            lu_id=f"LU9{k+1:03d}", class_code=6, landuse_class="Shrubland",
            area_ha=np.nan,
            geometry=Polygon([(cx, cy), (cx + 900, cy + 900),
                              (cx + 900, cy), (cx, cy + 900)]),
        ))
    # 3. two exact duplicate rows
    dup = lu.iloc[[10, 40]].copy()
    lu = pd.concat([lu, gpd.GeoDataFrame(bad, crs=CRS_UTM), dup], ignore_index=True)
    lu = gpd.GeoDataFrame(lu, geometry="geometry", crs=CRS_UTM)
    return lu


def build_buildings(blocks, roads, popdens, land_mask, dem, rng, n=5200):
    """Building footprints, denser where population density is higher."""
    dens = np.where(popdens > NODATA_F / 2, popdens, 0.0)
    flat = dens.ravel().astype(float)
    p = flat / flat.sum()
    picks = rng.choice(flat.size, size=n, p=p)
    nrows, ncols = dens.shape
    rr, cc = np.unravel_index(picks, (nrows, ncols))
    x = XMIN + (cc + rng.random(n)) * RES_POP
    y = YMAX - (rr + rng.random(n)) * RES_POP

    ang = rng.uniform(0, np.pi / 2, n)
    w = rng.gamma(3.0, 3.4, n) + 5.0
    h = w * rng.uniform(0.6, 1.6, n)
    polys = []
    for xi, yi, wi, hi, ai in zip(x, y, w, h, ang):
        ca, sa = np.cos(ai), np.sin(ai)
        pts = [(-wi / 2, -hi / 2), (wi / 2, -hi / 2), (wi / 2, hi / 2), (-wi / 2, hi / 2)]
        polys.append(Polygon([(xi + px * ca - py * sa, yi + px * sa + py * ca)
                              for px, py in pts]))
    b = gpd.GeoDataFrame({"building_id": [f"BLD{i+1:05d}" for i in range(n)]},
                         geometry=polys, crs=CRS_UTM)
    b["footprint_m2"] = b.geometry.area.round(1)
    d_core = np.hypot(x - CORE[0], y - CORE[1]) / 1000.0
    b["floors"] = np.clip(np.round(rng.gamma(2.0, 1.1, n) + 3.5 * np.exp(-d_core / 4.0)),
                          1, 22).astype(int)
    use = rng.choice(["residential", "commercial", "industrial", "public"],
                     size=n, p=[0.74, 0.15, 0.07, 0.04])
    b["use_type"] = use
    b["year_built"] = np.clip(
        np.round(rng.normal(1975, 26, n) + 12 * np.exp(-d_core / 6.0)), 1890, 2024).astype(float)
    b["construction"] = rng.choice(["masonry", "reinforced_concrete", "timber", "steel"],
                                   size=n, p=[0.42, 0.38, 0.15, 0.05])
    b["has_basement"] = rng.random(n) < 0.31
    elev = sample_raster_at(np.where(land_mask, dem, np.nan), RES_DEM, x, y, default=5.0)
    b["ground_elev_m"] = np.round(np.nan_to_num(elev, nan=5.0), 1)
    base_value = (0.9 + 2.6 * np.exp(-d_core / 5.5)) * b["footprint_m2"] * b["floors"]
    base_value *= 1 + 0.004 * (b["year_built"] - 1975)
    b["value_kvs"] = np.round(base_value * rng.lognormal(0, 0.22, n) / 10, 1)

    # ---- deliberate problems ----------------------------------------------
    b.loc[RNG.choice(b.index, 240, replace=False), "year_built"] = np.nan
    b.loc[RNG.choice(b.index, 12, replace=False), "year_built"] = 1066.0   # impossible
    b.loc[RNG.choice(b.index, 6, replace=False), "year_built"] = 2199.0    # impossible
    b.loc[RNG.choice(b.index, 95, replace=False), "value_kvs"] = np.nan
    return b


def build_facilities(districts, blocks, land, roads, rng):
    """Schools, clinics, hospitals, fire and police stations."""
    spec = [("school", 42, 6, "Vallmara {} School"),
            ("clinic", 18, 4, "{} Community Clinic"),
            ("hospital", 4, 2, "{} General Hospital"),
            ("fire_station", 9, 3, "{} Fire Station"),
            ("police_station", 7, 3, "{} Police Station")]
    # sample locations weighted by block population
    w = blocks["population"].to_numpy(dtype=float) + 1.0
    w = w / w.sum()
    recs, fid = [], 0
    for ftype, count, minkm, tmpl in spec:
        idx = rng.choice(len(blocks), size=count, replace=False, p=w)
        for k, bi in enumerate(idx):
            pt = blocks.geometry.iloc[bi].representative_point()
            fid += 1
            dname = str(blocks["district_id"].iloc[bi])
            cap = {
                "school": int(rng.integers(180, 1500)),
                "clinic": int(rng.integers(8, 60)),
                "hospital": int(rng.integers(120, 620)),
                "fire_station": int(rng.integers(2, 9)),
                "police_station": int(rng.integers(6, 40)),
            }[ftype]
            recs.append(dict(
                facility_id=f"F{fid:04d}",
                name=tmpl.format(DISTRICT_NAMES[k % len(DISTRICT_NAMES)]),
                facility_type=ftype,
                capacity=cap,
                staff=int(max(1, cap * rng.uniform(0.05, 0.35))),
                opening_year=int(rng.integers(1958, 2023)),
                is_24h=bool(ftype in ("hospital", "fire_station", "police_station")),
                geometry=pt,
            ))
    f = gpd.GeoDataFrame(recs, geometry="geometry", crs=CRS_UTM)
    # deliberate problems: -999 sentinel for unknown capacity, 3 duplicate points
    f.loc[RNG.choice(f.index, 11, replace=False), "capacity"] = -999
    f = pd.concat([f, f.iloc[[3, 21, 55]]], ignore_index=True)
    return gpd.GeoDataFrame(f, geometry="geometry", crs=CRS_UTM)


def build_transit(roads, land, rng):
    """Six bus routes riding on named road corridors, plus stops along them."""
    usable = roads[roads["road_class"].isin(["primary", "secondary", "motorway"])]
    corridors = (usable.groupby("name")["geometry"]
                 .apply(lambda gs: linemerge(unary_union(list(gs)))))
    cand = []
    for nm, g in corridors.items():
        if isinstance(g, MultiLineString):
            g = max(g.geoms, key=lambda q: q.length)
        if isinstance(g, LineString) and g.length > 2_000:
            cand.append((nm, g))
    cand.sort(key=lambda t: -t[1].length)
    cand = cand[:6]

    routes, stops, sid = [], [], 0
    for r, (nm, line) in enumerate(cand):
        routes.append(dict(route_id=f"BUS{r+1:02d}", name=f"Route {r+1} ({nm})",
                           mode="bus", headway_min=int(rng.integers(8, 45)),
                           daily_riders=int(rng.integers(900, 14_000)),
                           geometry=line))
        d = 0.0
        while d < line.length:
            p = line.interpolate(d)
            sid += 1
            stops.append(dict(stop_id=f"S{sid:04d}", route_id=f"BUS{r+1:02d}",
                              shelter=bool(rng.random() < 0.55),
                              boardings_daily=int(rng.integers(5, 900)),
                              geometry=p))
            d += 650.0 * rng.uniform(0.7, 1.4)
    rt = gpd.GeoDataFrame(routes, geometry="geometry", crs=CRS_UTM)
    rt["length_km"] = (rt.geometry.length / 1000).round(2)
    return rt, gpd.GeoDataFrame(stops, geometry="geometry", crs=CRS_UTM)


def build_protected_areas(lc, land, rng):
    """Six reserves over forest/wetland, exported as WGS84 GeoJSON."""
    cand = _mask_polygons(lc, {5, 8}, min_area=2_500_000)
    cand = sorted(cand, key=lambda g: -g.area)[:6]
    # Clip each candidate to a compact circle so reserves are 5-40 km2 rather
    # than swallowing the entire forest belt.
    forest = []
    for i, g in enumerate(cand):
        c = g.representative_point()
        rad = 1_800.0 + 700.0 * (i % 4)
        clipped = g.intersection(c.buffer(rad)).buffer(0)
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda q: q.area)
        if not clipped.is_empty and clipped.area > 1_000_000:
            forest.append(clipped)
    names = ["Halvorn Forest Reserve", "Fyrdal Wetland Park",
             "Corran Ridge Nature Reserve", "Ostrand Marshes",
             "Brannock Woods", "Highfen Moor"]
    recs = []
    for i, g in enumerate(forest):
        g2 = g.simplify(60).intersection(land).buffer(0)
        if g2.is_empty:
            continue
        recs.append(dict(
            pa_id=f"PA{i+1:02d}", name=names[i % len(names)],
            designation=rng.choice(["National Park", "Nature Reserve",
                                    "Ramsar Wetland", "Regional Park"]),
            year_designated=int(rng.integers(1968, 2019)),
            iucn_category=str(rng.choice(["Ia", "II", "IV", "V"])),
            geometry=g2))
    pa = gpd.GeoDataFrame(recs, geometry="geometry", crs=CRS_UTM)
    if len(pa) >= 2:
        # Merge the last two reserves into ONE row holding a MultiPolygon, so
        # the layer mixes Polygon and MultiPolygon geometry types on purpose.
        merged = MultiPolygon([g if isinstance(g, Polygon) else max(g.geoms, key=lambda q: q.area)
                               for g in pa.geometry.iloc[-2:]])
        pa = pa.iloc[:-2].copy()
        pa.loc[len(pa)] = dict(pa_id="PA90", name="Kestrian Coastal Reserve (2 sites)",
                               designation="National Park", year_designated=1994,
                               iucn_category="II", geometry=merged)
        pa = gpd.GeoDataFrame(pa, geometry="geometry", crs=CRS_UTM)
    pa["area_km2"] = (pa.geometry.area / 1e6).round(3)
    return pa.to_crs(CRS_WGS84)


def _mask_polygons(lc, codes, min_area=0.0):
    from rasterio.features import shapes
    f = 4
    small = lc[::f, ::f]
    tr = from_origin(XMIN, YMAX, RES_DEM * f, RES_DEM * f)
    m = np.isin(small, list(codes)).astype("uint8")
    out = []
    for geom, val in shapes(m, mask=m.astype(bool), transform=tr):
        poly = Polygon(geom["coordinates"][0], geom["coordinates"][1:]).buffer(0)
        if poly.area >= min_area:
            out.append(poly)
    return out


def build_sensors_and_readings(land, roads, dem, land_mask, rain, urban, rng):
    """24 environmental monitoring stations + 36 months of readings."""
    xy = poisson_points(land.buffer(-800), 24, 4_000.0, rng)
    mw = roads[roads.road_class == "motorway"].geometry.union_all()
    urban_pt = sample_raster_at(urban, RES_DEM, xy[:, 0], xy[:, 1], default=0.0)
    elev = np.nan_to_num(sample_raster_at(np.where(land_mask, dem, np.nan), RES_DEM,
                                          xy[:, 0], xy[:, 1], default=5.0), nan=5.0)
    rain_pt = sample_raster_at(np.where(rain > NODATA_F / 2, rain, np.nan),
                               RES_RAIN, xy[:, 0], xy[:, 1], default=650.0)
    rain_pt = np.nan_to_num(rain_pt, nan=650.0)
    d_mw = np.array([Point(p).distance(mw) for p in xy])

    stations = pd.DataFrame({
        "station_id": [f"ST{i+1:03d}" for i in range(len(xy))],
        "name": [f"{DISTRICT_NAMES[i % len(DISTRICT_NAMES)]} Monitoring Station"
                 for i in range(len(xy))],
        "station_type": rng.choice(["air_quality", "rain_gauge", "combined"],
                                   size=len(xy), p=[0.35, 0.3, 0.35]),
        "install_year": rng.integers(1995, 2021, len(xy)),
        "elevation_m": np.round(elev, 1),
        "x_utm": np.round(xy[:, 0], 2), "y_utm": np.round(xy[:, 1], 2),
    })
    g = gpd.GeoSeries([Point(p) for p in xy], crs=CRS_UTM).to_crs(CRS_WGS84)
    stations["lon"] = g.x.round(6)
    stations["lat"] = g.y.round(6)

    months = pd.date_range("2022-01-01", "2024-12-01", freq="MS")
    rows = []
    for i, sid in enumerate(stations["station_id"]):
        for m in months:
            doy = m.month
            seas = np.cos(2 * np.pi * (doy - 1) / 12)
            pm = (7.5 + 16.0 * urban_pt[i] + 9.0 * np.exp(-d_mw[i] / 1_800.0)
                  + 4.0 * seas + rng.normal(0, 2.2))
            rr = max(0.0, rain_pt[i] / 12 * (1 + 0.55 * seas) * rng.lognormal(0, 0.35))
            tc = 16.0 - 0.0062 * elev[i] - 8.5 * seas + rng.normal(0, 1.0) + 2.4 * urban_pt[i]
            rows.append((sid, m.date().isoformat(), round(float(max(1.0, pm)), 2),
                         round(float(rr), 1), round(float(tc), 2)))
    readings = pd.DataFrame(rows, columns=["station_id", "date", "pm25_ugm3",
                                           "rainfall_mm", "temp_c"])
    # deliberate problems: NaN gaps, -999 sentinels, one duplicated key
    ridx = RNG.choice(readings.index, 90, replace=False)
    readings.loc[ridx[:45], "pm25_ugm3"] = np.nan
    readings.loc[ridx[45:], "rainfall_mm"] = -999.0
    readings = pd.concat([readings, readings.iloc[[7, 501]]], ignore_index=True)
    return stations, readings, d_mw


def build_flood_incidents(flood_zones, dem, land_mask, rivers, buildings, popdens,
                          rng, n=380):
    """Historic (fictional) flood REPORTS.

    Two processes are superimposed on purpose:
      * the hazard - 72 % of events fall in the 100-year zone, 28 % in the
        500-year zone;
      * the reporting bias - a flood is only recorded if somebody was there to
        record it, so within a zone the probability of a report is weighted by
        local population density.
    The second process is what makes naive pseudo-absence sampling dangerous
    in Module 3, Lesson A10.
    """
    import shapely as _sh
    z100 = flood_zones[flood_zones.return_period_yr == 100].geometry.union_all()
    z500 = flood_zones[flood_zones.return_period_yr == 500].geometry.union_all()
    riv = rivers.geometry.union_all()
    dens_arr = np.where(popdens > NODATA_F / 2, popdens, 0.0)

    def sample_in(poly, k):
        """Population-weighted rejection sampling inside `poly`."""
        minx, miny, maxx, maxy = poly.bounds
        px, py = [], []
        while len(px) < k * 40:
            x = rng.uniform(minx, maxx, 4000)
            y = rng.uniform(miny, maxy, 4000)
            m = _sh.contains_xy(poly, x, y)
            px.extend(x[m].tolist()); py.extend(y[m].tolist())
        px = np.asarray(px); py = np.asarray(py)
        dens = sample_raster_at(dens_arr, RES_POP, px, py, default=0.0)
        w = np.clip(np.nan_to_num(dens, nan=0.0), 5.0, None) ** 0.75
        w = w / w.sum()
        idx = rng.choice(len(px), size=k, replace=False, p=w)
        return list(zip(px[idx], py[idx]))

    pts = sample_in(z100, int(n * 0.72)) + sample_in(z500, n - int(n * 0.72))
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    elev = np.nan_to_num(sample_raster_at(np.where(land_mask, dem, np.nan),
                                          RES_DEM, xs, ys, default=5.0), nan=5.0)
    dr = np.array([Point(a, b).distance(riv) for a, b in zip(xs, ys)])

    dates = pd.to_datetime("2015-01-01") + pd.to_timedelta(
        rng.integers(0, 3650, len(xs)), unit="D")
    depth = np.clip(rng.gamma(2.0, 22.0, len(xs)) + 45 * np.exp(-dr / 500.0)
                    - 0.05 * elev, 5, 320)
    damage = depth * rng.lognormal(2.6, 0.6, len(xs))

    g = gpd.GeoSeries([Point(a, b) for a, b in zip(xs, ys)], crs=CRS_UTM).to_crs(CRS_WGS84)
    df = pd.DataFrame({
        "incident_id": [f"FI{i+1:04d}" for i in range(len(xs))],
        "date": [d.date().isoformat() for d in dates],
        "lon": g.x.round(6), "lat": g.y.round(6),
        "depth_cm": depth.round(1),
        "damage_kvs": damage.round(1),
        "cause": rng.choice(["river_overflow", "flash_flood", "drainage_failure",
                             "coastal_surge"], size=len(xs), p=[.45, .28, .20, .07]),
        "injuries": rng.poisson(0.12, len(xs)),
    })
    # ---- deliberate problems ----------------------------------------------
    dirty = []
    for k in range(6):                     # lon/lat swapped
        r = df.iloc[k].copy()
        r["incident_id"] = f"FIX{k+1:03d}"
        r["lon"], r["lat"] = r["lat"], r["lon"]
        dirty.append(r)
    for k in range(4):                     # null island
        r = df.iloc[50 + k].copy()
        r["incident_id"] = f"FIZ{k+1:03d}"
        r["lon"], r["lat"] = 0.0, 0.0
        dirty.append(r)
    for k in range(3):                     # far outside the study region
        r = df.iloc[100 + k].copy()
        r["incident_id"] = f"FIO{k+1:03d}"
        r["lon"], r["lat"] = 2.35 + k, 48.85 + k
        dirty.append(r)
    df = pd.concat([df, pd.DataFrame(dirty)], ignore_index=True)
    df.loc[RNG.choice(df.index, 25, replace=False), "depth_cm"] = np.nan
    df.loc[RNG.choice(df.index, 8, replace=False), "damage_kvs"] = -1.0
    return df


def build_socioeconomic(districts, blocks, rng):
    agg = blocks.groupby("district_id", dropna=True).agg(
        population=("population", "sum"), households=("households", "sum")).reset_index()
    df = districts[["district_id", "name", "district_type", "dist_core_km"]].merge(
        agg, on="district_id", how="left")
    d = df["dist_core_km"].to_numpy(float)
    df["median_income_vs"] = np.round(
        34_000 * np.exp(-d / 45.0) * rng.lognormal(0, 0.14, len(df))
        + 6_500 * (df["district_type"] == "urban_core"), -2)
    df["unemployment_rate"] = np.round(
        np.clip(0.11 - 0.0015 * (df["median_income_vs"] - 30_000) / 1_000
                + rng.normal(0, 0.018, len(df)), 0.015, 0.31), 4)
    df["pct_over65"] = np.round(np.clip(rng.normal(0.19, 0.045, len(df))
                                        + 0.02 * (d > 12), 0.06, 0.38), 4)
    df["pct_tertiary_edu"] = np.round(np.clip(
        0.42 - 0.010 * d + rng.normal(0, 0.05, len(df)), 0.05, 0.72), 4)
    df["hospital_beds_per_1000"] = np.round(
        np.clip(rng.gamma(2.2, 1.1, len(df)) - 0.05 * d, 0.0, 12.0), 2)
    df["vehicles_per_household"] = np.round(
        np.clip(0.75 + 0.035 * d + rng.normal(0, 0.12, len(df)), 0.2, 2.6), 3)
    df["survey_year"] = 2024

    # ---- deliberate problems ----------------------------------------------
    df.loc[RNG.choice(df.index, 2, replace=False), "median_income_vs"] = np.nan
    df = pd.concat([df, df.iloc[[4, 11]]], ignore_index=True)      # duplicate keys
    orphan = df.iloc[[0]].copy()
    orphan["district_id"] = "D99"
    orphan["name"] = "Ghost District (no geometry)"
    df = pd.concat([df, orphan], ignore_index=True)
    return df.drop(columns=["dist_core_km"])


# -----------------------------------------------------------------------------
# 4. MAIN
# -----------------------------------------------------------------------------
def main(outdir="data"):
    out = Path(outdir)
    vec, ras, tab = out / "vector", out / "raster", out / "tabular"
    for d in (vec, ras, tab, out / "outputs"):
        d.mkdir(parents=True, exist_ok=True)
    (out / "outputs" / ".gitkeep").touch()

    rng = np.random.default_rng(SEED)
    print("Vallmara Basin synthetic GIS database")
    print("-" * 62)

    # --- physical layers ----------------------------------------------------
    print("[ 1/14] coastline & land mask ...")
    coast_line, sea, land = build_coast_and_land(rng)

    print("[ 2/14] digital elevation model ...")
    dem, land_mask = build_dem(land, rng)

    print("[ 3/14] river network ...")
    river_lines = trace_rivers(dem, land_mask, rng)
    rivers = build_rivers(river_lines, land, rng)

    print("[ 4/14] land cover ...")
    lc, urban = build_landcover(dem, land_mask, rivers.geometry.union_all(), rng)

    print("[ 5/14] climate & vegetation rasters ...")
    rain = build_rainfall(dem, land_mask, rng)
    ndvi, bands = build_ndvi_and_bands(lc, land_mask, rng)
    popdens = build_popdens(urban, land_mask, rng)

    write_raster(ras / "dem_25m.tif", dem, RES_DEM, CRS_UTM, NODATA_F, "float32",
                 ["Elevation above mean sea level (m)"])
    write_raster(ras / "landcover_25m.tif", lc, RES_DEM, CRS_UTM, NODATA_U8, "uint8",
                 ["Land cover class code"])
    write_raster(ras / "rainfall_annual_250m.tif", rain, RES_RAIN, CRS_UTM, NODATA_F,
                 "float32", ["Mean annual rainfall (mm)"])
    write_raster(ras / "ndvi_50m.tif", ndvi, RES_NDVI, CRS_UTM, NODATA_F, "float32",
                 ["NDVI (-1..1)"])
    write_raster(ras / "popdens_100m.tif", popdens, RES_POP, CRS_UTM, NODATA_F,
                 "float32", ["Population density (persons/km2)"])
    write_raster(ras / "multispectral_50m.tif", bands, RES_NDVI, CRS_UTM, 0, "uint16",
                 ["Blue (B)", "Green (G)", "Red (R)", "NIR"])
    build_lst_webmercator(dem, urban, land_mask, rng, ras / "lst_summer_100m_3857.tif")

    # --- vector layers ------------------------------------------------------
    print("[ 6/14] census blocks ...")
    blocks = build_blocks(land, rng, popdens)

    print("[ 7/14] districts (dissolved from blocks) ...")
    districts, seeds = build_districts(blocks, land, coast_line, rng, dem,
                                       land_mask, popdens)
    blocks = blocks.drop(columns=["cx", "cy"])

    print("[ 8/14] roads ...")
    roads = build_roads(districts, land, coast_line, rng)

    print("[ 9/14] flood hazard zones ...")
    flood = build_flood_zones(rivers, dem, land_mask, land, coast_line)

    print("[10/14] land use polygons ...")
    landuse = build_landuse(lc, land, rng)

    print("[11/14] buildings ...")
    buildings = build_buildings(blocks, roads, popdens, land_mask, dem, rng)

    print("[12/14] facilities, transit, protected areas ...")
    facilities = build_facilities(districts, blocks, land, roads, rng)
    routes, stops = build_transit(roads, land, rng)
    protected = build_protected_areas(lc, land, rng)

    print("[13/14] tabular data ...")
    stations, readings, _ = build_sensors_and_readings(
        land, roads, dem, land_mask, rain, urban, rng)
    incidents = build_flood_incidents(flood, dem, land_mask, rivers, buildings,
                                      popdens, rng)
    socio = build_socioeconomic(districts, blocks, rng)

    # District population already equals the sum of its blocks (dissolve).
    # Two districts lose their population value on purpose (missing-data lesson).
    districts["population"] = districts["population"].astype(float)
    districts.loc[[6, 17], "population"] = np.nan
    districts.loc[[6, 17], "pop_density_km2"] = np.nan

    # One census block is given an EMPTY geometry on purpose
    blocks.loc[blocks.index[300], "geometry"] = Polygon()

    # --- write everything ---------------------------------------------------
    print("[14/14] writing files ...")
    gpkg = vec / "vallmara.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    layers = {
        "districts": districts,
        "census_blocks": blocks,
        "landuse": landuse,
        "rivers": rivers,
        "flood_zones": flood,
        "buildings": buildings,
        "facilities": facilities,
        "transit_routes": routes,
        "bus_stops": stops,
        "sea": gpd.GeoDataFrame({"name": ["Kestrian Sea"]}, geometry=[sea], crs=CRS_UTM),
        "land_boundary": gpd.GeoDataFrame({"name": ["Vallmara Basin"]},
                                          geometry=[land], crs=CRS_UTM),
        "coastline": gpd.GeoDataFrame({"name": ["Vallmara coastline"]},
                                      geometry=[coast_line], crs=CRS_UTM),
    }
    for lname, gdf in layers.items():
        gdf.to_file(gpkg, layer=lname, driver="GPKG")

    # Roads deliberately live in WGS84 GeoJSON -> forces a CRS conversion
    roads.to_crs(CRS_WGS84).to_file(vec / "roads.geojson", driver="GeoJSON")
    protected.to_file(vec / "protected_areas.geojson", driver="GeoJSON")

    # A legacy ESRI Shapefile with truncated field names
    shp_dir = vec / "schools_shp"
    shp_dir.mkdir(exist_ok=True)
    sch = facilities[facilities.facility_type == "school"].copy()
    sch = sch.rename(columns={"facility_id": "facility_identifier",
                              "opening_year": "year_of_opening_of_facility"})
    sch.to_file(shp_dir / "schools.shp", driver="ESRI Shapefile")

    stations.to_csv(tab / "sensor_stations.csv", index=False)
    readings.to_csv(tab / "sensor_readings.csv", index=False)
    incidents.to_csv(tab / "flood_incidents.csv", index=False)
    socio.to_csv(tab / "district_socioeconomic.csv", index=False)
    pd.DataFrame(
        [{"class_code": k, "landuse_class": v} for k, v in LANDCOVER_CLASSES.items()]
    ).to_csv(tab / "landcover_legend.csv", index=False)

    _write_data_readme(out)

    print("-" * 62)
    print(f"Done. Files written under: {out.resolve()}")
    print(f"  districts      {len(districts):6d}")
    print(f"  census blocks  {len(blocks):6d}")
    print(f"  roads          {len(roads):6d}")
    print(f"  buildings      {len(buildings):6d}")
    print(f"  landuse polys  {len(landuse):6d}")
    print(f"  facilities     {len(facilities):6d}")
    print(f"  flood zones    {len(flood):6d}")
    print(f"  incidents      {len(incidents):6d}")
    print(f"  sensor rows    {len(readings):6d}")
    return 0


DATA_README = """# Vallmara Basin — Synthetic GIS Database (v1.0)

**Everything here is fictional.** No real place, person, agency or measurement is
represented. The data are generated by `generate_data.py` from a fixed random
seed (42), so regenerating reproduces identical files.

## The scenario

The *Vallmara Basin* is an imaginary 48 km x 36 km coastal region in the
fictional Republic of Kestria. The Kestrian Sea lies to the west; the land rises
eastwards to the Corran Ridge (~900 m). The regional capital, **Vallmara City**,
sits at UTM (418000 E, 4618000 N), with two secondary centres. Five rivers drain
the uplands and flood the coastal plain periodically. The regional government has
asked you — the spatial data scientist — to quantify flood risk, service
accessibility and environmental pressure.

## Coordinate reference systems (deliberately mixed)

| CRS | Where it is used | Why it matters |
|---|---|---|
| `EPSG:32633` (WGS 84 / UTM 33N, metres) | GeoPackage layers, most rasters | The **analysis CRS**: metres, so lengths/areas/buffers are meaningful |
| `EPSG:4326` (WGS 84, degrees) | `roads.geojson`, `protected_areas.geojson`, CSV lon/lat | Storage/exchange CRS; degrees are **not** a distance unit |
| `EPSG:3857` (Web Mercator, metres) | `lst_summer_100m_3857.tif` | Web-map CRS; badly distorts area — never use it for statistics |

## Files

### `vector/vallmara.gpkg` (EPSG:32633)

| Layer | Geometry | Rows (approx.) | Key attributes |
|---|---|---|---|
| `districts` | Polygon | 24 | `district_id`, `name`, `district_type`, `area_km2`, `dist_core_km`, `mean_elev_m`, `coastal`, `population`, `households`, `pop_density_km2` |
| `census_blocks` | Polygon | 460 | `block_id`, `district_id`, `area_km2`, `population`, `households`, `pop_density_km2` |
| `landuse` | Polygon | 441 | `lu_id`, `class_code`, `landuse_class`, `area_ha` |
| `rivers` | LineString | 6 | `river_id`, `name`, `strahler_order`, `mean_discharge_m3s`, `perennial`, `length_km` |
| `flood_zones` | Polygon | 181 | `zone_id`, `hazard_class`, `return_period_yr` (100 / 500), `area_km2` |
| `buildings` | Polygon | 5200 | `building_id`, `use_type`, `floors`, `footprint_m2`, `year_built`, `construction`, `has_basement`, `ground_elev_m`, `value_kvs` |
| `facilities` | Point | ~83 | `facility_id`, `facility_type` (school/clinic/hospital/fire_station/police_station), `capacity`, `staff`, `opening_year`, `is_24h` |
| `transit_routes` | LineString | 6 | `route_id`, `headway_min`, `daily_riders` |
| `bus_stops` | Point | 130 | `stop_id`, `route_id`, `shelter`, `boardings_daily` |
| `sea`, `land_boundary`, `coastline` | Polygon / LineString | 1 each | study-area masks |

*Money is in **VS** (fictional Vallmara Shilling); `value_kvs` = thousands of VS.*

### `vector/roads.geojson` (EPSG:4326) — 791 segments
`road_id`, `name`, `road_class` (motorway / primary / secondary / residential /
track), `lanes`, `speed_limit_kmh`, `surface`, `aadt` (annual average daily
traffic), `oneway`, `length_m` *(computed in metres before export — compare it
with what you get if you naively measure length in degrees!)*

### `vector/protected_areas.geojson` (EPSG:4326)
`pa_id`, `name`, `designation`, `year_designated`, `iucn_category`, `area_km2`.
Five rows; the last one is a **MultiPolygon** covering two separate sites,
so the layer deliberately mixes geometry types.

### `vector/schools_shp/schools.shp` (EPSG:32633)
The same schools as in `facilities`, stored as a legacy Shapefile so you can see
field names truncated to 10 characters.

### `raster/`

| File | Res. | CRS | dtype | NoData | Meaning |
|---|---|---|---|---|---|
| `dem_25m.tif` | 25 m | 32633 | float32 | -9999 | Elevation (m). Sea = NoData |
| `landcover_25m.tif` | 25 m | 32633 | uint8 | 0 | Classes 1–8 (see `tabular/landcover_legend.csv`) |
| `rainfall_annual_250m.tif` | 250 m | 32633 | float32 | -9999 | Mean annual rainfall (mm) |
| `ndvi_50m.tif` | 50 m | 32633 | float32 | -9999 | NDVI, with two circular "cloud" NoData gaps |
| `popdens_100m.tif` | 100 m | 32633 | float32 | -9999 | Persons per km² |
| `multispectral_50m.tif` | 50 m | 32633 | uint16 | 0 | 4 bands: Blue, Green, Red, NIR (reflectance x 10000) |
| `lst_summer_100m_3857.tif` | ~100 m | **3857** | float32 | -9999 | Summer land surface temperature (°C) |

`multispectral_50m.tif` is constructed so that `(NIR - RED) / (NIR + RED)`
reproduces `ndvi_50m.tif` — you can verify your band maths against it.

### `tabular/`

| File | Rows | Contents |
|---|---|---|
| `sensor_stations.csv` | 24 | `station_id`, `name`, `station_type`, `install_year`, `elevation_m`, `x_utm`, `y_utm`, `lon`, `lat` |
| `sensor_readings.csv` | 866 | Monthly 2022-01 → 2024-12: `pm25_ugm3`, `rainfall_mm`, `temp_c` |
| `flood_incidents.csv` | 393 | `incident_id`, `date`, `lon`, `lat`, `depth_cm`, `damage_kvs`, `cause`, `injuries` |
| `district_socioeconomic.csv` | 27 | `district_id`, `population`, `households`, `median_income_vs`, `unemployment_rate`, `pct_over65`, `pct_tertiary_edu`, `hospital_beds_per_1000`, `vehicles_per_household` |
| `landcover_legend.csv` | 8 | class code → label |

## Deliberate data-quality problems

These are **intentional teaching material**, not bugs:

1. `landuse.landuse_class` — inconsistent casing and stray whitespace (45 rows).
2. `landuse` — 3 self-intersecting "bow-tie" polygons (invalid geometry) and 2
   exact duplicate rows (`lu_id` repeated).
3. `census_blocks` — one row (index 300) has an **empty** geometry.
4. `districts.population` — 2 missing values.
5. `buildings.year_built` — 240 NaN, 12 rows dated 1066, 6 rows dated 2199.
6. `buildings.value_kvs` — 95 NaN.
7. `facilities.capacity` — 11 rows use the sentinel `-999`; 3 duplicated points.
8. `roads.speed_limit_kmh` — NaN for every residential street.
9. `sensor_readings` — 45 NaN PM2.5, 45 rainfall values of `-999`, 2 duplicate rows.
10. `flood_incidents.csv` — 6 rows with lon/lat swapped, 4 rows at (0, 0)
    ("null island"), 3 rows in another country, 25 NaN depths, 8 negative damages.
11. `district_socioeconomic.csv` — 2 duplicated `district_id` keys and one
    orphan key `D99` that matches no polygon; 2 NaN incomes.
12. Three different CRS across the files (see table above).

## Ground truth built into the simulation

Knowing the generating process lets you check whether your analysis recovers it:

* Elevation rises inland; rainfall = `470 + 0.62 x elevation + north-south gradient`.
* PM2.5 rises with urban intensity and **decays exponentially with distance from
  the motorway** (e-folding distance ≈ 1.8 km).
* Land surface temperature = `31.5 - 0.0062 x elevation + 6.4 x urban intensity`
  (a textbook urban heat island).
* Flood incidents were drawn 72% from the 100-year zone and 28% from the
  500-year zone, and **within a zone the probability of a report is weighted by
  local population density** (`w = max(density, 5)^0.75`) -- a flood is only
  recorded if somebody was there to record it. Depth increases near rivers and
  decreases with elevation. This reporting bias is exactly what makes naive
  pseudo-absence sampling dangerous.
* Urban intensity `u` is a smooth field peaking at the city core. Population
  density is `9500 * u^2.1 + 22` persons/km2, so `u` can be recovered from the
  density raster as `u = ((density - 22) / 9500)^(1/2.1)`.
* PM2.5 = `7.5 + 16*u + 9*exp(-d_motorway / 1800 m) + seasonal + noise`. Because
  `u` and `d_motorway` are correlated, fitting the decay term *without* `u` gives
  a badly biased e-folding distance -- a built-in lesson in confounding.
* Population density decays from the city core. **Districts are built by
  dissolving census blocks**, so a district polygon is exactly the union of its
  blocks and its population is exactly their sum (except for the two districts
  whose population was deliberately blanked).
* Building value decays exponentially with distance from the core.
"""


def _write_data_readme(out: Path):
    (out / "README_DATA.md").write_text(DATA_README, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the Vallmara Basin dataset")
    ap.add_argument("--out", default="data", help="output directory (default: ./data)")
    args = ap.parse_args()
    sys.exit(main(args.out))
