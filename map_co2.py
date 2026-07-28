#!/usr/bin/env python3
"""
map_co2.py
==========
CONUS maps of predicted groundwater CO2 with ecoregion / land-cover overlays.
Pure matplotlib + geopandas + rasterio.  No cartopy.

Optimised:
  - Rasterised geometry masks (rasterio.features) replace per-pixel
    point-in-polygon for grid &f raster masking.
  - Vector overlays bbox-clipped + simplified; visual CONUS clip via
    matplotlib PathPatch (no expensive polygon intersection).
  - CONUS boundary from Natural Earth 50m states (fast, cached).
  - Shapefile loads cached for the session.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from shapely.geometry import box as shapely_box
from shapely.validation import make_valid
import os, glob, gc, warnings

warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.enums import Resampling
    import rasterio.features
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio not installed — .tif layers and fast masking unavailable")


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONUS_EXTENT = [-125, -66, 24, 50]

ESA_WORLDCOVER = {
    10:  ("#006400", "Tree cover"),
    20:  ("#ffbb22", "Shrubland"),
    30:  ("#ffff4c", "Grassland"),
    40:  ("#f096ff", "Cropland"),
    50:  ("#fa0000", "Built-up"),
    60:  ("#b4b4b4", "Bare / sparse veg."),
    70:  ("#f0f0f0", "Snow and ice"),
    80:  ("#0064c8", "Water bodies"),
    90:  ("#0096a0", "Herbaceous wetland"),
    95:  ("#00cf75", "Mangroves"),
    100: ("#fae6a0", "Moss and lichen"),
}

_BIOME_PALETTE = [
    "#a8ddb5", "#bdd7e7", "#fdbe85", "#d4b9da", "#b2e2e2",
    "#fbb4b9", "#c7e9c0", "#9ecae1", "#fdd49e", "#cbc9e2",
    "#74c476", "#6baed6", "#fdae6b", "#9e9ac8", "#67a9cf",
    "#f768a1", "#41ab5d", "#3182bd", "#e6550d", "#756bb1",
    "#016c59", "#08519c", "#a63603", "#54278f", "#014636",
]


# ═══════════════════════════════════════════════════════════════════
# COLORMAP
# ═══════════════════════════════════════════════════════════════════
def get_co2_cmap():
    colors = [
        "#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc",
        "#9e9ac8", "#807dba", "#6a51a3", "#54278f", "#3f007d",
    ]
    return mcolors.LinearSegmentedColormap.from_list("co2_purple", colors, N=256)


def _is_raster(path):
    return isinstance(path, str) and path.lower().endswith((".tif", ".tiff"))


# ═══════════════════════════════════════════════════════════════════
# US BOUNDARY — Natural Earth states (fast, cached)
# ═══════════════════════════════════════════════════════════════════
_CONUS_CACHE  = None
_STATES_CACHE = None


def _get_states_wgs84():
    """CONUS state polygons in WGS84. Downloaded once, then cached."""
    global _STATES_CACHE
    if _STATES_CACHE is not None:
        return _STATES_CACHE

    for res in ("50m", "110m"):
        url = (f"https://naciscdn.org/naturalearth/{res}/cultural/"
               f"ne_{res}_admin_1_states_provinces_lakes.zip")
        try:
            print(f"  [clip] Fetching Natural Earth {res} states ...", flush=True)
            gdf = gpd.read_file(url)
            gdf = gdf[gdf["admin"] == "United States of America"].copy()
            break
        except Exception as exc:
            print(f"  WARNING ({res}): {exc}")
            gdf = None

    if gdf is None or gdf.empty:
        return None

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)

    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g)

    conus_box = shapely_box(CONUS_EXTENT[0], CONUS_EXTENT[2],
                            CONUS_EXTENT[1], CONUS_EXTENT[3])
    gdf = gdf[gdf.geometry.centroid.within(conus_box)].copy()
    gdf = gpd.clip(gdf, conus_box)

    _STATES_CACHE = gdf
    print(f"  [clip] {len(gdf)} CONUS states loaded", flush=True)
    return _STATES_CACHE


def get_conus_boundary(source=None, boundary_layers=None):
    """Dissolved CONUS boundary. Uses Natural Earth states by default."""
    global _CONUS_CACHE
    if _CONUS_CACHE is not None:
        return _CONUS_CACHE

    gdf = None

    if isinstance(source, gpd.GeoDataFrame):
        gdf = source.copy()
    elif isinstance(source, str) and os.path.isfile(source):
        print(f"  [clip] Loading: {source}", flush=True)
        gdf = gpd.read_file(source)
        if gdf.crs is not None and not gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=4326)
        elif gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        gdf["geometry"] = gdf["geometry"].apply(
            lambda g: make_valid(g) if g is not None and not g.is_valid else g)
        conus_box = shapely_box(CONUS_EXTENT[0], CONUS_EXTENT[2],
                                CONUS_EXTENT[1], CONUS_EXTENT[3])
        gdf = gpd.clip(gdf, conus_box)

    if gdf is None:
        gdf = _get_states_wgs84()

    if gdf is None:
        return None

    _CONUS_CACHE = gdf.dissolve()
    print("  [clip] CONUS boundary ready", flush=True)
    return _CONUS_CACHE


def _build_clip_patch(ax, boundary_gdf):
    """Matplotlib PathPatch from a dissolved GeoDataFrame (for set_clip_path)."""
    geom = boundary_gdf.geometry.iloc[0]
    geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]

    verts, codes = [], []
    for g in geoms:
        if not hasattr(g, "exterior"):
            continue
        ring = np.array(g.exterior.coords)
        n = len(ring)
        if n < 3:
            continue
        verts.extend(ring.tolist())
        codes.extend(
            [MplPath.MOVETO] + [MplPath.LINETO] * (n - 2) + [MplPath.CLOSEPOLY])
        for interior in g.interiors:
            h = np.array(interior.coords)
            nh = len(h)
            if nh < 3:
                continue
            verts.extend(h.tolist())
            codes.extend(
                [MplPath.MOVETO] + [MplPath.LINETO] * (nh - 2) + [MplPath.CLOSEPOLY])

    if not verts:
        return None

    patch = mpatches.PathPatch(
        MplPath(verts, codes), transform=ax.transData,
        facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    return patch


# ═══════════════════════════════════════════════════════════════════
# DATA MASK — rasterised geometry mask (replaces per-pixel PIP)
# ═══════════════════════════════════════════════════════════════════
_INSIDE_CACHE = {}


def _mask_data_to_boundary(lons, lats, data, boundary_gdf):
    """
    NaN-out grid cells outside CONUS.
    Uses rasterio.features.geometry_mask (C-level rasterisation)
    instead of per-point shapely .within().
    """
    cache_key = (data.shape, float(lons[0]), float(lons[-1]),
                 float(lats[0]), float(lats[-1]))

    if cache_key not in _INSIDE_CACHE:
        print("  [mask] Rasterising CONUS mask ...", flush=True)
        h, w = data.shape

        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())

        # Pixel spacing; half-pixel padding so centres align with grid pts
        dx = (lon_max - lon_min) / max(w - 1, 1) if w > 1 else 0.1
        dy = (lat_max - lat_min) / max(h - 1, 1) if h > 1 else 0.1

        transform = transform_from_bounds(
            lon_min - dx / 2, lat_min - dy / 2,
            lon_max + dx / 2, lat_max + dy / 2,
            w, h)

        geom = boundary_gdf.geometry.unary_union
        # geometry_mask: True = OUTSIDE geometry
        inside = ~rasterio.features.geometry_mask(
            [geom.__geo_interface__],
            out_shape=(h, w),
            transform=transform)

        # rasterio row-0 = north; flip if data row-0 = south (ascending lats)
        if len(lats) > 1 and lats[0] < lats[-1]:
            inside = inside[::-1]

        _INSIDE_CACHE[cache_key] = inside
        n_in = inside.sum()
        print(f"  [mask] {n_in}/{inside.size} cells inside CONUS "
              f"({100 * n_in / inside.size:.1f}%)", flush=True)

    inside = _INSIDE_CACHE[cache_key]
    masked = data.copy().astype(float)
    masked[~inside] = np.nan
    return masked


def _mask_raster_to_boundary(data, extent, boundary_gdf):
    """Zero-out raster pixels outside CONUS via rasterised mask."""
    h, w = data.shape
    # extent = [left, right, bottom, top]
    transform = transform_from_bounds(
        extent[0], extent[2], extent[1], extent[3], w, h)

    geom = boundary_gdf.geometry.unary_union
    outside = rasterio.features.geometry_mask(
        [geom.__geo_interface__],
        out_shape=(h, w),
        transform=transform)

    masked = data.copy()
    masked[outside] = 0
    return masked


# ═══════════════════════════════════════════════════════════════════
# RASTER LAYER  (.tif)
# ═══════════════════════════════════════════════════════════════════
def _read_raster_conus(tif_path, downsample=4, clip_to_conus=True):
    """Return (data_2d, [left,right,bottom,top], nodata)."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio required for .tif layers")

    with rasterio.open(tif_path) as src:
        if clip_to_conus:
            win = from_bounds(CONUS_EXTENT[0], CONUS_EXTENT[2],
                              CONUS_EXTENT[1], CONUS_EXTENT[3], src.transform)
            oh = max(1, int(win.height) // downsample)
            ow = max(1, int(win.width)  // downsample)
            data = src.read(1, window=win, out_shape=(oh, ow))
            ext = list(CONUS_EXTENT)
        else:
            oh = max(1, src.height // downsample)
            ow = max(1, src.width  // downsample)
            data = src.read(1, out_shape=(oh, ow))
            b = src.bounds
            ext = [b.left, b.right, b.bottom, b.top]
        nodata = src.nodata

    print(f"  [raster] {os.path.basename(tif_path)}  "
          f"{ow}x{oh} px  (ds{downsample}x)", flush=True)
    return data, ext, nodata


def _plot_raster_fill(ax, data, extent, nodata,
                      fill_alpha, class_map, zorder=1):
    if class_map is None:
        class_map = ESA_WORLDCOVER
    rgba = np.zeros((*data.shape, 4), dtype=np.float32)
    for cval, (hexc, _) in class_map.items():
        m = data == cval
        if m.any():
            r, g, b = mcolors.hex2color(hexc)
            rgba[m] = [r, g, b, fill_alpha]
    if nodata is not None:
        rgba[data == nodata] = 0
    rgba[data == 0] = 0
    im = ax.imshow(rgba, extent=extent, origin="upper",
                   aspect="equal", interpolation="nearest", zorder=zorder)
    present = set(np.unique(data))
    handles = [mpatches.Patch(facecolor=hexc, alpha=fill_alpha,
                              edgecolor="#333", linewidth=0.3, label=name)
               for cval in sorted(class_map)
               if cval in present and cval != 0
               for hexc, name in [class_map[cval]]]
    return im, handles


def _plot_raster_boundary(ax, data, extent, class_map,
                          linewidth=0.6, zorder=5):
    if class_map is None:
        class_map = ESA_WORLDCOVER
    h, w = data.shape
    xs = np.linspace(extent[0], extent[1], w)
    ys = np.linspace(extent[3], extent[2], h)
    present = set(np.unique(data))
    handles = []
    for cval in sorted(class_map):
        if cval not in present or cval == 0:
            continue
        hexc, name = class_map[cval]
        binary = (data == cval).astype(np.float32)
        ax.contour(xs, ys, binary, levels=[0.5],
                   colors=[hexc], linewidths=linewidth, zorder=zorder)
        handles.append(mpatches.Patch(
            facecolor="none", edgecolor=hexc, linewidth=1.5, label=name))
    return handles


# ═══════════════════════════════════════════════════════════════════
# VECTOR LAYER — bbox clip + simplify + visual clip_path
# ═══════════════════════════════════════════════════════════════════
_SHP_CACHE = {}


def _load_boundary_shp(shp_path):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)
    elif gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g)
    return gdf


def _plot_vector_layer(ax, shp_path, label_col=None, fill_alpha=0.25,
                       linewidth=0.8, edgecolor="#333333", clip_to_conus=True,
                       legend=True, zorder_fill=1, zorder_edge=3,
                       simplify_tol=0.01, clip_patch=None):
    """
    Plot vector overlay.  Bbox-clipped + simplified for speed;
    visual CONUS clip via clip_patch (no expensive polygon intersection).
    """
    # ── Load with session caching ─────────────────────────────────
    if isinstance(shp_path, str):
        cache_key = (shp_path, simplify_tol)
        if cache_key not in _SHP_CACHE:
            print(f"  [vector] Loading {os.path.basename(shp_path)} ...", flush=True)
            gdf = _load_boundary_shp(shp_path)
            # Fast bbox clip (NOT full polygon intersection with CONUS)
            conus_box = shapely_box(CONUS_EXTENT[0], CONUS_EXTENT[2],
                                    CONUS_EXTENT[1], CONUS_EXTENT[3])
            gdf = gpd.clip(gdf, conus_box)
            if simplify_tol > 0:
                gdf["geometry"] = gdf["geometry"].simplify(
                    simplify_tol, preserve_topology=True)
            _SHP_CACHE[cache_key] = gdf
            print(f"  [vector] {len(gdf)} features cached", flush=True)
        gdf = _SHP_CACHE[cache_key].copy()
    else:
        gdf = shp_path.copy()
        if clip_to_conus:
            conus_box = shapely_box(CONUS_EXTENT[0], CONUS_EXTENT[2],
                                    CONUS_EXTENT[1], CONUS_EXTENT[3])
            gdf = gpd.clip(gdf, conus_box)

    if gdf.empty:
        return [], gdf, edgecolor, linewidth

    # ── Assign colours ────────────────────────────────────────────
    color_map = {}
    if label_col and label_col in gdf.columns:
        cats = sorted(gdf[label_col].dropna().unique())
        color_map = {c: _BIOME_PALETTE[i % len(_BIOME_PALETTE)]
                     for i, c in enumerate(cats)}
        gdf["_fill"] = gdf[label_col].map(color_map).fillna("#e8e8e8")
    else:
        gdf["_fill"] = "#e8e8e8"

    # ── Plot fills (clipped visually to CONUS via clip_patch) ─────
    if fill_alpha > 0:
        n_before = len(ax.collections)
        for fc, sub in gdf.groupby("_fill"):
            sub.plot(ax=ax, facecolor=fc, edgecolor="none",
                     alpha=fill_alpha, linewidth=0, zorder=zorder_fill)
        if clip_patch is not None:
            for coll in ax.collections[n_before:]:
                coll.set_clip_path(clip_patch)

    # ── Plot edges (clipped visually) ─────────────────────────────
    n_before = len(ax.collections)
    gdf.boundary.plot(ax=ax, edgecolor=edgecolor, linewidth=linewidth,
                      zorder=zorder_edge)
    if clip_patch is not None:
        for coll in ax.collections[n_before:]:
            coll.set_clip_path(clip_patch)

    # ── Legend handles ────────────────────────────────────────────
    handles = []
    if legend and color_map:
        for cat in sorted(color_map):
            handles.append(mpatches.Patch(
                facecolor=color_map[cat], edgecolor=edgecolor, linewidth=0.3,
                alpha=max(fill_alpha, 0.3), label=cat))
    return handles, gdf, edgecolor, linewidth


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════
def load_all_time_mean(cache_path):
    if cache_path.endswith(".csv"):
        return pd.read_csv(cache_path)
    return pd.read_parquet(cache_path)


def load_predictions(pred_dir, year=None, month=None, date=None):
    dfs = []
    if date is not None:
        dt = pd.to_datetime(date)
        ym = [(dt.year, dt.month)]
    elif month is not None:
        p = month.split("-")
        ym = [(int(p[0]), int(p[1]))]
    elif year is not None:
        ym = [(year, m) for m in range(1, 13)]
    else:
        raise ValueError("Specify date, month, or year")

    for y, m in ym:
        for pat in [
            os.path.join(pred_dir, f"year={y}", f"month={m}", "*.parquet"),
            os.path.join(pred_dir, f"year_{y}", f"month_{m}", "*.parquet"),
        ]:
            for f in glob.glob(pat):
                df = pd.read_parquet(f)
                if "year"  not in df.columns: df["year"]  = y
                if "month" not in df.columns: df["month"] = m
                dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No files in {pred_dir}")
    big = pd.concat(dfs, ignore_index=True)
    if date is not None and "valid_time" in big.columns:
        big["valid_time"] = pd.to_datetime(big["valid_time"])
        big = big[big["valid_time"].dt.normalize() == pd.to_datetime(date).normalize()]
    print(f"  Loaded {len(big):,} rows", flush=True)
    return big


def aggregate_to_grid(df, agg="mean"):
    g = df.groupby(["latitude", "longitude"])["CO2_predicted"].agg(agg).reset_index()
    return g.pivot(index="latitude", columns="longitude",
                   values="CO2_predicted").sort_index(ascending=True)


def precompute_all_time_mean(pred_dir, cache_path=None, extra_cols=None):
    if cache_path is None:
        cache_path = os.path.join(pred_dir, "alltime_mean.parquet")
    files = glob.glob(os.path.join(pred_dir, "**", "*.parquet"), recursive=True)
    files = [f for f in files if os.path.basename(f) != os.path.basename(cache_path)]
    if not files:
        raise FileNotFoundError(f"No parquets under {pred_dir}")
    print(f"Precomputing from {len(files)} files ...", flush=True)
    vcols = ["CO2_predicted"] + (list(extra_cols) if extra_cols else [])
    rc = ["latitude", "longitude"] + vcols
    acc_s, acc_n = None, None
    for i, fp in enumerate(files):
        try:    df = pd.read_parquet(fp, columns=rc)
        except: df = pd.read_parquet(fp, columns=["latitude","longitude","CO2_predicted"])
        g = df.groupby(["latitude", "longitude"])
        s, n = g[vcols].sum(), g["CO2_predicted"].count().rename("n")
        if acc_s is None: acc_s, acc_n = s, n
        else: acc_s, acc_n = acc_s.add(s, fill_value=0), acc_n.add(n, fill_value=0)
        if (i+1) % 50 == 0: print(f"  {i+1}/{len(files)} ...", flush=True)
        del df; gc.collect()
    out = acc_s.div(acc_n, axis=0).reset_index()
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    (out.to_csv if cache_path.endswith(".csv") else out.to_parquet)(cache_path, index=False)
    print(f"  Saved: {cache_path}  ({len(out):,} cells)")
    return out


# ═══════════════════════════════════════════════════════════════════
# CORE PLOT — add show_states parameter
# ═══════════════════════════════════════════════════════════════════
def plot_co2_map(
    pivot, title,
    cmap=None, vmin=None, vmax=None, log_scale=False,
    co2_alpha=0.85,
    label="Predicted CO\u2082 (mol/L)",
    save_path=None,
    boundary_layers=None,
    clip_boundary="auto",
    figsize=(16, 10),
    show_states=True,          # <-- NEW
):
    lons = pivot.columns.values.astype(float)
    lats = pivot.index.values.astype(float)
    data = pivot.values

    if cmap is None:
        cmap = get_co2_cmap()
    if not isinstance(cmap, mcolors.Colormap):
        cmap = plt.get_cmap(cmap)
    cmap = cmap.copy()
    cmap.set_bad(color="none")

    valid = data[~np.isnan(data)]
    if vmin is None: vmin = np.percentile(valid, 2)  if len(valid) else 0
    if vmax is None: vmax = np.percentile(valid, 98) if len(valid) else 1
    norm = (mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax) if log_scale
            else mcolors.Normalize(vmin=vmin, vmax=vmax))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    clip_patch = None
    clip_gdf  = None
    if clip_boundary is not None:
        src = None if clip_boundary == "auto" else clip_boundary
        clip_gdf = get_conus_boundary(source=src,
                                      boundary_layers=boundary_layers)
        if clip_gdf is not None:
            clip_patch = _build_clip_patch(ax, clip_gdf)

    all_handles   = []
    vector_redraw = []
    raster_artists = []

    if boundary_layers:
        for layer in boundary_layers:
            lyr = layer.copy()
            path        = lyr.pop("shp")
            clip_c      = lyr.pop("clip_to_conus", True)
            show_legend = lyr.pop("legend", True)

            if _is_raster(path):
                style = lyr.pop("style", "fill")
                ds    = lyr.pop("downsample", 4)
                cm    = lyr.pop("class_map", None)
                rdata, rext, rnodata = _read_raster_conus(
                    path, downsample=ds, clip_to_conus=clip_c)

                if clip_gdf is not None:
                    rdata = _mask_raster_to_boundary(rdata, rext, clip_gdf)

                if style == "boundary":
                    hdls = _plot_raster_boundary(
                        ax, rdata, rext, cm,
                        linewidth=lyr.pop("linewidth", 0.6), zorder=5)
                else:
                    im_r, hdls = _plot_raster_fill(
                        ax, rdata, rext, rnodata,
                        fill_alpha=lyr.pop("fill_alpha", 0.45),
                        class_map=cm, zorder=1)
                    raster_artists.append(im_r)
                    if clip_patch:
                        im_r.set_clip_path(clip_patch)

                if show_legend:
                    all_handles.extend(hdls)

            else:
                lc  = lyr.pop("label_col", None)
                fa  = lyr.pop("fill_alpha", 0.25)
                lw  = lyr.pop("linewidth", 1.2)   # thicker default for vectors
                ec  = lyr.pop("edgecolor", "#333333")
                st  = lyr.pop("simplify", 0.01)
                hdls, gdf, ec_out, lw_out = _plot_vector_layer(
                    ax, path, label_col=lc, fill_alpha=fa,
                    linewidth=lw, edgecolor=ec,
                    clip_to_conus=clip_c, legend=show_legend,
                    zorder_fill=1, zorder_edge=3,
                    simplify_tol=st, clip_patch=clip_patch)
                if show_legend:
                    all_handles.extend(hdls)
                vector_redraw.append((gdf, ec_out, lw_out))

    if clip_gdf is not None:
        data = _mask_data_to_boundary(lons, lats, data, clip_gdf)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    im = ax.pcolormesh(
        lon_grid, lat_grid, data,
        cmap=cmap, norm=norm, alpha=co2_alpha,
        shading="auto", rasterized=True, zorder=2)

    if clip_patch:
        im.set_clip_path(clip_patch)

    for gdf, ec, lw in vector_redraw:
        if not gdf.empty:
            n_before = len(ax.collections)
            gdf.boundary.plot(ax=ax, edgecolor=ec, linewidth=lw, zorder=5)
            if clip_patch:
                for coll in ax.collections[n_before:]:
                    coll.set_clip_path(clip_patch)

    # ── 4) US state outlines — only if requested ─────────────────
    if show_states:
        states = _get_states_wgs84()
        if states is not None:
            states.boundary.plot(ax=ax, edgecolor="#222222",
                                 linewidth=0.5, zorder=6)

    ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
    ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, shrink=0.65, pad=0.02, aspect=30)
    cbar.set_label(label, fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    if all_handles:
        has_raster = any(_is_raster(l.get("shp", ""))
                         for l in (boundary_layers or []))
        has_vector = any(not _is_raster(l.get("shp", ""))
                         for l in (boundary_layers or []))
        if has_raster and has_vector:
            ltitle = "Layers"
        elif has_raster:
            ltitle = "Land Cover"
        else:
            ltitle = "Biome"

        ax.legend(handles=all_handles, loc="lower left", fontsize=5.5,
                  framealpha=0.92, ncol=2, title=ltitle,
                  title_fontsize=7, borderpad=0.6)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API — all updated with show_states
# ═══════════════════════════════════════════════════════════════════
def map_all_time_mean(
    pred_dir=None, cache_path=None,
    agg="mean", log_scale=False,
    vmin=None, vmax=None, cmap=None,
    co2_alpha=0.85,
    label="Predicted CO\u2082 (mol/L)",
    title=None,
    save_path=None, save_dir=None,
    show=True,
    boundary_layers=None,
    clip_boundary="auto",
    figsize=(16, 10),
    show_states=True,          # <-- NEW
):
    if cache_path is None and pred_dir is not None:
        cache_path = os.path.join(pred_dir, "alltime_mean.parquet")

    if cache_path and os.path.isfile(cache_path):
        print(f"Loading cached all-time mean: {cache_path}", flush=True)
        df_mean = load_all_time_mean(cache_path)
        print(f"  {len(df_mean):,} grid cells", flush=True)
    elif pred_dir is not None:
        df_mean = precompute_all_time_mean(pred_dir, cache_path=cache_path)
    else:
        raise ValueError("Provide cache_path or pred_dir")

    pivot = aggregate_to_grid(df_mean, agg="mean")
    print(f"  Grid: {pivot.shape[0]} lat x {pivot.shape[1]} lon")

    if title is None:
        title = f"Predicted Groundwater CO\u2082 \u2014 All-Time {agg.capitalize()}"
    fname = f"co2_map_alltime_{agg}.png"
    if save_dir and not save_path:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, fname)

    fig = plot_co2_map(
        pivot, title, cmap=cmap, label=label,
        vmin=vmin, vmax=vmax, log_scale=log_scale,
        co2_alpha=co2_alpha,
        save_path=save_path,
        boundary_layers=boundary_layers,
        clip_boundary=clip_boundary,
        figsize=figsize,
        show_states=show_states,   # <-- pass through
    )
    if show:
        plt.show()
    return fig


def map_co2(
    pred_dir,
    date=None, month=None, year=None, diff_year=None,
    agg="mean", log_scale=False,
    vmin=None, vmax=None, cmap=None,
    co2_alpha=0.85,
    save_path=None, save_dir=None,
    show=True, boundary_layers=None,
    clip_boundary="auto",
    figsize=(16, 10),
    show_states=True,          # <-- NEW
):
    if date is not None:
        df = load_predictions(pred_dir, date=date)
        tag, fname = date, f"co2_map_{date}.png"
    elif month is not None:
        df = load_predictions(pred_dir, month=month)
        tag, fname = f"{month} ({agg})", f"co2_map_{month}_{agg}.png"
    elif year is not None:
        df = load_predictions(pred_dir, year=year)
        tag, fname = f"{year} ({agg})", f"co2_map_{year}_{agg}.png"
    else:
        raise ValueError("Specify date, month, or year")

    pivot = aggregate_to_grid(df, agg=agg)
    lbl = "Predicted CO\u2082 (mol/L)"

    if diff_year is not None and year is not None:
        pr = aggregate_to_grid(load_predictions(pred_dir, year=diff_year), agg=agg)
        cl = pivot.index.intersection(pr.index)
        cc = pivot.columns.intersection(pr.columns)
        pivot = pivot.loc[cl, cc] - pr.loc[cl, cc]
        tag = f"Anomaly {year} - {diff_year} ({agg})"
        fname = f"co2_anomaly_{year}_vs_{diff_year}_{agg}.png"
        lbl = "\u0394CO\u2082 (mol/L)"

    if save_dir and not save_path:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, fname)

    return plot_co2_map(
        pivot, f"Predicted Groundwater CO\u2082 \u2014 {tag}",
        cmap=cmap, vmin=vmin, vmax=vmax, log_scale=log_scale,
        co2_alpha=co2_alpha, label=lbl, save_path=save_path,
        boundary_layers=boundary_layers, clip_boundary=clip_boundary,
        figsize=figsize,
        show_states=show_states,   # <-- pass through
    )


def map_seasonal(
    pred_dir, year, agg="mean", log_scale=False,
    vmin=None, vmax=None, save_dir=None, show=True,
    boundary_layers=None,
    clip_boundary="auto",
    show_states=True,          # <-- NEW
):
    seasons = {"DJF": [12,1,2], "MAM": [3,4,5],
               "JJA": [6,7,8],  "SON": [9,10,11]}
    cmap = get_co2_cmap().copy(); cmap.set_bad("none")
    df = load_predictions(pred_dir, year=year)
    if "month" not in df.columns and "valid_time" in df.columns:
        df["month"] = pd.to_datetime(df["valid_time"]).dt.month
    pivots = {n: (aggregate_to_grid(df[df["month"].isin(ms)], agg=agg)
                  if not df[df["month"].isin(ms)].empty else None)
              for n, ms in seasons.items()}
    allv = np.concatenate([p.values.ravel() for p in pivots.values() if p is not None])
    allv = allv[~np.isnan(allv)]
    if vmin is None: vmin = np.percentile(allv, 2)
    if vmax is None: vmax = np.percentile(allv, 98)
    norm = (mcolors.LogNorm(max(vmin,1e-10), vmax) if log_scale
            else mcolors.Normalize(vmin, vmax))

    clip_gdf = None
    if clip_boundary is not None:
        src = None if clip_boundary == "auto" else clip_boundary
        clip_gdf = get_conus_boundary(source=src,
                                      boundary_layers=boundary_layers)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12)); axes = axes.ravel(); im = None
    for i, (n, pv) in enumerate(pivots.items()):
        ax = axes[i]
        if pv is None:
            ax.text(.5,.5,"No data",ha="center",va="center",
                    transform=ax.transAxes)
            continue
        lo = pv.columns.values.astype(float)
        la = pv.index.values.astype(float)
        pv_data = pv.values

        if clip_gdf is not None:
            pv_data = _mask_data_to_boundary(lo, la, pv_data, clip_gdf)

        lg, ltg = np.meshgrid(lo, la)
        im = ax.pcolormesh(lg, ltg, pv_data, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True)

        if clip_gdf is not None:
            cp = _build_clip_patch(ax, clip_gdf)
            if cp is not None:
                im.set_clip_path(cp)
            if show_states:                        # <-- conditional
                states = _get_states_wgs84()
                if states is not None:
                    states.boundary.plot(ax=ax, edgecolor="#222222",
                                         linewidth=0.5, zorder=6)

        ax.set_xlim(*CONUS_EXTENT[:2]); ax.set_ylim(*CONUS_EXTENT[2:])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(n, fontsize=13, fontweight="bold")

    fig.suptitle(f"Seasonal CO\u2082 \u2014 {year}", fontsize=18,
                 fontweight="bold", y=1.01)
    if im:
        cax = fig.add_axes([.92,.15,.015,.7])
        fig.colorbar(im, cax=cax).set_label(
            "CO\u2082 (mol/L)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0,0,.91,1])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"co2_seasonal_{year}.png"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    if show: plt.show()
    return fig

def map_timeseries(pred_dir, years, lat=None, lon=None, agg="mean",
                   save_dir=None, show=True):
    rows = []
    for yr in years:
        try: df = load_predictions(pred_dir, year=yr)
        except FileNotFoundError: continue
        if lat is not None and lon is not None:
            df["d"] = (df["latitude"]-lat).abs()+(df["longitude"]-lon).abs()
            n = df.loc[df["d"].idxmin()]
            sub = df[(df["latitude"]==n["latitude"])&(df["longitude"]==n["longitude"])]
            rows.append({"year": yr, "CO2": sub["CO2_predicted"].agg(agg)})
        else:
            rows.append({"year": yr, "CO2": df["CO2_predicted"].agg(agg)})
    ts = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(ts["year"], ts["CO2"], "o-", color="#6a51a3", lw=2.5,
            markersize=6, mfc="white", mew=2)
    ax.fill_between(ts["year"], ts["CO2"], alpha=0.1, color="#6a51a3")
    ax.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"{agg.capitalize()} CO\u2082", fontsize=13, fontweight="bold")
    ax.set_title("CONUS Predicted Groundwater CO\u2082", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"co2_ts_{years[0]}-{years[-1]}.png"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    if show: plt.show()
    return fig, ts


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Map predicted groundwater CO2 over CONUS")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--date", default=None)
    parser.add_argument("--month", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--diff-year", type=int, default=None)
    parser.add_argument("--seasonal", action="store_true")
    parser.add_argument("--timeseries", nargs=2, type=int,
                        metavar=("START", "END"))
    parser.add_argument("--agg", default="mean",
                        choices=["mean", "median", "min", "max", "std"])
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-states", action="store_true",
                        help="Hide US state boundary lines")
    args = parser.parse_args()
    show = not args.no_show
    states = not args.no_states

    if args.seasonal and args.year:
        map_seasonal(args.pred_dir, args.year, agg=args.agg,
                     log_scale=args.log_scale, vmin=args.vmin, vmax=args.vmax,
                     save_dir=args.save_dir, show=show,
                     show_states=states)
    elif args.timeseries:
        yrs = list(range(args.timeseries[0], args.timeseries[1] + 1))
        map_timeseries(args.pred_dir, yrs, agg=args.agg,
                       save_dir=args.save_dir, show=show)
    elif args.date or args.month or args.year is not None:
        map_co2(args.pred_dir, date=args.date, month=args.month,
                year=args.year, diff_year=args.diff_year,
                agg=args.agg, log_scale=args.log_scale,
                vmin=args.vmin, vmax=args.vmax,
                save_path=args.save_path, save_dir=args.save_dir,
                show=show, show_states=states)
    else:
        parser.print_help()