# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 16:40:43 2025

@author: Fatih DİKBAŞ
"""

# SPDX-License-Identifier: GPL-3.0-only
#
# Groundwater Drought Dynamics in California — SGI/Event/Trend/GW-NDSPI Workflow
# Copyright (c) 2025 Fatih Dikbaş
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# Project repo: https://github.com/fdikbas/GWDroughtKonya
# Suggested citation (software):
#   Dikbaş, F. (2025). GWDroughtKonya: Python-based groundwater drought analysis for Konya [Source code].
#   GitHub. https://github.com/fdikbas/GWDroughtKonya

"""
Konya GW — ALL-IN-ONE analysis, figures and study-area map (EN labels + _EN folders)

Outputs (English labels; _EN folders)
  • Tables (CSV): out_tables/*, plus drought_metrics_yearly.csv, drought_clusters_dbscan.csv
  • Publication-ready TIFFs (400 dpi): out_figs_EN/*.tiff
  • Spatial maps of yearly drought metrics: out_SGI_spatial_EN/*.tiff
  • Study-area map (wells on OSM basemap, no title): out_figs_EN/fig0_study_area_map.tiff
  • SGI monthly time-series for each well: out_SGI_plots_EN/SGI_series_<WELL>.tiff
"""

from pathlib import Path
from typing import Optional
from matplotlib import patches
from matplotlib.transforms import Bbox
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import kendalltau
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib as mpl
import os, re
from tqdm import tqdm

# global defaults for all progress bars
import sys
TQDM_KW = {
    "ncols": 80,
    "dynamic_ncols": True,
    "leave": True,              # bittiğinde satır kalsın
    "smoothing": 0.1,
    "mininterval": 0.2,
    "miniters": 1,
    "ascii": None,              # True dersen ASCII bar; None otomatik
    "file": sys.stdout,         # Spyder/IPython için kritik
    "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
}

def _tqdm(iterable, **overrides):
    """Merge TQDM_KW with per-call overrides safely (no duplicate kwargs)."""
    kw = {**TQDM_KW, **overrides}
    return tqdm(iterable, **kw)

# --- Optional geospatial stack (for the study-area basemap figure) ---
try:
    import geopandas as gpd
    from shapely.geometry import Point
    import contextily as cx
    HAS_GEO = True
except Exception:
    HAS_GEO = False

# === Projection & cartography helpers (EPSG:3310, topo base, hillshade, scalebar, N arrow) ===
CA_CRS = "EPSG:3310"   # California Albers (meters)
WGS84  = "EPSG:4326"

def to_3310_coords_df(coords_df: pd.DataFrame) -> gpd.GeoDataFrame:
    gpts = gpd.GeoDataFrame(
        coords_df.copy(),
        geometry=[Point(float(x), float(y)) for x, y in zip(coords_df["X"], coords_df["Y"])],
        crs=WGS84
    ).to_crs(CA_CRS)
    out = coords_df.copy()
    out["X_3310"] = gpts.geometry.x.values
    out["Y_3310"] = gpts.geometry.y.values
    return out

def idw_grid_3310(x_m, y_m, z, xmin, xmax, ymin, ymax, nx=320, power=2):
    xg = np.linspace(xmin, xmax, nx)
    yg = np.linspace(ymin, ymax, nx)
    Xi, Yi = np.meshgrid(xg, yg)
    Zi = np.full_like(Xi, np.nan, dtype=float)
    xs = np.asarray(x_m, float); ys = np.asarray(y_m, float); zs = np.asarray(z, float)
    mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    xs, ys, zs = xs[mask], ys[mask], zs[mask]
    if xs.size < 3: return Xi, Yi, Zi, (xmin, xmax, ymin, ymax)
    for i in range(Yi.shape[0]):
        dy2 = (yg[i] - ys) ** 2
        for j in range(Xi.shape[1]):
            dx2 = (xg[j] - xs) ** 2
            d = np.sqrt(dx2 + dy2)
            if np.any(d == 0):
                Zi[i, j] = zs[d.argmin()]
            else:
                w = 1.0 / np.power(d, power)
                Zi[i, j] = np.nansum(w * zs) / np.nansum(w)
    return Xi, Yi, Zi, (xmin, xmax, ymin, ymax)

def idw_grid_3857(x_m, y_m, z, xmin, xmax, ymin, ymax, nx=320, power=2):
    return idw_grid_3310(x_m, y_m, z, xmin, xmax, ymin, ymax, nx=nx, power=power)
# (mevcut idw_grid_3310 zaten genel; istersen onu kullanıp sadece çağrıları 3857 isimli wrapper’dan yap)

# ---- contextily zoom helper (WebMerc / EPSG:3857 extents) ----
def _ctx_zoom_for_extent(extent, min_zoom=6, max_zoom=19):
    x0, x1, y0, y1 = map(float, extent)
    span = max(abs(x1 - x0), abs(y1 - y0))  # meters in EPSG:3857
    # simple, robust heuristic
    if span < 50_000:       z = 13
    elif span < 150_000:    z = 12
    elif span < 400_000:    z = 11
    elif span < 800_000:    z = 10
    else:                   z = 9
    return max(min(z, max_zoom), min_zoom)

# --- Map-only saver: tight_layout yok; bbox_inches ile kayıt ---
def save_map_tiff(fig, path, dpi=400):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os, re
    def _sanitize_filename(name: str) -> str:
        name = re.sub(r'[\\/:*?"<>|]+', "_", name).rstrip(" .")
        return name
    safe_name = _sanitize_filename(path.name)
    safe_path = path.with_name(safe_name)
    fig.savefig(os.fspath(safe_path), format="tiff", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def to_target_coords_df(coords_df: pd.DataFrame, target: str = "EPSG:3857") -> pd.DataFrame:
    import geopandas as gpd
    from shapely.geometry import Point
    src = _guess_crs_from_xy(coords_df)  # 4326 (lon/lat) ya da 32636 (UTM) dönecek
    gpts = gpd.GeoDataFrame(
        coords_df.copy(),
        geometry=[Point(float(x), float(y)) for x, y in zip(coords_df["X"], coords_df["Y"])],
        crs=src
    ).to_crs(target)
    out = coords_df.copy()
    out["X_3857"] = gpts.geometry.x.values
    out["Y_3857"] = gpts.geometry.y.values
    return out

def add_basemap_color_only(
    ax,
    crs="EPSG:3310",
    topo="Esri.WorldTopoMap",
    zoom=None,
    alpha=0.95,
):
    """Sadece renkli topo (veya fallback) ekler; hillshade YOK."""
    import contextily as cx

    def _resolve_provider(path: str):
        prov = cx.providers
        for part in str(path).split("."):
            prov = getattr(prov, part)
        return prov

    def _add_basemap_safe(ax, source, crs, zoom=None, **kwargs):
        if zoom is None:
            return cx.add_basemap(ax, source=source, crs=crs, **kwargs)
        else:
            return cx.add_basemap(ax, source=source, crs=crs, zoom=int(zoom), **kwargs)

    drawn = False
    # 1) Tercih edilen renkli topo
    try:
        prov_topo = _resolve_provider(topo)
        _add_basemap_safe(ax, source=prov_topo, crs=crs, zoom=zoom, attribution=False, alpha=alpha)
        drawn = True
    except Exception as e:
        print("[WARN] Topo basemap failed:", e)

    # 2) Fallback: Carto → OSM
    if not drawn:
        for fb in ("CartoDB.Voyager", "OpenStreetMap.Mapnik"):
            try:
                prov_fb = _resolve_provider(fb)
                _add_basemap_safe(ax, source=prov_fb, crs=crs, zoom=zoom, attribution=False, alpha=alpha)
                drawn = True
                break
            except Exception as e:
                print(f"[WARN] Basemap fallback failed ({fb}):", e)

    return drawn

def add_year_tag(ax, year, x=0.985, y=0.985):
    """Top-right year label inside the axes (survives cropping)."""
    txt = ax.text(x, y, f"{year}",
                  transform=ax.transAxes, ha="right", va="top",
                  fontsize=12, fontweight="bold",
                  color="rebeccapurple", zorder=20)
    try:
        import matplotlib.patheffects as pe
        txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground="lavender")])
    except Exception:
        pass
    return txt

def _safe_year_tag(ax, y):
    f = globals().get("add_year_tag", None)
    if callable(f):
        return f(ax, y)
    # fallback – pratikte çalışmazsa bile etiketi basar
    return ax.text(0.985, 0.985, f"{int(y)}",
                   transform=ax.transAxes, ha="right", va="top",
                   fontsize=12, fontweight="bold", color="rebeccapurple", zorder=50)

def export_wse_series(
    wse_wide: pd.DataFrame,
    out_dir: Path = Path("./out_WSE_series_EN"),
    fig_w_cm: float = 16.0,
    fig_h_cm: float = 8.0,
    add_roll_12: bool = True,
    add_roll_36: bool = True,
    add_baseline: bool = True,
) -> None:
    """
    Per-well Monthly DTW time series (publication style, same as California).
    - wse_wide: Monthly DTW (wide) with DatetimeIndex, columns=Well IDs (ft a.m.s.l.)
    - out_dir: output folder (created if absent)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _cm_to_inch(x_cm: float) -> float:
        return float(x_cm) / 2.54

    BG_FACE   = "#FAFAFA"   # panel background
    GRID_COL  = "#E6E6E6"   # grid color
    SPINE_COL = "#444444"   # left/bottom spines
    LINE_COL  = "steelblue"
    MARKER_FC = "lightskyblue"
    MARKER_EC = "darkblue"

    # rolling means (style aligned with California)
    ROLL12_COL    = "yellowgreen"
    ROLL36_COL    = "palevioletred"
    ROLL_ALPHA_12 = 0.95
    ROLL_ALPHA_36 = 0.95
    ROLL_LW_12    = 1.9
    ROLL_LW_36    = 2.1

    def _safe_ylim(yvals, pad_frac=0.06, fallback=(0.0, 1.0)):
        v = np.asarray(yvals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return fallback
        vmin, vmax = float(np.min(v)), float(np.max(v))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return fallback
        if vmax <= vmin:
            return (vmin - 0.5, vmax + 0.5)
        pad = (vmax - vmin) * pad_frac
        return (vmin - pad, vmax + pad)

    def _concise_date(ax):
        # yearly majors; quarterly minors; concise formatter
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        for label in ax.get_xticklabels():
            label.set(rotation=0, ha="center")

    # Ensure DatetimeIndex
    if not isinstance(wse_wide.index, pd.DatetimeIndex):
        try:
            wse_wide = wse_wide.copy()
            wse_wide.index = pd.to_datetime(wse_wide.index)
        except Exception:
            print("[WARN] export_wse_series: index is not datetime-like; skipping.")
            return

    # Sort by time (important for rolling and plotting)
    wse_wide = wse_wide.sort_index()

    for well in _tqdm(wse_wide.columns, desc="DTW time series (per well)"):
        try:
            s = pd.to_numeric(wse_wide[well], errors="coerce").dropna()
            if s.empty:
                continue

            # Optional overlays
            roll12 = s.rolling(12, min_periods=6).mean() if add_roll_12 else None
            roll36 = s.rolling(36, min_periods=12).mean() if add_roll_36 else None

            # Optional baseline: mean of first 3 full calendar years
            baseline_val = None
            if add_baseline:
                years = sorted(set(s.index.year))
                if len(years) >= 3:
                    first3 = years[:3]
                    ss = s[(s.index.year >= first3[0]) & (s.index.year <= first3[-1])]
                    if ss.notna().sum() >= 6:
                        baseline_val = float(ss.mean())

            y0, y1 = int(s.index.min().year), int(s.index.max().year)

            fig, ax = plt.subplots(figsize=(_cm_to_inch(fig_w_cm), _cm_to_inch(fig_h_cm)))
            ax.set_facecolor(BG_FACE)

            # Primary monthly series
            ax.plot(
                s.index, s.values,
                color=LINE_COL, linewidth=1.4, marker="o", markersize=2.4,
                markerfacecolor=MARKER_FC, markeredgecolor=MARKER_EC, markeredgewidth=0.6,
                label="Monthly DTW", zorder=5
            )

            # 12-month mean
            if add_roll_12 and roll12 is not None and roll12.notna().any():
                ax.plot(
                    roll12.index, roll12.values,
                    linewidth=ROLL_LW_12, alpha=ROLL_ALPHA_12,
                    color=ROLL12_COL, label="12-mo mean", zorder=3
                )

            # 36-month mean
            if add_roll_36 and roll36 is not None and roll36.notna().any():
                ax.plot(
                    roll36.index, roll36.values,
                    linewidth=ROLL_LW_36, alpha=ROLL_ALPHA_36,
                    color=ROLL36_COL, label="36-mo mean", zorder=4
                )

            # Baseline
            if baseline_val is not None and np.isfinite(baseline_val):
                ax.axhline(baseline_val, color="#8c8c8c", linewidth=1.1, linestyle="--",
                           label="Baseline (first 3 yrs)")

            # Grid & spines
            ax.grid(axis="y", color=GRID_COL, linewidth=0.8, alpha=1.0)
            ax.grid(axis="x", color=GRID_COL, linewidth=0.6, alpha=0.7)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            for side in ["left", "bottom"]:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(SPINE_COL)
                ax.spines[side].set_linewidth(1.0)

            # Ticks / formatters
            ax.tick_params(axis="both", which="major", length=4.0, width=0.9, colors="#222222")
            ax.tick_params(axis="both", which="minor", length=2.0, width=0.6, colors="#444444")
            _concise_date(ax)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune=None))

            # Labels & limits
            ax.set_xlabel("Year")
            ax.set_ylabel("Depth to groundwater (m bgs)")
            ax.set_title(f"DTW time series — {well} ({y0}–{y1})", pad=6)

            ymin, ymax = _safe_ylim(s.values, pad_frac=0.06)
            ax.set_ylim(ymin, ymax)
            ax.margins(x=0.01)

            # Legend (auto corner)
            ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8)

            # Save per well
            save_tiff(out_dir / f"DTW_series_{well}.tiff")
            plt.close(fig)

        except Exception as e:
            print(f"[WARN] Failed to export DTW series for well '{well}': {e}")


def add_scalebar(ax,
                 length_km: float = 50.0,
                 location: str | None = None,   # new name
                 loc: str | None = None,        # backward compatibility
                 pad_frac: float = 0.025,
                 height_frac: float = 0.012,
                 facecolor: str = "white",
                 edgecolor: str = "black",
                 text_color: str = "black",
                 lw: float = 0.9,
                 fontsize: int = 9):
    """
    Draw a simple metric scalebar on a map in EPSG:3857 (meters).
    Accepts both 'location=' and legacy 'loc='.
    Valid positions contain 'lower'/'upper' and 'left'/'right' (or 'center').
    """
    # resolve parameter naming
    if location is None and loc is not None:
        location = loc
    if location is None:
        location = "lower left"

    # current data extent (must be set *after* setting basemap/limits)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0 or dy <= 0:
        # nothing to draw against
        return

    # geometry in data units (EPSG:3857 meters)
    bar_len_m = float(length_km) * 1000.0
    bar_h = height_frac * dy
    pad_x = pad_frac * dx
    pad_y = pad_frac * dy

    # vertical placement
    if "lower" in location:
        y0 = ymin + pad_y
    else:
        y0 = ymax - pad_y - bar_h

    # horizontal placement
    if "left" in location:
        x0 = xmin + pad_x
    elif "right" in location:
        x0 = xmax - pad_x - bar_len_m
    else:  # center
        x0 = xmin + 0.5 * (dx - bar_len_m)

    # draw bar
    rect = patches.Rectangle((x0, y0), bar_len_m, bar_h,
                             facecolor=facecolor, edgecolor=edgecolor, lw=lw,
                             zorder=10)
    ax.add_patch(rect)

    # label (km)
    ax.text(x0 + bar_len_m / 2.0, y0 + bar_h * 1.2,
            f"{int(length_km)} km",
            ha="center", va="bottom",
            color=text_color, fontsize=fontsize,
            zorder=11)

def add_north_arrow(ax,
                    xy=(0.92, 0.14),
                    length=0.08,
                    color="black",
                    lw=1.2,
                    head_length=12,
                    fontsize=10,
                    loc=None):
    """
    North arrow in axes-fraction coordinates; independent of CRS.
    Accepts either 'xy=(x,y)' in axes fraction OR a convenience 'loc' like
    'upper left', 'upper right', 'lower left', 'lower right'.
    """
    if isinstance(loc, str):
        loc_map = {
            "upper left":  (0.12, 0.82),
            "upper right": (0.88, 0.82),
            "lower left":  (0.12, 0.12),
            "lower right": (0.88, 0.12),
        }
        xy = loc_map.get(loc.lower().strip(), xy)

    x, y = xy
    ax.annotate("",
                xy=(x, y + length), xytext=(x, y),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0,
                                mutation_scale=head_length),
                zorder=12)
    ax.text(x, y + length + 0.02, "N",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=fontsize, color=color, zorder=13)

def add_north_arrow_above_scalebar(ax,
                                   x=0.05,
                                   pad_frac=0.025,
                                   height_frac=0.012,
                                   extra=0.020,
                                   **kwargs):
    """Place a north arrow just above a lower-left scalebar."""
    y = float(pad_frac) + float(height_frac) + float(extra)
    add_north_arrow(ax, xy=(float(x), y), **kwargs)

def _sanitize_filename(name: str) -> str:
    # Remove characters illegal on Windows:  \ / : * ? " < > | and strip trailing dot/space
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = name.rstrip(" .")
    return name

def save_tiff(path: Path):
    path = Path(path)
    # 1) ensure parent exists
    path.parent.mkdir(parents=True, exist_ok=True)
    # 2) sanitize filename on Windows (no-ops on clean names)
    safe_name = _sanitize_filename(path.name)
    safe_path = path.with_name(safe_name)
    # 3) always hand PIL a plain str path (os.fspath is safest for path-likes)
    fname = os.fspath(safe_path)
    # 4) save
    plt.tight_layout()
    plt.savefig(fname, format="tiff", dpi=400, bbox_inches="tight")
    plt.close()

# === Utility: round a value to a given step (used by plot_sgi_series) ===
def _round_to_step(x, step=0.5, how="ceil"):
    """Round x to a multiple of `step` using the chosen mode.
    how: 'ceil' | 'floor' | 'round'
    """
    if x is None:
        return x
    try:
        # np.isfinite handles scalars/ndarrays robustly
        if not np.isfinite(x):
            return x
    except Exception:
        # if x is not numeric, just return it (defensive)
        return x

    q = x / step
    if how == "ceil":
        return step * np.ceil(q)
    elif how == "floor":
        return step * np.floor(q)
    else:
        return step * np.round(q)

# ---------------- PATHS & CONFIG ----------------
EXCEL_FILE = "well_data_imputed_SYNTHETIC.xlsx"        # must contain 'Yıl' and 'Ay'
COORD_FILE = "Konya.Well.Coordinates_SYNTHETIC.csv"
OPTIONAL_BOUNDARY_FILE: Optional[str] = None  # e.g., "konya_boundary.geojson" / ".shp"

# >>> All outputs now go to _EN folders <<<
OUT_FIGS = Path("./out_figs_EN")
OUT_TBLS = Path("./out_tables_EN")
OUT_SPATIAL = Path("./out_SGI_spatial_EN")
OUT_SGI_SERIES = Path("./out_SGI_plots_EN")
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_TBLS.mkdir(parents=True, exist_ok=True)
OUT_SPATIAL.mkdir(parents=True, exist_ok=True)
OUT_SGI_SERIES.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2016-01-01")
END   = pd.Timestamp("2024-12-31")

# ---- Global default for basemap provider ----
try:
    BASEMAP_PROVIDER  # existing?
except NameError:
    BASEMAP_PROVIDER = "OpenStreetMap.Mapnik"

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 400,
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.figsize": (6.0, 3.8),
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- Colorbar kenarını (outline) net ve tam çizmek için yardımcı ---
def _style_colorbar_fat_outline(cbar):
    try:
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1.2)
    except Exception:
        pass
    # Bazı stillerde spines kapalı olabiliyor; açıp kalınlık verelim
    for side in ("left", "right", "top", "bottom"):
        spine = getattr(cbar.ax.spines, side, None)
        if spine is not None:
            try:
                cbar.ax.spines[side].set_visible(True)
                cbar.ax.spines[side].set_linewidth(1.2)
                cbar.ax.spines[side].set_edgecolor("black")
            except Exception:
                pass
    cbar.ax.tick_params(length=4, width=1.0)
    return cbar

def add_year_tag(ax, year, x=0.985, y=0.985, color="rebeccapurple"):
    """
    Eksen içine sağ-üst yıl etiketi (crop sonrası da görünür).
    """
    txt = ax.text(
        float(x), float(y), f"{int(year)}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12, fontweight="bold", color=color, zorder=50
    )
    try:
        import matplotlib.patheffects as pe
        txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground="lavender")])
    except Exception:
        pass
    return txt

def _colorbar_same_height(ax, im, label=None, ticks=None, outline=True):
    # a slim cbar that matches the axes height 1:1
    cax = inset_axes(
        ax, width="3.2%", height="100%",  # same height as axes
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),  # shift to the right of axes
        bbox_transform=ax.transAxes,
        borderpad=0.0
    )
    cb = plt.colorbar(im, cax=cax)
    if ticks is not None:
        cb.set_ticks(ticks)
    if label:
        cb.set_label(label)
    if outline:
        try:
            cb.outline.set_visible(True)
            cb.outline.set_linewidth(1.0)
            cb.outline.set_edgecolor("black")
        except Exception:
            pass
    return cb

# ==========================================================
# Güvenli renk ölçeği seçici (tek-kutuplu ↔ iki-kutuplu otomatik)
# ==========================================================
def _safe_diverging_norm_and_cmap(data_min, data_max, p2=None, p98=None):
    """
    Veri aralığına göre uygun norm/cmap döndürür.
      - Tümü >= 0 ise: Normalize(0..vmax), 'YlOrRd'
      - Tümü <= 0 ise: Normalize(vmin..0), 'YlGnBu'
      - Karışık işaretli ise: TwoSlopeNorm(vmin<0<vmax), 'RdBu'
    Döndürür: norm, cmap (mpl colormap objesi), vmin, vmax
    """
    # Robust sınırları uygula (opsiyonel)
    vmin = float(data_min) if p2 is None or not np.isfinite(p2) else float(min(p2, data_min))
    vmax = float(data_max) if p98 is None or not np.isfinite(p98) else float(max(p98, data_max))

    # Tek-kutuplu (tamamı pozitif veya sıfır)
    if vmin >= 0:
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        cmap = mpl.colormaps.get("YlOrRd")
        return norm, cmap, 0.0, vmax

    # Tek-kutuplu (tamamı negatif veya sıfır)
    if vmax <= 0:
        if not np.isfinite(vmin) or vmin >= 0:
            vmin = -1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=0.0)
        cmap = mpl.colormaps.get("YlGnBu")
        return norm, cmap, vmin, 0.0

    # Karışık işaretli → diverging, 0 merkezli
    eps = 1e-9
    if vmin >= 0: vmin = -eps
    if vmax <= 0: vmax = +eps
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or (vmax - vmin) <= 1e-12:
        vmin, vmax = -1.0, 1.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = mpl.colormaps.get("RdBu")
    return norm, cmap, vmin, vmax

# ---------------- HELPERS ----------------
def _norm_well_id(x) -> str:
    """Normalize well codes: trim, remove BOM, handle '62555.0' → '62555'."""
    if pd.isna(x):
        return ""
    s = str(x).strip().replace("\u200b", "")
    s2 = s.replace(",", ".")
    try:
        f = float(s2)
        if np.isfinite(f) and abs(f - round(f)) < 1e-12:
            return str(int(round(f)))
        return s
    except Exception:
        return s

def _order_wells_alpha(cols) -> list[str]:
    """
    Wells on Y-axis: top→bottom alphabetically ascending (string sort).
    Works with numeric-looking IDs as strings too.
    Returns only columns that actually exist.
    """
    cols = [str(c) for c in cols]
    return sorted(cols)

# === Loader (Yıl + Ay; Turkish month names handled; YEAR FILL) ===
def turkish_month_to_int(month):
    MONTHS = {
        "Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4, "Mayıs": 5,
        "Haziran": 6, "Temmuz": 7, "Ağustos": 8, "Eylül": 9, "Ekim": 10,
        "Kasım": 11, "Aralık": 12
    }
    try:
        return int(month)
    except Exception:
        return MONTHS.get(str(month).strip(), None)

def build_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Build monthly time series exactly like your working snippet."""
    df.columns = [str(c).strip() for c in df.columns]
    if "Yıl" not in df.columns or "Ay" not in df.columns:
        raise ValueError("Excel must have columns 'Yıl' and 'Ay'")
    df["Yıl"] = df["Yıl"].ffill()
    df["Ay"] = df["Ay"].apply(turkish_month_to_int)
    df["Tarih"] = pd.to_datetime(
        dict(
            year=pd.to_numeric(df["Yıl"], errors="coerce"),
            month=pd.to_numeric(df["Ay"], errors="coerce"),
            day=1,
        ),
        errors="coerce"
    )
    df = df.dropna(subset=["Tarih"])
    ts_df = df.drop(columns=["Yıl", "Ay"]).set_index("Tarih")
    ts_df = ts_df.apply(pd.to_numeric, errors="coerce")
    ts_df.columns = [_norm_well_id(c) for c in ts_df.columns]
    ts_df = ts_df.sort_index().asfreq("MS")
    return ts_df

# === SGI (z-score) ===
def calculate_sgi(ts_df: pd.DataFrame) -> pd.DataFrame:
    return (ts_df - ts_df.mean()) / ts_df.std()

def extract_drought_events(series, thr=-1.0):
    events, in_evt = [], False
    start, sev, dur = None, 0.0, 0
    for date, val in series.items():
        if pd.notna(val) and val < thr:
            if not in_evt:
                in_evt = True
                start, sev, dur = date, 0.0, 0
            sev += abs(val); dur += 1
        else:
            if in_evt:
                events.append({"well": series.name, "start": start,
                               "end": date - pd.offsets.MonthBegin(1),
                               "duration_mo": dur, "severity_sgi": sev})
                in_evt = False
    if in_evt:
        events.append({"well": series.name, "start": start,
                       "end": series.index[-1], "duration_mo": dur, "severity_sgi": sev})
    return events

def drought_metrics_yearly(sgi_df: pd.DataFrame, threshold=-1.0) -> pd.DataFrame:
    rows = []
    for well in sgi_df.columns:
        s = sgi_df[well].dropna()
        if s.empty:
            continue
        for year in sorted(set(s.index.year)):
            s_year = s[s.index.year == year]
            mask = s_year < threshold
            max_dur, curr, ev_count = 0, 0, 0
            for flag in mask:
                if flag:
                    curr += 1
                else:
                    if curr > 0:
                        max_dur = max(max_dur, curr); ev_count += 1; curr = 0
            if curr > 0:
                max_dur = max(max_dur, curr); ev_count += 1
            min_sgi = s_year[mask].min() if mask.any() else np.nan
            cum_def = float((-s_year[mask]).sum()) if mask.any() else 0.0
            rows.append({
                "Well": _norm_well_id(well),
                "Year": int(year),
                "MaxDroughtDuration": int(max_dur),
                "MinSGI": float(min_sgi) if pd.notna(min_sgi) else np.nan,
                "CumulativeDeficit": cum_def,
                "NumEvents": int(ev_count)
            })
    return pd.DataFrame(rows)

def mann_kendall_sen(series_monthly: pd.Series):
    s = series_monthly.dropna()
    if len(s) < 10:
        return np.nan, np.nan, np.nan
    x = np.arange(len(s))
    tau, p = kendalltau(x, s.values)
    n = len(s); vals = s.values
    slopes = []
    for i in range(n - 1):
        dv = (vals[i + 1:] - vals[i]) / (np.arange(i + 1, n) - i)
        slopes.extend(dv.tolist())
    sen_month = np.median(slopes)
    return tau, p, sen_month * 12.0

def cluster_drought_periods(sgi_series: pd.Series, threshold=-1.0, eps=2, min_samples=2):
    idx = np.where(sgi_series < threshold)[0]
    if len(idx) < min_samples:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(idx.reshape(-1,1)).labels_
    clusters = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        members = idx[labels == lab]
        start = sgi_series.index[members[0]]
        end   = sgi_series.index[members[-1]]
        clusters.append((start, end, len(members)))
    return clusters

# ---------------- SPATIAL / STUDY-AREA MAP ----------------
def _read_csv_autosep(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        for sep in [",", ";", "\t", r"\s+"]:
            try:
                return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")
            except Exception:
                continue
    raise ValueError(f"Could not read CSV with common separators: {path}")

def read_well_coordinates(path: Path) -> pd.DataFrame:
    # Columnar layout (preferred)
    try:
        df = _read_csv_autosep(path)
        cols = [str(c).strip().replace("\u200b", "") for c in df.columns]
        df.columns = cols
        cand_well = [c for c in cols if c.lower() in
                     {"well","kuyu","kuyu_no","kuyuno","station","id","code","well_id","wellcode"}]
        cand_x = [c for c in cols if c.lower() in
                  {"x","easting","lon","longitude","utm_x","x_utm","xcoord","xcoord_utm"}]
        cand_y = [c for c in cols if c.lower() in
                  {"y","northing","lat","latitude","utm_y","y_utm","ycoord","ycoord_utm"}]
        cand_z = [c for c in cols if c.lower() in
                  {"groundelev","elev","elevation","z","height","elev_m","ground_elev"}]
        if cand_well:
            wcol = cand_well[0]
            xcol = cand_x[0] if cand_x else None
            ycol = cand_y[0] if cand_y else None
            zcol = cand_z[0] if cand_z else None
            out = pd.DataFrame({
                "Well": df[wcol].apply(_norm_well_id),
                "X": pd.to_numeric(df[xcol].astype(str).str.replace(",", ".", regex=False),
                                   errors="coerce") if xcol else np.nan,
                "Y": pd.to_numeric(df[ycol].astype(str).str.replace(",", ".", regex=False),
                                   errors="coerce") if ycol else np.nan,
                "GroundElev": pd.to_numeric(df[zcol].astype(str).str.replace(",", ".", regex=False),
                                            errors="coerce") if zcol else np.nan
            })
            out = out.dropna(subset=["Well"]).reset_index(drop=True)
            if len(out) > 0:
                return out
    except Exception:
        pass

    # 4-row transposed fallback
    df2 = pd.read_csv(path, header=None, sep=None, engine="python", encoding="utf-8-sig")
    if df2.shape[0] >= 4 and df2.shape[1] >= 2:
        wells = [_norm_well_id(df2.iloc[0, i]) for i in range(1, df2.shape[1])]
        gelev = [str(df2.iloc[1, i]) for i in range(1, df2.shape[1])]
        xs    = [str(df2.iloc[2, i]) for i in range(1, df2.shape[1])]
        ys    = [str(df2.iloc[3, i]) for i in range(1, df2.shape[1])]
        out = pd.DataFrame({
            "Well": wells,
            "X": pd.to_numeric(pd.Series(xs).str.replace(",", ".", regex=False), errors="coerce"),
            "Y": pd.to_numeric(pd.Series(ys).str.replace(",", ".", regex=False), errors="coerce"),
            "GroundElev": pd.to_numeric(pd.Series(gelev).str.replace(",", ".", regex=False), errors="coerce"),
        }).dropna(subset=["Well"]).reset_index(drop=True)
        if len(out) > 0:
            return out

    raise ValueError(f"Could not parse coordinates file layout: {path}")

def _guess_crs_from_xy(df_coords: pd.DataFrame) -> str:
    x = df_coords["X"].astype(float)
    y = df_coords["Y"].astype(float)
    if x.abs().max() <= 180 and y.abs().max() <= 90:
        return "EPSG:4326"   # lon/lat
    return "EPSG:32636"      # UTM Zone 36N

def _draw_north_arrow(ax, xy=(0.05, 0.95), size=0.06):
    ax.annotate('N', xy=xy, xytext=(xy[0], xy[1]-size),
                xycoords='axes fraction', textcoords='axes fraction',
                ha='center', va='center', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', lw=1.2, color='k'))

def _draw_scale_bar(ax, length_km=10):
    try:
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        xm = x0 + 0.05*(x1-x0); ym = y0 + 0.07*(y1-y0)
        length_m = length_km * 1000.0
        ax.plot([xm, xm+length_m], [ym, ym], color='k', lw=2)
        ax.plot([xm, xm], [ym-0.01*(y1-y0), ym+0.01*(y1-y0)], color='k', lw=2)
        ax.plot([xm+length_m, xm+length_m], [ym-0.01*(y1-y0), ym+0.01*(y1-y0)], color='k', lw=2)
        ax.text(xm + length_m/2, ym + 0.012*(y1-y0), f"{length_km} km",
                ha='center', va='bottom', fontsize=8)
    except Exception:
        pass

def plot_study_area_map(coords_df: pd.DataFrame,
                        boundary_path: Optional[str],
                        provider: str,
                        out_path: Path):
    """OSM basemap (+ optional boundary), no title, with north arrow & scale bar."""
    if not HAS_GEO:
        print("[INFO] geopandas/contextily not available; drawing simple scatter without basemap.")
        plt.figure(figsize=(7.2, 5.6))
        plt.scatter(coords_df["X"], coords_df["Y"], s=30, edgecolor='black')
        for _, r in coords_df.iterrows():
            plt.annotate(str(r["Well"]), (r["X"], r["Y"]), xytext=(3,3),
                         textcoords="offset points", fontsize=7)
        plt.xlabel("X"); plt.ylabel("Y")
        save_tiff(out_path); return

    crs_in = _guess_crs_from_xy(coords_df)
    gdf = gpd.GeoDataFrame(coords_df.copy(),
                           geometry=[Point(xy) for xy in zip(coords_df["X"], coords_df["Y"])],
                           crs=crs_in)

    boundary = None
    if boundary_path:
        try:
            boundary = gpd.read_file(boundary_path)
            if boundary.crs is None:
                boundary = boundary.set_crs("EPSG:4326", allow_override=True)
        except Exception as e:
            print(f"[WARN] Boundary could not be read: {e}")

    gdf3857 = gdf.to_crs(epsg=3857)
    if boundary is not None:
        boundary3857 = boundary.to_crs(epsg=3857)
        bounds = boundary3857.total_bounds
    else:
        bounds = gdf3857.total_bounds

    xmin, ymin, xmax, ymax = bounds
    dx = (xmax - xmin) * 0.10; dy = (ymax - ymin) * 0.10

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    if boundary is not None:
        boundary3857.plot(ax=ax, facecolor="none", edgecolor="k", linewidth=1.0)

    gdf3857.plot(ax=ax, color="royalblue", edgecolor="black", markersize=20, alpha=0.9)
    for _, r in gdf3857.iterrows():
        ax.annotate(str(r["Well"]), (r.geometry.x, r.geometry.y),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)

    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_xlim(xmin - dx, xmax + dx); ax.set_ylim(ymin - dy, ymax + dy)
    ax.tick_params(labelleft=False, labelbottom=False)
    add_scalebar(ax, length_km=10, location="lower left")
    add_north_arrow_above_scalebar(ax, length=0.08, fontsize=10, lw=1.2, head_length=12)
    save_tiff(out_path)
    
    # === Türkiye locator inset (400 dpi) ===
    try:
        from shapely.geometry import box

        # --- Havza bbox'ı (EPSG:3857) ---
        if boundary is not None:
            bminx, bminy, bmaxx, bmaxy = boundary3857.total_bounds
        else:
            bminx, bminy, bmaxx, bmaxy = gdf3857.total_bounds
        basin_rect_3857 = gpd.GeoSeries([box(bminx, bminy, bmaxx, bmaxy)], crs=3857)

        # --- Türkiye: Natural Earth (offline garanti) ---
        tr3857 = None
        try:
            world_path = gpd.datasets.get_path("naturalearth_lowres")
            world = gpd.read_file(world_path)
            if "iso_a3" in world.columns:
                tr = world[world["iso_a3"].str.upper().eq("TUR")]
            else:
                tr = world[world["name"].str.lower().eq("turkey")]
            if tr.empty and "name" in world.columns:
                tr = world[world["name"].str.contains("Turk", case=False, na=False)]
            if not tr.empty:
                tr3857 = tr.to_crs(3857)
                tr_minx, tr_miny, tr_maxx, tr_maxy = tr3857.total_bounds
            else:
                raise ValueError("Turkey polygon not found.")
        except Exception:
            tr_bb_4326 = gpd.GeoSeries([box(25.0, 35.5, 45.0, 42.5)], crs=4326).to_crs(3857)
            tr_minx, tr_miny, tr_maxx, tr_maxy = tr_bb_4326.iloc[0].bounds

        # Kenarlarda küçük tampon
        pad_x = (tr_maxx - tr_minx) * 0.03
        pad_y = (tr_maxy - tr_miny) * 0.03
        tr_minx -= pad_x; tr_maxx += pad_x
        tr_miny -= pad_y; tr_maxy += pad_y

        # --- 400 dpi kare figür ---
        fig2, ax2 = plt.subplots(figsize=(2.6, 2.6), dpi=400)  # <-- 400 dpi

        # Önce görünümü ayarla
        ax2.set_xlim(tr_minx, tr_maxx)
        ax2.set_ylim(tr_miny, tr_maxy)

        # OSM basemap (opsiyonel) – altta kalsın ve atfı kapat
        try:
            cx.add_basemap(
                ax2,
                source=cx.providers.OpenStreetMap.Mapnik,
                zoom=5,
                crs="EPSG:3857",
                zorder=-1,
                attribution=False   # <-- telif yazısını gösterme
            )
        except Exception:
            pass

        # Türkiye poligonu (offline garanti)
        if tr3857 is not None:
            tr3857.plot(ax=ax2, color="#e6e6e6", edgecolor="#666666", linewidth=0.6, zorder=1)

        # Havza DİKDÖRTGENİ: çok ince çizgi
        basin_rect_3857.plot(ax=ax2, facecolor="none", edgecolor="red", linewidth=0.6, zorder=10)

        # Temizlik
        ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_linewidth(0.8); sp.set_color("#333333")

        # --- Kayıt: 400 dpi ---
        loc_path = out_path.parent / "fig0_turkey_locator.tiff"
        plt.savefig(loc_path, dpi=400, format="tiff", bbox_inches="tight")  # <-- 400 dpi
        plt.close(fig2)
        print("[OK] Turkey locator inset saved ->", loc_path)

    except Exception as e:
        print("[WARN] Turkey locator inset could not be created:", e)

# === SGI severity shading (±max_abs, step=0.5; alpha +0.05/step) ===
def fill_sgi_severity_background(ax, max_abs=5.0, step=0.5, alpha_step=0.05,
                                 dry_color="tomato", wet_color="cornflowerblue"):
    dry_rgb = mcolors.to_rgb(dry_color)
    wet_rgb = mcolors.to_rgb(wet_color)

    n_steps = int(np.ceil(max_abs / step))

    # Negatiften 0'a (daha negatif → daha opak)
    for i in range(n_steps):
        y0 = -(i + 1) * step
        y1 = -i * step
        alpha = float(np.clip((i + 1) * alpha_step, 0.0, 0.6))
        ax.axhspan(y0, y1, facecolor=dry_rgb, alpha=alpha, zorder=0)

    # 0'dan pozitife (daha büyük → daha opak)
    for i in range(n_steps):
        y0 = i * step
        y1 = (i + 1) * step
        alpha = float(np.clip((i + 1) * alpha_step, 0.0, 0.6))
        ax.axhspan(y0, y1, facecolor=wet_rgb, alpha=alpha, zorder=0)

def plot_sgi_series(sgi_df, well, drought_threshold=-1.0, out_dir=None):
    # Seriyi al
    sgi = sgi_df[well]

    # --- Dinamik y-aralığı belirle ---
    # Veri min–maks + eşik dikkate alınır; 0.5'lik kademelere yuvarlanır
    data_min = np.nanmin([sgi.min(), drought_threshold,  0.0])
    data_max = np.nanmax([sgi.max(), drought_threshold,  0.0])

    # güvenli tampon
    span = data_max - data_min
    pad = 0.1 * span if (np.isfinite(span) and span > 0) else 0.5

    ymin_raw = data_min - pad
    ymax_raw = data_max + pad

    # 0.5 adımlı yumuşak sınırlar
    ymin = _round_to_step(ymin_raw, step=0.5, how="floor")
    ymax = _round_to_step(ymax_raw, step=0.5, how="ceil")
    if not (np.isfinite(ymin) and np.isfinite(ymax)) or (ymax <= ymin):
        ymin, ymax = -2.5, 2.5  # emniyetli bir varsayılan

    # Arka plan için ±max_abs (0 merkezli kapsayıcı)
    max_abs = _round_to_step(max(abs(ymin), abs(ymax)), step=0.5, how="ceil")

    # --- Çizim ---
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Arka plan şeritleri (dinamik ±max_abs)
    fill_sgi_severity_background(ax, max_abs=max_abs, step=0.5, alpha_step=0.05)

    # SGI çizgisi (arkada)
    ax.plot(
        sgi.index, sgi.values,
        label="SGI",
        color="cornflowerblue",
        linewidth=3,
        zorder=1
    )

    # Referans çizgileri (arkada)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.axhline(
        drought_threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Drought threshold (SGI<{drought_threshold})",
        zorder=1
    )

    # Kuraklık noktaları (önde, çerçeveli)
    drought_mask = sgi < drought_threshold
    ax.scatter(
        sgi.index[drought_mask],
        sgi.values[drought_mask],
        facecolors="lightcoral",
        edgecolors="firebrick",
        linewidths=0.6,
        s=30,
        label="Drought",
        zorder=2
    )

    # Eksen/legend
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("SGI (Standardized Groundwater Level)")
    ax.grid(alpha=0.3)
    ax.legend(title=f"Well: {well}")
    fig.tight_layout()

    # Kaydet
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"SGI_series_{well}.tiff"
    fig.savefig(out_path, dpi=400, format="tiff")
    plt.close(fig)

# ---------------- INTERPOLATED ANNUAL GW LEVEL MAPS ----------------
def _idw_grid(x, y, z, xmin, xmax, ymin, ymax, nx=300, power=2, eps=1e-12):
    """
    Inverse Distance Weighting (IDW) on a regular grid.
    x,y,z: 1D arrays of known points (z = values to interpolate)
    Returns (Xi, Yi, Zi, extent)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    xr = xmax - xmin
    yr = ymax - ymin
    nx = int(nx)
    ny = max(2, int(round(nx * (yr / xr)))) if xr > 0 else nx

    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    Xi, Yi = np.meshgrid(xi, yi)

    dx = Xi[None, :, :] - x[:, None, None]
    dy = Yi[None, :, :] - y[:, None, None]
    dist = np.hypot(dx, dy) + eps
    w = 1.0 / (dist ** power)

    valid = np.isfinite(z)
    if valid.sum() < 3:
        return Xi, Yi, np.full_like(Xi, np.nan), [xmin, xmax, ymin, ymax]

    w = w[valid]
    zv = z[valid][:, None, None]
    Zi = np.nansum(w * zv, axis=0) / np.nansum(w, axis=0)

    return Xi, Yi, Zi, [xmin, xmax, ymin, ymax]

def _interp_grid_safe(x, y, v, xmin, xmax, ymin, ymax, nx=None, power=None):
    """
    Güvenli enterpolasyon çağrısı:
    1) _interp_grid varsa onu dener,
    2) yoksa _idw_grid’i çağırır,
    3) ikisi de yoksa minimal yerel IDW uygular.
    Geri dönüş: (Xi, Yi, Zi, [xmin, xmax, ymin, ymax])
    """
    import numpy as np

    nx = int(globals().get("NX_IDW", 320) if nx is None else nx)
    power = int(globals().get("IDW_PWR", 2) if power is None else power)

    # 1) _interp_grid varsa ve çağrılabilir ise
    _ig = globals().get("_interp_grid", None)
    if callable(_ig):
        try:
            return _ig(x, y, v, xmin, xmax, ymin, ymax, nx=nx, power=power)
        except Exception:
            pass  # sessizce fallback

    # 2) _idw_grid varsa ve çağrılabilir ise
    _idw = globals().get("_idw_grid", None)
    if callable(_idw):
        return _idw(x, y, v, xmin, xmax, ymin, ymax, nx=nx, power=power)

    # 3) Minimal yerel IDW fallback (küçük veri kümeleri için yeterli)
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(v, float)
    xr = float(xmax - xmin); yr = float(ymax - ymin)
    ny = max(2, int(round(nx * (yr / xr)))) if xr > 0 else nx
    xi = np.linspace(xmin, xmax, nx); yi = np.linspace(ymin, ymax, ny)
    Xi, Yi = np.meshgrid(xi, yi)

    # uzaklık-ağırlıklı ortalama
    eps = 1e-12
    dx = Xi[None, :, :] - x[:, None, None]
    dy = Yi[None, :, :] - y[:, None, None]
    dist = np.hypot(dx, dy) + eps

    ok = np.isfinite(z)
    if ok.sum() < 3:
        return Xi, Yi, np.full_like(Xi, np.nan, dtype=float), [xmin, xmax, ymin, ymax]

    w = 1.0 / (dist[ok] ** power)
    zv = z[ok][:, None, None]
    Zi = np.nansum(w * zv, axis=0) / np.nansum(w, axis=0)
    return Xi, Yi, Zi, [xmin, xmax, ymin, ymax]

def _round_to_base(x, base=2, how="floor"):
    if not np.isfinite(x):
        return np.nan
    if how == "floor":
        return base * np.floor(x / base)
    elif how == "ceil":
        return base * np.ceil(x / base)
    else:
        return base * np.round(x / base)

# ==== EXTREMES: GEV fit for yearly minimum SGI =================================
from scipy.stats import genextreme as _gev
def fit_gev_return_periods_min_sgi(sgi_df: pd.DataFrame,
                                   years=(10, 25, 50),
                                   b: int = None,
                                   random_state: int = 42,
                                   n_jobs: int | None = None) -> pd.DataFrame:
    """
    Aylık SGI -> yıllık min -> GEV -> T-yıl eşikleri. 
    - Tek satırlık ilerleme: sadece "GEV (per well)" çubuğu.
    - Bootstrap sayısı: GEV_BOOTSTRAP_B (yoksa 120).
    - İsteğe bağlı paralel bootstrap (joblib varsa).
    """
    from scipy.stats import genextreme as _gev

    if b is None:
        b = int(globals().get("GEV_BOOTSTRAP_B", 120))  # 150→120 (hız/kararlılık dengesi)

    # paralellik
    if n_jobs is None:
        try:
            import os
            n_jobs = max(1, (os.cpu_count() or 2) - 1)
        except Exception:
            n_jobs = 1

    # tarih indeksini garanti et
    if not isinstance(sgi_df.index, pd.DatetimeIndex):
        sgi_df = sgi_df.copy()
        sgi_df.index = pd.to_datetime(sgi_df.index, errors="coerce")
        sgi_df = sgi_df.loc[sgi_df.index.notna()]

    ygrp = sgi_df.groupby(sgi_df.index.year)
    wells = list(sgi_df.columns)

    out = []
    rng = np.random.default_rng(random_state)

    # Tek satırlık dış bar
    for w in _tqdm(wells, desc="GEV (per well)"):
        s = pd.to_numeric(sgi_df[w], errors="coerce")
        amin = ygrp[w].min().dropna()
        if amin.size < 8:
            continue

        try:
            c, loc, scale = _gev.fit(amin.values)
        except Exception:
            continue

        # --- Bootstrap (isteğe bağlı joblib)
        boot_params = []
        def _one_boot(_):
            res = rng.choice(amin.values, size=len(amin), replace=True)
            return _gev.fit(res)

        try:
            from joblib import Parallel, delayed
            if n_jobs > 1:
                boot_params = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_one_boot)(i) for i in range(b)
                )
            else:
                for i in range(b):
                    boot_params.append(_one_boot(i))
        except Exception:
            # joblib yoksa/başarısızsa: seri
            for i in range(b):
                boot_params.append(_one_boot(i))

        for Ty in years:
            q = 1.0 - 1.0/float(Ty)
            xT = _gev.ppf(q, c, loc=loc, scale=scale)
            lo = hi = np.nan
            if boot_params:
                xs = []
                for (cb, lb, sb) in boot_params:
                    if np.isfinite(sb):
                        xs.append(_gev.ppf(q, cb, loc=lb, scale=sb))
                if xs:
                    xs = np.asarray(xs, float)
                    lo, hi = np.nanpercentile(xs, [5, 95])
            out.append({
                "Well": _norm_well_id(w),
                "T_year": int(Ty),
                "xT_minSGI": float(xT),
                "xT_lo5": float(lo) if np.isfinite(lo) else np.nan,
                "xT_hi95": float(hi) if np.isfinite(hi) else np.nan
            })

    return pd.DataFrame(out)

# ==== MULTIPLE TESTING: FDR BH ============================================
def fdr_bh(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return np.zeros_like(p, dtype=bool)
    idx = np.argsort(p[mask])
    ranked = p[mask][idx]
    m = float(mask.sum())
    thresh = alpha * (np.arange(1, int(m)+1) / m)
    passed = ranked <= thresh
    if passed.any():
        k = np.where(passed)[0].max() + 1
        cutoff = ranked[k-1]
        out = np.zeros_like(p, dtype=bool); out[mask] = p[mask] <= cutoff
        return out
    else:
        return np.zeros_like(p, dtype=bool)

# ==== CHANGE-POINT: Pettitt test (nonparametric) ==========================
def pettitt_test_series(x: pd.Series):
    """
    Basit Pettitt testi; döndürür: (k_index, p_value).
    x: zaman sıralı (yıllık) seri.
    """
    s = pd.to_numeric(x, errors="coerce").dropna()
    n = len(s)
    if n < 8:
        return (np.nan, np.nan)
    # U-statistik (O(n^2) ama n küçük olduğundan yeterli)
    U = np.zeros(n, dtype=float)
    for t in range(n):
        U[t] = np.sum(np.sign(s.values - s.values[t])) - 0  # i!=t zaten sign(0)=0
    K_cum = np.cumsum(U)
    K = int(np.argmax(np.abs(K_cum)))
    Kstar = float(np.max(np.abs(K_cum)))
    p = 2.0 * np.exp((-6.0 * (Kstar**2)) / (n**3 + n**2))
    return (K, p)

def pettitt_on_annual_depth(annual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annual DTW (index=year[int], cols=Well) için Pettitt: kırılma yılı ve p.
    Dönüş: [Well, cp_year, cp_p] (tqdm ile ilerleme gösterir)
    """
    wells = [c for c in annual_df.columns]
    rows = []
    for w in _tqdm(wells, desc="Pettitt (per well)"):
        s = pd.to_numeric(annual_df[w], errors="coerce").dropna()
        years = s.index.values
        if len(s) < 8:
            rows.append({"Well": _norm_well_id(w), "cp_year": np.nan, "cp_p": np.nan})
            continue
        k, p = pettitt_test_series(s)
        if np.isfinite(k):
            k = int(np.clip(k, 0, len(s)-1))
            cp_year = int(years[k])
        else:
            cp_year = np.nan
        rows.append({"Well": _norm_well_id(w), "cp_year": cp_year, "cp_p": float(p)})
    return pd.DataFrame(rows)

def build_annual_dtw(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Monthly DTW (wide) -> Annual mean DTW (index=year[int], cols=Well)."""
    x = df.copy()
    # index'i datetime’a çevir ve filtrele
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="coerce")
    x = x.loc[x.index.notna()]
    if start is not None: x = x.loc[x.index >= pd.Timestamp(start)]
    if end   is not None: x = x.loc[x.index <= pd.Timestamp(end)]
    # numerikleşme ve tamamen NaN kolonları at
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.dropna(axis=1, how="all")
    # yıllık ortalama
    a = x.resample("YE").mean()
    if a.empty:
        return a
    a.index = a.index.year.astype(int)
    # tek satırlık güvenlik: tüm satırı NaN olan yılları at
    a = a.dropna(how="all")
    return a

def _apply_year_axis_labels_and_boundaries(ax, years: list[int]):
    """
    Heatmap'te x-ekseni: yılları sütun merkezine yaz ve aralara dikey sınır çiz.
    years: matrisin sütun sırası ile birebir aynı yıl listesi (len = ncols).
    """
    import numpy as np
    ncols = int(len(years))
    if ncols == 0:
        return

    # 1) Sütun merkezleri ve etiketler
    centers = np.arange(ncols, dtype=float)  # 0..ncols-1 zaten hücre merkezleri
    labels  = [str(int(y)) for y in years]

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=0, ha="center", va="top")

    # 2) Sınır çizgileri: hücre kenarları i+0.5 (son sınırı çizme)
    for b in (np.arange(ncols - 1, dtype=float) + 0.5):
        ax.axvline(b, color="k", lw=0.5, alpha=0.25)

    # 3) Piksel-merkez hizası (kaymayı engeller)
    ax.set_xlim(-0.5, ncols - 0.5)

def _save_axes_cropped(
    fig, ax, path, *,
    dpi=400,
    pad_inches=0.0,
    spine_color="#7a7a7a",
    spine_lw=0.8,
    keep_ticks=False,
    hide_title=True
):
    """
    Save only the area covered by `ax` (no colorbar, no surrounding figure).
    Ensures a thin gray frame (spines). Optionally hides the axes title for the save.
    """
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # cache current spines and title
    spine_state = {name: (sp.get_visible(), sp.get_edgecolor(), sp.get_linewidth())
                   for name, sp in ax.spines.items()}
    old_title = ax.get_title()

    # style spines (thin gray frame)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(spine_color)
        sp.set_linewidth(float(spine_lw))

    # remove ticks/labels if requested
    if not keep_ticks:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")

    # hide title only for the export
    if hide_title and old_title:
        ax.set_title("")

    # render + crop to axes bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, dpi=dpi, format="tiff", bbox_inches=bbox, pad_inches=pad_inches)

    # restore title and spines
    if hide_title and old_title:
        ax.set_title(old_title)
    for name, (vis, col, lw) in spine_state.items():
        sp = ax.spines[name]
        sp.set_visible(vis)
        sp.set_edgecolor(col)
        sp.set_linewidth(lw)

def _figsize_from_extent(x0, x1, y0, y1, target_h_in=6.2, min_h=4.8, max_h=8.5):
    """
    Choose a figure size that preserves the map aspect ratio.
    Returns (width_in, height_in).
    """
    import numpy as np
    w = float(x1 - x0); h = float(y1 - y0)
    if not (np.isfinite(w) and np.isfinite(h)) or w <= 0 or h <= 0:
        return (7.2, 5.6)
    ar = w / h
    H = float(np.clip(target_h_in, min_h, max_h))
    W = float(np.clip(H * ar, 5.0, 12.0))
    return (W, H)

def fit_gev_return_periods_min_sgi(sgi_df: pd.DataFrame,
                                   years=(10, 25, 50),
                                   b: int = None,
                                   random_state: int = 42,
                                   n_jobs: int | None = None) -> pd.DataFrame:
    """
    Monthly SGI → yearly minima → GEV → return levels (x_T) with 5–95% bands.
    - Single, clean tqdm bar over wells (no nested bars).
    - Optional parallel bootstrap with joblib.
    """
    # defaults
    if b is None:
        b = int(globals().get("GEV_BOOTSTRAP_B", 120))  # keep reasonable
    if n_jobs is None:
        try:
            import os
            n_jobs = max(1, (os.cpu_count() or 2) - 1)
        except Exception:
            n_jobs = 1

    # ensure datetime index
    if not isinstance(sgi_df.index, pd.DatetimeIndex):
        sgi_df = sgi_df.copy()
        sgi_df.index = pd.to_datetime(sgi_df.index, errors="coerce")
        sgi_df = sgi_df.loc[sgi_df.index.notna()]

    ygrp = sgi_df.groupby(sgi_df.index.year)
    wells = list(sgi_df.columns)

    out = []
    rng = np.random.default_rng(random_state)

    # outer, single progress bar
    pbar = _tqdm(wells, desc="GEV (per well)")
    for w in pbar:
        s = pd.to_numeric(sgi_df[w], errors="coerce")
        amin = ygrp[w].min().dropna()
        if amin.size < 8:
            continue

        # MLE fit
        try:
            c, loc, scale = _gev.fit(amin.values)
        except Exception:
            continue

        # bootstrap worker
        def _one_boot(_):
            res = rng.choice(amin.values, size=len(amin), replace=True)
            return _gev.fit(res)

        # collect bootstrap params (parallel if available)
        boot_params = []
        try:
            from joblib import Parallel, delayed
            if n_jobs > 1:
                boot_params = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_one_boot)(i) for i in range(b)
                )
            else:
                for i in range(b):
                    boot_params.append(_one_boot(i))
        except Exception:
            for i in range(b):
                boot_params.append(_one_boot(i))

        # small, unobtrusive progress hint in the bar itself
        pbar.set_postfix_str(f"boot={b}, well={w}")

        for Ty in years:
            q = 1.0 - 1.0/float(Ty)
            xT = _gev.ppf(q, c, loc=loc, scale=scale)
            lo = hi = np.nan
            if boot_params:
                xs = []
                for (cb, lb, sb) in boot_params:
                    if np.isfinite(sb):
                        xs.append(_gev.ppf(q, cb, loc=lb, scale=sb))
                if xs:
                    xs = np.asarray(xs, float)
                    lo, hi = np.nanpercentile(xs, [5, 95])
            out.append({
                "Well": _norm_well_id(w),
                "T_year": int(Ty),
                "xT_minSGI": float(xT),
                "xT_lo5": float(lo) if np.isfinite(lo) else np.nan,
                "xT_hi95": float(hi) if np.isfinite(hi) else np.nan
            })

    return pd.DataFrame(out)

# ---------------- MAIN PIPELINE ----------------
def main():
    import numpy as np
    import pandas as pd          # <-- add this line
    import matplotlib.pyplot as plt  # (if you use plt in main)

    # Fix 1: make sure we use the module-level setting instead of creating a local
    global BASEMAP_PROVIDER

    # Fix 2: defensive local import so Path is always available in this scope
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    coords_df = read_well_coordinates(Path(COORD_FILE))
    
    # 1) Load Excel (must have Yıl & Ay); build monthly time series
    raw = pd.read_excel(EXCEL_FILE)
    df = build_timeseries(raw)

    df = df.loc[(df.index >= START) & (df.index <= END)]
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.empty or df.shape[1] == 0:
        raise ValueError("No data after filtering to 2016–2024 or no well columns found.")

    # --- MONTHLY -> ANNUAL (DTW) | single source of truth ---
    global annual_dtw
    annual_dtw = build_annual_dtw(df, START, END)
    
    if annual_dtw is None or annual_dtw.empty:
        print("[WARN] annual_dtw is empty after build_annual_dtw(); check input months/columns.")
    else:
        # bilgi amaçlı küçük rapor
        yr_list = list(map(int, annual_dtw.index.tolist()))
        print(f"[OK] annual_dtw built: years {min(yr_list)}–{max(yr_list)}, "
              f"{annual_dtw.shape[1]} wells with any data")

    annual = annual_dtw  # keep existing downstream code happy

    # 2) Annual summaries (depths)
    annual = df.resample("YE").mean()
    annual.index = annual.index.year.astype(int)
    annual_delta = annual.diff()
    basin_annual_mean = annual.mean(axis=1)

    # --- Annual depth trend: Sen slope (m/year) + Kendall tau, p ---
    def _sen_slope(y_vals: np.ndarray, t_vals: np.ndarray) -> float:
        """Sen slope: median of all pairwise slopes (per year)."""
        y_vals = np.asarray(y_vals, float)
        t_vals = np.asarray(t_vals, float)
        mask = np.isfinite(y_vals) & np.isfinite(t_vals)
        y = y_vals[mask]; t = t_vals[mask]
        n = len(y)
        if n < 2:
            return np.nan
        slopes = []
        for i in range(n - 1):
            dt = t[i+1:] - t[i]
            dy = y[i+1:] - y[i]
            valid = dt != 0
            if np.any(valid):
                slopes.append(dy[valid] / dt[valid])
        if not slopes:
            return np.nan
        return float(np.nanmedian(np.concatenate(slopes)))
    
    trend_rows_depth = []
    for well in annual.columns:
        yy = annual[well].values.astype(float)
        tt = annual.index.values.astype(float)
        if np.isfinite(yy).sum() >= 3:
            sen = _sen_slope(yy, tt)                        # m/year
            tau, p = kendalltau(np.arange(len(yy))[np.isfinite(yy)], yy[np.isfinite(yy)])
        else:
            sen, tau, p = np.nan, np.nan, np.nan
        trend_rows_depth.append({
            "Well": _norm_well_id(well),
            "slope_m_per_year": sen,
            "tau": tau,
            "p": p
        })
    
    trend_df_depth = pd.DataFrame(trend_rows_depth)
    trend_df_depth.to_csv(OUT_TBLS / "fig10_trend_table.csv", index=False, float_format="%.6f")
    
    trend_map = coords_df.merge(trend_df_depth, on="Well", how="left")

    # 3) SGI (z-score), drought events & yearly metrics
    sgi = calculate_sgi(df).sort_index()

    events = []
    for col in sgi.columns:
        events += extract_drought_events(sgi[col], thr=-1.0)
    events_df = pd.DataFrame(events)
    OUT_TBLS.mkdir(exist_ok=True, parents=True)
    events_df.to_csv(OUT_TBLS / "sgi_drought_events.csv", index=False)

    yearly_metrics = drought_metrics_yearly(sgi, threshold=-1.0)
    # (moved from root to _EN tables folder)
    yearly_metrics.to_csv(OUT_TBLS / "drought_metrics_yearly.csv", index=False)

    # 4) SGI trend (Mann–Kendall + Sen)
    trend_rows = []
    for col in sgi.columns:
        tau, p, sen = mann_kendall_sen(sgi[col])
        trend_rows.append({"well": col, "tau": tau, "p_value": p, "sen_slope_per_year": sen})
    trend_df = pd.DataFrame(trend_rows).set_index("well").sort_values("sen_slope_per_year")
    trend_df.to_csv(OUT_TBLS / "sgi_trend_analysis.csv", float_format="%.6f")

    # 5) Figures
    # Fig 1 — Basin annual mean depth (with 95% CI + LOWESS)
    # Not: df aylık (monthly) seriler; OUT_FIGS zaten tanımlı.
    
    # Yıllık ortalamaları ve havza ortalamasını güvenli şekilde üret
    annual = df.resample("YE").mean()
    annual.index = annual.index.year.astype(int)
    basin_annual_mean = annual.mean(axis=1)
    
    if basin_annual_mean.size >= 1:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
    
        # Şekil boyutu: Fig.2 ile aynı; global varsa onu kullan
        FIG_W = float(globals().get("FIG_W", 9.8))
        FIG_H = float(globals().get("FIG_H", 5.8))
    
        years = basin_annual_mean.index.values.astype(int)
        yvals = basin_annual_mean.values.astype(float)
    
        # ---- Bootstrap %95 GA (kuyular arası) ----
        B = 2000
        rng = np.random.default_rng(42)
        per_year_vals = [row.values[np.isfinite(row.values)].astype(float) for _, row in annual.iterrows()]
        boot = np.full((B, len(per_year_vals)), np.nan, dtype=float)
        for b in range(B):
            for i, v in enumerate(per_year_vals):
                if v.size:
                    picks = rng.integers(0, v.size, size=v.size)  # replacement ile örnekleme
                    boot[b, i] = np.mean(v[picks])
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
    
        # ---- Düşük frekanslı eğri: LOWESS (varsa) yoksa 5-yıllık ort. ----
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(yvals, years, frac=0.35, return_sorted=False)
            label_smooth = "LOWESS"
        except Exception:
            smooth = pd.Series(yvals, index=years).rolling(5, center=True, min_periods=3).mean().values
            label_smooth = "5-yr running mean"
    
        # ---- Çizim ----
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    
        ax.plot(
            years, yvals,
            marker="o",
            color="steelblue",
            linewidth=1.4,
            markerfacecolor="lightskyblue",
            markeredgecolor="darkblue",
            markeredgewidth=0.8,
            markersize=7,
            zorder=5,
            label="Basin mean (annual)"
        )
    
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(years, lo, hi, color="steelblue", alpha=0.18, linewidth=0, label="95% CI (bootstrap)")
    
        if smooth is not None and np.isfinite(smooth).any():
            ax.plot(years, smooth, color="#1f487e", linewidth=2.0, alpha=0.9, label=label_smooth)
    
        ax.set_xlabel("Year")
        ax.set_ylabel("Groundwater depth (m)")
        ax.set_title(f"Basin annual mean groundwater depth (Konya, {int(years.min())}–{int(years.max())})")
        ax.legend(loc="best", frameon=True, framealpha=0.9)
    
        save_tiff(OUT_FIGS / "fig1_basin_annual_mean.tiff")
        plt.close(fig)

    # === Fig. 2 — Annual mean heatmaps (SGI ve WSE ayrı; V2 ile uyumlu) ===
    # Not: Boyutlar tüm makale genelinde aynı olsun diye global FIG_W/FIG_H kullanılıyor.
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    FIG_W = float(globals().get("FIG_W", 9.8))
    FIG_H = float(globals().get("FIG_H", 5.8))
    
    # ---- 2a) SGI yıllık ortalama ısı haritası (kırmızı=negatif=kurak) ----
    # SGI aylık tablo (sgi) zaten aylık olmalı; yıllığa ortalamayla indirgenir.
    try:
        sgi2 = sgi.copy()
        if not isinstance(sgi2.index, pd.DatetimeIndex):
            sgi2.index = pd.to_datetime(sgi2.index, errors="coerce")
        sgi2 = sgi2.loc[sgi2.index.notna()]
    
        annual_sgi = sgi2.resample("YE").mean()
        if not annual_sgi.empty:
            annual_sgi.index = annual_sgi.index.year.astype(int)
    
            # Kuyu sırası: doğal/alfabetik
            try:
                wells = _order_wells_alpha(annual_sgi.columns)
            except Exception:
                wells = sorted([str(c) for c in annual_sgi.columns])
            A = annual_sgi.reindex(columns=wells)
    
            years = A.index.astype(int).tolist()
    
            # Renk aralığı: simetrik, sıfır merkezli
            vals = A.to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size >= 10:
                p2, p98 = np.nanpercentile(vals, [2, 98]).astype(float)
                vmax = float(max(abs(p2), abs(p98)))
                vmin, vmax = -vmax, +vmax
            elif vals.size >= 2:
                vmax = float(max(abs(np.nanmin(vals)), abs(np.nanmax(vals))))
                vmin, vmax = -vmax, +vmax
            else:
                vmin, vmax = -1.0, 1.0
    
            cmap = globals().get("SGI_CMAP", "RdBu")
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=False)
            im = ax.imshow(
                A.T.values,
                aspect="auto",               # sütunlar arası boşluk yok
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
                origin="upper"
            )

            _apply_year_axis_labels_and_boundaries(ax, years)
    
            # Hücre merkez hizası (net ayraç çizgileri için önemli)
            ncols = len(years)
            ax.set_xlim(-0.5, ncols - 0.5)
    
            ax.set_xlabel("Year"); ax.set_ylabel("Well")
    
            # X-etiketleri (seyrelt)
            if len(years) > 0:
                step = max(1, len(years) // 20)  # ~≤20 x-etiketi
                xt = np.arange(0, len(years), step)
                ax.set_xticks(xt)
                ax.set_xticklabels([str(years[i]) for i in xt], rotation=45, ha="right")
    
            # Y-etiketleri (tam liste)
            ax.set_yticks(np.arange(len(wells)))
            ax.set_yticklabels([str(w) for w in wells])
    
            # === Yıllar arası dikey ayırıcı çizgiler (Fig.4 ile aynı mantık) ===
            # Sütun sınırları: i + 0.5 (son sınırı çizme)
            boundaries = [i + 0.5 for i in range(ncols - 1)]
            for b in boundaries:
                ax.axvline(b, color="k", lw=0.5, alpha=0.25)
    
            # Sağda GENİŞ renk çubuğu
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3.5%", pad=0.18)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("SGI (unitless)")
    
            ax.set_title("Fig. 2a — Annual mean SGI (wells × years)")
    
            save_tiff(OUT_FIGS / "fig2_annual_mean_heatmap.tiff")
        else:
            print("[INFO] Fig.2 SGI heatmap skipped: annual_sgi is empty.")
    except Exception as e:
        print("[WARN] Fig.2 SGI heatmap failed:", e)
    
    
    # ---- 2b) DTW yıllık ortalama ısı haritası — sadece negatif değerler boyalı (koyu kırmızı = en negatif) ----
    try:
        # 0) FIG boyutları (inç) – FIG_W/FIG_H yoksa varsayılan
        if "FIG_W" in globals() and "FIG_H" in globals():
            _FIG_W, _FIG_H = float(FIG_W), float(FIG_H)
        else:
            _FIG_W, _FIG_H = (9.8, 5.8)
    
        # 1) Yıllık tablo (DTW)
        annual_dtw = df.resample("YE").mean()
        if isinstance(annual_dtw.index, pd.DatetimeIndex):
            annual_dtw.index = annual_dtw.index.year.astype(int)
        else:
            annual_dtw.index = pd.to_datetime(annual_dtw.index, errors="coerce").year.astype("Int64")
            annual_dtw = annual_dtw.loc[annual_dtw.index.notna()].copy()
            annual_dtw.index = annual_dtw.index.astype(int)
        annual_dtw = annual_dtw.sort_index()
    
        if not annual_dtw.empty:
            # 2) Kuyu sırası (alfabetik ↑)
            try:
                wells_wse = _order_wells_alpha(annual_dtw.columns)
            except Exception:
                wells_wse = sorted([str(c) for c in annual_dtw.columns])
    
            # 3) Çizilecek matris (yıllar × kuyular)
            W = annual_dtw.reindex(columns=wells_wse)
            years = W.index.astype(int).tolist()
    
            # 4) Negatif aralık belirleme (sadece negatifleri renklendir)
            wvals = pd.to_numeric(W.to_numpy(dtype=float).ravel(), errors="coerce")
            negvals = wvals[np.isfinite(wvals) & (wvals < 0)]
            if negvals.size >= 10:
                vmin2 = float(np.nanpercentile(negvals, 2))   # dip kuyruk için robust alt sınır
            elif negvals.size >= 1:
                vmin2 = float(np.nanmin(negvals))
            else:
                print("[INFO] Fig.2b: No negative DTW values; skipping negative-only heatmap.")
                raise RuntimeError("no_negative_values")
    
            vmax2 = 0.0  # üst sınır sıfır: 0 → açık, vmin → koyu kırmızı
    
            # 5) Veriyi (wells × years) olarak al, pozitif ve sıfırları maskele
            M = W.T.to_numpy(dtype=float)                     # rows=wells, cols=years
            data_masked = np.ma.masked_where(M >= 0.0, M)     # >=0 beyaz (görünmez)
    
            # 6) Colormap: Reds_r (vmin = en negatif → koyu kırmızı; vmax=0 → açık)
            try:
                cmap = plt.cm.Reds_r.copy()
            except Exception:
                cmap = plt.cm.get_cmap("Reds_r")
            try:
                cmap.set_bad("white")  # maskelenen değerler (>=0) beyaz
            except Exception:
                pass
    
            # 7) Çizim
            fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), constrained_layout=False)
            im = ax.imshow(
                data_masked,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin2, vmax=vmax2,
                origin="upper"
            )

            # --- YIL ETİKETLERİ MERKEZDE + DÜŞEY SINIR ÇİZGİLERİ (her zaman görünür) ---
            def _year_axis_with_boundaries(_ax, _years):
                import numpy as np
                ncols = int(len(_years))
                if ncols == 0:
                    return
                centers = np.arange(ncols, dtype=float)           # 0..n-1 sütun merkezleri
                labels  = [str(int(y)) for y in _years]
                _ax.set_xticks(centers)
                _ax.set_xticklabels(labels, rotation=0, ha="center", va="top")
                _ax.set_xlim(-0.5, ncols - 0.5)                   # piksel-merkez hizası
                # Çizgiler en üste gelsin:
                for b in (np.arange(ncols - 1, dtype=float) + 0.5):
                    _ax.axvline(b, color="k", lw=0.8, alpha=0.35, zorder=50, solid_capstyle="butt")
    
            if "_apply_year_axis_labels_and_boundaries" in globals() and callable(_apply_year_axis_labels_and_boundaries):
                # Eğer global yardımcı varsa onu çağır, ardından çizgileri ekstra güvenceyle tekrar çiz
                _apply_year_axis_labels_and_boundaries(ax, years)
                # bazı ortamlarda zorder düşük kalırsa ek güvence:
                for b in (np.arange(len(years) - 1, dtype=float) + 0.5):
                    ax.axvline(b, color="k", lw=0.8, alpha=0.35, zorder=50, solid_capstyle="butt")
            else:
                _year_axis_with_boundaries(ax, years)
    
            ax.set_xlabel("Year")
            ax.set_ylabel("Well")
    
            # (ÖNEMLİ) X-tick’leri yeniden ayarlamayın; merkezleri bozmayın!
            # Aşağıdaki blok kaldırıldı:
            # if len(years) > 0:
            #     step = max(1, len(years) // 20)
            #     xt = np.arange(0, len(years), step)
            #     ax.set_xticks(xt)
            #     ax.set_xticklabels([str(years[i]) for i in xt], rotation=0, ha="center", va="top")
    
            # Y-etiketleri (tam liste)
            ax.set_yticks(np.arange(len(wells_wse)))
            ax.set_yticklabels([str(w) for w in wells_wse])
    
            # DİKEY COLORBAR: sadece negatif aralık
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3.5%", pad=0.18)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("DTW (m bgs)")
            try:
                _style_colorbar_fat_outline(cbar)
            except Exception:
                pass
    
            ax.set_title("Fig. 2b — Annual mean groundwater depth (negatives highlighted)")
            save_tiff(OUT_FIGS / "fig2_annual_mean_heatmap_DTW_negonly.tiff")
    
        else:
            print("[INFO] Fig.2 DTW heatmap skipped: annual_dtw is empty.")
    
    except RuntimeError as _e:
        if str(_e) != "no_negative_values":
            print("[WARN] Fig.2b DTW heatmap failed:", _e)
    except Exception as e:
        print("[WARN] Fig.2b DTW heatmap failed:", e)
   
    # ---------- 2c) Annual mean DTW anomaly (per-well baseline = first observed year) ----------
    try:
        import numpy as np
        import matplotlib.colors as mcolors
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    
        # 0) FIG boyutları
        if "FIG_W" in globals() and "FIG_H" in globals():
            _FW, _FH = float(FIG_W), float(FIG_H)
        else:
            _FW, _FH = (24.0/2.54, 12.0/2.54)  # ≈ 24×12 cm
    
        # 1) Yıllık tablo kaynağı: annual_dtw (varsa) → yoksa df'den üret
        if "annual_dtw" in globals() and isinstance(annual_dtw, pd.DataFrame) and (not annual_dtw.empty):
            A0 = annual_dtw.copy()
        else:
            # defansif yedek: df'den üret
            _tmp = df.copy()
            if not isinstance(_tmp.index, pd.DatetimeIndex):
                _tmp.index = pd.to_datetime(_tmp.index, errors="coerce")
            _tmp = _tmp.loc[_tmp.index.notna()]
            A0 = _tmp.resample("YE").mean()
            if A0.empty:
                print("[INFO] Fig.2c anomaly heatmap skipped: annual_dtw is empty.")
                raise RuntimeError("annual_dtw_empty")
            A0.index = A0.index.year.astype(int)
        A0 = A0.sort_index()
    
        # 2) Kuyu sırası (alfabetik ↑)
        try:
            wells_anom = _order_wells_alpha(A0.columns)
        except Exception:
            wells_anom = sorted([str(c) for c in A0.columns])
        A0 = A0.reindex(columns=wells_anom)
    
        # 3) Per-kuyu baz çizgi: ilk finite (gözlenen) değer
        baselines = pd.Series(index=A0.columns, dtype=float)
        for col in A0.columns:
            s = pd.to_numeric(A0[col], errors="coerce")
            s = s.dropna()
            first_finite = s.iloc[0] if len(s) else np.nan
            baselines[col] = float(first_finite) if np.isfinite(first_finite) else np.nan
    
        # 4) Anomali = yıl ortalaması − baz çizgi (kuyuya özgü)
        A = A0.astype(float).subtract(baselines, axis=1)
    
        # 5) Renk aralığı & cmap: veri dağılımına göre
        vals = A.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 1:
            data_min, data_max = float(np.nanmin(vals)), float(np.nanmax(vals))
        else:
            data_min, data_max = -1.0, 1.0
    
        if (data_min < 0.0) and (data_max > 0.0):
            # Sıfırı kapsıyorsa diverging + TwoSlopeNorm
            cmap = "RdBu_r"  # pozitif (derinleşme) kırmızı, negatif (sığlaşma) mavi
            norm = mcolors.TwoSlopeNorm(vmin=data_min, vcenter=0.0, vmax=data_max)
        elif data_max <= 0.0:
            cmap = "Blues_r"  # tamamen negatif → daha az negatif açık, daha negatif koyu mavi
            norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
        else:  # data_min >= 0.0
            cmap = "Reds"    # tamamen pozitif → daha pozitif koyu kırmızı
            norm = mcolors.Normalize(vmin=data_min, vmax=data_max)
    
        # 6) Matris (wells × years) ve yıl listesi
        years = A.index.astype(int).tolist()          # sütun (x) ekseni: yıllar
        M = A.T.to_numpy(dtype=float)                 # rows=wells, cols=years
    
        fig, ax = plt.subplots(figsize=(_FW, _FH), constrained_layout=False)
        im = ax.imshow(
            M,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            origin="upper"  # ilk kuyu en üstte
        )
    
        # 7) X ekseni: yıllar sütun merkezinde ve dikey yıl sınır çizgileri
        def _year_axis_with_boundaries(_ax, _years):
            ncols = int(len(_years))
            if ncols == 0:
                return
            centers = np.arange(ncols, dtype=float)                  # 0..n-1 sütun merkezleri
            labels  = [str(int(y)) for y in _years]
            _ax.set_xticks(centers)
            _ax.set_xticklabels(labels, rotation=0, ha="center", va="top")
            _ax.set_xlim(-0.5, ncols - 0.5)                          # piksel-merkez hizası
            for b in (np.arange(ncols - 1, dtype=float) + 0.5):      # sınırlar
                _ax.axvline(b, color="k", lw=0.5, alpha=0.25)
    
        # Yardımcıyı çağır (global fonksiyon varsa onu kullan)
        if "_apply_year_axis_labels_and_boundaries" in globals() and callable(_apply_year_axis_labels_and_boundaries):
            _apply_year_axis_labels_and_boundaries(ax, years)
        else:
            _year_axis_with_boundaries(ax, years)
    
        # 8) Y-etiketleri (kuyular)
        ax.set_yticks(np.arange(len(wells_anom)))
        ax.set_yticklabels([str(w) for w in wells_anom])
    
        ax.set_xlabel("Year")
        ax.set_ylabel("Well")
        ax.set_title("Fig. 2c — Annual mean DTW anomaly (per-well baseline = first observed year)")
    
        # 9) Dikey colorbar (axes ile aynı yükseklik)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("DTW anomaly (m; + = deeper)")
        try:
            cbar.solids.set_edgecolor("face"); cbar.solids.set_linewidth(0)
            cbar.outline.set_linewidth(1.0)
            cbar.ax.tick_params(length=4, width=1.0)
        except Exception:
            pass
    
        save_tiff(OUT_FIGS / "fig2_annual_mean_heatmap_DTW_anom.tiff")
    
    except RuntimeError as _e:
        if str(_e) != "annual_dtw_empty":
            print("[WARN] Fig.2c anomaly heatmap failed:", _e)
    except Exception as e:
        print("[WARN] Fig.2c anomaly heatmap failed:", e)

    # ==== Pettitt: change-point year on annual DTW ==================================
    try:
        if "annual_dtw" in globals() and isinstance(annual_dtw, pd.DataFrame) and not annual_dtw.empty:
            # 1) Table (per well)
            cp_tbl = pettitt_on_annual_depth(annual_dtw)
            OUT_CP = Path("./out_change_point_EN"); OUT_CP.mkdir(parents=True, exist_ok=True)
            cp_tbl.to_csv(OUT_CP / "pettitt_change_points.csv", index=False)
    
            # 2) Coordinates -> EPSG:3857
            coordsX = coords_df.copy(); coordsX["Well"] = coordsX["Well"].apply(_norm_well_id)
            coords_3857 = to_target_coords_df(coordsX, target="EPSG:3857")
            M = coords_3857.merge(cp_tbl, on="Well", how="left")
    
            ok = (
                np.isfinite(M["cp_year"].to_numpy(float)) &
                np.isfinite(M["X_3857"].to_numpy(float)) &
                np.isfinite(M["Y_3857"].to_numpy(float))
            )
            if ok.sum() >= 3:
                x_all = coords_3857["X_3857"].to_numpy(float)
                y_all = coords_3857["Y_3857"].to_numpy(float)
                x0, x1 = float(np.nanmin(x_all)), float(np.nanmax(x_all))
                y0, y1 = float(np.nanmin(y_all)), float(np.nanmax(y_all))
                dx, dy = 0.10*(x1-x0), 0.10*(y1-y0)
                GX0, GX1, GY0, GY1 = x0-dx, x1+dx, y0-dy, y1+dy
    
                x_use = M.loc[ok, "X_3857"].to_numpy(float)
                y_use = M.loc[ok, "Y_3857"].to_numpy(float)
                v_use = M.loc[ok, "cp_year"].to_numpy(float)
    
                vmin_i = int(np.floor(np.nanmin(v_use)))
                vmax_i = int(np.ceil (np.nanmax(v_use)))
    
                Xi, Yi, Zi, extent = _interp_grid_safe(
                    x_use, y_use, v_use, GX0, GX1, GY0, GY1,
                    nx=int(globals().get("NX_IDW", 320)),
                    power=int(globals().get("IDW_PWR", 2))
                )
                if np.isfinite(Zi).any():
                    # ---- güvenli fig boyutu + eksen oluşturma
                    _fs = globals().get("_figsize_from_extent", None)
                    if callable(_fs):
                        W_in, H_in = _fs(*extent)
                    else:
                        ex0, ex1, ey0, ey1 = extent
                        ar = (ex1-ex0) / max((ey1-ey0), 1e-9)
                        H_in = 6.2; W_in = min(max(H_in*ar, 5.0), 12.0)
    
                    fig, ax = plt.subplots(figsize=(W_in, H_in))  # <--- ÖNCE aksı aç
                    # basemap
                    try:
                        zoom = _ctx_zoom_for_extent(extent)
                        add_basemap_color_only(ax, crs="EPSG:3857",
                                               topo=BASEMAP_PROVIDER, alpha=0.95, zoom=zoom)
                    except Exception:
                        pass
    
                    # raster
                    im = ax.imshow(Zi, origin="lower", extent=(GX0,GX1,GY0,GY1),
                                   cmap="viridis", vmin=vmin_i, vmax=vmax_i,
                                   interpolation="nearest", aspect="equal", zorder=2)
    
                    # noktalar
                    ax.scatter(x_all, y_all, s=10, c="lightgray", edgecolor="none", zorder=3)
                    ax.scatter(x_use, y_use, s=22, c="black", edgecolor="white", linewidth=0.4, zorder=4)
    
                    ax.set_xlim(GX0,GX1); ax.set_ylim(GY0,GY1)
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
    
                    try:
                        add_scalebar(ax, length_km=20, location="lower left")
                        add_north_arrow_above_scalebar(ax, length=0.08, fontsize=10, lw=1.2, head_length=12)
                    except Exception:
                        pass
    
                    ticks = np.arange(vmin_i, vmax_i + 1, 1)
                    cbar = _colorbar_same_height(
                        ax, im,
                        label="Change-point year (Pettitt)",
                        ticks=ticks,
                        outline=True
                    )
    
                    ax.set_title("Regime shift (Pettitt) — change-point year on annual DTW")
                    save_tiff(OUT_CP / "pettitt_change_point_year.tiff")
                    plt.close(fig)
            else:
                print("[INFO] Pettitt: insufficient valid cp_year points to map.")
        else:
            print("[INFO] Pettitt: annual_dtw empty; skipping.")
    except Exception as e:
        print("[WARN] Pettitt failed:", e)
    
        # Fig 3 — SGI Sen’s slope per well (colored; * = p<0.05), wells A→Z top→bottom
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import colors as mcolors
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        FIG_W = float(globals().get("FIG_W", 9.8))
        FIG_H = float(globals().get("FIG_H", 5.8))
        
        # --- trend_df yoksa sgi'den inşa et ---
        need_build = ('trend_df' not in locals()) or (trend_df is None) or (getattr(trend_df, "empty", True))
        if need_build:
            trend_rows = []
            # sgi datetime garantisi
            _sgi = sgi.copy()
            if not isinstance(_sgi.index, pd.DatetimeIndex):
                _sgi.index = pd.to_datetime(_sgi.index, errors="coerce")
                _sgi = _sgi.loc[_sgi.index.notna()]
        
            for w in _sgi.columns:
                s = pd.to_numeric(_sgi[w], errors="coerce").dropna()
                if s.empty:
                    continue
                tau, p, sen_per_year = mann_kendall_sen(s)  # tercih edilen yöntem
                # fallback: sen slope NaN ise lineer fit (aylık eğim * 12)
                if not np.isfinite(sen_per_year):
                    try:
                        slope_month, _ = np.polyfit(np.arange(len(s), dtype=float),
                                                    s.values.astype(float), 1)
                        sen_per_year = float(slope_month * 12.0)
                        if not np.isfinite(p):   p = 1.0
                        if not np.isfinite(tau): tau = 0.0
                    except Exception:
                        continue
                trend_rows.append({
                    "Well": str(w),
                    "sen_slope_per_year": float(sen_per_year),
                    "p_value": float(p) if np.isfinite(p) else 1.0,
                    "tau": float(tau) if np.isfinite(tau) else 0.0
                })
            trend_df = pd.DataFrame(trend_rows).set_index("Well") if trend_rows else pd.DataFrame()
        
        if (trend_df is not None) and (trend_df.shape[0] >= 1):
            # --- Alfabetik sıralama ve çizim verisi ---
            well_order = sorted([str(w) for w in trend_df.index])      # A→Z
            trend_plot = trend_df.reindex(well_order)
        
            vals = trend_plot["sen_slope_per_year"].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if finite.sum() == 0:
                print("[WARN] No finite Sen slopes; Fig.3 skipped.")
            else:
                # Dinamik aralık + küçük ped
                vmin = float(np.nanmin(vals[finite])); vmax = float(np.nanmax(vals[finite]))
                span = vmax - vmin
                if not np.isfinite(span) or span <= 0:
                    center = vmin if np.isfinite(vmin) else 0.0
                    half = max(abs(center), 0.25)
                    vmin, vmax = center - half, center + half
                    span = vmax - vmin
                pad = 0.06 * span
                x_min = vmin - pad; x_max = vmax + pad
                if x_min > 0: x_min = 0.0 - 0.04 * span
                if x_max < 0: x_max = 0.0 + 0.04 * span
        
                # Diverging colormap (negatif kırmızı, pozitif mavi)
                cmap = mpl.colormaps.get("RdBu")
                norm = mcolors.TwoSlopeNorm(vmin=x_min, vcenter=0.0, vmax=x_max)
                bar_colors = [cmap(norm(v)) if np.isfinite(v) else (0.7, 0.7, 0.7, 1.0) for v in vals]
        
                # Çizim
                fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                y = np.arange(len(well_order))
                ax.barh(y, vals, color=bar_colors, edgecolor="none", height=0.9)
        
                # p<0.05 yıldız
                for i, w in enumerate(well_order):
                    row = trend_plot.loc[w]
                    v = row.get("sen_slope_per_year", np.nan)
                    p = row.get("p_value", np.nan)
                    if np.isfinite(v) and np.isfinite(p) and (p < 0.05):
                        ax.plot(v, i, marker="*", markersize=7, color="black", zorder=3)
        
                ax.axvline(0, color="k", linewidth=0.8)
                ax.set_xlim(x_min, x_max)
                ax.set_xlabel("Sen’s slope of SGI (per year)")
        
                # Y ekseni: A→Z üstten alta (invert ile)
                ax.set_yticks(y)
                ax.set_yticklabels(well_order)
                ax.invert_yaxis()
        
                ax.set_title("SGI trend per well (Sen’s slope, * = p<0.05)")
        
                # Yan geniş colorbar (eksen sıkışmadan)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3.5%", pad=0.15)
                cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                cb.set_label("SGI trend (per year)")
        
                save_tiff(OUT_FIGS / "fig3_sgi_sen_slope_per_well.tiff")
        else:
            print("[WARN] Fig.3 skipped: trend_df empty or undefined.")

    # Fig 4 — SGI heatmap (wells × months), continuous bars across all months
    try:
        # SGI dizinini garanti şekilde datetime yap
        _sgi = sgi.copy()
        if not isinstance(_sgi.index, pd.DatetimeIndex):
            _sgi.index = pd.to_datetime(_sgi.index, errors="coerce")
            _sgi = _sgi.loc[_sgi.index.notna()]
    
        if (_sgi.shape[0] >= 1) and (_sgi.shape[1] >= 1):
            dates = _sgi.index
    
            # 1) Y-ekseni için kuyuları alfabetik (küçük→büyük) sırada tut
            try:
                well_order2 = _order_wells_alpha(_sgi.columns)  # küçük→büyük
            except Exception:
                well_order2 = sorted([str(c) for c in _sgi.columns])
    
            # 2) Veriyi bu sırada al ve (rows=wells, cols=months) şekline getir
            data = _sgi[well_order2].T.to_numpy(dtype=float)
            data_masked = np.ma.masked_invalid(data)
    
            # 3) Yıl merkezlerini ve sınırlarını hesapla (etiketler merkezde)
            years_unique = np.unique(dates.year.values)
            centers, labels, boundaries = [], [], []
            for y in years_unique:
                idxs = np.flatnonzero(dates.year.values == y)
                if idxs.size == 0:
                    continue
                centers.append((idxs[0] + idxs[-1]) / 2.0)   # yılın orta sütunu
                labels.append(str(int(y)))
                boundaries.append(idxs[-1] + 0.5)            # yılın sağ sınırı (ayırıcı çizgi)
    
            # 4) Figür boyutu (varsayılan: ~25×15 cm)
            if "FIG_W" in globals() and "FIG_H" in globals():
                _FW, _FH = float(FIG_W), float(FIG_H)
            else:
                _FW, _FH = (25.0/2.54, 15.0/2.54)
    
            fig, ax = plt.subplots(figsize=(_FW, _FH))
    
            # 5) Isı haritası (extent KULLANMADAN; piksel indeksleriyle tam hizalı)
            im = ax.imshow(
                data_masked,
                aspect="auto",
                cmap="RdBu",            # red = dry (low), blue = wet (high)
                vmin=-2.5, vmax=2.5,
                interpolation="none",
                origin="upper"          # first well (well_order2[0]) en üst satırda
            )

            _apply_year_axis_labels_and_boundaries(ax, years)
    
            # Piksel-merkez hizası için x-limit
            ax.set_xlim(-0.5, data_masked.shape[1] - 0.5)
    
            # 6) Eksen etiketleri
            ax.set_xlabel("Time")
            ax.set_ylabel("Well")
    
            # 7) Yıl etiketlerini yıl merkezlerine yatay ortalı koy
            if len(centers) > 0:
                ax.set_xticks(centers)
                ax.set_xticklabels(labels, rotation=0, ha="center", va="top")
    
            # 8) Yıl sınırlarını ince dikey çizgilerle ayır (son sınırı çizme)
            for b in boundaries[:-1]:
                ax.axvline(b, color="k", lw=0.5, alpha=0.25)
    
            # 9) Y-etiketleri (kuyular) — satır indeksleriyle birebir
            ax.set_yticks(np.arange(len(well_order2)))
            ax.set_yticklabels([str(w) for w in well_order2])
    
            # 10) Colorbar (dikey, eksenle aynı yükseklikte)
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.15)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("SGI")
            try:
                cbar.solids.set_edgecolor("face"); cbar.solids.set_linewidth(0)
            except Exception:
                pass
    
            ax.set_title("Standardized Groundwater Index (SGI) — wells × months")
            save_tiff(OUT_FIGS / "fig4_sgi_heatmap.tiff")
        else:
            print("[INFO] Fig.4 SGI heatmap skipped: SGI is empty or has no columns.")
    except Exception as e:
        print("[WARN] Fig.4 SGI heatmap failed:", e)


    # ---------- Fig. 5 — Drought timelines (SGI < -1), colored by EVENT TOTAL SGI ----------
    dates = sgi.index
    if len(dates) >= 2:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.colors as mcolors
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    
        # Well order: alphabetical (ascending, top→bottom)
        try:
            well_order3 = _order_wells_alpha(sgi.columns)
        except Exception:
            well_order3 = sorted([str(c) for c in sgi.columns])
    
        # Collect events and compute total_sgi per event
        evs = []
        for col in sgi.columns:
            s = pd.to_numeric(sgi[col], errors="coerce")
            for ev in extract_drought_events(s, thr=-1.0):
                if "total_sgi" not in ev or ev["total_sgi"] is None or not np.isfinite(ev["total_sgi"]):
                    s_dt = pd.to_datetime(ev["start"])
                    e_dt = pd.to_datetime(ev["end"])
                    seg = s.loc[s_dt:e_dt]
                    ev["total_sgi"] = float(np.nansum(seg.values))
                evs.append(ev)
    
        events_df2 = pd.DataFrame(evs)
        if not events_df2.empty:
            date_to_pos = {d: i for i, d in enumerate(dates)}
            year_starts = [i for i, d in enumerate(dates) if getattr(d, "month", pd.Timestamp(d).month) == 1]
    
            # Robust color scaling
            totals = events_df2["total_sgi"].to_numpy(dtype=float)
            finite = np.isfinite(totals)
            if finite.any():
                data_min = float(np.nanmin(totals[finite]))
                data_max = float(np.nanmax(totals[finite]))
                p2, p98 = np.nanpercentile(totals[finite], [2, 98]).astype(float)
            else:
                data_min, data_max, p2, p98 = -1.0, 1.0, None, None
    
            if data_min < 0 and data_max > 0:
                vmin = min(p2, data_min) if p2 is not None and np.isfinite(p2) else data_min
                vmax = max(p98, data_max) if p98 is not None and np.isfinite(p98) else data_max
                vmin = min(vmin, -1e-9); vmax = max(vmax, 1e-9)
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                cmap = mpl.colormaps.get("RdYlBu")  # red = drought (negative), blue = wet
            elif data_max <= 0:
                vmin = float(data_min) if np.isfinite(data_min) else -1.0
                vmax = 0.0
                if vmin >= vmax: vmin = vmax - 1.0
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = mpl.colormaps.get("RdYlGn")
            else:
                vmin = 0.0
                vmax = float(data_max) if np.isfinite(data_max) else 1.0
                if vmax <= vmin: vmax = vmin + 1.0
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = mpl.colormaps.get("YlGnBu")
    
            # Figure
            FIG_W = float(globals().get("FIG_W", 9.8))
            FIG_H = float(globals().get("FIG_H", 5.8))
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    
            # Bars with darkslategray borders
            for _, ev in events_df2.iterrows():
                w = str(ev["well"])
                if w not in well_order3:
                    continue
                y = well_order3.index(w)
    
                s_dt = pd.Timestamp(ev["start"])
                e_dt = pd.Timestamp(ev["end"])
                if (s_dt not in date_to_pos) or (e_dt not in date_to_pos):
                    continue
    
                x0 = date_to_pos[s_dt]
                x1 = date_to_pos[e_dt]
                if not np.isfinite(x0) or not np.isfinite(x1):
                    continue
                width = max(1, int(x1 - x0 + 1))  # at least one month wide
    
                face = cmap(norm(float(ev["total_sgi"]))) if np.isfinite(ev["total_sgi"]) else (0.7, 0.7, 0.7, 1.0)
                ax.broken_barh(
                    [(x0, width)],
                    (y - 0.4, 0.8),
                    facecolor=face,
                    edgecolor="darkslategray",
                    linewidth=0.6,
                    zorder=2
                )
    
            # Axes limits and ticks
            n = max(len(dates) - 1, 1)
            pad = 0.05 * n  # +5% on the right so bars aren't cut
            ax.set_xlim(-0.5, n + 0.5 + pad)
            ax.set_ylim(-1, len(well_order3))
            ax.set_yticks(np.arange(len(well_order3)))
            ax.set_yticklabels(well_order3)
    
            if len(year_starts) >= 1:
                ax.set_xticks(year_starts)
                ax.set_xticklabels([str(pd.Timestamp(dates[i]).year) for i in year_starts], rotation=45, ha="right")
    
            ax.set_xlabel("Time (monthly)")
            ax.set_ylabel("Well")
            ax.set_title("Fig. 5 — Groundwater drought timelines (SGI < −1), colored by event total SGI")
    
            # Light gray grid (vertical & horizontal), behind bars
            ax.set_axisbelow(True)
            ax.grid(True, which="major", axis="both", color="lightgray", linewidth=0.6, alpha=0.6)
    
            # Vertical colorbar (right)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.15)
            sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
            cbar.set_label("Event total SGI")
            cbar.outline.set_linewidth(1.0)
            cbar.ax.tick_params(length=4, width=1.0)
    
            save_tiff(OUT_FIGS / "fig5_drought_timelines.tiff")
        else:
            print("[INFO] Fig.5 skipped: no drought events found.")

    # === Fig 6 — Annual change (Δ) distributions by year (professional violin+boxplot) ===
    # Girdi: annual_delta -> index: years, columns: wells (Δ değerleri)
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    valid_years = [int(y) for y in annual_delta.index if pd.Series(annual_delta.loc[y]).notna().any()]
    data_box = [annual_delta.loc[y].dropna().values.astype(float) for y in valid_years]
    
    if (len(valid_years) >= 1) and (len(data_box) == len(valid_years)):
        FIG_W = float(globals().get("FIG_W", 9.8))
        FIG_H = float(globals().get("FIG_H", 5.8))
    
        x = np.arange(1, len(valid_years) + 1)
    
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    
        # Yıl -> renk eşlemesi
        cmap = mpl.colormaps.get("viridis")
        norm = mpl.colors.Normalize(vmin=min(valid_years), vmax=max(valid_years))
    
        # Violinplot (renkli gövdeler)
        vparts = ax.violinplot(
            data_box, positions=x, widths=0.85,
            showmeans=True, showmedians=True, showextrema=False
        )
        for i, b in enumerate(vparts['bodies']):
            yr = valid_years[i]
            b.set_facecolor(cmap(norm(yr)))
            b.set_edgecolor("black")
            b.set_linewidth(0.6)
            b.set_alpha(0.85)
    
        # Medyan ve ortalama çizgilerini belirginleştir
        if 'cmeans' in vparts:
            vparts['cmeans'].set_color("black")
            vparts['cmeans'].set_linewidth(1.2)
        if 'cmedians' in vparts:
            vparts['cmedians'].set_color("white")
            vparts['cmedians'].set_linewidth(1.6)
    
        # İnce bir boxplot overlay (kuartilleri göster, dış değerleri gizle)
        bparts = ax.boxplot(
            data_box, positions=x, widths=0.28, showfliers=False, patch_artist=True
        )
        for patch in bparts['boxes']:
            patch.set_facecolor("white")
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        for elem in ['whiskers', 'caps', 'medians']:
            for artist in bparts[elem]:
                artist.set_color("black")
                artist.set_linewidth(0.8)
    
        # Her yıl için ortalama noktası
        means = [np.mean(v) if len(v) else np.nan for v in data_box]
        ax.plot(x, means, marker='o', markersize=3.5, linestyle='None', color='black', zorder=3)
    
        # X-etiketleri: "Yıl\n(n=...)" biçiminde
        xticklabels = [f"{yr}\n(n={len(data_box[i])})" for i, yr in enumerate(valid_years)]
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=0, ha="center")
    
        # Yardımcı unsurlar
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)          # referans çizgisi
        ax.grid(axis='y', which='major', linestyle='--', linewidth=0.6, alpha=0.5)
    
        # Eksenler / başlık
        ax.set_xlim(0.4, len(valid_years) + 0.6)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual change Δ depth (m)")
        ax.set_title("Distribution of annual groundwater change across wells")
    
        # Yıl renk skalası için küçük colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.15)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Year")
        _style_colorbar_fat_outline(cbar)
    
        save_tiff(OUT_FIGS / "fig6_annual_change_boxplots.tiff")

    # Fig 7 — Basin monthly climatology with IQR across wells
    if df.shape[0] >= 12:
        months = np.arange(1, 13)
        monthly_basin = df.mean(axis=1)
        clim_mean = [monthly_basin[monthly_basin.index.month == m].mean() for m in months]
        iqr_low, iqr_high = [], []
        for m in months:
            vals = df[df.index.month == m].mean(axis=0)
            if vals.size == 0 or np.all(np.isnan(vals.values)):
                iqr_low.append(np.nan); iqr_high.append(np.nan)
            else:
                q25, q75 = np.nanpercentile(vals.values, [25, 75])
                iqr_low.append(q25); iqr_high.append(q75)
        plt.figure(figsize=(6.2, 3.8))
        plt.plot(months, clim_mean, marker="o")
        if not np.all(np.isnan(iqr_low)) and not np.all(np.isnan(iqr_high)):
            plt.fill_between(months, iqr_low, iqr_high, alpha=0.25)
        plt.xticks(months, ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        plt.xlabel("Month"); plt.ylabel("Groundwater depth (m)")
        plt.title("Basin monthly climatology (mean) with IQR across wells")
        save_tiff(OUT_FIGS / "fig7_monthly_climatology_iqr.tiff")

    # 6) SPATIAL MAPS (yearly drought metrics) — California-style minimalist grids
    # - Fixed extent for all panels (10% padding from all wells)
    # - Global (per-metric) color limits computed from ALL years
    # - IDW interpolation onto a regular grid
    # - Backdrop: all wells (light gray)
    # - Overlay: wells with data for the current panel (green)
    # - Minimalist: NO legend, NO north arrow, NO scale bar
    
    # ------------- inputs -------------
    coords_df = read_well_coordinates(Path(COORD_FILE))
    coords_df.to_csv(OUT_SPATIAL / "_coords_parsed.csv", index=False)
    
    # normalize Well IDs to be safe (same as you do elsewhere)
    cdf = coords_df.copy()
    cdf["Well"] = cdf["Well"].apply(_norm_well_id)
    
    mdf = yearly_metrics.copy()
    if not mdf.empty:
        mdf["Well"] = mdf["Well"].apply(_norm_well_id)
    
    metrics = ["MaxDroughtDuration", "MinSGI", "CumulativeDeficit", "NumEvents"]
    years   = sorted(mdf["Year"].unique().tolist()) if not mdf.empty else []
    
    if not years:
        print("[INFO] Spatial maps: yearly_metrics is empty — skipping.")
    else:
        # -------- fixed map extent from ALL wells (10% padding) --------
        _xmin, _xmax = float(cdf["X"].min()), float(cdf["X"].max())
        _ymin, _ymax = float(cdf["Y"].min()), float(cdf["Y"].max())
        _dx, _dy     = 0.10 * (_xmax - _xmin), 0.10 * (_ymax - _ymin)
        GX0, GX1, GY0, GY1 = _xmin - _dx, _xmax + _dx, _ymin - _dy, _ymax + _dy
    
        # backdrop lattice (all wells, always plotted the same)
        X_ALL = cdf["X"].to_numpy(dtype=float)
        Y_ALL = cdf["Y"].to_numpy(dtype=float)
    
        # -------- global per-metric limits (robust) --------
        limits = {}
        for metric in metrics:
            if metric not in mdf.columns:
                limits[metric] = (0.0, 1.0)
                continue
    
            vals = pd.to_numeric(mdf[metric], errors="coerce").to_numpy(dtype=float)
    
            if metric == "MinSGI":
                # lower (more negative) is worse → cap upper at >=0
                lo, hi = -2.0, 0.0
                try:
                    p2, p98 = np.nanpercentile(vals, [2, 98])
                    lo = float(p2); hi = float(max(p98, 0.0))
                except Exception:
                    pass
                if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                    lo, hi = -2.0, 0.0
            else:
                lo = 0.0
                try:
                    hi = float(np.nanpercentile(vals, 98))
                    if not np.isfinite(hi) or hi <= lo:
                        raise ValueError
                except Exception:
                    try:
                        vmax = float(np.nanmax(vals))
                        hi = vmax if np.isfinite(vmax) and vmax > lo else lo + 1.0
                    except Exception:
                        hi = lo + 1.0
    
            limits[metric] = (float(lo), float(hi))
    
        # -------- small interpolation wrapper (prefer global _idw_grid if present) --------
        def _interp_grid(x, y, v, gx0, gx1, gy0, gy1, nx=320, power=2):
            # Try global _idw_grid; if not available, fall back to local simple IDW
            if "_idw_grid" in globals() and callable(globals()["_idw_grid"]):
                return _idw_grid(x, y, v, gx0, gx1, gy0, gy1, nx=nx, power=power)
            elif "_idw_grid_local" in globals() and callable(globals()["_idw_grid_local"]):
                return _idw_grid_local(x, y, v, gx0, gx1, gy0, gy1, nx=nx, power=power)
            else:
                # minimal fallback: very small helper
                x = np.asarray(x, float); y = np.asarray(y, float); v = np.asarray(v, float)
                nx = int(nx); xr = gx1 - gx0; yr = gy1 - gy0
                ny = max(2, int(round(nx * (yr / xr)))) if xr > 0 else nx
                xi = np.linspace(gx0, gx1, nx); yi = np.linspace(gy0, gy1, ny)
                Xi, Yi = np.meshgrid(xi, yi)
                dx = Xi[None, :, :] - x[:, None, None]
                dy = Yi[None, :, :] - y[:, None, None]
                dist = np.hypot(dx, dy) + 1e-12
                ok = np.isfinite(v)
                if ok.sum() < 3:
                    return Xi, Yi, np.full_like(Xi, np.nan), [gx0, gx1, gy0, gy1]
                w = 1.0 / (dist[ok] ** power)
                Zi = np.nansum(w * v[ok][:, None, None], axis=0) / np.nansum(w, axis=0)
                return Xi, Yi, Zi, [gx0, gx1, gy0, gy1]
    
        # figure size (use globals if defined)
        if "FIG_W" in globals() and "FIG_H" in globals():
            _FW, _FH = float(FIG_W), float(FIG_H)
        else:
            _FW, _FH = (24.0/2.54, 12.0/2.54)  # ~ (9.45 in, 4.72 in)
    
        # tqdm wrapper (use your _tqdm if defined, else plain tqdm)
        _PB = _tqdm if "_tqdm" in globals() and callable(globals()["_tqdm"]) else tqdm
    
        for metric in metrics:
            if metric not in mdf.columns:
                continue
    
            vmin, vmax = limits[metric]
    
            # consistent colormap selection
            if metric == "MinSGI":
                cmap       = "Spectral"    # lower = worse → warm end
                cbar_label = "Min SGI (monthly, lower = worse)"
            else:
                cmap       = "Spectral_r"  # larger = worse → warm end
                cbar_label = {
                    "MaxDroughtDuration": "Max drought duration (months)",
                    "CumulativeDeficit": "Σ(−SGI) where SGI < −1",
                    "NumEvents": "Number of drought events"
                }.get(metric, metric)
    
            for y in _PB(years, desc=f"Spatial {metric} (per year)"):
                # join coords with metric of this year
                dfy    = mdf.loc[mdf["Year"] == y, ["Well", metric]].copy()
                merged = cdf.merge(dfy, on="Well", how="left")
    
                # valid points for THIS panel
                xvals = merged["X"].to_numpy(dtype=float)
                yvals = merged["Y"].to_numpy(dtype=float)
                vvals = pd.to_numeric(merged[metric], errors="coerce").to_numpy(dtype=float)
    
                ok = np.isfinite(xvals) & np.isfinite(yvals) & np.isfinite(vvals)
                if ok.sum() < 3:
                    continue
    
                x_use, y_use, v_use = xvals[ok], yvals[ok], vvals[ok]
    
                # interpolate on fixed extent
                Xi, Yi, Zi, extent = _interp_grid(
                    x_use, y_use, v_use,
                    GX0, GX1, GY0, GY1,
                    nx=int(globals().get("NX_IDW", 320)),
                    power=int(globals().get("IDW_PWR", 2))
                )
    
                if not np.isfinite(Zi).any():
                    continue
    
                fig, ax = plt.subplots(figsize=(_FW, _FH))
    
                im = ax.imshow(
                    Zi, origin="lower",
                    extent=(GX0, GX1, GY0, GY1),
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation="nearest",
                    aspect="equal", zorder=2
                )
    
                # backdrop: all wells (constant lattice)
                ax.scatter(X_ALL, Y_ALL, s=8, c="lightgray", edgecolor="none", zorder=3)
                # overlay: wells with data in this panel
                ax.scatter(x_use, y_use, s=16, c="xkcd:true green",
                           edgecolor="xkcd:pine green", linewidth=0.5, zorder=4)
    
                # clean frame
                ax.set_xlim(GX0, GX1); ax.set_ylim(GY0, GY1)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
    
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(cbar_label)
    
                ax.set_title(f"SGI spatial metric — {metric} ({y})")

                add_year_tag(ax, y)
    
                # California adlandırması: {metric}_{year}.tiff
                save_tiff(OUT_SPATIAL / f"{metric}_{y}.tiff")
                plt.close(fig)



    # 7) STUDY-AREA MAP (OSM)
    plot_study_area_map(coords_df, OPTIONAL_BOUNDARY_FILE, BASEMAP_PROVIDER,
                        OUT_FIGS / "fig0_study_area_map.tiff")

    # 8) SGI monthly time series for each well
    for well in sgi.columns:
        plot_sgi_series(sgi, well, drought_threshold=-1.0, out_dir=OUT_SGI_SERIES)

    # 9) Interpolated annual groundwater level maps (IDW) — EN
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    annual_maps_out = Path("./out_annual_gw_maps_EN")
    annual_maps_out.mkdir(parents=True, exist_ok=True)
    
    VMIN, VMAX = -120.0, 0.0  # fixed scale for comparability
    cmap_fixed = plt.get_cmap("RdYlBu").copy()
    
    xmin = float(coords_df["X"].min()); xmax = float(coords_df["X"].max())
    ymin = float(coords_df["Y"].min()); ymax = float(coords_df["Y"].max())
    dx = (xmax - xmin) * 0.10; dy = (ymax - ymin) * 0.10
    xminp, xmaxp = xmin - dx, xmax + dx
    yminp, ymaxp = ymin - dy, ymax + dy
    
    years = [int(y) for y in annual.index]
    
    for year in years:
        vals = annual.loc[year]
        df_year = pd.DataFrame({"Well": vals.index.map(_norm_well_id), "Value": vals.values})
    
        merged = coords_df.copy()
        merged["Well"] = merged["Well"].apply(_norm_well_id)
        merged = merged.merge(df_year, on="Well", how="left")
    
        ok = (
            merged["Value"].apply(np.isfinite)
            & merged["X"].apply(np.isfinite)
            & merged["Y"].apply(np.isfinite)
        )
        pts = merged.loc[ok, ["Well", "X", "Y", "Value"]].dropna(subset=["X", "Y", "Value"])
    
        print(f"[INFO] Year {year}: {len(pts)} wells with annual values")
        if len(pts) < 3:
            print(f"[WARN] Not enough wells with values to interpolate for year {year}. Skipping.")
            continue
    
        Xi, Yi, Zi, extent = _idw_grid(
            pts["X"].to_numpy(float),
            pts["Y"].to_numpy(float),
            pts["Value"].to_numpy(float),
            xminp, xmaxp, yminp, ymaxp,
            nx=300, power=2,
        )
    
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
    
        im = ax.imshow(
            Zi, origin="lower",
            extent=[extent[0], extent[1], extent[2], extent[3]],
            cmap=cmap_fixed, vmin=VMIN, vmax=VMAX,
            interpolation="nearest", zorder=3
        )
    
        # --- 10 m aralıklı DÜZ konturlar (solid lines), okunaklı etiket ---
        levels = np.arange(VMIN, VMAX + 1, 10)
        cs = ax.contour(Xi, Yi, Zi, levels=levels, colors="k",
                        linewidths=0.8, linestyles="solid", alpha=0.9, zorder=4)
        try:
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f", inline_spacing=2)
        except Exception:
            pass
    
        # Kuyular
        ax.scatter(
            pts["X"], pts["Y"], s=36, c="forestgreen",
            edgecolor="darkgreen", linewidth=0.7, zorder=5
        )
        for _, r in pts.iterrows():
            ax.annotate(
                str(r["Well"]), (r["X"], r["Y"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=7, color="k", zorder=6
            )
    
        # Çerçeve & eksen
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("black"); spine.set_linewidth(1.0)
    
        # --- YIL ETİKETİ: Sağ üstte, diğer haritalarla aynı stil ---
        add_year_tag(ax, year)
    
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Annual mean groundwater depth (m)")  # isterseniz: "DTW (m bgs)"
    
        out_path = annual_maps_out / f"annual_gw_map_{year}.tiff"
        fig.savefig(out_path, dpi=400, format="tiff", bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

    # === Fig. 10 — Sen slope (m/year) [IDW], dynamic scale + 0.5 m/year contours ===
    def _round_to_step(x, step=0.2, how="floor"):
        if not np.isfinite(x):
            return np.nan
        q = x / step
        if how == "floor":
            return step * np.floor(q)
        elif how == "ceil":
            return step * np.ceil(q)
        else:
            return step * np.round(q)
    
    pts_tr = trend_map[
        np.isfinite(trend_map["X"]) &
        np.isfinite(trend_map["Y"]) &
        np.isfinite(trend_map["slope_m_per_year"])
    ].copy()
    
    trend_map.loc[:, ["Well", "X", "Y", "slope_m_per_year", "tau", "p"]].to_csv(
        OUT_TBLS / "fig10_trend_values.csv", index=False
    )
    
    if len(pts_tr) < 3:
        print("[WARN] Fig. 10: at least 3 points are needed for IDW. Skipped.")
    else:
        xmin, xmax = coords_df["X"].min(), coords_df["X"].max()
        ymin, ymax = coords_df["Y"].min(), coords_df["Y"].max()
        dx, dy = (xmax - xmin) * 0.10, (ymax - ymin) * 0.10
        xminp, xmaxp = xmin - dx, xmax + dx
        yminp, ymaxp = ymin - dy, ymax + dy
    
        Xi, Yi, Zi, extent = _idw_grid(
            pts_tr["X"].values, pts_tr["Y"].values, pts_tr["slope_m_per_year"].values,
            xminp, xmaxp, yminp, ymaxp, nx=300, power=2
        )
    
        slopes = pts_tr["slope_m_per_year"].values.astype(float)
        vmin_raw = float(np.nanmin(slopes))
        vmax_raw = float(np.nanmax(slopes))
        if vmax_raw <= 0: vmax_raw = 0.0
        if vmin_raw >= 0: vmin_raw = 0.0
        vmin = _round_to_step(vmin_raw, step=0.2, how="floor")
        vmax = _round_to_step(vmax_raw, step=0.2, how="ceil")
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = -0.2, 0.2
    
        levels = np.arange(vmin, vmax + 0.5, 0.5)
    
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        im = ax.imshow(
            Zi, origin="lower",
            extent=[extent[0], extent[1], extent[2], extent[3]],
            cmap="RdYlBu", vmin=vmin, vmax=vmax
        )
    
        cs = ax.contour(Xi, Yi, Zi, levels=levels, colors="k", linewidths=0.5, alpha=0.7)
        if (vmin < 0) and (vmax > 0):
            try:
                ax.contour(Xi, Yi, Zi, levels=[0.0], colors="k", linewidths=1.2)
            except Exception:
                pass
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    
        ax.scatter(pts_tr["X"], pts_tr["Y"], s=26, c="white",
                   edgecolor="black", linewidth=0.8, zorder=3)
    
        sig = (trend_map["p"] < 0.05) & np.isfinite(trend_map["p"])
        ax.scatter(trend_map.loc[sig, "X"], trend_map.loc[sig, "Y"],
                   marker="+", s=80, c="black", linewidths=1.4, zorder=4)

        # After plotting all wells with slope-based color...
        try:
            if "trend_df_depth" in globals() and "sig_FDR05" in trend_df_depth.columns:
                sig = trend_df_depth["sig_FDR05"].to_numpy(dtype=bool)
                xS = trend_df_depth.loc[sig, "X"].to_numpy(float)
                yS = trend_df_depth.loc[sig, "Y"].to_numpy(float)
                if xS.size > 0:
                    ax.scatter(xS, yS, s=90, facecolors="none", edgecolors="black",
                               linewidths=1.5, zorder=10)  # kalın siyah halka
        except Exception:
            pass
    
        for _, r in pts_tr.iterrows():
            ax.annotate(str(r["Well"]), (r["X"], r["Y"]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color="k", zorder=5)
    
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
    
        ax.set_title("Fig. 10 — Sen slope (m/year) [IDW] and MK significance (p<0.05: +)")
    
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Sen slope (m/year)")
        cbar.set_ticks(levels)
    
        plt.savefig(OUT_FIGS / "fig10_trend_map.tiff", dpi=400, format="tiff")
        plt.close()

        # ==== Apply FDR to trend p-values, then use in Fig.10 mapping ============
        try:
            if "trend_df_depth" in globals() and isinstance(trend_df_depth, pd.DataFrame) and not trend_df_depth.empty:
                trend_df_depth["sig_FDR05"] = fdr_bh(trend_df_depth["p"].values, alpha=0.05)
                (OUT_TBLS if "OUT_TBLS" in globals() else OUT_FIGS).mkdir(parents=True, exist_ok=True)
                out_tbl_dir = OUT_TBLS if "OUT_TBLS" in globals() else OUT_FIGS
                trend_df_depth.to_csv(out_tbl_dir / "fig10_trend_table_FDR.csv", index=False)
            else:
                print("[INFO] FDR: trend_df_depth not found or empty; skipping.")
        except Exception as e:
            print("[WARN] FDR failed:", e)

        print("[OK] Fig. 10 saved ->", OUT_FIGS / "fig10_trend_map.tiff")

    # === 11) GW-NDSPI MAP (2016–2024): Sum of SGI < -1 — IDW ===
    if isinstance(sgi.index, pd.DatetimeIndex):
        sgi_period = sgi.loc["2016-01-01":"2024-12-31"]
    else:
        sgi_period = sgi.copy()
    
    sgi_ndspi = sgi_period.where(sgi_period < -1.0, 0.0).sum(axis=0)
    df_ndspi = pd.DataFrame({"Well": sgi_ndspi.index.map(_norm_well_id), "Value": sgi_ndspi.values})
    
    merged_b = coords_df.copy()
    merged_b["Well"] = merged_b["Well"].apply(lambda x: _norm_well_id(x))
    merged_b = merged_b.merge(df_ndspi, on="Well", how="left")
    
    ok_b = merged_b["Value"].apply(np.isfinite) & merged_b["X"].apply(np.isfinite) & merged_b["Y"].apply(np.isfinite)
    pts_b = merged_b.loc[ok_b, ["Well", "X", "Y", "Value"]].dropna(subset=["X", "Y", "Value"])
    
    if len(pts_b) < 3:
        print("[WARN] Fig. 11: at least 3 points are needed for IDW. Skipped.")
    else:
        xmin, xmax = coords_df["X"].min(), coords_df["X"].max()
        ymin, ymax = coords_df["Y"].min(), coords_df["Y"].max()
        dx, dy = (xmax - xmin)*0.10, (ymax - ymin)*0.10
        xminp, xmaxp, yminp, ymaxp = xmin - dx, xmax + dx, ymin - dy, ymax + dy
    
        Xi, Yi, Zi, extent = _idw_grid(pts_b["X"].values, pts_b["Y"].values, pts_b["Value"].values,
                                       xminp, xmaxp, yminp, ymaxp, nx=300, power=2)
    
        data_min_b = float(np.nanmin(pts_b["Value"].values))
        data_max_b = float(np.nanmax(pts_b["Value"].values))
        vmin_b = _round_to_base(data_min_b, base=2, how="floor")
        vmax_b = _round_to_base(data_max_b, base=2, how="ceil")
        if not np.isfinite(vmin_b) or not np.isfinite(vmax_b) or vmin_b == vmax_b:
            vmin_b, vmax_b = -2, 2
        levels_b = np.arange(vmin_b, vmax_b + 2, 2)
    
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        im = ax.imshow(Zi, origin="lower",
                       extent=[extent[0], extent[1], extent[2], extent[3]],
                       cmap="RdYlBu", vmin=vmin_b, vmax=vmax_b)
    
        cs = ax.contour(Xi, Yi, Zi, levels=levels_b, colors="k", linewidths=0.5, alpha=0.75)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    
        ax.scatter(pts_b["X"], pts_b["Y"], s=36, c="forestgreen", edgecolor="darkgreen", linewidth=0.7, zorder=3)
        for _, r in pts_b.iterrows():
            ax.annotate(str(r["Well"]), (r["X"], r["Y"]), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color="k")
    
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
        ax.set_title("Fig. 11 — GW-NDSPI (sum of SGI < -1, 2016–2024) — IDW")
    
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("GW-NDSPI (sum of SGI < -1)")
        cbar.set_ticks(levels_b)
    
        df_ndspi.to_csv(OUT_TBLS / "fig11b_gw_ndspi_table.csv", index=False)
        plt.savefig(OUT_FIGS / "fig11b_gw_ndspi_map.tiff", dpi=400, format="tiff")
        plt.close()
   
    print(f"[OK] Tables -> {OUT_TBLS.resolve()}")
    print(f"[OK] Yearly metrics -> { (OUT_TBLS / 'drought_metrics_yearly.csv').resolve() }")
    print(f"[OK] Figures -> {OUT_FIGS.resolve()}")
    print(f"[OK] Spatial plots -> {OUT_SPATIAL.resolve()}")
    print(f"[OK] SGI series -> {OUT_SGI_SERIES.resolve()}")

    # 12 ===================== Annual WSE & anomaly maps with COMMON scales (EPSG:3857, topo base) =====================
    # --- Kullanıcı global/varsayılanları (varsa sizin değerleriniz önceliklidir) ---
    NX_IDW   = int(globals().get("NX_IDW", 320))
    IDW_PWR  = int(globals().get("IDW_PWR", 2))
    
    WELL_FACE_COLOR = globals().get("WELL_FACE_COLOR", "green")
    WELL_EDGE_COLOR = globals().get("WELL_EDGE_COLOR", "black")
    WELL_SIZE       = int(globals().get("WELL_SIZE", 42))
    WELL_LINEWIDTH  = float(globals().get("WELL_LINEWIDTH", 0.6))
    
    # --- Çıktı klasörleri (3857) ---
    OUT_ANNUAL_WSE_COMMON_3857   = Path("./out_annual_gw_maps_EN_common_3857")
    OUT_ANNUAL_WSE_COMMON_3857_CROPPED = Path("./out_annual_gw_maps_EN_common_3857_cropped")
    OUT_ANNUAL_ANOM_COMMON_3857  = Path("./out_annual_anomaly_maps_EN_common_3857")
    OUT_ANOM_CROPPED_3857        = Path("./out_annual_anomaly_maps_EN_common_3857_cropped")
    for p in (OUT_ANNUAL_WSE_COMMON_3857, OUT_ANNUAL_ANOM_COMMON_3857, OUT_ANOM_CROPPED_3857):
        p.mkdir(parents=True, exist_ok=True)
    
    # --- Zoom kestirimi: EPSG:3857 extente göre güvenli seviye (0-19) ---
    def _ctx_zoom_for_extent(extent, min_zoom=6, max_zoom=19):
        x0, x1, y0, y1 = map(float, extent)
        span = max(abs(x1 - x0), abs(y1 - y0))
        if span < 50_000:       z = 13
        elif span < 150_000:    z = 12
        elif span < 400_000:    z = 11
        elif span < 800_000:    z = 10
        else:                   z = 9
        return max(min(z, max_zoom), min_zoom)
    
    # --- IDW (nx→ny oranı extente göre ayarlanır) ---
    def _idw_grid_local(x, y, z, xmin, xmax, ymin, ymax, nx=300, power=2, eps=1e-12):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); z = np.asarray(z, dtype=float)
        xr = xmax - xmin; yr = ymax - ymin
        nx = int(nx); ny = max(2, int(round(nx * (yr / xr)))) if xr > 0 else nx
        xi = np.linspace(xmin, xmax, nx); yi = np.linspace(ymin, ymax, ny)
        Xi, Yi = np.meshgrid(xi, yi)
        dx = Xi[None, :, :] - x[:, None, None]
        dy = Yi[None, :, :] - y[:, None, None]
        dist = np.hypot(dx, dy) + eps
        valid = np.isfinite(z)
        if valid.sum() < 3:
            return Xi, Yi, np.full_like(Xi, np.nan), [xmin, xmax, ymin, ymax]
        w = 1.0 / (dist[valid] ** power)
        zv = z[valid][:, None, None]
        Zi = np.nansum(w * zv, axis=0) / np.nansum(w, axis=0)
        return Xi, Yi, Zi, [xmin, xmax, ymin, ymax]
    
    # --- Basemap ekleyici (dosyanızdaki add_basemap_color_only ile uyumlu) ---
    def _add_basemap(ax, extent, provider=BASEMAP_PROVIDER, alpha=0.95):
        try:
            zoom = _ctx_zoom_for_extent(extent)
            add_basemap_color_only(ax, crs="EPSG:3857", topo=provider, alpha=alpha, zoom=zoom)
        except Exception as e:
            print("[WARN] Basemap add failed:", e)
    
    # --- Colorbar’ı eksenle aynı yükseklikte yapan yardımcı ---
    def _add_matched_cbar(ax, im, label, size="3.8%", pad=0.18, style_cb=True):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=size, pad=pad)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(label)
        try:
            _style_colorbar_fat_outline(cb)
        except Exception:
            pass
        return cb
   
    # --- Figür boyutunu extente göre belirle (oran korunur) ---
    def _figsize_from_extent(x0, x1, y0, y1, target_h_in=6.2, min_h=4.8, max_h=8.5):
        w = float(x1 - x0); h = float(y1 - y0)
        if not (np.isfinite(w) and np.isfinite(h)) or w <= 0 or h <= 0:
            return (7.2, 5.6)
        ar = w / h
        H = float(np.clip(target_h_in, min_h, max_h))
        W = float(np.clip(H * ar, 5.0, 12.0))
        return (W, H)
    
    # --- Koordinatları EPSG:3857’e çevir ---
    try:
        coords_3857 = to_target_coords_df(coords_df, target="EPSG:3857")
    except Exception as e:
        print("[WARN] to_target_coords_df failed:", e)
        coords_3857 = None
    
    # ---- 'annual' kontrolü ve sayısallaştırma ----
    if isinstance(annual, pd.DataFrame) and (not annual.empty):
        annual = annual.apply(pd.to_numeric, errors="coerce")
        years = list(annual.index)
    else:
        years = []
    
    if years and (coords_3857 is not None) and (not coords_3857.empty):
        # Hızlı erişim için Well index'i
        try:
            coords_3857_idx = coords_3857.set_index("Well", drop=False)
        except Exception:
            coords_3857_idx = coords_3857.copy()
            if "Well" not in coords_3857_idx.columns:
                raise ValueError("coords_3857 does not contain a 'Well' column.")
    
        # ----- Anomali (BASELINE_YEARS varsa o yılların ort., yoksa ilk 1–3 yıl) -----
        try:
            years_avail = list(annual.index)
            baseline_years = [y for y in globals().get("BASELINE_YEARS", []) if y in years_avail]
            if len(baseline_years) == 0:
                baseline_years = years_avail[:min(3, len(years_avail))]
            baseline_per_well = pd.to_numeric(annual.loc[baseline_years].mean(axis=0), errors="coerce")
            annual_anom = annual.subtract(baseline_per_well, axis=1).apply(pd.to_numeric, errors="coerce")
        except Exception as e:
            print("[WARN] Could not compute annual anomalies:", e)
            annual_anom = pd.DataFrame(index=annual.index, columns=annual.columns, dtype=float)
    
        # ----- Ortak renk aralıkları -----
        # WSE absolute: robust 2–98% (fallback min/max)
        try:
            flat_wse = np.asarray(annual.to_numpy(), dtype=float)
            flat_wse = flat_wse[np.isfinite(flat_wse)]
            if flat_wse.size >= 10:
                WSE_VMIN, WSE_VMAX = np.nanpercentile(flat_wse, [2, 98]).astype(float)
            elif flat_wse.size >= 2:
                WSE_VMIN, WSE_VMAX = float(np.nanmin(flat_wse)), float(np.nanmax(flat_wse))
            else:
                WSE_VMIN, WSE_VMAX = 0.0, 1.0
            if not (np.isfinite(WSE_VMIN) and np.isfinite(WSE_VMAX) and (WSE_VMAX > WSE_VMIN)):
                WSE_VMIN, WSE_VMAX = 0.0, 1.0
        except Exception:
            WSE_VMIN, WSE_VMAX = 0.0, 1.0
    
        # Anomali: simetrik ortak skala (robust genlik)
        try:
            flat_an = np.asarray(annual_anom.to_numpy(), dtype=float)
            flat_an = flat_an[np.isfinite(flat_an)]
            if flat_an.size >= 10:
                a2, a98 = np.nanpercentile(flat_an, [2, 98])
                A = float(max(abs(a2), abs(a98)))
                if not (np.isfinite(A) and A > 0):
                    A = float(np.nanmax(np.abs(flat_an))) if flat_an.size else 1.0
            elif flat_an.size >= 1:
                A = float(np.nanmax(np.abs(flat_an)))
                if not (np.isfinite(A) and A > 0):
                    A = 1.0
            else:
                A = 1.0
            ANOM_VMIN, ANOM_VMAX = -A, +A
        except Exception:
            ANOM_VMIN, ANOM_VMAX = -1.0, +1.0
    
        # ----- Grid sınırları (+%10 tampon) -----
        try:
            xmin = float(coords_3857["X_3857"].min()); xmax = float(coords_3857["X_3857"].max())
            ymin = float(coords_3857["Y_3857"].min()); ymax = float(coords_3857["Y_3857"].max())
            dx, dy = (xmax - xmin) * 0.10, (ymax - ymin) * 0.10
            gx0, gx1, gy0, gy1 = xmin - dx, xmax + dx, ymin - dy, ymax + dy
        except Exception as e:
            print("[WARN] 3857 bounds failed:", e)
            gx0=gx1=gy0=gy1=None
    
        if all(v is not None for v in (gx0, gx1, gy0, gy1)):
    
            # ========== ABSOLUTE WSE (Spectral), COMMON SCALE ==========
            for y in _tqdm(list(years), desc="Annual DTW (common, 3857)"):
                try:
                    ok_cols = [c for c in annual.columns if c in coords_3857_idx.index]
                    if not ok_cols:
                        continue
    
                    vals = pd.to_numeric(annual.loc[y, ok_cols], errors="coerce").to_numpy(dtype=float)
                    x_m  = coords_3857_idx.loc[ ok_cols, "X_3857"].to_numpy(dtype=float)
                    y_m  = coords_3857_idx.loc[ ok_cols, "Y_3857"].to_numpy(dtype=float)
                    mask = np.isfinite(vals) & np.isfinite(x_m) & np.isfinite(y_m)
                    if mask.sum() < 3:
                        continue
    
                    Xi, Yi, Zi, extent = _idw_grid_local(x_m[mask], y_m[mask], vals[mask],
                                                   gx0, gx1, gy0, gy1, nx=NX_IDW, power=IDW_PWR)
    
                    if not np.isfinite(Zi).any():
                        continue
    
                    W_in, H_in = _figsize_from_extent(*extent)
                    fig, ax = plt.subplots(figsize=(W_in, H_in))
    
                    # Basemap
                    _add_basemap(ax, extent, provider=BASEMAP_PROVIDER, alpha=0.95)
    
                    im = ax.imshow(Zi, origin="lower", extent=extent, cmap="Spectral",
                                   vmin=WSE_VMIN, vmax=WSE_VMAX, interpolation="nearest", zorder=3)

                    # --- 10 m aralıklı DÜZ konturlar (solid lines) ---
                    # Aralığı 10'a yuvarlayarak sabitle
                    lo = np.floor(WSE_VMIN / 10.0) * 10.0
                    hi = np.ceil (WSE_VMAX / 10.0) * 10.0
                    levels = np.arange(lo, hi + 1, 10.0)
                    
                    cs = ax.contour(
                        Xi, Yi, Zi, levels=levels,
                        colors="k", linewidths=0.8, linestyles="solid",
                        alpha=0.9, zorder=4
                    )
                    try:
                        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f", inline_spacing=2)
                    except Exception:
                        pass
    
                    ax.scatter(x_m[mask], y_m[mask], s=WELL_SIZE, c=WELL_FACE_COLOR,
                               edgecolor=WELL_EDGE_COLOR, linewidth=WELL_LINEWIDTH, zorder=5)
    
                    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
                    ax.set_aspect("equal", adjustable="box")
                    year_text = _safe_year_tag(ax, y)
    
                    try:
                        add_scalebar(ax, length_km=20, location="lower left")
                        add_north_arrow_above_scalebar(ax, length=0.08, fontsize=10, lw=1.2, head_length=12)
                    except Exception:
                        pass
    
                    _add_matched_cbar(ax, im, "DTW (m bgs)")
                    ax.set_title(f"Annual DTW — {y} (common scale, EPSG:3857)")
                    ax.set_xticks([]); ax.set_yticks([])
                    
                    # full panel (keeps colorbar/title)    
                    fig.savefig(OUT_ANNUAL_WSE_COMMON_3857 / f"annual_dtw_map_{y}.tiff",
                                dpi=400, format="tiff", bbox_inches="tight", pad_inches=0.08)
                    
                    # cropped map-only export with thin gray frame
                    cropped_path = OUT_ANNUAL_WSE_COMMON_3857_CROPPED / f"annual_dtw_map_{y}.tiff"
                    _save_axes_cropped(
                        fig, ax, cropped_path,
                        dpi=400,
                        pad_inches=0.0,
                        spine_color="#7a7a7a",   # light gray
                        spine_lw=0.8,
                        keep_ticks=False,         # keep it clean
                        hide_title=True
                    )
                    
                    plt.close(fig)
                except Exception as e:
                    print(f"[WARN] Annual DTW map {y} failed:", e)
    
            # ========== ANOMALY (RdBu; red = negative), COMMON SYMMETRIC SCALE ==========
            for y in _tqdm(list(years), desc="Annual DTW anomaly (common, 3857)"):
                try:
                    if annual_anom.empty:
                        break
                    ok_cols = [c for c in annual_anom.columns if c in coords_3857_idx.index]
                    if not ok_cols:
                        continue
    
                    vals = pd.to_numeric(annual_anom.loc[y, ok_cols], errors="coerce").to_numpy(dtype=float)
                    x_m  = coords_3857_idx.loc[ ok_cols, "X_3857"].to_numpy(dtype=float)
                    y_m  = coords_3857_idx.loc[ ok_cols, "Y_3857"].to_numpy(dtype=float)
                    mask = np.isfinite(vals) & np.isfinite(x_m) & np.isfinite(y_m)
                    if mask.sum() < 3:
                        continue
    
                    Xi, Yi, Zi, extent = _idw_grid_local(x_m[mask], y_m[mask], vals[mask],
                                                   gx0, gx1, gy0, gy1, nx=NX_IDW, power=IDW_PWR)
                    if not np.isfinite(Zi).any():
                        continue
    
                    W_in, H_in = _figsize_from_extent(*extent)
                    fig, ax = plt.subplots(figsize=(W_in, H_in))
    
                    _add_basemap(ax, extent, provider=BASEMAP_PROVIDER, alpha=0.95)
    
                    im = ax.imshow(Zi, origin="lower", extent=extent, cmap="RdBu",
                                   vmin=ANOM_VMIN, vmax=ANOM_VMAX, interpolation="nearest", zorder=3)

                    _apply_year_axis_labels_and_boundaries(ax, years)
    
                    ax.scatter(x_m[mask], y_m[mask], s=WELL_SIZE, c=WELL_FACE_COLOR,
                               edgecolor=WELL_EDGE_COLOR, linewidth=WELL_LINEWIDTH, zorder=5)
    
                    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
                    ax.set_aspect("equal", adjustable="box")
                    try:
                        add_scalebar(ax, length_km=20, location="lower left")
                        add_north_arrow_above_scalebar(ax, length=0.08, fontsize=10, lw=1.2, head_length=12)
                    except Exception:
                        pass
    
                    # Yıl etiketi (sağ üst)
                    year_text = ax.text(0.985, 0.985, f"{y}",
                                        transform=ax.transAxes, ha="right", va="top",
                                        fontsize=12, fontweight="bold", color="rebeccapurple", zorder=20)
                    try:
                        import matplotlib.patheffects as pe
                        year_text.set_path_effects([pe.withStroke(linewidth=1.2, foreground="lavender")])
                    except Exception:
                        pass
    
                    cb = _add_matched_cbar(ax, im, "DTW anomaly (m bgs)")
                    ax.set_title(f"Annual WSE anomaly — {y} (common symmetric scale, EPSG:3857)")
                    ax.set_xticks([]); ax.set_yticks([])
    
                    # --- FULL kayıt ---
                    full_path = OUT_ANNUAL_ANOM_COMMON_3857 / f"annual_dtw_anomaly_map_{y}.tiff"
                    fig.savefig(full_path, dpi=400, format="tiff", bbox_inches="tight", pad_inches=0.08)
                    ax.set_title("")   # no title in the cropped version
                    try:
                        cb.remove()    # no colorbar in the cropped version
                    except Exception:
                        pass    
                    # --- CROPPED (sadece harita ekseni) ---
                    crop_path = OUT_ANOM_CROPPED_3857 / f"annual_dtw_anomaly_map_{y}.tiff"
                    _save_axes_cropped(fig, ax, crop_path, dpi=400, pad_inches=0.08)
    
                    plt.close(fig)
                except Exception as e:
                    print(f"[WARN] Annual anomaly map {y} failed:", e)
    else:
        print("[INFO] Annual DTW/anomaly common-scale maps skipped: 'annual' empty or coords_3857 missing.")

    print("[STEP] WSE time series for each well…")   
    # 13 --- WSE time series for each well
    export_wse_series(df, out_dir=Path("./out_DTW_series_EN"))


    print("[STEP] GEV-based return levels for yearly minimum SGI…")
    # 14 === EXTREMES: GEV-based return levels for yearly minimum SGI ==================
    try:
        OUT_EXTREMES = Path("./out_SGI_extremes_EN")
        OUT_EXTREMES.mkdir(parents=True, exist_ok=True)
    
        # 1) hesapla ve tabloyu kaydet
        gev_tbl = fit_gev_return_periods_min_sgi(sgi, years=(10, 25, 50))  # b comes from GEV_BOOTSTRAP_B or defaults to 150
        if not gev_tbl.empty:
            (OUT_EXTREMES / "tables").mkdir(exist_ok=True, parents=True)
            gev_tbl.to_csv(OUT_EXTREMES / "tables" / "minSGI_GEV_return_levels.csv", index=False)
    
            # 2) mekânsal raster: her T için xT_minSGI’yi enterpole edip çiz
            # extent: tüm kuyular (10% padding) — California stilinizle aynı
            coordsX = coords_df.copy()
            coordsX["Well"] = coordsX["Well"].apply(_norm_well_id)
            _xmin, _xmax = float(coordsX["X"].min()), float(coordsX["X"].max())
            _ymin, _ymax = float(coordsX["Y"].min()), float(coordsX["Y"].max())
            _dx, _dy     = 0.10 * (_xmax - _xmin), 0.10 * (_ymax - _ymin)
            GX0, GX1, GY0, GY1 = _xmin - _dx, _xmax + _dx, _ymin - _dy, _ymax + _dy
            X_ALL = coordsX["X"].to_numpy(float); Y_ALL = coordsX["Y"].to_numpy(float)
    
            # colormap ve sınırlar (global, tüm kuyular için)
            # MinSGI return level → tipik aralık ~ [-3, 0]; robust aralık kullanalım
            try:
                all_xT = pd.to_numeric(gev_tbl["xT_minSGI"], errors="coerce").to_numpy(float)
                lo, hi = np.nanpercentile(all_xT[np.isfinite(all_xT)], [2, 98])
            except Exception:
                lo, hi = -3.0, 0.0
            vmin, vmax = float(lo), float(max(0.0, hi)) if np.isfinite(hi) else (-3.0, 0.0)
    
            # tqdm
            PB = _tqdm if "_tqdm" in globals() and callable(_tqdm) else tqdm
    
            for Ty in PB(sorted(gev_tbl["T_year"].unique()), desc="GEV return maps (MinSGI)"):
                G = gev_tbl.loc[gev_tbl["T_year"] == Ty, ["Well", "xT_minSGI"]].merge(
                        coordsX[["Well","X","Y"]], on="Well", how="inner")
                ok = np.isfinite(G["xT_minSGI"]) & np.isfinite(G["X"]) & np.isfinite(G["Y"])
                if ok.sum() < 3:
                    continue
                x_use = G.loc[ok, "X"].to_numpy(float)
                y_use = G.loc[ok, "Y"].to_numpy(float)
                v_use = G.loc[ok, "xT_minSGI"].to_numpy(float)
    
                Xi, Yi, Zi, extent = (_interp_grid if "_interp_grid" in globals() and callable(_interp_grid)
                                      else _idw_grid)(x_use, y_use, v_use, GX0, GX1, GY0, GY1,
                                                      nx=int(globals().get("NX_IDW", 320)),
                                                      power=int(globals().get("IDW_PWR", 2)))
                if not np.isfinite(Zi).any():
                    continue
    
                # figure size
                if "FIG_W" in globals() and "FIG_H" in globals():
                    FW, FH = float(FIG_W), float(FIG_H)
                else:
                    FW, FH = (24.0/2.54, 12.0/2.54)
    
                fig, ax = plt.subplots(figsize=(FW, FH))
                im = ax.imshow(Zi, origin="lower", extent=(GX0,GX1,GY0,GY1),
                               cmap="Spectral", vmin=vmin, vmax=vmax,
                               interpolation="nearest", aspect="equal", zorder=2)

                _apply_year_axis_labels_and_boundaries(ax, years)

                # backdrop + panel noktaları
                ax.scatter(X_ALL, Y_ALL, s=8, c="lightgray", edgecolor="none", zorder=3)
                ax.scatter(x_use, y_use, s=16, c="xkcd:true green",
                           edgecolor="xkcd:pine green", linewidth=0.5, zorder=4)
    
                ax.set_xlim(GX0,GX1); ax.set_ylim(GY0,GY1)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
    
                cbar = plt.colorbar(im, ax=ax); cbar.set_label(f"Return level of yearly min SGI (T={Ty} yr)")
                ax.set_title(f"GEV return level of yearly minimum SGI — T={Ty} years")
                save_tiff(OUT_EXTREMES / f"minSGI_returnlevel_T{Ty}.tiff")
                plt.close(fig)
        else:
            print("[INFO] GEV: No results; SGI series too short.")
    except Exception as e:
        print("[WARN] GEV return map failed:", e)

if __name__ == "__main__":
    main()

