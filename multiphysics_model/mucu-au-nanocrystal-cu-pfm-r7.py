"""
Au-assisted Cu-Cu diffusion bonding simulator — rectangular MPF edition (v3)
=============================================================================
Physics: MPF + Fickian/Hart diffusion + Solute Drag + Elastoplasticity.

MAJOR CHANGES in v3
-------------------
Bug fixes (from review):
  * handle_errors no longer silently returns None where callers unpack tuples;
    decorated methods now return a typed sentinel that callers check, or
    re-raise (configurable). UI-level wrappers catch and display.
  * Cahn-Hilliard path now auto-substeps to respect a nonlinear CFL.
  * "FIX C" mass conservation now *actually* rejects blown-up steps
    (rolls back to the previous c) instead of silently accepting them.
  * compute_local_grain_size now returns a physically meaningful
    distance-to-nearest-GB-based grain size instead of 2/|∇s|.
  * Plane-strain elastic moduli corrected: C11_2d=C11, C12_2d=C12, C44_2d=C44.
  * Plastic flow direction now uses a J2 return-mapping step driven by the
    local deviatoric stress, not a fixed xx+/yy- direction.
  * torch import wrapped in try/except (like h5py).
  * Animation vmin/vmax now computed over ALL frames (no clipping drift).
  * export_csv sigma_y branch made reachable.
  * SimulationDatabase gains clear_all() / bounded cache.

Visualization upgrades:
  * Full Visualization Settings tab: global theme, fonts, DPI, colorbar layout.
  * Per-field dynamic range: vmin/vmax overrides, log-norm, colormap invert.
  * Multi-field overlays: base heatmap + overlay heatmap OR contour lines,
    with alpha blending and optional grain-map ghosting.
  * Vector field visualization: quiver (matplotlib) and Plotly cone/scatter
    for ∇c, ∇σ_h, principal stress directions.
  * Plotly time-scrubbing animation (frames + slider + play button),
    no GIF rendering required.
  * Multi-time line profiles: overlay profiles from multiple frames.
  * Animation export: GIF/MP4, watermark, batch (all fields → zip).
  * Publication-ready matplotlib style presets (including dark mode).
"""

import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import (Colormap, LinearSegmentedColormap,
                                ListedColormap, LogNorm)
from matplotlib.ticker import AutoMinorLocator
import matplotlib.animation as animation
import plotly.graph_objects as go
from datetime import datetime
import json, zipfile, pickle, hashlib, traceback, warnings
from io import BytesIO, StringIO
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SENTINEL FOR DECORATED FAILURES
# ============================================================================
class _Sentinel:
    """Typed sentinel returned by decorated functions on failure."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reason = None
        return cls._instance
    def __bool__(self): return False
    def __repr__(self): return f"<FAILED: {self.reason}>"

FAILED = _Sentinel()


# ============================================================================
# ERROR HANDLING (BUG FIX P0)
# ============================================================================
def handle_errors(func=None, *, re_raise=False):
    """
    Decorator. On exception:
      - logs the traceback
      - if re_raise=True: re-raises
      - else: stores reason on FAILED sentinel and returns it
    Callers MUST check `if result is FAILED` or `if not result` before unpacking.
    """
    def deco(f):
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                FAILED.reason = f"{f.__name__}: {e}"
                logger.error(f"{e}\n{traceback.format_exc()}")
                try:
                    st.error(f"❌ Error in {f.__name__}: {e}")
                except Exception:
                    pass
                if re_raise:
                    raise
                return FAILED
        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__
        return wrapper

    if func is None:
        return deco
    return deco(func)


def safe_unpack(result, n, default=None):
    """Unpack a decorated-return; if FAILED, return `default` n-tuple."""
    if result is FAILED or result is None:
        if default is None:
            return tuple([None] * n)
        return default
    return result


# ============================================================================
# RECTANGULAR SPECTRAL OPERATORS
# ============================================================================
def make_k_vectors(Nx, Ny, dx, dy=None):
    if dy is None:
        dy = dx
    kx = 2 * np.pi * fftfreq(Nx, d=dx).reshape(1, -1)
    ky = 2 * np.pi * fftfreq(Ny, d=dy).reshape(-1, 1)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1e-12
    return kx, ky, k2


def spectral_gradients(field, kx, ky):
    fh = fft2(field)
    return (np.real(ifft2(1j * kx * fh)),
            np.real(ifft2(1j * ky * fh)))


def spectral_laplacian(field, k2):
    return np.real(ifft2(-k2 * fft2(field)))


def spectral_divergence(fx, fy, kx, ky):
    return np.real(ifft2(1j * kx * fft2(fx) + 1j * ky * fft2(fy)))


# ============================================================================
# INTERFACE PROFILING HELPERS
# ============================================================================
def _extract_interface_profile(interface_2d):
    a = np.asarray(interface_2d)
    if a.ndim == 0:
        return None
    if a.ndim == 1:
        return a
    if a.shape[0] > 1 and np.allclose(a[0, :], a[-1, :], atol=1e-9):
        return a[0, :]
    return np.mean(a, axis=0)


def _plot_interface(ax, x_coords, interface_2d, **line_kwargs):
    a = np.asarray(interface_2d)
    if a.ndim == 0 or (a.ndim == 1 and a.size == 1):
        ax.axhline(float(a), **line_kwargs)
        return
    profile = _extract_interface_profile(a)
    if profile is None:
        return
    ax.plot(x_coords, profile, **line_kwargs)


# ============================================================================
# COLORMAPS
# ============================================================================
def _build_custom_colormaps():
    au_conc = LinearSegmentedColormap.from_list('au_conc', [
        (0.00, '#0d1b2a'), (0.15, '#1b4f72'), (0.35, '#c9a227'),
        (0.60, '#f4d03f'), (0.85, '#fbea9a'), (1.00, '#fff8e0')])

    grain_map = ListedColormap([
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194',
        '#cedb9c', '#9c9ede', '#e7969c', '#e7ba52', '#e7cb94',
        '#de9ed6', '#adc6dc', '#bd9e39', '#c49c94', '#f7b6d2',
    ], name='grain_map')

    stress_pub = LinearSegmentedColormap.from_list('stress_pub', [
        (0.0, '#2c7bb6'), (0.25, '#abd9e9'), (0.5, '#ffffbf'),
        (0.75, '#fdae61'), (1.0, '#d7191c')])

    gb_hot = LinearSegmentedColormap.from_list('gb_hot', [
        (0.0, '#000004'), (0.3, '#420a68'), (0.6, '#932667'),
        (0.85, '#f8850f'), (1.0, '#fcffa4')])

    return {'au_conc': au_conc, 'grain_map': grain_map,
            'stress_pub': stress_pub, 'gb_hot': gb_hot}

_CUSTOM_COLORMAPS = _build_custom_colormaps()
for _n, _c in _CUSTOM_COLORMAPS.items():
    try:
        if _n not in mpl.colormaps:
            mpl.colormaps.register(_c, name=_n, force=True)
    except Exception as _e:
        logger.warning(f"Could not register colormap '{_n}': {_e}")


class ColormapRegistry:
    BASE = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo',
        'Greys', 'Blues', 'Greens', 'Reds', 'Oranges', 'Purples',
        'YlOrRd', 'YlOrBr', 'YlGnBu', 'YlGn', 'BuPu', 'PuRd', 'BuGn',
        'RdBu', 'RdBu_r', 'coolwarm', 'coolwarm_r', 'bwr', 'bwr_r',
        'seismic', 'seismic_r', 'Spectral', 'Spectral_r',
        'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
        'twilight', 'hsv', 'jet', 'hot', 'afmhot', 'bone', 'copper',
        'flag', 'prism', 'ocean', 'terrain', 'rainbow', 'nipy_spectral',
        'Pastel1', 'Paired', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20',
    ]
    @classmethod
    def list_all(cls):
        names = list(cls.BASE) + list(_CUSTOM_COLORMAPS.keys())
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n); out.append(n)
        return out

    @classmethod
    def resolve(cls, name, invert=False):
        if name is None:
            cmap = mpl.colormaps['viridis']
        elif isinstance(name, Colormap):
            cmap = name
        else:
            key = str(name)
            if key in mpl.colormaps:
                cmap = mpl.colormaps[key]
            elif key in _CUSTOM_COLORMAPS:
                cmap = _CUSTOM_COLORMAPS[key]
            else:
                logger.warning(f"Unknown colormap '{name}', using viridis")
                cmap = mpl.colormaps['viridis']
        if invert:
            cmap = cmap.reversed()
        return cmap

    @classmethod
    def to_plotly(cls, name, n=64, invert=False):
        cmap = cls.resolve(name, invert=invert)
        t = np.linspace(0, 1, n)
        rgb = (cmap(t)[:, :3] * 255).astype(int)
        return [[float(ti), f'rgb({r},{g},{b})'] for ti, (r, g, b) in zip(t, rgb)]


# ============================================================================
# PUBLICATION THEMES
# ============================================================================
MATPLOTLIB_THEMES = {
    'default':                None,
    'ggplot':                 'ggplot',
    'seaborn-whitegrid':      'seaborn-v0_8-whitegrid',
    'seaborn-darkgrid':       'seaborn-v0_8-darkgrid',
    'seaborn-ticks':          'seaborn-v0_8-ticks',
    'seaborn-paper':          'seaborn-v0_8-paper',
    'seaborn-talk':           'seaborn-v0_8-talk',
    'bmh':                    'bmh',
    'dark_background':        'dark_background',
    'classic':                'classic',
    'fivethirtyeight':        'fivethirtyeight',
    'Solarize_Light2':        'Solarize_Light2',
    'tableau-colorblind10':   'tableau-colorblind10',
    'nature-like (custom)':   '__nature__',
    'science-like (custom)':  '__science__',
}

_NATURE_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}

_SCIENCE_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.linewidth': 1.0,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
}


def apply_matplotlib_theme(theme_name: str, cfg: dict):
    """Apply a matplotlib theme and per-app font/DPI overrides."""
    plt.rcdefaults()
    style = MATPLOTLIB_THEMES.get(theme_name, None)
    if style == '__nature__':
        mpl.rcParams.update(_NATURE_RC)
    elif style == '__science__':
        mpl.rcParams.update(_SCIENCE_RC)
    elif style is not None:
        try:
            plt.style.use(style)
        except Exception as e:
            logger.warning(f"Theme '{style}' unavailable: {e}")
            plt.rcdefaults()
    # App-level overrides
    mpl.rcParams['axes.titlesize']  = cfg.get('title_font_size', 11)
    mpl.rcParams['axes.labelsize']  = cfg.get('label_font_size', 9)
    mpl.rcParams['xtick.labelsize'] = cfg.get('tick_font_size', 8)
    mpl.rcParams['ytick.labelsize'] = cfg.get('tick_font_size', 8)
    mpl.rcParams['legend.fontsize'] = cfg.get('legend_font_size', 8)
    mpl.rcParams['figure.dpi']      = cfg.get('dpi', 150)
    mpl.rcParams['savefig.dpi']     = cfg.get('dpi', 150)


# ============================================================================
# FIELD CONFIGURATION
# ============================================================================
FIELD_DEFAULT_CMAPS = {
    'phi':       'au_conc',
    'eta_poly':  'Blues',
    'eta_au':    'YlOrBr',
    'eta_nc':    'Greens',
    'grain_map': 'grain_map',
    'phi_gb':    'gb_hot',
    'D_eff':     'plasma',
    'sigma_eq':  'hot',
    'sigma_h':   'RdBu_r',
    'h':         'gray',
    'eps_p_mag': 'YlOrRd',
    'sigma_y':   'viridis',
    'rho_gnd':   'magma',
}

FIELD_LABELS = {
    'phi':       'c (Au concentration)',
    'eta_poly':  'η_poly (Poly-Cu aggregate)',
    'eta_au':    'η_Au (Au nanofilm)',
    'eta_nc':    'η_nc (nc-Cu aggregate)',
    'grain_map': 'Grain ID map (individual grains)',
    'phi_gb':    'φ_gb (GB indicator)',
    'D_eff':     'D_eff (Hart effective)',
    'sigma_eq':  'σ_eq (Von Mises)',
    'sigma_h':   'σ_h (Hydrostatic)',
    'h':         'Local grain size (nm)',
    'eps_p_mag': '|ε_p| (Plastic strain)',
    'sigma_y':   'σ_y (Yield stress)',
    'rho_gnd':   'ρ_GND (Dislocation density)',
}

FIELD_UNITS = {
    'phi': '', 'eta_poly': '', 'eta_au': '', 'eta_nc': '', 'grain_map': '',
    'phi_gb': '', 'D_eff': '',
    'sigma_eq': 'GPa', 'sigma_h': 'GPa', 'h': 'nm',
    'eps_p_mag': '', 'sigma_y': 'MPa', 'rho_gnd': '',
}

DISCRETE_FIELDS = {'grain_map', 'phi_gb'}

# Fields suitable for vector-field visualization (they have meaningful gradients)
VECTOR_FIELD_SOURCES = ['phi', 'sigma_h', 'sigma_eq', 'D_eff', 'phi_gb', 'rho_gnd']


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
def default_field_settings():
    return {
        f: {'auto': True, 'vmin': None, 'vmax': None,
            'log': False, 'invert': False}
        for f in FIELD_DEFAULT_CMAPS.keys()
    }


def default_viz_config():
    return {
        'theme': 'default',
        'title_font_size': 11,
        'label_font_size': 9,
        'tick_font_size': 8,
        'legend_font_size': 8,
        'dpi': 150,
        'colorbar_fraction': 0.046,
        'colorbar_pad': 0.04,
        'field_settings': default_field_settings(),
        'overlay': {
            'enabled': False,
            'base': 'sigma_eq',
            'overlay': 'phi_gb',
            'alpha': 0.5,
            'mode': 'contour',      # 'contour' or 'heatmap'
            'n_contours': 10,
            'show_grain_ghost': False,
        },
        'vector': {
            'enabled': False,
            'source': 'phi',        # which field to differentiate
            'mode': 'flux',         # 'flux' (J = -D∇c) or 'gradient' or 'principal_stress'
            'density': 24,
            'color': 'white',
            'scale': 1.0,
        },
        'animation': {
            'format': 'gif',
            'fps': 5,
            'dpi': 120,
            'watermark': True,
            'watermark_text': '',
            'watermark_pos': 'lower right',
        },
    }


def get_viz_config():
    if 'viz_cfg' not in st.session_state:
        st.session_state.viz_cfg = default_viz_config()
    return st.session_state.viz_cfg


# ============================================================================
# MATERIAL PROPERTIES
# ============================================================================
class MaterialProperties:
    @staticmethod
    def get_cu_properties():
        return {
            'elastic': {'C11': 168.4e9, 'C12': 121.4e9, 'C44': 75.4e9,
                        'source': 'PRB 73, 064112 (2006)'},
            'plasticity': {'mu': 48e9, 'nu': 0.34, 'b': 0.256e-9,
                           'sigma0': 50e6, 'gamma0_dot': 1e-3, 'm': 20},
            'misfit_beta': 0.04,
            'au_misfit_beta': 0.045,
        }

    @staticmethod
    def get_material(name='Cu'):
        return MaterialProperties.get_cu_properties()

    @staticmethod
    def validate_parameters(params):
        errors, warns = [], []
        if params.get('dt', 0) <= 0:  errors.append("dt must be positive")
        if params.get('dx', 0) <= 0:  errors.append("dx must be positive")
        if params.get('Nx', 0) < 32 or params.get('Ny', 0) < 32:
            warns.append("N < 32 per axis may be inaccurate")

        D_gb  = float(params.get('D_gb', 50.0))
        D_lat = float(params.get('D_lattice', 0.01))
        delta_D_max = max(D_gb - D_lat, 0.0)
        dx_min = min(params.get('dx', 1.0), params.get('dy', params.get('dx', 1.0)))
        dt_expl_max = 0.20 * dx_min**2 / max(delta_D_max + D_lat, 1e-12)
        if params.get('dt', 0) > dt_expl_max:
            warns.append(
                f"dt = {params.get('dt'):.2e} > CFL for explicit GB term "
                f"({dt_expl_max:.2e}); solver will auto-substep.")
        return errors, warns


# ============================================================================
# VORONOI GEOMETRY
# ============================================================================
def voronoi_assign(Nx, Ny, extent, seeds, mask=None, warp=None):
    x = np.linspace(extent[0], extent[1], Nx)
    y = np.linspace(extent[2], extent[3], Ny)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    if len(seeds) == 0:
        return np.full((Ny, Nx), -1, dtype=int)

    if warp is not None:
        wx, wy = warp
        pts_w = pts.copy()
        ix = np.clip(((pts[:, 0] - extent[0]) / (extent[1] - extent[0]) *
                      (wx.shape[1] - 1)).astype(int), 0, wx.shape[1] - 1)
        iy = np.clip(((pts[:, 1] - extent[2]) / (extent[3] - extent[2]) *
                      (wx.shape[0] - 1)).astype(int), 0, wx.shape[0] - 1)
        pts_w[:, 0] += wx[iy, ix]
        pts_w[:, 1] += wy[iy, ix]

        seed_ix = np.clip(
            ((seeds[:, 0] - extent[0]) / (extent[1] - extent[0]) *
             (wx.shape[1] - 1)).astype(int), 0, wx.shape[1] - 1)
        seed_iy = np.clip(
            ((seeds[:, 1] - extent[2]) / (extent[3] - extent[2]) *
             (wx.shape[0] - 1)).astype(int), 0, wx.shape[0] - 1)
        warped_seeds = seeds + np.column_stack([wx[seed_iy, seed_ix],
                                                 wy[seed_iy, seed_ix]])

        tree = cKDTree(warped_seeds)
        _, idx = tree.query(pts_w, k=1)
    else:
        tree = cKDTree(seeds)
        _, idx = tree.query(pts, k=1)

    gm = idx.reshape(Ny, Nx).astype(int)
    if mask is not None:
        gm = np.where(mask, gm, -1)
    return gm


def diffuse_field(binary, sigma_px):
    if sigma_px <= 0:
        return binary.astype(float)
    return gaussian_filter(binary.astype(float), sigma=sigma_px, mode='nearest')


def make_warp_field(Nx, Ny, extent, amplitude, wavelength_nm, seed=0):
    rng = np.random.default_rng(seed)
    Lx = extent[1] - extent[0]
    Ly = extent[3] - extent[2]
    nx_modes = max(2, int(Lx / wavelength_nm))
    ny_modes = max(2, int(Ly / wavelength_nm))
    x = np.linspace(0, 1, Nx); y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)
    wx = np.zeros_like(X); wy = np.zeros_like(Y)
    for i in range(1, nx_modes + 1):
        for j in range(1, ny_modes + 1):
            ph = rng.uniform(0, 2 * np.pi)
            amp = amplitude / (i * j)
            wx += amp * np.sin(2 * np.pi * i * X + ph) * np.cos(2 * np.pi * j * Y)
            wy += amp * np.cos(2 * np.pi * i * X) * np.sin(2 * np.pi * j * Y + ph)
    return wx, wy


class AuCuBondingGeometry:
    def __init__(self, Nx, Ny, dx, dy=None):
        self.Nx, self.Ny = Nx, Ny
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.extent = [-Nx * self.dx / 2, Nx * self.dx / 2,
                       -Ny * self.dy / 2, Ny * self.dy / 2]
        self.x = np.linspace(self.extent[0], self.extent[1], Nx)
        self.y = np.linspace(self.extent[2], self.extent[3], Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    @handle_errors
    def build(self,
              poly_thickness_frac=0.45,
              au_thickness_frac=0.04,
              n_poly_grains=2,
              poly_grain_size_nm=150.0,
              nc_grain_size_nm=15.0,
              gb_width_nm=1.5,
              poly_curve_amp=8.0,
              seed=1234,
              use_warp=True):
        rng = np.random.default_rng(seed)
        Nx, Ny = self.Nx, self.Ny
        y_top, y_bot = self.extent[3], self.extent[2]
        total_h = y_top - y_bot

        y_poly_bot = y_top - total_h * poly_thickness_frac
        y_au_bot   = y_poly_bot - total_h * au_thickness_frac

        y_poly_bot_eff = y_poly_bot
        y_au_bot_eff   = y_au_bot

        mask_poly = self.Y > y_poly_bot_eff
        mask_au   = (self.Y <= y_poly_bot_eff) & (self.Y > y_au_bot_eff)
        mask_nc   = self.Y <= y_au_bot_eff

        Lx = self.extent[1] - self.extent[0]
        area_nc = Lx * (y_au_bot - y_bot)
        n_nc = max(6, int(round(area_nc / (nc_grain_size_nm ** 2))))

        cy_poly = y_top - (y_top - y_poly_bot) / 2.0

        if n_poly_grains == 1:
            poly_seeds = np.array([[0.0, cy_poly]])
        elif n_poly_grains == 2:
            dy_offset = 0.25 * (y_top - y_poly_bot)
            poly_seeds = np.array([
                [-Lx / 4.0, cy_poly + dy_offset],
                [ Lx / 4.0, cy_poly - dy_offset],
            ])
        else:
            ys, xs = np.where(mask_poly)
            if len(ys) == 0:
                poly_seeds = np.zeros((0, 2))
            else:
                picks = rng.choice(len(ys),
                                    size=min(int(n_poly_grains), len(ys)),
                                    replace=False)
                poly_seeds = np.column_stack([self.X[ys[picks], xs[picks]],
                                               self.Y[ys[picks], xs[picks]]])

        ys_nc, xs_nc = np.where(mask_nc)
        if len(ys_nc) == 0:
            nc_seeds = np.zeros((0, 2))
        else:
            picks_nc = rng.choice(len(ys_nc),
                                   size=min(n_nc, len(ys_nc)),
                                   replace=False)
            nc_seeds = np.column_stack([self.X[ys_nc[picks_nc], xs_nc[picks_nc]],
                                         self.Y[ys_nc[picks_nc], xs_nc[picks_nc]]])

        if use_warp and poly_curve_amp > 1e-6:
            wx, wy = make_warp_field(Nx, Ny, self.extent,
                                      amplitude=poly_curve_amp,
                                      wavelength_nm=70.0, seed=seed)
            warp_poly = (wx, wy)
        else:
            warp_poly = None

        gm_poly = voronoi_assign(Nx, Ny, self.extent, poly_seeds, mask_poly,
                                  warp=warp_poly)
        gm_nc   = voronoi_assign(Nx, Ny, self.extent, nc_seeds,   mask_nc,
                                  warp=None)

        grain_map = np.full((Ny, Nx), -1, dtype=int)
        n_poly_actual = len(poly_seeds)
        n_nc_actual   = len(nc_seeds)
        au_idx = n_poly_actual
        if n_poly_actual > 0:
            grain_map[mask_poly] = gm_poly[mask_poly]
        if n_nc_actual > 0:
            grain_map[mask_nc] = gm_nc[mask_nc] + au_idx + 1
        grain_map[mask_au] = au_idx
        n_total = n_poly_actual + 1 + n_nc_actual

        sigma_poly = max(4.0 * gb_width_nm / self.dx, 1.5)
        sigma_au   = max(1.0 * gb_width_nm / self.dx, 0.8)
        sigma_nc   = max(0.8 * gb_width_nm / self.dx, 0.6)

        eta_list = []
        for g in range(n_total):
            m = (grain_map == g).astype(float)
            if g < n_poly_actual:
                sig = sigma_poly
            elif g == au_idx:
                sig = sigma_au
            else:
                sig = sigma_nc
            eta_list.append(np.clip(diffuse_field(m, sig), 0, 1))

        ssum = np.sum(eta_list, axis=0)
        ssum_safe = np.where(ssum > 1e-9, ssum, 1.0)
        eta_list = [e / ssum_safe for e in eta_list]

        c = np.clip(diffuse_field(mask_au.astype(float), sigma_au), 0, 1)

        layer_id = np.zeros((Ny, Nx), dtype=int)
        layer_id[mask_poly] = 1
        layer_id[mask_au]   = 2
        layer_id[mask_nc]   = 3

        meta = {
            'n_poly': n_poly_actual, 'n_nc': n_nc_actual,
            'au_idx': au_idx, 'n_total': n_total,
            'mask_poly': mask_poly, 'mask_au': mask_au, 'mask_nc': mask_nc,
            'layer_id': layer_id,
            'y_poly_bot': y_poly_bot, 'y_au_bot': y_au_bot,
            'y_poly_bot_eff': y_poly_bot_eff, 'y_au_bot_eff': y_au_bot_eff,
            'seeds_poly': poly_seeds, 'seeds_nc': nc_seeds,
        }
        return c, eta_list, grain_map, layer_id, meta


# ============================================================================
# SPECTRAL ELASTICITY  (BUG FIX: correct plane-strain moduli)
# ============================================================================
class SpectralElasticSolver:
    """
    Solve ∇·σ = 0 for σ = C:(ε_total − ε*) via plane-strain Fourier Green's fn.

    BUG FIX: previously used averaged moduli
        C11_2d = (C11 + C12 + 2 C44)/2,  etc.
    which is a plane-stress-like blend. For true 2D plane strain on {100}:
        σ_xx = C11 ε_xx + C12 ε_yy
        σ_yy = C12 ε_xx + C11 ε_yy
        σ_xy = 2 C44 ε_xy
    so we now use (C11, C12, C44) directly.
    """

    def __init__(self, Nx, Ny, dx, dy, elastic_params, kx=None, ky=None, k2=None):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        if kx is None:
            self.kx, self.ky, self.k2 = make_k_vectors(Nx, Ny, dx, dy)
        else:
            self.kx, self.ky, self.k2 = kx, ky, k2

        C11, C12, C44 = (elastic_params['C11'], elastic_params['C12'],
                         elastic_params['C44'])
        # Correct plane-strain moduli
        self.C11_2d, self.C12_2d, self.C44_2d = C11, C12, C44
        lam = C12
        mu  = C44
        self.lam, self.mu = lam, mu

        k2 = self.k2
        inv_mu_k2 = 1.0 / (mu * k2 + 1e-30)
        ratio = (lam + mu) / (lam + 2 * mu)
        self.G11 = inv_mu_k2 * (1.0 - ratio * (self.kx**2) / k2)
        self.G12 = inv_mu_k2 * (-ratio * self.kx * self.ky / k2)
        self.G22 = inv_mu_k2 * (1.0 - ratio * (self.ky**2) / k2)
        self.G11[0, 0] = self.G12[0, 0] = self.G22[0, 0] = 0.0

    @handle_errors
    def solve(self, exx_star, eyy_star, exy_star,
              app_xx=0.0, app_yy=0.0, app_xy=0.0):
        exx_h = fft2(exx_star)
        eyy_h = fft2(eyy_star)
        exy_h = fft2(exy_star)

        eps_kk_h = exx_h + eyy_h
        fx_h = self.lam * self.kx * eps_kk_h + 2 * self.mu * (
            self.kx * exx_h + self.ky * exy_h)
        fy_h = self.lam * self.ky * eps_kk_h + 2 * self.mu * (
            self.kx * exy_h + self.ky * eyy_h)

        ux_h = self.G11 * fx_h + self.G12 * fy_h
        uy_h = self.G12 * fx_h + self.G22 * fy_h

        exx_el = np.real(ifft2(1j * self.kx * ux_h))
        eyy_el = np.real(ifft2(1j * self.ky * uy_h))
        exy_el = 0.5 * np.real(ifft2(1j * (self.kx * uy_h + self.ky * ux_h)))

        exx = exx_el + exx_star
        eyy = eyy_el + eyy_star
        exy = exy_el + exy_star

        sxx = self.C11_2d * exx_el + self.C12_2d * eyy_el + app_xx
        syy = self.C12_2d * exx_el + self.C11_2d * eyy_el + app_yy
        sxy = 2 * self.C44_2d * exy_el + app_xy

        sigma_eq = np.sqrt(0.5 * ((sxx - syy)**2 + syy**2 + sxx**2 + 6 * sxy**2))
        sigma_eq = np.clip(sigma_eq, 0, 5e9)
        sigma_h = 0.5 * (sxx + syy)

        return sigma_eq, sxx, syy, sxy, sigma_h, exx, eyy, exy


# ============================================================================
# MAIN MPF SOLVER
# ============================================================================
class AuCuBondingSolver:
    def __init__(self, params):
        self.params = params
        self.Nx = params['Nx']
        self.Ny = params['Ny']
        self.dx = params['dx']
        self.dy = params.get('dy', params['dx'])
        self.dt = params['dt']
        self.mat = MaterialProperties.get_material(params.get('material', 'Cu'))

        self.kx, self.ky, self.k2 = make_k_vectors(self.Nx, self.Ny,
                                                   self.dx, self.dy)

        self.kappa_eta      = params.get('kappa_eta', 1.0)
        self.aniso_strength = params.get('aniso_strength', 0.3)
        self.L_ref          = params.get('L_ref', 1.0)

        self.geom = AuCuBondingGeometry(self.Nx, self.Ny, self.dx, self.dy)
        result = self.geom.build(
            poly_thickness_frac=params.get('poly_thickness_frac', 0.45),
            au_thickness_frac=params.get('au_thickness_frac', 0.04),
            n_poly_grains=params.get('n_poly_grains', 2),
            poly_grain_size_nm=params.get('poly_grain_size_nm', 150.0),
            nc_grain_size_nm=params.get('nc_grain_size_nm', 15.0),
            gb_width_nm=params.get('gb_width_nm', 1.5),
            poly_curve_amp=params.get('poly_curve_amp', 8.0),
            seed=params.get('seed', 1234),
            use_warp=params.get('use_warp', True))
        if result is FAILED or result is None:
            raise RuntimeError(f"Geometry build failed: {FAILED.reason}")
        (self.c, self.eta_list, self.grain_map,
         self.layer_id, self.geom_meta) = result

        self.eta_list = self._relax_grains(
            self.eta_list, n_steps=params.get('relax_steps', 10))

        self.n_grains = len(self.eta_list)
        self.au_idx   = self.geom_meta['au_idx']
        self.n_poly   = self.geom_meta['n_poly']

        rng = np.random.default_rng(params.get('seed', 1234))
        self.grain_theta = rng.uniform(0, 2 * np.pi, size=self.n_grains)

        self.eps_p_xx = np.zeros((self.Ny, self.Nx))
        self.eps_p_yy = np.zeros((self.Ny, self.Nx))
        self.eps_p_xy = np.zeros((self.Ny, self.Nx))

        self.elastic = SpectralElasticSolver(
            self.Nx, self.Ny, self.dx, self.dy, self.mat['elastic'],
            self.kx, self.ky, self.k2)

        self.use_cahn_hilliard = bool(params.get('use_cahn_hilliard', False))
        self.W        = float(params.get('W', 2.0))
        self.kappa_c  = float(params.get('kappa_c', 0.5))
        self.M_ref    = float(params.get('M_ref', 1.0))

        self.history = {
            'time': [], 'c_top': [], 'c_bot': [],
            'gb_top': [], 'gb_bot': [],
            'max_stress': [], 'avg_stress': [],
            'plastic_work': [], 'energy': [],
            'au_mass': [],
        }

    # ------------------------------------------------------------------
    def _relax_grains(self, etas, n_steps=10):
        if n_steps <= 0:
            return etas
        etas = [e.copy() for e in etas]
        dt_relax = 0.15 * min(self.dx**2, self.dy**2) / max(self.kappa_eta, 1e-9)
        for _ in range(n_steps):
            new = []
            for e in etas:
                lap = spectral_laplacian(e, self.k2)
                e_new = e + dt_relax * self.kappa_eta * lap
                new.append(np.clip(e_new, 0, 1))
            s = np.sum(new, axis=0)
            s = np.where(s > 1.0, s, 1.0)
            etas = [e / s for e in new]
        return etas

    # ------------------------------------------------------------------
    def compute_gb_indicator(self):
        s = np.zeros((self.Ny, self.Nx))
        for e in self.eta_list:
            s += e**2
        return np.clip(1.0 - s, 0, 1)

    # ------------------------------------------------------------------
    def compute_local_grain_size(self):
        """
        BUG FIX: previous version returned 2/|∇(Ση)| which is an inverse
        GB-thickness, not a grain size. We now use the distance to the
        nearest GB pixel (via distance transform) scaled to represent the
        local grain diameter.

        h(x) ≈ 2 × dist_to_nearest_GB(x)   [nm]
        """
        phi_gb = self.compute_gb_indicator()
        gb_mask = phi_gb > 0.5
        if not gb_mask.any():
            return np.full((self.Ny, self.Nx), 100.0)
        dist = distance_transform_edt(~gb_mask) * min(self.dx, self.dy)
        h = 2.0 * dist
        return np.clip(h, 2.0, 300.0)

    # ==================================================================
    # DIFFUSION
    # ==================================================================
    def evolve_concentration(self, sigma_h):
        phi_gb = self.compute_gb_indicator()

        D_lat = float(self.params.get('D_lattice', 0.01))
        D_gb  = float(self.params.get('D_gb', 50.0))
        Omega = float(self.params.get('Omega', 0.05))

        D_eff = D_lat + (D_gb - D_lat) * phi_gb

        if self.use_cahn_hilliard:
            self.c = self._cahn_hilliard_step(sigma_h, D_eff, Omega)
            return np.clip(self.c, 0.0, 1.0), D_eff

        delta_D_max = max(D_gb - D_lat, 0.0)
        delta_D     = delta_D_max * phi_gb

        dx_min      = min(self.dx, self.dy)
        dt_expl_max = 0.20 * dx_min**2 / max(delta_D_max + D_lat, 1e-12)
        n_sub       = max(1, int(np.ceil(self.dt / dt_expl_max)))
        dt_sub      = self.dt / n_sub

        for _ in range(n_sub):
            self.c = self._concentration_substep(
                dt_sub, phi_gb, D_lat, delta_D, Omega, sigma_h)

        return np.clip(self.c, 0.0, 1.0), D_eff

    # ------------------------------------------------------------------
    def _cahn_hilliard_step(self, sigma_h, D_eff, Omega):
        """
        Cahn-Hilliard step with:
          * nonlinear CFL auto-substepping (BUG FIX P0)
          * actual rejection of blown-up steps (rollback to previous c)
        """
        # --- CFL for the explicit nonlinear part -------------------------
        dx_min = min(self.dx, self.dy)
        # dt_cfl ~ dx^2 / (4 M W)   (linearized CH chemical potential)
        M_max = float(np.max(D_eff))
        W_eff = max(abs(self.W), 1e-9)
        dt_cfl = 0.20 * dx_min**2 / max(M_max * W_eff, 1e-12)
        n_sub = max(1, int(np.ceil(self.dt / dt_cfl)))
        dt_sub = self.dt / n_sub

        c_start = self.c.copy()
        c = self.c.copy()

        for _ in range(n_sub):
            sigma_h_norm = sigma_h / 1e9

            df_dc = self.W * (4.0 * c**3 - 6.0 * c**2 + 2.0 * c)
            lap_c = spectral_laplacian(c, self.k2)
            mu = df_dc - self.kappa_c * lap_c + Omega * sigma_h_norm

            mu_gx, mu_gy = spectral_gradients(mu, self.kx, self.ky)
            Jx = -D_eff * mu_gx
            Jy = -D_eff * mu_gy
            div_J = spectral_divergence(Jx, Jy, self.kx, self.ky)

            c_expl = c + dt_sub * div_J

            denom = 1.0 + dt_sub * self.M_ref * self.kappa_c * self.k2**2
            c_new = np.real(ifft2(fft2(c_expl) / denom))

            # Bounds
            c_clip = np.clip(c_new, 0.0, 1.0)

            # --- BUG FIX: actually reject blown-up steps -----------------
            m_old = float(np.sum(c))
            m_new = float(np.sum(c_clip))
            if m_old < 1e-12:
                c = c_clip
                continue
            ratio = m_new / m_old
            # Reject if the nonlinear explicit part produced a wildly
            # different mass or a huge local increment.
            max_inc = float(np.max(np.abs(c_clip - c)))
            if ratio < 0.5 or ratio > 2.0 or max_inc > 0.5:
                # Roll back: keep the previous state, don't apply this step.
                logger.warning(
                    f"CH substep rejected: mass ratio {ratio:.3e}, "
                    f"max increment {max_inc:.3e}")
                # Try a half step once
                c_attempt = self._ch_half_step(c, sigma_h, D_eff, Omega,
                                                dt_sub * 0.5)
                if c_attempt is not None:
                    c = c_attempt
                # else: keep c as-is (skip this substep)
            else:
                # Small correction
                c_clip *= m_old / m_new
                c = np.clip(c_clip, 0.0, 1.0)

        # Final safety: if the whole step went nowhere useful, keep c_start
        if not np.any(np.isfinite(c)):
            return np.clip(c_start, 0.0, 1.0)
        return np.clip(c, 0.0, 1.0)

    def _ch_half_step(self, c0, sigma_h, D_eff, Omega, dt_half):
        """Attempt a single half-size CH substep; return None on failure."""
        try:
            sigma_h_norm = sigma_h / 1e9
            df_dc = self.W * (4.0 * c0**3 - 6.0 * c0**2 + 2.0 * c0)
            lap_c = spectral_laplacian(c0, self.k2)
            mu = df_dc - self.kappa_c * lap_c + Omega * sigma_h_norm
            mu_gx, mu_gy = spectral_gradients(mu, self.kx, self.ky)
            div_J = spectral_divergence(-D_eff * mu_gx, -D_eff * mu_gy,
                                         self.kx, self.ky)
            c_expl = c0 + dt_half * div_J
            denom = 1.0 + dt_half * self.M_ref * self.kappa_c * self.k2**2
            c_new = np.real(ifft2(fft2(c_expl) / denom))
            c_new = np.clip(c_new, 0.0, 1.0)
            m_old = float(np.sum(c0)); m_new = float(np.sum(c_new))
            if m_old < 1e-12 or m_new < 1e-12:
                return None
            ratio = m_new / m_old
            if 0.5 < ratio < 2.0:
                c_new *= m_old / m_new
                return np.clip(c_new, 0.0, 1.0)
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    def _concentration_substep(self, dt, phi_gb, D_lat, delta_D, Omega, sigma_h):
        """
        Fickian path. Uses normalized σ_h and *actual* rollback on
        blow-up (BUG FIX P0).
        """
        sigma_h_norm = sigma_h / 1e9

        c_gx, c_gy = spectral_gradients(self.c, self.kx, self.ky)
        Jx_d = -delta_D * c_gx
        Jy_d = -delta_D * c_gy
        div_J_var = spectral_divergence(Jx_d, Jy_d, self.kx, self.ky)

        sig_gx, sig_gy = spectral_gradients(sigma_h_norm, self.kx, self.ky)
        D_eff = D_lat + delta_D
        Jx_s = -D_eff * Omega * self.c * sig_gx
        Jy_s = -D_eff * Omega * self.c * sig_gy
        div_J_s = spectral_divergence(Jx_s, Jy_s, self.kx, self.ky)

        c_expl = self.c + dt * (div_J_var + div_J_s)

        denom = 1.0 + dt * D_lat * self.k2
        c_new = np.real(ifft2(fft2(c_expl) / denom))
        c_new = np.clip(c_new, 0.0, 1.0)

        m_old = float(np.sum(self.c))
        m_new = float(np.sum(c_new))

        if m_new < 1e-12:
            c_safe = np.real(ifft2(fft2(self.c) / denom))
            return np.clip(c_safe, 0.0, 1.0)

        if m_old > 1e-12:
            ratio = m_new / m_old
            max_inc = float(np.max(np.abs(c_new - self.c)))
            # BUG FIX: actually reject, don't silently accept
            if ratio < 0.1 or ratio > 10.0 or max_inc > 0.5:
                logger.warning(
                    f"Fickian substep rejected: mass ratio {ratio:.3e}, "
                    f"max increment {max_inc:.3e}")
                # Fall back to pure lattice diffusion for this substep
                c_safe = np.real(ifft2(fft2(self.c) / denom))
                return np.clip(c_safe, 0.0, 1.0)
            c_new *= m_old / m_new
        return np.clip(c_new, 0.0, 1.0)

    # ==================================================================
    def evolve_grains(self):
        A = float(self.params.get('A', 5.0))
        B = float(self.params.get('B', 10.0))
        L0 = float(self.params.get('L_eta', 1.0))
        alpha_drag = float(self.params.get('alpha_drag', 50.0))
        kappa0 = self.kappa_eta
        delta_aniso = self.aniso_strength

        L_eff = L0 / (1.0 + alpha_drag * self.c**2)

        sq = [e**2 for e in self.eta_list]
        total_sq = np.sum(sq, axis=0)

        # Use a spatially-averaged L_eff for the implicit stabilizer
        # (BUG FIX: previously used L_ref only, decoupled from L_eff)
        L_stab = float(np.mean(L_eff))

        new_list = []
        for i, e in enumerate(self.eta_list):
            others_sq = total_sq - sq[i]
            df_de_loc = 2 * A * e * (1 - e) * (1 - 2 * e) + 2 * B * e * others_sq

            gx, gy = spectral_gradients(e, self.kx, self.ky)
            mag = np.sqrt(gx**2 + gy**2 + 1e-12)
            nx = gx / mag
            ny = gy / mag
            theta_n = np.arctan2(ny, nx)
            theta_i = self.grain_theta[i]
            kappa_aniso = kappa0 * (1.0 + delta_aniso *
                                    np.cos(2.0 * (theta_n - theta_i)))

            kx_gx, ky_gx = spectral_gradients(kappa_aniso * gx, self.kx, self.ky)
            kx_gy, ky_gy = spectral_gradients(kappa_aniso * gy, self.kx, self.ky)
            grad_term = kx_gx + ky_gy

            R_expl = -L_eff * (df_de_loc - grad_term)

            e_expl = e + self.dt * R_expl
            denom = 1.0 + self.dt * L_stab * kappa0 * self.k2
            e_new = np.real(ifft2(fft2(e_expl) / denom))
            new_list.append(np.clip(e_new, 0, 1))

        ssum = np.sum(new_list, axis=0)
        mask = ssum > 1.0
        if np.any(mask):
            ssum_safe = np.where(mask, ssum, 1.0)
            new_list = [np.where(mask, e / ssum_safe, e) for e in new_list]
        return new_list

    def compute_eigenstrain(self):
        beta = float(self.params.get('misfit_beta', 0.045))
        return beta * self.c, beta * self.c, np.zeros_like(self.c)

    # ------------------------------------------------------------------
    def compute_plasticity(self, sigma_eq, sxx, syy, sxy, h):
        """
        J2 return mapping (BUG FIX: flow direction driven by local
        deviatoric stress, not a fixed xx+/yy- increment).
        """
        pl = self.mat['plasticity']
        mu_p, nu, b = pl['mu'], pl['nu'], pl['b']
        sigma0 = float(self.params.get('sigma0', pl['sigma0']))
        gamma0_dot = pl['gamma0_dot']
        m = int(pl['m'])

        h_safe = np.maximum(h, 2.001 * b)
        sigma_y = sigma0 + (mu_p * b / (2 * np.pi * h_safe * (1 - nu))) * \
                  np.log(h_safe / b)
        sigma_y = np.where(h > 2 * b, sigma_y,
                           sigma0 + mu_p / (2 * np.pi * (1 - nu)))

        overstress = np.clip((sigma_eq - sigma_y) / np.maximum(sigma_y, 1e-9), 0, 1)
        gamma_dot = gamma0_dot * overstress ** m
        d_gamma = np.minimum(gamma_dot * self.dt, 1e-3)

        # Plane-strain mean: p = (σ_xx + σ_yy) / 2
        p = 0.5 * (sxx + syy)
        s_dev_xx = sxx - p
        s_dev_yy = syy - p
        s_dev_xy = sxy

        # Flow direction n = (3/2) s / σ_eq  (with σ_eq the von Mises)
        denom_eq = np.maximum(sigma_eq, 1e-9)
        n_xx = 1.5 * s_dev_xx / denom_eq
        n_yy = 1.5 * s_dev_yy / denom_eq
        n_xy = 1.5 * s_dev_xy / denom_eq

        # Plastic increment
        d_xx = d_gamma * n_xx
        d_yy = d_gamma * n_yy
        d_xy = d_gamma * n_xy

        self.eps_p_xx = np.clip(self.eps_p_xx + d_xx, -0.1, 0.1)
        self.eps_p_yy = np.clip(self.eps_p_yy + d_yy, -0.1, 0.1)
        self.eps_p_xy = np.clip(self.eps_p_xy + d_xy, -0.1, 0.1)

        eps_p_mag = np.sqrt(2.0 / 3.0 * (self.eps_p_xx**2 + self.eps_p_yy**2
                                          + 2 * self.eps_p_xy**2 + 1e-15))
        return sigma_y, eps_p_mag

    def compute_kam_density(self, exy_total, gb_indicator):
        exy_gx, exy_gy = spectral_gradients(exy_total, self.kx, self.ky)
        geom_factor = np.sqrt(exy_gx**2 + exy_gy**2)
        rho = geom_factor * (0.5 + 0.5 * gb_indicator)
        rho_max = np.max(rho) if np.max(rho) > 1e-15 else 1.0
        return np.clip(rho / rho_max, 0, 1)

    @handle_errors
    def step(self):
        sig = float(self.params.get('applied_stress', 30e6))
        theta = np.deg2rad(float(self.params.get('applied_stress_angle', 90.0)))
        app_xx = sig * np.cos(theta)**2
        app_yy = sig * np.sin(theta)**2
        app_xy = sig * np.sin(theta) * np.cos(theta)

        exx_s, eyy_s, exy_s = self.compute_eigenstrain()
        exx_t = exx_s + self.eps_p_xx
        eyy_t = eyy_s + self.eps_p_yy
        exy_t = exy_s + self.eps_p_xy

        el = self.elastic.solve(exx_t, eyy_t, exy_t, app_xx, app_yy, app_xy)
        if el is FAILED or el is None:
            raise RuntimeError("Elastic solve failed")
        (sigma_eq, sxx, syy, sxy, sigma_h,
         exx_tot, eyy_tot, exy_tot) = el

        h = self.compute_local_grain_size()
        sigma_y, eps_p_mag = self.compute_plasticity(sigma_eq, sxx, syy, sxy, h)
        phi_gb = self.compute_gb_indicator()
        rho_gnd = self.compute_kam_density(exy_tot, phi_gb)

        c_new, D_eff = self.evolve_concentration(sigma_h)
        self.c = c_new

        self.eta_list = self.evolve_grains()

        mtop = self.geom_meta['mask_poly']
        mbot = self.geom_meta['mask_nc']
        c_top = float(np.mean(self.c[mtop])) if mtop.any() else 0.0
        c_bot = float(np.mean(self.c[mbot])) if mbot.any() else 0.0
        phi_gb_new = self.compute_gb_indicator()
        gb_top = float(np.mean(phi_gb_new[mtop])) if mtop.any() else 0.0
        gb_bot = float(np.mean(phi_gb_new[mbot])) if mbot.any() else 0.0

        au_mass = float(np.sum(self.c) * self.dx * self.dy)

        grad_energy_sum = 0.0
        for e in self.eta_list:
            gx, gy = spectral_gradients(e, self.kx, self.ky)
            grad_energy_sum += float(np.sum(gx**2 + gy**2))
        energy = 0.5 * self.kappa_eta * grad_energy_sum * self.dx * self.dy

        t = (len(self.history['time']) + 1) * self.dt

        self.history['time'].append(t)
        self.history['c_top'].append(c_top)
        self.history['c_bot'].append(c_bot)
        self.history['gb_top'].append(gb_top)
        self.history['gb_bot'].append(gb_bot)
        self.history['max_stress'].append(float(np.max(sigma_eq)))
        self.history['avg_stress'].append(float(np.mean(sigma_eq)))
        self.history['plastic_work'].append(float(np.sum(eps_p_mag) * self.dx * self.dy))
        self.history['energy'].append(energy)
        self.history['au_mass'].append(au_mass)

        eta_poly = (np.sum([self.eta_list[i] for i in range(self.n_poly)], axis=0)
                    if self.n_poly else np.zeros_like(self.c))
        eta_au = self.eta_list[self.au_idx]
        eta_nc = (np.sum([self.eta_list[i]
                          for i in range(self.n_poly + 1, self.n_grains)], axis=0)
                  if self.n_grains > self.n_poly + 1 else np.zeros_like(self.c))

        eta_stack = np.stack(self.eta_list, axis=0)
        grain_map_soft = np.argmax(eta_stack, axis=0).astype(float)
        grain_map_soft[eta_stack.max(axis=0) < 0.5] = -1.0

        return {
            'phi':        self.c.copy(),
            'eta_poly':   eta_poly,
            'eta_au':     eta_au,
            'eta_nc':     eta_nc,
            'grain_map':  grain_map_soft,
            'phi_gb':     phi_gb_new,
            'D_eff':      D_eff.copy(),
            'sigma_eq':   sigma_eq.copy(),
            'sigma_h':    sigma_h.copy(),
            'h':          h.copy(),
            'eps_p_mag':  eps_p_mag.copy(),
            'sigma_y':    sigma_y.copy(),
            'rho_gnd':    rho_gnd,
            'sxx':        sxx.copy(),
            'syy':        syy.copy(),
            'sxy':        sxy.copy(),
            'convergence': {
                'c_top': c_top, 'c_bot': c_bot,
                'au_mass': au_mass,
                'max_stress': float(np.max(sigma_eq)),
                'avg_stress': float(np.mean(sigma_eq)),
                'energy': energy,
            }
        }


# ============================================================================
# LINE PROFILER  (now with multi-time profiles)
# ============================================================================
class EnhancedLineProfiler:
    def __init__(self, Nx, Ny, dx, dy):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        self.extent = [-Nx * dx / 2, Nx * dx / 2, -Ny * dy / 2, Ny * dy / 2]

    @handle_errors
    def extract_profile(self, data, profile_type='horizontal',
                        position_ratio=0.5, angle_deg=45):
        ptype = str(profile_type).lower().replace('-', '_')
        ny, nx = data.shape
        cx, cy = nx // 2, ny // 2
        off_x = int(nx * 0.35 * position_ratio)
        off_y = int(ny * 0.35 * position_ratio)

        if ptype == 'horizontal':
            r = min(max(cy + off_y, 0), ny - 1)
            return (np.linspace(self.extent[0], self.extent[1], nx),
                    data[r, :], (self.extent[0], r * self.dy + self.extent[2],
                                 self.extent[1], r * self.dy + self.extent[2]))
        if ptype == 'vertical':
            cidx = min(max(cx + off_x, 0), nx - 1)
            return (np.linspace(self.extent[2], self.extent[3], ny),
                    data[:, cidx], (cidx * self.dx + self.extent[0], self.extent[2],
                                     cidx * self.dx + self.extent[0], self.extent[3]))
        if ptype == 'diagonal':
            L = int(min(nx, ny) * 0.8)
            s = (cx - L // 2, cy - L // 2)
            vals, ds = [], []
            for i in range(L):
                x, y = s[0] + i, s[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    vals.append(data[y, x])
                    ds.append((i - L // 2) * self.dx * np.sqrt(2))
            return np.array(ds), np.array(vals), None
        if ptype in ('anti_diagonal', 'antidiagonal'):
            L = int(min(nx, ny) * 0.8)
            s = (cx + L // 2, cy - L // 2)
            vals, ds = [], []
            for i in range(L):
                x, y = s[0] - i, s[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    vals.append(data[y, x])
                    ds.append((i - L // 2) * self.dx * np.sqrt(2))
            return np.array(ds), np.array(vals), None
        if ptype == 'custom':
            ar = np.deg2rad(angle_deg)
            L = int(min(nx, ny) * 0.8)
            vals, ds = [], []
            for t in np.linspace(-L // 2, L // 2, L):
                x = cx + t * np.cos(ar) + off_y * np.cos(ar + np.pi / 2)
                y = cy + t * np.sin(ar) + off_y * np.sin(ar + np.pi / 2)
                if 0 <= x < nx - 1 and 0 <= y < ny - 1:
                    x0, y0 = int(x), int(y)
                    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
                    wx, wy = x - x0, y - y0
                    v = (data[y0, x0] * (1 - wx) * (1 - wy)
                         + data[y0, x1] * wx * (1 - wy)
                         + data[y1, x0] * (1 - wx) * wy
                         + data[y1, x1] * wx * wy)
                    vals.append(v); ds.append(t * self.dx)
            return np.array(ds), np.array(vals), None
        return None, None, None

    @handle_errors
    def extract_profile_at(self, data, x0, y0, x1, y1, n=200):
        """Extract a profile along an arbitrary segment in physical coords."""
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        # Convert to pixel coords
        i = ((xs - self.extent[0]) / self.dx).astype(float)
        j = ((ys - self.extent[2]) / self.dy).astype(float)
        i = np.clip(i, 0, self.Nx - 1)
        j = np.clip(j, 0, self.Ny - 1)
        i0 = i.astype(int); j0 = j.astype(int)
        i1 = np.clip(i0 + 1, 0, self.Nx - 1)
        j1 = np.clip(j0 + 1, 0, self.Ny - 1)
        wi = i - i0
        wj = j - j0
        v = (data[j0, i0] * (1 - wi) * (1 - wj)
             + data[j0, i1] * wi * (1 - wj)
             + data[j1, i0] * (1 - wi) * wj
             + data[j1, i1] * wi * wj)
        dist = np.sqrt((xs - x0)**2 + (ys - y0)**2)
        return dist, v


# ============================================================================
# FIELD RESCALING HELPER
# ============================================================================
def _rescale_field(field_name, data):
    d = data.copy()
    if field_name in ('sigma_eq', 'sigma_h', 'sxx', 'syy', 'sxy'):
        d = d / 1e9
    elif field_name == 'sigma_y':
        d = d / 1e6
    return d, FIELD_UNITS.get(field_name, '')


def compute_field_norm(field_name, data, fs):
    """
    Given per-field visualization settings, return (norm, vmin, vmax).
    norm is a matplotlib Normalize/LogNorm or None for auto.
    """
    if fs is None:
        fs = {}
    auto = fs.get('auto', True)
    use_log = fs.get('log', False)

    d = np.asarray(data)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return None, None, None

    if auto:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
        if field_name == 'sigma_h':
            vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
        if vmax - vmin < 1e-12:
            vmax = vmin + 1e-9
    else:
        vmin = fs.get('vmin', None)
        vmax = fs.get('vmax', None)
        if vmin is None: vmin = float(np.min(finite))
        if vmax is None: vmax = float(np.max(finite))
        if vmax - vmin < 1e-12:
            vmax = vmin + 1e-9

    if use_log:
        pos = finite[finite > 0]
        if pos.size == 0:
            return None, vmin, vmax
        if vmin <= 0: vmin = float(np.percentile(pos, 2))
        if vmax <= 0: vmax = float(np.percentile(pos, 98))
        if vmin >= vmax: vmin = vmax / 10.0
        try:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        except Exception:
            norm = None
    else:
        norm = None

    return norm, vmin, vmax


# ============================================================================
# VECTOR FIELD COMPUTATION
# ============================================================================
def compute_vector_field(results_dict, source, mode='gradient'):
    """
    Compute a 2-component vector field from the results.
    Returns (vx, vy) in the grid's native shape.

    mode = 'gradient'         → (∂f/∂x, ∂f/∂y)
    mode = 'flux'             → -D_eff ∇f  (only meaningful for phi)
    mode = 'principal_stress' → (cos θ1, sin θ1) of major principal stress
    """
    Nx = results_dict.get('phi').shape[1]
    Ny = results_dict.get('phi').shape[0]
    dx = 1.0
    # Use finite differences (we don't have k vectors here)
    if mode == 'principal_stress':
        sxx = results_dict.get('sxx')
        syy = results_dict.get('syy')
        sxy = results_dict.get('sxy')
        if sxx is None or syy is None or sxy is None:
            return None, None
        # Principal stress angle: θ = 0.5 atan2(2 sxy, sxx - syy)
        theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
        vx = np.cos(theta)
        vy = np.sin(theta)
        return vx, vy

    f = results_dict.get(source)
    if f is None:
        return None, None

    gy, gx = np.gradient(f)
    if mode == 'flux':
        # J = -D ∇c
        D = results_dict.get('D_eff')
        if D is None:
            D = np.ones_like(f)
        vx = -D * gx
        vy = -D * gy
    else:
        vx, vy = gx, gy
    return vx, vy


# ============================================================================
# VISUALIZER  (major upgrade)
# ============================================================================
class EnhancedTwinVisualizer:
    def __init__(self, Nx, Ny, dx, dy=None, dt=1e-4):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, (dy if dy is not None else dx)
        self.dt = dt
        self.extent = [-Nx * self.dx / 2, Nx * self.dx / 2,
                       -Ny * self.dy / 2, Ny * self.dy / 2]
        self.line_profiler = EnhancedLineProfiler(Nx, Ny, dx, self.dy)

    def get_colormap(self, name, invert=False):
        return ColormapRegistry.resolve(name, invert=invert)

    # ------------------------------------------------------------------
    # MULTI-FIELD COMPARISON
    # ------------------------------------------------------------------
    @handle_errors
    def create_multi_field_comparison(self, results_dict, style_params=None):
        if style_params is None:
            style_params = {}
        colormaps = style_params.get('colormaps', {})
        field_settings = style_params.get('field_settings', {})
        title_fs = float(style_params.get('title_font_size', 10))
        label_fs = float(style_params.get('label_font_size', 8))
        cbar_fraction = float(style_params.get('colorbar_fraction', 0.046))
        cbar_pad = float(style_params.get('colorbar_pad', 0.04))

        fields = [
            ('phi',       'c (Au concentration)',      [0, 1]),
            ('grain_map', 'Individual grains (Voronoi)', None),
            ('eta_poly',  'η_poly (Poly-Cu)',           [0, 1]),
            ('eta_au',    'η_Au (Au film)',              [0, 1]),
            ('eta_nc',    'η_nc (nc-Cu)',                [0, 1]),
            ('phi_gb',    'φ_gb (GB indicator)',         [0, 1]),
            ('D_eff',     'D_eff (Hart effective)',      None),
            ('sigma_eq',  'σ_eq (GPa)',                  None),
            ('sigma_h',   'σ_h (GPa)',                   None),
            ('h',         'Local grain size (nm)',       None),
            ('eps_p_mag', 'Plastic strain |ε_p|',        None),
            ('sigma_y',   'Yield stress (MPa)',          None),
            ('rho_gnd',   'ρ_GND (Dislocation density)', [0, 1]),
        ]
        avail = [(f, t, r) for (f, t, r) in fields if f in results_dict]
        if not avail:
            return None

        n = len(avail)
        cols = 3 if n >= 3 else n
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.8 * rows))
        axes = np.array(axes).flatten()

        for idx, (fname, title, rng) in enumerate(avail):
            ax = axes[idx]
            d, unit = _rescale_field(fname, results_dict[fname])

            fs = field_settings.get(fname, {})
            cmap_name = (colormaps.get(fname)
                         or style_params.get(f'{fname}_cmap')
                         or FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
            cmap = self.get_colormap(cmap_name, invert=fs.get('invert', False))

            interp = 'nearest' if fname in DISCRETE_FIELDS else 'bilinear'

            if fname == 'grain_map':
                im = ax.imshow(d, extent=self.extent, cmap=cmap,
                               origin='lower', aspect='equal',
                               interpolation='nearest')
                ax.set_title(title, fontsize=title_fs)
                ax.set_xlabel('x (nm)', fontsize=label_fs)
                ax.set_ylabel('y (nm)', fontsize=label_fs)
                continue

            if rng is not None and fs.get('auto', True) and not fs.get('log', False):
                vmin, vmax = rng
                norm = None
            else:
                norm, vmin, vmax = compute_field_norm(fname, d, fs)

            if norm is not None:
                im = ax.imshow(d, extent=self.extent, cmap=cmap, norm=norm,
                               origin='lower', aspect='equal', interpolation=interp)
            else:
                im = ax.imshow(d, extent=self.extent, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               origin='lower', aspect='equal', interpolation=interp)
            ax.set_title(title, fontsize=title_fs)
            ax.set_xlabel('x (nm)', fontsize=label_fs)
            ax.set_ylabel('y (nm)', fontsize=label_fs)
            cbar = plt.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
            if unit:
                cbar.set_label(unit)

        for k in range(n, len(axes)):
            axes[k].axis('off')
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # OVERLAY  (new)
    # ------------------------------------------------------------------
    @handle_errors
    def create_overlay(self, results_dict, base_field, overlay_field,
                       alpha=0.5, mode='contour', n_contours=10,
                       show_grain_ghost=False, grain_map=None,
                       style_params=None):
        """
        Composite visualization: base field as heatmap, overlay field as
        contour lines (default) or as an alpha-blended heatmap.
        """
        if style_params is None:
            style_params = {}
        field_settings = style_params.get('field_settings', {})
        colormaps = style_params.get('colormaps', {})
        title_fs = float(style_params.get('title_font_size', 11))
        label_fs = float(style_params.get('label_font_size', 9))

        if base_field not in results_dict or overlay_field not in results_dict:
            return None

        base, base_unit = _rescale_field(base_field, results_dict[base_field])
        ov, ov_unit = _rescale_field(overlay_field, results_dict[overlay_field])

        base_fs = field_settings.get(base_field, {})
        ov_fs = field_settings.get(overlay_field, {})

        base_cmap = self.get_colormap(
            colormaps.get(base_field, FIELD_DEFAULT_CMAPS.get(base_field, 'viridis')),
            invert=base_fs.get('invert', False))

        fig, ax = plt.subplots(figsize=(11, 5.5))

        # Base layer
        if base_field in DISCRETE_FIELDS:
            im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                origin='lower', aspect='equal',
                                interpolation='nearest')
        else:
            norm, vmin, vmax = compute_field_norm(base_field, base, base_fs)
            if norm is not None:
                im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                    norm=norm, origin='lower', aspect='equal',
                                    interpolation='bilinear')
            else:
                im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                    vmin=vmin, vmax=vmax,
                                    origin='lower', aspect='equal',
                                    interpolation='bilinear')

        # Optional grain-map ghost
        if show_grain_ghost and grain_map is not None:
            ghost = np.where(grain_map < 0, np.nan, grain_map)
            ax.imshow(ghost, extent=self.extent, cmap=ColormapRegistry.resolve('grain_map'),
                      origin='lower', aspect='equal', interpolation='nearest',
                      alpha=0.25 * alpha)

        # Overlay layer
        if mode == 'contour':
            ov_cmap = self.get_colormap(
                colormaps.get(overlay_field,
                              FIELD_DEFAULT_CMAPS.get(overlay_field, 'viridis')),
                invert=ov_fs.get('invert', False))
            x = np.linspace(self.extent[0], self.extent[1], self.Nx)
            y = np.linspace(self.extent[2], self.extent[3], self.Ny)
            X, Y = np.meshgrid(x, y)
            ov_for_contour = ov.copy()
            ov_for_contour[~np.isfinite(ov_for_contour)] = np.nan
            try:
                cs = ax.contour(X, Y, ov_for_contour, levels=n_contours,
                                cmap=ov_cmap, linewidths=1.0, alpha=alpha)
                ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
            except Exception as e:
                logger.warning(f"Overlay contour failed: {e}")
        else:
            ov_cmap = self.get_colormap(
                colormaps.get(overlay_field,
                              FIELD_DEFAULT_CMAPS.get(overlay_field, 'viridis')),
                invert=ov_fs.get('invert', False))
            _, ov_vmin, ov_vmax = compute_field_norm(overlay_field, ov, ov_fs)
            ax.imshow(ov, extent=self.extent, cmap=ov_cmap,
                      vmin=ov_vmin, vmax=ov_vmax,
                      origin='lower', aspect='equal',
                      interpolation='bilinear', alpha=alpha)

        cbar = plt.colorbar(im_base, ax=ax, fraction=0.046, pad=0.04)
        if base_unit:
            cbar.set_label(f"{base_field} ({base_unit})")

        ax.set_title(
            f"Overlay: {FIELD_LABELS.get(base_field, base_field)} + "
            f"{FIELD_LABELS.get(overlay_field, overlay_field)} "
            f"(mode={mode}, α={alpha:.2f})", fontsize=title_fs)
        ax.set_xlabel('x (nm)', fontsize=label_fs)
        ax.set_ylabel('y (nm)', fontsize=label_fs)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # VECTOR FIELD OVERLAY  (new)
    # ------------------------------------------------------------------
    @handle_errors
    def create_vector_overlay(self, results_dict, source, mode='gradient',
                              density=24, color='white', scale=1.0,
                              base_field=None, style_params=None):
        """
        Draw a quiver vector field on top of an optional scalar heatmap.
        """
        if style_params is None:
            style_params = {}
        field_settings = style_params.get('field_settings', {})
        title_fs = float(style_params.get('title_font_size', 11))
        label_fs = float(style_params.get('label_font_size', 9))

        vx, vy = compute_vector_field(results_dict, source, mode=mode)
        if vx is None:
            return None

        fig, ax = plt.subplots(figsize=(11, 5.5))

        if base_field and base_field in results_dict:
            base, base_unit = _rescale_field(base_field, results_dict[base_field])
            base_fs = field_settings.get(base_field, {})
            base_cmap = self.get_colormap(
                FIELD_DEFAULT_CMAPS.get(base_field, 'viridis'),
                invert=base_fs.get('invert', False))
            norm, vmin, vmax = compute_field_norm(base_field, base, base_fs)
            if norm is not None:
                im = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                               norm=norm, origin='lower', aspect='equal',
                               interpolation='bilinear')
            else:
                im = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                               vmin=vmin, vmax=vmax,
                               origin='lower', aspect='equal',
                               interpolation='bilinear')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=f"{base_field} ({base_unit})")
        else:
            ax.set_facecolor('#111111')

        # Downsample for quiver
        step = max(1, self.Nx // max(density, 1))
        x = np.linspace(self.extent[0], self.extent[1], self.Nx)[::step]
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)[::step]
        X, Y = np.meshgrid(x, y)
        U = vx[::step, ::step] * scale
        V = vy[::step, ::step] * scale

        # Normalize for visibility
        mag = np.sqrt(U**2 + V**2) + 1e-12
        U_n = U / mag
        V_n = V / mag

        ax.quiver(X, Y, U_n, V_n, mag, cmap='autumn', alpha=0.9,
                  scale=30, width=0.0025, pivot='mid')

        ax.set_title(f"Vector field: {mode} of {source}", fontsize=title_fs)
        ax.set_xlabel('x (nm)', fontsize=label_fs)
        ax.set_ylabel('y (nm)', fontsize=label_fs)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # PLOTLY HEATMAP  (log support, inversion)
    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_heatmap(self, results_dict, field_name, frame_idx=0,
                              cmap_name=None, zmid=None, viz_cfg=None):
        if field_name not in results_dict:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        d, unit = _rescale_field(field_name, results_dict[field_name])
        chosen = cmap_name or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis')
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))

        norm, vmin, vmax = compute_field_norm(field_name, d, fs)

        # Plotly log: apply log10 to z if requested (with clipping)
        z = d
        if fs.get('log', False):
            pos = d > 0
            z = np.where(pos, np.log10(np.maximum(d, 1e-30)), np.nan)

        if zmid is None and field_name == 'sigma_h':
            zmid = 0

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=z,
            x=np.linspace(self.extent[0], self.extent[1], self.Nx),
            y=np.linspace(self.extent[2], self.extent[3], self.Ny),
            colorscale=colorscale, zmid=zmid,
            zmin=(np.log10(vmin) if fs.get('log') and vmin and vmin > 0 else vmin),
            zmax=(np.log10(vmax) if fs.get('log') and vmax and vmax > 0 else vmax),
            colorbar=dict(title=f"{field_name} {unit}"),
            hovertemplate='x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>%{z:.4f}<extra></extra>'))
        fig.update_layout(title=f"{field_name} — Frame {frame_idx} ({chosen})",
                          xaxis_title="x (nm)", yaxis_title="y (nm)",
                          yaxis=dict(scaleanchor="x", scaleratio=self.dy / self.dx),
                          width=700, height=500, template="plotly_white")
        return fig

    # ------------------------------------------------------------------
    # PLOTLY 3D SURFACE  (with contours)
    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_3d_surface(self, results_dict, field_name, frame_idx=0,
                                 cmap_name=None, show_contours=True,
                                 viz_cfg=None):
        if field_name not in results_dict:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        d, unit = _rescale_field(field_name, results_dict[field_name])
        chosen = cmap_name or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis')
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))
        x = np.linspace(self.extent[0], self.extent[1], self.Nx)
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)
        X, Y = np.meshgrid(x, y)

        contours = None
        if show_contours:
            contours = dict(
                z=dict(show=True, usecolormap=True, project_z=True,
                       highlightcolor="limegreen"))

        fig = go.Figure(data=[go.Surface(
            z=d, x=X, y=Y, colorscale=colorscale, contours=contours,
            colorbar=dict(title=f"{field_name} {unit}"))])
        fig.update_layout(title=f"3D: {field_name} — Frame {frame_idx}",
                          scene=dict(xaxis_title='x (nm)', yaxis_title='y (nm)',
                                     zaxis_title=f'{field_name} {unit}',
                                     aspectmode='manual',
                                     aspectratio=dict(x=2, y=1, z=0.6)),
                          width=800, height=600, template="plotly_white")
        return fig

    # ------------------------------------------------------------------
    # PLOTLY LINE PROFILES
    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_line_profiles(self, results_dict, field_name, profile_types,
                                    position_ratio=0.5, viz_cfg=None):
        fig = go.Figure()
        d, unit = _rescale_field(field_name, results_dict[field_name])
        for pt in profile_types:
            res = self.line_profiler.extract_profile(d, pt, position_ratio)
            if res is FAILED or res is None:
                continue
            dist, prof, _ = res
            if dist is None:
                continue
            fig.add_trace(go.Scatter(x=dist, y=prof, mode='lines',
                                     name=pt.replace('_', ' ').title()))
        fig.update_layout(title=f"{field_name} Line Profiles",
                          xaxis_title="Position (nm)",
                          yaxis_title=f"{field_name} {unit}",
                          hovermode='x unified', template="plotly_white")
        return fig

    # ------------------------------------------------------------------
    # MULTI-TIME PROFILES  (new)
    # ------------------------------------------------------------------
    @handle_errors
    def create_multi_time_profiles(self, history, field_name, ptype='horizontal',
                                   frames=None, position_ratio=0.5,
                                   use_plotly=False):
        if frames is None:
            frames = list(range(len(history)))
        frames = [f for f in frames if 0 <= f < len(history)]
        if not frames:
            return None

        fig = go.Figure() if use_plotly else plt.figure(figsize=(11, 4.5)).gca()
        cmap = plt.cm.viridis if not use_plotly else None
        n = len(frames)
        for k, f in enumerate(frames):
            d, unit = _rescale_field(field_name, history[f][field_name])
            res = self.line_profiler.extract_profile(d, ptype, position_ratio)
            if res is FAILED or res is None:
                continue
            dist, prof, _ = res
            if dist is None:
                continue
            if use_plotly:
                fig.add_trace(go.Scatter(x=dist, y=prof, mode='lines',
                                         name=f"Frame {f}"))
            else:
                color = cmap(k / max(n - 1, 1))
                fig.plot(dist, prof, color=color, lw=1.8,
                         label=f"Frame {f}")

        if use_plotly:
            fig.update_layout(title=f"{field_name} — multi-time {ptype}",
                              xaxis_title="Position (nm)",
                              yaxis_title=f"{field_name}",
                              hovermode='x unified', template="plotly_white")
            return fig
        else:
            fig.set_xlabel('Position (nm)')
            fig.set_ylabel(f'{field_name}')
            fig.set_title(f'{field_name} — multi-time {ptype}')
            fig.grid(True, alpha=0.3)
            fig.legend(loc='best', fontsize=8)
            plt.tight_layout()
            return fig.figure

    # ------------------------------------------------------------------
    # PLOTLY TIME-SCRUBBING ANIMATION  (new)
    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_animation(self, history, field_name, dt,
                                cmap_name=None, viz_cfg=None):
        if not history:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        chosen = cmap_name or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis')
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))

        # Global range across all frames (BUG FIX for clipping drift)
        d0, unit = _rescale_field(field_name, history[0][field_name])
        if field_name in ('phi', 'eta_poly', 'eta_au', 'eta_nc', 'phi_gb'):
            zmin, zmax = 0, 1
        elif field_name == 'grain_map':
            zmin, zmax = None, None
        else:
            vmin_all, vmax_all = [], []
            for h in history:
                dd, _ = _rescale_field(field_name, h[field_name])
                finite = dd[np.isfinite(dd)]
                if finite.size:
                    vmin_all.append(np.percentile(finite, 2))
                    vmax_all.append(np.percentile(finite, 98))
            if vmin_all:
                zmin = float(np.min(vmin_all)); zmax = float(np.max(vmax_all))
            else:
                zmin, zmax = None, None
            if field_name == 'sigma_h' and zmin is not None:
                zm = max(abs(zmin), abs(zmax))
                zmin, zmax = -zm, zm

        x = np.linspace(self.extent[0], self.extent[1], self.Nx)
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)

        frames = []
        for i, h in enumerate(history):
            d, _ = _rescale_field(field_name, h[field_name])
            frames.append(go.Frame(
                data=[go.Heatmap(z=d, x=x, y=y, colorscale=colorscale,
                                 zmin=zmin, zmax=zmax,
                                 colorbar=dict(title=f"{field_name} {unit}"))],
                name=f"t={i*dt:.4f}"))

        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            title=f"{field_name} — interactive time scrubbing",
            xaxis_title="x (nm)", yaxis_title="y (nm)",
            yaxis=dict(scaleanchor="x", scaleratio=self.dy / self.dx),
            width=850, height=600, template="plotly_white",
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=150,
                                                     redraw=True),
                                          fromcurrent=True,
                                          transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0,
                                                       redraw=False),
                                            mode="immediate")]),
                ])],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[f.name],
                                  dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))],
                            label=f.name)
                       for f in frames],
                x=0.05, len=0.9, xanchor="left", y=0, yanchor="top",
                currentvalue=dict(prefix="Frame: ", visible=True,
                                  xanchor="right"))],
        )
        return fig

    # ------------------------------------------------------------------
    # STATIC ANIMATION (matplotlib GIF/MP4)  with watermark + global range
    # ------------------------------------------------------------------
    @handle_errors
    def create_animation_buffer(self, history, field_name, dt,
                                output_format='gif', fps=5, dpi=120,
                                cmap_name=None, watermark=False,
                                watermark_text='', watermark_pos='lower right',
                                viz_cfg=None):
        if not history:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
        cmap = ColormapRegistry.resolve(
            cmap_name or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'),
            invert=fs.get('invert', False))

        # Global vmin/vmax across all frames
        d0, unit = _rescale_field(field_name, history[0][field_name])
        if field_name in ('phi', 'eta_poly', 'eta_au', 'eta_nc', 'phi_gb'):
            vmin, vmax = 0, 1
        elif field_name == 'grain_map':
            vmin, vmax = None, None
        else:
            vmin_all, vmax_all = [], []
            for h in history:
                dd, _ = _rescale_field(field_name, h[field_name])
                finite = dd[np.isfinite(dd)]
                if finite.size:
                    vmin_all.append(np.percentile(finite, 2))
                    vmax_all.append(np.percentile(finite, 98))
            if vmin_all:
                vmin = float(np.min(vmin_all)); vmax = float(np.max(vmax_all))
            else:
                vmin, vmax = None, None
            if field_name == 'sigma_h' and vmin is not None:
                zm = max(abs(vmin), abs(vmax))
                vmin, vmax = -zm, zm

        interp = 'nearest' if field_name in DISCRETE_FIELDS else 'bilinear'
        im = ax.imshow(d0, extent=self.extent, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='lower', interpolation=interp, aspect='auto')
        if field_name != 'grain_map':
            cbar = plt.colorbar(im, ax=ax); cbar.set_label(f"{field_name} {unit}")
        title = ax.set_title(f"{field_name} — t = 0 ns")

        wm_artist = None
        if watermark and watermark_text:
            xmap = {'lower right': (0.98, 0.02, 'right', 'bottom'),
                    'lower left':  (0.02, 0.02, 'left',  'bottom'),
                    'upper right': (0.98, 0.98, 'right', 'top'),
                    'upper left':  (0.02, 0.98, 'left',  'top')}
            x, y, ha, va = xmap.get(watermark_pos, (0.98, 0.02, 'right', 'bottom'))
            wm_artist = ax.text(x, y, watermark_text, transform=ax.transAxes,
                                fontsize=8, color='white', alpha=0.75,
                                ha=ha, va=va,
                                bbox=dict(boxstyle='round,pad=0.25',
                                          facecolor='black', alpha=0.35,
                                          edgecolor='none'))

        def update(i):
            d, _ = _rescale_field(field_name, history[i][field_name])
            im.set_array(d)
            title.set_text(f"{field_name} — t = {i * dt:.4f} ns")
            return ([im, title] if wm_artist is None else [im, title, wm_artist])

        ani = animation.FuncAnimation(fig, update, frames=len(history),
                                       interval=1000 / max(fps, 1), blit=True)
        buf = BytesIO()
        try:
            if output_format == 'gif':
                ani.save(buf, writer='pillow', fps=fps, dpi=dpi)
            else:
                ani.save(buf, writer='ffmpeg', fps=fps, dpi=dpi)
        except Exception as e:
            logger.error(f"Animation export failed: {e}")
            plt.close(fig)
            return None
        plt.close(fig)
        buf.seek(0)
        return buf


# ============================================================================
# EXPORTERS
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


class DataExporter:
    @staticmethod
    @handle_errors
    def export_pkl(sim_data, params, history, name):
        buf = BytesIO()
        pickle.dump({'params': params, 'history': history,
                     'metadata': sim_data.get('metadata', {})}, buf)
        buf.seek(0)
        return buf, f"{name}.pkl"

    @staticmethod
    @handle_errors
    def export_pt(sim_data, params, history, name):
        if not TORCH_AVAILABLE:
            st.error("torch not installed — pip install torch")
            return None, None
        buf = BytesIO()
        out = {'params': params, 'metadata': sim_data.get('metadata', {}),
               'history': []}
        for frame in history:
            frame_out = {}
            for k, v in frame.items():
                if isinstance(v, np.ndarray):
                    frame_out[k] = torch.from_numpy(np.asarray(v))
                elif isinstance(v, dict):
                    frame_out[k] = {kk: (torch.from_numpy(np.asarray(vv))
                                          if isinstance(vv, np.ndarray) else vv)
                                    for kk, vv in v.items()}
            out['history'].append(frame_out)
        torch.save(out, buf)
        buf.seek(0)
        return buf, f"{name}.pt"

    @staticmethod
    @handle_errors
    def export_json(sim_data, params, history, name):
        def conv(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.generic): return o.item()
            if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
            if isinstance(o, list): return [conv(i) for i in o]
            return o
        s = json.dumps({'params': params,
                        'metadata': sim_data.get('metadata', {}),
                        'history': conv(history)},
                       indent=2, cls=NumpyEncoder)
        return BytesIO(s.encode()), f"{name}.json"

    @staticmethod
    @handle_errors
    def export_csv(history, name, extent, Nx, Ny, dx, dy):
        """BUG FIX: include sigma_y in the tuple so its branch is reachable."""
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            x = np.linspace(extent[0], extent[1], Nx)
            y = np.linspace(extent[2], extent[3], Ny)
            X, Y = np.meshgrid(x, y)
            keys = ('phi', 'eta_poly', 'eta_au', 'eta_nc', 'grain_map',
                    'sigma_eq', 'sigma_h', 'sigma_y', 'h', 'eps_p_mag',
                    'D_eff', 'rho_gnd')
            for i, frame in enumerate(history):
                d = {'x': X.flatten(), 'y': Y.flatten()}
                for k in keys:
                    if k in frame:
                        v = frame[k]
                        if k in ('sigma_eq', 'sigma_h'): v = v / 1e9
                        elif k == 'sigma_y': v = v / 1e6
                        d[k] = np.asarray(v).flatten()
                zf.writestr(f"{name}_frame_{i:04d}.csv",
                            pd.DataFrame(d).to_csv(index=False))
        buf.seek(0)
        return buf, f"{name}_csv.zip"

    @staticmethod
    @handle_errors
    def export_hdf5(sim_data, params, history, name, Nx, Ny, dx, dy):
        if not H5PY_AVAILABLE:
            st.error("h5py not installed — pip install h5py")
            return None, None
        buf = BytesIO()
        with h5py.File(buf, 'w') as f:
            pg = f.create_group('parameters')
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    pg.attrs[k] = v
            f.create_dataset('x', data=np.linspace(-Nx*dx/2, Nx*dx/2, Nx))
            f.create_dataset('y', data=np.linspace(-Ny*dy/2, Ny*dy/2, Ny))
            fg = f.create_group('frames')
            for i, frame in enumerate(history):
                g = fg.create_group(f'frame_{i:04d}')
                for k, v in frame.items():
                    if isinstance(v, np.ndarray):
                        g.create_dataset(k, data=v)
        buf.seek(0)
        return buf, f"{name}.h5"


# ============================================================================
# SIMULATION DATABASE  (with clear / bounded cache)
# ============================================================================
class SimulationDatabase:
    MAX_SIMS = 32   # BUG FIX P3: bounded cache

    @staticmethod
    def _hash(params):
        s = json.dumps({k: v for k, v in params.items()
                        if k not in ('history', 'results', 'geom_viz',
                                     'n_steps', 'save_frequency')},
                       sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]

    @staticmethod
    @handle_errors
    def save(params, history, geometry_data, metadata=None, run_time=0.0):
        if 'sims' not in st.session_state:
            st.session_state.sims = {}
        sid = SimulationDatabase._hash(params)
        st.session_state.sims[sid] = {
            'id': sid, 'params': params, 'results_history': history,
            'geometry_data': geometry_data,
            'metadata': metadata or {'created_at': datetime.now().isoformat(),
                                      'run_time': run_time},
            'created_at': datetime.now().isoformat(),
        }
        # Enforce bound: drop oldest if too many
        if len(st.session_state.sims) > SimulationDatabase.MAX_SIMS:
            oldest = min(st.session_state.sims.items(),
                         key=lambda kv: kv[1].get('created_at', ''))
            del st.session_state.sims[oldest[0]]
        return sid

    @staticmethod
    def get(sid):
        return st.session_state.get('sims', {}).get(sid)

    @staticmethod
    def delete(sid):
        if sid in st.session_state.get('sims', {}):
            del st.session_state.sims[sid]
            return True
        return False

    @staticmethod
    def clear_all():
        st.session_state.sims = {}

    @staticmethod
    def all():
        return st.session_state.get('sims', {})

    @staticmethod
    def list_for_ui():
        out = []
        for sid, sim in SimulationDatabase.all().items():
            p = sim.get('params', {})
            out.append({
                'id': sid,
                'name': (f"Au-Cu | n_poly={p.get('n_poly_grains', '?')} | "
                         f"nc={p.get('nc_grain_size_nm',0):.0f}nm | "
                         f"σ={p.get('applied_stress',0)/1e6:.0f}MPa | "
                         f"D_gb={p.get('D_gb',0):.1f}"),
                'params': p,
            })
        return out


# ============================================================================
# ANIMATION HELPER (kept for backward compat — delegates to visualizer)
# ============================================================================
@handle_errors
def make_animation(history, field_name, dt, extent, Nx, Ny, output_format='gif',
                   fps=5, dpi=120, cmap_name=None, watermark=False,
                   watermark_text='', watermark_pos='lower right'):
    vis = EnhancedTwinVisualizer(Nx, Ny,
                                 (extent[1] - extent[0]) / max(Nx - 1, 1),
                                 (extent[3] - extent[2]) / max(Ny - 1, 1),
                                 dt=dt)
    return vis.create_animation_buffer(
        history, field_name, dt, output_format=output_format, fps=fps, dpi=dpi,
        cmap_name=cmap_name, watermark=watermark,
        watermark_text=watermark_text, watermark_pos=watermark_pos)


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="Au-Cu Diffusion Bonding (Rectangular MPF)",
        layout="wide", initial_sidebar_state="expanded")

    # Initialize visualization config
    viz_cfg = get_viz_config()
    apply_matplotlib_theme(viz_cfg['theme'], viz_cfg)

    st.markdown("""
    <style>
    .main-header { font-size: 2.1rem; text-align: center; margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #8B4513, #C9A227);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .viz-badge { background:#eef; padding:2px 6px; border-radius:4px;
                 font-size:0.8rem; color:#335; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🥇 Au-Assisted Cu-Cu Diffusion Bonding '
                '(Rectangular MPF) — v3</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#FFF8E7; padding:1rem; border-radius:8px;
                border-left:5px solid #C9A227;">
    <b>Physics:</b> MPF + Fickian/Hart + Solute Drag + Elastoplasticity (J2).<br>
    • Top <b>Poly-Cu</b> · Middle <b>Au film</b> · Bottom <b>nc-Cu</b>.<br>
    • 30 MPa downward → ∇σ_h drives Au <b>selectively into nc-Cu</b>.<br>
    • <b>Auto-substepping</b> for both Fickian and Cahn–Hilliard paths.<br>
    • <b>Guarded mass conservation</b> with actual step rejection.<br>
    <span class="viz-badge">v3 adds: overlays · vector fields · log scaling ·
    Plotly time scrubbing · themes · watermarks</span>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        mode = st.radio("Mode",
                        ["Run New Simulation", "Single Simulation View",
                         "Compare Saved"],
                        index=0, key="mode")

        if mode == "Run New Simulation":
            st.subheader("🧱 Geometry")
            poly_frac = st.slider("Poly-Cu thickness fraction", 0.10, 0.60, 0.45, 0.05)
            au_frac   = st.slider("Au film thickness fraction", 0.01, 0.15, 0.04, 0.005)

            n_poly_grains = st.slider(
                "Number of Poly-Cu grains", 1, 5, 2, 1,
                help="Keep low (1–2) for a ~200 nm wide domain to represent "
                     "bulk grains with d > 100 nm.")

            poly_gs   = st.slider("Poly-Cu grain size target (nm)",
                                   50.0, 500.0, 150.0, 10.0)
            nc_gs     = st.slider("nc-Cu grain size (nm)", 5.0, 60.0, 15.0, 1.0)
            gb_w      = st.slider("GB width (nm)", 0.5, 5.0, 1.5, 0.1)
            curve_amp = st.slider("Poly-Cu boundary waviness (nm)",
                                   0.0, 20.0, 8.0, 1.0)
            use_warp = st.checkbox(
                "Apply warp to Voronoi boundaries", value=True)
            relax_st  = st.slider("Curvature relaxation steps", 0, 200, 10, 5,
                                   help="High values can absorb small grains.")
            seed_in   = st.number_input("Random seed", value=1234, step=1)

            st.subheader("🔲 Grid (rectangular)")
            Nx = st.slider("Nx (grid points in x)", 64, 512, 256, 32)
            Ny = st.slider("Ny (grid points in y)", 64, 512, 128, 32)
            dx = st.slider("dx (nm)", 0.2, 2.0, 0.8, 0.1)
            dy = st.slider("dy (nm)", 0.2, 2.0, 0.8, 0.1)
            dt = st.slider("dt (ns)", 1e-4, 5e-2, 5e-3, 1e-4, format="%.5f")
            st.caption(f"Domain: {Nx*dx:.1f} nm × {Ny*dy:.1f} nm")

            st.subheader("🌡️ Thermodynamics")
            W  = st.slider("W — chemical well depth", 0.1, 10.0, 2.0, 0.1)
            A  = st.slider("A — grain double-well",   0.1, 20.0, 5.0, 0.5)
            B  = st.slider("B — grain anti-overlap",  0.1, 30.0, 10.0, 0.5)
            kappa_eta = st.slider("κ_η (gradient)",  0.05, 5.0, 1.0, 0.05)
            aniso_str = st.slider("Anisotropy strength δ", 0.0, 0.8, 0.3, 0.05)

            st.subheader("🕳️ Diffusion (Hart)")
            use_ch = st.checkbox(
                "Use Cahn-Hilliard for Au (uses W)", value=False,
                help="CH is more robust to stiff drift terms.")
            D_lat = st.slider("D_lattice", 1e-4, 1.0, 0.01, 1e-3, format="%.4f")
            D_gb  = st.slider("D_gb (grain boundaries)", 1.0, 100.0, 50.0, 1.0)
            Omega = st.slider("Ω (stress-drift coupling)", 0.0, 0.5, 0.05, 0.005)
            if use_ch:
                kappa_c = st.slider("κ_c (CH gradient)", 0.05, 5.0, 0.5, 0.05)
                M_ref   = st.slider("M_ref (CH stabilizer)", 0.1, 10.0, 1.0, 0.1)
            else:
                kappa_c, M_ref = 0.5, 1.0

            st.subheader("🐌 Solute Drag")
            L_eta      = st.slider("L₀ (base GB mobility)", 0.1, 5.0, 1.0, 0.1)
            alpha_drag = st.slider("α (drag strength)", 0.0, 500.0, 50.0, 10.0)

            st.subheader("🔗 Mechanics")
            applied_mpa = st.slider("Applied stress (MPa)", 0.0, 300.0, 30.0, 5.0)
            theta_deg   = st.slider("Loading angle θ (°, 90=down)", 0.0, 180.0, 90.0, 5.0)
            beta_misfit = st.slider("β (Au-Cu misfit)", 0.0, 0.10, 0.045, 0.005)

            st.subheader("⏯️ Run Control")
            n_steps = st.slider("Steps", 20, 2000, 300, 20)
            save_freq = st.slider("Save every N steps", 1, 100, 10, 1)

            if st.button("🚀 Initialize", type="primary",
                         use_container_width=True):
                params = {
                    'Nx': Nx, 'Ny': Ny, 'dx': dx, 'dy': dy, 'dt': dt,
                    'poly_thickness_frac': poly_frac,
                    'au_thickness_frac':   au_frac,
                    'n_poly_grains':       int(n_poly_grains),
                    'poly_grain_size_nm':  poly_gs,
                    'nc_grain_size_nm':    nc_gs,
                    'gb_width_nm':         gb_w,
                    'poly_curve_amp':      curve_amp,
                    'use_warp':            bool(use_warp),
                    'relax_steps':         relax_st,
                    'seed':                int(seed_in),
                    'W': W, 'A': A, 'B': B,
                    'kappa_eta': kappa_eta, 'aniso_strength': aniso_str,
                    'L_ref': 1.0,
                    'use_cahn_hilliard':   bool(use_ch),
                    'kappa_c':             kappa_c,
                    'M_ref':               M_ref,
                    'D_lattice': D_lat, 'D_gb': D_gb, 'Omega': Omega,
                    'L_eta': L_eta, 'alpha_drag': alpha_drag,
                    'applied_stress': applied_mpa * 1e6,
                    'applied_stress_angle': theta_deg,
                    'misfit_beta': beta_misfit,
                    'n_steps': n_steps, 'save_frequency': save_freq,
                    'material': 'Cu',
                }
                errors, warns = MaterialProperties.validate_parameters(params)
                if errors:
                    st.error("; ".join(errors))
                else:
                    for w in warns: st.warning(w)
                    st.session_state.params = params
                    st.session_state.initialized = True
                    geom = AuCuBondingGeometry(Nx, Ny, dx, dy)
                    res = geom.build(
                        poly_thickness_frac=poly_frac,
                        au_thickness_frac=au_frac,
                        n_poly_grains=int(n_poly_grains),
                        poly_grain_size_nm=poly_gs,
                        nc_grain_size_nm=nc_gs,
                        gb_width_nm=gb_w,
                        poly_curve_amp=curve_amp,
                        seed=int(seed_in),
                        use_warp=bool(use_warp))
                    if res is FAILED or res is None:
                        st.error(f"Geometry build failed: {FAILED.reason}")
                    else:
                        c, eta_list, gm, lid, meta = res
                        st.session_state.geom = geom
                        st.session_state.c0 = c
                        st.session_state.eta0 = eta_list
                        st.session_state.gm0 = gm
                        st.session_state.meta0 = meta
                        st.success(
                            f"✅ Initialized: {meta['n_poly']} Poly grain(s) "
                            f"+ 1 Au + {meta['n_nc']} nc = {meta['n_total']} phases")

        elif mode == "Single Simulation View":
            sims = SimulationDatabase.list_for_ui()
            if not sims:
                st.info("No saved simulations.")
            else:
                names = [s['name'] for s in sims]
                chosen = st.selectbox("Select", names)
                st.session_state.view_id = sims[names.index(chosen)]['id']

        elif mode == "Compare Saved":
            sims = SimulationDatabase.list_for_ui()
            if not sims:
                st.info("No saved simulations.")
            else:
                names = [s['name'] for s in sims]
                picked = st.multiselect("Select", names,
                                         default=names[:min(2, len(names))])
                st.session_state.compare_ids = [sims[names.index(n)]['id']
                                                 for n in picked]

        # Database utilities
        st.markdown("---")
        with st.expander("🗄️ Simulation database", expanded=False):
            n_sims = len(SimulationDatabase.all())
            st.caption(f"{n_sims} saved simulations (max {SimulationDatabase.MAX_SIMS})")
            if st.button("🧹 Clear all saved simulations"):
                SimulationDatabase.clear_all()
                st.success("Cleared.")
                st.rerun()

    # ==================================================================
    # MAIN AREA
    # ==================================================================
    if mode == "Run New Simulation" and st.session_state.get('initialized'):
        params = st.session_state.params
        Nx, Ny = params['Nx'], params['Ny']
        dx, dy = params['dx'], params['dy']
        meta = st.session_state.meta0

        tabs = st.tabs(["📐 Geometry", "▶️ Run", "🎨 Visualization Settings",
                        "📊 Results", "🔬 Line Profiles", "🌐 Plotly",
                        "📤 Export"])

        # ---- Geometry ----------------------------------------------
        with tabs[0]:
            st.header("Initial 3-Layer Geometry (Rectangular)")
            st.caption(f"Poly-Cu: {meta['n_poly']} coarse grain(s) · "
                       f"Au: 1 flat slab · nc-Cu: {meta['n_nc']} fine grains · "
                       f"Total = {meta['n_total']} phases")
            st.caption(f"Domain: {Nx*dx:.1f} nm × {Ny*dy:.1f} nm  |  "
                       f"grid: {Nx} × {Ny
