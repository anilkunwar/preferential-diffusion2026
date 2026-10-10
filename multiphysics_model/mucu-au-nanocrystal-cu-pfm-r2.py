"""
Au-assisted Cu-Cu diffusion bonding simulator — rectangular MPF edition
=======================================================================
FIXES in this revision:
  • axhline() on 2D wavy interface arrays → replaced with ax.plot()
  • Added _extract_interface_profile() helper (robust to multi-valued interfaces)
  • Added layer-boundary overlay in initial-geometry Au-concentration plot
  • All other physics, geometry, visualization preserved unchanged
"""

import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import AutoMinorLocator
import matplotlib.animation as animation
import plotly.graph_objects as go
from datetime import datetime
import json, zipfile, pickle, torch, sqlite3, hashlib, traceback, warnings
from io import BytesIO, StringIO
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ERROR HANDLING
# ============================================================================
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Error in {func.__name__}: {e}")
            logger.error(f"{e}\n{traceback.format_exc()}")
            return None
    return wrapper


# ============================================================================
# RECTANGULAR SPECTRAL OPERATORS
# ============================================================================
def make_k_vectors(Nx, Ny, dx, dy=None):
    """2D wavenumber grids for a rectangular (Ny, Nx) domain."""
    if dy is None:
        dy = dx
    kx = 2 * np.pi * fftfreq(Nx, d=dx).reshape(1, -1)   # (1, Nx)
    ky = 2 * np.pi * fftfreq(Ny, d=dy).reshape(-1, 1)   # (Ny, 1)
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
# INTERFACE PROFILING HELPERS (fix for wavy 2D interface arrays)
# ============================================================================
def _extract_interface_profile(interface_2d):
    """
    Extract a representative 1D profile from a 2D interface-level field.

    The interface field y_*(x, y) is a scalar level-set of the layer boundary.
    For a boundary that only varies along x (which is our case: the wave was
    built from np.sin(2π x / λ) broadcast across y), every row is identical,
    so we simply return row 0.

    If the interface has genuine y-variation (advanced geometry), we return
    the column-wise mean so we still get a sensible 1D curve.
    """
    a = np.asarray(interface_2d)
    if a.ndim == 0:
        return None
    if a.ndim == 1:
        return a
    # Same row everywhere? then just take row 0
    if a.shape[0] > 1 and np.allclose(a[0, :], a[-1, :], atol=1e-9):
        return a[0, :]
    # Otherwise average over rows — graceful fallback
    return np.mean(a, axis=0)


def _plot_interface(ax, x_coords, interface_2d, **line_kwargs):
    """
    Plot a (possibly wavy) layer interface on an imshow axes.

    If the interface is scalar (legacy case), use axhline.
    If it is 1D/2D, use ax.plot with the extracted profile.
    """
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
    except Exception:
        pass


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
    def resolve(cls, name):
        if name is None: return mpl.colormaps['viridis']
        if isinstance(name, Colormap): return name
        key = str(name)
        if key in mpl.colormaps: return mpl.colormaps[key]
        if key in _CUSTOM_COLORMAPS: return _CUSTOM_COLORMAPS[key]
        logger.warning(f"Unknown colormap '{name}', using viridis")
        return mpl.colormaps['viridis']

    @classmethod
    def to_plotly(cls, name, n=64):
        cmap = cls.resolve(name)
        t = np.linspace(0, 1, n)
        rgb = (cmap(t)[:, :3] * 255).astype(int)
        return [[float(ti), f'rgb({r},{g},{b})'] for ti, (r, g, b) in zip(t, rgb)]


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

# Discrete fields get "nearest" interpolation (no bilinear blur)
DISCRETE_FIELDS = {'grain_map', 'phi_gb'}


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
        D_ref = max(params.get('D_gb', 1.0), params.get('D_lattice', 0.01))
        dt_stab = 0.25 * min(params.get('dx', 1.0)**2,
                             params.get('dy', 1.0)**2) / max(D_ref, 1e-9)
        if params.get('dt', 0) > dt_stab:
            warns.append(f"dt > dx²/(4 D_gb) ≈ {dt_stab:.4f} — "
                         f"semi-implicit stabilizer engaged")
        return errors, warns


# ============================================================================
# VORONOI GEOMETRY — RECTANGULAR, CURVED GRAINS
# ============================================================================
def voronoi_assign(Nx, Ny, extent, seeds, mask=None, warp=None):
    """Assign each pixel to nearest seed. warp=(sx, sy) optional distortion field."""
    x = np.linspace(extent[0], extent[1], Nx)
    y = np.linspace(extent[2], extent[3], Ny)
    X, Y = np.meshgrid(x, y)             # (Ny, Nx)
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
        tree = cKDTree(seeds + np.column_stack([
            np.interp(seeds[:, 0], x, np.mean(wx, axis=0)),
            np.interp(seeds[:, 1], y, np.mean(wy, axis=1))]))
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
    """Smooth random warp field for making Voronoi boundaries wavy."""
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
    """
    Rectangular 3-layer domain:
       Top    : Poly-Cu   (coarse Voronoi, curved via warp+relaxation)
       Middle : Au film   (single phase)
       Bottom : nc-Cu     (fine Voronoi, sharp)
    """

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
              poly_grain_size_nm=150.0,
              nc_grain_size_nm=15.0,
              gb_width_nm=1.5,
              poly_curve_amp=8.0,
              seed=1234):
        rng = np.random.default_rng(seed)
        Nx, Ny = self.Nx, self.Ny
        y_top, y_bot = self.extent[3], self.extent[2]
        total_h = y_top - y_bot

        y_poly_bot = y_top - total_h * poly_thickness_frac
        y_au_bot   = y_poly_bot - total_h * au_thickness_frac

        # Wavy layer interfaces (2D arrays — the source of the axhline issue)
        interface_wave_amp = 1.5
        wave = interface_wave_amp * np.sin(2 * np.pi * self.X / 60.0)
        y_poly_bot_eff = y_poly_bot + wave        # shape (Ny, Nx)
        y_au_bot_eff   = y_au_bot   + 0.5 * wave  # shape (Ny, Nx)

        mask_poly = self.Y > y_poly_bot_eff
        mask_au   = (self.Y <= y_poly_bot_eff) & (self.Y > y_au_bot_eff)
        mask_nc   = self.Y <= y_au_bot_eff

        # Seed counts
        Lx = self.extent[1] - self.extent[0]
        area_poly = Lx * (y_top - y_poly_bot)
        area_nc   = Lx * (y_au_bot - y_bot)
        n_poly = max(2, int(round(area_poly / (poly_grain_size_nm ** 2))))
        n_nc   = max(6, int(round(area_nc   / (nc_grain_size_nm   ** 2))))

        def sample_seeds(mask, n):
            ys, xs = np.where(mask)
            if len(ys) == 0: return np.zeros((0, 2))
            picks = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
            return np.column_stack([self.X[ys[picks], xs[picks]],
                                     self.Y[ys[picks], xs[picks]]])

        poly_seeds = sample_seeds(mask_poly, n_poly)
        nc_seeds   = sample_seeds(mask_nc,   n_nc)

        # Wavy warp for the poly layer (makes Voronoi edges curved)
        wx, wy = make_warp_field(Nx, Ny, self.extent,
                                  amplitude=poly_curve_amp,
                                  wavelength_nm=70.0, seed=seed)
        warp_poly = (wx, wy)

        gm_poly = voronoi_assign(Nx, Ny, self.extent, poly_seeds, mask_poly,
                                  warp=warp_poly)
        gm_nc   = voronoi_assign(Nx, Ny, self.extent, nc_seeds,   mask_nc,
                                  warp=None)

        # Global grain IDs
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

        # Diffuse fields (different widths per layer)
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
# 2D PLANE-STRAIN SPECTRAL ELASTICITY (correct Green's function)
# ============================================================================
class SpectralElasticSolver:
    """
    Solve ∇·σ = 0 for σ = C:(ε_total - ε*) via plane-strain Fourier Green's fn.
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
        C11_2d = (C11 + C12 + 2 * C44) / 2
        C12_2d = (C11 + C12 - 2 * C44) / 2
        lam = C12_2d
        mu  = (C11_2d - C12_2d) / 2
        self.C11_2d, self.C12_2d, self.C44_2d = C11_2d, C12_2d, C44
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
# MAIN MPF SOLVER (rectangular, Fickian + drift, anisotropic, solute drag)
# ============================================================================
class AuCuBondingSolver:
    """Fully coupled MPF + Fickian/Hart + Elastoplastic solver."""

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

        self.geom = AuCuBondingGeometry(self.Nx, self.Ny, self.dx, self.dy)
        (self.c, self.eta_list, self.grain_map,
         self.layer_id, self.geom_meta) = self.geom.build(
            poly_thickness_frac=params.get('poly_thickness_frac', 0.45),
            au_thickness_frac=params.get('au_thickness_frac', 0.04),
            poly_grain_size_nm=params.get('poly_grain_size_nm', 150.0),
            nc_grain_size_nm=params.get('nc_grain_size_nm', 15.0),
            gb_width_nm=params.get('gb_width_nm', 1.5),
            poly_curve_amp=params.get('poly_curve_amp', 8.0),
            seed=params.get('seed', 1234))

        self.eta_list = self._relax_grains(self.eta_list,
                                           n_steps=params.get('relax_steps', 40))

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

        self.kappa_eta = params.get('kappa_eta', 1.0)
        self.aniso_strength = params.get('aniso_strength', 0.3)
        self.L_ref = params.get('L_ref', 1.0)

        self.history = {
            'time': [], 'c_top': [], 'c_bot': [],
            'gb_top': [], 'gb_bot': [],
            'max_stress': [], 'avg_stress': [],
            'plastic_work': [], 'energy': [],
            'au_mass': [],
        }

    def _relax_grains(self, etas, n_steps=40):
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

    def compute_gb_indicator(self):
        s = np.zeros((self.Ny, self.Nx))
        for e in self.eta_list:
            s += e**2
        return np.clip(1.0 - s, 0, 1)

    def compute_local_grain_size(self):
        s = np.sum(self.eta_list, axis=0)
        gx, gy = spectral_gradients(s, self.kx, self.ky)
        gm = np.sqrt(gx**2 + gy**2 + 1e-15)
        return np.clip(2.0 / gm, 0, 300.0)

    def evolve_concentration(self, sigma_h):
        phi_gb = self.compute_gb_indicator()

        D_lat = float(self.params.get('D_lattice', 0.01))
        D_gb  = float(self.params.get('D_gb', 50.0))
        Omega = float(self.params.get('Omega', 0.05))

        D_eff = D_lat + (D_gb - D_lat) * phi_gb
        D_ref = float(np.max(D_eff)) + 1e-12
        delta_D = D_eff - D_ref

        c_gx, c_gy = spectral_gradients(self.c, self.kx, self.ky)
        Jx_d = -delta_D * c_gx
        Jy_d = -delta_D * c_gy

        sig_gx, sig_gy = spectral_gradients(sigma_h, self.kx, self.ky)
        Jx_s = -D_eff * Omega * self.c * sig_gx
        Jy_s = -D_eff * Omega * self.c * sig_gy

        Jx = Jx_d + Jx_s
        Jy = Jy_d + Jy_s
        div_J = spectral_divergence(Jx, Jy, self.kx, self.ky)

        c_hat = fft2(self.c + self.dt * div_J)
        c_new = np.real(ifft2(c_hat / (1.0 + self.dt * D_ref * self.k2)))
        return np.clip(c_new, 0, 1), D_eff

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
            denom = 1.0 + self.dt * self.L_ref * kappa0 * self.k2
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

    def compute_plasticity(self, sigma_eq, h):
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
        d_inc = np.minimum(2.0 / 3.0 * gamma_dot * self.dt, 1e-3)

        self.eps_p_xx = np.clip(self.eps_p_xx + d_inc,       -0.1, 0.1)
        self.eps_p_yy = np.clip(self.eps_p_yy - 0.5 * d_inc, -0.1, 0.1)
        self.eps_p_xy = np.clip(self.eps_p_xy + 0.5 * d_inc, -0.1, 0.1)

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

        (sigma_eq, sxx, syy, sxy, sigma_h,
         exx_tot, eyy_tot, exy_tot) = self.elastic.solve(
            exx_t, eyy_t, exy_t, app_xx, app_yy, app_xy)

        h = self.compute_local_grain_size()
        sigma_y, eps_p_mag = self.compute_plasticity(sigma_eq, h)
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
            'convergence': {
                'c_top': c_top, 'c_bot': c_bot,
                'au_mass': au_mass,
                'max_stress': float(np.max(sigma_eq)),
                'avg_stress': float(np.mean(sigma_eq)),
                'energy': energy,
            }
        }


# ============================================================================
# LINE PROFILER
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


# ============================================================================
# VISUALIZER
# ============================================================================
def _rescale_field(field_name, data):
    d = data.copy()
    if field_name in ('sigma_eq', 'sigma_h'):
        d = d / 1e9
    elif field_name == 'sigma_y':
        d = d / 1e6
    return d, FIELD_UNITS.get(field_name, '')


class EnhancedTwinVisualizer:
    def __init__(self, Nx, Ny, dx, dy=None, dt=1e-4):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, (dy if dy is not None else dx)
        self.dt = dt
        self.extent = [-Nx * self.dx / 2, Nx * self.dx / 2,
                       -Ny * self.dy / 2, Ny * self.dy / 2]
        self.line_profiler = EnhancedLineProfiler(Nx, Ny, dx, self.dy)

    def get_colormap(self, name):
        return ColormapRegistry.resolve(name)

    @handle_errors
    def create_multi_field_comparison(self, results_dict, style_params=None):
        if style_params is None:
            style_params = {}
        colormaps = style_params.get('colormaps', {})
        title_fs = float(style_params.get('title_font_size', 10))
        label_fs = float(style_params.get('label_font_size', 8))

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
            ('h',         'Local grain size (nm)',       [0, 100]),
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

            cmap_name = (colormaps.get(fname)
                         or style_params.get(f'{fname}_cmap')
                         or FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
            cmap = self.get_colormap(cmap_name)

            interp = 'nearest' if fname in DISCRETE_FIELDS else 'bilinear'

            if fname == 'grain_map':
                im = ax.imshow(d, extent=self.extent, cmap=cmap,
                               origin='lower', aspect='equal',
                               interpolation='nearest')
                ax.set_title(title, fontsize=title_fs)
                ax.set_xlabel('x (nm)', fontsize=label_fs)
                ax.set_ylabel('y (nm)', fontsize=label_fs)
                continue

            if rng is not None:
                vmin, vmax = rng
            else:
                vmin = np.percentile(d, 2)
                vmax = np.percentile(d, 98)
                if fname == 'sigma_h':
                    vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
                if vmax - vmin < 1e-9:
                    vmax = vmin + 1e-9

            im = ax.imshow(d, extent=self.extent, cmap=cmap, vmin=vmin, vmax=vmax,
                           origin='lower', aspect='equal', interpolation=interp)
            ax.set_title(title, fontsize=title_fs)
            ax.set_xlabel('x (nm)', fontsize=label_fs)
            ax.set_ylabel('y (nm)', fontsize=label_fs)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if unit:
                cbar.set_label(unit)

        for k in range(n, len(axes)):
            axes[k].axis('off')
        plt.tight_layout()
        return fig

    @handle_errors
    def create_plotly_heatmap(self, results_dict, field_name, frame_idx=0,
                              cmap_name=None, zmid=None):
        if field_name not in results_dict:
            return None
        d, unit = _rescale_field(field_name, results_dict[field_name])
        chosen = cmap_name or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis')
        colorscale = ColormapRegistry.to_plotly(chosen)
        if zmid is None and field_name == 'sigma_h':
            zmid = 0
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=d,
            x=np.linspace(self.extent[0], self.extent[1], self.Nx),
            y=np.linspace(self.extent[2], self.extent[3], self.Ny),
            colorscale=colorscale, zmid=zmid,
            colorbar=dict(title=f"{field_name} {unit}"),
            hovertemplate='x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>%{z:.4f}<extra></extra>'))
        fig.update_layout(title=f"{field_name} — Frame {frame_idx} ({chosen})",
                          xaxis_title="x (nm)", yaxis_title="y (nm)",
                          yaxis=dict(scaleanchor="x", scaleratio=self.dy / self.dx),
                          width=700, height=500, template="plotly_white")
        return fig

    @handle_errors
    def create_plotly_3d_surface(self, results_dict, field_name, frame_idx=0,
                                 cmap_name=None):
        if field_name not in results_dict:
            return None
        d, unit = _rescale_field(field_name, results_dict[field_name])
        chosen = cmap_name or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis')
        colorscale = ColormapRegistry.to_plotly(chosen)
        x = np.linspace(self.extent[0], self.extent[1], self.Nx)
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)
        X, Y = np.meshgrid(x, y)
        fig = go.Figure(data=[go.Surface(
            z=d, x=X, y=Y, colorscale=colorscale,
            colorbar=dict(title=f"{field_name} {unit}"))])
        fig.update_layout(title=f"3D: {field_name} — Frame {frame_idx}",
                          scene=dict(xaxis_title='x (nm)', yaxis_title='y (nm)',
                                     zaxis_title=f'{field_name} {unit}',
                                     aspectmode='manual',
                                     aspectratio=dict(x=2, y=1, z=0.6)),
                          width=800, height=600, template="plotly_white")
        return fig

    @handle_errors
    def create_plotly_line_profiles(self, results_dict, field_name, profile_types,
                                    position_ratio=0.5):
        fig = go.Figure()
        d, unit = _rescale_field(field_name, results_dict[field_name])
        for pt in profile_types:
            dist, prof, _ = self.line_profiler.extract_profile(d, pt, position_ratio)
            if dist is None: continue
            fig.add_trace(go.Scatter(x=dist, y=prof, mode='lines',
                                     name=pt.replace('_', ' ').title()))
        fig.update_layout(title=f"{field_name} Line Profiles",
                          xaxis_title="Position (nm)",
                          yaxis_title=f"{field_name} {unit}",
                          hovermode='x unified', template="plotly_white")
        return fig


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
        buf = BytesIO()
        out = {'params': params, 'metadata': sim_data.get('metadata', {}),
               'history': []}
        for frame in history:
            out['history'].append({
                k: torch.from_numpy(np.asarray(v))
                for k, v in frame.items() if isinstance(v, np.ndarray)})
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
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            x = np.linspace(extent[0], extent[1], Nx)
            y = np.linspace(extent[2], extent[3], Ny)
            X, Y = np.meshgrid(x, y)
            for i, frame in enumerate(history):
                d = {'x': X.flatten(), 'y': Y.flatten()}
                for k in ('phi', 'eta_poly', 'eta_au', 'eta_nc', 'grain_map',
                          'sigma_eq', 'sigma_h', 'h', 'eps_p_mag', 'D_eff', 'rho_gnd'):
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
                if isinstance(v, (int, float, str)):
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
# SIMULATION DATABASE
# ============================================================================
class SimulationDatabase:
    @staticmethod
    def _hash(params):
        s = json.dumps({k: v for k, v in params.items()
                        if k not in ('history', 'results', 'geom_viz')},
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
    def all():
        return st.session_state.get('sims', {})

    @staticmethod
    def list_for_ui():
        out = []
        for sid, sim in SimulationDatabase.all().items():
            p = sim.get('params', {})
            out.append({
                'id': sid,
                'name': (f"Au-Cu | nc={p.get('nc_grain_size_nm',0):.0f}nm | "
                         f"σ={p.get('applied_stress',0)/1e6:.0f}MPa | "
                         f"β={p.get('misfit_beta',0):.3f} | "
                         f"D_gb={p.get('D_gb',0):.1f}"),
                'params': p,
            })
        return out


# ============================================================================
# ANIMATION
# ============================================================================
@handle_errors
def make_animation(history, field_name, dt, extent, Nx, Ny, output_format='gif',
                   fps=5, dpi=120, cmap_name=None):
    if not history:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    cmap = ColormapRegistry.resolve(cmap_name or
                                     FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'))

    d0, unit = _rescale_field(field_name, history[0][field_name])
    if field_name in ('phi', 'eta_poly', 'eta_au', 'eta_nc', 'phi_gb'):
        vmin, vmax = 0, 1
    elif field_name == 'grain_map':
        vmin, vmax = None, None
    elif field_name == 'sigma_h':
        vmax = np.percentile(np.abs(d0), 98); vmin = -vmax
    else:
        vmin = np.percentile(d0, 2); vmax = np.percentile(d0, 98)

    interp = 'nearest' if field_name in DISCRETE_FIELDS else 'bilinear'
    im = ax.imshow(d0, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='lower', interpolation=interp, aspect='auto')
    if field_name != 'grain_map':
        cbar = plt.colorbar(im, ax=ax); cbar.set_label(f"{field_name} {unit}")
    title = ax.set_title(f"{field_name} — t = 0 ns")

    def update(i):
        d, _ = _rescale_field(field_name, history[i][field_name])
        im.set_array(d)
        title.set_text(f"{field_name} — t = {i * dt:.4f} ns")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(history),
                                   interval=1000/fps, blit=True)
    buf = BytesIO()
    if output_format == 'gif':
        ani.save(buf, writer='pillow', fps=fps, dpi=dpi)
    else:
        ani.save(buf, writer='ffmpeg', fps=fps, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="Au-Cu Diffusion Bonding (Rectangular MPF)",
        layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    .main-header { font-size: 2.1rem; text-align: center; margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #8B4513, #C9A227);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🥇 Au-Assisted Cu-Cu Diffusion Bonding '
                '(Rectangular MPF)</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#FFF8E7; padding:1rem; border-radius:8px;
                border-left:5px solid #C9A227;">
    <b>Fixes applied:</b><br>
    ✅ Rectangular domain (Nx × Ny, independent dx/dy — default 205×102 nm)<br>
    ✅ Fickian diffusion + stress drift (replaced phase-separating CH double-well)<br>
    ✅ Individual nc-Cu grains via categorical <code>grain_map</code> colormap<br>
    ✅ Anisotropic gradient energy κ(θ) and correct plane-strain Green's function<br>
    ✅ Curved Poly-Cu boundaries via Allen-Cahn relaxation of Voronoi map<br>
    ✅ KAM-like dislocation density ρ_GND and solute drag on GB mobility<br>
    ✅ Wavy layer interfaces drawn with <code>ax.plot</code> (axhline fix)
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
            poly_gs   = st.slider("Poly-Cu grain size (nm)", 60.0, 500.0, 150.0, 10.0)
            nc_gs     = st.slider("nc-Cu grain size (nm)", 5.0, 60.0, 15.0, 1.0)
            gb_w      = st.slider("GB width (nm)", 0.5, 5.0, 1.5, 0.1)
            curve_amp = st.slider("Poly-Cu boundary waviness (nm)", 0.0, 20.0, 8.0, 1.0)
            relax_st  = st.slider("Curvature relaxation steps", 0, 200, 40, 10)
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
            D_lat = st.slider("D_lattice", 1e-4, 1.0, 0.01, 1e-3, format="%.4f")
            D_gb  = st.slider("D_gb (grain boundaries)", 1.0, 100.0, 50.0, 1.0)
            Omega = st.slider("Ω (stress-drift coupling)", 0.0, 0.5, 0.05, 0.005)

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

            st.subheader("🎨 Global Colormaps")
            with st.expander("Edit field colormaps", expanded=False):
                cmap_list = ColormapRegistry.list_all()
                global_cmaps = {}
                for f, default in FIELD_DEFAULT_CMAPS.items():
                    idx = cmap_list.index(default) if default in cmap_list else 0
                    global_cmaps[f] = st.selectbox(
                        FIELD_LABELS.get(f, f), cmap_list, index=idx,
                        key=f"g_{f}")

            if st.button("🚀 Initialize", type="primary",
                         use_container_width=True):
                params = {
                    'Nx': Nx, 'Ny': Ny, 'dx': dx, 'dy': dy, 'dt': dt,
                    'poly_thickness_frac': poly_frac,
                    'au_thickness_frac':   au_frac,
                    'poly_grain_size_nm':  poly_gs,
                    'nc_grain_size_nm':    nc_gs,
                    'gb_width_nm':         gb_w,
                    'poly_curve_amp':      curve_amp,
                    'relax_steps':         relax_st,
                    'seed':                int(seed_in),
                    'W': W, 'A': A, 'B': B,
                    'kappa_eta': kappa_eta, 'aniso_strength': aniso_str,
                    'L_ref': 1.0,
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
                    st.session_state.global_cmaps = global_cmaps
                    st.session_state.initialized = True
                    geom = AuCuBondingGeometry(Nx, Ny, dx, dy)
                    c, eta_list, gm, lid, meta = geom.build(
                        poly_thickness_frac=poly_frac,
                        au_thickness_frac=au_frac,
                        poly_grain_size_nm=poly_gs,
                        nc_grain_size_nm=nc_gs,
                        gb_width_nm=gb_w,
                        poly_curve_amp=curve_amp,
                        seed=int(seed_in))
                    st.session_state.geom = geom
                    st.session_state.c0 = c
                    st.session_state.eta0 = eta_list
                    st.session_state.gm0 = gm
                    st.session_state.meta0 = meta
                    st.success(f"✅ Initialized: {meta['n_poly']} Poly + 1 Au "
                               f"+ {meta['n_nc']} nc = {meta['n_total']} phases")

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

    # ==================================================================
    # MAIN AREA
    # ==================================================================
    if mode == "Run New Simulation" and st.session_state.get('initialized'):
        params = st.session_state.params
        Nx, Ny = params['Nx'], params['Ny']
        dx, dy = params['dx'], params['dy']
        meta = st.session_state.meta0

        tabs = st.tabs(["📐 Geometry", "▶️ Run", "📊 Results",
                        "🔬 Line Profiles", "🌐 Plotly", "📤 Export"])

        # ---- Geometry ----------------------------------------------
        with tabs[0]:
            st.header("Initial 3-Layer Geometry (Rectangular)")
            st.caption(f"Poly-Cu: {meta['n_poly']} coarse grains · "
                       f"Au: 1 phase · nc-Cu: {meta['n_nc']} fine grains · "
                       f"Total = {meta['n_total']} phases")
            st.caption(f"Domain: {Nx*dx:.1f} nm × {Ny*dy:.1f} nm  |  "
                       f"grid: {Nx} × {Ny}")

            extent = [-Nx*dx/2, Nx*dx/2, -Ny*dy/2, Ny*dy/2]
            x_coords = np.linspace(extent[0], extent[1], Nx)
            gm = st.session_state.gm0

            # ---- Grain map with wavy interface overlay (FIXED) ------
            fig, ax = plt.subplots(figsize=(10, 4.5))
            cmap = ColormapRegistry.resolve('grain_map')
            im = ax.imshow(gm, extent=extent, origin='lower', cmap=cmap,
                           interpolation='nearest', aspect='equal')

            # FIX: The interfaces are 2D wavy arrays — plot them, don't axhline them.
            _plot_interface(ax, x_coords, meta['y_poly_bot_eff'],
                            color='white', lw=1.5, ls='--',
                            label='Poly-Cu / Au interface')
            _plot_interface(ax, x_coords, meta['y_au_bot_eff'],
                            color='yellow', lw=2.0,
                            label='Au / nc-Cu interface')

            ax.set_title("Voronoi grain map  "
                         "(white dashed = Poly/Au, yellow = Au/nc)")
            ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
            ax.legend(loc='upper right', framealpha=0.85)
            st.pyplot(fig); plt.close(fig)

            # ---- Au concentration with the same interface overlay --
            c0 = st.session_state.c0
            fig, ax = plt.subplots(figsize=(10, 4.5))
            cmap_c = ColormapRegistry.resolve(st.session_state.global_cmaps['phi'])
            im = ax.imshow(c0, extent=extent, origin='lower', cmap=cmap_c,
                           vmin=0, vmax=1, aspect='equal')

            _plot_interface(ax, x_coords, meta['y_poly_bot_eff'],
                            color='white', lw=1.5, ls='--')
            _plot_interface(ax, x_coords, meta['y_au_bot_eff'],
                            color='yellow', lw=2.0)

            plt.colorbar(im, ax=ax, label='c (Au)')
            ax.set_title(f"Initial Au concentration · film thickness ≈ "
                         f"{meta['y_poly_bot']-meta['y_au_bot']:.1f} nm")
            ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
            st.pyplot(fig); plt.close(fig)

            # ---- Individual Poly-Cu η fields -----------------------
            with st.expander("Individual Poly-Cu η fields (check curvature)",
                             expanded=False):
                eta0 = st.session_state.eta0
                n_show = min(4, meta['n_poly'])
                if n_show > 0:
                    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
                    axes = np.atleast_1d(axes)
                    for i in range(n_show):
                        axes[i].imshow(eta0[i], extent=extent, origin='lower',
                                       cmap='Blues', vmin=0, vmax=1,
                                       aspect='equal')
                        axes[i].set_title(f"η_{i}")
                        axes[i].set_xlabel('x (nm)')
                    plt.tight_layout()
                    st.pyplot(fig); plt.close(fig)

        # ---- Run ---------------------------------------------------
        with tabs[1]:
            st.header("Run the MPF Simulation")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                **Expected physics with current settings:**
                - D_lattice = `{params['D_lattice']:.4f}`, D_gb = `{params['D_gb']:.1f}`
                - Ω (stress drift) = `{params['Omega']:.3f}`
                - Applied stress = `{params['applied_stress']/1e6:.0f} MPa @ `
                  `{params['applied_stress_angle']:.0f}°`
                - Au-Cu misfit β = `{params['misfit_beta']:.3f}`
                - Solute drag α = `{params['alpha_drag']:.0f}`
                - Steps × dt = `{params['n_steps']} × {params['dt']:.4f} ns`
                  = `{params['n_steps']*params['dt']:.3f} ns` total
                """)
            if st.button("▶️ Start Evolution", type="primary",
                         use_container_width=True):
                with st.spinner("Simulating MPF + Fickian/Hart + Solute Drag..."):
                    solver = AuCuBondingSolver(params)
                    solver.c = st.session_state.c0.copy()
                    solver.eta_list = [e.copy() for e in st.session_state.eta0]

                    n_steps = params['n_steps']
                    save_freq = params['save_frequency']
                    history, timesteps = [], []
                    progress = st.progress(0)
                    status = st.empty()

                    for step in range(n_steps):
                        res = solver.step()
                        if step % save_freq == 0 or step == n_steps - 1:
                            history.append({k: (v.copy()
                                                if isinstance(v, np.ndarray)
                                                else v)
                                            for k, v in res.items()})
                            timesteps.append(step * params['dt'])
                        progress.progress((step+1) / n_steps)
                        if step % max(1, n_steps // 20) == 0:
                            status.text(
                                f"Step {step+1}/{n_steps} · "
                                f"c_top={solver.history['c_top'][-1]:.4f} · "
                                f"c_bot={solver.history['c_bot'][-1]:.4f} · "
                                f"Δc="
                                f"{solver.history['c_bot'][-1]-solver.history['c_top'][-1]:+.4f}")

                    st.success(f"✅ Done. {len(history)} frames saved.")
                    st.session_state.history = history
                    st.session_state.timesteps = timesteps
                    st.session_state.solver = solver
                    sid = SimulationDatabase.save(params, history, None,
                                                   run_time=n_steps * params['dt'])
                    st.session_state.last_sid = sid
                    st.balloons()

        # ---- Results -----------------------------------------------
        with tabs[2]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                history = st.session_state.history
                solver = st.session_state.solver
                idx = st.slider("Frame", 0, len(history)-1,
                                len(history)-1, key="res_frame")
                vis = EnhancedTwinVisualizer(Nx, Ny, dx, dy, dt=params['dt'])
                style = {'colormaps': st.session_state.global_cmaps}
                fig = vis.create_multi_field_comparison(history[idx], style)
                if fig is not None:
                    st.pyplot(fig); plt.close(fig)

                st.subheader("Region-Averaged Diagnostics")
                t = np.array(solver.history['time'])

                fig, axes = plt.subplots(2, 2, figsize=(13, 8))
                axes[0, 0].plot(t, solver.history['c_top'], 'b-', lw=2,
                                label='Poly-Cu (top)')
                axes[0, 0].plot(t, solver.history['c_bot'], 'r-', lw=2,
                                label='nc-Cu (bottom)')
                axes[0, 0].set_xlabel('t (ns)')
                axes[0, 0].set_ylabel('mean c (Au)')
                axes[0, 0].set_title('Au concentration: Poly-Cu vs nc-Cu')
                axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

                axes[0, 1].plot(t, solver.history['gb_top'], 'b-', lw=2,
                                label='Poly-Cu')
                axes[0, 1].plot(t, solver.history['gb_bot'], 'r-', lw=2,
                                label='nc-Cu')
                axes[0, 1].set_xlabel('t (ns)')
                axes[0, 1].set_ylabel('mean φ_gb')
                axes[0, 1].set_title('GB density (Hart indicator)')
                axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

                axes[1, 0].plot(t, np.array(solver.history['avg_stress'])/1e9,
                                'g-', lw=2, label='Avg')
                axes[1, 0].plot(t, np.array(solver.history['max_stress'])/1e9,
                                'g--', lw=1.5, label='Max')
                axes[1, 0].set_xlabel('t (ns)')
                axes[1, 0].set_ylabel('σ (GPa)')
                axes[1, 0].set_title('Stress evolution')
                axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

                au_mass = np.array(solver.history['au_mass'])
                axes[1, 1].plot(t, au_mass / au_mass[0], 'k-', lw=2,
                                label='Au mass / initial')
                c_top_arr = np.array(solver.history['c_top'])
                c_bot_arr = np.array(solver.history['c_bot'])
                ax2 = axes[1, 1].twinx()
                ax2.plot(t, c_top_arr, 'b--', lw=1.5, label='c_top')
                ax2.plot(t, c_bot_arr, 'r--', lw=1.5, label='c_bot')
                ax2.set_ylabel('mean c')
                axes[1, 1].set_xlabel('t (ns)')
                axes[1, 1].set_ylabel('mass ratio')
                axes[1, 1].set_title('Mass conservation')
                axes[1, 1].legend(loc='upper left')
                ax2.legend(loc='upper right')
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

        # ---- Line profiles ------------------------------------------
        with tabs[3]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                history = st.session_state.history
                idx = st.slider("Frame", 0, len(history)-1, len(history)-1,
                                key="lp_frame")
                res = history[idx]
                profiler = EnhancedLineProfiler(Nx, Ny, dx, dy)
                field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()),
                                      index=0, key="lp_field")
                ptype = st.selectbox("Profile direction",
                                      ["horizontal", "vertical", "diagonal",
                                       "anti_diagonal"], key="lp_ptype")
                pos = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05,
                                key="lp_pos")

                d, unit = _rescale_field(field, res[field])
                dist, prof, _ = profiler.extract_profile(d, ptype, pos)
                if dist is not None:
                    fig, ax = plt.subplots(figsize=(11, 4))
                    ax.plot(dist, prof, 'b-', lw=2)
                    ax.set_xlabel('Position (nm)')
                    ax.set_ylabel(f'{field} {unit}')
                    ax.set_title(f'{field} · {ptype} profile · frame {idx}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig); plt.close(fig)

        # ---- Plotly -------------------------------------------------
        with tabs[4]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                history = st.session_state.history
                idx = st.slider("Frame", 0, len(history)-1, len(history)-1,
                                key="px_frame")
                vis = EnhancedTwinVisualizer(Nx, Ny, dx, dy, dt=params['dt'])
                field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()),
                                      index=0, key="px_field")
                cmap = st.session_state.global_cmaps.get(
                    field, FIELD_DEFAULT_CMAPS[field])
                fig_h = vis.create_plotly_heatmap(history[idx], field, idx, cmap)
                if fig_h is not None:
                    st.plotly_chart(fig_h, use_container_width=True)
                fig_3d = vis.create_plotly_3d_surface(history[idx], field, idx,
                                                       cmap)
                if fig_3d is not None:
                    st.plotly_chart(fig_3d, use_container_width=True)

        # ---- Export -------------------------------------------------
        with tabs[5]:
            if 'history' not in st.session_state:
                st.info("Run the simulation first.")
            else:
                params = st.session_state.params
                history = st.session_state.history
                sim_data = {'metadata': {'created_at': datetime.now().isoformat()}}
                name = (f"au_cu_bonding_Nx{params['Nx']}_Ny{params['Ny']}"
                        f"_seed{params['seed']}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Pickle"):
                        buf, fname = DataExporter.export_pkl(sim_data, params,
                                                              history, name)
                        st.download_button("Download .pkl", buf, fname)
                    if st.button("JSON"):
                        buf, fname = DataExporter.export_json(sim_data, params,
                                                               history, name)
                        st.download_button("Download .json", buf, fname)
                with col2:
                    if st.button("CSV (zip)"):
                        extent = [-Nx*dx/2, Nx*dx/2, -Ny*dy/2, Ny*dy/2]
                        buf, fname = DataExporter.export_csv(history, name,
                                                              extent, Nx, Ny,
                                                              dx, dy)
                        st.download_button("Download .zip", buf, fname)
                    if st.button("HDF5"):
                        buf, fname = DataExporter.export_hdf5(sim_data, params,
                                                               history, name,
                                                               Nx, Ny, dx, dy)
                        if buf:
                            st.download_button("Download .h5", buf, fname)

                st.markdown("---")
                anim_field = st.selectbox("Animation field",
                                            list(FIELD_DEFAULT_CMAPS.keys()),
                                            index=0, key="anim_field")
                fps = st.slider("FPS", 1, 20, 5, key="anim_fps")
                if st.button("🎬 Generate GIF"):
                    extent = [-Nx*dx/2, Nx*dx/2, -Ny*dy/2, Ny*dy/2]
                    cmap = st.session_state.global_cmaps.get(
                        anim_field, FIELD_DEFAULT_CMAPS[anim_field])
                    buf = make_animation(history, anim_field, params['dt'],
                                          extent, Nx, Ny, 'gif', fps, 120, cmap)
                    if buf:
                        st.download_button("Download .gif", buf,
                                            f"{name}_{anim_field}.gif")

    # ==================================================================
    elif mode == "Single Simulation View" and st.session_state.get('view_id'):
        sid = st.session_state.view_id
        sim = SimulationDatabase.get(sid)
        if sim is None:
            st.error("Simulation not found.")
        else:
            st.header(f"Simulation: {sid}")
            params = sim['params']
            history = sim['results_history']
            Nx = params['Nx']; Ny = params['Ny']
            dx = params['dx']; dy = params.get('dy', dx)

            st.json({k: v for k, v in params.items()
                     if not isinstance(v, np.ndarray)})

            if history:
                idx = st.slider("Frame", 0, len(history)-1, len(history)-1,
                                key=f"view_frame_{sid}")
                vis = EnhancedTwinVisualizer(Nx, Ny, dx, dy, dt=params['dt'])
                style = {'colormaps': FIELD_DEFAULT_CMAPS}
                fig = vis.create_multi_field_comparison(history[idx], style)
                if fig is not None:
                    st.pyplot(fig); plt.close(fig)

            if st.button("🗑️ Delete"):
                SimulationDatabase.delete(sid)
                del st.session_state.view_id
                st.rerun()

    # ==================================================================
    elif mode == "Compare Saved" and st.session_state.get('compare_ids'):
        ids = st.session_state.compare_ids
        sims = [SimulationDatabase.get(i) for i in ids]
        sims = [s for s in sims if s is not None]
        if not sims:
            st.warning("No valid simulations selected.")
        else:
            st.header(f"Comparing {len(sims)} simulations")
            field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()),
                                  index=0, key="cmp_field")
            cmap = st.session_state.get('global_cmaps', {}).get(
                field, FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
            cols = min(3, len(sims))
            rows = (len(sims) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            axes = np.array(axes).flatten()
            for i, sim in enumerate(sims):
                p = sim['params']
                h = sim['results_history']
                if not h: continue
                last = h[-1]
                if field not in last: continue
                d, unit = _rescale_field(field, last[field])
                extent = [-p['Nx']*p['dx']/2, p['Nx']*p['dx']/2,
                          -p['Ny']*p['dy']/2, p['Ny']*p['dy']/2]
                ax = axes[i]
                interp = 'nearest' if field in DISCRETE_FIELDS else 'bilinear'
                im = ax.imshow(d, extent=extent, origin='lower',
                               cmap=ColormapRegistry.resolve(cmap),
                               interpolation=interp, aspect='equal')
                ax.set_title(f"{sim['id']} — {field}")
                if field != 'grain_map':
                    plt.colorbar(im, ax=ax, fraction=0.046)
            for k in range(len(sims), len(axes)):
                axes[k].axis('off')
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)


if __name__ == "__main__":
    main()
